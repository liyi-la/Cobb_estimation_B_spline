import torch.nn as nn
from torchvision import models
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

resnet50 = models.resnet50(weights=True)
class ResNetFeatures(nn.Module):
    def __init__(self, original_model):
        super(ResNetFeatures, self).__init__()
        # 提取ResNet的前几层
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x 
    
    
class FPN(nn.Module):
    def __init__(self, backbone):
        super(FPN, self).__init__()
        # 使用ResNet的前几层作为骨干网络
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])#backbone:resnet50
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 定义自顶向下的卷积层
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1)  # Reduce channels
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1)

        # 定义平滑层
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.'''
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # Bottom-up pathway
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down pathway
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        
        return p3

from timm.models.swin_transformer import SwinTransformer

class Swintrans(nn.Module):
    def __init__(self, img_size=(1024,256), patch_size=4, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super(Swintrans, self).__init__()
        # Swin Transformer 骨干网络
        self.backbone = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            num_classes=0  # 不进行分类，只提取特征
        )
        

    def forward(self, x):
        # 提取特征
        features = self.backbone.forward_features(x)
        bc,h,w,c=features.shape
        features=features.permute(0,3,1,2)
        
        return features

# 示例：初始化模型
# model = SwinHeatmapModel(num_keypoints=68, img_size=224)
# print(model)

# 示例输入
# x = torch.randn(1, 3, 224, 224)  # 输入图像 (B, C, H, W)
# heatmap = model(x)
# print(heatmap.shape)  # 输出热图 (B,out, H, W)
 ### Decoder of CenterNet
 # Decoder中采用UpSample + BN + Activation作为一个block，以此堆叠三次作为一个Decoder。
 # 其中CenterNet的UpSample为反卷积，激活函数为ReLU。需要注意的是，三个反卷积的核大小都为4x4，卷积核的数目分别为256，128，64。
 # 那么经过Decoder之后，feature map的宽高维度则变为原来1/4（比较重要，后面会反复用到），通道维度为64。   
class CenterNetDecoder(nn.Module):
    def __init__(self, in_channels, bn_momentum=0.1):
        super(CenterNetDecoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.in_channels = in_channels
        self.deconv_with_bias = False

        # h/32, w/32, 2048 -> h/16, w/16, 256 -> h/8, w/8, 128 -> h/4, w/4, 64
        self.deconv_layers = self._make_deconv_layer(
            num_layers=1,
            # num_filters=[256, 128, 64],##其他网络，需要多个反卷积
            num_filters=[64],#FPN,只有一个反卷积层，输出通道数为64
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):## 3层反卷积
            kernel = num_kernels[i]
            num_filter = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=num_filter,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(num_filter, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = num_filter
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)
    
 ### Head  CenterNet原论文中还需要进行分类，我们只需要检测关键点，所以不需要类别   
class CenterNetHead(nn.Module):
    def __init__(self, channel=256, bn_momentum=0.1):
        super(CenterNetHead, self).__init__()

        # heatmap
        self.cls_head = nn.Sequential(
            nn.Conv2d(channel,32,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,1,kernel_size=1, stride=1, padding=0),##num_classes=1
        )
        # center point offset
        self.offset_head = nn.Sequential(
            nn.Conv2d(channel,32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x)
        offset = self.offset_head(x)

        return hm, offset




class CenterNet(nn.Module):
    def __init__(self,backbone,init=False):
        super(CenterNet, self).__init__()
        
        if backbone == 'ResNet50':
            self.backbone=ResNetFeatures(resnet50)
        elif backbone == 'FPN':
            self.backbone=FPN(resnet50)
        elif backbone == 'SwinTransformer':
            self.backbone=Swintrans(img_size=(1024,256), patch_size=4, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
        else:
            raise Exception("There is no {}.".format(backbone))
        self.decoder=CenterNetDecoder(256)
        self.head=CenterNetHead(64)
        
        if init:#是否采用正态分布进行卷积参数初始化
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x=self.backbone(x)
        x=self.decoder(x)
        heatmap1,offset=self.head(x)
        # print(heatmap1.shape,offset.shape)
        return heatmap1


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Hourglass(nn.Module):
    def __init__(self, in_channels, intermediate_channels, num_keypoints):
        super(Hourglass, self).__init__()
        self.down1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(intermediate_channels)
        self.down2 = nn.Conv2d(intermediate_channels, intermediate_channels*2, kernel_size=3, stride=2, padding=1)
        self.res2 = ResidualBlock(intermediate_channels*2)
        self.bottleneck = ResidualBlock(intermediate_channels*2)
        self.up2 = nn.ConvTranspose2d(intermediate_channels*2, intermediate_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1 = nn.ConvTranspose2d(intermediate_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.heatmap_layer = nn.Conv2d(intermediate_channels*2, num_keypoints, kernel_size=1)
    
    def forward(self, x):
        down1 = self.down1(x)
        res1 = self.res1(down1)
        down2 = self.down2(res1)
        res2 = self.res2(down2)
        bottleneck = self.bottleneck(res2)
        up2 = self.up2(bottleneck)
        up1 = self.up1(up2)
        heatmap = self.heatmap_layer(bottleneck)
        heatmap = F.interpolate(heatmap, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return up1, heatmap
    
class StackedHourglass(nn.Module):
    def __init__(self, in_channels, num_stacks=2, intermediate_channels=128, num_keypoints=1):
        super(StackedHourglass, self).__init__()
        self.num_stacks = num_stacks
        self.hourglasses = nn.ModuleList([Hourglass(in_channels, intermediate_channels, num_keypoints) for _ in range(num_stacks)])
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
    
    def forward(self, x):
        x = self.init_conv(x)
        heatmaps = []
        for i in range(self.num_stacks):
            x, heatmap = self.hourglasses[i](x)
            heatmaps.append(heatmap)
        return heatmaps



##网络整体搭建  
    
