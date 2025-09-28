import csv
import torch
import cv2
import math
import numpy as np
from torch.utils.data import Dataset
from B_spline import BS_curve
from utils import keyP,cobb_angle_line


path_heatmap=r'D:\Project\Xiehe_Spinal_image_stitching\cobb\Heatmap'
path_image=r'D:\Project\Xiehe_Spinal_image_stitching\cobb\ke30_u7_AASCE2019-master\boostnet_labeldata'
class CobbNetDataset(Dataset):
    def __init__(self, path, path_heatmap, train=True):
        self.names = []
        self.labels = []
        self.cobb_angles=[]
        self.heatmap_names=[]
        self.scale=4#heatmap的缩放倍数
        self.train=train
        if train:
            image_path = path + "/data/training_preprocessed/"  # 原图
            heatmaps_path = path_heatmap+"/pred_training_cp/"   # 对应的热图
            names = csv.reader(open(path + "/labels/training/filenames.csv", 'r'))  # 文件名
            cobb_angles= csv.reader(open(path + "/labels/training/angles.csv", 'r')) # 对应的cobb角
            names=list(names)
            self.names=[image_path+n[0] for n in names]
            self.heatmap_names=[heatmaps_path+n[0] for n in names]
            landmarks = csv.reader(open(path + "/labels/training/landmarks.csv", 'r'))
        else:
            image_path = path + "/data/test/"
            heatmaps_path = path_heatmap+"/pred_test_cp/"
            names = csv.reader(open(path + "/labels/test/filenames.csv", 'r'))
            cobb_angles= csv.reader(open(path + "/labels/test/angles.csv", 'r'))
            names=list(names)
            self.names=[image_path+n[0] for n in names]
            self.heatmap_names=[heatmaps_path+n[0] for n in names]
            landmarks = csv.reader(open(path + "/labels/test/landmarks.csv", 'r'))
        
        for landmark_each_image in landmarks:  # 地标
            coordinate_list = []
            for coordinate in landmark_each_image:
                coordinate_list.append(float(coordinate))
            self.labels.append(coordinate_list)
            
        for cobb_each in cobb_angles:  # cobb角
            cobb_list = []
            for cobb in cobb_each:
                cobb_list.append(float(cobb))
            self.cobb_angles.append(cobb_list)
             

    def pad_img(self, img, flag=True): 
          
        h,w=img.shape[:2]
        if(flag):
            h_max=3840
            w_max=1536
        else:
            h_max=960
            w_max=384
        top = math.floor((h_max - h)/2)
        bottom = round((h_max - h)/2+0.1)
        left = math.floor((w_max - w) / 2)
        right = round((w_max - w) / 2+0.1)#四舍五入
        image_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        return image_padded

    def __getitem__(self, index):
        image_name = self.names[index]
        label = self.labels[index]
        cobb_angle_GT=np.array(self.cobb_angles[index])

        heatmap_name=self.heatmap_names[index]
        origin_image = cv2.imread(image_name)
        image_padded=self.pad_img(origin_image)

        target_height = image_padded.shape[0]//4 # 3840/4=960 缩小原图输入
        target_width =image_padded.shape[1]//4 # 1536/4=384
        image_resize=cv2.resize(image_padded, (target_width, target_height))#结果存在image
        heatmap = cv2.imread(heatmap_name,0)
        heatmap_padded=self.pad_img(heatmap,False)
        kp = keyP(heatmap)
        heatmap_y=[coord[0] for coord in kp]
        heatmap_x=[coord[1] for coord in kp]
        bs=BS_curve(9,3)  #10个控制点，3次B样条
        kp_pred = np.array([heatmap_y,heatmap_x]).T # 基于热图计算B样条参数
        paras = bs.estimate_parameters(kp_pred) # B样条参数
        knots = bs.get_knots() # 节点
        if bs.check():
            cp = bs.approximation(kp_pred) # 控制点
        uq = np.linspace(0,1,34)
        y_c = np.array(bs.bs(uq)) # 计算B样条曲线
        cobb_angle=np.array(cobb_angle_line(y_c)) # 基于预测的中心点计算角度
        xs=[]
        ys=[]
        p=label
        img_src_resize=cv2.resize(origin_image, (target_width, target_height))
        h,w=img_src_resize.shape[:2]
        num_p = len(p)//2
        for i in range(0,num_p,2):
            xs.append((p[i]+p[i+1])*w/2) # 原图像脊柱中心点
            ys.append((p[i+num_p]+p[i+num_p+1])*h/2)

        bs=BS_curve(9,3) # 10个控制点，3次B样条
        kp_GT = np.array([ys,xs]).T # 基于GT控制点计算B样条参数
        paras_GT = bs.estimate_parameters(kp_GT)
        knots_GT = bs.get_knots()
        if bs.check():
            cp_GT = bs.approximation(kp_GT)

 
        # heatmap_resize=cv2.resize(heatmap_padded ,(target_width//self.scale,target_height//self.scale)) #960/4=240 ,384/4=96缩小heatmap输入
        
        # image_resize = torch.tensor(image_resize, dtype=torch.float32)
        # heatmap = torch.tensor(heatmap, dtype=torch.float32)

        
        return origin_image.shape,label,image_name,p,kp_pred,cp,knots,paras,cp_GT,knots_GT,paras_GT,cobb_angle,cobb_angle_GT

    def __len__(self):
        return len(self.names)

