import torch
import torch.nn as nn

from B_spline_torch import BS_curve_torch
from utils import cobb_angle_line_torch

class KeypointBSplineNet(nn.Module):
    """
    基于关键点的B样条Cobb角测量网络
    输入：关键点坐标 (data_pred)
    输出：控制点 (cp) 和节点 (knots)
    通过B样条解析计算Cobb角
    """
    
    def __init__(self, num_keypoints=34, num_control_points=10, degree=3, num_angles=3, device='cpu'):
        super(KeypointBSplineNet, self).__init__()
        self.num_keypoints = num_keypoints
        self.num_control_points = num_control_points
        self.degree = degree
        self.num_angles = num_angles
        self.device = device
        
        # 输入维度：num_keypoints * 2 (x, y坐标)
        input_dim = num_keypoints * 2
        
        # 控制点输出维度：num_control_points * 2 (x, y坐标)
        cp_output_dim = num_control_points * 2
        
        # 节点输出维度：num_control_points + degree + 1
        knots_output_dim = num_control_points + degree + 1 - 8
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2)
        )
        
        # 预测变化控制点
        self.deta_kp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, self.num_keypoints * 2)
        )
        # 控制点预测头
        self.cp_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, cp_output_dim)
        )
        
        # 节点预测头
        self.knots_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, knots_output_dim),
            nn.Hardtanh(min_val=0.0, max_val=1.0)   # 把输出直接钳位到 [0,1] 
        )
        # Cobb角预测头
        self.cobb_angle_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, num_angles)
        )
        
        # B样条初始化
        self.bs_torch = BS_curve_torch(9,3,self.device)
        # 在模型 __init__ 最后或 build 阶段
        with torch.no_grad():
            # 把最后一层线性层的权重和偏置置 0
            self.deta_kp[-1].weight.zero_()
            self.deta_kp[-1].bias.zero_()
            self.cp_head[-1].weight.zero_()
            self.cp_head[-1].bias.zero_()
            self.knots_head[-2].weight.zero_()
            self.knots_head[-2].bias.zero_()
            
    def forward(self, keypoints, knots, cp):
        """
        前向传播
        Args:
            keypoints: 关键点坐标 [B, num_keypoints, 2] 或 [B, num_keypoints*2]
        Returns:
            cp: 预测的控制点 [B, num_control_points, 2]
            knots: 预测的节点 [B, num_control_points + degree + 1]
            cobb_angles: 计算的Cobb角 [B, num_angles]
        """
        batch_size = keypoints.shape[0]

        # 确保输入是2D的
        if keypoints.dim() == 3:
            keypoints = keypoints.view(batch_size, -1)  # [B, num_keypoints*2]
        
        features = self.feature_extractor(keypoints)  # [B, 256]
        # deta_keyp = self.deta_kp(features) # [B, num_keypoints * 2]
        # kps = deta_keyp + keypoints  
       # 预测控制点
        cp_flat = self.cp_head(features)  # [B, num_control_points*2]
        deta_cp = cp_flat.view(batch_size, self.num_control_points, 2)  # [B, num_control_points, 2]
        cps = deta_cp + cp.to(self.device)  # [B, num_control_points, 2]
        cps = cps.to(dtype=torch.float32)
        # 预测节点
        deta_knots_6 = self.knots_head(features)  # [B, num_control_points + degree + 1]
        zeros = torch.zeros(batch_size, 4).to(self.device)
        deta_knots_14 = torch.cat([zeros, deta_knots_6, zeros], dim=1)
        knots = deta_knots_14 + knots.to(self.device)  # [B, num_control_points + degree + 1]
        knots = knots.to(dtype=torch.float32)
   
        # 特征提取
        
        # 计算cobb角
        bs_torch = BS_curve_torch(9,3,self.device)
        cobb_angles = []
        for i in range(batch_size):
            kp = keypoints[i].view(self.num_keypoints, 2)  # [num_keypoints, 2]
            paras = bs_torch.estimate_parameters(kp)  # 
            cp_i = cps[i].view(self.num_control_points, 2)  # [num_control_points, 2]
            knot_i = knots[i]  # [num_control_points + degree + 1]
            bs_torch.cp = cp_i
            bs_torch.u = knot_i
            uq = torch.linspace(0,1,34).to(self.device)
            y_c_torch = bs_torch.bs(uq)
            cobb_angle_torch = cobb_angle_line_torch(y_c_torch)
            cobb_angles.append(cobb_angle_torch)
            
        #     
            
        #     knots = bs_torch.get_knots()
        #     cp = bs_torch.approximation(kp)
        #     uq = torch.linspace(0,1,34).to(self.device)
        #     y_c_torch = bs_torch.bs(uq)
        #     cobb_angle_torch = cobb_angle_line_torch(y_c_torch)

            
        #     cobb_angles.append(cobb_angle_torch)
        cobb_angles = torch.stack(cobb_angles).squeeze(-1).to(self.device) # [bs, 3]
  

        # cobb_angles = self.cobb_angle_head(features)  # [B, num_angles]
        return cobb_angles,deta_cp,deta_knots_6
    

    


    
