import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import FancyBboxPatch
import os

from mDataset import CobbNetDataset
from CobbNet import KeypointBSplineNet
from B_spline_torch import BS_curve_torch
from utils import cobb_angle_line_torch, line_intersection, find_intersection, calculate_perpendicular_lines

class CobbNetPredictor:
    """CobbNet预测器"""
    
    def __init__(self, model_path):
        """
        初始化预测器
        Args:
            model_path: 模型检查点路径
        """
        self.config = {
            'num_keypoints': 34,
            'num_control_points': 10,
            'degree': 3,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        self.device = self.config['device']
        self.model = KeypointBSplineNet(
            num_keypoints=self.config['num_keypoints'],
            num_control_points=self.config['num_control_points'],
            degree=self.config['degree'],
            device=self.device
        ).to(self.device)

        # 加载模型
        self._load_model(model_path)
    
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        print(f"加载模型: {model_path}")
  
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    
    def predict_single(self, keypoints):
        """
        预测单个样本
        Args:
            keypoints: 关键点坐标 [num_keypoints, 2] 或 [1, num_keypoints, 2]
            return_curve: 是否返回B样条曲线点
        Returns:
            result: 预测结果字典
        """
        with torch.no_grad():
            # 确保输入格式正确
            if isinstance(keypoints, np.ndarray):
                keypoints = torch.from_numpy(keypoints).float()
            
            if keypoints.dim() == 2:
                keypoints = keypoints.unsqueeze(0)  # 添加batch维度
            
            # 确保tensor是连续的
            keypoints = keypoints.contiguous().to(self.device)
            
            # 模型预测
            cobb_angles_pred,kp_deta = self.model(keypoints)
            
            # 计算Cobb角
            # cobb_angles = self._compute_cobb_angles(pred_cp, pred_knots)
            
            result = {
                # 'control_points': pred_cp.squeeze(0).cpu().numpy(),
                'kp_deta': kp_deta.squeeze(0),  # 保持为PyTorch张量
                'cobb_angles': cobb_angles_pred.squeeze(0).cpu().numpy()
            }
            
            return result
    
    def _compute_cobb_angles(self, cp, knots):
        """计算Cobb角"""
        batch_size = cp.shape[0]
        cobb_angles_list = []
        
        for b in range(batch_size):
            try:
                # 创建B样条对象
                bs = BS_curve_torch(self.config['num_control_points'] - 1, 
                                  self.config['degree'], device=self.device)
                bs.cp = cp[b]
                bs.u = knots[b]
                bs.m = knots.shape[1] - 1
                
                if bs.check():
                    # 采样34个点
                    uq = torch.linspace(0, 1, 34, device=self.device, dtype=cp.dtype)
                    curve_points = bs.bs(uq)
                    
                    # 计算Cobb角
                    cobb_angle = cobb_angle_line_torch(curve_points, device=self.device, dtype=cp.dtype)
                    cobb_angles_list.append(cobb_angle)
                else:
                    cobb_angles_list.append(torch.zeros(3, device=self.device, dtype=cp.dtype))
                    
            except Exception as e:
                print(f"B样条计算错误: {e}")
                cobb_angles_list.append(torch.zeros(3, device=self.device, dtype=cp.dtype))
        
        return torch.stack(cobb_angles_list)
    



def main():
    """主函数示例"""
    # 配置
    model_path = 'checkpoints_cobb2/best_model.pth'  # 模型路径
    
    # 数据集路径
    path_heatmap = r'D:\Project\Xiehe_Spinal_image_stitching\cobb\Heatmap'
    path_image = r'D:\Project\Xiehe_Spinal_image_stitching\cobb\ke30_u7_AASCE2019-master\boostnet_labeldata'
    
    # 创建预测器
    predictor = CobbNetPredictor(model_path)
    
    # 创建数据集
    test_dataset = CobbNetDataset(path_image, path_heatmap, train=False)
    
    # 预测前5个样本
    print("开始预测...")
    for batch_idx, (origin_shape,  label, image_name, 
                   kp_pred, cp, knots, paras, cp_GT, knots_GT, paras_GT, 
                   cobb_angle, cobb_angle_GT) in enumerate(test_dataset):
        # 数据集返回的是元组，根据train.py中的顺序获取数据
        # 将numpy数组转换为PyTorch张量
        kp_pred_tensor = torch.tensor(kp_pred, dtype=torch.float32, device=predictor.device)
        
        result= predictor.predict_single(kp_pred_tensor)
        deta_kp = result['kp_deta']
        kp_pred_tensor = kp_pred_tensor.reshape(1, -1)

        
        print(image_name)
        print(f"预测结果:")
        # print(f"  曲线：{y_c_torch}")
        print(f"  关键点坐标：{result['kp_deta'].shape}")
        print(f"  原Cobb角：{cobb_angle}")
        print(f"  修正后的Cobb角: {result['cobb_angles']}")
        print(f"  Cobb角GT: {cobb_angle_GT}")
        print("-" * 50)
        
        
        if batch_idx >= 4:  # 只处理前5个样本
            break
        
   
    


if __name__ == "__main__":
    main()
