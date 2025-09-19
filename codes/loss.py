import torch
import torch.nn as nn


class KeypointBSplineLoss(nn.Module):
    """KeypointBSplineNet的损失函数"""
    
    def __init__(self, cp_weight=1.0, knots_weight=10.0, angle_weight=1.0):
        super(KeypointBSplineLoss, self).__init__()
        self.cp_weight = cp_weight
        self.knots_weight = knots_weight
        self.angle_weight = angle_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_cp, pred_knots,
                gt_cp, gt_knots):
        """
        计算总损失
        Args:
            pred_cp: 预测控制点 [B, num_control_points, 2]
            pred_knots: 预测节点 [B, num_control_points + degree + 1]
            gt_cp: 真实控制点 [B, num_control_points, 2]
            gt_knots: 真实节点 [B, num_control_points + degree + 1]
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失详情
        """
        # 归一化控制点
        pred_cp_norm = normalize_control_points(pred_cp)
        gt_cp_norm = normalize_control_points(gt_cp)
        
        
        # 控制点损失
        cp_loss = self.mse_loss(pred_cp_norm, gt_cp_norm)
        
        # 节点损失
        knots_loss = self.mse_loss(pred_knots, gt_knots)
        
        # 总损失
        total_loss = (self.cp_weight * cp_loss 
                    + self.knots_weight * knots_loss)
                                                         
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'cp_loss': cp_loss.item(),
            'knots_loss': knots_loss.item()
        }
        
        return total_loss, loss_dict
    
def normalize_control_points(cp):
    """
    归一化控制点
    Args:
        cp: 控制点 [B, num_control_points, 2]
    Returns:
        normalized_cp: 归一化后的控制点
    """
    # 计算每个batch的均值和标准差
    mean = cp.mean(dim=1, keepdim=True)  # [B, 1, 2]
    std = cp.std(dim=1, keepdim=True)    # [B, 1, 2]
    
    # 避免除零
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    
    # 归一化
    normalized_cp = (cp - mean) / std
    return normalized_cp