import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import json
from mDataset import CobbNetDataset
from CobbNet import KeypointBSplineNet
from loss import KeypointBSplineLoss

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 配置参数
config = {
    'path_heatmap': r'D:\Project\Xiehe_Spinal_image_stitching\cobb\Heatmap',
    'path_image': r'D:\Project\Xiehe_Spinal_image_stitching\cobb\ke30_u7_AASCE2019-master\boostnet_labeldata',
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'num_keypoints': 34,
    'num_control_points': 10,
    'degree': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'checkpoints_cobb2',
    'log_dir': 'logs_cobb2'
}

# 创建保存目录
os.makedirs(config['save_dir'], exist_ok=True)
os.makedirs(config['log_dir'], exist_ok=True)

# 数据集和数据加载器
print("加载数据集...")
train_dataset = CobbNetDataset(config['path_image'], config['path_heatmap'], train=True)
train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=False, num_workers=0)

test_dataset = CobbNetDataset(config['path_image'], config['path_heatmap'], train=False)
test_loader = DataLoader(test_dataset, 8, shuffle=False, num_workers=0)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
mse_loss = nn.MSELoss()
# 模型
model = KeypointBSplineNet(
    num_keypoints=config['num_keypoints'],
    num_control_points=config['num_control_points'],
    degree=config['degree'],
    device=config['device']
).to(config['device'])



# 损失函数和优化器
criterion = KeypointBSplineLoss(cp_weight=1.0, knots_weight=10.0)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# 训练历史记录
train_history = {
    'epoch': [],
    'train_loss': [],
    'train_cp_loss': [],
    'train_knots_loss': [],
    'val_loss': [],
    'val_cp_loss': [],
    'val_knots_loss': [],
    'lr': []
}

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_cp_loss = 0.0
    total_knots_loss = 0.0
    num_batches = 0
    
    for batch_idx, (origin_shape, label,  image_name, 
                   kp_pred, cp, knots, paras, cp_GT, knots_GT, paras_GT, 
                   cobb_angle, cobb_angle_GT) in enumerate(train_loader):
        
        # 数据移到设备
        kp_pred = kp_pred.to(device, dtype=torch.float32)
        cobb_angle_GT = cobb_angle_GT.to(device, dtype=torch.float32) 
        # 前向传播
        # with torch.autograd.set_detect_anomaly(True):
        optimizer.zero_grad()
        
        cobb_angles_pred, deta_keyp = model(kp_pred)
        loss_batch = mse_loss(cobb_angles_pred, cobb_angle_GT)

        # print("deta_keyp", deta_keyp)
        
        # 反向传播
        loss_batch.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 记录损失
        total_loss += loss_batch.item()
        # total_cp_loss += loss_dict['cp_loss']
        # total_knots_loss += loss_dict['knots_loss']
        num_batches += 1
        
        # 打印进度
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {total_loss}, '
                  )
    
    return total_loss / num_batches

def validate_epoch(model, test_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    total_cp_loss = 0.0
    total_knots_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for origin_shape, label, image_name, \
            kp_pred, cp, knots, paras, cp_GT, knots_GT, paras_GT, \
            cobb_angle, cobb_angle_GT in test_loader:
            
            kp_pred = kp_pred.to(device, dtype=torch.float32)
            cobb_angle_GT = cobb_angle_GT.to(device, dtype=torch.float32) 
            
            # 前向传播
            cobb_angles_pred, deta_keyp = model(kp_pred)
            
            # 计算损失
            mse_loss = nn.MSELoss()
            total_loss_batch = mse_loss(cobb_angles_pred, cobb_angle_GT)
            
            # 记录损失
            total_loss += total_loss_batch.item()
            # total_cp_loss += loss_dict['cp_loss']
            # total_knots_loss += loss_dict['knots_loss']
            num_batches += 1
    
    return total_loss / num_batches
         

def plot_training_history(history, save_path):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    
    # 总损失
    axes[0].plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(history['epoch'], history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # # 控制点损失
    # axes[1].plot(history['epoch'], history['train_cp_loss'], 'b-', label='Train CP Loss')
    # axes[1].plot(history['epoch'], history['val_cp_loss'], 'r-', label='Val CP Loss')
    # axes[1].set_title('Control Points Loss')
    # axes[1].set_xlabel('Epoch')
    # axes[1].set_ylabel('Loss')
    # axes[1].legend()
    # axes[1].grid(True)
    
    # # 节点损失
    # axes[2].plot(history['epoch'], history['train_knots_loss'], 'b-', label='Train Knots Loss')
    # axes[2].plot(history['epoch'], history['val_knots_loss'], 'r-', label='Val Knots Loss')
    # axes[2].set_title('Knots Loss')
    # axes[2].set_xlabel('Epoch')
    # axes[2].set_ylabel('Loss')
    # axes[2].legend()
    # axes[2].grid(True)
    
    # 学习率
    axes[1].plot(history['epoch'], history['lr'], 'g-', label='Learning Rate')
    axes[1].set_title('Learning Rate')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

# 主训练循环
print("开始训练...")
start_time = time.time()
best_val_loss = float('inf')

for epoch in range(config['num_epochs']):
    epoch_start_time = time.time()
    
    # 训练
    train_loss = train_epoch(
        model, train_loader, criterion, optimizer, config['device'], epoch
    )
    
    # 验证
    val_loss = validate_epoch(
        model, test_loader, criterion, config['device']
    )
    
    # 学习率调度
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # 记录历史（确保转换为Python原生类型）
    train_history['epoch'].append(epoch)
    train_history['train_loss'].append(train_loss.item() if hasattr(train_loss, 'item') else train_loss)
    train_history['val_loss'].append(val_loss.item() if hasattr(val_loss, 'item') else val_loss)
    train_history['lr'].append(current_lr)
    
    # 打印epoch结果
    epoch_time = time.time() - epoch_start_time
    print(f'Epoch {epoch:3d}/{config["num_epochs"]} | '
          f'Train Loss: {train_loss:.6f} | '
          f'Val Loss: {val_loss:.6f} | '
          f'LR: {current_lr:.2e} | '
          f'Time: {epoch_time:.2f}s')
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, val_loss, 
                       os.path.join(config['save_dir'], 'best_model.pth'))
        print(f'新的最佳模型已保存，验证损失: {val_loss:.6f}')
    
    # # 定期保存检查点
    # if (epoch + 1) % 10 == 0:
    #     save_checkpoint(model, optimizer, epoch, val_loss,
    #                    os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}.pth'))
    
    # 定期绘制训练历史
    if (epoch + 1) % 5 == 0:
        plot_training_history(train_history, 
                             os.path.join(config['log_dir'], f'training_history_epoch_{epoch}.png'))
    
    # 保存训练历史（转换张量为Python原生类型）
    train_history_serializable = {}
    for key, value in train_history.items():
        if isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'item'):
            # 如果是张量列表，转换为Python原生类型
            train_history_serializable[key] = [v.item() if hasattr(v, 'item') else v for v in value]
        else:
            train_history_serializable[key] = value
    
    with open(os.path.join(config['log_dir'], 'training_history.json'), 'w') as f:
        json.dump(train_history_serializable, f, indent=2)

# 训练完成
total_time = time.time() - start_time
print(f'\n训练完成！总时间: {total_time/3600:.2f} 小时')
print(f'最佳验证损失: {best_val_loss:.6f}')

# 最终绘制训练历史
plot_training_history(train_history, 
                     os.path.join(config['log_dir'], 'final_training_history.png'))

# 保存最终模型
save_checkpoint(model, optimizer, config['num_epochs']-1, val_loss,
               os.path.join(config['save_dir'], 'final_model.pth'))

print(f'模型和日志已保存到: {config["save_dir"]} 和 {config["log_dir"]}')