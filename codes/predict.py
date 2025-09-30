import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from collections import defaultdict
from mDataset import CobbNetDataset
from CobbNet import KeypointBSplineNet
from B_spline_torch import BS_curve_torch
from B_spline import BS_curve
from utils import cobb_angle_line_torch, line_intersection, find_intersection, calculate_perpendicular_lines,calculate_slopes

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

    
    def predict_single(self, keypoints, knots, cp):
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
            knots = knots.unsqueeze(0)
            cp = cp.unsqueeze(0)
            cobb_angles_pred,deta_cp,deta_knots = self.model(keypoints, knots, cp)
            
            # 计算Cobb角
            # cobb_angles = self._compute_cobb_angles(pred_cp, pred_knots)
            
            result = {
                # 'control_points': pred_cp.squeeze(0).cpu().numpy(),
                'deta_cp': deta_cp.squeeze(0),  # 保持为PyTorch张量
                'deta_knots': deta_knots.squeeze(0),  # 保持为PyTorch张量
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
    

def visualize_result(img_name, p, y_c_pred, y_c_fixed):
    slopes_pred=np.array(calculate_slopes(y_c_pred))
    avg_slopes_pred=[-1/s for s in slopes_pred]


    if np.argmax(avg_slopes_pred)> np.argmin(avg_slopes_pred):#两种情况：左弯或者右弯
        lower_MT_pred= np.argmax(avg_slopes_pred)
        upper_MT_pred = np.argmin(avg_slopes_pred)
    else:
        upper_MT_pred= np.argmax(avg_slopes_pred)#斜率为正在上脊柱
        lower_MT_pred = np.argmin(avg_slopes_pred)#斜率为负在下脊柱,注意纵坐标是相反的
    def find_line(y_c_pred,upper,lower):
        mink1_x=[y_c_pred[upper][1],y_c_pred[upper+1][1]]
        mink1_y=[y_c_pred[upper][0],y_c_pred[upper+1][0]]
        maxk1_x=[y_c_pred[lower][1],y_c_pred[lower+1][1]]
        maxk1_y=[y_c_pred[lower][0],y_c_pred[lower+1][0]]
        mink1, mink2 =[mink1_x[0],mink1_y[0]], [mink1_x[1],mink1_y[1]]
        maxk1, maxk2 =[maxk1_x[0],maxk1_y[0]], [maxk1_x[1],maxk1_y[1]]
        return mink1, mink2, maxk1, maxk2
    mink1, mink2, maxk1, maxk2 = find_line(y_c_pred,upper_MT_pred,lower_MT_pred)

    upper_max_slope= np.amax(avg_slopes_pred[0:upper_MT_pred+1])
    upper_maxId= np.argmax(avg_slopes_pred[0:upper_MT_pred+1])
    upper_min_slope = np.amin(avg_slopes_pred[0:upper_MT_pred+1])
    upper_minId = np.argmin(avg_slopes_pred[0:upper_MT_pred+1])

    lower_max_slope=np.amax(avg_slopes_pred[lower_MT_pred:])
    lower_maxId=np.argmax(avg_slopes_pred[lower_MT_pred:])+lower_MT_pred
    lower_min_slope = np.amin(avg_slopes_pred[lower_MT_pred:])
    lower_minId = np.argmin(avg_slopes_pred[lower_MT_pred:])+lower_MT_pred 
    mink1_upper, mink2_upper,maxk1_upper, maxk2_upper = find_line(y_c_pred,upper_maxId,upper_minId)
    mink1_lower, mink2_lower,maxk1_lower, maxk2_lower = find_line(y_c_pred,lower_maxId,lower_minId)


    intersection = line_intersection(maxk1,maxk2,mink1,mink2)
    intersection_upper = line_intersection(mink1_upper, mink2_upper,maxk1_upper, maxk2_upper)
    intersection_lower = line_intersection(mink1_lower, mink2_lower,maxk1_lower, maxk2_lower)
    #MT
    mink_c=[(mink1[0]+mink2[0])/2,(mink1[1]+mink2[1])/2]
    maxk_c=[(maxk1[0]+maxk2[0])/2,(maxk1[1]+maxk2[1])/2]#计算中点
    intersection_c_pred= find_intersection([mink1,mink2],[maxk1,maxk2])#中垂线交点
    p_line1_pred, p_line2_pred = calculate_perpendicular_lines([mink1,mink2],[maxk1,maxk2], 20)#中垂线

    ##upper
    mink_c_upper=[(mink1_upper[0]+mink2_upper[0])/2,(mink1_upper[1]+mink2_upper[1])/2]
    maxk_c_upper=[(maxk1_upper[0]+maxk2_upper[0])/2,(maxk1_upper[1]+maxk2_upper[1])/2]#计算中点
    intersection_c_upper_pred= find_intersection([mink1_upper,mink2_upper],[maxk1_upper,maxk2_upper])#中垂线交点
    p_line1_upper_pred, p_line2_upper_pred = calculate_perpendicular_lines([mink1_upper,mink2_upper],[maxk1_upper,maxk2_upper], 20)#中垂线
    ##lower
    mink_c_lower=[(mink1_lower[0]+mink2_lower[0])/2,(mink1_lower[1]+mink2_lower[1])/2]
    maxk_c_lower=[(maxk1_lower[0]+maxk2_lower[0])/2,(maxk1_lower[1]+maxk2_lower[1])/2]#计算中点
    intersection_c_lower_pred= find_intersection([mink1_lower,mink2_lower],[maxk1_lower,maxk2_lower])#中垂线交点
    p_line1_lower_pred, p_line2_lower_pred = calculate_perpendicular_lines([mink1_lower,mink2_lower],[maxk1_lower,maxk2_lower], 20)#中垂线

    slopes_fixed=np.array(calculate_slopes(y_c_fixed))
    avg_slopes_fixed=[-1/s for s in slopes_fixed]
    if np.argmax(avg_slopes_fixed)> np.argmin(avg_slopes_fixed):#两种情况：左弯或者右弯
        lower_MT_fixed= np.argmax(avg_slopes_fixed)
        upper_MT_fixed = np.argmin(avg_slopes_fixed)
    else:
        upper_MT_fixed= np.argmax(avg_slopes_fixed)#斜率为正在上脊柱
        lower_MT_fixed = np.argmin(avg_slopes_fixed)#斜率为负在下脊柱,注意纵坐标是相反的
    def find_line(y_c_fixed,upper,lower):
        mink1_x=[y_c_fixed[upper][1],y_c_fixed[upper+1][1]]
        mink1_y=[y_c_fixed[upper][0],y_c_fixed[upper+1][0]]
        maxk1_x=[y_c_fixed[lower][1],y_c_fixed[lower+1][1]]
        maxk1_y=[y_c_fixed[lower][0],y_c_fixed[lower+1][0]]
        mink1, mink2 =[mink1_x[0],mink1_y[0]], [mink1_x[1],mink1_y[1]]
        maxk1, maxk2 =[maxk1_x[0],maxk1_y[0]], [maxk1_x[1],maxk1_y[1]]
        return mink1, mink2, maxk1, maxk2
    mink1, mink2, maxk1, maxk2 = find_line(y_c_fixed,upper_MT_fixed,lower_MT_fixed)

    upper_max_slope= np.amax(avg_slopes_fixed[0:upper_MT_fixed+1])
    upper_maxId= np.argmax(avg_slopes_fixed[0:upper_MT_fixed+1])
    upper_min_slope = np.amin(avg_slopes_fixed[0:upper_MT_fixed+1])
    upper_minId = np.argmin(avg_slopes_fixed[0:upper_MT_fixed+1])

    lower_max_slope=np.amax(avg_slopes_fixed[lower_MT_fixed:])
    lower_maxId=np.argmax(avg_slopes_fixed[lower_MT_fixed:])+lower_MT_fixed
    lower_min_slope = np.amin(avg_slopes_fixed[lower_MT_fixed:])
    lower_minId = np.argmin(avg_slopes_fixed[lower_MT_fixed:])+lower_MT_fixed 
    mink1_upper, mink2_upper,maxk1_upper, maxk2_upper = find_line(y_c_fixed,upper_maxId,upper_minId)
    mink1_lower, mink2_lower,maxk1_lower, maxk2_lower = find_line(y_c_fixed,lower_maxId,lower_minId)


    intersection = line_intersection(maxk1,maxk2,mink1,mink2)
    intersection_upper = line_intersection(mink1_upper, mink2_upper,maxk1_upper, maxk2_upper)
    intersection_lower = line_intersection(mink1_lower, mink2_lower,maxk1_lower, maxk2_lower)
    #MT
    mink_c=[(mink1[0]+mink2[0])/2,(mink1[1]+mink2[1])/2]
    maxk_c=[(maxk1[0]+maxk2[0])/2,(maxk1[1]+maxk2[1])/2]#计算中点
    intersection_c_fixed= find_intersection([mink1,mink2],[maxk1,maxk2])#中垂线交点
    p_line1_fixed, p_line2_fixed = calculate_perpendicular_lines([mink1,mink2],[maxk1,maxk2], 20)#中垂线

    ##upper
    mink_c_upper=[(mink1_upper[0]+mink2_upper[0])/2,(mink1_upper[1]+mink2_upper[1])/2]
    maxk_c_upper=[(maxk1_upper[0]+maxk2_upper[0])/2,(maxk1_upper[1]+maxk2_upper[1])/2]#计算中点
    intersection_c_upper_fixed= find_intersection([mink1_upper,mink2_upper],[maxk1_upper,maxk2_upper])#中垂线交点
    p_line1_upper_fixed, p_line2_upper_fixed = calculate_perpendicular_lines([mink1_upper,mink2_upper],[maxk1_upper,maxk2_upper], 20)#中垂线
    ##lower
    mink_c_lower=[(mink1_lower[0]+mink2_lower[0])/2,(mink1_lower[1]+mink2_lower[1])/2]
    maxk_c_lower=[(maxk1_lower[0]+maxk2_lower[0])/2,(maxk1_lower[1]+maxk2_lower[1])/2]#计算中点
    intersection_c_lower_fixed= find_intersection([mink1_lower,mink2_lower],[maxk1_lower,maxk2_lower])#中垂线交点
    p_line1_lower_fixed, p_line2_lower_fixed = calculate_perpendicular_lines([mink1_lower,mink2_lower],[maxk1_lower,maxk2_lower], 20)#中垂线

    img = cv2.imread(img_name)
    h,w=img.shape[:2]
    img = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
    fig = plt.figure(figsize=(4,7.5))
    ax = fig.add_subplot(111)
    # 绘制两对点到交点的延长线
    ax.imshow(img, aspect='auto', alpha=1)
    
    num_p = len(p)//2
    slopes =[]
    avg_slopes=[]
    lines = []
    c_lines = []
    xs = []
    ys = []

    for m in range (0,num_p,2):
        slope = ((img.shape[0]*p[m+1+num_p])-(img.shape[0]*p[m+num_p]))/((img.shape[1]*p[m+1])-(img.shape[1]*p[m]))#计算每两对点之间的斜率
        #print (p[m+num_p])
        slopes.append(slope)
        lines.append((((img.shape[1]*p[m+1]),(img.shape[0]*p[m+1+num_p])),((img.shape[1]*p[m]),(img.shape[0]*p[m+num_p]))))#连线(x1,y1),(x2,y2)
        xs.append((p[m]+p[m+1])*w/2) # 原图像脊柱中心点
        ys.append((p[m+num_p]+p[m+num_p+1])*h/2)
    for s in range (0,num_p//2,2):
        avg_slopes.append((slopes[s]+slopes[s+1])/2)#每个椎体的平均斜率
        p0_L=lines[s][0]
        p0_R=lines[s][1]
        p1_L=lines[s+1][0]
        p1_R=lines[s+1][1]
        pc_L=((p0_L[0]+p1_L[0])/2,(p0_L[1]+p1_L[1])/2)
        pc_R=((p0_R[0]+p1_R[0])/2,(p0_R[1]+p1_R[1])/2)
        c_lines.append(((pc_L),(pc_R)))

    if np.argmax(avg_slopes[1:num_p//4 - 1])> np.argmin(avg_slopes[1:num_p//4 - 1]):#两种情况：左弯或者右弯
        lower_MT= np.argmax(avg_slopes[1:num_p//4 - 1]) + 1
        upper_MT = np.argmin(avg_slopes[1:num_p//4 - 1]) + 1
        
    else:
        upper_MT= np.argmax(avg_slopes[1:num_p//4 - 1]) + 1#斜率为正在上脊柱
        lower_MT = np.argmin(avg_slopes[1:num_p//4 - 1]) + 1#斜率为负在下脊柱,注意纵坐标是相反的

    upper_maxId= np.argmax(avg_slopes[0:upper_MT+1])
    upper_minId = np.argmin(avg_slopes[0:upper_MT+1])
    lower_maxId=np.argmax(avg_slopes[lower_MT:num_p//4])+lower_MT
    lower_minId = np.argmin(avg_slopes[lower_MT:num_p//4])+lower_MT
    ax.plot([c_lines[upper_minId][0][0],c_lines[upper_minId][1][0]],[c_lines[upper_minId][0][1],c_lines[upper_minId][1][1]],color='w')##x1,x2,y1,y2
    ax.plot([c_lines[upper_maxId][0][0],c_lines[upper_maxId][1][0]],[c_lines[upper_maxId][0][1],c_lines[upper_maxId][1][1]],color='w')
    ax.plot([c_lines[lower_minId][0][0],c_lines[lower_minId][1][0]],[c_lines[lower_minId][0][1],c_lines[lower_minId][1][1]],color='w')
    ax.plot([c_lines[lower_maxId][0][0],c_lines[lower_maxId][1][0]],[c_lines[lower_maxId][0][1],c_lines[lower_maxId][1][1]],color='w')
    
    ax.plot(y_c_pred[:,1],y_c_pred[:,0],'-b')#B样条曲线

    # ax.plot(cp[:,1],cp[:,0],'b*')#输出拟合的控制点坐标
    ##center
    ax.plot([p_line1_pred[0][0],p_line1_pred[1][0]],[p_line1_pred[0][1],p_line1_pred[1][1]],color='r')
    ax.plot([p_line2_pred[0][0],p_line2_pred[1][0]],[p_line2_pred[0][1],p_line2_pred[1][1]],color='r')#最大斜率椎体
    ax.scatter(*intersection_c_pred, color='blue', label='Intersection')
    ##upper
    ax.plot([p_line1_upper_pred[0][0],p_line1_upper_pred[1][0]],[p_line1_upper_pred[0][1],p_line1_upper_pred[1][1]],color='r')
    ax.plot([p_line2_upper_pred[0][0],p_line2_upper_pred[1][0]],[p_line2_upper_pred[0][1],p_line2_upper_pred[1][1]],color='r')#最大斜率椎体
    ax.scatter(*intersection_c_upper_pred, color='blue', label='Intersection')
    ##lower
    ax.plot([p_line1_lower_pred[0][0],p_line1_lower_pred[1][0]],[p_line1_lower_pred[0][1],p_line1_lower_pred[1][1]],color='r')
    ax.plot([p_line2_lower_pred[0][0],p_line2_lower_pred[1][0]],[p_line2_lower_pred[0][1],p_line2_lower_pred[1][1]],color='r')#最大斜率椎体
    ax.scatter(*intersection_c_lower_pred, color='blue', label='Intersection')

    slopes_fixed=np.array(calculate_slopes(y_c_fixed))
    avg_slopes_fixed=[-1/s for s in slopes_fixed]


    # ax.scatter(xs,ys)#输入点

    ax.plot(y_c_fixed[:,1],y_c_fixed[:,0],'yellow')#B样条曲线
    # ax.plot(cp[:,1],cp[:,0],'b*')#输出拟合的控制点坐标
    ##center
    ax.plot([p_line1_fixed[0][0],p_line1_fixed[1][0]],[p_line1_fixed[0][1],p_line1_fixed[1][1]],color='g')
    ax.plot([p_line2_fixed[0][0],p_line2_fixed[1][0]],[p_line2_fixed[0][1],p_line2_fixed[1][1]],color='g')#最大斜率椎体
    ax.scatter(*intersection_c_fixed, color='yellow', label='Intersection')
    ##upper
    ax.plot([p_line1_upper_fixed[0][0],p_line1_upper_fixed[1][0]],[p_line1_upper_fixed[0][1],p_line1_upper_fixed[1][1]],color='g')
    ax.plot([p_line2_upper_fixed[0][0],p_line2_upper_fixed[1][0]],[p_line2_upper_fixed[0][1],p_line2_upper_fixed[1][1]],color='g')#最大斜率椎体
    ax.scatter(*intersection_c_upper_fixed, color='yellow', label='Intersection')
    ##lower
    ax.plot([p_line1_lower_fixed[0][0],p_line1_lower_fixed[1][0]],[p_line1_lower_fixed[0][1],p_line1_lower_fixed[1][1]],color='g')
    ax.plot([p_line2_lower_fixed[0][0],p_line2_lower_fixed[1][0]],[p_line2_lower_fixed[0][1],p_line2_lower_fixed[1][1]],color='g')#最大斜率椎体
    ax.scatter(*intersection_c_lower_fixed, color='yellow', label='Intersection')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.set_xlim(0, img.shape[1]) 
    ax.set_ylim(img.shape[0] ,0 )  

    ax.set_title(img_name.split('/')[-1])
    # ax.invert_yaxis()
    # ax.grid(True)
    plt.show()
    

def main():
    """主函数示例"""
    # 配置
    model_path = 'checkpoints_cp_knots/best_model.pth'  # 模型路径
    
    # 数据集路径
    path_heatmap = r'D:\Project\Xiehe_Spinal_image_stitching\cobb\Heatmap'
    path_image = r'D:\Project\Xiehe_Spinal_image_stitching\cobb\ke30_u7_AASCE2019-master\boostnet_labeldata'
    
    # 创建预测器
    predictor = CobbNetPredictor(model_path)
    
    # 创建数据集
    test_dataset = CobbNetDataset(path_image, path_heatmap, train=False)
    
    # 预测前5个样本
    mae_pred = defaultdict(list)  # 存 method1 的绝对误差和分母
    mae_fixed = defaultdict(list)  # 存 method2 的绝对误差和分母
    smape_pred = []  # 存所有角的 (abs_err, gt+pred)
    smape_fixed = []
    print("开始预测...")
    for batch_idx, (origin_shape,  label, image_name, p,
                   kp_pred, cp, knots, paras, cp_GT, knots_GT, paras_GT, 
                   cobb_angle, cobb_angle_GT) in enumerate(test_dataset):
        kp_pred_tensor = torch.tensor(kp_pred, dtype=torch.float32, device=predictor.device)
        knots_tensor = torch.tensor(knots, dtype=torch.float32, device=predictor.device)
        cp_tensor = torch.tensor(cp, dtype=torch.float32, device=predictor.device)
        result= predictor.predict_single(kp_pred_tensor, knots_tensor, cp_tensor)
        deta_cp = result['deta_cp']
        deta_knots = result['deta_knots']
        kp_pred_tensor = kp_pred_tensor.reshape(-1)
        # kp_fixed = kp_pred_tensor + deta_cp
        # kp_fixed_np = kp_fixed.cpu().numpy().reshape(34,2)

        ### 预测关键点计算B样条曲线可视化
        bs_pred=BS_curve(9,3)
        paras = bs_pred.estimate_parameters(kp_pred) # B样条参数
        knots = bs_pred.get_knots() # 节点
        cp = bs_pred.approximation(kp_pred) # 控制点
        # print(f"  预测的节点：{knots}")
        # print(f"  预测的控制点：{cp}")
        uq = np.linspace(0,1,34)
        y_c_pred = np.array(bs_pred.bs(uq)) # 计算B样条曲线

       
        ### 修正后的关键点计算B样条曲线可视化
        bs_fixed=BS_curve(9,3)
        # paras = bs_fixed.estimate_parameters(kp_fixed_np) # B样条参数
        zeros = torch.zeros(4).to(predictor.device)
        deta_knots_14 = torch.cat([zeros, deta_knots, zeros], dim=0)
        knots = deta_knots_14 + knots_tensor# 节点
        cp = deta_cp + cp_tensor
        paras = bs_fixed.estimate_parameters(kp_pred)
        cp = cp.cpu().numpy()
        knots = knots.cpu().numpy()
        bs_fixed.cp = cp
        bs_fixed.u = knots

        # print(f"  修正后的节点：{knots}")
        # print(f"  修正后的控制点：{cp}")
        # cp = bs_fixed.approximation(kp_fixed_np) # 控制点
        uq = np.linspace(0,1,34)
        y_c_fixed = np.array(bs_fixed.bs(uq)) # 计算B样条曲线
      
        visualize_result(image_name, p, y_c_pred, y_c_fixed)
        
        
        print(image_name)
        print(f"预测结果:")
        # print(f"  曲线：{y_c_torch}")
        print(f"  关键点坐标：{result['deta_cp'].shape}")
        print(f"  基于预测的中心点计算角度：{cobb_angle}")
        print(f"  修正后的Cobb角: {result['cobb_angles']}")
        print(f"  Cobb角GT: {cobb_angle_GT}")
        print("-" * 50)
        
        angles = ['MT', 'PT', 'TL']

        for i, angle_name in enumerate(angles):
            gt = cobb_angle_GT[i]
            pred1 = cobb_angle[i]
            pred2 = result['cobb_angles'][i]

            # MAE
            mae_pred[angle_name].append(abs(gt - pred1))
            mae_fixed[angle_name].append(abs(gt - pred2))

            # SMAPE
            smape_pred.append((abs(gt - pred1), gt + pred1))
            smape_fixed.append((abs(gt - pred2), gt + pred2))
        
        
        if batch_idx >= 5:  # 只处理前5个样本
            break
        
    # 计算 MAE
    def compute_mae(mae_dict):
        return {angle: sum(errors) / len(errors) for angle, errors in mae_dict.items()}

    mae1 = compute_mae(mae_pred)
    mae2 = compute_mae(mae_fixed)

    print("基于预测的中心点计算角度 MAE (MT, PT, TL):", mae1)
    print("基于修正后的中心点计算角度 MAE (MT, PT, TL):", mae2)

    # 计算 SMAPE
    def compute_smape(smape_list):
        total_err = sum(v[0] for v in smape_list)
        total_sum = sum(v[1] for v in smape_list)
        return (total_err / total_sum) * 100

    smape1 = compute_smape(smape_pred)
    smape2 = compute_smape(smape_fixed)

    print("基于预测的中心点计算角度 Overall SMAPE:", smape1)
    print("基于修正后的中心点计算角度 Overall SMAPE:", smape2)


if __name__ == "__main__":
    main()
