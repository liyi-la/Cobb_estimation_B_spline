"计算交点，角度"
import torch
import numpy as np
import cv2
from scipy.interpolate import interp1d
from typing import Union


def line_intersection(p1, p2, p3, p4):
    """
    计算两条直线的交点
    :param p1: 第一对点的第一个点
    :param p2: 第一对点的第二个点
    :param p3: 第二对点的第一个点
    :param p4: 第二对点的第二个点
    :return: 交点坐标
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:
        # 两条直线平行或重合
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return (px, py)
def find_intersection(line1, line2):
    "计算两条直线中垂线交点"
    # 计算第一条直线的中点
    midpoint1 = [(line1[0][0] + line1[1][0]) / 2, (line1[0][1] + line1[1][1]) / 2]
    # 计算第二条直线的中点
    midpoint2 = [(line2[0][0] + line2[1][0]) / 2, (line2[0][1] + line2[1][1]) / 2]

    # 计算第一条直线的斜率
    if line1[1][0] - line1[0][0] != 0:
        slope1 = (line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0])
    else:
        slope1 = float('inf')  # 垂直直线的斜率为无穷大

    # 计算第二条直线的斜率
    if line2[1][0] - line2[0][0] != 0:
        slope2 = (line2[1][1] - line2[0][1]) / (line2[1][0] - line2[0][0])
    else:
        slope2 = float('inf')  # 垂直直线的斜率为无穷大

    # 计算第一条直线中垂线的斜率
    if slope1 != 0 and slope1 != float('inf'):
        perpendicular_slope1 = -1 / slope1
    elif slope1 == 0:
        perpendicular_slope1 = float('inf')
    else:
        perpendicular_slope1 = 0

    # 计算第二条直线中垂线的斜率
    if slope2 != 0 and slope2 != float('inf'):
        perpendicular_slope2 = -1 / slope2
    elif slope2 == 0:
        perpendicular_slope2 = float('inf')
    else:
        perpendicular_slope2 = 0

    # 使用点斜式方程 y - y1 = m(x - x1) 表示中垂线方程
    # 转化为斜截式方程 y = mx + b
    if perpendicular_slope1 != float('inf'):
        b1 = midpoint1[1] - perpendicular_slope1 * midpoint1[0]
    if perpendicular_slope2 != float('inf'):
        b2 = midpoint2[1] - perpendicular_slope2 * midpoint2[0]

    # 求解两条中垂线的交点
    if perpendicular_slope1 == float('inf'):
        x = midpoint1[0]
        y = perpendicular_slope2 * x + b2
    elif perpendicular_slope2 == float('inf'):
        x = midpoint2[0]
        y = perpendicular_slope1 * x + b1
    else:
        x = (b2 - b1) / (perpendicular_slope1 - perpendicular_slope2)
        y = perpendicular_slope1 * x + b1

    return [x, y]


def angle_between_vectors(v1, v2):
    """
    计算两个向量之间的夹角
    :param v1: 第一个向量
    :param v2: 第二个向量
    :return: 夹角（弧度）
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(cos_angle)
    if angle>np.pi/2:
        angle=np.pi-angle
    return angle

def calculate_perpendicular_lines(line1, line2, d):
    def calculate_perpendicular_endpoints(line, d):
        # 计算直线的中点
        midpoint = np.array([(line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2])
        # 计算直线的向量
        vector = np.array([line[1][0] - line[0][0], line[1][1] - line[0][1]])
        # 计算垂直向量
        perpendicular_vector = np.array([-vector[1], vector[0]])
        # 归一化垂直向量
        perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
        # 计算中垂线的两个端点
        endpoint1 = midpoint + d * perpendicular_vector
        endpoint2 = midpoint - d * perpendicular_vector
        return [endpoint1.tolist(), endpoint2.tolist()]

    # 计算第一条直线的中垂线
    perpendicular_line1 = calculate_perpendicular_endpoints(line1, d)
    # 计算第二条直线的中垂线
    perpendicular_line2 = calculate_perpendicular_endpoints(line2, d)

    return perpendicular_line1, perpendicular_line2

def keyP(image, min_threshold=10, dilation_size=5, min_distance=2) -> list:
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    local_max = (image == dilated)#如果膨胀后的图像在某一点的值和原图像一样，那这个点就是一个局部最大值。
    local_max = local_max & (image > min_threshold)
    coords = np.column_stack(np.where(local_max))#返回[y,x]

    # 在图像上绘制圆心
    filtered_coords = []
    for (y,x) in coords:
        if all(np.sqrt((x-cx)**2+(y-cy)**2)>=min_distance for (cy,cx) in filtered_coords):
            filtered_coords.append((y,x))
    filtered_coords = normalize_keypoints(filtered_coords)
    return filtered_coords#返回[y,x]


def normalize_keypoints( keypoints, target_count = 34):
        """
        将关键点数量标准化到目标数量
        Args:
            keypoints: 原始关键点 [N, 2] 或 [(y,x), ...] 列表
            target_count: 目标关键点数量
        Returns:
            normalized_keypoints: 标准化后的关键点 [target_count, 2]
        """
        # 确保keypoints是numpy数组
        if isinstance(keypoints, list):
            keypoints = np.array(keypoints)
        
        if len(keypoints) == 0:
            # 如果没有检测到关键点，返回零填充
            return np.zeros((target_count, 2))
        
        if len(keypoints) == target_count:
            return keypoints
        
        if len(keypoints) < target_count:
            # 关键点不足，使用插值填充
            return interpolate_keypoints(keypoints, target_count)
        else:
            # 关键点过多，使用均匀采样
            return sample_keypoints(keypoints, target_count)
    
def interpolate_keypoints( keypoints, target_count):
    """
    使用插值方法填充关键点
    """
    # 确保keypoints是numpy数组
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints)
    
    if len(keypoints) < 2:
        # 如果关键点太少，直接重复填充
        repeated = np.tile(keypoints, (target_count // len(keypoints) + 1, 1))
        return repeated[:target_count]
    
    # 计算累积弧长参数
    distances = np.sqrt(np.sum(np.diff(keypoints, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    
    if cumulative_distances[-1] == 0:
        # 如果所有点都相同，返回重复的点
        return np.tile(keypoints[0:1], (target_count, 1))
    
    # 归一化参数
    t_old = cumulative_distances / cumulative_distances[-1]
    t_new = np.linspace(0, 1, target_count)
    
    # 插值
    
    f_x = interp1d(t_old, keypoints[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
    f_y = interp1d(t_old, keypoints[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
    
    x_new = f_x(t_new)
    y_new = f_y(t_new)
    return np.column_stack([x_new, y_new])


def sample_keypoints(keypoints, target_count):
    """
    均匀采样关键点
    """
    # 确保keypoints是numpy数组
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints)
    
    if len(keypoints) <= target_count:
        return keypoints
    
    indices = np.linspace(0, len(keypoints) - 1, target_count, dtype=int)
    return keypoints[indices]

def calculate_slopes(points):
    # 计算每对相邻点之间的斜率
    slopes = []
    for i in range(len(points) - 1):
        y1, x1 = points[i]
        y2, x2 = points[i + 1]
        # 计算斜率
        if x2 - x1 == 0:
            print("Warning: Vertical line detected, skipped")
            continue
        slope = (y2 - y1) / (x2 - x1)
       
        slopes.append(slope)
    return slopes

def cobb_angle_line(y_c) -> list:
    "通过B样条曲线采样点y_c计算COBB角度"
    slopes=np.array(calculate_slopes(y_c))
    avg_slopes=[-1/s for s in slopes]

    max_slope = np.amax(avg_slopes)
    min_slope= np.amin(avg_slopes)
    if np.argmax(avg_slopes)> np.argmin(avg_slopes):#两种情况：左弯或者右弯
        lower_MT= np.argmax(avg_slopes)
        upper_MT = np.argmin(avg_slopes)
    else:
        upper_MT= np.argmax(avg_slopes)#斜率为正在上脊柱
        lower_MT = np.argmin(avg_slopes)#斜率为负在下脊柱,注意纵坐标是相反的

    upper_max_slope= np.amax(avg_slopes[0:upper_MT+1])
    upper_maxId= np.argmax(avg_slopes[0:upper_MT+1])
    upper_min_slope = np.amin(avg_slopes[0:upper_MT+1])
    upper_minId = np.argmin(avg_slopes[0:upper_MT+1])

    lower_max_slope=np.amax(avg_slopes[lower_MT:])
    lower_maxId=np.argmax(avg_slopes[lower_MT:])+lower_MT
    lower_min_slope = np.amin(avg_slopes[lower_MT:])
    lower_minId = np.argmin(avg_slopes[lower_MT:])+lower_MT 


    cobb_angles= [0.0,0.0,0.0]  
    cobb_angles[0]= abs(np.rad2deg(np.arctan(max_slope))- np.rad2deg(np.arctan(min_slope)))
    cobb_angles[1]= abs(np.rad2deg(np.arctan(upper_max_slope))-np.rad2deg(np.arctan(upper_min_slope)))
    cobb_angles[2]= abs(np.rad2deg(np.arctan(lower_max_slope)) - np.rad2deg(np.arctan(lower_min_slope)))
    
    return cobb_angles

def calculate_slopes_torch(y_c: torch.Tensor) -> torch.Tensor:
    """
    Calculate slopes between consecutive points in a curve.
    
    Args:
        y_c: Curve points tensor of shape [N, 2] where N is number of points
        
    Returns:
        Slopes tensor of shape [N-1]
    """
    if len(y_c.shape) != 2 or y_c.shape[1] != 2:
        raise ValueError("y_c should have shape [N, 2]")

    
    # Calculate differences
    dx = y_c[1:, 1] - y_c[:-1, 1]  # x differences
    dy = y_c[1:, 0] - y_c[:-1, 0]  # y differences
    
    # Handle zero dx cases to avoid division by zero
    dx_safe = torch.where(torch.abs(dx) < 1e-6,  # 增加阈值，减少除零风险
                         torch.sign(dx) * 1e-6 + 1e-10,  # 确保不为零
                         dx)
    
    slopes = dy / dx_safe

    # [NaN 调试] 检查 slopes 是否有NaN/Inf
    # if torch.isnan(slopes).any() or torch.isinf(slopes).any():
    #     print(f"[calculate_slopes_torch] Error: slopes contains NaN or Inf!")
    #     raise ValueError("slopes contains NaN or Inf")

    return slopes

def cobb_angle_line_torch(y_c: torch.Tensor, 
                         device: Union[str, torch.device] = None,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Calculate COBB angles from B-spline curve sampling points using PyTorch.
    
    Args:
        y_c: Curve points tensor of shape [N, 2] where each point is [x, y]
        device: Device to perform computation on
        dtype: Data type for computation
        
    Returns:
        COBB angles tensor [total_angle, upper_angle, lower_angle]
    """
    # Convert to tensor if needed and move to specified device
    
    if device is not None:
        y_c = y_c.to(device)
    
    if dtype != y_c.dtype:
        y_c = y_c.to(dtype)
    
    # Calculate slopes
    slopes = calculate_slopes_torch(y_c)
    
    # [NaN 调试] 检查 slopes 是否有NaN/Inf
    if torch.isnan(slopes).any() or torch.isinf(slopes).any():
        print(f"[cobb_angle_line_torch] Error: slopes contains NaN or Inf!")
        raise ValueError("slopes contains NaN or Inf")

    # Calculate average slopes (perpendicular slopes: -1/s)
    # Handle division by zero for slopes
    slopes_safe = torch.where(torch.abs(slopes) < 1e-6,  # 增加阈值，减少除零风险
                             torch.sign(slopes) * 1e-6 + 1e-10,  # 确保不为零
                             slopes)
    avg_slopes = -1.0 / slopes_safe

    # [NaN 调试] 检查 avg_slopes 是否有NaN/Inf
    if torch.isnan(avg_slopes).any() or torch.isinf(avg_slopes).any():
        print(f"[cobb_angle_line_torch] Error: avg_slopes contains NaN or Inf!")
        raise ValueError("avg_slopes contains NaN or Inf")
    
    # Find max and min slopes
    max_slope = torch.max(avg_slopes)
    min_slope = torch.min(avg_slopes)
    max_idx = torch.argmax(avg_slopes)
    min_idx = torch.argmin(avg_slopes)
    
    # Determine upper and lower MT (Main Thoracic) based on curve direction
    if max_idx > min_idx:  # Left bend or right bend case
        lower_MT = max_idx
        upper_MT = min_idx
    else:
        upper_MT = max_idx  # Positive slope in upper spine
        lower_MT = min_idx  # Negative slope in lower spine
    
    # Calculate upper segment slopes
    upper_segment = avg_slopes[:upper_MT + 1]
    if len(upper_segment) > 0:
        upper_max_slope = torch.max(upper_segment)
        upper_max_idx = torch.argmax(upper_segment)
        upper_min_slope = torch.min(upper_segment)
        upper_min_idx = torch.argmin(upper_segment)
    else:
        upper_max_slope = avg_slopes[0]
        upper_min_slope = avg_slopes[0]
    
    # Calculate lower segment slopes
    lower_segment = avg_slopes[lower_MT:]
    if len(lower_segment) > 0:
        lower_max_slope = torch.max(lower_segment)
        lower_max_idx = torch.argmax(lower_segment) + lower_MT
        lower_min_slope = torch.min(lower_segment)
        lower_min_idx = torch.argmin(lower_segment) + lower_MT
    else:
        lower_max_slope = avg_slopes[-1]
        lower_min_slope = avg_slopes[-1]
    
    # Calculate COBB angles in degrees
    cobb_angles = torch.zeros(3, dtype=y_c.dtype, device=y_c.device)
    
    # Total COBB angle
    total_angle = torch.abs(
        torch.rad2deg(torch.atan(max_slope)) - 
        torch.rad2deg(torch.atan(min_slope))
    )
    
    # Upper COBB angle
    upper_angle = torch.abs(
        torch.rad2deg(torch.atan(upper_max_slope)) - 
        torch.rad2deg(torch.atan(upper_min_slope))
    )
    
    # Lower COBB angle  
    lower_angle = torch.abs(
        torch.rad2deg(torch.atan(lower_max_slope)) - 
        torch.rad2deg(torch.atan(lower_min_slope))
    )
    
    # 避免就地操作，使用torch.stack创建新张量
    cobb_angles = torch.stack([total_angle, upper_angle, lower_angle])
    
    return cobb_angles

## 计算SMAPE
def compute_smape(errors_dict):
    smape_per_angle = {}
    for angle_name, values in errors_dict.items():
        total_err = sum(v[0] for v in values)
        total_sum = sum(v[1] for v in values)
        smape = (total_err / total_sum) * 100
        smape_per_angle[angle_name] = smape
    return smape_per_angle

## 计算总体SMAPE
def compute_overall_smape(errors_dict):
    total_err = sum(v[0] for values in errors_dict.values() for v in values)
    total_sum = sum(v[1] for values in errors_dict.values() for v in values)
    return (total_err / total_sum) * 100