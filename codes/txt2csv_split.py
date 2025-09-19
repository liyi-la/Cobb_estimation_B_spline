import os
import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def cls_polygons(txt):
    # 分割出多边形
    polygons = []
    with open(txt) as f:
        for line in f:  # 每一行
            _, *points = line.split()  # cls 代表类别，points 代表点 xy
            points = [float(x) for x in points]
            x = points[0]
            y = points[1]
            polygons.append([x, y])
    return polygons


def split_points_by_curve(points):
    # 将点转换为 numpy 数组
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # 定义拟合函数（这里使用二次函数，通过 y 拟合 x）
    def func(y, a, b, c, d, e, f):
        return a * y ** 5 + b * y ** 4 + c * y ** 3 + d * y ** 2 + e * y + f

    # 进行曲线拟合
    popt, _ = curve_fit(func, y, x)

    # 根据拟合曲线将点分为左右两边
    left_points = []
    right_points = []
    for point in points:
        x_val = point[0]
        y_val = point[1]
        x_fit = func(y_val, *popt)
        if x_val < x_fit:
            left_points.append(point)
        else:
            right_points.append(point)

    # 按 y 坐标从小到大排序
    left_points.sort(key=lambda p: p[1])
    right_points.sort(key=lambda p: p[1])

    return left_points, right_points, popt


def write_to_csv(csv_file, all_points):
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for points in all_points:
                left_points, right_points, popt = split_points_by_curve(points)
                left_x = [p[0] for p in left_points]
                left_y = [p[1] for p in left_points]
                right_x = [p[0] for p in right_points]
                right_y = [p[1] for p in right_points]

                # 交替合并左右点的 x 坐标
                merged_x = []
                max_len = max(len(left_x), len(right_x))
                for i in range(max_len):
                    if i < len(left_x):
                        merged_x.append(left_x[i])
                    if i < len(right_x):
                        merged_x.append(right_x[i])

                # 交替合并左右点的 y 坐标
                merged_y = []
                for i in range(max_len):
                    if i < len(left_y):
                        merged_y.append(left_y[i])
                    if i < len(right_y):
                        merged_y.append(right_y[i])

                row = merged_x + merged_y
                writer.writerow(row)
        print(f"已将所有 txt 文件的数据按要求写入 {csv_file}")
    except Exception as e:
        print(f"写入文件 {csv_file} 时出现错误: {e}")


txt_folder = r"D:\Project\Xiehe_Spinal_image_stitching\cobb\labeled_spine\txt\test"
csv_file = r"D:\Project\Xiehe_Spinal_image_stitching\cobb\labeled_spine\landmarks_test.csv"

# 获取 txt 文件列表
txt_list = os.listdir(txt_folder)
txt_list.sort()
all_points = []

for txt_name in txt_list:
    txt_path = os.path.join(txt_folder, txt_name)
    points = cls_polygons(txt_path)
    all_points.append(points)

write_to_csv(csv_file, all_points)

# # 绘图
# for i, points in enumerate(all_points):
#     left_points, right_points, popt = split_points_by_curve(points)
#     points = np.array(points)
#     x = points[:, 0]
#     y = points[:, 1]

#     # 定义拟合函数
#     def func(y, a, b, c, d, e, f):
#         return a * y ** 5 + b * y ** 4 + c * y ** 3 + d * y ** 2 + e * y + f

#     # 生成拟合曲线的 y 值
#     y_fit = np.linspace(min(y), max(y), 100)
#     x_fit = func(y_fit, *popt)

#     # 绘制点
#     left_x = [p[0] for p in left_points]
#     left_y = [p[1] for p in left_points]
#     right_x = [p[0] for p in right_points]
#     right_y = [p[1] for p in right_points]
#     plt.scatter(left_x, left_y, color='blue', label='Left Points')
#     plt.scatter(right_x, right_y, color='red', label='Right Points')

#     # 绘制拟合曲线
#     plt.plot(x_fit, y_fit, color='green', label='Fitted Curve')

#     plt.title(f'{txt_list[i]},{len(left_points)==len(right_points)}')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.legend()
#     plt.show()
#     if i == 56:
#         break