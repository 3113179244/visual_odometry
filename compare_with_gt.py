import matplotlib.pyplot as plt
import numpy as np
import os

def compare_vo_with_ground_truth():
    file_vo = 'trajectory.txt'
    file_gt = '04.txt'

    # 1. 检查文件
    if not os.path.exists(file_vo) or not os.path.exists(file_gt):
        print("❌ 错误：请确保 trajectory.txt 和 07.txt 同时存在于当前目录下！")
        return

    # 2. 读取数据
    my_data = np.loadtxt(file_vo, comments='#')
    gt_data = np.loadtxt(file_gt)

    # 3. 截取相同帧数（对齐时间轴）
    # 你的算法可能因为熔断丢弃、或者只跑了数据集的一部分，需要保证两者数组长度完全一致
    min_frames = min(len(my_data), len(gt_data))
    my_data = my_data[:min_frames]
    gt_data = gt_data[:min_frames]

    print(f"================ 位姿对齐报告 ================")
    print(f"📊 成功对齐时间步，共同对比帧数: {min_frames} 帧")

    # 4. 提取坐标
    # 你的里程计坐标 (X, Y, Z)
    my_x = my_data[:, 0]
    my_z = my_data[:, 2] # 进深

    # KITTI 真值坐标 (从 3x4 变换矩阵中提取：第3列为 X，第11列为 Z)
    gt_x = gt_data[:, 3]
    gt_z = gt_data[:, 11]

    # 5. 【硬核算法指标】计算绝对轨迹误差 (Absolute Trajectory Error, ATE)
    # 计算每一帧你的估计值和真实值之间的欧氏距离
    errors = np.sqrt((my_x - gt_x)**2 + (my_z - gt_z)**2)
    rmse_error = np.sqrt(np.mean(errors**2))
    max_error = np.max(errors)
    
    print(f"📉 平均绝对轨迹漂移误差 (RMSE): {rmse_error:.3f} 米")
    print(f"🚨 最大漂移误差 (Max Error): {max_error:.3f} 米")
    print(f"==============================================")

    # 6. 开始高精度画图
    plt.figure(figsize=(11, 9))
    
    # 绘制官方真值轨迹（黑色虚线）
    plt.plot(gt_x, gt_z, 'k--', linewidth=2.5, label='KITTI Ground Truth (07.txt)', alpha=0.8)
    
    # 绘制你手写的里程计轨迹（蓝色实线）
    plt.plot(my_x, my_z, color='dodgerblue', linewidth=2.0, label='My Stereo VO (trajectory.txt)')
    
    # 特别标记起点
    plt.plot(gt_x[0], gt_z[0], 'go', markersize=10, label='Start Point (0, 0, 0)')
    # 标记两者的终点
    plt.plot(gt_x[-1], gt_z[-1], 'ks', markersize=8, label='GT End')
    plt.plot(my_x[-1], my_z[-1], 'ro', markersize=8, label='Your VO End')

    # 图表细节装饰
    plt.title("Stereo VO vs KITTI Ground Truth (Sequence 07)", fontsize=14, fontweight='bold')
    plt.xlabel("X (Left / Right) [meters]", fontsize=12)
    plt.ylabel("Z (Forward) [meters]", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11)
    
    # 🌟 再次强调：必须等比例，否则无法看出真正的漂移趋势
    plt.axis('equal')
    
    print("🖥️  正在渲染对比曲线，请查看弹窗...")
    plt.show()

if __name__ == '__main__':
    compare_vo_with_ground_truth()
