import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

class KittiEvaluator:
    def __init__(self, gt_dir, vo_path='trajectory.txt'):
        """
        初始化评估器
        :param gt_dir: KITTI 真值文件存放的根目录
        :param vo_path: 你的里程计生成的轨迹文件路径
        """
        self.gt_dir = gt_dir
        self.vo_path = vo_path

    def _load_data(self, sequence):
        """内部方法：加载并对齐数据"""
        seq_str = f"{int(sequence):02d}"
        file_gt = os.path.join(self.gt_dir, f"{seq_str}.txt")

        if not os.path.exists(self.vo_path):
            raise FileNotFoundError(f"❌ 找不到 VO 轨迹文件: {self.vo_path}")
        if not os.path.exists(file_gt):
            raise FileNotFoundError(f"❌ 找不到 KITTI 真值文件: {file_gt}")

        my_data = np.loadtxt(self.vo_path, comments='#')
        gt_data = np.loadtxt(file_gt)

        min_frames = min(len(my_data), len(gt_data))
        return my_data[:min_frames], gt_data[:min_frames], seq_str, min_frames

    def _compute_metrics(self, my_data, gt_data):
        """内部方法：提取坐标并执行标准的 Umeyama 仿射对齐，消除坐标轴方向不一致引起的虚假旋转"""
        # 1. 提取原始坐标
        my_x, my_z = my_data[:, 0], my_data[:, 2]
        gt_x, gt_z = gt_data[:, 3], gt_data[:, 11]

        # 2. 构造 2D 坐标矩阵用于对齐
        X_my = np.vstack((my_x, my_z)) # 2 x N
        Y_gt = np.vstack((gt_x, gt_z)) # 2 x N

        # 3. 严格执行工业级绝对轨迹旋转对齐 (Umeyama 算法)
        mu_X = np.mean(X_my, axis=1, keepdims=True)
        mu_Y = np.mean(Y_gt, axis=1, keepdims=True)
        
        X_centered = X_my - mu_X
        Y_centered = Y_gt - mu_Y
        
        H = X_centered @ Y_centered.T
        U, S, Vt = np.linalg.svd(H)
        
        # 计算旋转矩阵 R
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T
            
        # 计算平移向量 t
        t = mu_Y - R @ mu_X
        
        # 4. 得到完全消除起始方向偏差后的真实 VO 轨迹坐标
        X_aligned = R @ X_my + t
        my_x_aligned = X_aligned[0, :]
        my_z_aligned = X_aligned[1, :]

        # 5. 计算真实的 ATE (Absolute Trajectory Error)
        errors = np.sqrt((my_x_aligned - gt_x)**2 + (my_z_aligned - gt_z)**2)
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(errors)

        return my_x_aligned, my_z_aligned, gt_x, gt_z, rmse, max_error

    def _plot_trajectory(self, my_x, my_z, gt_x, gt_z, seq_str):
        """内部方法：渲染绘图"""
        plt.figure(figsize=(11, 9))
        
        # 绘制轨迹
        plt.plot(gt_x, gt_z, 'k--', linewidth=2.5, label=f'GT (Seq {seq_str})', alpha=0.8)
        plt.plot(my_x, my_z, color='dodgerblue', linewidth=2.0, label='Stereo VO (Aligned)')
        
        # 标记关键点
        plt.plot(gt_x[0], gt_z[0], 'go', markersize=10, label='Start (0,0,0)')
        plt.plot(gt_x[-1], gt_z[-1], 'ks', markersize=8, label='GT End')
        plt.plot(my_x[-1], my_z[-1], 'ro', markersize=8, label='VO End')

        plt.title(f"Stereo VO vs KITTI Ground Truth (Sequence {seq_str})", fontsize=14, fontweight='bold')
        plt.xlabel("X (Left / Right) [meters]")
        plt.ylabel("Z (Forward) [meters]")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')
        plt.axis('equal')
        
        plt.show()

    def evaluate(self, sequence):
        """核心对外接口：执行一键评估"""
        try:
            print(f"\n🚀 开始评估 Sequence {sequence} ...")
            
            my_data, gt_data, seq_str, frames = self._load_data(sequence)
            print(f"📊 成功对齐时间步，共同对比帧数: {frames} 帧")

            my_x, my_z, gt_x, gt_z, rmse, max_err = self._compute_metrics(my_data, gt_data)
            print("-" * 40)
            print(f"📉 RMSE (平均漂移): {rmse:.3f} m")
            print(f"🚨 Max Error (最大漂移): {max_err:.3f} m")
            print("-" * 40)

            print("🖥️  正在渲染对比曲线...")
            self._plot_trajectory(my_x, my_z, gt_x, gt_z, seq_str)

        except Exception as e:
            print(e)


if __name__ == '__main__':
    GT_DIRECTORY = '/home/wzj/stereovo/data_odometry_poses/dataset/poses'
    evaluator = KittiEvaluator(gt_dir=GT_DIRECTORY, vo_path='trajectory.txt')
    
    # 🌟 测试哪个序列，直接在这里修改数字即可（支持 4 或 7）
    evaluator.evaluate(4)