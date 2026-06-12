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

        # 检查文件存在性
        if not os.path.exists(self.vo_path):
            raise FileNotFoundError(f"❌ 找不到 VO 轨迹文件: {self.vo_path}")
        if not os.path.exists(file_gt):
            raise FileNotFoundError(f"❌ 找不到 KITTI 真值文件: {file_gt}")

        # 读取数据
        my_data = np.loadtxt(self.vo_path, comments='#')
        gt_data = np.loadtxt(file_gt)

        # 截取最小帧数对齐
        min_frames = min(len(my_data), len(gt_data))
        return my_data[:min_frames], gt_data[:min_frames], seq_str, min_frames

    def _compute_metrics(self, my_data, gt_data):
        """内部方法：提取坐标并计算误差指标"""
        # 提取 VO 坐标 (假设你的格式: x y z)
        my_x, my_z = my_data[:, 0], my_data[:, 2]
        
        # 提取 GT 坐标 (KITTI 格式: 3x4矩阵展平)
        gt_x, gt_z = gt_data[:, 3], gt_data[:, 11]

        # 计算 ATE (Absolute Trajectory Error)
        errors = np.sqrt((my_x - gt_x)**2 + (my_z - gt_z)**2)
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(errors)

        return my_x, my_z, gt_x, gt_z, rmse, max_error

    def _plot_trajectory(self, my_x, my_z, gt_x, gt_z, seq_str):
        """内部方法：渲染绘图"""
        plt.figure(figsize=(11, 9))
        
        # 绘制轨迹
        plt.plot(gt_x, gt_z, 'k--', linewidth=2.5, label=f'GT (Seq {seq_str})', alpha=0.8)
        plt.plot(my_x, my_z, color='dodgerblue', linewidth=2.0, label='Stereo VO')
        
        # 标记关键点
        plt.plot(gt_x[0], gt_z[0], 'go', markersize=10, label='Start (0,0,0)')
        plt.plot(gt_x[-1], gt_z[-1], 'ks', markersize=8, label='GT End')
        plt.plot(my_x[-1], my_z[-1], 'ro', markersize=8, label='VO End')

        # 图表设置
        plt.title(f"Stereo VO vs KITTI Ground Truth (Sequence {seq_str})", fontsize=14, fontweight='bold')
        plt.xlabel("X (Left / Right) [meters]")
        plt.ylabel("Z (Forward) [meters]")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')
        plt.axis('equal')
        
        plt.show()

    def evaluate(self, sequence):
        """
        核心对外接口：执行一键评估
        :param sequence: 序列号，支持数字 6 或字符串 '06'
        """
        try:
            print(f"\n🚀 开始评估 Sequence {sequence} ...")
            
            # 1. 获取数据
            my_data, gt_data, seq_str, frames = self._load_data(sequence)
            print(f"📊 成功对齐时间步，共同对比帧数: {frames} 帧")

            # 2. 计算误差
            my_x, my_z, gt_x, gt_z, rmse, max_err = self._compute_metrics(my_data, gt_data)
            print("-" * 40)
            print(f"📉 RMSE (平均漂移): {rmse:.3f} m")
            print(f"🚨 Max Error (最大漂移): {max_err:.3f} m")
            print("-" * 40)

            # 3. 绘图显示
            print("🖥️  正在渲染对比曲线...")
            self._plot_trajectory(my_x, my_z, gt_x, gt_z, seq_str)

        except Exception as e:
            print(e)


if __name__ == '__main__':
    # 配置绝对路径
    GT_DIRECTORY = '/home/wzj/stereovo/data_odometry_poses/dataset/poses'
    
    # 实例化评估器
    evaluator = KittiEvaluator(gt_dir=GT_DIRECTORY, vo_path='trajectory.txt')

    # ==========================================
    # 🌟 极简调用方式：想换哪个序列，直接改这里的数字即可
    # ==========================================
    
    evaluator.evaluate(4)  # 直接传数字 4 即可，自动对应 04.txt
    
    # 如果你想通过命令行运行 (例如 python script.py --seq 06)
    # 也可以取消下方代码的注释：
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--seq", type=str, default="06")
    # args = parser.parse_args()
    # evaluator.evaluate(args.seq)