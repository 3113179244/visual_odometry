#include "system.h"
#include <iostream>
#include <opencv2/core/core.hpp>

int main() {
    // 强制限制 OpenCV 使用单线程，防止底层计算抢占引发时序问题
    cv::setNumThreads(1); 

    // 硬编码路径配置
    std::string calib_file = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/04/calib.txt";
    std::string path_left = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/04/image_0";
    std::string path_right = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/04/image_1";
    std::string traj_out = "/home/wzj/stereovo/trajectory.txt";

    // 一键启动系统
    System slam_system(calib_file);
    slam_system.Run(path_left, path_right, traj_out);

    return 0;
}