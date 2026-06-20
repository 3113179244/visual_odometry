#include "Tracking.h"
#include <iostream>

Tracking::Tracking() : mIsInitialized(false) {
    // 纯净空壳构造函数
}

Eigen::Isometry3d Tracking::GrabImageStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp) {
    if (!mIsInitialized) {
        std::cout << ">>> [Tracking 空壳] 接收到第一帧图像，系统初始化成功！" << std::endl;
        mIsInitialized = true;
    }

    // 打印当前接收到的时间戳，证明 ROS 2 的数据已经成功传导进了算法层
    std::cout << "[Tracking 空壳] 正在处理时间戳为 " << std::fixed << timestamp << " 的双目图像..." << std::endl;

    // 算法暂未填充，直接无脑返回一个单位矩阵（Identity）作为虚拟位姿
    return Eigen::Isometry3d::Identity();
}