#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>

class Tracking {
public:
    Tracking();
    ~Tracking() = default;

    // 核心主入口空壳：只负责接收图像并返回一个初始位姿，不走任何算法
    Eigen::Isometry3d GrabImageStereo(const cv::Mat& imLeft, const cv::Mat& imRight, const double &timestamp);

private:
    bool mIsInitialized;        // 系统是否初始化成功
};

#endif // TRACKING_H