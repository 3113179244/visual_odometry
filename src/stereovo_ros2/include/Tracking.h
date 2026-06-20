#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include "Frame.h"
#include "Map.h"

class Tracking {
public:
    Tracking();
    ~Tracking() = default;

    // 核心主入口：被你的 ROS 2 节点调用
    Eigen::Isometry3d GrabImageStereo(const cv::Mat& imLeft, const cv::Mat& imRight, const double &timestamp);

protected:
    void Track();               // 前端主控制流
    void TrackLastFrame();      // 1. 稀疏光流追踪：上一帧左目 -> 当前帧左目
    void TrackRightFrame();     // 2. 双目视差匹配：当前帧左目 -> 当前帧右目
    void EstimatePosePnP();     // 3. 位姿求解：用已知的 3D-2D 点对求解 PnP
    void KeyFrameLoop();        // 4. 关键帧触发与地图点初始化/三角化

private:
    Frame mCurrentFrame;        // 当前帧
    Frame mLastFrame;           // 上一帧
    bool mIsInitialized;        // 系统是否初始化成功
    std::shared_ptr<Map> mpMap; // 地图指针
};

#endif // TRACKING_H