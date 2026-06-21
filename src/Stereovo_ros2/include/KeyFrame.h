#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <opencv2/opencv.hpp>

class KeyFrame {
public:
    // 修改构造函数：允许传入当前帧的特征观测数据
    KeyFrame(unsigned long id, double timestamp, const Eigen::Isometry3d& Twc,
             const std::map<int, cv::Point2f>& measurements)
        : mId(id), mTimeStamp(timestamp), mTwc(Twc), mmObservations(measurements) {}
        
    ~KeyFrame() = default;

    // 获取该关键帧在世界坐标系下的绝对位姿
    Eigen::Isometry3d GetPose();

public:
    unsigned long mId;        // 关键帧专属 ID
    double mTimeStamp;        // 时间戳
    Eigen::Isometry3d mTwc;   // 相机到世界的绝对位姿 (T_w_c)

    // 新增：存储当前关键帧的观测。Key: 特征点全局ID, Value: 2D像素坐标(u, v)
    std::map<int, cv::Point2f> mmObservations; 
};

#endif // KEYFRAME_H