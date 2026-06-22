#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <opencv2/opencv.hpp>
#include <mutex>

class KeyFrame
{
public:
    // 构造函数
    KeyFrame(unsigned long id, double timestamp, const Eigen::Isometry3d &Twc,
             const std::map<int, cv::Point2f> &measurements)
        : mId(id), mTimeStamp(timestamp), mTwc(Twc), mmObservations(measurements) {}

    ~KeyFrame() = default;

    // 供外部调用的线程安全接口
    Eigen::Isometry3d GetPose();
    void SetPose(const Eigen::Isometry3d &Twc_opt);

public:
    // 【核心关键】以下三个变量必须保持在第一个 private: 之前，供 Optimizer.cpp 直接读写
    unsigned long mId;                         // 关键帧专属 ID
    double mTimeStamp;                         // 时间戳
    std::map<int, cv::Point2f> mmObservations; // 2D 像素观测

private:
    // 只有位姿矩阵和对应的互斥锁需要放入私有区域，强制通过接口加锁访问
    Eigen::Isometry3d mTwc; // 相机到世界的绝对位姿 (T_w_c)
    std::mutex mMutexPose;  // 专用于保护位姿的互斥锁
};

#endif // KEYFRAME_H