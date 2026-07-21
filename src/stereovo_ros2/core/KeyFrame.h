#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>
#include <DBoW3/BowVector.h>
// 新增双目观测结构体
struct StereoObs {
    cv::Point2f ptLeft;    // 左目像素特征点
    cv::Point2f ptRight;   // 同一特征 ID 在右目的像素特征点
    bool hasRight = false; // 当前帧该点是否成功追到了右目
};

class KeyFrame
{
public:
    // 构造函数：输入变更为带有左右目特征的 StereoObs
    KeyFrame(unsigned long id, double timestamp, const Eigen::Isometry3d &Twc,
             const std::map<int, StereoObs> &measurements)
        : mId(id), mTimeStamp(timestamp), mTwc(Twc), mmObservations(measurements) {}

    ~KeyFrame() = default;

    // 供外部调用的线程安全接口
    Eigen::Isometry3d GetPose();
    void SetPose(const Eigen::Isometry3d &Twc_opt);

public:
    unsigned long mId;                         // 关键帧专属 ID
    double mTimeStamp;                         // 时间戳
    std::map<int, StereoObs> mmObservations;   // 双目 2D 像素观测
    cv::Mat mImgLeft, mImgRight; // 保存原始图像用于后端回环计算 ORB
    // ORB 特征与词袋
    std::vector<cv::KeyPoint> mvOrbKeysLeft;
    cv::Mat mDescriptorsLeft;
    DBoW3::BowVector mBowVec;
    // 对应左目 ORB 特征点的 3D 空间点 (相机系或世界系)
    std::vector<cv::Point3f> mvMapPoints3D; 

    // 回环边/图连接: <连接关键帧ID, 相对位姿 T_parent_child>
    std::map<unsigned long, Eigen::Isometry3d> mmLoopEdges;

private:
    Eigen::Isometry3d mTwc; // 相机到世界的绝对位姿 (T_w_c)
    std::mutex mMutexPose;  // 专用于保护位姿的互斥锁
};

#endif // KEYFRAME_H