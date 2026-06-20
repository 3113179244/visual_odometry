#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <Eigen/Core>
#include <Eigen/Geometry>

class KeyFrame {
public:
    // 构造函数：传入关键帧的基础属性
    KeyFrame(unsigned long id, double timestamp, const Eigen::Isometry3d& Twc);
    ~KeyFrame() = default;

    // 获取该关键帧在世界坐标系下的绝对位姿
    Eigen::Isometry3d GetPose();

public:
    unsigned long mId;        // 关键帧专属 ID
    double mTimeStamp;        // 时间戳
    Eigen::Isometry3d mTwc;   // 相机到世界的绝对位姿 (用于在 Rviz2 里画历史轨迹或优化)
};

#endif // KEYFRAME_H