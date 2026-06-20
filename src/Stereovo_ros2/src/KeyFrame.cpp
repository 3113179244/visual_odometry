#include "KeyFrame.h"

KeyFrame::KeyFrame(unsigned long id, double timestamp, const Eigen::Isometry3d& Twc)
    : mId(id), mTimeStamp(timestamp), mTwc(Twc) {
    // 构造函数直接初始化列表赋值
}

Eigen::Isometry3d KeyFrame::GetPose() {
    return mTwc;
}