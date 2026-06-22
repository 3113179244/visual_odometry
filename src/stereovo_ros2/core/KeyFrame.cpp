#include "KeyFrame.h"

Eigen::Isometry3d KeyFrame::GetPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTwc;
}

void KeyFrame::SetPose(const Eigen::Isometry3d &Twc_opt)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mTwc = Twc_opt;
}