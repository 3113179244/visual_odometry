#include "KeyFrame.h"

// 如果你在头文件中已经完全实现了构造函数，这里可以不写构造函数体
Eigen::Isometry3d KeyFrame::GetPose() {
    return mTwc;
}