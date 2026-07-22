#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>
#include <memory>
#include <DBoW3/DBoW3.h>

struct StereoObs {
    cv::Point2f ptLeft;
    cv::Point2f ptRight;
    bool hasRight = false;
};

class KeyFrame
{
public:
    KeyFrame(unsigned long id, double timestamp, const Eigen::Isometry3d &Twc,
             const std::map<int, StereoObs> &measurements, const cv::Mat &imgLeft);

    ~KeyFrame() = default;

    Eigen::Isometry3d GetPose();
    void SetPose(const Eigen::Isometry3d &Twc_opt);

    // 计算 DBoW3 词包向量
    void ComputeBoW(std::shared_ptr<DBoW3::Vocabulary> pVoc);

public:
    unsigned long mId;
    double mTimeStamp;
    std::map<int, StereoObs> mmObservations;
    cv::Mat mImgLeft; // 左目图像，用于ORB描述子计算

    // DBoW3 数据结构
    std::vector<cv::KeyPoint> mvKeys;
    cv::Mat mDescriptors;
    DBoW3::BowVector mBowVec;
    DBoW3::FeatureVector mFeatVec;
    std::vector<int> mvKeyFeatureIds;
private:
    Eigen::Isometry3d mTwc;
    std::mutex mMutexPose;
};

#endif // KEYFRAME_H