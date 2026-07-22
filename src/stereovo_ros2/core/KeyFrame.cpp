#include "core/KeyFrame.h"
#include <iostream>

KeyFrame::KeyFrame(unsigned long id, double timestamp, const Eigen::Isometry3d &Twc,
                   const std::map<int, StereoObs> &measurements, const cv::Mat &imgLeft)
    : mId(id), mTimeStamp(timestamp), mTwc(Twc), mmObservations(measurements)
{
    if (!imgLeft.empty())
    {
        mImgLeft = imgLeft.clone();

        mvKeys.clear();
        mvKeyFeatureIds.clear();
        mDescriptors = cv::Mat();

        const int border = 35;
        static cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);

        std::vector<cv::KeyPoint> kpsBatch;
        std::vector<int> featIdsBatch;

        for (const auto &obs : mmObservations)
        {
            cv::Point2f pt = obs.second.ptLeft;
            if (pt.x >= border && pt.x < mImgLeft.cols - border &&
                pt.y >= border && pt.y < mImgLeft.rows - border)
            {
                kpsBatch.emplace_back(pt, 31.0f);
                featIdsBatch.push_back(obs.first);
            }
        }

        if (!kpsBatch.empty())
        {
            cv::Mat descs;
            try {
                std::vector<cv::KeyPoint> kpsFiltered = kpsBatch;
                orb->compute(mImgLeft, kpsFiltered, descs);

                if (!descs.empty()) {
                    mvKeys = kpsFiltered;
                    mDescriptors = descs;
                    
                    // 💡【严格对齐】：确保只保留 compute 计算成功的 Feature ID[cite: 1]
                    mvKeyFeatureIds.clear();
                    for (const auto &kp : kpsFiltered) {
                        for (size_t i = 0; i < kpsBatch.size(); ++i) {
                            if (std::abs(kp.pt.x - kpsBatch[i].pt.x) < 1e-3 &&
                                std::abs(kp.pt.y - kpsBatch[i].pt.y) < 1e-3) {
                                mvKeyFeatureIds.push_back(featIdsBatch[i]);
                                break;
                            }
                        }
                    }
                }
            } catch (...) {}
        }
    }

    if (mDescriptors.empty() || mDescriptors.rows == 0)
    {
        mDescriptors = cv::Mat(0, 32, CV_8UC1);
    }
}

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

void KeyFrame::ComputeBoW(std::shared_ptr<DBoW3::Vocabulary> pVoc)
{
    if (mBowVec.empty() && pVoc && !mDescriptors.empty() && mDescriptors.rows > 0)
    {
        try
        {
            cv::Mat desc = mDescriptors;
            if (!desc.isContinuous()) {
                desc = desc.clone();
            }
            if (desc.type() != CV_8UC1) {
                desc.convertTo(desc, CV_8UC1);
            }

            pVoc->transform(desc, mBowVec);
        }
        catch (const std::exception &e)
        {
            std::cerr << "\033[31m[ComputeBoW Error] KF " << mId << " transform 异常: " << e.what() << "\033[0m" << std::endl;
            mBowVec.clear();
            mFeatVec.clear();
        }
        catch (...)
        {
            std::cerr << "\033[31m[ComputeBoW Error] KF " << mId << " transform 发生未知异常\033[0m" << std::endl;
            mBowVec.clear();
            mFeatVec.clear();
        }
    }
}