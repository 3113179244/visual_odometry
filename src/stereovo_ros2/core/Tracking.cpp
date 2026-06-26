#include "core/Tracking.h"
#include "core/Map.h"
#include "core/FeatureDetector.h"
#include "core/KeyFrame.h"
#include "backend/Optimizer.h"
#include "utils/Parameters.h"
#include <opencv2/core/eigen.hpp>
#include <iostream>

Tracking::Tracking(std::shared_ptr<Map> pMap)
    : mpMap(pMap), mIsInitialized(false), mIsRunning(true), mNeedOptimize(false), mNextKFId(0), mPrevTime(0.0)
{
    mFlowBack = (Parameters::FLOW_BACK != 0);
    mFx = Parameters::fx;
    mFy = Parameters::fy;
    mCx = Parameters::cx;
    mCy = Parameters::cy;
    mK1 = Parameters::k1;
    mK2 = Parameters::k2;
    mP1 = Parameters::p1;
    mP2 = Parameters::p2;
    mKeyframeParallax = Parameters::KEYFRAME_PARALLAX;
    mBodyTCam0 = Parameters::body_T_cam0;
    mBodyTCam1 = Parameters::body_T_cam1;

    mCurrentPose = Eigen::Isometry3d::Identity();

    mpFeatureDetector = std::make_unique<FeatureDetector>(Parameters::MAX_CNT, Parameters::MIN_DIST, mFlowBack);

    mTrackThread = std::thread(&Tracking::TrackLoop, this);
    mBackendThread = std::thread(&Tracking::BackendLoop, this);
}

Tracking::~Tracking()
{
    {
        std::unique_lock<std::mutex> lock1(mMutexBuf);
        std::unique_lock<std::mutex> lock2(mMutexBackend);
        mIsRunning = false;
    }
    mCondBuf.notify_all();
    mCondBackend.notify_all();
    if (mTrackThread.joinable())
        mTrackThread.join();
    if (mBackendThread.joinable())
        mBackendThread.join();
}

void Tracking::RegisterCallback(RenderCallback cb)
{
    mRenderCb = cb;
}

void Tracking::FeedStereoImages(const cv::Mat &imLeft, const cv::Mat &imRight, double timestamp)
{
    std::unique_lock<std::mutex> lock(mMutexBuf);
    mLeftBuf.push(imLeft.clone());
    mRightBuf.push(imRight.clone());
    mTimeBuf.push(timestamp);
    mCondBuf.notify_one();
}

void Tracking::TrackLoop()
{
    while (true)
    {
        cv::Mat matLeft, matRight;
        double timestamp = 0.0;
        {
            std::unique_lock<std::mutex> lock(mMutexBuf);
            mCondBuf.wait(lock, [this]
                          { return !mIsRunning || (!mLeftBuf.empty() && !mRightBuf.empty() && !mTimeBuf.empty()); });

            if (!mIsRunning)
                break;

            matLeft = mLeftBuf.front();
            mLeftBuf.pop();
            matRight = mRightBuf.front();
            mRightBuf.pop();
            timestamp = mTimeBuf.front();
            mTimeBuf.pop();
        }

        cv::Mat feat_img;
        std::vector<Eigen::Vector3d> vWorldPoints;
        std::vector<Eigen::Vector3d> vKFPositions;

        Eigen::Isometry3d Tcw = ProcessStereo(matLeft, matRight, timestamp, feat_img, vWorldPoints, vKFPositions);

        if (mRenderCb && !feat_img.empty())
        {
            mRenderCb(timestamp, feat_img, vWorldPoints, vKFPositions, Tcw,
                      mpFeatureDetector->mvCurPts, mpFeatureDetector->mvIds, mpFeatureDetector->mvPtsVel);
        }
    }
}

Eigen::Isometry3d Tracking::ProcessStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                                          const double &timestamp, cv::Mat &imgTrack,
                                          std::vector<Eigen::Vector3d> &vWorldPoints,
                                          std::vector<Eigen::Vector3d> &vKFPositions)
{
    cv::Mat grayLeft = imLeft;
    if (imLeft.channels() == 3)
        cv::cvtColor(imLeft, grayLeft, cv::COLOR_BGR2GRAY);
    cv::Mat grayRight = imRight;
    if (imRight.channels() == 3)
        cv::cvtColor(imRight, grayRight, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(grayLeft, grayLeft);
    clahe->apply(grayRight, grayRight);

    cv::Mat imgLeftBGR, imgRightBGR;
    cv::cvtColor(grayLeft, imgLeftBGR, cv::COLOR_GRAY2BGR);
    cv::cvtColor(grayRight, imgRightBGR, cv::COLOR_GRAY2BGR);
    cv::hconcat(imgLeftBGR, imgRightBGR, imgTrack);

    vWorldPoints.clear();
    vKFPositions.clear();

    Eigen::Isometry3d localPose;
    {
        std::unique_lock<std::mutex> lock(mMutexBackend);
        localPose = mCurrentPose;
    }

    // --- A. 第一帧系统初始化 ---
    if (!mIsInitialized)
    {
        std::cout << ">>> [SLAM前端] 第一帧初始化，固化为初始关键帧。" << std::endl;
        mpFeatureDetector->mvCurPts.clear();
        mpFeatureDetector->mvIds.clear();
        mpFeatureDetector->mvTrackCnt.clear();
        mmIDToMapPoint.clear();
        localPose = Eigen::Isometry3d::Identity();
        mpFeatureDetector->AddNewFeatures(grayLeft);
        mpFeatureDetector->mvPtsVel.resize(mpFeatureDetector->mvCurPts.size(), cv::Point2f(0.0f, 0.0f));
        mPrevTime = timestamp;
        mPrevImg = grayLeft.clone();

        std::map<int, cv::Point2f> initMeasurements;
        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)
        {
            initMeasurements[mpFeatureDetector->mvIds[i]] = mpFeatureDetector->mvCurPts[i];
        }

        auto pKF = std::make_shared<KeyFrame>(mNextKFId++, timestamp, localPose.inverse(), initMeasurements);
        mpMap->AddKeyFrame(pKF);

        mvpPrevKFPointsMap = initMeasurements;
        mIsInitialized = true;

        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)
        {
            double len = std::min(1.0, 1.0 * mpFeatureDetector->mvTrackCnt[i] / 20.0);
            cv::Scalar ptColor = cv::Scalar(255 * (1 - len), 0, 255 * len);
            cv::circle(imgTrack, mpFeatureDetector->mvCurPts[i], 2, ptColor, 2);
        }

        vKFPositions.push_back(localPose.inverse().translation());
        mpFeatureDetector->UpdatePreviousStatus(grayLeft);

        {
            std::unique_lock<std::mutex> lock(mMutexBackend);
            mCurrentPose = localPose;
        }
        return localPose;
    }

    // --- B. 正常帧间状态解算 ---
    // 1. 稀疏光流帧间追踪
    mpFeatureDetector->TrackFeaturesLK(mPrevImg, grayLeft);

    // 2. PnP 求解当前帧初估位姿
    bool pnp_succ = mpFeatureDetector->EstimatePosePnP(mmIDToMapPoint, mFx, mFy, mCx, mCy, mK1, mK2, mP1, mP2, localPose);

    // 3. 均匀分布特征圈刷新与特征点提取
    mpFeatureDetector->SetMask(grayLeft.rows, grayLeft.cols);
    mpFeatureDetector->AddNewFeatures(grayLeft);

    // 4. 关键帧判定
    bool isKeyFrame = false;
    if (pnp_succ)
    {
        isKeyFrame = NeedNewKeyFrame();
    }

    // 5. 立体匹配与自适应双目三角化
    mpFeatureDetector->TriangulateNewPoints(
        grayLeft, grayRight, localPose, mBodyTCam0, mBodyTCam1, mFx, mFy, mCx, mCy,
        mK1, mK2, mP1, mP2,
        mmIDToMapPoint, mpMap, isKeyFrame, vWorldPoints, imgTrack);

    // 6. 固化新关键帧并触发后端 BA
    if (isKeyFrame)
    {
        std::map<int, cv::Point2f> currentMeasurements;

        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)
        {
            currentMeasurements[mpFeatureDetector->mvIds[i]] = mpFeatureDetector->mvCurPts[i];
        }

        auto pKF = std::make_shared<KeyFrame>(mNextKFId++, timestamp, localPose.inverse(), currentMeasurements);
        mpMap->AddKeyFrame(pKF);

        mvpPrevKFPointsMap = currentMeasurements;

        {
            std::unique_lock<std::mutex> lock(mMutexBackend);
            mNeedOptimize = true;
        }
        mCondBackend.notify_one();
        mpMap->CullMapPoints();
    }

    if (!pnp_succ)
    {
        std::cout << ">>> [SLAM前端] 提示：PnP 暂无足够追踪点，已通过双目三角化构建结构。" << std::endl;
        for (const auto &pair : mmIDToMapPoint)
        {
            vWorldPoints.push_back(pair.second->GetWorldPos());
        }
    }

    auto allKFs = mpMap->GetAllKeyFrames();
    for (const auto &kf : allKFs)
    {
        vKFPositions.push_back(kf->GetPose().translation());
    }

    for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); i++)
    {
        double len = std::min(1.0, 1.0 * mpFeatureDetector->mvTrackCnt[i] / 20.0);
        cv::Scalar ptColor = cv::Scalar(255 * (1 - len), 0, 255 * len);
        cv::circle(imgTrack, mpFeatureDetector->mvCurPts[i], 2, ptColor, 2);
    }

    mpFeatureDetector->mvPtsVel.assign(mpFeatureDetector->mvCurPts.size(), cv::Point2f(0.0f, 0.0f));
    double dt = timestamp - mPrevTime;
    if (dt > 1e-5)
    {
        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)
        {
            int id = mpFeatureDetector->mvIds[i];
            auto it = mpFeatureDetector->mInversePrevPtsMap.find(id);
            if (it != mpFeatureDetector->mInversePrevPtsMap.end())
            {
                cv::Point2f pt_prev = it->second;
                cv::Point2f pt_curr = mpFeatureDetector->mvCurPts[i];
                mpFeatureDetector->mvPtsVel[i].x = (pt_curr.x - pt_prev.x) / dt;
                mpFeatureDetector->mvPtsVel[i].y = (pt_curr.y - pt_prev.y) / dt;
            }
        }
    }
    mPrevTime = timestamp;
    mPrevImg = grayLeft.clone();
    mpFeatureDetector->UpdatePreviousStatus(grayLeft);

    {
        std::unique_lock<std::mutex> lock(mMutexBackend);
        mCurrentPose = localPose;
    }
    return localPose;
}

bool Tracking::NeedNewKeyFrame()
{
    if (mpFeatureDetector->mvCurPts.size() < 20)
        return true;
    double total_parallax = 0.0;
    int common_tracked_pts_cnt = 0;

    for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)
    {
        int id = mpFeatureDetector->mvIds[i];
        auto it = mvpPrevKFPointsMap.find(id);
        if (it != mvpPrevKFPointsMap.end())
        {
            total_parallax += mpFeatureDetector->Distance(mpFeatureDetector->mvCurPts[i], it->second);
            common_tracked_pts_cnt++;
        }
    }
    if (common_tracked_pts_cnt == 0)
        return true;
    double average_parallax = total_parallax / common_tracked_pts_cnt;

    return average_parallax >= mKeyframeParallax;
}

void Tracking::BackendLoop()
{
    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(mMutexBackend);
            mCondBackend.wait(lock, [this]
                              { return mNeedOptimize || !mIsRunning; });

            if (!mIsRunning)
                break;
            mNeedOptimize = false;
        }
        Optimizer::LocalBundleAdjustment(mpMap, 10);
    }
}