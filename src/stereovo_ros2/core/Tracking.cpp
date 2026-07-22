#include "core/Tracking.h"
#include "core/Map.h"
#include "core/FeatureDetector.h"
#include "core/KeyFrame.h"
#include "backend/Optimizer.h"
#include "utils/Parameters.h"
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include "core/LoopClosing.h"

// ==================== 构造函数 / 析构函数 ====================

Tracking::Tracking(std::shared_ptr<Map> pMap)
    : mpMap(pMap), mIsInitialized(true), mIsRunning(true), mNeedOptimize(false), 
      mNextKFId(0), mPrevTime(0.0), mbPauseRequested(false), mbPaused(false)
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
    std::string voc_path = "/home/wzj/visual_odometry/src/stereovo_ros2/Vocabulary/ORBvoc.txt";
    mpLoopClosing = std::make_shared<LoopClosing>(mpMap, voc_path);
    
    // 💡【新增】：将 Tracking 指针关联给 LoopClosing
    if (mpLoopClosing) {
        mpLoopClosing->SetTracking(this);
    }
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

// ==================== 外部接口与同步控制 ====================

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

// 💡【新增】：控制线程暂停与恢复实现
void Tracking::RequestPause()
{
    std::unique_lock<std::mutex> lock(mMutexPause);
    mbPauseRequested = true;
    mCondBuf.notify_all(); 
}

bool Tracking::IsPaused()
{
    std::unique_lock<std::mutex> lock(mMutexPause);
    return mbPaused;
}

void Tracking::Resume()
{
    std::unique_lock<std::mutex> lock(mMutexPause);
    mbPauseRequested = false;
    mbPaused = false;
    std::cout << "\033[32m[Tracking] 前端线程已成功恢复运行。\033[0m" << std::endl;
}

bool Tracking::CheckPause()
{
    std::unique_lock<std::mutex> lock(mMutexPause);
    if (mbPauseRequested)
    {
        mbPaused = true;
        return true;
    }
    return false;
}

// ==================== 前端跟踪主循环 ====================

void Tracking::TrackLoop()
{
    while (true)
    {
        // 1. 检查并挂起线程
        while (CheckPause()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        cv::Mat matLeft, matRight;
        double timestamp = 0.0;
        {
            std::unique_lock<std::mutex> lock(mMutexBuf);
            
            // 💡【修复】：Predicate 加上 mbPauseRequested 条件，允许因暂停请求而唤醒
            mCondBuf.wait(lock, [this] { 
                return !mIsRunning || 
                       (!mLeftBuf.empty() && !mRightBuf.empty() && !mTimeBuf.empty()) || 
                       mbPauseRequested; 
            });

            if (!mIsRunning)
                break;

            // 💡【修复】：如果是为了相应暂停请求被唤醒且缓冲区为空，跳过本次处理，回到循环顶部触发 CheckPause()
            if (mbPauseRequested && (mLeftBuf.empty() || mRightBuf.empty())) {
                continue;
            }

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

// ==================== 单帧立体处理流程 ====================

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

    if (mIsInitialized)
    {
        mpFeatureDetector->mvCurPts.clear();
        mpFeatureDetector->mvIds.clear();
        mpFeatureDetector->mvTrackCnt.clear();
        mmIDToMapPoint.clear();

        localPose = Eigen::Isometry3d::Identity();

        mpFeatureDetector->AddNewFeatures(grayLeft);
        mpFeatureDetector->mvPtsVel.resize(mpFeatureDetector->mvCurPts.size(), cv::Point2f(0.0f, 0.0f));
        mPrevTime = timestamp;
        mPrevImg = grayLeft.clone();

        mpFeatureDetector->TriangulateNewPoints(
            grayLeft, grayRight, localPose, mBodyTCam0, mBodyTCam1, mFx, mFy, mCx, mCy,
            mK1, mK2, mP1, mP2,
            mmIDToMapPoint, mpMap, true, vWorldPoints, imgTrack);

        std::map<int, StereoObs> initMeasurements;
        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)
        {
            int id = mpFeatureDetector->mvIds[i];
            StereoObs obs;
            obs.ptLeft = mpFeatureDetector->mvCurPts[i];
            if (i < mpFeatureDetector->mvRightPts.size() && mpFeatureDetector->stereoStatus[i])
            {
                obs.ptRight = mpFeatureDetector->mvRightPts[i];
                obs.hasRight = true;
            }
            initMeasurements[id] = obs;
        }

        auto pKF = std::make_shared<KeyFrame>(mNextKFId++, timestamp, localPose, initMeasurements, grayLeft);
        mpMap->AddKeyFrame(pKF);
        if (mpLoopClosing)
        {
            mpLoopClosing->InsertKeyFrame(pKF);
        }
        mvpPrevKFPointsMap.clear();
        for (const auto &pair : initMeasurements)
        {
            mvpPrevKFPointsMap[pair.first] = pair.second.ptLeft;
        }
        mIsInitialized = false;

        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)
        {
            double len = std::min(1.0, 1.0 * mpFeatureDetector->mvTrackCnt[i] / 20.0);
            cv::Scalar ptColor = cv::Scalar(255 * (1 - len), 0, 255 * len);
            cv::circle(imgTrack, mpFeatureDetector->mvCurPts[i], 2, ptColor, 2);
        }

        vKFPositions.push_back(localPose.translation());
        mpFeatureDetector->UpdatePreviousStatus(grayLeft);

        {
            std::unique_lock<std::mutex> lock(mMutexBackend);
            mCurrentPose = localPose;
        }
        return localPose;
    }

    mpFeatureDetector->TrackFeaturesLK(mPrevImg, grayLeft);
    bool pnp_succ = mpFeatureDetector->EstimatePosePnP(mmIDToMapPoint, mFx, mFy, mCx, mCy, mK1, mK2, mP1, mP2, localPose);

    mpFeatureDetector->SetMask(grayLeft.rows, grayLeft.cols);
    mpFeatureDetector->AddNewFeatures(grayLeft);

    bool isKeyFrame = false;
    if (pnp_succ)
    {
        isKeyFrame = NeedNewKeyFrame();
    }

    if (isKeyFrame || mIsInitialized)
    {
        mpFeatureDetector->TriangulateNewPoints(
            grayLeft, grayRight, localPose, mBodyTCam0, mBodyTCam1, mFx, mFy, mCx, mCy,
            mK1, mK2, mP1, mP2,
            mmIDToMapPoint, mpMap, isKeyFrame, vWorldPoints, imgTrack);
    }

    if (isKeyFrame)
    {
        std::map<int, StereoObs> currentMeasurements;
        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)
        {
            int id = mpFeatureDetector->mvIds[i];
            StereoObs obs;
            obs.ptLeft = mpFeatureDetector->mvCurPts[i];
            if (i < mpFeatureDetector->mvRightPts.size() && mpFeatureDetector->stereoStatus[i])
            {
                obs.ptRight = mpFeatureDetector->mvRightPts[i];
                obs.hasRight = true;
            }
            currentMeasurements[id] = obs;
        }

        auto pKF = std::make_shared<KeyFrame>(mNextKFId++, timestamp, localPose, currentMeasurements, grayLeft);
        mpMap->AddKeyFrame(pKF);

        if (mpLoopClosing)
        {
            mpLoopClosing->InsertKeyFrame(pKF);
        }

        mvpPrevKFPointsMap.clear();
        for (const auto &pair : currentMeasurements)
        {
            mvpPrevKFPointsMap[pair.first] = pair.second.ptLeft;
        }

        {
            std::unique_lock<std::mutex> lock(mMutexBackend);
            mNeedOptimize = true;
        }
        mCondBackend.notify_one();

        CullMapPoints();
        CullRedundantKeyFrames();
    }

    if (!pnp_succ)
    {
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

// ==================== 辅助与后端优化函数 ====================

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

void Tracking::CullMapPoints()
{
    std::unique_lock<std::mutex> lock(mpMap->GetMutex());
    auto &mspMapPoints = mpMap->GetMapPoints();

    const int MIN_OBSERVATIONS = 2;
    const int MAX_CONSECUTIVE_OUTLIER = 3;

    mnCullCounter++;
    if (mnCullCounter < 3)
        return;
    mnCullCounter = 0;

    auto it = mspMapPoints.begin();
    while (it != mspMapPoints.end())
    {
        auto pMP = *it;
        if (pMP->IsBad())
        {
            it = mspMapPoints.erase(it);
            continue;
        }

        bool shouldRemove = false;
        if (pMP->GetObservationCount() < MIN_OBSERVATIONS)
        {
            shouldRemove = true;
        }
        else if (pMP->GetConsecutiveOutlier() >= MAX_CONSECUTIVE_OUTLIER)
        {
            shouldRemove = true;
        }

        if (shouldRemove)
        {
            pMP->SetBad();
            it = mspMapPoints.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void Tracking::CullRedundantKeyFrames()
{
    std::unique_lock<std::mutex> lock(mpMap->GetMutex());
    auto &mspKeyFrames = mpMap->GetKeyFrames();
    auto &mspMapPoints = mpMap->GetMapPoints();

    if (mspKeyFrames.size() <= 20)
        return;

    std::unordered_map<int, std::shared_ptr<MapPoint>> mapIdToMapPoint;
    for (const auto &mp : mspMapPoints)
    {
        mapIdToMapPoint[mp->GetFeatureId()] = mp;
    }

    auto it = mspKeyFrames.begin();
    size_t max_search_bound = mspKeyFrames.size() / 3;
    size_t processed_count = 0;

    while (it != mspKeyFrames.end() && processed_count < max_search_bound)
    {
        auto pKF = *it;
        int total_features = pKF->mmObservations.size();
        if (total_features == 0)
        {
            it = mspKeyFrames.erase(it);
            continue;
        }

        int redundant_features_count = 0;
        for (const auto &obs : pKF->mmObservations)
        {
            int mapPointId = obs.first;
            auto itMp = mapIdToMapPoint.find(mapPointId);
            if (itMp != mapIdToMapPoint.end())
            {
                if (itMp->second->GetObservationCount() > 3)
                    redundant_features_count++;
            }
            else
            {
                redundant_features_count++;
            }
        }

        if (redundant_features_count > 0.90 * total_features)
        {
            it = mspKeyFrames.erase(it);
        }
        else
        {
            ++it;
            ++processed_count;
        }
    }
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

bool Tracking::IsBufEmpty()
{
    std::unique_lock<std::mutex> lock(mMutexBuf);
    return mLeftBuf.empty();
}