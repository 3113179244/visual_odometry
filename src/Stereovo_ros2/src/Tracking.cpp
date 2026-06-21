#include "Tracking.h"
#include "Parameters.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "FeatureDetector.h"
#include <iostream>

Tracking::Tracking(std::shared_ptr<Map> pMap)
    : mIsInitialized(false), mpMap(pMap), mNextKFId(0), mIsRunning(true) {
    mCurrentPose = Eigen::Isometry3d::Identity();
    mFx = Parameters::fx; mFy = Parameters::fy; mCx = Parameters::cx; mCy = Parameters::cy;
    mK1 = Parameters::k1; mK2 = Parameters::k2; mP1 = Parameters::p1; mP2 = Parameters::p2;
    mKeyframeParallax = Parameters::KEYFRAME_PARALLAX;
    mBodyTCam0 = Parameters::body_T_cam0; mBodyTCam1 = Parameters::body_T_cam1;

    // 构造独立的特征管理器
    mpFeatureDetector = std::make_unique<FeatureDetector>(Parameters::MAX_CNT, Parameters::MIN_DIST, Parameters::FLOW_BACK != 0);
    mTrackThread = std::thread(&Tracking::TrackLoop, this);
}

Tracking::~Tracking() {
    { std::unique_lock<std::mutex> lock(mMutexBuf); mIsRunning = false; }
    mCondBuf.notify_all(); if (mTrackThread.joinable()) mTrackThread.join();
}

void Tracking::RegisterCallback(RenderCallback cb) { mRenderCb = cb; }

void Tracking::FeedStereoImages(const cv::Mat &imLeft, const cv::Mat &imRight, double timestamp) {
    std::unique_lock<std::mutex> lock(mMutexBuf);
    mLeftBuf.push(imLeft.clone()); mRightBuf.push(imRight.clone()); mTimeBuf.push(timestamp);
    mCondBuf.notify_one();
}

void Tracking::TrackLoop() {
    while (true) {
        cv::Mat matLeft, matRight; double timestamp = 0.0;
        {
            std::unique_lock<std::mutex> lock(mMutexBuf);
            mCondBuf.wait(lock, [this](){ return !mIsRunning || (!mLeftBuf.empty() && !mRightBuf.empty() && !mTimeBuf.empty()); });
            if (!mIsRunning) break;
            matLeft = mLeftBuf.front(); mLeftBuf.pop(); matRight = mRightBuf.front(); mRightBuf.pop(); timestamp = mTimeBuf.front(); mTimeBuf.pop();
        }
        cv::Mat feat_img; std::vector<Eigen::Vector3d> vWorldPoints; std::vector<Eigen::Vector3d> vKFPositions;
        Eigen::Isometry3d Tcw = ProcessStereo(matLeft, matRight, timestamp, feat_img, vWorldPoints, vKFPositions);
        if (mRenderCb && !feat_img.empty()) mRenderCb(timestamp, feat_img, vWorldPoints, vKFPositions, Tcw, mpFeatureDetector->mvCurPts, mpFeatureDetector->mvIds);
    }
}

Eigen::Isometry3d Tracking::ProcessStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, cv::Mat &imgTrack, std::vector<Eigen::Vector3d> &vWorldPoints, std::vector<Eigen::Vector3d> &vKFPositions) {
    cv::Mat grayLeft = imLeft; if (imLeft.channels() == 3) cv::cvtColor(imLeft, grayLeft, cv::COLOR_BGR2GRAY);
    cv::Mat grayRight = imRight; if (imRight.channels() == 3) cv::cvtColor(imRight, grayRight, cv::COLOR_BGR2GRAY);
    cv::cvtColor(grayLeft, imgTrack, cv::COLOR_GRAY2BGR); vWorldPoints.clear(); vKFPositions.clear();

    // 1. 第一帧初始化
    if (!mIsInitialized) {
        std::cout << ">>> [SLAM前端] 第一帧初始化，固化为初始关键帧。" << std::endl;
        mmIDToMapPoint.clear(); mCurrentPose = Eigen::Isometry3d::Identity();
        mpFeatureDetector->mvCurPts.clear(); mpFeatureDetector->mvIds.clear(); mpFeatureDetector->mvTrackCnt.clear();
        
        mpFeatureDetector->AddNewFeatures(grayLeft);
        mPrevImg = grayLeft.clone();
        
        std::map<int, cv::Point2f> initMeasurements;
        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i) initMeasurements[mpFeatureDetector->mvIds[i]] = mpFeatureDetector->mvCurPts[i];
        auto pKF = std::make_shared<KeyFrame>(mNextKFId++, timestamp, mCurrentPose.inverse(), initMeasurements);
        mpMap->AddKeyFrame(pKF);

        mvpPrevKFPointsMap = initMeasurements;
        mIsInitialized = true;
        
        for (const auto &pt : mpFeatureDetector->mvCurPts) cv::circle(imgTrack, pt, 2, cv::Scalar(0, 0, 255), 2);
        vKFPositions.push_back(mCurrentPose.inverse().translation());
        mpFeatureDetector->UpdatePreviousStatus(grayLeft);
        return mCurrentPose;
    }

    // 2. 光流帧间追踪 
    mpFeatureDetector->TrackFeaturesLK(mPrevImg, grayLeft);

    // 3. PnP 求解当前帧 Pose
    mpFeatureDetector->EstimatePosePnP(mmIDToMapPoint, mFx, mFy, mCx, mCy, mK1, mK2, mP1, mP2, mCurrentPose);

    // 4. 自适应 Mask 补盲提取新特征点
    mpFeatureDetector->SetMask(grayLeft.rows, grayLeft.cols);
    mpFeatureDetector->AddNewFeatures(grayLeft);

    bool isKeyFrame = NeedNewKeyFrame();

    // 5. 传递给提取器完成双目立体匹配和三角化
    mpFeatureDetector->TriangulateNewPoints(grayLeft, grayRight, mCurrentPose, mBodyTCam0, mBodyTCam1, mFx, mFy, mCx, mCy, mmIDToMapPoint, mpMap, isKeyFrame, vWorldPoints, imgTrack);

    // 6. 固化关键帧数据库
    if (isKeyFrame) {
        std::map<int, cv::Point2f> currentMeasurements;
        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i) currentMeasurements[mpFeatureDetector->mvIds[i]] = mpFeatureDetector->mvCurPts[i];
        auto pKF = std::make_shared<KeyFrame>(mNextKFId++, timestamp, mCurrentPose.inverse(), currentMeasurements);
        mpMap->AddKeyFrame(pKF);

        mvpPrevKFPointsMap = currentMeasurements;
        for (const auto &id : mpFeatureDetector->mvIds) {
            auto it = mmIDToMapPoint.find(id);
            if (it != mmIDToMapPoint.end()) mpMap->AddMapPoint(it->second);
        }
    }

    auto allKFs = mpMap->GetAllKeyFrames();
    for (const auto &kf : allKFs) vKFPositions.push_back(kf->GetPose().translation());
    std::cout << "[SLAM地图管理] 全局关键帧总数: " << allKFs.size() << " | 全局地图路标点数: " << mpMap->GetMapPointsSize() << std::endl;

    // 7. 画特征轨迹并更新状态缓存
    for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); i++) {
        double len = std::min(1.0, 1.0 * mpFeatureDetector->mvTrackCnt[i] / 20.0);
        cv::circle(imgTrack, mpFeatureDetector->mvCurPts[i], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        auto it = mpFeatureDetector->mInversePrevPtsMap.find(mpFeatureDetector->mvIds[i]);
        if (it != mpFeatureDetector->mInversePrevPtsMap.end()) cv::arrowedLine(imgTrack, it->second, mpFeatureDetector->mvCurPts[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }

    mPrevImg = grayLeft.clone();
    mpFeatureDetector->UpdatePreviousStatus(grayLeft);
    return mCurrentPose;
}

bool Tracking::NeedNewKeyFrame() {
    if (mpFeatureDetector->mvCurPts.size() < 20) return true;
    double total_parallax = 0.0; int common_tracked_pts_cnt = 0;
    for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i) {
        int id = mpFeatureDetector->mvIds[i];
        auto it = mvpPrevKFPointsMap.find(id);
        if (it != mvpPrevKFPointsMap.end()) {
            double dx = mpFeatureDetector->mvCurPts[i].x - it->second.x;
            double dy = mpFeatureDetector->mvCurPts[i].y - it->second.y;
            total_parallax += std::sqrt(dx * dx + dy * dy);
            common_tracked_pts_cnt++;
        }
    }
    if (common_tracked_pts_cnt == 0) return true;
    return (total_parallax / common_tracked_pts_cnt) >= mKeyframeParallax;
}