#include "core/Tracking.h"
#include "core/Map.h"
#include "core/FeatureDetector.h"
#include "core/KeyFrame.h"
#include "backend/Optimizer.h"
#include "utils/Parameters.h"
#include <opencv2/core/eigen.hpp>
#include <iostream>

/**
 * @file Tracking.cpp
 * @brief 前端视觉里程计核心类实现
 *
 * 负责：
 *   - 接收双目图像，执行光流跟踪、PnP位姿估计
 *   - 管理关键帧的创建与地图点的三角化
 *   - 触发后端局部BA优化
 *   - 提供可视化渲染回调
 *
 * 主要流程见 ProcessStereo() 函数。
 */

// ==================== 构造函数 / 析构函数 ====================

Tracking::Tracking(std::shared_ptr<Map> pMap)
    : mpMap(pMap), mIsInitialized(false), mIsRunning(true), mNeedOptimize(false), mNextKFId(0), mPrevTime(0.0)
{
    // 从全局参数中读取配置
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

    // 启动两个独立线程：前端跟踪循环 和 后端优化循环
    mTrackThread = std::thread(&Tracking::TrackLoop, this);
    mBackendThread = std::thread(&Tracking::BackendLoop, this);
}

Tracking::~Tracking()
{
    // 安全停止线程
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

// ==================== 外部接口 ====================

void Tracking::RegisterCallback(RenderCallback cb)
{
    mRenderCb = cb; // 注册可视化回调，由上层（Stereovo_node）实现
}

void Tracking::FeedStereoImages(const cv::Mat &imLeft, const cv::Mat &imRight, double timestamp)
{
    // 将新图像放入缓冲区，唤醒跟踪线程
    std::unique_lock<std::mutex> lock(mMutexBuf);
    mLeftBuf.push(imLeft.clone());
    mRightBuf.push(imRight.clone());
    mTimeBuf.push(timestamp);
    mCondBuf.notify_one();
}

// ==================== 前端跟踪主循环（线程） ====================

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

        // 处理当前帧（核心函数）
        cv::Mat feat_img;
        std::vector<Eigen::Vector3d> vWorldPoints;
        std::vector<Eigen::Vector3d> vKFPositions;

        Eigen::Isometry3d Tcw = ProcessStereo(matLeft, matRight, timestamp, feat_img, vWorldPoints, vKFPositions);

        // 如果有渲染回调，将结果传递给上层（用于发布ROS消息）
        if (mRenderCb && !feat_img.empty())
        {
            mRenderCb(timestamp, feat_img, vWorldPoints, vKFPositions, Tcw,
                      mpFeatureDetector->mvCurPts, mpFeatureDetector->mvIds, mpFeatureDetector->mvPtsVel);
        }
    }
}

// ==================== 核心：单帧立体处理流程 ====================

Eigen::Isometry3d Tracking::ProcessStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                                          const double &timestamp, cv::Mat &imgTrack,
                                          std::vector<Eigen::Vector3d> &vWorldPoints,
                                          std::vector<Eigen::Vector3d> &vKFPositions)
{
    /**
     * @brief 处理一对立体图像，执行完整的 VO 前端流程
     * @steps:
     *  1. 灰度化 + CLAHE 对比度增强
     *  2. 若未初始化：提取第一批特征，创建首个关键帧
     *  3. 否则：
     *     a. 光流跟踪上一帧特征到当前帧
     *     b. PnP 求解当前帧位姿（利用已有地图点）
     *     c. 剔除异常特征，补充新特征
     *     d. 判断是否创建关键帧
     *     e. 若是关键帧：三角化新的地图点，触发后端优化
     *  4. 更新速度、可视化、保存上一帧状态
     *  5. 返回当前帧的相机位姿 (Tcw)
     */

    // ---------- 1. 预处理 ----------
    cv::Mat grayLeft = imLeft;
    if (imLeft.channels() == 3)
        cv::cvtColor(imLeft, grayLeft, cv::COLOR_BGR2GRAY);
    cv::Mat grayRight = imRight;
    if (imRight.channels() == 3)
        cv::cvtColor(imRight, grayRight, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(grayLeft, grayLeft);
    clahe->apply(grayRight, grayRight);

    // 拼接左右图用于可视化
    cv::Mat imgLeftBGR, imgRightBGR;
    cv::cvtColor(grayLeft, imgLeftBGR, cv::COLOR_GRAY2BGR);
    cv::cvtColor(grayRight, imgRightBGR, cv::COLOR_GRAY2BGR);
    cv::hconcat(imgLeftBGR, imgRightBGR, imgTrack);

    vWorldPoints.clear();
    vKFPositions.clear();

    // 读取当前全局位姿（线程安全）
    Eigen::Isometry3d localPose;
    {
        std::unique_lock<std::mutex> lock(mMutexBackend);
        localPose = mCurrentPose;
    }

    // ---------- 2. 初始化（第一帧） ----------
    if (!mIsInitialized)
    {
        // 清空旧数据，检测第一帧特征
        mpFeatureDetector->mvCurPts.clear();
        mpFeatureDetector->mvIds.clear();
        mpFeatureDetector->mvTrackCnt.clear();
        mmIDToMapPoint.clear();

        localPose = Eigen::Isometry3d::Identity();

        // 1. 提取左目第一批初始特征点
        mpFeatureDetector->AddNewFeatures(grayLeft);
        mpFeatureDetector->mvPtsVel.resize(mpFeatureDetector->mvCurPts.size(), cv::Point2f(0.0f, 0.0f));
        mPrevTime = timestamp;
        mPrevImg = grayLeft.clone();

        // 2. 【新增核心】在第一帧立即进行双目匹配与三角化，构建初始 3D 地图点
        // 第一帧作为世界坐标系原点，传入 localPose (Identity)
        mpFeatureDetector->TriangulateNewPoints(
            grayLeft, grayRight, localPose, mBodyTCam0, mBodyTCam1, mFx, mFy, mCx, mCy,
            mK1, mK2, mP1, mP2,
            mmIDToMapPoint, mpMap, true, vWorldPoints, imgTrack);

        // 3. 创建第一个关键帧（收集当前帧成功生成 3D 地图点或被观测到的特征）
        std::map<int, cv::Point2f> initMeasurements;
        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)
        {
            initMeasurements[mpFeatureDetector->mvIds[i]] = mpFeatureDetector->mvCurPts[i];
        }

        // 注意：传入的位姿是相机的逆位姿 T_w_c
        auto pKF = std::make_shared<KeyFrame>(mNextKFId++, timestamp, localPose.inverse(), initMeasurements);
        mpMap->AddKeyFrame(pKF);

        mvpPrevKFPointsMap = initMeasurements;
        mIsInitialized = true;

        // 4. 绘制特征点
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

    // ---------- 3. 正常跟踪模式 ----------
    // a. 光流跟踪上一帧特征到当前帧
    mpFeatureDetector->TrackFeaturesLK(mPrevImg, grayLeft);

    // b. PnP 估计当前位姿（基于已有地图点）
    bool pnp_succ = mpFeatureDetector->EstimatePosePnP(mmIDToMapPoint, mFx, mFy, mCx, mCy, mK1, mK2, mP1, mP2, localPose);

    // c. 重新设置掩码并补充新特征（保持特征数量稳定）
    mpFeatureDetector->SetMask(grayLeft.rows, grayLeft.cols);
    mpFeatureDetector->AddNewFeatures(grayLeft);

    // d. 判断是否为关键帧
    bool isKeyFrame = false;
    if (pnp_succ)
    {
        isKeyFrame = NeedNewKeyFrame();
    }

    // e. 三角化新的地图点（只在关键帧执行）
    mpFeatureDetector->TriangulateNewPoints(
        grayLeft, grayRight, localPose, mBodyTCam0, mBodyTCam1, mFx, mFy, mCx, mCy,
        mK1, mK2, mP1, mP2,
        mmIDToMapPoint, mpMap, isKeyFrame, vWorldPoints, imgTrack);

    // f. 若为关键帧，则添加到地图，并触发后端优化
    if (isKeyFrame)
    {
        // 构建当前帧的观测
        std::map<int, cv::Point2f> currentMeasurements;
        for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); ++i)
        {
            currentMeasurements[mpFeatureDetector->mvIds[i]] = mpFeatureDetector->mvCurPts[i];
        }

        auto pKF = std::make_shared<KeyFrame>(mNextKFId++, timestamp, localPose.inverse(), currentMeasurements);
        mpMap->AddKeyFrame(pKF);
        mvpPrevKFPointsMap = currentMeasurements;

        // 通知后端线程进行局部BA
        {
            std::unique_lock<std::mutex> lock(mMutexBackend);
            mNeedOptimize = true;
        }
        mCondBackend.notify_one();

        // 清理劣质地图点和冗余关键帧
        CullMapPoints();
        CullRedundantKeyFrames();
    }

    // 如果 PnP 失败，至少将已有地图点加入可视化列表
    if (!pnp_succ)
    {
        for (const auto &pair : mmIDToMapPoint)
        {
            vWorldPoints.push_back(pair.second->GetWorldPos());
        }
    }

    // 收集所有关键帧位置用于可视化
    auto allKFs = mpMap->GetAllKeyFrames();
    for (const auto &kf : allKFs)
    {
        vKFPositions.push_back(kf->GetPose().translation());
    }

    // 绘制当前特征点（颜色表示跟踪长度）
    for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); i++)
    {
        double len = std::min(1.0, 1.0 * mpFeatureDetector->mvTrackCnt[i] / 20.0);
        cv::Scalar ptColor = cv::Scalar(255 * (1 - len), 0, 255 * len);
        cv::circle(imgTrack, mpFeatureDetector->mvCurPts[i], 2, ptColor, 2);
    }

    // 计算特征点速度（用于前端）
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

    // 更新全局位姿
    {
        std::unique_lock<std::mutex> lock(mMutexBackend);
        mCurrentPose = localPose;
    }
    return localPose;
}

// ==================== 关键帧决策 ====================

bool Tracking::NeedNewKeyFrame()
{
    /**
     * @brief 判断当前帧是否应该成为关键帧
     * 标准：当前帧与上一关键帧之间的平均视差超过阈值，或者特征数过少
     */
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
    // 1. 锁定地图数据
    std::unique_lock<std::mutex> lock(mpMap->GetMutex());
    auto &mspMapPoints = mpMap->GetMapPoints(); // 注意这里是引用，可直接修改

    // ===== 筛选参数 =====
    const int MIN_OBSERVATIONS = 2;
    const int MAX_CONSECUTIVE_OUTLIER = 3;

    int cnt_removed_obs = 0;
    int cnt_removed_outlier = 0;
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
            cnt_removed_obs++;
        }
        else if (pMP->GetConsecutiveOutlier() >= MAX_CONSECUTIVE_OUTLIER)
        {
            shouldRemove = true;
            cnt_removed_outlier++;
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
// ==================== 后端优化线程（独立） ====================

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
        // 调用静态优化函数，窗口大小为 10
        Optimizer::LocalBundleAdjustment(mpMap, 10);
    }
}