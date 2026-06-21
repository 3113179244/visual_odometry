#include "Tracking.h"
#include "Map.h"
#include "FeatureDetector.h"
#include "KeyFrame.h"
#include "Optimizer.h"
#include "Parameters.h"
#include <opencv2/core/eigen.hpp>
#include <iostream>

Tracking::Tracking(std::shared_ptr<Map> pMap)
    : mpMap(pMap), mIsInitialized(false), mIsRunning(true), mNeedOptimize(false), mNextKFId(0)
{
    // 对全局静态参数做一次性深拷贝，彻底杜绝数据竞争
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

    // 实例化前端算法管道
    mpFeatureDetector = std::make_unique<FeatureDetector>(Parameters::MAX_CNT, Parameters::MIN_DIST, mFlowBack);

    // 启动前后端多线程并发管道
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
                      mpFeatureDetector->mvCurPts, mpFeatureDetector->mvIds);
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

    // 将左右目图像横向拼接组合成双目拼接大图
    cv::Mat imgLeftBGR, imgRightBGR;
    cv::cvtColor(grayLeft, imgLeftBGR, cv::COLOR_GRAY2BGR);
    cv::cvtColor(grayRight, imgRightBGR, cv::COLOR_GRAY2BGR);
    cv::hconcat(imgLeftBGR, imgRightBGR, imgTrack);

    vWorldPoints.clear();
    vKFPositions.clear();

    // 提取局部变量，锁保护临界区拷贝
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
    // 步长 1：帧间 LK 光流追踪
    mpFeatureDetector->TrackFeaturesLK(mPrevImg, grayLeft);

    // 步长 2：PnP 求解当前帧初估位姿
    bool pnp_succ = mpFeatureDetector->EstimatePosePnP(mmIDToMapPoint, mFx, mFy, mCx, mCy, mK1, mK2, mP1, mP2, localPose);

    // 步长 3：均匀分布特征约束并补充提取新特征点
    mpFeatureDetector->SetMask(grayLeft.rows, grayLeft.cols);
    mpFeatureDetector->AddNewFeatures(grayLeft);

    // 步长 4：挑选并判定当前帧是否为关键帧
    bool isKeyFrame = false;
    if (pnp_succ)
    {
        isKeyFrame = NeedNewKeyFrame();
    }

    // 步长 5：立体匹配与双目自适应深度三角化
    // 【注意】TriangulateNewPoints 内部如果判定 isKeyFrame == true，已经自动执行了 mpMap->AddMapPoint(pMP)
    mpFeatureDetector->TriangulateNewPoints(
        grayLeft, grayRight, localPose, mBodyTCam0, mBodyTCam1, mFx, mFy, mCx, mCy,
        mmIDToMapPoint, mpMap, isKeyFrame, vWorldPoints, imgTrack);

    // 步长 6：新关键帧固化并触发异步后端 BA 优化
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

        // =========================================================================
        // 【核心修复一】彻底删除了原先这里的 mmIDToMapPoint 循环二次 AddMapPoint 逻辑
        // 原因：Step 5 内部已经把新三角化的路标点指针安全加入了 mpMap 数据库。
        // 如果这里再加一遍，会导致同一个 MapPoint 实例在 vector 容器里重复叠加，造成严重的内存臃肿。
        // =========================================================================

        // 异步唤醒常驻在后台的 Ceres 优化线程
        {
            std::unique_lock<std::mutex> lock(mMutexBackend);
            mNeedOptimize = true;
        }
        mCondBackend.notify_one();
    }

    // 优化控制台打印：提供清晰的系统提示
    if (!pnp_succ)
    {
        std::cout << ">>> [SLAM前端] 提示：PnP 暂无足够追踪点（如冷启动首帧或发生短暂遮挡），已通过双目实时三角化构建/恢复结构。" << std::endl;
        // PnP失败时，直接把双目刚刚三角化出来的局部点传给可视化组件，防止 RViz 画面变空
        for (const auto &pair : mmIDToMapPoint)
        {
            vWorldPoints.push_back(pair.second->GetWorldPos());
        }
    }

    // 获取全局轨迹供给可视化发布
    auto allKFs = mpMap->GetAllKeyFrames();
    for (const auto &kf : allKFs)
    {
        vKFPositions.push_back(kf->GetPose().translation());
    }

    std::cout << "[SLAM地图管理] 全局关键帧总数: " << allKFs.size()
              << " | 全局地图固化路标点数: " << mpMap->GetMapPointsSize() << std::endl;

    // 红蓝渐变渲染左图的追踪特征点
    for (size_t i = 0; i < mpFeatureDetector->mvCurPts.size(); i++)
    {
        double len = std::min(1.0, 1.0 * mpFeatureDetector->mvTrackCnt[i] / 20.0);
        cv::Scalar ptColor = cv::Scalar(255 * (1 - len), 0, 255 * len);
        cv::circle(imgTrack, mpFeatureDetector->mvCurPts[i], 2, ptColor, 2);
    }

    mPrevImg = grayLeft.clone();
    mpFeatureDetector->UpdatePreviousStatus(grayLeft);

    // 临界区锁保护，安全更新并刷回全局缓冲变量
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

        std::cout << ">>> [后端优化] 异步线程触发 10 帧滑动窗口局部 BA 优化..." << std::endl;
        Optimizer::LocalBundleAdjustment(mpMap, 10);

        // =========================================================================
        // 【核心修复二】彻底删除了原先后端强行重写前端实时 mCurrentPose 的破坏性代码
        // 原因：BA 优化只负责精调滑窗内的“历史关键帧位姿”和“地图点 3D 坐标”。
        // 前端通过 PnP 求解实时帧时，由于输入的是被后端优化后的最新地图点，它解算出来的 localPose
        // 会天然收敛至无漂移的正确平面，不需要在这里用历史的位姿进行生硬覆盖。
        // 删掉此处逻辑后，前端主线程的帧间流畅度将彻底恢复，彻底解决轨迹断层、跳变或 PnP 频繁丢失的底层逻辑死结。
        // =========================================================================
    }
}