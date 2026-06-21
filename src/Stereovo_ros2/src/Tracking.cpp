#include "Tracking.h"
#include "Parameters.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

// 构造函数传入全局地图组件引用
Tracking::Tracking(std::shared_ptr<Map> pMap) 
    : mIsInitialized(false), mNextId(0), mpMap(pMap), mNextKFId(0), mIsRunning(true) {
    mCurrentPose = Eigen::Isometry3d::Identity(); 
    mTrackThread = std::thread(&Tracking::TrackLoop, this);
}

Tracking::~Tracking() {
    {
        std::unique_lock<std::mutex> lock(mMutexBuf);
        mIsRunning = false;
    }
    mCondBuf.notify_all();
    if (mTrackThread.joinable()) {
        mTrackThread.join();
    }
}

void Tracking::RegisterCallback(RenderCallback cb) {
    mRenderCb = cb;
}

void Tracking::FeedStereoImages(const cv::Mat& imLeft, const cv::Mat& imRight, double timestamp) {
    std::unique_lock<std::mutex> lock(mMutexBuf);
    mLeftBuf.push(imLeft.clone());
    mRightBuf.push(imRight.clone());
    mTimeBuf.push(timestamp);
    mCondBuf.notify_one(); 
}

void Tracking::TrackLoop() {
    while (true) {
        cv::Mat matLeft, matRight;
        double timestamp = 0.0;

        {
            std::unique_lock<std::mutex> lock(mMutexBuf);
            mCondBuf.wait(lock, [this]() { 
                return !mIsRunning || (!mLeftBuf.empty() && !mRightBuf.empty() && !mTimeBuf.empty()); 
            });

            if (!mIsRunning) break;

            matLeft = mLeftBuf.front();   mLeftBuf.pop();
            matRight = mRightBuf.front(); mRightBuf.pop();
            timestamp = mTimeBuf.front(); mTimeBuf.pop();
        }

        cv::Mat feat_img;
        std::vector<Eigen::Vector3d> vWorldPoints;
        Eigen::Isometry3d Tcw = ProcessStereo(matLeft, matRight, timestamp, feat_img, vWorldPoints);

        if (mRenderCb && !feat_img.empty()) {
            mRenderCb(timestamp, feat_img, vWorldPoints, Tcw);
        }
    }
}

Eigen::Isometry3d Tracking::ProcessStereo(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                          const double &timestamp, cv::Mat& imgTrack,
                                          std::vector<Eigen::Vector3d>& vWorldPoints) {
    cv::Mat grayLeft = imLeft;
    if (imLeft.channels() == 3) cv::cvtColor(imLeft, grayLeft, cv::COLOR_BGR2GRAY);
    cv::Mat grayRight = imRight;
    if (imRight.channels() == 3) cv::cvtColor(imRight, grayRight, cv::COLOR_BGR2GRAY);

    cv::cvtColor(grayLeft, imgTrack, cv::COLOR_GRAY2BGR);
    vWorldPoints.clear();

    // 1. 第一帧初始化（系统无条件筛选为第一个关键帧）
    if (!mIsInitialized) {
        std::cout << ">>> [SLAM前端] 第一帧初始化，固化为初始关键帧。" << std::endl;
        mvCurPts.clear(); mvIds.clear(); mvTrackCnt.clear();
        mmIDToMapPoint.clear();
        mCurrentPose = Eigen::Isometry3d::Identity(); 

        AddNewFeatures(grayLeft);
        
        mPrevImg = grayLeft.clone();
        mvPrevPts = mvCurPts;
        mInversePrevPtsMap.clear();
        for (size_t i = 0; i < mvCurPts.size(); ++i) mInversePrevPtsMap[mvIds[i]] = mvCurPts[i];
        
        // 固化进全局地图
        auto pKF = std::make_shared<KeyFrame>(mNextKFId++, timestamp, mCurrentPose.inverse());
        mpMap->AddKeyFrame(pKF);

        // 记录当前关键帧下的特征点像素点，供后续计算视差使用
        mvpPrevKFPointsMap.clear();
        for (size_t i = 0; i < mvCurPts.size(); ++i) {
            mvpPrevKFPointsMap[mvIds[i]] = mvCurPts[i];
        }

        mIsInitialized = true;
        for (const auto &pt : mvCurPts) cv::circle(imgTrack, pt, 2, cv::Scalar(0, 0, 255), 2);
        return mCurrentPose;
    }

    // 2. 执行图像帧间 LK 光流追踪与正反向校验
    if (!mvPrevPts.empty()) {
        std::vector<cv::Point2f> currPts;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(mPrevImg, grayLeft, mvPrevPts, currPts, status, err, cv::Size(21, 21), 3);

        if (Parameters::FLOW_BACK) {
            std::vector<cv::Point2f> reversePts = mvPrevPts;
            std::vector<uchar> reverseStatus;
            cv::calcOpticalFlowPyrLK(grayLeft, mPrevImg, currPts, reversePts, reverseStatus, err, cv::Size(21, 21), 3);
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i] && reverseStatus[i]) {
                    if (Distance(mvPrevPts[i], reversePts[i]) > 0.5) status[i] = 0;
                } else status[i] = 0;
            }
        }

        mvCurPts.clear();
        std::vector<int> keepIds;
        std::vector<int> keepTrackCnt;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] && InBorder(currPts[i], grayLeft.cols, grayLeft.rows)) {
                mvCurPts.push_back(currPts[i]);
                keepIds.push_back(mvIds[i]);
                keepTrackCnt.push_back(mvTrackCnt[i]);
            }
        }
        mvIds = keepIds;
        mvTrackCnt = keepTrackCnt;
    }

    for (auto &n : mvTrackCnt) n++;

    // 3. 利用已被长期关联的 MapPoint 3D坐标与当前帧 2D 像素坐标求解 PnP 位姿
    if (mIsInitialized && !mvCurPts.empty()) {
        std::vector<cv::Point3f> objectPoints; 
        std::vector<cv::Point2f> imagePoints;  
        std::vector<int> pnpFeatureIndices;    

        for (size_t i = 0; i < mvCurPts.size(); ++i) {
            int id = mvIds[i];
            auto it = mmIDToMapPoint.find(id);
            // 如果这个特征点拥有永久的地图路标点户口
            if (it != mmIDToMapPoint.end()) {
                Eigen::Vector3d pos = it->second->GetWorldPos();
                objectPoints.push_back(cv::Point3f(pos.x(), pos.y(), pos.z()));
                imagePoints.push_back(mvCurPts[i]);
                pnpFeatureIndices.push_back(i);
            }
        }

        if (objectPoints.size() >= 5) {
            cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << Parameters::fx, 0, Parameters::cx, 0, Parameters::fy, Parameters::cy, 0, 0, 1);
            cv::Mat distCoeffs = (cv::Mat_<double>(4, 1) << Parameters::k1, Parameters::k2, Parameters::p1, Parameters::p2);
            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            
            bool pnp_succ = cv::solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);

            if (pnp_succ && inliers.size() >= 4) {
                cv::Mat R;
                cv::Rodrigues(rvec, R);
                Eigen::Matrix3d eigen_R; Eigen::Vector3d eigen_t;
                cv::cv2eigen(R, eigen_R); cv::cv2eigen(tvec, eigen_t);

                mCurrentPose.linear() = eigen_R;
                mCurrentPose.translation() = eigen_t;

                // 剔除 PnP 误匹配外点
                std::vector<bool> isInlier(mvCurPts.size(), true);
                for (size_t k = 0; k < pnpFeatureIndices.size(); ++k) isInlier[pnpFeatureIndices[k]] = false; 
                for (int idx : inliers) isInlier[pnpFeatureIndices[idx]] = true; 

                std::vector<cv::Point2f> compressedPts; std::vector<int> compressedIds; std::vector<int> compressedTrackCnt;
                for (size_t k = 0; k < mvCurPts.size(); ++k) {
                    if (isInlier[k]) {
                        compressedPts.push_back(mvCurPts[k]);
                        compressedIds.push_back(mvIds[k]);
                        compressedTrackCnt.push_back(mvTrackCnt[k]);
                        
                        // 对被成功追踪并在 PnP 中认定为有效内点的地图点增加一次观测寿命计数
                        auto it_mp = mmIDToMapPoint.find(mvIds[k]);
                        if (it_mp != mmIDToMapPoint.end()) {
                            it_mp->second->AddObservation();
                        }
                    } else {
                        mmIDToMapPoint.erase(mvIds[k]); 
                    }
                }
                mvCurPts = compressedPts; mvIds = compressedIds; mvTrackCnt = compressedTrackCnt;
            }
        }
    }

    // 4. 均匀化掩膜并补足新特征点
    SetMask(grayLeft);
    AddNewFeatures(grayLeft);

    // 5. 【判定屏障】：计算当前帧相对于上一个关键帧的视差，决定是否设为新关键帧
    bool isKeyFrame = NeedNewKeyFrame();

    // 6. 执行双目匹配与自适应三角化
    if (!mvCurPts.empty() && !grayRight.empty()) {
        std::vector<cv::Point2f> mvRightPts;
        std::vector<uchar> stereoStatus;
        std::vector<float> stereoErr;
        cv::calcOpticalFlowPyrLK(grayLeft, grayRight, mvCurPts, mvRightPts, stereoStatus, stereoErr, cv::Size(21, 21), 3);

        Eigen::Matrix4d T_w_body = mCurrentPose.inverse().matrix(); 
        Eigen::Matrix4d T_w_c0 = T_w_body * Parameters::body_T_cam0;
        Eigen::Matrix4d T_w_c1 = T_w_body * Parameters::body_T_cam1;

        for (size_t i = 0; i < mvCurPts.size(); i++) {
            if (stereoStatus[i] && InBorder(mvRightPts[i], grayRight.cols, grayRight.rows)) {
                auto it = mmIDToMapPoint.find(mvIds[i]);
                if (it == mmIDToMapPoint.end()) {
                    Eigen::Vector3d P_w;
                    // 三角化出新空间路标点
                    if (TriangulateStereo(mvCurPts[i], mvRightPts[i], T_w_c0, T_w_c1, P_w)) {
                        auto pMP = std::make_shared<MapPoint>(P_w);
                        mmIDToMapPoint[mvIds[i]] = pMP; // 登记在追踪列表中

                        // 如果当前帧是关键帧，直接将其固化进入全局大地图中！
                        if (isKeyFrame) {
                            mpMap->AddMapPoint(pMP);
                        }
                        vWorldPoints.push_back(P_w);
                        cv::circle(imgTrack, mvCurPts[i], 4, cv::Scalar(255, 255, 0), 1); 
                    }
                } else {
                    // 若是老点，直接导出供渲染点云展示
                    vWorldPoints.push_back(it->second->GetWorldPos());
                }
            }
        }
    }

    // 7. 【关键帧持久化】：如果确立为关键帧，执行结构持久化数据落盘
    if (isKeyFrame) {
        // A. 实例化全局关键帧并插入大地图
        auto pKF = std::make_shared<KeyFrame>(mNextKFId++, timestamp, mCurrentPose.inverse());
        mpMap->AddKeyFrame(pKF);

        // B. 更新参考像素基准，用于后续图像帧做视差计算对比
        mvpPrevKFPointsMap.clear();
        for (size_t i = 0; i < mvCurPts.size(); ++i) {
            mvpPrevKFPointsMap[mvIds[i]] = mvCurPts[i];
        }

        // C. 把这一帧中之前未录入过大地图的地图点全部补录固化进去
        for (const auto& id : mvIds) {
            auto it = mmIDToMapPoint.find(id);
            if (it != mmIDToMapPoint.end()) {
                // 此处 AddMapPoint 内部会进行防重入处理
                mpMap->AddMapPoint(it->second);
            }
        }
    }

    std::cout << "[SLAM地图管理] 全局关键帧总数: " << mpMap->GetAllKeyFrames().size() 
              << " | 全局地图固化路标点数: " << mpMap->GetMapPointsSize() << std::endl;

    // 绘制寿命色彩与运动趋势线
    for (size_t i = 0; i < mvCurPts.size(); i++) {
        double len = std::min(1.0, 1.0 * mvTrackCnt[i] / 20.0);
        cv::Scalar ptColor = cv::Scalar(255 * (1 - len), 0, 255 * len);
        cv::circle(imgTrack, mvCurPts[i], 2, ptColor, 2);
        auto it = mInversePrevPtsMap.find(mvIds[i]);
        if (it != mInversePrevPtsMap.end()) cv::arrowedLine(imgTrack, it->second, mvCurPts[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }

    mPrevImg = grayLeft.clone(); mvPrevPts = mvCurPts; mInversePrevPtsMap.clear();
    for (size_t i = 0; i < mvCurPts.size(); i++) mInversePrevPtsMap[mvIds[i]] = mvCurPts[i];

    return mCurrentPose;
}

bool Tracking::NeedNewKeyFrame() {
    // 如果跟踪点太少，必须立刻强制插入关键帧以防丢失目标
    if (mvCurPts.size() < 20) {
        return true;
    }

    double total_parallax = 0.0;
    int common_tracked_pts_cnt = 0;

    // 计算当前帧与上一个关键帧共同拥有的特征点之间的像素平均移动距离（视差）
    for (size_t i = 0; i < mvCurPts.size(); ++i) {
        int id = mvIds[i];
        auto it = mvpPrevKFPointsMap.find(id);
        if (it != mvpPrevKFPointsMap.end()) {
            total_parallax += Distance(mvCurPts[i], it->second);
            common_tracked_pts_cnt++;
        }
    }

    // 如果没有共同观测到的点，说明运动过于剧烈，强行设为关键帧
    if (common_tracked_pts_cnt == 0) {
        return true;
    }

    double average_parallax = total_parallax / common_tracked_pts_cnt;
    
    // 如果平均视差超过了配置文件设定的阈值 (keyframe_parallax: 15)
    return average_parallax >= Parameters::KEYFRAME_PARALLAX;
}

bool Tracking::TriangulateStereo(const cv::Point2f &ptLeft, const cv::Point2f &ptRight,
                                 const Eigen::Matrix4d &T_w_c0, const Eigen::Matrix4d &T_w_c1,
                                 Eigen::Vector3d &P_w) {
    double x0 = (ptLeft.x - Parameters::cx) / Parameters::fx;
    double y0 = (ptLeft.y - Parameters::cy) / Parameters::fy;
    double x1 = (ptRight.x - Parameters::cx) / Parameters::fx;
    double y1 = (ptRight.y - Parameters::cy) / Parameters::fy;

    Eigen::Matrix4d T_c0_w = T_w_c0.inverse(); Eigen::Matrix4d T_c1_w = T_w_c1.inverse();
    Eigen::Matrix4d A;
    A.row(0) = x0 * T_c0_w.row(2) - T_c0_w.row(0); A.row(1) = y0 * T_c0_w.row(2) - T_c0_w.row(1);
    A.row(2) = x1 * T_c1_w.row(2) - T_c1_w.row(0); A.row(3) = y1 * T_c1_w.row(2) - T_c1_w.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X_w = svd.matrixV().col(3);
    if (std::abs(X_w.w()) < 1e-6) return false;
    P_w = X_w.head<3>() / X_w.w();

    Eigen::Vector3d t_c0 = Parameters::body_T_cam0.block<3, 1>(0, 3);
    Eigen::Vector3d t_c1 = Parameters::body_T_cam1.block<3, 1>(0, 3);
    double baseline = (t_c1 - t_c0).norm();
    if (baseline < 1e-4) baseline = 0.53715;

    double max_reliable_depth = (Parameters::fx * baseline) / 1.2;
    Eigen::Vector4d P_w_homo(P_w.x(), P_w.y(), P_w.z(), 1.0);
    double depth_cam0 = (T_c0_w * P_w_homo).z(); double depth_cam1 = (T_c1_w * P_w_homo).z();

    if (depth_cam0 > 0.0 && depth_cam0 < max_reliable_depth && depth_cam1 > 0.0 && depth_cam1 < max_reliable_depth) return true;
    return false;
}

void Tracking::SetMask(const cv::Mat &img) {
    mMask = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(255));
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cntPtsId;
    for (size_t i = 0; i < mvCurPts.size(); i++) cntPtsId.push_back(std::make_pair(mvTrackCnt[i], std::make_pair(mvCurPts[i], mvIds[i])));
    std::sort(cntPtsId.begin(), cntPtsId.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    mvCurPts.clear(); mvIds.clear(); mvTrackCnt.clear();
    for (auto &it : cntPtsId) {
        cv::Point2f pt = it.second.first;
        if (mMask.at<uchar>(pt) == 255) {
            mvCurPts.push_back(pt); mvIds.push_back(it.second.second); mvTrackCnt.push_back(it.first);
            cv::circle(mMask, pt, Parameters::MIN_DIST, 0, -1);
        }
    }
}

void Tracking::AddNewFeatures(const cv::Mat &img) {
    int countToDetect = Parameters::MAX_CNT - (int)mvCurPts.size();
    if (countToDetect > 0) {
        std::vector<cv::Point2f> nPts;
        cv::goodFeaturesToTrack(img, nPts, countToDetect, 0.01, Parameters::MIN_DIST, mMask);
        for (const auto &pt : nPts) {
            mvCurPts.push_back(pt); mvIds.push_back(mNextId++); mvTrackCnt.push_back(1);
        }
    }
}

bool Tracking::InBorder(const cv::Point2f &pt, int cols, int rows) {
    const int BORDER_SIZE = 1; int img_x = cvRound(pt.x); int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < cols - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < rows - BORDER_SIZE;
}

double Tracking::Distance(const cv::Point2f &pt1, const cv::Point2f &pt2) {
    double dx = pt1.x - pt2.x; double dy = pt1.y - pt2.y; return std::sqrt(dx * dx + dy * dy);
}