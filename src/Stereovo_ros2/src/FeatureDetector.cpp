#include "FeatureDetector.h"
#include "MapPoint.h"
#include "Map.h"
#include <opencv2/core/eigen.hpp>
#include <algorithm>

FeatureDetector::FeatureDetector(int maxCnt, int minDist, bool flowBack)
    : mMaxCnt(maxCnt), mMinDist(minDist), mFlowBack(flowBack), mNextId(0) {}

void FeatureDetector::TrackFeaturesLK(const cv::Mat &prevImg, const cv::Mat &currImg)
{
    if (mvPrevPts.empty())
        return;

    std::vector<cv::Point2f> currPts;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevImg, currImg, mvPrevPts, currPts, status, err, cv::Size(21, 21), 3);

    if (mFlowBack)
    {
        std::vector<cv::Point2f> reversePts = mvPrevPts;
        std::vector<uchar> reverseStatus;
        cv::calcOpticalFlowPyrLK(currImg, prevImg, currPts, reversePts, reverseStatus, err, cv::Size(21, 21), 3);
        for (size_t i = 0; i < status.size(); i++)
        {
            if (status[i] && reverseStatus[i])
            {
                if (Distance(mvPrevPts[i], reversePts[i]) > 0.5)
                    status[i] = 0;
            }
            else
                status[i] = 0;
        }
    }

    mvCurPts.clear();
    std::vector<int> keepIds;
    std::vector<int> keepTrackCnt;
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i] && InBorder(currPts[i], currImg.cols, currImg.rows))
        {
            mvCurPts.push_back(currPts[i]);
            keepIds.push_back(mvIds[i]);
            keepTrackCnt.push_back(mvTrackCnt[i]);
        }
    }
    mvIds = keepIds;
    mvTrackCnt = keepTrackCnt;

    for (auto &n : mvTrackCnt)
        n++;
}

bool FeatureDetector::EstimatePosePnP(
    std::map<int, std::shared_ptr<MapPoint>> &mmIDToMapPoint,
    double fx, double fy, double cx, double cy,
    double k1, double k2, double p1, double p2,
    Eigen::Isometry3d &currentPose)
{
    if (mvCurPts.empty() || mmIDToMapPoint.empty())
        return false;

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    std::vector<int> pnpFeatureIndices;

    for (size_t i = 0; i < mvCurPts.size(); ++i)
    {
        int id = mvIds[i];
        auto it = mmIDToMapPoint.find(id);
        if (it != mmIDToMapPoint.end())
        {
            Eigen::Vector3d pos = it->second->GetWorldPos();
            objectPoints.push_back(cv::Point3f(pos.x(), pos.y(), pos.z()));
            imagePoints.push_back(mvCurPts[i]);
            pnpFeatureIndices.push_back(i);
        }
    }

    if (objectPoints.size() < 5)
        return false;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2);
    cv::Mat rvec, tvec;
    std::vector<int> inliers;

    bool pnp_succ = cv::solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, 100, 2.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);

    if (pnp_succ && inliers.size() >= 4)
    {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_t;
        cv::cv2eigen(R, eigen_R);
        cv::cv2eigen(tvec, eigen_t);

        currentPose.linear() = eigen_R;
        currentPose.translation() = eigen_t;

        std::vector<bool> isInlier(mvCurPts.size(), true);
        for (size_t k = 0; k < pnpFeatureIndices.size(); ++k)
            isInlier[pnpFeatureIndices[k]] = false;
        for (int idx : inliers)
            isInlier[pnpFeatureIndices[idx]] = true;

        std::vector<cv::Point2f> compressedPts;
        std::vector<int> compressedIds;
        std::vector<int> compressedTrackCnt;
        for (size_t k = 0; k < mvCurPts.size(); ++k)
        {
            if (isInlier[k])
            {
                compressedPts.push_back(mvCurPts[k]);
                compressedIds.push_back(mvIds[k]);
                compressedTrackCnt.push_back(mvTrackCnt[k]);

                auto it_mp = mmIDToMapPoint.find(mvIds[k]);
                if (it_mp != mmIDToMapPoint.end())
                    it_mp->second->AddObservation();
            }
            else
            {
                mmIDToMapPoint.erase(mvIds[k]);
            }
        }
        mvCurPts = compressedPts;
        mvIds = compressedIds;
        mvTrackCnt = compressedTrackCnt;
        return true;
    }
    return false;
}

void FeatureDetector::TriangulateNewPoints(
    const cv::Mat &grayLeft, const cv::Mat &grayRight,
    const Eigen::Isometry3d &currentPose,
    const Eigen::Matrix4d &bodyTCam0, const Eigen::Matrix4d &bodyTCam1,
    double fx, double fy, double cx, double cy,
    std::map<int, std::shared_ptr<MapPoint>> &mmIDToMapPoint,
    std::shared_ptr<Map> mpMap, bool isKeyFrame,
    std::vector<Eigen::Vector3d> &vWorldPoints, cv::Mat &imgTrack)
{
    if (mvCurPts.empty() || grayRight.empty())
        return;

    std::vector<cv::Point2f> mvRightPts;
    std::vector<uchar> stereoStatus;
    std::vector<float> stereoErr;
    // 正向匹配：从左图特征点追踪匹配到右图对应位置
    cv::calcOpticalFlowPyrLK(grayLeft, grayRight, mvCurPts, mvRightPts, stereoStatus, stereoErr, cv::Size(21, 21), 3);

    // =========================================================================
    // 【核心改进二】双目左右目匹配的正反向光流校验（Stereo Forward-Backward Check）
    // 目的：双目基线处的微小误匹配会引入巨大的深度估计噪声，对三角化点的精度是毁灭性的。
    // =========================================================================
    if (mFlowBack)
    {                                                       // 借用配置参数中的 flowback 开关来控制双目反向校验
        std::vector<cv::Point2f> reverseLeftPts = mvCurPts; // 初始化反向映射存储器
        std::vector<uchar> reverseStereoStatus;             // 反向状态矩阵
        // 反向追踪：从右图的预测匹配点逆向追踪回左图
        cv::calcOpticalFlowPyrLK(grayRight, grayLeft, mvRightPts, reverseLeftPts, reverseStereoStatus, stereoErr, cv::Size(21, 21), 3);

        for (size_t i = 0; i < stereoStatus.size(); i++)
        {
            if (stereoStatus[i] && reverseStereoStatus[i])
            {
                // 如果正向追踪出去再反向追踪回来的欧氏距离像素残差大于 0.5 像素，说明发生了滑移错位，强制剔除！
                if (Distance(mvCurPts[i], reverseLeftPts[i]) > 0.5)
                {
                    stereoStatus[i] = 0;
                }
            }
            else
            {
                stereoStatus[i] = 0; // 单边丢失的点直接扔掉
            }
        }
    }
    // =========================================================================

    Eigen::Matrix4d T_w_body = currentPose.inverse().matrix();
    Eigen::Matrix4d T_w_c0 = T_w_body * bodyTCam0;
    Eigen::Matrix4d T_w_c1 = T_w_body * bodyTCam1;

    Eigen::Vector3d t_c0 = bodyTCam0.block<3, 1>(0, 3);
    Eigen::Vector3d t_c1 = bodyTCam1.block<3, 1>(0, 3);
    double baseline = (t_c1 - t_c0).norm();
    if (baseline < 1e-4)
        baseline = 0.53715;
    double max_reliable_depth = (fx * baseline) / 1.2;

    Eigen::Matrix4d T_c0_w = T_w_c0.inverse();
    Eigen::Matrix4d T_c1_w = T_w_c1.inverse();

    for (size_t i = 0; i < mvCurPts.size(); i++)
    {
        if (stereoStatus[i] && InBorder(mvRightPts[i], grayRight.cols, grayRight.rows))
        {

            cv::Point2f ptRight = mvRightPts[i];
            ptRight.x += grayLeft.cols;
            cv::circle(imgTrack, ptRight, 2, cv::Scalar(0, 255, 0), 2);

            auto it = mmIDToMapPoint.find(mvIds[i]);
            if (it == mmIDToMapPoint.end())
            {
                Eigen::Vector3d P_w;
                double x0 = (mvCurPts[i].x - cx) / fx;
                double y0 = (mvCurPts[i].y - cy) / fy;
                double x1 = (mvRightPts[i].x - cx) / fx;
                double y1 = (mvRightPts[i].y - cy) / fy;

                Eigen::Matrix4d A;
                A.row(0) = x0 * T_c0_w.row(2) - T_c0_w.row(0);
                A.row(1) = y0 * T_c0_w.row(2) - T_c0_w.row(1);
                A.row(2) = x1 * T_c1_w.row(2) - T_c1_w.row(0);
                A.row(3) = y1 * T_c1_w.row(2) - T_c1_w.row(1);

                Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
                Eigen::Vector4d X_w = svd.matrixV().col(3);

                if (std::abs(X_w.w()) >= 1e-6)
                {
                    P_w = X_w.head<3>() / X_w.w();
                    double depth_cam0 = (T_c0_w * X_w).z() / X_w.w();
                    double depth_cam1 = (T_c1_w * X_w).z() / X_w.w();

                    if (depth_cam0 > 0.0 && depth_cam0 < max_reliable_depth && depth_cam1 > 0.0 && depth_cam1 < max_reliable_depth)
                    {
                        auto pMP = std::make_shared<MapPoint>(P_w, mvIds[i]);
                        mmIDToMapPoint[mvIds[i]] = pMP;
                        if (isKeyFrame)
                            mpMap->AddMapPoint(pMP);
                        vWorldPoints.push_back(P_w);
                    }
                }
            }
            else
            {
                vWorldPoints.push_back(it->second->GetWorldPos());
            }
        }
    }
}

void FeatureDetector::SetMask(int rows, int cols)
{
    mMask = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(255));
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cntPtsId;
    for (size_t i = 0; i < mvCurPts.size(); i++)
        cntPtsId.push_back(std::make_pair(mvTrackCnt[i], std::make_pair(mvCurPts[i], mvIds[i])));
    std::sort(cntPtsId.begin(), cntPtsId.end(), [](const auto &a, const auto &b)
              { return a.first > b.first; });

    mvCurPts.clear();
    mvIds.clear();
    mvTrackCnt.clear();
    for (auto &it : cntPtsId)
    {
        cv::Point2f pt = it.second.first;
        if (mMask.at<uchar>(pt) == 255)
        {
            mvCurPts.push_back(pt);
            mvIds.push_back(it.second.second);
            mvTrackCnt.push_back(it.first);
            cv::circle(mMask, pt, mMinDist, 0, -1);
        }
    }
}

void FeatureDetector::AddNewFeatures(const cv::Mat &img)
{
    int countToDetect = mMaxCnt - (int)mvCurPts.size();
    if (countToDetect > 0)
    {
        std::vector<cv::Point2f> nPts;
        cv::goodFeaturesToTrack(img, nPts, countToDetect, 0.01, mMinDist, mMask);

        // =========================================================================
        // 【核心改进一】引入亚像素级精化（Sub-pixel Refinement）
        // 目的：cv::goodFeaturesToTrack 提取出来的角点只有像素级（整数级）精度。
        // 通过图像灰度梯度进行高斯曲面拟合，能将 2D 精度拉高到 0.1 甚至 0.01 像素级，提高几何重投影精度。
        // =========================================================================
        if (!nPts.empty())
        {
            // 配置终止迭代条件：最大迭代 30 次，或者算力精化收敛半径小于 0.01 像素时停止
            cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.01);
            // 设定微调搜索窗口半径为 5x5 像素，零区域设为 (-1,-1) 防止死区阻碍
            cv::cornerSubPix(img, nPts, cv::Size(5, 5), cv::Size(-1, -1), criteria);
        }
        // =========================================================================

        for (const auto &pt : nPts)
        {
            mvCurPts.push_back(pt);
            mvIds.push_back(mNextId++);
            mvTrackCnt.push_back(1);
        }
    }
}

void FeatureDetector::UpdatePreviousStatus(const cv::Mat &grayLeft)
{
    mvPrevPts = mvCurPts;
    mInversePrevPtsMap.clear();
    for (size_t i = 0; i < mvCurPts.size(); i++)
        mInversePrevPtsMap[mvIds[i]] = mvCurPts[i];
}

bool FeatureDetector::InBorder(const cv::Point2f &pt, int cols, int rows)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < cols - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < rows - BORDER_SIZE;
}

double FeatureDetector::Distance(const cv::Point2f &pt1, const cv::Point2f &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return std::sqrt(dx * dx + dy * dy);
}