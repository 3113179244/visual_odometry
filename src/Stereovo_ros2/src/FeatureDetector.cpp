#include "FeatureDetector.h"
#include "MapPoint.h"
#include "Map.h"
#include <opencv2/core/eigen.hpp>
#include <algorithm>

FeatureDetector::FeatureDetector(int maxCnt, int minDist, bool flowBack)
    : mMaxCnt(maxCnt), mMinDist(minDist), mFlowBack(flowBack), mNextId(0) {}

void FeatureDetector::TrackFeaturesLK(const cv::Mat &prevImg, const cv::Mat &currImg) {
    if (mvPrevPts.empty()) return;

    std::vector<cv::Point2f> currPts;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevImg, currImg, mvPrevPts, currPts, status, err, cv::Size(21, 21), 3);

    if (mFlowBack) {
        std::vector<cv::Point2f> reversePts = mvPrevPts;
        std::vector<uchar> reverseStatus;
        cv::calcOpticalFlowPyrLK(currImg, prevImg, currPts, reversePts, reverseStatus, err, cv::Size(21, 21), 3);
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
        if (status[i] && InBorder(currPts[i], currImg.cols, currImg.rows)) {
            mvCurPts.push_back(currPts[i]);
            keepIds.push_back(mvIds[i]);
            keepTrackCnt.push_back(mvTrackCnt[i]);
        }
    }
    mvIds = keepIds;
    mvTrackCnt = keepTrackCnt;

    for (auto &n : mvTrackCnt) n++;
}

bool FeatureDetector::EstimatePosePnP(
    std::map<int, std::shared_ptr<MapPoint>> &mmIDToMapPoint,
    double fx, double fy, double cx, double cy,
    double k1, double k2, double p1, double p2,
    Eigen::Isometry3d &currentPose) 
{
    if (mvCurPts.empty()) return false;

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    std::vector<int> pnpFeatureIndices;

    for (size_t i = 0; i < mvCurPts.size(); ++i) {
        int id = mvIds[i];
        auto it = mmIDToMapPoint.find(id);
        if (it != mmIDToMapPoint.end()) {
            Eigen::Vector3d pos = it->second->GetWorldPos();
            objectPoints.push_back(cv::Point3f(pos.x(), pos.y(), pos.z()));
            imagePoints.push_back(mvCurPts[i]);
            pnpFeatureIndices.push_back(i);
        }
    }

    if (objectPoints.size() < 5) return false;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2);
    cv::Mat rvec, tvec; std::vector<int> inliers;

    bool pnp_succ = cv::solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);

    if (pnp_succ && inliers.size() >= 4) {
        cv::Mat R; cv::Rodrigues(rvec, R);
        Eigen::Matrix3d eigen_R; Eigen::Vector3d eigen_t;
        cv::cv2eigen(R, eigen_R); cv::cv2eigen(tvec, eigen_t);

        currentPose.linear() = eigen_R;
        currentPose.translation() = eigen_t;

        std::vector<bool> isInlier(mvCurPts.size(), true);
        for (size_t k = 0; k < pnpFeatureIndices.size(); ++k) isInlier[pnpFeatureIndices[k]] = false;
        for (int idx : inliers) isInlier[pnpFeatureIndices[idx]] = true;

        std::vector<cv::Point2f> compressedPts;
        std::vector<int> compressedIds;
        std::vector<int> compressedTrackCnt;
        for (size_t k = 0; k < mvCurPts.size(); ++k) {
            if (isInlier[k]) {
                compressedPts.push_back(mvCurPts[k]);
                compressedIds.push_back(mvIds[k]);
                compressedTrackCnt.push_back(mvTrackCnt[k]);

                auto it_mp = mmIDToMapPoint.find(mvIds[k]);
                if (it_mp != mmIDToMapPoint.end()) it_mp->second->AddObservation();
            } else {
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
    if (mvCurPts.empty() || grayRight.empty()) return;

    std::vector<cv::Point2f> mvRightPts;
    std::vector<uchar> stereoStatus;
    std::vector<float> stereoErr;
    cv::calcOpticalFlowPyrLK(grayLeft, grayRight, mvCurPts, mvRightPts, stereoStatus, stereoErr, cv::Size(21, 21), 3);

    Eigen::Matrix4d T_w_body = currentPose.inverse().matrix();
    Eigen::Matrix4d T_w_c0 = T_w_body * bodyTCam0;
    Eigen::Matrix4d T_w_c1 = T_w_body * bodyTCam1;

    // 基线解析计算
    Eigen::Vector3d t_c0 = bodyTCam0.block<3, 1>(0, 3);
    Eigen::Vector3d t_c1 = bodyTCam1.block<3, 1>(0, 3);
    double baseline = (t_c1 - t_c0).norm();
    if (baseline < 1e-4) baseline = 0.53715;
    double max_reliable_depth = (fx * baseline) / 1.2;

    Eigen::Matrix4d T_c0_w = T_w_c0.inverse();
    Eigen::Matrix4d T_c1_w = T_w_c1.inverse();

    for (size_t i = 0; i < mvCurPts.size(); i++) {
        if (stereoStatus[i] && InBorder(mvRightPts[i], grayRight.cols, grayRight.rows)) {
            auto it = mmIDToMapPoint.find(mvIds[i]);
            if (it == mmIDToMapPoint.end()) {
                Eigen::Vector3d P_w;
                // --- 核心数学公式移到了底层优化 ---
                // 计算去中心反投影点
                double x0 = (mvCurPts[i].x - cx) / fx; double y0 = (mvCurPts[i].y - cy) / fy;
                double x1 = (mvRightPts[i].x - cx) / fx; double y1 = (mvRightPts[i].y - cy) / fy;

                Eigen::Matrix4d A;
                A.row(0) = x0 * T_c0_w.row(2) - T_c0_w.row(0);
                A.row(1) = y0 * T_c0_w.row(2) - T_c0_w.row(1);
                A.row(2) = x1 * T_c1_w.row(2) - T_c1_w.row(0);
                A.row(3) = y1 * T_c1_w.row(2) - T_c1_w.row(1);

                Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
                Eigen::Vector4d X_w = svd.matrixV().col(3);
                
                if (std::abs(X_w.w()) >= 1e-6) {
                    P_w = X_w.head<3>() / X_w.w();
                    double depth_cam0 = (T_c0_w * X_w).z() / X_w.w();
                    double depth_cam1 = (T_c1_w * X_w).z() / X_w.w();

                    if (depth_cam0 > 0.0 && depth_cam0 < max_reliable_depth && depth_cam1 > 0.0 && depth_cam1 < max_reliable_depth) {
                        auto pMP = std::make_shared<MapPoint>(P_w, mvIds[i]);
                        mmIDToMapPoint[mvIds[i]] = pMP;
                        if (isKeyFrame) mpMap->AddMapPoint(pMP);
                        vWorldPoints.push_back(P_w);
                        cv::circle(imgTrack, mvCurPts[i], 4, cv::Scalar(255, 255, 0), 1);
                    }
                }
            } else {
                vWorldPoints.push_back(it->second->GetWorldPos());
            }
        }
    }
}

void FeatureDetector::SetMask(int rows, int cols) {
    mMask = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(255));
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cntPtsId;
    for (size_t i = 0; i < mvCurPts.size(); i++)
        cntPtsId.push_back(std::make_pair(mvTrackCnt[i], std::make_pair(mvCurPts[i], mvIds[i])));
    std::sort(cntPtsId.begin(), cntPtsId.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    
    mvCurPts.clear(); mvIds.clear(); mvTrackCnt.clear();
    for (auto &it : cntPtsId) {
        cv::Point2f pt = it.second.first;
        if (mMask.at<uchar>(pt) == 255) {
            mvCurPts.push_back(pt);
            mvIds.push_back(it.second.second);
            mvTrackCnt.push_back(it.first);
            cv::circle(mMask, pt, mMinDist, 0, -1);
        }
    }
}

void FeatureDetector::AddNewFeatures(const cv::Mat &img) {
    int countToDetect = mMaxCnt - (int)mvCurPts.size();
    if (countToDetect > 0) {
        std::vector<cv::Point2f> nPts;
        cv::goodFeaturesToTrack(img, nPts, countToDetect, 0.01, mMinDist, mMask);
        for (const auto &pt : nPts) {
            mvCurPts.push_back(pt); mvIds.push_back(mNextId++); mvTrackCnt.push_back(1);
        }
    }
}

void FeatureDetector::UpdatePreviousStatus(const cv::Mat &grayLeft) {
    mvPrevPts = mvCurPts;
    mInversePrevPtsMap.clear();
    for (size_t i = 0; i < mvCurPts.size(); i++)
        mInversePrevPtsMap[mvIds[i]] = mvCurPts[i];
}

bool FeatureDetector::InBorder(const cv::Point2f &pt, int cols, int rows) {
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x); int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < cols - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < rows - BORDER_SIZE;
}

double FeatureDetector::Distance(const cv::Point2f &pt1, const cv::Point2f &pt2) {
    double dx = pt1.x - pt2.x; double dy = pt1.y - pt2.y;
    return std::sqrt(dx * dx + dy * dy);
}