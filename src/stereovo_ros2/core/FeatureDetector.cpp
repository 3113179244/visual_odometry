#include "core/FeatureDetector.h"
#include "core/MapPoint.h"
#include "core/Map.h"
#include "utils/Parameters.h" // 【新增包含】引入全局参数，以便使用 F_THRESHOLD
#include <opencv2/core/eigen.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

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
    // --- 临时容器：收集通过基础光流筛选的点对 ---
    std::vector<cv::Point2f> tmpPrevPts, tmpCurPts;
    std::vector<int> tmpIds;
    std::vector<int> tmpTrackCnt;
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i] && InBorder(currPts[i], currImg.cols, currImg.rows))
        {
            tmpPrevPts.push_back(mvPrevPts[i]);
            tmpCurPts.push_back(currPts[i]);
            tmpIds.push_back(mvIds[i]);
            tmpTrackCnt.push_back(mvTrackCnt[i]);
        }
    }
    // =========================================================================
    // 【新增防线一】时间域对极约束剔除误匹配 (Temporal Epipolar RANSAC Check)
    // 作用：利用前后两帧之间的 2D-2D 约束，估计基础矩阵 F 并执行 RANSAC 清洗。
    // 激活了 YAML 配置文件中的 Parameters::F_THRESHOLD 属性 (默认为 1.0 像素误差)
    // =========================================================================
    mvCurPts.clear();
    mvIds.clear();
    mvTrackCnt.clear();
    if (tmpCurPts.size() >= 8) // 计算基础矩阵 F 至少需要 8 个点对
    {
        std::vector<uchar> fStatus;
        // 调用 OpenCV 利用对极残差剔除偏离刚体运动的误匹配
        cv::findFundamentalMat(tmpPrevPts, tmpCurPts, cv::FM_RANSAC, Parameters::F_THRESHOLD, 0.99, fStatus);
        for (size_t i = 0; i < fStatus.size(); i++)
        {
            if (fStatus[i]) // 仅保留符合对极约束的内点 (Inliers)
            {
                mvCurPts.push_back(tmpCurPts[i]);
                mvIds.push_back(tmpIds[i]);
                mvTrackCnt.push_back(tmpTrackCnt[i]);
            }
        }
    }
    else
    {
        // 极端防护：如果追踪点数严重不足，降级保留光流原生结果防止系统直接死锁
        mvCurPts = tmpCurPts;
        mvIds = tmpIds;
        mvTrackCnt = tmpTrackCnt;
    }
    // =========================================================================
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
    {
        std::cout << "[DEBUG-PnP] 跳过PnP：mvCurPts=" << mvCurPts.size()
                  << ", mmIDToMapPoint=" << mmIDToMapPoint.size() << std::endl;
        return false;
    }

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

    if (objectPoints.size() < 4) // solvePnPRansac 鲁棒要求
    {
        std::cout << "[DEBUG-PnP] 跳过PnP：参与PnP的3D点不足4个，只有 " << objectPoints.size() << " 个" << std::endl;
        return false;
    }

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

        std::vector<bool> isInlier(mvCurPts.size(), false);
        for (int idx : inliers)
            isInlier[pnpFeatureIndices[idx]] = true;

        std::vector<cv::Point2f> compressedPts;
        std::vector<int> compressedIds;
        std::vector<int> compressedTrackCnt;

        // ========== [DEBUG] 过滤统计 ==========
        int cnt_keep_inlier = 0;         // 保留：已三角化且PnP内点
        int cnt_remove_outlier = 0;      // 删除：已三角化但PnP外点
        int cnt_keep_untriangulated = 0; // 保留：未三角化（待后续三角化）
        std::vector<int> removed_outlier_ids;
        // =======================================

        for (size_t k = 0; k < mvCurPts.size(); ++k)
        {
            int id = mvIds[k];
            auto it_mp = mmIDToMapPoint.find(id);
            bool has_triangulated = (it_mp != mmIDToMapPoint.end());

            if (has_triangulated)
            {
                if (isInlier[k])
                {
                    // 情况1：已三角化 + PnP内点 → 保留
                    compressedPts.push_back(mvCurPts[k]);
                    compressedIds.push_back(id);
                    compressedTrackCnt.push_back(mvTrackCnt[k]);

                    // 标记为内点 + 观测次数+1
                    it_mp->second->MarkAsInlier();
                    it_mp->second->AddObservation();

                    cnt_keep_inlier++;
                }
                else
                {
                    // 情况2：已三角化 + PnP外点 → 删除
                    // 先标记为外点（连续外点计数+1）
                    it_mp->second->MarkAsOutlier();

                    mmIDToMapPoint.erase(id);
                    cnt_remove_outlier++;
                    removed_outlier_ids.push_back(id);
                }
            }
            else
            {
                // 情况3：未三角化的点 → 保留（BUG修复）
                compressedPts.push_back(mvCurPts[k]);
                compressedIds.push_back(id);
                compressedTrackCnt.push_back(mvTrackCnt[k]);
                cnt_keep_untriangulated++;
            }
        }

        mvCurPts = compressedPts;
        mvIds = compressedIds;
        mvTrackCnt = compressedTrackCnt;

        // ========== [DEBUG] PnP 后统计 ==========
        // std::cout << "[DEBUG-PnP] PnP求解成功，内点数: " << inliers.size() << " / " << objectPoints.size() << std::endl;
        // std::cout << "[DEBUG-PnP] 过滤后统计: " << std::endl;
        // std::cout << "[DEBUG-PnP]   - 保留(已三角化+内点): " << cnt_keep_inlier << std::endl;
        // std::cout << "[DEBUG-PnP]   - 删除(已三角化+外点): " << cnt_remove_outlier << std::endl;
        // if (cnt_remove_outlier > 0 && cnt_remove_outlier <= 10)
        // {
        //     std::cout << "[DEBUG-PnP]     被删外点ID: ";
        //     for (int rid : removed_outlier_ids) std::cout << rid << " ";
        //     std::cout << std::endl;
        // }
        // std::cout << "[DEBUG-PnP]   - 保留(未三角化,待后续三角化): " << cnt_keep_untriangulated << std::endl;
        // std::cout << "[DEBUG-PnP] 最终特征点总数: " << mvCurPts.size()
        //           << " (验证: " << cnt_keep_inlier + cnt_keep_untriangulated << ")" << std::endl;
        // std::cout << "[DEBUG-PnP] === PnP 结束 ===" << std::endl;
        // =========================================

        return true;
    }
    else
    {
        std::cout << "[DEBUG-PnP] PnP求解失败或内点不足4个，内点数: " << inliers.size() << std::endl;
        return false;
    }
}

void FeatureDetector::TriangulateNewPoints(
    const cv::Mat &grayLeft, const cv::Mat &grayRight,
    const Eigen::Isometry3d &currentPose,
    const Eigen::Matrix4d &bodyTCam0, const Eigen::Matrix4d &bodyTCam1,
    double fx, double fy, double cx, double cy,
    double k1, double k2, double p1, double p2,
    std::map<int, std::shared_ptr<MapPoint>> &mmIDToMapPoint,
    std::shared_ptr<Map> mpMap, bool isKeyFrame,
    std::vector<Eigen::Vector3d> &vWorldPoints, cv::Mat &imgTrack)
{
    if (mvCurPts.empty() || grayRight.empty())
        return;

    std::vector<cv::Point2f> mvRightPts;
    std::vector<uchar> stereoStatus;
    std::vector<float> stereoErr;
    cv::calcOpticalFlowPyrLK(grayLeft, grayRight, mvCurPts, mvRightPts, stereoStatus, stereoErr, cv::Size(21, 21), 3);
    if (mFlowBack)
    {
        std::vector<cv::Point2f> reverseLeftPts = mvCurPts;
        std::vector<uchar> reverseStereoStatus;
        cv::calcOpticalFlowPyrLK(grayRight, grayLeft, mvRightPts, reverseLeftPts, reverseStereoStatus, stereoErr, cv::Size(21, 21), 3);
        for (size_t i = 0; i < stereoStatus.size(); i++)
        {
            if (stereoStatus[i] && reverseStereoStatus[i])
            {
                if (Distance(mvCurPts[i], reverseLeftPts[i]) > 0.5)
                    stereoStatus[i] = 0;
            }
            else
                stereoStatus[i] = 0;
        }
    }

    // =========================================================================
    // 【新增防线二】双目极线约束剔除误匹配
    // =========================================================================
    for (size_t i = 0; i < stereoStatus.size(); i++)
    {
        if (stereoStatus[i])
        {
            if (std::abs(mvCurPts[i].y - mvRightPts[i].y) > 1.5)
            {
                stereoStatus[i] = 0;
            }
        }
    }
    // =========================================================================

    std::vector<cv::Point2f> mvCurPtsUn = mvCurPts;
    std::vector<cv::Point2f> mvRightPtsUn = mvRightPts;

    if (k1 != 0.0 || k2 != 0.0 || p1 != 0.0 || p2 != 0.0)
    {
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat distCoeffs = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2);
        cv::undistortPoints(mvCurPts, mvCurPtsUn, cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix);
        cv::undistortPoints(mvRightPts, mvRightPtsUn, cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix);
    }

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

    // ========== [DEBUG] 三角化质量统计 ==========
    int cnt_new_triangulated = 0;
    int cnt_tri_failed = 0;
    int cnt_fail_svd_ratio = 0; // SVD奇异值比值不通过
    int cnt_fail_reproj = 0;    // 重投影误差不通过
    int cnt_fail_disparity = 0; // 视差太小不通过
    int cnt_fail_depth = 0;     // 深度不通过
    // ===========================================

    // ===== 质量校验阈值 =====
    const double SVD_RATIO_THRESH = 0.1;  // SVD奇异值比值阈值（越小越确定）
    const double REPROJ_ERR_THRESH = 1.0; // 重投影误差阈值（像素）
    const double MIN_DISPARITY = 1.0;     // 最小视差（像素）
    // ========================

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
                // ========== 【校验1】视差检查 ==========
                double disparity = mvCurPts[i].x - mvRightPts[i].x;
                if (std::abs(disparity) < MIN_DISPARITY)
                {
                    cnt_fail_disparity++;
                    cnt_tri_failed++;
                    continue;
                }
                // =======================================

                Eigen::Vector3d P_w;
                double x0 = (mvCurPtsUn[i].x - cx) / fx;
                double y0 = (mvCurPtsUn[i].y - cy) / fy;
                double x1 = (mvRightPtsUn[i].x - cx) / fx;
                double y1 = (mvRightPtsUn[i].y - cy) / fy;

                Eigen::Matrix4d A;
                A.row(0) = x0 * T_c0_w.row(2) - T_c0_w.row(0);
                A.row(1) = y0 * T_c0_w.row(2) - T_c0_w.row(1);
                A.row(2) = x1 * T_c1_w.row(2) - T_c1_w.row(0);
                A.row(3) = y1 * T_c1_w.row(2) - T_c1_w.row(1);

                Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
                Eigen::Vector4d X_w = svd.matrixV().col(3);

                // ========== 【校验2】SVD奇异值比值检查 ==========
                // 最小奇异值 / 次小奇异值，衡量解的确定性
                // 比值越小，说明齐次方程的解越唯一，三角化越可靠
                Eigen::Vector4d singularValues = svd.singularValues();
                double sv_ratio = singularValues(3) / singularValues(2);
                if (sv_ratio > SVD_RATIO_THRESH)
                {
                    cnt_fail_svd_ratio++;
                    cnt_tri_failed++;
                    continue;
                }
                // ===============================================

                if (std::abs(X_w.w()) >= 1e-6)
                {
                    P_w = X_w.head<3>() / X_w.w();
                    double depth_cam0 = (T_c0_w * X_w).z() / X_w.w();
                    double depth_cam1 = (T_c1_w * X_w).z() / X_w.w();

                    // ========== 【校验3】深度范围检查 ==========
                    if (!(depth_cam0 > 0.0 && depth_cam0 < max_reliable_depth &&
                          depth_cam1 > 0.0 && depth_cam1 < max_reliable_depth))
                    {
                        cnt_fail_depth++;
                        cnt_tri_failed++;
                        continue;
                    }
                    // =========================================

                    // ========== 【校验4】重投影误差检查 ==========
                    // 将3D点反投影回左目相机
                    Eigen::Vector4d P_homo(P_w.x(), P_w.y(), P_w.z(), 1.0);
                    Eigen::Vector4d p_c0 = T_c0_w * P_homo;
                    double u0_reproj = fx * p_c0.x() / p_c0.z() + cx;
                    double v0_reproj = fy * p_c0.y() / p_c0.z() + cy;
                    double err0 = std::sqrt(
                        (u0_reproj - mvCurPtsUn[i].x) * (u0_reproj - mvCurPtsUn[i].x) +
                        (v0_reproj - mvCurPtsUn[i].y) * (v0_reproj - mvCurPtsUn[i].y));

                    // 反投影回右目相机
                    Eigen::Vector4d p_c1 = T_c1_w * P_homo;
                    double u1_reproj = fx * p_c1.x() / p_c1.z() + cx;
                    double v1_reproj = fy * p_c1.y() / p_c1.z() + cy;
                    double err1 = std::sqrt(
                        (u1_reproj - mvRightPtsUn[i].x) * (u1_reproj - mvRightPtsUn[i].x) +
                        (v1_reproj - mvRightPtsUn[i].y) * (v1_reproj - mvRightPtsUn[i].y));

                    if (err0 > REPROJ_ERR_THRESH || err1 > REPROJ_ERR_THRESH)
                    {
                        cnt_fail_reproj++;
                        cnt_tri_failed++;
                        continue;
                    }
                    // =============================================

                    // 所有校验都通过，接受这个点
                    auto pMP = std::make_shared<MapPoint>(P_w, mvIds[i]);
                    mmIDToMapPoint[mvIds[i]] = pMP;
                    if (isKeyFrame)
                        mpMap->AddMapPoint(pMP);
                    vWorldPoints.push_back(P_w);
                    cnt_new_triangulated++;
                }
                else
                {
                    cnt_fail_depth++;
                    cnt_tri_failed++;
                }
            }
            else
            {
                vWorldPoints.push_back(it->second->GetWorldPos());
            }
        }
    }

    // ========== [DEBUG] 三角化后统计 ==========
    // int cnt_after_tri = 0;
    // int cnt_after_untri = 0;
    // for (size_t i = 0; i < mvCurPts.size(); ++i)
    // {
    //     if (mmIDToMapPoint.count(mvIds[i]))
    //         cnt_after_tri++;
    //     else
    //         cnt_after_untri++;
    // }
    // std::cout << "[DEBUG-Tri] 本次新三角化成功: " << cnt_new_triangulated << " 个点" << std::endl;
    // std::cout << "[DEBUG-Tri] 三角化失败总计: " << cnt_tri_failed << " 个点" << std::endl;
    // std::cout << "[DEBUG-Tri]   - 视差太小: " << cnt_fail_disparity << std::endl;
    // std::cout << "[DEBUG-Tri]   - SVD奇异值比过大: " << cnt_fail_svd_ratio << std::endl;
    // std::cout << "[DEBUG-Tri]   - 深度异常: " << cnt_fail_depth << std::endl;
    // std::cout << "[DEBUG-Tri]   - 重投影误差过大: " << cnt_fail_reproj << std::endl;
    // std::cout << "[DEBUG-Tri] 三角化后: 已三角化=" << cnt_after_tri
    //           << ", 仍未三角化=" << cnt_after_untri << std::endl;
    // std::cout << "[DEBUG-Tri] === 三角化结束 ===" << std::endl;
    // =========================================
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
    if ((int)mvCurPts.size() >= mMaxCnt)
    {
        return;
    }
    if (mMask.empty() || mMask.rows != img.rows || mMask.cols != img.cols)
    {
        mMask = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(255));
    }
    int width = img.cols;
    int height = img.rows;
    const int GRID_WIDTH_TARGET = 150;
    const int GRID_HEIGHT_TARGET = 150;
    int GRID_COLS = std::max(2, width / GRID_WIDTH_TARGET);
    int GRID_ROWS = std::max(2, height / GRID_HEIGHT_TARGET);
    int grid_width = width / GRID_COLS;
    int grid_height = height / GRID_ROWS;
    int total_grids = GRID_ROWS * GRID_COLS;
    int max_per_grid = (mMaxCnt + total_grids - 1) / total_grids;
    if (max_per_grid < 1)
        max_per_grid = 1;
    std::vector<std::vector<int>> grid_counts(GRID_ROWS, std::vector<int>(GRID_COLS, 0));
    for (const auto &pt : mvCurPts)
    {
        int c = static_cast<int>(pt.x / grid_width);
        int r = static_cast<int>(pt.y / grid_height);
        if (c >= GRID_COLS)
            c = GRID_COLS - 1;
        if (r >= GRID_ROWS)
            r = GRID_ROWS - 1;
        if (c >= 0 && r >= 0)
        {
            grid_counts[r][c]++;
        }
    }
    for (int r = 0; r < GRID_ROWS; ++r)
    {
        for (int c = 0; c < GRID_COLS; ++c)
        {
            int global_needed = mMaxCnt - (int)mvCurPts.size();
            if (global_needed <= 0)
            {
                return;
            }
            int needed = max_per_grid - grid_counts[r][c];
            needed = std::min(needed, global_needed);
            if (needed <= 0)
            {
                continue;
            }
            int x = c * grid_width;
            int y = r * grid_height;
            int w = (c == GRID_COLS - 1) ? (width - x) : grid_width;
            int h = (r == GRID_ROWS - 1) ? (height - y) : grid_height;
            cv::Rect roi(x, y, w, h);
            cv::Mat sub_img = img(roi);
            cv::Mat sub_mask = mMask(roi);
            std::vector<cv::Point2f> nPts;
            cv::goodFeaturesToTrack(sub_img, nPts, needed, 0.01, mMinDist, sub_mask);
            if (!nPts.empty())
            {
                cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.01);
                cv::cornerSubPix(sub_img, nPts, cv::Size(5, 5), cv::Size(-1, -1), criteria);
                for (const auto &pt : nPts)
                {
                    cv::Point2f global_pt(pt.x + x, pt.y + y);
                    if (mMask.at<uchar>(global_pt) == 255)
                    {
                        mvCurPts.push_back(global_pt);
                        mvIds.push_back(mNextId++);
                        mvTrackCnt.push_back(1);
                        cv::circle(mMask, global_pt, mMinDist, 0, -1);
                        if ((int)mvCurPts.size() >= mMaxCnt)
                        {
                            return;
                        }
                    }
                }
            }
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