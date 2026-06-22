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
    // =========================================================================
    // 【优化一：全局容量提前熔断】
    // 如果当前光流追踪保留下来的老特征点数量已经达到了设定上限，直接收工，不再浪费算力
    // =========================================================================
    if ((int)mvCurPts.size() >= mMaxCnt) {
        return;
    }

    // 首帧冷启动或图像尺寸极端变更时的安全防御
    if (mMask.empty() || mMask.rows != img.rows || mMask.cols != img.cols) {
        mMask = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(255));
    }

    int width = img.cols;
    int height = img.rows;

    // =========================================================================
    // 【优化二：完全自适应物理网格划分】
    // 工业级经验值：设定单个网格的理想物理尺寸目标为 150x150 像素。
    // 这样无论以后换成 VGA(640x480)、720P、1080P 还是 KITTI 图像，
    // 算法都能全自动推导出最匹配当前视场的行、列等分比例（至少划分为 2x2 网格）。
    // =========================================================================
    const int GRID_WIDTH_TARGET = 150;
    const int GRID_HEIGHT_TARGET = 150;

    int GRID_COLS = std::max(2, width / GRID_WIDTH_TARGET);
    int GRID_ROWS = std::max(2, height / GRID_HEIGHT_TARGET);

    int grid_width = width / GRID_COLS;
    int grid_height = height / GRID_ROWS;

    // =========================================================================
    // 【优化三：向上取整扩容算法】
    // 使用开销极低的位算公式实现向上取整：(A + B - 1) / B 
    // 确保总网格容量能完美覆盖 mMaxCnt（例如 300 点 / 24网格 = 12.5 -> 向上取整为 13）。
    // 宁可让每个网格的理论容量富余，也绝不允许因整型除法向下取整导致总特征点数缩水。
    // =========================================================================
    int total_grids = GRID_ROWS * GRID_COLS;
    int max_per_grid = (mMaxCnt + total_grids - 1) / total_grids;
    if (max_per_grid < 1) max_per_grid = 1;

    // 1. 统计当前留在各个自适应网格内的已有特征点数量
    std::vector<std::vector<int>> grid_counts(GRID_ROWS, std::vector<int>(GRID_COLS, 0));
    for (const auto &pt : mvCurPts) {
        int c = static_cast<int>(pt.x / grid_width);
        int r = static_cast<int>(pt.y / grid_height);
        
        // 极值边界越界防御
        if (c >= GRID_COLS) c = GRID_COLS - 1;
        if (r >= GRID_ROWS) r = GRID_ROWS - 1;
        if (c >= 0 && r >= 0) {
            grid_counts[r][c]++;
        }
    }
    
    // 2. 逐个网格进行局域内精准补充特征
    for (int r = 0; r < GRID_ROWS; ++r) {
        for (int c = 0; c < GRID_COLS; ++c) {
            
            // =========================================================================
            // 【优化四：双重锁定与全局缺口控制】
            // 在扫描每个网格前，动态计算当前全局还差多少个点。
            // 即使局部网格内部很不饱和，它本次能补充的数量也绝对不能越界超过全局总缺口。
            // =========================================================================
            int global_needed = mMaxCnt - (int)mvCurPts.size();
            if (global_needed <= 0) {
                return; // 全局已经攒够了所需的特征点（如 300个），直接退出大循环
            }
            
            int needed = max_per_grid - grid_counts[r][c];
            needed = std::min(needed, global_needed); // 双重锁定，取二者极小值
            if (needed <= 0) {
                continue; // 当前网格已饱满，跳过
            }
            
            // 计算当前自适应网格在大图中的物理感兴趣区域（ROI），并处理右侧及下边界余数
            int x = c * grid_width;
            int y = r * grid_height;
            int w = (c == GRID_COLS - 1) ? (width - x) : grid_width;
            int h = (r == GRID_ROWS - 1) ? (height - y) : grid_height;
            
            cv::Rect roi(x, y, w, h);
            cv::Mat sub_img = img(roi);
            cv::Mat sub_mask = mMask(roi); // 裁剪共享子掩码块，完美继承已有老点的全局抑制圈
            
            std::vector<cv::Point2f> nPts;
            // 仅仅在局部网格块内提点，耗时极低
            cv::goodFeaturesToTrack(sub_img, nPts, needed, 0.01, mMinDist, sub_mask);
            
            if (!nPts.empty()) {
                // 3. 执行局域亚像素级精度细化
                cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.01);
                cv::cornerSubPix(sub_img, nPts, cv::Size(5, 5), cv::Size(-1, -1), criteria);
                
                // 4. 将子图像局部坐标还原为全局大图坐标并收集
                for (const auto &pt : nPts) {
                    cv::Point2f global_pt(pt.x + x, pt.y + y);
                    
                    if (mMask.at<uchar>(global_pt) == 255) {
                        mvCurPts.push_back(global_pt);
                        mvIds.push_back(mNextId++);
                        mvTrackCnt.push_back(1);
                        
                        // 实时在全局大掩码上刷圈，防御同网格内后续点或相邻网格提点时靠得太近
                        cv::circle(mMask, global_pt, mMinDist, 0, -1);
                        
                        // =========================================================================
                        // 【优化五：单点压入实时熔断】
                        // 随着特征点一个一个压入，一旦在循环内部某一刻刚好顶满了设定的总点数上限，
                        // 瞬间熔断终止所有程序，防止后面的网格继续超量提取，精确控制特征点总量。
                        // =========================================================================
                        if ((int)mvCurPts.size() >= mMaxCnt) {
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