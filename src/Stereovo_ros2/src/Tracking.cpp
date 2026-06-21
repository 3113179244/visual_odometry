#include "Tracking.h"
#include "Parameters.h"
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>

Tracking::Tracking() : mIsInitialized(false), mNextId(0)
{
}

Eigen::Isometry3d Tracking::GrabImageStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                                            const double &timestamp, cv::Mat &imgTrack,
                                            std::vector<Eigen::Vector3d> &vWorldPoints)
{
    // 1. 确保输入为单通道灰度图
    cv::Mat grayLeft = imLeft;
    if (imLeft.channels() == 3)
    {
        cv::cvtColor(imLeft, grayLeft, cv::COLOR_BGR2GRAY);
    }
    cv::Mat grayRight = imRight;
    if (imRight.channels() == 3)
    {
        cv::cvtColor(imRight, grayRight, cv::COLOR_BGR2GRAY);
    }

    // 初始化彩色画布用于画图查看效果
    cv::cvtColor(grayLeft, imgTrack, cv::COLOR_GRAY2BGR);
    vWorldPoints.clear();

    // 2. 第一帧初始化处理
    if (!mIsInitialized)
    {
        std::cout << ">>> [VINS前端] 接收到第一帧图像，正在提取初始特征点..." << std::endl;
        mvCurPts.clear();
        mvIds.clear();
        mvTrackCnt.clear();

        AddNewFeatures(grayLeft);

        mPrevImg = grayLeft.clone();
        mvPrevPts = mvCurPts;

        mInversePrevPtsMap.clear();
        for (size_t i = 0; i < mvCurPts.size(); ++i)
        {
            mInversePrevPtsMap[mvIds[i]] = mvCurPts[i];
        }

        mIsInitialized = true;
        for (const auto &pt : mvCurPts)
        {
            cv::circle(imgTrack, pt, 2, cv::Scalar(0, 0, 255), 2);
        }
        return Eigen::Isometry3d::Identity();
    }

    // 3. 执行左目图像的前后帧 LK 光流追踪与正反向校验
    if (!mvPrevPts.empty())
    {
        std::vector<cv::Point2f> currPts;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(mPrevImg, grayLeft, mvPrevPts, currPts, status, err, cv::Size(21, 21), 3);

        if (Parameters::FLOW_BACK)
        {
            std::vector<cv::Point2f> reversePts = mvPrevPts;
            std::vector<uchar> reverseStatus;
            cv::calcOpticalFlowPyrLK(grayLeft, mPrevImg, currPts, reversePts, reverseStatus, err, cv::Size(21, 21), 3);
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i] && reverseStatus[i])
                {
                    if (Distance(mvPrevPts[i], reversePts[i]) > 0.5)
                        status[i] = 0;
                }
                else
                {
                    status[i] = 0;
                }
            }
        }

        mvCurPts.clear();
        std::vector<int> keepIds;
        std::vector<int> keepTrackCnt;
        for (size_t i = 0; i < status.size(); i++)
        {
            if (status[i] && InBorder(currPts[i], grayLeft.cols, grayLeft.rows))
            {
                mvCurPts.push_back(currPts[i]);
                keepIds.push_back(mvIds[i]);
                keepTrackCnt.push_back(mvTrackCnt[i]);
            }
        }
        mvIds = keepIds;
        mvTrackCnt = keepTrackCnt;
    }

    for (auto &n : mvTrackCnt)
        n++;

    // 4. 非极大值抑制（NMS）确保特征点均匀分布
    SetMask(grayLeft);

    // 5. 补足新特征点
    AddNewFeatures(grayLeft);

    // 6. 双目匹配与 DLT + SVD 三角化计算深度
    if (!mvCurPts.empty() && !grayRight.empty())
    {
        std::vector<cv::Point2f> mvRightPts;
        std::vector<uchar> stereoStatus;
        std::vector<float> stereoErr;

        // 从当前左图特征点，通过光流强行匹配到右图对应横向位置
        cv::calcOpticalFlowPyrLK(grayLeft, grayRight, mvCurPts, mvRightPts, stereoStatus, stereoErr, cv::Size(21, 21), 3);

        // 获取相机相对于世界系下的绝对变换矩阵（当前暂设定虚拟位姿 Twc = Identity）
        Eigen::Matrix4d T_w_body = Eigen::Matrix4d::Identity(); // 虚拟位姿
        Eigen::Matrix4d T_w_c0 = T_w_body * Parameters::body_T_cam0;
        Eigen::Matrix4d T_w_c1 = T_w_body * Parameters::body_T_cam1;

        for (size_t i = 0; i < mvCurPts.size(); i++)
        {
            if (stereoStatus[i] && InBorder(mvRightPts[i], grayRight.cols, grayRight.rows))
            {
                Eigen::Vector3d P_w;
                // 执行 SVD 求解 DLT 方程
                if (TriangulateStereo(mvCurPts[i], mvRightPts[i], T_w_c0, T_w_c1, P_w))
                {
                    vWorldPoints.push_back(P_w);
                    // 在图上标记成功三角化的特征点（青色圆圈）
                    cv::circle(imgTrack, mvCurPts[i], 4, cv::Scalar(255, 255, 0), 1);
                }
            }
        }
    }

    std::cout << "[DLT几何] 提取点数: " << mvCurPts.size()
              << " | 成功通过 SVD 三角化出 3D 点数: " << vWorldPoints.size() << std::endl;

    // 7. 绘制寿命深度色彩与速度矢量箭头线
    for (size_t i = 0; i < mvCurPts.size(); i++)
    {
        double len = std::min(1.0, 1.0 * mvTrackCnt[i] / 20.0);
        cv::Scalar ptColor = cv::Scalar(255 * (1 - len), 0, 255 * len);
        cv::circle(imgTrack, mvCurPts[i], 2, ptColor, 2);

        auto it = mInversePrevPtsMap.find(mvIds[i]);
        if (it != mInversePrevPtsMap.end())
        {
            cv::arrowedLine(imgTrack, it->second, mvCurPts[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    // 8. 缓存更新
    mPrevImg = grayLeft.clone();
    mvPrevPts = mvCurPts;
    mInversePrevPtsMap.clear();
    for (size_t i = 0; i < mvCurPts.size(); i++)
    {
        mInversePrevPtsMap[mvIds[i]] = mvCurPts[i];
    }

    return Eigen::Isometry3d::Identity();
}

bool Tracking::TriangulateStereo(const cv::Point2f &ptLeft, const cv::Point2f &ptRight,
                                 const Eigen::Matrix4d &T_w_c0, const Eigen::Matrix4d &T_w_c1,
                                 Eigen::Vector3d &P_w)
{
    // A. 像素坐标反投影，利用内参转换为相机平面上的归一化射线坐标
    double x0 = (ptLeft.x - Parameters::cx) / Parameters::fx;
    double y0 = (ptLeft.y - Parameters::cy) / Parameters::fy;

    double x1 = (ptRight.x - Parameters::cx) / Parameters::fx;
    double y1 = (ptRight.y - Parameters::cy) / Parameters::fy;

    // B. 获取从世界系到相机系的投影变换投影矩阵 P = T_c_w = (T_w_c)^-1
    Eigen::Matrix4d T_c0_w = T_w_c0.inverse();
    Eigen::Matrix4d T_c1_w = T_w_c1.inverse();

    // C. 依据 DLT 约束构建超定线性方程组矩阵 A
    Eigen::Matrix4d A;
    A.row(0) = x0 * T_c0_w.row(2) - T_c0_w.row(0);
    A.row(1) = y0 * T_c0_w.row(2) - T_c0_w.row(1);
    A.row(2) = x1 * T_c1_w.row(2) - T_c1_w.row(0);
    A.row(3) = y1 * T_c1_w.row(2) - T_c1_w.row(1);

    // D. 实施 SVD 奇异值分解求解最小二乘解
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X_w = svd.matrixV().col(3);

    // 检查奇异解，防止零除
    if (std::abs(X_w.w()) < 1e-6)
    {
        return false;
    }

    // 齐次坐标非齐次化
    P_w = X_w.head<3>() / X_w.w();

    // E. 【自适应深度自剔除核心】
    // 1. 动态从参数文件中解析左右相机相对于 body 的平移向量
    Eigen::Vector3d t_c0 = Parameters::body_T_cam0.block<3, 1>(0, 3);
    Eigen::Vector3d t_c1 = Parameters::body_T_cam1.block<3, 1>(0, 3);

    // 2. 动态计算双目相机的物理基线长度（两相机光学中心欧氏距离）
    double baseline = (t_c1 - t_c0).norm();

    // 安全防御性机制：如果外参平移配置不正确导致基线过小，则退避到 KITTI 的默认基线 0.53715 米
    if (baseline < 1e-4)
    {
        baseline = 0.53715;
    }

    // 3. 依据参数文件中的相机焦距 fx 和动态基线，计算当前双目下的理论极限可靠测量距离
    // 这里的 1.2 像素表示：当视差小于 1.2 像素时，深度误差呈指数级剧增，该远点不应参与 SLAM 优化
    double max_reliable_depth = (Parameters::fx * baseline) / 1.2;

    // 4. 计算三维点投影回左右相机坐标系后的物理 Z 轴深度
    Eigen::Vector4d P_w_homo(P_w.x(), P_w.y(), P_w.z(), 1.0);
    double depth_cam0 = (T_c0_w * P_w_homo).z();
    double depth_cam1 = (T_c1_w * P_w_homo).z();

    // 5. 实施自适应合理区域筛选（既防负深度外点，又防几何失效的远噪点）
    if (depth_cam0 > 0.0 && depth_cam0 < max_reliable_depth &&
        depth_cam1 > 0.0 && depth_cam1 < max_reliable_depth)
    {
        return true;
    }

    return false;
}

void Tracking::SetMask(const cv::Mat &img)
{
    mMask = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(255));
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cntPtsId;
    for (size_t i = 0; i < mvCurPts.size(); i++)
    {
        cntPtsId.push_back(std::make_pair(mvTrackCnt[i], std::make_pair(mvCurPts[i], mvIds[i])));
    }
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
            cv::circle(mMask, pt, Parameters::MIN_DIST, 0, -1);
        }
    }
}

void Tracking::AddNewFeatures(const cv::Mat &img)
{
    int countToDetect = Parameters::MAX_CNT - (int)mvCurPts.size();
    if (countToDetect > 0)
    {
        std::vector<cv::Point2f> nPts;
        cv::goodFeaturesToTrack(img, nPts, countToDetect, 0.01, Parameters::MIN_DIST, mMask);
        for (const auto &pt : nPts)
        {
            mvCurPts.push_back(pt);
            mvIds.push_back(mNextId++);
            mvTrackCnt.push_back(1);
        }
    }
}

bool Tracking::InBorder(const cv::Point2f &pt, int cols, int rows)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < cols - BORDER_SIZE &&
           BORDER_SIZE <= img_y && img_y < rows - BORDER_SIZE;
}

double Tracking::Distance(const cv::Point2f &pt1, const cv::Point2f &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return std::sqrt(dx * dx + dy * dy);
}