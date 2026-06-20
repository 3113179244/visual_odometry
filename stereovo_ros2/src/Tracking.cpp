#include "Tracking.h"
#include "MapPoint.h"
#include "Parameters.h" // 包含你之前读入的全局内参和外参
#include <iostream>

Tracking::Tracking() : mIsInitialized(false)
{
  mpMap = std::make_shared<Map>(); // 实例化地图
}

Eigen::Isometry3d Tracking::GrabImageStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
  // 构建当前帧
  mCurrentFrame = Frame(imLeft, imRight, timestamp);

  // 主追踪控制流
  Track();

  // 状态传递：当前帧变成下一帧的“上一帧”
  mLastFrame = mCurrentFrame;
  return mCurrentFrame.mTcw;
}

void Tracking::Track()
{
  // 状态一：如果系统还没初始化（第一帧）
  if (!mIsInitialized)
  {
    // 在第一帧左目图像中提取 GFTT 角点
    cv::goodFeaturesToTrack(mCurrentFrame.mImgLeft, mCurrentFrame.mvKeys, 200, 0.01, 30);

    // 匹配右目，并进行第一次三角化
    TrackRightFrame();
    KeyFrameLoop();

    mIsInitialized = true;
    std::cout << ">>> SLAM 系统初始化完成，第一帧提取了 " << mCurrentFrame.mvKeys.size() << " 个特征点。" << std::endl;
    return;
  }

  // 状态二：正常追踪状态
  // 1. 光流追踪上一帧到当前帧
  TrackLastFrame();

  // 2. 用 PnP 求解当前帧位姿
  EstimatePosePnP();

  // 3. 动态检查：特征点变少时，重新补充角点，匹配右目并补充新地图点
  if (mCurrentFrame.mvKeys.size() < 100)
  {
    KeyFrameLoop();
  }
}

// 核心算法 1：用 LK 光流追踪上一帧左目到当前帧左目
void Tracking::TrackLastFrame()
{
  if (mLastFrame.mvKeys.empty())
    return;

  std::vector<cv::Point2f> next_keys;
  std::vector<uchar> status;
  std::vector<float> err;

  // 稀疏光流追踪
  cv::calcOpticalFlowPyrLK(mLastFrame.mImgLeft, mCurrentFrame.mImgLeft,
                           mLastFrame.mvKeys, next_keys, status, err, cv::Size(21, 21), 3);

  // 筛选追踪成功的点
  for (size_t i = 0; i < status.size(); i++)
  {
    if (status[i])
    {
      mCurrentFrame.mvKeys.push_back(next_keys[i]);
      // 继承上一帧对应的 3D 地图点指针
      mCurrentFrame.mvpMapPoints.push_back(mLastFrame.mvpMapPoints[i]);
    }
  }
}

// 核心算法 2：用 LK 光流追踪当前帧左目到当前帧右目（算双目视差）
void Tracking::TrackRightFrame()
{
  if (mCurrentFrame.mvKeys.empty())
    return;

  std::vector<cv::Point2f> right_keys;
  std::vector<uchar> status;
  std::vector<float> err;

  // 极线约束：因为双目已校正，左目角点在右目对应的位置必然在同一水平行上
  cv::calcOpticalFlowPyrLK(mCurrentFrame.mImgLeft, mCurrentFrame.mImgRight,
                           mCurrentFrame.mvKeys, right_keys, status, err, cv::Size(21, 21), 3);

  // 这里的地图点指针大小要跟特征点保持一致
  mCurrentFrame.mvpMapPoints.resize(mCurrentFrame.mvKeys.size(), nullptr);

  // 获取相机基线 Baseline (从之前 Parameters 读入的 body_T_cam1 中获取平移 X)
  double baseline = std::abs(Parameters::body_T_cam1(0, 3));

  // 遍历特征点，利用视差进行 3D 三角化
  for (size_t i = 0; i < status.size(); i++)
  {
    if (status[i])
    {
      double disparity = mCurrentFrame.mvKeys[i].x - right_keys[i].x;
      if (disparity > 0.0)
      { // 视差必须大于 0
        // 双目三角化公式
        double z = (Parameters::fx * baseline) / disparity;
        double x = (mCurrentFrame.mvKeys[i].x - Parameters::cx) * z / Parameters::fx;
        double y = (mCurrentFrame.mvKeys[i].y - Parameters::cy) * z / Parameters::fy;

        Eigen::Vector3d p_cam(x, y, z);
        // 将相机坐标系下的点，通过当前位姿转换到世界坐标系
        Eigen::Vector3d p_world = mCurrentFrame.mTcw * p_cam;

        // 创建新的地图点并存入地图和当前帧
        auto pMP = std::make_shared<MapPoint>(p_world);
        mpMap->AddMapPoint(pMP);
        mCurrentFrame.mvpMapPoints[i] = pMP;
      }
    }
  }
}

// 核心算法 3：3D-2D 匹配解算相机当前位姿
void Tracking::EstimatePosePnP()
{
  std::vector<cv::Point3f> pts_3d;
  std::vector<cv::Point2f> pts_2d;

  // 筛选出那些既有 2D 像素坐标，又拥有 3D 世界坐标的有效地图点
  for (size_t i = 0; i < mCurrentFrame.mvKeys.size(); i++)
  {
    if (mCurrentFrame.mvpMapPoints[i] != nullptr)
    {
      Eigen::Vector3d p_w = mCurrentFrame.mvpMapPoints[i]->GetWorldPos();
      pts_3d.push_back(cv::Point3f(p_w.x(), p_w.y(), p_w.z()));
      pts_2d.push_back(mCurrentFrame.mvKeys[i]);
    }
  }

  if (pts_3d.size() < 4)
  {
    std::cerr << "警告：有效 3D-2D 匹配点太少，PnP 解算失败！" << std::endl;
    return;
  }

  // 组装相机内参矩阵
  cv::Mat K = (cv::Mat_<double>(3, 3) << Parameters::fx, 0, Parameters::cx,
               0, Parameters::fy, Parameters::cy,
               0, 0, 1);
  cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F); // 假定图像已去畸变

  cv::Mat rvec, tvec;
  // 使用 RANSAC PnP 剔除误匹配，解算旋转向量和平移向量
  bool success = cv::solvePnPRansac(pts_3d, pts_2d, K, distCoeffs, rvec, tvec);

  if (success)
  {
    cv::Mat R;
    // 【核心修正】：去掉 & 符号，直接传入 R。OpenCV 内部会自动将其转换为 OutputArray 引用
    cv::Rodrigues(rvec, R);

    // 将 OpenCV 的矩阵转换并完全赋值给 Eigen 变换矩阵
    Eigen::Matrix3d eigen_R;
    Eigen::Vector3d eigen_t;
    for (int i = 0; i < 3; i++)
    {
      eigen_t(i) = tvec.at<double>(i, 0);
      for (int j = 0; j < 3; j++)
      {
        eigen_R(i, j) = R.at<double>(i, j);
      }
    }

    // 组装完整的 Tcw (从世界系到当前相机系的变换)
    Eigen::Isometry3d Tcw_calculated = Eigen::Isometry3d::Identity();
    Tcw_calculated.prerotate(eigen_R);
    Tcw_calculated.pretranslate(eigen_t);

    // 赋值给当前帧
    mCurrentFrame.mTcw = Tcw_calculated;
  }
}

// 核心算法 4：补充角点并重新触发双目三角化
void Tracking::KeyFrameLoop()
{
  std::vector<cv::Point2f> new_keys;
  // 在左目上重新检测新的 GFTT 角点
  cv::goodFeaturesToTrack(mCurrentFrame.mImgLeft, new_keys, 150, 0.01, 30);

  // 巧妙的去重：如果新检测的角点距离现有追踪点太近，就不要它了
  for (const auto &pt : new_keys)
  {
    bool too_close = false;
    for (const auto &old_pt : mCurrentFrame.mvKeys)
    {
      if (cv::norm(pt - old_pt) < 20.0)
      {
        too_close = true;
        break;
      }
    }
    if (!too_close)
    {
      mCurrentFrame.mvKeys.push_back(pt);
      mCurrentFrame.mvpMapPoints.push_back(nullptr); // 暂时置空，等待匹配
    }
  }

  // 对这批新加进去的特征点，单独做一次右目匹配和三角化
  TrackRightFrame();
}