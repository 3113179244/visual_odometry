#include "Frame.h"
#include "MapPoint.h"
#include "Camera.h"

// 初始化静态成员变量
unsigned long Frame::nNextId = 0;

// 优化：在构造函数中使用 std::move 转移 cv::Mat 矩阵头，不增加引用计数，开销为 0ms
Frame::Frame(const cv::Mat &left, const cv::Mat &right, double timestamp, std::shared_ptr<Camera> pCamera)
    : mTimeStamp(timestamp),
      mImgLeft(std::move(left)),
      mImgRight(std::move(right)),
      mpCamera(pCamera),
      mId(nNextId++)
{
    // 这里可以补充你的前端特征提取、双目匹配等初始化逻辑
}

// 示例实现：判断 3D 点是否在视锥内
bool Frame::isInFrustum(const Eigen::Vector3d &p_world, float boundary)
{
    if (!mpCamera)
        return false;

    // 1. 将世界坐标系下的点转换到当前相机坐标系下
    Eigen::Vector3d p_cam = mTcw * p_world;

    // 深度必须为正（在相机前方）
    if (p_cam.z() <= 0.0)
        return false;

    // 2. 利用传入的相机模型反投到像素平面
    cv::Point2f pt = mpCamera->worldToSpaceCv(p_cam);

    // 3. 判断是否在图像边界内（考虑边界留白 boundary）
    // 注意：这里的图像尺寸应当从参数或 mpCamera 中获取，这里以 Parameters 为例
    // 如果引入了静态变量，需要确保参数已经加载
    return (pt.x >= boundary && pt.x < mImgLeft.cols - boundary &&
            pt.y >= boundary && pt.y < mImgLeft.rows - boundary);
}

// 辅助函数：对左目特征点进行去畸变
void Frame::UndistortKeyPoints()
{
    if (mvKeys.empty())
        return;

    // 如果 KITTI 数据集本身已经去过畸变（k1,k2,p1,p2 均为 0）
    // 也可以直接把 mvKeys 赋给 mvKeysUn 占位
    if (mpCamera && mpCamera->m_k1 == 0.0 && mpCamera->m_k2 == 0.0)
    {
        mvKeysUn = mvKeys;
        return;
    }

    // 如果有畸变参数，则调用 OpenCV 去畸变
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << mpCamera->m_fx, 0, mpCamera->m_cx,
                            0, mpCamera->m_fy, mpCamera->m_cy,
                            0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(4, 1) << mpCamera->m_k1, mpCamera->m_k2, mpCamera->m_p1, mpCamera->m_p2);

    cv::undistortPoints(mvKeys, mvKeysUn, cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix);
}