#include "utils/Camera.h"
#include "utils/Parameters.h"

Camera::Camera()
{
    // 默认直接利用全局解析好的静态 Parameters 变量进行初始化
    m_fx = Parameters::fx;
    m_fy = Parameters::fy;
    m_cx = Parameters::cx;
    m_cy = Parameters::cy;

    m_k1 = Parameters::k1;
    m_k2 = Parameters::k2;
    m_p1 = Parameters::p1;
    m_p2 = Parameters::p2;
}

Camera::Camera(double fx, double fy, double cx, double cy,
               double k1, double k2, double p1, double p2)
    : m_fx(fx), m_fy(fy), m_cx(cx), m_cy(cy),
      m_k1(k1), m_k2(k2), m_p1(p1), m_p2(p2)
{
}

// 投影公式：u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy
// 暂不考虑去畸变，因为你的 KITTI 图像在输入前通常已经由驱动或数据集去过畸变
Eigen::Vector2d Camera::worldToSpace(const Eigen::Vector3d &p_cam) const
{
    double inv_z = 1.0 / p_cam.z();
    double u = m_fx * p_cam.x() * inv_z + m_cx;
    double v = m_fy * p_cam.y() * inv_z + m_cy;
    return Eigen::Vector2d(u, v);
}

// 投影公式（OpenCV 点格式输出，方便画图或对接 OpenCV 函数）
cv::Point2f Camera::worldToSpaceCv(const Eigen::Vector3d &p_cam) const
{
    double inv_z = 1.0 / p_cam.z();
    float u = static_cast<float>(m_fx * p_cam.x() * inv_z + m_cx);
    float v = static_cast<float>(m_fy * p_cam.y() * inv_z + m_cy);
    return cv::Point2f(u, v);
}

// 反投影公式：X_norm = (u - cx) / fx, Y_norm = (v - cy) / fy
Eigen::Vector3d Camera::spaceToWorld(const Eigen::Vector2d &p_2d) const
{
    double x_norm = (p_2d.x() - m_cx) / m_fx;
    double y_norm = (p_2d.y() - m_cy) / m_fy;
    return Eigen::Vector3d(x_norm, y_norm, 1.0);
}

// 反投影公式（OpenCV 点格式输入）
Eigen::Vector3d Camera::spaceToWorldCv(const cv::Point2f &p_2d) const
{
    double x_norm = (p_2d.x - m_cx) / m_fx;
    double y_norm = (p_2d.y - m_cy) / m_fy;
    return Eigen::Vector3d(x_norm, y_norm, 1.0);
}