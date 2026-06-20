#ifndef CAMERA_H
#define CAMERA_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

class Camera {
public:
    // 构造函数：默认会从 Parameters 静态变量中自动初始化相机参数
    Camera();
    
    // 自定义构造函数：允许手动传入特定内参初始化
    Camera(double fx, double fy, double cx, double cy, 
           double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0);
           
    ~Camera() = default;

    // 1. 投影：将相机坐标系下的 3D 点 (X, Y, Z) 投影到 2D 像素平面 (u, v)
    Eigen::Vector2d worldToSpace(const Eigen::Vector3d& p_cam) const;
    cv::Point2f worldToSpaceCv(const Eigen::Vector3d& p_cam) const;

    // 2. 反投影：将 2D 像素点 (u, v) 转换为归一化相机平面上的 3D 点 (X/Z, Y/Z, 1)
    Eigen::Vector3d spaceToWorld(const Eigen::Vector2d& p_2d) const;
    Eigen::Vector3d spaceToWorldCv(const cv::Point2f& p_2d) const;

public:
    // 相机内参
    double m_fx, m_fy, m_cx, m_cy;
    // 畸变参数 (k1, k2: 径向畸变; p1, p2: 切向畸变)
    double m_k1, m_k2, m_p1, m_p2; 
};

#endif // CAMERA_H