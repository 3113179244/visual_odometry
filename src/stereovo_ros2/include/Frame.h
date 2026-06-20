#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <memory>

class MapPoint; // 前向声明

class Frame {
public:
    Frame() = default;
    Frame(const cv::Mat& left, const cv::Mat& right, double timestamp);
    ~Frame() = default;

public:
    double mTimeStamp;
    cv::Mat mImgLeft;
    cv::Mat mImgRight;

    // 当前帧左目图像中的特征点 2D 像素坐标
    std::vector<cv::Point2f> mvKeys;
    
    // 每个特征点对应的 3D 地图点指针（如果没三角化或追踪丢了，则为 nullptr）
    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;

    // 当前帧相对于世界坐标系的位姿 Tcw (Camera -> World)
    Eigen::Isometry3d mTcw = Eigen::Isometry3d::Identity();
};

#endif // FRAME_H