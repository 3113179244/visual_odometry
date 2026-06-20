#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <memory>

class MapPoint; // 前向声明
class Camera;   // 前向声明

class Frame {
public:
    Frame() = default;
    
    // 核心构造函数：传入左右目图像、时间戳以及相机内参模型
    Frame(const cv::Mat& left, const cv::Mat& right, double timestamp, std::shared_ptr<Camera> pCamera);
    ~Frame() = default;

    // 辅助函数：判断一个 3D 空间点投影到当前帧上时，是否落在图像边界内
    bool isInFrustum(const Eigen::Vector3d& p_world, float boundary = 0.0);

    // 辅助函数：对左目提取出的 mvKeys 像素角点进行去畸变校正
    void UndistortKeyPoints();

public:
    // 基础属性
    double mTimeStamp;
    cv::Mat mImgLeft;
    cv::Mat mImgRight;
    std::shared_ptr<Camera> mpCamera; // 当前帧持有的相机内参模型

    // 唯一的帧 ID，方便后续后端和关键帧检索
    unsigned long mId;
    static unsigned long nNextId;

    // --- 特征点数据 ---
    // 当前帧左目图像中追踪/检测到的特征点 2D 像素坐标
    std::vector<cv::Point2f> mvKeys;
    
    // 去畸变校正后的特征点 2D 坐标（用于更精准地进行 PnP 位姿解算）
    std::vector<cv::Point2f> mvKeysUn;

    // 每个特征点对应的 3D 地图点（MapPoint）智能指针
    // 如果该点尚未三角化，或者被光流追踪丢了，则对应位置存储为 nullptr
    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;

    // --- 位姿数据 ---
    // 当前相机相对于世界坐标系的位姿 Tcw (从世界系变换到相机系)
    Eigen::Isometry3d mTcw = Eigen::Isometry3d::Identity();
};

#endif // FRAME_H