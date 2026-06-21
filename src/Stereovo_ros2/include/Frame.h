#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <memory>

// 前向声明，避免循环包含头文件
class MapPoint;
class Camera;

class Frame
{
public:
    Frame() = default;

    // 核心构造函数：传入左右目图像、时间戳以及相机内参模型
    Frame(const cv::Mat &left, const cv::Mat &right, double timestamp, std::shared_ptr<Camera> pCamera);
    ~Frame() = default;

    // 辅助函数：判断一个 3D 空间点投影到当前帧上时，是否落在图像边界内
    bool isInFrustum(const Eigen::Vector3d &p_world, float boundary = 0.0);

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
    std::vector<cv::Point2f> mvKeys;
    std::vector<cv::Point2f> mvKeysUn;
    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;

    // --- 位姿数据 ---
    Eigen::Isometry3d mTcw = Eigen::Isometry3d::Identity();
};

#endif // FRAME_H