#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>
#include <map>

class Tracking {
public:
    Tracking();
    ~Tracking() = default;

    // 核心主入口：接收左右目图像、时间戳，传出可视化画布，以及三角化计算出的 3D 世界空间点云
    Eigen::Isometry3d GrabImageStereo(const cv::Mat& imLeft, const cv::Mat& imRight, 
                                      const double &timestamp, cv::Mat& imgTrack,
                                      std::vector<Eigen::Vector3d>& vWorldPoints);

private:
    // VINS风格核心：设置均匀化掩膜并对老特征点进行非极大值抑制（NMS）操作
    void SetMask(const cv::Mat& img);

    // VINS风格核心：补充提取新特征点并分配全局唯一ID
    void AddNewFeatures(const cv::Mat& img);

    // 几何核心：使用 DLT 算法和 SVD 奇异值分解对单个双目特征点进行三角化深度计算
    bool TriangulateStereo(const cv::Point2f& ptLeft, const cv::Point2f& ptRight, 
                           const Eigen::Matrix4d& T_w_c0, const Eigen::Matrix4d& T_w_c1, 
                           Eigen::Vector3d& P_w);

    // 辅助工具：检查特征点是否在图像边界内
    bool InBorder(const cv::Point2f &pt, int cols, int rows);

    // 辅助工具：计算两点之间的欧式距离
    double Distance(const cv::Point2f &pt1, const cv::Point2f &pt2);

private:
    bool mIsInitialized;                // 系统是否初始化成功
    cv::Mat mPrevImg;                   // 上一帧的左目灰度图像
    
    // VINS-Fusion 风格前端维护的数据容器
    std::vector<cv::Point2f> mvCurPts;   // 当前帧左目图像中的特征像素坐标
    std::vector<cv::Point2f> mvPrevPts;  // 上一帧左目图像中的特征像素坐标
    std::vector<int> mvIds;              // 每个特征点对应的全局唯一 ID
    std::vector<int> mvTrackCnt;         // 每个特征点被成功追踪的次数（寿命）
    
    std::map<int, cv::Point2f> mInversePrevPtsMap; // 轨迹画图映射

    cv::Mat mMask;                       // 均匀化均衡掩膜
    int mNextId;                         // 全局特征点 ID 累加计数器
};

#endif // TRACKING_H