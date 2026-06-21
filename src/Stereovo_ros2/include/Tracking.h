#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>
#include <map>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <functional>

// 前向声明 SLAM 核心组件
class Map;
class MapPoint;
class KeyFrame;

class Tracking
{
public:
    // 回调函数类型：时间戳, 渲染后的特征图, 三角化出的3D世界点云, 全局所有关键帧的3D世界坐标, 当前帧位姿Tcw
    using RenderCallback = std::function<void(
        double timestamp,
        const cv::Mat &feat_img,
        const std::vector<Eigen::Vector3d> &vWorldPoints,
        const std::vector<Eigen::Vector3d> &vKFPositions,
        const Eigen::Isometry3d &Tcw,
        const std::vector<cv::Point2f> &curPts, 
        const std::vector<int> &ids             
    )>;

    // 构造函数传入全局地图组件指针，实现数据固化持久化
    Tracking(std::shared_ptr<Map> pMap);
    ~Tracking();

    // 注册发布回调函数
    void RegisterCallback(RenderCallback cb);

    // 供外部 ROS2 节点快速无阻碍塞入图像接口 (生产者入口)
    void FeedStereoImages(const cv::Mat &imLeft, const cv::Mat &imRight, double timestamp);

private:
    // Tracking 内部常驻处理线程循环 (消费者入口)
    void TrackLoop();

    // 核心光流追踪与解算业务逻辑（包含 PnP、关键帧筛选与地图点长期关联）
    Eigen::Isometry3d ProcessStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                                    const double &timestamp, cv::Mat &imgTrack,
                                    std::vector<Eigen::Vector3d> &vWorldPoints,
                                    std::vector<Eigen::Vector3d> &vKFPositions);

    // 判定当前帧是否应该被筛选为关键帧
    bool NeedNewKeyFrame();

    void SetMask(const cv::Mat &img);
    void AddNewFeatures(const cv::Mat &img);
    bool TriangulateStereo(const cv::Point2f &ptLeft, const cv::Point2f &ptRight,
                           const Eigen::Matrix4d &T_w_c0, const Eigen::Matrix4d &T_w_c1,
                           Eigen::Vector3d &P_w);
    bool InBorder(const cv::Point2f &pt, int cols, int rows);
    double Distance(const cv::Point2f &pt1, const cv::Point2f &pt2);

private:
    bool mIsInitialized;
    cv::Mat mPrevImg;

    // 特征容器
    std::vector<cv::Point2f> mvCurPts;
    std::vector<cv::Point2f> mvPrevPts;
    std::vector<int> mvIds;
    std::vector<int> mvTrackCnt;
    std::map<int, cv::Point2f> mInversePrevPtsMap;
    cv::Mat mMask;
    int mNextId;

    // 关键帧与全局地图点管理的核心数据结构
    std::shared_ptr<Map> mpMap;                              // 全局地图指针
    std::map<int, std::shared_ptr<MapPoint>> mmIDToMapPoint; // 维护全局唯一的 特征点ID 到 地图点智能指针 的寿命映射

    std::map<int, cv::Point2f> mvpPrevKFPointsMap; // 记录【上一个关键帧】中所有特征点的2D像素坐标（用于计算平均视差）
    unsigned long mNextKFId;                       // 关键帧 ID 累加计数器
    Eigen::Isometry3d mCurrentPose;                // 当前帧的位姿 Tcw

    // 线程安全与异步队列相关
    std::thread mTrackThread;
    bool mIsRunning;
    std::mutex mMutexBuf;
    std::condition_variable mCondBuf;

    std::queue<cv::Mat> mLeftBuf;
    std::queue<cv::Mat> mRightBuf;
    std::queue<double> mTimeBuf;

    RenderCallback mRenderCb;

    // =========================================================
    // 【方案一新增】本地化参数缓存，彻底杜绝异步线程高频读取引发的数据竞争
    // =========================================================
    bool mFlowBack;
    double mFx, mFy, mCx, mCy;
    double mK1, mK2, mP1, mP2;
    double mKeyframeParallax;
    int mMaxCnt;
    int mMinDist;
    Eigen::Matrix4d mBodyTCam0;
    Eigen::Matrix4d mBodyTCam1;
};

#endif // TRACKING_H