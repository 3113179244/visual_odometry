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
class Map;
class MapPoint;
class FeatureDetector;

class Tracking {
public:
    // 回调函数类型定义
    using RenderCallback = std::function<void(
        double timestamp,
        const cv::Mat &feat_img,
        const std::vector<Eigen::Vector3d> &vWorldPoints,
        const std::vector<Eigen::Vector3d> &vKFPositions,
        const Eigen::Isometry3d &Tcw,
        const std::vector<cv::Point2f> &curPts, 
        const std::vector<int> &ids             
    )>;

    Tracking(std::shared_ptr<Map> pMap);
    ~Tracking();

    void RegisterCallback(RenderCallback cb);
    void FeedStereoImages(const cv::Mat &imLeft, const cv::Mat &imRight, double timestamp);

private:
    // 线程 1：前端主处理线程
    void TrackLoop();

    // 前端核心物理计算管道（与 .cpp 完全对齐）
    Eigen::Isometry3d ProcessStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                                    const double &timestamp, cv::Mat &imgTrack,
                                    std::vector<Eigen::Vector3d> &vWorldPoints,
                                    std::vector<Eigen::Vector3d> &vKFPositions);

    bool NeedNewKeyFrame();

    // 【新增】线程 2：后端 Ceres 优化常驻后台线程
    void BackendLoop();

private:
    bool mIsInitialized;
    cv::Mat mPrevImg;

    // 核心解耦组件
    std::unique_ptr<FeatureDetector> mpFeatureDetector;
    std::shared_ptr<Map> mpMap;                              
    std::map<int, std::shared_ptr<MapPoint>> mmIDToMapPoint; 
    std::map<int, cv::Point2f> mvpPrevKFPointsMap; 
    unsigned long mNextKFId;                       
    Eigen::Isometry3d mCurrentPose;                

    // 前端同步双输入缓存队列
    std::thread mTrackThread;
    bool mIsRunning;
    std::mutex mMutexBuf;
    std::condition_variable mCondBuf;

    std::queue<cv::Mat> mLeftBuf;
    std::queue<cv::Mat> mRightBuf;
    std::queue<double> mTimeBuf;

    RenderCallback mRenderCb;

    // 本地化安全参数缓存
    bool mFlowBack;
    double mFx, mFy, mCx, mCy;
    double mK1, mK2, mP1, mP2;
    double mKeyframeParallax;
    Eigen::Matrix4d mBodyTCam0;
    Eigen::Matrix4d mBodyTCam1;

    // ==========================================
    // 后端异步 Ceres 优化控制信号
    // ==========================================
    std::thread mBackendThread;     
    std::mutex mMutexBackend;       
    std::condition_variable mCondBackend; 
    bool mNeedOptimize;             
};

#endif // TRACKING_H