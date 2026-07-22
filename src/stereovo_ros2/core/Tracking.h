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
class LoopClosing;

class Tracking
{
public:
    using RenderCallback = std::function<void(
        double timestamp,
        const cv::Mat &feat_img,
        const std::vector<Eigen::Vector3d> &vWorldPoints,
        const std::vector<Eigen::Vector3d> &vKFPositions,
        const Eigen::Isometry3d &Tcw,
        const std::vector<cv::Point2f> &curPts,
        const std::vector<int> &ids,
        const std::vector<cv::Point2f> &ptsVel)>;

    Tracking(std::shared_ptr<Map> pMap);
    ~Tracking();

    void RegisterCallback(RenderCallback cb);
    void FeedStereoImages(const cv::Mat &imLeft, const cv::Mat &imRight, double timestamp);

    // 💡【新增】：线程同步控制接口
    void RequestPause();
    bool IsPaused();
    void Resume();
    bool IsBufEmpty();

private:
    void TrackLoop();

    Eigen::Isometry3d ProcessStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                                    const double &timestamp, cv::Mat &imgTrack,
                                    std::vector<Eigen::Vector3d> &vWorldPoints,
                                    std::vector<Eigen::Vector3d> &vKFPositions);

    bool NeedNewKeyFrame();
    void BackendLoop();
    void CullMapPoints();
    void CullRedundantKeyFrames();

    // 💡【新增】：内部暂停检查函数
    bool CheckPause();

private:
    bool mIsInitialized;
    cv::Mat mPrevImg;
    std::unique_ptr<FeatureDetector> mpFeatureDetector;
    std::shared_ptr<Map> mpMap;
    std::map<int, std::shared_ptr<MapPoint>> mmIDToMapPoint;
    std::map<int, cv::Point2f> mvpPrevKFPointsMap;
    std::shared_ptr<LoopClosing> mpLoopClosing;
    unsigned long mNextKFId;
    Eigen::Isometry3d mCurrentPose;

    std::thread mTrackThread;
    bool mIsRunning;
    std::mutex mMutexBuf;
    std::condition_variable mCondBuf;

    std::queue<cv::Mat> mLeftBuf;
    std::queue<cv::Mat> mRightBuf;
    std::queue<double> mTimeBuf;

    RenderCallback mRenderCb;

    bool mFlowBack;
    double mFx, mFy, mCx, mCy;
    double mK1, mK2, mP1, mP2;
    double mKeyframeParallax;
    Eigen::Matrix4d mBodyTCam0;
    Eigen::Matrix4d mBodyTCam1;
    double mPrevTime;

    std::thread mBackendThread;
    std::mutex mMutexBackend;
    std::condition_variable mCondBackend;
    bool mNeedOptimize;
    int mnCullCounter = 0;

    // 💡【新增】：线程暂停状态标志与互斥锁
    bool mbPauseRequested = false;
    bool mbPaused = false;
    std::mutex mMutexPause;
};

#endif // TRACKING_H