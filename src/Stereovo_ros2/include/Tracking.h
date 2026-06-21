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
    using RenderCallback = std::function<void(
        double timestamp, const cv::Mat &feat_img,
        const std::vector<Eigen::Vector3d> &vWorldPoints,
        const std::vector<Eigen::Vector3d> &vKFPositions,
        const Eigen::Isometry3d &Tcw,
        const std::vector<cv::Point2f> &curPts, const std::vector<int> &ids)>;

    Tracking(std::shared_ptr<Map> pMap);
    ~Tracking();

    void RegisterCallback(RenderCallback cb);
    void FeedStereoImages(const cv::Mat &imLeft, const cv::Mat &imRight, double timestamp);

private:
    void TrackLoop();
    Eigen::Isometry3d ProcessStereo(const cv::Mat &imLeft, const cv::Mat &imRight,
                                    const double &timestamp, cv::Mat &imgTrack,
                                    std::vector<Eigen::Vector3d> &vWorldPoints,
                                    std::vector<Eigen::Vector3d> &vKFPositions);
    bool NeedNewKeyFrame();

private:
    bool mIsInitialized;
    cv::Mat mPrevImg;
    Eigen::Isometry3d mCurrentPose;

    // 核心解耦组件
    std::unique_ptr<FeatureDetector> mpFeatureDetector;
    std::shared_ptr<Map> mpMap;
    std::map<int, std::shared_ptr<MapPoint>> mmIDToMapPoint;
    std::map<int, cv::Point2f> mvpPrevKFPointsMap;
    unsigned long mNextKFId;

    // 线程多队列
    std::thread mTrackThread; bool mIsRunning; std::mutex mMutexBuf; std::condition_variable mCondBuf;
    std::queue<cv::Mat> mLeftBuf; std::queue<cv::Mat> mRightBuf; std::queue<double> mTimeBuf;
    RenderCallback mRenderCb;

    // 仅保留数据流控制所必需的配置
    double mFx, mFy, mCx, mCy, mK1, mK2, mP1, mP2, mKeyframeParallax;
    Eigen::Matrix4d mBodyTCam0; Eigen::Matrix4d mBodyTCam1;
};

#endif // TRACKING_H