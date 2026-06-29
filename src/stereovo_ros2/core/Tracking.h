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

private:
    bool mIsInitialized;
    cv::Mat mPrevImg;
    std::unique_ptr<FeatureDetector> mpFeatureDetector;
    std::shared_ptr<Map> mpMap;
    std::map<int, std::shared_ptr<MapPoint>> mmIDToMapPoint;
    std::map<int, cv::Point2f> mvpPrevKFPointsMap;
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
};

#endif // TRACKING_H