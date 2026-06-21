#ifndef FEATUREDETECTOR_H
#define FEATUREDETECTOR_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <map>
#include <memory>

class MapPoint;
class Map;

class FeatureDetector {
public:
    FeatureDetector(int maxCnt, int minDist, bool flowBack);
    ~FeatureDetector() = default;

    // 1. 光流追踪与边界校验
    void TrackFeaturesLK(const cv::Mat &prevImg, const cv::Mat &currImg);
    
    // 2. 位姿解算与 Inlier 过滤（内部维护 mmIDToMapPoint）
    bool EstimatePosePnP(
        std::map<int, std::shared_ptr<MapPoint>> &mmIDToMapPoint,
        double fx, double fy, double cx, double cy,
        double k1, double k2, double p1, double p2,
        Eigen::Isometry3d &currentPose);

    // 3. 双目匹配与新路标三角化
    void TriangulateNewPoints(
        const cv::Mat &grayLeft, const cv::Mat &grayRight,
        const Eigen::Isometry3d &currentPose,
        const Eigen::Matrix4d &bodyTCam0, const Eigen::Matrix4d &bodyTCam1,
        double fx, double fy, double cx, double cy,
        std::map<int, std::shared_ptr<MapPoint>> &mmIDToMapPoint,
        std::shared_ptr<Map> mpMap, bool isKeyFrame,
        std::vector<Eigen::Vector3d> &vWorldPoints, cv::Mat &imgTrack);

    // 4. 辅助掩码管理
    void SetMask(int rows, int cols);
    void AddNewFeatures(const cv::Mat &img);

    // 5. 状态同步更新
    void UpdatePreviousStatus(const cv::Mat &grayLeft);

public:
    // 所有特征容器彻底从 Tracking 中剥离
    std::vector<cv::Point2f> mvCurPts;
    std::vector<cv::Point2f> mvPrevPts;
    std::vector<int> mvIds;
    std::vector<int> mvTrackCnt;
    std::map<int, cv::Point2f> mInversePrevPtsMap;
    cv::Mat mMask;

private:
    bool InBorder(const cv::Point2f &pt, int cols, int rows);
    double Distance(const cv::Point2f &pt1, const cv::Point2f &pt2);

    int mMaxCnt;
    int mMinDist;
    bool mFlowBack;
    int mNextId;
};

#endif // FEATUREDETECTOR_H