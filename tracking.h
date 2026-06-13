#pragma once
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <memory>
#include <vector>
#include <mutex>
#include "stereo_vo.h"
#include "map.h"
#include "local_mapping.h"
#include "keyframe_selector.h"

// 定义轻量级结构体，用于记录每一帧的从属优化关系
struct FrameRecord {
    int frame_id;
    int anchor_kf_id;            // 当前帧依赖的关键帧ID
    Sophus::SE3d T_curr_anchor;  // 当前帧相对于该关键帧的相对变换
};

class Tracking {
public:
    Tracking(StereoVO* vo, std::shared_ptr<Map> pMap, LocalMapping* local_mapper, KeyframeSelector* selector);
    ~Tracking() = default;

    Sophus::SE3d GrabImageStereo(const cv::Mat& img_curr_l, const cv::Mat& img_curr_r, int frame_id);

    cv::Mat GetVizImage();

    // 暴露给外部一次性导出的黄金记录容器
    std::vector<FrameRecord> GetTrajectoryRecords() { return mTrajectoryRecords; }

private:
    StereoVO* mpVO;
    std::shared_ptr<Map> mpMap;
    LocalMapping* mpLocalMapper;
    KeyframeSelector* mpSelector;

    bool mbInitialized;
    Sophus::SE3d mGlobalPose;
    Sophus::SE3d mLastDeltaT;

    std::vector<std::shared_ptr<Keyframe>> mWindowKeyframes;
    std::shared_ptr<Keyframe> mpLastKeyframe;

    cv::Mat mImgPrevL, mImgPrevR;
    std::vector<cv::KeyPoint> mKpsPrevL, mKpsPrevR;
    cv::Mat mDescPrevL, mDescPrevR;
    std::vector<cv::Point3f> mPts3dPrev;
    std::vector<int> mPrevMpIndices;

    std::mutex mMutexViz;
    cv::Mat mImgViz;

    // 🌟 轨迹矩阵传导核心容器
    std::vector<FrameRecord> mTrajectoryRecords;
};