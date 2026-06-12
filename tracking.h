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

class Tracking {
public:
    Tracking(StereoVO* vo, std::shared_ptr<Map> pMap, LocalMapping* local_mapper, KeyframeSelector* selector);
    ~Tracking() = default;

    // 核心接口：吃进双目图像，吐出世界系位姿
    Sophus::SE3d GrabImageStereo(const cv::Mat& img_curr_l, const cv::Mat& img_curr_r, int frame_id);

    // 供 Viewer 线程读取的可视化图像
    cv::Mat GetVizImage();

private:
    StereoVO* mpVO;
    std::shared_ptr<Map> mpMap;
    LocalMapping* mpLocalMapper;
    KeyframeSelector* mpSelector;

    // 状态机与位姿累计
    bool mbInitialized;
    Sophus::SE3d mGlobalPose;
    Sophus::SE3d mLastDeltaT;

    // 滑动窗口副本与关键帧状态
    std::vector<std::shared_ptr<Keyframe>> mWindowKeyframes;
    std::shared_ptr<Keyframe> mpLastKeyframe;

    // 上一帧的历史数据 (用于 F2F 追踪)
    cv::Mat mImgPrevL, mImgPrevR;
    std::vector<cv::KeyPoint> mKpsPrevL, mKpsPrevR;
    cv::Mat mDescPrevL, mDescPrevR;
    std::vector<cv::Point3f> mPts3dPrev;
    std::vector<int> mPrevMpIndices;

    // 可视化相关
    std::mutex mMutexViz;
    cv::Mat mImgViz;
};