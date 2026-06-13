#pragma once
#include <vector>
#include <opencv2/core/core.hpp>
#include "map.h" // 🌟 引入核心 Map 结构

class KeyframeSelector
{
public:
    KeyframeSelector(double fx, double fy, double cx, double cy, double baseline, int img_width, int img_height);
    ~KeyframeSelector() = default;

    void setParameters(double near_depth_thresh = 10.0, double loss_ratio_thresh = 0.15, int min_near_points = 30,
                       double min_avg_disparity = 15.0, double trans_base_ratio = 0.5, int grid_cell_rows = 6,
                       int grid_cell_cols = 8, double coverage_ratio = 0.35, double redundancy_ratio = 0.85);

    // 🌟 接口优化：直接接收大地图指针，内部执行线程安全的近点生命周期判断
    bool decide(const cv::Mat &curr_rvec, const cv::Mat &curr_tvec,
                const std::vector<cv::Point2f> &curr_keypoints_2d,
                const std::vector<int> &matched_mp_indices,
                std::shared_ptr<Map> pMap, // 🌟 关键替换项
                const Keyframe *last_keyframe,
                const std::vector<Keyframe> &window_keyframes,
                bool is_first_frame = false);

private:
    bool isNearPoint(const cv::Point3f &point_world, const cv::Mat &R_curr, const cv::Mat &t_curr) const;
    double computeAverageDisparity(const Keyframe &last_kf, const std::vector<cv::Point2f> &curr_kps, const std::vector<int> &matched_ids) const;
    double computeGridCoverage(const std::vector<cv::Point2f> &keypoints_2d) const;
    double computeRedundancyRatio(const std::vector<int> &matched_mp_indices, const std::vector<Keyframe> &window_keyframes) const;

    double fx_, fy_, cx_, cy_, baseline_;
    int img_width_, img_height_;
    double near_depth_thresh_, loss_ratio_thresh_, min_avg_disparity_, trans_base_ratio_, coverage_ratio_, redundancy_ratio_;
    int min_near_points_, grid_rows_, grid_cols_;
};