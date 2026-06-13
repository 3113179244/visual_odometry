#pragma once
#include <vector>
#include <deque>
#include <memory>
#include <shared_mutex>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include "map.h"

class SlidingWindow
{
public:
    // 🌟 核心修改：直接接收具体的标定数值，杜绝 cv::Mat 隐式类型转换灾难
    SlidingWindow(int window_size, double fx, double fy, double cx, double cy, double baseline,
                  int img_width, int img_height, std::shared_ptr<Map> pMap);
    ~SlidingWindow() = default;

    void addKeyframe(const Keyframe &kf);
    void optimize();

private:
    Sophus::SE3d cvToSophus(const cv::Mat &rvec, const cv::Mat &tvec) const;
    void sophusToCv(const Sophus::SE3d &T, cv::Mat &rvec, cv::Mat &tvec) const;

    void buildBAProblem(std::vector<std::tuple<int, int, cv::Point2f>> &observations,
                        std::vector<Sophus::SE3d> &frame_poses,
                        std::vector<Eigen::Vector3d> &points) const;

    void updateOptimizedResults(const std::vector<Sophus::SE3d> &opt_poses,
                                const std::vector<Eigen::Vector3d> &opt_points,
                                const std::vector<int> &point_ids_in_window);

    int window_size_;
    double fx_, fy_, cx_, cy_; // 🌟 改用显式双精度变量存储内参
    double baseline_;
    int img_width_, img_height_;
    std::shared_ptr<Map> mpMap;
    std::deque<Keyframe> keyframes_;
};