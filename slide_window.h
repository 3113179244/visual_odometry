#pragma once
#include <vector>
#include <deque>
#include <memory>
#include <shared_mutex>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include "map.h"

class SlidingWindow {
public:
    // 🌟 修改：增加图像宽度、高度参数，用于检查 2D 观测点是否在图像内
    SlidingWindow(int window_size, const cv::Mat& K, double baseline, 
                  int img_width, int img_height, std::shared_ptr<Map> pMap);
    ~SlidingWindow() = default;

    void addKeyframe(const Keyframe& kf);
    void optimize();

private:
    Sophus::SE3d cvToSophus(const cv::Mat& rvec, const cv::Mat& tvec) const;
    void sophusToCv(const Sophus::SE3d& T, cv::Mat& rvec, cv::Mat& tvec) const;
    
    void buildBAProblem(std::vector<std::tuple<int, int, cv::Point2f>>& observations,
                        std::vector<Sophus::SE3d>& frame_poses,
                        std::vector<Eigen::Vector3d>& points) const;
                        
    void updateOptimizedResults(const std::vector<Sophus::SE3d>& opt_poses,
                               const std::vector<Eigen::Vector3d>& opt_points,
                               const std::vector<int>& point_ids_in_window);

    int window_size_;
    cv::Mat K_;
    double baseline_;
    int img_width_, img_height_;          // 🌟 新增：图像尺寸
    std::shared_ptr<Map> mpMap;
    std::deque<Keyframe> keyframes_;
};