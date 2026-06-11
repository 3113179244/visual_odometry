#ifndef SLIDE_WINDOW_H
#define SLIDE_WINDOW_H

#include <deque>
#include <vector>
#include <tuple>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include "keyframe_selector.h"
#include "stereo_vo.h"   // 提供 MapPoint 完整定义

/**
 * @brief 滑动窗口管理器，对窗口内关键帧进行局部BA优化
 * 
 * 维护一个固定大小的关键帧队列，每次添加新关键帧后触发优化，
 * 优化窗口内所有关键帧的位姿以及它们观测到的地图点。
 */
class SlidingWindow {
public:
    /**
     * @param window_size      窗口最大容量（关键帧数量）
     * @param K                相机内参矩阵 (3x3)
     * @param baseline         双目基线（米，用于深度阈值等）
     * @param local_map        全局地图点引用（用于读写地图点坐标）
     */
    SlidingWindow(int window_size, const cv::Mat& K, double baseline,
                  std::vector<MapPoint>& local_map);
    
    /**
     * @brief 添加新关键帧到窗口
     * @param kf              新关键帧（包含位姿、特征点、地图点索引）
     * 
     * 若窗口已满则移除最早的关键帧，然后添加新帧，
     * 最后自动触发局部BA优化。
     */
    void addKeyframe(const Keyframe& kf);
    
    /**
     * @brief 主动触发窗口内关键帧的局部BA优化
     * 
     * 优化窗口内所有关键帧的位姿以及它们共视的地图点，
     * 优化后更新关键帧位姿和全局地图点坐标。
     * 注：第一帧的位姿会被固定以消除自由度。
     */
    void optimize();
    
    /** 获取当前窗口内所有关键帧（只读） */
    const std::deque<Keyframe>& getKeyframes() const { return keyframes_; }

private:
    int window_size_;
    cv::Mat K_;                     // 相机内参
    double baseline_;
    std::vector<MapPoint>& local_map_;   // 全局地图点池的引用
    std::deque<Keyframe> keyframes_;     // 窗口内关键帧队列（按时间顺序）
    
    /**
     * @brief 将OpenCV旋转向量+平移转换为Sophus SE3
     */
    Sophus::SE3d cvToSophus(const cv::Mat& rvec, const cv::Mat& tvec) const;
    
    /**
     * @brief 将Sophus SE3转换为OpenCV旋转向量和平移
     */
    void sophusToCv(const Sophus::SE3d& T, cv::Mat& rvec, cv::Mat& tvec) const;
    
    /**
     * @brief 从窗口内关键帧提取所有观测（用于BA）
     * @param observations  输出：每个观测包含(帧索引, 地图点索引, 2D像素坐标)
     * @param frame_poses   输出：每个关键帧的初始位姿（Sophus SE3）
     * @param points        输出：被观测地图点的初始世界坐标（Eigen Vector3d）
     * 
     * 只收集那些至少被窗口内两个关键帧观测到的地图点，以保证BA的约束性。
     */
    void buildBAProblem(
        std::vector<std::tuple<int, int, cv::Point2f>>& observations,
        std::vector<Sophus::SE3d>& frame_poses,
        std::vector<Eigen::Vector3d>& points) const;
    
    /**
     * @brief 将优化后的位姿和地图点写回关键帧和local_map_
     */
    void updateOptimizedResults(const std::vector<Sophus::SE3d>& opt_poses,
                                const std::vector<Eigen::Vector3d>& opt_points,
                                const std::vector<int>& point_ids_in_window);
};

#endif // SLIDE_WINDOW_H