// keyframe_selector.cpp
#include "keyframe_selector.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>
#include <set>
#include <map>      // 新增
KeyframeSelector::KeyframeSelector(double fx, double fy, double cx, double cy,
                                   double baseline, int img_width, int img_height)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline),
      img_width_(img_width), img_height_(img_height)
{
    // 设置默认参数
    setParameters();
}

void KeyframeSelector::setParameters(double near_depth_thresh,
                                     double loss_ratio_thresh,
                                     int min_near_points,
                                     double min_avg_disparity,
                                     double trans_base_ratio,
                                     int grid_cell_rows,
                                     int grid_cell_cols,
                                     double coverage_ratio,
                                     double redundancy_ratio)
{
    near_depth_thresh_ = near_depth_thresh;
    loss_ratio_thresh_ = loss_ratio_thresh;
    min_near_points_ = min_near_points;
    min_avg_disparity_ = min_avg_disparity;
    trans_base_ratio_ = trans_base_ratio;
    grid_rows_ = grid_cell_rows;
    grid_cols_ = grid_cell_cols;
    coverage_ratio_ = coverage_ratio;
    redundancy_ratio_ = redundancy_ratio;
}

bool KeyframeSelector::isNearPoint(const cv::Point3f& point_world,
                                   const cv::Mat& R_curr, const cv::Mat& t_curr) const
{
    // 将世界坐标点转换到当前相机坐标系
    cv::Mat point_world_mat = (cv::Mat_<double>(3,1) << point_world.x, point_world.y, point_world.z);
    cv::Mat point_cam = R_curr * point_world_mat + t_curr;
    double depth = point_cam.at<double>(2,0);
    return (depth > 0 && depth < near_depth_thresh_);
}

double KeyframeSelector::computeAverageDisparity(const Keyframe& last_kf,
                                                 const std::vector<cv::Point2f>& curr_kps,
                                                 const std::vector<int>& matched_ids) const
{
    // 构建上一关键帧的地图点索引到其2D坐标的映射
    std::map<int, cv::Point2f> last_kf_map;
    for (size_t i = 0; i < last_kf.map_point_indices.size(); ++i) {
        int mp_idx = last_kf.map_point_indices[i];
        if (mp_idx >= 0) {
            last_kf_map[mp_idx] = last_kf.keypoints_2d[i];
        }
    }
    
    double sum_disparity = 0.0;
    int valid_count = 0;
    
    for (size_t i = 0; i < matched_ids.size() && i < curr_kps.size(); ++i) {
        int mp_idx = matched_ids[i];
        auto it = last_kf_map.find(mp_idx);
        if (it != last_kf_map.end()) {
            cv::Point2f last_pt = it->second;
            cv::Point2f curr_pt = curr_kps[i];
            double dx = last_pt.x - curr_pt.x;
            double dy = last_pt.y - curr_pt.y;
            sum_disparity += std::sqrt(dx*dx + dy*dy);
            valid_count++;
        }
    }
    
    return (valid_count > 0) ? (sum_disparity / valid_count) : 0.0;
}

double KeyframeSelector::computeGridCoverage(const std::vector<cv::Point2f>& keypoints_2d) const
{
    if (grid_rows_ <= 0 || grid_cols_ <= 0) return 0.0;
    
    int total_cells = grid_rows_ * grid_cols_;
    std::vector<bool> covered(total_cells, false);
    
    double cell_w = static_cast<double>(img_width_) / grid_cols_;
    double cell_h = static_cast<double>(img_height_) / grid_rows_;
    
    for (const auto& pt : keypoints_2d) {
        int col = static_cast<int>(pt.x / cell_w);
        int row = static_cast<int>(pt.y / cell_h);
        if (col >= 0 && col < grid_cols_ && row >= 0 && row < grid_rows_) {
            int idx = row * grid_cols_ + col;
            covered[idx] = true;
        }
    }
    
    int covered_count = std::count(covered.begin(), covered.end(), true);
    return static_cast<double>(covered_count) / total_cells;
}

double KeyframeSelector::computeRedundancyRatio(const std::vector<int>& matched_mp_indices,
                                                const std::vector<Keyframe>& window_keyframes) const
{
    if (matched_mp_indices.empty()) return 0.0;
    
    // 统计当前帧观测到的每个地图点是否被窗口内其他关键帧观测到
    // 使用 set 快速去重，并计数被观测到的点
    std::set<int> observed_elsewhere;
    
    for (int mp_idx : matched_mp_indices) {
        if (mp_idx < 0) continue;
        // 遍历窗口内所有关键帧
        for (const auto& kf : window_keyframes) {
            // 检查该地图点是否出现在此关键帧的观测列表中
            for (int kf_mp_idx : kf.map_point_indices) {
                if (kf_mp_idx == mp_idx) {
                    observed_elsewhere.insert(mp_idx);
                    break; // 已找到，跳出内层循环
                }
            }
        }
    }
    
    double ratio = static_cast<double>(observed_elsewhere.size()) / matched_mp_indices.size();
    return ratio;
}

bool KeyframeSelector::decide(const cv::Mat& curr_rvec, const cv::Mat& curr_tvec,
                              const std::vector<cv::Point2f>& curr_keypoints_2d,
                              const std::vector<int>& matched_mp_indices,
                              const std::vector<cv::Point3f>& local_map_points,
                              const Keyframe* last_keyframe,
                              const std::vector<Keyframe>& window_keyframes,
                              bool is_first_frame)
{
    // 第一帧总是作为关键帧
    if (is_first_frame) {
        return true;
    }
    
    // 确保上一关键帧存在
    if (last_keyframe == nullptr) {
        return true;
    }
    
    // 将当前帧的旋转向量转换为旋转矩阵
    cv::Mat R_curr;
    cv::Rodrigues(curr_rvec, R_curr);
    
    // ========== 1. 近点统计与丢失率/数量阈值检查 ==========
    // 构建当前帧匹配到的地图点中，哪些是近点
    std::vector<int> near_point_indices_curr;
    for (int mp_idx : matched_mp_indices) {
        if (mp_idx >= 0 && mp_idx < (int)local_map_points.size()) {
            const cv::Point3f& pt_w = local_map_points[mp_idx];
            if (isNearPoint(pt_w, R_curr, curr_tvec)) {
                near_point_indices_curr.push_back(mp_idx);
            }
        }
    }
    int curr_near_count = (int)near_point_indices_curr.size();
    
    // 检查近点数量下限
    bool force_insert_by_low_near = (curr_near_count < min_near_points_);
    
    // 检查丢失率：相对于上一关键帧的近点
    bool force_insert_by_loss = false;
    if (last_keyframe != nullptr) {
        // 统计上一关键帧的近点集合
        cv::Mat R_last;
        cv::Rodrigues(last_keyframe->rvec, R_last);
        std::set<int> last_near_indices;
        for (int mp_idx : last_keyframe->map_point_indices) {
            if (mp_idx >= 0 && mp_idx < (int)local_map_points.size()) {
                if (isNearPoint(local_map_points[mp_idx], R_last, last_keyframe->tvec)) {
                    last_near_indices.insert(mp_idx);
                }
            }
        }
        
        // 统计上一关键帧的近点中有多少在当前帧仍然被匹配到
        int last_near_reobserved = 0;
        for (int mp_idx : near_point_indices_curr) {
            if (last_near_indices.count(mp_idx)) {
                last_near_reobserved++;
            }
        }
        
        int last_near_total = (int)last_near_indices.size();
        if (last_near_total > 0) {
            double loss_ratio = 1.0 - static_cast<double>(last_near_reobserved) / last_near_total;
            force_insert_by_loss = (loss_ratio > loss_ratio_thresh_);
        } else {
            // 上一关键帧没有近点，则视为需要插入新关键帧
            force_insert_by_loss = true;
        }
    }
    
    // 如果近点数量不足或丢失率过高，强制插入关键帧，跳过后续否决检查（但仍需通过冗余检查？按需求应直接插入，但冗余检查可保留以防过度插入）
    if (force_insert_by_low_near || force_insert_by_loss) {
        // 但仍需通过冗余检查避免冗余帧
        double redundancy = computeRedundancyRatio(matched_mp_indices, window_keyframes);
        if (redundancy > redundancy_ratio_) {
            return false; // 虽然近点问题触发，但点都被其他帧看到，不插
        }
        return true;
    }
    
    // ========== 2. 相对视差/相对运动硬否决 ==========
    // 计算平均视差（像素）
    double avg_disparity = computeAverageDisparity(*last_keyframe, curr_keypoints_2d, matched_mp_indices);
    
    // 计算相对平移量（米）与基线的比值
    cv::Mat delta_t = curr_tvec - last_keyframe->tvec;
    double trans_norm = cv::norm(delta_t);
    double trans_ratio = trans_norm / baseline_;
    
    // 硬否决：视差低于阈值 且 平移小于基线的0.5倍
    if (avg_disparity < min_avg_disparity_ && trans_ratio < trans_base_ratio_) {
        return false; // 运动不明显，不插入关键帧
    }
    
    // ========== 3. 特征点覆盖网格比例检查 ==========
    double coverage = computeGridCoverage(curr_keypoints_2d);
    if (coverage < coverage_ratio_) {
        return false; // 覆盖不足，拒绝
    }
    
    // ========== 4. 共视窗口冗余观测检查 ==========
    double redundancy = computeRedundancyRatio(matched_mp_indices, window_keyframes);
    if (redundancy > redundancy_ratio_) {
        return false; // 大部分点已被其他关键帧观察到，无需新增
    }
    
    // 所有条件满足，允许插入关键帧
    return true;
}