#include "keyframe_selector.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>
#include <set>
#include <map>
#include <algorithm>

KeyframeSelector::KeyframeSelector(double fx, double fy, double cx, double cy,
                                   double baseline, int img_width, int img_height)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), img_width_(img_width), img_height_(img_height)
{
    setParameters();
}

void KeyframeSelector::setParameters(double near_depth_thresh, double loss_ratio_thresh, int min_near_points,
                                     double min_avg_disparity, double trans_base_ratio, int grid_cell_rows,
                                     int grid_cell_cols, double coverage_ratio, double redundancy_ratio)
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

bool KeyframeSelector::isNearPoint(const cv::Point3f &point_world, const cv::Mat &R_curr, const cv::Mat &t_curr) const
{
    cv::Mat point_world_mat = (cv::Mat_<double>(3, 1) << point_world.x, point_world.y, point_world.z);
    cv::Mat point_cam = R_curr * point_world_mat + t_curr;
    double depth = point_cam.at<double>(2, 0);
    return (depth > 0 && depth < near_depth_thresh_);
}

double KeyframeSelector::computeAverageDisparity(const Keyframe &last_kf, const std::vector<cv::Point2f> &curr_kps, const std::vector<int> &matched_ids) const
{
    std::map<int, cv::Point2f> last_kf_map;
    for (size_t i = 0; i < last_kf.map_point_indices.size(); ++i)
    {
        int mp_idx = last_kf.map_point_indices[i];
        if (mp_idx >= 0)
            last_kf_map[mp_idx] = last_kf.keypoints_2d[i];
    }
    double sum_disparity = 0.0;
    int valid_count = 0;
    for (size_t i = 0; i < matched_ids.size() && i < curr_kps.size(); ++i)
    {
        int mp_idx = matched_ids[i];
        auto it = last_kf_map.find(mp_idx);
        if (it != last_kf_map.end())
        {
            cv::Point2f last_pt = it->second;
            cv::Point2f curr_pt = curr_kps[i];
            double dx = last_pt.x - curr_pt.x;
            double dy = last_pt.y - curr_pt.y;
            sum_disparity += std::sqrt(dx * dx + dy * dy);
            valid_count++;
        }
    }
    return (valid_count > 0) ? (sum_disparity / valid_count) : 0.0;
}

double KeyframeSelector::computeGridCoverage(const std::vector<cv::Point2f> &keypoints_2d) const
{
    if (grid_rows_ <= 0 || grid_cols_ <= 0)
        return 0.0;
    int total_cells = grid_rows_ * grid_cols_;
    std::vector<bool> covered(total_cells, false);
    double cell_w = static_cast<double>(img_width_) / grid_cols_;
    double cell_h = static_cast<double>(img_height_) / grid_rows_;
    for (const auto &pt : keypoints_2d)
    {
        int col = static_cast<int>(pt.x / cell_w);
        int row = static_cast<int>(pt.y / cell_h);
        if (col >= 0 && col < grid_cols_ && row >= 0 && row < grid_rows_)
            covered[row * grid_cols_ + col] = true;
    }
    return static_cast<double>(std::count(covered.begin(), covered.end(), true)) / total_cells;
}

double KeyframeSelector::computeRedundancyRatio(const std::vector<int> &matched_mp_indices, const std::vector<Keyframe> &window_keyframes) const
{
    if (matched_mp_indices.empty())
        return 0.0;
    std::set<int> observed_elsewhere;
    for (int mp_idx : matched_mp_indices)
    {
        if (mp_idx < 0)
            continue;
        for (const auto &kf : window_keyframes)
        {
            for (int kf_mp_idx : kf.map_point_indices)
            {
                if (kf_mp_idx == mp_idx)
                {
                    observed_elsewhere.insert(mp_idx);
                    break;
                }
            }
        }
    }
    return static_cast<double>(observed_elsewhere.size()) / matched_mp_indices.size();
}

bool KeyframeSelector::decide(const cv::Mat &curr_rvec, const cv::Mat &curr_tvec,
                              const std::vector<cv::Point2f> &curr_keypoints_2d,
                              const std::vector<int> &matched_mp_indices,
                              std::shared_ptr<Map> pMap, // 🌟 优化项
                              const Keyframe *last_keyframe,
                              const std::vector<Keyframe> &window_keyframes,
                              bool is_first_frame)
{
    if (is_first_frame || last_keyframe == nullptr)
        return true;
    cv::Mat R_curr;
    cv::Rodrigues(curr_rvec, R_curr);

    // 获取全局大地图快照
    auto global_mps = pMap->GetAllMapPoints();
    std::shared_lock<std::shared_mutex> lock(pMap->mMutexMap); // 保护锁

    // ========== 1. 近点统计 ==========
    std::vector<int> near_point_indices_curr;
    for (int mp_idx : matched_mp_indices)
    {
        if (mp_idx >= 0 && static_cast<size_t>(mp_idx) < global_mps.size() && global_mps[mp_idx])
        {
            if (isNearPoint(global_mps[mp_idx]->pos_world, R_curr, curr_tvec))
            {
                near_point_indices_curr.push_back(mp_idx);
            }
        }
    }
    int curr_near_count = (int)near_point_indices_curr.size();
    bool force_insert_by_low_near = (curr_near_count < min_near_points_);

    // ========== 2. 丢失率校验 ==========
    bool force_insert_by_loss = false;
    cv::Mat R_last;
    cv::Rodrigues(last_keyframe->rvec, R_last);
    std::set<int> last_near_indices;
    for (int mp_idx : last_keyframe->map_point_indices)
    {
        if (mp_idx >= 0 && static_cast<size_t>(mp_idx) < global_mps.size() && global_mps[mp_idx])
        {
            if (isNearPoint(global_mps[mp_idx]->pos_world, R_last, last_keyframe->tvec))
            {
                last_near_indices.insert(mp_idx);
            }
        }
    }

    int last_near_reobserved = 0;
    for (int mp_idx : near_point_indices_curr)
    {
        if (last_near_indices.count(mp_idx))
            last_near_reobserved++;
    }
    int last_near_total = (int)last_near_indices.size();
    if (last_near_total > 0)
    {
        double loss_ratio = 1.0 - static_cast<double>(last_near_reobserved) / last_near_total;
        force_insert_by_loss = (loss_ratio > loss_ratio_thresh_);
    }
    else
    {
        force_insert_by_loss = true;
    }

    if (force_insert_by_low_near || force_insert_by_loss)
    {
        if (computeRedundancyRatio(matched_mp_indices, window_keyframes) > redundancy_ratio_)
            return false;
        return true;
    }

    // ========== 3. 硬否决检查 ==========
    double avg_disparity = computeAverageDisparity(*last_keyframe, curr_keypoints_2d, matched_mp_indices);
    double trans_ratio = cv::norm(curr_tvec - last_keyframe->tvec) / baseline_;
    if (avg_disparity < min_avg_disparity_ && trans_ratio < trans_base_ratio_)
        return false;

    // ========== 4. 网格覆盖度与冗余度 ==========
    if (computeGridCoverage(curr_keypoints_2d) < coverage_ratio_)
        return false;
    if (computeRedundancyRatio(matched_mp_indices, window_keyframes) > redundancy_ratio_)
        return false;

    return true;
}