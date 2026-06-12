// keyframe_selector.h
#ifndef KEYFRAME_SELECTOR_H
#define KEYFRAME_SELECTOR_H

#include <vector>
#include <opencv2/core/core.hpp>

// 前向声明
struct MapPoint;

/**
 * @brief 关键帧数据结构（用于共视窗口管理）
 */
struct Keyframe {
    int id;                                 // 帧序号
    cv::Mat rvec;                           // 旋转向量（世界到相机）
    cv::Mat tvec;                           // 平移向量（世界到相机）
    std::vector<cv::Point2f> keypoints_2d;  // 该帧提取的2D特征点（像素坐标）
    std::vector<int> map_point_indices;     // 对应的地图点索引（在全局local_map中的位置）
    
    // ==========================================
    // 新增：异步三角化所必须的原始图像与特征数据
    // ==========================================
    cv::Mat img_l;                          
    std::vector<cv::KeyPoint> kps_l, kps_r; 
    cv::Mat desc_l, desc_r;                 

    Keyframe() : id(-1) {} //
    Keyframe(int _id, const cv::Mat& _rvec, const cv::Mat& _tvec,
             const std::vector<cv::Point2f>& _kps, const std::vector<int>& _mp_ids,
             const cv::Mat& _img_l, const std::vector<cv::KeyPoint>& _kps_l, 
             const std::vector<cv::KeyPoint>& _kps_r, const cv::Mat& _desc_l, const cv::Mat& _desc_r)
        : id(_id), rvec(_rvec.clone()), tvec(_tvec.clone()),
          keypoints_2d(_kps), map_point_indices(_mp_ids),
          img_l(_img_l.clone()), kps_l(_kps_l), kps_r(_kps_r), 
          desc_l(_desc_l.clone()), desc_r(_desc_r.clone()) {}
};

/**
 * @brief 关键帧决策器
 * 
 * 实现严格的筛选逻辑：
 * 1. 基于基线划分近/远点，只统计高精度近点
 * 2. 相对视差/运动硬阈值一票否决
 * 3. 特征点覆盖网格比例要求
 * 4. 共视窗口内冗余观测剔除
 */
class KeyframeSelector {
public:
    /**
     * @param fx, fy, cx, cy      相机内参
     * @param baseline            双目基线（米）
     * @param img_width, img_height 图像尺寸
     */
    KeyframeSelector(double fx, double fy, double cx, double cy,
                     double baseline, int img_width, int img_height);
    
    /**
     * @brief 设置各项阈值参数
     * @param near_depth_thresh   近点深度阈值（米），小于此值为近点
     * @param loss_ratio_thresh   近点丢失率阈值（0~1），超过此值强制插关键帧
     * @param min_near_points      最小近点数量，低于此值强制插关键帧
     * @param min_avg_disparity    最小平均视差（像素），低于此值且平移较小时一票否决
     * @param trans_base_ratio     平移量/基线比值阈值，低于此值且视差不足时一票否决
     * @param grid_cell_rows       网格行数
     * @param grid_cell_cols       网格列数
     * @param coverage_ratio       特征点覆盖网格比例阈值（0~1），低于此值拒绝
     * @param redundancy_ratio     共视冗余比例阈值（0~1），高于此值拒绝
     */
    void setParameters(double near_depth_thresh = 10.0,
                       double loss_ratio_thresh = 0.15,
                       int min_near_points = 30,
                       double min_avg_disparity = 15.0,
                       double trans_base_ratio = 0.5,
                       int grid_cell_rows = 6,
                       int grid_cell_cols = 8,
                       double coverage_ratio = 0.35,
                       double redundancy_ratio = 0.85);
    
    /**
     * @brief 判断当前帧是否应作为关键帧插入
     * 
     * @param curr_rvec            当前帧旋转向量（世界到相机）
     * @param curr_tvec            当前帧平移向量（世界到相机）
     * @param curr_keypoints_2d    当前帧所有2D特征点（像素坐标）
     * @param matched_mp_indices   当前帧匹配到的地图点索引（对应local_map）
     * @param local_map_points     全局地图点列表（世界坐标）
     * @param last_keyframe        上一个关键帧（用于丢失率和相对运动计算）
     * @param window_keyframes     共视窗口内的其他关键帧（用于冗余检测）
     * @param is_first_frame       是否为第一帧（总是插入）
     * @return true  应该插入关键帧
     * @return false 拒绝插入
     */
    bool decide(const cv::Mat& curr_rvec, const cv::Mat& curr_tvec,
                const std::vector<cv::Point2f>& curr_keypoints_2d,
                const std::vector<int>& matched_mp_indices,
                const std::vector<cv::Point3f>& local_map_points,
                const Keyframe* last_keyframe,
                const std::vector<Keyframe>& window_keyframes,
                bool is_first_frame = false);
    
private:
    // 相机参数
    double fx_, fy_, cx_, cy_, baseline_;
    int img_width_, img_height_;
    
    // 决策阈值
    double near_depth_thresh_;      // 近点深度上限（米）
    double loss_ratio_thresh_;      // 近点丢失率阈值
    int min_near_points_;           // 最小近点数量
    double min_avg_disparity_;      // 最小平均视差（像素）
    double trans_base_ratio_;       // 平移/基线比例阈值
    int grid_rows_, grid_cols_;     // 网格划分
    double coverage_ratio_;         // 网格覆盖比例要求
    double redundancy_ratio_;       // 共视冗余比例上限
    
    /**
     * @brief 判断地图点是否为近点（深度 < near_depth_thresh_）
     * @param point_world   世界坐标点
     * @param R_curr        当前帧旋转矩阵（3x3）
     * @param t_curr        当前帧平移向量
     * @return true 近点, false 远点
     */
    bool isNearPoint(const cv::Point3f& point_world,
                     const cv::Mat& R_curr, const cv::Mat& t_curr) const;
    
    /**
     * @brief 计算当前帧相对于上一关键帧的平均视差（像素）
     * @param last_kf       上一关键帧
     * @param curr_kps      当前帧特征点（与上一关键帧匹配的点）
     * @param matched_ids   匹配的地图点索引（需要在上一关键帧的观测中）
     * @return 平均视差值，若无共视点返回0
     */
    double computeAverageDisparity(const Keyframe& last_kf,
                                   const std::vector<cv::Point2f>& curr_kps,
                                   const std::vector<int>& matched_ids) const;
    
    /**
     * @brief 计算当前帧特征点的网格覆盖率
     * @param keypoints_2d   当前帧所有2D特征点
     * @return 覆盖比例（0~1）
     */
    double computeGridCoverage(const std::vector<cv::Point2f>& keypoints_2d) const;
    
    /**
     * @brief 检查当前帧观测到的地图点是否在共视窗口中被过度冗余观测
     * @param matched_mp_indices   当前帧匹配到的地图点索引
     * @param window_keyframes     共视窗口内的关键帧（不包含当前帧）
     * @return 冗余比例（观测到的点中，被其他关键帧观测到的比例）
     */
    double computeRedundancyRatio(const std::vector<int>& matched_mp_indices,
                                  const std::vector<Keyframe>& window_keyframes) const;
};

#endif // KEYFRAME_SELECTOR_H