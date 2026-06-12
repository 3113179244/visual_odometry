#include "slide_window.h"
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <set>
#include <map>

using namespace std;
using namespace cv;

// ------------------------------------------------------------
// 代价函数：重投影误差（适用于单个观测）
// 残差 = 像素坐标预测值 - 观测值
// 参数块：位姿（李代数6维）+ 地图点（3维）
// ------------------------------------------------------------
class ReprojectionErrorBA : public ceres::SizedCostFunction<2, 6, 3> {
public:
    ReprojectionErrorBA(const cv::Point2f& obs, const cv::Mat& K)
        : obs_(obs.x, obs.y), K_(K) {
        fx_ = K.at<double>(0,0); fy_ = K.at<double>(1,1);
        cx_ = K.at<double>(0,2); cy_ = K.at<double>(1,2);
    }
    
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override {
        // parameters[0]: 位姿李代数 xi (6维)
        // parameters[1]: 地图点坐标 Pw (3维)
        Eigen::Map<const Eigen::Matrix<double,6,1>> xi(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> Pw(parameters[1]);
        
        // 当前帧位姿
        Sophus::SE3d Tcw = Sophus::SE3d::exp(xi);
        Eigen::Vector3d Pc = Tcw * Pw;   // 转换到相机坐标系
        double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
        
        // 预测投影
        double u_pred = fx_ * X / Z + cx_;
        double v_pred = fy_ * Y / Z + cy_;
        
        // 残差
        residuals[0] = u_pred - obs_.x();
        residuals[1] = v_pred - obs_.y();
        
        // 解析雅可比（可选，帮助更快收敛）
        if (jacobians) {
            double Z2 = Z * Z;
            double inv_Z = 1.0 / Z;
            // 对地图点的雅可比 (2x3)
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double,2,3,Eigen::RowMajor>> J_p(jacobians[1]);
                J_p(0,0) = fx_ * inv_Z;
                J_p(0,1) = 0;
                J_p(0,2) = -fx_ * X / Z2;
                J_p(1,0) = 0;
                J_p(1,1) = fy_ * inv_Z;
                J_p(1,2) = -fy_ * Y / Z2;
                // 链式法则：J_cam * Rcw
                J_p = J_p * Tcw.rotationMatrix();
            }
            // 对位姿的雅可比 (2x6)
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> J_xi(jacobians[0]);
                // 左乘扰动模型下的雅可比
                J_xi(0,0) = fx_ * inv_Z;
                J_xi(0,1) = 0;
                J_xi(0,2) = -fx_ * X / Z2;
                J_xi(0,3) = -fx_ * X * Y / Z2;
                J_xi(0,4) = fx_ + fx_ * X * X / Z2;
                J_xi(0,5) = -fx_ * Y / Z;
                
                J_xi(1,0) = 0;
                J_xi(1,1) = fy_ * inv_Z;
                J_xi(1,2) = -fy_ * Y / Z2;
                J_xi(1,3) = -fy_ - fy_ * Y * Y / Z2;
                J_xi(1,4) = fy_ * X * Y / Z2;
                J_xi(1,5) = fy_ * X / Z;
            }
        }
        return true;
    }
    
private:
    Eigen::Vector2d obs_;
    cv::Mat K_;
    double fx_, fy_, cx_, cy_;
};

// ------------------------------------------------------------
// SlidingWindow 实现
// ------------------------------------------------------------
SlidingWindow::SlidingWindow(int window_size, const cv::Mat& K, double baseline,
                             std::vector<MapPoint>& local_map, std::shared_mutex& map_mutex)
    : window_size_(window_size), K_(K.clone()), baseline_(baseline),
      local_map_(local_map), map_mutex_(map_mutex) {}

Sophus::SE3d SlidingWindow::cvToSophus(const cv::Mat& rvec, const cv::Mat& tvec) const {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d R_eigen;
    cv::cv2eigen(R, R_eigen);
    Eigen::Vector3d t_eigen(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    return Sophus::SE3d(R_eigen, t_eigen);
}

void SlidingWindow::sophusToCv(const Sophus::SE3d& T, cv::Mat& rvec, cv::Mat& tvec) const {
    Eigen::Matrix3d R = T.rotationMatrix();
    cv::Mat R_cv;
    cv::eigen2cv(R, R_cv);
    cv::Rodrigues(R_cv, rvec);
    Eigen::Vector3d t = T.translation();
    tvec = (cv::Mat_<double>(3,1) << t(0), t(1), t(2));
}

void SlidingWindow::buildBAProblem(
    std::vector<std::tuple<int, int, cv::Point2f>>& observations,
    std::vector<Sophus::SE3d>& frame_poses,
    std::vector<Eigen::Vector3d>& points) const {
    std::shared_lock<std::shared_mutex> lock(map_mutex_);
    // 统计每个地图点被哪些关键帧观测
    std::map<int, std::set<int>> point_to_frames;  // map_point_idx -> set<frame_idx_in_window>
    
    // 收集所有观测，同时记录每个地图点出现的帧
    for (size_t i = 0; i < keyframes_.size(); ++i) {
        const Keyframe& kf = keyframes_[i];
        for (size_t j = 0; j < kf.map_point_indices.size(); ++j) {
            int mp_idx = kf.map_point_indices[j];
            if (mp_idx >= 0 && mp_idx < (int)local_map_.size()) {
                cv::Point2f pt = kf.keypoints_2d[j];
                observations.emplace_back(i, mp_idx, pt);
                point_to_frames[mp_idx].insert(i);
            }
        }
    }
    
    // 只保留至少被2个关键帧观测到的地图点（提供足够的约束）
    std::set<int> valid_points;
    for (const auto& kv : point_to_frames) {
        if (kv.second.size() >= 2) {
            valid_points.insert(kv.first);
        }
    }
    
    // 过滤观测，只保留有效地图点
    std::vector<std::tuple<int, int, cv::Point2f>> filtered_obs;
    for (const auto& obs : observations) {
        int mp_idx = std::get<1>(obs);
        if (valid_points.count(mp_idx)) {
            filtered_obs.push_back(obs);
        }
    }
    observations.swap(filtered_obs);
    
    // 提取每个关键帧的初始位姿
    frame_poses.resize(keyframes_.size());
    for (size_t i = 0; i < keyframes_.size(); ++i) {
        frame_poses[i] = cvToSophus(keyframes_[i].rvec, keyframes_[i].tvec);
    }
    
    // 提取有效地图点的坐标（去重）
    std::map<int, Eigen::Vector3d> point_coords;
    for (int mp_idx : valid_points) {
        const cv::Point3f& p = local_map_[mp_idx].pos_world;
        point_coords[mp_idx] = Eigen::Vector3d(p.x, p.y, p.z);
    }
    points.clear();
    for (int mp_idx : valid_points) {
        points.push_back(point_coords[mp_idx]);
    }
    // 注意：observations中存储的mp_idx是原始索引，后面优化时需要映射到points列表的索引
    // 我们将在optimize()中处理映射
}

void SlidingWindow::updateOptimizedResults(const std::vector<Sophus::SE3d>& opt_poses,
                                           const std::vector<Eigen::Vector3d>& opt_points,
                                           const std::vector<int>& point_ids_in_window) {
    // 更新关键帧位姿（跳过第一帧，因为被固定）
    for (size_t i = 1; i < opt_poses.size() && i < keyframes_.size(); ++i) {
        sophusToCv(opt_poses[i], keyframes_[i].rvec, keyframes_[i].tvec);
    }
    // 更新地图点坐标
    std::unique_lock<std::shared_mutex> lock(map_mutex_);
    for (size_t i = 0; i < point_ids_in_window.size(); ++i) {
        int raw_idx = point_ids_in_window[i];
        if (raw_idx >= 0 && raw_idx < (int)local_map_.size()) {
            local_map_[raw_idx].pos_world = cv::Point3f(opt_points[i].x(), opt_points[i].y(), opt_points[i].z());
        }
    }
}

void SlidingWindow::optimize() {
    if (keyframes_.size() < 2) return;   // 少于两帧无法构成有效BA
    
    // 1. 收集观测和变量
    std::vector<std::tuple<int, int, cv::Point2f>> raw_obs; // (frame_idx, mp_raw_idx, uv)
    std::vector<Sophus::SE3d> frame_poses_init;
    std::vector<Eigen::Vector3d> points_init;
    buildBAProblem(raw_obs, frame_poses_init, points_init);
    
    if (raw_obs.empty() || points_init.empty()) return;
    
    // 2. 建立 mp_raw_idx -> point_idx_in_ba 的映射，以及反向列表
    std::map<int, int> raw_to_ba;
    std::vector<int> ba_to_raw;   // 用于更新结果
    for (const auto& obs : raw_obs) {
        int raw_idx = std::get<1>(obs);
        if (raw_to_ba.find(raw_idx) == raw_to_ba.end()) {
            raw_to_ba[raw_idx] = ba_to_raw.size();
            ba_to_raw.push_back(raw_idx);
        }
    }
    // 注意 points_init 和 ba_to_raw 的顺序对应，但 points_init 可能比 ba_to_raw 多？
    // 实际上 buildBAProblem 已经将 points_init 按照 valid_points 的顺序填充，和 raw_to_ba 中的顺序一致（都是递增遍历 valid_points）。
    // 为确保一致，我们重新按 raw_to_ba 的顺序填充 points_init。
    std::vector<Eigen::Vector3d> ordered_points(ba_to_raw.size());
    for (size_t i = 0; i < ba_to_raw.size(); ++i) {
        int raw = ba_to_raw[i];
        const cv::Point3f& p = local_map_[raw].pos_world;
        ordered_points[i] = Eigen::Vector3d(p.x, p.y, p.z);
    }
    points_init = ordered_points;
    
    // 3. 构建Ceres问题
    ceres::Problem problem;
    
    // 参数块：位姿（6维李代数）和地图点（3维）
    std::vector<double*> pose_params;
    for (size_t i = 0; i < keyframes_.size(); ++i) {
        double* xi = new double[6]{0};
        pose_params.push_back(xi);
        problem.AddParameterBlock(xi, 6);
    }
    // 固定第一帧的位姿（消除自由度）
    if (!pose_params.empty()) {
        problem.SetParameterBlockConstant(pose_params[0]);
    }
    
    // 为每个地图点创建参数
    std::vector<double*> point_params;
    for (size_t i = 0; i < points_init.size(); ++i) {
        double* pt = new double[3];
        pt[0] = points_init[i].x();
        pt[1] = points_init[i].y();
        pt[2] = points_init[i].z();
        point_params.push_back(pt);
        problem.AddParameterBlock(pt, 3);
    }
    
    // 添加残差块
    for (const auto& obs : raw_obs) {
        int frame_idx = std::get<0>(obs);
        int raw_mp_idx = std::get<1>(obs);
        cv::Point2f uv = std::get<2>(obs);
        int ba_pt_idx = raw_to_ba[raw_mp_idx];
        
        ceres::CostFunction* cost = new ReprojectionErrorBA(uv, K_);
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                                 pose_params[frame_idx], point_params[ba_pt_idx]);
    }
    
    // 4. 求解
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 20;
    options.num_threads = 4;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // 5. 更新结果
    std::vector<Sophus::SE3d> opt_poses(keyframes_.size());
    for (size_t i = 0; i < keyframes_.size(); ++i) {
        if (i == 0) {
            opt_poses[i] = frame_poses_init[i];  // 第一帧固定
        } else {
            Eigen::Map<const Eigen::Matrix<double,6,1>> xi(pose_params[i]);
            opt_poses[i] = Sophus::SE3d::exp(xi) * frame_poses_init[i];
        }
    }
    std::vector<Eigen::Vector3d> opt_points(points_init.size());
    for (size_t i = 0; i < points_init.size(); ++i) {
        opt_points[i] = Eigen::Vector3d(point_params[i][0], point_params[i][1], point_params[i][2]);
    }
    
    updateOptimizedResults(opt_poses, opt_points, ba_to_raw);
    
    // 清理动态分配的内存
    for (double* p : pose_params) delete[] p;
    for (double* p : point_params) delete[] p;
}

void SlidingWindow::addKeyframe(const Keyframe& kf) {
    keyframes_.push_back(kf);
    while ((int)keyframes_.size() > window_size_) {
        keyframes_.pop_front();
    }
    optimize();
}