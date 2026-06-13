#include "slide_window.h"
#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <set>
#include <map>
#include <cmath>
#include "math_utils.h"

using namespace std;
using namespace cv;

template <typename T>
inline double GetRawValue(const T &val) { return val.a; }
template <>
inline double GetRawValue<double>(const double &val) { return val; }

struct ReprojectionErrorAutoDiff
{
    ReprojectionErrorAutoDiff(const cv::Point2f &obs, double fx, double fy, double cx, double cy)
        : obs_(obs.x, obs.y), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    template <typename T>
    bool operator()(const T *const jet_delta_xi, const T *const jet_Pw, T *residuals) const
    {
        Eigen::Matrix<T, 6, 1> delta_vec;
        delta_vec << jet_delta_xi[0], jet_delta_xi[1], jet_delta_xi[2],
            jet_delta_xi[3], jet_delta_xi[4], jet_delta_xi[5];

        // 🌟 核心修正：直接调用经过充分验证、支持自动求导的 Sophus 官方李群指数映射 🌟
        Sophus::SE3<T> delta_T = Sophus::SE3<T>::exp(delta_vec);

        // 利用 Sophus 对象与初始初值自相乘，驱动李代数左乘扰动模型
        Sophus::SE3<T> Tcw_optimized = delta_T * Tcw_init_T_.template cast<T>();

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pw(jet_Pw);
        Eigen::Matrix<T, 3, 1> Pc = Tcw_optimized * Pw;
        T X = Pc.x();
        T Y = Pc.y();
        T Z = Pc.z();

        double z_val = GetRawValue(Z);
        double x_val = GetRawValue(X);
        double y_val = GetRawValue(Y);
        if (Z <= T(1e-4) || !std::isfinite(z_val) || !std::isfinite(x_val) || !std::isfinite(y_val))
        {
            residuals[0] = T(0.0);
            residuals[1] = T(0.0);
            return true;
        }

        residuals[0] = T(fx_) * X / Z + T(cx_) - T(obs_.x());
        residuals[1] = T(fy_) * Y / Z + T(cy_) - T(obs_.y());
        return true;
    }

    Eigen::Vector2d obs_;
    Sophus::SE3d Tcw_init_T_;
    double fx_, fy_, cx_, cy_;
};

SlidingWindow::SlidingWindow(int window_size, double fx, double fy, double cx, double cy, double baseline,
                             int img_width, int img_height, std::shared_ptr<Map> pMap)
    : window_size_(window_size), fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline),
      img_width_(img_width), img_height_(img_height), mpMap(pMap) {}

Sophus::SE3d SlidingWindow::cvToSophus(const cv::Mat &rvec, const cv::Mat &tvec) const
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d R_eigen;
    cv::cv2eigen(R, R_eigen);
    Eigen::Vector3d t_eigen(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    return Sophus::SE3d(R_eigen, t_eigen);
}

void SlidingWindow::sophusToCv(const Sophus::SE3d &T, cv::Mat &rvec, cv::Mat &tvec) const
{
    Eigen::Matrix3d R = T.rotationMatrix();
    cv::Mat R_cv;
    cv::eigen2cv(R, R_cv);
    cv::Rodrigues(R_cv, rvec);
    Eigen::Vector3d t = T.translation();
    tvec = (cv::Mat_<double>(3, 1) << t(0), t(1), t(2));
}

void SlidingWindow::buildBAProblem(std::vector<std::tuple<int, int, cv::Point2f>> &observations,
                                   std::vector<Sophus::SE3d> &frame_poses,
                                   std::vector<Eigen::Vector3d> &points) const
{
    auto global_mps = mpMap->GetAllMapPoints();
    std::shared_lock<std::shared_mutex> lock(mpMap->mMutexMap);

    std::map<int, std::set<int>> point_to_frames;
    for (size_t i = 0; i < keyframes_.size(); ++i)
    {
        const Keyframe &kf = keyframes_[i];
        for (size_t j = 0; j < kf.map_point_indices.size(); ++j)
        {
            int mp_idx = kf.map_point_indices[j];
            if (mp_idx < 0 || static_cast<size_t>(mp_idx) >= global_mps.size())
                continue;
            auto pMP = global_mps[mp_idx];
            if (!pMP || pMP->is_bad)
                continue;

            const cv::Point3f &pt_w = pMP->pos_world;
            if (!std::isfinite(pt_w.x) || !std::isfinite(pt_w.y) || !std::isfinite(pt_w.z))
                continue;
            if (std::abs(pt_w.x) > 1e4 || std::abs(pt_w.y) > 1e4 || std::abs(pt_w.z) > 1e4)
                continue;

            cv::Point2f pt_2d = kf.keypoints_2d[j];
            if (pt_2d.x < 0 || pt_2d.x >= img_width_ || pt_2d.y < 0 || pt_2d.y >= img_height_)
                continue;

            observations.emplace_back(i, mp_idx, pt_2d);
            point_to_frames[mp_idx].insert(i);
        }
    }

    std::set<int> valid_points;
    for (const auto &kv : point_to_frames)
    {
        if (kv.second.size() >= 2)
            valid_points.insert(kv.first);
    }

    std::vector<std::tuple<int, int, cv::Point2f>> filtered_obs;
    for (const auto &obs : observations)
    {
        int mp_idx = std::get<1>(obs);
        if (valid_points.count(mp_idx))
            filtered_obs.push_back(obs);
    }
    observations.swap(filtered_obs);

    frame_poses.resize(keyframes_.size());
    for (size_t i = 0; i < keyframes_.size(); ++i)
    {
        frame_poses[i] = cvToSophus(keyframes_[i].rvec, keyframes_[i].tvec);
    }

    points.clear();
    for (int mp_idx : valid_points)
    {
        const cv::Point3f &p = global_mps[mp_idx]->pos_world;
        points.push_back(Eigen::Vector3d(p.x, p.y, p.z));
    }
}

void SlidingWindow::updateOptimizedResults(const std::vector<Sophus::SE3d> &opt_poses,
                                           const std::vector<Eigen::Vector3d> &opt_points,
                                           const std::vector<int> &point_ids_in_window)
{
    for (size_t i = 0; i < opt_poses.size() && i < keyframes_.size(); ++i)
    {
        sophusToCv(opt_poses[i], keyframes_[i].rvec, keyframes_[i].tvec);
    }

    auto global_kfs = mpMap->GetAllKeyframes();
    auto global_mps = mpMap->GetAllMapPoints();

    std::unique_lock<std::shared_mutex> lock(mpMap->mMutexMap);

    for (size_t i = 0; i < opt_poses.size() && i < keyframes_.size(); ++i)
    {
        int kf_id = keyframes_[i].id;
        for (auto &pKF : global_kfs)
        {
            if (pKF && pKF->id == kf_id)
            {
                keyframes_[i].rvec.copyTo(pKF->rvec);
                keyframes_[i].tvec.copyTo(pKF->tvec);
                break;
            }
        }
    }

    for (size_t i = 0; i < point_ids_in_window.size(); ++i)
    {
        int raw_idx = point_ids_in_window[i];
        if (raw_idx >= 0 && static_cast<size_t>(raw_idx) < global_mps.size() && global_mps[raw_idx])
        {
            global_mps[raw_idx]->pos_world = cv::Point3f(opt_points[i].x(), opt_points[i].y(), opt_points[i].z());
        }
    }
}

void SlidingWindow::optimize()
{
    if (keyframes_.size() < 2)
        return;

    std::vector<std::tuple<int, int, cv::Point2f>> raw_obs;
    std::vector<Sophus::SE3d> frame_poses_init;
    std::vector<Eigen::Vector3d> points_init;
    buildBAProblem(raw_obs, frame_poses_init, points_init);
    if (raw_obs.empty() || points_init.empty())
        return;

    std::map<int, int> raw_to_ba;
    std::vector<int> ba_to_raw;
    for (const auto &obs : raw_obs)
    {
        int raw_idx = std::get<1>(obs);
        if (raw_to_ba.find(raw_idx) == raw_to_ba.end())
        {
            raw_to_ba[raw_idx] = ba_to_raw.size();
            ba_to_raw.push_back(raw_idx);
        }
    }

    auto global_mps = mpMap->GetAllMapPoints();
    std::vector<Eigen::Vector3d> ordered_points(ba_to_raw.size());
    {
        std::shared_lock<std::shared_mutex> lock(mpMap->mMutexMap);
        for (size_t i = 0; i < ba_to_raw.size(); ++i)
        {
            int raw = ba_to_raw[i];
            const cv::Point3f &p = global_mps[raw]->pos_world;
            ordered_points[i] = Eigen::Vector3d(p.x, p.y, p.z);
        }
    }
    points_init = ordered_points;

    ceres::Problem problem;

    std::vector<std::vector<double>> pose_vecs(keyframes_.size(), std::vector<double>(6, 0.0));
    for (size_t i = 0; i < keyframes_.size(); ++i)
    {
        problem.AddParameterBlock(pose_vecs[i].data(), 6);
    }
    if (!pose_vecs.empty())
        problem.SetParameterBlockConstant(pose_vecs[0].data());

    std::vector<std::vector<double>> point_vecs(points_init.size(), std::vector<double>(3, 0.0));
    for (size_t i = 0; i < points_init.size(); ++i)
    {
        point_vecs[i][0] = points_init[i].x();
        point_vecs[i][1] = points_init[i].y();
        point_vecs[i][2] = points_init[i].z();
        problem.AddParameterBlock(point_vecs[i].data(), 3);
    }

    for (const auto &obs : raw_obs)
    {
        int frame_idx = std::get<0>(obs);
        int raw_mp_idx = std::get<1>(obs);
        cv::Point2f uv = std::get<2>(obs);
        int ba_pt_idx = raw_to_ba[raw_mp_idx];

        ReprojectionErrorAutoDiff *cost_struct = new ReprojectionErrorAutoDiff(uv, fx_, fy_, cx_, cy_);
        cost_struct->Tcw_init_T_ = frame_poses_init[frame_idx];

        ceres::CostFunction *cost = new ceres::AutoDiffCostFunction<ReprojectionErrorAutoDiff, 2, 6, 3>(cost_struct);
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), pose_vecs[frame_idx].data(), point_vecs[ba_pt_idx].data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 12;
    options.num_threads = 4;
    options.use_nonmonotonic_steps = true;
    options.use_inner_iterations = true;
    options.function_tolerance = 1e-3;
    options.gradient_tolerance = 1e-3;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::vector<Sophus::SE3d> opt_poses(keyframes_.size());
    for (size_t i = 0; i < keyframes_.size(); ++i)
    {
        // 🌟 关键修复点：显式将双精度的一维数组映射回 Eigen 向量形式，解除 Map 的推导阻断 🌟
        Eigen::VectorXd xi = Eigen::Map<const Eigen::VectorXd>(pose_vecs[i].data(), 6);
        opt_poses[i] = Sophus::SE3d::exp(xi) * frame_poses_init[i];
    }
    std::vector<Eigen::Vector3d> opt_points(points_init.size());
    for (size_t i = 0; i < points_init.size(); ++i)
    {
        opt_points[i] = Eigen::Vector3d(point_vecs[i][0], point_vecs[i][1], point_vecs[i][2]);
    }

    updateOptimizedResults(opt_poses, opt_points, ba_to_raw);
}

void SlidingWindow::addKeyframe(const Keyframe &kf)
{
    keyframes_.push_back(kf);
    while ((int)keyframes_.size() > window_size_)
    {
        keyframes_.pop_front();
    }
    optimize();
}