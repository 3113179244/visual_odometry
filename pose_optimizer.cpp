// =========================================================================
// ====== DEBUG REFACTOR CODE START ======
// 文件名: pose_optimizer.cpp
// 修改点: 升级 bundleAdjustment 接口，向前端传出外点掩码，用于物理剔除自行车
// =========================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include "math_utils.h"

using namespace std;
using namespace Eigen;

class PoseOptimizationCostFunction : public ceres::SizedCostFunction<2, 6>
{
public:
    PoseOptimizationCostFunction(const Vector3d &pos_3d, const Vector2d &measurement,
                                 const Matrix3d &K, const Sophus::SE3d &T_initial)
        : _pos3d(pos_3d), _measurement(measurement), _K(K), _T_initial(T_initial) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Map<const Matrix<double, 6, 1>> xi(parameters[0]);
        Sophus::SE3d T_current = Sophus::SE3d::exp(xi) * _T_initial;
        Matrix3d R = T_current.rotationMatrix();
        Vector3d t = T_current.translation();

        double fx = _K(0, 0), fy = _K(1, 1), cx = _K(0, 2), cy = _K(1, 2);

        bool valid = vo_math::ComputeReprojectionError<double>(R, t, _pos3d, _measurement, fx, fy, cx, cy, residuals);
        if (!valid)
            return false;

        if (jacobians && jacobians[0])
        {
            Map<Matrix<double, 2, 6, RowMajor>> J(jacobians[0]);
            Vector3d pos_cam = T_current * _pos3d;
            J = vo_math::GetPoseJacobianAnalytical(pos_cam, fx, fy);
        }
        return true;
    }

private:
    Vector3d _pos3d;
    Vector2d _measurement;
    Matrix3d _K;
    Sophus::SE3d _T_initial;
};

// 🌟 核心接口升级：追加返回 is_inlier 状态表，大小与 points_3d 一致
std::vector<bool> bundleAdjustment(const vector<cv::Point3f> &points_3d,
                                   const vector<cv::Point2f> &points_2d,
                                   const cv::Mat &K,
                                   cv::Mat &R, cv::Mat &t)
{
    Matrix3d R_eigen;
    cv::cv2eigen(R, R_eigen);
    Vector3d t_eigen(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
    Sophus::SE3d pose_initial(R_eigen, t_eigen);

    Matrix3d K_eigen;
    cv::cv2eigen(K, K_eigen);

    double xi[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    ceres::Problem problem;

    for (size_t i = 0; i < points_3d.size(); ++i)
    {
        ceres::CostFunction *cost_function = new PoseOptimizationCostFunction(
            Vector3d(points_3d[i].x, points_3d[i].y, points_3d[i].z),
            Vector2d(points_2d[i].x, points_2d[i].y),
            K_eigen,
            pose_initial);

        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), xi);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Map<const Matrix<double, 6, 1>> xi_opt(xi);
    Sophus::SE3d pose_optimized = Sophus::SE3d::exp(xi_opt) * pose_initial;

    double fx = K_eigen(0, 0), fy = K_eigen(1, 1), cx = K_eigen(0, 2), cy = K_eigen(1, 2);
    int dynamic_outliers_count = 0;
    double total_residual = 0.0;

    std::vector<bool> is_inlier(points_3d.size(), true);

    for (size_t i = 0; i < points_3d.size(); ++i)
    {
        double res[2] = {0.0, 0.0};
        vo_math::ComputeReprojectionError<double>(
            pose_optimized.rotationMatrix(), pose_optimized.translation(),
            Vector3d(points_3d[i].x, points_3d[i].y, points_3d[i].z),
            Vector2d(points_2d[i].x, points_2d[i].y),
            fx, fy, cx, cy, res);

        double err_norm = std::sqrt(res[0] * res[0] + res[1] * res[1]);
        total_residual += err_norm;

        // 🔍 如果发现重投影误差大于 2.5 像素，判定为移动物体（自行车），在状态表中打上标记
        if (err_norm > 2.5)
        {
            is_inlier[i] = false;
            dynamic_outliers_count++;
        }
    }

    if (dynamic_outliers_count > 0)
    {
        cout << "⚠️  [DEBUG WARNING] 监测到突发动静态冲突！总点数: " << points_3d.size()
             << " | 已标记并阻击的自行车外点数: " << dynamic_outliers_count
             << " | 突发均方残差: " << (total_residual / points_3d.size()) << " 像素" << endl;
    }

    cv::eigen2cv(pose_optimized.rotationMatrix(), R);
    Vector3d t_opt = pose_optimized.translation();
    t = (cv::Mat_<double>(3, 1) << t_opt(0), t_opt(1), t_opt(2));

    return is_inlier;
}
// ====== DEBUG REFACTOR CODE END ======