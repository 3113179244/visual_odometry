#include <iostream>
#include <vector>

// 1. 强制锁死 Eigen 优先引入，杜绝局部不齐冲突
#include <Eigen/Core>
#include <Eigen/Dense>

// 2. 引入 OpenCV 及其转换矩阵
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

// 3. 引入非线性迭代求解核心
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include "math_utils.h" // 🌟 引入新构建的全量数学工具箱

using namespace std;
using namespace Eigen;

// ---------------------------------------------------------
// 全面提炼抽象后的 Ceres 前端位姿代价函数
// ---------------------------------------------------------
class PoseOptimizationCostFunction : public ceres::SizedCostFunction<2, 6>
{
public:
    PoseOptimizationCostFunction(const Vector3d &pos_3d, const Vector2d &measurement,
                                 const Matrix3d &K, const Sophus::SE3d &T_initial)
        : _pos3d(pos_3d), _measurement(measurement), _K(K), _T_initial(T_initial) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        // 1. 获取当前 Ceres 迭代探索下的微小扰动李代数 xi
        Map<const Matrix<double, 6, 1>> xi(parameters[0]);

        // 2. 左乘扰动恢复出相机系的位姿 R_cw 与 t_cw
        Sophus::SE3d T_current = Sophus::SE3d::exp(xi) * _T_initial;
        Matrix3d R = T_current.rotationMatrix();
        Vector3d t = T_current.translation();

        double fx = _K(0, 0), fy = _K(1, 1), cx = _K(0, 2), cy = _K(1, 2);

        // 3. 🌟 一键调用工具箱通用重投影残差算子
        bool valid = vo_math::ComputeReprojectionError<double>(R, t, _pos3d, _measurement, fx, fy, cx, cy, residuals);
        if (!valid)
        {
            return false; // 如果投影到背面发生奇异，直接拒绝此迭代步
        }

        // 4. 🌟 一键调用工具箱解析雅可比矩阵算子，填充 2x6 解析偏导数矩阵
        if (jacobians && jacobians[0])
        {
            Map<Matrix<double, 2, 6, RowMajor>> J(jacobians[0]);
            Vector3d pos_cam = T_current * _pos3d; // 转换到当前相机坐标系

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

// ---------------------------------------------------------
// BA 优化函数对外核心解算接口
// ---------------------------------------------------------
void bundleAdjustment(const vector<cv::Point3f> &points_3d,
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

    // 6 维零向量扰动参数块
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
    options.max_num_iterations = 8;

    cout << "\n[Ceres] 开始执行 Bundle Adjustment (跨文件调用工具箱解析求导)..." << endl;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;

    Map<const Matrix<double, 6, 1>> xi_opt(xi);
    Sophus::SE3d pose_optimized = Sophus::SE3d::exp(xi_opt) * pose_initial;

    cv::eigen2cv(pose_optimized.rotationMatrix(), R);
    Vector3d t_opt = pose_optimized.translation();
    t = (cv::Mat_<double>(3, 1) << t_opt(0), t_opt(1), t_opt(2));

    cout << "[Ceres] BA 优化完成！" << endl;
}