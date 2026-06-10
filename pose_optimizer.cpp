#include <iostream>
#include <vector>

// ==========================================
// 核心修复：Eigen 头文件必须在最前面！
// ==========================================
#include <Eigen/Core>
#include <Eigen/Dense>

// ==========================================
// 然后才能包含 OpenCV 及其 Eigen 转换头文件
// ==========================================
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

// ==========================================
// 最后是李代数和 Ceres
// ==========================================
#include <sophus/se3.hpp>
#include <ceres/ceres.h>

using namespace std;
using namespace Eigen;

// ... 下面的类定义和 bundleAdjustment 代码保持完全不变 ...

// ---------------------------------------------------------
// 定义 Ceres 代价函数：继承 SizedCostFunction<残差维度, 参数维度>
// 残差维度 = 2 (u, v 像素误差)
// 参数维度 = 6 (李代数 xi 的 6 个维度)
// ---------------------------------------------------------
class PoseOptimizationCostFunction : public ceres::SizedCostFunction<2, 6> {
public:
    PoseOptimizationCostFunction(const Vector3d& pos_3d, const Vector2d& measurement, 
                                 const Matrix3d& K, const Sophus::SE3d& T_initial)
        : _pos3d(pos_3d), _measurement(measurement), _K(K), _T_initial(T_initial) {}

    // Evaluate 函数会在 Ceres 每次迭代时被调用，用来计算当前残差和雅可比矩阵
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        // 1. 获取当前 Ceres 正在探索的李代数微小扰动 xi
        Map<const Matrix<double, 6, 1>> xi(parameters[0]);
        
        // 2. 将微小扰动左乘到初始位姿上，得到当前帧的测试位姿
        Sophus::SE3d T_current = Sophus::SE3d::exp(xi) * _T_initial;

        // 3. 提取内参并计算 3D 点在当前相机坐标系下的位置
        Vector3d pos_cam = T_current * _pos3d;
        double fx = _K(0, 0), fy = _K(1, 1), cx = _K(0, 2), cy = _K(1, 2);
        double X = pos_cam[0], Y = pos_cam[1], Z = pos_cam[2];
        double Z2 = Z * Z;

        // 4. 计算残差 (公式: 预测像素值 - 真实观测像素值)
        double u_pred = fx * X / Z + cx;
        double v_pred = fy * Y / Z + cy;
        residuals[0] = u_pred - _measurement[0];
        residuals[1] = v_pred - _measurement[1];

        // 5. 手动填写雅可比矩阵 (解析求导)
        if (jacobians && jacobians[0]) {
            Map<Matrix<double, 2, 6, RowMajor>> J(jacobians[0]);

            // 李代数左乘扰动模型下的 2x6 雅可比矩阵
            J(0, 0) = fx / Z;
            J(0, 1) = 0;
            J(0, 2) = -fx * X / Z2;
            J(0, 3) = -fx * X * Y / Z2;
            J(0, 4) = fx + fx * X * X / Z2;
            J(0, 5) = -fx * Y / Z;

            J(1, 0) = 0;
            J(1, 1) = fy / Z;
            J(1, 2) = -fy * Y / Z2;
            J(1, 3) = -fy - fy * Y * Y / Z2;
            J(1, 4) = fy * X * Y / Z2;
            J(1, 5) = fy * X / Z;
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
// BA 优化函数接口
// ---------------------------------------------------------
void bundleAdjustment(const vector<cv::Point3f>& points_3d,
                      const vector<cv::Point2f>& points_2d,
                      const cv::Mat& K,
                      cv::Mat& R, cv::Mat& t) {
    
    // 将 OpenCV 数据转为 Eigen / Sophus 格式作为初值
    Matrix3d R_eigen;
    cv::cv2eigen(R, R_eigen);
    Vector3d t_eigen(t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0));
    Sophus::SE3d pose_initial(R_eigen, t_eigen);

    Matrix3d K_eigen;
    cv::cv2eigen(K, K_eigen);

    // 【核心精髓】
    // 我们不把庞大的位姿矩阵喂给求解器，而是创建一个初始化为 0 的 6 维扰动向量
    // 让 Ceres 专门负责把这个 xi 优化到最佳
    double xi[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    ceres::Problem problem;
    
    for (size_t i = 0; i < points_3d.size(); ++i) {
        ceres::CostFunction* cost_function = new PoseOptimizationCostFunction(
            Vector3d(points_3d[i].x, points_3d[i].y, points_3d[i].z),
            Vector2d(points_2d[i].x, points_2d[i].y),
            K_eigen,
            pose_initial
        );
        
        // 挂载误差项，并加入 Huber 核函数以抵御误匹配带来的野点干扰
        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), xi);
    }

    // 配置并运行求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 20;
    
    cout << "\n[Ceres] 开始执行 Bundle Adjustment (手动解析求导)..." << endl;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;

    // 优化结束后，我们将最终算出的扰动 xi_opt 左乘回初始位姿
    Map<const Matrix<double, 6, 1>> xi_opt(xi);
    Sophus::SE3d pose_optimized = Sophus::SE3d::exp(xi_opt) * pose_initial;

    // 将优化后的结果写回 OpenCV 的 R 和 t 中
    cv::eigen2cv(pose_optimized.rotationMatrix(), R);
    Vector3d t_opt = pose_optimized.translation();
    t = (cv::Mat_<double>(3, 1) << t_opt(0), t_opt(1), t_opt(2));
    
    cout << "[Ceres] BA 优化完成！" << endl;
}