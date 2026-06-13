#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <sophus/se3.hpp>

namespace vo_math
{

    // =========================================================================
    // 模块 1：基础旋转几何与四元数工具（用于非 Ceres 纯 double 运算环境）
    // =========================================================================

    /**
     * @brief 旋转向量 -> 3x3 旋转矩阵 (标准罗德里格斯变换)
     */
    template <typename T>
    inline Eigen::Matrix<T, 3, 3> RodriguesVectorToMatrix(const Eigen::Matrix<T, 3, 1> &v)
    {
        Eigen::Matrix<T, 3, 3> R = Eigen::Matrix<T, 3, 3>::Identity();
        T theta = v.norm();
        if (theta < T(1e-6))
            return R;

        Eigen::Matrix<T, 3, 1> k = v / theta;
        Eigen::Matrix<T, 3, 3> K;
        K << T(0.0), -k.z(), k.y(),
            k.z(), T(0.0), -k.x(),
            -k.y(), k.x(), T(0.0);

        using std::cos;
        using std::sin;
        R = R + sin(theta) * K + (T(1.0) - cos(theta)) * K * K;
        return R;
    }

    /**
     * @brief 3x3 旋转矩阵 -> 旋转向量 (罗德里格斯逆变换)
     */
    template <typename T>
    inline Eigen::Matrix<T, 3, 1> RodriguesMatrixToVector(const Eigen::Matrix<T, 3, 3> &R)
    {
        Eigen::Matrix<T, 3, 1> v = Eigen::Matrix<T, 3, 1>::Zero();
        T tr = R.trace();
        T cos_theta = (tr - 1.0) / 2.0;

        if (cos_theta > T(1.0))
            cos_theta = T(1.0);
        if (cos_theta < T(-1.0))
            cos_theta = T(-1.0);

        using std::abs;
        using std::acos;
        using std::sin;
        T theta = acos(cos_theta);
        if (theta < T(1e-6))
            return v;

        T sin_theta = sin(theta);
        if (abs(sin_theta) > T(1e-5))
        {
            v.x() = (R(2, 1) - R(1, 2)) / (2.0 * sin_theta);
            v.y() = (R(0, 2) - R(2, 0)) / (2.0 * sin_theta);
            v.z() = (R(1, 0) - R(0, 1)) / (2.0 * sin_theta);
            v = v * theta;
        }
        return v;
    }

    /**
     * @brief 3维向量 -> 3x3 反对称矩阵 (Hat 算子)
     */
    template <typename T>
    inline Eigen::Matrix<T, 3, 3> VectorToSkewSymmetric(const Eigen::Matrix<T, 3, 1> &v)
    {
        Eigen::Matrix<T, 3, 3> K;
        K << T(0.0), -v.z(), v.y(),
            v.z(), T(0.0), -v.x(),
            -v.y(), v.x(), T(0.0);
        return K;
    }

    /**
     * @brief 模板化欧拉角(RPY) -> 3x3 旋转矩阵
     */
    template <typename T>
    inline Eigen::Matrix<T, 3, 3> EulerToMatrix(const Eigen::Matrix<T, 3, 1> &euler)
    {
        T r = euler[0];
        T p = euler[1];
        T y = euler[2];
        Eigen::Matrix<T, 3, 3> Rx, Ry, Rz;
        using std::cos;
        using std::sin;
        Rx << T(1.0), T(0.0), T(0.0), T(0.0), cos(r), -sin(r), T(0.0), sin(r), cos(r);
        Ry << cos(p), T(0.0), sin(p), T(0.0), T(1.0), T(0.0), -sin(p), T(0.0), cos(p);
        Rz << cos(y), -sin(y), T(0.0), sin(y), cos(y), T(0.0), T(0.0), T(0.0), T(1.0);
        return Rz * Ry * Rx;
    }

    /**
     * @brief 3x3 旋转矩阵 -> 欧拉角 [roll, pitch, yaw]
     */
    template <typename T>
    inline Eigen::Matrix<T, 3, 1> MatrixToEuler(const Eigen::Matrix<T, 3, 3> &R)
    {
        Eigen::Matrix<T, 3, 1> euler;
        using std::abs;
        using std::asin;
        using std::atan2;
        using std::cos;
        euler[1] = asin(-R(2, 0));
        if (abs(cos(euler[1])) > T(1e-5))
        {
            euler[0] = atan2(R(2, 1), R(2, 2));
            euler[2] = atan2(R(1, 0), R(0, 0));
        }
        else
        {
            euler[0] = T(0.0);
            euler[2] = atan2(-R(0, 1), R(1, 1));
        }
        return euler;
    }

    /**
     * @brief 3维硬件级加速计算汉明距离算子
     */
    inline int ComputeHammingDistance(const uint8_t *desc1, const uint8_t *desc2)
    {
        int dist = 0;
        for (size_t i = 0; i < 32; ++i)
        {
            dist += __builtin_popcount(desc1[i] ^ desc2[i]);
        }
        return dist;
    }

    // =========================================================================
    // 模块 2：通用重投影误差计算接口 (模板函数，完美兼容 double 与 Ceres::Jet)
    // =========================================================================
    template <typename T>
    inline bool ComputeReprojectionError(const Eigen::Matrix<T, 3, 3> &R, const Eigen::Matrix<T, 3, 1> &t,
                                         const Eigen::Matrix<T, 3, 1> &Pw, const Eigen::Vector2d &obs,
                                         double fx, double fy, double cx, double cy, T *out_residual)
    {
        Eigen::Matrix<T, 3, 1> Pc = R * Pw + t;
        if (Pc.z() <= T(1e-5))
        {
            out_residual[0] = T(0.0);
            out_residual[1] = T(0.0);
            return false;
        }
        T u_pred = T(fx) * Pc.x() / Pc.z() + T(cx);
        T v_pred = T(fy) * Pc.y() / Pc.z() + T(cy);
        out_residual[0] = u_pred - T(obs.x());
        out_residual[1] = v_pred - T(obs.y());
        return true;
    }

    // =========================================================================
    // 模块 3：其余高阶几何和消元算子保持完全稳定...
    // =========================================================================
    inline Eigen::Matrix<double, 2, 6> GetPoseJacobianAnalytical(const Eigen::Vector3d &Pc, double fx, double fy)
    {
        Eigen::Matrix<double, 2, 6> J = Eigen::Matrix<double, 2, 6>::Zero();
        double X = Pc.x();
        double Y = Pc.y();
        double Z = Pc.z();
        double Z2 = Z * Z;
        if (Z < 1e-5)
            return J;
        J(0, 0) = fx / Z;
        J(0, 2) = -fx * X / Z2;
        J(0, 3) = -fx * X * Y / Z2;
        J(0, 4) = fx + fx * X * X / Z2;
        J(0, 5) = -fx * Y / Z;
        J(1, 1) = fy / Z;
        J(1, 2) = -fy * Y / Z2;
        J(1, 3) = -fy - fy * Y * Y / Z2;
        J(1, 4) = fy * X * Y / Z2;
        J(1, 5) = fy * X / Z;
        return J;
    }

    inline bool SolveICPBySVD(const std::vector<Eigen::Vector3d> &src_pts, const std::vector<Eigen::Vector3d> &dst_pts, Eigen::Matrix3d &out_R, Eigen::Vector3d &out_t)
    {
        if (src_pts.size() != dst_pts.size() || src_pts.size() < 3)
            return false;
        size_t N = src_pts.size();
        Eigen::Vector3d c_src(0, 0, 0), c_dst(0, 0, 0);
        for (size_t i = 0; i < N; ++i)
        {
            c_src += src_pts[i];
            c_dst += dst_pts[i];
        }
        c_src /= (double)N;
        c_dst /= (double)N;
        Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < N; ++i)
        {
            W += (src_pts[i] - c_src) * (dst_pts[i] - c_dst).transpose();
        }
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        out_R = svd.matrixV() * svd.matrixU().transpose();
        if (out_R.determinant() < 0.0)
        {
            Eigen::Matrix3d V = svd.matrixV();
            V.col(2) *= -1.0;
            out_R = V * svd.matrixU().transpose();
        }
        out_t = c_dst - out_R * c_src;
        return true;
    }

    inline bool TriangulatePointSVD(const Sophus::SE3d &T_w_c1, const Sophus::SE3d &T_w_c2, const Eigen::Vector2d &kp1, const Eigen::Vector2d &kp2, Eigen::Vector3d &out_Pw)
    {
        Eigen::Matrix<double, 3, 4> P1 = T_w_c1.inverse().matrix().topLeftCorner<3, 4>();
        Eigen::Matrix<double, 3, 4> P2 = T_w_c2.inverse().matrix().topLeftCorner<3, 4>();
        Eigen::Matrix4d A;
        A.row(0) = kp1.x() * P1.row(2) - P1.row(0);
        A.row(1) = kp1.y() * P1.row(2) - P1.row(1);
        A.row(2) = kp2.x() * P2.row(2) - P2.row(0);
        A.row(3) = kp2.y() * P2.row(2) - P2.row(1);
        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4d X = svd.matrixV().col(3);
        if (std::abs(X[3]) < 1e-6)
            return false;
        out_Pw = X.head<3>() / X[3];
        if ((T_w_c1.inverse() * out_Pw).z() <= 0.0 || (T_w_c2.inverse() * out_Pw).z() <= 0.0)
            return false;
        return true;
    }

    template <typename T>
    inline bool ProjectWorldPointToPixel(const Eigen::Matrix<T, 3, 3> &R, const Eigen::Matrix<T, 3, 1> &t, const Eigen::Matrix<T, 3, 1> &Pw, double fx, double fy, double cx, double cy, Eigen::Matrix<T, 2, 1> &out_uv)
    {
        Eigen::Matrix<T, 3, 1> Pc = R * Pw + t;
        if (Pc.z() <= T(1e-4))
            return false;
        out_uv.x() = T(fx) * Pc.x() / Pc.z() + T(cx);
        out_uv.y() = T(fy) * Pc.y() / Pc.z() + T(cy);
        return true;
    }

} // namespace vo_math