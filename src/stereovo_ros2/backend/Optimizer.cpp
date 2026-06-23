#include "backend/Optimizer.h"      // 引入本类声明
#include "core/KeyFrame.h"          // 跨模块：引入前端关键帧
#include "core/MapPoint.h"          // 跨模块：引入前端地图点
#include "utils/Parameters.h"       // 跨模块：引入工具层参数
#include <ceres/ceres.h>    // 引入谷歌 Ceres Solver 非线性最小二乘求解器的核心头文件
#include <ceres/rotation.h> // 引入 Ceres 的旋转数学库
#include <Eigen/Core>       // 引入 Eigen 矩阵基础库
#include <Eigen/Geometry>   // 引入 Eigen 几何库
#include <Eigen/Cholesky>   // 【新增】引入矩阵 Cholesky 分解库，用于对 H_prior 进行 LLT 分解
#include <map>              // 引入标准字典容器
#include <unordered_map>    // 引入无序哈希字典容器
#include <vector>           // 引入标准动态数组容器
#include <iostream>         // 引入标准输入输出流

// =========================================================================
// 【新增辅助数学工具】计算三维向量的反对称矩阵 (Skew-Symmetric Matrix)
// 作用：用于解析求解相机旋转扰动的雅可比矩阵
// =========================================================================
inline Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
    return m;
}

// =========================================================================
// 1. 基于主导帧逆深度的重投影误差代价函数 (1D Inverse Depth Residual)
// 优点：省去 1/lambda 的除法运算，采用 VINS-Mono 经典的齐次形式乘法，数值极其稳定，无惧零深度突变
// =========================================================================
struct SREPROJECTION_INVERSE_DEPTH_ERROR
{
    // 构造函数：传入特征点在主导帧(Host)上的归一化平面坐标，以及目标帧(Target)上的实际测量像素坐标与相机内参
    SREPROJECTION_INVERSE_DEPTH_ERROR(double host_u_norm, double host_v_norm, 
                                      double target_u, double target_v,
                                      double fx, double fy, double cx, double cy)
        : host_un(host_u_norm), host_vn(host_v_norm), 
          target_u(target_u), target_v(target_v),
          fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T *const host_pose, const T *const target_pose, const T *const inv_depth, T *residuals) const
    {
        // 1. 解包主导帧位姿块 Host Pose
        T t_h[3] = {host_pose[0], host_pose[1], host_pose[2]};
        T q_h[4] = {host_pose[3], host_pose[4], host_pose[5], host_pose[6]};

        // 2. 解包当前观测目标帧位姿块 Target Pose
        T t_t[3] = {target_pose[0], target_pose[1], target_pose[2]};
        T q_t[4] = {target_pose[3], target_pose[4], target_pose[5], target_pose[6]};

        // 3. 提取 1 维逆深度状态量
        T lambda = inv_depth[0];

        // 4. 定义该点在主导帧下深度为 1 的虚拟 3D 射线坐标
        T p_host_scaled[3] = {T(host_un), T(host_vn), T(1.0)};

        // 5. 核心代数重构：将主导帧下的射线投影旋转到世界坐标系
        T p_w_rot[3];
        ceres::QuaternionRotatePoint(q_h, p_host_scaled, p_w_rot);

        // 6. 齐次线性变换技巧：将世界系位移叠加并引入逆深度缩放，完全规避传统除以 lambda 的不稳定性
        T p_w_trans[3] = {
            p_w_rot[0] + lambda * (t_h[0] - t_t[0]),
            p_w_rot[1] + lambda * (t_h[1] - t_t[1]),
            p_w_trans[2] = p_w_rot[2] + lambda * (t_h[2] - t_t[2])
        };

        // 7. 将合并后的齐次世界点重投影变换到目标相机坐标系下：P_target = R_cw_target * p_w_trans
        T q_t_inv[4] = {q_t[0], -q_t[1], -q_t[2], -q_t[3]}; // 对目标帧旋转求逆
        T p_target[3];
        ceres::QuaternionRotatePoint(q_t_inv, p_w_trans, p_target);

        // 极值保护：深度截断异常规避
        if (p_target[2] <= T(1e-4))
        {
            residuals[0] = T(1111.0);
            residuals[1] = T(1111.0);
            return true;
        }

        // 8. 投影到目标帧图像平面
        T xp = p_target[0] / p_target[2];
        T yp = p_target[1] / p_target[2];
        T predicted_u = T(fx) * xp + T(cx);
        T predicted_v = T(fy) * yp + T(cy);

        // 9. 计算像素残差
        residuals[0] = T(target_u) - predicted_u;
        residuals[1] = T(target_v) - predicted_v;

        return true;
    }

    // 静态工厂构建函数：2 维像素残差，7 维主导帧位姿，7 维目标观测帧位姿，1 维逆深度参数块
    static ceres::CostFunction *Create(double host_u_norm, double host_v_norm, 
                                       double target_u, double target_v,
                                       double fx, double fy, double cx, double cy)
    {
        return (new ceres::AutoDiffCostFunction<SREPROJECTION_INVERSE_DEPTH_ERROR, 2, 7, 7, 1>(
            new SREPROJECTION_INVERSE_DEPTH_ERROR(host_u_norm, host_v_norm, target_u, target_v, fx, fy, cx, cy)));
    }

    double host_un, host_vn;
    double target_u, target_v;
    double fx, fy, cx, cy;
};

// =========================================================================
// 2. 位姿边缘化状态先验代价函数 (Pose Prior Factor)
// =========================================================================
struct PosePriorFactor
{
    PosePriorFactor(const Eigen::Vector3d &prior_t, const Eigen::Quaterniond &prior_q, const Eigen::Matrix<double, 6, 6> &sqrt_info)
        : m_prior_t(prior_t), m_prior_q(prior_q), m_sqrt_info(sqrt_info) {}

    template <typename T>
    bool operator()(const T *const camera_pose, T *residuals) const
    {
        T t[3] = {camera_pose[0], camera_pose[1], camera_pose[2]};
        T q[4] = {camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6]};

        T qw1 = T(m_prior_q.w()), qx1 = T(-m_prior_q.x()), qy1 = T(-m_prior_q.y()), qz1 = T(-m_prior_q.z());
        T qw2 = q[0], qx2 = q[1], qy2 = q[2], qz2 = q[3];

        T dq_w = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2;
        T dq_x = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2;
        T dq_y = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2;
        T dq_z = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2;

        T sign = (dq_w >= T(0.0)) ? T(1.0) : T(-1.0);

        Eigen::Matrix<T, 6, 1> raw_residuals;
        raw_residuals[0] = t[0] - T(m_prior_t.x());
        raw_residuals[1] = t[1] - T(m_prior_t.y());
        raw_residuals[2] = t[2] - T(m_prior_t.z());
        raw_residuals[3] = T(2.0) * sign * dq_x;
        raw_residuals[4] = T(2.0) * sign * dq_y;
        raw_residuals[5] = T(2.0) * sign * dq_z;

        Eigen::Matrix<T, 6, 1> weighted_residuals = m_sqrt_info.cast<T>() * raw_residuals;
        for (int i = 0; i < 6; ++i) residuals[i] = weighted_residuals[i];
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d &prior_t, const Eigen::Quaterniond &prior_q, const Eigen::Matrix<double, 6, 6> &sqrt_info)
    {
        return (new ceres::AutoDiffCostFunction<PosePriorFactor, 6, 7>(new PosePriorFactor(prior_t, prior_q, sqrt_info)));
    }

    Eigen::Vector3d m_prior_t;
    Eigen::Quaterniond m_prior_q;
    Eigen::Matrix<double, 6, 6> m_sqrt_info;
};

// =========================================================================
// 3. 全新重构的局部滑窗逆深度 BA + 舒尔补边缘化传导算法主函数
// =========================================================================
void Optimizer::LocalBundleAdjustment(std::shared_ptr<Map> pMap, int windowSize)
{
    if (!pMap) return;

    std::vector<std::shared_ptr<KeyFrame>> allKFs = pMap->GetAllKeyFrames();
    if (allKFs.size() < 2) return;

    // 1. 筛选并截取滑动窗口内的最新激活帧
    std::vector<std::shared_ptr<KeyFrame>> activeKFs;
    int startIdx = std::max(0, static_cast<int>(allKFs.size()) - windowSize);
    for (size_t i = startIdx; i < allKFs.size(); ++i) activeKFs.push_back(allKFs[i]);

    ceres::Problem problem;

    std::map<unsigned long, std::vector<double>> mapKFIdToParameterBlock;
    std::map<int, double> mapMPIdToInvDepth;                 // 存储地图点 1 自由度的逆深度参数块
    std::map<int, std::shared_ptr<KeyFrame>> mapMPIdToHostKF;// 存储每个地图点在当前滑窗内的“主导关键帧”
    std::map<int, std::shared_ptr<MapPoint>> mapIdToMPRef;

    double cam_fx = Parameters::fx;
    double cam_fy = Parameters::fy;
    double cam_cx = Parameters::cx;
    double cam_cy = Parameters::cy;

    // 2. 装载激活帧位姿到 Ceres 参数块中
    for (size_t i = 0; i < activeKFs.size(); ++i)
    {
        auto kf = activeKFs[i];
        Eigen::Isometry3d Twc = kf->GetPose();
        Eigen::Vector3d t = Twc.translation();
        Eigen::Quaterniond q(Twc.rotation());

        std::vector<double> pose_block = {t.x(), t.y(), t.z(), q.w(), q.x(), q.y(), q.z()};
        mapKFIdToParameterBlock[kf->mId] = pose_block;
    }

    // 提前构建全局点云的 O(1) 哈希快速查找表，消除多层循环内部的线性搜索
    std::vector<std::shared_ptr<MapPoint>> allMPs = pMap->GetAllMapPoints();
    std::unordered_map<int, std::shared_ptr<MapPoint>> mapIdToMP;
    for (const auto &mp : allMPs) mapIdToMP[mp->GetFeatureId()] = mp;

    // 3. 为主窗口内的每一个地图点选拔选定它的“主导帧（Host）”并计算初始逆深度
    for (auto kf : activeKFs)
    {
        for (const auto &obs : kf->mmObservations)
        {
            int mp_id = obs.first;
            auto it_mp = mapIdToMP.find(mp_id);
            if (it_mp == mapIdToMP.end()) continue;

            // 如果该地图点还没有指定主导帧，则在滑窗内最早看见它的那一帧被确立为主导帧
            if (mapMPIdToHostKF.find(mp_id) == mapMPIdToHostKF.end())
            {
                std::shared_ptr<MapPoint> targetMP = it_mp->second;
                mapMPIdToHostKF[mp_id] = kf;
                mapIdToMPRef[mp_id] = targetMP;

                // 计算地图点关于主导帧的初始逆深度
                Eigen::Vector3d P_w = targetMP->GetWorldPos();
                Eigen::Isometry3d Twc_host = kf->GetPose();
                Eigen::Vector3d P_host = Twc_host.inverse() * P_w; // 变换到主导相机坐标系下

                double depth = P_host.z();
                if (depth < 0.1) depth = 1.0; 
                mapMPIdToInvDepth[mp_id] = 1.0 / depth; // 写入 1 维逆深度初值
            }
        }
    }

    // 4. 建立 1D 逆深度重投影误差约束
    for (auto kf : activeKFs)
    {
        unsigned long target_kf_id = kf->mId;
        for (const auto &obs : kf->mmObservations)
        {
            int mp_id = obs.first;
            cv::Point2f pt_target = obs.second; // 当前目标观测帧上的 2D 像素坐标

            if (mapMPIdToHostKF.find(mp_id) == mapMPIdToHostKF.end()) continue;
            auto hostKF = mapMPIdToHostKF[mp_id];

            // 【性能裁剪机制】如果当前观测帧本身就是该点的主导帧，则残差永远为 0，直接略过
            if (target_kf_id == hostKF->mId) continue;

            // 提取主导帧上的对应 2D 特征观测，并换算为去畸变后的归一化相机坐标
            cv::Point2f pt_host = hostKF->mmObservations[mp_id];
            double host_u_norm = (pt_host.x - cam_cx) / cam_fx;
            double host_v_norm = (pt_host.y - cam_cy) / cam_fy;

            // 实例化逆深度代价函数
            ceres::CostFunction *cost_function = SREPROJECTION_INVERSE_DEPTH_ERROR::Create(
                host_u_norm, host_v_norm, pt_target.x, pt_target.y, cam_fx, cam_fy, cam_cx, cam_cy);

            // 自适应 HuberLoss 核函数机制
            std::shared_ptr<MapPoint> targetMP = mapIdToMPRef[mp_id];
            int obs_count = targetMP->GetObservationCount();
            double huber_delta = (obs_count >= 5) ? 1.0 : ((obs_count >= 2) ? 1.5 : 3.0);
            ceres::LossFunction *adaptive_loss_function = new ceres::HuberLoss(huber_delta);

            // 向 Ceres 因子图中注入约束边
            problem.AddResidualBlock(cost_function, adaptive_loss_function,
                                     mapKFIdToParameterBlock[hostKF->mId].data(),
                                     mapKFIdToParameterBlock[target_kf_id].data(),
                                     &mapMPIdToInvDepth[mp_id]);
        }
    }

    // 5. 对 1D 逆深度施加物理界限有界约束，防止解算严重震荡出现负深度
    for (auto &pair : mapMPIdToInvDepth)
    {
        problem.AddParameterBlock(&pair.second, 1);
        problem.SetParameterLowerBound(&pair.second, 0, 0.001); // 限制最远深度为 1000 米
        problem.SetParameterUpperBound(&pair.second, 0, 10.0);  // 限制最近深度为 0.1 米
    }

    // 6. 流形设置与滑窗首帧【基于舒尔补的边缘化动态传导】
    for (size_t i = 0; i < activeKFs.size(); ++i)
    {
        auto kf = activeKFs[i];
        double *pose_data = mapKFIdToParameterBlock[kf->mId].data();

#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
        problem.SetManifold(pose_data, new ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::QuaternionManifold>(
                                           ceres::EuclideanManifold<3>(), ceres::QuaternionManifold()));
#else
        ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization();
        problem.SetParameterization(pose_data, new ceres::ProductParameterization(new ceres::IdentityParameterization(3), quaternion_parameterization));
#endif

        // =========================================================================
        // 【核心机制重构】基于舒尔补的“边缘化”（Marginalization）动态传导逻辑
        // 作用：彻底取消固定值 50.0 和 100.0，改为由当前老帧特征点几何结构动态解算信息矩阵！
        // =========================================================================
        if (i == 0)
        {
            Eigen::Isometry3d Twc_prior = kf->GetPose();
            Eigen::Vector3d prior_t = Twc_prior.translation();
            Eigen::Quaterniond prior_q(Twc_prior.rotation());

            // 初始化全动态海森矩阵 H 和梯度向量 g
            // 状态排布：最老帧位姿 x_m (6自由度: [平移, 旋转扰动]), 活跃地图点 x_r (3自由度)
            Eigen::Matrix<double, 6, 6> H_mm = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 3> H_mr = Eigen::Matrix<double, 6, 3>::Zero();
            Eigen::Matrix<double, 3, 3> H_rr_sum = Eigen::Matrix<double, 3, 3>::Zero();

            double pixel_noise_sigma = 1.5; // 设图像追踪像素噪声标准差为 1.5 像素
            double omega_pixel = 1.0 / (pixel_noise_sigma * pixel_noise_sigma);
            int effective_obs_count = 0;

            // 遍历该最老帧身上的所有 2D 观测，解析求解空间几何雅可比
            for (const auto &obs : kf->mmObservations)
            {
                int mp_id = obs.first;
                auto it_mp = mapIdToMP.find(mp_id);
                if (it_mp == mapIdToMP.end()) continue;
                std::shared_ptr<MapPoint> mp = it_mp->second;

                Eigen::Vector3d P_w = mp->GetWorldPos();
                Eigen::Vector3d P_c = Twc_prior.inverse() * P_w; // 变换到相机坐标系
                double X = P_c.x(), Y = P_c.y(), Z = P_c.z();
                if (Z < 0.2) continue;
                
                effective_obs_count++;

                // A. 求解 2x3 的图像平面投影对相机系 3D 点的雅可比 J_proj
                Eigen::Matrix<double, 2, 3> J_proj;
                double inv_z = 1.0 / Z;
                double inv_z2 = inv_z * inv_z;
                J_proj << cam_fx * inv_z, 0.0, -cam_fx * X * inv_z2,
                          0.0, cam_fy * inv_z, -cam_fy * Y * inv_z2;

                // B. 求解 3x6 的相机系 3D 点对局部李代数扰动 [平移, 旋转] 的雅可比 J_space
                // 注意：Ceres 流形内平移是世界系相加，四元数是右乘扰动
                Eigen::Matrix<double, 3, 6> J_space;
                J_space.block<3, 3>(0, 0) = -Twc_prior.linear().transpose(); // dP_c / dt_world
                J_space.block<3, 3>(0, 3) = SkewSymmetric(P_c);               // dP_c / d\theta

                Eigen::Matrix<double, 2, 6> J_pose = J_proj * J_space;

                // C. 求解 2x3 的图像平面对世界地图点 3D 坐标的雅可比 J_point
                Eigen::Matrix<double, 2, 3> J_point = J_proj * Twc_prior.linear().transpose();

                // D. 动态累加高斯牛顿矩阵分块
                H_mm += J_pose.transpose() * omega_pixel * J_pose;
                H_mr += J_pose.transpose() * omega_pixel * J_point;
                H_rr_sum += J_point.transpose() * omega_pixel * J_point;
            }

            // 执行矩阵层面的舒尔补消元传导 (Schur Complement)
            Eigen::Matrix<double, 6, 6> H_prior = Eigen::Matrix<double, 6, 6>::Identity() * 1.0; // 基础弱约束保底
            if (effective_obs_count > 4)
            {
                Eigen::Matrix3d H_rr_inv = H_rr_sum.inverse();
                H_prior += H_mm - H_mr * H_rr_inv * H_mr.transpose();
            }
            else
            {
                // 极端无共视点环境下的退化安全保护
                H_prior.block<3, 3>(0, 0) *= 50.0;
                H_prior.block<3, 3>(3, 3) *= 100.0;
            }

            // 对实时派生出的 H_prior 进行 LLT 分解，安全提取 Ceres 所需的平方根信息权重矩阵
            Eigen::LLT<Eigen::Matrix<double, 6, 6>> lltOfH(H_prior);
            Eigen::Matrix<double, 6, 6> sqrt_info = lltOfH.matrixL().transpose();

            // 构造软约束因子边注入因子图
            ceres::CostFunction *prior_cost_function = PosePriorFactor::Create(prior_t, prior_q, sqrt_info);
            problem.AddResidualBlock(prior_cost_function, nullptr, pose_data);
        }
    }

    // 7. 配置高度精简、低迭代、高响应的求解器参数
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR; 
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 6;                  // 逆深度参数空间特征极佳，6 次迭代即可完全收敛
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 8. 优化成功，将精调后的位姿和【重新反投影恢复出的 3D 世界坐标】刷回数据库
    if (summary.termination_type == ceres::CONVERGENCE || summary.termination_type == ceres::USER_SUCCESS)
    {
        // A. 关键帧位姿恢复写回
        for (auto kf : activeKFs)
        {
            auto &data = mapKFIdToParameterBlock[kf->mId];
            Eigen::Vector3d t_opt(data[0], data[1], data[2]);
            Eigen::Quaterniond q_opt(data[3], data[4], data[5], data[6]);

            Eigen::Isometry3d Twc_opt = Eigen::Isometry3d::Identity();
            Twc_opt.linear() = q_opt.toRotationMatrix();
            Twc_opt.translation() = t_opt;

            kf->SetPose(Twc_opt);
        }

        // B. 地图路标点坐标反投影转换写回
        // 数学公式：P_world = R_world_host * ( (1/lambda_opt) * [u_n, v_n, 1]^T ) + t_world_host
        for (const auto &pair : mapMPIdToInvDepth)
        {
            int mp_id = pair.first;
            double lambda_opt = pair.second; 
            auto hostKF = mapMPIdToHostKF[mp_id];

            auto &host_pose_data = mapKFIdToParameterBlock[hostKF->mId];
            Eigen::Vector3d t_h_opt(host_pose_data[0], host_pose_data[1], host_pose_data[2]);
            Eigen::Quaterniond q_h_opt(host_pose_data[3], host_pose_data[4], host_pose_data[5], host_pose_data[6]);

            cv::Point2f pt_host = hostKF->mmObservations[mp_id];
            double host_u_norm = (pt_host.x - cam_cx) / cam_fx;
            double host_v_norm = (pt_host.y - cam_cy) / cam_fy;

            // 逆深度反投影恢复
            Eigen::Vector3d p_h_scaled(host_u_norm, host_v_norm, 1.0);
            Eigen::Vector3d pos_world_opt = q_h_opt * (p_h_scaled / lambda_opt) + t_h_opt;

            mapIdToMPRef[mp_id]->SetWorldPos(pos_world_opt);
        }
    }
}