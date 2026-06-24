#include "backend/Optimizer.h"
#include "core/KeyFrame.h"
#include "core/MapPoint.h"
#include "utils/Parameters.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Cholesky>
#include <map>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <cmath>

// =========================================================================
// 静态工具函数（仅当前文件可见，已移除 namespace）
// =========================================================================

// 三维向量反对称矩阵
static inline Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
    return m;
}

/**
 * @brief 位姿右扰动更新（李代数右乘模型，保证四元数流形约束）
 */
static inline void UpdatePose(const Eigen::Vector3d& t_old, const Eigen::Quaterniond& q_old,
                       const Eigen::Matrix<double, 6, 1>& delta,
                       Eigen::Vector3d& t_new, Eigen::Quaterniond& q_new)
{
    // 平移增量从相机系转换到世界系
    t_new = t_old + q_old.toRotationMatrix() * delta.head<3>();

    // 旋转增量：小角度四元数近似 + 右乘更新
    Eigen::Vector3d delta_theta = delta.tail<3>();
    Eigen::Quaterniond delta_q(
        1.0,
        delta_theta.x() / 2.0,
        delta_theta.y() / 2.0,
        delta_theta.z() / 2.0
    );
    delta_q.normalize();
    q_new = q_old * delta_q;
    q_new.normalize();
}

/**
 * @brief 计算两个位姿之间的右扰动李代数误差
 */
static inline Eigen::Matrix<double, 6, 1> ComputePoseError(
    const Eigen::Vector3d& t_prior, const Eigen::Quaterniond& q_prior,
    const Eigen::Vector3d& t_curr, const Eigen::Quaterniond& q_curr)
{
    Eigen::Matrix<double, 6, 1> error;
    // 相对平移：转换到先验位姿的局部坐标系
    error.head<3>() = q_prior.inverse().toRotationMatrix() * (t_curr - t_prior);
    // 相对旋转：四元数差转旋转向量
    Eigen::Quaterniond q_rel = q_prior.inverse() * q_curr;
    Eigen::AngleAxisd aa_rel(q_rel);
    error.tail<3>() = aa_rel.angle() * aa_rel.axis();
    return error;
}

/**
 * @brief 计算逆深度重投影残差 + 解析雅可比
 */
static inline bool ComputeReprojectionResidualAndJacobians(
    const Eigen::Vector3d& t_host, const Eigen::Quaterniond& q_host,
    const Eigen::Vector3d& t_target, const Eigen::Quaterniond& q_target,
    double inv_depth,
    double host_un, double host_vn,
    double target_u, double target_v,
    double fx, double fy, double cx, double cy,
    Eigen::Vector2d& residual,
    Eigen::Matrix<double, 2, 6>& J_host,   // 残差对主导帧位姿的雅可比
    Eigen::Matrix<double, 2, 6>& J_target, // 残差对目标帧位姿的雅可比
    Eigen::Matrix<double, 2, 1>& J_lambda  // 残差对逆深度的雅可比
)
{
    Eigen::Vector3d p_h0(host_un, host_vn, 1.0); // 主导帧归一化射线
    double lambda = inv_depth;

    // 1. 计算世界系3D点
    Eigen::Vector3d P_h = p_h0 / lambda;
    Eigen::Vector3d P_w = q_host * P_h + t_host;

    // 2. 变换到目标相机系
    Eigen::Vector3d P_t = q_target.inverse() * (P_w - t_target);
    double X = P_t.x(), Y = P_t.y(), Z = P_t.z();

    // 深度异常保护
    if (Z < 1e-4)
    {
        residual.setConstant(1111.0);
        J_host.setZero();
        J_target.setZero();
        J_lambda.setZero();
        return false;
    }

    // 3. 投影雅可比：像素坐标对相机系3D点的导数
    double inv_z = 1.0 / Z;
    double inv_z2 = inv_z * inv_z;
    Eigen::Matrix<double, 2, 3> J_proj;
    J_proj << fx * inv_z, 0.0, -fx * X * inv_z2,
              0.0, fy * inv_z, -fy * Y * inv_z2;

    // 4. 对目标帧位姿的雅可比（右扰动模型）
    Eigen::Matrix<double, 3, 6> dP_t_dxi_target;
    dP_t_dxi_target.block<3,3>(0,0) = -Eigen::Matrix3d::Identity();
    dP_t_dxi_target.block<3,3>(0,3) = SkewSymmetric(P_t);
    J_target = - J_proj * dP_t_dxi_target;

    // 5. 对主导帧位姿的雅可比（右扰动模型）
    Eigen::Matrix<double, 3, 6> dP_w_dxi_host;
    dP_w_dxi_host.block<3,3>(0,0) = q_host.toRotationMatrix();
    dP_w_dxi_host.block<3,3>(0,3) = q_host.toRotationMatrix() * SkewSymmetric(P_h);
    Eigen::Matrix<double, 3, 6> dP_t_dxi_host = q_target.inverse().toRotationMatrix() * dP_w_dxi_host;
    J_host = - J_proj * dP_t_dxi_host;

    // 6. 对逆深度的雅可比
    Eigen::Vector3d dP_h_dlambda = - p_h0 / (lambda * lambda);
    Eigen::Vector3d dP_t_dlambda = q_target.inverse().toRotationMatrix() * q_host.toRotationMatrix() * dP_h_dlambda;
    J_lambda = - J_proj * dP_t_dlambda;

    // 7. 计算像素残差
    double predicted_u = fx * X / Z + cx;
    double predicted_v = fy * Y / Z + cy;
    residual(0) = target_u - predicted_u;
    residual(1) = target_v - predicted_v;

    return true;
}

// 观测边结构体（全局静态，已移除 namespace）
struct ObservationEdge
{
    int host_idx;       // 主导帧索引
    int target_idx;     // 目标观测帧索引
    int mp_idx;         // 地图点索引
    double host_un;     // 主导帧归一化u坐标
    double host_vn;     // 主导帧归一化v坐标
    double target_u;    // 目标帧像素u
    double target_v;    // 目标帧像素v
    double huber_delta; // Huber核阈值
};

// =========================================================================
// 主函数：纯 Eigen 局部滑窗逆深度 BA + 舒尔补边缘化
// =========================================================================
void Optimizer::LocalBundleAdjustment(std::shared_ptr<Map> pMap, int windowSize)
{
    if (!pMap) return;
    std::vector<std::shared_ptr<KeyFrame>> allKFs = pMap->GetAllKeyFrames();
    if (allKFs.size() < 2) return;

    // ====================== 1. 筛选滑动窗口激活帧 ======================
    std::vector<std::shared_ptr<KeyFrame>> activeKFs;
    int startIdx = std::max(0, static_cast<int>(allKFs.size()) - windowSize);
    for (size_t i = startIdx; i < allKFs.size(); ++i)
        activeKFs.push_back(allKFs[i]);
    
    int num_kfs = activeKFs.size();
    int pose_dim = num_kfs * 6; // 位姿总维度（每个6自由度李代数）

    // 位姿状态存储 + ID到索引映射
    std::vector<Eigen::Vector3d> pose_t(num_kfs);
    std::vector<Eigen::Quaterniond> pose_q(num_kfs);
    std::unordered_map<unsigned long, int> kf_id_to_idx;
    for (int i = 0; i < num_kfs; ++i)
    {
        Eigen::Isometry3d Twc = activeKFs[i]->GetPose();
        pose_t[i] = Twc.translation();
        pose_q[i] = Eigen::Quaterniond(Twc.rotation());
        kf_id_to_idx[activeKFs[i]->mId] = i;
    }

    // 相机内参
    double cam_fx = Parameters::fx;
    double cam_fy = Parameters::fy;
    double cam_cx = Parameters::cx;
    double cam_cy = Parameters::cy;

    // ====================== 2. 地图点预处理 + 主导帧选拔 ======================
    std::vector<std::shared_ptr<MapPoint>> allMPs = pMap->GetAllMapPoints();
    std::unordered_map<int, std::shared_ptr<MapPoint>> mapIdToMP;
    for (const auto &mp : allMPs)
        mapIdToMP[mp->GetFeatureId()] = mp;

    std::unordered_map<int, int> mp_id_to_idx;   // 地图点ID→索引
    std::vector<double> inv_depth_vec;           // 逆深度参数数组
    std::vector<int> mp_host_kf_idx;             // 每个地图点的主导帧索引
    std::vector<std::shared_ptr<MapPoint>> mp_ref_vec; // 地图点指针数组

    for (int i = 0; i < num_kfs; ++i)
    {
        auto kf = activeKFs[i];
        for (const auto &obs : kf->mmObservations)
        {
            int mp_id = obs.first;
            auto it_mp = mapIdToMP.find(mp_id);
            if (it_mp == mapIdToMP.end()) continue;

            // 首次出现的地图点：设置主导帧 + 初始化逆深度
            if (mp_id_to_idx.find(mp_id) == mp_id_to_idx.end())
            {
                int mp_idx = inv_depth_vec.size();
                mp_id_to_idx[mp_id] = mp_idx;
                mp_host_kf_idx.push_back(i);
                mp_ref_vec.push_back(it_mp->second);

                // 计算初始逆深度
                Eigen::Vector3d P_w = it_mp->second->GetWorldPos();
                Eigen::Vector3d P_host = kf->GetPose().inverse() * P_w;
                double depth = P_host.z() < 0.1 ? 1.0 : P_host.z();
                inv_depth_vec.push_back(1.0 / depth);
            }
        }
    }
    int num_mps = inv_depth_vec.size();

    // ====================== 3. 预构建所有观测约束边 ======================
    std::vector<ObservationEdge> edges;
    for (int target_idx = 0; target_idx < num_kfs; ++target_idx)
    {
        auto kf = activeKFs[target_idx];
        for (const auto &obs : kf->mmObservations)
        {
            int mp_id = obs.first;
            cv::Point2f pt_target = obs.second;
            auto it_mp_idx = mp_id_to_idx.find(mp_id);
            if (it_mp_idx == mp_id_to_idx.end()) continue;
            
            int mp_idx = it_mp_idx->second;
            int host_idx = mp_host_kf_idx[mp_idx];
            if (host_idx == target_idx) continue; // 主导帧自身无残差

            // 主导帧归一化坐标
            cv::Point2f pt_host = activeKFs[host_idx]->mmObservations[mp_id];
            double host_un = (pt_host.x - cam_cx) / cam_fx;
            double host_vn = (pt_host.y - cam_cy) / cam_fy;

            // 自适应Huber阈值
            int obs_count = mp_ref_vec[mp_idx]->GetObservationCount();
            double huber_delta = (obs_count >= 5) ? 1.0 : ((obs_count >= 2) ? 1.5 : 3.0);

            edges.push_back({host_idx, target_idx, mp_idx, 
                            host_un, host_vn, 
                            pt_target.x, pt_target.y, 
                            huber_delta});
        }
    }

    // ====================== 4. 计算第一帧边缘化先验信息矩阵 ======================
    Eigen::Matrix<double, 6, 6> H_prior = Eigen::Matrix<double, 6, 6>::Identity() * 1.0;
    Eigen::Vector3d prior_t = pose_t[0];
    Eigen::Quaterniond prior_q = pose_q[0];
    {
        Eigen::Matrix<double, 6, 6> H_mm = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 3> H_mr = Eigen::Matrix<double, 6, 3>::Zero();
        Eigen::Matrix<double, 3, 3> H_rr_sum = Eigen::Matrix<double, 3, 3>::Zero();
        double pixel_noise_sigma = 1.5;
        double omega_pixel = 1.0 / (pixel_noise_sigma * pixel_noise_sigma);
        int effective_obs_count = 0;

        Eigen::Isometry3d Twc_prior = activeKFs[0]->GetPose();
        Eigen::Matrix3d R_prior = Twc_prior.rotation();

        for (const auto &obs : activeKFs[0]->mmObservations)
        {
            int mp_id = obs.first;
            auto it_mp = mapIdToMP.find(mp_id);
            if (it_mp == mapIdToMP.end()) continue;

            Eigen::Vector3d P_w = it_mp->second->GetWorldPos();
            Eigen::Vector3d P_c = Twc_prior.inverse() * P_w;
            double X = P_c.x(), Y = P_c.y(), Z = P_c.z();
            if (Z < 0.2) continue;
            effective_obs_count++;

            // 投影雅可比
            Eigen::Matrix<double, 2, 3> J_proj;
            double inv_z = 1.0 / Z;
            double inv_z2 = inv_z * inv_z;
            J_proj << cam_fx * inv_z, 0.0, -cam_fx * X * inv_z2,
                      0.0, cam_fy * inv_z, -cam_fy * Y * inv_z2;

            // 位姿雅可比
            Eigen::Matrix<double, 3, 6> J_space;
            J_space.block<3, 3>(0, 0) = -R_prior.transpose();
            J_space.block<3, 3>(0, 3) = SkewSymmetric(P_c);
            Eigen::Matrix<double, 2, 6> J_pose = J_proj * J_space;

            // 地图点雅可比
            Eigen::Matrix<double, 2, 3> J_point = J_proj * R_prior.transpose();

            H_mm += J_pose.transpose() * omega_pixel * J_pose;
            H_mr += J_pose.transpose() * omega_pixel * J_point;
            H_rr_sum += J_point.transpose() * omega_pixel * J_point;
        }

        if (effective_obs_count > 4)
        {
            Eigen::Matrix3d H_rr_inv = H_rr_sum.inverse();
            H_prior += H_mm - H_mr * H_rr_inv * H_mr.transpose();
        }
        else
        {
            // 退化场景保底约束
            H_prior.block<3, 3>(0, 0) *= 50.0;
            H_prior.block<3, 3>(3, 3) *= 100.0;
        }
    }

    // ====================== 5. Levenberg-Marquardt 迭代求解 ======================
    double lambda_lm = 1e-3;       // LM阻尼因子
    const double lambda_boost = 10.0;
    const double lambda_shrink = 0.1;
    const int max_iter = 6;        // 与原代码保持一致
    const double eps_converge = 1e-6;

    // 状态备份（用于更新失败回滚）
    std::vector<Eigen::Vector3d> last_pose_t = pose_t;
    std::vector<Eigen::Quaterniond> last_pose_q = pose_q;
    std::vector<double> last_inv_depth = inv_depth_vec;
    double last_cost = 0.0;
    bool first_iter = true;

    const double min_inv_depth = 0.001; // 最远深度1000米
    const double max_inv_depth = 10.0;  // 最近深度0.1米

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // ---------- 5.1 构建全局Hessian矩阵与梯度向量 ----------
        Eigen::MatrixXd H_xx = Eigen::MatrixXd::Zero(pose_dim, pose_dim);
        Eigen::VectorXd g_x = Eigen::VectorXd::Zero(pose_dim);

        std::vector<double> H_ll(num_mps, 0.0);        // 逆深度Hessian对角元
        std::vector<Eigen::VectorXd> H_xl(num_mps, Eigen::VectorXd::Zero(pose_dim)); // 位姿-逆深度交叉项
        std::vector<double> g_l(num_mps, 0.0);        // 逆深度梯度

        double total_cost = 0.0;

        // 遍历所有观测边累加
        for (const auto& edge : edges)
        {
            int host_start = edge.host_idx * 6;
            int target_start = edge.target_idx * 6;
            int mp_idx = edge.mp_idx;

            Eigen::Vector2d residual;
            Eigen::Matrix<double, 2, 6> J_host, J_target;
            Eigen::Matrix<double, 2, 1> J_lambda;

            bool valid = ComputeReprojectionResidualAndJacobians(
                pose_t[edge.host_idx], pose_q[edge.host_idx],
                pose_t[edge.target_idx], pose_q[edge.target_idx],
                inv_depth_vec[mp_idx],
                edge.host_un, edge.host_vn,
                edge.target_u, edge.target_v,
                cam_fx, cam_fy, cam_cx, cam_cy,
                residual, J_host, J_target, J_lambda
            );
            if (!valid) continue;

            // 自适应Huber核加权
            double res_norm = residual.norm();
            double huber_w = 1.0;
            if (res_norm > edge.huber_delta)
                huber_w = edge.huber_delta / res_norm;
            
            residual *= huber_w;
            J_host *= huber_w;
            J_target *= huber_w;
            J_lambda *= huber_w;

            total_cost += residual.squaredNorm();

            // 累加位姿Hessian分块
            H_xx.block<6,6>(host_start, host_start) += J_host.transpose() * J_host;
            H_xx.block<6,6>(host_start, target_start) += J_host.transpose() * J_target;
            H_xx.block<6,6>(target_start, host_start) += J_target.transpose() * J_host;
            H_xx.block<6,6>(target_start, target_start) += J_target.transpose() * J_target;

            // 累加位姿梯度
            g_x.segment<6>(host_start) += J_host.transpose() * residual;
            g_x.segment<6>(target_start) += J_target.transpose() * residual;

            // 累加逆深度相关项
            H_ll[mp_idx] += (J_lambda.transpose() * J_lambda)(0,0);
            H_xl[mp_idx].segment<6>(host_start) += J_host.transpose() * J_lambda;
            H_xl[mp_idx].segment<6>(target_start) += J_target.transpose() * J_lambda;
            g_l[mp_idx] += (J_lambda.transpose() * residual)(0,0);
        }

        // ---------- 5.2 加入第一帧边缘化先验约束 ----------
        {
            int first_start = 0;
            Eigen::Matrix<double,6,1> prior_error = ComputePoseError(prior_t, prior_q, pose_t[0], pose_q[0]);
            H_xx.block<6,6>(first_start, first_start) += H_prior;
            g_x.segment<6>(first_start) += H_prior * prior_error;
            total_cost += prior_error.transpose() * H_prior * prior_error;
        }

        // ---------- 5.3 舒尔补消去逆深度（仅需求解位姿增量） ----------
        Eigen::MatrixXd H_schur = H_xx;
        Eigen::VectorXd g_schur = g_x;
        for (int j = 0; j < num_mps; ++j)
        {
            if (H_ll[j] < 1e-12) continue;
            H_schur -= H_xl[j] * H_xl[j].transpose() / H_ll[j];
            g_schur -= H_xl[j] * g_l[j] / H_ll[j];
        }

        // ---------- 5.4 LM阻尼正则化 ----------
        Eigen::MatrixXd H_reg = H_schur;
        for (int i = 0; i < pose_dim; ++i)
            H_reg(i,i) += lambda_lm;

        // ---------- 5.5 求解位姿增量 ----------
        Eigen::LLT<Eigen::MatrixXd> llt(H_reg);
        if (llt.info() != Eigen::Success)
        {
            lambda_lm *= lambda_boost;
            continue;
        }
        Eigen::VectorXd delta_x = llt.solve(-g_schur);

        // ---------- 5.6 回代求解逆深度增量 ----------
        Eigen::VectorXd delta_l(num_mps);
        for (int j = 0; j < num_mps; ++j)
        {
            if (H_ll[j] < 1e-12)
            {
                delta_l(j) = 0.0;
                continue;
            }
            double rhs = -g_l[j] - H_xl[j].dot(delta_x);
            delta_l(j) = rhs / H_ll[j];
        }

        // ---------- 5.7 状态更新 ----------
        last_pose_t = pose_t;
        last_pose_q = pose_q;
        last_inv_depth = inv_depth_vec;
        last_cost = total_cost;

        // 更新位姿（流形约束）
        for (int i = 0; i < num_kfs; ++i)
        {
            Eigen::Matrix<double,6,1> delta = delta_x.segment<6>(i*6);
            UpdatePose(pose_t[i], pose_q[i], delta, pose_t[i], pose_q[i]);
        }

        // 更新逆深度 + 有界约束截断
        for (int j = 0; j < num_mps; ++j)
        {
            inv_depth_vec[j] += delta_l(j);
            inv_depth_vec[j] = std::max(min_inv_depth, std::min(max_inv_depth, inv_depth_vec[j]));
        }

        // ---------- 5.8 代价校验 + 阻尼调整 ----------
        // 计算更新后的总代价
        double new_cost = 0.0;
        for (const auto& edge : edges)
        {
            Eigen::Vector2d r;
            Eigen::Matrix<double,2,6> Jh, Jt;
            Eigen::Matrix<double,2,1> Jl;
            ComputeReprojectionResidualAndJacobians(
                pose_t[edge.host_idx], pose_q[edge.host_idx],
                pose_t[edge.target_idx], pose_q[edge.target_idx],
                inv_depth_vec[edge.mp_idx],
                edge.host_un, edge.host_vn,
                edge.target_u, edge.target_v,
                cam_fx, cam_fy, cam_cx, cam_cy,
                r, Jh, Jt, Jl
            );
            double rn = r.norm();
            double w = rn > edge.huber_delta ? edge.huber_delta / rn : 1.0;
            new_cost += (r * w).squaredNorm();
        }
        // 加上先验代价
        {
            Eigen::Matrix<double,6,1> perr = ComputePoseError(prior_t, prior_q, pose_t[0], pose_q[0]);
            new_cost += perr.transpose() * H_prior * perr;
        }

        // LM阻尼策略
        if (new_cost < last_cost || first_iter)
        {
            lambda_lm *= lambda_shrink;
            first_iter = false;
            if (fabs(last_cost - new_cost) < eps_converge * last_cost)
                break; // 收敛提前退出
        }
        else
        {
            // 更新无效，回滚状态并增大阻尼
            pose_t = last_pose_t;
            pose_q = last_pose_q;
            inv_depth_vec = last_inv_depth;
            lambda_lm *= lambda_boost;
        }
    }

    // ====================== 6. 优化结果写回地图 ======================
    // A. 写回关键帧位姿
    for (int i = 0; i < num_kfs; ++i)
    {
        Eigen::Isometry3d Twc_opt = Eigen::Isometry3d::Identity();
        Twc_opt.linear() = pose_q[i].toRotationMatrix();
        Twc_opt.translation() = pose_t[i];
        activeKFs[i]->SetPose(Twc_opt);
    }

    // B. 逆深度反投影写回地图点世界坐标
    for (int j = 0; j < num_mps; ++j)
    {
        double lambda_opt = inv_depth_vec[j];
        int host_idx = mp_host_kf_idx[j];
        int mp_id = mp_ref_vec[j]->GetFeatureId();

        cv::Point2f pt_host = activeKFs[host_idx]->mmObservations[mp_id];
        double host_un = (pt_host.x - cam_cx) / cam_fx;
        double host_vn = (pt_host.y - cam_cy) / cam_fy;

        Eigen::Vector3d p_h_scaled(host_un, host_vn, 1.0);
        Eigen::Vector3d pos_world_opt = pose_q[host_idx] * (p_h_scaled / lambda_opt) + pose_t[host_idx];
        mp_ref_vec[j]->SetWorldPos(pos_world_opt);
    }
}
