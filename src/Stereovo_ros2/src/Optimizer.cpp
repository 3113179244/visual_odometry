#include "Optimizer.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Parameters.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <unordered_map> // 新增：用于高速哈希查找
#include <vector>
#include <iostream>

// =========================================================================
// 1. 工业级健壮重投影误差代价函数
// =========================================================================
struct SREPROJECTION_ERROR {
    SREPROJECTION_ERROR(double observed_u, double observed_v, double fx, double fy, double cx, double cy)
        : u(observed_u), v(observed_v), fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const camera_pose, const T* const point_3d, T* residuals) const {
        T p_w[3] = { point_3d[0], point_3d[1], point_3d[2] };
        T p_c[3];

        // 【核心修复】参数块排布：[t_x, t_y, t_z, q_w, q_x, q_y, q_z]
        T t[3] = { camera_pose[0], camera_pose[1], camera_pose[2] };
        T q[4] = { camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6] }; // w, x, y, z

        T p_w_minus_t[3] = { p_w[0] - t[0], p_w[1] - t[1], p_w[2] - t[2] };
        
        // 【核心修复】四元数求逆（虚部取反）。因为 q 排布现在是 w,x,y,z，求逆只需虚部乘以 -1 
        T q_inv[4] = { q[0], -q[1], -q[2], -q[3] };
        ceres::QuaternionRotatePoint(q_inv, p_w_minus_t, p_c);

        // 软保护机制：如果深度非正，赋予像素惩罚项，消除 "infinite cost" 警告
        if (p_c[2] <= T(1e-4)) {
            residuals[0] = T(1111.0); 
            residuals[1] = T(1111.0);
            return true; 
        }

        // 标准相机投影公式
        T xp = p_c[0] / p_c[2];
        T yp = p_c[1] / p_c[2];
        T predicted_u = T(fx) * xp + T(cx);
        T predicted_v = T(fy) * yp + T(cy);

        // 计算重投影残差
        residuals[0] = T(u) - predicted_u;
        residuals[1] = T(v) - predicted_v;

        return true;
    }

    static ceres::CostFunction* Create(double u, double v, double fx, double fy, double cx, double cy) {
        return (new ceres::AutoDiffCostFunction<SREPROJECTION_ERROR, 2, 7, 3>(
            new SREPROJECTION_ERROR(u, v, fx, fy, cx, cy)));
    }

    double u, v;
    double fx, fy, cx, cy;
};

// =========================================================================
// 2. 局部 BA 主函数实现
// =========================================================================
void Optimizer::LocalBundleAdjustment(std::shared_ptr<Map> pMap, int windowSize) {
    if (!pMap) return;

    std::vector<std::shared_ptr<KeyFrame>> allKFs = pMap->GetAllKeyFrames();
    if (allKFs.size() < 2) return; 

    // 1. 筛选滑动窗口激活帧
    std::vector<std::shared_ptr<KeyFrame>> activeKFs;
    int startIdx = std::max(0, static_cast<int>(allKFs.size()) - windowSize);
    for (size_t i = startIdx; i < allKFs.size(); ++i) {
        activeKFs.push_back(allKFs[i]);
    }

    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0); 

    std::map<unsigned long, std::vector<double>> mapKFIdToParameterBlock;
    std::map<int, std::vector<double>> mapMPIdToParameterBlock;
    std::map<int, std::shared_ptr<MapPoint>> mapIdToMPRef;

    double cam_fx = Parameters::fx;
    double cam_fy = Parameters::fy;
    double cam_cx = Parameters::cx;
    double cam_cy = Parameters::cy;

    // 2. 装载激活帧位姿
    for (size_t i = 0; i < activeKFs.size(); ++i) {
        auto kf = activeKFs[i];
        Eigen::Isometry3d Twc = kf->GetPose(); 
        Eigen::Vector3d t = Twc.translation();
        Eigen::Quaterniond q(Twc.rotation());

        // 【核心修复】强制将内存排列调整为 Ceres 标准的 [t, q_w, q_x, q_y, q_z]
        std::vector<double> pose_block = { t.x(), t.y(), t.z(), q.w(), q.x(), q.y(), q.z() };
        mapKFIdToParameterBlock[kf->mId] = pose_block;
    }

    // 【核心性能优化】获取所有全局地图点，并在内存中建立局部哈希表
    // 彻底杜绝原先每帧进行嵌套 for 循环线性查找导致的指数级耗时卡死（从 O(N) 降低到 O(1)）
    std::vector<std::shared_ptr<MapPoint>> allMPs = pMap->GetAllMapPoints();
    std::unordered_map<int, std::shared_ptr<MapPoint>> mapIdToMP;
    for (const auto& mp : allMPs) {
        mapIdToMP[mp->GetFeatureId()] = mp;
    }
    
    // 3. 建立多帧多点 2D-3D 重投影误差残差约束
    for (auto kf : activeKFs) {
        unsigned long kf_id = kf->mId;
        for (const auto& obs : kf->mmObservations) {
            int mp_id = obs.first;          
            cv::Point2f pt2d = obs.second;  

            // 【核心性能优化】利用哈希表瞬间定位地图点
            auto it_mp = mapIdToMP.find(mp_id);
            if (it_mp == mapIdToMP.end()) continue;
            std::shared_ptr<MapPoint> targetMP = it_mp->second;

            if (mapMPIdToParameterBlock.find(mp_id) == mapMPIdToParameterBlock.end()) {
                Eigen::Vector3d pos = targetMP->GetWorldPos();
                mapMPIdToParameterBlock[mp_id] = { pos.x(), pos.y(), pos.z() };
                mapIdToMPRef[mp_id] = targetMP;
            }

            ceres::CostFunction* cost_function = SREPROJECTION_ERROR::Create(
                pt2d.x, pt2d.y, cam_fx, cam_fy, cam_cx, cam_cy);

            problem.AddResidualBlock(cost_function, loss_function, 
                                     mapKFIdToParameterBlock[kf_id].data(), 
                                     mapMPIdToParameterBlock[mp_id].data());
        }
    }

    // 4. 设置四元数流形空间约束并固定首帧
    for (size_t i = 0; i < activeKFs.size(); ++i) {
        auto kf = activeKFs[i];
        double* pose_data = mapKFIdToParameterBlock[kf->mId].data();
        
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
        // 【核心修复】新版 Ceres 下，将原先错误的 SubsetManifold(7, {0,1,2}) 
        // 修正为标准的 3维欧氏空间 + 4维四元数空间的 ProductManifold
        problem.SetManifold(pose_data, new ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::QuaternionManifold>(
            ceres::EuclideanManifold<3>(), ceres::QuaternionManifold()));
#else
        ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization();
        problem.SetParameterization(pose_data, new ceres::ProductParameterization(
            new ceres::IdentityParameterization(3), quaternion_parameterization));
#endif

        // 固定窗口内第一帧关键帧作为锚定基准
        if (i == 0) {
            problem.SetParameterBlockConstant(pose_data);
        }
    }

    // 5. 求解器高级参数配置
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR; 
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 8;                  
    options.minimizer_progress_to_stdout = false;     

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 6. 优化成功，将精调后的结果平滑写回系统数据库中
    if (summary.termination_type == ceres::CONVERGENCE || summary.termination_type == ceres::USER_SUCCESS) {
        for (auto kf : activeKFs) {
            auto& data = mapKFIdToParameterBlock[kf->mId];
            Eigen::Vector3d t_opt(data[0], data[1], data[2]);
            // 【核心修复】对应修改写回时的解析顺序，data[3] 是 q_w
            Eigen::Quaterniond q_opt(data[3], data[4], data[5], data[6]); // Eigen默认构造顺序为 (w, x, y, z)

            Eigen::Isometry3d Twc_opt = Eigen::Isometry3d::Identity();
            Twc_opt.linear() = q_opt.toRotationMatrix();
            Twc_opt.translation() = t_opt;
            
            kf->mTwc = Twc_opt; 
        }

        for (const auto& pair : mapMPIdToParameterBlock) {
            int mp_id = pair.first;
            const auto& data = pair.second;
            Eigen::Vector3d pos_opt(data[0], data[1], data[2]);
            mapIdToMPRef[mp_id]->SetWorldPos(pos_opt);
        }
    }
}