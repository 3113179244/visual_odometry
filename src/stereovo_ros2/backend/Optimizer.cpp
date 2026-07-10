#include "backend/Optimizer.h"
#include "core/KeyFrame.h"
#include "core/MapPoint.h"
#include "utils/Parameters.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h> // 引入 Ceres 头文件

#include <map>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
// ==================== 辅助数学函数 ====================

/**
 * @brief 构造三维向量的反对称矩阵
 */
static inline Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m << 0.0, -v.z(), v.y(),
        v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return m;
}

/**
 * @brief 计算重投影残差及其对位姿和逆深度的雅可比 (完全基于 Sophus 右扰动模型规范化)
 * 位姿变化定义为: T_target_w * T_w_host
 */
static inline bool ComputeReprojectionResidualAndJacobians(
    const Eigen::Vector3d &hostTranslation,
    const Eigen::Quaterniond &hostRotation,
    const Eigen::Vector3d &targetTranslation,
    const Eigen::Quaterniond &targetRotation,
    double invDepth,
    double hostUn, double hostVn,
    double targetU, double targetV,
    double fx, double fy, double cx, double cy,
    Eigen::Vector2d &residual,
    Eigen::Matrix<double, 2, 6> &jacobianHost,
    Eigen::Matrix<double, 2, 6> &jacobianTarget,
    Eigen::Matrix<double, 2, 1> &jacobianInvDepth)
{
    // 使用 Sophus 构造位姿
    Sophus::SE3d T_w_host(hostRotation, hostTranslation);
    Sophus::SE3d T_w_target(targetRotation, targetTranslation);
    Sophus::SE3d T_target_host = T_w_target.inverse() * T_w_host;

    Eigen::Vector3d hostRay(hostUn, hostVn, 1.0);
    double lambda = invDepth;
    Eigen::Vector3d pointHost = hostRay / lambda;

    // 投影到目标帧
    Eigen::Vector3d pointTarget = T_target_host * pointHost;
    double X = pointTarget.x();
    double Y = pointTarget.y();
    double Z = pointTarget.z();

    if (Z < 1e-4)
    {
        return false;
    }

    double invZ = 1.0 / Z;
    double invZ2 = invZ * invZ;

    // 像素对相机坐标系 3D 点的导数
    Eigen::Matrix<double, 2, 3> jacobianProj;
    jacobianProj << fx * invZ, 0.0, -fx * X * invZ2,
                    0.0, fy * invZ, -fy * Y * invZ2;

    // 1. 对 Target 帧位姿的导数 (基于右扰动: T_w_target * exp(delta^wedge))
    // d(pointTarget) / d(delta_target) = - [I, -pointTarget^wedge]
    Eigen::Matrix<double, 3, 6> dPointTarget_dXiTarget;
    dPointTarget_dXiTarget.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    dPointTarget_dXiTarget.block<3, 3>(0, 3) = Sophus::SO3d::hat(pointTarget);
    jacobianTarget = -jacobianProj * dPointTarget_dXiTarget;

    // 2. 对 Host 帧位姿的导数 (基于右扰动: T_w_host * exp(delta^wedge))
    // d(pointTarget) / d(delta_host) = R_target_host * [I, -pointHost^wedge]
    Eigen::Matrix<double, 3, 6> dPointTarget_dXiHost;
    dPointTarget_dXiHost.block<3, 3>(0, 0) = T_target_host.rotationMatrix();
    dPointTarget_dXiHost.block<3, 3>(0, 3) = -T_target_host.rotationMatrix() * Sophus::SO3d::hat(pointHost);
    jacobianHost = -jacobianProj * dPointTarget_dXiHost;

    // 3. 对逆深度的导数
    Eigen::Vector3d dPointHost_dLambda = -hostRay / (lambda * lambda);
    Eigen::Vector3d dPointTarget_dLambda = T_target_host.rotationMatrix() * dPointHost_dLambda;
    jacobianInvDepth = -jacobianProj * dPointTarget_dLambda;

    // 计算残差
    double predictedU = fx * X / Z + cx;
    double predictedV = fy * Y / Z + cy;
    residual(0) = targetU - predictedU;
    residual(1) = targetV - predictedV;

    return true;
}

// ==================== Ceres 参数块与局部参数化定义 ====================

/**
 * @brief Ceres 局部参数化：使用 Sophus 进行标准的右扰动位姿更新
 */
class PoseLocalParameterization : public ceres::LocalParameterization
{
public:
    virtual ~PoseLocalParameterization() {}
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        Eigen::Map<const Eigen::Vector3d> oldTranslation(x);
        Eigen::Map<const Eigen::Quaterniond> oldRotation(x + 3);

        // delta 包含 [平移增量(3维), 旋转李代数(3维)]
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> deltaPose(delta);

        Eigen::Map<Eigen::Vector3d> newTranslation(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> newRotation(x_plus_delta + 3);

        // 构造当前的 Sophus SE(3) 对象
        Sophus::SE3d T_old(oldRotation, oldTranslation);
        
        // 使用 Sophus::SE3d::exp 进行标准的右扰动（Right Multiplicative）更新： T_new = T_old * exp(delta)
        Sophus::SE3d T_new = T_old * Sophus::SE3d::exp(deltaPose);

        // 写回内存
        newTranslation = T_new.translation();
        newRotation = T_new.unit_quaternion();
        return true;
    }
    virtual bool ComputeJacobian(const double *x, double *jacobian) const
    {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }
    virtual int GlobalSize() const { return 7; } 
    virtual int LocalSize() const { return 6; }  
};

// ==================== Ceres 残差块定义 ====================

/**
 * @brief 重投影误差 Ceres 协同代价函数
 */
class ReprojectionCostFunction : public ceres::CostFunction
{
public:
    ReprojectionCostFunction(double hostUn, double hostVn, double targetU, double targetV,
                             double fx, double fy, double cx, double cy)
        : hostUn_(hostUn), hostVn_(hostVn), targetU_(targetU), targetV_(targetV),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy)
    {
        // 参数块 0: Host 帧位姿 (7维)
        mutable_parameter_block_sizes()->push_back(7);
        // 参数块 1: Target 帧位姿 (7维)
        mutable_parameter_block_sizes()->push_back(7);
        // 参数块 2: 逆深度 (1维)
        mutable_parameter_block_sizes()->push_back(1);
        // 残差维数: 2维 (u, v 像素误差)
        set_num_residuals(2);
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Vector3d> hostT(parameters[0]);
        Eigen::Map<const Eigen::Quaterniond> hostQ(parameters[0] + 3);

        Eigen::Map<const Eigen::Vector3d> targetT(parameters[1]);
        Eigen::Map<const Eigen::Quaterniond> targetQ(parameters[1] + 3);

        double invDepth = parameters[2][0];

        Eigen::Vector2d res;
        Eigen::Matrix<double, 2, 6> J_host;
        Eigen::Matrix<double, 2, 6> J_target;
        Eigen::Matrix<double, 2, 1> J_invDepth;

        bool valid = ComputeReprojectionResidualAndJacobians(
            hostT, hostQ, targetT, targetQ, invDepth,
            hostUn_, hostVn_, targetU_, targetV_,
            fx_, fy_, cx_, cy_, res, J_host, J_target, J_invDepth);

        if (!valid)
            return false;

        // 填入残差
        residuals[0] = res(0);
        residuals[1] = res(1);

        // 填入解析雅可比矩阵 (Ceres要求行优先储存 RowMajor)
        if (jacobians)
        {
            if (jacobians[0]) // 对 Host 位姿的导数 (2 x 7)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J0(jacobians[0]);
                J0.setZero();
                J0.block<2, 6>(0, 0) = J_host; // 映射前6维自由度
            }
            if (jacobians[1]) // 对 Target 位姿的导数 (2 x 7)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J1(jacobians[1]);
                J1.setZero();
                J1.block<2, 6>(0, 0) = J_target;
            }
            if (jacobians[2]) // 对 逆深度 的导数 (2 x 1)
            {
                Eigen::Map<Eigen::Matrix<double, 2, 1>> J2(jacobians[2]);
                J2 = J_invDepth;
            }
        }
        return true;
    }

private:
    double hostUn_, hostVn_, targetU_, targetV_;
    double fx_, fy_, cx_, cy_;
};

// ==================== 主优化函数 ====================

void Optimizer::LocalBundleAdjustment(std::shared_ptr<Map> map, int windowSize)
{
    if (!map)
        return;

    // ----- 1. 提取窗口内关键帧 -----
    std::vector<std::shared_ptr<KeyFrame>> allKeyFrames = map->GetAllKeyFrames();
    if (allKeyFrames.size() < 2)
        return;

    std::vector<std::shared_ptr<KeyFrame>> activeKeyFrames;
    int startIdx = std::max(0, static_cast<int>(allKeyFrames.size()) - windowSize);
    for (size_t i = startIdx; i < allKeyFrames.size(); ++i)
        activeKeyFrames.push_back(allKeyFrames[i]);

    int numKeyFrames = activeKeyFrames.size();

    // ----- 2. 初始化变量数组 (Ceres 优化变量通常用连续内存 double 数组) -----
    // 每个位姿 7维：[x, y, z, qx, qy, qz, qw]
    std::vector<std::vector<double>> poseBlocks(numKeyFrames, std::vector<double>(7));
    std::unordered_map<unsigned long, int> keyFrameIdToIdx;

    for (int i = 0; i < numKeyFrames; ++i)
    {
        Eigen::Isometry3d Twc = activeKeyFrames[i]->GetPose();
        Eigen::Vector3d t = Twc.translation();
        Eigen::Quaterniond q(Twc.rotation());

        poseBlocks[i][0] = t.x();
        poseBlocks[i][1] = t.y();
        poseBlocks[i][2] = t.z();
        poseBlocks[i][3] = q.x();
        poseBlocks[i][4] = q.y();
        poseBlocks[i][5] = q.z();
        poseBlocks[i][6] = q.w();

        keyFrameIdToIdx[activeKeyFrames[i]->mId] = i;
    }

    double camFx = Parameters::fx;
    double camFy = Parameters::fy;
    double camCx = Parameters::cx;
    double camCy = Parameters::cy;

    // ----- 3. 建立地图点索引和逆深度初始值 -----
    std::vector<std::shared_ptr<MapPoint>> allMapPoints = map->GetAllMapPoints();
    std::unordered_map<int, std::shared_ptr<MapPoint>> mapIdToMapPoint;
    for (const auto &mp : allMapPoints)
        mapIdToMapPoint[mp->GetFeatureId()] = mp;

    std::unordered_map<int, int> mapPointIdToIdx;
    std::vector<double> invDepthVector;
    std::vector<int> mapPointHostFrameIdx;
    std::vector<std::shared_ptr<MapPoint>> mapPointRefVec;

    for (int i = 0; i < numKeyFrames; ++i)
    {
        auto keyFrame = activeKeyFrames[i];
        for (const auto &obs : keyFrame->mmObservations)
        {
            int mapPointId = obs.first;
            auto itMp = mapIdToMapPoint.find(mapPointId);
            if (itMp == mapIdToMapPoint.end())
                continue;

            if (mapPointIdToIdx.find(mapPointId) == mapPointIdToIdx.end())
            {
                int mpIdx = invDepthVector.size();
                mapPointIdToIdx[mapPointId] = mpIdx;
                mapPointHostFrameIdx.push_back(i);
                mapPointRefVec.push_back(itMp->second);

                Eigen::Vector3d pointWorld = itMp->second->GetWorldPos();
                Eigen::Vector3d pointHost = keyFrame->GetPose().inverse() * pointWorld;
                double depth = pointHost.z() < 0.1 ? 1.0 : pointHost.z();
                invDepthVector.push_back(1.0 / depth);
            }
        }
    }

    // ----- 4. 构建 Ceres 问题并添加残差块 -----
    ceres::Problem problem;

    PoseLocalParameterization *poseParameterization = new PoseLocalParameterization();

    for (int i = 0; i < numKeyFrames; ++i)
    {
        problem.AddParameterBlock(poseBlocks[i].data(), 7, poseParameterization);
    }
    // 1. 固定第一帧相机位姿块
    problem.SetParameterBlockConstant(poseBlocks[0].data());

    // 建立所有观测误差边
    for (int targetIdx = 0; targetIdx < numKeyFrames; ++targetIdx)
    {
        auto keyFrame = activeKeyFrames[targetIdx];
        for (const auto &obs : keyFrame->mmObservations)
        {
            int mapPointId = obs.first;
            cv::Point2f ptTarget = obs.second;

            auto itMpIdx = mapPointIdToIdx.find(mapPointId);
            if (itMpIdx == mapPointIdToIdx.end())
                continue;

            int mpIdx = itMpIdx->second;
            int hostIdx = mapPointHostFrameIdx[mpIdx];

            if (hostIdx == targetIdx)
                continue;

            cv::Point2f ptHost = activeKeyFrames[hostIdx]->mmObservations[mapPointId];
            double hostUn = (ptHost.x - camCx) / camFx;
            double hostVn = (ptHost.y - camCy) / camFy;

            int obsCount = mapPointRefVec[mpIdx]->GetObservationCount();
            double huberDelta = (obsCount >= 5) ? 0.5 : ((obsCount >= 2) ? 1.0 : 1.5);

            ceres::CostFunction *costFunction = new ReprojectionCostFunction(
                hostUn, hostVn, ptTarget.x, ptTarget.y, camFx, camFy, camCx, camCy);

            ceres::LossFunction *lossFunction = new ceres::HuberLoss(huberDelta);

            problem.AddResidualBlock(costFunction, lossFunction,
                                     poseBlocks[hostIdx].data(),
                                     poseBlocks[targetIdx].data(),
                                     &invDepthVector[mpIdx]);

            problem.SetParameterLowerBound(&invDepthVector[mpIdx], 0, 0.001);
            problem.SetParameterUpperBound(&invDepthVector[mpIdx], 0, 10.0);
        }
    }

    // ----- 5. 配置 Ceres 求解器选项 -----
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    // 3. 优化迭代策略：给足迭代次数，同时松开不必要的严苛公差，兼顾速度与完全收敛
    options.max_num_iterations = 15;
    options.function_tolerance = 1e-4;
    options.parameter_tolerance = 1e-4;
    options.minimizer_progress_to_stdout = false;

    // ----- 6. 执行优化并打印 Ceres 报告 -----
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 打印 Ceres 优化报告
    // std::cout << "================== Ceres LBA Report ==================\n";
    // std::cout << summary.FullReport() << "\n";
    // std::cout << "======================================================\n";

    // ----- 7. 将优化结果写回关键帧和地图点 -----
    // ----- 7. 将优化结果写回关键帧和地图点 -----
    for (int i = 0; i < numKeyFrames; ++i)
    {
        Eigen::Vector3d t(poseBlocks[i][0], poseBlocks[i][1], poseBlocks[i][2]);
        // 修复四元数构造顺序，Eigen标量构造顺序严格为 (w, x, y, z)
        Eigen::Quaterniond q(poseBlocks[i][6], poseBlocks[i][3], poseBlocks[i][4], poseBlocks[i][5]); 
        q.normalize();
        Eigen::Isometry3d TwcOpt = Eigen::Isometry3d::Identity();
        TwcOpt.linear() = q.toRotationMatrix();
        TwcOpt.translation() = t;
        activeKeyFrames[i]->SetPose(TwcOpt);
    }

    int numMapPoints = invDepthVector.size();
    for (int j = 0; j < numMapPoints; ++j)
    {
        double lambdaOpt = invDepthVector[j];
        int hostIdx = mapPointHostFrameIdx[j];
        int mapPointId = mapPointRefVec[j]->GetFeatureId();

        cv::Point2f ptHost = activeKeyFrames[hostIdx]->mmObservations[mapPointId];
        double hostUn = (ptHost.x - camCx) / camFx;
        double hostVn = (ptHost.y - camCy) / camFy;

        Eigen::Vector3d hostRayScaled(hostUn, hostVn, 1.0);
        // 重新获取优化后的位姿用于更新3D世界点坐标
        Eigen::Vector3d tHost(poseBlocks[hostIdx][0], poseBlocks[hostIdx][1], poseBlocks[hostIdx][2]);
        Eigen::Quaterniond qHost(poseBlocks[hostIdx][6], poseBlocks[hostIdx][3], poseBlocks[hostIdx][4], poseBlocks[hostIdx][5]);
        qHost.normalize(); // 确保写回时四元数已归一化
        
        Eigen::Vector3d posWorldOpt = qHost * (hostRayScaled / lambdaOpt) + tHost;
        mapPointRefVec[j]->SetWorldPos(posWorldOpt);
    }
}