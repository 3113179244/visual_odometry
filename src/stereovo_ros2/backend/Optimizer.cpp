#include "backend/Optimizer.h"
#include "core/KeyFrame.h"
#include "core/MapPoint.h"
#include "utils/Parameters.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h> 
#include <map>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

// ==================== 辅助数学函数 ====================

/**
 * @brief 计算双目重投影残差及其对位姿和逆深度的雅可比 (基于 Sophus 右扰动模型规范化)
 */
static inline bool ComputeStereoReprojectionResidualAndJacobians(
    const Eigen::Vector3d &hostTranslation, const Eigen::Quaterniond &hostRotation,
    const Eigen::Vector3d &targetTranslation, const Eigen::Quaterniond &targetRotation,
    double invDepth,
    double hostUn, double hostVn,
    double targetLeftU, double targetLeftV,
    double targetRightU, double targetRightV, bool hasRight,
    double fx, double fy, double cx, double cy,
    const Sophus::SE3d &T_c1_c0, 
    Eigen::Matrix<double, 4, 1> &residual,
    Eigen::Matrix<double, 4, 6> &jacobianHost,
    Eigen::Matrix<double, 4, 6> &jacobianTarget,
    Eigen::Matrix<double, 4, 1> &jacobianInvDepth)
{
    residual.setZero();
    jacobianHost.setZero();
    jacobianTarget.setZero();
    jacobianInvDepth.setZero();

    Sophus::SE3d T_w_host(hostRotation, hostTranslation);
    Sophus::SE3d T_w_target(targetRotation, targetTranslation);
    Sophus::SE3d T_target_host = T_w_target.inverse() * T_w_host;

    Eigen::Vector3d hostRay(hostUn, hostVn, 1.0);
    Eigen::Vector3d pointHost = hostRay / invDepth;

    // 1. 投影到 Target 帧左目
    Eigen::Vector3d pointTarget_c0 = T_target_host * pointHost;
    double X0 = pointTarget_c0.x(), Y0 = pointTarget_c0.y(), Z0 = pointTarget_c0.z();
    if (Z0 < 1e-4) return false;

    double invZ0 = 1.0 / Z0;
    double invZ0_2 = invZ0 * invZ0;

    Eigen::Matrix<double, 2, 3> jacProj0;
    jacProj0 << fx * invZ0, 0.0, -fx * X0 * invZ0_2,
                0.0, fy * invZ0, -fy * Y0 * invZ0_2;

    Eigen::Matrix<double, 3, 6> dPt0_dXiTarget, dPt0_dXiHost;
    dPt0_dXiTarget.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    dPt0_dXiTarget.block<3, 3>(0, 3) = Sophus::SO3d::hat(pointTarget_c0);
    jacobianTarget.block<2, 6>(0, 0) = -jacProj0 * dPt0_dXiTarget;

    dPt0_dXiHost.block<3, 3>(0, 0) = T_target_host.rotationMatrix();
    dPt0_dXiHost.block<3, 3>(0, 3) = -T_target_host.rotationMatrix() * Sophus::SO3d::hat(pointHost);
    jacobianHost.block<2, 6>(0, 0) = -jacProj0 * dPt0_dXiHost;

    Eigen::Vector3d dPt0_dLambda = T_target_host.rotationMatrix() * (-hostRay / (invDepth * invDepth));
    jacobianInvDepth.block<2, 1>(0, 0) = -jacProj0 * dPt0_dLambda;

    residual(0) = targetLeftU - (fx * X0 / Z0 + cx);
    residual(1) = targetLeftV - (fy * Y0 / Z0 + cy);

    // 2. 投影到 Target 帧右目 (如果右目追踪有效)
    if (hasRight)
    {
        Eigen::Vector3d pointTarget_c1 = T_c1_c0 * pointTarget_c0;
        double X1 = pointTarget_c1.x(), Y1 = pointTarget_c1.y(), Z1 = pointTarget_c1.z();
        if (Z1 > 1e-4)
        {
            double invZ1 = 1.0 / Z1;
            double invZ1_2 = invZ1 * invZ1;

            Eigen::Matrix<double, 2, 3> jacProj1;
            jacProj1 << fx * invZ1, 0.0, -fx * X1 * invZ1_2,
                        0.0, fy * invZ1, -fy * Y1 * invZ1_2;

            Eigen::Matrix3d R_c1_c0 = T_c1_c0.rotationMatrix();
            jacobianTarget.block<2, 6>(2, 0) = -jacProj1 * R_c1_c0 * dPt0_dXiTarget;
            jacobianHost.block<2, 6>(2, 0) = -jacProj1 * R_c1_c0 * dPt0_dXiHost;
            jacobianInvDepth.block<2, 1>(2, 0) = -jacProj1 * R_c1_c0 * dPt0_dLambda;

            residual(2) = targetRightU - (fx * X1 / Z1 + cx);
            residual(3) = targetRightV - (fy * Y1 / Z1 + cy);
        }
    }

    return true;
}

// ==================== Ceres 参数块与局部参数化定义 ====================

class PoseLocalParameterization : public ceres::LocalParameterization
{
public:
    virtual ~PoseLocalParameterization() {}
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        Eigen::Map<const Eigen::Vector3d> oldTranslation(x);
        Eigen::Map<const Eigen::Quaterniond> oldRotation(x + 3);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> deltaPose(delta);

        Eigen::Map<Eigen::Vector3d> newTranslation(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> newRotation(x_plus_delta + 3);

        Sophus::SE3d T_old(oldRotation, oldTranslation);
        Sophus::SE3d T_new = T_old * Sophus::SE3d::exp(deltaPose);

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

// ==================== Ceres 双目残差块定义 ====================

class StereoReprojectionCostFunction : public ceres::CostFunction
{
public:
    StereoReprojectionCostFunction(double hostUn, double hostVn, 
                                   double targetLeftU, double targetLeftV,
                                   double targetRightU, double targetRightV, bool hasRight,
                                   double fx, double fy, double cx, double cy,
                                   const Sophus::SE3d &T_c1_c0)
        : hostUn_(hostUn), hostVn_(hostVn), 
          tLU_(targetLeftU), tLV_(targetLeftV), tRU_(targetRightU), tRV_(targetRightV),
          hasRight_(hasRight), fx_(fx), fy_(fy), cx_(cx), cy_(cy), T_c1_c0_(T_c1_c0)
    {
        mutable_parameter_block_sizes()->push_back(7); // Host Pose
        mutable_parameter_block_sizes()->push_back(7); // Target Pose
        mutable_parameter_block_sizes()->push_back(1); // InvDepth
        set_num_residuals(4); // 4维残差
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Vector3d> hostT(parameters[0]);
        Eigen::Map<const Eigen::Quaterniond> hostQ(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> targetT(parameters[1]);
        Eigen::Map<const Eigen::Quaterniond> targetQ(parameters[1] + 3);
        double invDepth = parameters[2][0];

        Eigen::Matrix<double, 4, 1> res;
        Eigen::Matrix<double, 4, 6> J_host;
        Eigen::Matrix<double, 4, 6> J_target;
        Eigen::Matrix<double, 4, 1> J_invDepth;

        bool valid = ComputeStereoReprojectionResidualAndJacobians(
            hostT, hostQ, targetT, targetQ, invDepth, hostUn_, hostVn_,
            tLU_, tLV_, tRU_, tRV_, hasRight_, fx_, fy_, cx_, cy_, T_c1_c0_,
            res, J_host, J_target, J_invDepth);

        if (!valid) return false;

        for(int i = 0; i < 4; ++i) residuals[i] = res(i);

        if (!hasRight_) {
            residuals[2] = 0.0;
            residuals[3] = 0.0;
        }

        if (jacobians)
        {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor>> J0(jacobians[0]);
                J0.setZero();
                J0.block<4, 6>(0, 0) = J_host;
                if(!hasRight_) J0.block<2, 6>(2, 0).setZero();
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor>> J1(jacobians[1]);
                J1.setZero();
                J1.block<4, 6>(0, 0) = J_target;
                if(!hasRight_) J1.block<2, 6>(2, 0).setZero();
            }
            if (jacobians[2]) {
                Eigen::Map<Eigen::Matrix<double, 4, 1>> J2(jacobians[2]);
                J2 = J_invDepth;
                if(!hasRight_) J2.block<2, 1>(2, 0).setZero();
            }
        }
        return true;
    }

private:
    double hostUn_, hostVn_;
    double tLU_, tLV_, tRU_, tRV_;
    bool hasRight_;
    double fx_, fy_, cx_, cy_;
    Sophus::SE3d T_c1_c0_;
};

// ==================== 主优化函数 ====================

void Optimizer::LocalBundleAdjustment(std::shared_ptr<Map> map, int windowSize)
{
    if (!map) return;

    std::vector<std::shared_ptr<KeyFrame>> allKeyFrames = map->GetAllKeyFrames();
    if (allKeyFrames.size() < 2) return;

    std::vector<std::shared_ptr<KeyFrame>> activeKeyFrames;
    int startIdx = std::max(0, static_cast<int>(allKeyFrames.size()) - windowSize);
    for (size_t i = startIdx; i < allKeyFrames.size(); ++i)
        activeKeyFrames.push_back(allKeyFrames[i]);

    int numKeyFrames = activeKeyFrames.size();

    std::vector<std::vector<double>> poseBlocks(numKeyFrames, std::vector<double>(7));
    std::unordered_map<unsigned long, int> keyFrameIdToIdx;

    for (int i = 0; i < numKeyFrames; ++i)
    {
        Eigen::Isometry3d Twc = activeKeyFrames[i]->GetPose();
        Eigen::Vector3d t = Twc.translation();
        Eigen::Quaterniond q(Twc.rotation());

        poseBlocks[i][0] = t.x(); poseBlocks[i][1] = t.y(); poseBlocks[i][2] = t.z();
        poseBlocks[i][3] = q.x(); poseBlocks[i][4] = q.y(); poseBlocks[i][5] = q.z(); poseBlocks[i][6] = q.w();

        keyFrameIdToIdx[activeKeyFrames[i]->mId] = i;
    }

    double camFx = Parameters::fx; double camFy = Parameters::fy;
    double camCx = Parameters::cx; double camCy = Parameters::cy;

    // 计算相对双目外参 T_c1_c0
    Sophus::SE3d T_b_c0(Eigen::Quaterniond(Parameters::body_T_cam0.block<3,3>(0,0)), Parameters::body_T_cam0.block<3,1>(0,3));
    Sophus::SE3d T_b_c1(Eigen::Quaterniond(Parameters::body_T_cam1.block<3,3>(0,0)), Parameters::body_T_cam1.block<3,1>(0,3));
    Sophus::SE3d T_c1_c0 = T_b_c1.inverse() * T_b_c0;

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
            if (itMp == mapIdToMapPoint.end()) continue;

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

    ceres::Problem problem;
    PoseLocalParameterization *poseParameterization = new PoseLocalParameterization();

    for (int i = 0; i < numKeyFrames; ++i)
        problem.AddParameterBlock(poseBlocks[i].data(), 7, poseParameterization);
        
    problem.SetParameterBlockConstant(poseBlocks[0].data());

    for (int targetIdx = 0; targetIdx < numKeyFrames; ++targetIdx)
    {
        auto keyFrame = activeKeyFrames[targetIdx];
        for (const auto &obs : keyFrame->mmObservations)
        {
            int mapPointId = obs.first;
            StereoObs obsTarget = obs.second;

            auto itMpIdx = mapPointIdToIdx.find(mapPointId);
            if (itMpIdx == mapPointIdToIdx.end()) continue;

            int mpIdx = itMpIdx->second;
            int hostIdx = mapPointHostFrameIdx[mpIdx];
            if (hostIdx == targetIdx) continue;

            StereoObs obsHost = activeKeyFrames[hostIdx]->mmObservations[mapPointId];
            double hostUn = (obsHost.ptLeft.x - camCx) / camFx;
            double hostVn = (obsHost.ptLeft.y - camCy) / camFy;

            int obsCount = mapPointRefVec[mpIdx]->GetObservationCount();
            double huberDelta = (obsCount >= 5) ? 0.5 : ((obsCount >= 2) ? 1.0 : 1.5);

            ceres::CostFunction *costFunction = new StereoReprojectionCostFunction(
                hostUn, hostVn, 
                obsTarget.ptLeft.x, obsTarget.ptLeft.y, 
                obsTarget.ptRight.x, obsTarget.ptRight.y, obsTarget.hasRight,
                camFx, camFy, camCx, camCy, T_c1_c0);

            ceres::LossFunction *lossFunction = new ceres::HuberLoss(huberDelta);

            problem.AddResidualBlock(costFunction, lossFunction,
                                     poseBlocks[hostIdx].data(),
                                     poseBlocks[targetIdx].data(),
                                     &invDepthVector[mpIdx]);

            problem.SetParameterLowerBound(&invDepthVector[mpIdx], 0, 0.001);
            problem.SetParameterUpperBound(&invDepthVector[mpIdx], 0, 10.0);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 15;
    options.function_tolerance = 1e-4;
    options.parameter_tolerance = 1e-4;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    for (int i = 0; i < numKeyFrames; ++i)
    {
        Eigen::Vector3d t(poseBlocks[i][0], poseBlocks[i][1], poseBlocks[i][2]);
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

        StereoObs obsHost = activeKeyFrames[hostIdx]->mmObservations[mapPointId];
        double hostUn = (obsHost.ptLeft.x - camCx) / camFx;
        double hostVn = (obsHost.ptLeft.y - camCy) / camFy;

        Eigen::Vector3d hostRayScaled(hostUn, hostVn, 1.0);
        Eigen::Vector3d tHost(poseBlocks[hostIdx][0], poseBlocks[hostIdx][1], poseBlocks[hostIdx][2]);
        Eigen::Quaterniond qHost(poseBlocks[hostIdx][6], poseBlocks[hostIdx][3], poseBlocks[hostIdx][4], poseBlocks[hostIdx][5]);
        qHost.normalize(); 
        
        Eigen::Vector3d posWorldOpt = qHost * (hostRayScaled / lambdaOpt) + tHost;
        mapPointRefVec[j]->SetWorldPos(posWorldOpt);
    }
}