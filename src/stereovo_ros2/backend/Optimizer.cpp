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
#include <cmath>

/**
 * @file Optimizer.cpp
 * @brief 实现局部捆集调整（Local Bundle Adjustment, LBA）
 * 
 * 优化目标：
 *   - 优化滑动窗口内关键帧的位姿（6DOF）
 *   - 优化窗口内地图点的逆深度（1DOF）
 *   - 采用舒尔补（Schur complement）加速，先消去逆深度变量
 *   - 使用 Levenberg-Marquardt 迭代求解
 * 
 * 数学细节：
 *   - 使用观测边缘（ObservationEdge）表示每个特征点在两个关键帧之间的重投影误差
 *   - 第一帧添加先验约束（边缘化先验信息）来固定尺度
 *   - 代价函数：Huber 鲁棒核函数
 */

// ==================== 辅助数学函数 ====================

/**
 * @brief 构造三维向量的反对称矩阵（用于李代数求导）
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
 * @brief 根据6维增量更新位姿（平移 + 旋转）
 * @param oldTranslation  原始平移
 * @param oldRotation     原始旋转（四元数）
 * @param delta           6维增量（前3平移，后3旋转（so(3)））
 * @param newTranslation  更新后的平移
 * @param newRotation     更新后的旋转
 */
static inline void UpdatePose(
    const Eigen::Vector3d &oldTranslation,
    const Eigen::Quaterniond &oldRotation,
    const Eigen::Matrix<double, 6, 1> &delta,
    Eigen::Vector3d &newTranslation,
    Eigen::Quaterniond &newRotation)
{
    // 平移更新：在全局坐标系下叠加
    newTranslation = oldTranslation + oldRotation.toRotationMatrix() * delta.head<3>();

    // 旋转更新：用轴角近似（小增量）
    Eigen::Vector3d deltaTheta = delta.tail<3>();
    Eigen::Quaterniond deltaQ(
        1.0,
        deltaTheta.x() / 2.0,
        deltaTheta.y() / 2.0,
        deltaTheta.z() / 2.0);
    deltaQ.normalize();

    newRotation = oldRotation * deltaQ;
    newRotation.normalize();
}

/**
 * @brief 计算当前位姿相对于先验位姿的误差（6维）
 */
static inline Eigen::Matrix<double, 6, 1> ComputePoseError(
    const Eigen::Vector3d &priorTranslation,
    const Eigen::Quaterniond &priorRotation,
    const Eigen::Vector3d &currentTranslation,
    const Eigen::Quaterniond &currentRotation)
{
    Eigen::Matrix<double, 6, 1> error;
    error.head<3>() = priorRotation.inverse().toRotationMatrix() * (currentTranslation - priorTranslation);

    Eigen::Quaterniond qRel = priorRotation.inverse() * currentRotation;
    Eigen::AngleAxisd aaRel(qRel);
    error.tail<3>() = aaRel.angle() * aaRel.axis();

    return error;
}

/**
 * @brief 计算重投影残差及其对位姿和逆深度的雅可比
 * @return bool 是否有效（Z > 0）
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
    // 1. 根据host帧的归一化坐标和逆深度重构3D点
    Eigen::Vector3d hostRay(hostUn, hostVn, 1.0);
    double lambda = invDepth;
    Eigen::Vector3d pointHost = hostRay / lambda;
    Eigen::Vector3d pointWorld = hostRotation * pointHost + hostTranslation;

    // 2. 将点转换到target帧坐标系
    Eigen::Vector3d pointTarget = targetRotation.inverse() * (pointWorld - targetTranslation);
    double X = pointTarget.x();
    double Y = pointTarget.y();
    double Z = pointTarget.z();

    if (Z < 1e-4)
    {
        residual.setConstant(1111.0);
        jacobianHost.setZero();
        jacobianTarget.setZero();
        jacobianInvDepth.setZero();
        return false;
    }

    double invZ = 1.0 / Z;
    double invZ2 = invZ * invZ;

    // 3. 投影雅可比 (2×3)
    Eigen::Matrix<double, 2, 3> jacobianProj;
    jacobianProj << fx * invZ, 0.0, -fx * X * invZ2,
        0.0, fy * invZ, -fy * Y * invZ2;

    // 4. 对target帧位姿的雅可比
    Eigen::Matrix<double, 3, 6> dPointTarget_dXiTarget;
    dPointTarget_dXiTarget.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    dPointTarget_dXiTarget.block<3, 3>(0, 3) = SkewSymmetric(pointTarget);
    jacobianTarget = -jacobianProj * dPointTarget_dXiTarget;

    // 5. 对host帧位姿的雅可比
    Eigen::Matrix<double, 3, 6> dPointWorld_dXiHost;
    dPointWorld_dXiHost.block<3, 3>(0, 0) = hostRotation.toRotationMatrix();
    dPointWorld_dXiHost.block<3, 3>(0, 3) = -hostRotation.toRotationMatrix() * SkewSymmetric(pointHost);
    Eigen::Matrix<double, 3, 6> dPointTarget_dXiHost = targetRotation.inverse().toRotationMatrix() * dPointWorld_dXiHost;
    jacobianHost = -jacobianProj * dPointTarget_dXiHost;

    // 6. 对逆深度的雅可比
    Eigen::Vector3d dPointHost_dLambda = -hostRay / (lambda * lambda);
    Eigen::Vector3d dPointTarget_dLambda = targetRotation.inverse().toRotationMatrix() * hostRotation.toRotationMatrix() * dPointHost_dLambda;
    jacobianInvDepth = -jacobianProj * dPointTarget_dLambda;

    // 7. 计算残差（像素误差）
    double predictedU = fx * X / Z + cx;
    double predictedV = fy * Y / Z + cy;
    residual(0) = targetU - predictedU;
    residual(1) = targetV - predictedV;

    return true;
}

// ==================== 观测边数据结构 ====================

/**
 * @brief 一条重投影观测边，连接 host帧、target帧 和 一个地图点
 */
struct ObservationEdge
{
    int hostFrameIdx;
    int targetFrameIdx;
    int mapPointIdx;
    double hostUn;    // host帧归一化坐标 (去畸变)
    double hostVn;
    double targetU;   // target帧像素坐标
    double targetV;
    double huberDelta; // Huber 阈值（根据观测次数动态调整）
};

// ==================== 主优化函数 ====================

void Optimizer::LocalBundleAdjustment(std::shared_ptr<Map> map, int windowSize)
{
    /**
     * @brief 对最近的 windowSize 个关键帧进行局部BA
     * 
     * 步骤：
     *  1. 提取滑动窗口内的关键帧和地图点
     *  2. 建立观测边（所有窗口内帧的共视关系）
     *  3. 构建先验（固定第一个关键帧的位姿，加入边缘化信息）
     *  4. Levenberg-Marquardt 迭代：
     *     - 计算残差和雅可比
     *     - 构造 Hessian（含舒尔补）
     *     - 求解增量，更新状态
     *     - 若代价下降则接受，否则回退并增大阻尼
     *  5. 将优化结果写回关键帧和地图点
     */
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
    int poseDim = numKeyFrames * 6;

    // ----- 2. 初始化变量 -----
    std::vector<Eigen::Vector3d> poseTranslation(numKeyFrames);
    std::vector<Eigen::Quaterniond> poseRotation(numKeyFrames);
    std::unordered_map<unsigned long, int> keyFrameIdToIdx;

    for (int i = 0; i < numKeyFrames; ++i)
    {
        Eigen::Isometry3d Twc = activeKeyFrames[i]->GetPose();
        poseTranslation[i] = Twc.translation();
        poseRotation[i] = Eigen::Quaterniond(Twc.rotation());
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
    std::vector<int> mapPointHostFrameIdx;          // 每个地图点首次被观测到的关键帧索引
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

                // 初值：从host帧反算深度
                Eigen::Vector3d pointWorld = itMp->second->GetWorldPos();
                Eigen::Vector3d pointHost = keyFrame->GetPose().inverse() * pointWorld;
                double depth = pointHost.z() < 0.1 ? 1.0 : pointHost.z();
                invDepthVector.push_back(1.0 / depth);
            }
        }
    }

    int numMapPoints = invDepthVector.size();

    // ----- 4. 构建所有观测边 -----
    std::vector<ObservationEdge> edges;
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

            if (hostIdx == targetIdx)  // 跳过自身观测
                continue;

            // 获取host帧中的观测像素
            cv::Point2f ptHost = activeKeyFrames[hostIdx]->mmObservations[mapPointId];
            double hostUn = (ptHost.x - camCx) / camFx;
            double hostVn = (ptHost.y - camCy) / camFy;

            // 根据地图点观测次数动态调整Huber阈值
            int obsCount = mapPointRefVec[mpIdx]->GetObservationCount();
            double huberDelta = (obsCount >= 5) ? 1.0 : ((obsCount >= 2) ? 1.5 : 3.0);

            edges.push_back({hostIdx, targetIdx, mpIdx,
                             hostUn, hostVn,
                             ptTarget.x, ptTarget.y,
                             huberDelta});
        }
    }

    // ----- 5. 构造先验（固定第一个关键帧，边缘化先验信息） -----
    Eigen::Matrix<double, 6, 6> H_prior = Eigen::Matrix<double, 6, 6>::Identity() * 1.0;
    Eigen::Vector3d priorTranslation = poseTranslation[0];
    Eigen::Quaterniond priorRotation = poseRotation[0];

    {
        // 利用第一个关键帧与其观测的地图点构造先验信息（简化版）
        Eigen::Matrix<double, 6, 6> H_mm = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 3> H_mr = Eigen::Matrix<double, 6, 3>::Zero();
        Eigen::Matrix<double, 3, 3> H_rr_sum = Eigen::Matrix<double, 3, 3>::Zero();

        double pixelNoiseSigma = 1.5;
        double omegaPixel = 1.0 / (pixelNoiseSigma * pixelNoiseSigma);
        int effectiveObsCount = 0;

        Eigen::Isometry3d TwcPrior = activeKeyFrames[0]->GetPose();
        Eigen::Matrix3d R_prior = TwcPrior.rotation();

        for (const auto &obs : activeKeyFrames[0]->mmObservations)
        {
            int mapPointId = obs.first;
            auto itMp = mapIdToMapPoint.find(mapPointId);
            if (itMp == mapIdToMapPoint.end())
                continue;

            Eigen::Vector3d pointWorld = itMp->second->GetWorldPos();
            Eigen::Vector3d pointCamera = TwcPrior.inverse() * pointWorld;
            double X = pointCamera.x();
            double Y = pointCamera.y();
            double Z = pointCamera.z();

            if (Z < 0.2)
                continue;

            effectiveObsCount++;

            Eigen::Matrix<double, 2, 3> jacobianProj;
            double invZ = 1.0 / Z;
            double invZ2 = invZ * invZ;
            jacobianProj << camFx * invZ, 0.0, -camFx * X * invZ2,
                0.0, camFy * invZ, -camFy * Y * invZ2;

            Eigen::Matrix<double, 3, 6> jacobianPose;
            jacobianPose.block<3, 3>(0, 0) = -R_prior.transpose();
            jacobianPose.block<3, 3>(0, 3) = SkewSymmetric(pointCamera);
            Eigen::Matrix<double, 2, 6> J_pose = jacobianProj * jacobianPose;

            Eigen::Matrix<double, 2, 3> J_point = jacobianProj * R_prior.transpose();

            H_mm += J_pose.transpose() * omegaPixel * J_pose;
            H_mr += J_pose.transpose() * omegaPixel * J_point;
            H_rr_sum += J_point.transpose() * omegaPixel * J_point;
        }

        if (effectiveObsCount > 4)
        {
            Eigen::Matrix3d H_rr_inv = H_rr_sum.inverse();
            H_prior += H_mm - H_mr * H_rr_inv * H_mr.transpose();
        }
        else
        {
            // 观测不足时加大先验权重
            H_prior.block<3, 3>(0, 0) *= 50.0;
            H_prior.block<3, 3>(3, 3) *= 100.0;
        }
    }

    // ----- 6. Levenberg-Marquardt 迭代 -----
    double lambdaLm = 1e-3;
    const double lambdaBoost = 10.0;
    const double lambdaShrink = 0.1;
    const int maxIterations = 10;
    const double epsConverge = 1e-6;

    std::vector<Eigen::Vector3d> lastPoseT = poseTranslation;
    std::vector<Eigen::Quaterniond> lastPoseQ = poseRotation;
    std::vector<double> lastInvDepth = invDepthVector;
    double lastCost = 0.0;

    const double minInvDepth = 0.001;
    const double maxInvDepth = 10.0;

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // ----- 6.1 构建线性系统（Hessian 和 梯度） -----
        Eigen::MatrixXd H_xx = Eigen::MatrixXd::Zero(poseDim, poseDim);
        Eigen::VectorXd g_x = Eigen::VectorXd::Zero(poseDim);

        std::vector<double> H_ll(numMapPoints, 0.0);
        std::vector<Eigen::VectorXd> H_xl(numMapPoints, Eigen::VectorXd::Zero(poseDim));
        std::vector<double> g_l(numMapPoints, 0.0);

        double totalCost = 0.0;

        for (const auto &edge : edges)
        {
            int hostStart = edge.hostFrameIdx * 6;
            int targetStart = edge.targetFrameIdx * 6;
            int mpIdx = edge.mapPointIdx;

            Eigen::Vector2d residual;
            Eigen::Matrix<double, 2, 6> J_host, J_target;
            Eigen::Matrix<double, 2, 1> J_lambda;

            bool valid = ComputeReprojectionResidualAndJacobians(
                poseTranslation[edge.hostFrameIdx], poseRotation[edge.hostFrameIdx],
                poseTranslation[edge.targetFrameIdx], poseRotation[edge.targetFrameIdx],
                invDepthVector[mpIdx],
                edge.hostUn, edge.hostVn,
                edge.targetU, edge.targetV,
                camFx, camFy, camCx, camCy,
                residual, J_host, J_target, J_lambda);

            if (!valid)
                continue;

            // Huber 加权
            double resNorm = residual.norm();
            double huberW = 1.0;
            if (resNorm > edge.huberDelta)
                huberW = edge.huberDelta / resNorm;
            residual *= huberW;
            J_host *= huberW;
            J_target *= huberW;
            J_lambda *= huberW;

            totalCost += residual.squaredNorm();

            // 填充Hessian块（姿势部分）
            H_xx.block<6, 6>(hostStart, hostStart) += J_host.transpose() * J_host;
            H_xx.block<6, 6>(hostStart, targetStart) += J_host.transpose() * J_target;
            H_xx.block<6, 6>(targetStart, hostStart) += J_target.transpose() * J_host;
            H_xx.block<6, 6>(targetStart, targetStart) += J_target.transpose() * J_target;

            g_x.segment<6>(hostStart) += J_host.transpose() * residual;
            g_x.segment<6>(targetStart) += J_target.transpose() * residual;

            // 逆深度部分（标量）
            H_ll[mpIdx] += (J_lambda.transpose() * J_lambda)(0, 0);
            H_xl[mpIdx].segment<6>(hostStart) += J_host.transpose() * J_lambda;
            H_xl[mpIdx].segment<6>(targetStart) += J_target.transpose() * J_lambda;
            g_l[mpIdx] += (J_lambda.transpose() * residual)(0, 0);
        }

        // 添加先验项（第一个关键帧）
        {
            int firstStart = 0;
            Eigen::Matrix<double, 6, 1> priorError = ComputePoseError(
                priorTranslation, priorRotation,
                poseTranslation[0], poseRotation[0]);
            H_xx.block<6, 6>(firstStart, firstStart) += H_prior;
            g_x.segment<6>(firstStart) += H_prior * priorError;
            totalCost += priorError.transpose() * H_prior * priorError;
        }

        // ----- 6.2 舒尔补消去逆深度变量 -----
        Eigen::MatrixXd H_schur = H_xx;
        Eigen::VectorXd g_schur = g_x;
        for (int j = 0; j < numMapPoints; ++j)
        {
            if (H_ll[j] < 1e-12)
                continue;
            H_schur -= H_xl[j] * H_xl[j].transpose() / H_ll[j];
            g_schur -= H_xl[j] * g_l[j] / H_ll[j];
        }

        // ----- 6.3 求解增量（带阻尼） -----
        Eigen::MatrixXd H_reg = H_schur;
        for (int i = 0; i < poseDim; ++i)
            H_reg(i, i) += lambdaLm;

        Eigen::LLT<Eigen::MatrixXd> llt(H_reg);
        if (llt.info() != Eigen::Success)
        {
            DEBUG_WARN("Hessian matrix is singular at iter " << iter << ", expanding lambda.");
            lambdaLm *= lambdaBoost;
            continue;
        }

        Eigen::VectorXd delta_x = llt.solve(-g_schur);

        // 求解逆深度增量
        Eigen::VectorXd delta_l(numMapPoints);
        for (int j = 0; j < numMapPoints; ++j)
        {
            if (H_ll[j] < 1e-12)
            {
                delta_l(j) = 0.0;
                continue;
            }
            double rhs = -g_l[j] - H_xl[j].dot(delta_x);
            delta_l(j) = rhs / H_ll[j];
        }

        // 保存旧状态用于回退
        lastPoseT = poseTranslation;
        lastPoseQ = poseRotation;
        lastInvDepth = invDepthVector;
        lastCost = totalCost;

        // ----- 6.4 更新状态 -----
        for (int i = 0; i < numKeyFrames; ++i)
        {
            Eigen::Matrix<double, 6, 1> delta = delta_x.segment<6>(i * 6);
            UpdatePose(poseTranslation[i], poseRotation[i], delta,
                       poseTranslation[i], poseRotation[i]);
        }

        for (int j = 0; j < numMapPoints; ++j)
        {
            invDepthVector[j] += delta_l(j);
            invDepthVector[j] = std::max(minInvDepth, std::min(maxInvDepth, invDepthVector[j]));
        }

        // ----- 6.5 计算新代价并决定接受/拒绝 -----
        double newCost = 0.0;
        for (const auto &edge : edges)
        {
            Eigen::Vector2d r;
            Eigen::Matrix<double, 2, 6> Jh, Jt;
            Eigen::Matrix<double, 2, 1> Jl;

            ComputeReprojectionResidualAndJacobians(
                poseTranslation[edge.hostFrameIdx], poseRotation[edge.hostFrameIdx],
                poseTranslation[edge.targetFrameIdx], poseRotation[edge.targetFrameIdx],
                invDepthVector[edge.mapPointIdx],
                edge.hostUn, edge.hostVn,
                edge.targetU, edge.targetV,
                camFx, camFy, camCx, camCy,
                r, Jh, Jt, Jl);

            double rn = r.norm();
            double w = rn > edge.huberDelta ? edge.huberDelta / rn : 1.0;
            newCost += (r * w).squaredNorm();
        }

        // 先验代价
        {
            Eigen::Matrix<double, 6, 1> perr = ComputePoseError(
                priorTranslation, priorRotation,
                poseTranslation[0], poseRotation[0]);
            newCost += perr.transpose() * H_prior * perr;
        }

        DEBUG_INFO("LBA Iter: " << iter
                                << " | Prev Cost: " << lastCost
                                << " | New Cost: " << newCost
                                << " | Lambda: " << lambdaLm);

        if (newCost < lastCost)
        {
            lambdaLm *= lambdaShrink;
            if (fabs(lastCost - newCost) < epsConverge * lastCost)
            {
                DEBUG_INFO("LBA converged early at iter " << iter);
                break;
            }
        }
        else
        {
            // 拒绝更新，回退状态，增大阻尼
            DEBUG_WARN("Cost increased! Update rejected. Resetting and scaling lambda up.");
            poseTranslation = lastPoseT;
            poseRotation = lastPoseQ;
            invDepthVector = lastInvDepth;
            lambdaLm *= lambdaBoost;
        }
    }

    // ----- 7. 将优化结果写回关键帧和地图点 -----
    for (int i = 0; i < numKeyFrames; ++i)
    {
        Eigen::Isometry3d TwcOpt = Eigen::Isometry3d::Identity();
        TwcOpt.linear() = poseRotation[i].toRotationMatrix();
        TwcOpt.translation() = poseTranslation[i];
        activeKeyFrames[i]->SetPose(TwcOpt);
    }

    for (int j = 0; j < numMapPoints; ++j)
    {
        double lambdaOpt = invDepthVector[j];
        int hostIdx = mapPointHostFrameIdx[j];
        int mapPointId = mapPointRefVec[j]->GetFeatureId();

        cv::Point2f ptHost = activeKeyFrames[hostIdx]->mmObservations[mapPointId];
        double hostUn = (ptHost.x - camCx) / camFx;
        double hostVn = (ptHost.y - camCy) / camFy;

        Eigen::Vector3d hostRayScaled(hostUn, hostVn, 1.0);
        Eigen::Vector3d posWorldOpt = poseRotation[hostIdx] * (hostRayScaled / lambdaOpt) + poseTranslation[hostIdx];

        mapPointRefVec[j]->SetWorldPos(posWorldOpt);
    }
}