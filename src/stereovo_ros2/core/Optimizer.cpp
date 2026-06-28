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
// 内部工具函数（仅当前文件可见）
// =========================================================================

/**
 * @brief 三维向量的反对称矩阵（叉乘矩阵）
 * @param v 三维向量
 * @return 3x3反对称矩阵
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
 * @brief 位姿右扰动更新（李代数右乘模型，保证四元数流形约束）
 * @param oldTranslation 旧平移
 * @param oldRotation 旧旋转（四元数）
 * @param delta 李代数增量（前3维平移，后3维旋转）
 * @param newTranslation 输出：新平移
 * @param newRotation 输出：新旋转（四元数）
 */
static inline void UpdatePose(
    const Eigen::Vector3d &oldTranslation,
    const Eigen::Quaterniond &oldRotation,
    const Eigen::Matrix<double, 6, 1> &delta,
    Eigen::Vector3d &newTranslation,
    Eigen::Quaterniond &newRotation)
{
    // 平移增量：从相机系转换到世界系
    newTranslation = oldTranslation + oldRotation.toRotationMatrix() * delta.head<3>();

    // 旋转增量：小角度四元数近似 + 右乘更新
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
 * @brief 计算两个位姿之间的右扰动李代数误差
 * @param priorTranslation 先验位姿平移
 * @param priorRotation 先验位姿旋转
 * @param currentTranslation 当前位姿平移
 * @param currentRotation 当前位姿旋转
 * @return 6维误差向量（前3维平移，后3维旋转）
 */
static inline Eigen::Matrix<double, 6, 1> ComputePoseError(
    const Eigen::Vector3d &priorTranslation,
    const Eigen::Quaterniond &priorRotation,
    const Eigen::Vector3d &currentTranslation,
    const Eigen::Quaterniond &currentRotation)
{
    Eigen::Matrix<double, 6, 1> error;

    // 相对平移：转换到先验位姿的局部坐标系
    error.head<3>() = priorRotation.inverse().toRotationMatrix()
                      * (currentTranslation - priorTranslation);

    // 相对旋转：四元数差转旋转向量
    Eigen::Quaterniond qRel = priorRotation.inverse() * currentRotation;
    Eigen::AngleAxisd aaRel(qRel);
    error.tail<3>() = aaRel.angle() * aaRel.axis();

    return error;
}

/**
 * @brief 计算逆深度重投影残差 + 解析雅可比
 * 
 * @param hostTranslation 主导帧平移
 * @param hostRotation 主导帧旋转
 * @param targetTranslation 目标帧平移
 * @param targetRotation 目标帧旋转
 * @param invDepth 逆深度
 * @param hostUn 主导帧归一化u坐标
 * @param hostVn 主导帧归一化v坐标
 * @param targetU 目标帧像素u
 * @param targetV 目标帧像素v
 * @param fx, fy, cx, cy 相机内参
 * @param residual 输出：2维残差
 * @param jacobianHost 输出：残差对主导帧位姿的雅可比 (2x6)
 * @param jacobianTarget 输出：残差对目标帧位姿的雅可比 (2x6)
 * @param jacobianInvDepth 输出：残差对逆深度的雅可比 (2x1)
 * @return true 计算成功，false 深度异常
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
    // 主导帧归一化射线方向
    Eigen::Vector3d hostRay(hostUn, hostVn, 1.0);
    double lambda = invDepth;

    // ========== 步骤1：计算世界系3D点 ==========
    Eigen::Vector3d pointHost = hostRay / lambda;
    Eigen::Vector3d pointWorld = hostRotation * pointHost + hostTranslation;

    // ========== 步骤2：变换到目标相机系 ==========
    Eigen::Vector3d pointTarget = targetRotation.inverse() * (pointWorld - targetTranslation);
    double X = pointTarget.x();
    double Y = pointTarget.y();
    double Z = pointTarget.z();

    // 卫语句：深度异常（太近或负深度）
    if (Z < 1e-4)
    {
        residual.setConstant(1111.0);
        jacobianHost.setZero();
        jacobianTarget.setZero();
        jacobianInvDepth.setZero();
        return false;
    }

    // ========== 步骤3：投影雅可比（像素坐标对相机系3D点的导数） ==========
    double invZ = 1.0 / Z;
    double invZ2 = invZ * invZ;

    Eigen::Matrix<double, 2, 3> jacobianProj;
    jacobianProj << fx * invZ, 0.0, -fx * X * invZ2,
        0.0, fy * invZ, -fy * Y * invZ2;

    // ========== 步骤4：对目标帧位姿的雅可比（右扰动模型） ==========
    Eigen::Matrix<double, 3, 6> dPointTarget_dXiTarget;
    dPointTarget_dXiTarget.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    dPointTarget_dXiTarget.block<3, 3>(0, 3) = SkewSymmetric(pointTarget);

    jacobianTarget = -jacobianProj * dPointTarget_dXiTarget;

    // ========== 步骤5：对主导帧位姿的雅可比（右扰动模型） ==========
    Eigen::Matrix<double, 3, 6> dPointWorld_dXiHost;
    dPointWorld_dXiHost.block<3, 3>(0, 0) = hostRotation.toRotationMatrix();
    dPointWorld_dXiHost.block<3, 3>(0, 3) = -hostRotation.toRotationMatrix() * SkewSymmetric(pointHost);

    Eigen::Matrix<double, 3, 6> dPointTarget_dXiHost
        = targetRotation.inverse().toRotationMatrix() * dPointWorld_dXiHost;

    jacobianHost = -jacobianProj * dPointTarget_dXiHost;

    // ========== 步骤6：对逆深度的雅可比 ==========
    Eigen::Vector3d dPointHost_dLambda = -hostRay / (lambda * lambda);
    Eigen::Vector3d dPointTarget_dLambda
        = targetRotation.inverse().toRotationMatrix() * hostRotation.toRotationMatrix() * dPointHost_dLambda;

    jacobianInvDepth = -jacobianProj * dPointTarget_dLambda;

    // ========== 步骤7：计算像素残差 ==========
    double predictedU = fx * X / Z + cx;
    double predictedV = fy * Y / Z + cy;

    residual(0) = targetU - predictedU;
    residual(1) = targetV - predictedV;

    return true;
}

// =========================================================================
// 观测边结构体
// =========================================================================

/**
 * @brief 观测约束边
 * 
 * 表示一个地图点在两个关键帧之间的观测约束，
 * 包含主导帧、目标帧、地图点索引和观测像素坐标。
 */
struct ObservationEdge
{
    int hostFrameIdx;      ///< 主导帧索引（首次观测到该点的帧）
    int targetFrameIdx;    ///< 目标观测帧索引
    int mapPointIdx;       ///< 地图点索引
    double hostUn;         ///< 主导帧归一化u坐标
    double hostVn;         ///< 主导帧归一化v坐标
    double targetU;        ///< 目标帧像素u
    double targetV;        ///< 目标帧像素v
    double huberDelta;     ///< Huber核阈值
};

// =========================================================================
// 主函数：局部滑动窗口逆深度 BA + 舒尔补边缘化
// =========================================================================

void Optimizer::LocalBundleAdjustment(std::shared_ptr<Map> map, int windowSize)
{
    // 卫语句：地图为空
    if (!map)
        return;

    std::vector<std::shared_ptr<KeyFrame>> allKeyFrames = map->GetAllKeyFrames();

    // 卫语句：关键帧数量不足
    if (allKeyFrames.size() < 2)
        return;

    // ====================== 步骤1：筛选滑动窗口激活帧 ======================
    std::vector<std::shared_ptr<KeyFrame>> activeKeyFrames;
    int startIdx = std::max(0, static_cast<int>(allKeyFrames.size()) - windowSize);
    for (size_t i = startIdx; i < allKeyFrames.size(); ++i)
        activeKeyFrames.push_back(allKeyFrames[i]);

    int numKeyFrames = activeKeyFrames.size();
    int poseDim = numKeyFrames * 6; // 位姿总维度（每个6自由度李代数）

    // 位姿状态存储 + ID到索引映射
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

    // 相机内参
    double camFx = Parameters::fx;
    double camFy = Parameters::fy;
    double camCx = Parameters::cx;
    double camCy = Parameters::cy;

    // ====================== 步骤2：地图点预处理 + 主导帧选拔 ======================
    std::vector<std::shared_ptr<MapPoint>> allMapPoints = map->GetAllMapPoints();
    std::unordered_map<int, std::shared_ptr<MapPoint>> mapIdToMapPoint;

    for (const auto &mp : allMapPoints)
        mapIdToMapPoint[mp->GetFeatureId()] = mp;

    std::unordered_map<int, int> mapPointIdToIdx;       ///< 地图点ID → 索引
    std::vector<double> invDepthVector;                  ///< 逆深度参数数组
    std::vector<int> mapPointHostFrameIdx;               ///< 每个地图点的主导帧索引
    std::vector<std::shared_ptr<MapPoint>> mapPointRefVec; ///< 地图点指针数组

    for (int i = 0; i < numKeyFrames; ++i)
    {
        auto keyFrame = activeKeyFrames[i];
        for (const auto &obs : keyFrame->mObservations)
        {
            int mapPointId = obs.first;
            auto itMp = mapIdToMapPoint.find(mapPointId);

            // 卫语句：地图中找不到该点，跳过
            if (itMp == mapIdToMapPoint.end())
                continue;

            // 首次出现的地图点：设置主导帧 + 初始化逆深度
            if (mapPointIdToIdx.find(mapPointId) == mapPointIdToIdx.end())
            {
                int mpIdx = invDepthVector.size();
                mapPointIdToIdx[mapPointId] = mpIdx;
                mapPointHostFrameIdx.push_back(i);
                mapPointRefVec.push_back(itMp->second);

                // 计算初始逆深度
                Eigen::Vector3d pointWorld = itMp->second->GetWorldPos();
                Eigen::Vector3d pointHost = keyFrame->GetPose().inverse() * pointWorld;
                double depth = pointHost.z() < 0.1 ? 1.0 : pointHost.z();
                invDepthVector.push_back(1.0 / depth);
            }
        }
    }

    int numMapPoints = invDepthVector.size();

    // ====================== 步骤3：预构建所有观测约束边 ======================
    std::vector<ObservationEdge> edges;

    for (int targetIdx = 0; targetIdx < numKeyFrames; ++targetIdx)
    {
        auto keyFrame = activeKeyFrames[targetIdx];
        for (const auto &obs : keyFrame->mObservations)
        {
            int mapPointId = obs.first;
            cv::Point2f ptTarget = obs.second;

            auto itMpIdx = mapPointIdToIdx.find(mapPointId);
            // 卫语句：该地图点不在优化变量中，跳过
            if (itMpIdx == mapPointIdToIdx.end())
                continue;

            int mpIdx = itMpIdx->second;
            int hostIdx = mapPointHostFrameIdx[mpIdx];

            // 卫语句：主导帧自身无残差
            if (hostIdx == targetIdx)
                continue;

            // 主导帧归一化坐标
            cv::Point2f ptHost = activeKeyFrames[hostIdx]->mObservations[mapPointId];
            double hostUn = (ptHost.x - camCx) / camFx;
            double hostVn = (ptHost.y - camCy) / camFy;

            // 自适应Huber阈值（观测越多，阈值越严格）
            int obsCount = mapPointRefVec[mpIdx]->GetObservationCount();
            double huberDelta = (obsCount >= 5) ? 1.0 : ((obsCount >= 2) ? 1.5 : 3.0);

            edges.push_back({hostIdx, targetIdx, mpIdx,
                             hostUn, hostVn,
                             ptTarget.x, ptTarget.y,
                             huberDelta});
        }
    }

    // ====================== 步骤4：计算第一帧边缘化先验信息矩阵 ======================
    Eigen::Matrix<double, 6, 6> H_prior = Eigen::Matrix<double, 6, 6>::Identity() * 1.0;
    Eigen::Vector3d priorTranslation = poseTranslation[0];
    Eigen::Quaterniond priorRotation = poseRotation[0];

    {
        Eigen::Matrix<double, 6, 6> H_mm = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 3> H_mr = Eigen::Matrix<double, 6, 3>::Zero();
        Eigen::Matrix<double, 3, 3> H_rr_sum = Eigen::Matrix<double, 3, 3>::Zero();

        double pixelNoiseSigma = 1.5;
        double omegaPixel = 1.0 / (pixelNoiseSigma * pixelNoiseSigma);
        int effectiveObsCount = 0;

        Eigen::Isometry3d TwcPrior = activeKeyFrames[0]->GetPose();
        Eigen::Matrix3d R_prior = TwcPrior.rotation();

        for (const auto &obs : activeKeyFrames[0]->mObservations)
        {
            int mapPointId = obs.first;
            auto itMp = mapIdToMapPoint.find(mapPointId);

            // 卫语句：找不到该地图点，跳过
            if (itMp == mapIdToMapPoint.end())
                continue;

            Eigen::Vector3d pointWorld = itMp->second->GetWorldPos();
            Eigen::Vector3d pointCamera = TwcPrior.inverse() * pointWorld;
            double X = pointCamera.x();
            double Y = pointCamera.y();
            double Z = pointCamera.z();

            // 卫语句：深度太近，跳过
            if (Z < 0.2)
                continue;

            effectiveObsCount++;

            // 投影雅可比
            Eigen::Matrix<double, 2, 3> jacobianProj;
            double invZ = 1.0 / Z;
            double invZ2 = invZ * invZ;
            jacobianProj << camFx * invZ, 0.0, -camFx * X * invZ2,
                0.0, camFy * invZ, -camFy * Y * invZ2;

            // 位姿雅可比
            Eigen::Matrix<double, 3, 6> jacobianPose;
            jacobianPose.block<3, 3>(0, 0) = -R_prior.transpose();
            jacobianPose.block<3, 3>(0, 3) = SkewSymmetric(pointCamera);
            Eigen::Matrix<double, 2, 6> J_pose = jacobianProj * jacobianPose;

            // 地图点雅可比
            Eigen::Matrix<double, 2, 3> J_point = jacobianProj * R_prior.transpose();

            // 累加信息矩阵
            H_mm += J_pose.transpose() * omegaPixel * J_pose;
            H_mr += J_pose.transpose() * omegaPixel * J_point;
            H_rr_sum += J_point.transpose() * omegaPixel * J_point;
        }

        // 有效观测足够时，使用舒尔补计算边缘化先验
        if (effectiveObsCount > 4)
        {
            Eigen::Matrix3d H_rr_inv = H_rr_sum.inverse();
            H_prior += H_mm - H_mr * H_rr_inv * H_mr.transpose();
        }
        else
        {
            // 退化场景：使用保底约束（更强的先验权重）
            H_prior.block<3, 3>(0, 0) *= 50.0;
            H_prior.block<3, 3>(3, 3) *= 100.0;
        }
    }

    // ====================== 步骤5：Levenberg-Marquardt 迭代求解 ======================
    double lambdaLm = 1e-3;       // LM阻尼因子
    const double lambdaBoost = 10.0;
    const double lambdaShrink = 0.1;
    const int maxIterations = 10; // 最大迭代次数
    const double epsConverge = 1e-6;

    // 状态备份（用于更新失败回滚）
    std::vector<Eigen::Vector3d> lastPoseT = poseTranslation;
    std::vector<Eigen::Quaterniond> lastPoseQ = poseRotation;
    std::vector<double> lastInvDepth = invDepthVector;
    double lastCost = 0.0;
    bool firstIter = true;

    // 逆深度有界约束（最远1000米，最近0.1米）
    const double minInvDepth = 0.001;
    const double maxInvDepth = 10.0;

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // ---------- 5.1 构建全局Hessian矩阵与梯度向量 ----------
        Eigen::MatrixXd H_xx = Eigen::MatrixXd::Zero(poseDim, poseDim);
        Eigen::VectorXd g_x = Eigen::VectorXd::Zero(poseDim);

        std::vector<double> H_ll(numMapPoints, 0.0);                                      // 逆深度Hessian对角元
        std::vector<Eigen::VectorXd> H_xl(numMapPoints, Eigen::VectorXd::Zero(poseDim)); // 位姿-逆深度交叉项
        std::vector<double> g_l(numMapPoints, 0.0);                                       // 逆深度梯度

        double totalCost = 0.0;

        // 遍历所有观测边累加
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

            // 卫语句：计算失败，跳过该边
            if (!valid)
                continue;

            // 自适应Huber核加权
            double resNorm = residual.norm();
            double huberW = 1.0;
            if (resNorm > edge.huberDelta)
                huberW = edge.huberDelta / resNorm;

            residual *= huberW;
            J_host *= huberW;
            J_target *= huberW;
            J_lambda *= huberW;

            totalCost += residual.squaredNorm();

            // 累加位姿Hessian分块
            H_xx.block<6, 6>(hostStart, hostStart) += J_host.transpose() * J_host;
            H_xx.block<6, 6>(hostStart, targetStart) += J_host.transpose() * J_target;
            H_xx.block<6, 6>(targetStart, hostStart) += J_target.transpose() * J_host;
            H_xx.block<6, 6>(targetStart, targetStart) += J_target.transpose() * J_target;

            // 累加位姿梯度
            g_x.segment<6>(hostStart) += J_host.transpose() * residual;
            g_x.segment<6>(targetStart) += J_target.transpose() * residual;

            // 累加逆深度相关项
            H_ll[mpIdx] += (J_lambda.transpose() * J_lambda)(0, 0);
            H_xl[mpIdx].segment<6>(hostStart) += J_host.transpose() * J_lambda;
            H_xl[mpIdx].segment<6>(targetStart) += J_target.transpose() * J_lambda;
            g_l[mpIdx] += (J_lambda.transpose() * residual)(0, 0);
        }

        // ---------- 5.2 加入第一帧边缘化先验约束 ----------
        {
            int firstStart = 0;
            Eigen::Matrix<double, 6, 1> priorError = ComputePoseError(
                priorTranslation, priorRotation,
                poseTranslation[0], poseRotation[0]);

            H_xx.block<6, 6>(firstStart, firstStart) += H_prior;
            g_x.segment<6>(firstStart) += H_prior * priorError;
            totalCost += priorError.transpose() * H_prior * priorError;
        }

        // ---------- 5.3 舒尔补消去逆深度（仅需求解位姿增量） ----------
        Eigen::MatrixXd H_schur = H_xx;
        Eigen::VectorXd g_schur = g_x;

        for (int j = 0; j < numMapPoints; ++j)
        {
            // 卫语句：Hessian对角元太小，跳过
            if (H_ll[j] < 1e-12)
                continue;

            H_schur -= H_xl[j] * H_xl[j].transpose() / H_ll[j];
            g_schur -= H_xl[j] * g_l[j] / H_ll[j];
        }

        // ---------- 5.4 LM阻尼正则化 ----------
        Eigen::MatrixXd H_reg = H_schur;
        for (int i = 0; i < poseDim; ++i)
            H_reg(i, i) += lambdaLm;

        // ---------- 5.5 求解位姿增量 ----------
        Eigen::LLT<Eigen::MatrixXd> llt(H_reg);

        // 卫语句：Cholesky分解失败，增大阻尼继续
        if (llt.info() != Eigen::Success)
        {
            lambdaLm *= lambdaBoost;
            continue;
        }

        Eigen::VectorXd delta_x = llt.solve(-g_schur);

        // ---------- 5.6 回代求解逆深度增量 ----------
        Eigen::VectorXd delta_l(numMapPoints);
        for (int j = 0; j < numMapPoints; ++j)
        {
            // 卫语句：Hessian对角元太小，增量为0
            if (H_ll[j] < 1e-12)
            {
                delta_l(j) = 0.0;
                continue;
            }

            double rhs = -g_l[j] - H_xl[j].dot(delta_x);
            delta_l(j) = rhs / H_ll[j];
        }

        // ---------- 5.7 状态更新 ----------
        // 备份上一轮状态
        lastPoseT = poseTranslation;
        lastPoseQ = poseRotation;
        lastInvDepth = invDepthVector;
        lastCost = totalCost;

        // 更新位姿（流形约束）
        for (int i = 0; i < numKeyFrames; ++i)
        {
            Eigen::Matrix<double, 6, 1> delta = delta_x.segment<6>(i * 6);
            UpdatePose(poseTranslation[i], poseRotation[i], delta,
                       poseTranslation[i], poseRotation[i]);
        }

        // 更新逆深度 + 有界约束截断
        for (int j = 0; j < numMapPoints; ++j)
        {
            invDepthVector[j] += delta_l(j);
            invDepthVector[j] = std::max(minInvDepth, std::min(maxInvDepth, invDepthVector[j]));
        }

        // ---------- 5.8 代价校验 + 阻尼调整 ----------
        // 计算更新后的总代价
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

        // 加上先验代价
        {
            Eigen::Matrix<double, 6, 1> perr = ComputePoseError(
                priorTranslation, priorRotation,
                poseTranslation[0], poseRotation[0]);
            newCost += perr.transpose() * H_prior * perr;
        }

        // LM阻尼策略
        if (newCost < lastCost)
        {
            // 代价下降：减小阻尼，检查收敛
            lambdaLm *= lambdaShrink;

            // 卫语句：收敛，退出迭代
            if (fabs(lastCost - newCost) < epsConverge * lastCost)
                break;
        }
        else
        {
            // 代价上升：回滚状态，增大阻尼
            poseTranslation = lastPoseT;
            poseRotation = lastPoseQ;
            invDepthVector = lastInvDepth;
            lambdaLm *= lambdaBoost;
        }
    }

    // ====================== 步骤6：优化结果写回地图 ======================

    // A. 写回关键帧位姿
    for (int i = 0; i < numKeyFrames; ++i)
    {
        Eigen::Isometry3d TwcOpt = Eigen::Isometry3d::Identity();
        TwcOpt.linear() = poseRotation[i].toRotationMatrix();
        TwcOpt.translation() = poseTranslation[i];
        activeKeyFrames[i]->SetPose(TwcOpt);
    }

    // B. 逆深度反投影写回地图点世界坐标
    for (int j = 0; j < numMapPoints; ++j)
    {
        double lambdaOpt = invDepthVector[j];
        int hostIdx = mapPointHostFrameIdx[j];
        int mapPointId = mapPointRefVec[j]->GetFeatureId();

        cv::Point2f ptHost = activeKeyFrames[hostIdx]->mObservations[mapPointId];
        double hostUn = (ptHost.x - camCx) / camFx;
        double hostVn = (ptHost.y - camCy) / camFy;

        Eigen::Vector3d hostRayScaled(hostUn, hostVn, 1.0);
        Eigen::Vector3d posWorldOpt = poseRotation[hostIdx] * (hostRayScaled / lambdaOpt)
                                      + poseTranslation[hostIdx];

        mapPointRefVec[j]->SetWorldPos(posWorldOpt);
    }
}
