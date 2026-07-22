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
 * @brief 计算双目重投影残差及其对位姿和逆深度的雅可比矩阵 (基于 Sophus 右扰动模型规范化)
 *
 * @param hostTranslation    参考帧(Host)在世界坐标系下的平移向量 $t_{w,host}$
 * @param hostRotation       参考帧(Host)在世界坐标系下的旋转四元数 $q_{w,host}$
 * @param targetTranslation  当前帧(Target)在世界坐标系下的平移向量 $t_{w,target}$
 * @param targetRotation     当前帧(Target)在世界坐标系下的旋转四元数 $q_{w,target}$
 * @param invDepth           地图点在 Host 帧相机坐标系下的逆深度 $\lambda = 1/Z_{host}$
 * @param hostUn             地图点在 Host 帧左目归一化图像平面上的 $x$ 坐标
 * @param hostVn             地图点在 Host 帧左目归一化图像平面上的 $y$ 坐标
 * @param targetLeftU        地图点在 Target 帧左目的实际观测像素坐标 $u_L$
 * @param targetLeftV        地图点在 Target 帧左目的实际观测像素坐标 $v_L$
 * @param targetRightU       地图点在 Target 帧右目的实际观测像素坐标 $u_R$
 * @param targetRightV       地图点在 Target 帧右目的实际观测像素坐标 $v_R$
 * @param hasRight           当前 Target 帧是否成功追踪到了该点的右目观测
 * @param fx, fy, cx, cy     相机内参
 * @param T_c1_c0            左目相机到右目相机的相对外参 (即双目基线变换 $T_{right, left}$)
 *
 * @output residual          4维残差向量 $[r_{uL}, r_{vL}, r_{uR}, r_{vR}]^T$
 * @output jacobianHost      残差对 Host 位姿的雅可比矩阵 ($4 \times 6$)
 * @output jacobianTarget    残差对 Target 位姿的雅可比矩阵 ($4 \times 6$)
 * @output jacobianInvDepth  残差对地图点在 Host 帧下逆深度的雅可比矩阵 ($4 \times 1$)
 *
 * @return 是否计算成功（若点投影到 Target 帧后深度不合法，返回 false）
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
    // 初始化输出
    residual.setZero();
    jacobianHost.setZero();
    jacobianTarget.setZero();
    jacobianInvDepth.setZero();

    // 构造李群形式的位姿变换矩阵 T_w_host 和 T_w_target
    Sophus::SE3d T_w_host(hostRotation, hostTranslation);
    Sophus::SE3d T_w_target(targetRotation, targetTranslation);

    // 计算从 Host 帧到 Target 帧的相对位姿变换： $T_{target, host} = T_{w,target}^{-1} \cdot T_{w,host}$
    Sophus::SE3d T_target_host = T_w_target.inverse() * T_w_host;

    // 根据 Host 归一化坐标和逆深度，恢复地图点在 Host 左目相机坐标系下的 3D 坐标 $P_{host}$
    Eigen::Vector3d hostRay(hostUn, hostVn, 1.0);
    Eigen::Vector3d pointHost = hostRay / invDepth;

    // -------------------------------------------------------------------------
    // 1. 投影到 Target 帧左目 (c0) 坐标系并计算残差与雅可比
    // -------------------------------------------------------------------------
    Eigen::Vector3d pointTarget_c0 = T_target_host * pointHost;
    double X0 = pointTarget_c0.x(), Y0 = pointTarget_c0.y(), Z0 = pointTarget_c0.z();

    if (Z0 < 1e-4)
        return false; // 严格的非正正定深度校验，防止数值爆炸

    double invZ0 = 1.0 / Z0;
    double invZ0_2 = invZ0 * invZ0;

    // 计算像素坐标对相机系 3D 点的投影导数 (链式法则第一步) $\frac{\partial e}{\partial P_{target}}$
    Eigen::Matrix<double, 2, 3> jacProj0;
    jacProj0 << fx * invZ0, 0.0, -fx * X0 * invZ0_2,
        0.0, fy * invZ0, -fy * Y0 * invZ0_2;

    // 计算 Target 帧相机系 3D 点对 Target 位姿（李代数 $\delta \xi_{target}$）的右扰动导数
    Eigen::Matrix<double, 3, 6> dPt0_dXiTarget, dPt0_dXiHost;
    dPt0_dXiTarget.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    dPt0_dXiTarget.block<3, 3>(0, 3) = Sophus::SO3d::hat(pointTarget_c0); // 反对称矩阵

    // 最终残差对 Target 位姿的雅可比 (注意带负号，因为残差 = 观测 - 预测)
    jacobianTarget.block<2, 6>(0, 0) = -jacProj0 * dPt0_dXiTarget;

    // 计算 Target 帧相机系 3D 点对 Host 位姿（李代数 $\delta \xi_{host}$）的右扰动导数
    dPt0_dXiHost.block<3, 3>(0, 0) = T_target_host.rotationMatrix();
    dPt0_dXiHost.block<3, 3>(0, 3) = -T_target_host.rotationMatrix() * Sophus::SO3d::hat(pointHost);

    // 最终残差对 Host 位姿的雅可比
    jacobianHost.block<2, 6>(0, 0) = -jacProj0 * dPt0_dXiHost;

    // 计算 Target 帧相机系 3D 点对 Host 逆深度 $\lambda$ 的导数
    Eigen::Vector3d dPt0_dLambda = T_target_host.rotationMatrix() * (-hostRay / (invDepth * invDepth));
    jacobianInvDepth.block<2, 1>(0, 0) = -jacProj0 * dPt0_dLambda;

    // 计算左目重投影残差：残差 = 实际观测值 - 针孔模型预测值
    residual(0) = targetLeftU - (fx * X0 / Z0 + cx);
    residual(1) = targetLeftV - (fy * Y0 / Z0 + cy);

    // -------------------------------------------------------------------------
    // 2. 投影到 Target 帧右目 (c1) 并计算相应残差与雅可比
    // -------------------------------------------------------------------------
    if (hasRight)
    {
        // 将 Target 左目系下的点变换到 Target 右目系下：$P_{target\_c1} = T_{c1\_c0} \cdot P_{target\_c0}$
        Eigen::Vector3d pointTarget_c1 = T_c1_c0 * pointTarget_c0;
        double X1 = pointTarget_c1.x(), Y1 = pointTarget_c1.y(), Z1 = pointTarget_c1.z();

        if (Z1 > 1e-4)
        {
            double invZ1 = 1.0 / Z1;
            double invZ1_2 = invZ1 * invZ1;

            // 右目投影矩阵导数
            Eigen::Matrix<double, 2, 3> jacProj1;
            jacProj1 << fx * invZ1, 0.0, -fx * X1 * invZ1_2,
                0.0, fy * invZ1, -fy * Y1 * invZ1_2;

            Eigen::Matrix3d R_c1_c0 = T_c1_c0.rotationMatrix();

            // 利用链式法则，乘以旋转矩阵 $R_{c1\_c0}$ 将导数传递回去
            jacobianTarget.block<2, 6>(2, 0) = -jacProj1 * R_c1_c0 * dPt0_dXiTarget;
            jacobianHost.block<2, 6>(2, 0) = -jacProj1 * R_c1_c0 * dPt0_dXiHost;
            jacobianInvDepth.block<2, 1>(2, 0) = -jacProj1 * R_c1_c0 * dPt0_dLambda;

            // 计算右目重投影残差
            residual(2) = targetRightU - (fx * X1 / Z1 + cx);
            residual(3) = targetRightV - (fy * Y1 / Z1 + cy);
        }
    }

    return true;
}

// ==================== Ceres 参数块与局部参数化定义 ====================

/**
 * @brief 自定义 Ceres 局部参数化类，处理位姿在流形 (Manifold) 上的加法
 *
 * 传统的位姿参数块采用 7 维向量存储（3维平移 + 4维四元数旋转）。
 * 四元数具有过参数化特征（必须保持单位长度），直接进行常规加法会破坏流形约束。
 * 这里采用李群 SE(3) 的指数映射将 6 维切空间(李代数)的微增量更新到 7 维流形上位姿。
 */
class PoseLocalParameterization : public ceres::LocalParameterization
{
public:
    virtual ~PoseLocalParameterization() {}

    /**
     * @brief 流形上的加法算子： $T_{new} = T_{old} \cdot \exp(\delta \xi)$
     */
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        Eigen::Map<const Eigen::Vector3d> oldTranslation(x);
        Eigen::Map<const Eigen::Quaterniond> oldRotation(x + 3);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> deltaPose(delta); // 6维李代数微量

        Eigen::Map<Eigen::Vector3d> newTranslation(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> newRotation(x_plus_delta + 3);

        Sophus::SE3d T_old(oldRotation, oldTranslation);
        // 右乘指数映射更新： exp(delta)
        Sophus::SE3d T_new = T_old * Sophus::SE3d::exp(deltaPose);

        newTranslation = T_new.translation();
        newRotation = T_new.unit_quaternion();
        return true;
    }

    /**
     * @brief 计算全局参数对局部切空间的雅可比，由于我们在 Compute 内部已经完成了扰动推导，这里直接给单位阵。
     */
    virtual bool ComputeJacobian(const double *x, double *jacobian) const
    {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    virtual int GlobalSize() const { return 7; } // 内存存储大小 (x,y,z, qx,qy,qz,qw)
    virtual int LocalSize() const { return 6; }  // 实际自由度大小 (重构平移自由度3 + 旋转自由度3)
};

// ==================== Ceres 双目残差块定义 ====================

/**
 * @brief 双目重投影残差代价函数
 *
 * 继承自形式最灵活的基类 `ceres::CostFunction`，方便我们自己手动编写高效的解析雅可比。
 */
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
        // 声明优化的各个参数块维度
        mutable_parameter_block_sizes()->push_back(7); // 参数块0: Host 关键帧位姿 (7维)
        mutable_parameter_block_sizes()->push_back(7); // 参数块1: Target 关键帧位姿 (7维)
        mutable_parameter_block_sizes()->push_back(1); // 参数块2: 该地图点的 Host 逆深度 (1维)
        set_num_residuals(4);                          // 4维残差向量 $[u_L, v_L, u_R, v_R]^T$
    }

    /**
     * @brief Ceres 核心评估计算接口：计算残差以及对应的雅可比矩阵
     */
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        // 1. 映射输入参数块到 Eigen 结构
        Eigen::Map<const Eigen::Vector3d> hostT(parameters[0]);
        Eigen::Map<const Eigen::Quaterniond> hostQ(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> targetT(parameters[1]);
        Eigen::Map<const Eigen::Quaterniond> targetQ(parameters[1] + 3);
        double invDepth = parameters[2][0];

        Eigen::Matrix<double, 4, 1> res;
        Eigen::Matrix<double, 4, 6> J_host;
        Eigen::Matrix<double, 4, 6> J_target;
        Eigen::Matrix<double, 4, 1> J_invDepth;

        // 2. 调用核心数学辅助函数，计算残差以及各模块解析导数
        bool valid = ComputeStereoReprojectionResidualAndJacobians(
            hostT, hostQ, targetT, targetQ, invDepth, hostUn_, hostVn_,
            tLU_, tLV_, tRU_, tRV_, hasRight_, fx_, fy_, cx_, cy_, T_c1_c0_,
            res, J_host, J_target, J_invDepth);

        if (!valid)
            return false;

        // 3. 填充残差向量
        for (int i = 0; i < 4; ++i)
            residuals[i] = res(i);

        // 如果没有有效的右目追踪，将后两维（右目）的残差硬性清零，避免干扰优化
        if (!hasRight_)
        {
            residuals[2] = 0.0;
            residuals[3] = 0.0;
        }

        // 4. 填充雅可比矩阵块（供 Ceres 组装 H 矩阵和 g 向量）
        if (jacobians)
        {
            // 对 Host 位姿的导数块 ($4 \times 7$)
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor>> J0(jacobians[0]);
                J0.setZero();
                J0.block<4, 6>(0, 0) = J_host; // 仅填充前 6 列局部自由度切空间导数
                if (!hasRight_)
                    J0.block<2, 6>(2, 0).setZero();
            }
            // 对 Target 位姿的导数块 ($4 \times 7$)
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor>> J1(jacobians[1]);
                J1.setZero();
                J1.block<4, 6>(0, 0) = J_target;
                if (!hasRight_)
                    J1.block<2, 6>(2, 0).setZero();
            }
            // 对地图点逆深度的导数块 ($4 \times 1$)
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 4, 1>> J2(jacobians[2]);
                J2 = J_invDepth;
                if (!hasRight_)
                    J2.block<2, 1>(2, 0).setZero();
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

/**
 * @brief 执行局部滑动窗口光束法平差 (Local Bundle Adjustment)
 */
void Optimizer::LocalBundleAdjustment(std::shared_ptr<Map> map, int windowSize)
{
    if (!map)
        return;

    // 获取地图内全部关键帧
    std::vector<std::shared_ptr<KeyFrame>> allKeyFrames = map->GetAllKeyFrames();
    if (allKeyFrames.size() < 2)
        return; // 至少需要两帧才能进行相对位姿与结构交叉优化

    // -------------------------------------------------------------------------
    // 1. 提取并建立滑动窗口内的活跃关键帧 (Active KeyFrames) 集合
    // -------------------------------------------------------------------------
    std::vector<std::shared_ptr<KeyFrame>> activeKeyFrames;
    int startIdx = std::max(0, static_cast<int>(allKeyFrames.size()) - windowSize);
    for (size_t i = startIdx; i < allKeyFrames.size(); ++i)
        activeKeyFrames.push_back(allKeyFrames[i]);

    int numKeyFrames = activeKeyFrames.size();

    // 构建一个由连续双精度数组组成的容器作为位姿优化参数块，避免直接修改内部对象导致线程冲突
    std::vector<std::vector<double>> poseBlocks(numKeyFrames, std::vector<double>(7));
    std::unordered_map<unsigned long, int> keyFrameIdToIdx;

    for (int i = 0; i < numKeyFrames; ++i)
    {
        Eigen::Isometry3d Twc = activeKeyFrames[i]->GetPose(); // 获取世界系到相机系的变换矩阵 $T_{w,c}$
        Eigen::Vector3d t = Twc.translation();
        Eigen::Quaterniond q(Twc.rotation());

        // 格式转换为连续内存阵列：[x, y, z, qx, qy, qz, qw]
        poseBlocks[i][0] = t.x();
        poseBlocks[i][1] = t.y();
        poseBlocks[i][2] = t.z();
        poseBlocks[i][3] = q.x();
        poseBlocks[i][4] = q.y();
        poseBlocks[i][5] = q.z();
        poseBlocks[i][6] = q.w();

        // 记录全局关键帧 ID 到滑动窗口内部局部索引的映射关系
        keyFrameIdToIdx[activeKeyFrames[i]->mId] = i;
    }

    // 提取全局基础相机内参
    double camFx = Parameters::fx;
    double camFy = Parameters::fy;
    double camCx = Parameters::cx;
    double camCy = Parameters::cy;

    // 依靠解析好的车体/外参矩阵计算相对双目相机外参：$T_{c1,c0} = T_{body,c1}^{-1} \cdot T_{body,c0}$
    Sophus::SE3d T_b_c0(Eigen::Quaterniond(Parameters::body_T_cam0.block<3, 3>(0, 0)), Parameters::body_T_cam0.block<3, 1>(0, 3));
    Sophus::SE3d T_b_c1(Eigen::Quaterniond(Parameters::body_T_cam1.block<3, 3>(0, 0)), Parameters::body_T_cam1.block<3, 1>(0, 3));
    Sophus::SE3d T_c1_c0 = T_b_c1.inverse() * T_b_c0;

    // -------------------------------------------------------------------------
    // 2. 检索这些关键帧能观测到的所有有效地图点，并确定其参考帧 (Host Frame)
    // -------------------------------------------------------------------------
    std::vector<std::shared_ptr<MapPoint>> allMapPoints = map->GetAllMapPoints();
    std::unordered_map<int, std::shared_ptr<MapPoint>> mapIdToMapPoint;
    for (const auto &mp : allMapPoints)
        mapIdToMapPoint[mp->GetFeatureId()] = mp;

    std::unordered_map<int, int> mapPointIdToIdx;
    std::vector<double> invDepthVector;                    // 存储优化的地图点逆深度参数块一维连续数组
    std::vector<int> mapPointHostFrameIdx;                 // 记录每个地图点在滑窗内的哪个关键帧被确立为 Host 帧
    std::vector<std::shared_ptr<MapPoint>> mapPointRefVec; // 关联原始地图点的指针以便后续写回结果

    for (int i = 0; i < numKeyFrames; ++i)
    {
        auto keyFrame = activeKeyFrames[i];
        for (const auto &obs : keyFrame->mmObservations)
        {
            int mapPointId = obs.first;
            auto itMp = mapIdToMapPoint.find(mapPointId);
            if (itMp == mapIdToMapPoint.end())
                continue;

            // 如果该地图点未曾在优化队列中被初始化，则将其加入逆深度优化阵列
            if (mapPointIdToIdx.find(mapPointId) == mapPointIdToIdx.end())
            {
                int mpIdx = invDepthVector.size();
                mapPointIdToIdx[mapPointId] = mpIdx;
                mapPointHostFrameIdx.push_back(i); // 将该点首次在滑动窗口中出现的关键帧设为它的 Host 帧
                mapPointRefVec.push_back(itMp->second);

                // 根据当前世界坐标，计算出该地图点在它的 Host 帧坐标系下的 3D 点坐标，以此推算初始逆深度
                Eigen::Vector3d pointWorld = itMp->second->GetWorldPos();
                Eigen::Vector3d pointHost = keyFrame->GetPose().inverse() * pointWorld;
                double depth = pointHost.z() < 0.1 ? 1.0 : pointHost.z(); // 异常保护
                invDepthVector.push_back(1.0 / depth);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 3. 构建 Ceres 非线性最小二乘问题
    // -------------------------------------------------------------------------
    ceres::Problem problem;
    PoseLocalParameterization *poseParameterization = new PoseLocalParameterization();

    // 3a. 将关键帧位姿参数块加入优化问题，并绑定自定义的流形旋转更新算子
    for (int i = 0; i < numKeyFrames; ++i)
        problem.AddParameterBlock(poseBlocks[i].data(), 7, poseParameterization);

    // 关键核心：将当前滑动窗口内最早的第一帧关键帧固定 (Set Constant)
    // 作用是约束 SLAM 的标度与 7 自由度规范场不发生整体漂移（即规范场零点固定值：固定第一帧位姿）
    problem.SetParameterBlockConstant(poseBlocks[0].data());

    // 3b. 交叉关联滑窗内所有的观测，填充残差块
    for (int targetIdx = 0; targetIdx < numKeyFrames; ++targetIdx)
    {
        auto keyFrame = activeKeyFrames[targetIdx];
        for (const auto &obs : keyFrame->mmObservations)
        {
            int mapPointId = obs.first;
            StereoObs obsTarget = obs.second;

            auto itMpIdx = mapPointIdToIdx.find(mapPointId);
            if (itMpIdx == mapPointIdToIdx.end())
                continue;

            int mpIdx = itMpIdx->second;
            int hostIdx = mapPointHostFrameIdx[mpIdx];

            // 跳过自共视观测（即自己对自己的观测），只优化不同帧之间的跨帧重投影残差
            if (hostIdx == targetIdx)
                continue;

            // 获取 Host 帧的左目观测数据
            StereoObs obsHost = activeKeyFrames[hostIdx]->mmObservations[mapPointId];
            double hostUn = (obsHost.ptLeft.x - camCx) / camFx; // 转换到归一化平面坐标
            double hostVn = (obsHost.ptLeft.y - camCy) / camFy;

            // 根据该点的总历史观测次数，动态调整鲁棒核函数的阈值 (Huber Delta)
            // 观测次数越多说明越稳定，给予更小的 Huber 阈值防止误剔除
            int obsCount = mapPointRefVec[mpIdx]->GetObservationCount();
            double huberDelta = (obsCount >= 5) ? 0.5 : ((obsCount >= 2) ? 1.0 : 1.5);

            // 构造重投影残差实例
            ceres::CostFunction *costFunction = new StereoReprojectionCostFunction(
                hostUn, hostVn,
                obsTarget.ptLeft.x, obsTarget.ptLeft.y,
                obsTarget.ptRight.x, obsTarget.ptRight.y, obsTarget.hasRight,
                camFx, camFy, camCx, camCy, T_c1_c0);

            // 使用 Huber 鲁棒损失函数降低外点(Outliers, 错配点)对图优化 H 矩阵的负面抗扰能力
            ceres::LossFunction *lossFunction = new ceres::HuberLoss(huberDelta);

            // 向 Ceres 注册添加该残差块（该残差强耦合了 Host 位姿、Target 位姿以及点的局部逆深度）
            problem.AddResidualBlock(costFunction, lossFunction,
                                     poseBlocks[hostIdx].data(),
                                     poseBlocks[targetIdx].data(),
                                     &invDepthVector[mpIdx]);

            // 设置逆深度的物理意义边界，限制地图点在合理的深度可视范围内
            problem.SetParameterLowerBound(&invDepthVector[mpIdx], 0, 1.0 / 60.0); // 对应最大深度 300m
            problem.SetParameterUpperBound(&invDepthVector[mpIdx], 0, 1.0 / 0.5);   // 对应最小深度 0.5m
        }
    }

    // -------------------------------------------------------------------------
    // 4. 配置 Ceres 求解器选项并执行迭代优化
    // -------------------------------------------------------------------------
    ceres::Solver::Options options;
    // 核心加速：选择基于 舒尔消元 (Schur Complement) 的密集解算器 DENSE_SCHUR
    // 利用了 SLAM 图优化的稀疏二分图特性（先消去点参数块，优先求解位姿增量，实现极快速的局部 BA）
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT; // 使用经典的 LM 阻尼方法
    options.max_num_iterations = 15;                                 // 限制最大迭代次数，保证后端线程能快速实时响应
    options.function_tolerance = 1e-4;
    options.parameter_tolerance = 1e-4;
    options.minimizer_progress_to_stdout = false; // 关闭标准输出，保持终端整洁

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary); // 启动解算

    // -------------------------------------------------------------------------
    // 5. 优化完成，将收敛后的最优值写回原 SLAM 核心数据结构
    // -------------------------------------------------------------------------

    // 5a. 更新关键帧位姿
    for (int i = 0; i < numKeyFrames; ++i)
    {
        Eigen::Vector3d t(poseBlocks[i][0], poseBlocks[i][1], poseBlocks[i][2]);
        Eigen::Quaterniond q(poseBlocks[i][6], poseBlocks[i][3], poseBlocks[i][4], poseBlocks[i][5]);
        q.normalize(); // 必须重新归一化，消除累积数值舍入误差

        Eigen::Isometry3d TwcOpt = Eigen::Isometry3d::Identity();
        TwcOpt.linear() = q.toRotationMatrix();
        TwcOpt.translation() = t;
        activeKeyFrames[i]->SetPose(TwcOpt); // 写入到线程安全的关键帧对象中
    }

    // 5b. 根据优化后的位姿和最优逆深度，重新交会计算出地图点在世界坐标系下的 3D 绝对坐标
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

        // 获得优化后的 Host 帧最新真实位姿
        Eigen::Vector3d tHost(poseBlocks[hostIdx][0], poseBlocks[hostIdx][1], poseBlocks[hostIdx][2]);
        Eigen::Quaterniond qHost(poseBlocks[hostIdx][6], poseBlocks[hostIdx][3], poseBlocks[hostIdx][4], poseBlocks[hostIdx][5]);
        qHost.normalize();

        // 计算公式： $P_{world} = R_{w,host} \cdot (P_{host} / \lambda) + t_{w,host}$
        Eigen::Vector3d posWorldOpt = qHost * (hostRayScaled / lambdaOpt) + tHost;
        mapPointRefVec[j]->SetWorldPos(posWorldOpt); // 写入最优 3D 坐标
    }
}

// 自定义 SE(3) 位姿图约束残差块
struct SE3GraphCostFunction
{
    SE3GraphCostFunction(const Sophus::SE3d &T_ij_measured)
        : T_ij_measured_(T_ij_measured) {}

    template <typename T>
    bool operator()(const T *const pose_i, const T *const pose_j, T *residuals) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_i(pose_i);
        Eigen::Map<const Eigen::Quaternion<T>> q_i(pose_i + 3);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_j(pose_j);
        Eigen::Map<const Eigen::Quaternion<T>> q_j(pose_j + 3);

        Sophus::SE3<T> T_w_i(q_i, t_i);
        Sophus::SE3<T> T_w_j(q_j, t_j);

        // 预测的相对位姿 T_ij = T_w_i^-1 * T_w_j
        Sophus::SE3<T> T_ij_pred = T_w_i.inverse() * T_w_j;

        // 位姿残差 e = Log( T_ij_measured^-1 * T_ij_pred )
        Sophus::SE3<T> T_error = T_ij_measured_.template cast<T>().inverse() * T_ij_pred;
        Eigen::Matrix<T, 6, 1> error = T_error.log();

        for (int k = 0; k < 6; ++k)
            residuals[k] = error(k);

        return true;
    }

    Sophus::SE3d T_ij_measured_;
};

void Optimizer::PoseGraphOptimization(
    std::shared_ptr<Map> map,
    const std::vector<std::pair<unsigned long, unsigned long>> &loopEdges,
    const std::map<std::pair<unsigned long, unsigned long>, Sophus::SE3d> &loopRelativePoses)
{
    if (!map) return;
    auto keyframes = map->GetAllKeyFrames();
    if (keyframes.size() < 2) return;

    ceres::Problem problem;
    PoseLocalParameterization *poseParameterization = new PoseLocalParameterization();

    std::map<unsigned long, std::vector<double>> poseBlocks;
    std::map<unsigned long, std::shared_ptr<KeyFrame>> idToKF;

    for (auto &kf : keyframes)
    {
        idToKF[kf->mId] = kf;
        Eigen::Isometry3d Twc = kf->GetPose();
        Eigen::Vector3d t = Twc.translation();
        Eigen::Quaterniond q(Twc.rotation());

        poseBlocks[kf->mId] = {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
        problem.AddParameterBlock(poseBlocks[kf->mId].data(), 7, poseParameterization);
    }

    // 固定首个关键帧
    problem.SetParameterBlockConstant(poseBlocks[keyframes.front()->mId].data());

    // 1. 添加连续相邻帧的 Odometry 边
    for (size_t i = 0; i < keyframes.size() - 1; ++i)
    {
        auto kf1 = keyframes[i];
        auto kf2 = keyframes[i + 1];

        Sophus::SE3d T_w_1(kf1->GetPose().rotation(), kf1->GetPose().translation());
        Sophus::SE3d T_w_2(kf2->GetPose().rotation(), kf2->GetPose().translation());
        Sophus::SE3d T_12 = T_w_1.inverse() * T_w_2;

        ceres::CostFunction *cost_func =
            new ceres::AutoDiffCostFunction<SE3GraphCostFunction, 6, 7, 7>(
                new SE3GraphCostFunction(T_12));

        problem.AddResidualBlock(cost_func, nullptr, poseBlocks[kf1->mId].data(), poseBlocks[kf2->mId].data());
    }

    // 2. 添加回环检测约束边 (Loop Edges)
    for (const auto &edge : loopEdges)
    {
        unsigned long id1 = edge.first;
        unsigned long id2 = edge.second;
        if (poseBlocks.count(id1) && poseBlocks.count(id2))
        {
            Sophus::SE3d T_12_loop = loopRelativePoses.at(edge);
            ceres::CostFunction *cost_func =
                new ceres::AutoDiffCostFunction<SE3GraphCostFunction, 6, 7, 7>(
                    new SE3GraphCostFunction(T_12_loop));

            ceres::LossFunction *loss_func = new ceres::HuberLoss(1.0);
            problem.AddResidualBlock(cost_func, loss_func, poseBlocks[id1].data(), poseBlocks[id2].data());
        }
    }

    // 优化配置
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 30;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 写回优化后的关键帧位姿
    for (auto &pair : poseBlocks)
    {
        unsigned long id = pair.first;
        Eigen::Vector3d t(pair.second[0], pair.second[1], pair.second[2]);
        Eigen::Quaterniond q(pair.second[6], pair.second[3], pair.second[4], pair.second[5]);
        q.normalize();

        Eigen::Isometry3d Twc = Eigen::Isometry3d::Identity();
        Twc.linear() = q.toRotationMatrix();
        Twc.translation() = t;
        idToKF[id]->SetPose(Twc);
    }
}