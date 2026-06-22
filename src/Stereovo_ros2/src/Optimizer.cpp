#include "Optimizer.h"      // 引入优化器类的头文件，包含局部 BA 函数的静态接口声明
#include "KeyFrame.h"       // 引入关键帧类的头文件，用于获取滑窗内各关键帧的测量值与位姿
#include "MapPoint.h"       // 引入地图点类的头文件，用于读取及精调三维空间路标点的坐标
#include "Parameters.h"     // 引入全局参数类的头文件，主要用于在优化中提取相机内参
#include <ceres/ceres.h>    // 引入谷歌 Ceres Solver 非线性最小二乘求解器的核心头文件
#include <ceres/rotation.h> // 引入 Ceres 的旋转数学库，提供流形四元数对三维点旋转的方法
#include <Eigen/Core>       // 引入 Eigen 矩阵基础库，用于三维空间向量的核心数学运算
#include <Eigen/Geometry>   // 引入 Eigen 几何库，提供变换矩阵与四元数之间的互相转换
#include <map>              // 引入标准字典容器，用于创建有序的 ID 到优化参数块的映射
#include <unordered_map>    // 引入无序哈希字典容器，提供 O(1) 级别的超高速内存数据查找
#include <vector>           // 引入标准动态数组容器，用于打包和批量管理帧及地图点
#include <iostream>         // 引入标准输入输出流，方便打印一些必要的调试或者异常信息

// =========================================================================
// 1. 工业级健壮重投影误差代价函数：计算 3D 世界点投影到图像上与 2D 实际观测之间的像素残差
// =========================================================================
struct SREPROJECTION_ERROR
{ // 声明重投影残差代价结构体，提供给 Ceres 做自动求导
    // 构造函数：初始化时传入某特征点的 2D 实际像素像素值 (observed_u, observed_v) 以及相机内参
    SREPROJECTION_ERROR(double observed_u, double observed_v, double fx, double fy, double cx, double cy) //
        : u(observed_u), v(observed_v), fx(fx), fy(fy), cx(cx), cy(cy)
    {
    } // 使用初始化列表给结构体内部常量赋值

    template <typename T> // 使用 C++ 模板支撑 Ceres 的 Jet 喷气求导类型，实现全自动微分
    bool operator()(const T *const camera_pose, const T *const point_3d, T *residuals) const
    {                                                       // 重载圆括号运算符，定义残差具体计算法则
        T p_w[3] = {point_3d[0], point_3d[1], point_3d[2]}; // 从传入的三维点指针中解包出地图点的世界坐标 X, Y, Z
        T p_c[3];                                           // 声明一个局部三维数组，用来承载变换到相机归一化坐标系下的 3D 坐标

        // 【内存排列核心对齐】参数块排布：[t_x, t_y, t_z, q_w, q_x, q_y, q_z]，前 3 项是平移，后 4 项是四元数
        T t[3] = {camera_pose[0], camera_pose[1], camera_pose[2]};                 // 解包提取平移向量 t 坐标分量
        T q[4] = {camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6]}; // 解包提取四元数 q 分量，顺序依次为 w, x, y, z

        T p_w_minus_t[3] = {p_w[0] - t[0], p_w[1] - t[1], p_w[2] - t[2]}; // 坐标转换第一步：计算世界点坐标减去平移向量 (P_w - t)

        // 【核心数学机制修正】四元数求逆。因为当前已知的是相机到世界的变换 T_wc (R_wc, t_wc)，投影需要世界到相机系变换 T_cw
        // 故对变换矩阵求逆，对于旋转部分对应的四元数，求逆即代表将其虚部 (x, y, z) 取负号
        T q_inv[4] = {q[0], -q[1], -q[2], -q[3]};              // 构造逆四元数，实部保持不变，虚部乘以 -1
        ceres::QuaternionRotatePoint(q_inv, p_w_minus_t, p_c); // 调用 Ceres 内置函数，用逆四元数旋转坐标差，得到相机系 3D 点 P_c = R_cw * (P_w - t)

        // 软件级别的极值保护机制：如果优化过程中由于过度震荡导致点掉到了相机背后（深度小于零）
        if (p_c[2] <= T(1e-4))
        {                             // 判断 Z 轴深度值是否小于等于 0.0001
            residuals[0] = T(1111.0); // 赋予 X 轴一个巨大的像素惩罚残差值
            residuals[1] = T(1111.0); // 赋予 Y 轴一个巨大的像素惩罚残差值值，以此引导优化方向回归，消除除零异常
            return true;              // 返回真，告诉 Ceres 虽然产生异常但残差依旧可计算，保障程序继续运行
        } // 保护结束

        // 标准小孔相机投影公式：计算归一化成像平面坐标
        T xp = p_c[0] / p_c[2];             // xp = X / Z
        T yp = p_c[1] / p_c[2];             // yp = Y / Z
        T predicted_u = T(fx) * xp + T(cx); // 结合焦距与光心计算在图像平面上的预测像素坐标 X：predicted_u = fx * xp + cx
        T predicted_v = T(fy) * yp + T(cy); // 结合焦距与光心计算在图像平面上的预测像素坐标 Y：predicted_v = fy * yp + cy

        // 计算真实的像素重投影残差：实际观测值减去数学几何预测值
        residuals[0] = T(u) - predicted_u; // X 方向的像素级残差
        residuals[1] = T(v) - predicted_v; // Y 方向的像素级残差

        return true; // 顺利解出残差，返回 true
    } // operator() 结束

    // 静态工厂构建函数：实例化求导代价函数。2 代表残差是2维像素，7 代表优化帧位姿块是 7 维，3 代表路标点坐标是 3 维
    static ceres::CostFunction *Create(double u, double v, double fx, double fy, double cx, double cy)
    {                                                                          //
        return (new ceres::AutoDiffCostFunction<SREPROJECTION_ERROR, 2, 7, 3>( // 通过 new 实例化自动微分代价函数对象并直接返回其指针
            new SREPROJECTION_ERROR(u, v, fx, fy, cx, cy)));                   // 嵌套传入实际观测坐标与内参作为结构体实例构造参数
    } // Create 函数结束

    double u, v;           // 结构体内存储变量：保存特征点当前的实际 2D 像素观测坐标
    double fx, fy, cx, cy; // 结构体内存储变量：保存相机固有的焦距 and 主点内参
}; // 结构体结束


// =========================================================================
// 2. 【新增特性】位姿边缘化状态先验代价函数 (Pose Prior Factor)
// 作用：在切线空间 (Tangent Space) 下通过四元数乘积扰动和欧氏平移差计算 6 维残差，并融合高置信度信息矩阵
// =========================================================================
struct PosePriorFactor
{
    // 构造函数：传入历史累积出的先验平移、先验四元数以及代表置信度权重的信息矩阵平方根
    PosePriorFactor(const Eigen::Vector3d& prior_t, const Eigen::Quaterniond& prior_q, const Eigen::Matrix<double, 6, 6>& sqrt_info)
        : m_prior_t(prior_t), m_prior_q(prior_q), m_sqrt_info(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const camera_pose, T* residuals) const
    {
        // 1. 解包待优化位姿参数块：[t_x, t_y, t_z, q_w, q_x, q_y, q_z]
        T t[3] = {camera_pose[0], camera_pose[1], camera_pose[2]};
        T q[4] = {camera_pose[3], camera_pose[4], camera_pose[5], camera_pose[6]};

        // 2. 切线空间旋转残差计算：利用四元数扰动公式 dq = q_prior_inv * q
        T qw1 = T(m_prior_q.w()), qx1 = T(-m_prior_q.x()), qy1 = T(-m_prior_q.y()), qz1 = T(-m_prior_q.z());
        T qw2 = q[0], qx2 = q[1], qy2 = q[2], qz2 = q[3];

        // 执行标准哈密顿四元数乘法运算 (Hamilton Product)
        T dq_w = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2;
        T dq_x = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2;
        T dq_y = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2;
        T dq_z = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2;

        // 引入双轴正负号最短路径保护机制 (Ensure shortest path conversion)
        T sign = (dq_w >= T(0.0)) ? T(1.0) : T(-1.0);

        // 3. 拼装未加权的 6 自由度原始空间残差 (前3维平移残差，后3维李代数轴角残差)
        Eigen::Matrix<T, 6, 1> raw_residuals;
        raw_residuals[0] = t[0] - T(m_prior_t.x());
        raw_residuals[1] = t[1] - T(m_prior_t.y());
        raw_residuals[2] = t[2] - T(m_prior_t.z());
        raw_residuals[3] = T(2.0) * sign * dq_x;
        raw_residuals[4] = T(2.0) * sign * dq_y;
        raw_residuals[5] = T(2.0) * sign * dq_z;

        // 4. 左乘置信度信息权重矩阵，保留历史相干空间记忆
        Eigen::Matrix<T, 6, 1> weighted_residuals = m_sqrt_info.cast<T>() * raw_residuals;
        for (int i = 0; i < 6; ++i)
        {
            residuals[i] = weighted_residuals[i];
        }
        return true;
    }

    // 静态工厂构建函数：实例化 6 维残差、7 维位姿参数块的自动微分因子
    static ceres::CostFunction* Create(const Eigen::Vector3d& prior_t, const Eigen::Quaterniond& prior_q, const Eigen::Matrix<double, 6, 6>& sqrt_info)
    {
        return (new ceres::AutoDiffCostFunction<PosePriorFactor, 6, 7>(
            new PosePriorFactor(prior_t, prior_q, sqrt_info)));
    }

    Eigen::Vector3d m_prior_t;
    Eigen::Quaterniond m_prior_q;
    Eigen::Matrix<double, 6, 6> m_sqrt_info;
};


// =========================================================================
// 3. 局部 BA 主函数实现：精调滑动窗口内若干帧关键帧的绝对位姿以及对应的地图路标坐标
// =========================================================================
void Optimizer::LocalBundleAdjustment(std::shared_ptr<Map> pMap, int windowSize)
{ //
    if (!pMap)
        return; // 安全校验：如果传入的全局地图指针为空，则属于非法调用，直接结束退出

    std::vector<std::shared_ptr<KeyFrame>> allKFs = pMap->GetAllKeyFrames(); // 从全局地图中线程安全地拷贝出所有的历史关键帧列表
    if (allKFs.size() < 2)
        return; // 边界限制：系统中至少必须拥有 2 帧及以上的关键帧才能构建起约束关系，否则直接退出

    // 1. 筛选滑动窗口激活帧
    std::vector<std::shared_ptr<KeyFrame>> activeKFs;                         // 声明容器，存放当前局部图优化被激活的滑动窗口关键帧
    int startIdx = std::max(0, static_cast<int>(allKFs.size()) - windowSize); // 索引测算：截取最新的 windowSize 帧（比如最新的10帧）的起始位置
    for (size_t i = startIdx; i < allKFs.size(); ++i)
    {                                   // 循环遍历被裁减出来的高频最新滑窗帧
        activeKFs.push_back(allKFs[i]); // 将对应的关键帧指针塞入激活帧容器中
    } // 滑窗帧筛选结束

    ceres::Problem problem; // 实例化 Ceres 的非线性优化问题管理核心对象（因子图图基底）

    std::map<unsigned long, std::vector<double>> mapKFIdToParameterBlock; // 声明局部字典：建立从“关键帧专属 ID”到“7维双精度优化参数块”的参数化路由
    std::map<int, std::vector<double>> mapMPIdToParameterBlock;           // 声明局部字典：建立从“路标点特征 ID”到“3维双精度优化坐标块”的参数化路由
    std::map<int, std::shared_ptr<MapPoint>> mapIdToMPRef;                // 声明本地字典：建立特征 ID 到实体地图点智能指针的检索路由，供优化成功后快速写回

    double cam_fx = Parameters::fx; // 从全局静态配置缓存中拷贝出相机内参焦距 fx
    double cam_fy = Parameters::fy; // 从全局静态配置缓存中拷贝出相机内参焦距 fy
    double cam_cx = Parameters::cx; // 从全局静态配置缓存中拷贝出相机内参主点光心 cx
    double cam_cy = Parameters::cy; // 从全局静态配置缓存中拷贝出相机内参主点光心 cy

    // 2. 装载激活帧位姿到 Ceres 参数块中
    for (size_t i = 0; i < activeKFs.size(); ++i)
    {                                          // 轮询每一个当前处于激活状态的滑窗帧
        auto kf = activeKFs[i];                // 获取该激活关键帧的共享智能指针
        Eigen::Isometry3d Twc = kf->GetPose(); // 调用线程安全的 GetPose() 加锁接口，安全取出该帧在当前系统数据库中存储的绝对位姿 T_wc
        Eigen::Vector3d t = Twc.translation(); // 从变换矩阵中剥离提取出绝对三维平移向量 t 坐标
        Eigen::Quaterniond q(Twc.rotation());  // 从变换矩阵中剥离提取出当前绝对旋转矩阵，并构造转换为四元数 q

        // 【核心修复对齐】强制重塑内存分布：将数据以连续数组组装为标准的 [t_x, t_y, t_z, q_w, q_x, q_y, q_z] 排布送给 Ceres
        std::vector<double> pose_block = {t.x(), t.y(), t.z(), q.w(), q.x(), q.y(), q.z()}; //
        mapKFIdToParameterBlock[kf->mId] = pose_block;                                      // 将当前帧打包完毕的这 7 维位姿数组作为值，存入 ID 参数映射字典
    } // 装载帧位姿结束

    // 【算法重大性能优化】拉出所有的地图点，预先在堆内存中生成一个专属的局部哈希检索表
    // 作用：彻底杜绝了原先版本在双层 for 循环内部采用线性查找导致整个优化复杂度随地图膨胀呈 O(N) 指数级灾难暴涨的硬伤，降为常数级 O(1)。
    std::vector<std::shared_ptr<MapPoint>> allMPs = pMap->GetAllMapPoints(); // 线程安全拉取当前小地图中累积沉淀的全局 3D 地图点路标集合
    std::unordered_map<int, std::shared_ptr<MapPoint>> mapIdToMP;            // 声明高效局域无序哈希映射字典
    for (const auto &mp : allMPs)
    {                                       // 轮询所有的三维路标点
        mapIdToMP[mp->GetFeatureId()] = mp; // 在高能哈希表里建立映射：全局特征 ID -> 实体地图点对象指针
    } // 局部哈希字典构建结束

    // 3. 建立多帧多点 2D-3D 重投影误差残差约束（因子图因子添加阶段）
    for (auto kf : activeKFs)
    {                                  // 外层循环：轮询滑窗内的每一个激活关键帧
        unsigned long kf_id = kf->mId; // 提炼取出当前遍历到的关键帧 ID 编号
        for (const auto &obs : kf->mmObservations)
        {                                  // 内层循环：遍历本帧包含的所有 2D 特征观测，obs.first 是特征 ID，obs.second 是 2D 像素点
            int mp_id = obs.first;         // 提取当前被观测特征的唯一特征 ID
            cv::Point2f pt2d = obs.second; // 提取当前特征在当前帧左图平面上的真实测量像素坐标 (u, v)

            // 【核心性能优化处】利用刚刚在上面预先构建的内存哈希映射表进行 O(1) 级瞬间碰撞定位
            auto it_mp = mapIdToMP.find(mp_id); // 在哈希字典里定向查找当前特征 ID 对应的实体 3D 地图点路标
            if (it_mp == mapIdToMP.end())
                continue;                                       // 如果没找到（说明是边缘化的坏点），直接略过此点不建约束
            std::shared_ptr<MapPoint> targetMP = it_mp->second; // 顺利查到，取出对应的地图点实体智能指针

            // 如果当前路标点的 3D 坐标参数尚未被提取并加入 Ceres 待优化队列
            if (mapMPIdToParameterBlock.find(mp_id) == mapMPIdToParameterBlock.end())
            {                                                                 //
                Eigen::Vector3d pos = targetMP->GetWorldPos();                // 加锁安全获取当前路标点对应的世界绝对 3D 坐标位置
                mapMPIdToParameterBlock[mp_id] = {pos.x(), pos.y(), pos.z()}; // 组装为包含 3 个双精度浮点数的坐标数组块，写入优化字典
                mapIdToMPRef[mp_id] = targetMP;                               // 同时将指针写入临时路由字典，方便后续 BA 收敛后能精准找到实体写回
            } // 点提取判断结束

            // 调用上方的静态工厂 Create 函数，为当前 2D-3D 重投影几何关系构建一个自动求导代价函数
            ceres::CostFunction *cost_function = SREPROJECTION_ERROR::Create( //
                pt2d.x, pt2d.y, cam_fx, cam_fy, cam_cx, cam_cy);              // 将当前测量的像素像素点与相机内参数打包传送进去

            // =========================================================================
            // 【核心改进一：自适应鲁棒核函数 (Adaptive Robust Kernel)】
            // 原理：新三角化路标由于初始基线比和匹配滑移，深度具有极高的不确定度（初始噪声大）。
            // 通过获取地图点当前已被成功观测的频次梯队，动态绑定核函数保护阈值 delta，避免优质新点在优化初期被当做杂点强制剪枝。
            // =========================================================================
            int obs_count = targetMP->GetObservationCount();
            double huber_delta = 1.0;
            if (obs_count >= 5)
            {
                huber_delta = 1.0; // 高稳定梯队（≥5帧共视）：代表系统绝对锁定的骨干点，施加严苛阈值，绝不允许其残差大幅震荡
            }
            else if (obs_count >= 2)
            {
                huber_delta = 1.5; // 中等稳定梯队（2-4帧共视）：给予平滑过渡容忍区
            }
            else
            {
                huber_delta = 3.0; // 新三角化梯队（首帧观测点）：放宽深度搜索包容度，允许产生初期较大残差，给算法创造收敛空间
            }
            ceres::LossFunction *adaptive_loss_function = new ceres::HuberLoss(huber_delta); // 为该约束边量身分配带有自适应阈值的鲁棒核函数
            // =========================================================================

            // 向 Ceres 核心问题中添加残差边（构建因子约束图：代价函数，核函数，优化关键帧位姿参数块指针，优化地图点参数块指针）
            problem.AddResidualBlock(cost_function, adaptive_loss_function,  //
                                     mapKFIdToParameterBlock[kf_id].data(),  // 传入当前对应的 7 维帧位姿参数块的首地址
                                     mapMPIdToParameterBlock[mp_id].data()); // 传入当前对应的 3 维地图点路标参数块的首地址
        } // 内层循环结束
    } // 外层循环约束构建完成

    // 4. 设置四元数流形空间约束并对滑窗首帧施加边缘化先验软约束（李代数流形空间设置与先验因子添加阶段）
    for (size_t i = 0; i < activeKFs.size(); ++i)
    {                                                                // 重新轮询激活的关键帧
        auto kf = activeKFs[i];                                      // 获取当前处理帧的智能指针
        double *pose_data = mapKFIdToParameterBlock[kf->mId].data(); // 获取对应 7 维位姿参数数组首元素的数据指针

#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1 // 针对新版 Ceres 库（Ceres 2.1.0 以上）执行新 API 条件编译
        // 【核心API修复机制】新版下旧的 SubsetManifold(7, {0,1,2}) API 被彻底废弃。
        // 必须通过 ProductManifold 将 3维平移的欧氏流形和 4维旋转的四元数流形强力融合，从而正确约束优化边界。
        problem.SetManifold(pose_data, new ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::QuaternionManifold>( //
                                           ceres::EuclideanManifold<3>(), ceres::QuaternionManifold()));                   // 实例化并指定该 7 维块的前 3 项走常规欧氏相加，后 4 项走规范四元数流形更新（J_plus机制）
#else                                                                                                                      // 针对旧版 Ceres 库（Ceres 2.0 及以下）执行向下兼容条件编译
        ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization();              // 实例化旧版标准的四元数参数化空间几何对象
        problem.SetParameterization(pose_data, new ceres::ProductParameterization(                                        // 调用旧版 API 建立乘积空间局部参数化机制
                                                   new ceres::IdentityParameterization(3), quaternion_parameterization)); // 申明前 3 维用恒等参数化，后 4 维绑定刚才创建的四元数参数化结构
#endif                                                                                                                     // 条件编译分支结束

        // =========================================================================
        // 【核心改进二：边缘化先验因子软约束 (Pose Prior Factor)】
        // 原理：原代码采用 SetParameterBlockConstant 强行把滑窗首帧 (i=0) 锁死。这种生硬切断几何约束的做法会导致先验状态与后续滑动状态的协方差硬性解耦。
        // 现在的做法：将滑窗头帧从固定参数块中释放出来，转而添加一个自定义的 PosePriorFactor，将其前一步累积出的状态作为先验锚点，允许其在具有物理意义的高置信度范围内进行软微调。
        // =========================================================================
        if (i == 0)
        {
            Eigen::Isometry3d Twc_prior = kf->GetPose();       // 获取当前累积的最老历史位姿基准
            Eigen::Vector3d prior_t = Twc_prior.translation(); // 提炼绝对物理平移作为先验中心点
            Eigen::Quaterniond prior_q(Twc_prior.rotation());  // 提炼旋转四元数作为先验中心点

            // 构造代表历史空间记忆强度的 6x6 协方差逆矩阵（信息矩阵的平方根形式）
            Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::Matrix<double, 6, 6>::Identity();
            sqrt_info.block<3, 3>(0, 0) *= 50.0;  // 设定平移方向刚度，使其保持亚厘米级软约束
            sqrt_info.block<3, 3>(3, 3) *= 100.0; // 设定旋转方向刚度，由于扰动对坐标投影影响大，给予更高的置信度因子

            // 通过静态工厂实例化先验代价约束块
            ceres::CostFunction* prior_cost_function = PosePriorFactor::Create(prior_t, prior_q, sqrt_info);
            problem.AddResidualBlock(prior_cost_function, nullptr, pose_data); // 将该软约束边植入因子图中，从而完美消除 Gauge Freedom
        }
        // =========================================================================
    } // 流形约束与先验因子的赋予完成

    // 5. 求解器高级参数配置
    ceres::Solver::Options options;                                  // 声明 Ceres 求解器选项配置大结构体
    options.linear_solver_type = ceres::SPARSE_SCHUR;                // 指定使用稀疏舒尔消元算法（Sparse Schur Complement），它是专为解 Bundle Adjustment 稀疏海森矩阵定制的超高效线性求解器
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT; // 选择经典的 LM（Levenberg-Marquardt）信赖域下降策略作为非线性核心迭代优化算法
    options.max_num_iterations = 8;                                  // 极限压榨耗时：设定最大迭代下降次数为 8 次，充分保障后台优化在几毫秒内闪电收敛，绝不拖累实时帧率
    options.minimizer_progress_to_stdout = false;                    // 关闭每步迭代向系统控制台打印日志的开关，杜绝频繁高频 IO 打印导致的额外系统耗时

    ceres::Solver::Summary summary;            // 声明结果分析简报结构体，用于收集收敛状态及统计指标
    ceres::Solve(options, &problem, &summary); // 一声令下，召唤 Ceres 求解核心正式启动非线性最小二乘数智优化矩阵求解

    // 6. 优化成功，将精调后的高精度数学计算结果平滑写回系统数据库中
    if (summary.termination_type == ceres::CONVERGENCE || summary.termination_type == ceres::USER_SUCCESS)
    { // 如果求解器成功收敛或者被用户策略判定为成功
        for (auto kf : activeKFs)
        {                                                     // 位姿写回：再次轮询滑动窗口中的每一帧激活关键帧
            auto &data = mapKFIdToParameterBlock[kf->mId];    // 根据帧 ID 获取本地精调优化完毕后的最新 7 维双精度数组引用
            Eigen::Vector3d t_opt(data[0], data[1], data[2]); // 解析提取出数组前 3 项重新构造高精度的优化平移向量 t
            // 【核心修复】精确解析对应顺序。由于前面塞入时 data[3] 对应的是 q.w()
            Eigen::Quaterniond q_opt(data[3], data[4], data[5], data[6]); // Eigen 构造顺序规定依次为 (w, x, y, z)，此时内存排布完美吻合

            Eigen::Isometry3d Twc_opt = Eigen::Isometry3d::Identity(); // 声明高精度单位变换矩阵，用来存储最新的绝对变换关系
            Twc_opt.linear() = q_opt.toRotationMatrix();               // 将四元数旋转展开写回变换矩阵的 3x3 旋转矩阵部分
            Twc_opt.translation() = t_opt;                             // 将平移向量分量更新写入变换矩阵的第 4 列平移向量部分

            kf->SetPose(Twc_opt); // 调用带有并发线程安全互斥锁保护的接口，平滑无缝地重写刷回历史关键帧数据库中
        } // 关键帧位姿批量写回结束

        for (const auto &pair : mapMPIdToParameterBlock)
        {                                                       // 路标点写回：遍历所有本地提取优化的地图点路标坐标字典
            int mp_id = pair.first;                             // 取得对应的特征 ID
            const auto &data = pair.second;                     // 取得 Ceres 迭代收敛后的最新高精度 3 维世界绝对空间位置数组
            Eigen::Vector3d pos_opt(data[0], data[1], data[2]); // 利用精调数据封装回 Eigen 3D 绝对物理世界向量
            mapIdToMPRef[mp_id]->SetWorldPos(pos_opt);          // 通过原先备份持原有的路由指针，调用内置线程安全加锁接口 SetWorldPos，刷回全局实体小地图中
        } // 地图路标点位置批量写回完成
    } // 收敛判定写回分支完成
} // 局部滑动窗口图优化 LocalBundleAdjustment 函数主体圆满结束