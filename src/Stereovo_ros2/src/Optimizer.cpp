// src/Stereovo_ros2/src/Optimizer.cpp
#include "Optimizer.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Parameters.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <iostream>

Optimizer::Optimizer() {}

// ================= 1. 定义 Ceres 重投影误差代价仿函数 =================
struct SnavelyReprojectionError {
    // 传入观测到的2D像素坐标 (u, v) 以及相机内参
    SnavelyReprojectionError(double observed_x, double observed_y, double fx, double fy, double cx, double cy)
        : observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const camera_rotation,    // 3维轴角 (r_x, r_y, r_z) 表示 T_c_w 的旋转
                    const T* const camera_translation, // 3维平移 (t_x, t_y, t_z) 表示 T_c_w 的平移
                    const T* const point,              // 3维世界坐标点 (X, Y, Z)
                    T* residuals) const {
        
        T p[3];
        // 1. 将世界坐标点旋转到相机坐标系下：p = R * point
        ceres::AngleAxisRotatePoint(camera_rotation, point, p);

        // 2. 加上平移：p = R * point + t
        p[0] += camera_translation[0];
        p[1] += camera_translation[1];
        p[2] += camera_translation[2];

        // 3. 计算归一化相机坐标
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // 4. 利用内参投影到像素平面
        T predicted_x = T(fx) * xp + T(cx);
        T predicted_y = T(fy) * yp + T(cy);

        // 5. 计算残差（观测值 - 预测值）
        residuals[0] = T(observed_x) - predicted_x;
        residuals[1] = T(observed_y) - predicted_y;

        return true;
    }

    // 辅助创建 Ceres 代价函数
    static ceres::CostFunction* Create(const double observed_x, const double observed_y,
                                       double fx, double fy, double cx, double cy) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3, 3, 3>(
            new SnavelyReprojectionError(observed_x, observed_y, fx, fy, cx, cy)));
    }

    double observed_x;
    double observed_y;
    double fx, fy, cx, cy;
};

// ================= 2. 局部滑动窗口 BA 优化实现 =================
void Optimizer::LocalBundleAdjustment(std::shared_ptr<Map> pMap, int window_size) {
    // 获取全局所有关键帧
    std::vector<std::shared_ptr<KeyFrame>> vpAllKFs = pMap->GetAllKeyFrames();
    if (vpAllKFs.size() < 2) return; // 至少需要两帧才能优化位姿

    // 确定滑动窗口边界：保留最后的 window_size 帧作为优化对象
    int start_idx = std::max(0, static_cast<int>(vpAllKFs.size()) - window_size);
    
    std::vector<std::shared_ptr<KeyFrame>> vpLocalKFs;
    for (size_t i = start_idx; i < vpAllKFs.size(); ++i) {
        vpLocalKFs.push_back(vpAllKFs[i]);
    }

    // 提取系统内参（局部缓存或静态参数库）
    double fx = Parameters::fx;
    double fy = Parameters::fy;
    double cx = Parameters::cx;
    double cy = Parameters::cy;

    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0); // 引入鲁棒核函数拒绝错误匹配

    // 临时数据结构：存储位姿优化的参数块
    // KeyFrame ID -> [r_x, r_y, r_z, t_x, t_y, t_z] (大小为6的数组)
    std::map<unsigned long, std::vector<double>> mapCameraPoses;
    // MapPoint Feature ID -> [X, Y, Z] (大小为3的数组)
    std::map<int, std::vector<double>> mapPointPoses;

    // 记录涉及到的所有局部地图点
    std::map<int, std::shared_ptr<MapPoint>> mapIdToMP;

    // 遍历窗口内的局部关键帧，将其位姿转为 Ceres 参数格式
    for (size_t i = 0; i < vpLocalKFs.size(); ++i) {
        auto pKF = vpLocalKFs[i];
        // 转换变换 T_w_c 为误差公式所需的 T_c_w (世界系到相机系)
        Eigen::Isometry3d Tcw = pKF->GetPose().inverse();
        Eigen::Matrix3d R = Tcw.linear();
        Eigen::Vector3d t = Tcw.translation();

        // 旋转阵转轴角
        double angle_axis[3];
        double R_arr[9] = {R(0,0), R(0,1), R(0,2), R(1,0), R(1,1), R(1,2), R(2,0), R(2,1), R(2,2)};
        ceres::RotationMatrixToAngleAxis(R_arr, angle_axis);

        std::vector<double> camera_param = {angle_axis[0], angle_axis[1], angle_axis[2], t.x(), t.y(), t.z()};
        mapCameraPoses[pKF->mId] = camera_param;
    }

    // 获取地图中所有的地图点并进行筛选
    std::vector<std::shared_ptr<MapPoint>> vpAllMPs = pMap->GetAllMapPoints();
    
    // 遍历局部窗口帧，构建观测残差约束
    for (auto pKF : vpLocalKFs) {
        // 遍历该帧包含的所有 2D 特征观测
        for (const auto& obs : pKF->mmObservations) {
            int feat_id = obs.first;
            cv::Point2f uv = obs.second;

            // 寻找对应的全局三维地图点
            std::shared_ptr<MapPoint> pMP = nullptr;
            for (auto mp : vpAllMPs) {
                if (mp->GetFeatureId() == feat_id) {
                    pMP = mp;
                    break;
                }
            }
            if (!pMP) continue;

            // 如果该地图点还没加入 Ceres 数组，则转换并加入
            if (mapPointPoses.find(feat_id) == mapPointPoses.end()) {
                Eigen::Vector3d Pw = pMP->GetWorldPos();
                mapPointPoses[feat_id] = {Pw.x(), Pw.y(), Pw.z()};
                mapIdToMP[feat_id] = pMP;
            }

            // 获取指针
            double* camera_ptr = &mapCameraPoses[pKF->mId][0]; // 旋转在前3位
            double* point_ptr = &mapPointPoses[feat_id][0];

            // 添加 Ceres 残差块 (注意这里的 camera_ptr 拆分为 旋转块 和 平移块)
            ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(uv.x, uv.y, fx, fy, cx, cy);
            problem.AddResidualBlock(cost_function, loss_function, camera_ptr, camera_ptr + 3, point_ptr);
        }
    }

    // 约束设置：为防止规范化尺度飘移，滑动窗口的第 1 帧（通常是窗口内最老的关键帧）的位姿保持固定不优化
    if (!mapCameraPoses.empty()) {
        auto firstKFId = vpLocalKFs[0]->mId;
        problem.SetParameterBlockConstant(&mapCameraPoses[firstKFId][0]);     // 固定旋转
        problem.SetParameterBlockConstant(&mapCameraPoses[firstKFId][0] + 3); // 固定平移
    }

    // 配置 Ceres 求解器参数
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR; // 视觉 BA 推荐使用 Schur 消元法
    options.max_num_iterations = 8;                  // 保证前端实时性，迭代次数不宜过多
    options.gradient_tolerance = 1e-4;
    options.function_tolerance = 1e-4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // ================= 3. 将优化完后的精细化数据回写到系统 =================
    // 回写关键帧位姿
    for (auto pKF : vpLocalKFs) {
        double* camera_ptr = &mapCameraPoses[pKF->mId][0];
        double R_arr[9];
        ceres::AngleAxisToRotationMatrix(camera_ptr, R_arr);
        
        Eigen::Matrix3d R;
        R << R_arr[0], R_arr[3], R_arr[6],
             R_arr[1], R_arr[4], R_arr[7],
             R_arr[2], R_arr[5], R_arr[8]; // 注意行/列对应关系

        Eigen::Vector3d t(camera_ptr[3], camera_ptr[4], camera_ptr[5]);
        
        Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
        Tcw.linear() = R;
        Tcw.translation() = t;

        // 更新关键帧绝对位姿 T_w_c = T_c_w^-1
        pKF->mTwc = Tcw.inverse();
    }

    // 回写地图点 3D 坐标
    for (const auto& pair : mapPointPoses) {
        int feat_id = pair.first;
        const auto& Pw_arr = pair.second;
        Eigen::Vector3d Pw_updated(Pw_arr[0], Pw_arr[1], Pw_arr[2]);
        mapIdToMP[feat_id]->SetWorldPos(Pw_updated);
    }
}