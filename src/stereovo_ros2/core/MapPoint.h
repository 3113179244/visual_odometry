#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <Eigen/Core>
#include <mutex>

class MapPoint
{
public:
    // 【修改】构造函数：传入该点在世界坐标系下的 3D 位置，以及它对应的全局特征 ID
    MapPoint(const Eigen::Vector3d &pos, int feature_id);
    ~MapPoint() = default;

    // 获取当前地图点的 3D 坐标（加锁保证安全）
    Eigen::Vector3d GetWorldPos();

    // 更新地图点的 3D 坐标（后端优化器优化完坐标后会调用它）
    void SetWorldPos(const Eigen::Vector3d &pos);

    // 记录观测计数（有多少帧看到了这个点，用来评估这个点稳不稳定）
    void AddObservation();
    int GetObservationCount();

    // 【新增】获取该地图点绑定的全局特征 ID
    int GetFeatureId();

private:
    Eigen::Vector3d mWorldPos; // 世界坐标系下的 3D 坐标 (X, Y, Z)
    int nObservations;         // 被观测到的次数
    std::mutex mMutexPos;      // 保护单个点坐标读写的互斥锁

    // 【新增】绑定的全局特征 ID，用于滑动窗口优化时与 KeyFrame 的观测进行匹配
    int mFeatureId;
};

#endif // MAPPOINT_H