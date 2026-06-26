#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <Eigen/Core>
#include <mutex>

class MapPoint
{
public:
    // 构造函数：传入该点在世界坐标系下的 3D 位置，以及它对应的全局特征 ID
    MapPoint(const Eigen::Vector3d &pos, int feature_id);
    ~MapPoint() = default;

    // 获取当前地图点的 3D 坐标（加锁保证安全）
    Eigen::Vector3d GetWorldPos();
    // 更新地图点的 3D 坐标（后端优化器优化完坐标后会调用它）
    void SetWorldPos(const Eigen::Vector3d &pos);

    // 观测计数
    void AddObservation();
    int GetObservationCount();

    // 获取该地图点绑定的全局特征 ID
    int GetFeatureId();

    // ===== 新增：地图点质量相关 =====
    // 连续外点计数
    void MarkAsOutlier() { mnConsecutiveOutlier++; }
    void MarkAsInlier() { mnConsecutiveOutlier = 0; }
    int GetConsecutiveOutlier() { return mnConsecutiveOutlier; }

    // 坏点标记
    bool IsBad() { return mbBad; }
    void SetBad() { mbBad = true; }

private:
    Eigen::Vector3d mWorldPos;   // 世界坐标系下的 3D 坐标 (X, Y, Z)
    int nObservations;           // 被观测到的次数
    std::mutex mMutexPos;        // 保护单个点坐标读写的互斥锁
    int mFeatureId;              // 绑定的全局特征 ID

    // ===== 新增字段 =====
    int mnConsecutiveOutlier = 0;   // 连续外点次数
    bool mbBad = false;             // 是否是坏点
};

#endif // MAPPOINT_H
