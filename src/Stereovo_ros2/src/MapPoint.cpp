#include "MapPoint.h"

// 【修改】在初始化列表中增加 mFeatureId(feature_id)
MapPoint::MapPoint(const Eigen::Vector3d& pos, int feature_id) 
    : mWorldPos(pos), nObservations(1), mFeatureId(feature_id) {
    // 默认刚创建时被观测次数为 1
}

Eigen::Vector3d MapPoint::GetWorldPos() {
    std::unique_lock<std::mutex> lock(mMutexPos);
    return mWorldPos;
}

void MapPoint::SetWorldPos(const Eigen::Vector3d& pos) {
    std::unique_lock<std::mutex> lock(mMutexPos);
    mWorldPos = pos;
}

void MapPoint::AddObservation() {
    std::unique_lock<std::mutex> lock(mMutexPos);
    nObservations++;
}

int MapPoint::GetObservationCount() {
    std::unique_lock<std::mutex> lock(mMutexPos);
    return nObservations;
}

// 【新增】获取特征 ID 的具体实现（由于特征 ID 在构造后不可变，通常不需要加锁）
int MapPoint::GetFeatureId() {
    return mFeatureId;
}