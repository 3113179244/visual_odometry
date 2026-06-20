#include "MapPoint.h"

MapPoint::MapPoint(const Eigen::Vector3d& pos) 
    : mWorldPos(pos), nObservations(1) {
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