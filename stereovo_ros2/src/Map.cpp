#include "Map.h"

Map::Map() {
    // 构造函数
}

void Map::AddMapPoint(std::shared_ptr<MapPoint> pMP) {
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.push_back(pMP);
}

std::vector<std::shared_ptr<MapPoint>> Map::GetAllMapPoints() {
    std::unique_lock<std::mutex> lock(mMutexMap);
    // 返回容器的副本，这样即使其他线程修改了原容器，当前读取线程也不会崩溃
    return mspMapPoints; 
}

int Map::GetMapPointsSize() {
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

void Map::Clear() {
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.clear();
}