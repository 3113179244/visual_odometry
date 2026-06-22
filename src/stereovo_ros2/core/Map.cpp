#include "Map.h"

Map::Map()
{
    // 构造函数
}

void Map::AddMapPoint(std::shared_ptr<MapPoint> pMP)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.push_back(pMP);
}

std::vector<std::shared_ptr<MapPoint>> Map::GetAllMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mspMapPoints;
}

int Map::GetMapPointsSize()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

// --- 以下为新增关键帧数据结构实现 ---
void Map::AddKeyFrame(std::shared_ptr<KeyFrame> pKF)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspKeyFrames.push_back(pKF);
}

std::vector<std::shared_ptr<KeyFrame>> Map::GetAllKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mspKeyFrames;
}

void Map::Clear()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.clear();
    mspKeyFrames.clear(); // 同时清空关键帧
}