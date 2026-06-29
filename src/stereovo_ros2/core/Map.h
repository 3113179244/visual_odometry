#ifndef MAP_H
#define MAP_H

#include <vector>
#include <memory>
#include <mutex>
#include "core/MapPoint.h"
#include "core/KeyFrame.h"

class Map
{
public:
    Map();
    ~Map() = default;

    // --- 地图点管理接口 ---
    void AddMapPoint(std::shared_ptr<MapPoint> pMP);
    std::vector<std::shared_ptr<MapPoint>> GetAllMapPoints(); // 备用，返回副本
    int GetMapPointsSize();

    // --- 关键帧管理接口 ---
    void AddKeyFrame(std::shared_ptr<KeyFrame> pKF);
    std::vector<std::shared_ptr<KeyFrame>> GetAllKeyFrames();

    // 清空整个地图
    void Clear();

    // ===== 【新增】暴露可变容器引用和锁，供 Tracking 执行剔除 =====
    std::vector<std::shared_ptr<MapPoint>>& GetMapPoints() { return mspMapPoints; }
    std::vector<std::shared_ptr<KeyFrame>>& GetKeyFrames() { return mspKeyFrames; }
    std::mutex& GetMutex() { return mMutexMap; }

private:
    std::vector<std::shared_ptr<MapPoint>> mspMapPoints;
    std::vector<std::shared_ptr<KeyFrame>> mspKeyFrames;
    std::mutex mMutexMap;
};

#endif // MAP_H