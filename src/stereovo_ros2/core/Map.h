#ifndef MAP_H
#define MAP_H

#include <vector>
#include <memory>
#include <mutex>
#include "core/MapPoint.h"
#include "core/KeyFrame.h" // 引入新写好的关键帧类

class Map
{
public:
    Map();
    ~Map() = default;

    // --- 地图点管理接口 ---
    void AddMapPoint(std::shared_ptr<MapPoint> pMP);
    std::vector<std::shared_ptr<MapPoint>> GetAllMapPoints();
    int GetMapPointsSize();

    // --- 关键帧管理接口（新增） ---
    // 向地图中插入一帧新的关键帧
    void AddKeyFrame(std::shared_ptr<KeyFrame> pKF);
    // 获取地图中所有的关键帧
    std::vector<std::shared_ptr<KeyFrame>> GetAllKeyFrames();

    // 清空整个地图
    void Clear();

private:
    std::vector<std::shared_ptr<MapPoint>> mspMapPoints; // 存储全局所有地图点的容器
    std::vector<std::shared_ptr<KeyFrame>> mspKeyFrames; // 新增：存储全局所有关键帧的历史数据库
    std::mutex mMutexMap;                                // 地图数据全局互斥锁，保证多线程安全
};

#endif // MAP_H