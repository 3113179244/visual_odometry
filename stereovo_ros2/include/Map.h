#ifndef MAP_H
#define MAP_H

#include <vector>
#include <memory>
#include <mutex>
#include "MapPoint.h"

class Map {
public:
    Map();
    ~Map() = default;

    // 向地图中添加一个新的地图点
    void AddMapPoint(std::shared_ptr<MapPoint> pMP);

    // 获取地图中当前所有的地图点（加锁保护，防止多线程冲突）
    std::vector<std::shared_ptr<MapPoint>> GetAllMapPoints();

    // 获取当前地图中点的总数量
    int GetMapPointsSize();

    // 清空地图
    void Clear();

private:
    std::vector<std::shared_ptr<MapPoint>> mspMapPoints; // 存储全局所有地图点的容器
    std::mutex mMutexMap;                                // 地图数据互斥锁，保证多线程安全
};

#endif // MAP_H