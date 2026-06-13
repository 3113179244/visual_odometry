#include "map.h"
#include <algorithm>

// ----- 全局地图管理接口 -----

/**
 * @brief 将前端或后端打包好的关键帧指针塞入全局大地图
 */
void Map::AddKeyframe(KeyframePtr pKF)
{
    std::unique_lock<std::shared_mutex> lock(mMutexMap);
    vpSubKeyframes.push_back(pKF);
}

/**
 * @brief 将新三角化出来的 3D 地图点指针塞入全局大地图
 */
void Map::AddMapPoint(MapPointPtr pMP)
{
    std::unique_lock<std::shared_mutex> lock(mMutexMap);
    vpSubMapPoints.push_back(pMP);
}

/**
 * @brief 线程安全地获取全局所有的关键帧（用于回环检测或全局轨迹绘制）
 */
std::vector<KeyframePtr> Map::GetAllKeyframes()
{
    std::shared_lock<std::shared_mutex> lock(mMutexMap);
    return vpSubKeyframes;
}

/**
 * @brief 线程安全地获取全局所有的 3D 地图点
 */
std::vector<MapPointPtr> Map::GetAllMapPoints()
{
    std::shared_lock<std::shared_mutex> lock(mMutexMap);
    return vpSubMapPoints;
}

/**
 * @brief 获取全局大地图中三维点的绝对总数
 */
size_t Map::GetGlobalMapPointsSize()
{
    std::shared_lock<std::shared_mutex> lock(mMutexMap);
    return vpSubMapPoints.size();
}

// ----- 局部地图管理接口（面向后端局部滑窗 BA） -----

/**
 * @brief 根据当前激活帧，自适应搜索并刷新局部激活的“关键帧池”与“3D 锚点池”
 * @param pCurrentKF 当前前端刚确立并推入的关键帧指针
 * @param max_local_kfs 局部地图锁定的最大历史关键帧数量（滑动窗口大小）
 */
void Map::UpdateLocalMap(KeyframePtr pCurrentKF, int max_local_kfs)
{
    std::unique_lock<std::shared_mutex> lock(mMutexMap);

    // 1. 甄选局部关键帧：采用时间线反向追踪近邻法（最直观高效）
    vpLocalKeyframes.clear();
    int count = 0;
    for (auto ri = vpSubKeyframes.rbegin(); ri != vpSubKeyframes.rend(); ++ri)
    {
        if (count >= max_local_kfs)
            break;
        vpLocalKeyframes.push_back(*ri);
        count++;
    }
    // 恢复时间轴正向顺序
    std::reverse(vpLocalKeyframes.begin(), vpLocalKeyframes.end());

    // 2. 收集局部地图点：只要能被上面任何一个“局部滑窗关键帧”观测到的点，都算作活跃的局部地图点
    vpLocalMapPoints.clear();
    std::set<MapPointPtr> spLocalMPCandidates; // 利用 set 的天然特性进行指针去重

    for (const auto &pKF : vpLocalKeyframes)
    {
        for (int mp_idx : pKF->map_point_indices)
        {
            // 安全边界防越界校验
            if (mp_idx >= 0 && static_cast<size_t>(mp_idx) < vpSubMapPoints.size())
            {
                MapPointPtr pMP = vpSubMapPoints[mp_idx];
                // 只有非空且不是坏点的地图点，才有资格进入局部 BA 优化池
                if (pMP && !pMP->is_bad)
                {
                    spLocalMPCandidates.insert(pMP);
                }
            }
        }
    }

    // 将去重筛选后的黄金局部点成批倒入局部地图容器中
    vpLocalMapPoints.assign(spLocalMPCandidates.begin(), spLocalMPCandidates.end());
}

/**
 * @brief 共享读锁：获取当前被锁定的滑窗局部关键帧
 */
std::vector<KeyframePtr> Map::GetLocalKeyframes()
{
    std::shared_lock<std::shared_mutex> lock(mMutexMap);
    return vpLocalKeyframes;
}

/**
 * @brief 共享读锁：获取当前与滑窗关联的局部活跃地图点（给 FuseMapPoints 和滑窗 BA 使用）
 */
std::vector<MapPointPtr> Map::GetLocalMapPoints()
{
    std::shared_lock<std::shared_mutex> lock(mMutexMap);
    return vpLocalMapPoints;
}

// ----- 地图瘦身与垃圾清理接口 -----

/**
 * @brief 全局物理死点清理：将大地图中被后端标记为 is_bad 的点彻底从内存中抹除
 * @note 定期在 LocalMapping 的 Run() 循环中低频调用，防止系统长时间运行内存爆炸
 */
void Map::CleanBadMechanisms()
{
    std::unique_lock<std::shared_mutex> lock(mMutexMap);

    // 使用 C++ 标准库的 erase-remove_if 惯用法进行物理内存擦除
    vpSubMapPoints.erase(
        std::remove_if(vpSubMapPoints.begin(), vpSubMapPoints.end(),
                       [](const MapPointPtr &pMP)
                       {
                           return !pMP || pMP->is_bad;
                       }),
        vpSubMapPoints.end());
}