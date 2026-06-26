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

void Map::CullMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMap);

    // ===== 筛选参数 =====
    const int MIN_OBSERVATIONS = 2;        // 至少被观测2次才算靠谱
    const int MAX_CONSECUTIVE_OUTLIER = 3; // 连续3帧外点就删掉

    int cnt_removed_obs = 0;     // 观测次数不足被删
    int cnt_removed_outlier = 0; // 连续外点被删
    mnCullCounter++;
    if (mnCullCounter < 3)
        return;
    mnCullCounter = 0;
    // 遍历删除坏点
    auto it = mspMapPoints.begin();
    while (it != mspMapPoints.end())
    {
        auto pMP = *it; // vector的迭代器直接解引用就是shared_ptr

        if (pMP->IsBad())
        {
            it = mspMapPoints.erase(it);
            continue;
        }

        bool shouldRemove = false;

        // 规则1：观测次数太少
        if (pMP->GetObservationCount() < MIN_OBSERVATIONS)
        {
            shouldRemove = true;
            cnt_removed_obs++;
        }
        // 规则2：连续多帧外点
        else if (pMP->GetConsecutiveOutlier() >= MAX_CONSECUTIVE_OUTLIER)
        {
            shouldRemove = true;
            cnt_removed_outlier++;
        }

        if (shouldRemove)
        {
            pMP->SetBad();
            it = mspMapPoints.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // 打印统计
    std::cout << "[DEBUG-Map] === 地图点筛选 ===" << std::endl;
    std::cout << "[DEBUG-Map] 删除: " << cnt_removed_obs + cnt_removed_outlier
              << " 个 | 剩余: " << mspMapPoints.size() << " 个" << std::endl;
    std::cout << "[DEBUG-Map]   - 观测次数不足(<" << MIN_OBSERVATIONS << "次): "
              << cnt_removed_obs << std::endl;
    std::cout << "[DEBUG-Map]   - 连续外点(>=" << MAX_CONSECUTIVE_OUTLIER << "帧): "
              << cnt_removed_outlier << std::endl;
    std::cout << "[DEBUG-Map] ======================" << std::endl;
}
