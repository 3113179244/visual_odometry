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
}

void Map::CullRedundantKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    
    // 如果关键帧总数还很少，说明处于初始阶段，不需要剔除
    if (mspKeyFrames.size() <= 20)
        return;

    // 为了快速通过特征点ID查找地图点，先建立一个临时的全局地图点索引
    std::unordered_map<int, std::shared_ptr<MapPoint>> mapIdToMapPoint;
    for (const auto &mp : mspMapPoints)
    {
        mapIdToMapPoint[mp->GetFeatureId()] = mp;
    }

    // 迭代遍历较老的关键帧。由于最新的帧正在参与前端追踪和滑动窗口，我们只在前 1/3 的老帧里找冗余
    auto it = mspKeyFrames.begin();
    size_t max_search_bound = mspKeyFrames.size() / 3; 
    size_t processed_count = 0;

    while (it != mspKeyFrames.end() && processed_count < max_search_bound)
    {
        auto pKF = *it;
        
        int total_features = pKF->mmObservations.size();
        if (total_features == 0)
        {
            it = mspKeyFrames.erase(it);
            continue;
        }

        int redundant_features_count = 0;

        // 检查当前关键帧中的所有观测点
        for (const auto& obs : pKF->mmObservations)
        {
            int mapPointId = obs.first;
            auto itMp = mapIdToMapPoint.find(mapPointId);
            
            if (itMp != mapIdToMapPoint.end())
            {
                // If this point is seen by more than 3 keyframes, it means this KF is redundant for this point
                if (itMp->second->GetObservationCount() > 3)
                {
                    redundant_features_count++;
                }
            }
            else
            {
                // 如果这个点在全局地图里都找不到了（已经被CullMapPoints删了），也算作无用观测
                redundant_features_count++;
            }
        }

        // 如果 90% 的点对其他帧来说都是可观测的，说明当前帧提供了重复信息，属于冗余帧
        if (redundant_features_count > 0.90 * total_features)
        {
            it = mspKeyFrames.erase(it); // 从全局容器中剔除该关键帧
        }
        else
        {
            ++it;
            ++processed_count;
        }
    }
}