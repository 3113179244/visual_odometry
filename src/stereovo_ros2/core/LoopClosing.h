#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <DBoW3/DBoW3.h>
#include <sophus/se3.hpp>

#include "core/KeyFrame.h"
#include "core/Map.h"

class Tracking; // 前向声明

struct ConsistentGroup {
    std::set<unsigned long> sKFIds; // 候选帧近邻圈的 KeyFrame ID 集合
    int nConsistencyCount;          // 连续命中计数
};

class LoopClosing
{
public:
    LoopClosing(std::shared_ptr<Map> pMap, const std::string &vocPath);
    ~LoopClosing();

    void SetTracking(Tracking *pTracking);
    void InsertKeyFrame(std::shared_ptr<KeyFrame> pKF);

private:
    void Run();
    bool DetectLoop(std::shared_ptr<KeyFrame> pCurrentKF, std::shared_ptr<KeyFrame> &pMatchedKF);
    bool CorrectLoop(std::shared_ptr<KeyFrame> pCurrentKF, std::shared_ptr<KeyFrame> pMatchedKF);

    bool ComputeRelativePosePnP(std::shared_ptr<KeyFrame> pCurrentKF,
                                std::shared_ptr<KeyFrame> pMatchedKF,
                                const std::vector<cv::DMatch> &matches,
                                Sophus::SE3d &T_match_curr_meas);

    bool CheckConsistency(std::shared_ptr<KeyFrame> pCandidateKF);

    void FuseMapPoints(std::shared_ptr<KeyFrame> pCurrentKF,
                       std::shared_ptr<KeyFrame> pMatchedKF,
                       const std::vector<cv::DMatch> &matches);

    // 💡【新增】：PGO 优化后同步更新地图点 3D 坐标
    void UpdateMapPointsAfterPGO(const std::map<unsigned long, Eigen::Isometry3d> &oldPoses);

private:
    std::shared_ptr<Map> mpMap;
    Tracking *mpTracking = nullptr;
    std::shared_ptr<DBoW3::Vocabulary> mpVocabulary;
    std::unique_ptr<DBoW3::Database> mpDatabase;

    // 💡【修复】：替换 vector 为 map，绑定 DBoW EntryId -> KeyFrame，防止关键帧剔除后索引错位
    std::unordered_map<DBoW3::EntryId, std::shared_ptr<KeyFrame>> mmKeyFrameDB;

    std::thread mLoopThread;
    bool mIsRunning;
    std::mutex mMutexQueue;
    std::condition_variable mCondQueue;
    std::queue<std::shared_ptr<KeyFrame>> mKeyFrameQueue;

    // 闭环边记录
    std::vector<std::pair<unsigned long, unsigned long>> mvLoopEdges;
    std::map<std::pair<unsigned long, unsigned long>, Sophus::SE3d> mmLoopRelativePoses;

    std::vector<ConsistentGroup> mvConsistentGroups;
};

#endif // LOOPCLOSING_H