#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <memory>
#include <vector>
#include <map>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <DBoW3/DBoW3.h>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>

#include "core/KeyFrame.h"
#include "core/Map.h"

class LoopClosure
{
public:
    LoopClosure(std::shared_ptr<Map> pMap, const std::string &vocPath);
    ~LoopClosure();

    void PushKeyFrame(std::shared_ptr<KeyFrame> pKF);

private:
    void LoopClosureThread();
    void ExtractORBAndTriangulate(std::shared_ptr<KeyFrame> pKF);
    bool DetectLoop(std::shared_ptr<KeyFrame> pKF, std::shared_ptr<KeyFrame> &matchedKF, Eigen::Isometry3d &T_cur_match);
    void RunGlobalPoseGraphOptimization();

private:
    std::shared_ptr<Map> mpMap;
    
    std::queue<std::shared_ptr<KeyFrame>> mKeyFrameQueue;
    std::mutex mMutexQueue;
    std::condition_variable mCondQueue;

    std::thread mLoopThread;
    bool mIsRunning;

    DBoW3::Vocabulary mVocabulary;
    DBoW3::Database mDatabase;

    std::map<unsigned long, std::shared_ptr<KeyFrame>> mmKeyFramesDB;
    std::mutex mMutexDB;
};

#endif // LOOPCLOSING_H