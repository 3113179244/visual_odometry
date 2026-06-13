#ifndef LOOP_CLOSING_H
#define LOOP_CLOSING_H

#include <thread>
#include <atomic>
#include <memory>
#include <mutex>
#include <DBoW3/DBoW3.h>
#include "safe_queue.h"
#include "keyframe_selector.h"
#include "map.h" // 🌟 引入大地图类

typedef std::shared_ptr<Keyframe> KeyframePtr;

class LoopClosing
{
public:
    // 🌟 核心修正：构造函数对齐接收 2 个参数 (词袋路径 + 大地图指针)
    LoopClosing(const std::string &voc_file, std::shared_ptr<Map> pMap);
    ~LoopClosing();

    void Start();
    void Stop();

    void InsertKeyFrame(KeyframePtr pKF);

private:
    void Run();

    bool DetectLoop(KeyframePtr pKF, KeyframePtr &pMatchedKF);
    void CorrectLoop(KeyframePtr pCurrKF, KeyframePtr pMatchedKF);

    SafeQueue<KeyframePtr> mLoopQueue;
    std::thread *mptLoopClosing;
    std::atomic<bool> mbRunning;

    DBoW3::Vocabulary mVocabulary;
    DBoW3::Database mDatabase;

    std::vector<KeyframePtr> mHistoryKeyframes;

    // 🌟 核心新增：存储大地图指针，供回环修正时拉平轨迹位姿
    std::shared_ptr<Map> mpMap;

    std::mutex m_mutex;
};

#endif // LOOP_CLOSING_H