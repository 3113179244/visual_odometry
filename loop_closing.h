#ifndef LOOP_CLOSING_H
#define LOOP_CLOSING_H

#include <thread>
#include <atomic>
#include <memory>
#include <DBoW3/DBoW3.h> // 需要安装 DBoW3
#include "safe_queue.h"
#include "keyframe_selector.h"

typedef std::shared_ptr<Keyframe> KeyframePtr;

class LoopClosing {
public:
    LoopClosing(const std::string& voc_file);
    ~LoopClosing();

    void Start();
    void Stop();
    
    // 供 LocalMapping 调用的接口
    void InsertKeyFrame(KeyframePtr pKF);

private:
    void Run();
    
    // 回环检测核心步骤
    bool DetectLoop(KeyframePtr pKF, KeyframePtr& pMatchedKF);
    void CorrectLoop(KeyframePtr pCurrKF, KeyframePtr pMatchedKF);

    SafeQueue<KeyframePtr> mLoopQueue;
    std::thread* mptLoopClosing;
    std::atomic<bool> mbRunning;

    // DBoW3 核心成员
    DBoW3::Vocabulary mVocabulary;
    DBoW3::Database mDatabase;
    
    // 存储历史所有关键帧，用于回环匹配时提取数据
    std::vector<KeyframePtr> mHistoryKeyframes;
};

#endif // LOOP_CLOSING_H