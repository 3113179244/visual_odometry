#pragma once
#include <thread>
#include <mutex>
#include <memory>
#include "map.h"
#include "slide_window.h"
#include "safe_queue.h"

class StereoVO;

class LocalMapping
{
public:
    LocalMapping(SlidingWindow *slide_win, std::shared_ptr<Map> pMap, StereoVO *vo);
    ~LocalMapping();

    void Start();
    void Stop();
    void InsertKeyFrame(KeyframePtr pKF);

private:
    void Run();
    void FuseMapPoints(KeyframePtr pKF);
    void CullLocalMap();

    SlidingWindow *mpSlideWindow;
    SafeQueue<KeyframePtr> mKeyFrameQueue; // 阻塞线程安全队列
    std::thread *mptLocalMapping;
    bool mbRunning;

    StereoVO *mpVO;
    std::shared_ptr<Map> mpMap;
};