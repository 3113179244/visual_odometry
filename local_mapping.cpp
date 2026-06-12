#include "local_mapping.h"
#include <iostream>

LocalMapping::LocalMapping(SlidingWindow* slide_win) 
    : mpSlideWindow(slide_win), mptLocalMapping(nullptr), mbRunning(false) {}

LocalMapping::~LocalMapping() {
    Stop();
}

void LocalMapping::Start() {
    mbRunning = true;
    mptLocalMapping = new std::thread(&LocalMapping::Run, this);
    std::cout << "[LocalMapping] 后端优化线程已启动。" << std::endl;
}

void LocalMapping::Stop() {
    if (mbRunning) {
        mbRunning = false;
        mKeyFrameQueue.shutdown(); // 唤醒可能阻塞在 pop() 的线程
        if (mptLocalMapping && mptLocalMapping->joinable()) {
            mptLocalMapping->join();
        }
        delete mptLocalMapping;
        mptLocalMapping = nullptr;
        std::cout << "[LocalMapping] 后端优化线程已安全停止。" << std::endl;
    }
}

void LocalMapping::InsertKeyFrame(KeyframePtr pKF) {
    mKeyFrameQueue.push(pKF);
}

void LocalMapping::Run() {
    while (mbRunning) {
        KeyframePtr currKF;
        // 阻塞等待，直到前端发来新的关键帧，或者系统收到 shutdown 信号
        if (mKeyFrameQueue.pop(currKF)) {
            if (!currKF) continue;
            
            // ==========================================
            // 执行极其耗时的后端操作，完全不会卡住前端！
            // ==========================================
            std::cout << "\n[LocalMapping] 收到前端发来的新关键帧 ID: " << currKF->id << "，开始执行局部 BA..." << std::endl;
            
            // 将关键帧加入滑动窗口并触发优化
            mpSlideWindow->addKeyframe(*currKF);
            
            std::cout << "[LocalMapping] 关键帧 ID: " << currKF->id << " 优化完成。\n" << std::endl;
        }
    }
}