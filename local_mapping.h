#ifndef LOCAL_MAPPING_H
#define LOCAL_MAPPING_H

#include <thread>
#include <atomic>
#include <memory>
#include "safe_queue.h"
#include "slide_window.h"
#include "keyframe_selector.h" // 包含 Keyframe 定义

// 定义智能指针类型，多线程传递效率更高
typedef std::shared_ptr<Keyframe> KeyframePtr;

class LocalMapping {
public:
    // 传入滑动窗口优化器
    LocalMapping(SlidingWindow* slide_win);
    ~LocalMapping();

    // 启动后台线程
    void Start();

    // 停止后台线程（系统退出时调用）
    void Stop();

    // 暴露给前端的接口：插入新的关键帧
    void InsertKeyFrame(KeyframePtr pKF);

private:
    // 后台线程的主循环函数
    void Run();

    SlidingWindow* mpSlideWindow;           // 真正的优化器
    SafeQueue<KeyframePtr> mKeyFrameQueue;  // 关键帧接收队列
    
    std::thread* mptLocalMapping;           // 线程对象
    std::atomic<bool> mbRunning;            // 线程运行标志
};

#endif // LOCAL_MAPPING_H