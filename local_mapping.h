#ifndef LOCAL_MAPPING_H
#define LOCAL_MAPPING_H

#include <thread>
#include <atomic>
#include <memory>
#include <vector>
#include <shared_mutex>
#include "safe_queue.h"
#include "slide_window.h"
#include "keyframe_selector.h" // 包含 Keyframe 定义

// 定义智能指针类型
typedef std::shared_ptr<Keyframe> KeyframePtr;

// 显式声明来自 stereo_vo.h 的结构体
struct MapPoint;
class StereoVO;
class LocalMapping {
public:
    // 【核心修复】：确保头文件声明支持 3 个参数
    LocalMapping(SlidingWindow* slide_win, 
                 std::vector<MapPoint>& local_map, 
                 std::shared_mutex& map_mutex,
                 StereoVO* vo);
                 
    ~LocalMapping();

    // 启动后台线程
    void Start();

    // 停止后台线程（系统退出时调用）
    void Stop();

    // 暴露给前端的接口：插入新的关键帧
    void InsertKeyFrame(KeyframePtr pKF);

private:
    StereoVO* mpVO;
    // 后台线程的主循环函数
    void Run();

    // 地图点融合与剔除的核心函数
    void FuseMapPoints(KeyframePtr pKF);
    void CullLocalMap();

    SlidingWindow* mpSlideWindow;           // 真正的优化器
    SafeQueue<KeyframePtr> mKeyFrameQueue;  // 关键帧接收队列
    
    std::thread* mptLocalMapping;           // 线程对象
    std::atomic<bool> mbRunning;            // 线程运行标志

    // 局部地图的引用与读写锁
    std::vector<MapPoint>& mLocalMap;
    std::shared_mutex& mMapMutex;
};

#endif // LOCAL_MAPPING_H