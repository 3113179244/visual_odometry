#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include "map.h"
#include "stereo_vo.h"
#include "local_mapping.h"
#include "keyframe_selector.h"
#include "slide_window.h"
#include "tracking.h"

class System
{
public:
    System(const std::string &calib_file);
    ~System();

    // 启动系统
    void Run(const std::string &path_left, const std::string &path_right, const std::string &traj_out_path);
    void Stop();

private:
    // ==== 核心独立模块 ====
    std::unique_ptr<StereoVO> mpVO;
    std::shared_ptr<Map> mpMap;
    std::unique_ptr<SlidingWindow> mpSlideWindow;
    std::unique_ptr<LocalMapping> mpLocalMapper;
    std::unique_ptr<KeyframeSelector> mpSelector;

    // ==== 前端追踪器 ====
    std::unique_ptr<Tracking> mpTracker;

    std::atomic<bool> mbRunning;

    // ====== DEBUG CODE START ======
    // 用于控制单步执行的原子布尔变量，初始为 false (暂停状态)
    std::atomic<bool> mbStepToNextFrame{false};
    // ====== DEBUG CODE END ======
};