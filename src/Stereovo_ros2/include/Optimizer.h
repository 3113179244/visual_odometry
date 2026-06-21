// src/Stereovo_ros2/include/Optimizer.h
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include "Map.h"

class Optimizer {
public:
    Optimizer();
    ~Optimizer() = default;

    /**
     * @brief 滑动窗口局部局部 Bundle Adjustment 优化
     * @param pMap 全局地图指针（用于获取关键帧和地图点）
     * @param window_size 滑动窗口大小（如 10）
     */
    static void LocalBundleAdjustment(std::shared_ptr<Map> pMap, int window_size = 10);
};

#endif // OPTIMIZER_H