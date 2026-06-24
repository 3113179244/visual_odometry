#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <memory>
#include <vector>
#include "core/Map.h"

class Optimizer
{
public:
    Optimizer() = default;
    ~Optimizer() = default;
    /**
     * @brief 局部滑动窗口 BA 优化（纯 Eigen 实现）
     * @param pMap 全局地图指针
     * @param windowSize 滑动窗口大小
     */
    static void LocalBundleAdjustment(std::shared_ptr<Map> pMap, int windowSize);
};
#endif // OPTIMIZER_H
