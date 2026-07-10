#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include <vector>
#include "core/Map.h"

/**
 * @brief 后端图优化/光束法平差 (Bundle Adjustment, BA) 核心类
 * 
 * 该类主要负责在独立的后端线程中，对前端收集到的关键帧位姿 (Poses) 和地图点 3D 坐标 (MapPoints)
 * 进行局部非线性最小二乘优化，消除累积漂移，保证轨迹与地图的局部一致性。
 */
class Optimizer
{
public:
    // 默认构造与析构函数
    Optimizer() = default;
    ~Optimizer() = default;

    /**
     * @brief 局部光束法平差 (Local Bundle Adjustment)
     * @param map        全局地图对象的指针，用于获取关键帧和地图点
     * @param windowSize 滑动窗口的大小（即参与优化的最大关键帧数量）
     * 
     * @note 该函数被定义为静态函数，方便前端或独立的后端控制流直接调用。
     */
    static void LocalBundleAdjustment(std::shared_ptr<Map> map, int windowSize);
};

#endif // OPTIMIZER_H