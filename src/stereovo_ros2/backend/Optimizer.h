#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include <vector>
#include <map>
#include "core/Map.h"
#include <sophus/se3.hpp>
class Optimizer
{
public:
    Optimizer() = default;
    ~Optimizer() = default;

    static void LocalBundleAdjustment(std::shared_ptr<Map> map, int windowSize);

    // 位姿图优化接口：优化全局关键帧位姿
    static void PoseGraphOptimization(
        std::shared_ptr<Map> map,
        const std::vector<std::pair<unsigned long, unsigned long>> &loopEdges,
        const std::map<std::pair<unsigned long, unsigned long>, Sophus::SE3d> &loopRelativePoses);
};

#endif // OPTIMIZER_H