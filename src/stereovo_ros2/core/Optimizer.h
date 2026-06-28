#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include <vector>
#include "core/Map.h"

/**
 * @brief 后端优化器类
 * 
 * 实现SLAM后端的局部Bundle Adjustment（BA）优化，包括：
 * - 滑动窗口机制
 * - 逆深度参数化
 * - 舒尔补边缘化
 * - Levenberg-Marquardt求解
 * - Huber鲁棒核
 * 
 * 纯Eigen实现，不依赖第三方优化库。
 */
class Optimizer
{
public:
    /// @brief 默认构造函数
    Optimizer() = default;

    /// @brief 析构函数
    ~Optimizer() = default;

    /**
     * @brief 局部滑动窗口BA优化（纯Eigen实现）
     * @param map 全局地图指针
     * @param windowSize 滑动窗口大小（关键帧数量）
     * 
     * 优化流程：
     * 1. 筛选滑动窗口内的激活关键帧
     * 2. 预处理地图点，选拔主导帧，初始化逆深度
     * 3. 预构建所有观测约束边
     * 4. 计算第一帧边缘化先验信息矩阵
     * 5. LM迭代求解（舒尔补 + Huber核）
     */
    static void LocalBundleAdjustment(std::shared_ptr<Map> map, int windowSize);
};

#endif // OPTIMIZER_H
