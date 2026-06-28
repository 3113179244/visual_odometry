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

    static void LocalBundleAdjustment(std::shared_ptr<Map> map, int windowSize);
};

#endif // OPTIMIZER_H