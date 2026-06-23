#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "NvInfer.h"

class YoloSegDetector {
public:
    // 构造函数：传入 engine 绝对路径和模型规模（如 "n"）
    YoloSegDetector(const std::string& engine_path, const std::string& sub_type);
    ~YoloSegDetector();

    // 核心外部接口：输入当前帧彩图或灰度图，返回动态物体的二值 Mask (CV_8UC1)
    // 在该 Mask 中，属于动态高危区域（人、车等）的像素值为 255，静态背景为 0
    cv::Mat GetDynamicMask(const cv::Mat& img);

private:
    // 依据 COCO 数据集定义的潜在动态高危物体类别 ID
    // 0:人, 1:自行车, 2:汽车, 3:摩托车, 5:公交车, 7:卡车
    std::set<int> mDynamicClassIds = {0, 1, 2, 3, 5, 7};

    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream;

    float* device_buffers[3]; // GPU 缓冲区
    float* output_buffer_host = nullptr;
    float* output_seg_buffer_host = nullptr;
    int model_bboxes;
};