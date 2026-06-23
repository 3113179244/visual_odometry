#include "yolo_seg_detector.h"
#include <fstream>
#include <iostream>

// 注入原项目所需的 TensorRT 全局日志记录器与核心配置常量
#include "logging.h"
#include "cuda_utils.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

Logger gLogger; // 必须在此定义全局 Logger，供给 model.cpp 编译链使用

const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1; //
const static int kOutputSegSize = 32 * (kInputH / 4) * (kInputW / 4); //

// 完美移植自原项目的辅助分割尺寸缩放函数
static cv::Rect get_downscale_rect(float bbox[4], float scale) { //
    float left = bbox[0]; float top = bbox[1];
    float right = bbox[0] + bbox[2]; float bottom = bbox[1] + bbox[3];
    left = left < 0 ? 0 : left; top = top < 0 ? 0 : top;
    right = right > kInputW ? kInputW : right; bottom = bottom > kInputH ? kInputH : bottom;
    left /= scale; top /= scale; right /= scale; bottom /= scale;
    return cv::Rect(int(left), int(top), int(right - left), int(bottom - top));
}

// 完美移植自原项目的分割掩码恢复函数
static std::vector<cv::Mat> local_process_mask(const float* proto, int proto_size, std::vector<Detection>& dets) { //
    std::vector<cv::Mat> masks;
    for (size_t i = 0; i < dets.size(); i++) {
        cv::Mat mask_mat = cv::Mat::zeros(kInputH / 4, kInputW / 4, CV_32FC1);
        auto r = get_downscale_rect(dets[i].bbox, 4);
        for (int x = r.x; x < r.x + r.width; x++) {
            for (int y = r.y; y < r.y + r.height; y++) {
                float e = 0.0f;
                for (int j = 0; j < 32; j++) {
                    e += dets[i].mask[j] * proto[j * proto_size / 32 + y * mask_mat.cols + x];
                }
                e = 1.0f / (1.0f + expf(-e));
                mask_mat.at<float>(y, x) = e;
            }
        }
        cv::resize(mask_mat, mask_mat, cv::Size(kInputW, kInputH));
        masks.push_back(mask_mat);
    }
    return masks;
}

YoloSegDetector::YoloSegDetector(const std::string& engine_path, const std::string& sub_type) {
    // 1. 初始化独立 CUDA 设备设备流
    cudaSetDevice(kGpuId); //
    
    // 2. 反序列化模型引擎
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << ">>> [YOLO包装类] 严重错误：读不到 Engine 文件: " << engine_path << std::endl;
        return;
    }
    size_t size = 0; file.seekg(0, file.end); size = file.tellg(); file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    file.read(serialized_engine, size); file.close();

    runtime = nvinfer1::createInferRuntime(gLogger); //
    engine = runtime->deserializeCudaEngine(serialized_engine, size); //
    context = engine->createExecutionContext(); //
    delete[] serialized_engine;

    CUDA_CHECK(cudaStreamCreate(&stream)); //
    cuda_preprocess_init(kMaxInputImageSize); // 初始化原项目的预处理硬件存储

    auto out_dims = engine->getBindingDimensions(1); //
    model_bboxes = out_dims.d[0]; //

    // 3. 仿照 prepare_buffer 逻辑开辟 GPU 显存与 CPU 内存缓冲区
    CUDA_CHECK(cudaMalloc((void**)&device_buffers[0], 1 * 3 * kInputH * kInputW * sizeof(float))); //
    CUDA_CHECK(cudaMalloc((void**)&device_buffers[1], 1 * kOutputSize * sizeof(float))); //
    CUDA_CHECK(cudaMalloc((void**)&device_buffers[2], 1 * kOutputSegSize * sizeof(float))); //

    output_buffer_host = new float[1 * kOutputSize]; //
    output_seg_buffer_host = new float[1 * kOutputSegSize]; //
}

YoloSegDetector::~YoloSegDetector() {
    // 安全释放所有 GPU/CPU 推理资源，杜绝 SLAM 长跑时的内存泄漏
    cudaStreamDestroy(stream); //
    CUDA_CHECK(cudaFree(device_buffers[0])); //
    CUDA_CHECK(cudaFree(device_buffers[1])); //
    CUDA_CHECK(cudaFree(device_buffers[2])); //
    delete[] output_buffer_host; //
    delete[] output_seg_buffer_host; //
    cuda_preprocess_destroy(); //
    if (context) context->destroy(); //
    if (engine) engine->destroy(); //
    if (runtime) runtime->destroy(); //
}

cv::Mat YoloSegDetector::GetDynamicMask(const cv::Mat& img) {
    // 统一转换为原项目推理所需的 BGR 彩色图
    cv::Mat cv_img = img.clone();
    if (cv_img.channels() == 1) {
        cv::cvtColor(cv_img, cv_img, cv::COLOR_GRAY2BGR);
    }

    // 1. 调用原硬件加速预处理算子，将图像缩放并搬运至 GPU
    std::vector<cv::Mat> img_batch = {cv_img};
    cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream); //

    // 2. 触发 TensorRT 硬件级同步异步推理
    context->enqueue(1, (void**)device_buffers, stream, nullptr); //
    
    // 3. 将推理结果无损拷回 CPU
    CUDA_CHECK(cudaMemcpyAsync(output_buffer_host, device_buffers[1], 1 * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream)); //
    CUDA_CHECK(cudaMemcpyAsync(output_seg_buffer_host, device_buffers[2], 1 * kOutputSegSize * sizeof(float), cudaMemcpyDeviceToHost, stream)); //
    CUDA_CHECK(cudaStreamSynchronize(stream)); //

    // 4. 运行原生 NMS 过滤重叠边界框
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, output_buffer_host, 1, kOutputSize, kConfThresh, kNmsThresh); //

    // 5. 核心动态过滤：剥离并融合动态物体的 Mask 轮廓
    cv::Mat total_dynamic_mask = cv::Mat::zeros(cv_img.rows, cv_img.cols, CV_8UC1);
    auto& res = res_batch[0];

    if (!res.empty()) {
        // 解算获取整帧的原尺寸 Mask 矩阵队列
        auto masks = local_process_mask(output_seg_buffer_host, kOutputSegSize, res); //
        
        for (size_t i = 0; i < res.size(); i++) {
            // 拦截器：仅当物体的类别属于高危动态目标时，才提取它的轮廓进行点云清洗
            if (mDynamicClassIds.count(res[i].class_id)) {
                cv::Mat mask_mat = masks[i];
                // 原项目 process_mask 输出的是 0.0~1.0 的浮点概率，通过阈值将其二值化为 0 和 255
                cv::Mat mask_binary;
                cv::threshold(mask_mat, mask_binary, 0.5, 255, cv::THRESH_BINARY);
                mask_binary.convertTo(mask_binary, CV_8UC1);

                // 依据原项目 get_rect 方法，精准把剪裁好的 Mask 缩放到当前图像的物理像素 ROI 区域中
                cv::Rect roi = get_rect(cv_img, res[i].bbox); //
                cv::Mat roi_mask = mask_binary(roi);

                // 将该物体的轮廓合并入总动态结界图中
                cv::bitwise_or(total_dynamic_mask(roi), roi_mask, total_dynamic_mask(roi));
            }
        }
    }
    return total_dynamic_mask;
}