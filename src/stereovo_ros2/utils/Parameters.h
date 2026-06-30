#ifndef PARAMETERS_H
#define PARAMETERS_H

// ==========================================
// 【全局调试开关】不需要打印时，直接注释掉下面这行
// #define ENABLE_SLAM_DEBUG
// ==========================================

#ifdef ENABLE_SLAM_DEBUG
    #include <iostream>
    #define DEBUG_INFO(msg)  std::cout << "[VO-INFO] " << msg << std::endl
    #define DEBUG_WARN(msg)  std::cout << "[VO-WARN] " << msg << std::endl
    #define DEBUG_ERROR(msg) std::cerr << "[VO-ERR]  " << msg << std::endl
#else
    #define DEBUG_INFO(msg)
    #define DEBUG_WARN(msg)
    #define DEBUG_ERROR(msg)
#endif

#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

/**
 * @brief 全局参数管理类（从 YAML 配置文件读取）
 * 
 * 所有参数均为静态成员，在程序启动时由 readParameters() 加载。
 * 使用时直接通过 Parameters::参数名 访问。
 */
class Parameters
{
public:
    static void readParameters(const std::string &config_file);

    // -------- 系统基础配置 --------
    static int IMU;                 // 是否使用 IMU (0=禁用)
    static int NUM_OF_CAM;          // 相机数量（此处固定为2）
    static std::string IMU_TOPIC;   // IMU 话题名（未使用）
    static std::string IMAGE0_TOPIC; // 左目图像话题
    static std::string IMAGE1_TOPIC; // 右目图像话题
    static std::string OUTPUT_PATH;  // 轨迹输出目录
    static int IMAGE_WIDTH;         // 图像宽度
    static int IMAGE_HEIGHT;        // 图像高度

    // -------- 相机内参 --------
    static double fx, fy, cx, cy;   // 焦距和主点
    static double k1, k2, p1, p2;   // 畸变系数（KITTI 均为0）

    // -------- 相机外参（相对于车体/IMU） --------
    static Eigen::Matrix4d body_T_cam0;  // 左目相机到车体坐标系的变换
    static Eigen::Matrix4d body_T_cam1;  // 右目相机到车体坐标系的变换

    // -------- 特征跟踪参数 --------
    static int MAX_CNT;            // 每帧最多特征点数
    static int MIN_DIST;           // 特征点之间最小像素距离
    static int FREQ;               // 发布频率（未使用）
    static double F_THRESHOLD;     // RANSAC 基础矩阵阈值
    static int SHOW_TRACK;         // 是否发布跟踪图像
    static int FLOW_BACK;          // 是否进行双向光流校验

    // -------- 优化参数 --------
    static double MAX_SOLVER_TIME;     // 最大求解时间（未严格使用）
    static int MAX_NUM_ITERATIONS;     // 最大迭代次数
    static double KEYFRAME_PARALLAX;   // 关键帧选择视差阈值（像素）

    // -------- IMU 参数（未使用） --------
    static double ACC_N;
    static double GYR_N;
    static double ACC_W;
    static double GYR_W;
    static double G_NORM;

    // -------- 时间同步参数 --------
    static int ESTIMATE_TD;
    static double TD;
};

#endif // PARAMETERS_H