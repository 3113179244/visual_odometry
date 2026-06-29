#ifndef PARAMETERS_H
#define PARAMETERS_H

// ==========================================
// 【全局调试开关】不需要打印时，直接注释掉下面这行
#define ENABLE_SLAM_DEBUG
// ==========================================

#ifdef ENABLE_SLAM_DEBUG
    #include <iostream>
    // 纯文本输出，无任何控制台颜色字符
    #define DEBUG_INFO(msg)  std::cout << "[VO-INFO] " << msg << std::endl
    #define DEBUG_WARN(msg)  std::cout << "[VO-WARN] " << msg << std::endl
    #define DEBUG_ERROR(msg) std::cerr << "[VO-ERR]  " << msg << std::endl
#else
    // 关闭开关时宏为空，编译器自动忽略，实现零运行时开销
    #define DEBUG_INFO(msg)
    #define DEBUG_WARN(msg)
    #define DEBUG_ERROR(msg)
#endif

#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

// 声明全局静态参数，方便任何其他文件直接包含并使用
class Parameters
{
public:
    // 读取主函数，传入 kitti_config04-12.yaml 的绝对路径
    static void readParameters(const std::string &config_file);
    // 基础参数
    static int IMU;
    static int NUM_OF_CAM;
    static std::string IMU_TOPIC;
    static std::string IMAGE0_TOPIC;
    static std::string IMAGE1_TOPIC;
    static std::string OUTPUT_PATH;
    static int IMAGE_WIDTH;
    static int IMAGE_HEIGHT;

    // 相机内参 (来自关联的内参 YAML)
    static double fx, fy, cx, cy;
    static double k1, k2, p1, p2;

    // 相机外参矩阵 (Eigen 格式方便 SLAM 核心算法使用)
    static Eigen::Matrix4d body_T_cam0;
    static Eigen::Matrix4d body_T_cam1;

    // 特征跟踪参数
    static int MAX_CNT;
    static int MIN_DIST;
    static int FREQ;
    static double F_THRESHOLD;
    static int SHOW_TRACK;
    static int FLOW_BACK;

    // 优化器参数
    static double MAX_SOLVER_TIME;
    static int MAX_NUM_ITERATIONS;
    static double KEYFRAME_PARALLAX;
    // IMU 参数
    static double ACC_N;
    static double GYR_N;
    static double ACC_W;
    static double GYR_W;
    static double G_NORM;

    // 时间同步参数
    static int ESTIMATE_TD;
    static double TD;
};

#endif // PARAMETERS_H