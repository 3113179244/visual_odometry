#include "Parameters.h"
#include <iostream>

// 静态成员变量在类外初始化定义
int Parameters::IMU = 0;
int Parameters::NUM_OF_CAM = 0;
std::string Parameters::IMU_TOPIC = "";
std::string Parameters::IMAGE0_TOPIC = "";
std::string Parameters::IMAGE1_TOPIC = "";
std::string Parameters::OUTPUT_PATH = "";
std::string Parameters::VOC_PATH = "";
int Parameters::IMAGE_WIDTH = 0;
int Parameters::IMAGE_HEIGHT = 0;

double Parameters::fx = 0.0;
double Parameters::fy = 0.0;
double Parameters::cx = 0.0;
double Parameters::cy = 0.0;
double Parameters::k1 = 0.0;
double Parameters::k2 = 0.0;
double Parameters::p1 = 0.0;
double Parameters::p2 = 0.0;

Eigen::Matrix4d Parameters::body_T_cam0 = Eigen::Matrix4d::Identity();
Eigen::Matrix4d Parameters::body_T_cam1 = Eigen::Matrix4d::Identity();

int Parameters::MAX_CNT = 0;
int Parameters::MIN_DIST = 0;
int Parameters::FREQ = 0;
double Parameters::F_THRESHOLD = 0.0;
int Parameters::SHOW_TRACK = 0;
int Parameters::FLOW_BACK = 0;

double Parameters::MAX_SOLVER_TIME = 0.0;
int Parameters::MAX_NUM_ITERATIONS = 0;
double Parameters::KEYFRAME_PARALLAX = 0.0;

double Parameters::ACC_N = 0.0;
double Parameters::GYR_N = 0.0;
double Parameters::ACC_W = 0.0;
double Parameters::GYR_W = 0.0;
double Parameters::G_NORM = 0.0;

int Parameters::ESTIMATE_TD = 0;
double Parameters::TD = 0.0;

void Parameters::readParameters(const std::string &config_file)
{
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "错误：无法打开主配置文件: " << config_file << std::endl;
        return;
    }

    // 1. 读取主配置参数
    fs["imu"] >> IMU;
    fs["num_of_cam"] >> NUM_OF_CAM;
    fs["imu_topic"] >> IMU_TOPIC;
    fs["image0_topic"] >> IMAGE0_TOPIC;
    fs["image1_topic"] >> IMAGE1_TOPIC;
    fs["output_path"] >> OUTPUT_PATH;
    fs["image_width"] >> IMAGE_WIDTH;
    fs["image_height"] >> IMAGE_HEIGHT;

    // 2. 嵌套读取相机内参文件
    std::string cam0_calib, cam1_calib;
    fs["cam0_calib"] >> cam0_calib;

    // 获取主配置文件的当前所在目录路径，以便拼接相对路径
    std::string config_dir = config_file.substr(0, config_file.find_last_of("/\\") + 1);
    std::string cam0_path = config_dir + cam0_calib;

    cv::FileStorage fs_cam(cam0_path, cv::FileStorage::READ);
    if (fs_cam.isOpened())
    {
        fs_cam["projection_parameters"]["fx"] >> fx;
        fs_cam["projection_parameters"]["fy"] >> fy;
        fs_cam["projection_parameters"]["cx"] >> cx;
        fs_cam["projection_parameters"]["cy"] >> cy;

        fs_cam["distortion_parameters"]["k1"] >> k1;
        fs_cam["distortion_parameters"]["k2"] >> k2;
        fs_cam["distortion_parameters"]["p1"] >> p1;
        fs_cam["distortion_parameters"]["p2"] >> p2;
        fs_cam.release();
    }
    else
    {
        std::cerr << "警告：无法打开相机内参文件: " << cam0_path << " 将使用默认内参0。" << std::endl;
    }

    // 3. 读取外参矩阵 (cv::Mat 转换为 Eigen::Matrix4d)
    cv::Mat cv_T0, cv_T1;
    fs["body_T_cam0"] >> cv_T0;
    fs["body_T_cam1"] >> cv_T1;

    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            body_T_cam0(r, c) = cv_T0.at<double>(r, c);
            body_T_cam1(r, c) = cv_T1.at<double>(r, c);
        }
    }

    // 4. 读取前端与优化参数
    fs["max_cnt"] >> MAX_CNT;
    fs["min_dist"] >> MIN_DIST;
    fs["freq"] >> FREQ;
    fs["F_threshold"] >> F_THRESHOLD;
    fs["show_track"] >> SHOW_TRACK;
    fs["flow_back"] >> FLOW_BACK;

    fs["max_solver_time"] >> MAX_SOLVER_TIME;
    fs["max_num_iterations"] >> MAX_NUM_ITERATIONS;
    fs["keyframe_parallax"] >> KEYFRAME_PARALLAX;

    // 5. 读取 IMU 及同步参数
    fs["acc_n"] >> ACC_N;
    fs["gyr_n"] >> GYR_N;
    fs["acc_w"] >> ACC_W;
    fs["gyr_w"] >> GYR_W;
    fs["g_norm"] >> G_NORM;

    fs["estimate_td"] >> ESTIMATE_TD;
    fs["td"] >> TD;
    fs["voc_path"] >> VOC_PATH;

    fs.release();
    std::cout << ">>> 成功加载主配文件以及相机内参。图像宽度: " << IMAGE_WIDTH << ", fx: " << fx << std::endl;
}