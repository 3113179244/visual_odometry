#include "system.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
namespace fs = std::filesystem;

System::System(const std::string& calib_file) : mbRunning(true) {
    mpVO = std::make_unique<StereoVO>();
    if (!mpVO->loadCalibration(calib_file)) {
        cerr << "致命错误: 标定加载失败! 请检查路径: " << calib_file << endl;
        exit(-1);
    }
}

System::~System() {
    Stop();
}

void System::Run(const std::string& path_left, const std::string& path_right, const std::string& traj_out_path) {
    vector<string> files_left, files_right;
    for (const auto &e : fs::directory_iterator(path_left))
        if (e.path().extension() == ".png") files_left.push_back(e.path().string());
    for (const auto &e : fs::directory_iterator(path_right))
        if (e.path().extension() == ".png") files_right.push_back(e.path().string());
    sort(files_left.begin(), files_left.end());
    sort(files_right.begin(), files_right.end());

    if (files_left.empty()) {
        cerr << "没有找到图像数据!" << endl; return;
    }

    // 提前读取第一张图以获取图片分辨率，用于模块初始化
    int img_width = cv::imread(files_left[0], cv::IMREAD_GRAYSCALE).cols;
    int img_height = cv::imread(files_left[0], cv::IMREAD_GRAYSCALE).rows;

    // ==========================================
    // 实例化与分配各大子模块
    // ==========================================
    mpMap = std::make_shared<Map>();
    mpSlideWindow = std::make_unique<SlidingWindow>(10, mpVO->K, mpVO->baseline, img_width, img_height, mpMap);
    mpLocalMapper = std::make_unique<LocalMapping>(mpSlideWindow.get(), mpMap, mpVO.get());
    mpSelector = std::make_unique<KeyframeSelector>(mpVO->fx, mpVO->fy, mpVO->cx, mpVO->cy, mpVO->baseline, img_width, img_height);
    mpSelector->setParameters(10.0, 0.15, 30, 15.0, 0.5, 6, 8, 0.35, 0.85);
    
    // 组装前端追踪器
    mpTracker = std::make_unique<Tracking>(mpVO.get(), mpMap, mpLocalMapper.get(), mpSelector.get());

    // 启动后端线程
    mpLocalMapper->Start();

    cv::namedWindow("VO Feature Tracking (Viewer Thread)", cv::WINDOW_NORMAL);
    cv::resizeWindow("VO Feature Tracking (Viewer Thread)", 1200, 400);

    int max_frames = files_left.size() - 1;
    cout << "== SLAM 系统启动 (纯前端模式)，共 " << max_frames << " 帧 ==" << endl;

    // ==========================================
    // 多线程调度：分离追踪核心与 UI 显示
    // ==========================================
    std::thread tracker_thread([&]() {
        ofstream traj_out(traj_out_path);
        traj_out << "# x y z (Pure Frontend Odometry)" << endl;

        for (int i = 0; i <= max_frames && mbRunning; ++i) {
            cv::Mat img_curr_l = cv::imread(files_left[i], cv::IMREAD_GRAYSCALE);
            cv::Mat img_curr_r = cv::imread(files_right[i], cv::IMREAD_GRAYSCALE);
            if (img_curr_l.empty() || img_curr_r.empty()) break;

            // 核心推演
            Sophus::SE3d global_pose = mpTracker->GrabImageStereo(img_curr_l, img_curr_r, i);

            // 记录轨迹
            traj_out << global_pose.translation().x() << " "
                     << global_pose.translation().y() << " "
                     << global_pose.translation().z() << endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(50)); 
        }
        mpLocalMapper->Stop();
        traj_out.close();
        mbRunning = false; // 跑完通知主线程退出
    });

    // 主线程专心做 UI 渲染
    while (mbRunning) {
        cv::Mat img_show = mpTracker->GetVizImage();
        if (!img_show.empty()) {
            cv::imshow("VO Feature Tracking (Viewer Thread)", img_show);
        }
        int key = cv::waitKey(30);
        if (key == 27) { mbRunning = false; break; } // 按 ESC 退出
    }

    if (tracker_thread.joinable()) tracker_thread.join();
    cout << "== 追踪结束，系统安全退出 ==" << endl;
}

void System::Stop() {
    mbRunning = false;
    if (mpLocalMapper) mpLocalMapper->Stop();
}