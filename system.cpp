#include "system.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "math_utils.h" // 🌟 引入全量数学工具箱

using namespace std;
namespace fs = std::filesystem;

System::System(const std::string &calib_file) : mbRunning(true)
{
    mpVO = std::make_unique<StereoVO>();
    if (!mpVO->loadCalibration(calib_file))
    {
        cerr << "致命错误: 标定加载失败! 请检查路径: " << calib_file << endl;
        exit(-1);
    }
}

System::~System()
{
    Stop();
}

void System::Run(const std::string &path_left, const std::string &path_right, const std::string &traj_out_path)
{
    vector<string> files_left, files_right;
    for (const auto &e : fs::directory_iterator(path_left))
        if (e.path().extension() == ".png")
            files_left.push_back(e.path().string());
    for (const auto &e : fs::directory_iterator(path_right))
        if (e.path().extension() == ".png")
            files_right.push_back(e.path().string());
    sort(files_left.begin(), files_left.end());
    sort(files_right.begin(), files_right.end());

    if (files_left.empty())
    {
        cerr << "没有找到图像数据!" << endl;
        return;
    }

    int img_width = cv::imread(files_left[0], cv::IMREAD_GRAYSCALE).cols;
    int img_height = cv::imread(files_left[0], cv::IMREAD_GRAYSCALE).rows;

    mpMap = std::make_shared<Map>();

    mpSlideWindow = std::make_unique<SlidingWindow>(10, mpVO->fx, mpVO->fy, mpVO->cx, mpVO->cy, mpVO->baseline, img_width, img_height, mpMap);
    mpLocalMapper = std::make_unique<LocalMapping>(mpSlideWindow.get(), mpMap, mpVO.get());
    mpSelector = std::make_unique<KeyframeSelector>(mpVO->fx, mpVO->fy, mpVO->cx, mpVO->cy, mpVO->baseline, img_width, img_height);
    mpSelector->setParameters(10.0, 0.15, 30, 15.0, 0.5, 6, 8, 0.35, 0.85);

    mpTracker = std::make_unique<Tracking>(mpVO.get(), mpMap, mpLocalMapper.get(), mpSelector.get());

    mpLocalMapper->Start();

    cv::namedWindow("VO Feature Tracking (Viewer Thread)", cv::WINDOW_NORMAL);
    cv::resizeWindow("VO Feature Tracking (Viewer Thread)", 1200, 400);

    int max_frames = files_left.size() - 1;
    cout << "== SLAM 系统启动 (后端滑窗全量同步模式)，共 " << max_frames + 1 << " 帧 ==" << endl;

    std::vector<Sophus::SE3d> all_frame_poses;

    std::thread tracker_thread([&]()
                               {
        for (int i = 0; i <= max_frames && mbRunning; ++i) {
            cv::Mat img_curr_l = cv::imread(files_left[i], cv::IMREAD_GRAYSCALE);
            cv::Mat img_curr_r = cv::imread(files_right[i], cv::IMREAD_GRAYSCALE);
            if (img_curr_l.empty() || img_curr_r.empty()) break;

            Sophus::SE3d current_p = mpTracker->GrabImageStereo(img_curr_l, img_curr_r, i);
            all_frame_poses.push_back(current_p);

            std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
        }
        
        mpLocalMapper->Stop();
        
        // ====================================================================
        // 🌟【增量拉直对齐算子扩展】：支持无死锁的欧拉角姿态安全导出 🌟
        // ====================================================================
        cout << "💾 后端异步BA处理全部完毕，开始依附末尾关键帧对齐并倒出黄金轨迹..." << endl;
        ofstream final_traj(traj_out_path);
        final_traj << "# x y z roll pitch yaw" << endl; // 丰富外部输出维度
        
        auto optimized_keyframes = mpMap->GetAllKeyframes();
        
        Sophus::SE3d last_kf_frontend_pose;
        Sophus::SE3d last_kf_backend_pose;
        bool found_match = false;

        if (!optimized_keyframes.empty() && !all_frame_poses.empty()) {
            auto last_kf = optimized_keyframes.back();
            if (last_kf && last_kf->id >= 0 && last_kf->id < (int)all_frame_poses.size()) {
                last_kf_frontend_pose = all_frame_poses[last_kf->id];
                
                cv::Mat R_mat; cv::Rodrigues(last_kf->rvec, R_mat);
                Eigen::Matrix3d R_eigen; cv::cv2eigen(R_mat, R_eigen);
                Eigen::Vector3d t_eigen(last_kf->tvec.at<double>(0, 0), last_kf->tvec.at<double>(1, 0), last_kf->tvec.at<double>(2, 0));
                
                last_kf_backend_pose = Sophus::SE3d(R_eigen, t_eigen).inverse(); 
                found_match = true;
            }
        }

        Sophus::SE3d delta_correction = Sophus::SE3d(); 
        if (found_match) {
            delta_correction = last_kf_backend_pose * last_kf_frontend_pose.inverse();
        }

        for (const auto& raw_pose : all_frame_poses) {
            Sophus::SE3d corrected_pose = delta_correction * raw_pose;
            
            // 🌟 从对齐后的位姿中分离出 3x3 旋转矩阵，并一键调用稳健欧拉角转换算子
            Eigen::Matrix3d R_matrix = corrected_pose.rotationMatrix();
            Eigen::Vector3d rpy = vo_math::MatrixToEuler(R_matrix);
            
            // 弧度转角度，提高数据可读性
            double roll_deg  = rpy.x() * 180.0 / M_PI;
            double pitch_deg = rpy.y() * 180.0 / M_PI;
            double yaw_deg   = rpy.z() * 180.0 / M_PI;
            
            final_traj << corrected_pose.translation().x() << " "
                       << corrected_pose.translation().y() << " "
                       << corrected_pose.translation().z() << " "
                       << roll_deg << " " << pitch_deg << " " << yaw_deg << endl;
        }
        
        final_traj.close();
        cout << "💾 全量对齐轨迹导出成功！输出总计数：" << all_frame_poses.size() << " 帧。" << endl;
        mbRunning = false; });

    while (mbRunning)
    {
        cv::Mat img_show = mpTracker->GetVizImage();
        if (!img_show.empty())
        {
            cv::imshow("VO Feature Tracking (Viewer Thread)", img_show);
        }
        int key = cv::waitKey(30);
        if (key == 27)
        {
            mbRunning = false;
            break;
        }
    }

    if (tracker_thread.joinable())
        tracker_thread.join();
    cout << "== 系统安全退出 ==" << endl;
}

void System::Stop()
{
    mbRunning = false;
    if (mpLocalMapper)
        mpLocalMapper->Stop();
}