#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

// Eigen 必须在 OpenCV eigen 转换头文件之前
#include <Eigen/Core>
#include <Eigen/Dense>   // 可选，提供完整 Eigen 定义

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>   // 现在 Eigen 已包含，不会触发错误

#include <sophus/se3.hpp>

#include "stereo_vo.h"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// 外部声明的 BA 优化函数（实现在 pose_optimizer.cpp）
void bundleAdjustment(const vector<cv::Point3f>& points_3d,
                      const vector<cv::Point2f>& points_2d,
                      const cv::Mat& K, cv::Mat& R, cv::Mat& t);

// ==============================================================================
// 共享数据区 (线程间通信)
// ==============================================================================
mutex img_mutex;
Mat shared_viz_img;
atomic<bool> system_running{true};

// ==============================================================================
// 线程 1：追踪线程 (Tracking Thread)
// ==============================================================================
void TrackingThread() {
    StereoVO vo;    // 使用 stereo_vo.h 中定义的类
    if (!vo.loadCalibration("/home/wzj/KITTI/data_odometry_gray/dataset/sequences/07/calib.txt")) {
        cerr << "标定加载失败!" << endl;
        system_running = false;
        return;
    }

    string path_left = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/07/image_0";
    string path_right = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/07/image_1";
    vector<string> files_left, files_right;
    for (const auto& e : fs::directory_iterator(path_left))
        if (e.path().extension() == ".png") files_left.push_back(e.path().string());
    for (const auto& e : fs::directory_iterator(path_right))
        if (e.path().extension() == ".png") files_right.push_back(e.path().string());
    sort(files_left.begin(), files_left.end());
    sort(files_right.begin(), files_right.end());

    ofstream traj_out("trajectory.txt");
    traj_out << "# x y z (Global Trajectory)" << endl;
    Sophus::SE3d global_pose;      // 世界坐标系下的相机位姿，初始为单位阵
    traj_out << 0.0 << " " << 0.0 << " " << 0.0 << endl;

    // 读取第一帧左右图像
    Mat img_prev_l = imread(files_left[0], IMREAD_GRAYSCALE);
    Mat img_prev_r = imread(files_right[0], IMREAD_GRAYSCALE);
    if (img_prev_l.empty() || img_prev_r.empty()) {
        cerr << "无法读取第一帧图像!" << endl;
        system_running = false;
        return;
    }

    int max_frames = files_left.size() - 1;
    cout << "== 追踪线程启动，共 " << max_frames << " 帧 ==" << endl;

    for (int i = 1; i <= max_frames && system_running; ++i) {
        Mat img_curr_l = imread(files_left[i], IMREAD_GRAYSCALE);
        if (img_curr_l.empty()) break;

        // 1. 提取前后帧及右图特征点，并进行匹配
        vector<KeyPoint> kps_prev_l, kps_prev_r, kps_curr_l;
        vector<DMatch> stereo_matches, temporal_matches;

        vo.matchORB(img_prev_l, img_prev_r, kps_prev_l, kps_prev_r, stereo_matches);
        vo.matchORB(img_prev_l, img_curr_l, kps_prev_l, kps_curr_l, temporal_matches);

        // 2. 利用双目三角化构建 3D-2D 对应关系（前一帧的 3D 点 -> 当前帧的 2D 点）
        vector<Point3f> pts_3d;          // 前一帧相机坐标系下的 3D 点
        vector<Point2f> pts_2d;          // 当前帧对应的 2D 点
        vector<DMatch> viz_matches;      // 用于可视化的匹配点

        vo.build3D2DPoints(kps_prev_l, kps_prev_r, kps_curr_l,
                           stereo_matches, temporal_matches,
                           pts_3d, pts_2d, viz_matches);

        bool tracking_success = false;

        if (pts_3d.size() >= 10) {
            Mat rvec, tvec, inliers, R_pnp;
            // 求解 PnP，得到前一帧到当前帧的相对位姿
            solvePnPRansac(pts_3d, pts_2d, vo.K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);

            // 熔断保护：内点数足够且平移量合理
            bool pass_inlier_check = (!inliers.empty() && inliers.rows >= 15);
            bool pass_motion_check = (norm(tvec) < 3.0);

            if (pass_inlier_check && pass_motion_check) {
                Rodrigues(rvec, R_pnp);

                // Ceres BA 优化
                bundleAdjustment(pts_3d, pts_2d, vo.K, R_pnp, tvec);

                // 再次检查优化后平移量
                if (norm(tvec) < 3.0) {
                    Eigen::Matrix3d R_eigen;
                    cv::cv2eigen(R_pnp, R_eigen);
                    Eigen::Vector3d t_eigen(tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0));
                    Sophus::SE3d T_curr_to_prev(R_eigen, t_eigen);

                    // 累积全局位姿（注意相对位姿的定义：T_curr_to_prev 表示从 prev 到 curr）
                    global_pose = global_pose * T_curr_to_prev.inverse();
                    tracking_success = true;
                }
            }

            if (!tracking_success) {
                cout << "⚠️ 警告 [帧 " << i << "]: 触发位姿防暴走熔断！内点数: "
                     << (inliers.empty() ? 0 : inliers.rows) << ", 预测位移: " << norm(tvec) << " 米 (保持上一帧位姿)" << endl;
            }
        } else {
            cout << "⚠️ 警告 [帧 " << i << "]: 特征点过少 (" << pts_3d.size() << ")，无法求解 PnP！" << endl;
        }

        // 写入轨迹（若跟踪失败则保持上一帧位置）
        traj_out << global_pose.translation().x() << " "
                 << global_pose.translation().y() << " "
                 << global_pose.translation().z() << endl;

        // 可视化：绘制前一帧与当前帧的匹配线
        Mat img_viz;
        drawMatches(img_prev_l, kps_prev_l, img_curr_l, kps_curr_l, viz_matches, img_viz,
                    Scalar(0, 255, 0), Scalar(0, 0, 255), vector<char>(),
                    DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        {
            lock_guard<mutex> lock(img_mutex);
            shared_viz_img = img_viz.clone();
        }

        // 滑动窗口前进
        img_prev_l = img_curr_l.clone();
        img_prev_r = imread(files_right[i], IMREAD_GRAYSCALE);
        if (img_prev_r.empty()) break;

        this_thread::sleep_for(chrono::milliseconds(10));
    }

    traj_out.close();
    cout << "== 追踪线程结束，轨迹已保存 ==" << endl;
    system_running = false;
}

// ==============================================================================
// 线程 2：主线程 -> 可视化线程
// ==============================================================================
int main() {
    thread tracker_thread(TrackingThread);

    namedWindow("VO Feature Tracking (Viewer Thread)", WINDOW_NORMAL);
    resizeWindow("VO Feature Tracking (Viewer Thread)", 1200, 400);

    while (system_running) {
        Mat img_show;
        {
            lock_guard<mutex> lock(img_mutex);
            if (!shared_viz_img.empty()) {
                img_show = shared_viz_img.clone();
            }
        }
        if (!img_show.empty()) {
            imshow("VO Feature Tracking (Viewer Thread)", img_show);
        }
        int key = waitKey(30);
        if (key == 27) { // ESC 退出
            system_running = false;
            break;
        }
    }

    if (tracker_thread.joinable()) {
        tracker_thread.join();
    }
    cout << "系统安全退出。" << endl;
    return 0;
}