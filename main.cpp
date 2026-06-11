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
    StereoVO vo;
    if (!vo.loadCalibration("/home/wzj/KITTI/data_odometry_gray/dataset/sequences/04/calib.txt")) {
        cerr << "标定加载失败!" << endl;
        system_running = false;
        return;
    }

    string path_left = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/04/image_0";
    string path_right = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/04/image_1";
    vector<string> files_left, files_right;
    for (const auto& e : fs::directory_iterator(path_left)) if (e.path().extension() == ".png") files_left.push_back(e.path().string());
    for (const auto& e : fs::directory_iterator(path_right)) if (e.path().extension() == ".png") files_right.push_back(e.path().string());
    sort(files_left.begin(), files_left.end()); sort(files_right.begin(), files_right.end());

    ofstream traj_out("/home/wzj/stereovo/trajectory.txt");
    traj_out << "# x y z (Global Trajectory)" << endl;
    Sophus::SE3d global_pose;
    traj_out << 0.0 << " " << 0.0 << " " << 0.0 << endl;

    int max_frames = files_left.size() - 1;
    cout << "== 追踪线程启动，共 " << max_frames << " 帧 ==" << endl;

    // ==========================================
    // 🚀 系统启动：初始化第 0 帧的状态缓存
    // ==========================================
    Mat img_prev_l = imread(files_left[0], IMREAD_GRAYSCALE);
    Mat img_prev_r = imread(files_right[0], IMREAD_GRAYSCALE);
    vector<KeyPoint> kps_prev_l, kps_prev_r;
    Mat desc_prev_l, desc_prev_r;
    vector<DMatch> stereo_matches;

    // 并行提取第 0 帧左右图特征
    thread t_init_l([&](){ vo.extractORBWithQuadTree(img_prev_l, kps_prev_l, desc_prev_l, 2000); });
    thread t_init_r([&](){ vo.extractORBWithQuadTree(img_prev_r, kps_prev_r, desc_prev_r, 2000); });
    t_init_l.join(); t_init_r.join();
    
    // 计算第 0 帧的双目匹配
    vo.matchDescriptors(desc_prev_l, desc_prev_r, stereo_matches);

    // ==========================================
    // 主循环开始
    // ==========================================
    for (int i = 1; i <= max_frames && system_running; ++i) {
        Mat img_curr_l = imread(files_left[i], IMREAD_GRAYSCALE);
        Mat img_curr_r = imread(files_right[i], IMREAD_GRAYSCALE);
        if (img_curr_l.empty() || img_curr_r.empty()) break;

        vector<KeyPoint> kps_curr_l, kps_curr_r;
        Mat desc_curr_l, desc_curr_r;

        // 🚀 核心优化：只提取当前帧的新特征，上一帧直接用缓存！
        thread t_l([&](){ vo.extractORBWithQuadTree(img_curr_l, kps_curr_l, desc_curr_l, 2000); });
        thread t_r([&](){ vo.extractORBWithQuadTree(img_curr_r, kps_curr_r, desc_curr_r, 2000); });
        t_l.join(); t_r.join();

        // 帧间匹配：上一帧左图 vs 当前帧左图
        vector<DMatch> temporal_matches;
        vo.matchDescriptors(desc_prev_l, desc_curr_l, temporal_matches);

        vector<Point3f> pts_3d;
        vector<Point2f> pts_2d;
        vector<DMatch> viz_matches;

        // 构建 3D-2D 点 (依赖上一帧算好的 stereo_matches 和刚刚算好的 temporal_matches)
        vo.build3D2DPoints(kps_prev_l, kps_prev_r, kps_curr_l,
                           stereo_matches, temporal_matches,
                           pts_3d, pts_2d, viz_matches);

        bool tracking_success = false;

        if (pts_3d.size() >= 10) {
            Mat rvec, tvec, inliers, R_pnp;
            solvePnPRansac(pts_3d, pts_2d, vo.K, Mat(), rvec, tvec, false, 100, 2.0, 0.99, inliers);

            if (!inliers.empty() && inliers.rows >= 15 && norm(tvec) < 3.0) {
                Rodrigues(rvec, R_pnp);
                bundleAdjustment(pts_3d, pts_2d, vo.K, R_pnp, tvec); // 这里现在只迭代 5 次了！

                if (norm(tvec) < 3.0) {
                    Eigen::Matrix3d R_eigen;
                    cv::cv2eigen(R_pnp, R_eigen);
                    Eigen::Vector3d t_eigen(tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0));
                    global_pose = global_pose * Sophus::SE3d(R_eigen, t_eigen).inverse();
                    tracking_success = true;
                }
            }
        }
        if (!tracking_success) {
            cout << "⚠️ 警告 [帧 " << i << "]: 触发位姿防暴走熔断！保持上一帧位姿" << endl;
        }
        traj_out << global_pose.translation().x() << " " << global_pose.translation().y() << " " << global_pose.translation().z() << endl;

        // 可视化
        Mat img_viz;
        drawMatches(img_prev_l, kps_prev_l, img_curr_l, kps_curr_l, viz_matches, img_viz,
                    Scalar(0, 255, 0), Scalar(0, 0, 255), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        {
            lock_guard<mutex> lock(img_mutex);
            shared_viz_img = img_viz.clone();
        }

        // ==========================================
        // 🚀 状态滚动：为下一帧做准备
        // ==========================================
        img_prev_l = img_curr_l.clone();
        kps_prev_l = kps_curr_l;
        kps_prev_r = kps_curr_r;
        desc_prev_l = desc_curr_l.clone();
        desc_prev_r = desc_curr_r.clone();

        // 提前计算好当前帧的双目匹配，供下一帧的 build3D2DPoints 使用
        vo.matchDescriptors(desc_prev_l, desc_prev_r, stereo_matches);

        this_thread::sleep_for(chrono::milliseconds(10));
    }

    traj_out.close();
    cout << "== 追踪线程结束 ==" << endl;
    system_running = false;
}
// ==============================================================================
// 线程 2：主线程 -> 可视化线程
// ==============================================================================
int main() {
    
    // 限制 OpenCV 多线程
    cv::setNumThreads(1); 

    // 👇 确保这里只有一行启动线程的代码，删掉多余的！
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