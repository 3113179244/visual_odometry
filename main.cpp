// main.cpp (修正版)
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
#include <set>
#include <map>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <sophus/se3.hpp>
#include "slide_window.h"
#include "stereo_vo.h"
#include "keyframe_selector.h"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void bundleAdjustment(const vector<cv::Point3f>& points_3d,
                      const vector<cv::Point2f>& points_2d,
                      const cv::Mat& K, cv::Mat& R, cv::Mat& t);

vector<Point2f> points2f_from_kps(const vector<KeyPoint>& kps) {
    vector<Point2f> pts; pts.reserve(kps.size());
    for (auto& kp : kps) pts.push_back(kp.pt);
    return pts;
}

vector<Point3f> getLocalMapWorldPoints(const StereoVO& vo) {
    vector<Point3f> pts;
    pts.reserve(vo.local_map.size());
    for (auto& mp : vo.local_map) {
        pts.push_back(mp.pos_world);
    }
    return pts;
}

mutex img_mutex;
Mat shared_viz_img;
atomic<bool> system_running{true};

void TrackingThread()
{
    StereoVO vo;
    if (!vo.loadCalibration("/home/wzj/KITTI/data_odometry_gray/dataset/sequences/03/calib.txt"))
    {
        cerr << "标定加载失败!" << endl;
        system_running = false;
        return;
    }

    string path_left = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/03/image_0";
    string path_right = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/03/image_1";
    vector<string> files_left, files_right;
    for (const auto &e : fs::directory_iterator(path_left))
        if (e.path().extension() == ".png")
            files_left.push_back(e.path().string());
    for (const auto &e : fs::directory_iterator(path_right))
        if (e.path().extension() == ".png")
            files_right.push_back(e.path().string());
    sort(files_left.begin(), files_left.end());
    sort(files_right.begin(), files_right.end());

    ofstream traj_out("/home/wzj/stereovo/trajectory.txt");
    traj_out << "# x y z (Global Trajectory)" << endl;
    Sophus::SE3d global_pose;
    traj_out << 0.0 << " " << 0.0 << " " << 0.0 << endl;

    int max_frames = files_left.size() - 1;
    cout << "== 追踪线程启动，共 " << max_frames << " 帧 ==" << endl;

    // 关键帧管理初始化
    int img_width = imread(files_left[0], IMREAD_GRAYSCALE).cols;
    int img_height = imread(files_left[0], IMREAD_GRAYSCALE).rows;
    KeyframeSelector selector(vo.fx, vo.fy, vo.cx, vo.cy, vo.baseline, img_width, img_height);
    selector.setParameters(10.0, 0.15, 30, 15.0, 0.5, 6, 8, 0.35, 0.85);

    // 创建滑动窗口优化器（窗口大小10）
    SlidingWindow slide_win(10, vo.K, vo.baseline, vo.local_map);

    vector<Keyframe> window_keyframes;
    const int max_window_size = 15;
    const Keyframe *last_keyframe_ptr = nullptr;

    // 第一帧处理
    Mat img_prev_l = imread(files_left[0], IMREAD_GRAYSCALE);
    Mat img_prev_r = imread(files_right[0], IMREAD_GRAYSCALE);
    vector<KeyPoint> kps_prev_l, kps_prev_r;
    Mat desc_prev_l, desc_prev_r;
    vector<DMatch> stereo_matches;

    thread t_init_l([&]()
                    { vo.extractORBWithQuadTree(img_prev_l, kps_prev_l, desc_prev_l, 2000); });
    thread t_init_r([&]()
                    { vo.extractORBWithQuadTree(img_prev_r, kps_prev_r, desc_prev_r, 2000); });
    t_init_l.join();
    t_init_r.join();
    vo.matchDescriptors(desc_prev_l, desc_prev_r, stereo_matches);

    vector<DMatch> empty_temporal;
    vector<Point3f> init_pts_3d_cam;
    vector<Point2f> dummy_pts_2d;
    vector<DMatch> dummy_viz;
    vo.build3D2DPoints(kps_prev_l, kps_prev_r, kps_prev_l,
                       stereo_matches, empty_temporal,
                       init_pts_3d_cam, dummy_pts_2d, dummy_viz);

    Sophus::SE3d identity_pose;
    vo.updateLocalMap(init_pts_3d_cam, img_prev_l, kps_prev_l, identity_pose);

    vector<int> prev_mp_indices(kps_prev_l.size(), -1);
    size_t base_idx = vo.local_map.size() - init_pts_3d_cam.size();
    for (size_t i = 0; i < init_pts_3d_cam.size(); ++i)
    {
        prev_mp_indices[i] = base_idx + i;
    }

    cv::Mat rvec_init = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec_init = cv::Mat::zeros(3, 1, CV_64F);
    vector<Point2f> kps_prev_2d = points2f_from_kps(kps_prev_l);
    Keyframe first_kf(0, rvec_init, tvec_init, kps_prev_2d, prev_mp_indices);
    window_keyframes.push_back(first_kf);
    last_keyframe_ptr = &window_keyframes.back();
    // 将第一帧也加入滑动窗口优化器
    slide_win.addKeyframe(first_kf);
    cout << "第一帧已作为关键帧插入，local_map 大小: " << vo.local_map.size() << endl;

    // 主循环
    for (int i = 1; i <= max_frames && system_running; ++i)
    {
        Mat img_curr_l = imread(files_left[i], IMREAD_GRAYSCALE);
        Mat img_curr_r = imread(files_right[i], IMREAD_GRAYSCALE);
        if (img_curr_l.empty() || img_curr_r.empty())
            break;

        vector<KeyPoint> kps_curr_l, kps_curr_r;
        Mat desc_curr_l, desc_curr_r;
        thread t_l([&]()
                   { vo.extractORBWithQuadTree(img_curr_l, kps_curr_l, desc_curr_l, 2000); });
        thread t_r([&]()
                   { vo.extractORBWithQuadTree(img_curr_r, kps_curr_r, desc_curr_r, 2000); });
        t_l.join();
        t_r.join();

        vector<DMatch> temporal_matches;
        vo.matchDescriptors(desc_prev_l, desc_curr_l, temporal_matches);

        vector<Point3f> pts_3d_cam;
        vector<Point2f> pts_2d_curr;
        vector<DMatch> viz_matches;
        vo.build3D2DPoints(kps_prev_l, kps_prev_r, kps_curr_l,
                           stereo_matches, temporal_matches,
                           pts_3d_cam, pts_2d_curr, viz_matches);

        vector<int> curr_mp_indices(kps_curr_l.size(), -1);
        for (const auto &match : temporal_matches)
        {
            int prev_idx = match.queryIdx;
            int curr_idx = match.trainIdx;
            if (prev_idx < (int)prev_mp_indices.size() && prev_mp_indices[prev_idx] != -1)
            {
                curr_mp_indices[curr_idx] = prev_mp_indices[prev_idx];
            }
        }

        bool tracking_success = false;
        cv::Mat rvec_curr, tvec_curr;

        if (pts_3d_cam.size() >= 10)
        {
            Mat rvec_pnp, tvec_pnp, inliers, R_pnp;
            solvePnPRansac(pts_3d_cam, pts_2d_curr, vo.K, Mat(), rvec_pnp, tvec_pnp, false, 100, 2.0, 0.99, inliers);
            if (!inliers.empty() && inliers.rows >= 15 && norm(tvec_pnp) < 3.0)
            {
                Rodrigues(rvec_pnp, R_pnp);
                bundleAdjustment(pts_3d_cam, pts_2d_curr, vo.K, R_pnp, tvec_pnp);
                if (norm(tvec_pnp) < 3.0)
                {
                    Eigen::Matrix3d R_eigen;
                    cv::cv2eigen(R_pnp, R_eigen);
                    Eigen::Vector3d t_eigen(tvec_pnp.at<double>(0, 0), tvec_pnp.at<double>(1, 0), tvec_pnp.at<double>(2, 0));
                    Sophus::SE3d T_cw(R_eigen, t_eigen);
                    global_pose = global_pose * T_cw.inverse();
                    rvec_curr = rvec_pnp.clone();
                    tvec_curr = tvec_pnp.clone();
                    tracking_success = true;
                }
            }
        }

        if (!tracking_success)
        {
            cout << "⚠️ 警告 [帧 " << i << "]: 触发位姿防暴走熔断！保持上一帧位姿" << endl;
            if (last_keyframe_ptr)
            {
                rvec_curr = last_keyframe_ptr->rvec.clone();
                tvec_curr = last_keyframe_ptr->tvec.clone();
            }
            else
            {
                rvec_curr = Mat::zeros(3, 1, CV_64F);
                tvec_curr = Mat::zeros(3, 1, CV_64F);
            }
        }

        traj_out << global_pose.translation().x() << " "
                 << global_pose.translation().y() << " "
                 << global_pose.translation().z() << endl;

        // 关键帧决策
        vector<int> matched_indices;
        for (int idx : curr_mp_indices)
            if (idx != -1)
                matched_indices.push_back(idx);
        vector<Point2f> curr_kps_2d = points2f_from_kps(kps_curr_l);
        vector<Point3f> local_map_world = getLocalMapWorldPoints(vo);

        bool is_keyframe = false;
        if (i > 1)
        {
            is_keyframe = selector.decide(rvec_curr, tvec_curr,
                                          curr_kps_2d,
                                          matched_indices,
                                          local_map_world,
                                          last_keyframe_ptr,
                                          window_keyframes,
                                          false);
        }

        if (is_keyframe)
        {
            Keyframe new_kf(i, rvec_curr, tvec_curr, curr_kps_2d, matched_indices);
            window_keyframes.push_back(new_kf);
            if (window_keyframes.size() > max_window_size)
                window_keyframes.erase(window_keyframes.begin());
            last_keyframe_ptr = &window_keyframes.back();
            // 添加到滑动窗口（内部会触发局部BA优化）
            slide_win.addKeyframe(new_kf);
            cout << "📸 关键帧插入: 帧 " << i << ", 窗口大小=" << window_keyframes.size() << endl;
        }

        // 可视化
        Mat img_viz;
        drawMatches(img_prev_l, kps_prev_l, img_curr_l, kps_curr_l, viz_matches, img_viz,
                    Scalar(0, 255, 0), Scalar(0, 0, 255), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        {
            lock_guard<mutex> lock(img_mutex);
            shared_viz_img = img_viz.clone();
        }

        // 状态滚动
        img_prev_l = img_curr_l.clone();
        kps_prev_l = kps_curr_l;
        kps_prev_r = kps_curr_r;
        desc_prev_l = desc_curr_l.clone();
        desc_prev_r = desc_curr_r.clone();
        prev_mp_indices = std::move(curr_mp_indices);
        vo.matchDescriptors(desc_prev_l, desc_prev_r, stereo_matches);

        this_thread::sleep_for(chrono::milliseconds(10));
    }

    traj_out.close();
    cout << "== 追踪线程结束 ==" << endl;
    system_running = false;
}

int main() {
    cv::setNumThreads(1);
    thread tracker_thread(TrackingThread);
    namedWindow("VO Feature Tracking (Viewer Thread)", WINDOW_NORMAL);
    resizeWindow("VO Feature Tracking (Viewer Thread)", 1200, 400);
    while (system_running) {
        Mat img_show;
        {
            lock_guard<mutex> lock(img_mutex);
            if (!shared_viz_img.empty()) img_show = shared_viz_img.clone();
        }
        if (!img_show.empty()) imshow("VO Feature Tracking (Viewer Thread)", img_show);
        int key = waitKey(30);
        if (key == 27) { system_running = false; break; }
    }
    if (tracker_thread.joinable()) tracker_thread.join();
    cout << "系统安全退出。" << endl;
    return 0;
}