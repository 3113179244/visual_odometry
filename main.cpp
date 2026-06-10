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

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// 声明外部的 Ceres 优化函数 (保持 pose_optimizer.cpp 不变)
void bundleAdjustment(const vector<cv::Point3f>& points_3d, const vector<cv::Point2f>& points_2d,
                      const cv::Mat& K, cv::Mat& R, cv::Mat& t);

// ==============================================================================
// 共享数据区 (线程间通信)
// ==============================================================================
mutex img_mutex;                // 保护图像的互斥锁
Mat shared_viz_img;             // 追踪线程画好后，传给可视化线程的图像
atomic<bool> system_running{true}; // 控制系统退出的原子标志位

// ==============================================================================
// 算法封装类：双目视觉里程计前端
// ==============================================================================
class StereoVO {
public:
    double fx, fy, cx, cy, baseline;
    Mat K;

    bool loadCalibration(const string& calib_file_path) {
        ifstream file(calib_file_path);
        if (!file.is_open()) return false;
        string line; vector<double> P0(12), P1(12); bool has_P0 = false, has_P1 = false;
        while (getline(file, line)) {
            if (line.rfind("P0:", 0) == 0) { stringstream ss(line.substr(4)); for (int i = 0; i < 12; ++i) ss >> P0[i]; has_P0 = true; }
            if (line.rfind("P1:", 0) == 0) { stringstream ss(line.substr(4)); for (int i = 0; i < 12; ++i) ss >> P1[i]; has_P1 = true; }
        }
        if (has_P0 && has_P1) { 
            fx = P0[0]; cx = P0[2]; fy = P0[5]; cy = P0[6]; baseline = -P1[3] / fx; 
            K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
            return true; 
        }
        return false;
    }

    void matchORB(const Mat& img1, const Mat& img2, vector<KeyPoint>& kps1, vector<KeyPoint>& kps2, vector<DMatch>& good_matches) {
        Ptr<ORB> orb = ORB::create(1000); Mat desc1, desc2;
        orb->detectAndCompute(img1, noArray(), kps1, desc1);
        orb->detectAndCompute(img2, noArray(), kps2, desc2);
        if (desc1.empty() || desc2.empty()) return;
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        vector<DMatch> matches; matcher->match(desc1, desc2, matches);
        double min_dist = 10000;
        for (int i = 0; i < desc1.rows; i++) if (matches[i].distance < min_dist) min_dist = matches[i].distance;
        good_matches.clear();
        for (int i = 0; i < desc1.rows; i++) if (matches[i].distance <= max(2 * min_dist, 30.0)) good_matches.push_back(matches[i]);
    }

    void build3D2DPoints(const vector<KeyPoint>& kps1_l, const vector<KeyPoint>& kps1_r, const vector<KeyPoint>& kps2_l,
                         const vector<DMatch>& stereo_matches, const vector<DMatch>& temporal_matches,
                         vector<Point3f>& pts_3d, vector<Point2f>& pts_2d, vector<DMatch>& viz_matches) {
        for (const auto& t_m : temporal_matches) {
            int idx_l1 = t_m.queryIdx, idx_l2 = t_m.trainIdx;
            for (const auto& s_m : stereo_matches) {
                if (s_m.queryIdx == idx_l1) {
                    int idx_r1 = s_m.trainIdx;
                    if (abs(kps1_l[idx_l1].pt.y - kps1_r[idx_r1].pt.y) > 2.0) continue;
                    double disparity = kps1_l[idx_l1].pt.x - kps1_r[idx_r1].pt.x;
                    if (disparity <= 0.0) continue;
                    double z = (fx * baseline) / disparity;
                    if (z < 0.5 || z > 80.0) continue;
                    double x = (kps1_l[idx_l1].pt.x - cx) * z / fx;
                    double y = (kps1_l[idx_l1].pt.y - cy) * z / fy;

                    pts_3d.push_back(Point3f(x, y, z));
                    pts_2d.push_back(kps2_l[idx_l2].pt);
                    viz_matches.push_back(t_m); 
                    break;
                }
            }
        }
    }
};

// ==============================================================================
// 线程 1：追踪线程 (Tracking Thread)
// ==============================================================================
void TrackingThread() {
    StereoVO vo;
    if (!vo.loadCalibration("/home/wzj/KITTI/data_odometry_gray/dataset/sequences/07/calib.txt")) {
        cerr << "标定加载失败!" << endl; system_running = false; return;
    }

    string path_left = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/07/image_0";
    string path_right = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/07/image_1";
    vector<string> files_left, files_right;
    for (const auto& e : fs::directory_iterator(path_left)) if (e.path().extension() == ".png") files_left.push_back(e.path().string());
    for (const auto& e : fs::directory_iterator(path_right)) if (e.path().extension() == ".png") files_right.push_back(e.path().string());
    sort(files_left.begin(), files_left.end());
    sort(files_right.begin(), files_right.end());

    ofstream traj_out("trajectory.txt");
    traj_out << "# x y z (Global Trajectory)" << endl;
    Sophus::SE3d global_pose; 
    traj_out << 0.0 << " " << 0.0 << " " << 0.0 << endl;

    Mat img_prev_l = imread(files_left[0], IMREAD_GRAYSCALE);
    Mat img_prev_r = imread(files_right[0], IMREAD_GRAYSCALE);

    int max_frames = files_left.size() - 1;
    cout << "== 追踪线程启动，共 " << max_frames << " 帧 ==" << endl;

    for (int i = 1; i <= max_frames && system_running; i++) {
        Mat img_curr_l = imread(files_left[i], IMREAD_GRAYSCALE);

        vector<KeyPoint> kps1_l, kps1_r, kps2_l;
        vector<DMatch> stereo_matches, temporal_matches;
        vo.matchORB(img_prev_l, img_prev_r, kps1_l, kps1_r, stereo_matches);
        vo.matchORB(img_prev_l, img_curr_l, kps1_l, kps2_l, temporal_matches);

        vector<Point3f> pts_3d; vector<Point2f> pts_2d; vector<DMatch> viz_matches;
        vo.build3D2DPoints(kps1_l, kps1_r, kps2_l, stereo_matches, temporal_matches, pts_3d, pts_2d, viz_matches);

        // 默认将当前滑窗的数据集拷贝，用于防熔断时维持滑动窗口更新
        bool tracking_success = false;

        if (pts_3d.size() >= 10) {
            Mat rvec, tvec, inliers, R_pnp;
            solvePnPRansac(pts_3d, pts_2d, vo.K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
            
            // ==================================================================
            // 🔥【核心修改：智能熔断防跑飞保护罩】
            // ==================================================================
            // 检查 1：经过 RANSAC 过滤后的绝对内点数是否足够。少于15个点说明匹配质量极度不可信。
            bool pass_inlier_check = (!inliers.empty() && inliers.rows >= 15);
            
            // 检查 2：平移模长检查。0.1秒内如果平移超过3米（时速>108km/h），属于代数发散。
            bool pass_motion_check = (norm(tvec) < 3.0);

            if (pass_inlier_check && pass_motion_check) {
                Rodrigues(rvec, R_pnp);
                
                // 初值安全，才允许送入 Ceres 进行打磨优化
                bundleAdjustment(pts_3d, pts_2d, vo.K, R_pnp, tvec);

                // 再次确保 Ceres 优化没有把位姿强行带偏
                if (norm(tvec) < 3.0) {
                    Eigen::Matrix3d R_eigen; cv::cv2eigen(R_pnp, R_eigen);
                    Eigen::Vector3d t_eigen(tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0));
                    Sophus::SE3d T_curr_to_prev(R_eigen, t_eigen);
                    
                    // 累加绝对位姿
                    global_pose = global_pose * T_curr_to_prev.inverse();
                    tracking_success = true;
                }
            }
            
            // 打印异常警告
            if (!tracking_success) {
                cout << "⚠️ 警告 [帧 " << i << "]: 触发位姿防暴走熔断！内点数: " 
                     << (inliers.empty() ? 0 : inliers.rows) << ", 预测位移: " << norm(tvec) << " 米 (保持上一帧位姿)" << endl;
            }
        } else {
            cout << "⚠️ 警告 [帧 " << i << "]: 特征点过少 (" << pts_3d.size() << ")，无法求解 PnP！" << endl;
        }

        // 写入轨迹（如果熔断，则直接复制写入上一帧的绝对坐标，确保时间戳连续不空断）
        traj_out << global_pose.translation().x() << " " << global_pose.translation().y() << " " << global_pose.translation().z() << endl;

        // 绘制特征图（无论成功与否，继续绘制传给前端，方便可视化观察）
        Mat img_viz;
        drawMatches(img_prev_l, kps1_l, img_curr_l, kps2_l, viz_matches, img_viz, 
                    Scalar(0, 255, 0), Scalar(0, 0, 255), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        
        {
            lock_guard<mutex> lock(img_mutex);
            shared_viz_img = img_viz.clone();
        }

        // 滑动窗口无缝推进
        img_prev_l = img_curr_l.clone();
        img_prev_r = imread(files_right[i], IMREAD_GRAYSCALE);
        
        this_thread::sleep_for(chrono::milliseconds(10)); 
    }

    traj_out.close();
    cout << "== 追踪线程结束，轨迹已保存 ==" << endl;
    system_running = false; 
}

// ==============================================================================
// 线程 2：主线程 -> 可视化线程 (Viewer Thread)
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
        if (key == 27) { 
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