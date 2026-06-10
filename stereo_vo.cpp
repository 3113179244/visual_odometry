#include "stereo_vo.h"
#include <fstream>
#include <sstream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;

bool StereoVO::loadCalibration(const string& calib_file_path) {
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

void StereoVO::matchORB(const Mat& img1, const Mat& img2, vector<KeyPoint>& kps1, vector<KeyPoint>& kps2, vector<DMatch>& good_matches) {
    Ptr<ORB> orb = ORB::create(2000); Mat desc1, desc2;
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

void StereoVO::build3D2DPoints(const vector<KeyPoint>& kps1_l, const vector<KeyPoint>& kps1_r, const vector<KeyPoint>& kps2_l,
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

void StereoVO::trackLocalMap(const Mat& img_curr, const Sophus::SE3d& T_world_predict, 
                             vector<Point3f>& pts_3d, vector<Point2f>& pts_2d,
                             vector<KeyPoint>& kps_curr, vector<DMatch>& viz_matches) {
    if (local_map.empty()) return;

    Ptr<ORB> orb = ORB::create(2000);
    Mat desc_curr;
    orb->detectAndCompute(img_curr, noArray(), kps_curr, desc_curr);
    if (desc_curr.empty()) return;

    // 拼接 Local Map 的总描述子矩阵
    Mat desc_local;
    for (const auto& mp : local_map) {
        desc_local.push_back(mp.descriptor);
    }

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches;
    matcher->match(desc_local, desc_curr, matches);
    if (matches.empty()) return;

    double min_dist = 10000;
    for (size_t i = 0; i < matches.size(); i++) if (matches[i].distance < min_dist) min_dist = matches[i].distance;

    for (const auto& m : matches) {
        if (m.distance <= max(2 * min_dist, 30.0)) {
            int local_map_idx = m.queryIdx;
            int curr_kp_idx = m.trainIdx;

            Point3f p_w = local_map[local_map_idx].pos_world;
            Eigen::Vector3d P_w(p_w.x, p_w.y, p_w.z);
            
            // 核心变换：把世界坐标系的点，转换到当前预测的相机坐标系下
            Eigen::Vector3d P_c = T_world_predict * P_w;

            pts_3d.push_back(Point3f(P_c.x(), P_c.y(), P_c.z()));
            pts_2d.push_back(kps_curr[curr_kp_idx].pt);
            viz_matches.push_back(m);
        }
    }
}

void StereoVO::updateLocalMap(const vector<Point3f>& pts_3d_cam, const Mat& img_curr_l, 
                             const vector<KeyPoint>& kps_curr_l, const Sophus::SE3d& T_world_curr) {
    Ptr<ORB> orb = ORB::create(2000);
    Mat desc_curr; vector<KeyPoint> kps_tmp;
    orb->detectAndCompute(img_curr_l, noArray(), kps_tmp, desc_curr);

    // 世界位姿的逆，即当前相机系转到世界系的矩阵
    Sophus::SE3d T_camera_to_world = T_world_curr.inverse();

    for (size_t i = 0; i < pts_3d_cam.size() && i < (size_t)desc_curr.rows; ++i) {
        Eigen::Vector3d P_c(pts_3d_cam[i].x, pts_3d_cam[i].y, pts_3d_cam[i].z);
        Eigen::Vector3d P_w = T_camera_to_world * P_c; 

        MapPoint mp;
        mp.pos_world = Point3f(P_w.x(), P_w.y(), P_w.z());
        mp.descriptor = desc_curr.row(i).clone();

        local_map.push_back(mp);
    }

    // 滑动窗口剔除老旧锚点
    if (local_map.size() > max_local_points) {
        local_map.erase(local_map.begin(), local_map.begin() + (local_map.size() - max_local_points));
    }
    cout << " 📌 [Local Map] 缓存池中有效 3D 锚点总数: " << local_map.size() << endl;
}