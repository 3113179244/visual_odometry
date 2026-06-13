#include "stereo_vo.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
// ====== DEBUG REFACTOR CODE START ======
#include "feature_utils.h"
// ====== DEBUG REFACTOR CODE END ======

using namespace std;
using namespace cv;

void StereoVO::extractORBWithQuadTree(const Mat &img, vector<KeyPoint> &kps, Mat &desc, int num_features)
{
    Ptr<ORB> orb_handler = ORB::create(num_features * 2);

    // ====== DEBUG REFACTOR CODE START ======
    // 🌟🌟🌟 构建车载相机专用的 ROI 拦截掩码 (Mask) 🌟🌟🌟
    // 全白（255）代表允许提取，全黑（0）代表强行拦截禁止提取
    cv::Mat mask = cv::Mat::ones(img.size(), CV_8U) * 255;

    int rows = img.rows;
    int cols = img.cols;

    // 1. 拦截左侧及左上方的细碎树丛/天空噪声区 (x从0到 cols*0.35, y从0到 rows*0.45)
    cv::Rect left_tree_zone(0, 0, static_cast<int>(cols * 0.35), static_cast<int>(rows * 0.45));
    mask(left_tree_zone).setTo(0);

    // 2. 拦截最下方的本车引擎盖盲区 (y从 rows*0.88 直到图像底部)
    cv::Rect car_hood_zone(0, static_cast<int>(rows * 0.88), cols, rows - static_cast<int>(rows * 0.88));
    mask(car_hood_zone).setTo(0);

    vector<KeyPoint> vIniKeys;
    // 将构建好的 mask 注入 detect 算子，OpenCV 会自动避开黑色区域
    orb_handler->detect(img, vIniKeys, mask);
    // ====== DEBUG REFACTOR CODE END ======

    // 调用工具箱进行空间均匀分发
    kps = vo_feature::DistributeQuadTree(vIniKeys, 0, img.cols, 0, img.rows, num_features);

    if (!kps.empty())
    {
        // 先备份四叉树筛选出的黄金点合法的 response
        vector<float> response_backup(kps.size());
        for (size_t i = 0; i < kps.size(); ++i)
        {
            response_backup[i] = kps[i].response;
        }

        // 计算描述子 (防止 OpenCV 内部抹零篡改)
        orb_handler->compute(img, kps, desc);

        // 强行把正确的响应写回
        for (size_t i = 0; i < kps.size(); ++i)
        {
            kps[i].response = response_backup[i];
        }
    }
}

bool StereoVO::loadCalibration(const string &calib_file_path)
{
    ifstream file(calib_file_path);
    if (!file.is_open())
        return false;
    string line;
    vector<double> P0(12), P1(12);
    bool has_P0 = false, has_P1 = false;
    while (getline(file, line))
    {
        if (line.rfind("P0:", 0) == 0)
        {
            stringstream ss(line.substr(4));
            for (int i = 0; i < 12; ++i)
                ss >> P0[i];
            has_P0 = true;
        }
        if (line.rfind("P1:", 0) == 0)
        {
            stringstream ss(line.substr(4));
            for (int i = 0; i < 12; ++i)
                ss >> P1[i];
            has_P1 = true;
        }
    }
    if (has_P0 && has_P1)
    {
        fx = P0[0];
        cx = P0[2];
        fy = P0[5];
        cy = P0[6];
        baseline = -P1[3] / fx;
        K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        return true;
    }
    return false;
}

void StereoVO::matchORB(const Mat &img1, const Mat &img2, vector<KeyPoint> &kps1, vector<KeyPoint> &kps2, vector<DMatch> &good_matches)
{
    Mat desc1, desc2;
    std::thread t_left([&]()
                       { this->extractORBWithQuadTree(img1, kps1, desc1, 2000); });
    std::thread t_right([&]()
                        { this->extractORBWithQuadTree(img2, kps2, desc2, 2000); });
    t_left.join();
    t_right.join();

    if (desc1.empty() || desc2.empty())
        return;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches;
    matcher->match(desc1, desc2, matches);
    double min_dist = 10000;
    for (int i = 0; i < desc1.rows; i++)
        if (matches[i].distance < min_dist)
            min_dist = matches[i].distance;
    good_matches.clear();
    for (int i = 0; i < desc1.rows; i++)
        if (matches[i].distance <= max(2 * min_dist, 30.0))
            good_matches.push_back(matches[i]);
}

void StereoVO::build3D2DPoints(const vector<KeyPoint> &kps1_l, const vector<KeyPoint> &kps1_r, const vector<KeyPoint> &kps2_l,
                               const vector<DMatch> &stereo_matches, const vector<DMatch> &temporal_matches,
                               vector<Point3f> &pts_3d, vector<Point2f> &pts_2d, vector<DMatch> &viz_matches)
{
    for (const auto &t_m : temporal_matches)
    {
        int idx_l1 = t_m.queryIdx, idx_l2 = t_m.trainIdx;
        for (const auto &s_m : stereo_matches)
        {
            if (s_m.queryIdx == idx_l1)
            {
                int idx_r1 = s_m.trainIdx;
                if (abs(kps1_l[idx_l1].pt.y - kps1_r[idx_r1].pt.y) > 2.0)
                    continue;
                double disparity = kps1_l[idx_l1].pt.x - kps1_r[idx_r1].pt.x;
                if (disparity <= 0.0)
                    continue;
                double z = (fx * baseline) / disparity;
                if (z < 0.5 || z > 80.0)
                    continue;
                pts_3d.push_back(Point3f((kps1_l[idx_l1].pt.x - cx) * z / fx, (kps1_l[idx_l1].pt.y - cy) * z / fy, z));
                pts_2d.push_back(kps2_l[idx_l2].pt);
                viz_matches.push_back(t_m);
                break;
            }
        }
    }
}

void StereoVO::matchDescriptors(const Mat &desc1, const Mat &desc2, vector<DMatch> &good_matches)
{
    if (desc1.empty() || desc2.empty())
        return;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches;
    matcher->match(desc1, desc2, matches);
    double min_dist = 10000;
    for (int i = 0; i < desc1.rows; i++)
        if (matches[i].distance < min_dist)
            min_dist = matches[i].distance;
    good_matches.clear();
    for (int i = 0; i < desc1.rows; i++)
        if (matches[i].distance <= max(2 * min_dist, 30.0))
            good_matches.push_back(matches[i]);
}

void StereoVO::matchStereoEpipolar(const vector<KeyPoint> &kps_l, const vector<KeyPoint> &kps_r,
                                   const Mat &desc_l, const Mat &desc_r,
                                   vector<DMatch> &good_matches, float max_v_disp)
{
    good_matches.clear();
    const int TH_LOW = 50;

    for (size_t i = 0; i < kps_l.size(); i++)
    {
        const KeyPoint &kp_l = kps_l[i];
        int best_dist = 256;
        int best_idx = -1;

        for (size_t j = 0; j < kps_r.size(); j++)
        {
            const KeyPoint &kp_r = kps_r[j];
            if (abs(kp_l.pt.y - kp_r.pt.y) > max_v_disp)
                continue;
            float disp = kp_l.pt.x - kp_r.pt.x;
            if (disp <= 0.0f)
                continue;

            int dist = cv::norm(desc_l.row(i), desc_r.row(j), cv::NORM_HAMMING);
            if (dist < best_dist)
            {
                best_dist = dist;
                best_idx = j;
            }
        }
        if (best_idx >= 0 && best_dist < TH_LOW)
            good_matches.push_back(DMatch(i, best_idx, best_dist));
    }
}

void StereoVO::matchTemporalByProjection(const std::vector<cv::KeyPoint> &kps_prev,
                                         const std::vector<cv::Point3f> &pts_3d_prev,
                                         const cv::Mat &desc_prev,
                                         const std::vector<cv::KeyPoint> &kps_curr,
                                         const cv::Mat &desc_curr,
                                         const Sophus::SE3d &T_curr_prev,
                                         const cv::Mat &K,
                                         std::vector<cv::DMatch> &temporal_matches,
                                         float search_radius)
{
    temporal_matches.clear();
    double fx_ = K.at<double>(0, 0);
    double fy_ = K.at<double>(1, 1);
    double cx_ = K.at<double>(0, 2);
    double cy_ = K.at<double>(1, 2);
    const int TH_LOW = 50;

    for (size_t i = 0; i < pts_3d_prev.size(); i++)
    {
        if (i >= (size_t)desc_prev.rows)
            continue;

        const Point3f &pt3d = pts_3d_prev[i];
        if (pt3d.z <= 0)
            continue;

        Eigen::Vector3d p_prev(pt3d.x, pt3d.y, pt3d.z);
        Eigen::Vector3d p_curr = T_curr_prev * p_prev;
        if (p_curr.z() <= 0.1)
            continue;

        double u_pred = fx_ * p_curr.x() / p_curr.z() + cx_;
        double v_pred = fy_ * p_curr.y() / p_curr.z() + cy_;

        int best_dist = 256;
        int best_idx = -1;

        for (size_t j = 0; j < kps_curr.size(); j++)
        {
            const KeyPoint &kp_curr = kps_curr[j];
            double du = kp_curr.pt.x - u_pred;
            double dv = kp_curr.pt.y - v_pred;
            if (du * du + dv * dv > search_radius * search_radius)
                continue;

            int dist = cv::norm(desc_prev.row(i), desc_curr.row(j), cv::NORM_HAMMING);
            if (dist < best_dist)
            {
                best_dist = dist;
                best_idx = j;
            }
        }
        if (best_idx >= 0 && best_dist < TH_LOW)
            temporal_matches.push_back(DMatch(i, best_idx, best_dist));
    }
}