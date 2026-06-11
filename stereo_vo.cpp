#include "stereo_vo.h"
#include <fstream>
#include <sstream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;

// ==========================================
// 新增实现：四叉树节点分裂逻辑
// ==========================================
void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4) {
    const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
    const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

    n1.UL = UL; n1.UR = Point2i(UL.x + halfX, UL.y); n1.BL = Point2i(UL.x, UL.y + halfY); n1.BR = Point2i(UL.x + halfX, UL.y + halfY);
    n2.UL = n1.UR; n2.UR = UR; n2.BL = n1.BR; n2.BR = Point2i(UR.x, UL.y + halfY);
    n3.UL = n1.BL; n3.UR = n1.BR; n3.BL = BL; n3.BR = Point2i(n1.BR.x, BL.y);
    n4.UL = n3.UR; n4.UR = n2.BR; n4.BL = n3.BR; n4.BR = BR;

    n1.vKeys.reserve(vKeys.size()); n2.vKeys.reserve(vKeys.size());
    n3.vKeys.reserve(vKeys.size()); n4.vKeys.reserve(vKeys.size());

    for (size_t i = 0; i < vKeys.size(); i++) {
        const KeyPoint &kp = vKeys[i];
        if (kp.pt.x < n1.UR.x) {
            if (kp.pt.y < n1.BR.y) n1.vKeys.push_back(kp); else n3.vKeys.push_back(kp);
        } else {
            if (kp.pt.y < n1.BR.y) n2.vKeys.push_back(kp); else n4.vKeys.push_back(kp);
        }
    }

    if (n1.vKeys.size() == 1) n1.bNoMore = true;
    if (n2.vKeys.size() == 1) n2.bNoMore = true;
    if (n3.vKeys.size() == 1) n3.bNoMore = true;
    if (n4.vKeys.size() == 1) n4.bNoMore = true;
}

// ==========================================
// 新增实现：四叉树核心分配算法
// ==========================================
vector<KeyPoint> StereoVO::DistributeQuadTree(const vector<KeyPoint>& vToDistributeKeys, int minX, int maxX, int minY, int maxY, int N) {
    if (vToDistributeKeys.size() < (size_t)N) return vToDistributeKeys;

    const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));
    const float hX = static_cast<float>(maxX - minX) / nIni;

    list<ExtractorNode> lNodes;
    vector<ExtractorNode*> vpIniNodes(nIni);

    for (int i = 0; i < nIni; i++) {
        ExtractorNode ni;
        ni.UL = Point2i(hX * static_cast<float>(i), minY);
        ni.UR = Point2i(hX * static_cast<float>(i + 1), minY);
        ni.BL = Point2i(ni.UL.x, maxY);
        ni.BR = Point2i(ni.UR.x, maxY);
        ni.vKeys.reserve(vToDistributeKeys.size());
        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    for (size_t i = 0; i < vToDistributeKeys.size(); i++) {
        const KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }

    auto lit = lNodes.begin();
    while (lit != lNodes.end()) {
        if (lit->vKeys.size() == 1) { lit->bNoMore = true; lit++; } 
        else if (lit->vKeys.empty()) lit = lNodes.erase(lit); 
        else lit++;
    }

    bool bFinish = false;
    int iteration = 0;
    while (!bFinish) {
        iteration++;
        int prevSize = lNodes.size();
        lit = lNodes.begin();
        int nToExpand = 0;

        while (lit != lNodes.end()) {
            if (lit->bNoMore) { lit++; continue; }
            ExtractorNode n1, n2, n3, n4;
            lit->DivideNode(n1, n2, n3, n4);

            if (n1.vKeys.size() > 0) { lNodes.push_front(n1); if (n1.vKeys.size() > 1) { nToExpand++; lNodes.front().lit = lNodes.begin(); }}
            if (n2.vKeys.size() > 0) { lNodes.push_front(n2); if (n2.vKeys.size() > 1) { nToExpand++; lNodes.front().lit = lNodes.begin(); }}
            if (n3.vKeys.size() > 0) { lNodes.push_front(n3); if (n3.vKeys.size() > 1) { nToExpand++; lNodes.front().lit = lNodes.begin(); }}
            if (n4.vKeys.size() > 0) { lNodes.push_front(n4); if (n4.vKeys.size() > 1) { nToExpand++; lNodes.front().lit = lNodes.begin(); }}

            lit = lNodes.erase(lit);
        }
        if ((int)lNodes.size() >= N || nToExpand == 0) bFinish = true;
    }

    // 提取叶子节点中响应值最大的点
    vector<KeyPoint> vResultKeys;
    vResultKeys.reserve(lNodes.size());
    for (lit = lNodes.begin(); lit != lNodes.end(); lit++) {
        vector<KeyPoint> &vNodeKeys = lit->vKeys;
        KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;
        for (size_t k = 1; k < vNodeKeys.size(); k++) {
            if (vNodeKeys[k].response > maxResponse) { pKP = &vNodeKeys[k]; maxResponse = vNodeKeys[k].response; }
        }
        vResultKeys.push_back(*pKP);
    }
    return vResultKeys;
}

// ==========================================
// 新增实现：封装完整的 ORB 特征提取流程
// ==========================================
void StereoVO::extractORBWithQuadTree(const Mat& img, vector<KeyPoint>& kps, Mat& desc, int num_features) {
    // 1. 大量提取初始特征点以保证候选充足，并利用 ORB 自带提取机制保留金字塔尺度特性
    Ptr<ORB> orb_detector = ORB::create(num_features * 4); 
    vector<KeyPoint> vIniKeys;
    orb_detector->detect(img, vIniKeys);

    // 2. 通过四叉树筛选出分布均匀的特征点
    kps = DistributeQuadTree(vIniKeys, 0, img.cols, 0, img.rows, num_features);

    // 3. 计算描述子 (仅针对保留下来的高质量均匀点)
    Ptr<ORB> orb_descriptor = ORB::create();
    orb_descriptor->compute(img, kps, desc);
}

// ==========================================
// 原有业务函数修改
// ==========================================

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
    Mat desc1, desc2;
    // 替换：使用四叉树均匀化提取特征
    std::thread t_left([&]() { this->extractORBWithQuadTree(img1, kps1, desc1, 2000); });
    std::thread t_right([&]() { this->extractORBWithQuadTree(img2, kps2, desc2, 2000); });
    
    t_left.join();
    t_right.join();

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

    Mat desc_curr;
    // 替换：使用四叉树均匀化提取特征
    extractORBWithQuadTree(img_curr, kps_curr, desc_curr, 2000);
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
            
            Eigen::Vector3d P_c = T_world_predict * P_w;

            pts_3d.push_back(Point3f(P_c.x(), P_c.y(), P_c.z()));
            pts_2d.push_back(kps_curr[curr_kp_idx].pt);
            viz_matches.push_back(m);
        }
    }
}

void StereoVO::updateLocalMap(const vector<Point3f>& pts_3d_cam, const Mat& img_curr_l, 
                             const vector<KeyPoint>& kps_curr_l, const Sophus::SE3d& T_world_curr) {
    Mat desc_curr; vector<KeyPoint> kps_tmp;
    // 替换：使用四叉树均匀化提取特征
    extractORBWithQuadTree(img_curr_l, kps_tmp, desc_curr, 2000);

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

    if (local_map.size() > max_local_points) {
        local_map.erase(local_map.begin(), local_map.begin() + (local_map.size() - max_local_points));
    }
    cout << " 📌 [Local Map] 缓存池中有效 3D 锚点总数: " << local_map.size() << endl;
}

void StereoVO::matchDescriptors(const Mat& desc1, const Mat& desc2, vector<DMatch>& good_matches) {
    if (desc1.empty() || desc2.empty()) return;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches; 
    matcher->match(desc1, desc2, matches);
    
    double min_dist = 10000;
    for (int i = 0; i < desc1.rows; i++) {
        if (matches[i].distance < min_dist) min_dist = matches[i].distance;
    }
    good_matches.clear();
    for (int i = 0; i < desc1.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }
}