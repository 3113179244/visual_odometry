#include "stereo_vo.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;

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
    while (!bFinish) {
        int nToExpand = 0;
        lit = lNodes.begin();
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

void StereoVO::extractORBWithQuadTree(const Mat& img, vector<KeyPoint>& kps, Mat& desc, int num_features) {
    Ptr<ORB> orb_detector = ORB::create(num_features * 2); 
    vector<KeyPoint> vIniKeys;
    orb_detector->detect(img, vIniKeys);
    kps = DistributeQuadTree(vIniKeys, 0, img.cols, 0, img.rows, num_features);
    Ptr<ORB> orb_descriptor = ORB::create();
    orb_descriptor->compute(img, kps, desc);
}

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
    std::thread t_left([&]() { this->extractORBWithQuadTree(img1, kps1, desc1, 2000); });
    std::thread t_right([&]() { this->extractORBWithQuadTree(img2, kps2, desc2, 2000); });
    t_left.join(); t_right.join();

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
                pts_3d.push_back(Point3f((kps1_l[idx_l1].pt.x - cx) * z / fx, (kps1_l[idx_l1].pt.y - cy) * z / fy, z));
                pts_2d.push_back(kps2_l[idx_l2].pt);
                viz_matches.push_back(t_m); 
                break;
            }
        }
    }
}

void StereoVO::matchDescriptors(const Mat& desc1, const Mat& desc2, vector<DMatch>& good_matches) {
    if (desc1.empty() || desc2.empty()) return;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches; matcher->match(desc1, desc2, matches);
    double min_dist = 10000;
    for (int i = 0; i < desc1.rows; i++) if (matches[i].distance < min_dist) min_dist = matches[i].distance;
    good_matches.clear();
    for (int i = 0; i < desc1.rows; i++) if (matches[i].distance <= max(2 * min_dist, 30.0)) good_matches.push_back(matches[i]);
}

void StereoVO::matchStereoEpipolar(const vector<KeyPoint>& kps_l, const vector<KeyPoint>& kps_r,
                                   const Mat& desc_l, const Mat& desc_r,
                                   vector<DMatch>& good_matches, float max_v_disp) {
    good_matches.clear();
    const int TH_LOW = 50;

    for (size_t i = 0; i < kps_l.size(); i++) {
        const KeyPoint& kp_l = kps_l[i];
        int best_dist = 256; int best_idx = -1;

        for (size_t j = 0; j < kps_r.size(); j++) {
            const KeyPoint& kp_r = kps_r[j];
            if (abs(kp_l.pt.y - kp_r.pt.y) > max_v_disp) continue;
            float disp = kp_l.pt.x - kp_r.pt.x;
            if (disp <= 0.0f) continue;

            int dist = cv::norm(desc_l.row(i), desc_r.row(j), cv::NORM_HAMMING);
            if (dist < best_dist) { best_dist = dist; best_idx = j; }
        }
        if (best_idx >= 0 && best_dist < TH_LOW) good_matches.push_back(DMatch(i, best_idx, best_dist));
    }
}

void StereoVO::matchTemporalByProjection(const vector<KeyPoint>& kps_prev, 
                                         const vector<Point3f>& pts_3d_prev,
                                         const Mat& desc_prev, 
                                         const vector<KeyPoint>& kps_curr, 
                                         const Mat& desc_curr,
                                         const Sophus::SE3d& T_curr_prev,
                                         const Mat& K,
                                         vector<DMatch>& temporal_matches,
                                         float search_radius) {
    temporal_matches.clear();
    double fx_ = K.at<double>(0, 0); double fy_ = K.at<double>(1, 1);
    double cx_ = K.at<double>(0, 2); double cy_ = K.at<double>(1, 2);
    const int TH_LOW = 50;

    for (size_t i = 0; i < pts_3d_prev.size(); i++) {
        const Point3f& pt3d = pts_3d_prev[i];
        if (pt3d.z <= 0) continue;

        Eigen::Vector3d p_prev(pt3d.x, pt3d.y, pt3d.z);
        Eigen::Vector3d p_curr = T_curr_prev * p_prev;
        if (p_curr.z() <= 0.1) continue;

        double u_pred = fx_ * p_curr.x() / p_curr.z() + cx_;
        double v_pred = fy_ * p_curr.y() / p_curr.z() + cy_;

        int best_dist = 256; int best_idx = -1;

        for (size_t j = 0; j < kps_curr.size(); j++) {
            const KeyPoint& kp_curr = kps_curr[j];
            double du = kp_curr.pt.x - u_pred; double dv = kp_curr.pt.y - v_pred;
            if (du * du + dv * dv > search_radius * search_radius) continue;

            int dist = cv::norm(desc_prev.row(i), desc_curr.row(j), cv::NORM_HAMMING);
            if (dist < best_dist) { best_dist = dist; best_idx = j; }
        }
        if (best_idx >= 0 && best_dist < TH_LOW) temporal_matches.push_back(DMatch(i, best_idx, best_dist));
    }
}