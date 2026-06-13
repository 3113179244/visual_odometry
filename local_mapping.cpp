#include "local_mapping.h"
#include <iostream>
#include "stereo_vo.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <algorithm>

LocalMapping::LocalMapping(SlidingWindow* slide_win, std::shared_ptr<Map> pMap, StereoVO* vo) 
    : mpSlideWindow(slide_win), mKeyFrameQueue(), mptLocalMapping(nullptr), mbRunning(false),
      mpVO(vo), mpMap(pMap) {} 

LocalMapping::~LocalMapping() { Stop(); }

void LocalMapping::Start() {
    mbRunning = true;
    mptLocalMapping = new std::thread(&LocalMapping::Run, this);
    std::cout << "[LocalMapping] 后端优化线程已启动。" << std::endl;
}

void LocalMapping::Stop() {
    if (mbRunning) {
        mbRunning = false;
        mKeyFrameQueue.shutdown(); 
        if (mptLocalMapping && mptLocalMapping->joinable()) { mptLocalMapping->join(); }
        delete mptLocalMapping; mptLocalMapping = nullptr;
        std::cout << "[LocalMapping] 后端优化线程已安全停止。" << std::endl;
    }
}

void LocalMapping::InsertKeyFrame(KeyframePtr pKF) { mKeyFrameQueue.push(pKF); }

void LocalMapping::Run() {
    while (mbRunning) {
        KeyframePtr currKF;
        if (mKeyFrameQueue.pop(currKF)) {
            if (!currKF) continue;
            std::cout << "\n[LocalMapping] 后端收到新关键帧 ID: " << currKF->id << "，开始建图..." << std::endl;
            mpMap->AddKeyframe(currKF);

            std::vector<cv::DMatch> stereo_matches;
            mpVO->matchStereoEpipolar(currKF->kps_l, currKF->kps_r, currKF->desc_l, currKF->desc_r, stereo_matches);

            std::vector<cv::Point3f> init_pts_3d_cam;
            std::vector<cv::Mat> descriptors_new; 
            std::vector<int> stereo_l_indices; 

            for (const auto& sm : stereo_matches) {
                int idx_l = sm.queryIdx; int idx_r = sm.trainIdx;
                
                // 如果当前关键帧的这个特征点在前端已经被追踪并分配了合法的地图点全局ID，后端直接跳过，防止重复创建
                if (currKF->map_point_indices[idx_l] != -1) continue;

                double disparity = currKF->kps_l[idx_l].pt.x - currKF->kps_r[idx_r].pt.x;
                if (disparity > 0) {
                    double z = (mpVO->fx * mpVO->baseline) / disparity;
                    if (z < 0.5 || z > 80.0) continue;

                    double x = (currKF->kps_l[idx_l].pt.x - mpVO->cx) * z / mpVO->fx;
                    double y = (currKF->kps_l[idx_l].pt.y - mpVO->cy) * z / mpVO->fy;
                    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

                    init_pts_3d_cam.push_back(cv::Point3f(x, y, z));
                    descriptors_new.push_back(currKF->desc_l.row(idx_l).clone());
                    stereo_l_indices.push_back(idx_l); 
                }
            }

            Eigen::Matrix3d R_eigen; cv::Mat R_mat;
            cv::Rodrigues(currKF->rvec, R_mat); cv::cv2eigen(R_mat, R_eigen);
            Eigen::Vector3d t_eigen(currKF->tvec.at<double>(0, 0), currKF->tvec.at<double>(1, 0), currKF->tvec.at<double>(2, 0));
            Sophus::SE3d T_cw(R_eigen, t_eigen);
            Sophus::SE3d T_wc = T_cw.inverse(); 

            size_t start_global_idx = mpMap->GetGlobalMapPointsSize();
            int valid_added_count = 0; 

            for (size_t i = 0; i < init_pts_3d_cam.size(); ++i) {
                Eigen::Vector3d P_c(init_pts_3d_cam[i].x, init_pts_3d_cam[i].y, init_pts_3d_cam[i].z);
                Eigen::Vector3d P_w = T_wc * P_c;

                if (!std::isfinite(P_w.x()) || !std::isfinite(P_w.y()) || !std::isfinite(P_w.z())) continue;
                if (P_w.norm() > 200.0) continue;

                auto pMP = std::make_shared<MapPoint>();
                pMP->pos_world = cv::Point3f(P_w.x(), P_w.y(), P_w.z());
                pMP->descriptor = descriptors_new[i];
                pMP->is_bad = false;
                pMP->n_observed = 1;
                pMP->n_visible = 1;

                mpMap->AddMapPoint(pMP);
                
                currKF->map_point_indices[stereo_l_indices[i]] = start_global_idx + valid_added_count;
                valid_added_count++;
            }

            mpMap->UpdateLocalMap(currKF, 10);
            FuseMapPoints(currKF);
            mpSlideWindow->addKeyframe(*currKF);
            CullLocalMap();
            
            // 【核心修改点】：将此处的物理清理代码注释掉！
            // 防止大地图平移引起的关键帧整型坐标下标错位与索引坍塌。
            // if (currKF->id % 5 == 0) { mpMap->CleanBadMechanisms(); }
            
            std::cout << "[LocalMapping] 后端关键帧 ID: " << currKF->id << " 局部融合与优化全部完成。\n" << std::endl;
        }
    }
}

void LocalMapping::FuseMapPoints(KeyframePtr pKF) {
    std::vector<MapPointPtr> local_mps = mpMap->GetLocalMapPoints();
    std::vector<MapPointPtr> global_mps = mpMap->GetAllMapPoints(); 
    const float dist_th = 0.1f; const int TH_HIGH = 50;      
    std::unique_lock<std::shared_mutex> lock(mpMap->mMutexMap); 

    for (size_t i = 0; i < pKF->map_point_indices.size(); ++i) {
        int curr_idx = pKF->map_point_indices[i];
        if (curr_idx < 0 || static_cast<size_t>(curr_idx) >= global_mps.size()) continue;
        MapPointPtr mp_curr = global_mps[curr_idx];
        if (!mp_curr || mp_curr->is_bad) continue;

        int best_match_global_idx = -1; int best_dist = 256;
        for (auto& mp_cand : local_mps) {
            if (mp_cand == mp_curr || mp_cand->is_bad) continue;
            float dx = mp_curr->pos_world.x - mp_cand->pos_world.x;
            float dy = mp_curr->pos_world.y - mp_cand->pos_world.y;
            float dz = mp_curr->pos_world.z - mp_cand->pos_world.z;
            if ((dx*dx + dy*dy + dz*dz) > dist_th * dist_th) continue;

            int dist = cv::norm(mp_curr->descriptor, mp_cand->descriptor, cv::NORM_HAMMING);
            if (dist < best_dist) {
                best_dist = dist;
                auto it = std::find(global_mps.begin(), global_mps.end(), mp_cand);
                if (it != global_mps.end()) { best_match_global_idx = std::distance(global_mps.begin(), it); }
            }
        }
        if (best_match_global_idx >= 0 && best_dist < TH_HIGH) {
            pKF->map_point_indices[i] = best_match_global_idx;
            global_mps[best_match_global_idx]->n_observed++; mp_curr->is_bad = true; 
        } else {
            mp_curr->n_observed++; mp_curr->n_visible++;
        }
    }
}

void LocalMapping::CullLocalMap() {
    std::vector<MapPointPtr> local_mps = mpMap->GetLocalMapPoints();
    std::unique_lock<std::shared_mutex> lock(mpMap->mMutexMap);
    for (auto& mp : local_mps) {
        if (!mp || mp->is_bad) continue;
        mp->n_visible++;
        if ((static_cast<float>(mp->n_observed) / mp->n_visible) < 0.25f && mp->n_visible > 3) { mp->is_bad = true; }
    }
}