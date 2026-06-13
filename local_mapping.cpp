// =========================================================================
// ====== DEBUG REFACTOR CODE START ======
// 文件名: local_mapping.cpp
// 修改点: 1. 加入后端关键帧队列控流机制（防止滞后堆积）
//        2. 极度缩小锁的保护范围，优化双重循环匹配性能，消灭线程阻塞
// =========================================================================

#include "local_mapping.h"
#include <iostream>
#include "stereo_vo.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <algorithm>
#include "feature_utils.h"
#include "math_utils.h"

LocalMapping::LocalMapping(SlidingWindow *slide_win, std::shared_ptr<Map> pMap, StereoVO *vo)
    : mpSlideWindow(slide_win), mKeyFrameQueue(), mptLocalMapping(nullptr), mbRunning(false),
      mpVO(vo), mpMap(pMap) {}

LocalMapping::~LocalMapping() { Stop(); }

void LocalMapping::Start()
{
    mbRunning = true;
    mptLocalMapping = new std::thread(&LocalMapping::Run, this);
    std::cout << "[LocalMapping] 后端优化线程已启动。" << std::endl;
}

void LocalMapping::Stop()
{
    if (mbRunning)
    {
        mbRunning = false;
        mKeyFrameQueue.shutdown();
        if (mptLocalMapping && mptLocalMapping->joinable())
        {
            mptLocalMapping->join();
        }
        delete mptLocalMapping;
        mptLocalMapping = nullptr;
        std::cout << "[LocalMapping] 后端优化线程已安全停止。" << std::endl;
    }
}

void LocalMapping::InsertKeyFrame(KeyframePtr pKF) { mKeyFrameQueue.push(pKF); }

void LocalMapping::Run()
{
    while (mbRunning)
    {
        KeyframePtr currKF;
        if (mKeyFrameQueue.pop(currKF))
        {
            if (!currKF)
                continue;

            // 🌟🌟🌟 核心改动点 1：后端超载控流保护 🌟🌟🌟
            // 如果前端发关键帧的速度太快，导致后端堆积严重（比如处理到390帧，但最新堆到了430帧）
            // 如果当前积压的关键帧超过 2 个，并且这个关键帧不是刚初始化完的极重要帧
            // 我们直接放弃对它的后续高耗时建图，防止后端永远追不上前端
            if (currKF->id > 5 && mKeyFrameQueue.size() > 2)
            {
                // 如果这个关键帧跟前一个关键帧靠得太近，放弃处理，直接跳过
                std::cout << "⏩ [LocalMapping OVERLOAD] 后端积压严重！强行跳过并抛弃过时关键帧 ID: " << currKF->id << std::endl;
                continue;
            }

            std::cout << "\n[LocalMapping] 后端收到新关键帧 ID: " << currKF->id << "，开始建图..." << std::endl;

            auto global_kfs = mpMap->GetAllKeyframes();
            KeyframePtr prevKF = nullptr;
            if (!global_kfs.empty())
            {
                prevKF = global_kfs.back();
            }

            mpMap->AddKeyframe(currKF);

            std::vector<cv::DMatch> stereo_matches;
            mpVO->matchStereoEpipolar(currKF->kps_l, currKF->kps_r, currKF->desc_l, currKF->desc_r, stereo_matches);

            std::vector<bool> point_triangulated(currKF->kps_l.size(), false);
            std::vector<cv::Point3f> init_pts_3d_cam;
            std::vector<cv::Mat> descriptors_new;
            std::vector<int> stereo_l_indices;

            // --- 策略 A：经典立体双目深度恢复 ---
            for (const auto &sm : stereo_matches)
            {
                int idx_l = sm.queryIdx;
                int idx_r = sm.trainIdx;
                if (currKF->map_point_indices[idx_l] != -1)
                    continue;

                double disparity = currKF->kps_l[idx_l].pt.x - currKF->kps_r[idx_r].pt.x;
                if (disparity > 0)
                {
                    double z = (mpVO->fx * mpVO->baseline) / disparity;
                    if (z < 0.5 || z > 80.0)
                        continue;

                    double x = (currKF->kps_l[idx_l].pt.x - mpVO->cx) * z / mpVO->fx;
                    double y = (currKF->kps_l[idx_l].pt.y - mpVO->cy) * z / mpVO->fy;
                    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                        continue;

                    init_pts_3d_cam.push_back(cv::Point3f(x, y, z));
                    descriptors_new.push_back(currKF->desc_l.row(idx_l).clone());
                    stereo_l_indices.push_back(idx_l);

                    point_triangulated[idx_l] = true;
                }
            }

            // --- 策略 B：多视图时间轴三角化 ---
            Eigen::Matrix3d R_curr;
            cv::Mat R_mat_c;
            cv::Rodrigues(currKF->rvec, R_mat_c);
            cv::cv2eigen(R_mat_c, R_curr);
            Eigen::Vector3d t_curr(currKF->tvec.at<double>(0, 0), currKF->tvec.at<double>(1, 0), currKF->tvec.at<double>(2, 0));
            Sophus::SE3d T_cw_curr(R_curr, t_curr);
            Sophus::SE3d T_wc_curr = T_cw_curr.inverse();

            if (prevKF != nullptr && prevKF->id != currKF->id)
            {
                Eigen::Matrix3d R_prev;
                cv::Mat R_mat_p;
                cv::Rodrigues(prevKF->rvec, R_mat_p);
                cv::cv2eigen(R_mat_p, R_prev);
                Eigen::Vector3d t_prev(prevKF->tvec.at<double>(0, 0), prevKF->tvec.at<double>(1, 0), prevKF->tvec.at<double>(2, 0));
                Sophus::SE3d T_cw_prev(R_prev, t_prev);

                for (size_t i = 0; i < currKF->kps_l.size(); ++i)
                {
                    if (point_triangulated[i] || currKF->map_point_indices[i] != -1)
                        continue;

                    if (i < prevKF->map_point_indices.size() && prevKF->map_point_indices[i] != -1)
                    {
                        Eigen::Vector2d kp1((prevKF->keypoints_2d[i].x - mpVO->cx) / mpVO->fx,
                                            (prevKF->keypoints_2d[i].y - mpVO->cy) / mpVO->fy);
                        Eigen::Vector2d kp2((currKF->kps_l[i].pt.x - mpVO->cx) / mpVO->fx,
                                            (currKF->kps_l[i].pt.y - mpVO->cy) / mpVO->fy);

                        Eigen::Vector3d P_world_svd;
                        if (vo_math::TriangulatePointSVD(T_cw_prev.inverse(), T_cw_curr.inverse(), kp1, kp2, P_world_svd))
                        {
                            Eigen::Vector3d P_cam = T_cw_curr * P_world_svd;
                            init_pts_3d_cam.push_back(cv::Point3f(P_cam.x(), P_cam.y(), P_cam.z()));
                            descriptors_new.push_back(currKF->desc_l.row(i).clone());
                            stereo_l_indices.push_back(i);
                            point_triangulated[i] = true;
                        }
                    }
                }
            }

            // 统一合并灌装至大地图
            size_t start_global_idx = mpMap->GetGlobalMapPointsSize();
            int valid_added_count = 0;

            for (size_t i = 0; i < init_pts_3d_cam.size(); ++i)
            {
                Eigen::Vector3d P_c(init_pts_3d_cam[i].x, init_pts_3d_cam[i].y, init_pts_3d_cam[i].z);
                Eigen::Vector3d P_w = T_wc_curr * P_c;

                if (!std::isfinite(P_w.x()) || !std::isfinite(P_w.y()) || !std::isfinite(P_w.z()))
                    continue;
                if (P_w.norm() > 200.0)
                    continue;

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

            std::cout << "[LocalMapping] 后端关键帧 ID: " << currKF->id << " 局部融合与优化全部完成。\n"
                      << std::endl;
        }
    }
}

void LocalMapping::FuseMapPoints(KeyframePtr pKF)
{
    std::vector<MapPointPtr> local_mps = mpMap->GetLocalMapPoints();
    std::vector<MapPointPtr> global_mps = mpMap->GetAllMapPoints();
    const float dist_th = 0.1f;
    const int TH_HIGH = 50;

    // 🌟🌟🌟 核心改动点 2：将范围锁缩小为极精细的“局部更新写锁” 🌟🌟🌟
    // 不再锁住整个函数。我们在下方需要改动全局地图点状态时才按需加锁

    for (size_t i = 0; i < pKF->map_point_indices.size(); ++i)
    {
        int curr_idx = pKF->map_point_indices[i];
        if (curr_idx < 0 || static_cast<size_t>(curr_idx) >= global_mps.size())
            continue;
        MapPointPtr mp_curr = global_mps[curr_idx];
        if (!mp_curr || mp_curr->is_bad)
            continue;

        int best_match_global_idx = -1;
        int best_dist = 256;
        for (auto &mp_cand : local_mps)
        {
            if (mp_cand == mp_curr || mp_cand->is_bad)
                continue;
            float dx = mp_curr->pos_world.x - mp_cand->pos_world.x;
            float dy = mp_curr->pos_world.y - mp_cand->pos_world.y;
            float dz = mp_curr->pos_world.z - mp_cand->pos_world.z;
            if ((dx * dx + dy * dy + dz * dz) > dist_th * dist_th)
                continue;

            int dist = cv::norm(mp_curr->descriptor, mp_cand->descriptor, cv::NORM_HAMMING);
            if (dist < best_dist)
            {
                best_dist = dist;
                auto it = std::find(global_mps.begin(), global_mps.end(), mp_cand);
                if (it != global_mps.end())
                {
                    best_match_global_idx = std::distance(global_mps.begin(), it);
                }
            }
        }

        // 🔍 只在真正发生匹配、要修改共享地图数据的一瞬间才加锁保护，处理完立马释放！
        if (best_match_global_idx >= 0 && best_dist < TH_HIGH)
        {
            std::unique_lock<std::shared_mutex> lock(mpMap->mMutexMap);
            pKF->map_point_indices[i] = best_match_global_idx;
            global_mps[best_match_global_idx]->n_observed++;
            mp_curr->is_bad = true;
        }
        else
        {
            std::unique_lock<std::shared_mutex> lock(mpMap->mMutexMap);
            mp_curr->n_observed++;
            mp_curr->n_visible++;
        }
    }
}

void LocalMapping::CullLocalMap()
{
    std::vector<MapPointPtr> local_mps = mpMap->GetLocalMapPoints();
    std::unique_lock<std::shared_mutex> lock(mpMap->mMutexMap);
    for (auto &mp : local_mps)
    {
        if (!mp || mp->is_bad)
            continue;
        mp->n_visible++;
        if ((static_cast<float>(mp->n_observed) / mp->n_visible) < 0.25f && mp->n_visible > 3)
        {
            mp->is_bad = true;
        }
    }
}
// ====== DEBUG REFACTOR CODE END ======