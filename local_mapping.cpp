#include "local_mapping.h"
#include <iostream>
#include "stereo_vo.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <algorithm>
#include "math_utils.h" // 🌟 引入全量数学工具箱

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
            std::cout << "\n[LocalMapping] 后端收到新关键帧 ID: " << currKF->id << "，开始建图..." << std::endl;

            // 🌟 核心捕获点：获取大地图快照，抓取前一个历史关键帧指针，用作单目时间轴三角化
            auto global_kfs = mpMap->GetAllKeyframes();
            KeyframePtr prevKF = nullptr;
            if (!global_kfs.empty())
            {
                prevKF = global_kfs.back();
            }

            mpMap->AddKeyframe(currKF);

            std::vector<cv::DMatch> stereo_matches;
            mpVO->matchStereoEpipolar(currKF->kps_l, currKF->kps_r, currKF->desc_l, currKF->desc_r, stereo_matches);

            // 建立状态表，防止多视图重复建点导致内存亢余
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

            // --- 🌟 策略 B 扩展：如果双目失效，自动退化为时间轴多视图线性三角化 🌟 ---
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
                    // 仅对未分配 3D 坐标且双目失败的黄金特征点进行多视图追溯三角化
                    if (point_triangulated[i] || currKF->map_point_indices[i] != -1)
                        continue;

                    // 利用前后帧的特征点在像素层面的连续关联进行反向归一化平面映射
                    if (i < prevKF->map_point_indices.size() && prevKF->map_point_indices[i] != -1)
                    {

                        Eigen::Vector2d kp1((prevKF->keypoints_2d[i].x - mpVO->cx) / mpVO->fx,
                                            (prevKF->keypoints_2d[i].y - mpVO->cy) / mpVO->fy);
                        Eigen::Vector2d kp2((currKF->kps_l[i].pt.x - mpVO->cx) / mpVO->fx,
                                            (currKF->kps_l[i].pt.y - mpVO->cy) / mpVO->fy);

                        Eigen::Vector3d P_world_svd;
                        // 调用工具箱内带有两帧正深度约束防御的 SVD 线性三角化接口
                        if (vo_math::TriangulatePointSVD(T_cw_prev.inverse(), T_cw_curr.inverse(), kp1, kp2, P_world_svd))
                        {
                            // 三角化结果投影回当前帧相机中心坐标系下
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
    std::unique_lock<std::shared_mutex> lock(mpMap->mMutexMap);

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
        if (best_match_global_idx >= 0 && best_dist < TH_HIGH)
        {
            pKF->map_point_indices[i] = best_match_global_idx;
            global_mps[best_match_global_idx]->n_observed++;
            mp_curr->is_bad = true;
        }
        else
        {
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