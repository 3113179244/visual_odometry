#include "local_mapping.h"
#include <iostream>
#include "stereo_vo.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
LocalMapping::LocalMapping(SlidingWindow* slide_win, 
                           std::vector<MapPoint>& local_map, 
                           std::shared_mutex& map_mutex,
                           StereoVO* vo) 
    : mpSlideWindow(slide_win), mKeyFrameQueue(), mptLocalMapping(nullptr), mbRunning(false),
      mpVO(vo), mLocalMap(local_map), mMapMutex(map_mutex) {}

LocalMapping::~LocalMapping() {
    Stop();
}

void LocalMapping::Start() {
    mbRunning = true;
    mptLocalMapping = new std::thread(&LocalMapping::Run, this);
    std::cout << "[LocalMapping] 后端优化线程已启动。" << std::endl;
}

void LocalMapping::Stop() {
    if (mbRunning) {
        mbRunning = false;
        mKeyFrameQueue.shutdown(); // 唤醒可能阻塞在 pop() 的线程
        if (mptLocalMapping && mptLocalMapping->joinable()) {
            mptLocalMapping->join();
        }
        delete mptLocalMapping;
        mptLocalMapping = nullptr;
        std::cout << "[LocalMapping] 后端优化线程已安全停止。" << std::endl;
    }
}

void LocalMapping::InsertKeyFrame(KeyframePtr pKF) {
    mKeyFrameQueue.push(pKF);
}

void LocalMapping::Run() {
    while (mbRunning) {
        KeyframePtr currKF;
        if (mKeyFrameQueue.pop(currKF)) {
            if (!currKF) continue;
            
            std::cout << "\n[LocalMapping] 后端收到新关键帧 ID: " << currKF->id << "，开始在后台异步三角化建图..." << std::endl;
            
            // ----------------------------------------------------
            // 【史诗级重构步骤 1】：在后台异步进行双目匹配、生成新 3D 点
            // ----------------------------------------------------
            std::vector<cv::DMatch> stereo_matches;
            mpVO->matchStereoEpipolar(currKF->kps_l, currKF->kps_r, currKF->desc_l, currKF->desc_r, stereo_matches);

            std::vector<cv::Point3f> init_pts_3d_cam;
            for (const auto& sm : stereo_matches) {
                int idx_l = sm.queryIdx;
                int idx_r = sm.trainIdx;
                double disparity = currKF->kps_l[idx_l].pt.x - currKF->kps_r[idx_r].pt.x;
                if (disparity > 0) {
                    double z = (mpVO->fx * mpVO->baseline) / disparity;
                    double x = (currKF->kps_l[idx_l].pt.x - mpVO->cx) * z / mpVO->fx;
                    double y = (currKF->kps_l[idx_l].pt.y - mpVO->cy) * z / mpVO->fy;
                    init_pts_3d_cam.push_back(cv::Point3f(x, y, z));
                }
            }

            // 计算当前帧从本地转回世界坐标的位姿参数初值
            Eigen::Matrix3d R_eigen;
            cv::Mat R_mat;
            cv::Rodrigues(currKF->rvec, R_mat);
            cv::cv2eigen(R_mat, R_eigen);
            Eigen::Vector3d t_eigen(currKF->tvec.at<double>(0, 0), currKF->tvec.at<double>(1, 0), currKF->tvec.at<double>(2, 0));
            Sophus::SE3d T_cw(R_eigen, t_eigen);

            // 异步将新创建的点并入地图
            size_t old_map_size = 0;
            {
                std::unique_lock<std::shared_mutex> lock(mMapMutex);
                old_map_size = mLocalMap.size();
            }
            
            // 安全调用升级地图
            mpVO->updateLocalMap(init_pts_3d_cam, currKF->img_l, currKF->kps_l, T_cw);

            // 顺便把这一帧刚刚被丢空缺失的 2D 对应地图点索引补齐
            size_t new_points_added = init_pts_3d_cam.size();
            for (size_t i = 0; i < new_points_added; ++i) {
                // 如果当前左图特征点位置以前没有关联任何地图点，我们把它和刚刚新建的地图点索引关联
                // 注：此处作简化对齐，真实 SLAM 系统会采用 Feature 观测类统一对齐
                currKF->map_point_indices.push_back(old_map_size + i);
            }

            // ----------------------------------------------------
            // 步骤 2：执行点融合 (查重与更新)
            // ----------------------------------------------------
            FuseMapPoints(currKF);
            
            // ----------------------------------------------------
            // 步骤 3：加入滑动窗口执行局部 BA 优化
            // ----------------------------------------------------
            mpSlideWindow->addKeyframe(*currKF);
            
            // ----------------------------------------------------
            // 步骤 4：剔除劣质地图点
            // ----------------------------------------------------
            CullLocalMap();
            
            std::cout << "[LocalMapping] 后端关键帧 ID: " << currKF->id << " 异步建图与 BA 优化全部完成。\n" << std::endl;
        }
    }
}

void LocalMapping::FuseMapPoints(KeyframePtr pKF) {
    std::unique_lock<std::shared_mutex> lock(mMapMutex);

    const float dist_th = 0.1f; // 空间距离阈值 (例如 10 厘米，根据 KITTI 尺度可适当调整)
    const int TH_HIGH = 50;     // ORB 描述子汉明距离阈值

    // 遍历当前关键帧观测到的所有地图点
    for (size_t i = 0; i < pKF->map_point_indices.size(); ++i) {
        int curr_idx = pKF->map_point_indices[i];
        if (curr_idx < 0 || static_cast<size_t>(curr_idx) >= mLocalMap.size()) continue;

        MapPoint& mp_curr = mLocalMap[curr_idx];
        if (mp_curr.is_bad) continue;

        int best_match_idx = -1;
        int best_dist = 256;

        // 在局部地图中寻找重复点
        // (注：这里使用暴力搜索，由于 local_map 限制了 1500 个点，开销可控。未来提速可换 Kd-Tree)
        for (size_t j = 0; j < mLocalMap.size(); ++j) {
            // 跳过自己和坏点
            if (j == static_cast<size_t>(curr_idx) || mLocalMap[j].is_bad) continue;

            MapPoint& mp_cand = mLocalMap[j];

            // 1. 空间欧氏距离校验
            float dx = mp_curr.pos_world.x - mp_cand.pos_world.x;
            float dy = mp_curr.pos_world.y - mp_cand.pos_world.y;
            float dz = mp_curr.pos_world.z - mp_cand.pos_world.z;
            float dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq > dist_th * dist_th) continue;

            // 2. 外观描述子校验
            int dist = cv::norm(mp_curr.descriptor, mp_cand.descriptor, cv::NORM_HAMMING);
            if (dist < best_dist) {
                best_dist = dist;
                best_match_idx = j;
            }
        }

        // 3. 执行融合
        if (best_match_idx >= 0 && best_dist < TH_HIGH) {
            // 发现老点！将当前关键帧的观测指针“移花接木”到老点上
            pKF->map_point_indices[i] = best_match_idx;
            
            // 增加老点的观测权重
            mLocalMap[best_match_idx].n_observed++;
            
            // 将当前这个多余的“重影”点打上坏点标记，等待清理
            mp_curr.is_bad = true; 
        } else {
            // 这是一个全新的、合法的独立点
            mp_curr.n_observed++;
            mp_curr.n_visible++;
        }
    }
}

// ==========================================
// 骨架：地图点剔除
// ==========================================
void LocalMapping::CullLocalMap() {
    std::unique_lock<std::shared_mutex> lock(mMapMutex);

    for (size_t i = 0; i < mLocalMap.size(); ++i) {
        MapPoint& mp = mLocalMap[i];
        if (mp.is_bad) continue; // 已经是坏点就不管了

        // 增加点的理论可见次数 (假设每次调用 Cull 时点都在视野内，这里做简化处理)
        mp.n_visible++;

        // 核心规则：观测比例过低 (被匹配上的次数 / 理论上应该看到的次数)
        // 比如一个点存在了很久 (n_visible 很大)，但只被看到了一两次 (n_observed 很小)
        float observe_ratio = static_cast<float>(mp.n_observed) / mp.n_visible;
        
        // 如果观测比例低于 25%，说明这是个极其不稳定的点 (可能是动态车、树叶摇晃、或者误匹配产生的飞点)
        if (observe_ratio < 0.25f && mp.n_visible > 3) {
            mp.is_bad = true;
        }
    }
    
    // 注意：我们绝不在这里调用 mLocalMap.erase()，防止索引雪崩！
}