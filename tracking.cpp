// =========================================================================
// ====== DEBUG REFACTOR CODE START ======
// 文件名: tracking.cpp
// 修改点: 1. 在断开全局索引的同时，物理清除 mPts3dPrev，彻底封死自行车的“转世连接”
//        2. 将可视化渲染逻辑彻底移到函数最末尾，确保 MaxResp 不再被任何中间算子抹零
// =========================================================================

#include "tracking.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <thread>
#include "math_utils.h"

using namespace std;
using namespace cv;

extern std::vector<bool> bundleAdjustment(const vector<cv::Point3f> &points_3d,
                                          const vector<cv::Point2f> &points_2d,
                                          const cv::Mat &K, cv::Mat &R, cv::Mat &t);

namespace
{
    vector<Point2f> points2f_from_kps(const vector<KeyPoint> &kps)
    {
        vector<Point2f> pts;
        pts.reserve(kps.size());
        for (auto &kp : kps)
            pts.push_back(kp.pt);
        return pts;
    }
}

Tracking::Tracking(StereoVO *vo, std::shared_ptr<Map> pMap, LocalMapping *local_mapper, KeyframeSelector *selector)
    : mpVO(vo), mpMap(pMap), mpLocalMapper(local_mapper), mpSelector(selector), mbInitialized(false)
{
    mGlobalPose = Sophus::SE3d();
    mLastDeltaT = Sophus::SE3d();
}

Sophus::SE3d Tracking::GrabImageStereo(const cv::Mat &img_curr_l, const cv::Mat &img_curr_r, int frame_id)
{
    // ==========================================
    // 1. 系统初始化阶段 (第 0 帧)
    // ==========================================
    if (!mbInitialized)
    {
        thread t_init_l([&]()
                        { mpVO->extractORBWithQuadTree(img_curr_l, mKpsPrevL, mDescPrevL, 2000); });
        thread t_init_r([&]()
                        { mpVO->extractORBWithQuadTree(img_curr_r, mKpsPrevR, mDescPrevR, 2000); });
        t_init_l.join();
        t_init_r.join();

        vector<DMatch> stereo_matches;
        mpVO->matchStereoEpipolar(mKpsPrevL, mKpsPrevR, mDescPrevL, mDescPrevR, stereo_matches);

        mPts3dPrev.assign(mKpsPrevL.size(), Point3f(0, 0, 0));
        mPrevMpIndices.assign(mKpsPrevL.size(), -1);

        size_t start_global_idx = mpMap->GetGlobalMapPointsSize();
        int valid_added_count = 0;

        for (const auto &sm : stereo_matches)
        {
            int idx_l = sm.queryIdx;
            int idx_r = sm.trainIdx;
            double disparity = mKpsPrevL[idx_l].pt.x - mKpsPrevR[idx_r].pt.x;
            if (disparity > 0)
            {
                double z = (mpVO->fx * mpVO->baseline) / disparity;
                if (z < 0.5 || z > 80.0)
                    continue;

                double x = (mKpsPrevL[idx_l].pt.x - mpVO->cx) * z / mpVO->fx;
                double y = (mKpsPrevL[idx_l].pt.y - mpVO->cy) * z / mpVO->fy;

                Point3f pt_c(x, y, z);
                mPts3dPrev[idx_l] = pt_c;

                auto pMP = std::make_shared<MapPoint>();
                pMP->pos_world = pt_c;
                pMP->descriptor = mDescPrevL.row(idx_l).clone();
                pMP->is_bad = false;
                pMP->n_observed = 1;
                pMP->n_visible = 1;

                mpMap->AddMapPoint(pMP);
                mPrevMpIndices[idx_l] = start_global_idx + valid_added_count;
                valid_added_count++;
            }
        }

        cv::Mat rvec_init = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat tvec_init = cv::Mat::zeros(3, 1, CV_64F);
        vector<Point2f> kps_prev_2d = points2f_from_kps(mKpsPrevL);

        auto first_kf = std::make_shared<Keyframe>(
            frame_id, rvec_init, tvec_init, kps_prev_2d, mPrevMpIndices,
            img_curr_l, mKpsPrevL, mKpsPrevR, mDescPrevL, mDescPrevR);

        mWindowKeyframes.push_back(first_kf);
        mpLastKeyframe = mWindowKeyframes.back();
        mpLocalMapper->InsertKeyFrame(first_kf);

        mTrajectoryRecords.push_back({frame_id, frame_id, Sophus::SE3d()});

        mImgPrevL = img_curr_l.clone();
        mImgPrevR = img_curr_r.clone();
        mbInitialized = true;
        return mGlobalPose;
    }

    // ==========================================
    // 2. 正常追踪阶段 (第 1 帧及以后)
    // ==========================================
    vector<KeyPoint> kps_curr_l, kps_curr_r;
    Mat desc_curr_l, desc_curr_r;
    thread t_l([&]()
               { mpVO->extractORBWithQuadTree(img_curr_l, kps_curr_l, desc_curr_l, 2000); });
    thread t_r([&]()
               { mpVO->extractORBWithQuadTree(img_curr_r, kps_curr_r, desc_curr_r, 2000); });
    t_l.join();
    t_r.join();

    // 💡 高亮核心备份：在最干净的时候把响应值全部锁在主线程栈里
    std::vector<float> original_response_holder(kps_curr_l.size(), 10.0f);
    for (size_t i = 0; i < kps_curr_l.size(); ++i)
    {
        original_response_holder[i] = kps_curr_l[i].response;
    }

    if (mpLastKeyframe)
    {
        auto global_kfs = mpMap->GetAllKeyframes();
        std::shared_lock<std::shared_mutex> lock(mpMap->mMutexMap);
        for (const auto &pKF : global_kfs)
        {
            if (pKF && pKF->id == mpLastKeyframe->id)
            {
                cv::Mat R_mat;
                cv::Rodrigues(pKF->rvec, R_mat);
                Eigen::Matrix3d R_eigen;
                cv::cv2eigen(R_mat, R_eigen);
                Eigen::Vector3d t_eigen(pKF->tvec.at<double>(0, 0), pKF->tvec.at<double>(1, 0), pKF->tvec.at<double>(2, 0));
                Sophus::SE3d T_cw_optimized(R_eigen, t_eigen);
                mGlobalPose = T_cw_optimized.inverse();
                break;
            }
        }
    }

    vector<DMatch> temporal_matches;
    {
        double fx_ = mpVO->fx;
        double fy_ = mpVO->fy;
        double cx_ = mpVO->cx;
        double cy_ = mpVO->cy;
        const double search_radius2 = 15.0 * 15.0;
        const int TH_LOW = 50;

        Eigen::Matrix3d R_curr_prev = mLastDeltaT.rotationMatrix();
        Eigen::Vector3d t_curr_prev = mLastDeltaT.translation();

        for (size_t i = 0; i < mPts3dPrev.size(); ++i)
        {
            if (i >= (size_t)mDescPrevL.rows || mPts3dPrev[i].z <= 0)
                continue;

            Eigen::Vector3d P_prev(mPts3dPrev[i].x, mPts3dPrev[i].y, mPts3dPrev[i].z);
            Eigen::Matrix<double, 2, 1> uv_pred;

            if (!vo_math::ProjectWorldPointToPixel(R_curr_prev, t_curr_prev, P_prev, fx_, fy_, cx_, cy_, uv_pred))
            {
                continue;
            }

            int best_dist = 256;
            int best_idx = -1;

            for (size_t j = 0; j < kps_curr_l.size(); ++j)
            {
                if (j >= (size_t)desc_curr_l.rows)
                    continue;

                double du = kps_curr_l[j].pt.x - uv_pred.x();
                double dv = kps_curr_l[j].pt.y - uv_pred.y();
                if (du * du + dv * dv > search_radius2)
                    continue;

                int dist = vo_math::ComputeHammingDistance(mDescPrevL.ptr(i), desc_curr_l.ptr(j));
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_idx = j;
                }
            }

            if (best_idx >= 0 && best_dist < TH_LOW)
            {
                temporal_matches.push_back(DMatch(i, best_idx, best_dist));
            }
        }
    }

    vector<Point3f> pts_3d_world;
    vector<Point2f> pts_2d_curr;
    vector<DMatch> viz_matches;
    Sophus::SE3d T_w_prev = mGlobalPose;

    for (const auto &tm : temporal_matches)
    {
        int idx_prev = tm.queryIdx;
        int idx_curr = tm.trainIdx;
        Point3f pt_c = mPts3dPrev[idx_prev];
        Eigen::Vector3d P_c(pt_c.x, pt_c.y, pt_c.z);
        Eigen::Vector3d P_w = T_w_prev * P_c;
        pts_3d_world.push_back(Point3f(P_w.x(), P_w.y(), P_w.z()));
        pts_2d_curr.push_back(kps_curr_l[idx_curr].pt);
        viz_matches.push_back(tm);
    }

    vector<int> curr_mp_indices(kps_curr_l.size(), -1);
    for (const auto &match : temporal_matches)
    {
        int prev_idx = match.queryIdx;
        int curr_idx = match.trainIdx;
        if (prev_idx < (int)mPrevMpIndices.size() && mPrevMpIndices[prev_idx] != -1)
        {
            curr_mp_indices[curr_idx] = mPrevMpIndices[prev_idx];
        }
    }

    bool tracking_success = false;
    cv::Mat rvec_curr, tvec_curr;

    if (pts_3d_world.size() >= 10)
    {
        Mat rvec_pnp, tvec_pnp, inliers, R_pnp;
        solvePnPRansac(pts_3d_world, pts_2d_curr, mpVO->K, Mat(), rvec_pnp, tvec_pnp, false, 100, 2.0, 0.99, inliers);
        if (!inliers.empty() && inliers.rows >= 15)
        {
            Rodrigues(rvec_pnp, R_pnp);

            // 执行后端 BA 优化并拿到内点表
            std::vector<bool> is_static_inlier = bundleAdjustment(pts_3d_world, pts_2d_curr, mpVO->K, R_pnp, tvec_pnp);

            // 🌟🌟🌟 终极物理清洗：彻底斩断移动车辆的所有传导线 🌟🌟🌟
            for (size_t i = 0; i < is_static_inlier.size(); ++i)
            {
                if (!is_static_inlier[i])
                {
                    int curr_idx_l = temporal_matches[i].trainIdx; // 当前帧的特征点索引
                    int prev_idx_l = temporal_matches[i].queryIdx; // 上一帧的特征点索引

                    if (curr_idx_l >= 0 && curr_idx_l < (int)curr_mp_indices.size())
                    {
                        curr_mp_indices[curr_idx_l] = -1; // 1. 断开与历史大地图地图点的连接
                    }
                    if (prev_idx_l >= 0 && prev_idx_l < (int)mPts3dPrev.size())
                    {
                        mPts3dPrev[prev_idx_l] = Point3f(0, 0, 0); // 2. 核心补丁：强行将上一帧的3D坐标抹零，彻底阻止自行车在前线转世
                    }
                }
            }

            Eigen::Matrix3d R_eigen;
            cv::cv2eigen(R_pnp, R_eigen);
            Eigen::Vector3d t_eigen(tvec_pnp.at<double>(0, 0), tvec_pnp.at<double>(1, 0), tvec_pnp.at<double>(2, 0));
            Sophus::SE3d T_cw_curr(R_eigen, t_eigen);
            Sophus::SE3d T_curr_prev = T_cw_curr * T_w_prev;

            if (T_curr_prev.translation().norm() < 3.0)
            {
                mGlobalPose = T_cw_curr.inverse();
                mLastDeltaT = T_curr_prev;
                tracking_success = true;
            }
        }
    }

    if (!tracking_success)
    {
        mGlobalPose = mGlobalPose * mLastDeltaT.inverse();
    }

    if (mpLastKeyframe)
    {
        Sophus::SE3d T_w_curr = mGlobalPose;
        Sophus::SE3d T_w_kf = T_w_prev;

        Sophus::SE3d T_kf_curr = T_w_kf.inverse() * T_w_curr;
        mTrajectoryRecords.push_back({frame_id, mpLastKeyframe->id, T_kf_curr});
    }

    Sophus::SE3d T_cw_curr = mGlobalPose.inverse();
    cv::Mat R_curr_mat;
    cv::eigen2cv(T_cw_curr.rotationMatrix(), R_curr_mat);
    cv::Rodrigues(R_curr_mat, rvec_curr);
    Eigen::Vector3d t_curr_eigen = T_cw_curr.translation();
    tvec_curr = (cv::Mat_<double>(3, 1) << t_curr_eigen(0), t_curr_eigen(1), t_curr_eigen(2));

    vector<int> curr_mp_indices_for_selector;
    for (int idx : curr_mp_indices)
        if (idx != -1)
            curr_mp_indices_for_selector.push_back(idx);
    vector<Point2f> curr_kps_2d = points2f_from_kps(kps_curr_l);

    bool is_keyframe = false;
    std::vector<Keyframe> raw_window_keyframes;
    for (auto &kf_ptr : mWindowKeyframes)
        raw_window_keyframes.push_back(*kf_ptr);
    is_keyframe = mpSelector->decide(rvec_curr, tvec_curr, curr_kps_2d, curr_mp_indices_for_selector,
                                     mpMap, mpLastKeyframe.get(), raw_window_keyframes, false);

    if (is_keyframe)
    {
        auto new_kf = std::make_shared<Keyframe>(
            frame_id, rvec_curr.clone(), tvec_curr.clone(), curr_kps_2d, curr_mp_indices,
            img_curr_l, kps_curr_l, kps_curr_r, desc_curr_l, desc_curr_r);
        mWindowKeyframes.push_back(new_kf);
        if (mWindowKeyframes.size() > 15)
            mWindowKeyframes.erase(mWindowKeyframes.begin());
        mpLastKeyframe = mWindowKeyframes.back();
        mpLocalMapper->InsertKeyFrame(new_kf);
        cout << "📸 [前端主线程] 关键帧已生成: 帧 " << frame_id << endl;
    }

    // 执行下一帧必备的双目融合提取
    vector<DMatch> stereo_matches;
    mpVO->matchStereoEpipolar(kps_curr_l, kps_curr_r, desc_curr_l, desc_curr_r, stereo_matches);
    mPts3dPrev.assign(kps_curr_l.size(), Point3f(0, 0, 0));
    for (const auto &sm : stereo_matches)
    {
        int idx_l = sm.queryIdx;
        int idx_r = sm.trainIdx;
        double disparity = kps_curr_l[idx_l].pt.x - kps_curr_r[idx_r].pt.x;
        if (disparity > 0)
        {
            double z = (mpVO->fx * mpVO->baseline) / disparity;
            double x = (kps_curr_l[idx_l].pt.x - mpVO->cx) * z / mpVO->fx;
            double y = (kps_curr_l[idx_l].pt.y - mpVO->cy) * z / mpVO->fy;
            mPts3dPrev[idx_l] = Point3f(x, y, z);
        }
    }

    // ==== 🌟🌟🌟 核心修改点：把图片渲染彻底放到赋值大循环最后！🌟🌟🌟 ====
    Mat img_viz;
    if (img_curr_l.channels() == 1)
    {
        cv::cvtColor(img_curr_l, img_viz, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img_viz = img_curr_l.clone();
    }

    if (!kps_curr_l.empty())
    {
        // 💡 强行刷回保存在栈里的最纯净 response，杜绝中间被任何 OpenCV 函数篡改抹零
        for (size_t i = 0; i < kps_curr_l.size(); ++i)
        {
            kps_curr_l[i].response = original_response_holder[i];
        }

        float min_resp = kps_curr_l[0].response;
        float max_resp = kps_curr_l[0].response;
        for (const auto &kp : kps_curr_l)
        {
            if (kp.response < min_resp)
                min_resp = kp.response;
            if (kp.response > max_resp)
                max_resp = kp.response;
        }
        float resp_range = max_resp - min_resp;
        if (resp_range < 1e-4f)
            resp_range = 1.0f;

        for (const auto &kp : kps_curr_l)
        {
            float ratio = (kp.response - min_resp) / resp_range;

            int r = static_cast<int>(ratio * 255);
            int g = static_cast<int>((1.0f - std::abs(ratio - 0.5f) * 2.0f) * 255);
            int b = static_cast<int>((1.0f - ratio) * 255);
            cv::Scalar color(b, g, r); // BGR

            cv::circle(img_viz, kp.pt, 4, color, -1);
            if (ratio > 0.8f)
            {
                cv::circle(img_viz, kp.pt, 6, cv::Scalar(0, 255, 255), 1);
            }
        }

        string debug_text = "Frame: " + to_string(frame_id) +
                            " | Features: " + to_string(kps_curr_l.size()) +
                            " | MaxResp: " + to_string(static_cast<int>(max_resp));
        cv::putText(img_viz, debug_text, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }

    {
        lock_guard<mutex> lock(mMutexViz);
        mImgViz = img_viz.clone();
    }

    mImgPrevL = img_curr_l.clone();
    mKpsPrevL = kps_curr_l;
    mKpsPrevR = kps_curr_r;
    mDescPrevL = desc_curr_l.clone();
    mDescPrevR = desc_curr_r.clone();
    mPrevMpIndices = std::move(curr_mp_indices);

    return mGlobalPose;
}

cv::Mat Tracking::GetVizImage()
{
    lock_guard<mutex> lock(mMutexViz);
    return mImgViz.empty() ? cv::Mat() : mImgViz.clone();
}