#include "tracking.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <thread>

using namespace std;
using namespace cv;

// 外部声明：链接 pose_optimizer.cpp 中的 BA 函数
extern void bundleAdjustment(const vector<cv::Point3f>& points_3d,
                             const vector<cv::Point2f>& points_2d,
                             const cv::Mat& K, cv::Mat& R, cv::Mat& t);

namespace {
    vector<Point2f> points2f_from_kps(const vector<KeyPoint>& kps) {
        vector<Point2f> pts; pts.reserve(kps.size());
        for (auto& kp : kps) pts.push_back(kp.pt);
        return pts;
    }
}

Tracking::Tracking(StereoVO* vo, std::shared_ptr<Map> pMap, LocalMapping* local_mapper, KeyframeSelector* selector)
    : mpVO(vo), mpMap(pMap), mpLocalMapper(local_mapper), mpSelector(selector), mbInitialized(false) {
    mGlobalPose = Sophus::SE3d();
    mLastDeltaT = Sophus::SE3d();
}

Sophus::SE3d Tracking::GrabImageStereo(const cv::Mat& img_curr_l, const cv::Mat& img_curr_r, int frame_id) {
    // ==========================================
    // 1. 系统初始化阶段 (第 0 帧)
    // ==========================================
    if (!mbInitialized) {
        thread t_init_l([&]() { mpVO->extractORBWithQuadTree(img_curr_l, mKpsPrevL, mDescPrevL, 2000); });
        thread t_init_r([&]() { mpVO->extractORBWithQuadTree(img_curr_r, mKpsPrevR, mDescPrevR, 2000); });
        t_init_l.join();
        t_init_r.join();

        vector<DMatch> stereo_matches;
        mpVO->matchStereoEpipolar(mKpsPrevL, mKpsPrevR, mDescPrevL, mDescPrevR, stereo_matches);

        mPts3dPrev.assign(mKpsPrevL.size(), Point3f(0, 0, 0));
        for (const auto& sm : stereo_matches) {
            int idx_l = sm.queryIdx;
            int idx_r = sm.trainIdx;
            double disparity = mKpsPrevL[idx_l].pt.x - mKpsPrevR[idx_r].pt.x;
            if (disparity > 0) {
                double z = (mpVO->fx * mpVO->baseline) / disparity;
                double x = (mKpsPrevL[idx_l].pt.x - mpVO->cx) * z / mpVO->fx;
                double y = (mKpsPrevL[idx_l].pt.y - mpVO->cy) * z / mpVO->fy;
                mPts3dPrev[idx_l] = Point3f(x, y, z);
            }
        }

        mPrevMpIndices.assign(mKpsPrevL.size(), -1);
        cv::Mat rvec_init = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat tvec_init = cv::Mat::zeros(3, 1, CV_64F);
        vector<Point2f> kps_prev_2d = points2f_from_kps(mKpsPrevL);

        auto first_kf = std::make_shared<Keyframe>(
            frame_id, rvec_init, tvec_init, kps_prev_2d, mPrevMpIndices,
            img_curr_l, mKpsPrevL, mKpsPrevR, mDescPrevL, mDescPrevR
        );
        
        mWindowKeyframes.push_back(first_kf);
        mpLastKeyframe = mWindowKeyframes.back();
        mpLocalMapper->InsertKeyFrame(first_kf);
        cout << "第一帧成功送入后端优化队列。" << endl;

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
    thread t_l([&]() { mpVO->extractORBWithQuadTree(img_curr_l, kps_curr_l, desc_curr_l, 2000); });
    thread t_r([&]() { mpVO->extractORBWithQuadTree(img_curr_r, kps_curr_r, desc_curr_r, 2000); });
    t_l.join();
    t_r.join();

    vector<DMatch> temporal_matches;
    mpVO->matchTemporalByProjection(mKpsPrevL, mPts3dPrev, mDescPrevL,
                                    kps_curr_l, desc_curr_l,
                                    mLastDeltaT, mpVO->K, temporal_matches, 15.0f);

    vector<Point3f> pts_3d_cam;
    vector<Point2f> pts_2d_curr;
    vector<DMatch> viz_matches;

    for (const auto& tm : temporal_matches) {
        int idx_prev = tm.queryIdx;
        int idx_curr = tm.trainIdx;
        pts_3d_cam.push_back(mPts3dPrev[idx_prev]);
        pts_2d_curr.push_back(kps_curr_l[idx_curr].pt);
        viz_matches.push_back(tm);
    }

    vector<int> curr_mp_indices(kps_curr_l.size(), -1);
    for (const auto &match : temporal_matches) {
        int prev_idx = match.queryIdx;
        int curr_idx = match.trainIdx;
        if (prev_idx < (int)mPrevMpIndices.size() && mPrevMpIndices[prev_idx] != -1) {
            curr_mp_indices[curr_idx] = mPrevMpIndices[prev_idx];
        }
    }

    bool tracking_success = false;
    cv::Mat rvec_curr, tvec_curr;

    if (pts_3d_cam.size() >= 10) {
        Mat rvec_pnp, tvec_pnp, inliers, R_pnp;
        solvePnPRansac(pts_3d_cam, pts_2d_curr, mpVO->K, Mat(), rvec_pnp, tvec_pnp, false, 100, 2.0, 0.99, inliers);
        if (!inliers.empty() && inliers.rows >= 15 && norm(tvec_pnp) < 3.0) {
            Rodrigues(rvec_pnp, R_pnp);
            bundleAdjustment(pts_3d_cam, pts_2d_curr, mpVO->K, R_pnp, tvec_pnp);
            if (norm(tvec_pnp) < 3.0) {
                Eigen::Matrix3d R_eigen;
                cv::cv2eigen(R_pnp, R_eigen);
                Eigen::Vector3d t_eigen(tvec_pnp.at<double>(0, 0), tvec_pnp.at<double>(1, 0), tvec_pnp.at<double>(2, 0));
                Sophus::SE3d T_curr_prev(R_eigen, t_eigen);

                mLastDeltaT = T_curr_prev;
                tracking_success = true;
                
                // 纯前端模式核心：世界坐标系 = 历史世界坐标系 * 帧间运动逆矩阵
                mGlobalPose = mGlobalPose * T_curr_prev.inverse();
            }
        }
    }

    if (!tracking_success) {
        cout << "⚠️ 警告 [帧 " << frame_id << "]: 触发位姿防暴走熔断！启用恒速模型！" << endl;
        mGlobalPose = mGlobalPose * mLastDeltaT.inverse();
    }

    // 提取当前绝对坐标供 Selector 使用
    Sophus::SE3d T_cw_curr = mGlobalPose.inverse();
    cv::Mat R_curr_mat;
    cv::eigen2cv(T_cw_curr.rotationMatrix(), R_curr_mat);
    cv::Rodrigues(R_curr_mat, rvec_curr);
    Eigen::Vector3d t_curr_eigen = T_cw_curr.translation();
    tvec_curr = (cv::Mat_<double>(3, 1) << t_curr_eigen(0), t_curr_eigen(1), t_curr_eigen(2));

    vector<int> curr_mp_indices_for_selector;
    for (int idx : curr_mp_indices) if (idx != -1) curr_mp_indices_for_selector.push_back(idx);
    vector<Point2f> curr_kps_2d = points2f_from_kps(kps_curr_l);

    // ==========================================
    // 3. 关键帧逻辑判断与推入
    // ==========================================
    bool is_keyframe = false;
    std::vector<Keyframe> raw_window_keyframes;
    for (auto& kf_ptr : mWindowKeyframes) raw_window_keyframes.push_back(*kf_ptr);
    is_keyframe = mpSelector->decide(rvec_curr, tvec_curr, curr_kps_2d, curr_mp_indices_for_selector,
                                     mpMap, mpLastKeyframe.get(), raw_window_keyframes, false);

    if (is_keyframe) {
        auto new_kf = std::make_shared<Keyframe>(
            frame_id, rvec_curr.clone(), tvec_curr.clone(), curr_kps_2d, curr_mp_indices,
            img_curr_l, kps_curr_l, kps_curr_r, desc_curr_l, desc_curr_r
        );
        mWindowKeyframes.push_back(new_kf);
        if (mWindowKeyframes.size() > 15) mWindowKeyframes.erase(mWindowKeyframes.begin());
        mpLastKeyframe = mWindowKeyframes.back();
        
        mpLocalMapper->InsertKeyFrame(new_kf);
        cout << "📸 [前端主线程] 关键帧已生成: 帧 " << frame_id << endl;
    }

    // 更新可视化图像
    Mat img_viz;
    drawMatches(mImgPrevL, mKpsPrevL, img_curr_l, kps_curr_l, viz_matches, img_viz,
                Scalar(0, 255, 0), Scalar(0, 0, 255), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    {
        lock_guard<mutex> lock(mMutexViz);
        mImgViz = img_viz.clone();
    }

    // ==========================================
    // 4. 为下一帧准备历史数据
    // ==========================================
    vector<DMatch> stereo_matches;
    mpVO->matchStereoEpipolar(kps_curr_l, kps_curr_r, desc_curr_l, desc_curr_r, stereo_matches);
    
    mPts3dPrev.assign(kps_curr_l.size(), Point3f(0, 0, 0));
    for (const auto& sm : stereo_matches) {
        int idx_l = sm.queryIdx;
        int idx_r = sm.trainIdx;
        double disparity = kps_curr_l[idx_l].pt.x - kps_curr_r[idx_r].pt.x;
        if (disparity > 0) {
            double z = (mpVO->fx * mpVO->baseline) / disparity;
            double x = (kps_curr_l[idx_l].pt.x - mpVO->cx) * z / mpVO->fx;
            double y = (kps_curr_l[idx_l].pt.y - mpVO->cy) * z / mpVO->fy;
            mPts3dPrev[idx_l] = Point3f(x, y, z);
        }
    }

    mImgPrevL = img_curr_l.clone();
    mKpsPrevL = kps_curr_l;
    mKpsPrevR = kps_curr_r;
    mDescPrevL = desc_curr_l.clone();
    mDescPrevR = desc_curr_r.clone();
    mPrevMpIndices = std::move(curr_mp_indices);

    return mGlobalPose;
}

cv::Mat Tracking::GetVizImage() {
    lock_guard<mutex> lock(mMutexViz);
    return mImgViz.empty() ? cv::Mat() : mImgViz.clone();
}