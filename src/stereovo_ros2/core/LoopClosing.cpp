#include "core/LoopClosing.h"
#include "utils/Parameters.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>

// ==================== Ceres 位姿图残差结构体 ====================
struct PoseGraphErrorTerm {
    PoseGraphErrorTerm(const Eigen::Isometry3d& T_ij) {
        // 预先转为 Sophus::SE3d 避免在 Jet 模板中做隐式转换
        Eigen::Quaterniond q(T_ij.rotation());
        Eigen::Vector3d t = T_ij.translation();
        T_ij_meas_ = Sophus::SE3d(q, t);
    }

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_i(pose_i);
        Eigen::Map<const Eigen::Quaternion<T>> q_i(pose_i + 3);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_j(pose_j);
        Eigen::Map<const Eigen::Quaternion<T>> q_j(pose_j + 3);

        Sophus::SE3<T> T_i(q_i, t_i);
        Sophus::SE3<T> T_j(q_j, t_j);
        
        // 类型安全转型
        Sophus::SE3<T> T_ij_meas = T_ij_meas_.template cast<T>();

        Sophus::SE3<T> T_ij_est = T_i.inverse() * T_j;
        
        Eigen::Matrix<T, 6, 1> error = (T_ij_meas.inverse() * T_ij_est).log();

        for (int k = 0; k < 6; ++k) {
            residuals[k] = error(k);
        }
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Isometry3d& T_ij) {
        return new ceres::AutoDiffCostFunction<PoseGraphErrorTerm, 6, 7, 7>(
            new PoseGraphErrorTerm(T_ij));
    }

    Sophus::SE3d T_ij_meas_;
};

// ==================== 2. LoopClosure 方法实现 ====================

LoopClosure::LoopClosure(std::shared_ptr<Map> pMap, const std::string &vocPath)
    : mpMap(pMap), mIsRunning(true)
{
    if (!vocPath.empty()) {
        mVocabulary.load(vocPath);
        mDatabase.setVocabulary(mVocabulary, false, 0);
    }
    mLoopThread = std::thread(&LoopClosure::LoopClosureThread, this);
}

LoopClosure::~LoopClosure()
{
    {
        std::unique_lock<std::mutex> lock(mMutexQueue);
        mIsRunning = false;
    }
    mCondQueue.notify_all();
    if (mLoopThread.joinable())
        mLoopThread.join();
}

void LoopClosure::PushKeyFrame(std::shared_ptr<KeyFrame> pKF)
{
    std::unique_lock<std::mutex> lock(mMutexQueue);
    mKeyFrameQueue.push(pKF);
    mCondQueue.notify_one();
}

void LoopClosure::LoopClosureThread()
{
    while (true)
    {
        std::shared_ptr<KeyFrame> pKF = nullptr;
        {
            std::unique_lock<std::mutex> lock(mMutexQueue);
            mCondQueue.wait(lock, [this] { return !mKeyFrameQueue.empty() || !mIsRunning; });

            if (!mIsRunning && mKeyFrameQueue.empty())
                break;

            pKF = mKeyFrameQueue.front();
            mKeyFrameQueue.pop();
        }

        if (!pKF) continue;

        ExtractORBAndTriangulate(pKF);

        std::shared_ptr<KeyFrame> matchedKF = nullptr;
        Eigen::Isometry3d T_cur_match = Eigen::Isometry3d::Identity();

        bool hasLoop = DetectLoop(pKF, matchedKF, T_cur_match);

        {
            std::unique_lock<std::mutex> lockDB(mMutexDB);
            if (hasLoop && matchedKF)
            {
                pKF->mmLoopEdges[matchedKF->mId] = T_cur_match;
                std::cout << "[LoopClosure] 回环检测成功! 当前帧: " << pKF->mId 
                          << " <---> 匹配帧: " << matchedKF->mId << std::endl;

                RunGlobalPoseGraphOptimization();
            }
            
            mDatabase.add(pKF->mBowVec);
            mmKeyFramesDB[pKF->mId] = pKF;
        }
    }
}

void LoopClosure::ExtractORBAndTriangulate(std::shared_ptr<KeyFrame> pKF)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
    orb->detectAndCompute(pKF->mImgLeft, cv::noArray(), pKF->mvOrbKeysLeft, pKF->mDescriptorsLeft);

    std::vector<cv::Point2f> leftPts, rightPts;
    for (const auto& kp : pKF->mvOrbKeysLeft) leftPts.push_back(kp.pt);

    if (leftPts.empty()) return;

    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(pKF->mImgLeft, pKF->mImgRight, leftPts, rightPts, status, err, cv::Size(21, 21), 3);

    mVocabulary.transform(pKF->mDescriptorsLeft, pKF->mBowVec);

    double fx = Parameters::fx, fy = Parameters::fy, cx = Parameters::cx, cy = Parameters::cy;
    double baseline = std::abs(Parameters::body_T_cam1(0, 3) - Parameters::body_T_cam0(0, 3));
    if (baseline < 1e-4) baseline = 0.53715;

    pKF->mvMapPoints3D.resize(pKF->mvOrbKeysLeft.size(), cv::Point3f(0, 0, 0));

    for (size_t i = 0; i < leftPts.size(); ++i) {
        if (status[i]) {
            double disp = leftPts[i].x - rightPts[i].x;
            if (disp > 1.0) {
                double depth = (fx * baseline) / disp;
                double X = (leftPts[i].x - cx) * depth / fx;
                double Y = (leftPts[i].y - cy) * depth / fy;
                pKF->mvMapPoints3D[i] = cv::Point3f(X, Y, depth);
            }
        }
    }
}

bool LoopClosure::DetectLoop(std::shared_ptr<KeyFrame> pKF, std::shared_ptr<KeyFrame> &matchedKF, Eigen::Isometry3d &T_cur_match)
{
    if (mmKeyFramesDB.size() < 10) return false;

    DBoW3::QueryResults ret;
    mDatabase.query(pKF->mBowVec, ret, 4);

    unsigned long bestMatchId = 0;
    double maxScore = 0.05;

    for (const auto& result : ret) {
        if (pKF->mId > 10 && result.Id < (pKF->mId - 10)) {
            if (result.Score > maxScore) {
                maxScore = result.Score;
                bestMatchId = result.Id;
            }
        }
    }

    if (bestMatchId == 0 || mmKeyFramesDB.find(bestMatchId) == mmKeyFramesDB.end())
        return false;

    matchedKF = mmKeyFramesDB[bestMatchId];

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(pKF->mDescriptorsLeft, matchedKF->mDescriptorsLeft, matches);

    std::vector<cv::Point3f> pts3D;
    std::vector<cv::Point2f> pts2D;

    for (const auto& m : matches) {
        cv::Point3f p3d = matchedKF->mvMapPoints3D[m.trainIdx];
        if (p3d.z > 0.1) {
            pts3D.push_back(p3d);
            pts2D.push_back(pKF->mvOrbKeysLeft[m.queryIdx].pt);
        }
    }

    if (pts3D.size() < 12) return false;

    cv::Mat K = (cv::Mat_<double>(3, 3) << Parameters::fx, 0, Parameters::cx, 0, Parameters::fy, Parameters::cy, 0, 0, 1);
    cv::Mat rvec, tvec;
    std::vector<int> inliers;

    bool pnpOk = cv::solvePnPRansac(pts3D, pts2D, K, cv::noArray(), rvec, tvec, false, 100, 2.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);

    if (pnpOk && inliers.size() >= 10) {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        Eigen::Matrix3d eigenR;
        Eigen::Vector3d eigenT;
        cv::cv2eigen(R, eigenR);
        cv::cv2eigen(tvec, eigenT);

        T_cur_match.linear() = eigenR;
        T_cur_match.translation() = eigenT;
        return true;
    }

    return false;
}

void LoopClosure::RunGlobalPoseGraphOptimization()
{
    ceres::Problem problem;

    std::map<unsigned long, std::vector<double>> poseMap;

    for (auto& pair : mmKeyFramesDB) {
        auto kf = pair.second;
        Eigen::Isometry3d Twc = kf->GetPose();
        Eigen::Vector3d t = Twc.translation();
        Eigen::Quaterniond q(Twc.rotation());

        poseMap[kf->mId] = {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
        problem.AddParameterBlock(poseMap[kf->mId].data(), 7);
        problem.SetParameterization(poseMap[kf->mId].data() + 3, new ceres::QuaternionParameterization());
    }

    if (!poseMap.empty()) {
        problem.SetParameterBlockConstant(poseMap.begin()->second.data());
    }

    auto it = mmKeyFramesDB.begin();
    auto prevIt = it;
    ++it;
    for (; it != mmKeyFramesDB.end(); ++prevIt, ++it) {
        Eigen::Isometry3d T_w_i = prevIt->second->GetPose();
        Eigen::Isometry3d T_w_j = it->second->GetPose();
        Eigen::Isometry3d T_ij = T_w_i.inverse() * T_w_j;

        ceres::CostFunction* cost_function = PoseGraphErrorTerm::Create(T_ij);
        problem.AddResidualBlock(cost_function, nullptr, poseMap[prevIt->first].data(), poseMap[it->first].data());
    }

    for (auto& pair : mmKeyFramesDB) {
        auto kf = pair.second;
        for (auto& loopEdge : kf->mmLoopEdges) {
            unsigned long matchedId = loopEdge.first;
            Eigen::Isometry3d T_loop = loopEdge.second;

            if (poseMap.find(matchedId) != poseMap.end()) {
                ceres::CostFunction* cost_function = PoseGraphErrorTerm::Create(T_loop);
                problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), poseMap[kf->mId].data(), poseMap[matchedId].data());
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    for (auto& pair : mmKeyFramesDB) {
        auto kf = pair.second;
        auto& p = poseMap[kf->mId];
        Eigen::Vector3d t(p[0], p[1], p[2]);
        Eigen::Quaterniond q(p[6], p[3], p[4], p[5]);

        Eigen::Isometry3d TwcOpt = Eigen::Isometry3d::Identity();
        TwcOpt.linear() = q.toRotationMatrix();
        TwcOpt.translation() = t;
        kf->SetPose(TwcOpt);
    }
}