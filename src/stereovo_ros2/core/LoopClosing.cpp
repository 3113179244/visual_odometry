#include "core/LoopClosing.h"
#include "core/Tracking.h"
#include "backend/Optimizer.h"
#include "utils/Parameters.h"
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

// ----------------------------------------------------------------------
// ORBSLAM2 格式 Vocabulary 解析派生类
// ----------------------------------------------------------------------
class ORBSLAM2Vocabulary : public DBoW3::Vocabulary
{
public:
    bool loadFromORBSLAM2Text(const std::string &filename)
    {
        std::cout << "[VocLoader] 开始加载字典文件: " << filename << std::endl;
        std::ifstream f(filename.c_str());
        if (!f.is_open())
        {
            std::cerr << "\033[31m[VocLoader-Error] 无法打开词袋文件！\033[0m" << std::endl;
            return false;
        }

        m_words.clear();
        m_nodes.clear();

        std::string line;
        if (!std::getline(f, line))
            return false;

        std::stringstream ss(line);
        int s, w;
        ss >> m_k >> m_L >> s >> w;

        m_scoring = static_cast<DBoW3::ScoringType>(s);
        m_weighting = static_cast<DBoW3::WeightingType>(w);
        createScoringObject();

        struct TempNode
        {
            DBoW3::NodeId id = 0;
            DBoW3::NodeId parent = 0;
            DBoW3::WordValue weight = 0.0;
            std::vector<uchar> desc_data;
            bool valid = false;
        };

        std::vector<TempNode> temp_nodes;
        temp_nodes.reserve(150000);

        while (std::getline(f, line))
        {
            if (line.empty())
                continue;
            std::stringstream ss_node(line);

            DBoW3::NodeId id, parent;
            DBoW3::WordValue weight;
            if (!(ss_node >> id >> parent >> weight))
                continue;

            if (id >= temp_nodes.size())
            {
                temp_nodes.resize(id + 1);
            }

            TempNode &tnode = temp_nodes[id];
            tnode.id = id;
            tnode.parent = parent;
            tnode.weight = weight;
            tnode.desc_data.resize(32);
            for (int i = 0; i < 32; ++i)
            {
                int val = 0;
                ss_node >> val;
                tnode.desc_data[i] = static_cast<uchar>(val);
            }
            tnode.valid = true;
        }

        if (temp_nodes.empty())
            return false;

        m_nodes.resize(temp_nodes.size());
        for (size_t i = 0; i < m_nodes.size(); ++i)
        {
            m_nodes[i].id = i;
            m_nodes[i].parent = 0;
            m_nodes[i].weight = 0;
            m_nodes[i].descriptor = cv::Mat::zeros(1, 32, CV_8UC1);
        }

        for (size_t i = 0; i < temp_nodes.size(); ++i)
        {
            if (temp_nodes[i].valid)
            {
                m_nodes[i].parent = temp_nodes[i].parent;
                m_nodes[i].weight = temp_nodes[i].weight;
                cv::Mat(1, 32, CV_8UC1, temp_nodes[i].desc_data.data()).copyTo(m_nodes[i].descriptor);
            }
        }

        for (size_t i = 0; i < m_nodes.size(); ++i)
        {
            if (temp_nodes[i].valid && m_nodes[i].id != 0)
            {
                DBoW3::NodeId parent = m_nodes[i].parent;
                if (parent < m_nodes.size())
                {
                    m_nodes[parent].children.push_back(i);
                }
            }
        }

        for (size_t i = 0; i < m_nodes.size(); ++i)
        {
            if (i != 0 && temp_nodes[i].valid && m_nodes[i].children.empty())
            {
                m_nodes[i].word_id = m_words.size();
                m_words.push_back(&m_nodes[i]);
            }
        }

        std::cout << "\033[32m[VocLoader] 字典树构建成功！节点数: " << m_nodes.size()
                  << " | 单词数(Words): " << m_words.size() << "\033[0m" << std::endl;

        return !m_words.empty();
    }

    virtual void transform(const cv::Mat &features, DBoW3::BowVector &v) const override
    {
        v.clear();
        if (m_nodes.empty() || features.empty())
            return;

        cv::Mat feat = features;
        if (feat.type() != CV_8UC1)
        {
            feat.convertTo(feat, CV_8UC1);
        }

        for (int r = 0; r < feat.rows; ++r)
        {
            const uchar *feat_ptr = feat.ptr<uchar>(r);
            DBoW3::NodeId curr_id = 0;

            while (!m_nodes[curr_id].children.empty())
            {
                const auto &children = m_nodes[curr_id].children;
                int min_dist = 999999;
                DBoW3::NodeId best_child = children[0];

                for (DBoW3::NodeId child_id : children)
                {
                    if (child_id >= m_nodes.size())
                        continue;
                    const uchar *node_desc = m_nodes[child_id].descriptor.ptr<uchar>(0);

                    const uint64_t *p1 = reinterpret_cast<const uint64_t *>(feat_ptr);
                    const uint64_t *p2 = reinterpret_cast<const uint64_t *>(node_desc);
                    int dist = __builtin_popcountll(p1[0] ^ p2[0]) +
                               __builtin_popcountll(p1[1] ^ p2[1]) +
                               __builtin_popcountll(p1[2] ^ p2[2]) +
                               __builtin_popcountll(p1[3] ^ p2[3]);

                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        best_child = child_id;
                    }
                }
                curr_id = best_child;
            }

            DBoW3::WordId word_id = m_nodes[curr_id].word_id;
            DBoW3::WordValue weight = m_nodes[curr_id].weight;

            v[word_id] += weight;
        }

        v.normalize(DBoW3::L1);
    }

    virtual void transform(const std::vector<cv::Mat> &features, DBoW3::BowVector &v, DBoW3::FeatureVector &fv, int levelsup) const override
    {
        v.clear();
        fv.clear();
        if (m_nodes.empty() || features.empty())
            return;

        for (size_t i = 0; i < features.size(); ++i)
        {
            if (features[i].empty())
                continue;
            cv::Mat feat = features[i];
            if (feat.type() != CV_8UC1)
                feat.convertTo(feat, CV_8UC1);

            const uchar *feat_ptr = feat.ptr<uchar>(0);
            DBoW3::NodeId curr_id = 0;

            while (!m_nodes[curr_id].children.empty())
            {
                const auto &children = m_nodes[curr_id].children;
                int min_dist = 999999;
                DBoW3::NodeId best_child = children[0];

                for (DBoW3::NodeId child_id : children)
                {
                    if (child_id >= m_nodes.size())
                        continue;
                    const uchar *node_desc = m_nodes[child_id].descriptor.ptr<uchar>(0);

                    const uint64_t *p1 = reinterpret_cast<const uint64_t *>(feat_ptr);
                    const uint64_t *p2 = reinterpret_cast<const uint64_t *>(node_desc);
                    int dist = __builtin_popcountll(p1[0] ^ p2[0]) +
                               __builtin_popcountll(p1[1] ^ p2[1]) +
                               __builtin_popcountll(p1[2] ^ p2[2]) +
                               __builtin_popcountll(p1[3] ^ p2[3]);

                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        best_child = child_id;
                    }
                }
                curr_id = best_child;
            }

            DBoW3::WordId word_id = m_nodes[curr_id].word_id;
            DBoW3::WordValue weight = m_nodes[curr_id].weight;

            v[word_id] += weight;
            fv.addFeature(curr_id, i);
        }

        v.normalize(DBoW3::L1);
    }
};

static bool LoadORBSLAM2Vocabulary(const std::string &vocPath, std::shared_ptr<DBoW3::Vocabulary> &pVoc)
{
    auto voc = std::make_shared<ORBSLAM2Vocabulary>();
    if (voc->loadFromORBSLAM2Text(vocPath))
    {
        pVoc = voc;
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------
// LoopClosing 实现
// ----------------------------------------------------------------------

LoopClosing::LoopClosing(std::shared_ptr<Map> pMap, const std::string &vocPath)
    : mpMap(pMap), mIsRunning(true)
{
    std::cout << "\033[33m[LoopClosing] 正在加载词袋文件: " << vocPath << " ...\033[0m" << std::endl;

    bool succ = LoadORBSLAM2Vocabulary(vocPath, mpVocabulary);

    if (succ && mpVocabulary && !mpVocabulary->empty())
    {
        try
        {
            mpDatabase = std::make_unique<DBoW3::Database>(*mpVocabulary, false, 0);
            cv::Mat testDesc = cv::Mat::ones(1, 32, CV_8UC1);
            DBoW3::BowVector testBow;
            mpVocabulary->transform(testDesc, testBow);
            std::cout << "\033[32m[LoopClosing] 词袋加载与 transform 测试完全成功！\033[0m" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "\033[31m[LoopClosing-WARN] 数据库初始化警告: " << e.what() << "\033[0m" << std::endl;
        }
    }
    else
    {
        std::cerr << "\033[31m[LoopClosing-WARN] 词袋文件格式不兼容或路径错误。\033[0m" << std::endl;
    }

    mLoopThread = std::thread(&LoopClosing::Run, this);
}

LoopClosing::~LoopClosing()
{
    {
        std::unique_lock<std::mutex> lock(mMutexQueue);
        mIsRunning = false;
    }
    mCondQueue.notify_all();
    if (mLoopThread.joinable())
        mLoopThread.join();
}

void LoopClosing::SetTracking(Tracking *pTracking)
{
    mpTracking = pTracking;
}

void LoopClosing::InsertKeyFrame(std::shared_ptr<KeyFrame> pKF)
{
    std::unique_lock<std::mutex> lock(mMutexQueue);
    mKeyFrameQueue.push(pKF);
    mCondQueue.notify_one();
}

void LoopClosing::Run()
{
    while (true)
    {
        std::shared_ptr<KeyFrame> pKF;
        {
            std::unique_lock<std::mutex> lock(mMutexQueue);
            mCondQueue.wait(lock, [this]
                            { return !mIsRunning || !mKeyFrameQueue.empty(); });

            if (!mIsRunning)
                break;

            pKF = mKeyFrameQueue.front();
            mKeyFrameQueue.pop();
        }

        if (!mpVocabulary || !mpDatabase)
            continue;

        if (pKF->mDescriptors.empty() || pKF->mDescriptors.rows == 0)
        {
            continue;
        }

        pKF->ComputeBoW(mpVocabulary);

        if (pKF->mBowVec.empty())
        {
            continue;
        }

        std::shared_ptr<KeyFrame> pMatchedKF;
        if (DetectLoop(pKF, pMatchedKF))
        {
            std::cout << "\033[32m[Loop closing detected!] KF " << pKF->mId
                      << " <---> Candidate KF " << pMatchedKF->mId << "\033[0m" << std::endl;

            CorrectLoop(pKF, pMatchedKF);
        }

        try
        {
            DBoW3::EntryId entryId = mpDatabase->add(pKF->mBowVec);
            mmKeyFrameDB[entryId] = pKF;
        }
        catch (...)
        {
        }
    }
}

bool LoopClosing::CheckConsistency(std::shared_ptr<KeyFrame> pCandidateKF)
{
    if (!pCandidateKF)
        return false;

    std::set<unsigned long> currentGroup;
    long candId = pCandidateKF->mId;
    
    // 扩大近邻组半径 (±6 帧)，提升对光流特征 BoW 检索跳变的容错
    for (long offset = -6; offset <= 6; ++offset)
    {
        if (candId + offset >= 0)
        {
            currentGroup.insert(static_cast<unsigned long>(candId + offset));
        }
    }

    bool bConsistentFound = false;
    std::vector<ConsistentGroup> vNewConsistentGroups;

    for (auto &group : mvConsistentGroups)
    {
        bool bOverlap = false;
        for (auto id : currentGroup)
        {
            if (group.sKFIds.count(id))
            {
                bOverlap = true;
                break;
            }
        }

        if (bOverlap)
        {
            ConsistentGroup newGroup;
            newGroup.sKFIds = currentGroup;
            newGroup.nConsistencyCount = group.nConsistencyCount + 1;
            vNewConsistentGroups.push_back(newGroup);

            // 连续命中 2 次即通过校验
            if (newGroup.nConsistencyCount >= 2)
            {
                bConsistentFound = true;
            }
        }
    }

    if (!bConsistentFound && vNewConsistentGroups.empty())
    {
        ConsistentGroup newGroup;
        newGroup.sKFIds = currentGroup;
        newGroup.nConsistencyCount = 1;
        vNewConsistentGroups.push_back(newGroup);
    }

    mvConsistentGroups = vNewConsistentGroups;
    return bConsistentFound;
}

bool LoopClosing::DetectLoop(std::shared_ptr<KeyFrame> pCurrentKF, std::shared_ptr<KeyFrame> &pMatchedKF)
{
    if (!mpDatabase || pCurrentKF->mBowVec.empty())
        return false;
    if (pCurrentKF->mId < 30)
        return false;

    DBoW3::QueryResults ret;
    try
    {
        mpDatabase->query(pCurrentKF->mBowVec, ret, 5);
    }
    catch (...)
    {
        return false;
    }

    double maxScore = 0.0;
    double maxValidScore = 0.0;
    pMatchedKF = nullptr;

    for (size_t i = 0; i < ret.size(); ++i)
    {
        DBoW3::EntryId entryId = ret[i].Id;
        auto itDB = mmKeyFrameDB.find(entryId);
        if (itDB == mmKeyFrameDB.end())
            continue;

        auto pCandidateKF = itDB->second;
        if (!pCandidateKF)
            continue;

        if (ret[i].Score > maxScore)
        {
            maxScore = ret[i].Score;
        }

        if ((pCurrentKF->mId > pCandidateKF->mId) && (pCurrentKF->mId - pCandidateKF->mId > 80))
        {
            if (ret[i].Score > 0.12 && ret[i].Score > maxValidScore)
            {
                maxValidScore = ret[i].Score;
                pMatchedKF = pCandidateKF;
            }
        }
    }

    if (pCurrentKF->mId % 20 == 0)
    {
        std::cout << "[LoopClosing Debug] KF " << pCurrentKF->mId 
                  << " | 最高检索得分: " << maxScore 
                  << " | 候选帧得分: " << maxValidScore << std::endl;
    }

    if (pMatchedKF != nullptr)
    {
        if (!CheckConsistency(pMatchedKF))
        {
            pMatchedKF = nullptr;
        }
    }

    return (pMatchedKF != nullptr);
}

void LoopClosing::FuseMapPoints(std::shared_ptr<KeyFrame> pCurrentKF,
                                std::shared_ptr<KeyFrame> pMatchedKF,
                                const std::vector<cv::DMatch> &matches)
{
    if (!mpMap)
        return;

    auto allMPs = mpMap->GetAllMapPoints();
    std::unordered_map<int, std::shared_ptr<MapPoint>> mapIdToMP;
    for (const auto &mp : allMPs)
    {
        if (mp && !mp->IsBad())
        {
            mapIdToMP[mp->GetFeatureId()] = mp;
        }
    }

    int replacedCount = 0;
    int addedObsCount = 0;

    for (const auto &m : matches)
    {
        int queryIdx = m.queryIdx;
        int trainIdx = m.trainIdx;

        if (queryIdx < 0 || queryIdx >= (int)pCurrentKF->mvKeyFeatureIds.size() ||
            trainIdx < 0 || trainIdx >= (int)pMatchedKF->mvKeyFeatureIds.size())
        {
            continue;
        }

        int currFeatId = pCurrentKF->mvKeyFeatureIds[queryIdx];
        int matchFeatId = pMatchedKF->mvKeyFeatureIds[trainIdx];

        auto itCurr = mapIdToMP.find(currFeatId);
        auto itMatch = mapIdToMP.find(matchFeatId);

        std::shared_ptr<MapPoint> pMPCurr = (itCurr != mapIdToMP.end()) ? itCurr->second : nullptr;
        std::shared_ptr<MapPoint> pMPMatch = (itMatch != mapIdToMP.end()) ? itMatch->second : nullptr;

        if (pMPCurr && pMPMatch && pMPCurr != pMPMatch)
        {
            pMPCurr->SetWorldPos(pMPMatch->GetWorldPos());
            pMPCurr->SetBad();
            replacedCount++;
        }
        else if (pMPMatch && !pMPCurr)
        {
            addedObsCount++;
        }
    }

    std::cout << "\033[32m[Map Fusion] 地图点融合完成! 融合重复点: "
              << replacedCount << " | 建立观测关联: " << addedObsCount << "\033[0m" << std::endl;
}

void LoopClosing::UpdateMapPointsAfterPGO(const std::map<unsigned long, Eigen::Isometry3d> &oldPoses)
{
    if (!mpMap) return;

    auto allMPs = mpMap->GetAllMapPoints();
    auto allKFs = mpMap->GetAllKeyFrames();

    for (auto &pMP : allMPs)
    {
        if (!pMP || pMP->IsBad()) continue;

        int featId = pMP->GetFeatureId();
        
        std::shared_ptr<KeyFrame> pRefKF = nullptr;
        for (const auto &kf : allKFs)
        {
            if (kf->mmObservations.count(featId))
            {
                pRefKF = kf;
                break;
            }
        }

        if (pRefKF)
        {
            auto itOldPose = oldPoses.find(pRefKF->mId);
            if (itOldPose != oldPoses.end())
            {
                Eigen::Isometry3d Twc_old = itOldPose->second;
                Eigen::Isometry3d Twc_new = pRefKF->GetPose();

                Eigen::Vector3d P_world_old = pMP->GetWorldPos();
                Eigen::Vector3d P_local = Twc_old.inverse() * P_world_old;
                Eigen::Vector3d P_world_new = Twc_new * P_local;
                
                pMP->SetWorldPos(P_world_new);
            }
        }
    }
}

bool LoopClosing::CorrectLoop(std::shared_ptr<KeyFrame> pCurrentKF, std::shared_ptr<KeyFrame> pMatchedKF)
{
    if (pCurrentKF->mDescriptors.empty() || pMatchedKF->mDescriptors.empty() ||
        pCurrentKF->mDescriptors.rows == 0 || pMatchedKF->mDescriptors.rows == 0)
    {
        return false;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    try
    {
        matcher.knnMatch(pCurrentKF->mDescriptors, pMatchedKF->mDescriptors, knnMatches, 2);
    }
    catch (...)
    {
        return false;
    }

    std::vector<cv::DMatch> goodMatches;
    std::vector<cv::Point2f> ptsCurr, ptsMatch;
    for (const auto &m : knnMatches)
    {
        if (m.size() >= 2 && m[0].distance < 0.85f * m[1].distance && m[0].distance < 75)
        {
            if (m[0].queryIdx < (int)pCurrentKF->mvKeys.size() &&
                m[0].trainIdx < (int)pMatchedKF->mvKeys.size())
            {
                goodMatches.push_back(m[0]);
                ptsCurr.push_back(pCurrentKF->mvKeys[m[0].queryIdx].pt);
                ptsMatch.push_back(pMatchedKF->mvKeys[m[0].trainIdx].pt);
            }
        }
    }

    if (ptsCurr.size() < 8)
    {
        std::cout << "\033[31m[Loop Correct Fail] KF " << pCurrentKF->mId 
                  << " <-> KF " << pMatchedKF->mId << " 匹配点数不足: " << ptsCurr.size() << "\033[0m" << std::endl;
        return false;
    }

    std::vector<uchar> inliers;
    cv::findFundamentalMat(ptsCurr, ptsMatch, cv::FM_RANSAC, 3.0, 0.99, inliers);

    int inlierCount = 0;
    std::vector<cv::DMatch> ransacMatches;
    for (size_t i = 0; i < inliers.size(); ++i)
    {
        if (inliers[i])
        {
            inlierCount++;
            ransacMatches.push_back(goodMatches[i]);
        }
    }

    if (inlierCount < 10)
    {
        std::cout << "\033[31m[Loop Correct Fail] KF " << pCurrentKF->mId 
                  << " <-> KF " << pMatchedKF->mId << " RANSAC 内点不足: " << inlierCount << "\033[0m" << std::endl;
        return false;
    }

    Sophus::SE3d T_match_curr_meas;
    bool pnpSucc = ComputeRelativePosePnP(pCurrentKF, pMatchedKF, ransacMatches, T_match_curr_meas);
    if (!pnpSucc)
    {
        std::cout << "\033[31m[Loop Correct Fail] KF " << pCurrentKF->mId 
                  << " <-> KF " << pMatchedKF->mId << " PnP 解算失败\033[0m" << std::endl;
        return false;
    }

    auto edge = std::make_pair(pMatchedKF->mId, pCurrentKF->mId);
    mvLoopEdges.push_back(edge);
    mmLoopRelativePoses[edge] = T_match_curr_meas;

    std::cout << "\033[32m[Loop Verified!] 闭环几何校验通过！KF " << pCurrentKF->mId
              << " <---> 匹配帧 KF " << pMatchedKF->mId
              << " | 相对平移: [" << T_match_curr_meas.translation().transpose() << "]\033[0m"
              << std::endl;

    if (mpTracking)
    {
        mpTracking->RequestPause();
        while (!mpTracking->IsPaused())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    {
        std::unique_lock<std::mutex> lockMap(mpMap->GetMutex());

        std::map<unsigned long, Eigen::Isometry3d> oldPoses;
        auto allKFs = mpMap->GetAllKeyFrames();
        for (const auto &kf : allKFs)
        {
            oldPoses[kf->mId] = kf->GetPose();
        }

        FuseMapPoints(pCurrentKF, pMatchedKF, ransacMatches);
        
        // std::cout << "\033[33m[PoseGraph] 开始执行位姿图优化 ...\033[0m" << std::endl;
        // Optimizer::PoseGraphOptimization(mpMap, mvLoopEdges, mmLoopRelativePoses);
        // std::cout << "\033[32m[PoseGraph] 全局位姿图优化完成！\033[0m" << std::endl;

        UpdateMapPointsAfterPGO(oldPoses);
    }

    if (mpTracking)
    {
        mpTracking->Resume();
    }

    return true;
}

bool LoopClosing::ComputeRelativePosePnP(std::shared_ptr<KeyFrame> pCurrentKF,
                                         std::shared_ptr<KeyFrame> pMatchedKF,
                                         const std::vector<cv::DMatch> &matches,
                                         Sophus::SE3d &T_match_curr_meas)
{
    double fx = Parameters::fx;
    double fy = Parameters::fy;
    double cx = Parameters::cx;
    double cy = Parameters::cy;

    Eigen::Matrix4d T_c0_c1 = Parameters::body_T_cam0.inverse() * Parameters::body_T_cam1;
    double baseline = T_c0_c1.block<3, 1>(0, 3).norm();
    if (baseline < 1e-4)
        baseline = 0.11;
    double bf = baseline * fx;

    std::vector<cv::Point3f> pts3D_match;
    std::vector<cv::Point2f> pts2D_curr;

    for (const auto &m : matches)
    {
        int queryIdx = m.queryIdx;
        int trainIdx = m.trainIdx;

        if (trainIdx < 0 || trainIdx >= (int)pMatchedKF->mvKeyFeatureIds.size())
            continue;

        int realFeatureId = pMatchedKF->mvKeyFeatureIds[trainIdx];

        auto itMatch = pMatchedKF->mmObservations.find(realFeatureId);
        if (itMatch == pMatchedKF->mmObservations.end())
            continue;

        const StereoObs &obsMatch = itMatch->second;
        if (!obsMatch.hasRight)
            continue;

        double disparity = obsMatch.ptLeft.x - obsMatch.ptRight.x;
        if (disparity <= 0.1)
            continue;

        double z = bf / disparity;
        double x = (obsMatch.ptLeft.x - cx) * z / fx;
        double y = (obsMatch.ptLeft.y - cy) * z / fy;

        pts3D_match.push_back(cv::Point3f(x, y, z));
        pts2D_curr.push_back(pCurrentKF->mvKeys[queryIdx].pt);
    }

    if (pts3D_match.size() < 8)
        return false;

    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
    cv::Mat rvec, tvec;
    std::vector<int> inliersPnP;

    bool success = cv::solvePnPRansac(
        pts3D_match, pts2D_curr, K, distCoeffs,
        rvec, tvec, false, 100, 2.0, 0.99, inliersPnP, cv::SOLVEPNP_ITERATIVE);

    if (!success || inliersPnP.size() < 8)
        return false;

    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);

    Eigen::Matrix3d R_curr_match;
    Eigen::Vector3d t_curr_match;
    for (int i = 0; i < 3; ++i)
    {
        t_curr_match(i) = tvec.at<double>(i);
        for (int j = 0; j < 3; ++j)
        {
            R_curr_match(i, j) = R_mat.at<double>(i, j);
        }
    }

    Sophus::SE3d T_curr_match(R_curr_match, t_curr_match);
    T_match_curr_meas = T_curr_match.inverse();

    return true;
}