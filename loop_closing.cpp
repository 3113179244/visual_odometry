#include "loop_closing.h"
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <algorithm>

using namespace std;

// 🌟 核心修正：构造函数初始化列表对齐接收 pMap 指针
LoopClosing::LoopClosing(const std::string &voc_file, std::shared_ptr<Map> pMap)
    : mbRunning(false), mVocabulary(voc_file), mDatabase(mVocabulary, false, 0), mpMap(pMap)
{
}

LoopClosing::~LoopClosing()
{
    Stop();
}

void LoopClosing::Start()
{
    mbRunning = true;
    mptLoopClosing = new std::thread(&LoopClosing::Run, this);
    std::cout << "[LoopClosing] 后端回环检测线程已启动。" << std::endl;
}

void LoopClosing::Stop()
{
    if (mbRunning)
    {
        mbRunning = false;
        mLoopQueue.shutdown();
        if (mptLoopClosing && mptLoopClosing->joinable())
        {
            mptLoopClosing->join();
        }
        delete mptLoopClosing;
        mptLoopClosing = nullptr;
        std::cout << "[LoopClosing] 后端回环检测线程已停止。" << std::endl;
    }
}

void LoopClosing::InsertKeyFrame(KeyframePtr pKF)
{
    mLoopQueue.push(pKF);
}

void LoopClosing::Run()
{
    while (mbRunning)
    {
        KeyframePtr pKF;
        if (mLoopQueue.pop(pKF))
        {
            if (!pKF)
                continue;

            // 1. 将当前帧左图描述子转换为词袋向量
            DBoW3::BowVector bowVec;
            mVocabulary.transform(pKF->desc_l, bowVec);
            pKF->bow_vec = bowVec;

            // 2. 检测回环
            KeyframePtr pMatchedKF;
            if (DetectLoop(pKF, pMatchedKF))
            {
                std::cout << "🔄 [LoopClosing] 🌟 触发全局大回环！当前帧 ID: " << pKF->id
                          << " 成功匹配到历史关键帧 ID: " << pMatchedKF->id << " 🌟" << std::endl;

                // 3. 执行闭环位姿修正
                CorrectLoop(pKF, pMatchedKF);
            }

            // 4. 将当前帧加入词袋数据库与历史记录
            mDatabase.add(pKF->bow_vec);
            mHistoryKeyframes.push_back(pKF);
        }
    }
}

bool LoopClosing::DetectLoop(KeyframePtr pKF, KeyframePtr &pMatchedKF)
{
    if (mHistoryKeyframes.size() < 50)
        return false;

    DBoW3::QueryResults ret;
    mDatabase.query(pKF->bow_vec, ret, 4);

    if (ret.empty())
        return false;

    for (const auto &res : ret)
    {
        int hist_idx = res.Id;
        if (hist_idx >= 0 && hist_idx < (int)mHistoryKeyframes.size())
        {
            KeyframePtr pCand = mHistoryKeyframes[hist_idx];
            // 确保回环两帧在时间上相隔 60 个关键帧以上，防止原地误触发
            if (pCand && (pKF->id - pCand->id > 60) && res.Score > 0.05)
            {
                pMatchedKF = pCand;
                return true;
            }
        }
    }
    return false;
}

void LoopClosing::CorrectLoop(KeyframePtr pCurrKF, KeyframePtr pMatchedKF)
{
    cv::Mat R_curr;
    cv::Rodrigues(pCurrKF->rvec, R_curr);
    cv::Mat R_match;
    cv::Rodrigues(pMatchedKF->rvec, R_match);

    // 计算当前普通增量累加位姿与回环目标位姿之间的相对旋转/平移漂移误差
    cv::Mat R_drift = R_match * R_curr.t();
    cv::Mat t_drift = pMatchedKF->tvec - R_drift * pCurrKF->tvec;

    // 获取全局大地图所有关键帧快照并加锁保护
    auto global_kfs = mpMap->GetAllKeyframes();
    std::unique_lock<std::shared_mutex> lock(mpMap->mMutexMap);

    int start_id = pMatchedKF->id;
    int end_id = pCurrKF->id;
    int total_loop_frames = end_id - start_id;
    if (total_loop_frames <= 0)
        return;

    // 将漂移误差成比例、平滑地分摊分摊回这一段环路内的所有历史关键帧中
    for (auto &pKF : global_kfs)
    {
        if (pKF && pKF->id > start_id && pKF->id <= end_id)
        {
            double weight = static_cast<double>(pKF->id - start_id) / total_loop_frames;

            cv::Mat R_k;
            cv::Rodrigues(pKF->rvec, R_k);
            pKF->tvec = pKF->tvec + t_drift * weight;

            cv::Mat R_k_corrected = R_k;
            cv::Rodrigues(R_k_corrected, pKF->rvec);
        }
    }
}