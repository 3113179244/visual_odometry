#include "loop_closing.h"
#include <opencv2/core/core.hpp>

LoopClosing::LoopClosing(const std::string& voc_file) 
    : mbRunning(false), mVocabulary(voc_file), mDatabase(mVocabulary, false, 0) {
    // 加载字典：mVocabulary.load(voc_file);
}

void LoopClosing::Run() {
    while (mbRunning) {
        KeyframePtr pKF;
        if (mLoopQueue.pop(pKF)) {
            // 1. 转换为 DBoW3 的 BowVector
            DBoW3::BowVector bowVec;
            // 假设你的 Keyframe 类里存有描述子列表 descriptors
            mVocabulary.transform(pKF->descriptors, bowVec);
            pKF->bow_vec = bowVec;

            // 2. 数据库检索 (Detect Loop)
            KeyframePtr pMatchedKF;
            if (DetectLoop(pKF, pMatchedKF)) {
                // 3. 计算相对变换并执行全局优化 (Correct Loop)
                CorrectLoop(pKF, pMatchedKF);
            }

            // 4. 将当前帧存入数据库
            mDatabase.add(pKF->bow_vec);
            mHistoryKeyframes.push_back(pKF);
        }
    }
}