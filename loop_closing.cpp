#include "loop_closing.h"
#include <opencv2/core/core.hpp>

LoopClosing::LoopClosing(const std::string& voc_file) 
    : mbRunning(false), mVocabulary(voc_file), mDatabase(mVocabulary, false, 0) {
}

void LoopClosing::Run() {
    // ====================================================================
    // 【调试前端修改】：强制回环检测大循环直接跳出
    // ====================================================================
    return;

    while (mbRunning) {
        KeyframePtr pKF;
        if (mLoopQueue.pop(pKF)) {
            DBoW3::BowVector bowVec;
            mVocabulary.transform(pKF->descriptors, bowVec);
            pKF->bow_vec = bowVec;

            KeyframePtr pMatchedKF;
            if (DetectLoop(pKF, pMatchedKF)) {
                CorrectLoop(pKF, pMatchedKF);
            }

            mDatabase.add(pKF->bow_vec);
            mHistoryKeyframes.push_back(pKF);
        }
    }
}