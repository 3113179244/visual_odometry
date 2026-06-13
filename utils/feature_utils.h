// =========================================================================
// ====== DEBUG REFACTOR CODE START ======
// 文件名: utils/feature_utils.h
// 修改点: 在四叉树分发前，加入绝对硬阈值初筛，并修复潜在的越界问题
// =========================================================================

#pragma once
#include <vector>
#include <list>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>

namespace vo_feature
{
  // 内部使用的四叉树网格节点结构体
  struct ExtractorNode
  {
    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore = false;

    inline void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
      const int halfX = std::ceil(static_cast<float>(UR.x - UL.x) / 2);
      const int halfY = std::ceil(static_cast<float>(BR.y - UL.y) / 2);

      n1.UL = UL;
      n1.UR = cv::Point2i(UL.x + halfX, UL.y);
      n1.BL = cv::Point2i(UL.x, UL.y + halfY);
      n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
      n2.UL = n1.UR;
      n2.UR = UR;
      n2.BL = n1.BR;
      n2.BR = cv::Point2i(UR.x, UL.y + halfY);
      n3.UL = n1.BL;
      n3.UR = n1.BR;
      n3.BL = BL;
      n3.BR = cv::Point2i(n1.BR.x, BL.y);
      n4.UL = n3.UR;
      n4.UR = n2.BR;
      n4.BL = n3.BR;
      n4.BR = BR;

      n1.vKeys.reserve(vKeys.size());
      n2.vKeys.reserve(vKeys.size());
      n3.vKeys.reserve(vKeys.size());
      n4.vKeys.reserve(vKeys.size());

      for (size_t i = 0; i < vKeys.size(); i++)
      {
        const cv::KeyPoint &kp = vKeys[i];
        if (kp.pt.x < n1.UR.x)
        {
          if (kp.pt.y < n1.BR.y)
            n1.vKeys.push_back(kp);
          else
            n3.vKeys.push_back(kp);
        }
        else
        {
          if (kp.pt.y < n1.BR.y)
            n2.vKeys.push_back(kp);
          else
            n4.vKeys.push_back(kp);
        }
      }

      if (n1.vKeys.size() == 1)
        n1.bNoMore = true;
      if (n2.vKeys.size() == 1)
        n2.bNoMore = true;
      if (n3.vKeys.size() == 1)
        n3.bNoMore = true;
      if (n4.vKeys.size() == 1)
        n4.bNoMore = true;
    }
  };

  /**
   * @brief 工业级无状态四叉树均匀化抑制与分发算子 (已追加调试初筛阈值)
   */
  inline std::vector<cv::KeyPoint> DistributeQuadTree(const std::vector<cv::KeyPoint> &vToDistributeKeys,
                                                      int minX, int maxX, int minY, int maxY, int N)
  {
    if (vToDistributeKeys.empty())
      return vToDistributeKeys;

    // 🌟🌟🌟 调试硬阈值过滤点池 🌟🌟🌟
    std::vector<cv::KeyPoint> vFilteredKeys;
    vFilteredKeys.reserve(vToDistributeKeys.size());

    float min_response_threshold = 20.0f; // 🔍 设定硬阈值底线，低于此响应值的点一律视为垃圾细碎噪声直接丢弃

    for (const auto &kp : vToDistributeKeys)
    {
      if (kp.response >= min_response_threshold)
      {
        vFilteredKeys.push_back(kp);
      }
    }

    // 💡 防御性降级机制：如果过滤完剩下的点还没有期望保留的点数 N 多，说明本帧太暗或纹理太差
    // 此时为了防止断掉跟踪，放宽要求，回退使用原始点池
    if (vFilteredKeys.size() < (size_t)N)
    {
      vFilteredKeys = vToDistributeKeys;
    }

    if (vFilteredKeys.size() < (size_t)N)
      return vFilteredKeys;

    const int nIni = std::round(static_cast<float>(maxX - minX) / (maxY - minY));
    const float hX = static_cast<float>(maxX - minX) / (nIni <= 0 ? 1 : nIni);

    std::list<ExtractorNode> lNodes;
    std::vector<ExtractorNode *> vpIniNodes(nIni <= 0 ? 1 : nIni);

    for (int i = 0; i < nIni; i++)
    {
      ExtractorNode ni;
      ni.UL = cv::Point2i(hX * static_cast<float>(i), minY);
      ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), minY);
      ni.BL = cv::Point2i(ni.UL.x, maxY);
      ni.BR = cv::Point2i(ni.UR.x, maxY);
      ni.vKeys.reserve(vFilteredKeys.size());
      lNodes.push_back(ni);
      vpIniNodes[i] = &lNodes.back();
    }

    for (size_t i = 0; i < vFilteredKeys.size(); i++)
    {
      const cv::KeyPoint &kp = vFilteredKeys[i];
      int idx = kp.pt.x / hX;
      if (idx >= 0 && idx < (int)vpIniNodes.size())
      {
        vpIniNodes[idx]->vKeys.push_back(kp);
      }
    }

    auto lit = lNodes.begin();
    while (lit != lNodes.end())
    {
      if (lit->vKeys.size() == 1)
      {
        lit->bNoMore = true;
        lit++;
      }
      else if (lit->vKeys.empty())
        lit = lNodes.erase(lit);
      else
        lit++;
    }

    bool bFinish = false;
    while (!bFinish)
    {
      int nToExpand = 0;
      lit = lNodes.begin();
      while (lit != lNodes.end())
      {
        if (lit->bNoMore)
        {
          lit++;
          continue;
        }
        ExtractorNode n1, n2, n3, n4;
        lit->DivideNode(n1, n2, n3, n4);

        if (n1.vKeys.size() > 0)
        {
          lNodes.push_front(n1);
          if (n1.vKeys.size() > 1)
          {
            nToExpand++;
            lNodes.front().lit = lNodes.begin();
          }
        }
        if (n2.vKeys.size() > 0)
        {
          lNodes.push_front(n2);
          if (n2.vKeys.size() > 1)
          {
            nToExpand++;
            lNodes.front().lit = lNodes.begin();
          }
        }
        if (n3.vKeys.size() > 0)
        {
          lNodes.push_front(n3);
          if (n3.vKeys.size() > 1)
          {
            nToExpand++;
            lNodes.front().lit = lNodes.begin();
          }
        }
        if (n4.vKeys.size() > 0)
        {
          lNodes.push_front(n4);
          if (n4.vKeys.size() > 1)
          {
            nToExpand++;
            lNodes.front().lit = lNodes.begin();
          }
        }

        lit = lNodes.erase(lit);
      }
      if ((int)lNodes.size() >= N || nToExpand == 0)
        bFinish = true;
    }

    std::vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(lNodes.size());
    for (lit = lNodes.begin(); lit != lNodes.end(); lit++)
    {
      std::vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
      cv::KeyPoint *pKP = &vNodeKeys[0];
      float maxResponse = pKP->response;
      for (size_t k = 1; k < vNodeKeys.size(); k++)
      {
        if (vNodeKeys[k].response > maxResponse)
        {
          pKP = &vNodeKeys[k];
          maxResponse = vNodeKeys[k].response;
        }
      }
      vResultKeys.push_back(*pKP);
    }
    return vResultKeys;
  }
} // namespace vo_feature
// ====== DEBUG REFACTOR CODE END ======