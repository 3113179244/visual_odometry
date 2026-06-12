#pragma once
#include <vector>
#include <set>
#include <memory>
#include <shared_mutex>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

// ====================================================================
// 🌟 核心修正：取消不完整的向前声明，直接在此处暴露完整的结构体与类实体定义
// ====================================================================

struct MapPoint {
    cv::Point3f pos_world;
    cv::Mat descriptor;
    bool is_bad = false;
    int n_observed = 0;
    int n_visible = 0;
};

class Keyframe {
public:
    int id;
    cv::Mat rvec, tvec;
    std::vector<cv::Point2f> keypoints_2d;
    std::vector<int> map_point_indices; 

    cv::Mat img_l;
    std::vector<cv::KeyPoint> kps_l, kps_r;
    cv::Mat desc_l, desc_r;

    Keyframe(int _id, const cv::Mat& _r, const cv::Mat& _t, 
             const std::vector<cv::Point2f>& _k2d, const std::vector<int>& _mp_idx,
             const cv::Mat& _img, const std::vector<cv::KeyPoint>& _kl, const std::vector<cv::KeyPoint>& _kr,
             const cv::Mat& _dl, const cv::Mat& _dr)
        : id(_id), rvec(_r.clone()), tvec(_t.clone()), keypoints_2d(_k2d), map_point_indices(_mp_idx),
          img_l(_img.clone()), kps_l(_kl), kps_r(_kr), desc_l(_dl.clone()), desc_r(_dr.clone()) {}
};

// 智能指针别名定义保持不变
using MapPointPtr = std::shared_ptr<MapPoint>;
using KeyframePtr = std::shared_ptr<Keyframe>;

class Map {
public:
    Map() = default; // 🌟 顺便修正：如果你在cpp里没写 Map::Map(){}，这里加上 = default 避免链接错误
    ~Map() = default;

    // ----- 全局地图管理接口 -----
    void AddKeyframe(KeyframePtr pKF);
    void AddMapPoint(MapPointPtr pMP);
    
    std::vector<KeyframePtr> GetAllKeyframes();
    std::vector<MapPointPtr> GetAllMapPoints();
    size_t GetGlobalMapPointsSize();

    // ----- 局部地图管理接口（面向后端局部BA） -----
    void UpdateLocalMap(KeyframePtr pCurrentKF, int max_local_kfs = 10);
    std::vector<KeyframePtr> GetLocalKeyframes();
    std::vector<MapPointPtr> GetLocalMapPoints();

    // ----- 地图清理接口 -----
    void CleanBadMechanisms();

    // 用于多线程安全的读写锁
    mutable std::shared_mutex mMutexMap;

private:
    // 全局地图容器
    std::vector<KeyframePtr> vpSubKeyframes;
    std::vector<MapPointPtr> vpSubMapPoints;

    // 局部地图容器（滑动窗口/共视近邻）
    std::vector<KeyframePtr> vpLocalKeyframes;
    std::vector<MapPointPtr> vpLocalMapPoints;
};