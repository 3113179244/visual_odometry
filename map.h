#pragma once
#include <vector>
#include <set>
#include <memory>
#include <shared_mutex>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

// 🌟 修复编译错误：引入 DBoW3 第三方头文件，让关键帧支持词袋向量
#include <DBoW3/DBoW3.h>

struct MapPoint
{
    cv::Point3f pos_world;
    cv::Mat descriptor;
    bool is_bad = false;
    int n_observed = 0;
    int n_visible = 0;
};

class Keyframe
{
public:
    int id;
    cv::Mat rvec, tvec;
    std::vector<cv::Point2f> keypoints_2d;
    std::vector<int> map_point_indices;

    cv::Mat img_l;
    std::vector<cv::KeyPoint> kps_l, kps_r;
    cv::Mat desc_l, desc_r;

    // 🌟 修复编译错误：在关键帧内部追加词袋向量成员变量
    DBoW3::BowVector bow_vec;

    Keyframe(int _id, const cv::Mat &_r, const cv::Mat &_t,
             const std::vector<cv::Point2f> &_k2d, const std::vector<int> &_mp_idx,
             const cv::Mat &_img, const std::vector<cv::KeyPoint> &_kl, const std::vector<cv::KeyPoint> &_kr,
             const cv::Mat &_dl, const cv::Mat &_dr)
        : id(_id), rvec(_r.clone()), tvec(_t.clone()), keypoints_2d(_k2d), map_point_indices(_mp_idx),
          img_l(_img.clone()), kps_l(_kl), kps_r(_kr), desc_l(_dl.clone()), desc_r(_dr.clone()) {}
};

using MapPointPtr = std::shared_ptr<MapPoint>;
using KeyframePtr = std::shared_ptr<Keyframe>;

class Map
{
public:
    Map() = default;
    ~Map() = default;

    void AddKeyframe(KeyframePtr pKF);
    void AddMapPoint(MapPointPtr pMP);

    std::vector<KeyframePtr> GetAllKeyframes();
    std::vector<MapPointPtr> GetAllMapPoints();
    size_t GetGlobalMapPointsSize();

    void UpdateLocalMap(KeyframePtr pCurrentKF, int max_local_kfs = 10);
    std::vector<KeyframePtr> GetLocalKeyframes();
    std::vector<MapPointPtr> GetLocalMapPoints();

    void CleanBadMechanisms();

    mutable std::shared_mutex mMutexMap;

private:
    std::vector<KeyframePtr> vpSubKeyframes;
    std::vector<MapPointPtr> vpSubMapPoints;

    std::vector<KeyframePtr> vpLocalKeyframes;
    std::vector<MapPointPtr> vpLocalMapPoints;
};