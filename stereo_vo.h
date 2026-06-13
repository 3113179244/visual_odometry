#pragma once
#include <vector>
#include <string>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include "map.h" // 🌟 引入新创建的地图类
#include <list>
// 四叉树节点结构体定义
struct ExtractorNode
{
    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore = false;
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);
};

class StereoVO
{
public:
    StereoVO() = default;
    ~StereoVO() = default;

    bool loadCalibration(const std::string &calib_file_path);

    // 四叉树 ORB 均匀分配核心
    std::vector<cv::KeyPoint> DistributeQuadTree(const std::vector<cv::KeyPoint> &vToDistributeKeys,
                                                 int minX, int maxX, int minY, int maxY, int N);
    void extractORBWithQuadTree(const cv::Mat &img, std::vector<cv::KeyPoint> &kps, cv::Mat &desc, int num_features);

    // 双目极线匹配与恒速投影匹配加速
    void matchStereoEpipolar(const std::vector<cv::KeyPoint> &kps_l, const std::vector<cv::KeyPoint> &kps_r,
                             const cv::Mat &desc_l, const cv::Mat &desc_r,
                             std::vector<cv::DMatch> &good_matches, float max_v_disp = 2.0f);

    void matchTemporalByProjection(const std::vector<cv::KeyPoint> &kps_prev,
                                   const std::vector<cv::Point3f> &pts_3d_prev,
                                   const cv::Mat &desc_prev,
                                   const std::vector<cv::KeyPoint> &kps_curr,
                                   const cv::Mat &desc_curr,
                                   const Sophus::SE3d &T_curr_prev,
                                   const cv::Mat &K,
                                   std::vector<cv::DMatch> &temporal_matches,
                                   float search_radius = 15.0f);

    // 🌟 原有业务函数（匹配与3D点构建接口保持不变）
    void matchORB(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2, std::vector<cv::DMatch> &good_matches);
    void build3D2DPoints(const std::vector<cv::KeyPoint> &kps1_l, const std::vector<cv::KeyPoint> &kps1_r, const std::vector<cv::KeyPoint> &kps2_l,
                         const std::vector<cv::DMatch> &stereo_matches, const std::vector<cv::DMatch> &temporal_matches,
                         std::vector<cv::Point3f> &pts_3d, std::vector<cv::Point2f> &pts_2d, std::vector<cv::DMatch> &viz_matches);
    void matchDescriptors(const cv::Mat &desc1, const cv::Mat &desc2, std::vector<cv::DMatch> &good_matches);

    // 相机内参及基线
    double fx, fy, cx, cy, baseline;
    cv::Mat K;

    // 🌟 移除原有的 local_map 成员，改由外部的 Map 类统一纳管，锁也交由 Map 内部接管
};