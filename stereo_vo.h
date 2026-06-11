#ifndef STEREO_VO_H
#define STEREO_VO_H

#include <iostream>
#include <vector>
#include <string>
#include <list>      // 新增：用于四叉树节点链表
#include <cmath>     // 新增：用于数学计算
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <sophus/se3.hpp>

// 局部地图点结构体
struct MapPoint {
    cv::Point3f pos_world; // 世界坐标系下的绝对 3D 坐标
    cv::Mat descriptor;    // 特征描述子
};

// ==========================================
// 新增：四叉树节点结构
// ==========================================
class ExtractorNode {
public:
    ExtractorNode() : bNoMore(false) {}
    // 节点分裂函数声明
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    cv::Point2i UL, UR, BL, BR;           // 节点边界坐标
    std::vector<cv::KeyPoint> vKeys;      // 节点内包含的特征点
    std::list<ExtractorNode>::iterator lit; // 迭代器
    bool bNoMore;                         // 标志位：是否无法再分
};

class StereoVO {
public:
    double fx, fy, cx, cy, baseline;
    cv::Mat K;
    
    // 局部地图缓存池
    std::vector<MapPoint> local_map; 
    const size_t max_local_points = 1500; // 限制滑窗大小

    StereoVO() = default;

    // 解析标定文件
    bool loadCalibration(const std::string& calib_file_path);

    // ==========================================
    // 新增：四叉树均分与 ORB 提取核心函数
    // ==========================================
    std::vector<cv::KeyPoint> DistributeQuadTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, 
                                                 int minX, int maxX, int minY, int maxY, int N);
    
    void extractORBWithQuadTree(const cv::Mat& img, std::vector<cv::KeyPoint>& kps, cv::Mat& desc, int num_features);

    // 特征提取与帧间匹配
    void matchORB(const cv::Mat& img1, const cv::Mat& img2, 
                  std::vector<cv::KeyPoint>& kps1, std::vector<cv::KeyPoint>& kps2, 
                  std::vector<cv::DMatch>& good_matches);

    // 双目初始三角化构建 3D-2D 点 (用于第一帧地图开张)
    void build3D2DPoints(const std::vector<cv::KeyPoint>& kps1_l, const std::vector<cv::KeyPoint>& kps1_r, 
                         const std::vector<cv::KeyPoint>& kps2_l, const std::vector<cv::DMatch>& stereo_matches, 
                         const std::vector<cv::DMatch>& temporal_matches, std::vector<cv::Point3f>& pts_3d, 
                         std::vector<cv::Point2f>& pts_2d, std::vector<cv::DMatch>& viz_matches);

    // 局部地图追踪核心：新帧向整个 Local Map 匹配和投影
    void trackLocalMap(const cv::Mat& img_curr, const Sophus::SE3d& T_world_predict, 
                       std::vector<cv::Point3f>& pts_3d, std::vector<cv::Point2f>& pts_2d,
                       std::vector<cv::KeyPoint>& kps_curr, std::vector<cv::DMatch>& viz_matches);

    // 局部地图更新核心：将当前帧优化对齐后的高精度新点存回地图
    void updateLocalMap(const std::vector<cv::Point3f>& pts_3d_cam, const cv::Mat& img_curr_l, 
                        const std::vector<cv::KeyPoint>& kps_curr_l, const Sophus::SE3d& T_world_curr);
    
    // 新增：直接使用描述子进行匹配，避免重复提取特征
    void matchDescriptors(const cv::Mat& desc1, const cv::Mat& desc2, 
                          std::vector<cv::DMatch>& good_matches);
};

#endif // STEREO_VO_H