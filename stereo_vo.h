#ifndef STEREO_VO_H
#define STEREO_VO_H

#include <iostream>
#include <vector>
#include <string>
#include <list>      
#include <cmath>     
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <sophus/se3.hpp>

struct MapPoint {
    cv::Point3f pos_world; 
    cv::Mat descriptor;    
};

class ExtractorNode {
public:
    ExtractorNode() : bNoMore(false) {}
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    cv::Point2i UL, UR, BL, BR;           
    std::vector<cv::KeyPoint> vKeys;      
    std::list<ExtractorNode>::iterator lit; 
    bool bNoMore;                         
};

class StereoVO {
public:
    double fx, fy, cx, cy, baseline;
    cv::Mat K;
    
    std::vector<MapPoint> local_map; 
    const size_t max_local_points = 1500; 

    StereoVO() = default;

    bool loadCalibration(const std::string& calib_file_path);

    std::vector<cv::KeyPoint> DistributeQuadTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, 
                                                 int minX, int maxX, int minY, int maxY, int N);
    
    void extractORBWithQuadTree(const cv::Mat& img, std::vector<cv::KeyPoint>& kps, cv::Mat& desc, int num_features);

    // ==========================================
    // 新增：利用对极几何约束的双目匹配 (取代全局暴力匹配)
    // ==========================================
    void matchStereoEpipolar(const std::vector<cv::KeyPoint>& kps_l, const std::vector<cv::KeyPoint>& kps_r,
                             const cv::Mat& desc_l, const cv::Mat& desc_r,
                             std::vector<cv::DMatch>& good_matches, float max_v_disp = 2.0f);

    // ==========================================
    // 新增：利用恒速运动模型进行帧间投影匹配 (取代全局暴力匹配)
    // ==========================================
    void matchTemporalByProjection(const std::vector<cv::KeyPoint>& kps_prev, 
                                   const std::vector<cv::Point3f>& pts_3d_prev,
                                   const cv::Mat& desc_prev, 
                                   const std::vector<cv::KeyPoint>& kps_curr, 
                                   const cv::Mat& desc_curr,
                                   const Sophus::SE3d& T_curr_prev,
                                   const cv::Mat& K,
                                   std::vector<cv::DMatch>& temporal_matches,
                                   float search_radius = 15.0f);

    // 原有函数保留，以防其他模块调用
    void matchORB(const cv::Mat& img1, const cv::Mat& img2, 
                  std::vector<cv::KeyPoint>& kps1, std::vector<cv::KeyPoint>& kps2, 
                  std::vector<cv::DMatch>& good_matches);

    void build3D2DPoints(const std::vector<cv::KeyPoint>& kps1_l, const std::vector<cv::KeyPoint>& kps1_r, 
                         const std::vector<cv::KeyPoint>& kps2_l, const std::vector<cv::DMatch>& stereo_matches, 
                         const std::vector<cv::DMatch>& temporal_matches, std::vector<cv::Point3f>& pts_3d, 
                         std::vector<cv::Point2f>& pts_2d, std::vector<cv::DMatch>& viz_matches);

    void trackLocalMap(const cv::Mat& img_curr, const Sophus::SE3d& T_world_predict, 
                       std::vector<cv::Point3f>& pts_3d, std::vector<cv::Point2f>& pts_2d,
                       std::vector<cv::KeyPoint>& kps_curr, std::vector<cv::DMatch>& viz_matches);

    void updateLocalMap(const std::vector<cv::Point3f>& pts_3d_cam, const cv::Mat& img_curr_l, 
                        const std::vector<cv::KeyPoint>& kps_curr_l, const Sophus::SE3d& T_world_curr);
    
    void matchDescriptors(const cv::Mat& desc1, const cv::Mat& desc2, 
                          std::vector<cv::DMatch>& good_matches);
};

#endif // STEREO_VO_H