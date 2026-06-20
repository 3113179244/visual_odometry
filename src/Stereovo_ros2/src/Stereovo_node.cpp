#include <memory>   // 引入智能指针相关库
#include <queue>    // 引入队列容器，用于缓存传感器图像数据
#include <mutex>    // 引入互斥锁相关库，用于多线程间的线程安全保护
#include <thread>   // 引入多线程相关库
#include <chrono>   // 引入时间相关库
#include <iostream> 
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"                         // ROS 2 核心 C++ 客户端库
#include "sensor_msgs/msg/image.hpp"                 // ROS 2 图像消息定义
#include "nav_msgs/msg/odometry.hpp"                 // ROS 2 里程计消息定义
#include "nav_msgs/msg/path.hpp"                     // ROS 2 轨迹路径消息定义
#include "sensor_msgs/msg/point_cloud2.hpp"         // ROS 2 3D点云标准消息定义
#include "geometry_msgs/msg/pose_stamped.hpp"       // 用于构建 Path 路径的单个姿态点消息

#include <cv_bridge/cv_bridge.h>                     // 负责 ROS 图像与 OpenCV Mat 的转换
#include <opencv2/opencv.hpp>                        // 引入 OpenCV 核心库
#include <Eigen/Core>                                // 引入 Eigen 核心库
#include <Eigen/Geometry>                            // 引入 Eigen 几何库

#include "Tracking.h"   // 引入前端追踪模块空壳
#include "Map.h"        // 引入地图管理模块
#include "Parameters.h" // 引入参数解析模块

class StereoVONode {
public:
    StereoVONode() {
        // 1. 在内部创建独立的 ROS 2 节点对象
        node_ = std::make_shared<rclcpp::Node>("stereovo_node");

        // 2. 实例化全局唯一的 SLAM 核心组件
        mpMap = std::make_shared<Map>();         
        mpTracker = std::make_shared<Tracking>(); 

        // 3. 创建核心数据发布者
        pub_odom_   = node_->create_publisher<nav_msgs::msg::Odometry>("/odometry", rclcpp::QoS(100));
        pub_path_   = node_->create_publisher<nav_msgs::msg::Path>("/path", rclcpp::QoS(10));
        pub_cloud_  = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/point_cloud", rclcpp::QoS(1000));
        pub_feat_img_ = node_->create_publisher<sensor_msgs::msg::Image>("/feature_image", rclcpp::QoS(10));

        // 4. 从 Parameters 中动态读取外部话题名称进行订阅
        sub_img0_ = node_->create_subscription<sensor_msgs::msg::Image>(
            Parameters::IMAGE0_TOPIC, rclcpp::SensorDataQoS(),
            std::bind(&StereoVONode::img0_callback, this, std::placeholders::_1));

        sub_img1_ = node_->create_subscription<sensor_msgs::msg::Image>(
            Parameters::IMAGE1_TOPIC, rclcpp::SensorDataQoS(),
            std::bind(&StereoVONode::img1_callback, this, std::placeholders::_1));

        // 5. 开启独立同步处理线程
        sync_thread_ = std::thread(&StereoVONode::sync_process, this);

        RCLCPP_INFO(node_->get_logger(), "ROS2 节点启动成功！已挂载话题并拉起同步线程。");
    }

    ~StereoVONode() {
        if (sync_thread_.joinable()) {
            sync_thread_.join(); 
        }
    }

    std::shared_ptr<rclcpp::Node> get_node() const { return node_; }

private:
    void img0_callback(const sensor_msgs::msg::Image::SharedPtr img_msg) {
        std::lock_guard<std::mutex> lock(m_buf_); 
        img0_buf_.push(img_msg);                  
    }

    void img1_callback(const sensor_msgs::msg::Image::SharedPtr img_msg) {
        std::lock_guard<std::mutex> lock(m_buf_); 
        img1_buf_.push(img_msg);                  
    }

    cv::Mat getImageFromMsg(const sensor_msgs::msg::Image::SharedPtr &img_msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(node_->get_logger(), "cv_bridge 转换失败: %s", e.what());
            return cv::Mat();
        }
        return cv_ptr->image;
    }

    void sync_process() {
        while (rclcpp::ok()) {
            sensor_msgs::msg::Image::SharedPtr img0 = nullptr; 
            sensor_msgs::msg::Image::SharedPtr img1 = nullptr; 

            m_buf_.lock(); 
            if (!img0_buf_.empty() && !img1_buf_.empty()) {
                double time0 = img0_buf_.front()->header.stamp.sec + img0_buf_.front()->header.stamp.nanosec * 1e-9;
                double time1 = img1_buf_.front()->header.stamp.sec + img1_buf_.front()->header.stamp.nanosec * 1e-9;

                // 严格时间戳对齐 (视差小于 3ms 视为同一瞬间捕获的双目图像)
                if (time0 < time1 - 0.003) {
                    img0_buf_.pop(); 
                } 
                else if (time0 > time1 + 0.003) {
                    img1_buf_.pop(); 
                } 
                else {
                    img0 = img0_buf_.front(); 
                    img0_buf_.pop();          
                    img1 = img1_buf_.front(); 
                    img1_buf_.pop();          

                    while (img0_buf_.size() > 1 && img1_buf_.size() > 1) {
                        img0_buf_.pop(); 
                        img1_buf_.pop(); 
                    }
                }
            }
            m_buf_.unlock(); 

            if (img0 && img1) {
                double sync_time = img0->header.stamp.sec + img0->header.stamp.nanosec * 1e-9;

                cv::Mat mat0 = getImageFromMsg(img0);
                cv::Mat mat1 = getImageFromMsg(img1);

                if (!mat0.empty() && !mat1.empty()) {
                    // 调用你的纯空壳 Tracking 入口，验证控制流是否接通
                    Eigen::Isometry3d Tcw = mpTracker->GrabImageStereo(mat0, mat1, sync_time);
                    
                    // 触发 ROS 2 数据话题对外发布
                    publishTopics(img0->header, img0, Tcw); 
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    void publishTopics(const std_msgs::msg::Header& header, 
                       const sensor_msgs::msg::Image::SharedPtr raw_img,
                       const Eigen::Isometry3d& Tcw) 
    {
        Eigen::Isometry3d Twc = Tcw.inverse();
        Eigen::Vector3d translation = Twc.translation();
        Eigen::Quaterniond rotation(Twc.rotation());

        // 1. 发布里程计话题 (/odometry)
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header = header;                 
        odom_msg.header.frame_id = "world";       
        odom_msg.child_frame_id = "body";         
        
        odom_msg.pose.pose.position.x = translation.x();
        odom_msg.pose.pose.position.y = translation.y();
        odom_msg.pose.pose.position.z = translation.z();
        odom_msg.pose.pose.orientation.w = rotation.w();
        odom_msg.pose.pose.orientation.x = rotation.x();
        odom_msg.pose.pose.orientation.y = rotation.y();
        odom_msg.pose.pose.orientation.z = rotation.z();
        pub_odom_->publish(odom_msg);             

        // 2. 发布历史轨迹话题 (/path)
        geometry_msgs::msg::PoseStamped current_pose;
        current_pose.header = header;
        current_pose.header.frame_id = "world";
        current_pose.pose = odom_msg.pose.pose;
        mv_path_poses_.push_back(current_pose);

        auto path_msg = nav_msgs::msg::Path();
        path_msg.header = header;                 
        path_msg.header.frame_id = "world";       
        path_msg.poses = mv_path_poses_;        
        pub_path_->publish(path_msg);             

        // 3. 发布空白点云话题 (/point_cloud)
        auto cloud_msg = sensor_msgs::msg::PointCloud2();
        cloud_msg.header = header;                 
        cloud_msg.header.frame_id = "world";       
        cloud_msg.height = 1;
        cloud_msg.width = 0; // 算法暂时为空，点云数量为 0
        pub_cloud_->publish(cloud_msg);           

        // 4. 发布原特征图传
        if (raw_img) {
            pub_feat_img_->publish(*raw_img);    
        }
    }

private:
    std::shared_ptr<rclcpp::Node> node_; 
    std::shared_ptr<Map> mpMap;           
    std::shared_ptr<Tracking> mpTracker;   

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img0_; 
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img1_; 
    
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;       
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;           
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_; 
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_feat_img_;            

    std::queue<sensor_msgs::msg::Image::SharedPtr> img0_buf_; 
    std::queue<sensor_msgs::msg::Image::SharedPtr> img1_buf_; 
    std::mutex m_buf_;                                       
    std::thread sync_thread_;                                

    std::vector<geometry_msgs::msg::PoseStamped> mv_path_poses_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv); 
    
    auto non_ros_args = rclcpp::remove_ros_arguments(argc, argv);
    if(non_ros_args.size() != 2)
    {
        std::cout << "使用方法: ros2 run stereovo_ros2 stereovo_node [参数文件绝对路径]" << std::endl;
        return 1;
    }

    std::string config_file = non_ros_args[1];
    Parameters::readParameters(config_file);
    
    auto stereo_vo_app = std::make_shared<StereoVONode>();
    rclcpp::spin(stereo_vo_app->get_node());
    
    rclcpp::shutdown(); 
    return 0;           
}