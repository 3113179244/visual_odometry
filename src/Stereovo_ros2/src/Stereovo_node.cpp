#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "Tracking.h"
#include "Map.h"
#include "Parameters.h"

class StereoVONode
{
public:
    StereoVONode()
    {
        node_ = std::make_shared<rclcpp::Node>("stereovo_node");
        mpMap = std::make_shared<Map>();
        mpTracker = std::make_shared<Tracking>(mpMap);

        // 注册回调：当且仅当 Tracking 专属子线程算完了，会自动跨线程执行主节点的发布逻辑
        mpTracker->RegisterCallback(std::bind(&StereoVONode::OnTrackingRendered, this,
                                              std::placeholders::_1, std::placeholders::_2,
                                              std::placeholders::_3, std::placeholders::_4));

        pub_odom_ = node_->create_publisher<nav_msgs::msg::Odometry>("/odometry", rclcpp::QoS(100));
        pub_path_ = node_->create_publisher<nav_msgs::msg::Path>("/path", rclcpp::QoS(10));
        pub_cloud_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/point_cloud", rclcpp::QoS(1000));
        pub_feat_img_ = node_->create_publisher<sensor_msgs::msg::Image>("/feature_image", rclcpp::QoS(10));

        sub_img0_ = node_->create_subscription<sensor_msgs::msg::Image>(
            Parameters::IMAGE0_TOPIC, rclcpp::SensorDataQoS(),
            std::bind(&StereoVONode::img0_callback, this, std::placeholders::_1));

        sub_img1_ = node_->create_subscription<sensor_msgs::msg::Image>(
            Parameters::IMAGE1_TOPIC, rclcpp::SensorDataQoS(),
            std::bind(&StereoVONode::img1_callback, this, std::placeholders::_1));

        sync_thread_ = std::thread(&StereoVONode::sync_process, this);
        RCLCPP_INFO(node_->get_logger(), "ROS2 节点启动成功！异步多线程追踪架构已激活。");
    }

    ~StereoVONode()
    {
        if (sync_thread_.joinable())
            sync_thread_.join();
    }

    std::shared_ptr<rclcpp::Node> get_node() const { return node_; }

private:
    void img0_callback(const sensor_msgs::msg::Image::SharedPtr img_msg)
    {
        std::lock_guard<std::mutex> lock(m_buf_);
        img0_buf_.push(img_msg);
    }

    void img1_callback(const sensor_msgs::msg::Image::SharedPtr img_msg)
    {
        std::lock_guard<std::mutex> lock(m_buf_);
        img1_buf_.push(img_msg);
    }

    cv::Mat getImageFromMsg(const sensor_msgs::msg::Image::SharedPtr &img_msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(node_->get_logger(), "cv_bridge 转换失败: %s", e.what());
            return cv::Mat();
        }
        return cv_ptr->image;
    }

    // 主节点内部的时间戳严格同步器依然保留，但只负责对齐后快速推送（生产者）
    void sync_process()
    {
        while (rclcpp::ok())
        {
            sensor_msgs::msg::Image::SharedPtr img0 = nullptr;
            sensor_msgs::msg::Image::SharedPtr img1 = nullptr;

            m_buf_.lock();
            if (!img0_buf_.empty() && !img1_buf_.empty())
            {
                double time0 = img0_buf_.front()->header.stamp.sec + img0_buf_.front()->header.stamp.nanosec * 1e-9;
                double time1 = img1_buf_.front()->header.stamp.sec + img1_buf_.front()->header.stamp.nanosec * 1e-9;

                if (time0 < time1 - 0.003)
                    img0_buf_.pop();
                else if (time0 > time1 + 0.003)
                    img1_buf_.pop();
                else
                {
                    img0 = img0_buf_.front();
                    img0_buf_.pop();
                    img1 = img1_buf_.front();
                    img1_buf_.pop();
                    while (img0_buf_.size() > 1 && img1_buf_.size() > 1)
                    {
                        img0_buf_.pop();
                        img1_buf_.pop();
                    }
                }
            }
            m_buf_.unlock();

            if (img0 && img1)
            {
                double sync_time = img0->header.stamp.sec + img0->header.stamp.nanosec * 1e-9;
                cv::Mat mat0 = getImageFromMsg(img0);
                cv::Mat mat1 = getImageFromMsg(img1);

                if (!mat0.empty() && !mat1.empty())
                {
                    // 【重大调整】：直接将双目图像无阻塞塞入 Tracking 维护的内部队列中
                    mpTracker->FeedStereoImages(mat0, mat1, sync_time);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    // 由异步线程触发的回调代理发布器
    void OnTrackingRendered(double timestamp, const cv::Mat &feat_img,
                            const std::vector<Eigen::Vector3d> &vWorldPoints,
                            const Eigen::Isometry3d &Tcw)
    {
        // 构造虚拟 Header
        std_msgs::msg::Header header;
        header.frame_id = "world";
        header.stamp.sec = static_cast<int32_t>(timestamp);
        header.stamp.nanosec = static_cast<uint32_t>((timestamp - header.stamp.sec) * 1e9);

        publishTopics(header, feat_img, vWorldPoints, Tcw);
    }

    void publishTopics(const std_msgs::msg::Header &header,
                       const cv::Mat &feat_img,
                       const std::vector<Eigen::Vector3d> &vWorldPoints,
                       const Eigen::Isometry3d &Tcw)
    {
        // 1. 计算当前相机在世界坐标系下的绝对位姿 Twc
        Eigen::Isometry3d Twc = Tcw.inverse();

        // 2. 提取当前帧相机原汁原味的三维平移和旋转（相机系：X右, Y下, Z前）
        Eigen::Vector3d t_cam = Twc.translation();
        Eigen::Matrix3d R_cam = Twc.rotation();

        // 3. 【核心修复】：转换至 ROS 2 标准坐标系（符合常规物理认知：X前, Y左, Z上）
        // 映射规则：
        // ROS_X (前进) = 对应相机的 Z 轴 (镜头前)
        // ROS_Y (向左) = 对应相机的 -X 轴 (因为相机 X 向右)
        // ROS_Z (向上) = 对应相机的 -Y 轴 (因为相机 Y 向下)
        Eigen::Vector3d t_ros;
        t_ros.x() = t_cam.z();
        t_ros.y() = -t_cam.x();
        t_ros.z() = -t_cam.y();

        // 对旋转矩阵进行相同的轴向转换
        Eigen::Matrix3d R_ros;
        // ROS的 X轴 来自相机的 Z轴
        R_ros.col(0) = R_cam.col(2);
        // ROS的 Y轴 来自相机的 -X轴
        R_ros.col(1) = -R_cam.col(0);
        // ROS的 Z轴 来自相机的 -Y轴
        R_ros.col(2) = -R_cam.col(1);

        Eigen::Quaterniond rotation_ros(R_ros);

        // 4. 将转换好轴向的 ROS 坐标数据打包发布
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header = header;
        odom_msg.header.frame_id = "world";
        odom_msg.child_frame_id = "body";

        // 使用修正后的 ROS 轴向数据
        odom_msg.pose.pose.position.x = t_ros.x();
        odom_msg.pose.pose.position.y = t_ros.y();
        odom_msg.pose.pose.position.z = t_ros.z();
        odom_msg.pose.pose.orientation.w = rotation_ros.w();
        odom_msg.pose.pose.orientation.x = rotation_ros.x();
        odom_msg.pose.pose.orientation.y = rotation_ros.y();
        odom_msg.pose.pose.orientation.z = rotation_ros.z();
        pub_odom_->publish(odom_msg);

        // 5. 发布历史轨迹话题 (/path)
        geometry_msgs::msg::PoseStamped current_pose;
        current_pose.header = header;
        current_pose.header.frame_id = "world";
        current_pose.pose = odom_msg.pose.pose;

        static std::mutex mutex_path;
        {
            std::lock_guard<std::mutex> lock(mutex_path);
            mv_path_poses_.push_back(current_pose);

            auto path_msg = nav_msgs::msg::Path();
            path_msg.header = header;
            path_msg.header.frame_id = "world";
            path_msg.poses = mv_path_poses_;
            pub_path_->publish(path_msg);
        }

        // 6. 实时生成的 3D 点云也需要进行同步的轴向重映射
        auto cloud_msg = sensor_msgs::msg::PointCloud2();
        cloud_msg.header = header;
        cloud_msg.header.frame_id = "world";
        cloud_msg.height = 1;
        cloud_msg.width = vWorldPoints.size();
        cloud_msg.is_bigendian = false;
        cloud_msg.is_dense = true;

        sensor_msgs::msg::PointField f_x, f_y, f_z;
        f_x.name = "x";
        f_x.offset = 0;
        f_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
        f_x.count = 1;
        f_y.name = "y";
        f_y.offset = 4;
        f_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
        f_y.count = 1;
        f_z.name = "z";
        f_z.offset = 8;
        f_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
        f_z.count = 1;
        cloud_msg.fields = {f_x, f_y, f_z};
        cloud_msg.point_step = 12;
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
        cloud_msg.data.resize(cloud_msg.row_step);

        float *data_ptr = reinterpret_cast<float *>(cloud_msg.data.data());
        for (size_t i = 0; i < vWorldPoints.size(); i++)
        {
            // 点云的空间坐标同步转换：将原相机的 Z (前) 映射到 ROS 的 X (前)
            data_ptr[i * 3 + 0] = static_cast<float>(vWorldPoints[i].z());
            data_ptr[i * 3 + 1] = static_cast<float>(-vWorldPoints[i].x());
            data_ptr[i * 3 + 2] = static_cast<float>(-vWorldPoints[i].y());
        }
        pub_cloud_->publish(cloud_msg);

        // 7. 发布特征图传
        if (!feat_img.empty())
        {
            auto img_msg_ptr = cv_bridge::CvImage(header, "bgr8", feat_img).toImageMsg();
            pub_feat_img_->publish(*img_msg_ptr);
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

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto non_ros_args = rclcpp::remove_ros_arguments(argc, argv);
    if (non_ros_args.size() != 2)
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