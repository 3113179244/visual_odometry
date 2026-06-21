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
        mpTracker = std::make_shared<Tracking>();

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
        RCLCPP_INFO(node_->get_logger(), "ROS2 节点启动成功！三角化点云机制已就绪。");
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
                    cv::Mat feat_img;
                    std::vector<Eigen::Vector3d> vWorldPoints; // 接收计算出的 3D 深度坐标点

                    // 传入点云引用，执行 DLT 分解
                    Eigen::Isometry3d Tcw = mpTracker->GrabImageStereo(mat0, mat1, sync_time, feat_img, vWorldPoints);

                    // 分发对外发布
                    publishTopics(img0->header, feat_img, vWorldPoints, Tcw);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    void publishTopics(const std_msgs::msg::Header &header,
                       const cv::Mat &feat_img,
                       const std::vector<Eigen::Vector3d> &vWorldPoints,
                       const Eigen::Isometry3d &Tcw)
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

        // 3. 发布实时生成的 3D 三角化点云数据 (/point_cloud)
        auto cloud_msg = sensor_msgs::msg::PointCloud2();
        cloud_msg.header = header;
        cloud_msg.header.frame_id = "world";
        cloud_msg.height = 1;
        cloud_msg.width = vWorldPoints.size();
        cloud_msg.is_bigendian = false;
        cloud_msg.is_dense = true;

        // 设置 PointCloud2 内存字段 (x, y, z 均为 FLOAT32)
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
        cloud_msg.point_step = 12; // 3 * 4 字节
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
        cloud_msg.data.resize(cloud_msg.row_step);

        float *data_ptr = reinterpret_cast<float *>(cloud_msg.data.data());
        for (size_t i = 0; i < vWorldPoints.size(); i++)
        {
            data_ptr[i * 3 + 0] = static_cast<float>(vWorldPoints[i].x());
            data_ptr[i * 3 + 1] = static_cast<float>(vWorldPoints[i].y());
            data_ptr[i * 3 + 2] = static_cast<float>(vWorldPoints[i].z());
        }
        pub_cloud_->publish(cloud_msg);

        // 4. 发布特征图传
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