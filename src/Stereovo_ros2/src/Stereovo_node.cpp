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
#include "sensor_msgs/msg/point_cloud.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/bool.hpp"

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
        node_ = std::make_shared<rclcpp::Node>("vins_estimator");
        mpMap = std::make_shared<Map>();
        mpTracker = std::make_shared<Tracking>(mpMap);

        mpTracker->RegisterCallback(
            std::bind(&StereoVONode::OnTrackingRendered, this,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4,
                      std::placeholders::_5, std::placeholders::_6,
                      std::placeholders::_7));

        pub_latest_odometry_ = node_->create_publisher<nav_msgs::msg::Odometry>("imu_propagate", 1000);
        pub_odometry_ = node_->create_publisher<nav_msgs::msg::Odometry>("odometry", 1000);
        pub_path_ = node_->create_publisher<nav_msgs::msg::Path>("path", 1000);
        pub_point_cloud_ = node_->create_publisher<sensor_msgs::msg::PointCloud>("point_cloud", 1000);
        pub_margin_cloud_ = node_->create_publisher<sensor_msgs::msg::PointCloud>("margin_cloud", 1000);
        pub_key_poses_ = node_->create_publisher<visualization_msgs::msg::Marker>("key_poses", 1000);
        pub_camera_pose_ = node_->create_publisher<nav_msgs::msg::Odometry>("camera_pose", 1000);
        pub_camera_pose_visual_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("camera_pose_visual", 1000);
        pub_extrinsic_ = node_->create_publisher<nav_msgs::msg::Odometry>("extrinsic", 1000);
        pub_image_track_ = node_->create_publisher<sensor_msgs::msg::Image>("image_track", 1000);
        pub_features_ = node_->create_publisher<sensor_msgs::msg::PointCloud>("/feature_tracker/feature", 1000);

        sub_img0_ = node_->create_subscription<sensor_msgs::msg::Image>(
            Parameters::IMAGE0_TOPIC, rclcpp::SensorDataQoS(),
            std::bind(&StereoVONode::img0_callback, this, std::placeholders::_1));
        sub_img1_ = node_->create_subscription<sensor_msgs::msg::Image>(
            Parameters::IMAGE1_TOPIC, rclcpp::SensorDataQoS(),
            std::bind(&StereoVONode::img1_callback, this, std::placeholders::_1));
        sub_restart_ = node_->create_subscription<std_msgs::msg::Bool>(
            "/vins_restart", rclcpp::QoS(rclcpp::KeepLast(100)),
            std::bind(&StereoVONode::restart_callback, this, std::placeholders::_1));

        sync_thread_ = std::thread(&StereoVONode::sync_process, this);
        PublishExtrinsicOnce();

        RCLCPP_INFO(node_->get_logger(), "VINS-Fusion 兼容模式双目前端已启动（无路径长度限制）");
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
    void restart_callback(const std_msgs::msg::Bool::SharedPtr restart_msg)
    {
        if (restart_msg->data)
        {
            RCLCPP_WARN(node_->get_logger(), "重置轨迹");
            std::lock_guard<std::mutex> lock(path_mutex_);
            mv_path_poses_.clear();
        }
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
            RCLCPP_ERROR(node_->get_logger(), "cv_bridge 错误: %s", e.what());
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
                    mpTracker->FeedStereoImages(mat0, mat1, sync_time);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    void PublishLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, double t)
    {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp.sec = static_cast<int32_t>(t);
        odom.header.stamp.nanosec = static_cast<uint32_t>((t - odom.header.stamp.sec) * 1e9);
        odom.header.frame_id = "world";
        odom.pose.pose.position.x = P.x();
        odom.pose.pose.position.y = P.y();
        odom.pose.pose.position.z = P.z();
        odom.pose.pose.orientation.x = Q.x();
        odom.pose.pose.orientation.y = Q.y();
        odom.pose.pose.orientation.z = Q.z();
        odom.pose.pose.orientation.w = Q.w();
        pub_latest_odometry_->publish(odom);
    }

    // ========== 修改点：移除了路径长度限制 ==========
    void PublishOdometry(const std_msgs::msg::Header &header,
                         const Eigen::Vector3d &P, const Eigen::Quaterniond &Q)
    {
        nav_msgs::msg::Odometry odom;
        odom.header = header;
        odom.header.frame_id = "world";
        odom.child_frame_id = "world";
        odom.pose.pose.position.x = P.x();
        odom.pose.pose.position.y = P.y();
        odom.pose.pose.position.z = P.z();
        odom.pose.pose.orientation.x = Q.x();
        odom.pose.pose.orientation.y = Q.y();
        odom.pose.pose.orientation.z = Q.z();
        odom.pose.pose.orientation.w = Q.w();
        pub_odometry_->publish(odom);

        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odom.pose.pose;

        std::lock_guard<std::mutex> lock(path_mutex_);
        mv_path_poses_.push_back(pose_stamped);
        // 已移除截断逻辑，路径将无限增长

        nav_msgs::msg::Path path_msg;
        path_msg.header = header;
        path_msg.header.frame_id = "world";
        path_msg.poses = mv_path_poses_;
        pub_path_->publish(path_msg);
    }

    void PublishCameraPose(const std_msgs::msg::Header &header,
                           const Eigen::Vector3d &P, const Eigen::Quaterniond &Q)
    {
        nav_msgs::msg::Odometry odom;
        odom.header = header;
        odom.header.frame_id = "world";
        odom.pose.pose.position.x = P.x();
        odom.pose.pose.position.y = P.y();
        odom.pose.pose.position.z = P.z();
        odom.pose.pose.orientation.x = Q.x();
        odom.pose.pose.orientation.y = Q.y();
        odom.pose.pose.orientation.z = Q.z();
        odom.pose.pose.orientation.w = Q.w();
        pub_camera_pose_->publish(odom);

        visualization_msgs::msg::MarkerArray frustum_array;
        visualization_msgs::msg::Marker edge_marker;
        edge_marker.header = header;
        edge_marker.ns = "camera_pose_visual";
        edge_marker.id = 0;
        edge_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        edge_marker.action = visualization_msgs::msg::Marker::ADD;
        edge_marker.scale.x = 0.02;
        edge_marker.color.r = 0.0;
        edge_marker.color.g = 1.0;
        edge_marker.color.b = 0.0;
        edge_marker.color.a = 1.0;

        double scale = 0.2;
        auto create_pt = [&](double cx, double cy, double cz)
        {
            Eigen::Vector3d p_cam(cx * scale, cy * scale, cz * scale);
            Eigen::Vector3d p_w = Q * p_cam + P;
            geometry_msgs::msg::Point pt;
            pt.x = p_w.x();
            pt.y = p_w.y();
            pt.z = p_w.z();
            return pt;
        };

        geometry_msgs::msg::Point r_o = create_pt(0, 0, 0);
        geometry_msgs::msg::Point r_p1 = create_pt(-1, -0.5, 2), r_p2 = create_pt(1, -0.5, 2);
        geometry_msgs::msg::Point r_p3 = create_pt(1, 0.5, 2), r_p4 = create_pt(-1, 0.5, 2);

        edge_marker.points = {r_o, r_p1, r_o, r_p2, r_o, r_p3, r_o, r_p4,
                              r_p1, r_p2, r_p2, r_p3, r_p3, r_p4, r_p4, r_p1};
        frustum_array.markers.push_back(edge_marker);
        pub_camera_pose_visual_->publish(frustum_array);
    }

    void PublishPointCloud(const std_msgs::msg::Header &header,
                           const std::vector<Eigen::Vector3d> &vWorldPoints)
    {
        sensor_msgs::msg::PointCloud cloud, margin_cloud;
        cloud.header = header;
        margin_cloud.header = header;

        for (size_t i = 0; i < vWorldPoints.size(); ++i)
        {
            geometry_msgs::msg::Point32 p;
            p.x = static_cast<float>(vWorldPoints[i].x());
            p.y = static_cast<float>(vWorldPoints[i].y());
            p.z = static_cast<float>(vWorldPoints[i].z());
            cloud.points.push_back(p);
            if (i % 8 == 0)
                margin_cloud.points.push_back(p);
        }
        pub_point_cloud_->publish(cloud);
        pub_margin_cloud_->publish(margin_cloud);
    }

    void PublishKeyPoses(const std_msgs::msg::Header &header,
                         const std::vector<Eigen::Vector3d> &vKFPositions)
    {
        if (vKFPositions.empty())
            return;
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.header.frame_id = "world";
        marker.ns = "key_poses";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.scale.z = 0.05;
        marker.color.r = 1.0;
        marker.color.a = 1.0;
        for (const auto &pos : vKFPositions)
        {
            geometry_msgs::msg::Point pt;
            pt.x = pos.x();
            pt.y = pos.y();
            pt.z = pos.z();
            marker.points.push_back(pt);
        }
        pub_key_poses_->publish(marker);
    }

    void PublishTrackImage(const cv::Mat &imgTrack, double t)
    {
        if (imgTrack.empty())
            return;
        std_msgs::msg::Header header;
        header.frame_id = "world";
        header.stamp.sec = static_cast<int32_t>(t);
        header.stamp.nanosec = static_cast<uint32_t>((t - header.stamp.sec) * 1e9);
        auto img_msg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
        pub_image_track_->publish(*img_msg);
    }

    void PublishExtrinsicOnce()
    {
        std_msgs::msg::Header header;
        header.frame_id = "world";
        header.stamp = node_->now();
        nav_msgs::msg::Odometry odom;
        odom.header = header;
        odom.header.frame_id = "world";
        odom.pose.pose.position.x = 0.0;
        odom.pose.pose.position.y = 0.0;
        odom.pose.pose.position.z = 0.0;
        odom.pose.pose.orientation.w = 1.0;
        pub_extrinsic_->publish(odom);
    }

    void PublishFeatures(const std_msgs::msg::Header &header,
                         const std::vector<cv::Point2f> &curPts,
                         const std::vector<int> &ids,
                         int camera_id = 0)
    {
        if (curPts.empty())
            return;
        sensor_msgs::msg::PointCloud feature_msg;
        feature_msg.header = header;
        feature_msg.header.frame_id = "world";

        sensor_msgs::msg::ChannelFloat32 id_ch, cam_ch, u_ch, v_ch, vx_ch, vy_ch;
        id_ch.name = "id";
        cam_ch.name = "camera_id";
        u_ch.name = "p_u";
        v_ch.name = "p_v";
        vx_ch.name = "velocity_x";
        vy_ch.name = "velocity_y";

        for (size_t i = 0; i < curPts.size(); ++i)
        {
            double u_norm = (curPts[i].x - Parameters::cx) / Parameters::fx;
            double v_norm = (curPts[i].y - Parameters::cy) / Parameters::fy;
            geometry_msgs::msg::Point32 p;
            p.x = static_cast<float>(u_norm);
            p.y = static_cast<float>(v_norm);
            p.z = 1.0f;
            feature_msg.points.push_back(p);

            id_ch.values.push_back(static_cast<float>(ids[i]));
            cam_ch.values.push_back(static_cast<float>(camera_id));
            u_ch.values.push_back(static_cast<float>(curPts[i].x));
            v_ch.values.push_back(static_cast<float>(curPts[i].y));
            vx_ch.values.push_back(0.0f);
            vy_ch.values.push_back(0.0f);
        }

        feature_msg.channels.push_back(id_ch);
        feature_msg.channels.push_back(cam_ch);
        feature_msg.channels.push_back(u_ch);
        feature_msg.channels.push_back(v_ch);
        feature_msg.channels.push_back(vx_ch);
        feature_msg.channels.push_back(vy_ch);
        pub_features_->publish(feature_msg);
    }

    void OnTrackingRendered(double timestamp,
                            const cv::Mat &feat_img,
                            const std::vector<Eigen::Vector3d> &vWorldPoints,
                            const std::vector<Eigen::Vector3d> &vKFPositions,
                            const Eigen::Isometry3d &Tcw,
                            const std::vector<cv::Point2f> &curPts,
                            const std::vector<int> &ids)
    {
        std_msgs::msg::Header header;
        header.frame_id = "world";
        header.stamp.sec = static_cast<int32_t>(timestamp);
        header.stamp.nanosec = static_cast<uint32_t>((timestamp - header.stamp.sec) * 1e9);

        Eigen::Isometry3d Twc = Tcw.inverse();
        Eigen::Vector3d P_cam = Twc.translation();
        Eigen::Quaterniond Q_cam(Twc.rotation());

        PublishLatestOdometry(P_cam, Q_cam, timestamp);
        PublishOdometry(header, P_cam, Q_cam);
        PublishCameraPose(header, P_cam, Q_cam);
        PublishPointCloud(header, vWorldPoints);
        PublishKeyPoses(header, vKFPositions);
        PublishTrackImage(feat_img, timestamp);
        PublishFeatures(header, curPts, ids, 0);
    }

private:
    std::shared_ptr<rclcpp::Node> node_;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Tracking> mpTracker;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img0_, sub_img1_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_restart_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_latest_odometry_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_margin_cloud_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_camera_pose_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_extrinsic_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_track_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_features_;

    std::queue<sensor_msgs::msg::Image::SharedPtr> img0_buf_, img1_buf_;
    std::mutex m_buf_;
    std::thread sync_thread_;

    std::mutex path_mutex_;
    std::vector<geometry_msgs::msg::PoseStamped> mv_path_poses_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto non_ros_args = rclcpp::remove_ros_arguments(argc, argv);
    if (non_ros_args.size() != 2)
    {
        std::cout << "用法: ros2 run stereovo_ros2 stereovo_node [配置文件绝对路径]" << std::endl;
        return 1;
    }
    Parameters::readParameters(non_ros_args[1]);

    auto app = std::make_shared<StereoVONode>();
    rclcpp::spin(app->get_node());
    rclcpp::shutdown();
    return 0;
}