#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip> // 用于 std::setw 和 std::setfill

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/point_cloud.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/bool.hpp"

#include "tf2_ros/transform_broadcaster.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "core/Tracking.h"
#include "core/Map.h"
#include "utils/Parameters.h"

class StereoVONode
{
public:
    StereoVONode(const std::string &sequence_path)
    {
        node_ = std::make_shared<rclcpp::Node>("vins_estimator");
        mpMap = std::make_shared<Map>();
        mpTracker = std::make_shared<Tracking>(mpMap);

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);

        mpTracker->RegisterCallback(
            std::bind(&StereoVONode::OnTrackingRendered, this,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4,
                      std::placeholders::_5, std::placeholders::_6,
                      std::placeholders::_7, std::placeholders::_8));

        // 保留原有的 Rviz 可视化话题发布
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

        sub_restart_ = node_->create_subscription<std_msgs::msg::Bool>(
            "/vins_restart", rclcpp::QoS(rclcpp::KeepLast(100)),
            std::bind(&StereoVONode::restart_callback, this, std::placeholders::_1));

        PublishExtrinsicOnce();
        fout_trajectory_.open(Parameters::OUTPUT_PATH + "trajectory.txt", std::ios::trunc);
        if (!fout_trajectory_.is_open())
        {
            RCLCPP_ERROR(node_->get_logger(), "错误：无法创建或打开 trajectory.txt！");
        }

        // 启动本地文件读取线程
        dataset_thread_ = std::thread(&StereoVONode::load_kitti_dataset, this, sequence_path);
    }

    ~StereoVONode()
    {
        if (dataset_thread_.joinable())
            dataset_thread_.join();

        if (fout_trajectory_.is_open())
            fout_trajectory_.close();
    }

    std::shared_ptr<rclcpp::Node> get_node() const { return node_; }

private:
    void restart_callback(const std_msgs::msg::Bool::SharedPtr restart_msg)
    {
        if (restart_msg->data)
        {
            RCLCPP_WARN(node_->get_logger(), "重置轨迹");
            std::lock_guard<std::mutex> lock(path_mutex_);
            mv_path_poses_.clear();
        }
    }

    // 新增：核心本地数据集加载逻辑
    void load_kitti_dataset(const std::string &sequence_path)
    {
        std::string times_file = sequence_path + "/times.txt";
        std::ifstream fTimes(times_file);
        if (!fTimes.is_open())
        {
            RCLCPP_ERROR(node_->get_logger(), "无法打开时间戳文件: %s", times_file.c_str());
            return;
        }

        RCLCPP_INFO(node_->get_logger(), "开始直接从本地读取 KITTI 数据集: %s", sequence_path.c_str());

        std::string line;
        int frame_id = 0;

        // 逐行读取时间戳并加载对应的图片
        while (std::getline(fTimes, line) && rclcpp::ok())
        {
            if (line.empty()) continue;
            double timestamp = std::stod(line);

            // 拼接 KITTI 命名的 6 位数字，如 000000.png
            std::stringstream ss;
            ss << std::setw(6) << std::setfill('0') << frame_id;
            std::string img_name = ss.str() + ".png";

            std::string left_img_path = sequence_path + "/image_0/" + img_name;
            std::string right_img_path = sequence_path + "/image_1/" + img_name;

            // 直接用 OpenCV 读取灰度图，完全跳过了 ROS 传输层
            cv::Mat mat0 = cv::imread(left_img_path, cv::IMREAD_GRAYSCALE);
            cv::Mat mat1 = cv::imread(right_img_path, cv::IMREAD_GRAYSCALE);

            if (mat0.empty() || mat1.empty())
            {
                RCLCPP_WARN(node_->get_logger(), "图片读取结束或失败，总计处理了 %d 帧。", frame_id);
                break;
            }

            // 直接喂入前端进行追踪与后端 BA 优化
            mpTracker->FeedStereoImages(mat0, mat1, timestamp);
            frame_id++;

            // 【性能平衡控制】
            // 本地 IO 极其快速，前端算法缓冲区如果塞得太满会导致内存飙升。
            // 这里控制每帧读取完稍微休眠一下（10ms ≈ 100Hz 播放速度），确保有足够的时间留给后端优化和 RViz 渲染展示。
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        RCLCPP_INFO(node_->get_logger(), "数据集加载线程执行完毕。");
    }

    // =========================================================================
    // 以下原有的可视化发布函数逻辑完全保持不变，直接兼容原有的系统工作流
    // =========================================================================
    void PublishLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, double t)
    {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp.sec = static_cast<int32_t>(t);
        odom.header.stamp.nanosec = static_cast<uint32_t>((t - odom.header.stamp.sec) * 1e9);
        odom.header.frame_id = "world";
        odom.child_frame_id = "body";
        odom.pose.pose.position.x = P.x();
        odom.pose.pose.position.y = P.y();
        odom.pose.pose.position.z = P.z();
        odom.pose.pose.orientation.x = Q.x();
        odom.pose.pose.orientation.y = Q.y();
        odom.pose.pose.orientation.z = Q.z();
        odom.pose.pose.orientation.w = Q.w();
        pub_latest_odometry_->publish(odom);
    }

    void PublishOdometry(const std_msgs::msg::Header &header,
                         const Eigen::Vector3d &P_ros, const Eigen::Quaterniond &Q_ros,
                         const Eigen::Vector3d &P_cam, const Eigen::Matrix3d &R_cam)
    {
        nav_msgs::msg::Odometry odom;
        odom.header = header;
        odom.header.frame_id = "world";
        odom.child_frame_id = "body";
        odom.pose.pose.position.x = P_ros.x();
        odom.pose.pose.position.y = P_ros.y();
        odom.pose.pose.position.z = P_ros.z();
        odom.pose.pose.orientation.x = Q_ros.x();
        odom.pose.pose.orientation.y = Q_ros.y();
        odom.pose.pose.orientation.z = Q_ros.z();
        odom.pose.pose.orientation.w = Q_ros.w();
        pub_odometry_->publish(odom);

        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odom.pose.pose;

        std::lock_guard<std::mutex> lock(path_mutex_);
        if (mv_path_poses_.size() > 2000)
        {
            mv_path_poses_.erase(mv_path_poses_.begin());
        }
        mv_path_poses_.push_back(pose_stamped);

        nav_msgs::msg::Path path_msg;
        path_msg.header = header;
        path_msg.header.frame_id = "world";
        path_msg.poses = mv_path_poses_;
        pub_path_->publish(path_msg);

        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header = header;
        transformStamped.child_frame_id = "body";
        transformStamped.transform.translation.x = P_ros.x();
        transformStamped.transform.translation.y = P_ros.y();
        transformStamped.transform.translation.z = P_ros.z();
        transformStamped.transform.rotation.x = Q_ros.x();
        transformStamped.transform.rotation.y = Q_ros.y();
        transformStamped.transform.rotation.z = Q_ros.z();
        transformStamped.transform.rotation.w = Q_ros.w();
        tf_broadcaster_->sendTransform(transformStamped);

        if (fout_trajectory_.is_open())
        {
            fout_trajectory_.setf(std::ios::fixed, std::ios::floatfield);
            fout_trajectory_.precision(6);
            fout_trajectory_ << R_cam(0, 0) << " " << R_cam(0, 1) << " " << R_cam(0, 2) << " " << P_cam.x() << " "
                             << R_cam(1, 0) << " " << R_cam(1, 1) << " " << R_cam(1, 2) << " " << P_cam.y() << " "
                             << R_cam(2, 0) << " " << R_cam(2, 1) << " " << R_cam(2, 2) << " " << P_cam.z() << "\n";
            fout_trajectory_.flush();
        }
    }

    void PublishCameraPose(const std_msgs::msg::Header &header,
                           const Eigen::Vector3d &P, const Eigen::Quaterniond &Q)
    {
        nav_msgs::msg::Odometry odom;
        odom.header = header;
        odom.header.frame_id = "world";
        odom.child_frame_id = "body";
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
        edge_marker.header.frame_id = "body";
        edge_marker.ns = "camera_pose_visual";
        edge_marker.id = 0;
        edge_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        edge_marker.action = visualization_msgs::msg::Marker::ADD;
        edge_marker.scale.x = 0.03;
        edge_marker.color.g = 1.0;
        edge_marker.color.a = 1.0;

        double scale = 0.2;
        auto create_ros_body_pt = [&](double fwd, double lft, double up) {
            geometry_msgs::msg::Point pt;
            pt.x = fwd * scale; pt.y = lft * scale; pt.z = up * scale;
            return pt;
        };
        geometry_msgs::msg::Point r_o = create_ros_body_pt(0, 0, 0);
        geometry_msgs::msg::Point r_p1 = create_ros_body_pt(2, 1, 0.5);
        geometry_msgs::msg::Point r_p2 = create_ros_body_pt(2, -1, 0.5);
        geometry_msgs::msg::Point r_p3 = create_ros_body_pt(2, -1, -0.5);
        geometry_msgs::msg::Point r_p4 = create_ros_body_pt(2, 1, -0.5);

        edge_marker.points = {r_o, r_p1, r_o, r_p2, r_o, r_p3, r_o, r_p4,
                              r_p1, r_p2, r_p2, r_p3, r_p3, r_p4, r_p4, r_p1};
        frustum_array.markers.push_back(edge_marker);
        pub_camera_pose_visual_->publish(frustum_array);
    }

    void PublishPointCloud(const std_msgs::msg::Header &header, const std::vector<Eigen::Vector3d> &vWorldPoints)
    {
        sensor_msgs::msg::PointCloud cloud, margin_cloud;
        cloud.header = header; margin_cloud.header = header;
        Eigen::Matrix3d R_ros_cam;
        R_ros_cam << 0, 0, 1, -1, 0, 0, 0, -1, 0;

        for (size_t i = 0; i < vWorldPoints.size(); ++i)
        {
            Eigen::Vector3d p_ros = R_ros_cam * vWorldPoints[i];
            geometry_msgs::msg::Point32 p;
            p.x = static_cast<float>(p_ros.x()); p.y = static_cast<float>(p_ros.y()); p.z = static_cast<float>(p_ros.z());
            cloud.points.push_back(p);
            if (i % 8 == 0) margin_cloud.points.push_back(p);
        }
        pub_point_cloud_->publish(cloud);
        pub_margin_cloud_->publish(margin_cloud);
    }

    void PublishKeyPoses(const std_msgs::msg::Header &header, const std::vector<Eigen::Vector3d> &vKFPositions)
    {
        if (vKFPositions.empty()) return;
        visualization_msgs::msg::Marker marker;
        marker.header = header; marker.header.frame_id = "world";
        marker.ns = "key_poses"; marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.05; marker.scale.y = 0.05; marker.scale.z = 0.05;
        marker.color.r = 1.0; marker.color.a = 1.0;
        Eigen::Matrix3d R_ros_cam;
        R_ros_cam << 0, 0, 1, -1, 0, 0, 0, -1, 0;

        for (const auto &pos : vKFPositions)
        {
            Eigen::Vector3d pos_ros = R_ros_cam * pos;
            geometry_msgs::msg::Point pt;
            pt.x = pos_ros.x(); pt.y = pos_ros.y(); pt.z = pos_ros.z();
            marker.points.push_back(pt);
        }
        pub_key_poses_->publish(marker);
    }

    void PublishTrackImage(const cv::Mat &imgTrack, double t)
    {
        if (imgTrack.empty()) return;
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
        header.frame_id = "world"; header.stamp = node_->now();
        nav_msgs::msg::Odometry odom; odom.header = header; odom.header.frame_id = "world";
        odom.pose.pose.orientation.w = 1.0;
        pub_extrinsic_->publish(odom);
    }

    void PublishFeatures(const std_msgs::msg::Header &header, const std::vector<cv::Point2f> &curPts, const std::vector<int> &ids, const std::vector<cv::Point2f> &ptsVel, int camera_id = 0)
    {
        if (curPts.empty()) return;
        sensor_msgs::msg::PointCloud feature_msg;
        feature_msg.header = header; feature_msg.header.frame_id = "body";
        sensor_msgs::msg::ChannelFloat32 id_ch, cam_ch, u_ch, v_ch, vx_ch, vy_ch;
        id_ch.name = "id"; cam_ch.name = "camera_id"; u_ch.name = "p_u"; v_ch.name = "p_v"; vx_ch.name = "velocity_x"; vy_ch.name = "velocity_y";

        for (size_t i = 0; i < curPts.size(); ++i)
        {
            double u_norm = (curPts[i].x - Parameters::cx) / Parameters::fx;
            double v_norm = (curPts[i].y - Parameters::cy) / Parameters::fy;
            geometry_msgs::msg::Point32 p;
            p.x = 1.0f; p.y = static_cast<float>(-u_norm); p.z = static_cast<float>(-v_norm);
            feature_msg.points.push_back(p);
            id_ch.values.push_back(static_cast<float>(ids[i]));
            cam_ch.values.push_back(static_cast<float>(camera_id));
            u_ch.values.push_back(static_cast<float>(curPts[i].x));
            v_ch.values.push_back(static_cast<float>(curPts[i].y));
            vx_ch.values.push_back(ptsVel[i].x); vy_ch.values.push_back(ptsVel[i].y);
        }
        feature_msg.channels.push_back(id_ch); feature_msg.channels.push_back(cam_ch); feature_msg.channels.push_back(u_ch); feature_msg.channels.push_back(v_ch); feature_msg.channels.push_back(vx_ch); feature_msg.channels.push_back(vy_ch);
        pub_features_->publish(feature_msg);
    }

    void OnTrackingRendered(double timestamp, const cv::Mat &feat_img, const std::vector<Eigen::Vector3d> &vWorldPoints, const std::vector<Eigen::Vector3d> &vKFPositions, const Eigen::Isometry3d &Tcw, const std::vector<cv::Point2f> &curPts, const std::vector<int> &ids, const std::vector<cv::Point2f> &ptsVel)
    {
        std_msgs::msg::Header header;
        header.frame_id = "world";
        header.stamp.sec = static_cast<int32_t>(timestamp);
        header.stamp.nanosec = static_cast<uint32_t>((timestamp - header.stamp.sec) * 1e9);

        Eigen::Isometry3d Twc = Tcw;
        Eigen::Vector3d P_cam = Twc.translation();
        Eigen::Matrix3d R_cam = Twc.rotation();

        Eigen::Matrix3d R_ros_cam;
        R_ros_cam << 0, 0, 1, -1, 0, 0, 0, -1, 0;

        Eigen::Vector3d P_ros = R_ros_cam * P_cam;
        Eigen::Matrix3d R_ros = R_ros_cam * R_cam * R_ros_cam.transpose();
        Eigen::Quaterniond Q_ros(R_ros);

        PublishLatestOdometry(P_ros, Q_ros, timestamp);
        PublishOdometry(header, P_ros, Q_ros, P_cam, R_cam);
        PublishCameraPose(header, P_ros, Q_ros);
        PublishPointCloud(header, vWorldPoints);
        PublishKeyPoses(header, vKFPositions);
        PublishTrackImage(feat_img, timestamp);
        PublishFeatures(header, curPts, ids, ptsVel, 0);
    }

private:
    std::shared_ptr<rclcpp::Node> node_;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Tracking> mpTracker;

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

    std::thread dataset_thread_; // 替换 sync_thread_
    std::ofstream fout_trajectory_;
    std::mutex path_mutex_;
    std::vector<geometry_msgs::msg::PoseStamped> mv_path_poses_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto non_ros_args = rclcpp::remove_ros_arguments(argc, argv);
    
    // 调整参数检测：现在需要接受两个参数 [配置文件路径] [KITTI文件夹路径]
    if (non_ros_args.size() != 3)
    {
        std::cout << "用法: ros2 run stereovo_ros2 Stereovo_node [配置文件绝对路径] [KITTI序列文件夹绝对路径]" << std::endl;
        std::cout << "示例: ros2 run stereovo_ros2 Stereovo_node /path/to/kitti_config05.yaml /home/wzj/KITTI/data_odometry_gray/dataset/sequences/05" << std::endl;
        return 1;
    }
    
    Parameters::readParameters(non_ros_args[1]);

    // 创建节点并将第二个参数传递进去
    auto app = std::make_shared<StereoVONode>(non_ros_args[2]);
    rclcpp::spin(app->get_node());
    rclcpp::shutdown();
    return 0;
}