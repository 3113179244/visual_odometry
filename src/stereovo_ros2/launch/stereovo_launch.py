import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 定义包名
    package_name = 'stereovo_ros2'

    # 2. 声明参数文件的绝对路径（默认指向你指定的 KITTI yaml 文件）
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='/home/wzj/visual_odometry/src/stereovo_ros2/config/kitti_config04-12.yaml',
        description='SLAM 主配置文件绝对路径'
    )

    # 3. 声明 rviz2 配置文件的绝对路径
    rviz_file_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value='/home/wzj/visual_odometry/src/stereovo_ros2/rviz2/deafult.rviz',
        description='RViz2 布局配置文件绝对路径'
    )

    # 4. 定义你的编译节点 Stereovo_node
    stereovo_node = Node(
        package=package_name,
        executable='Stereovo_node', # 对应 CMake 中的 add_executable
        name='Stereovo_node',
        output='screen',
        # 将参数文件作为命令行非 ROS 参数传递（匹配 main 函数中的 non_ros_args[1]）
        arguments=[LaunchConfiguration('config_file')]
    )

    # 5. 定义 RViz2 节点，并自动加载指定的 .rviz 界面
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen'
    )

    # 6. 组合图层返回
    return LaunchDescription([
        config_file_arg,
        rviz_file_arg,
        stereovo_node,
        rviz2_node
    ])