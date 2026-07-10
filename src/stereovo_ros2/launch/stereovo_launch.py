import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 定义包名
    package_name = 'stereovo_ros2'

    # 2. 声明参数文件的绝对路径
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='/home/wzj/visual_odometry/src/stereovo_ros2/config/kitti_config04-12.yaml',
        description='SLAM 主配置文件绝对路径'
    )

    # 【新增】3. 声明 KITTI 数据集序列文件夹的路径参数
    sequence_path_arg = DeclareLaunchArgument(
        'sequence_path',
        default_value='/home/wzj/KITTI/data_odometry_gray/dataset/sequences/00',
        description='KITTI 序列文件夹绝对路径'
    )

    # 4. 声明 rviz2 配置文件的绝对路径
    rviz_file_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value='/home/wzj/visual_odometry/src/stereovo_ros2/rviz2/deafult.rviz',
        description='RViz2 布局配置文件绝对路径'
    )

    # 5. 定义你的编译节点 Stereovo_node
    stereovo_node = Node(
        package=package_name,
        executable='Stereovo_node', # 对应 CMake 中的 add_executable
        name='Stereovo_node',
        output='screen',
        arguments=[
            LaunchConfiguration('config_file'),
            LaunchConfiguration('sequence_path')
        ]
    )

    # 6. 定义 RViz2 节点，并自动加载指定的 .rviz 界面
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen'
    )

    # 7. 组合图层返回（别忘了把新增的参数放进列表中）
    return LaunchDescription([
        config_file_arg,
        sequence_path_arg,
        rviz_file_arg,
        stereovo_node,
        rviz2_node
    ])