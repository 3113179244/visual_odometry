cmake_minimum_required(VERSION 3.8)
project(Stereovo_ros2)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14) 
endif()

add_compile_options(-Wextra -Wpedantic)
set(CMAKE_BUILD_TYPE Release)

# -----------------------------------------------------------------------------
# 1. 寻找 ROS 2 官方依赖包 (同步 VINS)
# -----------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcpputils REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# -----------------------------------------------------------------------------
# 2. 寻找第三方 SLAM 核心数学/视觉库
# -----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

# -----------------------------------------------------------------------------
# 3. 头文件包含路径
# -----------------------------------------------------------------------------
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
)

# -----------------------------------------------------------------------------
# 4. 创建 SLAM 核心算法库 (类似 VINS_lib)
# -----------------------------------------------------------------------------
add_library(${PROJECT_NAME}_lib
  src/Camera.cpp
  src/FeatureDetector.cpp
  src/Frame.cpp
  src/KeyFrame.cpp
  src/Map.cpp
  src/MapPoint.cpp
  src/Optimizer.cpp
  src/Tracking.cpp
  src/Parameters.cpp
)

# 链接第三方数学库
target_link_libraries(${PROJECT_NAME}_lib 
  ${OpenCV_LIBS} 
  ${CERES_LIBRARIES} 
)

# 链接 ROS 2 依赖组件
ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp 
  rcpputils 
  std_msgs 
  visualization_msgs 
  geometry_msgs 
  nav_msgs 
  sensor_msgs
  tf2 
  tf2_ros 
  cv_bridge 
  image_transport
)

# -----------------------------------------------------------------------------
# 5. 创建 ROS 2 可执行节点主入口
# -----------------------------------------------------------------------------
add_executable(Stereovo_node src/Stereovo_node.cpp)

# 链接内部核心算法库
target_link_libraries(Stereovo_node ${PROJECT_NAME}_lib) 

# 链接节点所需的 ROS 2 属性
ament_target_dependencies(Stereovo_node
  rclcpp 
  rcpputils 
  std_msgs 
  visualization_msgs 
  geometry_msgs 
  nav_msgs 
  sensor_msgs
  tf2 
  tf2_ros 
  cv_bridge 
  image_transport
)

# -----------------------------------------------------------------------------
# 6. 安装与导出规则 (ROS 2 标准)
# -----------------------------------------------------------------------------
# 安装二进制可执行文件与核心库
install(TARGETS Stereovo_node ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# 安装 launch 和 config 资源文件
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()