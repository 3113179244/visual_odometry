cmake_minimum_required(VERSION 3.8)
project(stereovo_ros2)

# -----------------------------------------------------------------------------
# 0. 启用 C++ 和 CUDA 支持
# -----------------------------------------------------------------------------
enable_language(CUDA)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14) 
endif()

add_compile_options(-Wextra -Wpedantic -w)
set(CMAKE_BUILD_TYPE Release)

set(YOLOV8_CODE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tensorrtx/yolov8")

set(CMAKE_CUDA_ARCHITECTURES "86")

# -----------------------------------------------------------------------------
# 1. 寻找 ROS 2 官方依赖包
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
find_package(Sophus REQUIRED)
# -----------------------------------------------------------------------------
# 3. 头文件包含路径
# -----------------------------------------------------------------------------
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}   
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  /usr/local/TensorRT/include
  ${YOLOV8_CODE_DIR}/include     
  ${YOLOV8_CODE_DIR}/plugin      
)

link_directories(/usr/local/TensorRT/lib)

# -----------------------------------------------------------------------------
# 4. 创建纯 C++ SLAM 核心算法库
# -----------------------------------------------------------------------------
add_library(${PROJECT_NAME}_lib
  # 基础与工具模块
  utils/Camera.cpp
  utils/Parameters.cpp
  core/FeatureDetector.cpp
  core/Frame.cpp
  core/KeyFrame.cpp
  core/Tracking.cpp
  core/Map.cpp
  core/MapPoint.cpp
  backend/Optimizer.cpp
  ${YOLOV8_CODE_DIR}/src/model.cpp
  ${YOLOV8_CODE_DIR}/src/preprocess.cu
  ${YOLOV8_CODE_DIR}/src/postprocess.cpp
  ${YOLOV8_CODE_DIR}/plugin/yololayer.cu
)

# 链接第三方数学库与 TensorRT 核心库
target_link_libraries(${PROJECT_NAME}_lib 
  ${OpenCV_LIBS} 
  ${CERES_LIBRARIES} 
  Sophus::Sophus
  nvinfer                      
  cudart                       
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

target_link_libraries(Stereovo_node ${PROJECT_NAME}_lib) 

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
# 6. 安装与导出规则
# -----------------------------------------------------------------------------
install(TARGETS
  ${PROJECT_NAME}_lib
  Stereovo_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config rviz2
  DESTINATION share/${PROJECT_NAME}
)

ament_package()