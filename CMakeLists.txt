cmake_minimum_required(VERSION 3.10)
project(MyTestProject)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -Wall -g")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 寻找所有的核心依赖库
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

# 声明可执行文件
add_executable(main main.cpp pose_optimizer.cpp stereo_vo.cpp keyframe_selector.cpp slide_window.cpp local_mapping.cpp)

# 链接所有的库
target_link_libraries(main 
    Eigen3::Eigen
    Sophus::Sophus
    ${OpenCV_LIBS}
    ceres
)