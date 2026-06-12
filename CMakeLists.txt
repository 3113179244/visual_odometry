cmake_minimum_required(VERSION 3.10)
project(MyTestProject)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -Wall -g")

# 🌟 显式锁定 C++17，以支持 std::shared_mutex 读写锁的高级多线程特性
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 寻找核心第三方依赖库
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

add_executable(main 
    main.cpp 
    system.cpp
    tracking.cpp
    map.cpp 
    pose_optimizer.cpp 
    stereo_vo.cpp 
    keyframe_selector.cpp 
    slide_window.cpp 
    local_mapping.cpp
)

target_link_libraries(main 
    Eigen3::Eigen
    Sophus::Sophus
    ${OpenCV_LIBS}
    ceres
)