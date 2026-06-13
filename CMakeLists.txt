cmake_minimum_required(VERSION 3.10)
project(MyTestProject)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -Wall -g")

# 显式锁定 C++17，支持高级多线程特性
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 寻找核心第三方依赖库
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# 🌟 核心修复点：显式搜索系统中安装的 DBoW3 词袋图优化库
find_package(DBoW3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${DBoW3_INCLUDE_DIRS}) # 包含词袋头文件路径
include_directories(utils)

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
    loop_closing.cpp  # 将回环检测源文件绑定到可执行对象
)

target_link_libraries(main 
    Eigen3::Eigen
    Sophus::Sophus
    ${OpenCV_LIBS}
    ceres
    ${DBoW3_LIBS} # 🌟 核心修复点：将 libDBoW3.so 显式链接给二进制文件，彻底消灭链接报错
)