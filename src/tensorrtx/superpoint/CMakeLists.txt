cmake_minimum_required(VERSION 2.8.12) # 顺便顺应警告，调高了最低版本要求

project(SuperPointNet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

# ==================== CUDA 路径配置 ====================
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# ==================== TensorRT 路径配置 ====================
# 如果命令行没有传 -DTRT_DIR，则默认使用 /usr/local/TensorRT
if(NOT DEFINED TRT_DIR)
    set(TRT_DIR /usr/local/TensorRT)
endif()

message(STATUS "Using TensorRT directory: ${TRT_DIR}")

# 动态引入你指定的 TensorRT 路径
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)
# ========================================================

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(supernet ${PROJECT_SOURCE_DIR}/supernet.cpp ${PROJECT_SOURCE_DIR}/utils.cpp)
target_link_libraries(supernet nvinfer)
target_link_libraries(supernet cudart)
target_link_libraries(supernet ${OpenCV_LIBS})

add_definitions(-O2 -pthread)