cmake_minimum_required(VERSION 3.5)
project(bytetrack-onnxrun-cpp)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
set(onnxruntime_INCLUDE_DIRS /Users/tunm/Downloads/onnxruntime-osx-x86_64-1.12.1/include)
include_directories(${onnxruntime_INCLUDE_DIRS})
include_directories(include)
include_directories(/usr/local/include/eigen3)
include_directories(${OpenCV_INCLUDE_DIRS})


set(MNN_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/MNN/darwin/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/MNN/darwin/include)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${MNN_LIBS})


include_directories(src/)
file(GLOB My_Source_Files src/*.cpp)

add_executable(scrfd_track ${My_Source_Files} exec/scrfd_track.cpp)
target_link_libraries(scrfd_track ${OpenCV_LIBS} MNN)