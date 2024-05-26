cmake_minimum_required(VERSION 3.5)
project(bytetrack-onnxrun-cpp)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})


set(MNN_LIBS /home/tunm/softwate/MNN-2.7.2/build_cuda/)
include_directories(/home/tunm/softwate/MNN-2.7.2/build_cuda/include)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${MNN_LIBS})


include_directories(src/)
file(GLOB My_Source_Files src/*.cpp)

add_executable(scrfd_track ${My_Source_Files} exec/scrfd_track.cpp)
target_link_libraries(scrfd_track ${OpenCV_LIBS} MNN)