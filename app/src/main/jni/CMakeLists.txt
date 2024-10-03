project(yolopv2ncnn)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.6.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230223-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolopv2ncnn SHARED yolopv2ncnn.cpp yolopv2.cpp ndkcamera.cpp yolov8.cpp yolov8.h)

target_link_libraries(yolopv2ncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
