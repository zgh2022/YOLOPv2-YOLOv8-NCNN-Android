// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#pragma once

#include <opencv2/opencv.hpp>
#include <android/log.h>
#include <net.h>
#include "cpu.h"
#include "layer.h"
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include "yolov8.h" // 添加这行


extern bool g_enable_drivable_area;
extern bool g_enable_lane_detection;
extern bool g_enable_object_detection;
extern float g_zoom;

//struct Object {
//    cv::Rect_<float> rect;
//    int label;
//    float prob;
//};

struct TimingInfo {
    double model_inference;
    double lane_area_draw;
    double lane_and_area;
    double object_detection;
    double object_drawing;
    double total_time;
};

class Yolopv2 {
public:
    Yolopv2();
    ~Yolopv2();
    int load(AAssetManager* mgr, bool use_gpu = false);
    void startThreads();
    void stopThreads();
    cv::Mat getLatestProcessedFrame();
    void updateLatestFrame(const cv::Mat& frame);
    TimingInfo getLatestTimingInfo() const;

private:
    Yolov8 yolov8; // 添加这个成员

    std::unique_ptr<ncnn::Net> yolopv2;  // 使用智能指针管理 ncnn::Net
    std::mutex net_mutex;                // 添加互斥锁保护网络访问

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;

    const int yolopv2_target_size = 320;
    const int yolov8_target_size = 640;
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

    std::vector<Object> objects;
    std::mutex objects_mutex;

    cv::Mat latest_frame;
    cv::Mat latest_processed_frame;
    std::mutex frame_mutex;
    std::condition_variable frame_cv;

    std::thread inference_thread;
    std::atomic<bool> stop_threads;

    void inferenceThreadFunction();
    int detect(cv::Mat& rgb, TimingInfo& timing);
    mutable std::mutex timing_mutex;
    TimingInfo latest_timing_info;
};