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

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "yolopv2.h"
#include "ndkcamera.h"
#include <chrono>
#include <mutex>
#include <memory>

#define TAG "Yolopv2Ncnn"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static std::unique_ptr<Yolopv2> g_yolopv2;
static std::mutex g_mutex;

bool g_enable_drivable_area = true;
bool g_enable_lane_detection = true;
bool g_enable_object_detection = true;
float g_zoom = 1.0f;

static TimingInfo g_timing_info;
static std::chrono::time_point<std::chrono::high_resolution_clock> g_last_frame_time;

class MyNdkCamera : public NdkCameraWindow {
public:
    virtual void on_image_render(cv::Mat &rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat &rgb) const {
    TimingInfo timing_info;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_yolopv2) {
            g_yolopv2->updateLatestFrame(rgb);
            cv::Mat processed = g_yolopv2->getLatestProcessedFrame();
            if (!processed.empty()) {
                processed.copyTo(rgb);
            }
            timing_info = g_yolopv2->getLatestTimingInfo();
        }
    }

//    // 计算 FPS
//    auto current_time = std::chrono::high_resolution_clock::now();
//    double fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(current_time - g_last_frame_time).count();
//    g_last_frame_time = current_time;

    // 准备时间信息字符串
    std::vector<std::string> info_lines = {
//            "FPS: " + std::to_string(static_cast<int>(std::round(fps))),
            "Model: " + std::to_string(static_cast<int>(timing_info.model_inference)) + " ms",
            "L/A: " + std::to_string(static_cast<int>(timing_info.lane_and_area)) + " ms",
            "Obj Det: " + std::to_string(static_cast<int>(timing_info.object_detection)) + " ms",
    };

    // 绘制时间信息
    int base_line = 0;
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.4;
    int thickness = 1;
    cv::Scalar text_color(255, 255, 255);  // 白色
    cv::Scalar bg_color(0, 0, 0, 128);     // 半透明黑色背景

    int y = 10;
    for (const auto& line : info_lines) {
        cv::Size text_size = cv::getTextSize(line, font_face, font_scale, thickness, &base_line);
        cv::rectangle(rgb, cv::Point(5, y - text_size.height),
                      cv::Point(10 + text_size.width, y + base_line),
                      bg_color, cv::FILLED);
        cv::putText(rgb, line, cv::Point(10, y), font_face, font_scale, text_color, thickness);
        y += text_size.height + 5;
    }
}

static std::unique_ptr<MyNdkCamera> g_camera;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    LOGI("JNI_OnLoad");
    g_camera.reset(new MyNdkCamera());
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    LOGI("JNI_OnUnload");
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_yolopv2.reset();
    }
    g_camera.reset();
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolopv2ncnn_Yolopv2Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint core) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) {
        LOGE("AssetManager 为空");
        return JNI_FALSE;
    }

    bool use_gpu = (int)core == 1;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (use_gpu && ncnn::get_gpu_count() == 0) {
            return JNI_FALSE;
        }
        g_yolopv2.reset(new Yolopv2());
        if (g_yolopv2->load(mgr, use_gpu) == 0) {
            g_yolopv2->startThreads();
            return JNI_TRUE;
        }
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolopv2ncnn_Yolopv2Ncnn_openCamera(JNIEnv* env, jobject thiz) {
    if (g_camera) {
        return g_camera->open() ? JNI_TRUE : JNI_FALSE;
    }
    return JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolopv2ncnn_Yolopv2Ncnn_closeCamera(JNIEnv* env, jobject thiz) {
    if (g_camera) {
        g_camera->close();
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolopv2ncnn_Yolopv2Ncnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface) {
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);
    if (g_camera && win) {
        g_camera->set_window(win);
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_tencent_yolopv2ncnn_Yolopv2Ncnn_enableDrivableArea(JNIEnv *env, jobject thiz, jboolean enable) {
    g_enable_drivable_area = enable;
}

JNIEXPORT void JNICALL
Java_com_tencent_yolopv2ncnn_Yolopv2Ncnn_enableLaneDetection(JNIEnv *env, jobject thiz, jboolean enable) {
    g_enable_lane_detection = enable;
}

JNIEXPORT void JNICALL
Java_com_tencent_yolopv2ncnn_Yolopv2Ncnn_enableObjectDetection(JNIEnv *env, jobject thiz, jboolean enable) {
    g_enable_object_detection = enable;
}

JNIEXPORT void JNICALL
Java_com_tencent_yolopv2ncnn_Yolopv2Ncnn_setZoom(JNIEnv *env, jobject thiz, jfloat zoom) {
    g_zoom = zoom;
    __android_log_print(ANDROID_LOG_DEBUG, "Yolopv2Ncnn", "Zoom set to %f", g_zoom);
}

}