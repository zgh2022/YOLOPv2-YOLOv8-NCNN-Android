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

#include "yolopv2.h"
#include <chrono>

#define MAX_STRIDE 32

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "yolopv2", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "yolopv2", __VA_ARGS__)

static void slice(const ncnn::Mat &in, ncnn::Mat &out, int start, int end, int axis) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_arithmetic = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_packed = true;

    std::unique_ptr<ncnn::Layer> op(ncnn::create_layer("Crop"));

    // set param
    ncnn::ParamDict pd;

    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);
    ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1);
    starts.fill(start);
    pd.set(9, starts);// start
    pd.set(10, ends);// end
    pd.set(11, axes);//axes

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);
}

static void interp(const ncnn::Mat &in, const float &scale, const int &out_w, const int &out_h,
                   ncnn::Mat &out) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_arithmetic = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_packed = true;

    std::unique_ptr<ncnn::Layer> op(ncnn::create_layer("Interp"));

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);// resize_type
    pd.set(1, scale);// height_scale
    pd.set(2, scale);// width_scale
    pd.set(3, out_h);// height
    pd.set(4, out_w);// width

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);
}

static inline float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}


static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}


TimingInfo Yolopv2::getLatestTimingInfo() const {
    std::lock_guard<std::mutex> lock(timing_mutex);
    return latest_timing_info;
}


Yolopv2::Yolopv2() : stop_threads(false) {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

Yolopv2::~Yolopv2() {
    stopThreads();

    std::lock_guard<std::mutex> lock(net_mutex);
    yolopv2.reset();  // 确保网络资源在其他操作之前被释放

    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
}

int Yolopv2::load(AAssetManager *mgr, bool use_gpu) {
    std::lock_guard<std::mutex> lock(net_mutex);

    // 重置并重新创建网络
    yolopv2.reset(new ncnn::Net());

    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolopv2->opt = ncnn::Option();
    yolopv2->opt.use_fp16_arithmetic = true;
    yolopv2->opt.use_fp16_packed = true;
    yolopv2->opt.use_fp16_storage = true;
#if NCNN_VULKAN
    yolopv2->opt.use_vulkan_compute = use_gpu;
#endif

    yolopv2->opt.num_threads = ncnn::get_big_cpu_count();
    yolopv2->opt.blob_allocator = &blob_pool_allocator;
    yolopv2->opt.workspace_allocator = &workspace_pool_allocator;

    int ret = yolopv2->load_param(mgr, "yolopv2.param");
    if (ret != 0) {
        return ret;
    }
    ret = yolopv2->load_model(mgr, "yolopv2.bin");
    if (ret != 0) {
        return ret;
    }

    const char* modeltype = "n"; // 或 "s"

    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};

    yolov8.load(mgr, modeltype, yolov8_target_size, mean_vals, norm_vals, use_gpu);

    return 0;
}
void Yolopv2::updateLatestFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex);
    frame.copyTo(latest_frame);
    frame_cv.notify_one();
}

void Yolopv2::startThreads() {
    stop_threads = false;
    inference_thread = std::thread(&Yolopv2::inferenceThreadFunction, this);
}

void Yolopv2::stopThreads() {
    stop_threads = true;
    frame_cv.notify_all();
    if (inference_thread.joinable()) {
        inference_thread.join();
    }
}

void Yolopv2::inferenceThreadFunction() {
    while (!stop_threads) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(frame_mutex);
            frame_cv.wait(lock, [this] { return !latest_frame.empty() || stop_threads; });
            if (stop_threads) break;
            latest_frame.copyTo(frame);
        }

        if (frame.empty()) {
            LOGE("Empty frame in inference thread");
            continue;
        }

        TimingInfo timing;
        int ret = detect(frame, timing);
        if (ret != 0) {
            LOGE("Detection failed with error code: %d", ret);
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest_processed_frame = frame;
        }
    }
}

cv::Mat Yolopv2::getLatestProcessedFrame() {
    std::lock_guard<std::mutex> lock(frame_mutex);
    return latest_processed_frame.clone();
}


int Yolopv2::detect(cv::Mat &rgb, TimingInfo& timing) {

    std::lock_guard<std::mutex> lock(net_mutex);  // 保护网络访问
    auto start = std::chrono::high_resolution_clock::now();

    // 清空检测结果
    {
        std::lock_guard<std::mutex> lock(objects_mutex);
        objects.clear();
    }
    ncnn::Mat da_seg_mask, ll_seg_mask;

    // 图像信息
    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // 应用缩放
    cv::Mat zoomed;
    cv::resize(rgb, zoomed, cv::Size(), g_zoom, g_zoom, cv::INTER_LINEAR);

    // 裁剪到原始大小
    int crop_x = (zoomed.cols - img_w) / 2;
    int crop_y = (zoomed.rows - img_h) / 2;
    cv::Rect roi(crop_x, crop_y, img_w, img_h);
    zoomed(roi).copyTo(rgb);

    // 图像缩放
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float) yolopv2_target_size / w;
        w = yolopv2_target_size;
        h = h * scale;
    } else {
        scale = (float) yolopv2_target_size / h;
        h = yolopv2_target_size;
        w = w * scale;
    }
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

    //输入tmp图像
    ncnn::Mat in, in_pad;

    //padding
    in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 114.f);
    in_pad.substract_mean_normalize(0, norm_vals);

    //run network
    {
        auto model_start = std::chrono::high_resolution_clock::now();

        ncnn::Extractor ex = yolopv2->create_extractor();
        ex.input("images", in_pad);

        if (g_enable_object_detection) {
            auto obj_start = std::chrono::high_resolution_clock::now();

            std::vector<Object> detected_objects;
            yolov8.detect(rgb, detected_objects);

            {
                std::lock_guard<std::mutex> lock(objects_mutex);
                objects = detected_objects; // 更新类成员
            }

            yolov8.draw(rgb, detected_objects);

            auto obj_end = std::chrono::high_resolution_clock::now();
            timing.object_detection = std::chrono::duration_cast<std::chrono::milliseconds>(obj_end - obj_start).count();
        }


        if (g_enable_drivable_area || g_enable_lane_detection) {
            auto da_ll_start = std::chrono::high_resolution_clock::now();
            //make mask for da,ll
            ncnn::Mat da, ll;
            ex.extract("677", da);
            ex.extract("769", ll);
            slice(da, da_seg_mask, hpad / 2, in_pad.h - hpad / 2, 1);
            slice(ll, ll_seg_mask, hpad / 2, in_pad.h - hpad / 2, 1);
            slice(da_seg_mask, da_seg_mask, wpad / 2, in_pad.w - wpad / 2, 2);
            slice(ll_seg_mask, ll_seg_mask, wpad / 2, in_pad.w - wpad / 2, 2);
            interp(da_seg_mask, 1 / scale, 0, 0, da_seg_mask);
            interp(ll_seg_mask, 1 / scale, 0, 0, ll_seg_mask);
            auto da_ll_end = std::chrono::high_resolution_clock::now();
            timing.lane_and_area = std::chrono::duration_cast<std::chrono::milliseconds>(da_ll_end - da_ll_start).count();
        }

        auto model_end = std::chrono::high_resolution_clock::now();
        timing.model_inference = std::chrono::duration_cast<std::chrono::milliseconds>(model_end - model_start).count();
    }


    if (g_enable_drivable_area || g_enable_lane_detection) {
        auto da_ll_start = std::chrono::high_resolution_clock::now();

        const float* da_ptr = (float*)da_seg_mask.data;
        const float* ll_ptr = (float*)ll_seg_mask.data;
        int ww = da_seg_mask.w;
        int hh = da_seg_mask.h;
        for (int i = 0; i < hh; i++) {
            auto* image_ptr = rgb.ptr<cv::Vec3b>(i);
            for (int j = 0; j < ww; j++) {
                if (g_enable_drivable_area && da_ptr[i * ww + j] < da_ptr[ww * hh + i * ww + j]) {
                    image_ptr[j] = cv::Vec3b(0, 255, 0);
                }

                if (g_enable_lane_detection && std::round(ll_ptr[i * ww + j]) == 1.0) {
                    image_ptr[j] = cv::Vec3b(255, 0, 0);
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        if (g_enable_drivable_area || g_enable_lane_detection) {
            timing.lane_area_draw = std::chrono::duration_cast<std::chrono::milliseconds>(end - da_ll_start).count();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    timing.total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    {
        std::lock_guard<std::mutex> lock(timing_mutex);
        latest_timing_info = timing;
    }

    return 0;
}