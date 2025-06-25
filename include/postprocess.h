#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

void cuda_postprocess(
        const float *det_output, // [attr_size, num_boxes]
        int num_boxes, int num_classes, float conf_threshold,
        float *output, // [6, num_boxes]，在外部分配好！用来存核函数的计算结果
        cudaStream_t stream);