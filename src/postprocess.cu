#include "include/cuda_utils.h"
#include "include/postprocess.h"
#include "device_launch_parameters.h"

__global__ void
postprocess_kernel(const float *det_output, // [attr_size, num_boxes]
                   int num_boxes, int num_classes, float conf_threshold,
                   float *out_boxes // [6, num_boxes]: cx,cy,w,h,class_id,conf
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boxes)
        return;

    // attr-major: det_output[attr * num_boxes + i]
    float cx = det_output[0 * num_boxes + i];
    float cy = det_output[1 * num_boxes + i];
    float w = det_output[2 * num_boxes + i];
    float h = det_output[3 * num_boxes + i];

    int best_class = -1;
    float best_score = -1e10f;
    for (int c = 0; c < num_classes; ++c) {
        float score = det_output[(4 + c) * num_boxes + i];
        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }

    if (best_score > conf_threshold) {
        out_boxes[i * 6 + 0] = cx;
        out_boxes[i * 6 + 1] = cy;
        out_boxes[i * 6 + 2] = w;
        out_boxes[i * 6 + 3] = h;
        out_boxes[i * 6 + 4] = static_cast<float>(best_class);
        out_boxes[i * 6 + 5] = best_score;
    } else {
        out_boxes[i * 6 + 4] = -1;
        out_boxes[i * 6 + 5] = 0.0f;
    }
}

void cuda_postprocess(
        const float *det_output, // [attr_size, num_boxes]
        int num_boxes, int num_classes, float conf_threshold,
        float *output, // [6, num_boxes] -> cx,cy,w,h,class_id,conf
        cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_boxes + threads - 1) / threads;
    postprocess_kernel<<<blocks, threads, 0, stream>>>(det_output, num_boxes, num_classes, conf_threshold, output);
    // 注意：output 需要在 host 端进一步筛选 class_id >= 0 的结果
}