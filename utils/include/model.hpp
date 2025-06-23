#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

struct Result
{
    int idx;          // 类别编号
    float confidence; // 置信度
    cv::Rect box;     // 检测框
};

class Model
{
private:
    int inputSize;        // 模型输入图像尺寸
    float scoreThreshold; // 置信度阈值
    float nmsThreshold;   // 非极大值抑制阈值

    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;
    void *buffers[2] = {nullptr, nullptr};
    std::vector<float> prob;

    int input_h, input_w;
    int output_h, output_w;
    float rx, ry;

    cv::Mat resizeFrame;

    cv::Mat preprocessing(cv::Mat &frame);
    void postprocessing();

public:
    std::vector<Result> detectResults;

    Model(const std::string modelPath, const int &inputSize, const float &scoreThreshol, const float &nmsThreshold);
    ~Model();

    bool Detect(cv::Mat &frame);
};

#endif