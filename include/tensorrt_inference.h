#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct Detection
{
    float conf;
    int class_id;
    Rect bbox;
};

class TensorrtInference
{

public:

    TensorrtInference(string model_path, nvinfer1::ILogger& logger);
    ~TensorrtInference();

    void preprocess(Mat& image) const;
    void infer() const;
    void postprocess(vector<Detection>& output) const;
    void draw(Mat& image, const vector<Detection>& output);

private:
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    float* gpu_buffers[3]{};               //!< The vector of device buffers needed for engine execution
    float* cpu_output_buffer{};

    cudaStream_t stream{}, pre_process_stream{} , post_process_stream{};
    IRuntime* runtime{};                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine{};               //!< The TensorRT engine used to run the network
    IExecutionContext* context{};        //!< The context for executing inference using an ICudaEngine

    // Model parameters
    int input_w;
    int input_h;
    int num_detections{};
    int detection_attribute_size{};
    int num_classes = 80;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    float conf_threshold = 0.3f;
    float nms_threshold = 0.4f;

    vector<Scalar> colors;

    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    bool saveEngine(const std::string& filename);
};