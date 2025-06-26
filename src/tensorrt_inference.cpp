#include "include/tensorrt_inference.h"
#include "include/logging.h"
#include "include/cuda_utils.h"
#include "include/preprocess.h"
#include "include/postprocess.h"
#include <NvOnnxParser.h>
#include "include/common.h"
#include <fstream>
#include <iostream>


static Logger logger;
#define isFP16 true
#define warmup true


TensorrtInference::TensorrtInference(string model_path, nvinfer1::ILogger &logger) {
    // Deserialize an engine
    if (model_path.find(".onnx") == std::string::npos) {
        init(model_path, logger);
    }
    // Build an engine from an onnx model
    else {
        build(model_path, logger);
        saveEngine(model_path);
    }

#if NV_TENSORRT_MAJOR < 10
    // Define input dimensions
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#endif
}


void TensorrtInference::init(std::string engine_path, nvinfer1::ILogger &logger) {
    // Read the engine file
    ifstream engineStream(engine_path, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the TensorrtInference engine
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // Get input and output sizes of the model


#if NV_TENSORRT_MAJOR < 10
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];
#else
    const auto [inputDims, input] = this->engine->getTensorShape("images");
    const auto [outputDims,  output] = this->engine->getTensorShape("output0");
    input_h = static_cast<int>(input[2]);
    input_w = static_cast<int>(input[3]);
    detection_attribute_size = static_cast<int>(output[1]);
    num_detections = static_cast<int>(output[2]);
#endif
    num_classes = detection_attribute_size - 4;
    // Initialize input buffers
    CUDA_CHECK(cudaMallocHost(&cpu_output_buffer, detection_attribute_size * num_detections * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    // Initialize output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));
    //
    CUDA_CHECK(cudaMalloc(&gpu_buffers[2], detection_attribute_size * num_detections * sizeof(float)));
    cuda_preprocess_init(MAX_IMAGE_SIZE);
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamCreate(&pre_process_stream));
    CUDA_CHECK(cudaStreamCreate(&post_process_stream));
    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

TensorrtInference::~TensorrtInference() {
    // Release stream and buffers
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (const auto & gpu_buffer : gpu_buffers)
        CUDA_CHECK(cudaFree(gpu_buffer));
    CUDA_CHECK(cudaFreeHost(cpu_output_buffer));
    // Destroy the engine
    cuda_preprocess_destroy();
    delete context;
    delete engine;
    delete runtime;
}

void TensorrtInference::preprocess(Mat &image) const {
    // Preprocessing data on gpu
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, pre_process_stream);
    CUDA_CHECK(cudaStreamSynchronize(pre_process_stream));
}

void TensorrtInference::infer() const {
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2((void **) gpu_buffers, stream, nullptr);
#else
    // 设置输入输出张量
    context->setTensorAddress("images", gpu_buffers[0]);
    context->setTensorAddress("output0",gpu_buffers[1]);
    this->context->enqueueV3(stream);
//    this->context->executeV2((void **) gpu_buffers);
#endif
}

void TensorrtInference::postprocess(vector<Detection> &output) const {
    const auto cuda_postprocess_start_time = static_cast<double>(cv::getTickCount());
    cuda_postprocess(gpu_buffers[1], num_detections, num_classes, conf_threshold, gpu_buffers[2], post_process_stream);
    std::cout <<"cuda_postprocess: " << ((static_cast<double>(cv::getTickCount()) - cuda_postprocess_start_time) / cv::getTickFrequency()) * 1000 << "ms,";
    const auto cuda_memcpy_start_time = static_cast<double>(cv::getTickCount());
    // 添加精确计时
    cudaEvent_t copy_start, copy_stop;
    cudaEventCreate(&copy_start);
    cudaEventCreate(&copy_stop);
    cudaEventRecord(copy_start, post_process_stream);
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[2], num_detections * detection_attribute_size * sizeof(float),cudaMemcpyDeviceToHost, post_process_stream));
    cudaEventRecord(copy_stop, post_process_stream);
    cudaEventSynchronize(copy_stop);
    float copy_ms = 0;
    cudaEventElapsedTime(&copy_ms, copy_start, copy_stop);
    std::cout << "D2H Copy: " << copy_ms << " ms, Data: " << (num_detections * detection_attribute_size * sizeof(float)/1024.0)<< "KB,";
    CUDA_CHECK(cudaStreamSynchronize(post_process_stream));
    std::cout << "cuda_memcpy_time: " <<  ((static_cast<double>(cv::getTickCount()) - cuda_memcpy_start_time) / cv::getTickFrequency()) *  1000 << "ms,";
    const auto nms_box_start_time =  static_cast<double>(cv::getTickCount());
    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;
    for (int i = 0; i < num_detections; ++i) {
        float class_id = cpu_output_buffer[i * 6 + 4];
        float conf = cpu_output_buffer[i * 6 + 5];
        if (class_id >= 0 && conf > conf_threshold) {
            float cx = cpu_output_buffer[i * 6 + 0];
            float cy = cpu_output_buffer[i * 6 + 1];
            float ow = cpu_output_buffer[i * 6 + 2];
            float oh = cpu_output_buffer[i * 6 + 3];
            Rect box;
            box.x = static_cast<int>(cx - 0.5f * ow);
            box.y = static_cast<int>(cy - 0.5f * oh);
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);
            boxes.push_back(box);
            class_ids.push_back(static_cast<int>(class_id));
            confidences.push_back(conf);
        }
    }
    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
    }
    std::cout << "nms_box_time: " <<  ((static_cast<double>(cv::getTickCount()) - nms_box_start_time) / cv::getTickFrequency()) *  1000 << "ms,";

}

void TensorrtInference::build(std::string onnxPath, nvinfer1::ILogger &logger) {
    auto builder = createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    IBuilderConfig *config = builder->createBuilderConfig();
    if (isFP16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger);
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    IHostMemory *plan{builder->buildSerializedNetwork(*network, *config)};

    runtime = createInferRuntime(logger);

    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    context = engine->createExecutionContext();

    delete network;
    delete config;
    delete parser;
    delete plan;
}

bool TensorrtInference::saveEngine(const std::string &onnxpath) {
    // Create an engine path from onnx path
    std::string engine_path;
    size_t dotIndex = onnxpath.find_last_of(".");
    if (dotIndex != std::string::npos) {
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    } else {
        return false;
    }

    // Save the engine to the path
    if (engine) {
        nvinfer1::IHostMemory *data = engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open()) {
            std::cout << "Create engine file" << engine_path << " failed" << std::endl;
            return 0;
        }
        file.write((const char *) data->data(), data->size());
        file.close();

        delete data;
    }
    return true;
}

void TensorrtInference::draw(Mat &image, const vector<Detection> &output) {
    const float ratio_h = input_h / (float) image.rows;
    const float ratio_w = input_w / (float) image.cols;

    for (int i = 0; i < output.size(); i++) {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        if (ratio_h > ratio_w) {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        } else {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }

        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Detection box text
        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, color, FILLED);
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}



