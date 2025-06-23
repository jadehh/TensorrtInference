#include <fstream>
#include <algorithm>
#include <numeric>

#include "include/model.hpp"

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kINFO) std::cout << msg << std::endl;
    }
} gLogger;

Model::Model(const std::string modelPath, const int &inputSize, const float &scoreThreshold, const float &nmsThreshold)
{
    this->inputSize = inputSize;
    this->scoreThreshold = scoreThreshold;
    this->nmsThreshold = nmsThreshold;

    std::ifstream engineFile(modelPath, std::ios::binary);
    std::vector<char> engineData;
    int fsize = 0;

    if (engineFile.good())
    {
        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        engineData.resize(fsize);
        engineFile.read(engineData.data(), fsize);
        engineFile.close();
    }

    this->runtime = nvinfer1::createInferRuntime(gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(engineData.data(), fsize);
    assert(this->engine != nullptr);

    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    // 获取输入输出维度
    nvinfer1::Dims inputDims = this->engine->getTensorShape("images");
    nvinfer1::Dims outputDims = this->engine->getTensorShape("output0");

    this->input_h = inputDims.d[2];
    this->input_w = inputDims.d[3];

    this->output_h = outputDims.d[1];
    this->output_w = outputDims.d[2];

    cudaMalloc(&(this->buffers[0]), this->input_h * this->input_w * 3 * sizeof(float));
    cudaMalloc(&(this->buffers[1]), this->output_h * this->output_w * sizeof(float));

    this->prob.resize(this->output_h * this->output_w);

    // Create stream
    cudaStreamCreate(&(this->stream));
}

Model::~Model()
{
    // 释放资源
    for (auto &buffer : this->buffers)
    {
        if (buffer)
            cudaFree(buffer);
    }
    if (this->context)
        delete this->context;
    if (this->engine)
        delete this->engine;
    if (this->runtime)
        delete this->runtime;
    if (this->stream)
        cudaStreamDestroy(this->stream);
}

cv::Mat Model::preprocessing(cv::Mat &frame)
{
    float frame_width = frame.cols;
    float frame_height = frame.rows;

    float r = float(this->inputSize / std::max(frame_width, frame_height));

    this->rx = frame_width / this->inputSize;
    this->ry = frame_height / this->inputSize;
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(0, 0, 0), true, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "blobFromImage time: " << elapsed.count()*1000 << " ms" << std::endl;

    return blob;
}

void Model::postprocessing()
{
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    cv::Mat det_output(this->output_h, this->output_w, CV_32F, (float *)prob.data());

    cv::Rect box;
    double score;
    cv::Point class_id_point;
    cv::Mat classes_scores;

    for (int idx = 0; idx < det_output.cols; ++idx)
    {
        classes_scores = det_output.col(idx).rowRange(4, this->output_h);
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > this->scoreThreshold)
        {
            const float cx = det_output.at<float>(0, idx);
            const float cy = det_output.at<float>(1, idx);
            const float ow = det_output.at<float>(2, idx);
            const float oh = det_output.at<float>(3, idx);

            box.x = static_cast<int>((cx - 0.5 * ow) * this->rx);
            box.y = static_cast<int>((cy - 0.5 * oh) * this->ry);
            box.width = static_cast<int>(ow * this->rx);
            box.height = static_cast<int>(oh * this->ry);

            boxes.push_back(box);
            classIds.push_back(class_id_point.x);
            confidences.push_back(score);
        }
    }

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, this->scoreThreshold, this->nmsThreshold, indexes);

    this->detectResults.reserve(indexes.size()); // 预分配空间
    for (int idx : indexes)
    {
        this->detectResults.emplace_back(Result{idx, confidences.at(idx), boxes.at(idx)});
    }
}

bool Model::Detect(cv::Mat &frame)
{
    try
    {
        this->detectResults.clear();
        auto start_time = static_cast<double>(cv::getTickCount());
        cv::Mat blob = preprocessing(frame);
        std::cout << "Speed:" << ((static_cast<double>(cv::getTickCount()) - start_time) / cv::getTickFrequency() )*1000 << "ms " << "preprocess,";
        auto inference_start_time = static_cast<double>(cv::getTickCount());
        cudaMemcpyAsync(buffers[0], blob.ptr<float>(), 3 * this->input_h * this->input_w * sizeof(float), cudaMemcpyHostToDevice, this->stream);
        context->executeV2(buffers);
        cudaMemcpyAsync(this->prob.data(), this->buffers[1], this->output_h * this->output_w * sizeof(float), cudaMemcpyDeviceToHost, this->stream);
        cudaStreamSynchronize(this->stream);
        std::cout <<  ((static_cast<double>(cv::getTickCount()) - inference_start_time) / cv::getTickFrequency())*1000 << "ms inference, " ;
        auto postprocess_start_time = static_cast<double>(cv::getTickCount());
        postprocessing();
        std::cout << ((static_cast<double>(cv::getTickCount()) - postprocess_start_time) / cv::getTickFrequency())*1000 << "ms postprocess, ";
        std::cout  << ( (static_cast<double>(cv::getTickCount()) - start_time) / cv::getTickFrequency())*1000 << "ms total" << std::endl;
        return !detectResults.empty();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }
}