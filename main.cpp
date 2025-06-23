#include <yaml-cpp/yaml.h>

#include "utils/include/draw.hpp"

const YAML::Node modelConfig = YAML::LoadFile("/app/configs/detectConfig.yaml");                                      // 模型配置

const std::string modelPath = modelConfig["modelPath"].as<std::string>(); // 模型路径
const int inputSize = modelConfig["inputSize"].as<int>();                 // 图像输入大小
const float scoreThreshold = modelConfig["scoreThreshold"].as<float>();   // 置信度阈值
const float nmsThreshold = modelConfig["nmsThreshold"].as<float>();       // 非极大值抑制阈值
const std::vector<std::string> classNames = []()
{
    std::vector<std::string> tmp;
    for (const auto &item : modelConfig["classNames"])
        tmp.emplace_back(item.second.as<std::string>());

    return tmp;
}(); // 模型类别列表
const bool showFlag = modelConfig["showFlag"].as<bool>(); // 绘制, 展示检测结果标志

char waitKey_Flag;

int main(int argc, char const *argv[])
{

    Model detectModel = Model(modelPath, inputSize, scoreThreshold, nmsThreshold);
    cv::Mat frame = cv::imread("/app/assets/bus.jpg");
    if (frame.empty()) {
        assert("The Image is Empty, check the image path");
    }
    bool detectFlag;
    while (true)
    {
        detectFlag = detectModel.Detect(frame);
        if (! detectFlag){
            assert("Detect Failed");
        }
        // 绘制, 展示检测结果
        if (showFlag)
        {
            drawDetect(frame, detectModel.detectResults, classNames);
            cv::imshow("img", frame);
            waitKey_Flag = cv::waitKey(1);
            if (waitKey_Flag == 113) // q
                break;
        }
    }

    return 0;
}
