#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "include/tensorrt_inference.h"


bool IsPathExist(const string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer{};
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

/**
 * @brief Setting up Tensorrt logger
*/
class Logger final : public nvinfer1::ILogger {
    void log(const Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

int main(int argc, char** argv)
{
    const string engine_file_path{ argv[1] };
    const string path{ argv[2] };
    vector<string> imagePathList;
    bool                     isVideo{ false };
    assert(argc == 3);

    if (IsFile(path))
    {
        string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv" || suffix == "webm")
        {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            abort();
        }
    }
    else if (IsPathExist(path))
    {
        glob(path + "/*.jpg", imagePathList);
    }

    // Assume it's a folder, add logic to handle folders
    // init model
    TensorrtInference model(engine_file_path, logger);

    if (isVideo) {
        //path to video
        cv::VideoCapture cap(path);

        while (true)
        {
            Mat image;
            cap >> image;

            if (image.empty()) break;

            vector<Detection> objects;
            model.preprocess(image);

            auto start = std::chrono::system_clock::now();
            model.infer();
            auto end = std::chrono::system_clock::now();

            model.postprocess(objects);
            model.draw(image, objects);

            auto tc = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.;
            printf("cost %2.4lf ms\n", tc);

            imshow("prediction", image);
            waitKey(1);
        }

        // Release resources
        destroyAllWindows();
        cap.release();
    }
    else {
        std::cout << std::fixed << std::setprecision(2); // 固定小数格式 + 两位精度[1,3,4,6](@ref)
        int count = 1000;
        auto start_time = static_cast<double>(cv::getTickCount());
        // path to folder saves images
        for (const auto& imagePath : imagePathList) {
            // open image
            for (int i = 0; i< count; i++) {
                Mat image = imread(imagePath, cv::IMREAD_COLOR);
                if (image.empty()) {
                   cerr << "Error reading image: " << imagePath << endl;
                   continue;
               }
//               auto preprocess_start_time = static_cast<double>(cv::getTickCount());
//               vector<Detection> objects;
//               model.preprocess(image);
//               std::cout << "preprocess: " << (static_cast<double>(cv::getTickCount()) - preprocess_start_time) / cv::getTickFrequency() * 1000 << "ms,";
//               auto inference_start_time = static_cast<double>(cv::getTickCount());
//               model.infer();
//               std::cout << "inference: " << ((static_cast<double>(cv::getTickCount()) - inference_start_time) / cv::getTickFrequency()) * 1000 << "ms,";
//               auto postprocess_start_time = static_cast<double>(cv::getTickCount());
//               model.postprocess(objects);
//               std::cout <<"postprocess: "<<  ((static_cast<double>(cv::getTickCount()) - postprocess_start_time) / cv::getTickFrequency()) * 1000 << "ms,";
//               std::cout << "total: "<<  ((static_cast<double>(cv::getTickCount()) - preprocess_start_time) / cv::getTickFrequency()) * 1000 << "ms," << std::endl;
//                namedWindow("Result",0);
//                model.draw(image, objects);
//                imshow("Result", image);
//                waitKey(1);
           }

        }
        std::cout << "all time: " <<((static_cast<double>(cv::getTickCount()) - start_time) / cv::getTickFrequency()) << "ms" << std::endl;
    }

    return 0;
}