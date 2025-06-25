# 简介
> ultralytics的TensorRT10 C++实现

# 环境配置
## ocker环境

```bash
docker pull jadehh/tensorrt:10.6-cuda-11.6.2-cudnn8-opencv4.8-yaml-devel
docker run -it --name tensorrt --gpus=all  -v ${PWD}:/app  -w /app  jadehh/tensorrt:10.6-cuda-11.6.2-cudnn8-opencv4.8-yaml-devel
trtexec --onnx=models/yolo11n.onnx --saveEngine=models/yolo11n.engine  --tacticSources="+cublas,-cublasLt"
mkdir build && cd build
cmake -D OPENCV_DIR=/usr/local/opencv -D TENSORRT_DIR=/usr/local/tensorrt -D CUDA_DIR=/usr/local/cuda  ..
make -j8
```

## Windows环境
```bash
cmake -G "Visual Studio 15 2017" -A x64 -D OPENCV_DIR=D:\SDKS\opencv-4.5.5 -D TENSORRT_DIR=D:\SDKS\TensorRT-10.6.0.26 -D CUDA_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"
trtexec --onnx=models/yolo11n.onnx --saveEngine=models/yolo11n.engine  --tacticSources="+cublas,-cublasLt"
```

