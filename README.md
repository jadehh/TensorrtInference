# 简介
> ultralytics的TensorRT10 C++实现

# 环境配置
使用Docker环境

```bash
docker pull jadehh/tensorrt:10.6-cuda-11.6.2-cudnn8-opencv4.8-yaml-devel
docker run -it --name tensorrt --gpus=all  -v ${PWD}:/app  -w /app  jadehh/tensorrt:10.6-cuda-11.6.2-cudnn8-opencv4.8-yaml-devel
```



# 部署
当环境配置正常，则该仓库代码部署也应该没什么问题.
一般`git`下来后，修改`CMakeLists.txt`中`TensorRT`的路径，以及`main.cpp`中的文件路径后，进行编译即可。

``` sh
trtexec --onnx=models/yolo11n.onnx --saveEngine=models/yolo11n.engine  --tacticSources="+cublas,-cublasLt"
mkdir build && cd build
cmake -D OPENCV_DIR=/usr/local/opencv -D TENSORRT_DIR=/usr/local/tensorrt -D CUDA_DIR=/usr/local/cuda -D YAML_DIR=/usr/local/yaml ..
make -j8
```