# Opencv

## Windows下编译Opencv CUDA 

以下是**Windows平台下编译支持CUDA的OpenCV的详细文档**，整合了多篇权威指南的核心步骤和避坑要点：

---

### 一、环境准备（关键！）
1. **硬件与驱动**
    - NVIDIA显卡（支持CUDA，算力≥3.5）
    - 安装最新[NVIDIA驱动](https://www.nvidia.com/drivers)

2. **软件依赖**
    - **CUDA Toolkit**：推荐11.x或12.x（[官网下载](https://developer.nvidia.com/cuda-toolkit-archive)）
    - **cuDNN**：需与CUDA版本匹配（[官网下载](https://developer.nvidia.com/cudnn)）
    - **Visual Studio**：2017/2019/2022（安装时勾选“C++桌面开发”）
    - **CMake** ≥ 3.10（[官网下载](https://cmake.org/download/)）
    - **OpenCV源码**：包含主库和contrib模块（版本需一致）
      ```bash
      git clone https://github.com/opencv/opencv.git
      git clone https://github.com/opencv/opencv_contrib.git
      ```
---

### 二、编译配置（CMake关键步骤）
1. **打开CMake GUI**
    - `Where is the source code`：指向`opencv`源码目录
    - `Where to build the binaries`：新建一个构建目录（如`build`）

2. **首次配置**
    - 点击`Configure`，选择匹配的Visual Studio版本和`x64`架构

3. **启用CUDA相关选项**

   | **选项** | **值/操作** | **说明** |
   |---|---|----|
   | `WITH_CUDA` | ✅勾选 | 启用CUDA支持 |
   | `OPENCV_DNN_CUDA` | ✅勾选 | 启用DNN模块的CUDA加速 |
   | `OPENCV_EXTRA_MODULES_PATH` | 指向`opencv_contrib/modules` | 启用额外算法（如SIFT） |
   | `CUDA_ARCH_BIN` | 填写显卡算力（如`7.5`） | [算力查询表](https://developer.nvidia.com/cuda-gpus) |
   | `BUILD_opencv_world` | ✅勾选（可选） | 生成单个整合库文件 |
   | `OPENCV_ENABLE_NONFREE` | ✅勾选 | 启用专利算法（如SURF） |
   | `ENABLE_FAST_MATH` + `CUDA_FAST_MATH` | ✅勾选 | 加速数学计算 |

4. **解决依赖下载问题**
    - 若出现`ippicv`、`ffmpeg`下载失败：  
      手动下载缺失文件（根据CMake报错中的URL），放入`opencv/.cache`对应子目录

5. **生成VS工程**
    - 点击`Generate`生成Visual Studio解决方案（`.sln`文件）

---

### 三、编译与安装（Visual Studio）
1. **打开解决方案**
    - 在构建目录（如`build`）中找到`OpenCV.sln`，用VS打开

2. **编译选项**
    - 顶部菜单切换为`Release` + `x64`
    - 右键解决方案 → `生成` → 选择`ALL_BUILD`（先编译） → 再选`INSTALL`（安装）

3. **等待编译完成**
    - 耗时约1-3小时（取决于硬件）
    - 成功后在`install`目录生成库文件（默认路径：`build/install`）

---

### 四、验证安装
1. **Python环境验证**
   ```python
   import cv2
   print("OpenCV Version:", cv2.__version__)
   print("CUDA Devices:", cv2.cuda.getCudaEnabledDeviceCount())
   ```
    - 若输出设备数≥1，则CUDA启用成功

2. **C++项目配置**
    - 包含目录：添加`install/include`
    - 库目录：添加`install/x64/vc17/lib`（根据VS版本调整）
    - 链接器 → 输入：添加`opencv_world4xx.lib`（若启用world库）

---

### 五、常见问题解决
| **问题** | **解决方案** |
|----------|--------------|
| CMake报错`Could NOT find CUDA` | 检查环境变量`CUDA_PATH`是否存在（安装CUDA后自动生成） |
| 编译卡死在99% | 关闭杀毒软件，避免占用进程 |
| 运行时报`cudnn64_8.dll`缺失 | 将cuDNN的`bin`目录加入系统PATH |
| GPU算力不匹配 | 重编译时修正`CUDA_ARCH_BIN`（如RTX 3060填`8.6`） |
| OpenCV DNN无法使用CUDA | 检查`OPENCV_DNN_CUDA=ON`是否启用，并安装对应cuDNN版本 |

---

### 六、优化建议
1. **多线程编译**：在VS中启用`/MP`选项（项目属性 → C/C++ → 所有选项 → `Multi-processor Compilation`设为`Yes`）
2. **减少编译时间**：在CMake中禁用不需要的模块（如`BUILD_TESTS=OFF`）
3. **版本兼容性**：OpenCV 4.6+需CUDA ≥11.0，推荐使用最新稳定版组合

> 完整编译好的库文件（含CUDA）可移植条件：目标机器需相同CUDA版本+相同显卡算力。

**参考资料**：
- [CUDA算力对照表](https://developer.nvidia.com/cuda-gpus)
- [OpenCV官方编译指南](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html)