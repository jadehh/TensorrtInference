cmake_minimum_required(VERSION 3.22.1)
set(DEMO_NAME "tensorrt_inference")
project(${DEMO_NAME} LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)
SET(OPENCV_DIR "/usr/local/opencv" CACHE PATH "Location of libraries")
SET(TENSORRT_DIR "/usr/local/tensorrt" CACHE PATH "Location of libraries")
SET(CUDA_DIR "/usr/local/cuda" CACHE PATH "Location of libraries")
set(YAML_CPP_DIR "/usr/local/yaml" CACHE PATH "location of libraries")

if (NOT DEFINED OPENCV_DIR OR OPENCV_DIR STREQUAL "")
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/usr/local/opencv")
endif ()

if (NOT DEFINED TENSORRT_DIR OR TENSORRT_DIR STREQUAL "")
    message(FATAL_ERROR "please set TENSORRT_DIR with -DTENSORRT_DIR=/usr/local/tensorrt")
endif ()

if (NOT DEFINED CUDA_DIR OR CUDA_DIR STREQUAL "")
    message(FATAL_ERROR "please set CUDA_DIR with -DCUDA_DIR=/usr/local/cuda")
endif ()

if (NOT DEFINED YAML_CPP_DIR OR YAML_CPP_DIR STREQUAL "")
    message(FATAL_ERROR "please set YAML_CPP_DIR with -DYAML_CPP_DIR=/usr/local/yaml")
endif ()

if (NOT WIN32)
    set(OPENCV_INCLUDE_DIR "${OPENCV_DIR}/include/opencv4")
    set(OPENCV_LIB_DIR "${OPENCV_DIR}/lib")
    set(OPENCV_LIBS
            opencv_calib3d
            opencv_core
            opencv_dnn
            opencv_features2d
            opencv_flann
            opencv_highgui
            opencv_imgcodecs
            opencv_imgproc
            opencv_ml
            opencv_objdetect
            opencv_photo
            opencv_stitching
            opencv_video
            opencv_videoio)
else ()
    set(OPENCV_INCLUDE_DIR "${OPENCV_DIR}/build/include/")
    set(OPENCV_LIB_DIR "${OPENCV_DIR}/build/x64/vc15/lib")
    set(OPENCV_LIBS opencv_world455)
endif ()
set(TENSORRT_INCLUDE_DIR "${TENSORRT_DIR}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_DIR}/lib")

set(CUDA_INCLUDE_DIR "${CUDA_DIR}/include")
if (NOT WIN32)
    set(CUDA_LIB_DIR "${CUDA_DIR}/lib64")
else ()
    set(CUDA_LIB_DIR "${CUDA_DIR}/lib/x64")
endif ()

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${OPENCV_INCLUDE_DIR}
        ${TENSORRT_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIR}
)
link_directories(
        ${OPENCV_LIB_DIR}
        ${TENSORRT_LIB_DIR}
        ${CUDA_LIB_DIR}
)
set(CUDA_LIBS cudart)
if (NOT WIN32)
    set(TENSORRT_LIBS
            nvinfer
            nvonnxparser
            nvinfer_plugin
    )
else ()
    set(TENSORRT_LIBS
            nvinfer_10
            nvonnxparser_10
            nvinfer_plugin_10
    )
endif ()


# 消除警告
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-deprecated-declarations)
endif ()
AUX_SOURCE_DIRECTORY(./src SRCS)
add_executable(
        ${DEMO_NAME}
        main.cpp
        ${SRCS}
)
if (NOT WIN32)
    target_link_libraries(
            ${DEMO_NAME}
            pthread
            ${CUDA_LIBS}
            ${TENSORRT_LIBS}
            ${OPENCV_LIBS}
    )
else ()
    target_link_libraries(
            ${DEMO_NAME}
            ${CUDA_LIBS}
            ${TENSORRT_LIBS}
            ${OPENCV_LIBS}
    )
endif ()
if (WIN32)
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DIR}/build/x64/vc15/bin/opencv_world455d.dll ./DEBUG/opencv_world455d.dll
    )
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DIR}/build/x64/vc15/bin/opencv_world455.dll ./Release/opencv_world455.dll
    )
endif ()
