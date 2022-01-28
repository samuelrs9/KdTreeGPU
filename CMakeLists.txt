cmake_minimum_required (VERSION 3.13)

project (kdtree_gpu LANGUAGES CUDA CXX)

set (CMAKE_CXX_STANDARD 11)

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")

set(COMMON_SOURCE_LIST
    "src/buildKdTree.cu"
    "src/KdNode.cu"
    "src/Gpu.cu"
    "src/mergeSort.cu"
    "src/removeDups.cu")

set(COMMON_HEADERS_LIST
    "src/buildKdTree_common.h"
    "src/Gpu.h"
    "src/KdNode.h"
    "src/mergeSort_common.h"
    "src/removeDups_common.h")

set(EXEC_SOURCE_LIST
	"${COMMON_SOURCE_LIST}"
    #"src/main.cu"
    "main.cu")

set(EXEC_HEADERS_LIST
    "${COMMON_HEADERS_LIST}")
    
add_executable(${PROJECT_NAME} ${EXEC_SOURCE_LIST})

target_link_libraries(${PROJECT_NAME} cudart)
target_include_directories(${PROJECT_NAME} PUBLIC  "/usr/local/cuda/samples/common/inc")