#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

constexpr int BLOCKSIZE = 128;
constexpr int MAX_PREVS = 3;
constexpr int MAX_ARGS = 5;
constexpr int MAX_PARAM_TENSORS = 10;

// Op codes
enum class OpType {
    NONE = -1,
    MATMUL = 0,
    MEAN = 1,
    MUL = 2,
    RELU = 3,
    LOGSOFTMAX = 4
};

// CUDA kernel declarations
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K);
__global__ void matmul_transpose(float* A, float* B, float* C, int M, int N, int K);
__global__ void transpose_matmul(float* A, float* B, float* C, int M, int N, int K);
__global__ void relu_kernel(float* out, float* inp, size_t n);
__global__ void relu_backward_kernel(float* dinp, float* dout, float* out, size_t n);
__global__ void logsoftmax_kernel(float* out, float* inp, int B, int C, int strides_0, int strides_1);
__global__ void logsoftmax_backward_kernel(float* dinp, float* dout, float* out, int B, int C);
__global__ void mul_kernel(float* out, float* a, float* b, size_t n);
__global__ void mean_kernel(float* out, float* inp, size_t n);
__global__ void mean_backward_kernel(float* dinp, float* dout, size_t n);
__global__ void update_weights_kernel(float* w, float* grad, float lr, int size);

// Custom deleter for CUDA memory
struct CudaDeleter {
    void operator()(float* ptr) const {
        if (ptr) cudaFree(ptr);
    }
};

class Arr {
public:
    std::vector<float> values;
    std::unique_ptr<float[], CudaDeleter> cuda_values;
    std::vector<int> shape;
    std::vector<int> strides;
    int ndim;
    size_t size;

    Arr(const std::vector<float>& data, const std::vector<int>& shape_);
    Arr(const std::vector<int>& shape_); // Zero initialization
    
    void cpu_to_cuda();
    void cuda_to_cpu();
    
    ~Arr() = default;
};

class Tensor {
public:
    std::unique_ptr<Arr> data;
    std::unique_ptr<Arr> grad;
    OpType op;
    std::array<Tensor*, MAX_PREVS> prevs;
    int num_prevs;

    Tensor(const std::vector<float>& data, const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape); // Zero tensor
    
    ~Tensor() = default;
};

void checkCudaError(cudaError_t error);

std::unique_ptr<Tensor> mul(Tensor* a, Tensor* b);
std::unique_ptr<Tensor> mean(Tensor* t);
std::unique_ptr<Tensor> matmul(Tensor* a, Tensor* b);
std::unique_ptr<Tensor> relu(Tensor* inp);
std::unique_ptr<Tensor> logsoftmax(Tensor* inp);

void backward(Tensor* t);
void mul_backward(Tensor* out);
void mean_backward(Tensor* out);
void matmul_backward(Tensor* out);
void relu_backward(Tensor* out);
void logsoftmax_backward(Tensor* out);
Tensor* softmax(Tensor* x);

void update_weights(Tensor* w, float lr);

void print_tensor(Tensor* t);
std::unique_ptr<Tensor> im2col(Tensor* im, int kernel_height, int kernel_width, int padding);
std::unique_ptr<Tensor> col2im(Tensor* col, int num_filters, int kernel_height, int kernel_width, 
                                int im_height, int im_width);
std::unique_ptr<Tensor> conv2d(Tensor* input, Tensor* kernel, Tensor* bias, int padding = 0);
std::unique_ptr<Tensor> conv_transpose(Tensor* input, Tensor* kernel, Tensor* bias);
std::unique_ptr<Tensor> max_pool(Tensor* input, int pool_size);
std::unique_ptr<Tensor> cat(Tensor* a, Tensor* b);
std::unique_ptr<Tensor> argmax(Tensor* input);