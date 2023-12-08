#include "tensor.cuh"
#include <algorithm>
#include <numeric>

// CUDA error checking
void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
    }
    cudaDeviceSynchronize();
}

Arr::Arr(const std::vector<float>& data, const std::vector<int>& shape_)
    : shape(shape_), ndim(static_cast<int>(shape_.size())), cuda_values(nullptr) {
    
    strides.resize(ndim);
    size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = size;
        size *= shape[i];
    }
    
    values = data;
    values.resize(size, 0.0f);
}

Arr::Arr(const std::vector<int>& shape_)
    : shape(shape_), ndim(static_cast<int>(shape_.size())), cuda_values(nullptr) {
    
    strides.resize(ndim);
    size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = size;
        size *= shape[i];
    }
    
    values.resize(size, 0.0f);
}

void Arr::cpu_to_cuda() {
    if (cuda_values) return;
    
    float* raw_ptr = nullptr;
    checkCudaError(cudaMalloc(&raw_ptr, size * sizeof(float)));
    cuda_values.reset(raw_ptr);
    
    checkCudaError(cudaMemcpy(cuda_values.get(), values.data(), 
                              size * sizeof(float), cudaMemcpyHostToDevice));
}

void Arr::cuda_to_cpu() {
    if (!cuda_values) return;
    
    checkCudaError(cudaMemcpy(values.data(), cuda_values.get(), 
                              size * sizeof(float), cudaMemcpyDeviceToHost));
}

Tensor::Tensor(const std::vector<float>& data_vec, const std::vector<int>& shape)
    : data(std::make_unique<Arr>(data_vec, shape)),
      grad(std::make_unique<Arr>(shape)),
      op(OpType::NONE),
      num_prevs(0) {
    prevs.fill(nullptr);
    data->cpu_to_cuda();
}

Tensor::Tensor(const std::vector<int>& shape)
    : data(std::make_unique<Arr>(shape)),
      grad(std::make_unique<Arr>(shape)),
      op(OpType::NONE),
      num_prevs(0) {
    prevs.fill(nullptr);
}

__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_transpose(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

__global__ void transpose_matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[k * M + row] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void relu_kernel(float* out, float* inp, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fmaxf(inp[i], 0.0f);
    }
}

__global__ void relu_backward_kernel(float* dinp, float* dout, float* out, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dinp[i] += (out[i] > 0.0f) ? dout[i] : 0.0f;
    }
}

__global__ void logsoftmax_kernel(float* out, float* inp, int B, int C, 
                                   int strides_0, int strides_1) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float max_val = -INFINITY;
        for (int c = 0; c < C; c++) {
            float val = inp[b * strides_0 + c * strides_1];
            max_val = fmaxf(max_val, val);
        }
        
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            sum += expf(inp[b * strides_0 + c * strides_1] - max_val);
        }
        
        float logsum = logf(sum);
        for (int c = 0; c < C; c++) {
            out[b * strides_0 + c * strides_1] = 
                inp[b * strides_0 + c * strides_1] - max_val - logsum;
        }
    }
}

__global__ void logsoftmax_backward_kernel(float* dinp, float* dout, float* out, 
                                            int B, int C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float sum_dout = 0.0f;
        for (int c = 0; c < C; c++) {
            sum_dout += dout[b * C + c];
        }
        for (int c = 0; c < C; c++) {
            float sm = expf(out[b * C + c]);
            dinp[b * C + c] += dout[b * C + c] - sm * sum_dout;
        }
    }
}

__global__ void mul_kernel(float* out, float* a, float* b, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void mean_kernel(float* out, float* inp, size_t n) {
    __shared__ float sharedSum;
    if (threadIdx.x == 0) sharedSum = 0.0f;
    __syncthreads();
    
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += inp[i];
    }
    atomicAdd(&sharedSum, sum);
    __syncthreads();
    
    if (threadIdx.x == 0) out[0] = sharedSum / n;
}

__global__ void mean_backward_kernel(float* dinp, float* dout, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dinp[i] += dout[0] / n;
}

__global__ void update_weights_kernel(float* w, float* grad, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        w[i] -= lr * grad[i];
        grad[i] = 0.0f;
    }
}

std::unique_ptr<Tensor> exp(Tensor* x);
std::unique_ptr<Tensor> sub(Tensor* a, Tensor* b);
std::unique_ptr<Tensor> div(Tensor* a, Tensor* b);
std::unique_ptr<Tensor> reduce_max(Tensor* x, int dim, bool keepdim = false);
std::unique_ptr<Tensor> reduce_sum(Tensor* x, int dim, bool keepdim = false);

std::unique_ptr<Tensor> mul(Tensor* a, Tensor* b) {
    auto t = std::make_unique<Tensor>(a->data->shape);
    t->data->cpu_to_cuda();
    t->grad->cpu_to_cuda();
    
    int numBlocks = (t->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    mul_kernel<<<numBlocks, BLOCKSIZE>>>(t->data->cuda_values.get(), 
                                         a->data->cuda_values.get(), 
                                         b->data->cuda_values.get(), 
                                         t->data->size);
    checkCudaError(cudaGetLastError());
    
    t->op = OpType::MUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}

std::unique_ptr<Tensor> mean(Tensor* t) {
    auto m = std::make_unique<Tensor>(std::vector<int>{1});
    m->data->cpu_to_cuda();
    m->grad->cpu_to_cuda();
    
    mean_kernel<<<1, BLOCKSIZE>>>(m->data->cuda_values.get(), 
                                  t->data->cuda_values.get(), 
                                  t->data->size);
    checkCudaError(cudaGetLastError());
    
    m->op = OpType::MEAN;
    m->num_prevs = 1;
    m->prevs[0] = t;
    return m;
}

std::unique_ptr<Tensor> matmul(Tensor* a, Tensor* b) {
    int M = a->data->shape[0];
    int K = a->data->shape[1];
    int N = b->data->shape[1];
    
    auto t = std::make_unique<Tensor>(std::vector<int>{M, N});
    t->data->cpu_to_cuda();
    t->grad->cpu_to_cuda();
    
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(a->data->cuda_values.get(), 
                                                   b->data->cuda_values.get(), 
                                                   t->data->cuda_values.get(), 
                                                   M, N, K);
    checkCudaError(cudaGetLastError());
    t->op = OpType::MATMUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}

std::unique_ptr<Tensor> relu(Tensor* inp) {
    auto t = std::make_unique<Tensor>(inp->data->shape);
    t->data->cpu_to_cuda();
    t->grad->cpu_to_cuda();

    int numBlocks = (t->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    relu_kernel<<<numBlocks, BLOCKSIZE>>>(t->data->cuda_values.get(),
                                          inp->data->cuda_values.get(),
                                          t->data->size);
    checkCudaError(cudaGetLastError());

    t->op = OpType::RELU;
    t->num_prevs = 1;
    t->prevs[0] = inp;
    return t;
}

std::unique_ptr<Tensor> logsoftmax(Tensor* inp) {
    auto t = std::make_unique<Tensor>(inp->data->shape);
    t->data->cpu_to_cuda();
    t->grad->cpu_to_cuda();

    int numBlocks = (inp->data->shape[0] + BLOCKSIZE - 1) / BLOCKSIZE;
    logsoftmax_kernel<<<numBlocks, BLOCKSIZE>>>(t->data->cuda_values.get(),
                                                inp->data->cuda_values.get(),
                                                inp->data->shape[0],
                                                inp->data->shape[1],
                                                inp->data->strides[0],
                                                inp->data->strides[1]);
    checkCudaError(cudaGetLastError());

    t->op = OpType::LOGSOFTMAX;
    t->num_prevs = 1;
    t->prevs[0] = inp;
    return t;
}

void mul_backward(Tensor* out) {
    int numBlocks = (out->data->size + BLOCKSIZE - 1) / BLOCKSIZE;

    mul_kernel<<<numBlocks, BLOCKSIZE>>>(out->prevs[0]->grad->cuda_values.get(),
                                         out->grad->cuda_values.get(),
                                         out->prevs[1]->data->cuda_values.get(),
                                         out->data->size);

    mul_kernel<<<numBlocks, BLOCKSIZE>>>(out->prevs[1]->grad->cuda_values.get(),
                                         out->grad->cuda_values.get(),
                                         out->prevs[0]->data->cuda_values.get(),
                                         out->data->size);
    checkCudaError(cudaGetLastError());
}

void mean_backward(Tensor* out) {
    int numBlocks = (out->prevs[0]->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    mean_backward_kernel<<<numBlocks, BLOCKSIZE>>>(
        out->prevs[0]->grad->cuda_values.get(),
        out->grad->cuda_values.get(),
        out->prevs[0]->data->size);
    checkCudaError(cudaGetLastError());
}

void matmul_backward(Tensor* out) {
    int P = out->prevs[0]->data->shape[0];
    int Q = out->prevs[0]->data->shape[1];
    int R = out->prevs[1]->data->shape[1];

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocksA((Q + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (P + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_transpose<<<numBlocksA, threadsPerBlock>>>(
        out->grad->cuda_values.get(),
        out->prevs[1]->data->cuda_values.get(),
        out->prevs[0]->grad->cuda_values.get(),
        P, Q, R);

    dim3 numBlocksB((R + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (Q + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose_matmul<<<numBlocksB, threadsPerBlock>>>(
        out->prevs[0]->data->cuda_values.get(),
        out->grad->cuda_values.get(),
        out->prevs[1]->grad->cuda_values.get(),
        Q, R, P);

    checkCudaError(cudaGetLastError());
}

void relu_backward(Tensor* out) {
    int numBlocks = (out->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    relu_backward_kernel<<<numBlocks, BLOCKSIZE>>>(
        out->prevs[0]->grad->cuda_values.get(),
        out->grad->cuda_values.get(),
        out->data->cuda_values.get(),
        out->data->size);
    checkCudaError(cudaGetLastError());
}

void logsoftmax_backward(Tensor* out) {
    int numBlocks = (out->data->shape[0] + BLOCKSIZE - 1) / BLOCKSIZE;
    logsoftmax_backward_kernel<<<numBlocks, BLOCKSIZE>>>(
        out->prevs[0]->grad->cuda_values.get(),
        out->grad->cuda_values.get(),
        out->data->cuda_values.get(),
        out->data->shape[0],
        out->data->shape[1]);
    checkCudaError(cudaGetLastError());
}

void backward(Tensor* t) {
    switch(t->op) {
        case OpType::MUL:
            mul_backward(t);
            break;
        case OpType::MEAN:
            mean_backward(t);
            break;
        case OpType::MATMUL:
            matmul_backward(t);
            break;
        case OpType::RELU:
            relu_backward(t);
            break;
        case OpType::LOGSOFTMAX:
            logsoftmax_backward(t);
            break;
        case OpType::NONE:
            return;
    }

    for (int i = 0; i < t->num_prevs; i++) {
        if (t->prevs[i]) {
            backward(t->prevs[i]);
        }
    }
}

void update_weights(Tensor* w, float lr) {
    int numBlocks = (w->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    update_weights_kernel<<<numBlocks, BLOCKSIZE>>>(
        w->data->cuda_values.get(),
        w->grad->cuda_values.get(),
        lr,
        w->data->size);
    checkCudaError(cudaGetLastError());
}

void print_tensor(Tensor* t) {
    t->data->cuda_to_cpu();
    t->grad->cuda_to_cpu();

    std::cout << "Tensor(\n";
    std::cout << "\tdata: ";
    for (size_t i = 0; i < t->data->size; i++) {
        std::cout << t->data->values[i] << ",";
    }
    std::cout << "\n\tshape: ";
    for (int i = 0; i < t->data->ndim; i++) {
        std::cout << t->data->shape[i] << ",";
    }
    std::cout << "\n\tgrad: ";
    for (size_t i = 0; i < t->grad->size; i++) {
        std::cout << t->grad->values[i] << ",";
    }
    std::cout << "\n)\n";
}

Tensor* softmax(Tensor* x) {
    auto max_val = reduce_max(x, -1, true); // keepdim=true
    auto exp_x = exp(sub(x, max_val));
    auto sum_exp = reduce_sum(exp_x, -1, true);
    return div(exp_x, sum_exp);
}
    

__global__ void exp_kernel(float* out, float* inp, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(inp[i]);
}

__global__ void sub_kernel(float* out, float* a, float* b, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

__global__ void div_kernel(float* out, float* a, float* b, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / (b[i] + 1e-8f); // avoid div by zero
}

__global__ void im2col_kernel(const float* im, float* col,
                               int channels, int height, int width,
                               int kernel_height, int kernel_width,
                               int padding, int out_height, int out_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * kernel_height * kernel_width * out_height * out_width;
    
    if (idx < total) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int kw = (idx / (out_width * out_height)) % kernel_width;
        int kh = (idx / (out_width * out_height * kernel_width)) % kernel_height;
        int c = idx / (out_width * out_height * kernel_width * kernel_height);
        
        int h_in = h_out - padding + kh;
        int w_in = w_out - padding + kw;
        
        int col_idx = (c * kernel_height * kernel_width + kh * kernel_width + kw) * 
                      (out_height * out_width) + h_out * out_width + w_out;
        
        if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
            col[col_idx] = im[c * height * width + h_in * width + w_in];
        } else {
            col[col_idx] = 0.0f;
        }
    }
}

std::unique_ptr<Tensor> im2col(Tensor* im, int kernel_height, int kernel_width, int padding) {
    // im shape: [channels, height, width]
    int channels = im->data->shape[0];
    int height = im->data->shape[1];
    int width = im->data->shape[2];
    
    int out_height = height + 2 * padding - kernel_height + 1;
    int out_width = width + 2 * padding - kernel_width + 1;
    
    // col shape: [kernel_height * kernel_width * channels, out_height * out_width]
    auto col = std::make_unique<Tensor>(std::vector<int>{
        kernel_height * kernel_width * channels, 
        out_height * out_width
    });
    
    int total = channels * kernel_height * kernel_width * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    im2col_kernel<<<blocks, threads>>>(
        im->data->cuda_values.get(),
        col->data->cuda_values.get(),
        channels, height, width,
        kernel_height, kernel_width,
        padding, out_height, out_width
    );
    
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
    
    return col;
}

__global__ void col2im_kernel(const float* col, float* im,
                               int num_filters, int kernel_height, int kernel_width,
                               int im_height, int im_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_window_rows = im_height / kernel_height;
    int num_window_cols = im_width / kernel_width;
    int total = num_filters * im_height * im_width;
    
    if (idx < total) {
        int w_out = idx % im_width;
        int h_out = (idx / im_width) % im_height;
        int d = idx / (im_width * im_height);
        
        int kh = h_out % kernel_height;
        int kw = w_out % kernel_width;
        int h_window = h_out / kernel_height;
        int w_window = w_out / kernel_width;
        
        if (h_window < num_window_rows && w_window < num_window_cols) {
            int col_idx = (d * kernel_height * kernel_width + kh * kernel_width + kw) * 
                         (num_window_rows * num_window_cols) + h_window * num_window_cols + w_window;
            im[idx] = col[col_idx];
        } else {
            im[idx] = 0.0f;
        }
    }
}

std::unique_ptr<Tensor> col2im(Tensor* col, int num_filters, int kernel_height, int kernel_width,
                                int im_height, int im_width) {
    auto im = std::make_unique<Tensor>(std::vector<int>{num_filters, im_height, im_width});
    
    int total = num_filters * im_height * im_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    col2im_kernel<<<blocks, threads>>>(
        col->data->cuda_values.get(),
        im->data->cuda_values.get(),
        num_filters, kernel_height, kernel_width,
        im_height, im_width
    );
    
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
    
    return im;
}

std::unique_ptr<Tensor> conv2d(Tensor* input, Tensor* kernel, Tensor* bias, int padding) {
    // input shape: [channels, height, width]
    // kernel shape: [out_channels, in_channels, kernel_height, kernel_width]
    // bias shape: [out_channels]
    
    int in_channels = input->data->shape[0];
    int in_height = input->data->shape[1];
    int in_width = input->data->shape[2];
    
    int out_channels = kernel->data->shape[0];
    int kernel_height = kernel->data->shape[2];
    int kernel_width = kernel->data->shape[3];
    
    int out_height = in_height + 2 * padding - kernel_height + 1;
    int out_width = in_width + 2 * padding - kernel_width + 1;
    
    // Apply im2col
    auto col = im2col(input, kernel_height, kernel_width, padding);
    
    // Reshape kernel for matrix multiplication
    // kernel: [out_channels, in_channels * kernel_height * kernel_width]
    auto kernel_reshaped = std::make_unique<Tensor>(std::vector<int>{
        out_channels,
        in_channels * kernel_height * kernel_width
    });
    
    // Copy kernel data with reshaping
    int kernel_flat_size = in_channels * kernel_height * kernel_width;
    checkCudaError(cudaMemcpy(
        kernel_reshaped->data->cuda_values.get(),
        kernel->data->cuda_values.get(),
        out_channels * kernel_flat_size * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));
    
    // Matrix multiply: [out_channels, kernel_size] x [kernel_size, out_height * out_width]
    // Result: [out_channels, out_height * out_width]
    auto output_flat = matmul(kernel_reshaped.get(), col.get());
    
    // Reshape output to 3D
    auto output = std::make_unique<Tensor>(std::vector<int>{out_channels, out_height, out_width});
    
    // Copy and add bias
    int total = out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    auto add_bias_kernel = [] __device__ (float* output, const float* flat_output, const float* bias,
                                          int out_channels, int out_height, int out_width) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = out_channels * out_height * out_width;
        
        if (idx < total) {
            int c = idx / (out_height * out_width);
            output[idx] = flat_output[idx] + bias[c];
        }
    };
    
    add_bias_kernel<<<blocks, threads>>>(
        output->data->cuda_values.get(),
        output_flat->data->cuda_values.get(),
        bias->data->cuda_values.get(),
        out_channels, out_height, out_width
    );
    
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
    
    return output;
}

// ============================================================================
// Transposed Convolution Kernel
// ============================================================================
__global__ void conv_transpose_kernel(const float* input, const float* kernel, float* col,
                                      int in_channels, int in_height, int in_width,
                                      int out_channels, int kernel_height, int kernel_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_window_vol = out_channels * kernel_height * kernel_width;
    int out_num_windows = in_height * in_width;
    int total = out_window_vol * out_num_windows;
    
    if (idx < total) {
        int window_idx = idx % out_num_windows;
        int vol_idx = idx / out_num_windows;
        
        int h_in = window_idx / in_width;
        int w_in = window_idx % in_width;
        
        int kw = vol_idx % kernel_width;
        int kh = (vol_idx / kernel_width) % kernel_height;
        int f = vol_idx / (kernel_width * kernel_height);
        
        float sum = 0.0f;
        for (int d = 0; d < in_channels; ++d) {
            int kernel_idx = ((d * out_channels + f) * kernel_height + kh) * kernel_width + kw;
            int input_idx = (d * in_height + h_in) * in_width + w_in;
            sum += input[input_idx] * kernel[kernel_idx];
        }
        
        col[idx] = sum;
    }
}

std::unique_ptr<Tensor> conv_transpose(Tensor* input, Tensor* kernel, Tensor* bias) {
    // input shape: [in_channels, height, width]
    // kernel shape: [in_channels, out_channels, kernel_height, kernel_width]
    // bias shape: [out_channels]
    
    int in_channels = input->data->shape[0];
    int in_height = input->data->shape[1];
    int in_width = input->data->shape[2];
    
    int out_channels = kernel->data->shape[1];
    int kernel_height = kernel->data->shape[2];
    int kernel_width = kernel->data->shape[3];
    
    int out_height = in_height * kernel_height;
    int out_width = in_width * kernel_width;
    
    // Create column matrix
    int out_window_vol = out_channels * kernel_height * kernel_width;
    int out_num_windows = in_height * in_width;
    auto col = std::make_unique<Tensor>(std::vector<int>{out_window_vol, out_num_windows});
    
    // Initialize col to zero
    checkCudaError(cudaMemset(col->data->cuda_values.get(), 0, 
                              col->data->size * sizeof(float)));
    
    int total = out_window_vol * out_num_windows;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    conv_transpose_kernel<<<blocks, threads>>>(
        input->data->cuda_values.get(),
        kernel->data->cuda_values.get(),
        col->data->cuda_values.get(),
        in_channels, in_height, in_width,
        out_channels, kernel_height, kernel_width
    );
    
    checkCudaError(cudaGetLastError());
    
    // Convert col to image
    auto output = col2im(col.get(), out_channels, kernel_height, kernel_width, 
                        out_height, out_width);
    
    // Add bias
    total = out_channels * out_height * out_width;
    blocks = (total + threads - 1) / threads;
    
    auto add_bias_3d_kernel = [] __device__ (float* output, const float* bias,
                                             int out_channels, int out_height, int out_width) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = out_channels * out_height * out_width;
        
        if (idx < total) {
            int c = idx / (out_height * out_width);
            output[idx] += bias[c];
        }
    };
    
    add_bias_3d_kernel<<<blocks, threads>>>(
        output->data->cuda_values.get(),
        bias->data->cuda_values.get(),
        out_channels, out_height, out_width
    );
    
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
    
    return output;
}

__global__ void max_pool_kernel(const float* input, float* output,
                                int channels, int in_height, int in_width,
                                int pool_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_height = in_height / pool_size;
    int out_width = in_width / pool_size;
    int total = channels * out_height * out_width;
    
    if (idx < total) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c = idx / (out_width * out_height);
        
        float max_val = -INFINITY;
        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {
                int h_in = h_out * pool_size + i;
                int w_in = w_out * pool_size + j;
                int in_idx = (c * in_height + h_in) * in_width + w_in;
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
        
        output[idx] = max_val;
    }
}