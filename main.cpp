#include "tensor.cuh"
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <iostream>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/time.h>
#endif

constexpr float M_PI_CONST = 3.14159265358979323846f;

// Timer utility class
class Timer {
#ifdef _WIN32
    LARGE_INTEGER start_time;
    LARGE_INTEGER frequency;
    
public:
    Timer() {
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start_time);
    }
    
    void reset() {
        QueryPerformanceCounter(&start_time);
    }
    
    double elapsed() const {
        LARGE_INTEGER end_time;
        QueryPerformanceCounter(&end_time);
        return static_cast<double>(end_time.QuadPart - start_time.QuadPart) / 
               static_cast<double>(frequency.QuadPart);
    }
#else
    struct timeval start_time;
    
public:
    Timer() {
        gettimeofday(&start_time, nullptr);
    }
    
    void reset() {
        gettimeofday(&start_time, nullptr);
    }
    
    double elapsed() const {
        struct timeval end_time;
        gettimeofday(&end_time, nullptr);
        return (end_time.tv_sec - start_time.tv_sec) + 
               (end_time.tv_usec - start_time.tv_usec) / 1e6;
    }
#endif
    
    void print_start() const {
        auto t = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(t);
        std::cout << "Start Time: " << std::ctime(&time_t);
    }
    
    void print_elapsed(const std::string& label = "Elapsed Time") const {
        std::cout << label << ": " << std::fixed << std::setprecision(6) 
                  << elapsed() << " seconds\n";
    }
};

void load_csv(Tensor* x, Tensor* y, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << "\n";
        std::exit(1);
    }

    std::string line;
    int batch_idx = 0;
    
    while (std::getline(file, line) && batch_idx < 60000) {
        std::stringstream ss(line);
        std::string value;
        int idx = 0;
        
        while (std::getline(ss, value, ',')) {
            if (idx >= 785) {
                std::cerr << "CSV format error: too many columns\n";
                file.close();
                std::exit(1);
            }
            
            float val = std::stof(value);
            
            if (idx == 0) {
                // First column is the label - convert to one-hot
                int label = static_cast<int>(val);
                for (int k = 0; k < 10; k++) {
                    if (k == label) {
                        y->data->values[batch_idx * 10 + k] = -1.0f;
                    } else {
                        y->data->values[batch_idx * 10 + k] = 0.0f;
                    }
                }
            } else {
                // Pixels (optionally normalize)
                x->data->values[batch_idx * 784 + (idx - 1)] = val / 255.0f;
            }
            idx++;
        }
        
        if (idx != 785) {
            std::cerr << "CSV format error: expected 785 columns, got " << idx 
                      << " at row " << batch_idx << "\n";
            file.close();
            std::exit(1);
        }
        
        batch_idx++;
    }
    
    if (batch_idx < 60000) {
        std::cerr << "Warning: Expected 60000 samples, but loaded " 
                  << batch_idx << " samples\n";
    }
    
    file.close();
    std::cout << "Successfully loaded " << batch_idx << " samples\n";
}

void load_test_csv(Tensor* x, Tensor* y, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open test file: " << filename << "\n";
        std::exit(1);
    }

    std::string line;
    int batch_idx = 0;
    constexpr int N_TEST = 10000;

    while (std::getline(file, line) && batch_idx < N_TEST) {
        std::stringstream ss(line);
        std::string value;
        int idx = 0;

        while (std::getline(ss, value, ',')) {
            if (idx >= 785) {
                std::cerr << "Test CSV format error: too many columns\n";
                file.close();
                std::exit(1);
            }

            float val = std::stof(value);

            if (idx == 0) {
                int label = static_cast<int>(val);
                for (int k = 0; k < 10; k++) {
                    y->data->values[batch_idx * 10 + k] = (k == label) ? -1.0f : 0.0f;
                }
            } else {
                x->data->values[batch_idx * 784 + (idx - 1)] = val / 255.0f;
            }
            idx++;
        }

        if (idx != 785) {
            std::cerr << "Test CSV format error at row " << batch_idx << ": got " << idx << " cols\n";
            file.close();
            std::exit(1);
        }
        batch_idx++;
    }

    file.close();
    std::cout << "Loaded " << batch_idx << " test samples\n";
}

void get_random_batch(Tensor* batch_x, Tensor* batch_y, 
                      Tensor* x, Tensor* y, int B) {
    static std::mt19937 gen(0);  
    static bool initialized = false;
    
    if (!initialized) {
        gen.seed(0);
        initialized = true;
    }
    
    std::uniform_int_distribution<> dis(0, x->data->shape[0] - 1);
    std::vector<bool> used_indices(x->data->shape[0], false);
    
    for (int i = 0; i < B; i++) {
        int index;
        do {
            index = dis(gen);
        } while (used_indices[index]);
        used_indices[index] = true;

        for (int j = 0; j < 784; j++) {
            int x_index = index * x->data->strides[0] + j;
            int batch_x_index = i * batch_x->data->strides[0] + j;
            batch_x->data->values[batch_x_index] = x->data->values[x_index];
        }

        for (int k = 0; k < 10; k++) {
            int y_index = index * y->data->strides[0] + k * y->data->strides[1];
            int batch_y_index = i * batch_y->data->strides[0] + k * batch_y->data->strides[1];
            batch_y->data->values[batch_y_index] = y->data->values[y_index];
        }
    }

    batch_x->data->cpu_to_cuda();
    batch_y->data->cpu_to_cuda();
    batch_x->grad->cpu_to_cuda();
    batch_y->grad->cpu_to_cuda();
}

class RandomGenerator {
    std::mt19937 gen;
    
public:
    explicit RandomGenerator(unsigned seed = 42) : gen(seed) {}
    
    float random_normal() {
        static std::normal_distribution<float> dist(0.0f, 1.0f);
        return dist(gen);
    }
    
    float rand_float() {
        static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(gen);
    }
    
    float rand_range(float min, float max) {
        return min + rand_float() * (max - min);
    }
    
    float kaiming_uniform(int fan_in) {
        float gain = std::sqrt(2.0f);  // for ReLU activation
        float std = gain / std::sqrt(static_cast<float>(fan_in));
        float bound = std::sqrt(3.0f) * std;
        return rand_range(-bound, bound);
    }
    
    float kaiming_init(int fan_in) {
        float std_dev = std::sqrt(2.0f / fan_in);
        return random_normal() * std_dev;
    }
};

float evaluate_accuracy(Tensor* x_test, Tensor* y_test, Tensor* w1, Tensor* w2) {
    const int N = x_test->data->shape[0]; 
    int correct = 0;

    x_test->data->cpu_to_cuda();
    y_test->data->cpu_to_cuda();

    auto w1_out = matmul(x_test, w1);
    auto relu_out = relu(w1_out.get());
    auto w2_out = matmul(relu_out.get(), w2);
    auto probs = softmax(w2_out.get()); 

    probs->data->cuda_to_cpu();

    for (int i = 0; i < N; ++i) {
        float max_prob = -1.0f;
        int pred = -1;
        int true_label = -1;

        for (int k = 0; k < 10; ++k) {
            if (probs->data->values[i * 10 + k] > max_prob) {
                max_prob = probs->data->values[i * 10 + k];
                pred = k;
            }
            if (y_test->data->values[i * 10 + k] == -1.0f) {
                true_label = k;
            }
        }

        if (pred == true_label) correct++;
    }

    return static_cast<float>(correct) / N;
}

int main() {
    try {
        checkCudaError(cudaSetDevice(0));
        
        auto x = std::make_unique<Tensor>(std::vector<int>{60000, 784});
        auto y = std::make_unique<Tensor>(std::vector<int>{60000, 10});

        // Load CSV
        load_csv(x.get(), y.get(), "mnist_train.csv");
        std::cout << "loaded csv\n";

        auto x_test = std::make_unique<Tensor>(std::vector<int>{10000, 784});
        auto y_test = std::make_unique<Tensor>(std::vector<int>{10000, 10});
        load_test_csv(x_test.get(), y_test.get(), "mnist_test.csv");
        
        // Transfer to GPU after loading
        x->data->cuda_to_cpu();  
        y->data->cuda_to_cpu();

        auto w1 = std::make_unique<Tensor>(std::vector<int>{784, 128});
        auto w2 = std::make_unique<Tensor>(std::vector<int>{128, 10});

        // Initialize weights with Kaiming uniform
        RandomGenerator rng(42);
        for (size_t i = 0; i < w1->data->size; i++) {
            w1->data->values[i] = rng.kaiming_uniform(784);
        }
        for (size_t i = 0; i < w2->data->size; i++) {
            w2->data->values[i] = rng.kaiming_uniform(128);
        }
        
        // Transfer weights to GPU
        w1->data->cpu_to_cuda();
        w2->data->cpu_to_cuda();
        w1->grad->cpu_to_cuda();
        w2->grad->cpu_to_cuda();

        // Training parameters
        constexpr int B = 128;
        constexpr float lr = 0.005f;
        constexpr int num_iterations = 5000;
        
        // Create batch tensors
        auto batch_x = std::make_unique<Tensor>(std::vector<int>{B, 784});
        auto batch_y = std::make_unique<Tensor>(std::vector<int>{B, 10});

        // Initial batch test
        get_random_batch(batch_x.get(), batch_y.get(), x.get(), y.get(), B);
        batch_x->data->cuda_to_cpu();
        batch_y->data->cuda_to_cpu();
        
        // Start timing
        Timer timer;
        timer.print_start();

        // Training loop
        for (int i = 0; i < num_iterations; i++) {
            // Get random batch
            get_random_batch(batch_x.get(), batch_y.get(), x.get(), y.get(), B);
            
            // Forward pass
            auto w1_out = matmul(batch_x.get(), w1.get());
            auto relu_out = relu(w1_out.get());
            auto w2_out = matmul(relu_out.get(), w2.get());
            auto lout = logsoftmax(w2_out.get());
            auto mul_out = mul(lout.get(), batch_y.get());
            auto loss = mean(mul_out.get());
            
            // Set loss gradient to 1
            float one = 1.0f;
            checkCudaError(cudaMemcpy(loss->grad->cuda_values.get(), &one, 
                                      sizeof(float), cudaMemcpyHostToDevice));
            
            if (i == 0) {
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::cerr << "CUDA error after loss grad set: " 
                              << cudaGetErrorString(err) << "\n";
                    return 1;
                }
            }
            
            // Zero out weight gradients
            checkCudaError(cudaMemset(w1->grad->cuda_values.get(), 0, 
                                      w1->grad->size * sizeof(float)));
            checkCudaError(cudaMemset(w2->grad->cuda_values.get(), 0, 
                                      w2->grad->size * sizeof(float)));
            
            // Backward pass
            backward(loss.get());

            // Print loss every 100 iterations
            if (i % 100 == 0) {
                loss->data->cuda_to_cpu();
                std::cout << "batch: " << i << " loss: " << loss->data->values[0] << "\n";
            }

            // Update weights
            update_weights(w1.get(), lr);
            update_weights(w2.get(), lr);

            checkCudaError(cudaDeviceSynchronize());
        }

        timer.print_elapsed();
        // std::cout << "Evaluating test accuracy...\n";
        // float acc = evaluate_accuracy(x_test.get(), y_test.get(), w1.get(), w2.get());
        // std::cout << "Test Accuracy: " << std::fixed << std::setprecision(4)
        //           << acc * 100.0f << "%\n";

        checkCudaError(cudaDeviceReset());
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        cudaDeviceReset();
        return 1;
    }
}
