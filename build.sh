 nvcc -std=c++17 -o train tensor.cu main.cpp -O3 --expt-relaxed-constexpr
