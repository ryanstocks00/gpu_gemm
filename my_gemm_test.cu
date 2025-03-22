#include <iomanip>
#include <iostream>
#include <curand.h>
#include <cublas_v2.h>
#include <chrono>

#include "cute_impl.cuh"

#define CUBLAS_CHECK(call) { \
    cublasStatus_t err; \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error in file '" << __FILE__ \
                  << "' at line " << __LINE__ << ": " \
                  << err << std::endl; \
        exit(1); \
    } \
}

class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    void reset() {
        start = std::chrono::high_resolution_clock::now();
    }
    double get() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
};

template <typename T>
void run_gemms(int m, int n, int k, int reps) {
  T *A, *B, *C, *C_ref;
  cudaMalloc(&A, m * k * sizeof(T));
  cudaMalloc(&B, k * n * sizeof(T));
  cudaMalloc(&C, m * n * sizeof(T));
  cudaMalloc(&C_ref, m * n * sizeof(T));

  double n_flops = 2.0 * m * n * k;

  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 42);
  if constexpr (std::is_same_v<T, float>) {
    static_assert(std::is_same_v<T, float>);
    curandGenerateUniform(gen, A, m * k);
    curandGenerateUniform(gen, B, k * n);
  } else if constexpr (std::is_same_v<T, double>) {
    curandGenerateUniformDouble(gen, A, m * k);
    curandGenerateUniformDouble(gen, B, k * n);
  }

  cudaMemset(C, 0, m * n * sizeof(T));

  T alpha = 1.0;
  T beta = 0.0;
  {
  cublasHandle_t handle;
  cublasCreate(&handle);
  CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, m, B, n, &beta, C_ref, m));
  
  cudaDeviceSynchronize();
  Timer timer;
  for (int i = 0; i < reps; i++) {
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, m, B, n, &beta, C, m));
  }
  cudaDeviceSynchronize();
  double per_gemm = timer.get() / reps;
  std::cout << "Time per CUBLAS GEMM: " << per_gemm << " s, GFLOPS: " << n_flops / per_gemm / 1e9 << std::endl;
  }

  {
    cudaMemset(C, 0, m * n * sizeof(T));
    gemm('N', 'T', m, n, k, alpha, A, m, B, k, beta, C, m);
    cudaDeviceSynchronize();
    Timer timer;
    for (int i = 0; i < reps; i++) {
        gemm('N', 'T', m, n, k, alpha, A, m, B, k, beta, C, m);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << __FILE__ << "' at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    double per_gemm = timer.get() / reps;
    std::cout << "Time per CuTE GEMM: " << per_gemm
              << " s, GFLOPS: " << n_flops / per_gemm / 1e9 << std::endl;

    // Check result
    std::vector<T> C_host(m * n, 0);
    std::vector<T> C_ref_host(m * n, 0);
    cudaMemcpy(C_host.data(), C, m * n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_ref_host.data(), C_ref, m * n * sizeof(T), cudaMemcpyDeviceToHost);
    double max_diff = 0;
    double max_val = 0;
    for (int i = 0; i < m * n; i++) {
      max_diff = std::max(max_diff, std::abs(C_host[i] - C_ref_host[i]));
      max_val = std::max(max_val, std::abs(C_ref_host[i]));
    }
    std::cout << "Max diff: " << std::setprecision(15) << max_diff << ", Max val: " << max_val << std::endl;
  }

  {
    cudaMemset(C, 0, m * n * sizeof(T));
    my_gemm('N', 'T', m, n, k, alpha, A, m, B, k, beta, C, m);
    cudaDeviceSynchronize();
    Timer timer;
    for (int i = 0; i < reps; i++) {
        my_gemm('N', 'T', m, n, k, alpha, A, m, B, k, beta, C, m);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << __FILE__ << "' at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    double per_gemm = timer.get() / reps;
    std::cout << "Time per MyCuTE GEMM: " << per_gemm
              << " s, GFLOPS: " << n_flops / per_gemm / 1e9 << std::endl;

    // Check result
    std::vector<T> C_host(m * n, 0);
    std::vector<T> C_ref_host(m * n, 0);
    cudaMemcpy(C_host.data(), C, m * n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_ref_host.data(), C_ref, m * n * sizeof(T), cudaMemcpyDeviceToHost);
    double max_diff = 0;
    double max_val = 0;
    for (int i = 0; i < m * n; i++) {
      max_diff = std::max(max_diff, std::abs(C_host[i] - C_ref_host[i]));
      max_val = std::max(max_val, std::abs(C_ref_host[i]));
    }
    std::cout << "Max diff: " << std::setprecision(15) << max_diff << ", Max val: " << max_val << std::endl;
  }
}

int main(int argc, char *argv[])
{
  int m = 512;
  int n = 512;
  int k = 512;
  int reps = 10;

  if (argc >= 2)
    m = atoi(argv[1]);
  if (argc >= 3)
    n = atoi(argv[2]);
  if (argc >= 4)
    k = atoi(argv[3]);

  std::cout << "m = " << m << ", n = " << n << ", k = " << k << ", reps = " << reps << std::endl;

  using T = double;
  run_gemms<T>(m, n, k, reps);
}
