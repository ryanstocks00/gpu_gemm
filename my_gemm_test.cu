#include <iomanip>
#include <iostream>
#include <curand.h>
#include <cublas_v2.h>
#include <chrono>

#include "cute_impl.cuh"
#include "cutlass/layout/matrix.h"
#include "my_impl.cuh"

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

#include "cutlass/gemm/device/gemm.h"

void print_matrix(const double* A, int m, int n) {
    for (int i = 0; i < m; i++) {
        printf("[ ");
        for (int j = 0; j < n; j++)
            printf("%f ", A[i + j * m]);
        printf("]\n");
    }
    printf("\n");
}

void cutlass_gemm(char transA, char transB, int m, int n, int k,
     double alpha,
     double const* A, int ldA,
     double const* B, int ldB,
     double beta,
     double      * C, int ldC,
     cudaStream_t stream = 0)
{
    using namespace cutlass;
    using ThreadBlockShape = cutlass::gemm::GemmShape<32, 32, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;
    using Gemm = cutlass::gemm::device::Gemm<
        double, cutlass::layout::ColumnMajor, double, cutlass::layout::RowMajor,
        double, cutlass::layout::ColumnMajor, double,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadBlockShape, WarpShape>;
    //using Gemm = cutlass::gemm::device::Gemm<
        //double, cutlass::layout::ColumnMajor, double, cutlass::layout::RowMajor,
        //double, cutlass::layout::ColumnMajor, double,
        //cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80>;
    Gemm gemm_op;
    Gemm::Arguments args({m, n, k}, {A, ldA}, {B, ldB}, {C, ldC}, {C, ldC}, {alpha, beta});
    gemm_op(args);
}



template <typename T>
void test_gemm(const std::string& name, std::function<void()> f, int m, int n, int k, int reps, T* C, T* C_ref) {
    std::cout << "Testing " << name << " GEMM ... " << std::endl;
    double n_flops = 2.0 * m * n * k;
    cudaMemset(C, 0, m * n * sizeof(T));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << __FILE__ << "' at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    f();
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << __FILE__ << "' at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    Timer timer;
    for (int i = 0; i < reps; i++) {
        f();
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << __FILE__ << "' at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    double per_gemm = timer.get() / reps;
    std::cout << "Time per " + name + " GEMM: " << per_gemm
              << " s, GFLOPS: " << n_flops / per_gemm / 1e9 << std::endl;

    // Check result
    std::vector<T> C_host(m * n, 0);
    std::vector<T> C_ref_host(m * n, 0);
    cudaMemcpy(C_host.data(), C, m * n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_ref_host.data(), C_ref, m * n * sizeof(T), cudaMemcpyDeviceToHost);
    double max_diff = 0;
    double max_val = 0;
    bool diff = false;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
          //printf("%f ", C_host[i + j * m]);
          if (std::abs(C_host[i + j * m]-C_ref_host[i + j * m]) > 100) {
              diff = true;
              printf("\n");
              printf("diff: %d %d %f \n", i, j, C_host[i + j * m] - C_ref_host[i + j * m]);
          }
      }
      if (diff)
          exit(1);
      //printf("\n");
    }
    //printf("\n");

    for (int i = 0; i < m * n; i++) {
      max_diff = std::max(max_diff, std::abs(C_host[i] - C_ref_host[i]));
      max_val = std::max(max_val, std::abs(C_ref_host[i]));
    }
    std::cout << "Max diff: " << std::setprecision(15) << max_diff << ", Max val: " << max_val << std::endl;
    std::cout << std::endl;
}

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
  std::cout << std::endl;

  test_gemm("CuTE", [&]() {
    gemm('N', 'T', m, n, k, alpha, A, m, B, k, beta, C, m);
  }, m, n, k, reps, C, C_ref);

  test_gemm("Naive", [&]() {
    naive_gemm('N', 'T', m, n, k, alpha, A, m, B, k, beta, C, m);
  }, m, n, k, reps, C, C_ref);

  test_gemm("My cute", [&]() {
    my_gemm('N', 'T', m, n, k, alpha, A, m, B, k, beta, C, m);
  }, m, n, k, reps, C, C_ref);

  test_gemm("Cutlass", [&]() {
    cutlass_gemm('N', 'T', m, n, k, alpha, A, m, B, k, beta, C, m);
  }, m, n, k, reps, C, C_ref);
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
  if (argc >= 5)
    reps = atoi(argv[4]);

  std::cout << "m = " << m << ", n = " << n << ", k = " << k << ", reps = " << reps << std::endl;

  using T = double;
  run_gemms<T>(m, n, k, reps);
}
