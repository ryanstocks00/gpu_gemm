#include <cassert>
#include <cute/tensor.hpp>


template <typename thread_block_shape, class T, typename MMA>
__global__ void my_gemm_device(int m, int n, int k,
        T alpha,
        T const* A, int ldA,
        T const* B, int ldB,
        T beta,
        T      * C, int ldC, MMA mma)
{
  using namespace cute;

  auto block_coord = make_coord(blockIdx.x, blockIdx.y);

  Tensor A_global_tensor = make_tensor(make_gmem_ptr(A), make_shape(m, k), make_shape(1, ldA));
  Tensor B_global_tensor = make_tensor(make_gmem_ptr(B), make_shape(k, n), make_shape(1, ldB));
  Tensor C_global_tensor = make_tensor(make_gmem_ptr(C), make_shape(m, n), make_shape(1, ldC));

  Tensor thread_block_slice_A = local_tile(A_global_tensor, select<0, 2>(thread_block_shape{}), make_shape(blockIdx.x, _));
  Tensor thread_block_slice_B = local_tile(B_global_tensor, select<1, 2>(thread_block_shape{}), make_shape(blockIdx.y, _));
  Tensor thread_block_slice_C = local_tile(C_global_tensor, select<0, 1>(thread_block_shape{}), block_coord);

  auto shared_block_A_shape = select<0, 2>(thread_block_shape{});
  auto shared_block_B_shape = select<1, 2>(thread_block_shape{});
  __shared__ T shared_block_A_data[size(shared_block_A_shape)];
  __shared__ T shared_block_B_data[size(shared_block_B_shape)];
  Tensor shared_block_A = make_tensor(make_smem_ptr(shared_block_A_data), shared_block_A_shape);
  Tensor shared_block_B = make_tensor(make_smem_ptr(shared_block_B_data), shared_block_B_shape);

  TiledCopy tiled_copy = make_tiled_copy(Copy_Atom<UniversalCopy<T>, T>{},
                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 m-major

  ThrCopy thread_copy = tiled_copy.get_slice(threadIdx.x);
  Tensor thread_copy_subset_global_A = thread_copy.partition_S(thread_block_slice_A);
  Tensor thread_copy_subset_global_B = thread_copy.partition_S(thread_block_slice_B);

  Tensor thread_copy_subset_shared_block_A = thread_copy.partition_D(shared_block_A);
  Tensor thread_copy_subset_shared_block_B = thread_copy.partition_D(shared_block_B);


  ThrMMA thr_mma_slice = mma.get_slice(threadIdx.x);
  Tensor thread_mma_subset_A = thr_mma_slice.partition_A(shared_block_A);
  Tensor thread_mma_subset_B = thr_mma_slice.partition_B(shared_block_B);
  Tensor thread_mma_subset_C = thr_mma_slice.partition_C(thread_block_slice_C);

  Tensor A_reg = thr_mma_slice.make_fragment_A(thread_mma_subset_A);
  Tensor B_reg = thr_mma_slice.make_fragment_B(thread_mma_subset_B);
  Tensor C_reg = thr_mma_slice.make_fragment_C(thread_mma_subset_C);
  clear(C_reg);

//
/*  if (thread0()) {*/
      /*print(A_global_tensor);*/
      /*printf("\n");*/
      /*print(thread_block_slice_A);*/
      /*printf("\n");*/
      /*print(tiled_copy);*/
      /*printf("\n");*/
      /*print(thread_copy);*/
      /*printf("\n");*/
      /*print(thread_copy_subset_global_A);*/
      /*printf("\n");*/
      /*print(mma);*/
      /*printf("\n");*/
      /*print(thr_mma_slice);*/
      /*printf("\n");*/
      /*print(shared_block_A);*/
      /*printf("\n");*/
      /*print(thread_mma_subset_A);*/
      /*printf("\n");*/
      /*print(A_reg);*/
      /*printf("\n");*/
      /*print(B_reg);*/
      /*printf("\n");*/
      /*print(C_reg);*/
      /*printf("\n\n");*/
  /*}*/
  for (int k_tile = 0; k_tile < size<2>(thread_block_slice_A); k_tile++) {
    __syncthreads();
    copy(tiled_copy, thread_copy_subset_global_A(_, _, _, k_tile), thread_copy_subset_shared_block_A);
    copy(tiled_copy, thread_copy_subset_global_B(_, _, _, k_tile), thread_copy_subset_shared_block_B);
    __syncthreads();
   copy(thread_mma_subset_A, A_reg);
    copy(thread_mma_subset_B, B_reg);
    __syncthreads();
    gemm(mma, A_reg, B_reg, C_reg);
    __syncthreads();
  }
  axpby(alpha, C_reg, beta, thread_mma_subset_C);
}


// Setup params for a NT GEMM
template <class T>
void
my_gemm_nt(int m, int n, int k,
        T alpha,
        T const* A, int ldA,
        T const* B, int ldB,
        T beta,
        T      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;
   
  using thread_block_shape = Shape<_128, _128, _8>;
  //using thread_block_shape = Shape<_4, _2, _2>;

  //TiledMMA mma = make_tiled_mma(UniversalFMA<T>{},
                                ////Layout<Shape<_16, _16, _4>>{});
                                //Layout<Shape<_16, _16, _1>>{});
  TiledMMA mma = make_tiled_mma(SM80_8x8x4_F64F64F64F64_TN{},
                                 Layout<Shape<_2,_4,_1>>{});  // 16x16x1 TiledMMA

  static_assert(size(mma) == 256);
  dim3 dimBlock(size(mma));
  //dim3 dimBlock(2, 2, 1);
  dim3 dimGrid(size(ceil_div(m, select<0>(thread_block_shape{}))),
               size(ceil_div(n, select<1>(thread_block_shape{}))));
  //dim3 dimGrid(1, 1, 1);
  my_gemm_device<thread_block_shape><<<dimGrid, dimBlock, 0, stream>>>(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, mma);
}




template <typename T>
void
my_gemm(char transA, char transB, int m, int n, int k,
     T alpha,
     T const* A, int ldA,
     T const* B, int ldB,
     T beta,
     T      * C, int ldC,
     cudaStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T') {
    return my_gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  }
  assert(false && "Not implemented");
}

