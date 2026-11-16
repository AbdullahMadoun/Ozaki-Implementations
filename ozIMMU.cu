#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <type_traits>

// ============================================================================
// Challenge #1: axpby_kernel & axpby & padding
// ============================================================================

__forceinline__ size_t padding(const size_t n) {
  // 32* ((31 + n ) / 32) finds closest multiple of 32 that is larger than n
  return 32 * ((31 + n) / 32);
}

inline constexpr int numThreads = 256;

template <bool beta_zero>
__global__ void
axpby_kernel(const unsigned rows, const unsigned cols, const double alpha,
             const double *const __restrict__ C_64f, const size_t ld_c64f,
             const double beta, double *const __restrict__ C_out,
             const size_t ld_cout) {
  // [0, rows*cols] actual logical index in subarray
  const size_t idx = threadIdx.x + size_t(blockDim.x) * blockIdx.x;

  // Safety check
  if (idx < size_t(rows) * cols) {
    // col major ( cols move fast rows move slow )
    const size_t col = idx / rows;
    const size_t row = idx % rows;
    // converts it to jump through padded elements both C arrays have the
    // rows*cols portion stored at top most corner
    const size_t c_out_idx = col * ld_cout + row;
    const size_t c_64f_idx = col * ld_c64f + row;

    if constexpr (beta_zero) {
      // perform C_out = alpha * C_64f
      C_out[c_out_idx] = alpha * C_64f[c_64f_idx];
    } else {
      // fma (multiply add) alpha * C64f then added to beta * C_out
      C_out[c_out_idx] = fma(alpha, C_64f[c_64f_idx], C_out[c_out_idx] * beta);
    }
  }
}

inline void axpby(const unsigned rows, const unsigned cols, const double alpha,
                  const double *const C_64f, const size_t ld_c64f,
                  const double beta, double *const C_out,
                  const size_t ld_cout) {
  if (rows == 0 || cols == 0) {
    return;
  }
  const size_t grid = (numThreads - 1 + size_t(rows) * cols) / numThreads;
  if (beta == 0.0) {
    axpby_kernel<true><<<grid, numThreads>>>(rows, cols, alpha, C_64f, ld_c64f,
                                             beta, C_out, ld_cout);
  } else {
    axpby_kernel<false><<<grid, numThreads>>>(rows, cols, alpha, C_64f, ld_c64f,
                                              beta, C_out, ld_cout);
  }
}

// ============================================================================
// Challenge #2: accum_kernel & accumulation
// ============================================================================

template <bool init>
__global__ void accum_kernel(const unsigned rows, const unsigned cols,
                             const int32_t *const __restrict__ C_32i,
                             double *const __restrict__ C_64f, const size_t ldc,
                             const int16_t *const __restrict__ ufp_max_A,
                             const int16_t *const __restrict__ ufp_max_B,
                             const int sft) {
  const size_t idx = threadIdx.x + size_t(blockDim.x) * blockIdx.x;
  if (idx < size_t(rows) * cols) {
    // row  idx % rows  col idx / rows
    const unsigned int row = idx % rows;
    const unsigned int col = idx / rows;
    const size_t idx_C = row + col * ldc;
    // find the actual shift of the product
    const int sft_C = ufp_max_A[row] + ufp_max_B[col] - sft;
    // applies shift using scalbn
    const double C_32i_v = scalbn(__int2double_rn(C_32i[idx_C]), sft_C);

    if constexpr (init) {
      C_64f[idx_C] = C_32i_v;
    } else {
      C_64f[idx_C] += C_32i_v;
    }
  }
}

inline void accumulation(const int bits_per_slice, const unsigned rows,
                         const unsigned cols, const int32_t *const C_32i,
                         double *const C_64f, const size_t ldc,
                         const int16_t *const ufp_max_A, const int idx_A,
                         const int16_t *const ufp_max_B, const int idx_B) {
  if (rows == 0 || cols == 0) {
    return;
  }
  const size_t grid = (numThreads - 1 + size_t(rows) * cols) / numThreads;
  const int sft = bits_per_slice * (idx_A + idx_B + 2) - 2;
  if (idx_A + idx_B == 0) {
    accum_kernel<true><<<grid, numThreads>>>(rows, cols, C_32i, C_64f, ldc,
                                             ufp_max_A, ufp_max_B, sft);
  } else {
    accum_kernel<false><<<grid, numThreads>>>(rows, cols, C_32i, C_64f, ldc,
                                              ufp_max_A, ufp_max_B, sft);
  }
}

// ============================================================================
// Challenge #3: gemm_8i_TN
// ============================================================================

inline void gemm_8i_TN(cublasHandle_t handle, const int m, const int n,
                       const int k, const int32_t *const alpha_32i,
                       const int8_t *const A_8i, const size_t ld_a8i,
                       const int8_t *const B_8i, const size_t ld_b8i,
                       const int32_t *const beta_32i, int32_t *const C_32i,
                       const size_t ld_c32i) {
  const int blk = 8192;
  int rem = n;
  int offset = 0;
  while (rem > 0) {
    int nn = rem <= 12288 ? rem : blk;
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, nn, k, alpha_32i, A_8i,
                 CUDA_R_8I, ld_a8i, B_8i + offset * ld_b8i, CUDA_R_8I, ld_b8i,
                 beta_32i, C_32i + offset * ld_c32i, CUDA_R_32I, ld_c32i,
                 CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    rem -= nn;
    offset += nn;
  }
}
// ============================================================================
// Challenge #4A: UINT_TYPE & conv_64f_2_8i_bitmask
// ============================================================================

template <int numSlice>
using UINT_TYPE = std::conditional_t<
    (numSlice <= 4), uint32_t,
    std::conditional_t<(numSlice <= 9), uint64_t, __uint128_t>>;
template <int numSlice>
__device__ __forceinline__ void
conv_64f_2_8i_bitmask(const int bits_per_slice, const double in,
                      const int ufp_max, int8_t *const __restrict__ out,
                      const size_t inc_out) {
  using uint_t = UINT_TYPE<numSlice>;
  constexpr int bit_width = sizeof(uint_t) * 8;
  const int in_sft = bit_width - ufp_max - 1;
  const bool sgn = (in < 0);
  const uint_t v = static_cast<uint_t>(scalbn(fabs(in), in_sft));
  const uint_t mask = (uint_t(1) << bits_per_slice) - 1;
#pragma unroll
  for (int s = 0; s < numSlice; ++s) {
    const int sft = bit_width - bits_per_slice * (s + 1);
    int8_t slice_v = uint8_t((v >> sft) & mask);
    if (sgn) {
      slice_v *= -1;
    }
    out[s * inc_out] = slice_v;
  }
}

// ============================================================================
// Challenge #4B-1: find_amax_colwise
// ============================================================================

inline constexpr int TILE_DIM = 32;

__device__ __forceinline__ double
find_amax_colwise(const unsigned rows, const double *const __restrict__ in,
                  double *smem) {

  int laneId = threadIdx.x;
  double amax = 0.0;
  // Per thread coalsced at each step they all access consecutive parts of the
  // memory jumped by block size
  for (unsigned int i = laneId; i < rows; i += blockDim.x) {
    amax = fmax(amax, fabs(in[i]));
  }
// Intra-Warp
#pragma unroll
  for (int i = TILE_DIM / 2; i > 0; i >>= 1) {
    amax = fmax(amax, __shfl_down_sync(0xFFFFFFFFu, amax, i, TILE_DIM));
  }
  // Warp master writes its warp val into the first smem positions no conflicts
  if ((laneId & 31) == 0) {
    // laneId >> 5 is basically warpId
    smem[laneId >> 5] = amax;
  }
  __syncthreads();

  if (laneId < 32) {

    if (laneId < blockDim.x >> 5) {
      amax = smem[laneId];
    } else {
      amax = 0.0;
    }
    // Inter-Warp now every amax is within registers we share them on warp 0

#pragma unroll
    for (int i = TILE_DIM / 2; i > 0; i >>= 1) {
      amax = fmax(amax, __shfl_down_sync(0xFFFFFFFFu, amax, i, TILE_DIM));
    }
  }

  // Thread master writes max
  if (laneId == 0) {
    smem[0] = amax;
  }
  __syncthreads();

  return smem[0];
}

// ============================================================================
// Challenge #4C-1: find_amax_rowwise
// ============================================================================

__device__ __forceinline__ double
find_amax_rowwise(const unsigned row_idx, const unsigned rows,
                  const unsigned cols, const double *const __restrict__ in,
                  const size_t ld_in, double smem[][TILE_DIM + 1]) {
  double amax = 0.0;
  // coalsced reads to get partial maxes for each sub row (colwise)
  // We keep row fixed and we get partial of columns for that row
  if (row_idx < rows) {
    for (unsigned int c = threadIdx.y; c < cols; c += blockDim.y) {
      // each warp (y fixed, x=0..31)
      // reads same col c across 32 consecutive rows -> coalesced
      amax = fmax(fabs(in[row_idx + c * ld_in]), amax);
    }
  }
  // warps are organized based on x N*(0,..,31) the purpose of this is to
  // organize values in smem read such that values accessed on x get the same
  // row partials write col partial of that row. now all threads [y][x] write
  // the partial for cols y in all x
  smem[threadIdx.y][threadIdx.x] = amax;
  __syncthreads();
  // now read partials by going over consecutive cols in the same row the first
  // dim represents the cols for same row
  amax = smem[threadIdx.x][threadIdx.y];
// share valeus to reduce to final row level amax
#pragma unroll
  for (int i = TILE_DIM / 2; i > 0; i >>= 1) {
    amax = fmax(__shfl_down_sync(0xFFFFFFFFu, amax, i, TILE_DIM), amax);
  }
  // warp master is the only one that can get the full reduction
  // neat idea here is that the caller ( aware of threadid per return ) can
  // directly only use the warp master's return checking if the t_id = 0
  return amax;
}

// ============================================================================
// Challenge #4C-2: compute_ufp_max_rowwise_kernel
// ============================================================================
__global__ void
compute_ufp_max_rowwise_kernel(const unsigned rows, const unsigned cols,
                               const double *const __restrict__ in,
                               const size_t ld_in,
                               int16_t *const __restrict__ ufp_max) {
  // find true row
  unsigned int row = threadIdx.x + blockIdx.x * TILE_DIM;
  // allocate smem
  __shared__ double smem[TILE_DIM][TILE_DIM + 1];
  // per row amax called by all threads
  const double amax = find_amax_rowwise(row, rows, cols, in, ld_in, smem);
  // serialized row inside ufp_max; it is diagonal matrix writen as serial
  row = threadIdx.y + blockIdx.x * TILE_DIM;
  // if row is valid and warp master is the one writing the amax
  if (row < rows && threadIdx.x == 0) {
    if (amax == 0.0) {
      ufp_max[row] = 0;
    } else {
      ufp_max[row] = static_cast<int16_t>(ilogb(amax));
    }
  }
}

// ============================================================================
// Challenge #4C-3: conv_64f_2_8i_rowwise_kernel & conv_64f_2_8i_rowwise
// ============================================================================

template <int numSlice>
__global__ void conv_64f_2_8i_rowwise_kernel(
    const int bits_per_slice, const unsigned rows, const unsigned cols,
    const double *const __restrict__ in, const size_t ld_in,
    int8_t *const __restrict__ out, const size_t ld_out, const size_t inc_out,
    int16_t *const __restrict__ ufp_max) {
  __shared__ double smem[TILE_DIM][TILE_DIM + 1];
  // in is (rows, cols) in col-major
  const unsigned int rowBase = blockDim.x * blockIdx.x;
  const unsigned int colBase = blockDim.y * blockIdx.y;
  const unsigned int in_col = colBase + threadIdx.y;
  const unsigned int in_row = rowBase + threadIdx.x;
  if (in_row < rows && in_col < cols) {
    smem[threadIdx.y][threadIdx.x] = in[in_col * ld_in + in_row];
  } else {
    smem[threadIdx.y][threadIdx.x] = 0.0;
  }
  __syncthreads();
  // out is (ld_out, rows) in col-major (zero-padded transpose of in):
  // threadIdx.x walks out_row (0..ld_out-1 -> coalesced writes), threadIdx.y
  // walks out_col (0..rows-1)
  const unsigned int out_row = colBase + threadIdx.x;
  const unsigned int out_col = rowBase + threadIdx.y;
  if (out_col >= rows || out_row >= ld_out) {
    return;
  }
  // output index col major
  const size_t idx_out = out_col * ld_out + out_row;

  // Phase 1 wrote smem[y][x] un-transposed so reads from in coalesced;
  // reading smem[x][y] transposes on-chip so writes to transposed out also
  // coalesce
  conv_64f_2_8i_bitmask<numSlice>(bits_per_slice,
                                  smem[threadIdx.x][threadIdx.y],
                                  ufp_max[out_col], out + idx_out, inc_out);
}

template <int numSlice>
inline void conv_64f_2_8i_rowwise(const int bits_per_slice, const unsigned rows,
                                  const unsigned cols, const double *const in,
                                  const size_t ld_in, int8_t *const out,
                                  const size_t ld_out, const size_t inc_out,
                                  int16_t *const ufp_max) {
  const dim3 threads(TILE_DIM, TILE_DIM);
  const dim3 grid((TILE_DIM - 1 + rows) / TILE_DIM,
                  (TILE_DIM - 1 + ld_out) / TILE_DIM);
  compute_ufp_max_rowwise_kernel<<<grid.x, threads>>>(rows, cols, in, ld_in,
                                                      ufp_max);
  conv_64f_2_8i_rowwise_kernel<numSlice><<<grid, threads>>>(
      bits_per_slice, rows, cols, in, ld_in, out, ld_out, inc_out, ufp_max);
}

// ============================================================================
// Challenge #4B-2: conv_64f_2_8i_colwise_kernel & conv_64f_2_8i_colwise
// ============================================================================

template <int numSlice>
__global__ void
conv_64f_2_8i_colwise_kernel(const int bits_per_slice, const unsigned rows,
                             const double *const __restrict__ in,
                             const size_t ld_in, int8_t *const __restrict__ out,
                             const size_t ld_out, const size_t inc_out,
                             int16_t *const __restrict__ ufp_max) {
  __shared__ double smem[TILE_DIM];
  const unsigned int col_idx = blockIdx.x;
  find_amax_colwise(rows, in + col_idx * ld_in, smem);
  const double amax = smem[0];
  const int16_t ufp_max_j =
      static_cast<int16_t>((amax == 0.0) ? 0 : ilogb(amax));
  if (threadIdx.x == 0) {
    ufp_max[col_idx] = ufp_max_j;
  }
  unsigned int row_idx = threadIdx.x;
  for (; row_idx < rows; row_idx += blockDim.x) {
    const size_t idx_in = col_idx * ld_in + row_idx;
    const size_t idx_out = col_idx * ld_out + row_idx;
    conv_64f_2_8i_bitmask<numSlice>(bits_per_slice, in[idx_in], ufp_max_j,
                                    out + idx_out, inc_out);
  }
  for (; row_idx < ld_out; row_idx += blockDim.x) {
    const size_t idx_out = col_idx * ld_out + row_idx;
#pragma unroll
    for (int s = 0; s < numSlice; ++s) {
      out[s * inc_out + idx_out] = 0;
    }
  }
}

template <int numSlice>
inline void conv_64f_2_8i_colwise(const int bits_per_slice, const unsigned rows,
                                  const unsigned cols, const double *const in,
                                  const size_t ld_in, int8_t *const out,
                                  const size_t ld_out, const size_t inc_out,
                                  int16_t *const ufp_max) {
  const int numThreads = 256;
  conv_64f_2_8i_colwise_kernel<numSlice><<<cols, numThreads>>>(
      bits_per_slice, rows, in, ld_in, out, ld_out, inc_out, ufp_max);
}

// ============================================================================
// Challenge #4D: conv_64f_2_8i_launch & conv_64f_2_8i
// ============================================================================

template <int numSlice>
inline void conv_64f_2_8i_launch(
    const int bits_per_slice, const unsigned m, const unsigned n,
    const unsigned k, const cublasOperation_t transa, const double *const A_64f,
    const size_t ld_a64f, const cublasOperation_t transb,
    const double *const B_64f, const size_t ld_b64f, int8_t *const A_8i,
    const size_t ld_a8i, const size_t inc_a8i, int16_t *const ufp_max_A,
    int8_t *const B_8i, const size_t ld_b8i, const size_t inc_b8i,
    int16_t *const ufp_max_B) {
  if (transa == CUBLAS_OP_N) {
    // A_64f: m*k -> A_8i: k*m (transposed and stored)
    conv_64f_2_8i_rowwise<numSlice>(bits_per_slice, m, k, A_64f, ld_a64f, A_8i,
                                    ld_a8i, inc_a8i, ufp_max_A);
  } else {
    // A_64f: k*m -> A_8i: k*m (stored without transposing)
    conv_64f_2_8i_colwise<numSlice>(bits_per_slice, k, m, A_64f, ld_a64f, A_8i,
                                    ld_a8i, inc_a8i, ufp_max_A);
  }

  if (transb == CUBLAS_OP_N) {
    // B_64f: k*n -> B_8i: k*n (stored without transposing)
    conv_64f_2_8i_colwise<numSlice>(bits_per_slice, k, n, B_64f, ld_b64f, B_8i,
                                    ld_b8i, inc_b8i, ufp_max_B);
  } else {
    // B_64f: n*k -> B_8i: k*n (transposed and stored)
    conv_64f_2_8i_rowwise<numSlice>(bits_per_slice, n, k, B_64f, ld_b64f, B_8i,
                                    ld_b8i, inc_b8i, ufp_max_B);
  }
}

inline void conv_64f_2_8i(const int numSlice, const int bits_per_slice,
                          const unsigned m, const unsigned n, const unsigned k,
                          const cublasOperation_t transa,
                          const double *const A_64f, const size_t ld_a64f,
                          const cublasOperation_t transb,
                          const double *const B_64f, const size_t ld_b64f,
                          int8_t *const A_8i, const size_t ld_a8i,
                          const size_t inc_a8i, int16_t *const ufp_max_A,
                          int8_t *const B_8i, const size_t ld_b8i,
                          const size_t inc_b8i, int16_t *const ufp_max_B) {
  switch (numSlice) {
  case 3:
    conv_64f_2_8i_launch<3>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                            transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                            ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 4:
    conv_64f_2_8i_launch<4>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                            transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                            ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 5:
    conv_64f_2_8i_launch<5>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                            transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                            ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 6:
    conv_64f_2_8i_launch<6>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                            transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                            ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 7:
    conv_64f_2_8i_launch<7>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                            transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                            ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 8:
    conv_64f_2_8i_launch<8>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                            transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                            ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 9:
    conv_64f_2_8i_launch<9>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                            transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                            ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 10:
    conv_64f_2_8i_launch<10>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                             transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                             ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 11:
    conv_64f_2_8i_launch<11>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                             transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                             ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 12:
    conv_64f_2_8i_launch<12>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                             transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                             ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 13:
    conv_64f_2_8i_launch<13>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                             transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                             ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 14:
    conv_64f_2_8i_launch<14>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                             transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                             ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 15:
    conv_64f_2_8i_launch<15>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                             transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                             ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 16:
    conv_64f_2_8i_launch<16>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                             transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                             ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 17:
    conv_64f_2_8i_launch<17>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                             transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                             ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  case 18:
    conv_64f_2_8i_launch<18>(bits_per_slice, m, n, k, transa, A_64f, ld_a64f,
                             transb, B_64f, ld_b64f, A_8i, ld_a8i, inc_a8i,
                             ufp_max_A, B_8i, ld_b8i, inc_b8i, ufp_max_B);
    break;
  }
}

// ============================================================================
// Challenge #5A: workSize
// ============================================================================

inline size_t workSize(const int m, const int n, const int k,
                       const int numSlice) {
  const size_t ld_a8i = padding(k);
  const size_t ld_b8i = ld_a8i;
  const size_t ld_c32i = padding(m);
  const size_t cols_a8i = ld_c32i;
  const size_t cols_b8i = n;
  const size_t cols_c32i = n;
  const size_t size_A = ld_a8i * cols_a8i;
  const size_t size_B = ld_b8i * cols_b8i;
  const size_t size_C = ld_c32i * cols_c32i;

  return sizeof(int8_t) * size_A * numSlice +
         sizeof(int8_t) * size_B * numSlice + sizeof(int32_t) * size_C +
         sizeof(double) * size_C + sizeof(int16_t) * cols_a8i +
         sizeof(int16_t) * cols_b8i;
}

// ============================================================================
// Challenge #5B: ozIMMU
// ============================================================================

inline void ozIMMU(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const double *alpha, const double *A, int lda,
                   const double *B, int ldb, const double *beta, double *C,
                   int ldc, void *const work, const int numSlice) {
  const size_t ld_a8i = padding(k);
  const size_t ld_b8i = ld_a8i;
  const size_t ld_c32i = padding(m);
  const size_t cols_a8i = ld_c32i;
  const size_t cols_b8i = n;
  const size_t cols_c32i = n;
  const size_t size_A = ld_a8i * cols_a8i;
  const size_t size_B = ld_b8i * cols_b8i;
  const size_t size_C = ld_c32i * cols_c32i;

  int8_t *const A_8i = reinterpret_cast<int8_t *>(work);
  int8_t *const B_8i = A_8i + size_A * numSlice;
  int32_t *const C_32i = reinterpret_cast<int32_t *>(B_8i + size_B * numSlice);
  double *const C_64f = reinterpret_cast<double *>(C_32i + size_C);
  int16_t *const ufp_max_A = reinterpret_cast<int16_t *>(C_64f + size_C);
  int16_t *const ufp_max_B = ufp_max_A + cols_a8i;

  const int clz = __builtin_clz(uint32_t(k));
  const int ufp_k = 31 - clz;
  const int ceil_log2_k = ((1u << ufp_k) == k) ? ufp_k : (ufp_k + 1);
  const int bits_per_slice = std::min(7, (31 - ceil_log2_k) / 2);

  conv_64f_2_8i(numSlice, bits_per_slice, m, n, k, transa, A, lda, transb, B,
                ldb, A_8i, ld_a8i, size_A, ufp_max_A, B_8i, ld_b8i, size_B,
                ufp_max_B);

  constexpr int32_t one = 1, zero = 0;
  for (int bandIdx = 0; bandIdx < numSlice; ++bandIdx) {
    for (int idx_A = 0; idx_A <= bandIdx; ++idx_A) {
      int idx_B = bandIdx - idx_A;
      gemm_8i_TN(handle, ld_c32i, n, ld_a8i, &one, A_8i + idx_A * size_A,
                 ld_a8i, B_8i + idx_B * size_B, ld_b8i, &zero, C_32i, ld_c32i);
      accumulation(bits_per_slice, m, n, C_32i, C_64f, ld_c32i, ufp_max_A,
                   idx_A, ufp_max_B, idx_B);
    }
  }
  axpby(m, n, *alpha, C_64f, ld_c32i, *beta, C, ldc);
}

// ============================================================================
// Challenge #5C: ozIMMU_EF (Final Function!)
// ============================================================================

inline void ozIMMU_EF(cublasHandle_t handle, cublasOperation_t transa,
                      cublasOperation_t transb, int m, int n, int k,
                      const double *alpha, const double *A, int lda,
                      const double *B, int ldb, const double *beta, double *C,
                      int ldc, void *const work, const int numSlice) {
  const size_t ld_a8i = padding(k);
  const size_t ld_b8i = ld_a8i;
  const size_t ld_c32i = padding(m);
  const size_t cols_a8i = ld_c32i;
  const size_t cols_b8i = n;
  const size_t cols_c32i = n;
  const size_t size_A = ld_a8i * cols_a8i;
  const size_t size_B = ld_b8i * cols_b8i;
  const size_t size_C = ld_c32i * cols_c32i;

  int8_t *const A_8i = reinterpret_cast<int8_t *>(work);
  int8_t *const B_8i = A_8i + size_A * numSlice;
  int32_t *const C_32i = reinterpret_cast<int32_t *>(B_8i + size_B * numSlice);
  double *const C_64f = reinterpret_cast<double *>(C_32i + size_C);
  int16_t *const ufp_max_A = reinterpret_cast<int16_t *>(C_64f + size_C);
  int16_t *const ufp_max_B = ufp_max_A + cols_a8i;

  const int clz = __builtin_clz(uint32_t(k));
  const int ufp_k = 31 - clz;
  const int ceil_log2_k = ((1u << ufp_k) == k) ? ufp_k : (ufp_k + 1);
  const int bits_per_slice = std::min(7, (31 - ceil_log2_k) / 2);

  conv_64f_2_8i(numSlice, bits_per_slice, m, n, k, transa, A, lda, transb, B,
                ldb, A_8i, ld_a8i, size_A, ufp_max_A, B_8i, ld_b8i, size_B,
                ufp_max_B);

  const int numAccum32i = 1
                          << std::max(0, 31 - 2 * bits_per_slice - ceil_log2_k);
  constexpr int32_t one = 1;
  for (int bandIdx = 0; bandIdx < numSlice; ++bandIdx) {
    int32_t numAccum = 0;
    int32_t beta_i = 0;
    for (int idx_A = 0; idx_A <= bandIdx; ++idx_A) {
      int idx_B = bandIdx - idx_A;
      gemm_8i_TN(handle, ld_c32i, n, ld_a8i, &one, A_8i + idx_A * size_A,
                 ld_a8i, B_8i + idx_B * size_B, ld_b8i, &beta_i, C_32i,
                 ld_c32i);
      ++numAccum;
      beta_i = 1;
      if (numAccum >= numAccum32i || idx_A == bandIdx) {
        accumulation(bits_per_slice, m, n, C_32i, C_64f, ld_c32i, ufp_max_A,
                     idx_A, ufp_max_B, idx_B);
        beta_i = 0;
        numAccum = 0;
      }
    }
  }
  axpby(m, n, *alpha, C_64f, ld_c32i, *beta, C, ldc);
}
