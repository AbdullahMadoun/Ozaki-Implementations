# Ozaki-Implementations

Annotated, from-scratch CUDA implementations of the **Ozaki Scheme** for emulating FP64 matrix multiplication (`DGEMM`) on **INT8 Tensor Cores** (`CUBLAS_COMPUTE_32I`).

## Structure

```text
├── kernels/
│   └── ozIMMU.cu       # Ozaki Scheme I (ozIMMU & error-free INT32-batched ozIMMU_EF)
└── utils/
    └── allclose.cuh    # Modular torch.allclose-style accuracy checker (host & GPU)
```

### `kernels/`
- **[`kernels/ozIMMU.cu`](kernels/ozIMMU.cu)** — Annotated Ozaki Scheme I (`ozIMMU` & `ozIMMU_EF`):
  - **#1 ([`L10`](kernels/ozIMMU.cu#L10)):** `padding`, `axpby_kernel`, `axpby` — 32-element alignment and BLAS $\alpha/\beta$ epilogue.
  - **#2 ([`L67`](kernels/ozIMMU.cu#L67)):** `accum_kernel`, `accumulation` — Exponent reconstruction (`scalbn`) from `int32_t` products into FP64.
  - **#3 ([`L116`](kernels/ozIMMU.cu#L116)):** `gemm_8i_TN` — Panel-blocked `cublasGemmEx` (`CUBLAS_OP_T`, `CUBLAS_OP_N`, `CUBLAS_COMPUTE_32I`).
  - **#4A–#4D ([`L139`](kernels/ozIMMU.cu#L139)–[`L515`](kernels/ozIMMU.cu#L515)):** `conv_64f_2_8i` — Block-floating-point bitmask slicing (`rowwise` shared-memory transpose and `colwise` warp reduction).
  - **#5A–#5C ([`L517`](kernels/ozIMMU.cu#L517)–[`L643`](kernels/ozIMMU.cu#L643)):** `workSize`, `ozIMMU`, `ozIMMU_EF` — Workspace layout, diagonal-band GEMM loop, and error-free INT32 batching.

### `utils/`
- **[`utils/allclose.cuh`](utils/allclose.cuh)** — Modular `torch.allclose`-style accuracy checker (`|a - b| <= atol + rtol * |b|`) for 1D/2D column-major host and GPU matrices (`allclose`, `allclose_host`, `allclose_device`).

## Usage

Compile with C++20 and link `cublas` (`nvcc -std=c++20 -O3 -I. ... -lcublas -lcudart`):

```cpp
#include "kernels/ozIMMU.cu"
#include "utils/allclose.cuh"

const int numSlice = 8; // 3..18
void *d_work = nullptr;
cudaMalloc(&d_work, workSize(m, n, k, numSlice));

double alpha = 1.0, beta = 0.0;
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC_ref, m);
ozIMMU_EF(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m, d_work, numSlice);

auto res = allclose(m, n, dC, m, dC_ref, m, /*rtol=*/1e-12, /*atol=*/1e-14);
res.print("ozIMMU_EF vs cuBLAS");

cudaFree(d_work);
```

## References

- **Ozaki Scheme (2012):** K. Ozaki, T. Ogita, S. Oishi, & S. M. Rump, *"Error-free transformations of matrix multiplication by using fast routines of matrix multiplication and its applications,"* Numerical Algorithms, 59, 95–118.
- **ozIMMU (2024):** H. Ootomo, K. Ozaki, & R. Yokota, *"DGEMM on integer matrix multiplication unit,"* IJHPCA, 38(4), 297–313.
- **Accelerator for ozIMMU (2025):** Y. Uchino, K. Ozaki, & T. Imamura, *"Performance enhancement of the Ozaki Scheme on integer matrix multiplication unit,"* IJHPCA, 39(3), 462–476.
