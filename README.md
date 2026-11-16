# Ozaki-Implementations

Annotated, from-scratch CUDA implementations of the **Ozaki Scheme** for emulating IEEE-754 FP64 matrix multiplication (`DGEMM`: $C \leftarrow \alpha\,\mathrm{op}(A)\,\mathrm{op}(B) + \beta C$) using **INT8 Tensor Cores** (`CUBLAS_COMPUTE_32I`).

> This repository documents hand-written kernels and personal inline notes studying how block-floating-point bit-slicing, coalesced shared-memory transposes, Tensor Core INT8 panel blocking, and error-free INT32 batch accumulation work together end-to-end.

---

## Architecture & Pipeline (`ozIMMU.cu`)

```text
Inputs: A (FP64), B (FP64), C (FP64) in GPU VRAM   [column-major cuBLAS layout]
   │
   ▼  (1) Block-Floating-Point Slicing  [conv_64f_2_8i]
      ├── Rows of op(A) ──> A_8i[0 .. numSlice-1]  (k_pad x m_pad, INT8) + ufp_max_A[m_pad]
      │     └─ OP_N uses rowwise coalesced load + smem[32][33] transpose + warp shuffle
      └── Cols of op(B) ──> B_8i[0 .. numSlice-1]  (k_pad x n,     INT8) + ufp_max_B[n]
            └─ OP_N uses single-pass 256-thread block column reduction + zero-padding
   │
   ▼  (2) Diagonal-Band INT8 Tensor Core Loop  [bandIdx = 0 .. numSlice-1]
      for idx_A = 0 .. bandIdx,  idx_B = bandIdx - idx_A:
        ├── gemm_8i_TN : C_32i = A_8i[idx_A]^T * B_8i[idx_B] + beta_i * C_32i
        │     └─ 8192-column panel blocking (<= 12288 lookahead) over cublasGemmEx
        └── ozIMMU_EF  : keeps beta_i = 1 in C_32i up to numAccum32i before flushing!
   │
   ▼  (3) Exponent Reconstruction  [accumulation -> C_64f]
      C_64f[r,c] (+)= scalbn((double)C_32i[r,c], ufp_max_A[r] + ufp_max_B[c] - sft)
        └─ idx_A + idx_B == 0 initializes C_64f directly (no memset kernel needed)
   │
   ▼  (4) BLAS Epilogue  [axpby -> C]
      C = alpha * C_64f + beta * C   (compile-time beta_zero guard avoids 0 * NaN)
```

---

## Mathematical Formulation

1. **Overflow-Free Slice Width (`bits_per_slice`):**
   Each `int8_t` slice element holds signed magnitude $< 2^b$ ($b \le 7$). A length-$k$ INT8 dot product summed into signed `int32_t` ($2^{31}-1$ max) satisfies:
   $$k \cdot (2^b - 1)^2 < 2^{\lceil \log_2 k \rceil + 2b} \le 2^{31} \implies b = \min\!\left(7,\; \left\lfloor \frac{31 - \lceil \log_2 k \rceil}{2} \right\rfloor \right)$$

2. **Non-Iterative Bitmask Slicing (`conv_64f_2_8i_bitmask`):**
   Instead of iteratively subtracting residuals per slice, each vector's maximum exponent $e_{\max} = \lfloor \log_2 (\max_t |x_t|) \rfloor$ (`ufp_max`) aligns every element in that row/column to a fixed binary point inside `UINT_TYPE<numSlice>` (`uint32_t` for $\le 4$ slices, `uint64_t` for $\le 9$, `__uint128_t` for $\le 18$):
   $$\text{in\_sft} = W - e_{\max} - 1, \qquad v = \left\lfloor |x| \cdot 2^{\text{in\_sft}} \right\rfloor \in [0, 2^W)$$
   Slice $s \in \{0, \dots, \text{numSlice}-1\}$ extracts bits $[W - b(s+1),\; W - bs - 1]$ directly via `(v >> (W - b*(s+1))) & ((1 << b) - 1)` and reapplies the sign of $x$.

3. **Why `gemm_8i_TN` Always Uses `(CUBLAS_OP_T, CUBLAS_OP_N)` & `padding(32)`:**
   Both `A_8i` ($k_{\text{pad}} \times m_{\text{pad}}$) and `B_8i` ($k_{\text{pad}} \times n$) store the reduction dimension $k$ padded to a multiple of `32` (`padding`), with zero-filled `[k, k_pad)` padding so `cublasGemmEx(CUBLAS_COMPUTE_32I)` executes safely on INT8 Tensor Cores regardless of user `transa`/`transb`.

4. **Exponent Reconstruction & Diagonal Truncation (`accumulation`):**
   Slice pairs with $i_A + i_B \ge \text{numSlice}$ contribute below the precision floor of `numSlice` slices and are skipped, halving the GEMMs from $S^2$ to $\frac{S(S+1)}{2}$. The `bit_width` $W$ cancels identically across $A$ and $B$:
   $$s_{A,i_A} + s_{B,i_B} \implies \text{sft\_C}[r, c] = \text{ufp\_max}_A[r] + \text{ufp\_max}_B[c] - \big[b\,(i_A + i_B + 2) - 2\big]$$

5. **Error-Free INT32 Batching (`ozIMMU_EF`):**
   Pairs on the same diagonal (`idx_A + idx_B == bandIdx`) share the same bit weight. Whenever `k` and `b` leave spare bits below `31`, up to
   $$\text{numAccum32i} = 2^{\max(0,\; 31 - 2b - \lceil \log_2 k \rceil)}$$
   diagonal GEMMs are accumulated directly inside `C_32i` via `beta_i = 1` before a single FP64 `accumulation` flush.

---

## Annotated Code Map (`ozIMMU.cu`)

| Section | Symbols (`ozIMMU.cu`) | Key Implementation Details & Notes |
|---|---|---|
| **#1** ([`L10`](ozIMMU.cu#L10)) | `padding`, `axpby_kernel<beta_zero>`, `axpby` | `32`-byte ceiling padding; compile-time `if constexpr (beta_zero)` avoids reading uninitialized `C` (`0.0 * NaN = NaN` safety), plus 64-bit `idx % rows` / `idx / rows`. |
| **#2** ([`L67`](ozIMMU.cu#L67)) | `accum_kernel<init>`, `accumulation` | Scales `int32_t` products via `__int2double_rn` + `scalbn(..., sft_C)`. Initializes `C_64f` directly when `idx_A + idx_B == 0` (`init = true`), avoiding an extra zeroing kernel; 64-bit `size_t` indexing (`idx % rows`, `idx / rows`) avoids 32-bit `col * rows` wraparound. |
| **#3** ([`L116`](ozIMMU.cu#L116)) | `gemm_8i_TN` | `cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_N, ..., CUBLAS_COMPUTE_32I)` with `blk = 8192` column-panel blocking and `<= 12288` lookahead to prevent tail splits and large-`n` Tensor Core slowdowns. |
| **#4A** ([`L139`](ozIMMU.cu#L139)) | `UINT_TYPE<numSlice>`, `conv_64f_2_8i_bitmask` | `std::conditional_t` integer selection (`32`/`64`/`128`-bit) and `#pragma unroll` MSB-to-LSB static bitmask extraction. |
| **#4B** ([`L169`](ozIMMU.cu#L169), [`L342`](ozIMMU.cu#L342)) | `find_amax_colwise`, `conv_64f_2_8i_colwise_kernel`, `conv_64f_2_8i_colwise` | 256-thread single-pass column kernel: intra-warp `__shfl_down_sync` + warp-leader shared-memory pass + broadcast via `smem[0]`, followed by active-row slicing and `[rows, ld_out)` zero-padding. |
| **#4C** ([`L222`](ozIMMU.cu#L222)–[`L339`](ozIMMU.cu#L339)) | `find_amax_rowwise`, `compute_ufp_max_rowwise_kernel`, `conv_64f_2_8i_rowwise_kernel`, `conv_64f_2_8i_rowwise` | `32 × 32` 2D blocks coalesce VRAM reads down columns, transpose on-chip via bank-conflict-padded `__shared__ double smem[TILE_DIM][TILE_DIM + 1]`, extract `ilogb(amax)` (guarding `0.0 -> 0`), and write transposed `A_8i` with `[cols, ld_out)` zero-padding. |
| **#4D** ([`L389`](ozIMMU.cu#L389)) | `conv_64f_2_8i_launch`, `conv_64f_2_8i` | Routes `(transa, transb)` to rowwise vs. colwise converters and lifts runtime `numSlice` (`3..18`) into compile-time `<S>` templates. |
| **#5A–#5C** ([`L517`](ozIMMU.cu#L517)–[`L643`](ozIMMU.cu#L643)) | `workSize`, `ozIMMU`, `ozIMMU_EF` | Sequential aligned workspace carving from a single GPU buffer (`A_8i` → `B_8i` → `C_32i` → `C_64f` → `ufp_max_A` → `ufp_max_B`), `bits_per_slice` via `__builtin_clz`, diagonal GEMM loop, and INT32 error-free batching (`ozIMMU_EF`). |

---

## Accuracy Verification (`allclose.cuh`)

[`allclose.cuh`](allclose.cuh) provides a modular, PyTorch-style `torch.allclose` checker (`|actual - expected| <= atol + rtol * |expected|`) for any floating-point type across 2D column-major matrices (`m, n, ld`) and 1D contiguous arrays:

- **Automatic host / device detection:** `allclose(...)` inspects `cudaPointerGetAttributes` and copies GPU buffers (with 2D strided `cudaMemcpy2D` when `ld > m`) transparently when either operand lives in device/managed memory; explicit `allclose_host` and `allclose_device` entry points are also available.
- **Diagnostics (`AllCloseResult`):** Evaluates to `bool` directly and prints pass/fail status, mismatch count/percentage, `max_abs_err`, `max_rel_err`, overflow-safe `frobenius_rel_err`, and the worst `(row, col)` entry.

---

## Minimal Usage Example

Include `ozIMMU.cu` and `allclose.cuh`, compile with C++20, and link `cublas` (`nvcc -std=c++20 -O3 ... -lcublas -lcudart`):

```cpp
#include "ozIMMU.cu"
#include "allclose.cuh"

// Given initialized cublasHandle_t handle and device matrices dA, dB, dC, dC_ref
const int numSlice = 8; // 3..18 (~8 slices for FP64-level accuracy)
void *d_work = nullptr;
cudaMalloc(&d_work, workSize(m, n, k, numSlice));

double alpha = 1.0, beta = 0.0;
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC_ref, m);
ozIMMU_EF(handle, CUBLAS_OP_N, CUBLAS_OP_N,
          m, n, k, &alpha, dA, m, dB, k, &beta, dC, m,
          d_work, numSlice);

auto res = allclose(m, n, dC, m, dC_ref, m, /*rtol=*/1e-12, /*atol=*/1e-14);
res.print("ozIMMU_EF vs cuBLAS");

cudaFree(d_work);
```

---

## References

- **Ozaki Scheme (2012):** K. Ozaki, T. Ogita, S. Oishi, & S. M. Rump, *"Error-free transformations of matrix multiplication by using fast routines of matrix multiplication and its applications,"* Numerical Algorithms, 59, 95–118.
- **ozIMMU (2024):** H. Ootomo, K. Ozaki, & R. Yokota, *"DGEMM on integer matrix multiplication unit,"* IJHPCA, 38(4), 297–313.
- **Accelerator for ozIMMU / ozIMMU_simple (2025):** Y. Uchino, K. Ozaki, & T. Imamura, *"Performance enhancement of the Ozaki Scheme on integer matrix multiplication unit,"* IJHPCA, 39(3), 462–476.
