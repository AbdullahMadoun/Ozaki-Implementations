#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__CUDACC__) || defined(__CUDA__) || __has_include(<cuda_runtime.h>)
#define OZAKI_ALLCLOSE_HAS_CUDA 1
#include <cuda_runtime.h>
#else
#define OZAKI_ALLCLOSE_HAS_CUDA 0
#endif

// ============================================================================
// Modular PyTorch-style `torch.allclose` Accuracy Checker
// ----------------------------------------------------------------------------
// Checks: |actual - expected| <= atol + rtol * |expected|
// Supports:
//   - Any floating-point type (double, float, long double, __half, __nv_bfloat16)
//   - 2D column-major matrices (both tight and strided ld >= m) and 1D arrays
//   - Automatic host / GPU device / mixed pointer handling via CUDA runtime
//   - Rich error diagnostics (mismatch count, max abs/rel, Frobenius, worst idx)
// ============================================================================

struct AllCloseResult {
  bool passed = true;
  size_t total_elements = 0;
  size_t mismatched_elements = 0;
  double rtol = 1e-5;
  double atol = 1e-8;
  double max_abs_err = 0.0;
  double max_rel_err = 0.0;
  double frobenius_rel_err = 0.0;
  size_t worst_row = 0;
  size_t worst_col = 0;
  double worst_actual = 0.0;
  double worst_expected = 0.0;

  constexpr explicit operator bool() const noexcept { return passed; }

  inline void print(const char *label = "allclose", FILE *out = stdout) const {
    const double fail_pct =
        (total_elements > 0)
            ? (100.0 * static_cast<double>(mismatched_elements) /
               static_cast<double>(total_elements))
            : 0.0;
    std::fprintf(
        out,
        "[%s] %s | mismatched: %zu / %zu (%.4f%%) | rtol=%.3e, atol=%.3e\n"
        "  max_abs_err=%.6e | max_rel_err=%.6e | frobenius_rel_err=%.6e\n",
        label, passed ? "PASS" : "FAIL", mismatched_elements, total_elements,
        fail_pct, rtol, atol, max_abs_err, max_rel_err, frobenius_rel_err);
    if (total_elements > 0) {
      std::fprintf(out,
                   "  worst @ (%zu, %zu): actual=%.17g, expected=%.17g\n",
                   worst_row, worst_col, worst_actual, worst_expected);
    }
  }
};

namespace detail {

template <typename T> inline double to_double(const T &val) noexcept {
  if constexpr (std::is_arithmetic_v<T>) {
    return static_cast<double>(val);
  } else {
    return static_cast<double>(static_cast<float>(val));
  }
}

struct ScaledSumSq {
  double scale = 0.0;
  double ssq = 1.0;
  bool has_inf = false;
  bool has_nan = false;

  inline void add(double x) noexcept {
    if (std::isnan(x)) {
      has_nan = true;
      return;
    }
    const double ax = std::abs(x);
    if (std::isinf(ax)) {
      has_inf = true;
      return;
    }
    if (ax == 0.0) {
      return;
    }
    if (scale < ax) {
      const double r = scale / ax;
      ssq = 1.0 + ssq * r * r;
      scale = ax;
    } else {
      const double r = ax / scale;
      ssq += r * r;
    }
  }

  inline double norm() const noexcept {
    if (has_nan) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    if (has_inf) {
      return std::numeric_limits<double>::infinity();
    }
    return (scale == 0.0) ? 0.0 : scale * std::sqrt(ssq);
  }
};

inline void validate_args(size_t m, size_t n, const void *actual,
                          size_t ld_actual, const void *expected,
                          size_t ld_expected, double rtol, double atol) {
  if (std::isnan(rtol) || rtol < 0.0 || std::isnan(atol) || atol < 0.0) {
    throw std::invalid_argument("allclose: rtol and atol must be non-negative");
  }
  if (m > 0 && n > 0) {
    if (actual == nullptr || expected == nullptr) {
      throw std::invalid_argument("allclose: input pointer must not be null");
    }
    if (ld_actual < m || ld_expected < m) {
      throw std::invalid_argument(
          "allclose: leading dimension must be >= row count m");
    }
  }
}

#if OZAKI_ALLCLOSE_HAS_CUDA

inline void check_cuda(cudaError_t err, const char *ctx) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(ctx) + ": " + cudaGetErrorString(err));
  }
}

inline bool is_device_pointer(const void *ptr) {
  if (ptr == nullptr) {
    return false;
  }
  cudaPointerAttributes attr{};
  const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  if (err == cudaErrorInvalidValue) {
    (void)cudaGetLastError();
    return false;
  }
  check_cuda(err, "cudaPointerGetAttributes");
  return attr.type == cudaMemoryTypeDevice ||
         attr.type == cudaMemoryTypeManaged;
}

template <typename T>
inline std::vector<T> copy_col_major_from_device(size_t m, size_t n,
                                                  const T *d_ptr, size_t ld) {
  std::vector<T> host(m * n);
  if (m == 0 || n == 0) {
    return host;
  }
  if (ld == m) {
    check_cuda(cudaMemcpy(host.data(), d_ptr, m * n * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy");
  } else {
    check_cuda(cudaMemcpy2D(host.data(), m * sizeof(T), d_ptr, ld * sizeof(T),
                            m * sizeof(T), n, cudaMemcpyDeviceToHost),
               "cudaMemcpy2D");
  }
  return host;
}

#endif // OZAKI_ALLCLOSE_HAS_CUDA

} // namespace detail

template <typename T>
inline bool isclose_elem(const T actual, const T expected,
                         const double rtol = 1e-5, const double atol = 1e-8,
                         const bool equal_nan = false) noexcept {
  const double a = detail::to_double(actual);
  const double e = detail::to_double(expected);
  if (std::isnan(a) || std::isnan(e)) {
    return equal_nan && std::isnan(a) && std::isnan(e);
  }
  if (std::isinf(a) || std::isinf(e)) {
    return a == e;
  }
  return std::abs(a - e) <= atol + rtol * std::abs(e);
}

// 2D column-major host matrix comparison
template <typename T>
inline AllCloseResult
allclose_host(const size_t m, const size_t n, const T *const actual,
              const size_t ld_actual, const T *const expected,
              const size_t ld_expected, const double rtol = 1e-5,
              const double atol = 1e-8, const bool equal_nan = false) {
  detail::validate_args(m, n, actual, ld_actual, expected, ld_expected, rtol,
                        atol);

  AllCloseResult res{};
  res.rtol = rtol;
  res.atol = atol;
  res.total_elements = m * n;
  if (res.total_elements == 0) {
    res.passed = true;
    return res;
  }

  detail::ScaledSumSq diff_sq;
  detail::ScaledSumSq ref_sq;
  int worst_tier = -1; // 0: finite, 1: inf mismatch, 2: nan mismatch
  double worst_ratio = -1.0;
  double worst_abs_tie = -1.0;

  for (size_t col = 0; col < n; ++col) {
    const T *const col_a = actual + col * ld_actual;
    const T *const col_e = expected + col * ld_expected;
    for (size_t row = 0; row < m; ++row) {
      const double a = detail::to_double(col_a[row]);
      const double e = detail::to_double(col_e[row]);
      const bool ok = isclose_elem(a, e, rtol, atol, equal_nan);
      if (!ok) {
        ++res.mismatched_elements;
      }

      int tier = 0;
      double abs_err = 0.0;
      double rel_err = 0.0;
      double ratio = 0.0;

      if (std::isnan(a) || std::isnan(e)) {
        if (ok) {
          abs_err = 0.0;
          rel_err = 0.0;
        } else {
          tier = 2;
          abs_err = std::numeric_limits<double>::quiet_NaN();
          rel_err = std::numeric_limits<double>::quiet_NaN();
          res.max_abs_err = abs_err;
          res.max_rel_err = rel_err;
          diff_sq.has_nan = true;
        }
      } else if (std::isinf(a) || std::isinf(e)) {
        ref_sq.add(e);
        if (a == e) {
          abs_err = 0.0;
          rel_err = 0.0;
        } else {
          tier = 1;
          abs_err = std::numeric_limits<double>::infinity();
          rel_err = std::numeric_limits<double>::infinity();
          if (!std::isnan(res.max_abs_err)) {
            res.max_abs_err = abs_err;
          }
          if (!std::isnan(res.max_rel_err)) {
            res.max_rel_err = rel_err;
          }
          diff_sq.has_inf = true;
        }
      } else {
        abs_err = std::abs(a - e);
        rel_err = (e != 0.0) ? (abs_err / std::abs(e))
                             : ((abs_err == 0.0)
                                    ? 0.0
                                    : std::numeric_limits<double>::infinity());
        diff_sq.add(a - e);
        ref_sq.add(e);

        if (!std::isnan(res.max_abs_err) && abs_err > res.max_abs_err) {
          res.max_abs_err = abs_err;
        }
        if (!std::isnan(res.max_rel_err) && rel_err > res.max_rel_err) {
          res.max_rel_err = rel_err;
        }

        const double tol = atol + rtol * std::abs(e);
        if (tol > 0.0) {
          ratio = abs_err / tol;
        } else {
          ratio = (abs_err == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
        }
      }

      const bool is_worse =
          (tier > worst_tier) ||
          (tier == worst_tier &&
           (ratio > worst_ratio ||
            (ratio == worst_ratio && abs_err > worst_abs_tie)));
      if (is_worse) {
        worst_tier = tier;
        worst_ratio = ratio;
        worst_abs_tie = abs_err;
        res.worst_row = row;
        res.worst_col = col;
        res.worst_actual = a;
        res.worst_expected = e;
      }
    }
  }

  const double diff_norm = diff_sq.norm();
  const double ref_norm = ref_sq.norm();
  if (std::isnan(diff_norm) || std::isnan(ref_norm)) {
    res.frobenius_rel_err = std::numeric_limits<double>::quiet_NaN();
  } else if (diff_norm == 0.0) {
    res.frobenius_rel_err = 0.0;
  } else if (ref_norm == 0.0 || std::isinf(diff_norm)) {
    res.frobenius_rel_err = std::numeric_limits<double>::infinity();
  } else if (std::isinf(ref_norm)) {
    res.frobenius_rel_err = 0.0;
  } else {
    res.frobenius_rel_err = diff_norm / ref_norm;
  }

  res.passed = (res.mismatched_elements == 0);
  return res;
}

// 1D contiguous host array comparison
template <typename T>
inline AllCloseResult allclose_host(const size_t count, const T *const actual,
                                    const T *const expected,
                                    const double rtol = 1e-5,
                                    const double atol = 1e-8,
                                    const bool equal_nan = false) {
  return allclose_host(count, size_t{1}, actual, count, expected, count, rtol,
                       atol, equal_nan);
}

#if OZAKI_ALLCLOSE_HAS_CUDA

// 2D column-major GPU device matrix comparison
template <typename T>
inline AllCloseResult
allclose_device(const size_t m, const size_t n, const T *const d_actual,
                const size_t ld_actual, const T *const d_expected,
                const size_t ld_expected, const double rtol = 1e-5,
                const double atol = 1e-8, const bool equal_nan = false) {
  detail::validate_args(m, n, d_actual, ld_actual, d_expected, ld_expected,
                        rtol, atol);
  if (m == 0 || n == 0) {
    return allclose_host(m, n, d_actual, ld_actual, d_expected, ld_expected,
                         rtol, atol, equal_nan);
  }
  const std::vector<T> h_actual =
      detail::copy_col_major_from_device(m, n, d_actual, ld_actual);
  const std::vector<T> h_expected =
      detail::copy_col_major_from_device(m, n, d_expected, ld_expected);
  return allclose_host(m, n, h_actual.data(), m, h_expected.data(), m, rtol,
                       atol, equal_nan);
}

// 1D contiguous GPU device array comparison
template <typename T>
inline AllCloseResult allclose_device(const size_t count,
                                      const T *const d_actual,
                                      const T *const d_expected,
                                      const double rtol = 1e-5,
                                      const double atol = 1e-8,
                                      const bool equal_nan = false) {
  return allclose_device(count, size_t{1}, d_actual, count, d_expected, count,
                         rtol, atol, equal_nan);
}

// Pointer-agnostic 2D column-major comparison (host, device, or mixed)
template <typename T>
inline AllCloseResult
allclose(const size_t m, const size_t n, const T *const actual,
         const size_t ld_actual, const T *const expected,
         const size_t ld_expected, const double rtol = 1e-5,
         const double atol = 1e-8, const bool equal_nan = false) {
  detail::validate_args(m, n, actual, ld_actual, expected, ld_expected, rtol,
                        atol);
  if (m == 0 || n == 0) {
    return allclose_host(m, n, actual, ld_actual, expected, ld_expected, rtol,
                         atol, equal_nan);
  }

  const bool actual_dev = detail::is_device_pointer(actual);
  const bool expected_dev = detail::is_device_pointer(expected);
  if (!actual_dev && !expected_dev) {
    return allclose_host(m, n, actual, ld_actual, expected, ld_expected, rtol,
                         atol, equal_nan);
  }

  std::vector<T> h_actual;
  std::vector<T> h_expected;
  const T *ptr_a = actual;
  size_t stride_a = ld_actual;
  const T *ptr_e = expected;
  size_t stride_e = ld_expected;

  if (actual_dev) {
    h_actual = detail::copy_col_major_from_device(m, n, actual, ld_actual);
    ptr_a = h_actual.data();
    stride_a = m;
  }
  if (expected_dev) {
    h_expected =
        detail::copy_col_major_from_device(m, n, expected, ld_expected);
    ptr_e = h_expected.data();
    stride_e = m;
  }
  return allclose_host(m, n, ptr_a, stride_a, ptr_e, stride_e, rtol, atol,
                       equal_nan);
}

#else

template <typename T>
inline AllCloseResult
allclose(const size_t m, const size_t n, const T *const actual,
         const size_t ld_actual, const T *const expected,
         const size_t ld_expected, const double rtol = 1e-5,
         const double atol = 1e-8, const bool equal_nan = false) {
  return allclose_host(m, n, actual, ld_actual, expected, ld_expected, rtol,
                       atol, equal_nan);
}

#endif // OZAKI_ALLCLOSE_HAS_CUDA

// Convenience tight 2D (ld == m) and 1D contiguous overloads
template <typename T>
inline AllCloseResult
allclose(const size_t m, const size_t n, const T *const actual,
         const T *const expected, const double rtol = 1e-5,
         const double atol = 1e-8, const bool equal_nan = false) {
  return allclose(m, n, actual, m, expected, m, rtol, atol, equal_nan);
}

template <typename T>
inline AllCloseResult
allclose(const size_t count, const T *const actual, const T *const expected,
         const double rtol = 1e-5, const double atol = 1e-8,
         const bool equal_nan = false) {
  return allclose(count, size_t{1}, actual, count, expected, count, rtol, atol,
                  equal_nan);
}
