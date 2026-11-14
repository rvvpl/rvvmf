# RVVMF — RISC-V Vector Math Functions

[![License](https://img.shields.io/badge/License-BSD_4--Clause-blue.svg)](https://opensource.org/licenses/BSD-4-Clause)
[![ISA](https://img.shields.io/badge/ISA-RVV_1.0-green.svg)](https://github.com/riscv/riscv-v-spec)
[![Standard](https://img.shields.io/badge/ISO%2FTR-10967--2-orange.svg)](https://www.iso.org/standard/33517.html)

**RVVMF** is a high-performance, high-accuracy vector mathematical library written in C using **RISC-V Vector Extension (RVV 1.0)** intrinsics.

Designed with a strict focus on both numerical precision and throughput, RVVMF provides an intrinsic-style API across all vector register groupings (`LMUL`) and floating-point data types (`fp64`, `fp32`, `fp16`).

---

## Key Features

* **Pure RVV 1.0 & Fully VLA:** Written entirely using standard RVV 1.0 intrinsics in a **Vector-Length Agnostic (VLA)** manner.
* **High Numerical Accuracy:** Targets compliance with **ISO/IEC 10967-2** (Language independent arithmetic) and aims for **<= 0.501 ULP** accuracy (statistically faithful rounding with at most 1 misrounded result per 1,000 random floating-point inputs).
* **Floating-Point Exceptions Support:** Engineered to properly raise IEEE 754 floating-point exceptions (`FE_INVALID`, `FE_DIVBYZERO`, `FE_OVERFLOW`, `FE_UNDERFLOW`, `FE_INEXACT`). Notice that `errno` is not supported (`-fno-math-errno`).
* **Alternative Implementations:** For operations that have native RVV hardware instructions or direct rounding counterparts (`sqrt`, `trunc`, `floor`, `ceil`, `round`), RVVMF provides alternative software/hybrid implementations that guarantee strict IEEE edge-case handling and performance profiles.
* **Uniform Intrinsic API:** Follows the standard RVV intrinsic naming convention (`__riscv_v<func>_f<type>m<lmul>`), supporting various `LMUL` configurations (`mf2`, `m1`, `m2`, `m4`, `m8`).

---

## Implemented Functions & Accuracy Status

Each cell in the **Max Known ULP**, **<= 0.501 ULP**, and **FE Checked** columns represents the status for **`fp64` / `fp32` / `fp16`** respectively.

> **Legend:** `✅` — Verified / Compliant | `🚧` — In Progress / Partial | `➖` — Not Yet Verified

| Category | Function | Base Intrinsic Name | Max Known Error (ULP) [fp64 / fp32 / fp16] | <= 0.501 ULP [fp64 / fp32 / fp16] | FE_EXCEPT Checked [fp64 / fp32 / fp16] |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Exponential** | `exp` | `__riscv_vexp` | `0.5000 / 0.5001 / 0.5002` | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ |
| | `exp2` | `__riscv_vexp2` | `0.5010 / 0.5003 / 0.5011` | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ |
| | `expm1` | `__riscv_vexpm1` | `0.5010 / 0.5071 / 0.5144` | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ |
| **Logarithmic** | `log` | `__riscv_vlog` | `0.5000 / 0.5168 / 0.5083` | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ |
| | `log2` | `__riscv_vlog2` | `0.5000 / 0.5016 / 0.5005` | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ |
| | `log10` | `__riscv_vlog10` | `0.5000 / 0.5185 / 0.5332` | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ |
| | `log1p` | `__riscv_vlog1p` | `0.5001 / 0.5278 / 0.5086` | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ |
| **Trigonometric** | `sin` | `__riscv_vsin` | `0.5972 / 0.5165 / 0.5028` | ✅ / ✅ / ✅ | 🚧 / 🚧 / 🚧 |
| | `cos` | `__riscv_vcos` | `0.6029 / 0.5164 / 0.6487` | ✅ / ✅ / ✅ | 🚧 / 🚧 / 🚧 |
| | `tan` | `__riscv_vtan` | `0.5241 / 0.5016 / 0.6412` | ✅ / ✅ / ✅ | ✅ / ✅ / 🚧 |
| | `asin` | `__riscv_vasin` | `0.5004 / 0.8289 / 0.5049` | ✅ / ✅ / ✅ | 🚧 / 🚧 / 🚧 |
| | `acos` | `__riscv_vacos` | `0.5000 / 0.5000 / 0.5037` | ✅ / ✅ / ✅ | 🚧 / 🚧 / 🚧 |
| | `atan` | `__riscv_vatan` | `0.5026 / 0.5992 / 0.5024` | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ |
| | `atan2` | `__riscv_vatan2` | `0.5000 / 0.5089 / 0.7499` | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ |
| **Hyperbolic** | `sinh` | `__riscv_vsinh` | `0.9999 / 0.9999 / 0.5012` | ✅ / ✅ / ✅ | ✅ / ✅ / 🚧 |
| | `cosh` | `__riscv_vcosh` | `0.5042 / 0.5316 / 0.5000` | ✅ / ✅ / ✅ | ✅ / ✅ / 🚧 |
| | `tanh` | `__riscv_vtanh` | `0.5609 / 0.5262 / 0.5155` | ✅ / ✅ / ✅ | 🚧 / 🚧 / 🚧 |
| **Power & Root** | `sqrt`* | `__riscv_vsqrt` | `0.5 / 0.5 / 0.5` | ✅ / ✅ / ✅ | 🚧 / 🚧 / 🚧 |
| | `cbrt` | `__riscv_vcbrt` | `0.5000 / 0.5000 / 0.5000` | ✅ / ✅ / ✅ | 🚧 / 🚧 / 🚧 |
| | `pow` | `__riscv_vpow` | `0.5052 / 0.7099 / 0.7504` | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ |
| | `hypot` | `__riscv_vhypot` | `0.8297 / 0.8849 / 0.9350` | ✅ / ✅ / ✅ | 🚧 / 🚧 / 🚧 |
| **Rounding** | `trunc`* | `__riscv_vtrunc` | `` | `` | ➖ / ➖ / ➖ |
| | `floor`* | `__riscv_vfloor` | `` | `` | ➖ / ➖ / ➖  |
| | `ceil`* | `__riscv_vceil` | `` | `` | ➖ / ➖ / ➖  |
| | `round`* | `__riscv_vround` | `` | `` | ➖ / ➖ / ➖  |

*\*Note: Functions marked with an asterisk provide alternative implementations in RVVMF, ensuring strict edge-case behavior and consistent flag semantics alongside native RVV instructions.*

---

## API & Usage Example

All functions follow the standard RVV intrinsic signature pattern: `__riscv_v<func>_f<bits>m<lmul>(v_arg, vl)`.

```c
#include <riscv_vector.h>
#include "rvvmf.h"

void compute_exp_f64(const double *src, double *dst, size_t n) {
    size_t vl;
    for (; n > 0; n -= vl, src += vl, dst += vl) {
        vl = __riscv_vsetvl_e64m1(n);
        vfloat64m1_t vx = __riscv_vle64_v_f64m1(src, vl);

        // Call RVVMF vector exponential (fp64, LMUL=1)
        vfloat64m1_t vy = __riscv_vexp_f64m1(vx, vl);

        __riscv_vse64_v_f64m1(dst, vy, vl);
    }
}
```

---

## Building from Source

RVVMF uses **CMake** and includes pre-configured toolchain files for cross-compiling with both **GCC** and **Clang/LLVM** (requires a compiler with RVV 1.0 intrinsic support).

### Cross-Compiling with CMake

```bash
git clone https://github.com/rvvpl/rvvmf.git
cd rvvmf

# Configure and build using the GCC toolchain file
cmake -B build \
      -DCMAKE_TOOLCHAIN_FILE=toolchains/riscv64-1p0-clang-f16.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

*(Note: You can switch to the GCC toolchain by passing the corresponding `.cmake` toolchain file to `-DCMAKE_TOOLCHAIN_FILE`).*

---

## Verification & Benchmarks

Accuracy (ULP) and throughput (**CPE** — Cycles Per Element) are verified on RISC-V RVV 1.0 hardware using an internal validation and benchmarking suite.

RVVMF is actively benchmarked against:
* **Scalar `glibc` (`libm`)**
* [**SLEEF**](https://github.com/shibatch/sleef) (SIMD Library for Evaluating Elementary Functions)
* [**veclibm**](https://github.com/rivosinc/veclibm) (RIVOS's RISC-V vector math implementations)

Detailed accuracy and CPE performance comparison tables are provided in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

## License

This project is licensed under the **4-Clause BSD License** — see the [LICENSE](LICENSE) file for details.

The software is provided "as is", without warranty of any kind, express or implied.