/*
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of
 * Nizhny Novgorod and its affiliates. All rights reserved.
 *
 * Copyright 2026 The RVVMF Authors (Kabalova Valeriia)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  sinh_cosh_macro.inl                          *
 *   Contains: helper macros for sinh, cosh functions    *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#ifndef __RVVMF_SC_HELPER_MACRO__
#define __RVVMF_SC_HELPER_MACRO__

#ifndef forceinline
    #if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
        #define forceinline __attribute__((always_inline)) inline
    #else
        #define forceinline inline
    #endif
#endif

/* fe exception macros */
#define RVVMF_SC_CALL_FE_INVALID() volatile double exception = 0.0 / 0.0

#define RVVMF_SC_CALL_FE_OVERFLOW() volatile double exception = DBL_MAX * 2.0

#define RVVMF_SC_CALL_FE_UNDERFLOW() volatile double exception = nextafter(DBL_MIN / (double((uint64_t)1 << 52)), 0.0)

/* c reinterpret macros */
#define RVVMF_SC_AS_FP64(x) (*(double*)(&x))

#define RVVMF_SC_AS_FP32(x) (*(float*)(&x))

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
  #define RVVMF_SC_AS_FP16(x) (*(_Float16*)(&x))
#endif

/* rvv reinterpret macros */
#define RVVMF_U64_TO_F64(x, postfix) \
  (__riscv_vreinterpret_v_u64##postfix##_f64##postfix(x))
#define RVVMF_F64_TO_U64(x, postfix) \
  (__riscv_vreinterpret_v_f64##postfix##_u64##postfix(x))
#define RVVMF_U64_TO_I64(x, postfix) \
  (__riscv_vreinterpret_v_u64##postfix##_i64##postfix(x))

#define RVVMF_U32_TO_F32(x, postfix) \
  (__riscv_vreinterpret_v_u32##postfix##_f32##postfix(x))
#define RVVMF_F32_TO_U32(x, postfix) \
  (__riscv_vreinterpret_v_f32##postfix##_u32##postfix(x))
#define RVVMF_U32_TO_I32(x, postfix) \
  (__riscv_vreinterpret_v_u32##postfix##_i32##postfix(x))

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
#define RVVMF_U16_TO_F16(x, postfix) \
  (__riscv_vreinterpret_v_u16##postfix##_f16##postfix(x))
#define RVVMF_F16_TO_U16(x, postfix) \
  (__riscv_vreinterpret_v_f16##postfix##_u16##postfix(x))
#define RVVMF_U16_TO_I16(x, postfix) \
  (__riscv_vreinterpret_v_u16##postfix##_i16##postfix(x))
#endif

/* double-FP arithmetic macros */
#define RVVMF_SC_FAST2SUM_VV(postfix, vtype, a, b, sh, sl, vl) /* |a| > |b| */ \
  sh = __riscv_vfadd_vv_##postfix(a, b, vl); \
  sl = __riscv_vfsub_vv_##postfix(b, __riscv_vfsub_vv_##postfix(sh, a, vl), vl)

#define RVVMF_SC_MUL12_VV(postfix, vtype, ah, bh, zh, zl, vl) \
  zh = __riscv_vfmul_vv_##postfix(ah, bh, vl); \
  zl = __riscv_vfmsub_vv_##postfix(ah, bh, zh, vl); \

#define RVVMF_SC_MUL22_VV(postfix, vtype, ah, al, bh, bl, zh, zl, vl) \
  zh = __riscv_vfmul_vv_##postfix(ah, bh, vl); \
  zl = __riscv_vfmsub_vv_##postfix(ah, bh, zh, vl); \
  zl = __riscv_vfadd_vv_##postfix(zl, __riscv_vfmadd_vv_##postfix(ah, bl, __riscv_vfmul_vv_##postfix(al, bh, vl), vl), vl);\

#define RVVMF_SC_FMA12_VV(postfix, vtype, ah, bh, ch, zh, zl, vl) \
  zh = __riscv_vfmadd_vv_##postfix(ah, bh, ch, vl); \
  /* fast2sum vector-vector (-zh, ch, sh, sl)*/ \
  vtype sh = __riscv_vfsub_vv_##postfix(ch, zh, vl); \
  vtype sl = __riscv_vfsub_vv_##postfix(ch, __riscv_vfadd_vv_##postfix(sh, zh, vl), vl); \
  zl = __riscv_vfmadd_vv_##postfix(ah, bh, sh, vl); \
  zl = __riscv_vfadd_vv_##postfix(zl, sl, vl);

  #define RVVMF_SC_FMA12_VER2_VF(postfix, vtype, ah, bh, ch, zh, zl, vl, literal) \
  zh = __riscv_vfmadd_vf_##postfix(ah, bh, ch, vl); \
  /* fast2sum vector-vector (ch, -zh, sh, sl)*/ \
  vtype sh = __riscv_vfsub_vv_##postfix(ch, zh, vl); \
  vtype sl = __riscv_vfsub_vv_##postfix( \
             __riscv_vfsub_vv_##postfix( \
             __riscv_vfmv_v_f_##postfix(0.0##literal, vl), \
             __riscv_vfsub_vv_##postfix(sh, ch, vl), vl), zh, vl);\
  zl = __riscv_vfmadd_vf_##postfix(ah, bh, sh, vl); \
  zl = __riscv_vfadd_vv_##postfix(zl, sl, vl);

#define RVVMF_SC_ADD22_VV(postfix, vtype, ah, al, bh, bl, zh, zl, vl) \
  vtype sh = __riscv_vfadd_vv_##postfix(ah, bh, vl); \
  vtype sl = __riscv_vfadd_vv_##postfix( \
          __riscv_vfadd_vv_##postfix( \
          __riscv_vfadd_vv_##postfix( \
          __riscv_vfsub_vv_##postfix(ah, sh, vl), bh, vl), bl, vl), al, vl); \
  zh = __riscv_vfadd_vv_##postfix(sh, sl, vl); \
  zl = __riscv_vfadd_vv_##postfix(__riscv_vfsub_vv_##postfix(sh, zh, vl), sl, vl);

#define RVVMF_SC_SUB22_VV(postfix, vtype, ah, al, bh, bl, zh, zl, vl) \
  vtype sh = __riscv_vfsub_vv_##postfix(ah, bh, vl); \
  vtype sl = __riscv_vfadd_vv_##postfix( \
          __riscv_vfsub_vv_##postfix( \
          __riscv_vfsub_vv_##postfix( \
          __riscv_vfsub_vv_##postfix(ah, sh, vl), bh, vl), bl, vl), al, vl); \
  zh = __riscv_vfadd_vv_##postfix(sh, sl, vl); \
  zl = __riscv_vfadd_vv_##postfix(__riscv_vfsub_vv_##postfix(sh, zh, vl), sl, vl);

#define RVVMF_SC_ADD21_VV(postfix, vtype, ah, al, bh, bl, zh, vl) \
  vtype sh = __riscv_vfadd_vv_##postfix(ah, bh, vl); \
  vtype sl = __riscv_vfadd_vv_##postfix( \
          __riscv_vfadd_vv_##postfix( \
          __riscv_vfadd_vv_##postfix( \
          __riscv_vfsub_vv_##postfix(ah, sh, vl), bh, vl), bl, vl), al, vl); \
  zh = __riscv_vfadd_vv_##postfix(sh, sl, vl);

#define RVVMF_SC_ADD21_2_VV(postfix, vtype, ah, al, bh, bl, zh, vl, literal) \
  vtype sh = __riscv_vfmadd_vv_##postfix(__riscv_vfmv_v_f_##postfix(2.0##literal, vl), ah, bh, vl); \
  vtype sl = __riscv_vfmadd_vv_##postfix(__riscv_vfmv_v_f_##postfix(2.0##literal, vl), ah, \
             __riscv_vfsub_vv_##postfix( \
             __riscv_vfmv_v_f_##postfix(0.0##literal, vl), sh, vl), vl);\
  sl = __riscv_vfadd_vv_##postfix(sl, bh, vl); \
  sl = __riscv_vfadd_vv_##postfix(sl, bl, vl); \
  sl = __riscv_vfmadd_vv_##postfix(__riscv_vfmv_v_f_##postfix(2.0##literal, vl), al, sl, vl); \
  zh = __riscv_vfadd_vv_##postfix(sh, sl, vl);

#endif
