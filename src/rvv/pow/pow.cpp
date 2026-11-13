/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2026 The RVVMF Authors (Egor Korotin, 
 * Vladislav Polozov)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  pow.cpp                                      *
 *   Contains:  intrinsic function pow for f64, f32, f16 *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input VL number of elements in vector register        *
 *                                                       *
 * Algorithm:                                            *
 *    1) Calculate log2 function from x (first argument) *
 *    2) Calculate exp2 function from y * log2(x)        *
 *                                                       *
 * Based on:						 *
 *    1) log2 implementation from src/rvv/log            *
 *    2) exp2 implementation from src/rvv/exp            *
 *                                                       *
 *********************************************************
*/


#include "pow_utilities.inl"
#include <iostream>
#include <fenv.h>

#include "pow_f64.data"
#include "pow_f32.data"
#ifdef __riscv_zvfh
#include "pow_f16.data"
#endif

#include "exp2_components.inl"
#include "log2_components.inl"

#define RVVMF_EXP_CALL_FE_DIVBYZERO() volatile double exception = 1.0/0.0
#define RVVMF_EXP_CALL_FE_INVALID() volatile double exception = 0.0/0.0

#ifndef __FAST_MATH__

#define RISCV_POW_F64(LMUL, BOOL)   \
void check_pow_special_cases_f64##LMUL(vfloat64##LMUL##_t& x, vfloat64##LMUL##_t& y, vfloat64##LMUL##_t& specRes, vint64##LMUL##_t& s_ans, vbool##BOOL##_t& specMask, size_t vl) {\
    vfloat64##LMUL##_t abs_y;\
    vint64##LMUL##_t sx, int_y, diff_y;\
    vuint64##LMUL##_t specResU;\
    vbool##BOOL##_t zero_x, inf_x, inf_y, spec_x, spec_y, uncorrect, triv_ans, mask_nan, y_lt0, x_nem1;\
    sx = __riscv_vreinterpret_v_u64##LMUL##_i64##LMUL(__riscv_vand_vx_u64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(x), SIGN_MASK_F64, vl));\
\
    x_nem1 = __riscv_vmfne_vf_f64##LMUL##_b##BOOL(x, -1.0, vl);\
    x = __riscv_vreinterpret_v_i64##LMUL##_f64##LMUL(__riscv_vand_vx_i64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(x), ABS_MASK_F64, vl));\
    triv_ans =  __riscv_vmor_mm_b##BOOL(__riscv_vmfeq_vf_f64##LMUL##_b##BOOL(x, 1.0, vl), __riscv_vmfeq_vf_f64##LMUL##_b##BOOL(y, 0.0, vl), vl);\
\
    zero_x = __riscv_vmfeq_vf_f64##LMUL##_b##BOOL(x, 0.0, vl);\
    inf_x = __riscv_vmseq_vx_u64##LMUL##_b##BOOL(__riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(x), EXP_MASK_F64, vl);\
    spec_y = __riscv_vmsgtu_vx_u64##LMUL##_b##BOOL(__riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(x), ONE_MASK_F64, vl);\
    spec_x = __riscv_vmor_mm_b##BOOL(zero_x, inf_x, vl);\
\
    abs_y = __riscv_vreinterpret_v_i64##LMUL##_f64##LMUL(__riscv_vand_vx_i64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(y), ABS_MASK_F64, vl));\
    inf_y = __riscv_vmseq_vx_u64##LMUL##_b##BOOL(__riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(abs_y), EXP_MASK_F64, vl);\
    abs_y = __riscv_vreinterpret_v_i64##LMUL##_f64##LMUL(__riscv_vxor_vv_i64##LMUL##_mu(__riscv_vmsge_vx_i64##LMUL##_b##BOOL(__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(abs_y), INT_MAX_P1_I64, vl),\
									__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(abs_y), \
									__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(abs_y), \
									__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(abs_y), \
									vl));\
    int_y = __riscv_vfcvt_rtz_x_f_v_i64##LMUL(abs_y, vl);\
\
    s_ans = __riscv_vand_vv_i64##LMUL(sx, __riscv_vsll_vx_i64##LMUL(__riscv_vand_vx_i64##LMUL(int_y, 1, vl), BITS_F64, vl), vl);\
\
    diff_y = __riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(__riscv_vfsub_vv_f64##LMUL(abs_y, __riscv_vfcvt_f_x_v_f64##LMUL(int_y, vl), vl));\
    uncorrect = __riscv_vmsne_vx_i64##LMUL##_b##BOOL(__riscv_vand_vv_i64##LMUL(diff_y, __riscv_vsra_vx_i64##LMUL(sx, BITS_F64, vl), vl), 0, vl);\
    s_ans = __riscv_vxor_vv_i64##LMUL##_mu(__riscv_vmand_mm_b##BOOL(uncorrect, spec_x, vl), s_ans, s_ans, s_ans, vl);\
    uncorrect = __riscv_vmandn_mm_b##BOOL(uncorrect, spec_x, vl);\
\
    if (__riscv_vcpop_m_b##BOOL(uncorrect, vl)) RVVMF_EXP_CALL_FE_INVALID();\
    mask_nan = __riscv_vmor_mm_b##BOOL(__riscv_vmandn_mm_b##BOOL(__riscv_vmor_mm_b##BOOL(__riscv_vmfne_vv_f64##LMUL##_b##BOOL(x, x, vl), __riscv_vmfne_vv_f64##LMUL##_b##BOOL(y, y, vl), vl), __riscv_vmand_mm_b##BOOL(triv_ans, x_nem1, vl), vl), uncorrect, vl);\
\
    y_lt0 = __riscv_vmsgtu_vx_u64##LMUL##_b##BOOL(__riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(y), SIGN_MASK_F64, vl);\
    spec_y = __riscv_vmxor_mm_b##BOOL(y_lt0, spec_y, vl);\
    if (__riscv_vcpop_m_b##BOOL(__riscv_vmand_mm_b##BOOL(y_lt0, zero_x, vl), vl)) RVVMF_EXP_CALL_FE_DIVBYZERO();\
    specMask = __riscv_vmor_mm_b##BOOL(__riscv_vmor_mm_b##BOOL(mask_nan, spec_x, vl), inf_y, vl);\
\
    specResU = __riscv_vmerge_vxm_u64##LMUL(__riscv_vmv_v_x_u64##LMUL(uint64_t(0), vl), EXP_MASK_F64, spec_y, vl);\
\
    specResU = __riscv_vmerge_vxm_u64##LMUL(specResU, ONE_MASK_F64, triv_ans, vl);\
    specRes =  __riscv_vreinterpret_v_u64##LMUL##_f64##LMUL(__riscv_vmerge_vxm_u64##LMUL(specResU, NAN_MASK_F64, mask_nan, vl));\
\
    triv_ans = __riscv_vmor_mm_b##BOOL(specMask, triv_ans, vl);\
\
    y = __riscv_vreinterpret_v_i64##LMUL##_f64##LMUL(__riscv_vxor_vv_i64##LMUL##_mu(triv_ans, __riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(y), __riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(y), __riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(y), vl));\
    x = __riscv_vfmerge_vfm_f64##LMUL(x, 1.0, triv_ans, vl);\
}\
\
vfloat64##LMUL##_t __riscv_vpow_f64##LMUL(vfloat64##LMUL##_t x, vfloat64##LMUL##_t y, size_t vl) {\
  vfloat64##LMUL##_t logxh, logxl, specRes;\
  vfloat64##LMUL##_t l, h, r, res;\
  vbool##BOOL##_t specMask, small_exp;\
  vint64##LMUL##_t s_ans, r_sign;\
  vuint64##LMUL##_t specialExp;\
  vuint64##LMUL##_t r_exp;\
  vbool##BOOL##_t exp_check;\
\
  r_exp = __riscv_vand_vx_u64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(y), EXP_MASK_F64, vl);\
\
  check_pow_special_cases_f64##LMUL(x, y, specRes, s_ans, specMask, vl);\
\
  log22_f64##LMUL(x, logxh, logxl, vl);\
\
  r_exp = __riscv_vadd_vv_u64##LMUL(r_exp, __riscv_vand_vx_u64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(logxh), EXP_MASK_F64, vl), vl);\
  r_sign = __riscv_vsra_vx_i64##LMUL(__riscv_vxor_vv_i64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(y), __riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(logxh), vl), BITS_F64, vl);\
\
  exp_check = __riscv_vmsltu_vx_u64##LMUL##_b##BOOL(r_exp, UF_EXP_TRESHOLD_F64, vl);\
\
  specialExp = __riscv_vor_vx_u64##LMUL(__riscv_vand_vx_u64##LMUL(__riscv_vreinterpret_v_i64##LMUL##_u64##LMUL(r_sign), UF_TRESHOLD_U64, vl), OF_TRESHOLD_U64, vl);\
\
  specialExp = __riscv_vxor_vv_u64##LMUL##_mu(exp_check, specialExp, specialExp, specialExp, vl);\
  exp_check = __riscv_vmandn_mm_b##BOOL(__riscv_vmsltu_vx_u64##LMUL##_b##BOOL(r_exp, OF_EXP_TRESHOLD_F64, vl), exp_check, vl);\
\
  r = __riscv_vfmul_vv_f64##LMUL##_mu(exp_check, __riscv_vfmv_v_f_f64##LMUL(0.0, vl), y, logxh, vl);\
  small_exp = __riscv_vmand_mm_b##BOOL(__riscv_vmsltu_vv_u64##LMUL##_b##BOOL(__riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(r), specialExp, vl), exp_check, vl);\
  l = __riscv_vfmsac_vv_f64##LMUL##_mu(exp_check, r, y, logxh, vl);\
  r = __riscv_vmerge_vvm_f64##LMUL(__riscv_vreinterpret_v_u64##LMUL##_f64##LMUL(specialExp), r, small_exp, vl);\
  l = __riscv_vfmacc_vv_f64##LMUL##_mu(small_exp, l, y, logxl, vl);\
\
  res = special_exp_f64##LMUL(r, l, vl);\
\
  res = __riscv_vmerge_vvm_f64##LMUL(res, specRes, specMask, vl);\
  res = __riscv_vreinterpret_v_i64##LMUL##_f64##LMUL(__riscv_vor_vv_i64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(res), s_ans, vl));\
\
  return __riscv_vfmadd_vf_f64##LMUL(res, 0x1p-60, res, vl);\
}

#define RISCV_POW_F32(LMUL, BOOL)   \
void check_pow_special_cases_f32##LMUL(vfloat32##LMUL##_t& x, vfloat32##LMUL##_t& y, vfloat32##LMUL##_t& specRes, vint32##LMUL##_t& s_ans, vbool##BOOL##_t& specMask, size_t vl) {\
    vfloat32##LMUL##_t abs_y;\
    vint32##LMUL##_t sx, int_y, diff_y;\
    vuint32##LMUL##_t specResU;\
    vbool##BOOL##_t zero_x, inf_x, inf_y, spec_x, spec_y, uncorrect, triv_ans, mask_nan, y_lt0, x_nem1;\
    sx = __riscv_vreinterpret_v_u32##LMUL##_i32##LMUL(__riscv_vand_vx_u32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(x), SIGN_MASK_F32, vl));\
\
    x_nem1 = __riscv_vmfne_vf_f32##LMUL##_b##BOOL(x, -1.0f, vl);\
    x = __riscv_vreinterpret_v_i32##LMUL##_f32##LMUL(__riscv_vand_vx_i32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(x), ABS_MASK_F32, vl));\
    triv_ans =  __riscv_vmor_mm_b##BOOL(__riscv_vmfeq_vf_f32##LMUL##_b##BOOL(x, 1.0f, vl), __riscv_vmfeq_vf_f32##LMUL##_b##BOOL(y, 0.0f, vl), vl);\
\
    zero_x = __riscv_vmfeq_vf_f32##LMUL##_b##BOOL(x, 0.0f, vl);\
    inf_x = __riscv_vmseq_vx_u32##LMUL##_b##BOOL(__riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(x), EXP_MASK_F32, vl);\
    spec_y = __riscv_vmsgtu_vx_u32##LMUL##_b##BOOL(__riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(x), ONE_MASK_F32, vl);\
    spec_x = __riscv_vmor_mm_b##BOOL(zero_x, inf_x, vl);\
\
    abs_y = __riscv_vreinterpret_v_i32##LMUL##_f32##LMUL(__riscv_vand_vx_i32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(y), ABS_MASK_F32, vl));\
    inf_y = __riscv_vmseq_vx_u32##LMUL##_b##BOOL(__riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(abs_y), EXP_MASK_F32, vl);\
    abs_y = __riscv_vreinterpret_v_i32##LMUL##_f32##LMUL(__riscv_vxor_vv_i32##LMUL##_mu(__riscv_vmsge_vx_i32##LMUL##_b##BOOL(__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(abs_y), INT_MAX_P1_I32, vl),\
									__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(abs_y), \
									__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(abs_y), \
									__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(abs_y), \
									vl));\
    int_y = __riscv_vfcvt_rtz_x_f_v_i32##LMUL(abs_y, vl);\
\
    s_ans = __riscv_vand_vv_i32##LMUL(sx, __riscv_vsll_vx_i32##LMUL(__riscv_vand_vx_i32##LMUL(int_y, 1, vl), BITS_F32, vl), vl);\
\
    diff_y = __riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(__riscv_vfsub_vv_f32##LMUL(abs_y, __riscv_vfcvt_f_x_v_f32##LMUL(int_y, vl), vl));\
    uncorrect = __riscv_vmsne_vx_i32##LMUL##_b##BOOL(__riscv_vand_vv_i32##LMUL(diff_y, __riscv_vsra_vx_i32##LMUL(sx, BITS_F32, vl), vl), 0, vl);\
    s_ans = __riscv_vxor_vv_i32##LMUL##_mu(__riscv_vmand_mm_b##BOOL(uncorrect, spec_x, vl), s_ans, s_ans, s_ans, vl);\
    uncorrect = __riscv_vmandn_mm_b##BOOL(uncorrect, spec_x, vl);\
\
    if (__riscv_vcpop_m_b##BOOL(uncorrect, vl)) RVVMF_EXP_CALL_FE_INVALID();\
    mask_nan = __riscv_vmor_mm_b##BOOL(__riscv_vmandn_mm_b##BOOL(__riscv_vmor_mm_b##BOOL(__riscv_vmfne_vv_f32##LMUL##_b##BOOL(x, x, vl), __riscv_vmfne_vv_f32##LMUL##_b##BOOL(y, y, vl), vl), __riscv_vmand_mm_b##BOOL(triv_ans, x_nem1, vl), vl), uncorrect, vl);\
\
    y_lt0 = __riscv_vmsgtu_vx_u32##LMUL##_b##BOOL(__riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(y), SIGN_MASK_F32, vl);\
    spec_y = __riscv_vmxor_mm_b##BOOL(y_lt0, spec_y, vl);\
    if (__riscv_vcpop_m_b##BOOL(__riscv_vmand_mm_b##BOOL(y_lt0, zero_x, vl), vl)) RVVMF_EXP_CALL_FE_DIVBYZERO();\
    specMask = __riscv_vmor_mm_b##BOOL(__riscv_vmor_mm_b##BOOL(mask_nan, spec_x, vl), inf_y, vl);\
\
    specResU = __riscv_vmerge_vxm_u32##LMUL(__riscv_vmv_v_x_u32##LMUL(uint32_t(0), vl), EXP_MASK_F32, spec_y, vl);\
\
    specResU = __riscv_vmerge_vxm_u32##LMUL(specResU, ONE_MASK_F32, triv_ans, vl);\
    specRes =  __riscv_vreinterpret_v_u32##LMUL##_f32##LMUL(__riscv_vmerge_vxm_u32##LMUL(specResU, NAN_MASK_F32, mask_nan, vl));\
\
    triv_ans = __riscv_vmor_mm_b##BOOL(specMask, triv_ans, vl);\
\
    y = __riscv_vreinterpret_v_i32##LMUL##_f32##LMUL(__riscv_vxor_vv_i32##LMUL##_mu(triv_ans, __riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(y), __riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(y), __riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(y), vl));\
    x = __riscv_vfmerge_vfm_f32##LMUL(x, 1.0f, triv_ans, vl);\
}\
\
vfloat32##LMUL##_t __riscv_vpow_f32##LMUL(vfloat32##LMUL##_t x, vfloat32##LMUL##_t y, size_t vl) {\
  vfloat32##LMUL##_t logxh, logxl, specRes;\
  vfloat32##LMUL##_t l, h, r, res;\
  vbool##BOOL##_t specMask, small_exp;\
  vint32##LMUL##_t s_ans, r_sign;\
  vuint32##LMUL##_t specialExp;\
  vuint32##LMUL##_t r_exp;\
  vbool##BOOL##_t exp_check;\
\
  r_exp = __riscv_vand_vx_u32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(y), EXP_MASK_F32, vl);\
\
  check_pow_special_cases_f32##LMUL(x, y, specRes, s_ans, specMask, vl);\
  log22_f32##LMUL(x, logxh, logxl, vl);\
\
  r_exp = __riscv_vadd_vv_u32##LMUL(r_exp, __riscv_vand_vx_u32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(logxh), EXP_MASK_F32, vl), vl);\
  r_sign = __riscv_vsra_vx_i32##LMUL(__riscv_vxor_vv_i32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(y), __riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(logxh), vl), BITS_F32, vl);\
\
  exp_check = __riscv_vmsltu_vx_u32##LMUL##_b##BOOL(r_exp, UF_EXP_TRESHOLD_F32, vl);\
\
  specialExp = __riscv_vor_vx_u32##LMUL(__riscv_vand_vx_u32##LMUL(__riscv_vreinterpret_v_i32##LMUL##_u32##LMUL(r_sign), UF_TRESHOLD_U32, vl), OF_TRESHOLD_U32, vl);\
\
  specialExp = __riscv_vxor_vv_u32##LMUL##_mu(exp_check, specialExp, specialExp, specialExp, vl);\
  exp_check = __riscv_vmandn_mm_b##BOOL(__riscv_vmsltu_vx_u32##LMUL##_b##BOOL(r_exp, OF_EXP_TRESHOLD_F32, vl), exp_check, vl);\
\
  r = __riscv_vfmul_vv_f32##LMUL##_mu(exp_check, __riscv_vfmv_v_f_f32##LMUL(0.0f, vl), y, logxh, vl);\
  small_exp = __riscv_vmand_mm_b##BOOL(__riscv_vmsltu_vv_u32##LMUL##_b##BOOL(__riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(r), specialExp, vl), exp_check, vl);\
  l = __riscv_vfmsac_vv_f32##LMUL##_mu(exp_check, r, y, logxh, vl);\
\
  r = __riscv_vmerge_vvm_f32##LMUL(__riscv_vreinterpret_v_u32##LMUL##_f32##LMUL(specialExp), r, small_exp, vl);\
  l = __riscv_vfmacc_vv_f32##LMUL##_mu(small_exp, l, y, logxl, vl);\
\
  res = special_exp_f32##LMUL(r, l, vl);\
\
  res = __riscv_vmerge_vvm_f32##LMUL(res, specRes, specMask, vl);\
  res = __riscv_vreinterpret_v_i32##LMUL##_f32##LMUL(__riscv_vor_vv_i32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(res), s_ans, vl));\
\
  return __riscv_vfmadd_vf_f32##LMUL(res, 0x1p-30f, res, vl);\
}

#else

#define RISCV_POW_F64(LMUL, BOOL)   \
void check_pow_special_cases_f64##LMUL(vfloat64##LMUL##_t& x, vfloat64##LMUL##_t& y, vfloat64##LMUL##_t& specRes, vint64##LMUL##_t& s_ans, vbool##BOOL##_t& specMask, size_t vl) {\
    vfloat64##LMUL##_t abs_y;\
    vint64##LMUL##_t sx, int_y, diff_y;\
    vuint64##LMUL##_t specResU;\
    vbool##BOOL##_t zero_x, inf_x, spec_x, uncorrect, triv_ans, mask_nan, y_lt0, x_nem1;\
    sx = __riscv_vreinterpret_v_u64##LMUL##_i64##LMUL(__riscv_vand_vx_u64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(x), SIGN_MASK_F64, vl));\
\
    specMask = __riscv_vmfeq_vf_f64##LMUL##_b##BOOL(x, 0.0, vl);\
    abs_y = __riscv_vreinterpret_v_i64##LMUL##_f64##LMUL(__riscv_vand_vx_i64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(y), ABS_MASK_F64, vl));\
    abs_y = __riscv_vreinterpret_v_i64##LMUL##_f64##LMUL(__riscv_vxor_vv_i64##LMUL##_mu(__riscv_vmsge_vx_i64##LMUL##_b##BOOL(__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(abs_y), INT_MAX_P1_I64, vl),\
									__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(abs_y), \
									__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(abs_y), \
									__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(abs_y), \
									vl));\
    int_y = __riscv_vfcvt_rtz_x_f_v_i64##LMUL(abs_y, vl);\
\
    s_ans = __riscv_vand_vv_i64##LMUL(sx, __riscv_vsll_vx_i64##LMUL(__riscv_vand_vx_i64##LMUL(int_y, 1, vl), BITS_F64, vl), vl);\
\
    diff_y = __riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(__riscv_vfsub_vv_f64##LMUL(abs_y, __riscv_vfcvt_f_x_v_f64##LMUL(int_y, vl), vl));\
    uncorrect = __riscv_vmsne_vx_i64##LMUL##_b##BOOL(diff_y, 0, vl);\
\
    s_ans = __riscv_vxor_vv_i64##LMUL##_mu(__riscv_vmand_mm_b##BOOL(uncorrect, specMask, vl), s_ans, s_ans, s_ans, vl);\
    x = __riscv_vfmerge_vfm_f64##LMUL(x, 1.0, specMask, vl);\
    specRes = __riscv_vfmv_v_f_f64##LMUL(0.0, vl);\
}\
\
vfloat64##LMUL##_t __riscv_vpow_f64##LMUL(vfloat64##LMUL##_t x, vfloat64##LMUL##_t y, size_t vl) {\
  vfloat64##LMUL##_t logxh, logxl, specRes;\
  vfloat64##LMUL##_t l, h, r, res;\
  vbool##BOOL##_t specMask, small_exp;\
  vint64##LMUL##_t s_ans, r_sign;\
  vuint64##LMUL##_t specialExp;\
\
  check_pow_special_cases_f64##LMUL(x, y, specRes, s_ans, specMask, vl);\
  log22_f64##LMUL(x, logxh, logxl, vl);\
\
  r_sign = __riscv_vsra_vx_i64##LMUL(__riscv_vxor_vv_i64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(y), __riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(logxh), vl), BITS_F64, vl);\
\
\
  specialExp = __riscv_vor_vx_u64##LMUL(__riscv_vand_vx_u64##LMUL(__riscv_vreinterpret_v_i64##LMUL##_u64##LMUL(r_sign), UF_TRESHOLD_U64, vl), OF_TRESHOLD_U64, vl);\
\
  r = __riscv_vfmul_vv_f64##LMUL(y, logxh, vl);\
  small_exp = __riscv_vmsgeu_vv_u64##LMUL##_b##BOOL(__riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(r), specialExp, vl);\
  l = __riscv_vfmsub_vv_f64##LMUL(y, logxh, r, vl);\
  r = __riscv_vmerge_vvm_f64##LMUL(r, __riscv_vreinterpret_v_u64##LMUL##_f64##LMUL(specialExp), small_exp, vl);\
  l = __riscv_vfmadd_vv_f64##LMUL(y, logxl, l, vl);\
  l = __riscv_vreinterpret_v_i64##LMUL##_f64##LMUL(__riscv_vxor_vv_i64##LMUL##_mu(small_exp, __riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(l), __riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(l), __riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(l), vl));\
  res = special_exp_f64##LMUL(r, l, vl);\
\
  res = __riscv_vmerge_vvm_f64##LMUL(res, specRes, specMask, vl);\
  res = __riscv_vreinterpret_v_i64##LMUL##_f64##LMUL(__riscv_vor_vv_i64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(res), s_ans, vl));\
\
  return res;\
}

#define RISCV_POW_F32(LMUL, BOOL)   \
void check_pow_special_cases_f32##LMUL(vfloat32##LMUL##_t& x, vfloat32##LMUL##_t& y, vfloat32##LMUL##_t& specRes, vint32##LMUL##_t& s_ans, vbool##BOOL##_t& specMask, size_t vl) {\
    vfloat32##LMUL##_t abs_y;\
    vint32##LMUL##_t sx, int_y, diff_y;\
    vuint32##LMUL##_t specResU;\
    vbool##BOOL##_t zero_x, inf_x, spec_x, uncorrect, triv_ans, mask_nan, y_lt0, x_nem1;\
    sx = __riscv_vreinterpret_v_u32##LMUL##_i32##LMUL(__riscv_vand_vx_u32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(x), SIGN_MASK_F32, vl));\
\
    specMask = __riscv_vmfeq_vf_f32##LMUL##_b##BOOL(x, 0.0f, vl);\
    abs_y = __riscv_vreinterpret_v_i32##LMUL##_f32##LMUL(__riscv_vand_vx_i32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(y), ABS_MASK_F32, vl));\
    abs_y = __riscv_vreinterpret_v_i32##LMUL##_f32##LMUL(__riscv_vxor_vv_i32##LMUL##_mu(__riscv_vmsge_vx_i32##LMUL##_b##BOOL(__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(abs_y), INT_MAX_P1_I32, vl),\
									__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(abs_y), \
									__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(abs_y), \
									__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(abs_y), \
									vl));\
    int_y = __riscv_vfcvt_rtz_x_f_v_i32##LMUL(abs_y, vl);\
\
    s_ans = __riscv_vand_vv_i32##LMUL(sx, __riscv_vsll_vx_i32##LMUL(__riscv_vand_vx_i32##LMUL(int_y, 1, vl), BITS_F32, vl), vl);\
\
    diff_y = __riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(__riscv_vfsub_vv_f32##LMUL(abs_y, __riscv_vfcvt_f_x_v_f32##LMUL(int_y, vl), vl));\
    uncorrect = __riscv_vmsne_vx_i32##LMUL##_b##BOOL(diff_y, 0, vl);\
\
    s_ans = __riscv_vxor_vv_i32##LMUL##_mu(__riscv_vmand_mm_b##BOOL(uncorrect, specMask, vl), s_ans, s_ans, s_ans, vl);\
    x = __riscv_vfmerge_vfm_f32##LMUL(x, 1.0f, specMask, vl);\
    specRes = __riscv_vfmv_v_f_f32##LMUL(0.0f, vl);\
}\
\
vfloat32##LMUL##_t __riscv_vpow_f32##LMUL(vfloat32##LMUL##_t x, vfloat32##LMUL##_t y, size_t vl) {\
  vfloat32##LMUL##_t logxh, logxl, specRes;\
  vfloat32##LMUL##_t l, h, r, res;\
  vbool##BOOL##_t specMask, small_exp;\
  vint32##LMUL##_t s_ans, r_sign;\
  vuint32##LMUL##_t specialExp;\
\
  check_pow_special_cases_f32##LMUL(x, y, specRes, s_ans, specMask, vl);\
  log22_f32##LMUL(x, logxh, logxl, vl);\
\
  r_sign = __riscv_vsra_vx_i32##LMUL(__riscv_vxor_vv_i32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(y), __riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(logxh), vl), BITS_F32, vl);\
\
  specialExp = __riscv_vor_vx_u32##LMUL(__riscv_vand_vx_u32##LMUL(__riscv_vreinterpret_v_i32##LMUL##_u32##LMUL(r_sign), UF_TRESHOLD_U32, vl), OF_TRESHOLD_U32, vl);\
\
  r = __riscv_vfmul_vv_f32##LMUL(y, logxh, vl);\
  small_exp = __riscv_vmsgeu_vv_u32##LMUL##_b##BOOL(__riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(r), specialExp, vl);\
  l = __riscv_vfmsub_vv_f32##LMUL(y, logxh, r, vl);\
  r = __riscv_vmerge_vvm_f32##LMUL(r, __riscv_vreinterpret_v_u32##LMUL##_f32##LMUL(specialExp), small_exp, vl);\
  l = __riscv_vfmadd_vv_f32##LMUL(y, logxl, l, vl);\
  l = __riscv_vreinterpret_v_i32##LMUL##_f32##LMUL(__riscv_vxor_vv_i32##LMUL##_mu(small_exp, __riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(l), __riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(l), __riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(l), vl));\
\
  res = special_exp_f32##LMUL(r, l, vl);\
\
  res = __riscv_vmerge_vvm_f32##LMUL(res, specRes, specMask, vl);\
  res = __riscv_vreinterpret_v_i32##LMUL##_f32##LMUL(__riscv_vor_vv_i32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(res), s_ans, vl));\
\
  return res;\
}

#endif

RISCV_POW_F64(m1, 64)
RISCV_POW_F64(m2, 32)
RISCV_POW_F64(m4, 16)
RISCV_POW_F64(m8, 8)
RISCV_POW_F32(m1, 32)
RISCV_POW_F32(m2, 16)
RISCV_POW_F32(m4, 8)
RISCV_POW_F32(m8, 4)


#ifdef __riscv_zvfh

#ifndef __FAST_MATH__
#define RISCV_POW_F16(LMUL, BOOL)   \
void check_pow_special_cases_f16##LMUL(vfloat16##LMUL##_t& x, vfloat16##LMUL##_t& y, vfloat16##LMUL##_t& specRes, vint16##LMUL##_t& s_ans, vbool##BOOL##_t& specMask, vbool##BOOL##_t& subnormalY, vbool##BOOL##_t& Ytozero, size_t vl) {\
    vfloat16##LMUL##_t abs_y;\
    vint16##LMUL##_t sx, int_y, diff_y;\
    vuint16##LMUL##_t specResU;\
    vbool##BOOL##_t zero_x, inf_x, inf_y, spec_x, spec_y, uncorrect, triv_ans, mask_nan, y_lt0, x_nem1;\
    sx = __riscv_vreinterpret_v_u16##LMUL##_i16##LMUL(__riscv_vand_vx_u16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(x), SIGN_MASK_F16, vl));\
\
    x_nem1 = __riscv_vmfne_vf_f16##LMUL##_b##BOOL(x, -1.0f16, vl);\
    x = __riscv_vreinterpret_v_i16##LMUL##_f16##LMUL(__riscv_vand_vx_i16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(x), ABS_MASK_F16, vl));\
    triv_ans =  __riscv_vmor_mm_b##BOOL(__riscv_vmfeq_vf_f16##LMUL##_b##BOOL(x, 1.0f16, vl), __riscv_vmfeq_vf_f16##LMUL##_b##BOOL(y, 0.0f16, vl), vl);\
\
    zero_x = __riscv_vmfeq_vf_f16##LMUL##_b##BOOL(x, 0.0f16, vl);\
    inf_x = __riscv_vmseq_vx_u16##LMUL##_b##BOOL(__riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(x), EXP_MASK_F16, vl);\
    spec_y = __riscv_vmsgtu_vx_u16##LMUL##_b##BOOL(__riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(x), ONE_MASK_F16, vl);\
    spec_x = __riscv_vmor_mm_b##BOOL(zero_x, inf_x, vl);\
\
    abs_y = __riscv_vreinterpret_v_i16##LMUL##_f16##LMUL(__riscv_vand_vx_i16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(y), ABS_MASK_F16, vl));\
    inf_y = __riscv_vmseq_vx_u16##LMUL##_b##BOOL(__riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(abs_y), EXP_MASK_F16, vl);\
    subnormalY = __riscv_vmflt_vf_f16##LMUL##_b##BOOL(abs_y, 0x1p-9f16, vl);\
    Ytozero = __riscv_vmflt_vf_f16##LMUL##_b##BOOL(abs_y, 0x1p-16f16, vl);\
    abs_y = __riscv_vreinterpret_v_i16##LMUL##_f16##LMUL(__riscv_vxor_vv_i16##LMUL##_mu(__riscv_vmsge_vx_i16##LMUL##_b##BOOL(__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(abs_y), INT_MAX_P1_I16, vl),\
									__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(abs_y), \
									__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(abs_y), \
									__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(abs_y), \
									vl));\
    int_y = __riscv_vfcvt_rtz_x_f_v_i16##LMUL(abs_y, vl);\
\
    s_ans = __riscv_vand_vv_i16##LMUL(sx, __riscv_vsll_vx_i16##LMUL(__riscv_vand_vx_i16##LMUL(int_y, 1, vl), BITS_F16, vl), vl);\
\
    diff_y = __riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(__riscv_vfsub_vv_f16##LMUL(abs_y, __riscv_vfcvt_f_x_v_f16##LMUL(int_y, vl), vl));\
    uncorrect = __riscv_vmsne_vx_i16##LMUL##_b##BOOL(__riscv_vand_vv_i16##LMUL(diff_y, __riscv_vsra_vx_i16##LMUL(sx, BITS_F16, vl), vl), 0, vl);\
    s_ans = __riscv_vxor_vv_i16##LMUL##_mu(__riscv_vmand_mm_b##BOOL(uncorrect, spec_x, vl), s_ans, s_ans, s_ans, vl);\
    uncorrect = __riscv_vmandn_mm_b##BOOL(uncorrect, spec_x, vl);\
\
    if (__riscv_vcpop_m_b##BOOL(uncorrect, vl)) RVVMF_EXP_CALL_FE_INVALID();\
    mask_nan = __riscv_vmor_mm_b##BOOL(__riscv_vmandn_mm_b##BOOL(__riscv_vmor_mm_b##BOOL(__riscv_vmfne_vv_f16##LMUL##_b##BOOL(x, x, vl), __riscv_vmfne_vv_f16##LMUL##_b##BOOL(y, y, vl), vl), __riscv_vmand_mm_b##BOOL(triv_ans, x_nem1, vl), vl), uncorrect, vl);\
\
    y_lt0 = __riscv_vmsgtu_vx_u16##LMUL##_b##BOOL(__riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(y), SIGN_MASK_F16, vl);\
    spec_y = __riscv_vmxor_mm_b##BOOL(y_lt0, spec_y, vl);\
    if (__riscv_vcpop_m_b##BOOL(__riscv_vmand_mm_b##BOOL(y_lt0, zero_x, vl), vl)) RVVMF_EXP_CALL_FE_DIVBYZERO();\
    specMask = __riscv_vmor_mm_b##BOOL(__riscv_vmor_mm_b##BOOL(mask_nan, spec_x, vl), inf_y, vl);\
\
    specResU = __riscv_vmerge_vxm_u16##LMUL(__riscv_vmv_v_x_u16##LMUL(uint16_t(0), vl), EXP_MASK_F16, spec_y, vl);\
\
    specResU = __riscv_vmerge_vxm_u16##LMUL(specResU, ONE_MASK_F16, triv_ans, vl);\
    specRes =  __riscv_vreinterpret_v_u16##LMUL##_f16##LMUL(__riscv_vmerge_vxm_u16##LMUL(specResU, NAN_MASK_F16, mask_nan, vl));\
\
    triv_ans = __riscv_vmor_mm_b##BOOL(specMask, triv_ans, vl);\
\
    y = __riscv_vreinterpret_v_i16##LMUL##_f16##LMUL(__riscv_vxor_vv_i16##LMUL##_mu(triv_ans, __riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(y), __riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(y), __riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(y), vl));\
    x = __riscv_vfmerge_vfm_f16##LMUL(x, 1.0f16, triv_ans, vl);\
}\
\
vfloat16##LMUL##_t __riscv_vpow_f16##LMUL(vfloat16##LMUL##_t x, vfloat16##LMUL##_t y, size_t vl) {\
  vfloat16##LMUL##_t logxh, logxl, specRes;\
  vfloat16##LMUL##_t l, h, r, res;\
  vbool##BOOL##_t specMask, subnormalY, Ytozero, small_exp;\
  vint16##LMUL##_t s_ans, r_sign;\
  vuint16##LMUL##_t specialExp;\
  vuint16##LMUL##_t r_exp, exp_diff;\
  vbool##BOOL##_t exp_check;\
\
  r_exp = __riscv_vand_vx_u16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(y), EXP_MASK_F16, vl);\
\
  check_pow_special_cases_f16##LMUL(x, y, specRes, s_ans, specMask, subnormalY, Ytozero, vl);\
\
  log22_f16##LMUL(x, logxh, logxl, vl);\
\
  r_exp = __riscv_vadd_vv_u16##LMUL(r_exp, __riscv_vand_vx_u16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(logxh), EXP_MASK_F16, vl), vl);\
  r_sign = __riscv_vsra_vx_i16##LMUL(__riscv_vxor_vv_i16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(y), __riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(logxh), vl), BITS_F16, vl);\
\
  exp_check = __riscv_vmor_mm_b##BOOL(__riscv_vmsltu_vx_u16##LMUL##_b##BOOL(r_exp, UF_EXP_TRESHOLD_P10_F16, vl), Ytozero, vl);\
\
  specialExp = __riscv_vor_vx_u16##LMUL(__riscv_vand_vx_u16##LMUL(__riscv_vreinterpret_v_i16##LMUL##_u16##LMUL(r_sign), UF_TRESHOLD_P10_U16, vl), OF_TRESHOLD_P10_U16, vl);\
\
  specialExp = __riscv_vxor_vv_u16##LMUL##_mu(exp_check, specialExp, specialExp, specialExp, vl);\
  exp_check = __riscv_vmandn_mm_b##BOOL(__riscv_vmsltu_vx_u16##LMUL##_b##BOOL(r_exp, OF_EXP_TRESHOLD_P10_F16, vl), exp_check, vl);\
\
  r = __riscv_vfmul_vv_f16##LMUL##_mu(exp_check, __riscv_vfmv_v_f_f16##LMUL(0.0f16, vl), y, logxh, vl);\
  small_exp = __riscv_vmand_mm_b##BOOL(__riscv_vmsltu_vv_u16##LMUL##_b##BOOL(__riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(r), specialExp, vl), exp_check, vl);\
  l = __riscv_vfmsac_vv_f16##LMUL##_mu(exp_check, r, y, logxh, vl);\
  r = __riscv_vmerge_vvm_f16##LMUL(__riscv_vreinterpret_v_u16##LMUL##_f16##LMUL(specialExp), r, small_exp, vl);\
  logxl = __riscv_vfmul_vf_f16##LMUL(logxl, 0x1p6f16, vl);\
  l = __riscv_vfmul_vf_f16##LMUL(l, 0x1p10f16, vl);\
  l = __riscv_vfmacc_vv_f16##LMUL##_mu(__riscv_vmandn_mm_b##BOOL(small_exp, subnormalY, vl), l, y, logxl, vl);\
\
  r = __riscv_vfmul_vf_f16##LMUL(r, 0x1p-10f16, vl);\
  res = special_exp_f16##LMUL(r, l, vl);\
\
  res = __riscv_vmerge_vvm_f16##LMUL(res, specRes, specMask, vl);\
  res = __riscv_vreinterpret_v_i16##LMUL##_f16##LMUL(__riscv_vor_vv_i16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(res), s_ans, vl));\
\
  return __riscv_vfmadd_vf_f16##LMUL(res, 0x1p-14f16, res, vl);\
}

#else
#define RISCV_POW_F16(LMUL, BOOL)   \
void check_pow_special_cases_f16##LMUL(vfloat16##LMUL##_t& x, vfloat16##LMUL##_t& y, vfloat16##LMUL##_t& specRes, vint16##LMUL##_t& s_ans, vbool##BOOL##_t& specMask, size_t vl) {\
    vfloat16##LMUL##_t abs_y;\
    vint16##LMUL##_t sx, int_y, diff_y;\
    vuint16##LMUL##_t specResU;\
    vbool##BOOL##_t zero_x, inf_x, spec_x, uncorrect, triv_ans, mask_nan, y_lt0, x_nem1;\
    sx = __riscv_vreinterpret_v_u16##LMUL##_i16##LMUL(__riscv_vand_vx_u16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(x), SIGN_MASK_F16, vl));\
\
    specMask = __riscv_vmfeq_vf_f16##LMUL##_b##BOOL(x, 0.0f16, vl);\
    abs_y = __riscv_vreinterpret_v_i16##LMUL##_f16##LMUL(__riscv_vand_vx_i16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(y), ABS_MASK_F16, vl));\
    abs_y = __riscv_vreinterpret_v_i16##LMUL##_f16##LMUL(__riscv_vxor_vv_i16##LMUL##_mu(__riscv_vmsge_vx_i16##LMUL##_b##BOOL(__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(abs_y), INT_MAX_P1_I16, vl),\
									__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(abs_y), \
									__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(abs_y), \
									__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(abs_y), \
									vl));\
    int_y = __riscv_vfcvt_rtz_x_f_v_i16##LMUL(abs_y, vl);\
\
    s_ans = __riscv_vand_vv_i16##LMUL(sx, __riscv_vsll_vx_i16##LMUL(__riscv_vand_vx_i16##LMUL(int_y, 1, vl), BITS_F16, vl), vl);\
\
    diff_y = __riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(__riscv_vfsub_vv_f16##LMUL(abs_y, __riscv_vfcvt_f_x_v_f16##LMUL(int_y, vl), vl));\
    uncorrect = __riscv_vmsne_vx_i16##LMUL##_b##BOOL(diff_y, 0, vl);\
\
    s_ans = __riscv_vxor_vv_i16##LMUL##_mu(__riscv_vmand_mm_b##BOOL(uncorrect, specMask, vl), s_ans, s_ans, s_ans, vl);\
    x = __riscv_vfmerge_vfm_f16##LMUL(x, 1.0f16, specMask, vl);\
    specRes = __riscv_vfmv_v_f_f16##LMUL(0.0f16, vl);\
}\
\
vfloat16##LMUL##_t __riscv_vpow_f16##LMUL(vfloat16##LMUL##_t x, vfloat16##LMUL##_t y, size_t vl) {\
  vfloat16##LMUL##_t logxh, logxl, specRes;\
  vfloat16##LMUL##_t l, h, r, res;\
  vbool##BOOL##_t specMask, small_exp;\
  vint16##LMUL##_t s_ans, r_sign;\
  vuint16##LMUL##_t specialExp;\
\
  check_pow_special_cases_f16##LMUL(x, y, specRes, s_ans, specMask, vl);\
\
  log22_f16##LMUL(x, logxh, logxl, vl);\
  r_sign = __riscv_vsra_vx_i16##LMUL(__riscv_vxor_vv_i16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(y), __riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(logxh), vl), BITS_F16, vl);\
\
  specialExp = __riscv_vor_vx_u16##LMUL(__riscv_vand_vx_u16##LMUL(__riscv_vreinterpret_v_i16##LMUL##_u16##LMUL(r_sign), UF_TRESHOLD_U16, vl), OF_TRESHOLD_U16, vl);\
\
  r = __riscv_vfmul_vv_f16##LMUL(y, logxh, vl);\
  small_exp = __riscv_vmsltu_vv_u16##LMUL##_b##BOOL(__riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(r), specialExp, vl);\
  l = __riscv_vfmsub_vv_f16##LMUL(y, logxh, r, vl);\
  r = __riscv_vmerge_vvm_f16##LMUL(__riscv_vreinterpret_v_u16##LMUL##_f16##LMUL(specialExp), r, small_exp, vl);\
  l = __riscv_vfmacc_vv_f16##LMUL##_mu(small_exp,\
				 __riscv_vfmul_vf_f16##LMUL##_mu(small_exp, __riscv_vfmv_v_f_f16##LMUL(0.0f16, vl), l, 0x1p10f16, vl),\
				 y,\
				 logxl,\
				 vl);\
\
  res = special_exp_f16##LMUL(r, l, vl);\
\
  res = __riscv_vmerge_vvm_f16##LMUL(res, specRes, specMask, vl);\
  res = __riscv_vreinterpret_v_i16##LMUL##_f16##LMUL(__riscv_vor_vv_i16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(res), s_ans, vl));\
\
  return res;\
}

#endif

RISCV_POW_F16(m1, 16)
RISCV_POW_F16(m2, 8)
RISCV_POW_F16(m4, 4)
RISCV_POW_F16(m8, 2)

#endif