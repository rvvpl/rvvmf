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
 *   File:  sinh_cosh_double.inl                         *
 *   Contains: helper built-in functions for sinh, cosh  *
 *             functions (float64_t)                     *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#include "sinh_cosh_utilities.inl"

const double ZERO_F64 = 0.0;
const double ONE_F64 = 1.0;
const double OVERFLOWTHRESHOLD_F64 = 0x1.633ce8fb9f87ep9;
const double MAGIC_CONST_1_F64 = 6755399441055744.0;    // =1.5*2^52
const double INV_LOG_22_K_F64 = 0x1.71547652b82fep6;    // =1 / (log(2)*2^(-k))
const double M_LOG22_K_H_F64 = -0x1.62e42fefap-7;       // =-log(2)*2^(-k) (high bits)
const double M_LOG22_K_L_F64 = -0x1.cf79abc9e3b3ap-46;  // =-log(2)*2^(-k) (low bits)
const size_t TABLE_SIZE_DEG_F64 = 6;
const size_t TABLE_SIZE_SMALL_DEG_F64 = 3;
const uint64_t MASK_FI_SMALL_BIT_F64 = 0x0000000000000007;
// const double POL_COEFF_1_F64 = 0x1p-1;
// const double POL_COEFF_2_F64 = 0x1.55555555548bap-3;
// const double POL_COEFF_3_F64 = 0x1.5555555555abcp-5;
// const double POL_COEFF_4_F64 = 0x1.111123cf189c3p-7;
// const double POL_COEFF_5_F64 = 0x1.6c16c6a1679b3p-10;
const double POL_COEFF_1_F64 = 0x1p510;
const double POL_COEFF_2_F64 = 0x1.55555555548bap508;
const double POL_COEFF_3_F64 = 0x1.5555555555abcp506;
const double POL_COEFF_4_F64 = 0x1.111123cf189c3p504;
const double POL_COEFF_5_F64 = 0x1.6c16c6a1679b3p501;
const uint64_t MASK_ABS_F64 = 0x7fffffffffffffff;
const uint64_t MASK_SIGN_F64 = 0x8000000000000000;
const uint64_t MASK_FI_BIT_F64 = 0x000000000000003f;  // =(uint64_t(1) << k) - uint64_t(1)
const uint64_t MASK_HI_BIT_F64 = 0x000000000003ffff;  // =(uint64_t(1) << (k + 11 + 1)) - uint64_t(1);

const double FACTOR_F64 =  0x1p511;
const uint64_t FACTOR_DEG_F64 = 512; // 1023 - 511
const double ADD_COEFF_F64 = 0x1p-86;
const uint64_t SHIFT_CONST_F64 = 0x0000000000000dcc;
const uint64_t MAGIC_DIFF_F64 = 338;
const double MUL_COEFF_F64 = 0x1p-2;
const double MASK_NUMBER_1_F64 = 0x1p-846;
const double MASK_NUMBER_2_F64 = 0x1p-1022;

static const double LOOKUP_TABLE_1_H_F64[8] = {0x1p0,
                                               0x1.172b83c7d517bp0,
                                               0x1.306fe0a31b715p0,
                                               0x1.4bfdad5362a27p0,
                                               0x1.6a09e667f3bcdp0,
                                               0x1.8ace5422aa0dbp0,
                                               0x1.ae89f995ad3adp0,
                                               0x1.d5818dcfba487p0};
static const double LOOKUP_TABLE_1_L_F64[8] = {0,
                                               -0x1.19041b9d78a76p-55,
                                               0x1.6f46ad23182e4p-55,
                                               0x1.d4397afec42e2p-56,
                                               -0x1.bdd3413b26456p-54,
                                               0x1.6e9f156864b27p-54,
                                               0x1.7a1cd345dcc81p-54,
                                               0x1.2ed02d75b3707p-55};
static const double LOOKUP_TABLE_2_H_F64[8] = {0x1p0,
                                               0x1.02c9a3e778061p0,
                                               0x1.059b0d3158574p0,
                                               0x1.0874518759bc8p0,
                                               0x1.0b5586cf9890fp0,
                                               0x1.0e3ec32d3d1a2p0,
                                               0x1.11301d0125b51p0,
                                               0x1.1429aaea92dep0};
static const double LOOKUP_TABLE_2_L_F64[8] = {0,
                                               -0x1.19083535b085dp-56,
                                               0x1.d73e2a475b465p-55,
                                               0x1.186be4bb284ffp-57,
                                               0x1.8a62e4adc610bp-54,
                                               0x1.03a1727c57b53p-59,
                                               -0x1.6c51039449b3ap-54,
                                               -0x1.32fbf9af1369ep-54};

// ---------------------------- m1 ----------------------------

forceinline void check_special_cases_f64m1(const vfloat64m1_t& x, vfloat64m1_t& special, vbool64_t& specialMask, size_t vl) {
  // check +inf
  const uint64_t pinf = 0x7ff0000000000000;
  specialMask = __riscv_vmfeq_vf_f64m1_b64(x, RVVMF_SC_AS_FP64(pinf), vl);
  special = __riscv_vfmerge_vfm_f64m1(x, RVVMF_SC_AS_FP64(pinf), specialMask, vl);

  // check overflow
  // overflowThreshold < x < pinf
  vbool64_t mask = __riscv_vmand_mm_b64(__riscv_vmfgt_vf_f64m1_b64(x, OVERFLOWTHRESHOLD_F64, vl),
                              __riscv_vmflt_vf_f64m1_b64(x, RVVMF_SC_AS_FP64(pinf), vl), vl);
  special = __riscv_vfmerge_vfm_f64m1(special, RVVMF_SC_AS_FP64(pinf), mask, vl);
  specialMask = __riscv_vmor_mm_b64(specialMask, mask, vl);
  if (__riscv_vcpop_m_b64(mask, vl)) RVVMF_SC_CALL_FE_OVERFLOW();

  const uint64_t qnan = 0x7ff8000000000000;
  // check qNaN (pinf < x)
  mask = __riscv_vmsgtu_vx_u64m1_b64(RVVMF_F64_TO_U64(x, m1), pinf, vl);
  special = __riscv_vfmerge_vfm_f64m1(special, RVVMF_SC_AS_FP64(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b64(specialMask, mask, vl);

  // check sNaN (pinf < x < qnan)
  mask = __riscv_vmand_mm_b64(mask,
                              __riscv_vmsltu_vx_u64m1_b64(RVVMF_F64_TO_U64(x, m1),
                              qnan, vl), vl);
  special = __riscv_vfmerge_vfm_f64m1(special, RVVMF_SC_AS_FP64(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b64(specialMask, mask, vl);
  if (__riscv_vcpop_m_b64(mask, vl)) RVVMF_SC_CALL_FE_INVALID();
}

forceinline void argument_reduction_f64m1(const vfloat64m1_t& x, vuint64m1_t& ei, vuint64m1_t& fi, vfloat64m1_t& h, size_t vl) {
  vfloat64m1_t vmagicConst1 = __riscv_vfmv_v_f_f64m1(MAGIC_CONST_1_F64, vl);
  h = __riscv_vfmadd_vf_f64m1(x, INV_LOG_22_K_F64, vmagicConst1, vl);
  fi = __riscv_vand_vx_u64m1(RVVMF_F64_TO_U64(h, m1), MASK_HI_BIT_F64, vl);
  ei = __riscv_vsrl_vx_u64m1(fi, TABLE_SIZE_DEG_F64, vl);
  fi = __riscv_vand_vx_u64m1(fi, MASK_FI_BIT_F64, vl);
  h = __riscv_vfsub_vv_f64m1(h, vmagicConst1, vl);
}

forceinline void argument_reduction_h_f64m1(const vfloat64m1_t& x, vfloat64m1_t& y, vuint64m1_t& ei, vuint64m1_t& fi, size_t vl) {
  vfloat64m1_t h;
  argument_reduction_f64m1(x, ei, fi, h, vl);
  y = __riscv_vfmadd_vf_f64m1(h, M_LOG22_K_L_F64, __riscv_vfmadd_vf_f64m1(h, M_LOG22_K_H_F64, x, vl), vl);
}

forceinline void get_table_values_f64m1(const vuint64m1_t index, vfloat64m1_t& th, vfloat64m1_t& tl, size_t vl){
  vuint64m1_t index1 = __riscv_vsrl_vx_u64m1(index, TABLE_SIZE_SMALL_DEG_F64, vl);
  vuint64m1_t index2 = __riscv_vand_vx_u64m1(index, MASK_FI_SMALL_BIT_F64, vl);
  index1 = __riscv_vmul_vx_u64m1(index1, uint64_t(sizeof(double)), vl);
  index2 = __riscv_vmul_vx_u64m1(index2, uint64_t(sizeof(double)), vl);

  vfloat64m1_t t1h = __riscv_vloxei64_v_f64m1(LOOKUP_TABLE_1_H_F64, index1, vl);
  vfloat64m1_t t1l = __riscv_vloxei64_v_f64m1(LOOKUP_TABLE_1_L_F64, index1, vl);
  vfloat64m1_t t2h = __riscv_vloxei64_v_f64m1(LOOKUP_TABLE_2_H_F64, index2, vl);
  vfloat64m1_t t2l = __riscv_vloxei64_v_f64m1(LOOKUP_TABLE_2_L_F64, index2, vl);
  mul22_vv_f64m1(t1h, t1l, t2h, t2l, th, tl, vl);
}

forceinline void calculate_polynom_f64m1(const vfloat64m1_t& sqry, vfloat64m1_t& odd, vfloat64m1_t& even, size_t vl){
  odd = __riscv_vfmadd_vf_f64m1(sqry, POL_COEFF_4_F64, __riscv_vfmv_v_f_f64m1(POL_COEFF_2_F64, vl), vl);
  odd = __riscv_vfmul_vv_f64m1(odd, sqry, vl);

  even = __riscv_vfmadd_vf_f64m1(sqry, POL_COEFF_5_F64, __riscv_vfmv_v_f_f64m1(POL_COEFF_3_F64, vl), vl);
  even = __riscv_vfmadd_vv_f64m1(sqry, even, __riscv_vfmv_v_f_f64m1(POL_COEFF_1_F64, vl), vl);
}

forceinline void calculate_polynom_sinh_f64m1(const vfloat64m1_t& y, vfloat64m1_t& odd, vfloat64m1_t& even, vfloat64m1_t& oddtail, vfloat64m1_t& eventail, size_t vl) {
  vfloat64m1_t yc = __riscv_vfadd_vf_f64m1(y, ADD_COEFF_F64, vl);
  vfloat64m1_t sqry = __riscv_vfmul_vv_f64m1(yc, yc, vl);
  calculate_polynom_f64m1(sqry, odd, even, vl);

  fma12_vv_f64m1(y, odd, __riscv_vfmul_vf_f64m1(y, FACTOR_F64, vl), odd, oddtail, vl);
  mul12_vv_f64m1(even, sqry, even, eventail, vl);
}

forceinline void calculate_polynom_cosh_f64m1(const vfloat64m1_t& y, vfloat64m1_t& odd, vfloat64m1_t& even, size_t vl) {
  vfloat64m1_t yc = __riscv_vfadd_vf_f64m1(y, ADD_COEFF_F64, vl);
  vfloat64m1_t sqry = __riscv_vfmul_vv_f64m1(yc, yc, vl);
  calculate_polynom_f64m1(sqry, odd, even, vl);

  odd = __riscv_vfmadd_vv_f64m1(y, odd, __riscv_vfmul_vf_f64m1(y, FACTOR_F64, vl), vl);
  even = __riscv_vfmul_vv_f64m1(even, sqry, vl);
}

forceinline void update_exponent_f64m1(const vuint64m1_t& ei, vfloat64m1_t& res, size_t vl){
  res = __riscv_vfmul_vv_f64m1(res, RVVMF_U64_TO_F64(__riscv_vsll_vx_u64m1(ei, (size_t)52, vl), m1), vl);
}

forceinline void get_sign_f64m1(vfloat64m1_t& x, vuint64m1_t& sign, size_t vl){
  sign = __riscv_vand_vx_u64m1(RVVMF_F64_TO_U64(x, m1), MASK_SIGN_F64, vl);
  x = RVVMF_U64_TO_F64(__riscv_vand_vx_u64m1(RVVMF_F64_TO_U64(x, m1), MASK_ABS_F64, vl), m1);
}

forceinline void set_sign_f64m1(vfloat64m1_t& res, const vuint64m1_t& sign, size_t vl){
  res = RVVMF_U64_TO_F64(__riscv_vor_vv_u64m1(RVVMF_F64_TO_U64(res, m1), sign, vl), m1);
}

forceinline void get_abs_f64m1(vfloat64m1_t& x, size_t vl){
  x = RVVMF_U64_TO_F64(__riscv_vand_vx_u64m1(RVVMF_F64_TO_U64(x, m1), MASK_ABS_F64, vl), m1);
}

// ---------------------------- m2 ----------------------------

forceinline void check_special_cases_f64m2(const vfloat64m2_t& x, vfloat64m2_t& special, vbool32_t& specialMask, size_t vl) {
  // check +inf
  const uint64_t pinf = 0x7ff0000000000000;
  specialMask = __riscv_vmfeq_vf_f64m2_b32(x, RVVMF_SC_AS_FP64(pinf), vl);
  special = __riscv_vfmerge_vfm_f64m2(x, RVVMF_SC_AS_FP64(pinf), specialMask, vl);

  // check overflow
  // overflowThreshold < x < pinf
  vbool32_t mask = __riscv_vmand_mm_b32(__riscv_vmfgt_vf_f64m2_b32(x, RVVMF_SC_AS_FP64(OVERFLOWTHRESHOLD_F64), vl),
                              __riscv_vmflt_vf_f64m2_b32(x, RVVMF_SC_AS_FP64(pinf), vl), vl);
  special = __riscv_vfmerge_vfm_f64m2(special, RVVMF_SC_AS_FP64(pinf), mask, vl);
  specialMask = __riscv_vmor_mm_b32(specialMask, mask, vl);
  if (__riscv_vcpop_m_b32(mask, vl)) RVVMF_SC_CALL_FE_OVERFLOW();

  const uint64_t qnan = 0x7ff8000000000000;
  // check qNaN (pinf < x)
  mask = __riscv_vmsgtu_vx_u64m2_b32(RVVMF_F64_TO_U64(x, m2), pinf, vl);
  special = __riscv_vfmerge_vfm_f64m2(special, RVVMF_SC_AS_FP64(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b32(specialMask, mask, vl);
  // check sNaN (pinf < x < qnan)
  mask = __riscv_vmand_mm_b32(mask,
                              __riscv_vmsltu_vx_u64m2_b32(RVVMF_F64_TO_U64(x, m2),
                              qnan, vl), vl);
  special = __riscv_vfmerge_vfm_f64m2(special, RVVMF_SC_AS_FP64(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b32(specialMask, mask, vl);
  if (__riscv_vcpop_m_b32(mask, vl)) RVVMF_SC_CALL_FE_INVALID();
}

forceinline void argument_reduction_f64m2(const vfloat64m2_t& x, vuint64m2_t& ei, vuint64m2_t& fi, vfloat64m2_t& h, size_t vl) {
  vfloat64m2_t vmagicConst1 = __riscv_vfmv_v_f_f64m2(MAGIC_CONST_1_F64, vl);
  h = __riscv_vfmadd_vf_f64m2(x, INV_LOG_22_K_F64, vmagicConst1, vl);
  fi = __riscv_vand_vx_u64m2(RVVMF_F64_TO_U64(h, m2), MASK_HI_BIT_F64, vl);
  ei = __riscv_vsrl_vx_u64m2(fi, TABLE_SIZE_DEG_F64, vl);
  fi = __riscv_vand_vx_u64m2(fi, MASK_FI_BIT_F64, vl);
  h = __riscv_vfsub_vv_f64m2(h, vmagicConst1, vl);
}

forceinline void argument_reduction_h_f64m2(const vfloat64m2_t& x, vfloat64m2_t& y, vuint64m2_t& ei, vuint64m2_t& fi, size_t vl) {
  vfloat64m2_t h;
  argument_reduction_f64m2(x, ei, fi, h, vl);
  y = __riscv_vfmadd_vf_f64m2(h, M_LOG22_K_L_F64, __riscv_vfmadd_vf_f64m2(h, M_LOG22_K_H_F64, x, vl), vl);
}

forceinline void get_table_values_f64m2(const vuint64m2_t index, vfloat64m2_t& th, vfloat64m2_t& tl, size_t vl){
  vuint64m2_t index1 = __riscv_vsrl_vx_u64m2(index, TABLE_SIZE_SMALL_DEG_F64, vl);
  vuint64m2_t index2 = __riscv_vand_vx_u64m2(index, MASK_FI_SMALL_BIT_F64, vl);
  index1 = __riscv_vmul_vx_u64m2(index1, uint64_t(sizeof(double)), vl);
  index2 = __riscv_vmul_vx_u64m2(index2, uint64_t(sizeof(double)), vl);

  vfloat64m2_t t1h = __riscv_vloxei64_v_f64m2(LOOKUP_TABLE_1_H_F64, index1, vl);
  vfloat64m2_t t1l = __riscv_vloxei64_v_f64m2(LOOKUP_TABLE_1_L_F64, index1, vl);
  vfloat64m2_t t2h = __riscv_vloxei64_v_f64m2(LOOKUP_TABLE_2_H_F64, index2, vl);
  vfloat64m2_t t2l = __riscv_vloxei64_v_f64m2(LOOKUP_TABLE_2_L_F64, index2, vl);
  mul22_vv_f64m2(t1h, t1l, t2h, t2l, th, tl, vl);
}

forceinline void calculate_polynom_f64m2(const vfloat64m2_t& sqry, vfloat64m2_t& odd, vfloat64m2_t& even, size_t vl){
  odd = __riscv_vfmadd_vf_f64m2(sqry, POL_COEFF_4_F64, __riscv_vfmv_v_f_f64m2(POL_COEFF_2_F64, vl), vl);
  odd = __riscv_vfmul_vv_f64m2(odd, sqry, vl);

  even = __riscv_vfmadd_vf_f64m2(sqry, POL_COEFF_5_F64, __riscv_vfmv_v_f_f64m2(POL_COEFF_3_F64, vl), vl);
  even = __riscv_vfmadd_vv_f64m2(sqry, even, __riscv_vfmv_v_f_f64m2(POL_COEFF_1_F64, vl), vl);
}

forceinline void calculate_polynom_sinh_f64m2(const vfloat64m2_t& y, vfloat64m2_t& odd, vfloat64m2_t& even, vfloat64m2_t& oddtail, vfloat64m2_t& eventail, size_t vl) {
  vfloat64m2_t yc = __riscv_vfadd_vf_f64m2(y, ADD_COEFF_F64, vl);
  vfloat64m2_t sqry = __riscv_vfmul_vv_f64m2(yc, yc, vl);
  calculate_polynom_f64m2(sqry, odd, even, vl);

  fma12_vv_f64m2(y, odd, __riscv_vfmul_vf_f64m2(y, FACTOR_F64, vl), odd, oddtail, vl);
  mul12_vv_f64m2(even, sqry, even, eventail, vl);
}

forceinline void calculate_polynom_cosh_f64m2(const vfloat64m2_t& y, vfloat64m2_t& odd, vfloat64m2_t& even, size_t vl) {
  vfloat64m2_t yc = __riscv_vfadd_vf_f64m2(y, ADD_COEFF_F64, vl);
  vfloat64m2_t sqry = __riscv_vfmul_vv_f64m2(yc, yc, vl);
  calculate_polynom_f64m2(sqry, odd, even, vl);

  odd = __riscv_vfmadd_vv_f64m2(y, odd, __riscv_vfmul_vf_f64m2(y, FACTOR_F64, vl), vl);
  even = __riscv_vfmul_vv_f64m2(even, sqry, vl);
}

forceinline void update_exponent_f64m2(const vuint64m2_t& ei, vfloat64m2_t& res, size_t vl){
  res = __riscv_vfmul_vv_f64m2(res, RVVMF_U64_TO_F64(__riscv_vsll_vx_u64m2(ei, (size_t)52, vl), m2), vl);
}

forceinline void get_sign_f64m2(vfloat64m2_t& x, vuint64m2_t& sign, size_t vl){
  sign = __riscv_vand_vx_u64m2(RVVMF_F64_TO_U64(x, m2), MASK_SIGN_F64, vl);
  x = RVVMF_U64_TO_F64(__riscv_vand_vx_u64m2(RVVMF_F64_TO_U64(x, m2), MASK_ABS_F64, vl), m2);
}

forceinline void set_sign_f64m2(vfloat64m2_t& res, const vuint64m2_t& sign, size_t vl){
  res = RVVMF_U64_TO_F64(__riscv_vor_vv_u64m2(RVVMF_F64_TO_U64(res, m2), sign, vl), m2);
}

forceinline void get_abs_f64m2(vfloat64m2_t& x, size_t vl){
  x = RVVMF_U64_TO_F64(__riscv_vand_vx_u64m2(RVVMF_F64_TO_U64(x, m2), MASK_ABS_F64, vl), m2);
}

// ---------------------------- m4 ----------------------------

forceinline void check_special_cases_f64m4(const vfloat64m4_t& x, vfloat64m4_t& special, vbool16_t& specialMask, size_t vl) {
  // check +inf
  const uint64_t pinf = 0x7ff0000000000000;
  specialMask = __riscv_vmfeq_vf_f64m4_b16(x, RVVMF_SC_AS_FP64(pinf), vl);
  special = __riscv_vfmerge_vfm_f64m4(x, RVVMF_SC_AS_FP64(pinf), specialMask, vl);

  // check overflow
  // overflowThreshold < x < pinf
  vbool16_t mask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f64m4_b16(x, RVVMF_SC_AS_FP64(OVERFLOWTHRESHOLD_F64), vl),
                              __riscv_vmflt_vf_f64m4_b16(x, RVVMF_SC_AS_FP64(pinf), vl), vl);
  special = __riscv_vfmerge_vfm_f64m4(special, RVVMF_SC_AS_FP64(pinf), mask, vl);
  specialMask = __riscv_vmor_mm_b16(specialMask, mask, vl);
  if (__riscv_vcpop_m_b16(mask, vl)) RVVMF_SC_CALL_FE_OVERFLOW();

  const uint64_t qnan = 0x7ff8000000000000;
  // check qNaN (pinf < x)
  mask = __riscv_vmsgtu_vx_u64m4_b16(RVVMF_F64_TO_U64(x, m4), pinf, vl);
  special = __riscv_vfmerge_vfm_f64m4(special, RVVMF_SC_AS_FP64(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b16(specialMask, mask, vl);
  // check sNaN (pinf < x < qnan)
  mask = __riscv_vmand_mm_b16(mask,
                              __riscv_vmsltu_vx_u64m4_b16(RVVMF_F64_TO_U64(x, m4),
                              qnan, vl), vl);
  special = __riscv_vfmerge_vfm_f64m4(special, RVVMF_SC_AS_FP64(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b16(specialMask, mask, vl);
  if (__riscv_vcpop_m_b16(mask, vl)) RVVMF_SC_CALL_FE_INVALID();
}

forceinline void argument_reduction_f64m4(const vfloat64m4_t& x, vuint64m4_t& ei, vuint64m4_t& fi, vfloat64m4_t& h, size_t vl) {
  vfloat64m4_t vmagicConst1 = __riscv_vfmv_v_f_f64m4(MAGIC_CONST_1_F64, vl);
  h = __riscv_vfmadd_vf_f64m4(x, INV_LOG_22_K_F64, vmagicConst1, vl);
  fi = __riscv_vand_vx_u64m4(RVVMF_F64_TO_U64(h, m4), MASK_HI_BIT_F64, vl);
  ei = __riscv_vsrl_vx_u64m4(fi, TABLE_SIZE_DEG_F64, vl);
  fi = __riscv_vand_vx_u64m4(fi, MASK_FI_BIT_F64, vl);
  h = __riscv_vfsub_vv_f64m4(h, vmagicConst1, vl);
}

forceinline void argument_reduction_h_f64m4(const vfloat64m4_t& x, vfloat64m4_t& y, vuint64m4_t& ei, vuint64m4_t& fi, size_t vl) {
  vfloat64m4_t h;
  argument_reduction_f64m4(x, ei, fi, h, vl);
  y = __riscv_vfmadd_vf_f64m4(h, M_LOG22_K_L_F64, __riscv_vfmadd_vf_f64m4(h, M_LOG22_K_H_F64, x, vl), vl);
}

forceinline void get_table_values_f64m4(const vuint64m4_t index, vfloat64m4_t& th, vfloat64m4_t& tl, size_t vl){
  vuint64m4_t index1 = __riscv_vsrl_vx_u64m4(index, TABLE_SIZE_SMALL_DEG_F64, vl);
  vuint64m4_t index2 = __riscv_vand_vx_u64m4(index, MASK_FI_SMALL_BIT_F64, vl);
  index1 = __riscv_vmul_vx_u64m4(index1, uint64_t(sizeof(double)), vl);
  index2 = __riscv_vmul_vx_u64m4(index2, uint64_t(sizeof(double)), vl);

  vfloat64m4_t t1h = __riscv_vloxei64_v_f64m4(LOOKUP_TABLE_1_H_F64, index1, vl);
  vfloat64m4_t t1l = __riscv_vloxei64_v_f64m4(LOOKUP_TABLE_1_L_F64, index1, vl);
  vfloat64m4_t t2h = __riscv_vloxei64_v_f64m4(LOOKUP_TABLE_2_H_F64, index2, vl);
  vfloat64m4_t t2l = __riscv_vloxei64_v_f64m4(LOOKUP_TABLE_2_L_F64, index2, vl);
  mul22_vv_f64m4(t1h, t1l, t2h, t2l, th, tl, vl);
}

forceinline void calculate_polynom_f64m4(const vfloat64m4_t& sqry, vfloat64m4_t& odd, vfloat64m4_t& even, size_t vl){
  odd = __riscv_vfmadd_vf_f64m4(sqry, POL_COEFF_4_F64, __riscv_vfmv_v_f_f64m4(POL_COEFF_2_F64, vl), vl);
  odd = __riscv_vfmul_vv_f64m4(odd, sqry, vl);

  even = __riscv_vfmadd_vf_f64m4(sqry, POL_COEFF_5_F64, __riscv_vfmv_v_f_f64m4(POL_COEFF_3_F64, vl), vl);
  even = __riscv_vfmadd_vv_f64m4(sqry, even, __riscv_vfmv_v_f_f64m4(POL_COEFF_1_F64, vl), vl);
}

forceinline void calculate_polynom_sinh_f64m4(const vfloat64m4_t& y, vfloat64m4_t& odd, vfloat64m4_t& even, vfloat64m4_t& oddtail, vfloat64m4_t& eventail, size_t vl) {
  vfloat64m4_t yc = __riscv_vfadd_vf_f64m4(y, ADD_COEFF_F64, vl);
  vfloat64m4_t sqry = __riscv_vfmul_vv_f64m4(yc, yc, vl);
  calculate_polynom_f64m4(sqry, odd, even, vl);

  fma12_vv_f64m4(y, odd, __riscv_vfmul_vf_f64m4(y, FACTOR_F64, vl), odd, oddtail, vl);
  mul12_vv_f64m4(even, sqry, even, eventail, vl);
}

forceinline void calculate_polynom_cosh_f64m4(const vfloat64m4_t& y, vfloat64m4_t& odd, vfloat64m4_t& even, size_t vl) {
  vfloat64m4_t yc = __riscv_vfadd_vf_f64m4(y, ADD_COEFF_F64, vl);
  vfloat64m4_t sqry = __riscv_vfmul_vv_f64m4(yc, yc, vl);
  calculate_polynom_f64m4(sqry, odd, even, vl);

  odd = __riscv_vfmadd_vv_f64m4(y, odd, __riscv_vfmul_vf_f64m4(y, FACTOR_F64, vl), vl);
  even = __riscv_vfmul_vv_f64m4(even, sqry, vl);
}

forceinline void update_exponent_f64m4(const vuint64m4_t& ei, vfloat64m4_t& res, size_t vl){
  res = __riscv_vfmul_vv_f64m4(res, RVVMF_U64_TO_F64(__riscv_vsll_vx_u64m4(ei, (size_t)52, vl), m4), vl);
}

forceinline void get_sign_f64m4(vfloat64m4_t& x, vuint64m4_t& sign, size_t vl){
  sign = __riscv_vand_vx_u64m4(RVVMF_F64_TO_U64(x, m4), MASK_SIGN_F64, vl);
  x = RVVMF_U64_TO_F64(__riscv_vand_vx_u64m4(RVVMF_F64_TO_U64(x, m4), MASK_ABS_F64, vl), m4);
}

forceinline void set_sign_f64m4(vfloat64m4_t& res, const vuint64m4_t& sign, size_t vl){
  res = RVVMF_U64_TO_F64(__riscv_vor_vv_u64m4(RVVMF_F64_TO_U64(res, m4), sign, vl), m4);
}

forceinline void get_abs_f64m4(vfloat64m4_t& x, size_t vl){
  x = RVVMF_U64_TO_F64(__riscv_vand_vx_u64m4(RVVMF_F64_TO_U64(x, m4), MASK_ABS_F64, vl), m4);
}
