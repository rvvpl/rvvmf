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
 *   File:  sinh_cosh_half.inl                           *
 *   Contains: helper built-in functions for sinh, cosh  *
 *             functions (float16_t)                     *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#include "sinh_cosh_utilities.inl"

#if defined(__riscv_zfh) || defined(__riscv_zvfh)

const _Float16 ZERO_F16 = 0.0f16;
const _Float16 ONE_F16 = 1.0f16;
const _Float16 TWO_F16 = 2.0f16;
const _Float16 COEFF_F16 = 0x1p-4f16;
const _Float16 OVERFLOWTHRESHOLD_F16 = 0x1.79p3f16;
const _Float16 MAGIC_CONST_1_F16 = 1536.0f16;    // =1.5*2^10
const _Float16 INV_LOG_22_K_F16 = 0x1.714p3f16;    // =1 / (log(2)*2^(-k))
const _Float16 M_LOG22_K_H_F16 = -0x1.6p-4f16;      // =-log(2)*2^(-k) (high bits)
const _Float16 M_LOG22_K_L_F16 = -0x1.72p-11f16;  // =-log(2)*2^(-k) (middle bits)
const _Float16 M_LOG22_K_LL_F16 = -0x1.8p-23f16;    // =-log(2)*2^(-k) (low bits)
const size_t TABLE_SIZE_DEG_F16 = 3;
const _Float16 POL_COEFF_1_F16 = 0x1p-1f16;
const _Float16 POL_COEFF_2_F16 = 0x1.554p-3f16;
const _Float16 POL_COEFF_3_F16 = 0x1.548p-5f16;
const uint16_t MASK_ABS_F16 = 0x7fff;
const uint16_t MASK_SIGN_F16 = 0x8000;
const uint16_t MASK_FI_BIT_F16 = 0x0007;    // =(uint16_t(1) << k) - uint16_t(1)
const uint16_t MASK_HI_BIT_F16 = 0x01ff;    // =(uint16_t(1) << (k + 5 + 1)) - uint16_t(1)

static const _Float16 LOOKUP_TABLE_H_F16[8] = {0x1p0f16, 0x1.174p0f16, 0x1.308p0f16, 0x1.4cp0f16,
                                              0x1.6ap0f16, 0x1.8acp0f16, 0x1.ae8p0f16, 0x1.d58p0f16};
static const _Float16 LOOKUP_TABLE_L_F16[8] = {0.0f16, -0x1.47cp-12f16, -0x1.02p-12f16, -0x1.298p-15f16,
                                            0x1.3ccp-13f16, 0x1.ca8p-13f16, 0x1.3f4p-13f16, 0x1.8ep-16f16};

// ---------------------------- m1 ----------------------------

forceinline void check_special_cases_f16m1(const vfloat16m1_t& x, vfloat16m1_t& special, vbool16_t& specialMask, size_t vl) {
  // check +inf
  const uint16_t pinf = 0x7c00;
  specialMask = __riscv_vmfeq_vf_f16m1_b16(x, RVVMF_SC_AS_FP16(pinf), vl);
  special = __riscv_vfmerge_vfm_f16m1(x, RVVMF_SC_AS_FP16(pinf), specialMask, vl);

  // check overflow
  // overflowThreshold < x < pinf
  vbool16_t mask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f16m1_b16(x, OVERFLOWTHRESHOLD_F16, vl),
                                        __riscv_vmflt_vf_f16m1_b16(x, RVVMF_SC_AS_FP16(pinf), vl), vl);
  special = __riscv_vfmerge_vfm_f16m1(special, RVVMF_SC_AS_FP16(pinf), mask, vl);
  specialMask = __riscv_vmor_mm_b16(specialMask, mask, vl);
  if (__riscv_vcpop_m_b16(mask, vl)) RVVMF_SC_CALL_FE_OVERFLOW();

  const uint16_t qnan = 0x7e00;
  // check qNaN (pinf < x)
  mask = __riscv_vmsgtu_vx_u16m1_b16(__riscv_vreinterpret_v_f16m1_u16m1(x), pinf, vl);
  special = __riscv_vfmerge_vfm_f16m1(special, RVVMF_SC_AS_FP16(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b16(specialMask, mask, vl);

  // check sNaN (pinf < x < qnan)
  mask = __riscv_vmand_mm_b16(mask,
                              __riscv_vmsltu_vx_u16m1_b16(__riscv_vreinterpret_v_f16m1_u16m1(x),
                              qnan, vl), vl);
  special = __riscv_vfmerge_vfm_f16m1(special, RVVMF_SC_AS_FP16(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b16(specialMask, mask, vl);
  if (__riscv_vcpop_m_b16(mask, vl)) RVVMF_SC_CALL_FE_INVALID();
}

forceinline void argument_reduction_f16m1(const vfloat16m1_t& x, vuint16m1_t& ei, vuint16m1_t& fi, vfloat16m1_t& h, size_t vl) {
  vfloat16m1_t vmagicConst1 = __riscv_vfmv_v_f_f16m1(MAGIC_CONST_1_F16, vl);
  h = __riscv_vfmadd_vf_f16m1(x, INV_LOG_22_K_F16, vmagicConst1, vl);
  fi = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(h), MASK_HI_BIT_F16, vl);
  ei = __riscv_vsrl_vx_u16m1(fi, TABLE_SIZE_DEG_F16, vl);
  fi = __riscv_vand_vx_u16m1(fi, MASK_FI_BIT_F16, vl);
  h = __riscv_vfsub_vv_f16m1(h, vmagicConst1, vl);
}

forceinline void argument_reduction_h_f16m1(const vfloat16m1_t& x, vfloat16m1_t& y, vuint16m1_t& ei, vuint16m1_t& fi, size_t vl) {
  vfloat16m1_t h;
  argument_reduction_f16m1(x, ei, fi, h, vl);
  y = __riscv_vfmadd_vf_f16m1(h, M_LOG22_K_L_F16, __riscv_vfmadd_vf_f16m1(h, M_LOG22_K_H_F16, x, vl), vl);
}

forceinline void argument_reduction_hl_f16m1(const vfloat16m1_t& x, vfloat16m1_t& yh, vfloat16m1_t& yl, vuint16m1_t& ei, vuint16m1_t& fi, size_t vl) {
  vfloat16m1_t h;
  argument_reduction_f16m1(x, ei, fi, h, vl);
  fma12_ver2_vf_f16m1(h, M_LOG22_K_L_F16, __riscv_vfmadd_vf_f16m1(h, M_LOG22_K_H_F16, x, vl), yh, yl, vl);
  yl = __riscv_vfmadd_vf_f16m1(h, M_LOG22_K_LL_F16, yl, vl);
  fast2sum_vv_f16m1(yh, yl, yh, yl, vl);
}

forceinline void get_table_values_f16m1(vuint16m1_t index, vfloat16m1_t& th, vfloat16m1_t& tl, size_t vl){
  index = __riscv_vmul_vx_u16m1(index, uint16_t(sizeof(_Float16)), vl);

  th = __riscv_vloxei16_v_f16m1(LOOKUP_TABLE_H_F16, index, vl);
  tl = __riscv_vloxei16_v_f16m1(LOOKUP_TABLE_L_F16, index, vl);
}

forceinline void calculate_polynom_f16m1(const vfloat16m1_t& yh, const vfloat16m1_t& yl, vfloat16m1_t& odd, vfloat16m1_t& even, vfloat16m1_t& oddtail, vfloat16m1_t& eventail, size_t vl){
  vfloat16m1_t sqry = __riscv_vfmul_vv_f16m1(yh, yh, vl);

  odd = __riscv_vfmul_vf_f16m1(sqry, POL_COEFF_2_F16, vl);
  even = __riscv_vfmadd_vf_f16m1(sqry, POL_COEFF_3_F16, __riscv_vfmv_v_f_f16m1(POL_COEFF_1_F16, vl), vl);

  fma12_vv_f16m1(yh, odd, yh, odd, oddtail, vl);
  oddtail = __riscv_vfadd_vv_f16m1(oddtail, __riscv_vfmadd_vv_f16m1(yl, oddtail, yl, vl), vl);
  mul12_vv_f16m1(even, sqry, even, eventail, vl);
}

forceinline void update_exponent_f16m1(const vuint16m1_t& ei, vfloat16m1_t& res, const size_t addition, size_t vl){
  res = __riscv_vfmul_vv_f16m1(res, __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vsll_vx_u16m1(__riscv_vadd_vx_u16m1(ei, addition, vl), (size_t)10, vl)), vl);
}

forceinline void get_sign_f16m1(vfloat16m1_t& x, vuint16m1_t& sign, size_t vl){
  sign = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(x), MASK_SIGN_F16, vl);
  x = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(x), MASK_ABS_F16, vl));
}

forceinline void set_sign_f16m1(vfloat16m1_t& res, const vuint16m1_t& sign, size_t vl){
  res = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vor_vv_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(res), sign, vl));
}

forceinline void get_abs_f16m1(vfloat16m1_t& x, size_t vl){
  x = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(x), MASK_ABS_F16, vl));
}

// ---------------------------- m2 ----------------------------

forceinline void check_special_cases_f16m2(const vfloat16m2_t& x, vfloat16m2_t& special, vbool8_t& specialMask, size_t vl) {
  // check +inf
  const uint16_t pinf = 0x7c00;
  specialMask = __riscv_vmfeq_vf_f16m2_b8(x, RVVMF_SC_AS_FP16(pinf), vl);
  special = __riscv_vfmerge_vfm_f16m2(x, RVVMF_SC_AS_FP16(pinf), specialMask, vl);

  // check overflow
  // overflowThreshold < x < pinf
  vbool8_t mask = __riscv_vmand_mm_b8(__riscv_vmfgt_vf_f16m2_b8(x, OVERFLOWTHRESHOLD_F16, vl),
                              __riscv_vmflt_vf_f16m2_b8(x, RVVMF_SC_AS_FP16(pinf), vl), vl);
  special = __riscv_vfmerge_vfm_f16m2(special, RVVMF_SC_AS_FP16(pinf), mask, vl);
  specialMask = __riscv_vmor_mm_b8(specialMask, mask, vl);
  if (__riscv_vcpop_m_b8(mask, vl)) RVVMF_SC_CALL_FE_OVERFLOW();

  const uint16_t qnan = 0x7e00;
  // check qNaN (pinf < x)
  mask = __riscv_vmsgtu_vx_u16m2_b8(__riscv_vreinterpret_v_f16m2_u16m2(x), pinf, vl);
  special = __riscv_vfmerge_vfm_f16m2(special, RVVMF_SC_AS_FP16(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b8(specialMask, mask, vl);

  // check sNaN (pinf < x < qnan)
  mask = __riscv_vmand_mm_b8(mask,
                              __riscv_vmsltu_vx_u16m2_b8(__riscv_vreinterpret_v_f16m2_u16m2(x),
                              qnan, vl), vl);
  special = __riscv_vfmerge_vfm_f16m2(special, RVVMF_SC_AS_FP16(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b8(specialMask, mask, vl);
  if (__riscv_vcpop_m_b8(mask, vl)) RVVMF_SC_CALL_FE_INVALID();
}

forceinline void argument_reduction_f16m2(const vfloat16m2_t& x, vuint16m2_t& ei, vuint16m2_t& fi, vfloat16m2_t& h, size_t vl) {
  vfloat16m2_t vmagicConst1 = __riscv_vfmv_v_f_f16m2(MAGIC_CONST_1_F16, vl);
  h = __riscv_vfmadd_vf_f16m2(x, INV_LOG_22_K_F16, vmagicConst1, vl);
  fi = __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(h), MASK_HI_BIT_F16, vl);
  ei = __riscv_vsrl_vx_u16m2(fi, TABLE_SIZE_DEG_F16, vl);
  fi = __riscv_vand_vx_u16m2(fi, MASK_FI_BIT_F16, vl);
  h = __riscv_vfsub_vv_f16m2(h, vmagicConst1, vl);
}

forceinline void argument_reduction_h_f16m2(const vfloat16m2_t& x, vfloat16m2_t& y, vuint16m2_t& ei, vuint16m2_t& fi, size_t vl) {
  vfloat16m2_t h;
  argument_reduction_f16m2(x, ei, fi, h, vl);
  y = __riscv_vfmadd_vf_f16m2(h, M_LOG22_K_L_F16, __riscv_vfmadd_vf_f16m2(h, M_LOG22_K_H_F16, x, vl), vl);
}

forceinline void argument_reduction_hl_f16m2(const vfloat16m2_t& x, vfloat16m2_t& yh, vfloat16m2_t& yl, vuint16m2_t& ei, vuint16m2_t& fi, size_t vl) {
  vfloat16m2_t h;
  argument_reduction_f16m2(x, ei, fi, h, vl);
  fma12_ver2_vf_f16m2(h, M_LOG22_K_L_F16, __riscv_vfmadd_vf_f16m2(h, M_LOG22_K_H_F16, x, vl), yh, yl, vl);
  yl = __riscv_vfmadd_vf_f16m2(h, M_LOG22_K_LL_F16, yl, vl);
  fast2sum_vv_f16m2(yh, yl, yh, yl, vl);
}

forceinline void get_table_values_f16m2(vuint16m2_t index, vfloat16m2_t& th, vfloat16m2_t& tl, size_t vl){
  index = __riscv_vmul_vx_u16m2(index, uint16_t(sizeof(_Float16)), vl);

  th = __riscv_vloxei16_v_f16m2(LOOKUP_TABLE_H_F16, index, vl);
  tl = __riscv_vloxei16_v_f16m2(LOOKUP_TABLE_L_F16, index, vl);
}

forceinline void calculate_polynom_f16m2(const vfloat16m2_t& yh, const vfloat16m2_t& yl, vfloat16m2_t& odd, vfloat16m2_t& even, vfloat16m2_t& oddtail, vfloat16m2_t& eventail, size_t vl){
  vfloat16m2_t sqry = __riscv_vfmul_vv_f16m2(yh, yh, vl);

  odd = __riscv_vfmul_vf_f16m2(sqry, POL_COEFF_2_F16, vl);
  even = __riscv_vfmadd_vf_f16m2(sqry, POL_COEFF_3_F16, __riscv_vfmv_v_f_f16m2(POL_COEFF_1_F16, vl), vl);

  fma12_vv_f16m2(yh, odd, yh, odd, oddtail, vl);
  oddtail = __riscv_vfadd_vv_f16m2(oddtail, __riscv_vfmadd_vv_f16m2(yl, oddtail, yl, vl), vl);
  mul12_vv_f16m2(even, sqry, even, eventail, vl);
}

forceinline void update_exponent_f16m2(const vuint16m2_t& ei, vfloat16m2_t& res, const size_t addition, size_t vl){
  res = __riscv_vfmul_vv_f16m2(res, __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vsll_vx_u16m2(__riscv_vadd_vx_u16m2(ei, addition, vl), (size_t)10, vl)), vl);
}

forceinline void get_sign_f16m2(vfloat16m2_t& x, vuint16m2_t& sign, size_t vl){
  sign = __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(x), MASK_SIGN_F16, vl);
  x = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(x), MASK_ABS_F16, vl));
}

forceinline void set_sign_f16m2(vfloat16m2_t& res, const vuint16m2_t& sign, size_t vl){
  res = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vor_vv_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(res), sign, vl));
}

forceinline void get_abs_f16m2(vfloat16m2_t& x, size_t vl){
  x = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(x), MASK_ABS_F16, vl));
}

// ---------------------------- m4 ----------------------------

forceinline void check_special_cases_f16m4(const vfloat16m4_t& x, vfloat16m4_t& special, vbool4_t& specialMask, size_t vl) {
  // check +inf
  const uint16_t pinf = 0x7c00;
  specialMask = __riscv_vmfeq_vf_f16m4_b4(x, RVVMF_SC_AS_FP16(pinf), vl);
  special = __riscv_vfmerge_vfm_f16m4(x, RVVMF_SC_AS_FP16(pinf), specialMask, vl);

  // check overflow
  // overflowThreshold < x < pinf
  vbool4_t mask = __riscv_vmand_mm_b4(__riscv_vmfgt_vf_f16m4_b4(x, OVERFLOWTHRESHOLD_F16, vl),
                              __riscv_vmflt_vf_f16m4_b4(x, RVVMF_SC_AS_FP16(pinf), vl), vl);
  special = __riscv_vfmerge_vfm_f16m4(special, RVVMF_SC_AS_FP16(pinf), mask, vl);
  specialMask = __riscv_vmor_mm_b4(specialMask, mask, vl);
  if (__riscv_vcpop_m_b4(mask, vl)) RVVMF_SC_CALL_FE_OVERFLOW();

  const uint16_t qnan = 0x7e00;
  // check qNaN (pinf < x)
  mask = __riscv_vmsgtu_vx_u16m4_b4(__riscv_vreinterpret_v_f16m4_u16m4(x), pinf, vl);
  special = __riscv_vfmerge_vfm_f16m4(special, RVVMF_SC_AS_FP16(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b4(specialMask, mask, vl);

  // check sNaN (pinf < x < qnan)
  mask = __riscv_vmand_mm_b4(mask,
                              __riscv_vmsltu_vx_u16m4_b4(__riscv_vreinterpret_v_f16m4_u16m4(x),
                              qnan, vl), vl);
  special = __riscv_vfmerge_vfm_f16m4(special, RVVMF_SC_AS_FP16(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b4(specialMask, mask, vl);
  if (__riscv_vcpop_m_b4(mask, vl)) RVVMF_SC_CALL_FE_INVALID();
}

forceinline void argument_reduction_f16m4(const vfloat16m4_t& x, vuint16m4_t& ei, vuint16m4_t& fi, vfloat16m4_t& h, size_t vl) {
  vfloat16m4_t vmagicConst1 = __riscv_vfmv_v_f_f16m4(MAGIC_CONST_1_F16, vl);
  h = __riscv_vfmadd_vf_f16m4(x, INV_LOG_22_K_F16, vmagicConst1, vl);
  fi = __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(h), MASK_HI_BIT_F16, vl);
  ei = __riscv_vsrl_vx_u16m4(fi, TABLE_SIZE_DEG_F16, vl);
  fi = __riscv_vand_vx_u16m4(fi, MASK_FI_BIT_F16, vl);
  h = __riscv_vfsub_vv_f16m4(h, vmagicConst1, vl);
}

forceinline void argument_reduction_h_f16m4(const vfloat16m4_t& x, vfloat16m4_t& y, vuint16m4_t& ei, vuint16m4_t& fi, size_t vl) {
  vfloat16m4_t h;
  argument_reduction_f16m4(x, ei, fi, h, vl);
  y = __riscv_vfmadd_vf_f16m4(h, M_LOG22_K_L_F16, __riscv_vfmadd_vf_f16m4(h, M_LOG22_K_H_F16, x, vl), vl);
}

forceinline void argument_reduction_hl_f16m4(const vfloat16m4_t& x, vfloat16m4_t& yh, vfloat16m4_t& yl, vuint16m4_t& ei, vuint16m4_t& fi, size_t vl) {
  vfloat16m4_t h;
  argument_reduction_f16m4(x, ei, fi, h, vl);
  fma12_ver2_vf_f16m4(h, M_LOG22_K_L_F16, __riscv_vfmadd_vf_f16m4(h, M_LOG22_K_H_F16, x, vl), yh, yl, vl);
  yl = __riscv_vfmadd_vf_f16m4(h, M_LOG22_K_LL_F16, yl, vl);
  fast2sum_vv_f16m4(yh, yl, yh, yl, vl);
}

forceinline void get_table_values_f16m4(vuint16m4_t index, vfloat16m4_t& th, vfloat16m4_t& tl, size_t vl){
  index = __riscv_vmul_vx_u16m4(index, uint16_t(sizeof(_Float16)), vl);

  th = __riscv_vloxei16_v_f16m4(LOOKUP_TABLE_H_F16, index, vl);
  tl = __riscv_vloxei16_v_f16m4(LOOKUP_TABLE_L_F16, index, vl);
}

forceinline void calculate_polynom_f16m4(const vfloat16m4_t& yh, const vfloat16m4_t& yl, vfloat16m4_t& odd, vfloat16m4_t& even, vfloat16m4_t& oddtail, vfloat16m4_t& eventail, size_t vl){
  vfloat16m4_t sqry = __riscv_vfmul_vv_f16m4(yh, yh, vl);

  odd = __riscv_vfmul_vf_f16m4(sqry, POL_COEFF_2_F16, vl);
  even = __riscv_vfmadd_vf_f16m4(sqry, POL_COEFF_3_F16, __riscv_vfmv_v_f_f16m4(POL_COEFF_1_F16, vl), vl);

  fma12_vv_f16m4(yh, odd, yh, odd, oddtail, vl);
  oddtail = __riscv_vfadd_vv_f16m4(oddtail, __riscv_vfmadd_vv_f16m4(yl, oddtail, yl, vl), vl);
  mul12_vv_f16m4(even, sqry, even, eventail, vl);
}

forceinline void update_exponent_f16m4(const vuint16m4_t& ei, vfloat16m4_t& res, const size_t addition, size_t vl){
  res = __riscv_vfmul_vv_f16m4(res, __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vsll_vx_u16m4(__riscv_vadd_vx_u16m4(ei, addition, vl), (size_t)10, vl)), vl);
}

forceinline void get_sign_f16m4(vfloat16m4_t& x, vuint16m4_t& sign, size_t vl){
  sign = __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(x), MASK_SIGN_F16, vl);
  x = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(x), MASK_ABS_F16, vl));
}

forceinline void set_sign_f16m4(vfloat16m4_t& res, const vuint16m4_t& sign, size_t vl){
  res = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vor_vv_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(res), sign, vl));
}

forceinline void get_abs_f16m4(vfloat16m4_t& x, size_t vl){
  x = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(x), MASK_ABS_F16, vl));
}

#endif
