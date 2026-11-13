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
 *   File:  sinh_cosh_float.inl                          *
 *   Contains: helper built-in functions for sinh, cosh  *
 *             functions (float32_t)                     *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#include "sinh_cosh_utilities.inl"

const float ZERO_F32 = 0.0f;
const float ONE_F32 = 1.0f;
const float TWO_F32 = 2.0f;
const float OVERFLOWTHRESHOLD_F32 = 0x1.65a9f8p6f;
const float MAGIC_CONST_1_F32 = 12582912.0f;    // =1.5*2^23
const float INV_LOG_22_K_F32 = 0x1.715476p4f;    // =1 / (log(2)*2^(-k))
const float M_LOG22_K_H_F32 = -0x1.62ep-5f;      // =-log(2)*2^(-k) (high bits)
const float M_LOG22_K_L_F32 = -0x1.0bfbe8p-19f;  // =-log(2)*2^(-k) (middle bits))
const float M_LOG22_K_LL_F32 = -0x1.cf79acp-44f;    // =-log(2)*2^(-k) (low bits)
const size_t TABLE_SIZE_DEG_F32 = 4;
// const float POL_COEFF_1_F32 = 0x1p-1f;
// const float POL_COEFF_2_F32 = 0x1.555556p-3f;
// const float POL_COEFF_3_F32 = 0x1.55569p-5f;
// const float POL_COEFF_4_F32 = 0x1.10aeeap-7f;
const float POL_COEFF_1_F32 = 0x1p56f;
const float POL_COEFF_2_F32 = 0x1.555556p54f;
const float POL_COEFF_3_F32 = 0x1.55569p52f;
const float POL_COEFF_4_F32 = 0x1.10aeeap50f;
const uint32_t MASK_ABS_F32 = 0x7fffffff;
const uint32_t MASK_SIGN_F32 = 0x80000000;
const uint32_t MASK_FI_BIT_F32 = 0x0000000f;  // =(uint32_t(1) << k) - uint32_t(1)
const uint32_t MASK_HI_BIT_F32 = 0x00001fff;  // =(uint32_t(1) << (k + 8 + 1)) - uint32_t(1);

const float FACTOR_F32 =  0x1p57f;
const uint32_t FACTOR_DEG_F32 = 70; // 127 - 57
const float ADD_COEFF_F32 = 0x1p-41f;
const uint32_t SHIFT_CONST_F32 = 0x000001af;
const int32_t MAGIC_DIFF_SINH_F32 = -1;
const uint32_t MAGIC_DIFF_COSH_F32 = 9;
const float MUL_COEFF_F32 = 0x1p-2f;
const float MASK_NUMBER_1_F32 = 0x1p-40f;
const float MASK_NUMBER_2_F32 = 0x1p-126f;
const float MASK_NUMBER_3_F32 = 0x1p4f;

static const float LOOKUP_TABLE_H_F32[16] = {0x1p0f,        0x1.0b5586p0f, 0x1.172b84p0f, 0x1.2387a6p0f,
                                            0x1.306fep0f,  0x1.3dea64p0f, 0x1.4bfdaep0f, 0x1.5ab07ep0f,
                                            0x1.6a09e6p0f, 0x1.7a1148p0f, 0x1.8ace54p0f, 0x1.9c4918p0f,
                                            0x1.ae89fap0f, 0x1.c199bep0f, 0x1.d5818ep0f, 0x1.ea4afap0f};

static const float LOOKUP_TABLE_L_F32[16] = {0.0f, 0x1.9f3122p-25f, -0x1.c15742p-27f, 0x1.ceac48p-25f,
                                            0x1.4636e2p-25f, 0x1.824684p-25f, -0x1.593abcp-25f, -0x1.5bd5ecp-27f,
                                            0x1.9fcef4p-26f, -0x1.829fdp-25f, 0x1.15506ep-27f, 0x1.51f848p-27f,
                                            -0x1.a94b14p-26f, -0x1.3d56b2p-27f, -0x1.822dbcp-27f, 0x1.52486cp-27f};

// ---------------------------- m1 ----------------------------

forceinline void check_special_cases_f32m1(const vfloat32m1_t& x, vfloat32m1_t& special, vbool32_t& specialMask, size_t vl) {
  // check +inf
  const uint32_t pinf = 0x7f800000;
  specialMask = __riscv_vmfeq_vf_f32m1_b32(x, RVVMF_SC_AS_FP32(pinf), vl);
  special = __riscv_vfmerge_vfm_f32m1(x, RVVMF_SC_AS_FP32(pinf), specialMask, vl);

  // check overflow
  // overflowThreshold < x < pinf
  vbool32_t mask = __riscv_vmand_mm_b32(__riscv_vmfgt_vf_f32m1_b32(x, OVERFLOWTHRESHOLD_F32, vl),
                              __riscv_vmflt_vf_f32m1_b32(x, RVVMF_SC_AS_FP32(pinf), vl), vl);
  special = __riscv_vfmerge_vfm_f32m1(special, RVVMF_SC_AS_FP32(pinf), mask, vl);
  specialMask = __riscv_vmor_mm_b32(specialMask, mask, vl);
  if (__riscv_vcpop_m_b32(mask, vl)) RVVMF_SC_CALL_FE_OVERFLOW();

  const uint32_t qnan = 0x7fc00000;
  // check qNaN (pinf < x)
  mask = __riscv_vmsgtu_vx_u32m1_b32(RVVMF_F32_TO_U32(x, m1), pinf, vl);
  special = __riscv_vfmerge_vfm_f32m1(special, RVVMF_SC_AS_FP32(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b32(specialMask, mask, vl);

  // check sNaN (pinf < x < qnan)
  mask = __riscv_vmand_mm_b32(mask,
                              __riscv_vmsltu_vx_u32m1_b32(RVVMF_F32_TO_U32(x, m1),
                              qnan, vl), vl);
  special = __riscv_vfmerge_vfm_f32m1(special, RVVMF_SC_AS_FP32(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b32(specialMask, mask, vl);
  if (__riscv_vcpop_m_b32(mask, vl)) RVVMF_SC_CALL_FE_INVALID();
}

forceinline void argument_reduction_f32m1(const vfloat32m1_t& x, vuint32m1_t& ei, vuint32m1_t& fi, vfloat32m1_t& h, size_t vl) {
  vfloat32m1_t vmagicConst1 = __riscv_vfmv_v_f_f32m1(MAGIC_CONST_1_F32, vl);
  h = __riscv_vfmadd_vf_f32m1(x, INV_LOG_22_K_F32, vmagicConst1, vl);
  fi = __riscv_vand_vx_u32m1(RVVMF_F32_TO_U32(h, m1), MASK_HI_BIT_F32, vl);
  ei = __riscv_vsrl_vx_u32m1(fi, TABLE_SIZE_DEG_F32, vl);
  fi = __riscv_vand_vx_u32m1(fi, MASK_FI_BIT_F32, vl);
  h = __riscv_vfsub_vv_f32m1(h, vmagicConst1, vl);
}

forceinline void argument_reduction_h_f32m1(const vfloat32m1_t& x, vfloat32m1_t& y, vuint32m1_t& ei, vuint32m1_t& fi, size_t vl) {
  vfloat32m1_t h;
  argument_reduction_f32m1(x, ei, fi, h, vl);
  y = __riscv_vfmadd_vf_f32m1(h, M_LOG22_K_L_F32, __riscv_vfmadd_vf_f32m1(h, M_LOG22_K_H_F32, x, vl), vl);
}

forceinline void argument_reduction_hl_f32m1(const vfloat32m1_t& x, vfloat32m1_t& yh, vfloat32m1_t& yl, vuint32m1_t& ei, vuint32m1_t& fi, size_t vl) {
  vfloat32m1_t h;
  argument_reduction_f32m1(x, ei, fi, h, vl);
  fma12_ver2_vf_f32m1(h, M_LOG22_K_L_F32, __riscv_vfmadd_vf_f32m1(h, M_LOG22_K_H_F32, x, vl), yh, yl, vl);
  yl = __riscv_vfmadd_vf_f32m1(h, M_LOG22_K_LL_F32, yl, vl);
  fast2sum_vv_f32m1(yh, yl, yh, yl, vl);
}

forceinline void get_table_values_f32m1(vuint32m1_t index, vfloat32m1_t& th, vfloat32m1_t& tl, size_t vl){
  index = __riscv_vmul_vx_u32m1(index, uint32_t(sizeof(float)), vl);

  th = __riscv_vloxei32_v_f32m1(LOOKUP_TABLE_H_F32, index, vl);
  tl = __riscv_vloxei32_v_f32m1(LOOKUP_TABLE_L_F32, index, vl);
}

forceinline void calculate_polynom_f32m1(const vfloat32m1_t& sqry, vfloat32m1_t& odd, vfloat32m1_t& even, size_t vl){
  odd = __riscv_vfmadd_vf_f32m1(sqry, POL_COEFF_4_F32, __riscv_vfmv_v_f_f32m1(POL_COEFF_2_F32, vl), vl);
  odd = __riscv_vfmul_vv_f32m1(odd, sqry, vl);

  even = __riscv_vfmadd_vf_f32m1(sqry, POL_COEFF_3_F32, __riscv_vfmv_v_f_f32m1(POL_COEFF_1_F32, vl), vl);
}

forceinline void calculate_polynom_sinh_f32m1(const vfloat32m1_t& yh, const vfloat32m1_t& yl, vfloat32m1_t& odd, vfloat32m1_t& even, vfloat32m1_t& oddtail, vfloat32m1_t& eventail, size_t vl){
  vfloat32m1_t yhc = __riscv_vfadd_vf_f32m1(yh, ADD_COEFF_F32, vl);
  vfloat32m1_t sqry = __riscv_vfmul_vv_f32m1(yhc, yhc, vl);
  calculate_polynom_f32m1(sqry, odd, even, vl);

  fma12_vv_f32m1(yh, odd, __riscv_vfmul_vf_f32m1(yh, FACTOR_F32, vl), odd, oddtail, vl);
  oddtail = __riscv_vfadd_vv_f32m1(__riscv_vfmadd(yl, oddtail, __riscv_vfmul_vf_f32m1(yl, FACTOR_F32, vl), vl), oddtail, vl);;
  mul12_vv_f32m1(even, sqry, even, eventail, vl);
}

forceinline void calculate_polynom_cosh_f32m1(const vfloat32m1_t& y, vfloat32m1_t& odd, vfloat32m1_t& even, size_t vl){
  vfloat32m1_t yc = __riscv_vfadd_vf_f32m1(y, ADD_COEFF_F32, vl);
  vfloat32m1_t sqry = __riscv_vfmul_vv_f32m1(yc, yc, vl);
  calculate_polynom_f32m1(sqry, odd, even, vl);

  odd = __riscv_vfmadd_vv_f32m1(y, odd, __riscv_vfmul_vf_f32m1(y, FACTOR_F32, vl), vl);
  even = __riscv_vfmul_vv_f32m1(even, sqry, vl);
}

forceinline void update_exponent_f32m1(const vuint32m1_t& ei, vfloat32m1_t& res, size_t vl){
  res = __riscv_vfmul_vv_f32m1(res, RVVMF_U32_TO_F32(__riscv_vsll_vx_u32m1(ei, (size_t)23, vl), m1), vl);
}

forceinline void get_sign_f32m1(vfloat32m1_t& x, vuint32m1_t& sign, size_t vl){
  sign = __riscv_vand_vx_u32m1(RVVMF_F32_TO_U32(x, m1), MASK_SIGN_F32, vl);
  x = RVVMF_U32_TO_F32(__riscv_vand_vx_u32m1(RVVMF_F32_TO_U32(x, m1), MASK_ABS_F32, vl), m1);
}

forceinline void set_sign_f32m1(vfloat32m1_t& res, const vuint32m1_t& sign, size_t vl){
  res = RVVMF_U32_TO_F32(__riscv_vor_vv_u32m1(RVVMF_F32_TO_U32(res, m1), sign, vl), m1);
}

forceinline void get_abs_f32m1(vfloat32m1_t& x, size_t vl){
  x = RVVMF_U32_TO_F32(__riscv_vand_vx_u32m1(RVVMF_F32_TO_U32(x, m1), MASK_ABS_F32, vl), m1);
}

// ---------------------------- m2 ----------------------------

forceinline void check_special_cases_f32m2(const vfloat32m2_t& x, vfloat32m2_t& special, vbool16_t& specialMask, size_t vl) {
  // check +inf
  const uint32_t pinf = 0x7f800000;
  specialMask = __riscv_vmfeq_vf_f32m2_b16(x, RVVMF_SC_AS_FP32(pinf), vl);
  special = __riscv_vfmerge_vfm_f32m2(x, RVVMF_SC_AS_FP32(pinf), specialMask, vl);

  // check overflow
  // overflowThreshold < x < pinf
  vbool16_t mask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f32m2_b16(x, OVERFLOWTHRESHOLD_F32, vl),
                                        __riscv_vmflt_vf_f32m2_b16(x, RVVMF_SC_AS_FP32(pinf), vl), vl);
  special = __riscv_vfmerge_vfm_f32m2(special, RVVMF_SC_AS_FP32(pinf), mask, vl);
  specialMask = __riscv_vmor_mm_b16(specialMask, mask, vl);
  if (__riscv_vcpop_m_b16(mask, vl)) RVVMF_SC_CALL_FE_OVERFLOW();

  const uint32_t qnan = 0x7fc00000;
  // check qNaN (pinf < x)
  mask = __riscv_vmsgtu_vx_u32m2_b16(RVVMF_F32_TO_U32(x, m2), pinf, vl);
  special = __riscv_vfmerge_vfm_f32m2(special, RVVMF_SC_AS_FP32(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b16(specialMask, mask, vl);

  // check sNaN (pinf < x < qnan)
  mask = __riscv_vmand_mm_b16(mask,
                              __riscv_vmsltu_vx_u32m2_b16(RVVMF_F32_TO_U32(x, m2),
                              qnan, vl), vl);
  special = __riscv_vfmerge_vfm_f32m2(special, RVVMF_SC_AS_FP32(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b16(specialMask, mask, vl);
  if (__riscv_vcpop_m_b16(mask, vl)) RVVMF_SC_CALL_FE_INVALID();
}

forceinline void argument_reduction_f32m2(const vfloat32m2_t& x, vuint32m2_t& ei, vuint32m2_t& fi, vfloat32m2_t& h, size_t vl) {
  vfloat32m2_t vmagicConst1 = __riscv_vfmv_v_f_f32m2(MAGIC_CONST_1_F32, vl);
  h = __riscv_vfmadd_vf_f32m2(x, INV_LOG_22_K_F32, vmagicConst1, vl);
  fi = __riscv_vand_vx_u32m2(RVVMF_F32_TO_U32(h, m2), MASK_HI_BIT_F32, vl);
  ei = __riscv_vsrl_vx_u32m2(fi, TABLE_SIZE_DEG_F32, vl);
  fi = __riscv_vand_vx_u32m2(fi, MASK_FI_BIT_F32, vl);
  h = __riscv_vfsub_vv_f32m2(h, vmagicConst1, vl);
}

forceinline void argument_reduction_h_f32m2(const vfloat32m2_t& x, vfloat32m2_t& y, vuint32m2_t& ei, vuint32m2_t& fi, size_t vl) {
  vfloat32m2_t h;
  argument_reduction_f32m2(x, ei, fi, h, vl);
  y = __riscv_vfmadd_vf_f32m2(h, M_LOG22_K_L_F32, __riscv_vfmadd_vf_f32m2(h, M_LOG22_K_H_F32, x, vl), vl);
}

forceinline void argument_reduction_hl_f32m2(const vfloat32m2_t& x, vfloat32m2_t& yh, vfloat32m2_t& yl, vuint32m2_t& ei, vuint32m2_t& fi, size_t vl) {
  vfloat32m2_t h;
  argument_reduction_f32m2(x, ei, fi, h, vl);
  fma12_ver2_vf_f32m2(h, M_LOG22_K_L_F32, __riscv_vfmadd_vf_f32m2(h, M_LOG22_K_H_F32, x, vl), yh, yl, vl);
  yl = __riscv_vfmadd_vf_f32m2(h, M_LOG22_K_LL_F32, yl, vl);
  fast2sum_vv_f32m2(yh, yl, yh, yl, vl);
}

forceinline void get_table_values_f32m2(vuint32m2_t index, vfloat32m2_t& th, vfloat32m2_t& tl, size_t vl){
  index = __riscv_vmul_vx_u32m2(index, uint32_t(sizeof(float)), vl);

  th = __riscv_vloxei32_v_f32m2(LOOKUP_TABLE_H_F32, index, vl);
  tl = __riscv_vloxei32_v_f32m2(LOOKUP_TABLE_L_F32, index, vl);
}

forceinline void calculate_polynom_f32m2(const vfloat32m2_t& sqry, vfloat32m2_t& odd, vfloat32m2_t& even, size_t vl){
  odd = __riscv_vfmadd_vf_f32m2(sqry, POL_COEFF_4_F32, __riscv_vfmv_v_f_f32m2(POL_COEFF_2_F32, vl), vl);
  odd = __riscv_vfmul_vv_f32m2(odd, sqry, vl);

  even = __riscv_vfmadd_vf_f32m2(sqry, POL_COEFF_3_F32, __riscv_vfmv_v_f_f32m2(POL_COEFF_1_F32, vl), vl);
}

forceinline void calculate_polynom_sinh_f32m2(const vfloat32m2_t& yh, const vfloat32m2_t& yl, vfloat32m2_t& odd, vfloat32m2_t& even, vfloat32m2_t& oddtail, vfloat32m2_t& eventail, size_t vl){
  vfloat32m2_t yhc = __riscv_vfadd_vf_f32m2(yh, ADD_COEFF_F32, vl);
  vfloat32m2_t sqry = __riscv_vfmul_vv_f32m2(yhc, yhc, vl);
  calculate_polynom_f32m2(sqry, odd, even, vl);

  fma12_vv_f32m2(yh, odd, __riscv_vfmul_vf_f32m2(yh, FACTOR_F32, vl), odd, oddtail, vl);
  oddtail = __riscv_vfadd_vv_f32m2(__riscv_vfmadd(yl, oddtail, __riscv_vfmul_vf_f32m2(yl, FACTOR_F32, vl), vl), oddtail, vl);;
  mul12_vv_f32m2(even, sqry, even, eventail, vl);
}

forceinline void calculate_polynom_cosh_f32m2(const vfloat32m2_t& y, vfloat32m2_t& odd, vfloat32m2_t& even, size_t vl){
  vfloat32m2_t yc = __riscv_vfadd_vf_f32m2(y, ADD_COEFF_F32, vl);
  vfloat32m2_t sqry = __riscv_vfmul_vv_f32m2(yc, yc, vl);
  calculate_polynom_f32m2(sqry, odd, even, vl);

  odd = __riscv_vfmadd_vv_f32m2(y, odd, __riscv_vfmul_vf_f32m2(y, FACTOR_F32, vl), vl);
  even = __riscv_vfmul_vv_f32m2(even, sqry, vl);
}

forceinline void update_exponent_f32m2(const vuint32m2_t& ei, vfloat32m2_t& res, size_t vl){
  res = __riscv_vfmul_vv_f32m2(res, RVVMF_U32_TO_F32(__riscv_vsll_vx_u32m2(ei, (size_t)23, vl), m2), vl);
}

forceinline void get_sign_f32m2(vfloat32m2_t& x, vuint32m2_t& sign, size_t vl){
  sign = __riscv_vand_vx_u32m2(RVVMF_F32_TO_U32(x, m2), MASK_SIGN_F32, vl);
  x = RVVMF_U32_TO_F32(__riscv_vand_vx_u32m2(RVVMF_F32_TO_U32(x, m2), MASK_ABS_F32, vl), m2);
}

forceinline void set_sign_f32m2(vfloat32m2_t& res, const vuint32m2_t& sign, size_t vl){
  res = RVVMF_U32_TO_F32(__riscv_vor_vv_u32m2(RVVMF_F32_TO_U32(res, m2), sign, vl), m2);
}

forceinline void get_abs_f32m2(vfloat32m2_t& x, size_t vl){
  x = RVVMF_U32_TO_F32(__riscv_vand_vx_u32m2(RVVMF_F32_TO_U32(x, m2), MASK_ABS_F32, vl), m2);
}

// ---------------------------- m4 ----------------------------

forceinline void check_special_cases_f32m4(const vfloat32m4_t& x, vfloat32m4_t& special, vbool8_t& specialMask, size_t vl) {
  // check +inf
  const uint32_t pinf = 0x7f800000;
  specialMask = __riscv_vmfeq_vf_f32m4_b8(x, RVVMF_SC_AS_FP32(pinf), vl);
  special = __riscv_vfmerge_vfm_f32m4(x, RVVMF_SC_AS_FP32(pinf), specialMask, vl);

  // check overflow
  // overflowThreshold < x < pinf
  vbool8_t mask = __riscv_vmand_mm_b8(__riscv_vmfgt_vf_f32m4_b8(x, OVERFLOWTHRESHOLD_F32, vl),
                              __riscv_vmflt_vf_f32m4_b8(x, RVVMF_SC_AS_FP32(pinf), vl), vl);
  special = __riscv_vfmerge_vfm_f32m4(special, RVVMF_SC_AS_FP32(pinf), mask, vl);
  specialMask = __riscv_vmor_mm_b8(specialMask, mask, vl);
  if (__riscv_vcpop_m_b8(mask, vl)) RVVMF_SC_CALL_FE_OVERFLOW();

  const uint32_t qnan = 0x7fc00000;
  // check qNaN (pinf < x)
  mask = __riscv_vmsgtu_vx_u32m4_b8(RVVMF_F32_TO_U32(x, m4), pinf, vl);
  special = __riscv_vfmerge_vfm_f32m4(special, RVVMF_SC_AS_FP32(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b8(specialMask, mask, vl);

  // check sNaN (pinf < x < qnan)
  mask = __riscv_vmand_mm_b8(mask,
                              __riscv_vmsltu_vx_u32m4_b8(RVVMF_F32_TO_U32(x, m4),
                              qnan, vl), vl);
  special = __riscv_vfmerge_vfm_f32m4(special, RVVMF_SC_AS_FP32(qnan), mask, vl);
  specialMask = __riscv_vmor_mm_b8(specialMask, mask, vl);
  if (__riscv_vcpop_m_b8(mask, vl)) RVVMF_SC_CALL_FE_INVALID();
}

forceinline void argument_reduction_f32m4(const vfloat32m4_t& x, vuint32m4_t& ei, vuint32m4_t& fi, vfloat32m4_t& h, size_t vl) {
  vfloat32m4_t vmagicConst1 = __riscv_vfmv_v_f_f32m4(MAGIC_CONST_1_F32, vl);
  h = __riscv_vfmadd_vf_f32m4(x, INV_LOG_22_K_F32, vmagicConst1, vl);
  fi = __riscv_vand_vx_u32m4(RVVMF_F32_TO_U32(h, m4), MASK_HI_BIT_F32, vl);
  ei = __riscv_vsrl_vx_u32m4(fi, TABLE_SIZE_DEG_F32, vl);
  fi = __riscv_vand_vx_u32m4(fi, MASK_FI_BIT_F32, vl);
  h = __riscv_vfsub_vv_f32m4(h, vmagicConst1, vl);
}

forceinline void argument_reduction_h_f32m4(const vfloat32m4_t& x, vfloat32m4_t& y, vuint32m4_t& ei, vuint32m4_t& fi, size_t vl) {
  vfloat32m4_t h;
  argument_reduction_f32m4(x, ei, fi, h, vl);
  y = __riscv_vfmadd_vf_f32m4(h, M_LOG22_K_L_F32, __riscv_vfmadd_vf_f32m4(h, M_LOG22_K_H_F32, x, vl), vl);
}

forceinline void argument_reduction_hl_f32m4(const vfloat32m4_t& x, vfloat32m4_t& yh, vfloat32m4_t& yl, vuint32m4_t& ei, vuint32m4_t& fi, size_t vl) {
  vfloat32m4_t h;
  argument_reduction_f32m4(x, ei, fi, h, vl);
  fma12_ver2_vf_f32m4(h, M_LOG22_K_L_F32, __riscv_vfmadd_vf_f32m4(h, M_LOG22_K_H_F32, x, vl), yh, yl, vl);
  yl = __riscv_vfmadd_vf_f32m4(h, M_LOG22_K_LL_F32, yl, vl);
  fast2sum_vv_f32m4(yh, yl, yh, yl, vl);
}

forceinline void get_table_values_f32m4(vuint32m4_t index, vfloat32m4_t& th, vfloat32m4_t& tl, size_t vl){
  index = __riscv_vmul_vx_u32m4(index, uint32_t(sizeof(float)), vl);

  th = __riscv_vloxei32_v_f32m4(LOOKUP_TABLE_H_F32, index, vl);
  tl = __riscv_vloxei32_v_f32m4(LOOKUP_TABLE_L_F32, index, vl);
}

forceinline void calculate_polynom_f32m4(const vfloat32m4_t& sqry, vfloat32m4_t& odd, vfloat32m4_t& even, size_t vl){
  odd = __riscv_vfmadd_vf_f32m4(sqry, POL_COEFF_4_F32, __riscv_vfmv_v_f_f32m4(POL_COEFF_2_F32, vl), vl);
  odd = __riscv_vfmul_vv_f32m4(odd, sqry, vl);

  even = __riscv_vfmadd_vf_f32m4(sqry, POL_COEFF_3_F32, __riscv_vfmv_v_f_f32m4(POL_COEFF_1_F32, vl), vl);
}

forceinline void calculate_polynom_sinh_f32m4(const vfloat32m4_t& yh, const vfloat32m4_t& yl, vfloat32m4_t& odd, vfloat32m4_t& even, vfloat32m4_t& oddtail, vfloat32m4_t& eventail, size_t vl){
  vfloat32m4_t yhc = __riscv_vfadd_vf_f32m4(yh, ADD_COEFF_F32, vl);
  vfloat32m4_t sqry = __riscv_vfmul_vv_f32m4(yhc, yhc, vl);
  calculate_polynom_f32m4(sqry, odd, even, vl);

  fma12_vv_f32m4(yh, odd, __riscv_vfmul_vf_f32m4(yh, FACTOR_F32, vl), odd, oddtail, vl);
  oddtail = __riscv_vfadd_vv_f32m4(__riscv_vfmadd(yl, oddtail, __riscv_vfmul_vf_f32m4(yl, FACTOR_F32, vl), vl), oddtail, vl);;
  mul12_vv_f32m4(even, sqry, even, eventail, vl);
}

forceinline void calculate_polynom_cosh_f32m4(const vfloat32m4_t& y, vfloat32m4_t& odd, vfloat32m4_t& even, size_t vl){
  vfloat32m4_t yc = __riscv_vfadd_vf_f32m4(y, ADD_COEFF_F32, vl);
  vfloat32m4_t sqry = __riscv_vfmul_vv_f32m4(yc, yc, vl);
  calculate_polynom_f32m4(sqry, odd, even, vl);

  odd = __riscv_vfmadd_vv_f32m4(y, odd, __riscv_vfmul_vf_f32m4(y, FACTOR_F32, vl), vl);
  even = __riscv_vfmul_vv_f32m4(even, sqry, vl);
}

forceinline void update_exponent_f32m4(const vuint32m4_t& ei, vfloat32m4_t& res, size_t vl){
  res = __riscv_vfmul_vv_f32m4(res, RVVMF_U32_TO_F32(__riscv_vsll_vx_u32m4(ei, (size_t)23, vl), m4), vl);
}

forceinline void get_sign_f32m4(vfloat32m4_t& x, vuint32m4_t& sign, size_t vl){
  sign = __riscv_vand_vx_u32m4(RVVMF_F32_TO_U32(x, m4), MASK_SIGN_F32, vl);
  x = RVVMF_U32_TO_F32(__riscv_vand_vx_u32m4(RVVMF_F32_TO_U32(x, m4), MASK_ABS_F32, vl), m4);
}

forceinline void set_sign_f32m4(vfloat32m4_t& res, const vuint32m4_t& sign, size_t vl){
  res = RVVMF_U32_TO_F32(__riscv_vor_vv_u32m4(RVVMF_F32_TO_U32(res, m4), sign, vl), m4);
}

forceinline void get_abs_f32m4(vfloat32m4_t& x, size_t vl){
  x = RVVMF_U32_TO_F32(__riscv_vand_vx_u32m4(RVVMF_F32_TO_U32(x, m4), MASK_ABS_F32, vl), m4);
}
