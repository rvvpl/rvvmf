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
 *   File:  sinh_cosh.cpp                                *
 *   Contains: intrinsic function sinh for f64, f32, f16 *
 *             intrinsic function cosh for f64, f32, f16 *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the hyperbolic sine or hyperbolic cosine     *
 *          of input vector V                            *
 *                                                       *
 * Algorithm:                                            *
 *    1) Argument reduction to a small interval near 0   *
 *    2) Additional reduction using the look-up table    *
 *       of size 2^k (k: f64 - 6, f32 - 4, f16 - 3)      *
 *    3) Polynomial degrees: f64 - 6, f32 - 4, f16 - 3   *
 *    4) Exponents scaling to prevent underflow          *
 *    4) Reconstruction of the result according to the   *
 *       formulas of hyperbolic functions                *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "sinh_cosh_double.inl"
#include "sinh_cosh_float.inl"

vfloat64m1_t __riscv_vsinh_f64m1(vfloat64m1_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e64m1(avl);
  vfloat64m1_t result;
  vuint64m1_t sign;
  get_sign_f64m1(x, sign, vl);

#ifndef __FAST_MATH__
  vfloat64m1_t special;
  vbool64_t specialMask;
  check_special_cases_f64m1(x, special, specialMask, vl);
#endif

  vfloat64m1_t y, th, tl;
  vuint64m1_t powerT1, fi;

  vfloat64m1_t x_minus = __riscv_vfsub_vv_f64m1(__riscv_vfmv_v_f_f64m1(0.0, vl), x, vl);  // -x
  vfloat64m1_t th_minus, tl_minus;
  vuint64m1_t powerT2, fi_minus;

  vfloat64m1_t odd, even, oddtail, eventail;

  argument_reduction_f64m1(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_h_f64m1(x, y, powerT1, fi, vl);

  vbool64_t flag_mask = __riscv_vmand_mm_b64(__riscv_vmflt_vf_f64m1_b64(x, MASK_NUMBER_1_F64, vl),
                                             __riscv_vmfge_vf_f64m1_b64(x, MASK_NUMBER_2_F64, vl), vl);

  y = __riscv_vfmerge_vfm_f64m1(y, 0.0, flag_mask, vl);

  get_table_values_f64m1(fi, th, tl, vl);
  get_table_values_f64m1(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_sinh_f64m1(y, odd, even, oddtail, eventail, vl);

  vuint64m1_t sign_powerT1 = __riscv_vsrl_vx_u64m1(powerT1, (size_t)11, vl);
  vuint64m1_t sign_powerT2 = __riscv_vsrl_vx_u64m1(powerT2, (size_t)11, vl);
  vuint64m1_t shiftT1 = __riscv_vmul_vx_u64m1(sign_powerT1, SHIFT_CONST_F64, vl);
  vuint64m1_t shiftT2 = __riscv_vmul_vx_u64m1(sign_powerT2, SHIFT_CONST_F64, vl);
  vuint64m1_t powT1 = __riscv_vadd_vv_u64m1(__riscv_vrsub_vx_u64m1(shiftT1, FACTOR_DEG_F64, vl), powerT1, vl);
  vuint64m1_t powT2 = __riscv_vadd_vv_u64m1(__riscv_vrsub_vx_u64m1(shiftT2, FACTOR_DEG_F64, vl), powerT2, vl);
  vuint64m1_t factorT1 = __riscv_vadd_vx_u64m1(shiftT1, 1023, vl);
  vuint64m1_t factorT2 = __riscv_vadd_vx_u64m1(shiftT2, 1023, vl);

  vint64m1_t exponents_diff = __riscv_vsub_vv_i64m1(RVVMF_U64_TO_I64(powT1, m1),
                                                    RVVMF_U64_TO_I64(powT2, m1), vl);
  vbool64_t exponents_mask = __riscv_vmsgt_vx_i64m1_b64(exponents_diff, MAGIC_DIFF_F64, vl);
  powT2 = __riscv_vmerge_vxm_u64m1(powT2, 0, exponents_mask, vl);
  factorT2 = __riscv_vmerge_vxm_u64m1(factorT2, 0, exponents_mask, vl);

  update_exponent_f64m1(powT1, th, vl);
  update_exponent_f64m1(powT1, tl, vl);
  update_exponent_f64m1(powT2, th_minus, vl);
  update_exponent_f64m1(powT2, tl_minus, vl);

  update_exponent_f64m1(factorT1, th, vl);
  update_exponent_f64m1(factorT1, tl, vl);
  update_exponent_f64m1(factorT2, th_minus, vl);
  update_exponent_f64m1(factorT2, tl_minus, vl);

  th = __riscv_vfmul_vf_f64m1(th, MUL_COEFF_F64, vl);
  tl = __riscv_vfmul_vf_f64m1(tl, MUL_COEFF_F64, vl);
  th_minus = __riscv_vfmul_vf_f64m1(th_minus, MUL_COEFF_F64, vl);
  tl_minus = __riscv_vfmul_vf_f64m1(tl_minus, MUL_COEFF_F64, vl);

  vfloat64m1_t sum, sumtail;
  add22_vv_f64m1(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat64m1_t diff, difftail;
  sub22_vv_f64m1(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 - T2)even(y) + (T1+T2)odd(y) + (T1-T2)
  result = __riscv_vfmadd_vv_f64m1(difftail, eventail, __riscv_vfmul_vf_f64m1(difftail, FACTOR_F64, vl), vl);
  result = __riscv_vfmadd_vv_f64m1(sumtail, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f64m1(even, difftail, result, vl);
  result = __riscv_vfmadd_vv_f64m1(diff, eventail, result, vl);
  result = __riscv_vfmadd_vv_f64m1(sum, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f64m1(sumtail, odd, result, vl);
  result = __riscv_vfmadd_vv_f64m1(diff, even, result, vl);
  result = __riscv_vfmadd_vv_f64m1(sum, odd, result, vl);
  result = __riscv_vfadd_vv_f64m1(result, __riscv_vfmul_vf_f64m1(diff, FACTOR_F64, vl), vl);
  result = __riscv_vfmul_vf_f64m1(result, 2.0, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f64m1(result, special, specialMask, vl);
#endif

  result = __riscv_vmerge_vvm_f64m1(result, x, flag_mask, vl);
  set_sign_f64m1(result, sign, vl);
  return result;
}

vfloat64m2_t __riscv_vsinh_f64m2(vfloat64m2_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e64m2(avl);
  vfloat64m2_t result;
  vuint64m2_t sign;
  get_sign_f64m2(x, sign, vl);

#ifndef __FAST_MATH__
  vfloat64m2_t special;
  vbool32_t specialMask;
  check_special_cases_f64m2(x, special, specialMask, vl);
#endif

  vfloat64m2_t y, th, tl;
  vuint64m2_t powerT1, fi;

  vfloat64m2_t x_minus = __riscv_vfsub_vv_f64m2(__riscv_vfmv_v_f_f64m2(0.0, vl), x, vl);  // -x
  vfloat64m2_t th_minus, tl_minus;
  vuint64m2_t powerT2, fi_minus;

  vfloat64m2_t odd, oddtail, even, eventail;

  argument_reduction_f64m2(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_h_f64m2(x, y, powerT1, fi, vl);

  vbool32_t flag_mask = __riscv_vmand_mm_b32(__riscv_vmflt_vf_f64m2_b32(x, MASK_NUMBER_1_F64, vl),
                                             __riscv_vmfge_vf_f64m2_b32(x, MASK_NUMBER_2_F64, vl), vl);

  y = __riscv_vfmerge_vfm_f64m2(y, 0.0, flag_mask, vl);

  get_table_values_f64m2(fi, th, tl, vl);
  get_table_values_f64m2(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_sinh_f64m2(y, odd, even, oddtail, eventail, vl);

  vuint64m2_t sign_powerT1 = __riscv_vsrl_vx_u64m2(powerT1, (size_t)11, vl);
  vuint64m2_t sign_powerT2 = __riscv_vsrl_vx_u64m2(powerT2, (size_t)11, vl);
  vuint64m2_t shiftT1 = __riscv_vmul_vx_u64m2(sign_powerT1, SHIFT_CONST_F64, vl);
  vuint64m2_t shiftT2 = __riscv_vmul_vx_u64m2(sign_powerT2, SHIFT_CONST_F64, vl);
  vuint64m2_t powT1 = __riscv_vadd_vv_u64m2(__riscv_vrsub_vx_u64m2(shiftT1, FACTOR_DEG_F64, vl), powerT1, vl);
  vuint64m2_t powT2 = __riscv_vadd_vv_u64m2(__riscv_vrsub_vx_u64m2(shiftT2, FACTOR_DEG_F64, vl), powerT2, vl);
  vuint64m2_t factorT1 = __riscv_vadd_vx_u64m2(shiftT1, 1023, vl);
  vuint64m2_t factorT2 = __riscv_vadd_vx_u64m2(shiftT2, 1023, vl);

  vint64m2_t exponents_diff = __riscv_vsub_vv_i64m2(RVVMF_U64_TO_I64(powT1, m2),
                                                    RVVMF_U64_TO_I64(powT2, m2), vl);
  vbool32_t exponents_mask = __riscv_vmsgt_vx_i64m2_b32(exponents_diff, MAGIC_DIFF_F64, vl);
  powT2 = __riscv_vmerge_vxm_u64m2(powT2, 0, exponents_mask, vl);
  factorT2 = __riscv_vmerge_vxm_u64m2(factorT2, 0, exponents_mask, vl);

  update_exponent_f64m2(powT1, th, vl);
  update_exponent_f64m2(powT1, tl, vl);
  update_exponent_f64m2(powT2, th_minus, vl);
  update_exponent_f64m2(powT2, tl_minus, vl);

  update_exponent_f64m2(factorT1, th, vl);
  update_exponent_f64m2(factorT1, tl, vl);
  update_exponent_f64m2(factorT2, th_minus, vl);
  update_exponent_f64m2(factorT2, tl_minus, vl);

  th = __riscv_vfmul_vf_f64m2(th, MUL_COEFF_F64, vl);
  tl = __riscv_vfmul_vf_f64m2(tl, MUL_COEFF_F64, vl);
  th_minus = __riscv_vfmul_vf_f64m2(th_minus, MUL_COEFF_F64, vl);
  tl_minus = __riscv_vfmul_vf_f64m2(tl_minus, MUL_COEFF_F64, vl);

  vfloat64m2_t sum, sumtail;
  add22_vv_f64m2(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat64m2_t diff, difftail;
  sub22_vv_f64m2(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 - T2)even(y) + (T1+T2)odd(y) + (T1-T2)
result = __riscv_vfmadd_vv_f64m2(difftail, eventail, __riscv_vfmul_vf_f64m2(difftail, FACTOR_F64, vl), vl);
  result = __riscv_vfmadd_vv_f64m2(sumtail, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f64m2(even, difftail, result, vl);
  result = __riscv_vfmadd_vv_f64m2(diff, eventail, result, vl);
  result = __riscv_vfmadd_vv_f64m2(sum, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f64m2(sumtail, odd, result, vl);
  result = __riscv_vfmadd_vv_f64m2(diff, even, result, vl);
  result = __riscv_vfmadd_vv_f64m2(sum, odd, result, vl);
  result = __riscv_vfadd_vv_f64m2(result, __riscv_vfmul_vf_f64m2(diff, FACTOR_F64, vl), vl);
  result = __riscv_vfmul_vf_f64m2(result, 2.0, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f64m2(result, special, specialMask, vl);
#endif

  result = __riscv_vmerge_vvm_f64m2(result, x, flag_mask, vl);
  set_sign_f64m2(result, sign, vl);
  return result;
}

vfloat64m4_t __riscv_vsinh_f64m4(vfloat64m4_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e64m4(avl);
  vfloat64m4_t result;
  vuint64m4_t sign;
  get_sign_f64m4(x, sign, vl);

#ifndef __FAST_MATH__
  vfloat64m4_t special;
  vbool16_t specialMask;
  check_special_cases_f64m4(x, special, specialMask, vl);
#endif

  vfloat64m4_t y, th, tl;
  vuint64m4_t powerT1, fi;

  vfloat64m4_t x_minus = __riscv_vfsub_vv_f64m4(__riscv_vfmv_v_f_f64m4(0.0, vl), x, vl);  // -x
  vfloat64m4_t th_minus, tl_minus;
  vuint64m4_t powerT2, fi_minus;

  vfloat64m4_t odd, oddtail, even, eventail;

  argument_reduction_f64m4(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_h_f64m4(x, y, powerT1, fi, vl);

  vbool16_t flag_mask = __riscv_vmand_mm_b16(__riscv_vmflt_vf_f64m4_b16(x, MASK_NUMBER_1_F64, vl),
                                             __riscv_vmfge_vf_f64m4_b16(x, MASK_NUMBER_2_F64, vl), vl);

  y = __riscv_vfmerge_vfm_f64m4(y, 0.0, flag_mask, vl);

  get_table_values_f64m4(fi, th, tl, vl);
  get_table_values_f64m4(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_sinh_f64m4(y, odd, even, oddtail, eventail, vl);

  vuint64m4_t sign_powerT1 = __riscv_vsrl_vx_u64m4(powerT1, (size_t)11, vl);
  vuint64m4_t sign_powerT2 = __riscv_vsrl_vx_u64m4(powerT2, (size_t)11, vl);
  vuint64m4_t shiftT1 = __riscv_vmul_vx_u64m4(sign_powerT1, SHIFT_CONST_F64, vl);
  vuint64m4_t shiftT2 = __riscv_vmul_vx_u64m4(sign_powerT2, SHIFT_CONST_F64, vl);
  vuint64m4_t powT1 = __riscv_vadd_vv_u64m4(__riscv_vrsub_vx_u64m4(shiftT1, FACTOR_DEG_F64, vl), powerT1, vl);
  vuint64m4_t powT2 = __riscv_vadd_vv_u64m4(__riscv_vrsub_vx_u64m4(shiftT2, FACTOR_DEG_F64, vl), powerT2, vl);
  vuint64m4_t factorT1 = __riscv_vadd_vx_u64m4(shiftT1, 1023, vl);
  vuint64m4_t factorT2 = __riscv_vadd_vx_u64m4(shiftT2, 1023, vl);

  vint64m4_t exponents_diff = __riscv_vsub_vv_i64m4(RVVMF_U64_TO_I64(powT1, m4),
                                                    RVVMF_U64_TO_I64(powT2, m4), vl);
  vbool16_t exponents_mask = __riscv_vmsgt_vx_i64m4_b16(exponents_diff, MAGIC_DIFF_F64, vl);
  powT2 = __riscv_vmerge_vxm_u64m4(powT2, 0, exponents_mask, vl);
  factorT2 = __riscv_vmerge_vxm_u64m4(factorT2, 0, exponents_mask, vl);

  update_exponent_f64m4(powT1, th, vl);
  update_exponent_f64m4(powT1, tl, vl);
  update_exponent_f64m4(powT2, th_minus, vl);
  update_exponent_f64m4(powT2, tl_minus, vl);

  update_exponent_f64m4(factorT1, th, vl);
  update_exponent_f64m4(factorT1, tl, vl);
  update_exponent_f64m4(factorT2, th_minus, vl);
  update_exponent_f64m4(factorT2, tl_minus, vl);

  th = __riscv_vfmul_vf_f64m4(th, MUL_COEFF_F64, vl);
  tl = __riscv_vfmul_vf_f64m4(tl, MUL_COEFF_F64, vl);
  th_minus = __riscv_vfmul_vf_f64m4(th_minus, MUL_COEFF_F64, vl);
  tl_minus = __riscv_vfmul_vf_f64m4(tl_minus, MUL_COEFF_F64, vl);

  vfloat64m4_t sum, sumtail;
  add22_vv_f64m4(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat64m4_t diff, difftail;
  sub22_vv_f64m4(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 - T2)even(y) + (T1+T2)odd(y) + (T1-T2)
result = __riscv_vfmadd_vv_f64m4(difftail, eventail, __riscv_vfmul_vf_f64m4(difftail, FACTOR_F64, vl), vl);
  result = __riscv_vfmadd_vv_f64m4(sumtail, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f64m4(even, difftail, result, vl);
  result = __riscv_vfmadd_vv_f64m4(diff, eventail, result, vl);
  result = __riscv_vfmadd_vv_f64m4(sum, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f64m4(sumtail, odd, result, vl);
  result = __riscv_vfmadd_vv_f64m4(diff, even, result, vl);
  result = __riscv_vfmadd_vv_f64m4(sum, odd, result, vl);
  result = __riscv_vfadd_vv_f64m4(result, __riscv_vfmul_vf_f64m4(diff, FACTOR_F64, vl), vl);
  result = __riscv_vfmul_vf_f64m4(result, 2.0, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f64m4(result, special, specialMask, vl);
#endif

  result = __riscv_vmerge_vvm_f64m4(result, x, flag_mask, vl);
  set_sign_f64m4(result, sign, vl);
  return result;
}

vfloat64m8_t __riscv_vsinh_f64m8(vfloat64m8_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e64m4(avl);
  vfloat64m8_t result;
  vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
  x1 = __riscv_vsinh_f64m4(x1, vl);
  result = __riscv_vset_v_f64m4_f64m8(result, 0, x1);
  if (avl > vl) {
    vl = __riscv_vsetvl_e64m4(avl - vl);
    x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
    x1 = __riscv_vsinh_f64m4(x1, vl);
    result = __riscv_vset_v_f64m4_f64m8(result, 1, x1);
  }
  return result;
}

vfloat32m1_t __riscv_vsinh_f32m1(vfloat32m1_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e32m1(avl);
  vfloat32m1_t result;
  vuint32m1_t sign;
  get_sign_f32m1(x, sign, vl);

#ifndef __FAST_MATH__
  vfloat32m1_t special;
  vbool32_t specialMask;
  check_special_cases_f32m1(x, special, specialMask, vl);
#endif

  vfloat32m1_t yh, yl;
  vfloat32m1_t th, tl;
  vuint32m1_t powerT1, fi;

  vfloat32m1_t x_minus = __riscv_vfsub_vv_f32m1(__riscv_vfmv_v_f_f32m1(0.0f, vl), x, vl);  // -x
  vfloat32m1_t th_minus, tl_minus;
  vuint32m1_t powerT2, fi_minus;

  vfloat32m1_t odd, oddtail, even, eventail;

  argument_reduction_f32m1(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_hl_f32m1(x, yh, yl, powerT1, fi, vl);

  vbool32_t flag_mask = __riscv_vmand_mm_b32(__riscv_vmflt_vf_f32m1_b32(x, MASK_NUMBER_1_F32, vl),
                                             __riscv_vmfge_vf_f32m1_b32(x, MASK_NUMBER_2_F32, vl), vl);

  yh = __riscv_vfmerge_vfm_f32m1(yh, 0.0f, flag_mask, vl);

  get_table_values_f32m1(fi, th, tl, vl);
  get_table_values_f32m1(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_sinh_f32m1(yh, yl, odd, even, oddtail, eventail, vl);

  vuint32m1_t sign_powerT1 = __riscv_vsrl_vx_u32m1(powerT1, (size_t)8, vl);
  vuint32m1_t sign_powerT2 = __riscv_vsrl_vx_u32m1(powerT2, (size_t)8, vl);
  vuint32m1_t shiftT1 = __riscv_vmul_vx_u32m1(sign_powerT1, SHIFT_CONST_F32, vl);
  vuint32m1_t shiftT2 = __riscv_vmul_vx_u32m1(sign_powerT2, SHIFT_CONST_F32, vl);
  vuint32m1_t powT1 = __riscv_vadd_vv_u32m1(__riscv_vrsub_vx_u32m1(shiftT1, FACTOR_DEG_F32, vl), powerT1, vl);
  vuint32m1_t powT2 = __riscv_vadd_vv_u32m1(__riscv_vrsub_vx_u32m1(shiftT2, FACTOR_DEG_F32, vl), powerT2, vl);
  vuint32m1_t factorT1 = __riscv_vadd_vx_u32m1(shiftT1, 127, vl);
  vuint32m1_t factorT2 = __riscv_vadd_vx_u32m1(shiftT2, 127, vl);

  vint32m1_t exponents_diff = __riscv_vsub_vv_i32m1(RVVMF_U32_TO_I32(powT1, m1),
                                                    RVVMF_U32_TO_I32(powT2, m1), vl);
  vbool32_t exponents_mask = __riscv_vmand_mm_b32(__riscv_vmsgt_vx_i32m1_b32(exponents_diff, MAGIC_DIFF_SINH_F32, vl),
                                                  __riscv_vmfge_vf_f32m1_b32(x, MASK_NUMBER_3_F32, vl), vl);
  powT2 = __riscv_vmerge_vxm_u32m1(powT2, 0, exponents_mask, vl);
  factorT2 = __riscv_vmerge_vxm_u32m1(factorT2, 0, exponents_mask, vl);

  update_exponent_f32m1(powT1, th, vl);
  update_exponent_f32m1(powT1, tl, vl);
  update_exponent_f32m1(powT2, th_minus, vl);
  update_exponent_f32m1(powT2, tl_minus, vl);

  update_exponent_f32m1(factorT1, th, vl);
  update_exponent_f32m1(factorT1, tl, vl);
  update_exponent_f32m1(factorT2, th_minus, vl);
  update_exponent_f32m1(factorT2, tl_minus, vl);

  th = __riscv_vfmul_vf_f32m1(th, MUL_COEFF_F32, vl);
  tl = __riscv_vfmul_vf_f32m1(tl, MUL_COEFF_F32, vl);
  th_minus = __riscv_vfmul_vf_f32m1(th_minus, MUL_COEFF_F32, vl);
  tl_minus = __riscv_vfmul_vf_f32m1(tl_minus, MUL_COEFF_F32, vl);

  vfloat32m1_t sum, sumtail;
  add22_vv_f32m1(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat32m1_t diff, difftail;
  sub22_vv_f32m1(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 - T2)even(y) + (T1 + T2)odd(y) + (T1 - T2)
  result = __riscv_vfmul_vv_f32m1(difftail, eventail, vl);
  result = __riscv_vfmadd_vv_f32m1(sumtail, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f32m1(even, difftail, result, vl);
  result = __riscv_vfmadd_vv_f32m1(diff, eventail, result, vl);
  result = __riscv_vfmadd_vv_f32m1(sum, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f32m1(sumtail, odd, result, vl);

  fma12_vv_f32m1(diff, even, result, th, tl, vl);

  fma12_vv_f32m1(sum, odd, th, th_minus, tl_minus, vl);
  tl_minus = __riscv_vfadd_vv_f32m1(tl_minus, tl, vl);

  add21_vv_f32m1(__riscv_vfmul_vf_f32m1(diff, FACTOR_F32, vl), __riscv_vfmul_vf_f32m1(difftail, FACTOR_F32, vl), th_minus, tl_minus, result, vl);
  result = __riscv_vfmul_vf_f32m1(result, 2.0f, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f32m1(result, special, specialMask, vl);
#endif

  result = __riscv_vmerge_vvm_f32m1(result, x, flag_mask, vl);
  set_sign_f32m1(result, sign, vl);
  return result;
}

vfloat32m2_t __riscv_vsinh_f32m2(vfloat32m2_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e32m2(avl);
  vfloat32m2_t result;
  vuint32m2_t sign;
  get_sign_f32m2(x, sign, vl);

#ifndef __FAST_MATH__
  vfloat32m2_t special;
  vbool16_t specialMask;
  check_special_cases_f32m2(x, special, specialMask, vl);
#endif

  vfloat32m2_t yh, yl;
  vfloat32m2_t th, tl;
  vuint32m2_t powerT1, fi;

  vfloat32m2_t x_minus = __riscv_vfsub_vv_f32m2(__riscv_vfmv_v_f_f32m2(0.0f, vl), x, vl);  // -x
  vfloat32m2_t th_minus, tl_minus;
  vuint32m2_t powerT2, fi_minus;

  vfloat32m2_t odd, oddtail, even, eventail;

  argument_reduction_f32m2(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_hl_f32m2(x, yh, yl, powerT1, fi, vl);

  vbool16_t flag_mask = __riscv_vmand_mm_b16(__riscv_vmflt_vf_f32m2_b16(x, MASK_NUMBER_1_F32, vl),
                                             __riscv_vmfge_vf_f32m2_b16(x, MASK_NUMBER_2_F32, vl), vl);

  yh = __riscv_vfmerge_vfm_f32m2(yh, 0.0f, flag_mask, vl);

  get_table_values_f32m2(fi, th, tl, vl);
  get_table_values_f32m2(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_sinh_f32m2(yh, yl, odd, even, oddtail, eventail, vl);

  vuint32m2_t sign_powerT1 = __riscv_vsrl_vx_u32m2(powerT1, (size_t)8, vl);
  vuint32m2_t sign_powerT2 = __riscv_vsrl_vx_u32m2(powerT2, (size_t)8, vl);
  vuint32m2_t shiftT1 = __riscv_vmul_vx_u32m2(sign_powerT1, SHIFT_CONST_F32, vl);
  vuint32m2_t shiftT2 = __riscv_vmul_vx_u32m2(sign_powerT2, SHIFT_CONST_F32, vl);
  vuint32m2_t powT1 = __riscv_vadd_vv_u32m2(__riscv_vrsub_vx_u32m2(shiftT1, FACTOR_DEG_F32, vl), powerT1, vl);
  vuint32m2_t powT2 = __riscv_vadd_vv_u32m2(__riscv_vrsub_vx_u32m2(shiftT2, FACTOR_DEG_F32, vl), powerT2, vl);
  vuint32m2_t factorT1 = __riscv_vadd_vx_u32m2(shiftT1, 127, vl);
  vuint32m2_t factorT2 = __riscv_vadd_vx_u32m2(shiftT2, 127, vl);

  vint32m2_t exponents_diff = __riscv_vsub_vv_i32m2(RVVMF_U32_TO_I32(powT1, m2),
                                                    RVVMF_U32_TO_I32(powT2, m2), vl);
  vbool16_t exponents_mask = __riscv_vmand_mm_b16(__riscv_vmsgt_vx_i32m2_b16(exponents_diff, MAGIC_DIFF_SINH_F32, vl),
                                                  __riscv_vmfge_vf_f32m2_b16(x, MASK_NUMBER_3_F32, vl), vl);
  powT2 = __riscv_vmerge_vxm_u32m2(powT2, 0, exponents_mask, vl);
  factorT2 = __riscv_vmerge_vxm_u32m2(factorT2, 0, exponents_mask, vl);

  update_exponent_f32m2(powT1, th, vl);
  update_exponent_f32m2(powT1, tl, vl);
  update_exponent_f32m2(powT2, th_minus, vl);
  update_exponent_f32m2(powT2, tl_minus, vl);

  update_exponent_f32m2(factorT1, th, vl);
  update_exponent_f32m2(factorT1, tl, vl);
  update_exponent_f32m2(factorT2, th_minus, vl);
  update_exponent_f32m2(factorT2, tl_minus, vl);

  th = __riscv_vfmul_vf_f32m2(th, MUL_COEFF_F32, vl);
  tl = __riscv_vfmul_vf_f32m2(tl, MUL_COEFF_F32, vl);
  th_minus = __riscv_vfmul_vf_f32m2(th_minus, MUL_COEFF_F32, vl);
  tl_minus = __riscv_vfmul_vf_f32m2(tl_minus, MUL_COEFF_F32, vl);

  vfloat32m2_t sum, sumtail;
  add22_vv_f32m2(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat32m2_t diff, difftail;
  sub22_vv_f32m2(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 - T2)even(y) + (T1 + T2)odd(y) + (T1 - T2)
  result = __riscv_vfmul_vv_f32m2(difftail, eventail, vl);
  result = __riscv_vfmadd_vv_f32m2(sumtail, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f32m2(even, difftail, result, vl);
  result = __riscv_vfmadd_vv_f32m2(diff, eventail, result, vl);
  result = __riscv_vfmadd_vv_f32m2(sum, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f32m2(sumtail, odd, result, vl);

  fma12_vv_f32m2(diff, even, result, th, tl, vl);

  fma12_vv_f32m2(sum, odd, th, th_minus, tl_minus, vl);
  tl_minus = __riscv_vfadd_vv_f32m2(tl_minus, tl, vl);

  add21_vv_f32m2(__riscv_vfmul_vf_f32m2(diff, FACTOR_F32, vl), __riscv_vfmul_vf_f32m2(difftail, FACTOR_F32, vl), th_minus, tl_minus, result, vl);
  result = __riscv_vfmul_vf_f32m2(result, 2.0f, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f32m2(result, special, specialMask, vl);
#endif

  result = __riscv_vmerge_vvm_f32m2(result, x, flag_mask, vl);
  set_sign_f32m2(result, sign, vl);
  return result;
}

vfloat32m4_t __riscv_vsinh_f32m4(vfloat32m4_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e32m4(avl);
  vfloat32m4_t result;
  vuint32m4_t sign;
  get_sign_f32m4(x, sign, vl);

#ifndef __FAST_MATH__
  vfloat32m4_t special;
  vbool8_t specialMask;
  check_special_cases_f32m4(x, special, specialMask, vl);
#endif

  vfloat32m4_t yh, yl;
  vfloat32m4_t th, tl;
  vuint32m4_t powerT1, fi;

  vfloat32m4_t x_minus = __riscv_vfsub_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.0f, vl), x, vl);  // -x
  vfloat32m4_t th_minus, tl_minus;
  vuint32m4_t powerT2, fi_minus;

  vfloat32m4_t odd, oddtail, even, eventail;

  argument_reduction_f32m4(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_hl_f32m4(x, yh, yl, powerT1, fi, vl);

  vbool8_t flag_mask = __riscv_vmand_mm_b8(__riscv_vmflt_vf_f32m4_b8(x, MASK_NUMBER_1_F32, vl),
                                           __riscv_vmfge_vf_f32m4_b8(x, MASK_NUMBER_2_F32, vl), vl);

  yh = __riscv_vfmerge_vfm_f32m4(yh, 0.0f, flag_mask, vl);

  get_table_values_f32m4(fi, th, tl, vl);
  get_table_values_f32m4(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_sinh_f32m4(yh, yl, odd, even, oddtail, eventail, vl);

  vuint32m4_t sign_powerT1 = __riscv_vsrl_vx_u32m4(powerT1, (size_t)8, vl);
  vuint32m4_t sign_powerT2 = __riscv_vsrl_vx_u32m4(powerT2, (size_t)8, vl);
  vuint32m4_t shiftT1 = __riscv_vmul_vx_u32m4(sign_powerT1, SHIFT_CONST_F32, vl);
  vuint32m4_t shiftT2 = __riscv_vmul_vx_u32m4(sign_powerT2, SHIFT_CONST_F32, vl);
  vuint32m4_t powT1 = __riscv_vadd_vv_u32m4(__riscv_vrsub_vx_u32m4(shiftT1, FACTOR_DEG_F32, vl), powerT1, vl);
  vuint32m4_t powT2 = __riscv_vadd_vv_u32m4(__riscv_vrsub_vx_u32m4(shiftT2, FACTOR_DEG_F32, vl), powerT2, vl);
  vuint32m4_t factorT1 = __riscv_vadd_vx_u32m4(shiftT1, 127, vl);
  vuint32m4_t factorT2 = __riscv_vadd_vx_u32m4(shiftT2, 127, vl);

  vint32m4_t exponents_diff = __riscv_vsub_vv_i32m4(RVVMF_U32_TO_I32(powT1, m4),
                                                    RVVMF_U32_TO_I32(powT2, m4), vl);
  vbool8_t exponents_mask = __riscv_vmand_mm_b8(__riscv_vmsgt_vx_i32m4_b8(exponents_diff, MAGIC_DIFF_SINH_F32, vl),
                                                __riscv_vmfge_vf_f32m4_b8(x, MASK_NUMBER_3_F32, vl), vl);
  powT2 = __riscv_vmerge_vxm_u32m4(powT2, 0, exponents_mask, vl);
  factorT2 = __riscv_vmerge_vxm_u32m4(factorT2, 0, exponents_mask, vl);

  update_exponent_f32m4(powT1, th, vl);
  update_exponent_f32m4(powT1, tl, vl);
  update_exponent_f32m4(powT2, th_minus, vl);
  update_exponent_f32m4(powT2, tl_minus, vl);

  update_exponent_f32m4(factorT1, th, vl);
  update_exponent_f32m4(factorT1, tl, vl);
  update_exponent_f32m4(factorT2, th_minus, vl);
  update_exponent_f32m4(factorT2, tl_minus, vl);

  th = __riscv_vfmul_vf_f32m4(th, MUL_COEFF_F32, vl);
  tl = __riscv_vfmul_vf_f32m4(tl, MUL_COEFF_F32, vl);
  th_minus = __riscv_vfmul_vf_f32m4(th_minus, MUL_COEFF_F32, vl);
  tl_minus = __riscv_vfmul_vf_f32m4(tl_minus, MUL_COEFF_F32, vl);

  vfloat32m4_t sum, sumtail;
  add22_vv_f32m4(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat32m4_t diff, difftail;
  sub22_vv_f32m4(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 - T2)even(y) + (T1 + T2)odd(y) + (T1 - T2)
  result = __riscv_vfmul_vv_f32m4(difftail, eventail, vl);
  result = __riscv_vfmadd_vv_f32m4(sumtail, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f32m4(even, difftail, result, vl);
  result = __riscv_vfmadd_vv_f32m4(diff, eventail, result, vl);
  result = __riscv_vfmadd_vv_f32m4(sum, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f32m4(sumtail, odd, result, vl);

  fma12_vv_f32m4(diff, even, result, th, tl, vl);

  fma12_vv_f32m4(sum, odd, th, th_minus, tl_minus, vl);
  tl_minus = __riscv_vfadd_vv_f32m4(tl_minus, tl, vl);

  add21_vv_f32m4(__riscv_vfmul_vf_f32m4(diff, FACTOR_F32, vl), __riscv_vfmul_vf_f32m4(difftail, FACTOR_F32, vl), th_minus, tl_minus, result, vl);
  result = __riscv_vfmul_vf_f32m4(result, 2.0f, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f32m4(result, special, specialMask, vl);
#endif

  result = __riscv_vmerge_vvm_f32m4(result, x, flag_mask, vl);
  set_sign_f32m4(result, sign, vl);
  return result;
}

vfloat32m8_t __riscv_vsinh_f32m8(vfloat32m8_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e32m4(avl);
  vfloat32m8_t result;
  vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
  x1 = __riscv_vsinh_f32m4(x1, vl);
  result = __riscv_vset_v_f32m4_f32m8(result, 0, x1);
  if (avl > vl) {
    vl = __riscv_vsetvl_e32m4(avl - vl);
    x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
    x1 = __riscv_vsinh_f32m4(x1, vl);
    result = __riscv_vset_v_f32m4_f32m8(result, 1, x1);
  }
  return result;
}

vfloat64m1_t __riscv_vcosh_f64m1(vfloat64m1_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e64m1(avl);
  vfloat64m1_t result;
  get_abs_f64m1(x, vl);

#ifndef __FAST_MATH__
  vfloat64m1_t special;
  vbool64_t specialMask;
  check_special_cases_f64m1(x, special, specialMask, vl);
#endif

  vfloat64m1_t y, th, tl;
  vuint64m1_t powerT1, fi;

  vfloat64m1_t x_minus = __riscv_vfsub_vv_f64m1(__riscv_vfmv_v_f_f64m1(0.0, vl), x, vl);  // -x
  vfloat64m1_t th_minus, tl_minus;
  vuint64m1_t powerT2, fi_minus;

  vfloat64m1_t odd, even;

  argument_reduction_f64m1(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_h_f64m1(x, y, powerT1, fi, vl);

  get_table_values_f64m1(fi, th, tl, vl);
  get_table_values_f64m1(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_cosh_f64m1(y, odd, even, vl);

  vuint64m1_t sign_powerT1 = __riscv_vsrl_vx_u64m1(powerT1, 11, vl);
  vuint64m1_t sign_powerT2 = __riscv_vsrl_vx_u64m1(powerT2, 11, vl);
  vuint64m1_t shiftT1 = __riscv_vmul_vx_u64m1(sign_powerT1, SHIFT_CONST_F64, vl);
  vuint64m1_t shiftT2 = __riscv_vmul_vx_u64m1(sign_powerT2, SHIFT_CONST_F64, vl);
  vuint64m1_t powT1 = __riscv_vadd_vv_u64m1(__riscv_vrsub_vx_u64m1(shiftT1, FACTOR_DEG_F64, vl), powerT1, vl);
  vuint64m1_t powT2 = __riscv_vadd_vv_u64m1(__riscv_vrsub_vx_u64m1(shiftT2, FACTOR_DEG_F64, vl), powerT2, vl);
  vuint64m1_t factorT1 = __riscv_vadd_vx_u64m1(shiftT1, 1023, vl);
  vuint64m1_t factorT2 = __riscv_vadd_vx_u64m1(shiftT2, 1023, vl);

  vint64m1_t exponents_diff = __riscv_vsub_vv_i64m1(RVVMF_U64_TO_I64(powT1, m1),
                                                    RVVMF_U64_TO_I64(powT2, m1), vl);
  vbool64_t exponents_mask = __riscv_vmsgt_vx_i64m1_b64(exponents_diff, MAGIC_DIFF_F64, vl);
  powT2 = __riscv_vmerge_vxm_u64m1(powT2, 0, exponents_mask, vl);
  factorT2 = __riscv_vmerge_vxm_u64m1(factorT2, 0, exponents_mask, vl);

  update_exponent_f64m1(powT1, th, vl);
  update_exponent_f64m1(powT1, tl, vl);
  update_exponent_f64m1(powT2, th_minus, vl);
  update_exponent_f64m1(powT2, tl_minus, vl);

  update_exponent_f64m1(factorT1, th, vl);
  update_exponent_f64m1(factorT1, tl, vl);
  update_exponent_f64m1(factorT2, th_minus, vl);
  update_exponent_f64m1(factorT2, tl_minus, vl);

  th = __riscv_vfmul_vf_f64m1(th, MUL_COEFF_F64, vl);
  tl = __riscv_vfmul_vf_f64m1(tl, MUL_COEFF_F64, vl);
  th_minus = __riscv_vfmul_vf_f64m1(th_minus, MUL_COEFF_F64, vl);
  tl_minus = __riscv_vfmul_vf_f64m1(tl_minus, MUL_COEFF_F64, vl);

  vfloat64m1_t sum, sumtail;
  add22_vv_f64m1(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat64m1_t diff, difftail;
  sub22_vv_f64m1(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 + T2)even(y) + (T1 - T2)odd(y) + (T1 + T2)
  result = __riscv_vfmadd_vv_f64m1(difftail, odd, __riscv_vfmul_vf_f64m1(sumtail, FACTOR_F64, vl), vl);
  result = __riscv_vfmadd_vv_f64m1(sumtail, even, result, vl);
  result = __riscv_vfmadd_vv_f64m1(diff, odd, result, vl);
  result = __riscv_vfmadd_vv_f64m1(sum, even, result, vl);
  result = __riscv_vfadd_vv_f64m1(result, __riscv_vfmul_vf_f64m1(sum, FACTOR_F64, vl), vl);
  result = __riscv_vfmul_vf_f64m1(result, 2.0, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f64m1(result, special, specialMask, vl);
#endif

  return result;
}

vfloat64m2_t __riscv_vcosh_f64m2(vfloat64m2_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e64m2(avl);
  vfloat64m2_t result;
  get_abs_f64m2(x, vl);

#ifndef __FAST_MATH__
  vfloat64m2_t special;
  vbool32_t specialMask;
  check_special_cases_f64m2(x, special, specialMask, vl);
#endif

  vfloat64m2_t y, th, tl;
  vuint64m2_t powerT1, fi;

  vfloat64m2_t x_minus = __riscv_vfsub_vv_f64m2(__riscv_vfmv_v_f_f64m2(0.0, vl), x, vl);  // -x
  vfloat64m2_t th_minus, tl_minus;
  vuint64m2_t powerT2, fi_minus;

  vfloat64m2_t odd, even;

  argument_reduction_f64m2(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_h_f64m2(x, y, powerT1, fi, vl);

  get_table_values_f64m2(fi, th, tl, vl);
  get_table_values_f64m2(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_cosh_f64m2(y, odd, even, vl);

  vuint64m2_t sign_powerT1 = __riscv_vsrl_vx_u64m2(powerT1, 11, vl);
  vuint64m2_t sign_powerT2 = __riscv_vsrl_vx_u64m2(powerT2, 11, vl);
  vuint64m2_t shiftT1 = __riscv_vmul_vx_u64m2(sign_powerT1, SHIFT_CONST_F64, vl);
  vuint64m2_t shiftT2 = __riscv_vmul_vx_u64m2(sign_powerT2, SHIFT_CONST_F64, vl);
  vuint64m2_t powT1 = __riscv_vadd_vv_u64m2(__riscv_vrsub_vx_u64m2(shiftT1, FACTOR_DEG_F64, vl), powerT1, vl);
  vuint64m2_t powT2 = __riscv_vadd_vv_u64m2(__riscv_vrsub_vx_u64m2(shiftT2, FACTOR_DEG_F64, vl), powerT2, vl);
  vuint64m2_t factorT1 = __riscv_vadd_vx_u64m2(shiftT1, 1023, vl);
  vuint64m2_t factorT2 = __riscv_vadd_vx_u64m2(shiftT2, 1023, vl);

  vint64m2_t exponents_diff = __riscv_vsub_vv_i64m2(RVVMF_U64_TO_I64(powT1, m2),
                                                    RVVMF_U64_TO_I64(powT2, m2), vl);
  vbool32_t exponents_mask = __riscv_vmsgt_vx_i64m2_b32(exponents_diff, MAGIC_DIFF_F64, vl);
  powT2 = __riscv_vmerge_vxm_u64m2(powT2, 0, exponents_mask, vl);
  factorT2 = __riscv_vmerge_vxm_u64m2(factorT2, 0, exponents_mask, vl);

  update_exponent_f64m2(powT1, th, vl);
  update_exponent_f64m2(powT1, tl, vl);
  update_exponent_f64m2(powT2, th_minus, vl);
  update_exponent_f64m2(powT2, tl_minus, vl);

  update_exponent_f64m2(factorT1, th, vl);
  update_exponent_f64m2(factorT1, tl, vl);
  update_exponent_f64m2(factorT2, th_minus, vl);
  update_exponent_f64m2(factorT2, tl_minus, vl);

  th = __riscv_vfmul_vf_f64m2(th, MUL_COEFF_F64, vl);
  tl = __riscv_vfmul_vf_f64m2(tl, MUL_COEFF_F64, vl);
  th_minus = __riscv_vfmul_vf_f64m2(th_minus, MUL_COEFF_F64, vl);
  tl_minus = __riscv_vfmul_vf_f64m2(tl_minus, MUL_COEFF_F64, vl);

  vfloat64m2_t sum, sumtail;
  add22_vv_f64m2(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat64m2_t diff, difftail;
  sub22_vv_f64m2(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 + T2)even(y) + (T1 - T2)odd(y) + (T1 + T2)
  result = __riscv_vfmadd_vv_f64m2(difftail, odd, __riscv_vfmul_vf_f64m2(sumtail, FACTOR_F64, vl), vl);
  result = __riscv_vfmadd_vv_f64m2(sumtail, even, result, vl);
  result = __riscv_vfmadd_vv_f64m2(diff, odd, result, vl);
  result = __riscv_vfmadd_vv_f64m2(sum, even, result, vl);
  result = __riscv_vfadd_vv_f64m2(result, __riscv_vfmul_vf_f64m2(sum, FACTOR_F64, vl), vl);
  result = __riscv_vfmul_vf_f64m2(result, 2.0, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f64m2(result, special, specialMask, vl);
#endif

  return result;
}

vfloat64m4_t __riscv_vcosh_f64m4(vfloat64m4_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e64m4(avl);
  vfloat64m4_t result;
  get_abs_f64m4(x, vl);

#ifndef __FAST_MATH__
  vfloat64m4_t special;
  vbool16_t specialMask;
  check_special_cases_f64m4(x, special, specialMask, vl);
#endif

  vfloat64m4_t y, th, tl;
  vuint64m4_t powerT1, fi;

  vfloat64m4_t x_minus = __riscv_vfsub_vv_f64m4(__riscv_vfmv_v_f_f64m4(0.0, vl), x, vl);  // -x
  vfloat64m4_t th_minus, tl_minus;
  vuint64m4_t powerT2, fi_minus;

  vfloat64m4_t odd, even;

  argument_reduction_f64m4(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_h_f64m4(x, y, powerT1, fi, vl);

  get_table_values_f64m4(fi, th, tl, vl);
  get_table_values_f64m4(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_cosh_f64m4(y, odd, even, vl);

  vuint64m4_t sign_powerT1 = __riscv_vsrl_vx_u64m4(powerT1, 11, vl);
  vuint64m4_t sign_powerT2 = __riscv_vsrl_vx_u64m4(powerT2, 11, vl);
  vuint64m4_t shiftT1 = __riscv_vmul_vx_u64m4(sign_powerT1, SHIFT_CONST_F64, vl);
  vuint64m4_t shiftT2 = __riscv_vmul_vx_u64m4(sign_powerT2, SHIFT_CONST_F64, vl);
  vuint64m4_t powT1 = __riscv_vadd_vv_u64m4(__riscv_vrsub_vx_u64m4(shiftT1, FACTOR_DEG_F64, vl), powerT1, vl);
  vuint64m4_t powT2 = __riscv_vadd_vv_u64m4(__riscv_vrsub_vx_u64m4(shiftT2, FACTOR_DEG_F64, vl), powerT2, vl);
  vuint64m4_t factorT1 = __riscv_vadd_vx_u64m4(shiftT1, 1023, vl);
  vuint64m4_t factorT2 = __riscv_vadd_vx_u64m4(shiftT2, 1023, vl);

  vint64m4_t exponents_diff = __riscv_vsub_vv_i64m4(RVVMF_U64_TO_I64(powT1, m4),
                                                    RVVMF_U64_TO_I64(powT2, m4), vl);
  vbool16_t exponents_mask = __riscv_vmsgt_vx_i64m4_b16(exponents_diff, MAGIC_DIFF_F64, vl);
  powT2 = __riscv_vmerge_vxm_u64m4(powT2, 0, exponents_mask, vl);
  factorT2 = __riscv_vmerge_vxm_u64m4(factorT2, 0, exponents_mask, vl);

  update_exponent_f64m4(powT1, th, vl);
  update_exponent_f64m4(powT1, tl, vl);
  update_exponent_f64m4(powT2, th_minus, vl);
  update_exponent_f64m4(powT2, tl_minus, vl);

  update_exponent_f64m4(factorT1, th, vl);
  update_exponent_f64m4(factorT1, tl, vl);
  update_exponent_f64m4(factorT2, th_minus, vl);
  update_exponent_f64m4(factorT2, tl_minus, vl);

  th = __riscv_vfmul_vf_f64m4(th, MUL_COEFF_F64, vl);
  tl = __riscv_vfmul_vf_f64m4(tl, MUL_COEFF_F64, vl);
  th_minus = __riscv_vfmul_vf_f64m4(th_minus, MUL_COEFF_F64, vl);
  tl_minus = __riscv_vfmul_vf_f64m4(tl_minus, MUL_COEFF_F64, vl);

  vfloat64m4_t sum, sumtail;
  add22_vv_f64m4(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat64m4_t diff, difftail;
  sub22_vv_f64m4(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 + T2)even(y) + (T1 - T2)odd(y) + (T1 + T2)
  result = __riscv_vfmadd_vv_f64m4(difftail, odd, __riscv_vfmul_vf_f64m4(sumtail, FACTOR_F64, vl), vl);
  result = __riscv_vfmadd_vv_f64m4(sumtail, even, result, vl);
  result = __riscv_vfmadd_vv_f64m4(diff, odd, result, vl);
  result = __riscv_vfmadd_vv_f64m4(sum, even, result, vl);
  result = __riscv_vfadd_vv_f64m4(result, __riscv_vfmul_vf_f64m4(sum, FACTOR_F64, vl), vl);
  result = __riscv_vfmul_vf_f64m4(result, 2.0, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f64m4(result, special, specialMask, vl);
#endif

  return result;
}

vfloat64m8_t __riscv_vcosh_f64m8(vfloat64m8_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e64m4(avl);
  vfloat64m8_t result;
  vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
  x1 = __riscv_vcosh_f64m4(x1, vl);
  result = __riscv_vset_v_f64m4_f64m8(result, 0, x1);
  if (avl > vl) {
    vl = __riscv_vsetvl_e64m4(avl - vl);
    x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
    x1 = __riscv_vcosh_f64m4(x1, vl);
    result = __riscv_vset_v_f64m4_f64m8(result, 1, x1);
  }
  return result;
}

vfloat32m1_t __riscv_vcosh_f32m1(vfloat32m1_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e32m1(avl);
  vfloat32m1_t result;
  get_abs_f32m1(x, vl);

#ifndef __FAST_MATH__
  vfloat32m1_t special;
  vbool32_t specialMask;
  check_special_cases_f32m1(x, special, specialMask, vl);
#endif

  vfloat32m1_t y;
  vfloat32m1_t th, tl;
  vuint32m1_t powerT1, fi;

  vfloat32m1_t x_minus = __riscv_vfsub_vv_f32m1(__riscv_vfmv_v_f_f32m1(0.0f, vl), x, vl);  // -x
  vfloat32m1_t th_minus, tl_minus;
  vuint32m1_t powerT2, fi_minus;

  vfloat32m1_t odd, even;

  argument_reduction_f32m1(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_h_f32m1(x, y, powerT1, fi, vl);

  get_table_values_f32m1(fi, th, tl, vl);
  get_table_values_f32m1(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_cosh_f32m1(y, odd, even, vl);

  vuint32m1_t sign_powerT1 = __riscv_vsrl_vx_u32m1(powerT1, (size_t)8, vl);
  vuint32m1_t sign_powerT2 = __riscv_vsrl_vx_u32m1(powerT2, (size_t)8, vl);
  vuint32m1_t shiftT1 = __riscv_vmul_vx_u32m1(sign_powerT1, SHIFT_CONST_F32, vl);
  vuint32m1_t shiftT2 = __riscv_vmul_vx_u32m1(sign_powerT2, SHIFT_CONST_F32, vl);
  vuint32m1_t powT1 = __riscv_vadd_vv_u32m1(__riscv_vrsub_vx_u32m1(shiftT1, FACTOR_DEG_F32, vl), powerT1, vl);
  vuint32m1_t powT2 = __riscv_vadd_vv_u32m1(__riscv_vrsub_vx_u32m1(shiftT2, FACTOR_DEG_F32, vl), powerT2, vl);
  vuint32m1_t factorT1 = __riscv_vadd_vx_u32m1(shiftT1, 127, vl);
  vuint32m1_t factorT2 = __riscv_vadd_vx_u32m1(shiftT2, 127, vl);

  vint32m1_t exponents_diff = __riscv_vsub_vv_i32m1(RVVMF_U32_TO_I32(powT1, m1),
                                                    RVVMF_U32_TO_I32(powT2, m1), vl);
  vbool32_t exponents_mask = __riscv_vmsgt_vx_i32m1_b32(exponents_diff, MAGIC_DIFF_COSH_F32, vl);
  powT2 = __riscv_vmerge_vxm_u32m1(powT2, 0, exponents_mask, vl);
  factorT2 = __riscv_vmerge_vxm_u32m1(factorT2, 0, exponents_mask, vl);

  update_exponent_f32m1(powT1, th, vl);
  update_exponent_f32m1(powT1, tl, vl);
  update_exponent_f32m1(powT2, th_minus, vl);
  update_exponent_f32m1(powT2, tl_minus, vl);

  update_exponent_f32m1(factorT1, th, vl);
  update_exponent_f32m1(factorT1, tl, vl);
  update_exponent_f32m1(factorT2, th_minus, vl);
  update_exponent_f32m1(factorT2, tl_minus, vl);

  th = __riscv_vfmul_vf_f32m1(th, MUL_COEFF_F32, vl);
  tl = __riscv_vfmul_vf_f32m1(tl, MUL_COEFF_F32, vl);
  th_minus = __riscv_vfmul_vf_f32m1(th_minus, MUL_COEFF_F32, vl);
  tl_minus = __riscv_vfmul_vf_f32m1(tl_minus, MUL_COEFF_F32, vl);

  vfloat32m1_t sum, sumtail;
  add22_vv_f32m1(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat32m1_t diff, difftail;
  sub22_vv_f32m1(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 + T2)even(y) + (T1 - T2)odd(y) + (T1 + T2)
  result = __riscv_vfmadd_vv_f32m1(difftail, odd, __riscv_vfmul_vf_f32m1(sumtail, FACTOR_F32, vl), vl);
  result = __riscv_vfmadd_vv_f32m1(sumtail, even, result, vl);
  result = __riscv_vfmadd_vv_f32m1(diff, odd, result, vl);
  result = __riscv_vfmadd_vv_f32m1(sum, even, result, vl);
  result = __riscv_vfadd_vv_f32m1(result, __riscv_vfmul_vf_f32m1(sum, FACTOR_F32, vl), vl);
  result = __riscv_vfmul_vf_f32m1(result, 2.0f, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f32m1(result, special, specialMask, vl);
#endif

  return result;
}

vfloat32m2_t __riscv_vcosh_f32m2(vfloat32m2_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e32m2(avl);
  vfloat32m2_t result;
  get_abs_f32m2(x, vl);

#ifndef __FAST_MATH__
  vfloat32m2_t special;
  vbool16_t specialMask;
  check_special_cases_f32m2(x, special, specialMask, vl);
#endif

  vfloat32m2_t y;
  vfloat32m2_t th, tl;
  vuint32m2_t powerT1, fi;

  vfloat32m2_t x_minus = __riscv_vfsub_vv_f32m2(__riscv_vfmv_v_f_f32m2(0.0f, vl), x, vl);  // -x
  vfloat32m2_t th_minus, tl_minus;
  vuint32m2_t powerT2, fi_minus;

  vfloat32m2_t odd, even;

  argument_reduction_f32m2(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_h_f32m2(x, y, powerT1, fi, vl);

  get_table_values_f32m2(fi, th, tl, vl);
  get_table_values_f32m2(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_cosh_f32m2(y, odd, even, vl);

  vuint32m2_t sign_powerT1 = __riscv_vsrl_vx_u32m2(powerT1, (size_t)8, vl);
  vuint32m2_t sign_powerT2 = __riscv_vsrl_vx_u32m2(powerT2, (size_t)8, vl);
  vuint32m2_t shiftT1 = __riscv_vmul_vx_u32m2(sign_powerT1, SHIFT_CONST_F32, vl);
  vuint32m2_t shiftT2 = __riscv_vmul_vx_u32m2(sign_powerT2, SHIFT_CONST_F32, vl);
  vuint32m2_t powT1 = __riscv_vadd_vv_u32m2(__riscv_vrsub_vx_u32m2(shiftT1, FACTOR_DEG_F32, vl), powerT1, vl);
  vuint32m2_t powT2 = __riscv_vadd_vv_u32m2(__riscv_vrsub_vx_u32m2(shiftT2, FACTOR_DEG_F32, vl), powerT2, vl);
  vuint32m2_t factorT1 = __riscv_vadd_vx_u32m2(shiftT1, 127, vl);
  vuint32m2_t factorT2 = __riscv_vadd_vx_u32m2(shiftT2, 127, vl);

  vint32m2_t exponents_diff = __riscv_vsub_vv_i32m2(RVVMF_U32_TO_I32(powT1, m2),
                                                    RVVMF_U32_TO_I32(powT2, m2), vl);
  vbool16_t exponents_mask = __riscv_vmsgt_vx_i32m2_b16(exponents_diff, MAGIC_DIFF_COSH_F32, vl);
  powT2 = __riscv_vmerge_vxm_u32m2(powT2, 0, exponents_mask, vl);
  factorT2 = __riscv_vmerge_vxm_u32m2(factorT2, 0, exponents_mask, vl);

  update_exponent_f32m2(powT1, th, vl);
  update_exponent_f32m2(powT1, tl, vl);
  update_exponent_f32m2(powT2, th_minus, vl);
  update_exponent_f32m2(powT2, tl_minus, vl);

  update_exponent_f32m2(factorT1, th, vl);
  update_exponent_f32m2(factorT1, tl, vl);
  update_exponent_f32m2(factorT2, th_minus, vl);
  update_exponent_f32m2(factorT2, tl_minus, vl);

  th = __riscv_vfmul_vf_f32m2(th, MUL_COEFF_F32, vl);
  tl = __riscv_vfmul_vf_f32m2(tl, MUL_COEFF_F32, vl);
  th_minus = __riscv_vfmul_vf_f32m2(th_minus, MUL_COEFF_F32, vl);
  tl_minus = __riscv_vfmul_vf_f32m2(tl_minus, MUL_COEFF_F32, vl);

  vfloat32m2_t sum, sumtail;
  add22_vv_f32m2(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat32m2_t diff, difftail;
  sub22_vv_f32m2(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 + T2)even(y) + (T1 - T2)odd(y) + (T1 + T2)
  result = __riscv_vfmadd_vv_f32m2(difftail, odd, __riscv_vfmul_vf_f32m2(sumtail, FACTOR_F32, vl), vl);
  result = __riscv_vfmadd_vv_f32m2(sumtail, even, result, vl);
  result = __riscv_vfmadd_vv_f32m2(diff, odd, result, vl);
  result = __riscv_vfmadd_vv_f32m2(sum, even, result, vl);
  result = __riscv_vfadd_vv_f32m2(result, __riscv_vfmul_vf_f32m2(sum, FACTOR_F32, vl), vl);
  result = __riscv_vfmul_vf_f32m2(result, 2.0f, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f32m2(result, special, specialMask, vl);
#endif

  return result;
}

vfloat32m4_t __riscv_vcosh_f32m4(vfloat32m4_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e32m4(avl);
  vfloat32m4_t result;
  get_abs_f32m4(x, vl);

#ifndef __FAST_MATH__
  vfloat32m4_t special;
  vbool8_t specialMask;
  check_special_cases_f32m4(x, special, specialMask, vl);
#endif

  vfloat32m4_t y;
  vfloat32m4_t th, tl;
  vuint32m4_t powerT1, fi;

  vfloat32m4_t x_minus = __riscv_vfsub_vv_f32m4(__riscv_vfmv_v_f_f32m4(0.0f, vl), x, vl);  // -x
  vfloat32m4_t th_minus, tl_minus;
  vuint32m4_t powerT2, fi_minus;

  vfloat32m4_t odd, even;

  argument_reduction_f32m4(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_h_f32m4(x, y, powerT1, fi, vl);

  get_table_values_f32m4(fi, th, tl, vl);
  get_table_values_f32m4(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_cosh_f32m4(y, odd, even, vl);

  vuint32m4_t sign_powerT1 = __riscv_vsrl_vx_u32m4(powerT1, (size_t)8, vl);
  vuint32m4_t sign_powerT2 = __riscv_vsrl_vx_u32m4(powerT2, (size_t)8, vl);
  vuint32m4_t shiftT1 = __riscv_vmul_vx_u32m4(sign_powerT1, SHIFT_CONST_F32, vl);
  vuint32m4_t shiftT2 = __riscv_vmul_vx_u32m4(sign_powerT2, SHIFT_CONST_F32, vl);
  vuint32m4_t powT1 = __riscv_vadd_vv_u32m4(__riscv_vrsub_vx_u32m4(shiftT1, FACTOR_DEG_F32, vl), powerT1, vl);
  vuint32m4_t powT2 = __riscv_vadd_vv_u32m4(__riscv_vrsub_vx_u32m4(shiftT2, FACTOR_DEG_F32, vl), powerT2, vl);
  vuint32m4_t factorT1 = __riscv_vadd_vx_u32m4(shiftT1, 127, vl);
  vuint32m4_t factorT2 = __riscv_vadd_vx_u32m4(shiftT2, 127, vl);

  vint32m4_t exponents_diff = __riscv_vsub_vv_i32m4(RVVMF_U32_TO_I32(powT1, m4),
                                                    RVVMF_U32_TO_I32(powT2, m4), vl);
  vbool8_t exponents_mask = __riscv_vmsgt_vx_i32m4_b8(exponents_diff, MAGIC_DIFF_COSH_F32, vl);
  powT2 = __riscv_vmerge_vxm_u32m4(powT2, 0, exponents_mask, vl);
  factorT2 = __riscv_vmerge_vxm_u32m4(factorT2, 0, exponents_mask, vl);

  update_exponent_f32m4(powT1, th, vl);
  update_exponent_f32m4(powT1, tl, vl);
  update_exponent_f32m4(powT2, th_minus, vl);
  update_exponent_f32m4(powT2, tl_minus, vl);

  update_exponent_f32m4(factorT1, th, vl);
  update_exponent_f32m4(factorT1, tl, vl);
  update_exponent_f32m4(factorT2, th_minus, vl);
  update_exponent_f32m4(factorT2, tl_minus, vl);

  th = __riscv_vfmul_vf_f32m4(th, MUL_COEFF_F32, vl);
  tl = __riscv_vfmul_vf_f32m4(tl, MUL_COEFF_F32, vl);
  th_minus = __riscv_vfmul_vf_f32m4(th_minus, MUL_COEFF_F32, vl);
  tl_minus = __riscv_vfmul_vf_f32m4(tl_minus, MUL_COEFF_F32, vl);

  vfloat32m4_t sum, sumtail;
  add22_vv_f32m4(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat32m4_t diff, difftail;
  sub22_vv_f32m4(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 + T2)even(y) + (T1 - T2)odd(y) + (T1 + T2)
  result = __riscv_vfmadd_vv_f32m4(difftail, odd, __riscv_vfmul_vf_f32m4(sumtail, FACTOR_F32, vl), vl);
  result = __riscv_vfmadd_vv_f32m4(sumtail, even, result, vl);
  result = __riscv_vfmadd_vv_f32m4(diff, odd, result, vl);
  result = __riscv_vfmadd_vv_f32m4(sum, even, result, vl);
  result = __riscv_vfadd_vv_f32m4(result, __riscv_vfmul_vf_f32m4(sum, FACTOR_F32, vl), vl);
  result = __riscv_vfmul_vf_f32m4(result, 2.0f, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f32m4(result, special, specialMask, vl);
#endif

  return result;
}

vfloat32m8_t __riscv_vcosh_f32m8(vfloat32m8_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e32m4(avl);
  vfloat32m8_t result;
  vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
  x1 = __riscv_vcosh_f32m4(x1, vl);
  result = __riscv_vset_v_f32m4_f32m8(result, 0, x1);
  if (avl > vl) {
    vl = __riscv_vsetvl_e32m4(avl - vl);
    x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
    x1 = __riscv_vcosh_f32m4(x1, vl);
    result = __riscv_vset_v_f32m4_f32m8(result, 1, x1);
  }
  return result;
}

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
#include "sinh_cosh_half.inl"

vfloat16m1_t __riscv_vsinh_f16m1(vfloat16m1_t x, size_t avl){
  size_t vl = __riscv_vsetvl_e16m1(avl);
  vfloat16m1_t result;
  vuint16m1_t sign;
  get_sign_f16m1(x, sign, vl);

#ifndef __FAST_MATH__
  vfloat16m1_t special;
  vbool16_t specialMask;
  check_special_cases_f16m1(x, special, specialMask, vl);
#endif

  vfloat16m1_t yh, yl;
  vfloat16m1_t th, tl;
  vuint16m1_t powerT1, fi;

  vfloat16m1_t x_minus = __riscv_vfsub_vv_f16m1(__riscv_vfmv_v_f_f16m1(0.0f16, vl), x, vl);  // -x
  vfloat16m1_t th_minus, tl_minus;
  vuint16m1_t powerT2, fi_minus;

  vfloat16m1_t odd, oddtail, even, eventail;

  argument_reduction_f16m1(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_hl_f16m1(x, yh, yl, powerT1, fi, vl);

  get_table_values_f16m1(fi, th, tl, vl);
  get_table_values_f16m1(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_f16m1(yh, yl, odd, even, oddtail, eventail, vl);

  update_exponent_f16m1(powerT1, th, 13, vl);
  update_exponent_f16m1(powerT1, tl, 13, vl);
  update_exponent_f16m1(powerT2, th_minus, 17, vl);
  update_exponent_f16m1(powerT2, tl_minus, 17, vl);
  th_minus = __riscv_vfmul_vf_f16m1(th_minus, 0.0625f16, vl);
  tl_minus = __riscv_vfmul_vf_f16m1(tl_minus, 0.0625f16, vl);

  vfloat16m1_t sum, sumtail;
  add22_vv_f16m1(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat16m1_t diff, difftail;
  sub22_vv_f16m1(th, tl, th_minus, tl_minus, diff, difftail, vl);

  even = __riscv_vfmul_vf_f16m1(even, 2.0f16, vl);
  eventail = __riscv_vfmul_vf_f16m1(eventail, 2.0f16, vl);
  odd = __riscv_vfmul_vf_f16m1(odd, 2.0f16, vl);
  oddtail = __riscv_vfmul_vf_f16m1(oddtail, 2.0f16, vl);

  //(T1 - T2)even(y) + (T1+T2)odd(y) + (T1-T2)
  result = __riscv_vfmul_vv_f16m1(difftail, eventail, vl);
  result = __riscv_vfmadd_vv_f16m1(sumtail, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f16m1(even, difftail, result, vl);
  result = __riscv_vfmadd_vv_f16m1(diff, eventail, result, vl);
  result = __riscv_vfmadd_vv_f16m1(sum, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f16m1(sumtail, odd, result, vl);

  fma12_vv_f16m1(diff, even, result, th, tl, vl);

  fma12_vv_f16m1(sum, odd, th, th_minus, tl_minus, vl);
  tl_minus = __riscv_vfadd_vv_f16m1(tl_minus, tl, vl);

  add21_2_vv_f16m1(diff, difftail, th_minus, tl_minus, result, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f16m1(result, special, specialMask, vl);
#endif

  set_sign_f16m1(result, sign, vl);
  return result;
}

vfloat16m2_t __riscv_vsinh_f16m2(vfloat16m2_t x, size_t avl){
  size_t vl = __riscv_vsetvl_e16m2(avl);
  vfloat16m2_t result;
  vuint16m2_t sign;
  get_sign_f16m2(x, sign, vl);

#ifndef __FAST_MATH__
  vfloat16m2_t special;
  vbool8_t specialMask;
  check_special_cases_f16m2(x, special, specialMask, vl);
#endif

  vfloat16m2_t yh, yl;
  vfloat16m2_t th, tl;
  vuint16m2_t powerT1, fi;

  vfloat16m2_t x_minus = __riscv_vfsub_vv_f16m2(__riscv_vfmv_v_f_f16m2(0.0f16, vl), x, vl);  // -x
  vfloat16m2_t th_minus, tl_minus;
  vuint16m2_t powerT2, fi_minus;

  vfloat16m2_t odd, oddtail, even, eventail;

  argument_reduction_f16m2(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_hl_f16m2(x, yh, yl, powerT1, fi, vl);

  get_table_values_f16m2(fi, th, tl, vl);
  get_table_values_f16m2(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_f16m2(yh, yl, odd, even, oddtail, eventail, vl);

  update_exponent_f16m2(powerT1, th, 13, vl);
  update_exponent_f16m2(powerT1, tl, 13, vl);
  update_exponent_f16m2(powerT2, th_minus, 17, vl);
  update_exponent_f16m2(powerT2, tl_minus, 17, vl);

  th_minus = __riscv_vfmul_vf_f16m2(th_minus, 0.0625f16, vl);
  tl_minus = __riscv_vfmul_vf_f16m2(tl_minus, 0.0625f16, vl);

  vfloat16m2_t sum, sumtail;
  add22_vv_f16m2(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat16m2_t diff, difftail;
  sub22_vv_f16m2(th, tl, th_minus, tl_minus, diff, difftail, vl);

  even = __riscv_vfmul_vf_f16m2(even, 2.0f16, vl);
  eventail = __riscv_vfmul_vf_f16m2(eventail, 2.0f16, vl);
  odd = __riscv_vfmul_vf_f16m2(odd, 2.0f16, vl);
  oddtail = __riscv_vfmul_vf_f16m2(oddtail, 2.0f16, vl);

  //(T1 - T2)even(y) + (T1+T2)odd(y) + (T1-T2)
  result = __riscv_vfmul_vv_f16m2(difftail, eventail, vl);
  result = __riscv_vfmadd_vv_f16m2(sumtail, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f16m2(even, difftail, result, vl);
  result = __riscv_vfmadd_vv_f16m2(diff, eventail, result, vl);
  result = __riscv_vfmadd_vv_f16m2(sum, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f16m2(sumtail, odd, result, vl);

  fma12_vv_f16m2(diff, even, result, th, tl, vl);

  fma12_vv_f16m2(sum, odd, th, th_minus, tl_minus, vl);
  tl_minus = __riscv_vfadd_vv_f16m2(tl_minus, tl, vl);

  add21_2_vv_f16m2(diff, difftail, th_minus, tl_minus, result, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f16m2(result, special, specialMask, vl);
#endif

  set_sign_f16m2(result, sign, vl);
  return result;
}

vfloat16m4_t __riscv_vsinh_f16m4(vfloat16m4_t x, size_t avl){
  size_t vl = __riscv_vsetvl_e16m4(avl);
  vfloat16m4_t result;
  vuint16m4_t sign;
  get_sign_f16m4(x, sign, vl);

#ifndef __FAST_MATH__
  vfloat16m4_t special;
  vbool4_t specialMask;
  check_special_cases_f16m4(x, special, specialMask, vl);
#endif

  vfloat16m4_t yh, yl;
  vfloat16m4_t th, tl;
  vuint16m4_t powerT1, fi;

  vfloat16m4_t x_minus = __riscv_vfsub_vv_f16m4(__riscv_vfmv_v_f_f16m4(0.0f16, vl), x, vl);  // -x
  vfloat16m4_t th_minus, tl_minus;
  vuint16m4_t powerT2, fi_minus;

  vfloat16m4_t odd, oddtail, even, eventail;

  argument_reduction_f16m4(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_hl_f16m4(x, yh, yl, powerT1, fi, vl);

  get_table_values_f16m4(fi, th, tl, vl);
  get_table_values_f16m4(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_f16m4(yh, yl, odd, even, oddtail, eventail, vl);

  update_exponent_f16m4(powerT1, th, 13, vl);
  update_exponent_f16m4(powerT1, tl, 13, vl);
  update_exponent_f16m4(powerT2, th_minus, 17, vl);
  update_exponent_f16m4(powerT2, tl_minus, 17, vl);

  th_minus = __riscv_vfmul_vf_f16m4(th_minus, 0.0625f16, vl);
  tl_minus = __riscv_vfmul_vf_f16m4(tl_minus, 0.0625f16, vl);

  vfloat16m4_t sum, sumtail;
  add22_vv_f16m4(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat16m4_t diff, difftail;
  sub22_vv_f16m4(th, tl, th_minus, tl_minus, diff, difftail, vl);

  even = __riscv_vfmul_vf_f16m4(even, 2.0f16, vl);
  eventail = __riscv_vfmul_vf_f16m4(eventail, 2.0f16, vl);
  odd = __riscv_vfmul_vf_f16m4(odd, 2.0f16, vl);
  oddtail = __riscv_vfmul_vf_f16m4(oddtail, 2.0f16, vl);

  // (T1 - T2)even(y) + (T1+T2)odd(y) + (T1-T2)
  result = __riscv_vfmul_vv_f16m4(difftail, eventail, vl);
  result = __riscv_vfmadd_vv_f16m4(sumtail, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f16m4(even, difftail, result, vl);
  result = __riscv_vfmadd_vv_f16m4(diff, eventail, result, vl);
  result = __riscv_vfmadd_vv_f16m4(sum, oddtail, result, vl);
  result = __riscv_vfmadd_vv_f16m4(sumtail, odd, result, vl);

  fma12_vv_f16m4(diff, even, result, th, tl, vl);

  fma12_vv_f16m4(sum, odd, th, th_minus, tl_minus, vl);
  tl_minus = __riscv_vfadd_vv_f16m4(tl_minus, tl, vl);

  add21_2_vv_f16m4(diff, difftail, th_minus, tl_minus, result, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f16m4(result, special, specialMask, vl);
#endif

  set_sign_f16m4(result, sign, vl);
  return result;
}

vfloat16m8_t __riscv_vsinh_f16m8(vfloat16m8_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e16m4(avl);
  vfloat16m8_t result;
  vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
  x1 = __riscv_vsinh_f16m4(x1, vl);
  result = __riscv_vset_v_f16m4_f16m8(result, 0, x1);
  if (avl > vl) {
    vl = __riscv_vsetvl_e16m4(avl - vl);
    x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
    x1 = __riscv_vsinh_f16m4(x1, vl);
    result = __riscv_vset_v_f16m4_f16m8(result, 1, x1);
  }
  return result;
}

vfloat16m1_t __riscv_vcosh_f16m1(vfloat16m1_t x, size_t avl){
  size_t vl = __riscv_vsetvl_e16m1(avl);
  vfloat16m1_t result;
  get_abs_f16m1(x, vl);

#ifndef __FAST_MATH__
  vfloat16m1_t special;
  vbool16_t specialMask;
  check_special_cases_f16m1(x, special, specialMask, vl);
#endif

  vfloat16m1_t yh, yl;
  vfloat16m1_t th, tl;
  vuint16m1_t powerT1, fi;

  vfloat16m1_t x_minus = __riscv_vfsub_vv_f16m1(__riscv_vfmv_v_f_f16m1(0.0f16, vl), x, vl);  // -x
  vfloat16m1_t th_minus, tl_minus;
  vuint16m1_t powerT2, fi_minus;

  vfloat16m1_t odd, even, oddtail, eventail;

  argument_reduction_f16m1(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_hl_f16m1(x, yh, yl, powerT1, fi, vl);

  get_table_values_f16m1(fi, th, tl, vl);
  get_table_values_f16m1(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_f16m1(yh, yl, odd, even, oddtail, eventail, vl);

  update_exponent_f16m1(powerT1, th, 13, vl);
  update_exponent_f16m1(powerT1, tl, 13, vl);
  update_exponent_f16m1(powerT2, th_minus, 17, vl);
  update_exponent_f16m1(powerT2, tl_minus, 17, vl);
  th_minus = __riscv_vfmul_vf_f16m1(th_minus, 0.0625f16, vl);
  tl_minus = __riscv_vfmul_vf_f16m1(tl_minus, 0.0625f16, vl);

  vfloat16m1_t sum, sumtail;
  add22_vv_f16m1(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat16m1_t diff, difftail;
  sub22_vv_f16m1(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 + T2)even(y) + (T1 - T2)odd(y) + (T1 + T2)
  result = __riscv_vfmul_vv_f16m1(difftail, oddtail, vl);
  result = __riscv_vfmadd_vv_f16m1(sumtail, eventail, result, vl);
  result = __riscv_vfmadd_vv_f16m1(even, sumtail, result, vl);
  result = __riscv_vfmadd_vv_f16m1(odd, difftail, result, vl);
  result = __riscv_vfmadd_vv_f16m1(oddtail, diff, result, vl);
  result = __riscv_vfmadd_vv_f16m1(eventail, sum, result, vl);
  result = __riscv_vfmadd_vv_f16m1(even, sum, result, vl);

  fma12_vv_f16m1(odd, diff, result, th, tl, vl);
  add21_vv_f16m1(sum, sumtail, th, tl, result, vl);
  result = __riscv_vfmul_vf_f16m1(result, 2.0f16, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f16m1(result, special, specialMask, vl);
#endif

  return result;
}

vfloat16m2_t __riscv_vcosh_f16m2(vfloat16m2_t x, size_t avl){
  size_t vl = __riscv_vsetvl_e16m2(avl);
  vfloat16m2_t result;
  get_abs_f16m2(x, vl);

#ifndef __FAST_MATH__
  vfloat16m2_t special;
  vbool8_t specialMask;
  check_special_cases_f16m2(x, special, specialMask, vl);
#endif

  vfloat16m2_t yh, yl;
  vfloat16m2_t th, tl;
  vuint16m2_t powerT1, fi;

  vfloat16m2_t x_minus = __riscv_vfsub_vv_f16m2(__riscv_vfmv_v_f_f16m2(0.0f16, vl), x, vl);  // -x
  vfloat16m2_t th_minus, tl_minus;
  vuint16m2_t powerT2, fi_minus;

  vfloat16m2_t odd, even, oddtail, eventail;

  argument_reduction_f16m2(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_hl_f16m2(x, yh, yl, powerT1, fi, vl);

  get_table_values_f16m2(fi, th, tl, vl);
  get_table_values_f16m2(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_f16m2(yh, yl, odd, even, oddtail, eventail, vl);

  update_exponent_f16m2(powerT1, th, 13, vl);
  update_exponent_f16m2(powerT1, tl, 13, vl);
  update_exponent_f16m2(powerT2, th_minus, 17, vl);
  update_exponent_f16m2(powerT2, tl_minus, 17, vl);
  th_minus = __riscv_vfmul_vf_f16m2(th_minus, 0.0625f16, vl);
  tl_minus = __riscv_vfmul_vf_f16m2(tl_minus, 0.0625f16, vl);

  vfloat16m2_t sum, sumtail;
  add22_vv_f16m2(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat16m2_t diff, difftail;
  sub22_vv_f16m2(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 + T2)even(y) + (T1 - T2)odd(y) + (T1 + T2)
  result = __riscv_vfmul_vv_f16m2(difftail, oddtail, vl);
  result = __riscv_vfmadd_vv_f16m2(sumtail, eventail, result, vl);
  result = __riscv_vfmadd_vv_f16m2(even, sumtail, result, vl);
  result = __riscv_vfmadd_vv_f16m2(odd, difftail, result, vl);
  result = __riscv_vfmadd_vv_f16m2(oddtail, diff, result, vl);
  result = __riscv_vfmadd_vv_f16m2(eventail, sum, result, vl);
  result = __riscv_vfmadd_vv_f16m2(even, sum, result, vl);

  fma12_vv_f16m2(odd, diff, result, th, tl, vl);
  add21_vv_f16m2(sum, sumtail, th, tl, result, vl);
  result = __riscv_vfmul_vf_f16m2(result, 2.0f16, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f16m2(result, special, specialMask, vl);
#endif

  return result;
}

vfloat16m4_t __riscv_vcosh_f16m4(vfloat16m4_t x, size_t avl){
  size_t vl = __riscv_vsetvl_e16m4(avl);
  vfloat16m4_t result;
  get_abs_f16m4(x, vl);

#ifndef __FAST_MATH__
  vfloat16m4_t special;
  vbool4_t specialMask;
  check_special_cases_f16m4(x, special, specialMask, vl);
#endif

  vfloat16m4_t yh, yl;
  vfloat16m4_t th, tl;
  vuint16m4_t powerT1, fi;

  vfloat16m4_t x_minus = __riscv_vfsub_vv_f16m4(__riscv_vfmv_v_f_f16m4(0.0f16, vl), x, vl);  // -x
  vfloat16m4_t th_minus, tl_minus;
  vuint16m4_t powerT2, fi_minus;

  vfloat16m4_t odd, even, oddtail, eventail;

  argument_reduction_f16m4(x_minus, powerT2, fi_minus, th_minus, vl); // th_minus for "h-argument". overwrites later
  argument_reduction_hl_f16m4(x, yh, yl, powerT1, fi, vl);

  get_table_values_f16m4(fi, th, tl, vl);
  get_table_values_f16m4(fi_minus, th_minus, tl_minus, vl);

  calculate_polynom_f16m4(yh, yl, odd, even, oddtail, eventail, vl);

  update_exponent_f16m4(powerT1, th, 13, vl);
  update_exponent_f16m4(powerT1, tl, 13, vl);
  update_exponent_f16m4(powerT2, th_minus, 17, vl);
  update_exponent_f16m4(powerT2, tl_minus, 17, vl);
  th_minus = __riscv_vfmul_vf_f16m4(th_minus, 0.0625f16, vl);
  tl_minus = __riscv_vfmul_vf_f16m4(tl_minus, 0.0625f16, vl);

  vfloat16m4_t sum, sumtail;
  add22_vv_f16m4(th, tl, th_minus, tl_minus, sum, sumtail, vl);
  vfloat16m4_t diff, difftail;
  sub22_vv_f16m4(th, tl, th_minus, tl_minus, diff, difftail, vl);

  // (T1 + T2)even(y) + (T1 - T2)odd(y) + (T1 + T2)
  result = __riscv_vfmul_vv_f16m4(difftail, oddtail, vl);
  result = __riscv_vfmadd_vv_f16m4(sumtail, eventail, result, vl);
  result = __riscv_vfmadd_vv_f16m4(even, sumtail, result, vl);
  result = __riscv_vfmadd_vv_f16m4(odd, difftail, result, vl);
  result = __riscv_vfmadd_vv_f16m4(oddtail, diff, result, vl);
  result = __riscv_vfmadd_vv_f16m4(eventail, sum, result, vl);
  result = __riscv_vfmadd_vv_f16m4(even, sum, result, vl);

  fma12_vv_f16m4(odd, diff, result, th, tl, vl);
  add21_vv_f16m4(sum, sumtail, th, tl, result, vl);
  result = __riscv_vfmul_vf_f16m4(result, 2.0f16, vl);

#ifndef __FAST_MATH__
  result = __riscv_vmerge_vvm_f16m4(result, special, specialMask, vl);
#endif

  return result;
}

vfloat16m8_t __riscv_vcosh_f16m8(vfloat16m8_t x, size_t avl) {
  size_t vl = __riscv_vsetvl_e16m4(avl);
  vfloat16m8_t result;
  vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
  x1 = __riscv_vcosh_f16m4(x1, vl);
  result = __riscv_vset_v_f16m4_f16m8(result, 0, x1);
  if (avl > vl) {
    vl = __riscv_vsetvl_e16m4(avl - vl);
    x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
    x1 = __riscv_vcosh_f16m4(x1, vl);
    result = __riscv_vset_v_f16m4_f16m8(result, 1, x1);
  }
  return result;
}

#endif

#endif
