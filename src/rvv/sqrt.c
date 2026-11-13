/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025-2026 The RVVMF Authors (Alexander Sysoyev)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  sqrt.c                                       *
 *   Contains: intrinsic function sqrt for f64, f32, f16 *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input VL number of elements in vector register        *
 *                                                       *
 * Return value: square root of the elements of vector V *
 *                                                       *
 * Algorithms:                                           *
 *   1) Goldschmidt's algorithm                          *
 *   2) Fast inverse square root                         *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

vfloat16m1_t __riscv_vsqrt_f16m1(vfloat16m1_t x, size_t vl)
{
  vuint16m1_t xi = __riscv_vreinterpret_v_f16m1_u16m1(x);
  vuint16m1_t x_int = __riscv_vand_vx_u16m1(xi, 0x7fff, vl);
  vbool16_t zero_mask = __riscv_vmseq_vx_u16m1_b16(x_int, 0, vl);
    
#ifndef __FAST_MATH__
  unsigned short inf_si = 0x7c00; 
  _Float16 inf_f16 = *(_Float16*)(&inf_si);
  
  vbool16_t zero_less_mask = __riscv_vmsgtu_vx_u16m1_b16(xi, 0x8000, vl);
  vbool16_t inf_mask = __riscv_vmseq_vx_u16m1_b16(x_int, 0x7c00, vl);
  vbool16_t nan_mask = __riscv_vmsgtu_vx_u16m1_b16(x_int, 0x7c00, vl);
  vbool16_t mask_sNaN = __riscv_vmand_mm_b16(__riscv_vmsltu_vx_u16m1_b16 (x_int, 0x7e00, vl), nan_mask, vl);
  mask_sNaN = __riscv_vmor_mm_b16(mask_sNaN, zero_less_mask, vl);
  
  vbool16_t special_mask = __riscv_vmor_mm_b16(zero_less_mask, nan_mask, vl);
  vfloat16m1_t x_spec = __riscv_vfmerge_vfm_f16m1(x, 0, 
                             __riscv_vmor_mm_b16(special_mask, inf_mask, vl), vl);

  vfloat16m1_t one = __riscv_vfmv_v_f_f16m1(1.0f16, vl);
  vbool16_t denormal_mask = __riscv_vmsltu_vx_u16m1_b16(x_int, 0x0400, vl);
  vfloat16m1_t denorm_mult = 
    __riscv_vfmerge_vfm_f16m1(one, 0x1.0p+12f16, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f16m1(x_spec, denorm_mult, vl);
  
  x_int = __riscv_vreinterpret_v_f16m1_u16m1(x_spec);
#endif

  vuint16m1_t mantissa_in_x = __riscv_vand_vx_u16m1(x_int, 0x03ff, vl);
  vuint16m1_t order_in_x = __riscv_vand_vx_u16m1(x_int, 0x7c00, vl);
  order_in_x = __riscv_vsrl_vx_u16m1(order_in_x, 10, vl);

  vuint16m1_t reduced_x_int = __riscv_vor_vx_u16m1(mantissa_in_x, 0x3c00, vl);
  vfloat16m1_t reduced_x = __riscv_vreinterpret_v_u16m1_f16m1(reduced_x_int);

  vuint16m1_t low_ind_bit = __riscv_vand_vx_u16m1(order_in_x, 0x1, vl);

  vfloat16m1_t y0 = __riscv_vreinterpret_v_u16m1_f16m1(
    __riscv_vrsub_vx_u16m1(__riscv_vsrl_vx_u16m1(reduced_x_int, 1, vl), 0x59d8, vl));

  vfloat16m1_t xx = __riscv_vfmul_vv_f16m1(y0, reduced_x, vl);
  vfloat16m1_t h = __riscv_vfmul_vf_f16m1(y0, 0.5f16, vl);
  vfloat16m1_t r = __riscv_vfrsub_vf_f16m1(
    __riscv_vfmul_vv_f16m1(xx, h, vl), 0.5f16, vl);

  xx = __riscv_vfmacc_vv_f16m1(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f16m1(h, h, r, vl);
  r = __riscv_vfrsub_vf_f16m1(__riscv_vfmul_vv_f16m1(xx, h, vl), 0.5f16, vl);
  xx = __riscv_vfmacc_vv_f16m1(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f16m1(h, h, r, vl);
  
  r = __riscv_vfnmsac_vv_f16m1(reduced_x, xx, xx, vl);
  
  vfloat16m1_t zh = __riscv_vfmacc_vv_f16m1(xx, r, h, vl); // high-part of sqrt value
  vfloat16m1_t zl = __riscv_vfmacc_vv_f16m1(__riscv_vfsub_vv_f16m1(xx, zh, vl), r, h, vl);// low-part of sqrt value
    
  vuint16m1_t new_exp = __riscv_vadd_vx_u16m1(order_in_x, 15, vl);
  new_exp = __riscv_vsrl_vx_u16m1(new_exp, 1, vl);
  vfloat16m1_t order_exp = __riscv_vreinterpret_v_u16m1_f16m1(
                           __riscv_vsll_vx_u16m1(new_exp, 10, vl));
  
  vbool16_t is_odd = __riscv_vmseq_vx_u16m1_b16(low_ind_bit, 1, vl);
  vfloat16m1_t order_low = __riscv_vfmerge_vfm_f16m1(
    __riscv_vfmv_v_f_f16m1(0x1.6ap+0f16, vl), 1.0f16, is_odd, vl);
  vfloat16m1_t order_low_ = __riscv_vfmerge_vfm_f16m1(
    __riscv_vfmv_v_f_f16m1(0x1.3ccp-13f16, vl), 0.0f16, is_odd, vl);
  vuint16m1_t red_exp = __riscv_vrsub_vx_u16m1(low_ind_bit, 16, vl);
  vfloat16m1_t order_reduce = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vsll_vx_u16m1(red_exp, 10, vl));
   
  vfloat16m1_t sh = __riscv_vfmul_vv_f16m1(zh, order_low, vl);
  vfloat16m1_t sl = __riscv_vfmsac_vv_f16m1(sh, zh, order_low, vl);

  sl = __riscv_vfmacc_vv_f16m1(sl, zh, order_low_, vl);
  sl = __riscv_vfmacc_vv_f16m1(sl, zl, order_low, vl);
  
  vfloat16m1_t sqrt_value = __riscv_vfadd_vv_f16m1(sh, sl, vl);
  
  reduced_x = __riscv_vfmul_vv_f16m1(reduced_x, order_reduce, vl);
  
  h = __riscv_vfmul_vf_f16m1(h, 0.5f16, vl);
  r = __riscv_vfnmsac_vv_f16m1(reduced_x, sqrt_value, sqrt_value, vl);
  r = __riscv_vfmul_vv_f16m1(r, order_low, vl);
  
  sqrt_value = __riscv_vfmacc_vv_f16m1(sqrt_value, r, h, vl);
  sqrt_value = __riscv_vfmul_vv_f16m1(sqrt_value, order_exp, vl);

#ifndef __FAST_MATH__
  sqrt_value = __riscv_vfmerge_vfm_f16m1(sqrt_value, inf_f16, inf_mask, vl);
  
  sqrt_value = __riscv_vmerge_vvm_f16m1(sqrt_value, x, nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
  if (issNaN) {
    volatile _Float16 x1 = 0.0f16 / 0.0f16;
    sqrt_value = __riscv_vfmerge_vfm_f16m1(sqrt_value, x1, mask_sNaN, vl);
  }
    
  denorm_mult = __riscv_vfmerge_vfm_f16m1(one, 0x1.0p-6f16, denormal_mask, vl);
  sqrt_value = __riscv_vfmul_vv_f16m1(sqrt_value, denorm_mult, vl);
#endif

  sqrt_value = __riscv_vmerge_vvm_f16m1(sqrt_value, x, zero_mask, vl);
  return sqrt_value; 
}

vfloat16m2_t __riscv_vsqrt_f16m2(vfloat16m2_t x, size_t vl)
{
  vuint16m2_t xi = __riscv_vreinterpret_v_f16m2_u16m2(x);
  vuint16m2_t x_int = __riscv_vand_vx_u16m2(xi, 0x7fff, vl);
  vbool8_t zero_mask = __riscv_vmseq_vx_u16m2_b8(x_int, 0, vl);
  
#ifndef __FAST_MATH__
  unsigned short inf_si = 0x7c00; 
  _Float16 inf_f16 = *(_Float16*)(&inf_si);
  
  vbool8_t zero_less_mask = __riscv_vmsgtu_vx_u16m2_b8(xi, 0x8000, vl);
  vbool8_t inf_mask = __riscv_vmseq_vx_u16m2_b8(x_int, 0x7c00, vl);
  vbool8_t nan_mask = __riscv_vmsgtu_vx_u16m2_b8(x_int, 0x7c00, vl);
  vbool8_t mask_sNaN = __riscv_vmand_mm_b8(__riscv_vmsltu_vx_u16m2_b8 (x_int, 0x7e00, vl), nan_mask, vl);
  mask_sNaN = __riscv_vmor_mm_b8(mask_sNaN, zero_less_mask, vl);
  
  vbool8_t special_mask = __riscv_vmor_mm_b8(zero_less_mask, nan_mask, vl);
  vfloat16m2_t x_spec = __riscv_vfmerge_vfm_f16m2(x, 0, 
                             __riscv_vmor_mm_b8(special_mask, inf_mask, vl), vl);

  vfloat16m2_t one = __riscv_vfmv_v_f_f16m2(1.0f16, vl);
  vbool8_t denormal_mask = __riscv_vmsltu_vx_u16m2_b8(x_int, 0x0400, vl);
  vfloat16m2_t denorm_mult = 
    __riscv_vfmerge_vfm_f16m2(one, 0x1.0p+12f16, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f16m2(x_spec, denorm_mult, vl);
  
  x_int = __riscv_vreinterpret_v_f16m2_u16m2(x_spec);
#endif

  vuint16m2_t mantissa_in_x = __riscv_vand_vx_u16m2(x_int, 0x03ff, vl);
  vuint16m2_t order_in_x = __riscv_vand_vx_u16m2(x_int, 0x7c00, vl);
  order_in_x = __riscv_vsrl_vx_u16m2(order_in_x, 10, vl);

  vuint16m2_t reduced_x_int = __riscv_vor_vx_u16m2(mantissa_in_x, 0x3c00, vl);
  vfloat16m2_t reduced_x = __riscv_vreinterpret_v_u16m2_f16m2(reduced_x_int);

  vuint16m2_t low_ind_bit = __riscv_vand_vx_u16m2(order_in_x, 0x1, vl);

  vfloat16m2_t y0 = __riscv_vreinterpret_v_u16m2_f16m2(
    __riscv_vrsub_vx_u16m2(__riscv_vsrl_vx_u16m2(reduced_x_int, 1, vl), 0x59d8, vl));

  vfloat16m2_t xx = __riscv_vfmul_vv_f16m2(y0, reduced_x, vl);
  vfloat16m2_t h = __riscv_vfmul_vf_f16m2(y0, 0.5f16, vl);
  vfloat16m2_t r = __riscv_vfrsub_vf_f16m2(
    __riscv_vfmul_vv_f16m2(xx, h, vl), 0.5f16, vl);

  xx = __riscv_vfmacc_vv_f16m2(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f16m2(h, h, r, vl);
  r = __riscv_vfrsub_vf_f16m2(__riscv_vfmul_vv_f16m2(xx, h, vl), 0.5f16, vl);
  xx = __riscv_vfmacc_vv_f16m2(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f16m2(h, h, r, vl);
  
  r = __riscv_vfnmsac_vv_f16m2(reduced_x, xx, xx, vl);
  
  vfloat16m2_t zh = __riscv_vfmacc_vv_f16m2(xx, r, h, vl); // high-part of sqrt value
  vfloat16m2_t zl = __riscv_vfmacc_vv_f16m2(__riscv_vfsub_vv_f16m2(xx, zh, vl), r, h, vl);// low-part of sqrt value
    
  vuint16m2_t new_exp = __riscv_vadd_vx_u16m2(order_in_x, 15, vl);
  new_exp = __riscv_vsrl_vx_u16m2(new_exp, 1, vl);
  vfloat16m2_t order_exp = __riscv_vreinterpret_v_u16m2_f16m2(
                           __riscv_vsll_vx_u16m2(new_exp, 10, vl));
  
  vbool8_t is_odd = __riscv_vmseq_vx_u16m2_b8(low_ind_bit, 1, vl);
  vfloat16m2_t order_low = __riscv_vfmerge_vfm_f16m2(
    __riscv_vfmv_v_f_f16m2(0x1.6ap+0f16, vl), 1.0f16, is_odd, vl);
  vfloat16m2_t order_low_ = __riscv_vfmerge_vfm_f16m2(
    __riscv_vfmv_v_f_f16m2(0x1.3ccp-13f16, vl), 0.0f16, is_odd, vl);
  vuint16m2_t red_exp = __riscv_vrsub_vx_u16m2(low_ind_bit, 16, vl);
  vfloat16m2_t order_reduce = __riscv_vreinterpret_v_u16m2_f16m2(
                                __riscv_vsll_vx_u16m2(red_exp, 10, vl));
   
  vfloat16m2_t sh = __riscv_vfmul_vv_f16m2(zh, order_low, vl);
  vfloat16m2_t sl = __riscv_vfmsac_vv_f16m2(sh, zh, order_low, vl);

  sl = __riscv_vfmacc_vv_f16m2(sl, zh, order_low_, vl);
  sl = __riscv_vfmacc_vv_f16m2(sl, zl, order_low, vl);
  
  vfloat16m2_t sqrt_value = __riscv_vfadd_vv_f16m2(sh, sl, vl);
  
  reduced_x = __riscv_vfmul_vv_f16m2(reduced_x, order_reduce, vl);
  
  h = __riscv_vfmul_vf_f16m2(h, 0.5f16, vl);
  r = __riscv_vfnmsac_vv_f16m2(reduced_x, sqrt_value, sqrt_value, vl);
  r = __riscv_vfmul_vv_f16m2(r, order_low, vl);
  
  sqrt_value = __riscv_vfmacc_vv_f16m2(sqrt_value, r, h, vl);
  sqrt_value = __riscv_vfmul_vv_f16m2(sqrt_value, order_exp, vl);

#ifndef __FAST_MATH__
  sqrt_value = __riscv_vfmerge_vfm_f16m2(sqrt_value, inf_f16, inf_mask, vl);
  
  sqrt_value = __riscv_vmerge_vvm_f16m2(sqrt_value, x, nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b8(mask_sNaN, vl);
  if (issNaN) {
    volatile _Float16 x1 = 0.0f16 / 0.0f16;
    sqrt_value = __riscv_vfmerge_vfm_f16m2(sqrt_value, x1, mask_sNaN, vl);
  }
    
  denorm_mult = __riscv_vfmerge_vfm_f16m2(one, 0x1.0p-6f16, denormal_mask, vl);
  sqrt_value = __riscv_vfmul_vv_f16m2(sqrt_value, denorm_mult, vl);
#endif

  sqrt_value = __riscv_vmerge_vvm_f16m2(sqrt_value, x, zero_mask, vl);
  return sqrt_value; 
}

vfloat16m4_t __riscv_vsqrt_f16m4(vfloat16m4_t x, size_t vl)
{
  vuint16m4_t xi = __riscv_vreinterpret_v_f16m4_u16m4(x);
  vuint16m4_t x_int = __riscv_vand_vx_u16m4(xi, 0x7fff, vl);
  vbool4_t zero_mask = __riscv_vmseq_vx_u16m4_b4(x_int, 0, vl);
  
#ifndef __FAST_MATH__
  unsigned short inf_si = 0x7c00; 
  _Float16 inf_f16 = *(_Float16*)(&inf_si);
  
  vbool4_t zero_less_mask = __riscv_vmsgtu_vx_u16m4_b4(xi, 0x8000, vl);
  vbool4_t inf_mask = __riscv_vmseq_vx_u16m4_b4(x_int, 0x7c00, vl);
  vbool4_t nan_mask = __riscv_vmsgtu_vx_u16m4_b4(x_int, 0x7c00, vl);
  vbool4_t mask_sNaN = __riscv_vmand_mm_b4(__riscv_vmsltu_vx_u16m4_b4 (x_int, 0x7e00, vl), nan_mask, vl);
  mask_sNaN = __riscv_vmor_mm_b4(mask_sNaN, zero_less_mask, vl);
  
  vbool4_t special_mask = __riscv_vmor_mm_b4(zero_less_mask, nan_mask, vl);
  vfloat16m4_t x_spec = __riscv_vfmerge_vfm_f16m4(x, 0, 
                             __riscv_vmor_mm_b4(special_mask, inf_mask, vl), vl);

  vfloat16m4_t one = __riscv_vfmv_v_f_f16m4(1.0f16, vl);
  vbool4_t denormal_mask = __riscv_vmsltu_vx_u16m4_b4(x_int, 0x0400, vl);
  vfloat16m4_t denorm_mult = 
    __riscv_vfmerge_vfm_f16m4(one, 0x1.0p+12f16, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f16m4(x_spec, denorm_mult, vl);
  
  x_int = __riscv_vreinterpret_v_f16m4_u16m4(x_spec);
#endif

  vuint16m4_t mantissa_in_x = __riscv_vand_vx_u16m4(x_int, 0x03ff, vl);
  vuint16m4_t order_in_x = __riscv_vand_vx_u16m4(x_int, 0x7c00, vl);
  order_in_x = __riscv_vsrl_vx_u16m4(order_in_x, 10, vl);

  vuint16m4_t reduced_x_int = __riscv_vor_vx_u16m4(mantissa_in_x, 0x3c00, vl);
  vfloat16m4_t reduced_x = __riscv_vreinterpret_v_u16m4_f16m4(reduced_x_int);

  vuint16m4_t low_ind_bit = __riscv_vand_vx_u16m4(order_in_x, 0x1, vl);

  vfloat16m4_t y0 = __riscv_vreinterpret_v_u16m4_f16m4(
    __riscv_vrsub_vx_u16m4(__riscv_vsrl_vx_u16m4(reduced_x_int, 1, vl), 0x59d8, vl));

  vfloat16m4_t xx = __riscv_vfmul_vv_f16m4(y0, reduced_x, vl);
  vfloat16m4_t h = __riscv_vfmul_vf_f16m4(y0, 0.5f16, vl);
  vfloat16m4_t r = __riscv_vfrsub_vf_f16m4(
    __riscv_vfmul_vv_f16m4(xx, h, vl), 0.5f16, vl);

  xx = __riscv_vfmacc_vv_f16m4(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f16m4(h, h, r, vl);
  r = __riscv_vfrsub_vf_f16m4(__riscv_vfmul_vv_f16m4(xx, h, vl), 0.5f16, vl);
  xx = __riscv_vfmacc_vv_f16m4(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f16m4(h, h, r, vl);
  
  r = __riscv_vfnmsac_vv_f16m4(reduced_x, xx, xx, vl);
  
  vfloat16m4_t zh = __riscv_vfmacc_vv_f16m4(xx, r, h, vl); // high-part of sqrt value
  vfloat16m4_t zl = __riscv_vfmacc_vv_f16m4(__riscv_vfsub_vv_f16m4(xx, zh, vl), r, h, vl);// low-part of sqrt value
    
  vuint16m4_t new_exp = __riscv_vadd_vx_u16m4(order_in_x, 15, vl);
  new_exp = __riscv_vsrl_vx_u16m4(new_exp, 1, vl);
  vfloat16m4_t order_exp = __riscv_vreinterpret_v_u16m4_f16m4(
                           __riscv_vsll_vx_u16m4(new_exp, 10, vl));
  
  vbool4_t is_odd = __riscv_vmseq_vx_u16m4_b4(low_ind_bit, 1, vl);
  vfloat16m4_t order_low = __riscv_vfmerge_vfm_f16m4(
    __riscv_vfmv_v_f_f16m4(0x1.6ap+0f16, vl), 1.0f16, is_odd, vl);
  vfloat16m4_t order_low_ = __riscv_vfmerge_vfm_f16m4(
    __riscv_vfmv_v_f_f16m4(0x1.3ccp-13f16, vl), 0.0f16, is_odd, vl);
  vuint16m4_t red_exp = __riscv_vrsub_vx_u16m4(low_ind_bit, 16, vl);
  vfloat16m4_t order_reduce = __riscv_vreinterpret_v_u16m4_f16m4(
                                __riscv_vsll_vx_u16m4(red_exp, 10, vl));
   
  vfloat16m4_t sh = __riscv_vfmul_vv_f16m4(zh, order_low, vl);
  vfloat16m4_t sl = __riscv_vfmsac_vv_f16m4(sh, zh, order_low, vl);

  sl = __riscv_vfmacc_vv_f16m4(sl, zh, order_low_, vl);
  sl = __riscv_vfmacc_vv_f16m4(sl, zl, order_low, vl);
  
  vfloat16m4_t sqrt_value = __riscv_vfadd_vv_f16m4(sh, sl, vl);
  
  reduced_x = __riscv_vfmul_vv_f16m4(reduced_x, order_reduce, vl);
  
  h = __riscv_vfmul_vf_f16m4(h, 0.5f16, vl);
  r = __riscv_vfnmsac_vv_f16m4(reduced_x, sqrt_value, sqrt_value, vl);
  r = __riscv_vfmul_vv_f16m4(r, order_low, vl);
  
  sqrt_value = __riscv_vfmacc_vv_f16m4(sqrt_value, r, h, vl);
  sqrt_value = __riscv_vfmul_vv_f16m4(sqrt_value, order_exp, vl);

#ifndef __FAST_MATH__
  sqrt_value = __riscv_vfmerge_vfm_f16m4(sqrt_value, inf_f16, inf_mask, vl);
  
  sqrt_value = __riscv_vmerge_vvm_f16m4(sqrt_value, x, nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b4(mask_sNaN, vl);
  if (issNaN) {
    volatile _Float16 x1 = 0.0f16 / 0.0f16;
    sqrt_value = __riscv_vfmerge_vfm_f16m4(sqrt_value, x1, mask_sNaN, vl);
  }
    
  denorm_mult = __riscv_vfmerge_vfm_f16m4(one, 0x1.0p-6f16, denormal_mask, vl);
  sqrt_value = __riscv_vfmul_vv_f16m4(sqrt_value, denorm_mult, vl);
#endif

  sqrt_value = __riscv_vmerge_vvm_f16m4(sqrt_value, x, zero_mask, vl);
  return sqrt_value; 
}

vfloat16m8_t __riscv_vsqrt_f16m8(vfloat16m8_t x, size_t vl)
{
  vuint16m8_t xi = __riscv_vreinterpret_v_f16m8_u16m8(x);
  vuint16m8_t x_int = __riscv_vand_vx_u16m8(xi, 0x7fff, vl);
  vbool2_t zero_mask = __riscv_vmseq_vx_u16m8_b2(x_int, 0, vl);
  
#ifndef __FAST_MATH__
  unsigned short inf_si = 0x7c00; 
  _Float16 inf_f16 = *(_Float16*)(&inf_si);
  
  vbool2_t zero_less_mask = __riscv_vmsgtu_vx_u16m8_b2(xi, 0x8000, vl);
  vbool2_t inf_mask = __riscv_vmseq_vx_u16m8_b2(x_int, 0x7c00, vl);
  vbool2_t nan_mask = __riscv_vmsgtu_vx_u16m8_b2(x_int, 0x7c00, vl);
  vbool2_t mask_sNaN = __riscv_vmand_mm_b2(__riscv_vmsltu_vx_u16m8_b2 (x_int, 0x7e00, vl), nan_mask, vl);
  mask_sNaN = __riscv_vmor_mm_b2(mask_sNaN, zero_less_mask, vl);
  
  vbool2_t special_mask = __riscv_vmor_mm_b2(zero_less_mask, nan_mask, vl);
  vfloat16m8_t x_spec = __riscv_vfmerge_vfm_f16m8(x, 0, 
                             __riscv_vmor_mm_b2(special_mask, inf_mask, vl), vl);

  vfloat16m8_t one = __riscv_vfmv_v_f_f16m8(1.0f16, vl);
  vbool2_t denormal_mask = __riscv_vmsltu_vx_u16m8_b2(x_int, 0x0400, vl);
  vfloat16m8_t denorm_mult = 
    __riscv_vfmerge_vfm_f16m8(one, 0x1.0p+12f16, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f16m8(x_spec, denorm_mult, vl);
  
  x_int = __riscv_vreinterpret_v_f16m8_u16m8(x_spec);
#endif

  vuint16m8_t mantissa_in_x = __riscv_vand_vx_u16m8(x_int, 0x03ff, vl);
  vuint16m8_t order_in_x = __riscv_vand_vx_u16m8(x_int, 0x7c00, vl);
  order_in_x = __riscv_vsrl_vx_u16m8(order_in_x, 10, vl);

  vuint16m8_t reduced_x_int = __riscv_vor_vx_u16m8(mantissa_in_x, 0x3c00, vl);
  vfloat16m8_t reduced_x = __riscv_vreinterpret_v_u16m8_f16m8(reduced_x_int);

  vuint16m8_t low_ind_bit = __riscv_vand_vx_u16m8(order_in_x, 0x1, vl);

  vfloat16m8_t y0 = __riscv_vreinterpret_v_u16m8_f16m8(
    __riscv_vrsub_vx_u16m8(__riscv_vsrl_vx_u16m8(reduced_x_int, 1, vl), 0x59d8, vl));

  vfloat16m8_t xx = __riscv_vfmul_vv_f16m8(y0, reduced_x, vl);
  vfloat16m8_t h = __riscv_vfmul_vf_f16m8(y0, 0.5f16, vl);
  vfloat16m8_t r = __riscv_vfrsub_vf_f16m8(
    __riscv_vfmul_vv_f16m8(xx, h, vl), 0.5f16, vl);

  xx = __riscv_vfmacc_vv_f16m8(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f16m8(h, h, r, vl);
  r = __riscv_vfrsub_vf_f16m8(__riscv_vfmul_vv_f16m8(xx, h, vl), 0.5f16, vl);
  xx = __riscv_vfmacc_vv_f16m8(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f16m8(h, h, r, vl);
  
  r = __riscv_vfnmsac_vv_f16m8(reduced_x, xx, xx, vl);
  
  vfloat16m8_t zh = __riscv_vfmacc_vv_f16m8(xx, r, h, vl); // high-part of sqrt value
  vfloat16m8_t zl = __riscv_vfmacc_vv_f16m8(__riscv_vfsub_vv_f16m8(xx, zh, vl), r, h, vl);// low-part of sqrt value
    
  vuint16m8_t new_exp = __riscv_vadd_vx_u16m8(order_in_x, 15, vl);
  new_exp = __riscv_vsrl_vx_u16m8(new_exp, 1, vl);
  vfloat16m8_t order_exp = __riscv_vreinterpret_v_u16m8_f16m8(
                           __riscv_vsll_vx_u16m8(new_exp, 10, vl));
  
  vbool2_t is_odd = __riscv_vmseq_vx_u16m8_b2(low_ind_bit, 1, vl);
  vfloat16m8_t order_low = __riscv_vfmerge_vfm_f16m8(
    __riscv_vfmv_v_f_f16m8(0x1.6ap+0f16, vl), 1.0f16, is_odd, vl);
  vfloat16m8_t order_low_ = __riscv_vfmerge_vfm_f16m8(
    __riscv_vfmv_v_f_f16m8(0x1.3ccp-13f16, vl), 0.0f16, is_odd, vl);
  vuint16m8_t red_exp = __riscv_vrsub_vx_u16m8(low_ind_bit, 16, vl);
  vfloat16m8_t order_reduce = __riscv_vreinterpret_v_u16m8_f16m8(
                                __riscv_vsll_vx_u16m8(red_exp, 10, vl));
   
  vfloat16m8_t sh = __riscv_vfmul_vv_f16m8(zh, order_low, vl);
  vfloat16m8_t sl = __riscv_vfmsac_vv_f16m8(sh, zh, order_low, vl);

  sl = __riscv_vfmacc_vv_f16m8(sl, zh, order_low_, vl);
  sl = __riscv_vfmacc_vv_f16m8(sl, zl, order_low, vl);
  
  vfloat16m8_t sqrt_value = __riscv_vfadd_vv_f16m8(sh, sl, vl);
  
  reduced_x = __riscv_vfmul_vv_f16m8(reduced_x, order_reduce, vl);
  
  h = __riscv_vfmul_vf_f16m8(h, 0.5f16, vl);
  r = __riscv_vfnmsac_vv_f16m8(reduced_x, sqrt_value, sqrt_value, vl);
  r = __riscv_vfmul_vv_f16m8(r, order_low, vl);
  
  sqrt_value = __riscv_vfmacc_vv_f16m8(sqrt_value, r, h, vl);
  sqrt_value = __riscv_vfmul_vv_f16m8(sqrt_value, order_exp, vl);

#ifndef __FAST_MATH__
  sqrt_value = __riscv_vfmerge_vfm_f16m8(sqrt_value, inf_f16, inf_mask, vl);
  
  sqrt_value = __riscv_vmerge_vvm_f16m8(sqrt_value, x, nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b2(mask_sNaN, vl);
  if (issNaN) {
    volatile _Float16 x1 = 0.0f16 / 0.0f16;
    sqrt_value = __riscv_vfmerge_vfm_f16m8(sqrt_value, x1, mask_sNaN, vl);
  }
    
  denorm_mult = __riscv_vfmerge_vfm_f16m8(one, 0x1.0p-6f16, denormal_mask, vl);
  sqrt_value = __riscv_vfmul_vv_f16m8(sqrt_value, denorm_mult, vl);
#endif

  sqrt_value = __riscv_vmerge_vvm_f16m8(sqrt_value, x, zero_mask, vl);
  return sqrt_value; 
}

vfloat32m1_t __riscv_vsqrt_f32m1(vfloat32m1_t x, size_t vl)
{
  vuint32m1_t xi = __riscv_vreinterpret_v_f32m1_u32m1(x);
  vuint32m1_t x_int = __riscv_vand_vx_u32m1(xi, 0x7fffffff, vl);
  vbool32_t zero_mask = __riscv_vmseq_vx_u32m1_b32(x_int, 0, vl);
  
#ifndef __FAST_MATH__
  unsigned int inf_ui = 0x7f800000; // mask for +inf
  float inf_f = *(float*)(&inf_ui); // +inf
  
  vbool32_t zero_less_mask = __riscv_vmsgtu_vx_u32m1_b32(xi, 0x80000000, vl);
  vbool32_t inf_mask = __riscv_vmseq_vx_u32m1_b32(x_int, inf_ui, vl);
  vbool32_t nan_mask = __riscv_vmsgtu_vx_u32m1_b32(x_int, 0x7f800000, vl);
  vbool32_t mask_sNaN = __riscv_vmand_mm_b32(
            __riscv_vmsltu_vx_u32m1_b32(x_int, 0x7fc00000, vl), nan_mask, vl);
  mask_sNaN = __riscv_vmor_mm_b32(mask_sNaN, zero_less_mask, vl);
  
  vbool32_t special_mask = __riscv_vmor_mm_b32(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b32(special_mask, nan_mask, vl);
  vfloat32m1_t x_spec = __riscv_vfmerge_vfm_f32m1(x, 0, special_mask, vl);

  vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
  vbool32_t denormal_mask = __riscv_vmsltu_vx_u32m1_b32(x_int, 0x00800000, vl);
  vfloat32m1_t denorm_mult = 
           __riscv_vfmerge_vfm_f32m1(one, 0x1.0p+24f, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f32m1(x_spec, denorm_mult, vl);
  
  
  x_int = __riscv_vreinterpret_v_f32m1_u32m1(x_spec);
#endif

  vuint32m1_t mantissa_in_x = __riscv_vand_vx_u32m1(x_int, 0x007fffff, vl);
  vuint32m1_t order_in_x = __riscv_vand_vx_u32m1(x_int, 0x7f800000, vl);
  order_in_x = __riscv_vsrl_vx_u32m1(order_in_x, 23, vl);

  vuint32m1_t reduced_x_int = __riscv_vor_vx_u32m1(mantissa_in_x, 0x3f800000, vl);
  vfloat32m1_t reduced_x = __riscv_vreinterpret_v_u32m1_f32m1(reduced_x_int);

  vuint32m1_t low_ind_bit = __riscv_vand_vx_u32m1(order_in_x, 0x1, vl);

  vfloat32m1_t y0 = __riscv_vreinterpret_v_u32m1_f32m1(
    __riscv_vrsub_vx_u32m1(__riscv_vsrl_vx_u32m1(reduced_x_int, 1, vl), 0x5f3759df, vl));
  
  vfloat32m1_t xx = __riscv_vfmul_vv_f32m1(y0, reduced_x, vl);
  vfloat32m1_t h = __riscv_vfmul_vf_f32m1(y0, 0.5f, vl);
  vfloat32m1_t r = __riscv_vfrsub_vf_f32m1(__riscv_vfmul_vv_f32m1(xx, h, vl), 0.5f, vl);

  xx = __riscv_vfmacc_vv_f32m1(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f32m1(h, h, r, vl);
  r = __riscv_vfrsub_vf_f32m1(__riscv_vfmul_vv_f32m1(xx, h, vl), 0.5f, vl);
  xx = __riscv_vfmacc_vv_f32m1(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f32m1(h, h, r, vl);
  r = __riscv_vfrsub_vf_f32m1(__riscv_vfmul_vv_f32m1(xx, h, vl), 0.5f, vl);
  xx = __riscv_vfmacc_vv_f32m1(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f32m1(h, h, r, vl);
  
  r = __riscv_vfnmsac_vv_f32m1(reduced_x, xx, xx, vl);
  
  vfloat32m1_t zh = __riscv_vfmacc_vv_f32m1(xx, r, h, vl); // high-part of sqrt value
  vfloat32m1_t zl = __riscv_vfmacc_vv_f32m1(__riscv_vfsub_vv_f32m1(xx, zh, vl), r, h, vl); // low-part of sqrt value
  
  vuint32m1_t new_exp = __riscv_vadd_vx_u32m1(order_in_x, 127, vl);
  new_exp = __riscv_vsrl_vx_u32m1(new_exp, 1, vl);
  vfloat32m1_t order_exp = __riscv_vreinterpret_v_u32m1_f32m1(
                           __riscv_vsll_vx_u32m1(new_exp, 23, vl));
  
  vbool32_t is_odd = __riscv_vmseq_vx_u32m1_b32(low_ind_bit, 1, vl);
  vfloat32m1_t order_low = __riscv_vfmerge_vfm_f32m1(
    __riscv_vfmv_v_f_f32m1(0x1.6a09e6p+0f, vl), 1.0f, is_odd, vl);
  vfloat32m1_t order_low_ = __riscv_vfmerge_vfm_f32m1(
    __riscv_vfmv_v_f_f32m1(0x1.9fcef4p-26f, vl), 0.0f, is_odd, vl);
  vfloat32m1_t h_factor = __riscv_vfmerge_vfm_f32m1(
      __riscv_vfmv_v_f_f32m1(0x1.6a09e8p-1f, vl), 1.0f, is_odd, vl);
  vuint32m1_t red_exp = __riscv_vrsub_vx_u32m1(low_ind_bit, 128, vl);
  vfloat32m1_t order_reduce = __riscv_vreinterpret_v_u32m1_f32m1(
                                __riscv_vsll_vx_u32m1(red_exp, 23, vl));
  
  vfloat32m1_t sh = __riscv_vfmul_vv_f32m1(zh, order_low, vl);
  vfloat32m1_t sl = __riscv_vfmsac_vv_f32m1(sh, zh, order_low, vl);
  
  sl = __riscv_vfmacc_vv_f32m1(sl, zh, order_low_, vl);
  sl = __riscv_vfmacc_vv_f32m1(sl, zl, order_low, vl);
   
  vfloat32m1_t sqrt_value = __riscv_vfadd_vv_f32m1(sh, sl, vl);
  
  reduced_x = __riscv_vfmul_vv_f32m1(reduced_x, order_reduce, vl);
  r = __riscv_vfnmsac_vv_f32m1(reduced_x, sqrt_value, sqrt_value, vl);
  r = __riscv_vfmul_vv_f32m1(r, h_factor, vl);
  
  sqrt_value = __riscv_vfmacc_vv_f32m1(sqrt_value, r, h, vl);
  sqrt_value = __riscv_vfmul_vv_f32m1(sqrt_value, order_exp, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m1(sqrt_value, inf_f, inf_mask, vl);
  
  sqrt_value = __riscv_vmerge_vvm_f32m1(sqrt_value, x, nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b32(mask_sNaN, vl);
  if (issNaN) {
    volatile float x1 = 0.0f / 0.0f;
    sqrt_value = __riscv_vfmerge_vfm_f32m1(sqrt_value, x1, mask_sNaN, vl);
  }
  
  denorm_mult = 
    __riscv_vfmerge_vfm_f32m1(one, 0x1.0p-12f, denormal_mask, vl);
  sqrt_value = __riscv_vfmul_vv_f32m1(sqrt_value, denorm_mult, vl);
#endif

  sqrt_value = __riscv_vmerge_vvm_f32m1(sqrt_value, x, zero_mask, vl);
  return sqrt_value; 
}

vfloat32m2_t __riscv_vsqrt_f32m2(vfloat32m2_t x, size_t vl)
{  
  vuint32m2_t xi = __riscv_vreinterpret_v_f32m2_u32m2(x);
  vuint32m2_t x_int = __riscv_vand_vx_u32m2(xi, 0x7fffffff, vl);
  vbool16_t zero_mask = __riscv_vmseq_vx_u32m2_b16(x_int, 0, vl);
  
#ifndef __FAST_MATH__
  unsigned int inf_ui = 0x7f800000; // mask for +inf
  float inf_f = *(float*)(&inf_ui); // +inf
  
  vbool16_t zero_less_mask = __riscv_vmsgtu_vx_u32m2_b16(xi, 0x80000000, vl);
  vbool16_t inf_mask = __riscv_vmseq_vx_u32m2_b16(x_int, inf_ui, vl);
  vbool16_t nan_mask = __riscv_vmsgtu_vx_u32m2_b16(x_int, 0x7f800000, vl);
  vbool16_t mask_sNaN = __riscv_vmand_mm_b16(
            __riscv_vmsltu_vx_u32m2_b16(x_int, 0x7fc00000, vl), nan_mask, vl);
  mask_sNaN = __riscv_vmor_mm_b16(mask_sNaN, zero_less_mask, vl);
  
  vbool16_t special_mask = __riscv_vmor_mm_b16(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b16(special_mask, nan_mask, vl);
  vfloat32m2_t x_spec = __riscv_vfmerge_vfm_f32m2(x, 0, special_mask, vl);

  vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
  vbool16_t denormal_mask = __riscv_vmsltu_vx_u32m2_b16(x_int, 0x00800000, vl);
  vfloat32m2_t denorm_mult = 
           __riscv_vfmerge_vfm_f32m2(one, 0x1.0p+24f, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f32m2(x_spec, denorm_mult, vl);
  
  
  x_int = __riscv_vreinterpret_v_f32m2_u32m2(x_spec);
#endif

  vuint32m2_t mantissa_in_x = __riscv_vand_vx_u32m2(x_int, 0x007fffff, vl);
  vuint32m2_t order_in_x = __riscv_vand_vx_u32m2(x_int, 0x7f800000, vl);
  order_in_x = __riscv_vsrl_vx_u32m2(order_in_x, 23, vl);

  vuint32m2_t reduced_x_int = __riscv_vor_vx_u32m2(mantissa_in_x, 0x3f800000, vl);
  vfloat32m2_t reduced_x = __riscv_vreinterpret_v_u32m2_f32m2(reduced_x_int);

  vuint32m2_t low_ind_bit = __riscv_vand_vx_u32m2(order_in_x, 0x1, vl);

  vfloat32m2_t y0 = __riscv_vreinterpret_v_u32m2_f32m2(
    __riscv_vrsub_vx_u32m2(__riscv_vsrl_vx_u32m2(reduced_x_int, 1, vl), 0x5f3759df, vl));
  
  vfloat32m2_t xx = __riscv_vfmul_vv_f32m2(y0, reduced_x, vl);
  vfloat32m2_t h = __riscv_vfmul_vf_f32m2(y0, 0.5f, vl);
  vfloat32m2_t r = __riscv_vfrsub_vf_f32m2(__riscv_vfmul_vv_f32m2(xx, h, vl), 0.5f, vl);

  xx = __riscv_vfmacc_vv_f32m2(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f32m2(h, h, r, vl);
  r = __riscv_vfrsub_vf_f32m2(__riscv_vfmul_vv_f32m2(xx, h, vl), 0.5f, vl);
  xx = __riscv_vfmacc_vv_f32m2(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f32m2(h, h, r, vl);
  r = __riscv_vfrsub_vf_f32m2(__riscv_vfmul_vv_f32m2(xx, h, vl), 0.5f, vl);
  xx = __riscv_vfmacc_vv_f32m2(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f32m2(h, h, r, vl);
  
  r = __riscv_vfnmsac_vv_f32m2(reduced_x, xx, xx, vl);
  
  vfloat32m2_t zh = __riscv_vfmacc_vv_f32m2(xx, r, h, vl); // high-part of sqrt value
  vfloat32m2_t zl = __riscv_vfmacc_vv_f32m2(__riscv_vfsub_vv_f32m2(xx, zh, vl), r, h, vl); // low-part of sqrt value
  
  vuint32m2_t new_exp = __riscv_vadd_vx_u32m2(order_in_x, 127, vl);
  new_exp = __riscv_vsrl_vx_u32m2(new_exp, 1, vl);
  vfloat32m2_t order_exp = __riscv_vreinterpret_v_u32m2_f32m2(
                           __riscv_vsll_vx_u32m2(new_exp, 23, vl));
  
  vbool16_t is_odd = __riscv_vmseq_vx_u32m2_b16(low_ind_bit, 1, vl);
  vfloat32m2_t order_low = __riscv_vfmerge_vfm_f32m2(
    __riscv_vfmv_v_f_f32m2(0x1.6a09e6p+0f, vl), 1.0f, is_odd, vl);
  vfloat32m2_t order_low_ = __riscv_vfmerge_vfm_f32m2(
    __riscv_vfmv_v_f_f32m2(0x1.9fcef4p-26f, vl), 0.0f, is_odd, vl);
  vfloat32m2_t h_factor = __riscv_vfmerge_vfm_f32m2(
      __riscv_vfmv_v_f_f32m2(0x1.6a09e8p-1f, vl), 1.0f, is_odd, vl);
  vuint32m2_t red_exp = __riscv_vrsub_vx_u32m2(low_ind_bit, 128, vl);
  vfloat32m2_t order_reduce = __riscv_vreinterpret_v_u32m2_f32m2(
                                __riscv_vsll_vx_u32m2(red_exp, 23, vl));
  
  vfloat32m2_t sh = __riscv_vfmul_vv_f32m2(zh, order_low, vl);
  vfloat32m2_t sl = __riscv_vfmsac_vv_f32m2(sh, zh, order_low, vl);
  
  sl = __riscv_vfmacc_vv_f32m2(sl, zh, order_low_, vl);
  sl = __riscv_vfmacc_vv_f32m2(sl, zl, order_low, vl);
   
  vfloat32m2_t sqrt_value = __riscv_vfadd_vv_f32m2(sh, sl, vl);
  
  reduced_x = __riscv_vfmul_vv_f32m2(reduced_x, order_reduce, vl);
  r = __riscv_vfnmsac_vv_f32m2(reduced_x, sqrt_value, sqrt_value, vl);
  r = __riscv_vfmul_vv_f32m2(r, h_factor, vl);
  
  sqrt_value = __riscv_vfmacc_vv_f32m2(sqrt_value, r, h, vl);
  sqrt_value = __riscv_vfmul_vv_f32m2(sqrt_value, order_exp, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m2(sqrt_value, inf_f, inf_mask, vl);
  
  sqrt_value = __riscv_vmerge_vvm_f32m2(sqrt_value, x, nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
  if (issNaN) {
    volatile float x1 = 0.0f / 0.0f;
    sqrt_value = __riscv_vfmerge_vfm_f32m2(sqrt_value, x1, mask_sNaN, vl);
  }
  
  denorm_mult = 
    __riscv_vfmerge_vfm_f32m2(one, 0x1.0p-12f, denormal_mask, vl);
  sqrt_value = __riscv_vfmul_vv_f32m2(sqrt_value, denorm_mult, vl);
#endif

  sqrt_value = __riscv_vmerge_vvm_f32m2(sqrt_value, x, zero_mask, vl);
  return sqrt_value; 
}

vfloat32m4_t __riscv_vsqrt_f32m4(vfloat32m4_t x, size_t vl)
{
  vuint32m4_t xi = __riscv_vreinterpret_v_f32m4_u32m4(x);
  vuint32m4_t x_int = __riscv_vand_vx_u32m4(xi, 0x7fffffff, vl);
  vbool8_t zero_mask = __riscv_vmseq_vx_u32m4_b8(x_int, 0, vl);
  
#ifndef __FAST_MATH__
  unsigned int inf_ui = 0x7f800000; // mask for +inf
  float inf_f = *(float*)(&inf_ui); // +inf
  
  vbool8_t zero_less_mask = __riscv_vmsgtu_vx_u32m4_b8(xi, 0x80000000, vl);
  vbool8_t inf_mask = __riscv_vmseq_vx_u32m4_b8(x_int, inf_ui, vl);
  vbool8_t nan_mask = __riscv_vmsgtu_vx_u32m4_b8(x_int, 0x7f800000, vl);
  vbool8_t mask_sNaN = __riscv_vmand_mm_b8(
            __riscv_vmsltu_vx_u32m4_b8(x_int, 0x7fc00000, vl), nan_mask, vl);
  mask_sNaN = __riscv_vmor_mm_b8(mask_sNaN, zero_less_mask, vl);
  
  vbool8_t special_mask = __riscv_vmor_mm_b8(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b8(special_mask, nan_mask, vl);
  vfloat32m4_t x_spec = __riscv_vfmerge_vfm_f32m4(x, 0, special_mask, vl);

  vfloat32m4_t one = __riscv_vfmv_v_f_f32m4(1.0f, vl);
  vbool8_t denormal_mask = __riscv_vmsltu_vx_u32m4_b8(x_int, 0x00800000, vl);
  vfloat32m4_t denorm_mult = 
           __riscv_vfmerge_vfm_f32m4(one, 0x1.0p+24f, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f32m4(x_spec, denorm_mult, vl);
  
  x_int = __riscv_vreinterpret_v_f32m4_u32m4(x_spec);
#endif

  vuint32m4_t mantissa_in_x = __riscv_vand_vx_u32m4(x_int, 0x007fffff, vl);
  vuint32m4_t order_in_x = __riscv_vand_vx_u32m4(x_int, 0x7f800000, vl);
  order_in_x = __riscv_vsrl_vx_u32m4(order_in_x, 23, vl);

  vuint32m4_t reduced_x_int = __riscv_vor_vx_u32m4(mantissa_in_x, 0x3f800000, vl);
  vfloat32m4_t reduced_x = __riscv_vreinterpret_v_u32m4_f32m4(reduced_x_int);

  vuint32m4_t low_ind_bit = __riscv_vand_vx_u32m4(order_in_x, 0x1, vl);

  vfloat32m4_t y0 = __riscv_vreinterpret_v_u32m4_f32m4(
    __riscv_vrsub_vx_u32m4(__riscv_vsrl_vx_u32m4(reduced_x_int, 1, vl), 0x5f3759df, vl));
  
  vfloat32m4_t xx = __riscv_vfmul_vv_f32m4(y0, reduced_x, vl);
  vfloat32m4_t h = __riscv_vfmul_vf_f32m4(y0, 0.5f, vl);
  vfloat32m4_t r = __riscv_vfrsub_vf_f32m4(__riscv_vfmul_vv_f32m4(xx, h, vl), 0.5f, vl);

  xx = __riscv_vfmacc_vv_f32m4(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f32m4(h, h, r, vl);
  r = __riscv_vfrsub_vf_f32m4(__riscv_vfmul_vv_f32m4(xx, h, vl), 0.5f, vl);
  xx = __riscv_vfmacc_vv_f32m4(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f32m4(h, h, r, vl);
  r = __riscv_vfrsub_vf_f32m4(__riscv_vfmul_vv_f32m4(xx, h, vl), 0.5f, vl);
  xx = __riscv_vfmacc_vv_f32m4(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f32m4(h, h, r, vl);
  
  r = __riscv_vfnmsac_vv_f32m4(reduced_x, xx, xx, vl);
  
  vfloat32m4_t zh = __riscv_vfmacc_vv_f32m4(xx, r, h, vl); // high-part of sqrt value
  vfloat32m4_t zl = __riscv_vfmacc_vv_f32m4(__riscv_vfsub_vv_f32m4(xx, zh, vl), r, h, vl); // low-part of sqrt value
  
  vuint32m4_t new_exp = __riscv_vadd_vx_u32m4(order_in_x, 127, vl);
  new_exp = __riscv_vsrl_vx_u32m4(new_exp, 1, vl);
  vfloat32m4_t order_exp = __riscv_vreinterpret_v_u32m4_f32m4(
                           __riscv_vsll_vx_u32m4(new_exp, 23, vl));
  
  vbool8_t is_odd = __riscv_vmseq_vx_u32m4_b8(low_ind_bit, 1, vl);
  vfloat32m4_t order_low = __riscv_vfmerge_vfm_f32m4(
    __riscv_vfmv_v_f_f32m4(0x1.6a09e6p+0f, vl), 1.0f, is_odd, vl);
  vfloat32m4_t order_low_ = __riscv_vfmerge_vfm_f32m4(
    __riscv_vfmv_v_f_f32m4(0x1.9fcef4p-26f, vl), 0.0f, is_odd, vl);
  vfloat32m4_t h_factor = __riscv_vfmerge_vfm_f32m4(
      __riscv_vfmv_v_f_f32m4(0x1.6a09e8p-1f, vl), 1.0f, is_odd, vl);
  vuint32m4_t red_exp = __riscv_vrsub_vx_u32m4(low_ind_bit, 128, vl);
  vfloat32m4_t order_reduce = __riscv_vreinterpret_v_u32m4_f32m4(
                                __riscv_vsll_vx_u32m4(red_exp, 23, vl));
  
  vfloat32m4_t sh = __riscv_vfmul_vv_f32m4(zh, order_low, vl);
  vfloat32m4_t sl = __riscv_vfmsac_vv_f32m4(sh, zh, order_low, vl);
  
  sl = __riscv_vfmacc_vv_f32m4(sl, zh, order_low_, vl);
  sl = __riscv_vfmacc_vv_f32m4(sl, zl, order_low, vl);
   
  vfloat32m4_t sqrt_value = __riscv_vfadd_vv_f32m4(sh, sl, vl);
  
  reduced_x = __riscv_vfmul_vv_f32m4(reduced_x, order_reduce, vl);
  r = __riscv_vfnmsac_vv_f32m4(reduced_x, sqrt_value, sqrt_value, vl);
  r = __riscv_vfmul_vv_f32m4(r, h_factor, vl);
  
  sqrt_value = __riscv_vfmacc_vv_f32m4(sqrt_value, r, h, vl);
  sqrt_value = __riscv_vfmul_vv_f32m4(sqrt_value, order_exp, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m4(sqrt_value, inf_f, inf_mask, vl);
  
  sqrt_value = __riscv_vmerge_vvm_f32m4(sqrt_value, x, nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b8(mask_sNaN, vl);
  if (issNaN) {
    volatile float x1 = 0.0f / 0.0f;
    sqrt_value = __riscv_vfmerge_vfm_f32m4(sqrt_value, x1, mask_sNaN, vl);
  }
  
  denorm_mult = 
    __riscv_vfmerge_vfm_f32m4(one, 0x1.0p-12f, denormal_mask, vl);
  sqrt_value = __riscv_vfmul_vv_f32m4(sqrt_value, denorm_mult, vl);
#endif

  sqrt_value = __riscv_vmerge_vvm_f32m4(sqrt_value, x, zero_mask, vl);
  return sqrt_value; 
}

vfloat32m8_t __riscv_vsqrt_f32m8(vfloat32m8_t x, size_t vl)
{
  vuint32m8_t xi = __riscv_vreinterpret_v_f32m8_u32m8(x);
  vuint32m8_t x_int = __riscv_vand_vx_u32m8(xi, 0x7fffffff, vl);
  vbool4_t zero_mask = __riscv_vmseq_vx_u32m8_b4(x_int, 0, vl);
  
#ifndef __FAST_MATH__
  unsigned int inf_ui = 0x7f800000; // mask for +inf
  float inf_f = *(float*)(&inf_ui); // +inf
  
  vbool4_t zero_less_mask = __riscv_vmsgtu_vx_u32m8_b4(xi, 0x80000000, vl);
  vbool4_t inf_mask = __riscv_vmseq_vx_u32m8_b4(x_int, inf_ui, vl);
  vbool4_t nan_mask = __riscv_vmsgtu_vx_u32m8_b4(x_int, 0x7f800000, vl);
  vbool4_t mask_sNaN = __riscv_vmand_mm_b4(
            __riscv_vmsltu_vx_u32m8_b4(x_int, 0x7fc00000, vl), nan_mask, vl);
  mask_sNaN = __riscv_vmor_mm_b4(mask_sNaN, zero_less_mask, vl);
  
  vbool4_t special_mask = __riscv_vmor_mm_b4(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b4(special_mask, nan_mask, vl);
  vfloat32m8_t x_spec = __riscv_vfmerge_vfm_f32m8(x, 0, special_mask, vl);

  vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1.0f, vl);
  vbool4_t denormal_mask = __riscv_vmsltu_vx_u32m8_b4(x_int, 0x00800000, vl);
  vfloat32m8_t denorm_mult = 
           __riscv_vfmerge_vfm_f32m8(one, 0x1.0p+24f, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f32m8(x_spec, denorm_mult, vl);
  
  
  x_int = __riscv_vreinterpret_v_f32m8_u32m8(x_spec);
#endif

  vuint32m8_t mantissa_in_x = __riscv_vand_vx_u32m8(x_int, 0x007fffff, vl);
  vuint32m8_t order_in_x = __riscv_vand_vx_u32m8(x_int, 0x7f800000, vl);
  order_in_x = __riscv_vsrl_vx_u32m8(order_in_x, 23, vl);

  vuint32m8_t reduced_x_int = __riscv_vor_vx_u32m8(mantissa_in_x, 0x3f800000, vl);
  vfloat32m8_t reduced_x = __riscv_vreinterpret_v_u32m8_f32m8(reduced_x_int);

  vuint32m8_t low_ind_bit = __riscv_vand_vx_u32m8(order_in_x, 0x1, vl);

  vfloat32m8_t y0 = __riscv_vreinterpret_v_u32m8_f32m8(
    __riscv_vrsub_vx_u32m8(__riscv_vsrl_vx_u32m8(reduced_x_int, 1, vl), 0x5f3759df, vl));
  
  vfloat32m8_t xx = __riscv_vfmul_vv_f32m8(y0, reduced_x, vl);
  vfloat32m8_t h = __riscv_vfmul_vf_f32m8(y0, 0.5f, vl);
  vfloat32m8_t r = __riscv_vfrsub_vf_f32m8(__riscv_vfmul_vv_f32m8(xx, h, vl), 0.5f, vl);

  xx = __riscv_vfmacc_vv_f32m8(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f32m8(h, h, r, vl);
  r = __riscv_vfrsub_vf_f32m8(__riscv_vfmul_vv_f32m8(xx, h, vl), 0.5f, vl);
  xx = __riscv_vfmacc_vv_f32m8(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f32m8(h, h, r, vl);
  r = __riscv_vfrsub_vf_f32m8(__riscv_vfmul_vv_f32m8(xx, h, vl), 0.5f, vl);
  xx = __riscv_vfmacc_vv_f32m8(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f32m8(h, h, r, vl);
  
  r = __riscv_vfnmsac_vv_f32m8(reduced_x, xx, xx, vl);
  
  vfloat32m8_t zh = __riscv_vfmacc_vv_f32m8(xx, r, h, vl); // high-part of sqrt value
  vfloat32m8_t zl = __riscv_vfmacc_vv_f32m8(__riscv_vfsub_vv_f32m8(xx, zh, vl), r, h, vl); // low-part of sqrt value
  
  vuint32m8_t new_exp = __riscv_vadd_vx_u32m8(order_in_x, 127, vl);
  new_exp = __riscv_vsrl_vx_u32m8(new_exp, 1, vl);
  vfloat32m8_t order_exp = __riscv_vreinterpret_v_u32m8_f32m8(
                           __riscv_vsll_vx_u32m8(new_exp, 23, vl));
  
  vbool4_t is_odd = __riscv_vmseq_vx_u32m8_b4(low_ind_bit, 1, vl);
  vfloat32m8_t order_low = __riscv_vfmerge_vfm_f32m8(
    __riscv_vfmv_v_f_f32m8(0x1.6a09e6p+0f, vl), 1.0f, is_odd, vl);
  vfloat32m8_t order_low_ = __riscv_vfmerge_vfm_f32m8(
    __riscv_vfmv_v_f_f32m8(0x1.9fcef4p-26f, vl), 0.0f, is_odd, vl);
  vfloat32m8_t h_factor = __riscv_vfmerge_vfm_f32m8(
      __riscv_vfmv_v_f_f32m8(0x1.6a09e8p-1f, vl), 1.0f, is_odd, vl);
  vuint32m8_t red_exp = __riscv_vrsub_vx_u32m8(low_ind_bit, 128, vl);
  vfloat32m8_t order_reduce = __riscv_vreinterpret_v_u32m8_f32m8(
                                __riscv_vsll_vx_u32m8(red_exp, 23, vl));
  
  vfloat32m8_t sh = __riscv_vfmul_vv_f32m8(zh, order_low, vl);
  vfloat32m8_t sl = __riscv_vfmsac_vv_f32m8(sh, zh, order_low, vl);
  
  sl = __riscv_vfmacc_vv_f32m8(sl, zh, order_low_, vl);
  sl = __riscv_vfmacc_vv_f32m8(sl, zl, order_low, vl);
   
  vfloat32m8_t sqrt_value = __riscv_vfadd_vv_f32m8(sh, sl, vl);
  
  reduced_x = __riscv_vfmul_vv_f32m8(reduced_x, order_reduce, vl);
  r = __riscv_vfnmsac_vv_f32m8(reduced_x, sqrt_value, sqrt_value, vl);
  r = __riscv_vfmul_vv_f32m8(r, h_factor, vl);
  
  sqrt_value = __riscv_vfmacc_vv_f32m8(sqrt_value, r, h, vl);
  sqrt_value = __riscv_vfmul_vv_f32m8(sqrt_value, order_exp, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m8(sqrt_value, inf_f, inf_mask, vl);
  
  sqrt_value = __riscv_vmerge_vvm_f32m8(sqrt_value, x, nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b4(mask_sNaN, vl);
  if (issNaN) {
    volatile float x1 = 0.0f / 0.0f;
    sqrt_value = __riscv_vfmerge_vfm_f32m8(sqrt_value, x1, mask_sNaN, vl);
  }
  
  denorm_mult = 
    __riscv_vfmerge_vfm_f32m8(one, 0x1.0p-12f, denormal_mask, vl);
  sqrt_value = __riscv_vfmul_vv_f32m8(sqrt_value, denorm_mult, vl);
#endif

  sqrt_value = __riscv_vmerge_vvm_f32m8(sqrt_value, x, zero_mask, vl);
  return sqrt_value; 
}

vfloat64m1_t __riscv_vsqrt_f64m1(vfloat64m1_t x, size_t vl)
{
  vuint64m1_t xi = __riscv_vreinterpret_v_f64m1_u64m1(x);
  vuint64m1_t x_int = __riscv_vand_vx_u64m1(xi, 0x7fffffffffffffff, vl);
  vbool64_t zero_mask = __riscv_vmseq_vx_u64m1_b64(x_int, 0, vl);
  
#ifndef __FAST_MATH__
  unsigned long long inf_ull = 0x7ff0000000000000; // mask for +inf
  double inf_d = *(double*)(&inf_ull); // +inf

  vbool64_t zero_less_mask = __riscv_vmsgtu_vx_u64m1_b64(xi, 0x8000000000000000ULL, vl);
  vbool64_t inf_mask = __riscv_vmseq_vx_u64m1_b64(x_int, inf_ull, vl);
  vbool64_t nan_mask = __riscv_vmsgtu_vx_u64m1_b64(x_int, 0x7ff0000000000000, vl);
  vbool64_t mask_sNaN = __riscv_vmand_mm_b64(
            __riscv_vmsltu_vx_u64m1_b64(x_int, 0x7ff8000000000000, vl), nan_mask, vl);
  mask_sNaN = __riscv_vmor_mm_b64(mask_sNaN, zero_less_mask, vl);
  
  vbool64_t special_mask = __riscv_vmor_mm_b64(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b64(special_mask, nan_mask, vl);

  vfloat64m1_t x_spec = __riscv_vfmerge_vfm_f64m1(x, 0, special_mask, vl);

  vfloat64m1_t one = __riscv_vfmv_v_f_f64m1(1.0, vl);
  vbool64_t denormal_mask = __riscv_vmsltu_vx_u64m1_b64(x_int, 0x0010000000000000, vl);
  vfloat64m1_t denorm_mult = 
           __riscv_vfmerge_vfm_f64m1(one, 0x1.0p+60, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f64m1(x_spec, denorm_mult, vl);

  x_int = __riscv_vreinterpret_v_f64m1_u64m1(x_spec);
#endif

  vuint64m1_t mantissa_in_x = __riscv_vand_vx_u64m1(x_int, 0x000fffffffffffff, vl);
  vuint64m1_t order_in_x = __riscv_vand_vx_u64m1(x_int, 0x7ff0000000000000LL, vl);
  order_in_x = __riscv_vsrl_vx_u64m1(order_in_x, 52, vl);

  vuint64m1_t reduced_x_int = __riscv_vor_vx_u64m1(mantissa_in_x, 0x3ff0000000000000, vl);
  vfloat64m1_t reduced_x = __riscv_vreinterpret_v_u64m1_f64m1(reduced_x_int);

  vuint64m1_t low_ind_bit = __riscv_vand_vx_u64m1(order_in_x, 0x1, vl);

  vfloat64m1_t y0 = __riscv_vreinterpret_v_u64m1_f64m1(
    __riscv_vrsub_vx_u64m1(__riscv_vsrl_vx_u64m1(reduced_x_int, 1, vl), 0x5fe6eb50c7b537a9, vl));
  
  vfloat64m1_t xx = __riscv_vfmul_vv_f64m1(y0, reduced_x, vl);
  vfloat64m1_t h = __riscv_vfmul_vf_f64m1(y0, 0.5, vl);
  vfloat64m1_t r = __riscv_vfrsub_vf_f64m1(__riscv_vfmul_vv_f64m1(xx, h, vl), 0.5, vl);

  xx = __riscv_vfmacc_vv_f64m1(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m1(h, h, r, vl);
  r = __riscv_vfrsub_vf_f64m1(__riscv_vfmul_vv_f64m1(xx, h, vl), 0.5, vl);
  xx = __riscv_vfmacc_vv_f64m1(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m1(h, h, r, vl);
  r = __riscv_vfrsub_vf_f64m1(__riscv_vfmul_vv_f64m1(xx, h, vl), 0.5, vl);
  xx = __riscv_vfmacc_vv_f64m1(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m1(h, h, r, vl);
  r = __riscv_vfrsub_vf_f64m1(__riscv_vfmul_vv_f64m1(xx, h, vl), 0.5, vl);
  xx = __riscv_vfmacc_vv_f64m1(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m1(h, h, r, vl);
  
  r = __riscv_vfnmsac_vv_f64m1(reduced_x, xx, xx, vl);

  vfloat64m1_t zh = __riscv_vfmacc_vv_f64m1(xx, r, h, vl); // high-part of sqrt value
  vfloat64m1_t zl = __riscv_vfmacc_vv_f64m1(__riscv_vfsub_vv_f64m1(xx, zh, vl), r, h, vl);

  vuint64m1_t new_exp = __riscv_vadd_vx_u64m1(order_in_x, 1023, vl);
  new_exp = __riscv_vsrl_vx_u64m1(new_exp, 1, vl);
  vfloat64m1_t order_exp = __riscv_vreinterpret_v_u64m1_f64m1(
                           __riscv_vsll_vx_u64m1(new_exp, 52, vl));
  
  vbool64_t is_odd = __riscv_vmseq_vx_u64m1_b64(low_ind_bit, 1, vl);
  vfloat64m1_t order_low = __riscv_vfmerge_vfm_f64m1(
    __riscv_vfmv_v_f_f64m1(0x1.6a09e667f3bccp+0, vl), 1.0, is_odd, vl);
  vfloat64m1_t order_low_ = __riscv_vfmerge_vfm_f64m1(
    __riscv_vfmv_v_f_f64m1(0x1.21165f626cdd5p-53f, vl), 0.0, is_odd, vl);
  vfloat64m1_t h_factor = __riscv_vfmerge_vfm_f64m1(
      __riscv_vfmv_v_f_f64m1(0x1.6a09e667f3bcdp-1, vl), 1.0, is_odd, vl);
  vuint64m1_t red_exp = __riscv_vrsub_vx_u64m1(low_ind_bit, 1024, vl);
  vfloat64m1_t order_reduce = __riscv_vreinterpret_v_u64m1_f64m1(
                                __riscv_vsll_vx_u64m1(red_exp, 52, vl));
  
  vfloat64m1_t sh = __riscv_vfmul_vv_f64m1(zh, order_low, vl);
  vfloat64m1_t sl = __riscv_vfmsac_vv_f64m1(sh, zh, order_low, vl);
  
  sl = __riscv_vfmacc_vv_f64m1(sl, zh, order_low_, vl);
  sl = __riscv_vfmacc_vv_f64m1(sl, zl, order_low, vl);
  
  vfloat64m1_t sqrt_value = __riscv_vfadd_vv_f64m1(sh, sl, vl);
  
  reduced_x = __riscv_vfmul_vv_f64m1(reduced_x, order_reduce, vl);
  r = __riscv_vfnmsac_vv_f64m1(reduced_x, sqrt_value, sqrt_value, vl);
  r = __riscv_vfmul_vv_f64m1(r, h_factor, vl);
  
  sqrt_value = __riscv_vfmacc_vv_f64m1(sqrt_value, r, h, vl);
  sqrt_value = __riscv_vfmul_vv_f64m1(sqrt_value, order_exp, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m1(sqrt_value, inf_d, inf_mask, vl);

  sqrt_value = __riscv_vmerge_vvm_f64m1(sqrt_value, x, nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b64(mask_sNaN, vl);
  if (issNaN) {
    volatile double x1 = 0.0 / 0.0;
    sqrt_value = __riscv_vfmerge_vfm_f64m1(sqrt_value, x1, mask_sNaN, vl);
  }

  denorm_mult = 
    __riscv_vfmerge_vfm_f64m1(one, 0x1.0p-30, denormal_mask, vl);
  sqrt_value = __riscv_vfmul_vv_f64m1(sqrt_value, denorm_mult, vl);
#endif

  sqrt_value = __riscv_vmerge_vvm_f64m1(sqrt_value, x, zero_mask, vl);
  return sqrt_value; 
}

vfloat64m2_t __riscv_vsqrt_f64m2(vfloat64m2_t x, size_t vl)
{
  vuint64m2_t xi = __riscv_vreinterpret_v_f64m2_u64m2(x);
  vuint64m2_t x_int = __riscv_vand_vx_u64m2(xi, 0x7fffffffffffffff, vl);
  vbool32_t zero_mask = __riscv_vmseq_vx_u64m2_b32(x_int, 0, vl);
  
#ifndef __FAST_MATH__
  unsigned long long inf_ull = 0x7ff0000000000000; // mask for +inf
  double inf_d = *(double*)(&inf_ull); // +inf

  vbool32_t zero_less_mask = __riscv_vmsgtu_vx_u64m2_b32(xi, 0x8000000000000000ULL, vl);
  vbool32_t inf_mask = __riscv_vmseq_vx_u64m2_b32(x_int, inf_ull, vl);
  vbool32_t nan_mask = __riscv_vmsgtu_vx_u64m2_b32(x_int, 0x7ff0000000000000, vl);
  vbool32_t mask_sNaN = __riscv_vmand_mm_b32(
            __riscv_vmsltu_vx_u64m2_b32(x_int, 0x7ff8000000000000, vl), nan_mask, vl);
  mask_sNaN = __riscv_vmor_mm_b32(mask_sNaN, zero_less_mask, vl);
  
  vbool32_t special_mask = __riscv_vmor_mm_b32(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b32(special_mask, nan_mask, vl);

  vfloat64m2_t x_spec = __riscv_vfmerge_vfm_f64m2(x, 0, special_mask, vl);

  vfloat64m2_t one = __riscv_vfmv_v_f_f64m2(1.0, vl);
  vbool32_t denormal_mask = __riscv_vmsltu_vx_u64m2_b32(x_int, 0x0010000000000000, vl);
  vfloat64m2_t denorm_mult = 
           __riscv_vfmerge_vfm_f64m2(one, 0x1.0p+60, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f64m2(x_spec, denorm_mult, vl);

  x_int = __riscv_vreinterpret_v_f64m2_u64m2(x_spec);
#endif

  vuint64m2_t mantissa_in_x = __riscv_vand_vx_u64m2(x_int, 0x000fffffffffffff, vl);
  vuint64m2_t order_in_x = __riscv_vand_vx_u64m2(x_int, 0x7ff0000000000000LL, vl);
  order_in_x = __riscv_vsrl_vx_u64m2(order_in_x, 52, vl);

  vuint64m2_t reduced_x_int = __riscv_vor_vx_u64m2(mantissa_in_x, 0x3ff0000000000000, vl);
  vfloat64m2_t reduced_x = __riscv_vreinterpret_v_u64m2_f64m2(reduced_x_int);

  vuint64m2_t low_ind_bit = __riscv_vand_vx_u64m2(order_in_x, 0x1, vl);

  vfloat64m2_t y0 = __riscv_vreinterpret_v_u64m2_f64m2(
    __riscv_vrsub_vx_u64m2(__riscv_vsrl_vx_u64m2(reduced_x_int, 1, vl), 0x5fe6eb50c7b537a9, vl));
  
  vfloat64m2_t xx = __riscv_vfmul_vv_f64m2(y0, reduced_x, vl);
  vfloat64m2_t h = __riscv_vfmul_vf_f64m2(y0, 0.5, vl);
  vfloat64m2_t r = __riscv_vfrsub_vf_f64m2(__riscv_vfmul_vv_f64m2(xx, h, vl), 0.5, vl);

  xx = __riscv_vfmacc_vv_f64m2(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m2(h, h, r, vl);
  r = __riscv_vfrsub_vf_f64m2(__riscv_vfmul_vv_f64m2(xx, h, vl), 0.5, vl);
  xx = __riscv_vfmacc_vv_f64m2(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m2(h, h, r, vl);
  r = __riscv_vfrsub_vf_f64m2(__riscv_vfmul_vv_f64m2(xx, h, vl), 0.5, vl);
  xx = __riscv_vfmacc_vv_f64m2(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m2(h, h, r, vl);
  r = __riscv_vfrsub_vf_f64m2(__riscv_vfmul_vv_f64m2(xx, h, vl), 0.5, vl);
  xx = __riscv_vfmacc_vv_f64m2(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m2(h, h, r, vl);
  
  r = __riscv_vfnmsac_vv_f64m2(reduced_x, xx, xx, vl);

  vfloat64m2_t zh = __riscv_vfmacc_vv_f64m2(xx, r, h, vl); // high-part of sqrt value
  vfloat64m2_t zl = __riscv_vfmacc_vv_f64m2(__riscv_vfsub_vv_f64m2(xx, zh, vl), r, h, vl);

  vuint64m2_t new_exp = __riscv_vadd_vx_u64m2(order_in_x, 1023, vl);
  new_exp = __riscv_vsrl_vx_u64m2(new_exp, 1, vl);
  vfloat64m2_t order_exp = __riscv_vreinterpret_v_u64m2_f64m2(
                           __riscv_vsll_vx_u64m2(new_exp, 52, vl));
  
  vbool32_t is_odd = __riscv_vmseq_vx_u64m2_b32(low_ind_bit, 1, vl);
  vfloat64m2_t order_low = __riscv_vfmerge_vfm_f64m2(
    __riscv_vfmv_v_f_f64m2(0x1.6a09e667f3bccp+0, vl), 1.0, is_odd, vl);
  vfloat64m2_t order_low_ = __riscv_vfmerge_vfm_f64m2(
    __riscv_vfmv_v_f_f64m2(0x1.21165f626cdd5p-53f, vl), 0.0, is_odd, vl);
  vfloat64m2_t h_factor = __riscv_vfmerge_vfm_f64m2(
      __riscv_vfmv_v_f_f64m2(0x1.6a09e667f3bcdp-1, vl), 1.0, is_odd, vl);
  vuint64m2_t red_exp = __riscv_vrsub_vx_u64m2(low_ind_bit, 1024, vl);
  vfloat64m2_t order_reduce = __riscv_vreinterpret_v_u64m2_f64m2(
                                __riscv_vsll_vx_u64m2(red_exp, 52, vl));
  
  vfloat64m2_t sh = __riscv_vfmul_vv_f64m2(zh, order_low, vl);
  vfloat64m2_t sl = __riscv_vfmsac_vv_f64m2(sh, zh, order_low, vl);
  
  sl = __riscv_vfmacc_vv_f64m2(sl, zh, order_low_, vl);
  sl = __riscv_vfmacc_vv_f64m2(sl, zl, order_low, vl);
  
  vfloat64m2_t sqrt_value = __riscv_vfadd_vv_f64m2(sh, sl, vl);
  
  reduced_x = __riscv_vfmul_vv_f64m2(reduced_x, order_reduce, vl);
  r = __riscv_vfnmsac_vv_f64m2(reduced_x, sqrt_value, sqrt_value, vl);
  r = __riscv_vfmul_vv_f64m2(r, h_factor, vl);
  
  sqrt_value = __riscv_vfmacc_vv_f64m2(sqrt_value, r, h, vl);
  sqrt_value = __riscv_vfmul_vv_f64m2(sqrt_value, order_exp, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m2(sqrt_value, inf_d, inf_mask, vl);

  sqrt_value = __riscv_vmerge_vvm_f64m2(sqrt_value, x, nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b32(mask_sNaN, vl);
  if (issNaN) {
    volatile double x1 = 0.0 / 0.0;
    sqrt_value = __riscv_vfmerge_vfm_f64m2(sqrt_value, x1, mask_sNaN, vl);
  }

  denorm_mult = 
    __riscv_vfmerge_vfm_f64m2(one, 0x1.0p-30, denormal_mask, vl);
  sqrt_value = __riscv_vfmul_vv_f64m2(sqrt_value, denorm_mult, vl);
#endif

  sqrt_value = __riscv_vmerge_vvm_f64m2(sqrt_value, x, zero_mask, vl);
  return sqrt_value; 
}

vfloat64m4_t __riscv_vsqrt_f64m4(vfloat64m4_t x, size_t vl)
{
  vuint64m4_t xi = __riscv_vreinterpret_v_f64m4_u64m4(x);
  vuint64m4_t x_int = __riscv_vand_vx_u64m4(xi, 0x7fffffffffffffff, vl);
  vbool16_t zero_mask = __riscv_vmseq_vx_u64m4_b16(x_int, 0, vl);
 
#ifndef __FAST_MATH__
  unsigned long long inf_ull = 0x7ff0000000000000; // mask for +inf
  double inf_d = *(double*)(&inf_ull); // +inf

  vbool16_t zero_less_mask = __riscv_vmsgtu_vx_u64m4_b16(xi, 0x8000000000000000ULL, vl);
  vbool16_t inf_mask = __riscv_vmseq_vx_u64m4_b16(x_int, inf_ull, vl);
  vbool16_t nan_mask = __riscv_vmsgtu_vx_u64m4_b16(x_int, 0x7ff0000000000000, vl);
  vbool16_t mask_sNaN = __riscv_vmand_mm_b16(
            __riscv_vmsltu_vx_u64m4_b16(x_int, 0x7ff8000000000000, vl), nan_mask, vl);
  mask_sNaN = __riscv_vmor_mm_b16(mask_sNaN, zero_less_mask, vl);
  
  vbool16_t special_mask = __riscv_vmor_mm_b16(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b16(special_mask, nan_mask, vl);

  vfloat64m4_t x_spec = __riscv_vfmerge_vfm_f64m4(x, 0, special_mask, vl);

  vfloat64m4_t one = __riscv_vfmv_v_f_f64m4(1.0, vl);
  vbool16_t denormal_mask = __riscv_vmsltu_vx_u64m4_b16(x_int, 0x0010000000000000, vl);
  vfloat64m4_t denorm_mult = 
           __riscv_vfmerge_vfm_f64m4(one, 0x1.0p+60, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f64m4(x_spec, denorm_mult, vl);

  x_int = __riscv_vreinterpret_v_f64m4_u64m4(x_spec);
#endif

  vuint64m4_t mantissa_in_x = __riscv_vand_vx_u64m4(x_int, 0x000fffffffffffff, vl);
  vuint64m4_t order_in_x = __riscv_vand_vx_u64m4(x_int, 0x7ff0000000000000LL, vl);
  order_in_x = __riscv_vsrl_vx_u64m4(order_in_x, 52, vl);

  vuint64m4_t reduced_x_int = __riscv_vor_vx_u64m4(mantissa_in_x, 0x3ff0000000000000, vl);
  vfloat64m4_t reduced_x = __riscv_vreinterpret_v_u64m4_f64m4(reduced_x_int);

  vuint64m4_t low_ind_bit = __riscv_vand_vx_u64m4(order_in_x, 0x1, vl);

  vfloat64m4_t y0 = __riscv_vreinterpret_v_u64m4_f64m4(
    __riscv_vrsub_vx_u64m4(__riscv_vsrl_vx_u64m4(reduced_x_int, 1, vl), 0x5fe6eb50c7b537a9, vl));
  
  vfloat64m4_t xx = __riscv_vfmul_vv_f64m4(y0, reduced_x, vl);
  vfloat64m4_t h = __riscv_vfmul_vf_f64m4(y0, 0.5, vl);
  vfloat64m4_t r = __riscv_vfrsub_vf_f64m4(__riscv_vfmul_vv_f64m4(xx, h, vl), 0.5, vl);

  xx = __riscv_vfmacc_vv_f64m4(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m4(h, h, r, vl);
  r = __riscv_vfrsub_vf_f64m4(__riscv_vfmul_vv_f64m4(xx, h, vl), 0.5, vl);
  xx = __riscv_vfmacc_vv_f64m4(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m4(h, h, r, vl);
  r = __riscv_vfrsub_vf_f64m4(__riscv_vfmul_vv_f64m4(xx, h, vl), 0.5, vl);
  xx = __riscv_vfmacc_vv_f64m4(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m4(h, h, r, vl);
  r = __riscv_vfrsub_vf_f64m4(__riscv_vfmul_vv_f64m4(xx, h, vl), 0.5, vl);
  xx = __riscv_vfmacc_vv_f64m4(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m4(h, h, r, vl);
  
  r = __riscv_vfnmsac_vv_f64m4(reduced_x, xx, xx, vl);

  vfloat64m4_t zh = __riscv_vfmacc_vv_f64m4(xx, r, h, vl); // high-part of sqrt value
  vfloat64m4_t zl = __riscv_vfmacc_vv_f64m4(__riscv_vfsub_vv_f64m4(xx, zh, vl), r, h, vl);

  vuint64m4_t new_exp = __riscv_vadd_vx_u64m4(order_in_x, 1023, vl);
  new_exp = __riscv_vsrl_vx_u64m4(new_exp, 1, vl);
  vfloat64m4_t order_exp = __riscv_vreinterpret_v_u64m4_f64m4(
                           __riscv_vsll_vx_u64m4(new_exp, 52, vl));
  
  vbool16_t is_odd = __riscv_vmseq_vx_u64m4_b16(low_ind_bit, 1, vl);
  vfloat64m4_t order_low = __riscv_vfmerge_vfm_f64m4(
    __riscv_vfmv_v_f_f64m4(0x1.6a09e667f3bccp+0, vl), 1.0, is_odd, vl);
  vfloat64m4_t order_low_ = __riscv_vfmerge_vfm_f64m4(
    __riscv_vfmv_v_f_f64m4(0x1.21165f626cdd5p-53f, vl), 0.0, is_odd, vl);
  vfloat64m4_t h_factor = __riscv_vfmerge_vfm_f64m4(
      __riscv_vfmv_v_f_f64m4(0x1.6a09e667f3bcdp-1, vl), 1.0, is_odd, vl);
  vuint64m4_t red_exp = __riscv_vrsub_vx_u64m4(low_ind_bit, 1024, vl);
  vfloat64m4_t order_reduce = __riscv_vreinterpret_v_u64m4_f64m4(
                                __riscv_vsll_vx_u64m4(red_exp, 52, vl));
  
  vfloat64m4_t sh = __riscv_vfmul_vv_f64m4(zh, order_low, vl);
  vfloat64m4_t sl = __riscv_vfmsac_vv_f64m4(sh, zh, order_low, vl);
  
  sl = __riscv_vfmacc_vv_f64m4(sl, zh, order_low_, vl);
  sl = __riscv_vfmacc_vv_f64m4(sl, zl, order_low, vl);
  
  vfloat64m4_t sqrt_value = __riscv_vfadd_vv_f64m4(sh, sl, vl);
  
  reduced_x = __riscv_vfmul_vv_f64m4(reduced_x, order_reduce, vl);
  r = __riscv_vfnmsac_vv_f64m4(reduced_x, sqrt_value, sqrt_value, vl);
  r = __riscv_vfmul_vv_f64m4(r, h_factor, vl);
  
  sqrt_value = __riscv_vfmacc_vv_f64m4(sqrt_value, r, h, vl);
  sqrt_value = __riscv_vfmul_vv_f64m4(sqrt_value, order_exp, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m4(sqrt_value, inf_d, inf_mask, vl);

  sqrt_value = __riscv_vmerge_vvm_f64m4(sqrt_value, x, nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
  if (issNaN) {
    volatile double x1 = 0.0 / 0.0;
    sqrt_value = __riscv_vfmerge_vfm_f64m4(sqrt_value, x1, mask_sNaN, vl);
  }

  denorm_mult = 
    __riscv_vfmerge_vfm_f64m4(one, 0x1.0p-30, denormal_mask, vl);
  sqrt_value = __riscv_vfmul_vv_f64m4(sqrt_value, denorm_mult, vl);
#endif

  sqrt_value = __riscv_vmerge_vvm_f64m4(sqrt_value, x, zero_mask, vl);
  return sqrt_value; 
}

vfloat64m8_t __riscv_vsqrt_f64m8(vfloat64m8_t x, size_t vl)
{
  vuint64m8_t xi = __riscv_vreinterpret_v_f64m8_u64m8(x);
  vuint64m8_t x_int = __riscv_vand_vx_u64m8(xi, 0x7fffffffffffffff, vl);
  vbool8_t zero_mask = __riscv_vmseq_vx_u64m8_b8(x_int, 0, vl);
 
#ifndef __FAST_MATH__
  unsigned long long inf_ull = 0x7ff0000000000000; // mask for +inf
  double inf_d = *(double*)(&inf_ull); // +inf

  vbool8_t zero_less_mask = __riscv_vmsgtu_vx_u64m8_b8(xi, 0x8000000000000000ULL, vl);
  vbool8_t inf_mask = __riscv_vmseq_vx_u64m8_b8(x_int, inf_ull, vl);
  vbool8_t nan_mask = __riscv_vmsgtu_vx_u64m8_b8(x_int, 0x7ff0000000000000, vl);
  vbool8_t mask_sNaN = __riscv_vmand_mm_b8(
            __riscv_vmsltu_vx_u64m8_b8(x_int, 0x7ff8000000000000, vl), nan_mask, vl);
  mask_sNaN = __riscv_vmor_mm_b8(mask_sNaN, zero_less_mask, vl);
  
  vbool8_t special_mask = __riscv_vmor_mm_b8(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b8(special_mask, nan_mask, vl);

  vfloat64m8_t x_spec = __riscv_vfmerge_vfm_f64m8(x, 0, special_mask, vl);

  vfloat64m8_t one = __riscv_vfmv_v_f_f64m8(1.0, vl);
  vbool8_t denormal_mask = __riscv_vmsltu_vx_u64m8_b8(x_int, 0x0010000000000000, vl);
  vfloat64m8_t denorm_mult = 
           __riscv_vfmerge_vfm_f64m8(one, 0x1.0p+60, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f64m8(x_spec, denorm_mult, vl);

  x_int = __riscv_vreinterpret_v_f64m8_u64m8(x_spec);
#endif

  vuint64m8_t mantissa_in_x = __riscv_vand_vx_u64m8(x_int, 0x000fffffffffffff, vl);
  vuint64m8_t order_in_x = __riscv_vand_vx_u64m8(x_int, 0x7ff0000000000000LL, vl);
  order_in_x = __riscv_vsrl_vx_u64m8(order_in_x, 52, vl);

  vuint64m8_t reduced_x_int = __riscv_vor_vx_u64m8(mantissa_in_x, 0x3ff0000000000000, vl);
  vfloat64m8_t reduced_x = __riscv_vreinterpret_v_u64m8_f64m8(reduced_x_int);

  vuint64m8_t low_ind_bit = __riscv_vand_vx_u64m8(order_in_x, 0x1, vl);

  vfloat64m8_t y0 = __riscv_vreinterpret_v_u64m8_f64m8(
    __riscv_vrsub_vx_u64m8(__riscv_vsrl_vx_u64m8(reduced_x_int, 1, vl), 0x5fe6eb50c7b537a9, vl));
  
  vfloat64m8_t xx = __riscv_vfmul_vv_f64m8(y0, reduced_x, vl);
  vfloat64m8_t h = __riscv_vfmul_vf_f64m8(y0, 0.5, vl);
  vfloat64m8_t r = __riscv_vfrsub_vf_f64m8(__riscv_vfmul_vv_f64m8(xx, h, vl), 0.5, vl);

  xx = __riscv_vfmacc_vv_f64m8(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m8(h, h, r, vl);
  r = __riscv_vfrsub_vf_f64m8(__riscv_vfmul_vv_f64m8(xx, h, vl), 0.5, vl);
  xx = __riscv_vfmacc_vv_f64m8(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m8(h, h, r, vl);
  r = __riscv_vfrsub_vf_f64m8(__riscv_vfmul_vv_f64m8(xx, h, vl), 0.5, vl);
  xx = __riscv_vfmacc_vv_f64m8(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m8(h, h, r, vl);
  r = __riscv_vfrsub_vf_f64m8(__riscv_vfmul_vv_f64m8(xx, h, vl), 0.5, vl);
  xx = __riscv_vfmacc_vv_f64m8(xx, xx, r, vl);
  h = __riscv_vfmacc_vv_f64m8(h, h, r, vl);
  
  r = __riscv_vfnmsac_vv_f64m8(reduced_x, xx, xx, vl);

  vfloat64m8_t zh = __riscv_vfmacc_vv_f64m8(xx, r, h, vl); // high-part of sqrt value
  vfloat64m8_t zl = __riscv_vfmacc_vv_f64m8(__riscv_vfsub_vv_f64m8(xx, zh, vl), r, h, vl);

  vuint64m8_t new_exp = __riscv_vadd_vx_u64m8(order_in_x, 1023, vl);
  new_exp = __riscv_vsrl_vx_u64m8(new_exp, 1, vl);
  vfloat64m8_t order_exp = __riscv_vreinterpret_v_u64m8_f64m8(
                           __riscv_vsll_vx_u64m8(new_exp, 52, vl));
  
  vbool8_t is_odd = __riscv_vmseq_vx_u64m8_b8(low_ind_bit, 1, vl);
  vfloat64m8_t order_low = __riscv_vfmerge_vfm_f64m8(
    __riscv_vfmv_v_f_f64m8(0x1.6a09e667f3bccp+0, vl), 1.0, is_odd, vl);
  vfloat64m8_t order_low_ = __riscv_vfmerge_vfm_f64m8(
    __riscv_vfmv_v_f_f64m8(0x1.21165f626cdd5p-53f, vl), 0.0, is_odd, vl);
  vfloat64m8_t h_factor = __riscv_vfmerge_vfm_f64m8(
      __riscv_vfmv_v_f_f64m8(0x1.6a09e667f3bcdp-1, vl), 1.0, is_odd, vl);
  vuint64m8_t red_exp = __riscv_vrsub_vx_u64m8(low_ind_bit, 1024, vl);
  vfloat64m8_t order_reduce = __riscv_vreinterpret_v_u64m8_f64m8(
                                __riscv_vsll_vx_u64m8(red_exp, 52, vl));
  
  vfloat64m8_t sh = __riscv_vfmul_vv_f64m8(zh, order_low, vl);
  vfloat64m8_t sl = __riscv_vfmsac_vv_f64m8(sh, zh, order_low, vl);
  
  sl = __riscv_vfmacc_vv_f64m8(sl, zh, order_low_, vl);
  sl = __riscv_vfmacc_vv_f64m8(sl, zl, order_low, vl);
  
  vfloat64m8_t sqrt_value = __riscv_vfadd_vv_f64m8(sh, sl, vl);
  
  reduced_x = __riscv_vfmul_vv_f64m8(reduced_x, order_reduce, vl);
  r = __riscv_vfnmsac_vv_f64m8(reduced_x, sqrt_value, sqrt_value, vl);
  r = __riscv_vfmul_vv_f64m8(r, h_factor, vl);
  
  sqrt_value = __riscv_vfmacc_vv_f64m8(sqrt_value, r, h, vl);
  sqrt_value = __riscv_vfmul_vv_f64m8(sqrt_value, order_exp, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m8(sqrt_value, inf_d, inf_mask, vl);

  sqrt_value = __riscv_vmerge_vvm_f64m8(sqrt_value, x, nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b8(mask_sNaN, vl);
  if (issNaN) {
    volatile double x1 = 0.0 / 0.0;
    sqrt_value = __riscv_vfmerge_vfm_f64m8(sqrt_value, x1, mask_sNaN, vl);
  }

  denorm_mult = 
    __riscv_vfmerge_vfm_f64m8(one, 0x1.0p-30, denormal_mask, vl);
  sqrt_value = __riscv_vfmul_vv_f64m8(sqrt_value, denorm_mult, vl);
#endif

  sqrt_value = __riscv_vmerge_vvm_f64m8(sqrt_value, x, zero_mask, vl);
  return sqrt_value; 
}
#endif