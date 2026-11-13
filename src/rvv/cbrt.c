/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2026 The RVVMF Authors (Alexander Sysoyev)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  cbrt.c                                       *
 *   Contains: intrinsic function sqrt for f64, f32, f16 *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input VL number of elements in vector register        *
 *                                                       *
 * Return value: cube root of the elements of vector V   *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

static _Float16 tab_high_f16[3] =
{
  0x1.0p+0f16,   // cbrt(1)
  0x1.428p+0f16, // cbrt(2)
  0x1.964p+0f16  // cbrt(4)
};
static _Float16 tab_low_f16[3] =
{
  0x0.0p+0f16,    // cbrt(1)
  0x1.464p-13f16, // cbrt(2)
  0x1.ff8p-12f16  // cbrt(4)
};
static float tab_high_flt[3] =
{
  0x1.0p+0f,      // cbrt(1)
  0x1.428a30p+0f, // cbrt(2)
  0x1.965feap+0f  // cbrt(4)
};
static float tab_low_flt[3] =
{
  0x0.0p+0f,        // cbrt(1)
  -0x1.9ca36p-26f,  // cbrt(2)
  0x1.4f5b9p-26f    // cbrt(4)
};
static double tab_high_dbl[3] =
{
  0x1.0p+0,             // cbrt(1)
  0x1.428a2f98d728ap+0, // cbrt(2)
  0x1.965fea53d6e3cp+0  // cbrt(4)
};
static double tab_low_dbl[3] =
{
  0x0.0p+0,              // cbrt(1)
  0x1.c447bb56e2b7cp-53, // cbrt(2)
  0x1.0560b3335687cp-53  // cbrt(4)
};

vfloat16m1_t __riscv_vcbrt_f16m1(vfloat16m1_t x, size_t vl)
{
  vuint16m1_t orig_x = __riscv_vreinterpret_v_f16m1_u16m1(x);
  vuint16m1_t orig_x_abs = __riscv_vand_vx_u16m1(orig_x, 0x7fff, vl);
  vbool16_t zero_mask = __riscv_vmseq_vx_u16m1_b16(orig_x_abs, 0, vl);
   
  vfloat16m1_t one = __riscv_vfmv_v_f_f16m1(1.0f16, vl);
  
#ifndef __FAST_MATH__
  vbool16_t nan_mask = __riscv_vmsgtu_vx_u16m1_b16(orig_x_abs, 0x7c00, vl);
  
  vbool16_t special_mask = __riscv_vmor_mm_b16(zero_mask,
                        __riscv_vmsgeu_vx_u16m1_b16(orig_x_abs, 0x7c00, vl), vl);

  vfloat16m1_t x_abs = __riscv_vreinterpret_v_u16m1_f16m1(orig_x_abs);
  vfloat16m1_t x_spec = __riscv_vfmerge_vfm_f16m1(x_abs, 1.0f16, special_mask, vl);
  
  vbool16_t denormal_mask = __riscv_vmsltu_vx_u16m1_b16(orig_x_abs, 0x0400, vl);
  vfloat16m1_t denorm_mult = 
    __riscv_vfmerge_vfm_f16m1(one, 0x1.0p+12f16, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f16m1(x_spec, denorm_mult, vl);

  vuint16m1_t x_int = __riscv_vreinterpret_v_f16m1_u16m1(x_spec);
#else
  vuint16m1_t x_int = orig_x_abs;
#endif

  vuint16m1_t mantissa_in_x = __riscv_vand_vx_u16m1(x_int, 0x03ff, vl);
  vuint16m1_t order_in_x = __riscv_vsrl_vx_u16m1(x_int, 10, vl);

  vuint16m1_t order_plus = __riscv_vadd_vx_u16m1(order_in_x, 15, vl);
  vuint16m1_t div3floorf = __riscv_vsrl_vx_u16m1(
      __riscv_vmul_vx_u16m1(order_plus, 342, vl), 10, vl);
  vuint16m1_t index = __riscv_vsub_vv_u16m1(order_plus, div3floorf, vl);
  index = __riscv_vsub_vv_u16m1(index, __riscv_vsll_vx_u16m1(div3floorf, 1, vl), vl);
  vuint16m1_t order_in_cbrt = __riscv_vadd_vx_u16m1(div3floorf, 5, vl);
  
  vuint16m1_t reduced_x_int = __riscv_vor_vx_u16m1(mantissa_in_x, 0x3c00, vl);
  vfloat16m1_t reduced_x = __riscv_vreinterpret_v_u16m1_f16m1(reduced_x_int);
  
  vfloat16m1_t yn = __riscv_vfmv_v_f_f16m1(0x1.7fp-1f16, vl);
  yn = __riscv_vfnmsac_vf_f16m1(yn, 0x1.eb8p-4f16, reduced_x, vl);
  yn = __riscv_vfmadd_vv_f16m1(yn, reduced_x, __riscv_vfmv_v_f_f16m1(-0x1.c6p0f16, vl), vl);
  yn = __riscv_vfmadd_vv_f16m1(yn, reduced_x, __riscv_vfmv_v_f_f16m1(0x1.128p1f16, vl), vl);

  //y_(n + 1) = y_n + y_n / 3 (1 - x ^ 2 y_n ^ 3)
  //                         x * x * y * y * y = (x * y) * (x * y) * y
  vfloat16m1_t r = __riscv_vfmul_vv_f16m1(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f16m1(r, r, vl);
  r = __riscv_vfmul_vv_f16m1(r, yn, vl);
  r = __riscv_vfnmsub_vv_f16m1(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f16m1(yn, 0x1.554p-2f16, r, vl);

  vfloat16m1_t rh = __riscv_vfmul_vv_f16m1(reduced_x, yn, vl);
  vfloat16m1_t rl = __riscv_vfmsac_vv_f16m1(rh, reduced_x, yn, vl);

  // rh^2
  vfloat16m1_t hi = __riscv_vfmul_vv_f16m1(rh, rh, vl);
  vfloat16m1_t lo = __riscv_vfmsac_vv_f16m1(hi, rh, rh, vl);
  vfloat16m1_t cross = __riscv_vfmul_vv_f16m1(rl, rh, vl);
  cross = __riscv_vfadd_vv_f16m1(cross, cross, vl);
  lo = __riscv_vfadd_vv_f16m1(lo, cross, vl);
  
  rh = __riscv_vfadd_vv_f16m1(hi, lo, vl);
  vfloat16m1_t rl_tmp = __riscv_vfsub_vv_f16m1(rh, hi, vl);
  rl = __riscv_vfsub_vv_f16m1(lo, rl_tmp, vl);

  vfloat16m1_t rh1 = __riscv_vfnmsac_vv_f16m1(one, rh, yn, vl); 
  vfloat16m1_t rl1 = __riscv_vfmul_vv_f16m1(rl, yn, vl);

  hi = __riscv_vfmul_vv_f16m1(rh1, yn, vl);
  lo = __riscv_vfmsac_vv_f16m1(hi, rh1, yn, vl);
  lo = __riscv_vfnmsac_vv_f16m1(lo, rl1, yn, vl);
  
  vfloat16m1_t yh = __riscv_vfmacc_vf_f16m1(yn, 0x1.554p-2f16, hi, vl);
  vfloat16m1_t yl = __riscv_vfsub_vv_f16m1(yn, yh, vl);
  yl = __riscv_vfmacc_vf_f16m1(yl, 0x1.554p-2f16, hi, vl);
  yl = __riscv_vfmacc_vf_f16m1(yl, 0x1.554p-2f16, lo, vl);

  vbool16_t is_1 = __riscv_vmseq_vx_u16m1_b16(index, 1, vl);
  vbool16_t is_2 = __riscv_vmseq_vx_u16m1_b16(index, 2, vl);
  
  vfloat16m1_t ord_hi = __riscv_vfmv_v_f_f16m1(tab_high_f16[0], vl);
  ord_hi = __riscv_vfmerge_vfm_f16m1(ord_hi, tab_high_f16[1], is_1, vl);
  ord_hi = __riscv_vfmerge_vfm_f16m1(ord_hi, tab_high_f16[2], is_2, vl);

  vfloat16m1_t ord_lo = __riscv_vfmv_v_f_f16m1(tab_low_f16[0], vl);
  ord_lo = __riscv_vfmerge_vfm_f16m1(ord_lo, tab_low_f16[1], is_1, vl);
  ord_lo = __riscv_vfmerge_vfm_f16m1(ord_lo, tab_low_f16[2], is_2, vl);
  
  hi = __riscv_vfmul_vv_f16m1(yh, reduced_x, vl);
  lo = __riscv_vfmsac_vv_f16m1(hi, yh, reduced_x, vl);
  lo = __riscv_vfmacc_vv_f16m1(lo, yl, reduced_x, vl);
  
  rh = __riscv_vfmul_vv_f16m1(ord_hi, hi, vl);
  rl = __riscv_vfmsac_vv_f16m1(rh, ord_hi, hi, vl);
  rl = __riscv_vfmacc_vv_f16m1(rl, ord_lo, hi, vl);
  rl = __riscv_vfmacc_vv_f16m1(rl, ord_hi, lo, vl);
  rh = __riscv_vfadd_vv_f16m1(rh, rl, vl);
  
  vuint16m1_t mask_for_order = __riscv_vsll_vx_u16m1(order_in_cbrt, 10, vl);
  vfloat16m1_t order = __riscv_vreinterpret_v_u16m1_f16m1(mask_for_order);
  rh = __riscv_vfmul_vv_f16m1(rh, order, vl);

  vfloat16m1_t cbrt_value = __riscv_vfsgnj_vv_f16m1(rh, x, vl);

#ifndef __FAST_MATH__
  vfloat16m1_t denorm_mult_back = 
    __riscv_vfmerge_vfm_f16m1(one, 0x1.0p-4f16, denormal_mask, vl);
  cbrt_value = __riscv_vfmul_vv_f16m1(cbrt_value, denorm_mult_back, vl);

  cbrt_value = __riscv_vmerge_vvm_f16m1(cbrt_value, x, special_mask, vl);

  vbool16_t mask_sNaN = __riscv_vmand_mm_b16(
            __riscv_vmsltu_vx_u16m1_b16(orig_x_abs, 0x7e00, vl), nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
  if (issNaN) {
    volatile _Float16 x1 = 0.0f16 / 0.0f16;
    cbrt_value = __riscv_vfmerge_vfm_f16m1(cbrt_value, x1, mask_sNaN, vl);
  }
#else
  cbrt_value = __riscv_vmerge_vvm_f16m1(cbrt_value, x, zero_mask, vl);  
#endif

  return cbrt_value; 
}

vfloat16m2_t __riscv_vcbrt_f16m2(vfloat16m2_t x, size_t vl)
{
  vuint16m2_t orig_x = __riscv_vreinterpret_v_f16m2_u16m2(x);
  vuint16m2_t orig_x_abs = __riscv_vand_vx_u16m2(orig_x, 0x7fff, vl);
  vbool8_t zero_mask = __riscv_vmseq_vx_u16m2_b8(orig_x_abs, 0, vl);
   
  vfloat16m2_t one = __riscv_vfmv_v_f_f16m2(1.0f16, vl);
  
#ifndef __FAST_MATH__
  vbool8_t nan_mask = __riscv_vmsgtu_vx_u16m2_b8(orig_x_abs, 0x7c00, vl);
  
  vbool8_t special_mask = __riscv_vmor_mm_b8(zero_mask,
                        __riscv_vmsgeu_vx_u16m2_b8(orig_x_abs, 0x7c00, vl), vl);

  vfloat16m2_t x_abs = __riscv_vreinterpret_v_u16m2_f16m2(orig_x_abs);
  vfloat16m2_t x_spec = __riscv_vfmerge_vfm_f16m2(x_abs, 1.0f16, special_mask, vl);
  
  vbool8_t denormal_mask = __riscv_vmsltu_vx_u16m2_b8(orig_x_abs, 0x0400, vl);
  vfloat16m2_t denorm_mult = 
    __riscv_vfmerge_vfm_f16m2(one, 0x1.0p+12f16, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f16m2(x_spec, denorm_mult, vl);

  vuint16m2_t x_int = __riscv_vreinterpret_v_f16m2_u16m2(x_spec);
#else
  vuint16m2_t x_int = orig_x_abs;
#endif

  vuint16m2_t mantissa_in_x = __riscv_vand_vx_u16m2(x_int, 0x03ff, vl);
  vuint16m2_t order_in_x = __riscv_vsrl_vx_u16m2(x_int, 10, vl);

  vuint16m2_t order_plus = __riscv_vadd_vx_u16m2(order_in_x, 15, vl);
  vuint16m2_t div3floorf = __riscv_vsrl_vx_u16m2(
      __riscv_vmul_vx_u16m2(order_plus, 342, vl), 10, vl);
  vuint16m2_t index = __riscv_vsub_vv_u16m2(order_plus, div3floorf, vl);
  index = __riscv_vsub_vv_u16m2(index, __riscv_vsll_vx_u16m2(div3floorf, 1, vl), vl);
  vuint16m2_t order_in_cbrt = __riscv_vadd_vx_u16m2(div3floorf, 5, vl);
  
  vuint16m2_t reduced_x_int = __riscv_vor_vx_u16m2(mantissa_in_x, 0x3c00, vl);
  vfloat16m2_t reduced_x = __riscv_vreinterpret_v_u16m2_f16m2(reduced_x_int);
  
  vfloat16m2_t yn = __riscv_vfmv_v_f_f16m2(0x1.7fp-1f16, vl);
  yn = __riscv_vfnmsac_vf_f16m2(yn, 0x1.eb8p-4f16, reduced_x, vl);
  yn = __riscv_vfmadd_vv_f16m2(yn, reduced_x, __riscv_vfmv_v_f_f16m2(-0x1.c6p0f16, vl), vl);
  yn = __riscv_vfmadd_vv_f16m2(yn, reduced_x, __riscv_vfmv_v_f_f16m2(0x1.128p1f16, vl), vl);

  //y_(n + 1) = y_n + y_n / 3 (1 - x ^ 2 y_n ^ 3)
  //                         x * x * y * y * y = (x * y) * (x * y) * y
  vfloat16m2_t r = __riscv_vfmul_vv_f16m2(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f16m2(r, r, vl);
  r = __riscv_vfmul_vv_f16m2(r, yn, vl);
  r = __riscv_vfnmsub_vv_f16m2(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f16m2(yn, 0x1.554p-2f16, r, vl);

  vfloat16m2_t rh = __riscv_vfmul_vv_f16m2(reduced_x, yn, vl);
  vfloat16m2_t rl = __riscv_vfmsac_vv_f16m2(rh, reduced_x, yn, vl);

  // rh^2
  vfloat16m2_t hi = __riscv_vfmul_vv_f16m2(rh, rh, vl);
  vfloat16m2_t lo = __riscv_vfmsac_vv_f16m2(hi, rh, rh, vl);
  vfloat16m2_t cross = __riscv_vfmul_vv_f16m2(rl, rh, vl);
  cross = __riscv_vfadd_vv_f16m2(cross, cross, vl);
  lo = __riscv_vfadd_vv_f16m2(lo, cross, vl);
  
  rh = __riscv_vfadd_vv_f16m2(hi, lo, vl);
  vfloat16m2_t rl_tmp = __riscv_vfsub_vv_f16m2(rh, hi, vl);
  rl = __riscv_vfsub_vv_f16m2(lo, rl_tmp, vl);

  vfloat16m2_t rh1 = __riscv_vfnmsac_vv_f16m2(one, rh, yn, vl); 
  vfloat16m2_t rl1 = __riscv_vfmul_vv_f16m2(rl, yn, vl);

  hi = __riscv_vfmul_vv_f16m2(rh1, yn, vl);
  lo = __riscv_vfmsac_vv_f16m2(hi, rh1, yn, vl);
  lo = __riscv_vfnmsac_vv_f16m2(lo, rl1, yn, vl);
  
  vfloat16m2_t yh = __riscv_vfmacc_vf_f16m2(yn, 0x1.554p-2f16, hi, vl);
  vfloat16m2_t yl = __riscv_vfsub_vv_f16m2(yn, yh, vl);
  yl = __riscv_vfmacc_vf_f16m2(yl, 0x1.554p-2f16, hi, vl);
  yl = __riscv_vfmacc_vf_f16m2(yl, 0x1.554p-2f16, lo, vl);

  vbool8_t is_1 = __riscv_vmseq_vx_u16m2_b8(index, 1, vl);
  vbool8_t is_2 = __riscv_vmseq_vx_u16m2_b8(index, 2, vl);
  
  vfloat16m2_t ord_hi = __riscv_vfmv_v_f_f16m2(tab_high_f16[0], vl);
  ord_hi = __riscv_vfmerge_vfm_f16m2(ord_hi, tab_high_f16[1], is_1, vl);
  ord_hi = __riscv_vfmerge_vfm_f16m2(ord_hi, tab_high_f16[2], is_2, vl);

  vfloat16m2_t ord_lo = __riscv_vfmv_v_f_f16m2(tab_low_f16[0], vl);
  ord_lo = __riscv_vfmerge_vfm_f16m2(ord_lo, tab_low_f16[1], is_1, vl);
  ord_lo = __riscv_vfmerge_vfm_f16m2(ord_lo, tab_low_f16[2], is_2, vl);
  
  hi = __riscv_vfmul_vv_f16m2(yh, reduced_x, vl);
  lo = __riscv_vfmsac_vv_f16m2(hi, yh, reduced_x, vl);
  lo = __riscv_vfmacc_vv_f16m2(lo, yl, reduced_x, vl);
  
  rh = __riscv_vfmul_vv_f16m2(ord_hi, hi, vl);
  rl = __riscv_vfmsac_vv_f16m2(rh, ord_hi, hi, vl);
  rl = __riscv_vfmacc_vv_f16m2(rl, ord_lo, hi, vl);
  rl = __riscv_vfmacc_vv_f16m2(rl, ord_hi, lo, vl);
  rh = __riscv_vfadd_vv_f16m2(rh, rl, vl);
  
  vuint16m2_t mask_for_order = __riscv_vsll_vx_u16m2(order_in_cbrt, 10, vl);
  vfloat16m2_t order = __riscv_vreinterpret_v_u16m2_f16m2(mask_for_order);
  rh = __riscv_vfmul_vv_f16m2(rh, order, vl);

  vfloat16m2_t cbrt_value = __riscv_vfsgnj_vv_f16m2(rh, x, vl);

#ifndef __FAST_MATH__
  vfloat16m2_t denorm_mult_back = 
    __riscv_vfmerge_vfm_f16m2(one, 0x1.0p-4f16, denormal_mask, vl);
  cbrt_value = __riscv_vfmul_vv_f16m2(cbrt_value, denorm_mult_back, vl);

  cbrt_value = __riscv_vmerge_vvm_f16m2(cbrt_value, x, special_mask, vl);

  vbool8_t mask_sNaN = __riscv_vmand_mm_b8(
            __riscv_vmsltu_vx_u16m2_b8(orig_x_abs, 0x7e00, vl), nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b8(mask_sNaN, vl);
  if (issNaN) {
    volatile _Float16 x1 = 0.0f16 / 0.0f16;
    cbrt_value = __riscv_vfmerge_vfm_f16m2(cbrt_value, x1, mask_sNaN, vl);
  }
#else
  cbrt_value = __riscv_vmerge_vvm_f16m2(cbrt_value, x, zero_mask, vl);  
#endif

  return cbrt_value; 
}

vfloat16m4_t __riscv_vcbrt_f16m4(vfloat16m4_t x, size_t vl)
{
  vuint16m4_t orig_x = __riscv_vreinterpret_v_f16m4_u16m4(x);
  vuint16m4_t orig_x_abs = __riscv_vand_vx_u16m4(orig_x, 0x7fff, vl);
  vbool4_t zero_mask = __riscv_vmseq_vx_u16m4_b4(orig_x_abs, 0, vl);
   
  vfloat16m4_t one = __riscv_vfmv_v_f_f16m4(1.0f16, vl);
  
#ifndef __FAST_MATH__
  vbool4_t nan_mask = __riscv_vmsgtu_vx_u16m4_b4(orig_x_abs, 0x7c00, vl);
  
  vbool4_t special_mask = __riscv_vmor_mm_b4(zero_mask,
                        __riscv_vmsgeu_vx_u16m4_b4(orig_x_abs, 0x7c00, vl), vl);

  vfloat16m4_t x_abs = __riscv_vreinterpret_v_u16m4_f16m4(orig_x_abs);
  vfloat16m4_t x_spec = __riscv_vfmerge_vfm_f16m4(x_abs, 1.0f16, special_mask, vl);
  
  vbool4_t denormal_mask = __riscv_vmsltu_vx_u16m4_b4(orig_x_abs, 0x0400, vl);
  vfloat16m4_t denorm_mult = 
    __riscv_vfmerge_vfm_f16m4(one, 0x1.0p+12f16, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f16m4(x_spec, denorm_mult, vl);

  vuint16m4_t x_int = __riscv_vreinterpret_v_f16m4_u16m4(x_spec);
#else
  vuint16m4_t x_int = orig_x_abs;
#endif

  vuint16m4_t mantissa_in_x = __riscv_vand_vx_u16m4(x_int, 0x03ff, vl);
  vuint16m4_t order_in_x = __riscv_vsrl_vx_u16m4(x_int, 10, vl);

  vuint16m4_t order_plus = __riscv_vadd_vx_u16m4(order_in_x, 15, vl);
  vuint16m4_t div3floorf = __riscv_vsrl_vx_u16m4(
      __riscv_vmul_vx_u16m4(order_plus, 342, vl), 10, vl);
  vuint16m4_t index = __riscv_vsub_vv_u16m4(order_plus, div3floorf, vl);
  index = __riscv_vsub_vv_u16m4(index, __riscv_vsll_vx_u16m4(div3floorf, 1, vl), vl);
  vuint16m4_t order_in_cbrt = __riscv_vadd_vx_u16m4(div3floorf, 5, vl);
  
  vuint16m4_t reduced_x_int = __riscv_vor_vx_u16m4(mantissa_in_x, 0x3c00, vl);
  vfloat16m4_t reduced_x = __riscv_vreinterpret_v_u16m4_f16m4(reduced_x_int);
  
  vfloat16m4_t yn = __riscv_vfmv_v_f_f16m4(0x1.7fp-1f16, vl);
  yn = __riscv_vfnmsac_vf_f16m4(yn, 0x1.eb8p-4f16, reduced_x, vl);
  yn = __riscv_vfmadd_vv_f16m4(yn, reduced_x, __riscv_vfmv_v_f_f16m4(-0x1.c6p0f16, vl), vl);
  yn = __riscv_vfmadd_vv_f16m4(yn, reduced_x, __riscv_vfmv_v_f_f16m4(0x1.128p1f16, vl), vl);

  //y_(n + 1) = y_n + y_n / 3 (1 - x ^ 2 y_n ^ 3)
  //                         x * x * y * y * y = (x * y) * (x * y) * y
  vfloat16m4_t r = __riscv_vfmul_vv_f16m4(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f16m4(r, r, vl);
  r = __riscv_vfmul_vv_f16m4(r, yn, vl);
  r = __riscv_vfnmsub_vv_f16m4(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f16m4(yn, 0x1.554p-2f16, r, vl);

  vfloat16m4_t rh = __riscv_vfmul_vv_f16m4(reduced_x, yn, vl);
  vfloat16m4_t rl = __riscv_vfmsac_vv_f16m4(rh, reduced_x, yn, vl);

  // rh^2
  vfloat16m4_t hi = __riscv_vfmul_vv_f16m4(rh, rh, vl);
  vfloat16m4_t lo = __riscv_vfmsac_vv_f16m4(hi, rh, rh, vl);
  vfloat16m4_t cross = __riscv_vfmul_vv_f16m4(rl, rh, vl);
  cross = __riscv_vfadd_vv_f16m4(cross, cross, vl);
  lo = __riscv_vfadd_vv_f16m4(lo, cross, vl);
  
  rh = __riscv_vfadd_vv_f16m4(hi, lo, vl);
  vfloat16m4_t rl_tmp = __riscv_vfsub_vv_f16m4(rh, hi, vl);
  rl = __riscv_vfsub_vv_f16m4(lo, rl_tmp, vl);

  vfloat16m4_t rh1 = __riscv_vfnmsac_vv_f16m4(one, rh, yn, vl); 
  vfloat16m4_t rl1 = __riscv_vfmul_vv_f16m4(rl, yn, vl);

  hi = __riscv_vfmul_vv_f16m4(rh1, yn, vl);
  lo = __riscv_vfmsac_vv_f16m4(hi, rh1, yn, vl);
  lo = __riscv_vfnmsac_vv_f16m4(lo, rl1, yn, vl);
  
  vfloat16m4_t yh = __riscv_vfmacc_vf_f16m4(yn, 0x1.554p-2f16, hi, vl);
  vfloat16m4_t yl = __riscv_vfsub_vv_f16m4(yn, yh, vl);
  yl = __riscv_vfmacc_vf_f16m4(yl, 0x1.554p-2f16, hi, vl);
  yl = __riscv_vfmacc_vf_f16m4(yl, 0x1.554p-2f16, lo, vl);

  vbool4_t is_1 = __riscv_vmseq_vx_u16m4_b4(index, 1, vl);
  vbool4_t is_2 = __riscv_vmseq_vx_u16m4_b4(index, 2, vl);
  
  vfloat16m4_t ord_hi = __riscv_vfmv_v_f_f16m4(tab_high_f16[0], vl);
  ord_hi = __riscv_vfmerge_vfm_f16m4(ord_hi, tab_high_f16[1], is_1, vl);
  ord_hi = __riscv_vfmerge_vfm_f16m4(ord_hi, tab_high_f16[2], is_2, vl);

  vfloat16m4_t ord_lo = __riscv_vfmv_v_f_f16m4(tab_low_f16[0], vl);
  ord_lo = __riscv_vfmerge_vfm_f16m4(ord_lo, tab_low_f16[1], is_1, vl);
  ord_lo = __riscv_vfmerge_vfm_f16m4(ord_lo, tab_low_f16[2], is_2, vl);
  
  hi = __riscv_vfmul_vv_f16m4(yh, reduced_x, vl);
  lo = __riscv_vfmsac_vv_f16m4(hi, yh, reduced_x, vl);
  lo = __riscv_vfmacc_vv_f16m4(lo, yl, reduced_x, vl);
  
  rh = __riscv_vfmul_vv_f16m4(ord_hi, hi, vl);
  rl = __riscv_vfmsac_vv_f16m4(rh, ord_hi, hi, vl);
  rl = __riscv_vfmacc_vv_f16m4(rl, ord_lo, hi, vl);
  rl = __riscv_vfmacc_vv_f16m4(rl, ord_hi, lo, vl);
  rh = __riscv_vfadd_vv_f16m4(rh, rl, vl);
  
  vuint16m4_t mask_for_order = __riscv_vsll_vx_u16m4(order_in_cbrt, 10, vl);
  vfloat16m4_t order = __riscv_vreinterpret_v_u16m4_f16m4(mask_for_order);
  rh = __riscv_vfmul_vv_f16m4(rh, order, vl);

  vfloat16m4_t cbrt_value = __riscv_vfsgnj_vv_f16m4(rh, x, vl);

#ifndef __FAST_MATH__
  vfloat16m4_t denorm_mult_back = 
    __riscv_vfmerge_vfm_f16m4(one, 0x1.0p-4f16, denormal_mask, vl);
  cbrt_value = __riscv_vfmul_vv_f16m4(cbrt_value, denorm_mult_back, vl);

  cbrt_value = __riscv_vmerge_vvm_f16m4(cbrt_value, x, special_mask, vl);

  vbool4_t mask_sNaN = __riscv_vmand_mm_b4(
            __riscv_vmsltu_vx_u16m4_b4(orig_x_abs, 0x7e00, vl), nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b4(mask_sNaN, vl);
  if (issNaN) {
    volatile _Float16 x1 = 0.0f16 / 0.0f16;
    cbrt_value = __riscv_vfmerge_vfm_f16m4(cbrt_value, x1, mask_sNaN, vl);
  }
#else
  cbrt_value = __riscv_vmerge_vvm_f16m4(cbrt_value, x, zero_mask, vl);  
#endif

  return cbrt_value; 
}

vfloat16m8_t __riscv_vcbrt_f16m8(vfloat16m8_t x, size_t vl)
{
  vuint16m8_t orig_x = __riscv_vreinterpret_v_f16m8_u16m8(x);
  vuint16m8_t orig_x_abs = __riscv_vand_vx_u16m8(orig_x, 0x7fff, vl);
  vbool2_t zero_mask = __riscv_vmseq_vx_u16m8_b2(orig_x_abs, 0, vl);
   
  vfloat16m8_t one = __riscv_vfmv_v_f_f16m8(1.0f16, vl);
  
#ifndef __FAST_MATH__
  vbool2_t nan_mask = __riscv_vmsgtu_vx_u16m8_b2(orig_x_abs, 0x7c00, vl);
  
  vbool2_t special_mask = __riscv_vmor_mm_b2(zero_mask,
                        __riscv_vmsgeu_vx_u16m8_b2(orig_x_abs, 0x7c00, vl), vl);

  vfloat16m8_t x_abs = __riscv_vreinterpret_v_u16m8_f16m8(orig_x_abs);
  vfloat16m8_t x_spec = __riscv_vfmerge_vfm_f16m8(x_abs, 1.0f16, special_mask, vl);
  
  vbool2_t denormal_mask = __riscv_vmsltu_vx_u16m8_b2(orig_x_abs, 0x0400, vl);
  vfloat16m8_t denorm_mult = 
    __riscv_vfmerge_vfm_f16m8(one, 0x1.0p+12f16, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f16m8(x_spec, denorm_mult, vl);

  vuint16m8_t x_int = __riscv_vreinterpret_v_f16m8_u16m8(x_spec);
#else
  vuint16m8_t x_int = orig_x_abs;
#endif

  vuint16m8_t mantissa_in_x = __riscv_vand_vx_u16m8(x_int, 0x03ff, vl);
  vuint16m8_t order_in_x = __riscv_vsrl_vx_u16m8(x_int, 10, vl);

  vuint16m8_t order_plus = __riscv_vadd_vx_u16m8(order_in_x, 15, vl);
  vuint16m8_t div3floorf = __riscv_vsrl_vx_u16m8(
      __riscv_vmul_vx_u16m8(order_plus, 342, vl), 10, vl);
  vuint16m8_t index = __riscv_vsub_vv_u16m8(order_plus,  div3floorf, vl);
  index = __riscv_vsub_vv_u16m8(index, __riscv_vsll_vx_u16m8(div3floorf, 1, vl), vl);
  vuint16m8_t order_in_cbrt = __riscv_vadd_vx_u16m8(div3floorf, 5, vl);
  
  vuint16m8_t reduced_x_int = __riscv_vor_vx_u16m8(mantissa_in_x, 0x3c00, vl);
  vfloat16m8_t reduced_x = __riscv_vreinterpret_v_u16m8_f16m8(reduced_x_int);
  
  vfloat16m8_t yn = __riscv_vfmv_v_f_f16m8(0x1.7fp-1f16, vl);
  yn = __riscv_vfnmsac_vf_f16m8(yn, 0x1.eb8p-4f16, reduced_x, vl);
  yn = __riscv_vfmadd_vv_f16m8(yn, reduced_x, __riscv_vfmv_v_f_f16m8(-0x1.c6p0f16, vl), vl);
  yn = __riscv_vfmadd_vv_f16m8(yn, reduced_x, __riscv_vfmv_v_f_f16m8(0x1.128p1f16, vl), vl);

  //y_(n + 1) = y_n + y_n / 3 (1 - x ^ 2 y_n ^ 3)
  //                         x * x * y * y * y = (x * y) * (x * y) * y
  vfloat16m8_t r = __riscv_vfmul_vv_f16m8(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f16m8(r, r, vl);
  r = __riscv_vfmul_vv_f16m8(r, yn, vl);
  r = __riscv_vfnmsub_vv_f16m8(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f16m8(yn, 0x1.554p-2f16, r, vl);

  vfloat16m8_t rh = __riscv_vfmul_vv_f16m8(reduced_x, yn, vl);
  vfloat16m8_t rl = __riscv_vfmsac_vv_f16m8(rh, reduced_x, yn, vl);

  // rh^2
  vfloat16m8_t hi = __riscv_vfmul_vv_f16m8(rh, rh, vl);
  vfloat16m8_t lo = __riscv_vfmsac_vv_f16m8(hi, rh, rh, vl);
  vfloat16m8_t cross = __riscv_vfmul_vv_f16m8(rl, rh, vl);
  cross = __riscv_vfadd_vv_f16m8(cross, cross, vl);
  lo = __riscv_vfadd_vv_f16m8(lo, cross, vl);
  
  rh = __riscv_vfadd_vv_f16m8(hi, lo, vl);
  vfloat16m8_t rl_tmp = __riscv_vfsub_vv_f16m8(rh, hi, vl);
  rl = __riscv_vfsub_vv_f16m8(lo, rl_tmp, vl);

  vfloat16m8_t rh1 = __riscv_vfnmsac_vv_f16m8(one, rh, yn, vl); 
  vfloat16m8_t rl1 = __riscv_vfmul_vv_f16m8(rl, yn, vl);

  hi = __riscv_vfmul_vv_f16m8(rh1, yn, vl);
  lo = __riscv_vfmsac_vv_f16m8(hi, rh1, yn, vl);
  lo = __riscv_vfnmsac_vv_f16m8(lo, rl1, yn, vl);
  
  vfloat16m8_t yh = __riscv_vfmacc_vf_f16m8(yn, 0x1.554p-2f16, hi, vl);
  vfloat16m8_t yl = __riscv_vfsub_vv_f16m8(yn, yh, vl);
  yl = __riscv_vfmacc_vf_f16m8(yl, 0x1.554p-2f16, hi, vl);
  yl = __riscv_vfmacc_vf_f16m8(yl, 0x1.554p-2f16, lo, vl);

  vbool2_t is_1 = __riscv_vmseq_vx_u16m8_b2(index, 1, vl);
  vbool2_t is_2 = __riscv_vmseq_vx_u16m8_b2(index, 2, vl);
  
  vfloat16m8_t ord_hi = __riscv_vfmv_v_f_f16m8(tab_high_f16[0], vl);
  ord_hi = __riscv_vfmerge_vfm_f16m8(ord_hi, tab_high_f16[1], is_1, vl);
  ord_hi = __riscv_vfmerge_vfm_f16m8(ord_hi, tab_high_f16[2], is_2, vl);

  vfloat16m8_t ord_lo = __riscv_vfmv_v_f_f16m8(tab_low_f16[0], vl);
  ord_lo = __riscv_vfmerge_vfm_f16m8(ord_lo, tab_low_f16[1], is_1, vl);
  ord_lo = __riscv_vfmerge_vfm_f16m8(ord_lo, tab_low_f16[2], is_2, vl);
  
  hi = __riscv_vfmul_vv_f16m8(yh, reduced_x, vl);
  lo = __riscv_vfmsac_vv_f16m8(hi, yh, reduced_x, vl);
  lo = __riscv_vfmacc_vv_f16m8(lo, yl, reduced_x, vl);
  
  rh = __riscv_vfmul_vv_f16m8(ord_hi, hi, vl);
  rl = __riscv_vfmsac_vv_f16m8(rh, ord_hi, hi, vl);
  rl = __riscv_vfmacc_vv_f16m8(rl, ord_lo, hi, vl);
  rl = __riscv_vfmacc_vv_f16m8(rl, ord_hi, lo, vl);
  rh = __riscv_vfadd_vv_f16m8(rh, rl, vl);
  
  vuint16m8_t mask_for_order = __riscv_vsll_vx_u16m8(order_in_cbrt, 10, vl);
  vfloat16m8_t order = __riscv_vreinterpret_v_u16m8_f16m8(mask_for_order);
  rh = __riscv_vfmul_vv_f16m8(rh, order, vl);

  vfloat16m8_t cbrt_value = __riscv_vfsgnj_vv_f16m8(rh, x, vl);

#ifndef __FAST_MATH__
  vfloat16m8_t denorm_mult_back = 
    __riscv_vfmerge_vfm_f16m8(one, 0x1.0p-4f16, denormal_mask, vl);
  cbrt_value = __riscv_vfmul_vv_f16m8(cbrt_value, denorm_mult_back, vl);

  cbrt_value = __riscv_vmerge_vvm_f16m8(cbrt_value, x, special_mask, vl);

  vbool2_t mask_sNaN = __riscv_vmand_mm_b2(
            __riscv_vmsltu_vx_u16m8_b2(orig_x_abs, 0x7e00, vl), nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b2(mask_sNaN, vl);
  if (issNaN) {
    volatile _Float16 x1 = 0.0f16 / 0.0f16;
    cbrt_value = __riscv_vfmerge_vfm_f16m8(cbrt_value, x1, mask_sNaN, vl);
  }
#else
  cbrt_value = __riscv_vmerge_vvm_f16m8(cbrt_value, x, zero_mask, vl);  
#endif

  return cbrt_value; 
}

vfloat32m1_t __riscv_vcbrt_f32m1(vfloat32m1_t x, size_t vl)
{
  vuint32m1_t orig_x = __riscv_vreinterpret_v_f32m1_u32m1(x);
  vuint32m1_t orig_x_abs = __riscv_vand_vx_u32m1(orig_x, 0x7fffffff, vl);
  vbool32_t zero_mask = __riscv_vmseq_vx_u32m1_b32(orig_x_abs, 0, vl);

  vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);

#ifndef __FAST_MATH__
  vbool32_t nan_mask = __riscv_vmsgtu_vx_u32m1_b32(orig_x_abs, 0x7f800000, vl);
  vbool32_t special_mask = __riscv_vmor_mm_b32(zero_mask,
                         __riscv_vmsgeu_vx_u32m1_b32(orig_x_abs, 0x7f800000, vl), vl);

  vfloat32m1_t x_abs = __riscv_vreinterpret_v_u32m1_f32m1(orig_x_abs);
  vfloat32m1_t x_spec = __riscv_vfmerge_vfm_f32m1(x_abs, 1.0f, special_mask, vl);

  vbool32_t denormal_mask = __riscv_vmsltu_vx_u32m1_b32(orig_x_abs, 0x00800000, vl);
  vfloat32m1_t denorm_mult = 
    __riscv_vfmerge_vfm_f32m1(one, 0x1.0p+24f, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f32m1(x_spec, denorm_mult, vl);

  vuint32m1_t x_int = __riscv_vreinterpret_v_f32m1_u32m1(x_spec);
#else
  vuint32m1_t x_int = orig_x_abs;
#endif

  vuint32m1_t mantissa_in_x = __riscv_vand_vx_u32m1(x_int, 0x007fffff, vl);
  vuint32m1_t order_in_x = __riscv_vsrl_vx_u32m1(x_int, 23, vl);

  vuint32m1_t order_plus = __riscv_vadd_vx_u32m1(order_in_x, 173, vl);
  vuint32m1_t div3floorf = __riscv_vsrl_vx_u32m1(
      __riscv_vmul_vx_u32m1(order_plus, 21846, vl), 16, vl);
  vuint32m1_t index = __riscv_vsub_vv_u32m1(order_plus, div3floorf, vl);
  index = __riscv_vsub_vv_u32m1(index, __riscv_vsll_vx_u32m1(div3floorf, 1, vl), vl);
  vuint32m1_t order_in_cbrt = __riscv_vadd_vx_u32m1(div3floorf, 27, vl);

  vuint32m1_t reduced_x_int = __riscv_vor_vx_u32m1(mantissa_in_x, 0x3f800000, vl);
  vfloat32m1_t reduced_x = __riscv_vreinterpret_v_u32m1_f32m1(reduced_x_int);

  //float: x ^ (-2 / 3)
  //  0x1.c446e4p0 + x * (-0x1.f40db6p-1 + x * 0x1.a3b686p-3)
  vfloat32m1_t yn = __riscv_vfmv_v_f_f32m1(-0x1.f40db6p-1f, vl);
  yn = __riscv_vfmacc_vf_f32m1(yn, 0x1.a3b686p-3f, reduced_x, vl);
  yn = __riscv_vfmadd_vv_f32m1(yn, reduced_x, __riscv_vfmv_v_f_f32m1(0x1.c446e4p0f, vl), vl);

  //y_(n + 1) = y_n + y_n / 3 (1 - x ^ 2 y_n ^ 3)
  //                         x * x * y * y * y = (x * y) * (x * y) * y
  vfloat32m1_t r = __riscv_vfmul_vv_f32m1(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f32m1(r, r, vl);
  r = __riscv_vfmul_vv_f32m1(r, yn, vl);
  r = __riscv_vfnmsub_vv_f32m1(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f32m1(yn, 0x1.555556p-2f, r, vl);

  r = __riscv_vfmul_vv_f32m1(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f32m1(r, r, vl);
  r = __riscv_vfmul_vv_f32m1(r, yn, vl);
  r = __riscv_vfnmsub_vv_f32m1(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f32m1(yn, 0x1.555556p-2f, r, vl);

  vfloat32m1_t rh = __riscv_vfmul_vv_f32m1(reduced_x, yn, vl);
  vfloat32m1_t rl = __riscv_vfmsac_vv_f32m1(rh, reduced_x, yn, vl);

  vfloat32m1_t hi = __riscv_vfmul_vv_f32m1(rh, rh, vl);
  vfloat32m1_t lo = __riscv_vfmsac_vv_f32m1(hi, rh, rh, vl);
  vfloat32m1_t cross = __riscv_vfmul_vv_f32m1(rl, rh, vl);
  cross = __riscv_vfadd_vv_f32m1(cross, cross, vl);
  lo = __riscv_vfadd_vv_f32m1(lo, cross, vl);

  rh = __riscv_vfadd_vv_f32m1(hi, lo, vl);
  vfloat32m1_t rl_tmp = __riscv_vfsub_vv_f32m1(rh, hi, vl);
  rl = __riscv_vfsub_vv_f32m1(lo, rl_tmp, vl);

  vfloat32m1_t rh1 = __riscv_vfnmsac_vv_f32m1(one, rh, yn, vl);
  vfloat32m1_t r2 = __riscv_vfmul_vv_f32m1(rh1, rh1, vl);
  vfloat32m1_t rl1 = __riscv_vfmul_vf_f32m1(r2, 0x1.555556p-1f, vl);
  rl1 = __riscv_vfnmsac_vv_f32m1(rl1, rl, yn, vl); // rl1 = (2/3)*rh1^2 - rl*yn

  hi = __riscv_vfmul_vv_f32m1(rh1, yn, vl);
  lo = __riscv_vfmsac_vv_f32m1(hi, rh1, yn, vl);
  lo = __riscv_vfmacc_vv_f32m1(lo, rl1, yn, vl);
  rh = __riscv_vfadd_vv_f32m1(hi, lo, vl);
  rl = __riscv_vfsub_vv_f32m1(rh, hi, vl);
  rl = __riscv_vfsub_vv_f32m1(lo, rl, vl);


  vfloat32m1_t yh = __riscv_vfmacc_vf_f32m1(yn, 0x1.555556p-2f, rh, vl);
  vfloat32m1_t yl = __riscv_vfsub_vv_f32m1(yn, yh, vl);
  yl = __riscv_vfmacc_vf_f32m1(yl, 0x1.555556p-2f, rh, vl);
  yl = __riscv_vfmacc_vf_f32m1(yl, 0x1.555556p-2f, rl, vl);

  vbool32_t is_1 = __riscv_vmseq_vx_u32m1_b32(index, 1, vl);
  vbool32_t is_2 = __riscv_vmseq_vx_u32m1_b32(index, 2, vl);
  
  vfloat32m1_t ord_hi = __riscv_vfmv_v_f_f32m1(tab_high_flt[0], vl);
  ord_hi = __riscv_vfmerge_vfm_f32m1(ord_hi, tab_high_flt[1], is_1, vl);
  ord_hi = __riscv_vfmerge_vfm_f32m1(ord_hi, tab_high_flt[2], is_2, vl);

  vfloat32m1_t ord_lo = __riscv_vfmv_v_f_f32m1(tab_low_flt[0], vl);
  ord_lo = __riscv_vfmerge_vfm_f32m1(ord_lo, tab_low_flt[1], is_1, vl);
  ord_lo = __riscv_vfmerge_vfm_f32m1(ord_lo, tab_low_flt[2], is_2, vl);

  hi = __riscv_vfmul_vv_f32m1(yh, reduced_x, vl);
  lo = __riscv_vfmsac_vv_f32m1(hi, yh, reduced_x, vl);
  lo = __riscv_vfmacc_vv_f32m1(lo, yl, reduced_x, vl);
 
  rh = __riscv_vfmul_vv_f32m1(ord_hi, hi, vl);
  rl = __riscv_vfmsac_vv_f32m1(rh, ord_hi, hi, vl);
  rl = __riscv_vfmacc_vv_f32m1(rl, ord_lo, hi, vl);
  rl = __riscv_vfmacc_vv_f32m1(rl, ord_hi, lo, vl);
  rh = __riscv_vfadd_vv_f32m1(rh, rl, vl);
  
  vuint32m1_t mask_for_order = __riscv_vsll_vx_u32m1(order_in_cbrt, 23, vl);
  vfloat32m1_t order = __riscv_vreinterpret_v_u32m1_f32m1(mask_for_order);
  rh = __riscv_vfmul_vv_f32m1(rh, order, vl);

  vfloat32m1_t cbrt_value = __riscv_vfsgnj_vv_f32m1(rh, x, vl);

#ifndef __FAST_MATH__
  vfloat32m1_t denorm_mult_back = 
    __riscv_vfmerge_vfm_f32m1(one, 0x1.0p-8f, denormal_mask, vl);
  cbrt_value = __riscv_vfmul_vv_f32m1(cbrt_value, denorm_mult_back, vl);

  cbrt_value = __riscv_vmerge_vvm_f32m1(cbrt_value, x, special_mask, vl);

  vbool32_t mask_sNaN = __riscv_vmand_mm_b32(
            __riscv_vmsltu_vx_u32m1_b32(orig_x_abs, 0x7fc00000, vl), nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b32(mask_sNaN, vl);
  if (issNaN) {
    volatile float x1 = 0.0f / 0.0f;
    cbrt_value = __riscv_vfmerge_vfm_f32m1(cbrt_value, x1, mask_sNaN, vl);
  }
#else
  cbrt_value = __riscv_vmerge_vvm_f32m1(cbrt_value, x, zero_mask, vl);
#endif

  return cbrt_value; 
}

vfloat32m2_t __riscv_vcbrt_f32m2(vfloat32m2_t x, size_t vl)
{
  vuint32m2_t orig_x = __riscv_vreinterpret_v_f32m2_u32m2(x);
  vuint32m2_t orig_x_abs = __riscv_vand_vx_u32m2(orig_x, 0x7fffffff, vl);
  vbool16_t zero_mask = __riscv_vmseq_vx_u32m2_b16(orig_x_abs, 0, vl);

  vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);

#ifndef __FAST_MATH__
  vbool16_t nan_mask = __riscv_vmsgtu_vx_u32m2_b16(orig_x_abs, 0x7f800000, vl);
  vbool16_t special_mask = __riscv_vmor_mm_b16(zero_mask,
                         __riscv_vmsgeu_vx_u32m2_b16(orig_x_abs, 0x7f800000, vl), vl);

  vfloat32m2_t x_abs = __riscv_vreinterpret_v_u32m2_f32m2(orig_x_abs);
  vfloat32m2_t x_spec = __riscv_vfmerge_vfm_f32m2(x_abs, 1.0f, special_mask, vl);

  vbool16_t denormal_mask = __riscv_vmsltu_vx_u32m2_b16(orig_x_abs, 0x00800000, vl);
  vfloat32m2_t denorm_mult = 
    __riscv_vfmerge_vfm_f32m2(one, 0x1.0p+24f, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f32m2(x_spec, denorm_mult, vl);

  vuint32m2_t x_int = __riscv_vreinterpret_v_f32m2_u32m2(x_spec);
#else
  vuint32m2_t x_int = orig_x_abs;
#endif

  vuint32m2_t mantissa_in_x = __riscv_vand_vx_u32m2(x_int, 0x007fffff, vl);
  vuint32m2_t order_in_x = __riscv_vsrl_vx_u32m2(x_int, 23, vl);

  vuint32m2_t order_plus = __riscv_vadd_vx_u32m2(order_in_x, 173, vl);
  vuint32m2_t div3floorf = __riscv_vsrl_vx_u32m2(
      __riscv_vmul_vx_u32m2(order_plus, 21846, vl), 16, vl);
  vuint32m2_t index = __riscv_vsub_vv_u32m2(order_plus, div3floorf, vl);
  index = __riscv_vsub_vv_u32m2(index, __riscv_vsll_vx_u32m2(div3floorf, 1, vl), vl);
  vuint32m2_t order_in_cbrt = __riscv_vadd_vx_u32m2(div3floorf, 27, vl);

  vuint32m2_t reduced_x_int = __riscv_vor_vx_u32m2(mantissa_in_x, 0x3f800000, vl);
  vfloat32m2_t reduced_x = __riscv_vreinterpret_v_u32m2_f32m2(reduced_x_int);

  //float: x ^ (-2 / 3)
  //  0x1.c446e4p0 + x * (-0x1.f40db6p-1 + x * 0x1.a3b686p-3)
  vfloat32m2_t yn = __riscv_vfmv_v_f_f32m2(-0x1.f40db6p-1f, vl);
  yn = __riscv_vfmacc_vf_f32m2(yn, 0x1.a3b686p-3f, reduced_x, vl);
  yn = __riscv_vfmadd_vv_f32m2(yn, reduced_x, __riscv_vfmv_v_f_f32m2(0x1.c446e4p0f, vl), vl);

  //y_(n + 1) = y_n + y_n / 3 (1 - x ^ 2 y_n ^ 3)
  //                         x * x * y * y * y = (x * y) * (x * y) * y
  vfloat32m2_t r = __riscv_vfmul_vv_f32m2(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f32m2(r, r, vl);
  r = __riscv_vfmul_vv_f32m2(r, yn, vl);
  r = __riscv_vfnmsub_vv_f32m2(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f32m2(yn, 0x1.555556p-2f, r, vl);

  r = __riscv_vfmul_vv_f32m2(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f32m2(r, r, vl);
  r = __riscv_vfmul_vv_f32m2(r, yn, vl);
  r = __riscv_vfnmsub_vv_f32m2(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f32m2(yn, 0x1.555556p-2f, r, vl);

  vfloat32m2_t rh = __riscv_vfmul_vv_f32m2(reduced_x, yn, vl);
  vfloat32m2_t rl = __riscv_vfmsac_vv_f32m2(rh, reduced_x, yn, vl);

  vfloat32m2_t hi = __riscv_vfmul_vv_f32m2(rh, rh, vl);
  vfloat32m2_t lo = __riscv_vfmsac_vv_f32m2(hi, rh, rh, vl);
  vfloat32m2_t cross = __riscv_vfmul_vv_f32m2(rl, rh, vl);
  cross = __riscv_vfadd_vv_f32m2(cross, cross, vl);
  lo = __riscv_vfadd_vv_f32m2(lo, cross, vl);

  rh = __riscv_vfadd_vv_f32m2(hi, lo, vl);
  vfloat32m2_t rl_tmp = __riscv_vfsub_vv_f32m2(rh, hi, vl);
  rl = __riscv_vfsub_vv_f32m2(lo, rl_tmp, vl);

  vfloat32m2_t rh1 = __riscv_vfnmsac_vv_f32m2(one, rh, yn, vl);
  vfloat32m2_t r2 = __riscv_vfmul_vv_f32m2(rh1, rh1, vl);
  vfloat32m2_t rl1 = __riscv_vfmul_vf_f32m2(r2, 0x1.555556p-1f, vl);
  rl1 = __riscv_vfnmsac_vv_f32m2(rl1, rl, yn, vl); // rl1 = (2/3)*rh1^2 - rl*yn

  hi = __riscv_vfmul_vv_f32m2(rh1, yn, vl);
  lo = __riscv_vfmsac_vv_f32m2(hi, rh1, yn, vl);
  lo = __riscv_vfmacc_vv_f32m2(lo, rl1, yn, vl);
  rh = __riscv_vfadd_vv_f32m2(hi, lo, vl);
  rl = __riscv_vfsub_vv_f32m2(rh, hi, vl);
  rl = __riscv_vfsub_vv_f32m2(lo, rl, vl);


  vfloat32m2_t yh = __riscv_vfmacc_vf_f32m2(yn, 0x1.555556p-2f, rh, vl);
  vfloat32m2_t yl = __riscv_vfsub_vv_f32m2(yn, yh, vl);
  yl = __riscv_vfmacc_vf_f32m2(yl, 0x1.555556p-2f, rh, vl);
  yl = __riscv_vfmacc_vf_f32m2(yl, 0x1.555556p-2f, rl, vl);

  vbool16_t is_1 = __riscv_vmseq_vx_u32m2_b16(index, 1, vl);
  vbool16_t is_2 = __riscv_vmseq_vx_u32m2_b16(index, 2, vl);
  
  vfloat32m2_t ord_hi = __riscv_vfmv_v_f_f32m2(tab_high_flt[0], vl);
  ord_hi = __riscv_vfmerge_vfm_f32m2(ord_hi, tab_high_flt[1], is_1, vl);
  ord_hi = __riscv_vfmerge_vfm_f32m2(ord_hi, tab_high_flt[2], is_2, vl);

  vfloat32m2_t ord_lo = __riscv_vfmv_v_f_f32m2(tab_low_flt[0], vl);
  ord_lo = __riscv_vfmerge_vfm_f32m2(ord_lo, tab_low_flt[1], is_1, vl);
  ord_lo = __riscv_vfmerge_vfm_f32m2(ord_lo, tab_low_flt[2], is_2, vl);

  hi = __riscv_vfmul_vv_f32m2(yh, reduced_x, vl);
  lo = __riscv_vfmsac_vv_f32m2(hi, yh, reduced_x, vl);
  lo = __riscv_vfmacc_vv_f32m2(lo, yl, reduced_x, vl);
 
  rh = __riscv_vfmul_vv_f32m2(ord_hi, hi, vl);
  rl = __riscv_vfmsac_vv_f32m2(rh, ord_hi, hi, vl);
  rl = __riscv_vfmacc_vv_f32m2(rl, ord_lo, hi, vl);
  rl = __riscv_vfmacc_vv_f32m2(rl, ord_hi, lo, vl);
  rh = __riscv_vfadd_vv_f32m2(rh, rl, vl);
  
  vuint32m2_t mask_for_order = __riscv_vsll_vx_u32m2(order_in_cbrt, 23, vl);
  vfloat32m2_t order = __riscv_vreinterpret_v_u32m2_f32m2(mask_for_order);
  rh = __riscv_vfmul_vv_f32m2(rh, order, vl);

  vfloat32m2_t cbrt_value = __riscv_vfsgnj_vv_f32m2(rh, x, vl);

#ifndef __FAST_MATH__
  vfloat32m2_t denorm_mult_back = 
    __riscv_vfmerge_vfm_f32m2(one, 0x1.0p-8f, denormal_mask, vl);
  cbrt_value = __riscv_vfmul_vv_f32m2(cbrt_value, denorm_mult_back, vl);

  cbrt_value = __riscv_vmerge_vvm_f32m2(cbrt_value, x, special_mask, vl);

  vbool16_t mask_sNaN = __riscv_vmand_mm_b16(
            __riscv_vmsltu_vx_u32m2_b16(orig_x_abs, 0x7fc00000, vl), nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
  if (issNaN) {
    volatile float x1 = 0.0f / 0.0f;
    cbrt_value = __riscv_vfmerge_vfm_f32m2(cbrt_value, x1, mask_sNaN, vl);
  }
#else
  cbrt_value = __riscv_vmerge_vvm_f32m2(cbrt_value, x, zero_mask, vl);
#endif

  return cbrt_value; 
}

vfloat32m4_t __riscv_vcbrt_f32m4(vfloat32m4_t x, size_t vl)
{
  vuint32m4_t orig_x = __riscv_vreinterpret_v_f32m4_u32m4(x);
  vuint32m4_t orig_x_abs = __riscv_vand_vx_u32m4(orig_x, 0x7fffffff, vl);
  vbool8_t zero_mask = __riscv_vmseq_vx_u32m4_b8(orig_x_abs, 0, vl);

  vfloat32m4_t one = __riscv_vfmv_v_f_f32m4(1.0f, vl);

#ifndef __FAST_MATH__
  vbool8_t nan_mask = __riscv_vmsgtu_vx_u32m4_b8(orig_x_abs, 0x7f800000, vl);
  vbool8_t special_mask = __riscv_vmor_mm_b8(zero_mask,
                         __riscv_vmsgeu_vx_u32m4_b8(orig_x_abs, 0x7f800000, vl), vl);

  vfloat32m4_t x_abs = __riscv_vreinterpret_v_u32m4_f32m4(orig_x_abs);
  vfloat32m4_t x_spec = __riscv_vfmerge_vfm_f32m4(x_abs, 1.0f, special_mask, vl);

  vbool8_t denormal_mask = __riscv_vmsltu_vx_u32m4_b8(orig_x_abs, 0x00800000, vl);
  vfloat32m4_t denorm_mult = 
    __riscv_vfmerge_vfm_f32m4(one, 0x1.0p+24f, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f32m4(x_spec, denorm_mult, vl);

  vuint32m4_t x_int = __riscv_vreinterpret_v_f32m4_u32m4(x_spec);
#else
  vuint32m4_t x_int = orig_x_abs;
#endif

  vuint32m4_t mantissa_in_x = __riscv_vand_vx_u32m4(x_int, 0x007fffff, vl);
  vuint32m4_t order_in_x = __riscv_vsrl_vx_u32m4(x_int, 23, vl);

  vuint32m4_t order_plus = __riscv_vadd_vx_u32m4(order_in_x, 173, vl);
  vuint32m4_t div3floorf = __riscv_vsrl_vx_u32m4(
      __riscv_vmul_vx_u32m4(order_plus, 21846, vl), 16, vl);
  vuint32m4_t index = __riscv_vsub_vv_u32m4(order_plus, div3floorf, vl);
  index = __riscv_vsub_vv_u32m4(index, __riscv_vsll_vx_u32m4(div3floorf, 1, vl), vl);
  vuint32m4_t order_in_cbrt = __riscv_vadd_vx_u32m4(div3floorf, 27, vl);

  vuint32m4_t reduced_x_int = __riscv_vor_vx_u32m4(mantissa_in_x, 0x3f800000, vl);
  vfloat32m4_t reduced_x = __riscv_vreinterpret_v_u32m4_f32m4(reduced_x_int);

  //float: x ^ (-2 / 3)
  //  0x1.c446e4p0 + x * (-0x1.f40db6p-1 + x * 0x1.a3b686p-3)
  vfloat32m4_t yn = __riscv_vfmv_v_f_f32m4(-0x1.f40db6p-1f, vl);
  yn = __riscv_vfmacc_vf_f32m4(yn, 0x1.a3b686p-3f, reduced_x, vl);
  yn = __riscv_vfmadd_vv_f32m4(yn, reduced_x, __riscv_vfmv_v_f_f32m4(0x1.c446e4p0f, vl), vl);

  //y_(n + 1) = y_n + y_n / 3 (1 - x ^ 2 y_n ^ 3)
  //                         x * x * y * y * y = (x * y) * (x * y) * y
  vfloat32m4_t r = __riscv_vfmul_vv_f32m4(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f32m4(r, r, vl);
  r = __riscv_vfmul_vv_f32m4(r, yn, vl);
  r = __riscv_vfnmsub_vv_f32m4(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f32m4(yn, 0x1.555556p-2f, r, vl);

  r = __riscv_vfmul_vv_f32m4(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f32m4(r, r, vl);
  r = __riscv_vfmul_vv_f32m4(r, yn, vl);
  r = __riscv_vfnmsub_vv_f32m4(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f32m4(yn, 0x1.555556p-2f, r, vl);

  vfloat32m4_t rh = __riscv_vfmul_vv_f32m4(reduced_x, yn, vl);
  vfloat32m4_t rl = __riscv_vfmsac_vv_f32m4(rh, reduced_x, yn, vl);

  vfloat32m4_t hi = __riscv_vfmul_vv_f32m4(rh, rh, vl);
  vfloat32m4_t lo = __riscv_vfmsac_vv_f32m4(hi, rh, rh, vl);
  vfloat32m4_t cross = __riscv_vfmul_vv_f32m4(rl, rh, vl);
  cross = __riscv_vfadd_vv_f32m4(cross, cross, vl);
  lo = __riscv_vfadd_vv_f32m4(lo, cross, vl);

  rh = __riscv_vfadd_vv_f32m4(hi, lo, vl);
  vfloat32m4_t rl_tmp = __riscv_vfsub_vv_f32m4(rh, hi, vl);
  rl = __riscv_vfsub_vv_f32m4(lo, rl_tmp, vl);

  vfloat32m4_t rh1 = __riscv_vfnmsac_vv_f32m4(one, rh, yn, vl);
  vfloat32m4_t r2 = __riscv_vfmul_vv_f32m4(rh1, rh1, vl);
  vfloat32m4_t rl1 = __riscv_vfmul_vf_f32m4(r2, 0x1.555556p-1f, vl);
  rl1 = __riscv_vfnmsac_vv_f32m4(rl1, rl, yn, vl); // rl1 = (2/3)*rh1^2 - rl*yn

  hi = __riscv_vfmul_vv_f32m4(rh1, yn, vl);
  lo = __riscv_vfmsac_vv_f32m4(hi, rh1, yn, vl);
  lo = __riscv_vfmacc_vv_f32m4(lo, rl1, yn, vl);
  rh = __riscv_vfadd_vv_f32m4(hi, lo, vl);
  rl = __riscv_vfsub_vv_f32m4(rh, hi, vl);
  rl = __riscv_vfsub_vv_f32m4(lo, rl, vl);


  vfloat32m4_t yh = __riscv_vfmacc_vf_f32m4(yn, 0x1.555556p-2f, rh, vl);
  vfloat32m4_t yl = __riscv_vfsub_vv_f32m4(yn, yh, vl);
  yl = __riscv_vfmacc_vf_f32m4(yl, 0x1.555556p-2f, rh, vl);
  yl = __riscv_vfmacc_vf_f32m4(yl, 0x1.555556p-2f, rl, vl);

  vbool8_t is_1 = __riscv_vmseq_vx_u32m4_b8(index, 1, vl);
  vbool8_t is_2 = __riscv_vmseq_vx_u32m4_b8(index, 2, vl);
  
  vfloat32m4_t ord_hi = __riscv_vfmv_v_f_f32m4(tab_high_flt[0], vl);
  ord_hi = __riscv_vfmerge_vfm_f32m4(ord_hi, tab_high_flt[1], is_1, vl);
  ord_hi = __riscv_vfmerge_vfm_f32m4(ord_hi, tab_high_flt[2], is_2, vl);

  vfloat32m4_t ord_lo = __riscv_vfmv_v_f_f32m4(tab_low_flt[0], vl);
  ord_lo = __riscv_vfmerge_vfm_f32m4(ord_lo, tab_low_flt[1], is_1, vl);
  ord_lo = __riscv_vfmerge_vfm_f32m4(ord_lo, tab_low_flt[2], is_2, vl);

  hi = __riscv_vfmul_vv_f32m4(yh, reduced_x, vl);
  lo = __riscv_vfmsac_vv_f32m4(hi, yh, reduced_x, vl);
  lo = __riscv_vfmacc_vv_f32m4(lo, yl, reduced_x, vl);
 
  rh = __riscv_vfmul_vv_f32m4(ord_hi, hi, vl);
  rl = __riscv_vfmsac_vv_f32m4(rh, ord_hi, hi, vl);
  rl = __riscv_vfmacc_vv_f32m4(rl, ord_lo, hi, vl);
  rl = __riscv_vfmacc_vv_f32m4(rl, ord_hi, lo, vl);
  rh = __riscv_vfadd_vv_f32m4(rh, rl, vl);
  
  vuint32m4_t mask_for_order = __riscv_vsll_vx_u32m4(order_in_cbrt, 23, vl);
  vfloat32m4_t order = __riscv_vreinterpret_v_u32m4_f32m4(mask_for_order);
  rh = __riscv_vfmul_vv_f32m4(rh, order, vl);

  vfloat32m4_t cbrt_value = __riscv_vfsgnj_vv_f32m4(rh, x, vl);

#ifndef __FAST_MATH__
  vfloat32m4_t denorm_mult_back = 
    __riscv_vfmerge_vfm_f32m4(one, 0x1.0p-8f, denormal_mask, vl);
  cbrt_value = __riscv_vfmul_vv_f32m4(cbrt_value, denorm_mult_back, vl);

  cbrt_value = __riscv_vmerge_vvm_f32m4(cbrt_value, x, special_mask, vl);

  vbool8_t mask_sNaN = __riscv_vmand_mm_b8(
            __riscv_vmsltu_vx_u32m4_b8(orig_x_abs, 0x7fc00000, vl), nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b8(mask_sNaN, vl);
  if (issNaN) {
    volatile float x1 = 0.0f / 0.0f;
    cbrt_value = __riscv_vfmerge_vfm_f32m4(cbrt_value, x1, mask_sNaN, vl);
  }
#else
  cbrt_value = __riscv_vmerge_vvm_f32m4(cbrt_value, x, zero_mask, vl);
#endif

  return cbrt_value; 
}

vfloat32m8_t __riscv_vcbrt_f32m8(vfloat32m8_t x, size_t vl)
{
  vuint32m8_t orig_x = __riscv_vreinterpret_v_f32m8_u32m8(x);
  vuint32m8_t orig_x_abs = __riscv_vand_vx_u32m8(orig_x, 0x7fffffff, vl);
  vbool4_t zero_mask = __riscv_vmseq_vx_u32m8_b4(orig_x_abs, 0, vl);

  vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1.0f, vl);

#ifndef __FAST_MATH__
  vbool4_t nan_mask = __riscv_vmsgtu_vx_u32m8_b4(orig_x_abs, 0x7f800000, vl);
  vbool4_t special_mask = __riscv_vmor_mm_b4(zero_mask,
                         __riscv_vmsgeu_vx_u32m8_b4(orig_x_abs, 0x7f800000, vl), vl);

  vfloat32m8_t x_abs = __riscv_vreinterpret_v_u32m8_f32m8(orig_x_abs);
  vfloat32m8_t x_spec = __riscv_vfmerge_vfm_f32m8(x_abs, 1.0f, special_mask, vl);

  vbool4_t denormal_mask = __riscv_vmsltu_vx_u32m8_b4(orig_x_abs, 0x00800000, vl);
  vfloat32m8_t denorm_mult = 
    __riscv_vfmerge_vfm_f32m8(one, 0x1.0p+24f, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f32m8(x_spec, denorm_mult, vl);

  vuint32m8_t x_int = __riscv_vreinterpret_v_f32m8_u32m8(x_spec);
#else
  vuint32m8_t x_int = orig_x_abs;
#endif

  vuint32m8_t mantissa_in_x = __riscv_vand_vx_u32m8(x_int, 0x007fffff, vl);
  vuint32m8_t order_in_x = __riscv_vsrl_vx_u32m8(x_int, 23, vl);

  vuint32m8_t order_plus = __riscv_vadd_vx_u32m8(order_in_x, 173, vl);
  vuint32m8_t div3floorf = __riscv_vsrl_vx_u32m8(
      __riscv_vmul_vx_u32m8(order_plus, 21846, vl), 16, vl);
  vuint32m8_t index = __riscv_vsub_vv_u32m8(order_plus, div3floorf, vl);
  index = __riscv_vsub_vv_u32m8(index, __riscv_vsll_vx_u32m8(div3floorf, 1, vl), vl);
  vuint32m8_t order_in_cbrt = __riscv_vadd_vx_u32m8(div3floorf, 27, vl);

  vuint32m8_t reduced_x_int = __riscv_vor_vx_u32m8(mantissa_in_x, 0x3f800000, vl);
  vfloat32m8_t reduced_x = __riscv_vreinterpret_v_u32m8_f32m8(reduced_x_int);

  //float: x ^ (-2 / 3)
  //  0x1.c446e4p0 + x * (-0x1.f40db6p-1 + x * 0x1.a3b686p-3)
  vfloat32m8_t yn = __riscv_vfmv_v_f_f32m8(-0x1.f40db6p-1f, vl);
  yn = __riscv_vfmacc_vf_f32m8(yn, 0x1.a3b686p-3f, reduced_x, vl);
  yn = __riscv_vfmadd_vv_f32m8(yn, reduced_x, __riscv_vfmv_v_f_f32m8(0x1.c446e4p0f, vl), vl);

  //y_(n + 1) = y_n + y_n / 3 (1 - x ^ 2 y_n ^ 3)
  //                         x * x * y * y * y = (x * y) * (x * y) * y
  vfloat32m8_t r = __riscv_vfmul_vv_f32m8(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f32m8(r, r, vl);
  r = __riscv_vfmul_vv_f32m8(r, yn, vl);
  r = __riscv_vfnmsub_vv_f32m8(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f32m8(yn, 0x1.555556p-2f, r, vl);

  r = __riscv_vfmul_vv_f32m8(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f32m8(r, r, vl);
  r = __riscv_vfmul_vv_f32m8(r, yn, vl);
  r = __riscv_vfnmsub_vv_f32m8(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f32m8(yn, 0x1.555556p-2f, r, vl);

  vfloat32m8_t rh = __riscv_vfmul_vv_f32m8(reduced_x, yn, vl);
  vfloat32m8_t rl = __riscv_vfmsac_vv_f32m8(rh, reduced_x, yn, vl);

  vfloat32m8_t hi = __riscv_vfmul_vv_f32m8(rh, rh, vl);
  vfloat32m8_t lo = __riscv_vfmsac_vv_f32m8(hi, rh, rh, vl);
  vfloat32m8_t cross = __riscv_vfmul_vv_f32m8(rl, rh, vl);
  cross = __riscv_vfadd_vv_f32m8(cross, cross, vl);
  lo = __riscv_vfadd_vv_f32m8(lo, cross, vl);

  rh = __riscv_vfadd_vv_f32m8(hi, lo, vl);
  vfloat32m8_t rl_tmp = __riscv_vfsub_vv_f32m8(rh, hi, vl);
  rl = __riscv_vfsub_vv_f32m8(lo, rl_tmp, vl);

  vfloat32m8_t rh1 = __riscv_vfnmsac_vv_f32m8(one, rh, yn, vl);
  vfloat32m8_t r2 = __riscv_vfmul_vv_f32m8(rh1, rh1, vl);
  vfloat32m8_t rl1 = __riscv_vfmul_vf_f32m8(r2, 0x1.555556p-1f, vl);
  rl1 = __riscv_vfnmsac_vv_f32m8(rl1, rl, yn, vl); // rl1 = (2/3)*rh1^2 - rl*yn

  hi = __riscv_vfmul_vv_f32m8(rh1, yn, vl);
  lo = __riscv_vfmsac_vv_f32m8(hi, rh1, yn, vl);
  lo = __riscv_vfmacc_vv_f32m8(lo, rl1, yn, vl);
  rh = __riscv_vfadd_vv_f32m8(hi, lo, vl);
  rl = __riscv_vfsub_vv_f32m8(rh, hi, vl);
  rl = __riscv_vfsub_vv_f32m8(lo, rl, vl);


  vfloat32m8_t yh = __riscv_vfmacc_vf_f32m8(yn, 0x1.555556p-2f, rh, vl);
  vfloat32m8_t yl = __riscv_vfsub_vv_f32m8(yn, yh, vl);
  yl = __riscv_vfmacc_vf_f32m8(yl, 0x1.555556p-2f, rh, vl);
  yl = __riscv_vfmacc_vf_f32m8(yl, 0x1.555556p-2f, rl, vl);

  vbool4_t is_1 = __riscv_vmseq_vx_u32m8_b4(index, 1, vl);
  vbool4_t is_2 = __riscv_vmseq_vx_u32m8_b4(index, 2, vl);
  
  vfloat32m8_t ord_hi = __riscv_vfmv_v_f_f32m8(tab_high_flt[0], vl);
  ord_hi = __riscv_vfmerge_vfm_f32m8(ord_hi, tab_high_flt[1], is_1, vl);
  ord_hi = __riscv_vfmerge_vfm_f32m8(ord_hi, tab_high_flt[2], is_2, vl);

  vfloat32m8_t ord_lo = __riscv_vfmv_v_f_f32m8(tab_low_flt[0], vl);
  ord_lo = __riscv_vfmerge_vfm_f32m8(ord_lo, tab_low_flt[1], is_1, vl);
  ord_lo = __riscv_vfmerge_vfm_f32m8(ord_lo, tab_low_flt[2], is_2, vl);

  hi = __riscv_vfmul_vv_f32m8(yh, reduced_x, vl);
  lo = __riscv_vfmsac_vv_f32m8(hi, yh, reduced_x, vl);
  lo = __riscv_vfmacc_vv_f32m8(lo, yl, reduced_x, vl);
 
  rh = __riscv_vfmul_vv_f32m8(ord_hi, hi, vl);
  rl = __riscv_vfmsac_vv_f32m8(rh, ord_hi, hi, vl);
  rl = __riscv_vfmacc_vv_f32m8(rl, ord_lo, hi, vl);
  rl = __riscv_vfmacc_vv_f32m8(rl, ord_hi, lo, vl);
  rh = __riscv_vfadd_vv_f32m8(rh, rl, vl);
  
  vuint32m8_t mask_for_order = __riscv_vsll_vx_u32m8(order_in_cbrt, 23, vl);
  vfloat32m8_t order = __riscv_vreinterpret_v_u32m8_f32m8(mask_for_order);
  rh = __riscv_vfmul_vv_f32m8(rh, order, vl);

  vfloat32m8_t cbrt_value = __riscv_vfsgnj_vv_f32m8(rh, x, vl);

#ifndef __FAST_MATH__
  vfloat32m8_t denorm_mult_back = 
    __riscv_vfmerge_vfm_f32m8(one, 0x1.0p-8f, denormal_mask, vl);
  cbrt_value = __riscv_vfmul_vv_f32m8(cbrt_value, denorm_mult_back, vl);

  cbrt_value = __riscv_vmerge_vvm_f32m8(cbrt_value, x, special_mask, vl);

  vbool4_t mask_sNaN = __riscv_vmand_mm_b4(
            __riscv_vmsltu_vx_u32m8_b4(orig_x_abs, 0x7fc00000, vl), nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b4(mask_sNaN, vl);
  if (issNaN) {
    volatile float x1 = 0.0f  /0.0f;
    cbrt_value = __riscv_vfmerge_vfm_f32m8(cbrt_value, x1, mask_sNaN, vl);
  }
#else
  cbrt_value = __riscv_vmerge_vvm_f32m8(cbrt_value, x, zero_mask, vl);
#endif

  return cbrt_value; 
}



vfloat64m1_t __riscv_vcbrt_f64m1(vfloat64m1_t x, size_t vl)
{
  vuint64m1_t orig_x = __riscv_vreinterpret_v_f64m1_u64m1(x);
  vuint64m1_t orig_x_abs = __riscv_vand_vx_u64m1(orig_x, 0x7fffffffffffffffULL, vl);
  vbool64_t zero_mask = __riscv_vmseq_vx_u64m1_b64(orig_x_abs, 0, vl);

  vfloat64m1_t one = __riscv_vfmv_v_f_f64m1(1.0, vl);

#ifndef __FAST_MATH__
  vbool64_t nan_mask = __riscv_vmsgtu_vx_u64m1_b64(orig_x_abs, 0x7ff0000000000000ULL, vl);
  vbool64_t special_mask = __riscv_vmor_mm_b64(zero_mask,
                         __riscv_vmsgeu_vx_u64m1_b64(orig_x_abs, 0x7ff0000000000000ULL, vl), vl);

  vfloat64m1_t x_abs = __riscv_vreinterpret_v_u64m1_f64m1(orig_x_abs);
  vfloat64m1_t x_spec = __riscv_vfmerge_vfm_f64m1(x_abs, 1.0, special_mask, vl);

  vbool64_t denormal_mask = __riscv_vmsltu_vx_u64m1_b64(orig_x_abs, 0x0010000000000000ULL, vl);
  vfloat64m1_t denorm_mult = 
    __riscv_vfmerge_vfm_f64m1(one, 0x1.0p+60, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f64m1(x_spec, denorm_mult, vl);

  vuint64m1_t x_int = __riscv_vreinterpret_v_f64m1_u64m1(x_spec);
#else
  vuint64m1_t x_int = orig_x_abs;
#endif

  vuint64m1_t mantissa_in_x = __riscv_vand_vx_u64m1(x_int, 0x000fffffffffffffULL, vl);
  vuint64m1_t order_in_x = __riscv_vsrl_vx_u64m1(x_int, 52, vl);

  vuint64m1_t order_plus = __riscv_vadd_vx_u64m1(order_in_x, 177, vl);
  vuint64m1_t div3floor = __riscv_vsrl_vx_u64m1(
      __riscv_vmul_vx_u64m1(order_plus, 349526, vl), 20, vl);
  vuint64m1_t index = __riscv_vsub_vv_u64m1(order_plus, div3floor, vl);
  index = __riscv_vsub_vv_u64m1(index, __riscv_vsll_vx_u64m1(div3floor, 1, vl), vl);
  vuint64m1_t order_in_cbrt = __riscv_vadd_vx_u64m1(div3floor, 623, vl);

  vuint64m1_t reduced_x_int = 
    __riscv_vor_vx_u64m1(mantissa_in_x, 0x3ff0000000000000ULL, vl);
  vfloat64m1_t reduced_x = __riscv_vreinterpret_v_u64m1_f64m1(reduced_x_int);

  vfloat64m1_t yn = __riscv_vfmv_v_f_f64m1(-0x1.f40dc09665846p-1, vl);
  yn = __riscv_vfmacc_vf_f64m1(yn, 0x1.a3b694a2e22eap-3, reduced_x, vl);
  yn = __riscv_vfmadd_vv_f64m1(yn, reduced_x, __riscv_vfmv_v_f_f64m1(0x1.c446e7c491267p0, vl), vl);

  vfloat64m1_t r = __riscv_vfmul_vv_f64m1(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f64m1(r, r, vl);
  r = __riscv_vfmul_vv_f64m1(r, yn, vl);
  r = __riscv_vfnmsub_vv_f64m1(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f64m1(yn, 0x1.5555555555555p-2, r, vl);

  r = __riscv_vfmul_vv_f64m1(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f64m1(r, r, vl);
  r = __riscv_vfmul_vv_f64m1(r, yn, vl);
  r = __riscv_vfnmsub_vv_f64m1(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f64m1(yn, 0x1.5555555555555p-2, r, vl);

  r = __riscv_vfmul_vv_f64m1(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f64m1(r, r, vl);
  r = __riscv_vfmul_vv_f64m1(r, yn, vl);
  r = __riscv_vfnmsub_vv_f64m1(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f64m1(yn, 0x1.5555555555555p-2, r, vl);

  vfloat64m1_t rh = __riscv_vfmul_vv_f64m1(reduced_x, yn, vl);
  vfloat64m1_t rl = __riscv_vfmsac_vv_f64m1(rh, reduced_x, yn, vl);

  vfloat64m1_t hi = __riscv_vfmul_vv_f64m1(rh, rh, vl);
  vfloat64m1_t lo = __riscv_vfmsac_vv_f64m1(hi, rh, rh, vl);
  vfloat64m1_t cross = __riscv_vfmul_vv_f64m1(rl, rh, vl);
  cross = __riscv_vfadd_vv_f64m1(cross, cross, vl);
  lo = __riscv_vfadd_vv_f64m1(lo, cross, vl);

  rh = __riscv_vfadd_vv_f64m1(hi, lo, vl);
  vfloat64m1_t rl_tmp = __riscv_vfsub_vv_f64m1(rh, hi, vl);
  rl = __riscv_vfsub_vv_f64m1(lo, rl_tmp, vl);

  vfloat64m1_t rh1 = __riscv_vfnmsac_vv_f64m1(one, rh, yn, vl);
  vfloat64m1_t r2 = __riscv_vfmul_vv_f64m1(rh1, rh1, vl);
  vfloat64m1_t rl1 = __riscv_vfmul_vf_f64m1(r2, 0x1.5555555555555p-1, vl);
  rl1 = __riscv_vfnmsac_vv_f64m1(rl1, rl, yn, vl); 

  hi = __riscv_vfmul_vv_f64m1(rh1, yn, vl);
  lo = __riscv_vfmsac_vv_f64m1(hi, rh1, yn, vl);
  lo = __riscv_vfmacc_vv_f64m1(lo, rl1, yn, vl);
  
  rh = __riscv_vfadd_vv_f64m1(hi, lo, vl);
  rl_tmp = __riscv_vfsub_vv_f64m1(rh, hi, vl);
  rl = __riscv_vfsub_vv_f64m1(lo, rl_tmp, vl);

  vfloat64m1_t yh = __riscv_vfmacc_vf_f64m1(yn, 0x1.5555555555555p-2, rh, vl);
  vfloat64m1_t yl = __riscv_vfsub_vv_f64m1(yn, yh, vl);
  yl = __riscv_vfmacc_vf_f64m1(yl, 0x1.5555555555555p-2, rh, vl);
  yl = __riscv_vfmacc_vf_f64m1(yl, 0x1.5555555555555p-2, rl, vl);

  vbool64_t is_1 = __riscv_vmseq_vx_u64m1_b64(index, 1, vl);
  vbool64_t is_2 = __riscv_vmseq_vx_u64m1_b64(index, 2, vl);
  
  vfloat64m1_t ord_hi = __riscv_vfmv_v_f_f64m1(tab_high_dbl[0], vl);
  ord_hi = __riscv_vfmerge_vfm_f64m1(ord_hi, tab_high_dbl[1], is_1, vl);
  ord_hi = __riscv_vfmerge_vfm_f64m1(ord_hi, tab_high_dbl[2], is_2, vl);

  vfloat64m1_t ord_lo = __riscv_vfmv_v_f_f64m1(tab_low_dbl[0], vl);
  ord_lo = __riscv_vfmerge_vfm_f64m1(ord_lo, tab_low_dbl[1], is_1, vl);
  ord_lo = __riscv_vfmerge_vfm_f64m1(ord_lo, tab_low_dbl[2], is_2, vl);

  hi = __riscv_vfmul_vv_f64m1(yh, reduced_x, vl);
  lo = __riscv_vfmsac_vv_f64m1(hi, yh, reduced_x, vl);
  lo = __riscv_vfmacc_vv_f64m1(lo, yl, reduced_x, vl);
 
  rh = __riscv_vfmul_vv_f64m1(ord_hi, hi, vl);
  rl = __riscv_vfmsac_vv_f64m1(rh, ord_hi, hi, vl);
  rl = __riscv_vfmacc_vv_f64m1(rl, ord_lo, hi, vl);
  rl = __riscv_vfmacc_vv_f64m1(rl, ord_hi, lo, vl);
  rh = __riscv_vfadd_vv_f64m1(rh, rl, vl);

  vuint64m1_t mask_for_order = __riscv_vsll_vx_u64m1(order_in_cbrt, 52, vl);
  vfloat64m1_t order = __riscv_vreinterpret_v_u64m1_f64m1(mask_for_order);
  rh = __riscv_vfmul_vv_f64m1(rh, order, vl);

  vfloat64m1_t cbrt_value = __riscv_vfsgnj_vv_f64m1(rh, x, vl);

#ifndef __FAST_MATH__
  vfloat64m1_t denorm_mult_back = 
    __riscv_vfmerge_vfm_f64m1(one, 0x1.0p-20, denormal_mask, vl);
  cbrt_value = __riscv_vfmul_vv_f64m1(cbrt_value, denorm_mult_back, vl);

  cbrt_value = __riscv_vmerge_vvm_f64m1(cbrt_value, x, special_mask, vl);

  vbool64_t mask_sNaN = __riscv_vmand_mm_b64(
            __riscv_vmsltu_vx_u64m1_b64(orig_x_abs, 0x7ff8000000000000ULL, vl), nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b64(mask_sNaN, vl);
  if (issNaN) {
    volatile double x1 = 0.0 / 0.0;
    cbrt_value = __riscv_vfmerge_vfm_f64m1(cbrt_value, x1, mask_sNaN, vl);
  }
#else
  cbrt_value = __riscv_vmerge_vvm_f64m1(cbrt_value, x, zero_mask, vl);
#endif

  return cbrt_value; 
}


vfloat64m2_t __riscv_vcbrt_f64m2(vfloat64m2_t x, size_t vl)
{
  vuint64m2_t orig_x = __riscv_vreinterpret_v_f64m2_u64m2(x);
  vuint64m2_t orig_x_abs = __riscv_vand_vx_u64m2(orig_x, 0x7fffffffffffffffULL, vl);
  vbool32_t zero_mask = __riscv_vmseq_vx_u64m2_b32(orig_x_abs, 0, vl);

  vfloat64m2_t one = __riscv_vfmv_v_f_f64m2(1.0, vl);

#ifndef __FAST_MATH__
  vbool32_t nan_mask = __riscv_vmsgtu_vx_u64m2_b32(orig_x_abs, 0x7ff0000000000000ULL, vl);
  vbool32_t special_mask = __riscv_vmor_mm_b32(zero_mask,
                         __riscv_vmsgeu_vx_u64m2_b32(orig_x_abs, 0x7ff0000000000000ULL, vl), vl);

  vfloat64m2_t x_abs = __riscv_vreinterpret_v_u64m2_f64m2(orig_x_abs);
  vfloat64m2_t x_spec = __riscv_vfmerge_vfm_f64m2(x_abs, 1.0, special_mask, vl);

  vbool32_t denormal_mask = __riscv_vmsltu_vx_u64m2_b32(orig_x_abs, 0x0010000000000000ULL, vl);
  vfloat64m2_t denorm_mult = 
    __riscv_vfmerge_vfm_f64m2(one, 0x1.0p+60, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f64m2(x_spec, denorm_mult, vl);

  vuint64m2_t x_int = __riscv_vreinterpret_v_f64m2_u64m2(x_spec);
#else
  vuint64m2_t x_int = orig_x_abs;
#endif

  vuint64m2_t mantissa_in_x = __riscv_vand_vx_u64m2(x_int, 0x000fffffffffffffULL, vl);
  vuint64m2_t order_in_x = __riscv_vsrl_vx_u64m2(x_int, 52, vl);

  vuint64m2_t order_plus = __riscv_vadd_vx_u64m2(order_in_x, 177, vl);
  vuint64m2_t div3floor = __riscv_vsrl_vx_u64m2(
      __riscv_vmul_vx_u64m2(order_plus, 349526, vl), 20, vl);
  vuint64m2_t index = __riscv_vsub_vv_u64m2(order_plus, div3floor, vl);
  index = __riscv_vsub_vv_u64m2(index, __riscv_vsll_vx_u64m2(div3floor, 1, vl), vl);
  vuint64m2_t order_in_cbrt = __riscv_vadd_vx_u64m2(div3floor, 623, vl);

  vuint64m2_t reduced_x_int = 
    __riscv_vor_vx_u64m2(mantissa_in_x, 0x3ff0000000000000ULL, vl);
  vfloat64m2_t reduced_x = __riscv_vreinterpret_v_u64m2_f64m2(reduced_x_int);

  vfloat64m2_t yn = __riscv_vfmv_v_f_f64m2(-0x1.f40dc09665846p-1, vl);
  yn = __riscv_vfmacc_vf_f64m2(yn, 0x1.a3b694a2e22eap-3, reduced_x, vl);
  yn = __riscv_vfmadd_vv_f64m2(yn, reduced_x, __riscv_vfmv_v_f_f64m2(0x1.c446e7c491267p0, vl), vl);

  vfloat64m2_t r = __riscv_vfmul_vv_f64m2(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f64m2(r, r, vl);
  r = __riscv_vfmul_vv_f64m2(r, yn, vl);
  r = __riscv_vfnmsub_vv_f64m2(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f64m2(yn, 0x1.5555555555555p-2, r, vl);

  r = __riscv_vfmul_vv_f64m2(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f64m2(r, r, vl);
  r = __riscv_vfmul_vv_f64m2(r, yn, vl);
  r = __riscv_vfnmsub_vv_f64m2(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f64m2(yn, 0x1.5555555555555p-2, r, vl);

  r = __riscv_vfmul_vv_f64m2(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f64m2(r, r, vl);
  r = __riscv_vfmul_vv_f64m2(r, yn, vl);
  r = __riscv_vfnmsub_vv_f64m2(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f64m2(yn, 0x1.5555555555555p-2, r, vl);

  vfloat64m2_t rh = __riscv_vfmul_vv_f64m2(reduced_x, yn, vl);
  vfloat64m2_t rl = __riscv_vfmsac_vv_f64m2(rh, reduced_x, yn, vl);

  vfloat64m2_t hi = __riscv_vfmul_vv_f64m2(rh, rh, vl);
  vfloat64m2_t lo = __riscv_vfmsac_vv_f64m2(hi, rh, rh, vl);
  vfloat64m2_t cross = __riscv_vfmul_vv_f64m2(rl, rh, vl);
  cross = __riscv_vfadd_vv_f64m2(cross, cross, vl);
  lo = __riscv_vfadd_vv_f64m2(lo, cross, vl);

  rh = __riscv_vfadd_vv_f64m2(hi, lo, vl);
  vfloat64m2_t rl_tmp = __riscv_vfsub_vv_f64m2(rh, hi, vl);
  rl = __riscv_vfsub_vv_f64m2(lo, rl_tmp, vl);

  vfloat64m2_t rh1 = __riscv_vfnmsac_vv_f64m2(one, rh, yn, vl);
  vfloat64m2_t r2 = __riscv_vfmul_vv_f64m2(rh1, rh1, vl);
  vfloat64m2_t rl1 = __riscv_vfmul_vf_f64m2(r2, 0x1.5555555555555p-1, vl);
  rl1 = __riscv_vfnmsac_vv_f64m2(rl1, rl, yn, vl); 

  hi = __riscv_vfmul_vv_f64m2(rh1, yn, vl);
  lo = __riscv_vfmsac_vv_f64m2(hi, rh1, yn, vl);
  lo = __riscv_vfmacc_vv_f64m2(lo, rl1, yn, vl);
  
  rh = __riscv_vfadd_vv_f64m2(hi, lo, vl);
  rl_tmp = __riscv_vfsub_vv_f64m2(rh, hi, vl);
  rl = __riscv_vfsub_vv_f64m2(lo, rl_tmp, vl);

  vfloat64m2_t yh = __riscv_vfmacc_vf_f64m2(yn, 0x1.5555555555555p-2, rh, vl);
  vfloat64m2_t yl = __riscv_vfsub_vv_f64m2(yn, yh, vl);
  yl = __riscv_vfmacc_vf_f64m2(yl, 0x1.5555555555555p-2, rh, vl);
  yl = __riscv_vfmacc_vf_f64m2(yl, 0x1.5555555555555p-2, rl, vl);

  vbool32_t is_1 = __riscv_vmseq_vx_u64m2_b32(index, 1, vl);
  vbool32_t is_2 = __riscv_vmseq_vx_u64m2_b32(index, 2, vl);
  
  vfloat64m2_t ord_hi = __riscv_vfmv_v_f_f64m2(tab_high_dbl[0], vl);
  ord_hi = __riscv_vfmerge_vfm_f64m2(ord_hi, tab_high_dbl[1], is_1, vl);
  ord_hi = __riscv_vfmerge_vfm_f64m2(ord_hi, tab_high_dbl[2], is_2, vl);

  vfloat64m2_t ord_lo = __riscv_vfmv_v_f_f64m2(tab_low_dbl[0], vl);
  ord_lo = __riscv_vfmerge_vfm_f64m2(ord_lo, tab_low_dbl[1], is_1, vl);
  ord_lo = __riscv_vfmerge_vfm_f64m2(ord_lo, tab_low_dbl[2], is_2, vl);

  hi = __riscv_vfmul_vv_f64m2(yh, reduced_x, vl);
  lo = __riscv_vfmsac_vv_f64m2(hi, yh, reduced_x, vl);
  lo = __riscv_vfmacc_vv_f64m2(lo, yl, reduced_x, vl);
 
  rh = __riscv_vfmul_vv_f64m2(ord_hi, hi, vl);
  rl = __riscv_vfmsac_vv_f64m2(rh, ord_hi, hi, vl);
  rl = __riscv_vfmacc_vv_f64m2(rl, ord_lo, hi, vl);
  rl = __riscv_vfmacc_vv_f64m2(rl, ord_hi, lo, vl);
  rh = __riscv_vfadd_vv_f64m2(rh, rl, vl);

  vuint64m2_t mask_for_order = __riscv_vsll_vx_u64m2(order_in_cbrt, 52, vl);
  vfloat64m2_t order = __riscv_vreinterpret_v_u64m2_f64m2(mask_for_order);
  rh = __riscv_vfmul_vv_f64m2(rh, order, vl);

  vfloat64m2_t cbrt_value = __riscv_vfsgnj_vv_f64m2(rh, x, vl);

#ifndef __FAST_MATH__
  vfloat64m2_t denorm_mult_back = 
    __riscv_vfmerge_vfm_f64m2(one, 0x1.0p-20, denormal_mask, vl);
  cbrt_value = __riscv_vfmul_vv_f64m2(cbrt_value, denorm_mult_back, vl);

  cbrt_value = __riscv_vmerge_vvm_f64m2(cbrt_value, x, special_mask, vl);

  vbool32_t mask_sNaN = __riscv_vmand_mm_b32(
            __riscv_vmsltu_vx_u64m2_b32(orig_x_abs, 0x7ff8000000000000ULL, vl), nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b32(mask_sNaN, vl);
  if (issNaN) {
    volatile double x1 = 0.0 / 0.0;
    cbrt_value = __riscv_vfmerge_vfm_f64m2(cbrt_value, x1, mask_sNaN, vl);
  }
#else
  cbrt_value = __riscv_vmerge_vvm_f64m2(cbrt_value, x, zero_mask, vl);
#endif

  return cbrt_value; 
}


vfloat64m4_t __riscv_vcbrt_f64m4(vfloat64m4_t x, size_t vl)
{
  vuint64m4_t orig_x = __riscv_vreinterpret_v_f64m4_u64m4(x);
  vuint64m4_t orig_x_abs = __riscv_vand_vx_u64m4(orig_x, 0x7fffffffffffffffULL, vl);
  vbool16_t zero_mask = __riscv_vmseq_vx_u64m4_b16(orig_x_abs, 0, vl);

  vfloat64m4_t one = __riscv_vfmv_v_f_f64m4(1.0, vl);

#ifndef __FAST_MATH__
  vbool16_t nan_mask = __riscv_vmsgtu_vx_u64m4_b16(orig_x_abs, 0x7ff0000000000000ULL, vl);
  vbool16_t special_mask = __riscv_vmor_mm_b16(zero_mask,
                         __riscv_vmsgeu_vx_u64m4_b16(orig_x_abs, 0x7ff0000000000000ULL, vl), vl);

  vfloat64m4_t x_abs = __riscv_vreinterpret_v_u64m4_f64m4(orig_x_abs);
  vfloat64m4_t x_spec = __riscv_vfmerge_vfm_f64m4(x_abs, 1.0, special_mask, vl);

  vbool16_t denormal_mask = __riscv_vmsltu_vx_u64m4_b16(orig_x_abs, 0x0010000000000000ULL, vl);
  vfloat64m4_t denorm_mult = 
    __riscv_vfmerge_vfm_f64m4(one, 0x1.0p+60, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f64m4(x_spec, denorm_mult, vl);

  vuint64m4_t x_int = __riscv_vreinterpret_v_f64m4_u64m4(x_spec);
#else
  vuint64m4_t x_int = orig_x_abs;
#endif

  vuint64m4_t mantissa_in_x = __riscv_vand_vx_u64m4(x_int, 0x000fffffffffffffULL, vl);
  vuint64m4_t order_in_x = __riscv_vsrl_vx_u64m4(x_int, 52, vl);

  vuint64m4_t order_plus = __riscv_vadd_vx_u64m4(order_in_x, 177, vl);
  vuint64m4_t div3floor = __riscv_vsrl_vx_u64m4(
      __riscv_vmul_vx_u64m4(order_plus, 349526, vl), 20, vl);
  vuint64m4_t index = __riscv_vsub_vv_u64m4(order_plus, div3floor, vl);
  index = __riscv_vsub_vv_u64m4(index, __riscv_vsll_vx_u64m4(div3floor, 1, vl), vl);
  vuint64m4_t order_in_cbrt = __riscv_vadd_vx_u64m4(div3floor, 623, vl);

  vuint64m4_t reduced_x_int = 
    __riscv_vor_vx_u64m4(mantissa_in_x, 0x3ff0000000000000ULL, vl);
  vfloat64m4_t reduced_x = __riscv_vreinterpret_v_u64m4_f64m4(reduced_x_int);

  vfloat64m4_t yn = __riscv_vfmv_v_f_f64m4(-0x1.f40dc09665846p-1, vl);
  yn = __riscv_vfmacc_vf_f64m4(yn, 0x1.a3b694a2e22eap-3, reduced_x, vl);
  yn = __riscv_vfmadd_vv_f64m4(yn, reduced_x, __riscv_vfmv_v_f_f64m4(0x1.c446e7c491267p0, vl), vl);

  vfloat64m4_t r = __riscv_vfmul_vv_f64m4(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f64m4(r, r, vl);
  r = __riscv_vfmul_vv_f64m4(r, yn, vl);
  r = __riscv_vfnmsub_vv_f64m4(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f64m4(yn, 0x1.5555555555555p-2, r, vl);

  r = __riscv_vfmul_vv_f64m4(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f64m4(r, r, vl);
  r = __riscv_vfmul_vv_f64m4(r, yn, vl);
  r = __riscv_vfnmsub_vv_f64m4(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f64m4(yn, 0x1.5555555555555p-2, r, vl);

  r = __riscv_vfmul_vv_f64m4(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f64m4(r, r, vl);
  r = __riscv_vfmul_vv_f64m4(r, yn, vl);
  r = __riscv_vfnmsub_vv_f64m4(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f64m4(yn, 0x1.5555555555555p-2, r, vl);

  vfloat64m4_t rh = __riscv_vfmul_vv_f64m4(reduced_x, yn, vl);
  vfloat64m4_t rl = __riscv_vfmsac_vv_f64m4(rh, reduced_x, yn, vl);

  vfloat64m4_t hi = __riscv_vfmul_vv_f64m4(rh, rh, vl);
  vfloat64m4_t lo = __riscv_vfmsac_vv_f64m4(hi, rh, rh, vl);
  vfloat64m4_t cross = __riscv_vfmul_vv_f64m4(rl, rh, vl);
  cross = __riscv_vfadd_vv_f64m4(cross, cross, vl);
  lo = __riscv_vfadd_vv_f64m4(lo, cross, vl);

  rh = __riscv_vfadd_vv_f64m4(hi, lo, vl);
  vfloat64m4_t rl_tmp = __riscv_vfsub_vv_f64m4(rh, hi, vl);
  rl = __riscv_vfsub_vv_f64m4(lo, rl_tmp, vl);

  vfloat64m4_t rh1 = __riscv_vfnmsac_vv_f64m4(one, rh, yn, vl);
  vfloat64m4_t r2 = __riscv_vfmul_vv_f64m4(rh1, rh1, vl);
  vfloat64m4_t rl1 = __riscv_vfmul_vf_f64m4(r2, 0x1.5555555555555p-1, vl);
  rl1 = __riscv_vfnmsac_vv_f64m4(rl1, rl, yn, vl); 

  hi = __riscv_vfmul_vv_f64m4(rh1, yn, vl);
  lo = __riscv_vfmsac_vv_f64m4(hi, rh1, yn, vl);
  lo = __riscv_vfmacc_vv_f64m4(lo, rl1, yn, vl);
  
  rh = __riscv_vfadd_vv_f64m4(hi, lo, vl);
  rl_tmp = __riscv_vfsub_vv_f64m4(rh, hi, vl);
  rl = __riscv_vfsub_vv_f64m4(lo, rl_tmp, vl);

  vfloat64m4_t yh = __riscv_vfmacc_vf_f64m4(yn, 0x1.5555555555555p-2, rh, vl);
  vfloat64m4_t yl = __riscv_vfsub_vv_f64m4(yn, yh, vl);
  yl = __riscv_vfmacc_vf_f64m4(yl, 0x1.5555555555555p-2, rh, vl);
  yl = __riscv_vfmacc_vf_f64m4(yl, 0x1.5555555555555p-2, rl, vl);

  vbool16_t is_1 = __riscv_vmseq_vx_u64m4_b16(index, 1, vl);
  vbool16_t is_2 = __riscv_vmseq_vx_u64m4_b16(index, 2, vl);
  
  vfloat64m4_t ord_hi = __riscv_vfmv_v_f_f64m4(tab_high_dbl[0], vl);
  ord_hi = __riscv_vfmerge_vfm_f64m4(ord_hi, tab_high_dbl[1], is_1, vl);
  ord_hi = __riscv_vfmerge_vfm_f64m4(ord_hi, tab_high_dbl[2], is_2, vl);

  vfloat64m4_t ord_lo = __riscv_vfmv_v_f_f64m4(tab_low_dbl[0], vl);
  ord_lo = __riscv_vfmerge_vfm_f64m4(ord_lo, tab_low_dbl[1], is_1, vl);
  ord_lo = __riscv_vfmerge_vfm_f64m4(ord_lo, tab_low_dbl[2], is_2, vl);

  hi = __riscv_vfmul_vv_f64m4(yh, reduced_x, vl);
  lo = __riscv_vfmsac_vv_f64m4(hi, yh, reduced_x, vl);
  lo = __riscv_vfmacc_vv_f64m4(lo, yl, reduced_x, vl);
 
  rh = __riscv_vfmul_vv_f64m4(ord_hi, hi, vl);
  rl = __riscv_vfmsac_vv_f64m4(rh, ord_hi, hi, vl);
  rl = __riscv_vfmacc_vv_f64m4(rl, ord_lo, hi, vl);
  rl = __riscv_vfmacc_vv_f64m4(rl, ord_hi, lo, vl);
  rh = __riscv_vfadd_vv_f64m4(rh, rl, vl);

  vuint64m4_t mask_for_order = __riscv_vsll_vx_u64m4(order_in_cbrt, 52, vl);
  vfloat64m4_t order = __riscv_vreinterpret_v_u64m4_f64m4(mask_for_order);
  rh = __riscv_vfmul_vv_f64m4(rh, order, vl);

  vfloat64m4_t cbrt_value = __riscv_vfsgnj_vv_f64m4(rh, x, vl);

#ifndef __FAST_MATH__
  vfloat64m4_t denorm_mult_back = 
    __riscv_vfmerge_vfm_f64m4(one, 0x1.0p-20, denormal_mask, vl);
  cbrt_value = __riscv_vfmul_vv_f64m4(cbrt_value, denorm_mult_back, vl);

  cbrt_value = __riscv_vmerge_vvm_f64m4(cbrt_value, x, special_mask, vl);

  vbool16_t mask_sNaN = __riscv_vmand_mm_b16(
            __riscv_vmsltu_vx_u64m4_b16(orig_x_abs, 0x7ff8000000000000ULL, vl), nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
  if (issNaN) {
    volatile double x1 = 0.0 / 0.0;
    cbrt_value = __riscv_vfmerge_vfm_f64m4(cbrt_value, x1, mask_sNaN, vl);
  }
#else
  cbrt_value = __riscv_vmerge_vvm_f64m4(cbrt_value, x, zero_mask, vl);
#endif

  return cbrt_value; 
}


vfloat64m8_t __riscv_vcbrt_f64m8(vfloat64m8_t x, size_t vl)
{
  vuint64m8_t orig_x = __riscv_vreinterpret_v_f64m8_u64m8(x);
  vuint64m8_t orig_x_abs = __riscv_vand_vx_u64m8(orig_x, 0x7fffffffffffffffULL, vl);
  vbool8_t zero_mask = __riscv_vmseq_vx_u64m8_b8(orig_x_abs, 0, vl);

  vfloat64m8_t one = __riscv_vfmv_v_f_f64m8(1.0, vl);

#ifndef __FAST_MATH__
  vbool8_t nan_mask = __riscv_vmsgtu_vx_u64m8_b8(orig_x_abs, 0x7ff0000000000000ULL, vl);
  vbool8_t special_mask = __riscv_vmor_mm_b8(zero_mask,
                         __riscv_vmsgeu_vx_u64m8_b8(orig_x_abs, 0x7ff0000000000000ULL, vl), vl);

  vfloat64m8_t x_abs = __riscv_vreinterpret_v_u64m8_f64m8(orig_x_abs);
  vfloat64m8_t x_spec = __riscv_vfmerge_vfm_f64m8(x_abs, 1.0, special_mask, vl);

  vbool8_t denormal_mask = __riscv_vmsltu_vx_u64m8_b8(orig_x_abs, 0x0010000000000000ULL, vl);
  vfloat64m8_t denorm_mult = 
    __riscv_vfmerge_vfm_f64m8(one, 0x1.0p+60, denormal_mask, vl);
  x_spec = __riscv_vfmul_vv_f64m8(x_spec, denorm_mult, vl);

  vuint64m8_t x_int = __riscv_vreinterpret_v_f64m8_u64m8(x_spec);
#else
  vuint64m8_t x_int = orig_x_abs;
#endif

  vuint64m8_t mantissa_in_x = __riscv_vand_vx_u64m8(x_int, 0x000fffffffffffffULL, vl);
  vuint64m8_t order_in_x = __riscv_vsrl_vx_u64m8(x_int, 52, vl);

  vuint64m8_t order_plus = __riscv_vadd_vx_u64m8(order_in_x, 177, vl);
  vuint64m8_t div3floor = __riscv_vsrl_vx_u64m8(
      __riscv_vmul_vx_u64m8(order_plus, 349526, vl), 20, vl);
  vuint64m8_t index = __riscv_vsub_vv_u64m8(order_plus, div3floor, vl);
  index = __riscv_vsub_vv_u64m8(index, __riscv_vsll_vx_u64m8(div3floor, 1, vl), vl);
  vuint64m8_t order_in_cbrt = __riscv_vadd_vx_u64m8(div3floor, 623, vl);

  vuint64m8_t reduced_x_int = 
    __riscv_vor_vx_u64m8(mantissa_in_x, 0x3ff0000000000000ULL, vl);
  vfloat64m8_t reduced_x = __riscv_vreinterpret_v_u64m8_f64m8(reduced_x_int);

  vfloat64m8_t yn = __riscv_vfmv_v_f_f64m8(-0x1.f40dc09665846p-1, vl);
  yn = __riscv_vfmacc_vf_f64m8(yn, 0x1.a3b694a2e22eap-3, reduced_x, vl);
  yn = __riscv_vfmadd_vv_f64m8(yn, reduced_x, __riscv_vfmv_v_f_f64m8(0x1.c446e7c491267p0, vl), vl);

  vfloat64m8_t r = __riscv_vfmul_vv_f64m8(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f64m8(r, r, vl);
  r = __riscv_vfmul_vv_f64m8(r, yn, vl);
  r = __riscv_vfnmsub_vv_f64m8(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f64m8(yn, 0x1.5555555555555p-2, r, vl);

  r = __riscv_vfmul_vv_f64m8(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f64m8(r, r, vl);
  r = __riscv_vfmul_vv_f64m8(r, yn, vl);
  r = __riscv_vfnmsub_vv_f64m8(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f64m8(yn, 0x1.5555555555555p-2, r, vl);

  r = __riscv_vfmul_vv_f64m8(reduced_x, yn, vl);
  r = __riscv_vfmul_vv_f64m8(r, r, vl);
  r = __riscv_vfmul_vv_f64m8(r, yn, vl);
  r = __riscv_vfnmsub_vv_f64m8(r, yn, yn, vl);
  yn = __riscv_vfmacc_vf_f64m8(yn, 0x1.5555555555555p-2, r, vl);

  vfloat64m8_t rh = __riscv_vfmul_vv_f64m8(reduced_x, yn, vl);
  vfloat64m8_t rl = __riscv_vfmsac_vv_f64m8(rh, reduced_x, yn, vl);

  vfloat64m8_t hi = __riscv_vfmul_vv_f64m8(rh, rh, vl);
  vfloat64m8_t lo = __riscv_vfmsac_vv_f64m8(hi, rh, rh, vl);
  vfloat64m8_t cross = __riscv_vfmul_vv_f64m8(rl, rh, vl);
  cross = __riscv_vfadd_vv_f64m8(cross, cross, vl);
  lo = __riscv_vfadd_vv_f64m8(lo, cross, vl);

  rh = __riscv_vfadd_vv_f64m8(hi, lo, vl);
  vfloat64m8_t rl_tmp = __riscv_vfsub_vv_f64m8(rh, hi, vl);
  rl = __riscv_vfsub_vv_f64m8(lo, rl_tmp, vl);

  vfloat64m8_t rh1 = __riscv_vfnmsac_vv_f64m8(one, rh, yn, vl);
  vfloat64m8_t r2 = __riscv_vfmul_vv_f64m8(rh1, rh1, vl);
  vfloat64m8_t rl1 = __riscv_vfmul_vf_f64m8(r2, 0x1.5555555555555p-1, vl);
  rl1 = __riscv_vfnmsac_vv_f64m8(rl1, rl, yn, vl); 

  hi = __riscv_vfmul_vv_f64m8(rh1, yn, vl);
  lo = __riscv_vfmsac_vv_f64m8(hi, rh1, yn, vl);
  lo = __riscv_vfmacc_vv_f64m8(lo, rl1, yn, vl);
  
  rh = __riscv_vfadd_vv_f64m8(hi, lo, vl);
  rl_tmp = __riscv_vfsub_vv_f64m8(rh, hi, vl);
  rl = __riscv_vfsub_vv_f64m8(lo, rl_tmp, vl);

  vfloat64m8_t yh = __riscv_vfmacc_vf_f64m8(yn, 0x1.5555555555555p-2, rh, vl);
  vfloat64m8_t yl = __riscv_vfsub_vv_f64m8(yn, yh, vl);
  yl = __riscv_vfmacc_vf_f64m8(yl, 0x1.5555555555555p-2, rh, vl);
  yl = __riscv_vfmacc_vf_f64m8(yl, 0x1.5555555555555p-2, rl, vl);

  vbool8_t is_1 = __riscv_vmseq_vx_u64m8_b8(index, 1, vl);
  vbool8_t is_2 = __riscv_vmseq_vx_u64m8_b8(index, 2, vl);
  
  vfloat64m8_t ord_hi = __riscv_vfmv_v_f_f64m8(tab_high_dbl[0], vl);
  ord_hi = __riscv_vfmerge_vfm_f64m8(ord_hi, tab_high_dbl[1], is_1, vl);
  ord_hi = __riscv_vfmerge_vfm_f64m8(ord_hi, tab_high_dbl[2], is_2, vl);

  vfloat64m8_t ord_lo = __riscv_vfmv_v_f_f64m8(tab_low_dbl[0], vl);
  ord_lo = __riscv_vfmerge_vfm_f64m8(ord_lo, tab_low_dbl[1], is_1, vl);
  ord_lo = __riscv_vfmerge_vfm_f64m8(ord_lo, tab_low_dbl[2], is_2, vl);

  hi = __riscv_vfmul_vv_f64m8(yh, reduced_x, vl);
  lo = __riscv_vfmsac_vv_f64m8(hi, yh, reduced_x, vl);
  lo = __riscv_vfmacc_vv_f64m8(lo, yl, reduced_x, vl);
 
  rh = __riscv_vfmul_vv_f64m8(ord_hi, hi, vl);
  rl = __riscv_vfmsac_vv_f64m8(rh, ord_hi, hi, vl);
  rl = __riscv_vfmacc_vv_f64m8(rl, ord_lo, hi, vl);
  rl = __riscv_vfmacc_vv_f64m8(rl, ord_hi, lo, vl);
  rh = __riscv_vfadd_vv_f64m8(rh, rl, vl);

  vuint64m8_t mask_for_order = __riscv_vsll_vx_u64m8(order_in_cbrt, 52, vl);
  vfloat64m8_t order = __riscv_vreinterpret_v_u64m8_f64m8(mask_for_order);
  rh = __riscv_vfmul_vv_f64m8(rh, order, vl);

  vfloat64m8_t cbrt_value = __riscv_vfsgnj_vv_f64m8(rh, x, vl);

#ifndef __FAST_MATH__
  vfloat64m8_t denorm_mult_back = 
    __riscv_vfmerge_vfm_f64m8(one, 0x1.0p-20, denormal_mask, vl);
  cbrt_value = __riscv_vfmul_vv_f64m8(cbrt_value, denorm_mult_back, vl);

  cbrt_value = __riscv_vmerge_vvm_f64m8(cbrt_value, x, special_mask, vl);

  vbool8_t mask_sNaN = __riscv_vmand_mm_b8(
            __riscv_vmsltu_vx_u64m8_b8(orig_x_abs, 0x7ff8000000000000ULL, vl), nan_mask, vl);
  unsigned int issNaN = __riscv_vcpop_m_b8(mask_sNaN, vl);
  if (issNaN) {
    volatile double x1 = 0.0 / 0.0;
    cbrt_value = __riscv_vfmerge_vfm_f64m8(cbrt_value, x1, mask_sNaN, vl);
  }
#else
  cbrt_value = __riscv_vmerge_vvm_f64m8(cbrt_value, x, zero_mask, vl);
#endif

  return cbrt_value; 
}
#endif