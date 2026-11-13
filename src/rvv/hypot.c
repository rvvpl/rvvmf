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
 *   File:  hypot.c                                      *
 *   Contains: intrinsic function hypot for f64, f32     *
 *             and f16                                   *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input VL number of elements in vector register        *
 *                                                       *
 * Return value: square root of the sum of the squares   *
 *   of corresponding elements of vectors x and y        *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

static _Float16 order_tab_high_f16[2] =
{ 0x1.0p+0f16, 0x1.0p+8f16 };
static _Float16 order_tab_low_f16[16] =
{ 0x1.6a0p-8f16, 0x1.0p-7f16, 0x1.6a0p-7f16, 0x1.0p-6f16,
  0x1.6a0p-6f16, 0x1.0p-5f16, 0x1.6a0p-5f16, 0x1.0p-4f16,
  0x1.6a0p-4f16, 0x1.0p-3f16, 0x1.6a0p-3f16, 0x1.0p-2f16,
  0x1.6a0p-2f16, 0x1.0p-1f16, 0x1.6a0p-1f16, 0x1.0p+0f16 };
static _Float16 order_tab_low_f16_[16] =
{ 0x0.9e6p-19f16, 0x0.0p+0f16, 0x0.9e6p-18f16, 0x0.0p+0f16,
  0x0.9e6p-17f16, 0x0.0p+0f16, 0x0.9e6p-16f16, 0x0.0p+0f16,
  0x0.9e6p-15f16, 0x0.0p+0f16, 0x0.9e6p-14f16, 0x0.0p+0f16,
  0x0.9e6p-13f16, 0x0.0p+0f16, 0x0.9e6p-12f16, 0x0.0p+0f16 };

static float order_tab_high_flt[16] =
{ 0x1.0p-56, 0x1.0p-48, 0x1.0p-40, 0x1.0p-32,
  0x1.0p-24, 0x1.0p-16, 0x1.0p-8,  0x1.0p+0,
  0x1.0p+8,  0x1.0p+16, 0x1.0p+24, 0x1.0p+32,
  0x1.0p+40, 0x1.0p+48, 0x1.0p+56, 0x1.0p+64 };
static float order_tab_low_flt[16] =
{ 0x1.6a09e6p-8, 0x1.0p-7, 0x1.6a09e6p-7, 0x1.0p-6,
  0x1.6a09e6p-6, 0x1.0p-5, 0x1.6a09e6p-5, 0x1.0p-4,
  0x1.6a09e6p-4, 0x1.0p-3, 0x1.6a09e6p-3, 0x1.0p-2,
  0x1.6a09e6p-2, 0x1.0p-1, 0x1.6a09e6p-1, 0x1.0p+0 };
static float order_tab_low_flt_[16] =
{ 0x0.67f3bcdp-32, 0x0.0p+0, 0x0.67f3bcdp-31, 0x0.0p+0,
  0x0.67f3bcdp-30, 0x0.0p+0, 0x0.67f3bcdp-29, 0x0.0p+0,
  0x0.67f3bcdp-28, 0x0.0p+0, 0x0.67f3bcdp-27, 0x0.0p+0,
  0x0.67f3bcdp-26, 0x0.0p+0, 0x0.67f3bcdp-25, 0x0.0p+0 };

static double order_tab_high[8] =
{ 0x1.0p-384, 0x1.0p-256, 0x1.0p-128, 0x1.0p+0,
  0x1.0p+128, 0x1.0p+256, 0x1.0p+384, 0x1.0p+512 };
static double order_tab_mid[16] =
{ 0x1.0p-120, 0x1.0p-112, 0x1.0p-104, 0x1.0p-96,
  0x1.0p-88,  0x1.0p-80,  0x1.0p-72,  0x1.0p-64,
  0x1.0p-56,  0x1.0p-48,  0x1.0p-40,  0x1.0p-32,
  0x1.0p-24,  0x1.0p-16,  0x1.0p-8,   0x1.0p+0 };
static double order_tab_low[16] =
{ 0x1.6a09e667f3bccp-8, 0x1.0p-7, 0x1.6a09e667f3bccp-7, 0x1.0p-6,
  0x1.6a09e667f3bccp-6, 0x1.0p-5, 0x1.6a09e667f3bccp-5, 0x1.0p-4,
  0x1.6a09e667f3bccp-4, 0x1.0p-3, 0x1.6a09e667f3bccp-3, 0x1.0p-2,
  0x1.6a09e667f3bccp-2, 0x1.0p-1, 0x1.6a09e667f3bccp-1, 0x1.0p+0 };
static double order_tab_low_[16] =
{ 0x1.21165f626cdd5p-61, 0x0.0p+0, 0x1.21165f626cdd5p-60, 0x0.0p+0,
  0x1.21165f626cdd5p-59, 0x0.0p+0, 0x1.21165f626cdd5p-58, 0x0.0p+0,
  0x1.21165f626cdd5p-57, 0x0.0p+0, 0x1.21165f626cdd5p-56, 0x0.0p+0,
  0x1.21165f626cdd5p-55, 0x0.0p+0, 0x1.21165f626cdd5p-54, 0x0.0p+0 };

vfloat16m1_t __riscv_vsqrt_f16m1(vfloat16m1_t x, size_t vl);
vfloat16m2_t __riscv_vsqrt_f16m2(vfloat16m2_t x, size_t vl);
vfloat16m4_t __riscv_vsqrt_f16m4(vfloat16m4_t x, size_t vl);
vfloat16m8_t __riscv_vsqrt_f16m8(vfloat16m8_t x, size_t vl);
vfloat32m1_t __riscv_vsqrt_f32m1(vfloat32m1_t x, size_t vl);
vfloat32m2_t __riscv_vsqrt_f32m2(vfloat32m2_t x, size_t vl);
vfloat32m4_t __riscv_vsqrt_f32m4(vfloat32m4_t x, size_t vl);
vfloat32m8_t __riscv_vsqrt_f32m8(vfloat32m8_t x, size_t vl);
vfloat64m1_t __riscv_vsqrt_f64m1(vfloat64m1_t x, size_t vl);
vfloat64m2_t __riscv_vsqrt_f64m2(vfloat64m2_t x, size_t vl);
vfloat64m4_t __riscv_vsqrt_f64m4(vfloat64m4_t x, size_t vl);
vfloat64m8_t __riscv_vsqrt_f64m8(vfloat64m8_t x, size_t vl);

vfloat16m1_t __riscv_vhypot_f16m1(vfloat16m1_t x, vfloat16m1_t y, size_t vl)
{
  vuint16m1_t x_int = 
    __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(x), 0x7fff, vl);
  vuint16m1_t y_int = 
    __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(y), 0x7fff, vl);

#ifndef __FAST_MATH__
  unsigned short nan_si = 0x7e00; // mask for NaN
  _Float16 nan_f16 = *(_Float16*)(&nan_si); // NaN
  unsigned short inf_si = 0x7c00; // mask for +inf
  _Float16 inf_f16 = *(_Float16*)(&inf_si); // +inf

  vbool16_t x_inf_mask = __riscv_vmseq_vx_u16m1_b16(x_int, 0x7c00, vl); //vmfeq x==inf bugs
  vbool16_t y_inf_mask = __riscv_vmseq_vx_u16m1_b16(y_int, 0x7c00, vl);
  vbool16_t x_nan_mask = __riscv_vmsgtu_vx_u16m1_b16(x_int, 0x7c00, vl);
  vbool16_t y_nan_mask = __riscv_vmsgtu_vx_u16m1_b16(y_int, 0x7c00, vl);

  vbool16_t special_mask = __riscv_vmor_mm_b16(x_inf_mask, y_inf_mask, vl);
  special_mask = __riscv_vmor_mm_b16(special_mask, x_nan_mask, vl);
  special_mask = __riscv_vmor_mm_b16(special_mask, y_nan_mask, vl);
  x_int = __riscv_vmerge_vxm_u16m1(x_int, 1, special_mask, vl);
  y_int = __riscv_vmerge_vxm_u16m1(y_int, 1, special_mask, vl);
#endif

  vuint16m1_t u_max = __riscv_vmaxu_vv_u16m1(x_int, y_int, vl);
  vuint16m1_t u_min = __riscv_vminu_vv_u16m1(x_int, y_int, vl);

  vuint16m1_t diff = __riscv_vsub_vv_u16m1(u_max, u_min, vl);
  vbool16_t cutoff_mask = __riscv_vmsgeu_vx_u16m1_b16(diff, 0x3000, vl);
  u_min = __riscv_vmerge_vxm_u16m1(u_min, 0, cutoff_mask, vl);

  vuint16m1_t E_bits = __riscv_vand_vx_u16m1(u_max, 0x7c00, vl);
  vuint16m1_t M_bit = __riscv_vand_vx_u16m1(u_max, 0x4000, vl); 
  vuint16m1_t base = __riscv_vsub_vv_u16m1(E_bits, __riscv_vsrl_vx_u16m1(M_bit, 3, vl), vl);
  
  // a-subnormal => M=0 => e_in = 29 => sa in [2^-10, 0.5)
  // a < 2  => M = 0 => e_in = 29 - e => sa in [0.5, 1) 
  // a >= 2 => M = 1 => e_in = 31 - e => sa in [2, 4) 
  vuint16m1_t e_out = __riscv_vadd_vx_u16m1(base, 0x0400, vl);
  vuint16m1_t e_in = __riscv_vrsub_vx_u16m1(base, 0x7400, vl);

  vfloat16m1_t scale_in  = __riscv_vreinterpret_v_u16m1_f16m1(e_in);
  vfloat16m1_t scale_out = __riscv_vreinterpret_v_u16m1_f16m1(e_out);

  vfloat16m1_t sa = __riscv_vfmul_vv_f16m1(__riscv_vreinterpret_v_u16m1_f16m1(u_max), scale_in, vl);
  vfloat16m1_t sb = __riscv_vfmul_vv_f16m1(__riscv_vreinterpret_v_u16m1_f16m1(u_min), scale_in, vl);

  vfloat16m1_t sb2 = __riscv_vfmul_vv_f16m1(sb, sb, vl);
  vfloat16m1_t sum2 = __riscv_vfmadd_vv_f16m1(sa, sa, sb2, vl);
  
  vfloat16m1_t r = __riscv_vsqrt_f16m1(sum2, vl);  
  vfloat16m1_t sqrt_value = __riscv_vfmul_vv_f16m1(r, scale_out, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m1(sqrt_value, nan_si, x_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m1(sqrt_value, nan_si, y_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m1(sqrt_value, inf_si, x_inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m1(sqrt_value, inf_si, y_inf_mask, vl);
#endif

  return sqrt_value;
}

vfloat16m2_t __riscv_vhypot_f16m2(vfloat16m2_t x, vfloat16m2_t y, size_t vl)
{
  vuint16m2_t x_int = 
    __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(x), 0x7fff, vl);
  vuint16m2_t y_int = 
    __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(y), 0x7fff, vl);

#ifndef __FAST_MATH__
  unsigned short nan_si = 0x7e00; // mask for NaN
  _Float16 nan_f16 = *(_Float16*)(&nan_si); // NaN
  unsigned short inf_si = 0x7c00; // mask for +inf
  _Float16 inf_f16 = *(_Float16*)(&inf_si); // +inf

  vbool8_t x_inf_mask = __riscv_vmseq_vx_u16m2_b8(x_int, 0x7c00, vl); //vmfeq x==inf bugs
  vbool8_t y_inf_mask = __riscv_vmseq_vx_u16m2_b8(y_int, 0x7c00, vl);
  vbool8_t x_nan_mask = __riscv_vmsgtu_vx_u16m2_b8(x_int, 0x7c00, vl);
  vbool8_t y_nan_mask = __riscv_vmsgtu_vx_u16m2_b8(y_int, 0x7c00, vl);

  vbool8_t special_mask = __riscv_vmor_mm_b8(x_inf_mask, y_inf_mask, vl);
  special_mask = __riscv_vmor_mm_b8(special_mask, x_nan_mask, vl);
  special_mask = __riscv_vmor_mm_b8(special_mask, y_nan_mask, vl);
  x_int = __riscv_vmerge_vxm_u16m2(x_int, 1, special_mask, vl);
  y_int = __riscv_vmerge_vxm_u16m2(y_int, 1, special_mask, vl);
#endif

  vuint16m2_t u_max = __riscv_vmaxu_vv_u16m2(x_int, y_int, vl);
  vuint16m2_t u_min = __riscv_vminu_vv_u16m2(x_int, y_int, vl);

  vuint16m2_t diff = __riscv_vsub_vv_u16m2(u_max, u_min, vl);
  vbool8_t cutoff_mask = __riscv_vmsgeu_vx_u16m2_b8(diff, 0x3000, vl);
  u_min = __riscv_vmerge_vxm_u16m2(u_min, 0, cutoff_mask, vl);

  vuint16m2_t E_bits = __riscv_vand_vx_u16m2(u_max, 0x7c00, vl);
  vuint16m2_t M_bit = __riscv_vand_vx_u16m2(u_max, 0x4000, vl); 
  vuint16m2_t base = __riscv_vsub_vv_u16m2(E_bits, __riscv_vsrl_vx_u16m2(M_bit, 3, vl), vl);
  
  // a-subnormal => M=0 => e_in = 29 => sa in [2^-10, 0.5)
  // a < 2  => M = 0 => e_in = 29 - e => sa in [0.5, 1) 
  // a >= 2 => M = 1 => e_in = 31 - e => sa in [2, 4) 
  vuint16m2_t e_out = __riscv_vadd_vx_u16m2(base, 0x0400, vl);
  vuint16m2_t e_in = __riscv_vrsub_vx_u16m2(base, 0x7400, vl);

  vfloat16m2_t scale_in  = __riscv_vreinterpret_v_u16m2_f16m2(e_in);
  vfloat16m2_t scale_out = __riscv_vreinterpret_v_u16m2_f16m2(e_out);

  vfloat16m2_t sa = __riscv_vfmul_vv_f16m2(__riscv_vreinterpret_v_u16m2_f16m2(u_max), scale_in, vl);
  vfloat16m2_t sb = __riscv_vfmul_vv_f16m2(__riscv_vreinterpret_v_u16m2_f16m2(u_min), scale_in, vl);

  vfloat16m2_t sb2 = __riscv_vfmul_vv_f16m2(sb, sb, vl);
  vfloat16m2_t sum2 = __riscv_vfmadd_vv_f16m2(sa, sa, sb2, vl);
  
  vfloat16m2_t r = __riscv_vsqrt_f16m2(sum2, vl);  
  vfloat16m2_t sqrt_value = __riscv_vfmul_vv_f16m2(r, scale_out, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m2(sqrt_value, nan_si, x_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m2(sqrt_value, nan_si, y_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m2(sqrt_value, inf_si, x_inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m2(sqrt_value, inf_si, y_inf_mask, vl);
#endif

  return sqrt_value;
}

vfloat16m4_t __riscv_vhypot_f16m4(vfloat16m4_t x, vfloat16m4_t y, size_t vl)
{
  vuint16m4_t x_int = 
    __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(x), 0x7fff, vl);
  vuint16m4_t y_int = 
    __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(y), 0x7fff, vl);

#ifndef __FAST_MATH__
  unsigned short nan_si = 0x7e00; // mask for NaN
  _Float16 nan_f16 = *(_Float16*)(&nan_si); // NaN
  unsigned short inf_si = 0x7c00; // mask for +inf
  _Float16 inf_f16 = *(_Float16*)(&inf_si); // +inf

  vbool4_t x_inf_mask = __riscv_vmseq_vx_u16m4_b4(x_int, 0x7c00, vl); //vmfeq x==inf bugs
  vbool4_t y_inf_mask = __riscv_vmseq_vx_u16m4_b4(y_int, 0x7c00, vl);
  vbool4_t x_nan_mask = __riscv_vmsgtu_vx_u16m4_b4(x_int, 0x7c00, vl);
  vbool4_t y_nan_mask = __riscv_vmsgtu_vx_u16m4_b4(y_int, 0x7c00, vl);

  vbool4_t special_mask = __riscv_vmor_mm_b4(x_inf_mask, y_inf_mask, vl);
  special_mask = __riscv_vmor_mm_b4(special_mask, x_nan_mask, vl);
  special_mask = __riscv_vmor_mm_b4(special_mask, y_nan_mask, vl);
  x_int = __riscv_vmerge_vxm_u16m4(x_int, 1, special_mask, vl);
  y_int = __riscv_vmerge_vxm_u16m4(y_int, 1, special_mask, vl);
#endif
  
  vuint16m4_t u_max = __riscv_vmaxu_vv_u16m4(x_int, y_int, vl);
  vuint16m4_t u_min = __riscv_vminu_vv_u16m4(x_int, y_int, vl);

  vuint16m4_t diff = __riscv_vsub_vv_u16m4(u_max, u_min, vl);
  vbool4_t cutoff_mask = __riscv_vmsgeu_vx_u16m4_b4(diff, 0x3000, vl);
  u_min = __riscv_vmerge_vxm_u16m4(u_min, 0, cutoff_mask, vl);

  vuint16m4_t E_bits = __riscv_vand_vx_u16m4(u_max, 0x7c00, vl);
  vuint16m4_t M_bit = __riscv_vand_vx_u16m4(u_max, 0x4000, vl); 
  vuint16m4_t base = __riscv_vsub_vv_u16m4(E_bits, __riscv_vsrl_vx_u16m4(M_bit, 3, vl), vl);
  
  // a-subnormal => M=0 => e_in = 29 => sa in [2^-10, 0.5)
  // a < 2  => M = 0 => e_in = 29 - e => sa in [0.5, 1) 
  // a >= 2 => M = 1 => e_in = 31 - e => sa in [2, 4) 
  vuint16m4_t e_out = __riscv_vadd_vx_u16m4(base, 0x0400, vl);
  vuint16m4_t e_in = __riscv_vrsub_vx_u16m4(base, 0x7400, vl);

  vfloat16m4_t scale_in  = __riscv_vreinterpret_v_u16m4_f16m4(e_in);
  vfloat16m4_t scale_out = __riscv_vreinterpret_v_u16m4_f16m4(e_out);

  vfloat16m4_t sa = __riscv_vfmul_vv_f16m4(__riscv_vreinterpret_v_u16m4_f16m4(u_max), scale_in, vl);
  vfloat16m4_t sb = __riscv_vfmul_vv_f16m4(__riscv_vreinterpret_v_u16m4_f16m4(u_min), scale_in, vl);

  vfloat16m4_t sb2 = __riscv_vfmul_vv_f16m4(sb, sb, vl);
  vfloat16m4_t sum2 = __riscv_vfmadd_vv_f16m4(sa, sa, sb2, vl);
  
  vfloat16m4_t r = __riscv_vsqrt_f16m4(sum2, vl);  
  vfloat16m4_t sqrt_value = __riscv_vfmul_vv_f16m4(r, scale_out, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m4(sqrt_value, nan_si, x_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m4(sqrt_value, nan_si, y_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m4(sqrt_value, inf_si, x_inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m4(sqrt_value, inf_si, y_inf_mask, vl);
#endif

  return sqrt_value;
}

vfloat16m8_t __riscv_vhypot_f16m8(vfloat16m8_t x, vfloat16m8_t y, size_t vl)
{
  vuint16m8_t x_int = 
    __riscv_vand_vx_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(x), 0x7fff, vl);
  vuint16m8_t y_int = 
    __riscv_vand_vx_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(y), 0x7fff, vl);

#ifndef __FAST_MATH__
  unsigned short nan_si = 0x7e00; // mask for NaN
  _Float16 nan_f16 = *(_Float16*)(&nan_si); // NaN
  unsigned short inf_si = 0x7c00; // mask for +inf
  _Float16 inf_f16 = *(_Float16*)(&inf_si); // +inf

  vbool2_t x_inf_mask = __riscv_vmseq_vx_u16m8_b2(x_int, 0x7c00, vl); //vmfeq x==inf bugs
  vbool2_t y_inf_mask = __riscv_vmseq_vx_u16m8_b2(y_int, 0x7c00, vl);
  vbool2_t x_nan_mask = __riscv_vmsgtu_vx_u16m8_b2(x_int, 0x7c00, vl);
  vbool2_t y_nan_mask = __riscv_vmsgtu_vx_u16m8_b2(y_int, 0x7c00, vl);

  vbool2_t special_mask = __riscv_vmor_mm_b2(x_inf_mask, y_inf_mask, vl);
  special_mask = __riscv_vmor_mm_b2(special_mask, x_nan_mask, vl);
  special_mask = __riscv_vmor_mm_b2(special_mask, y_nan_mask, vl);
  x_int = __riscv_vmerge_vxm_u16m8(x_int, 1, special_mask, vl);
  y_int = __riscv_vmerge_vxm_u16m8(y_int, 1, special_mask, vl);
#endif

  vuint16m8_t u_max = __riscv_vmaxu_vv_u16m8(x_int, y_int, vl);
  vuint16m8_t u_min = __riscv_vminu_vv_u16m8(x_int, y_int, vl);

  vuint16m8_t diff = __riscv_vsub_vv_u16m8(u_max, u_min, vl);
  vbool2_t cutoff_mask = __riscv_vmsgeu_vx_u16m8_b2(diff, 0x3000, vl);
  u_min = __riscv_vmerge_vxm_u16m8(u_min, 0, cutoff_mask, vl);

  vuint16m8_t E_bits = __riscv_vand_vx_u16m8(u_max, 0x7c00, vl);
  vuint16m8_t M_bit = __riscv_vand_vx_u16m8(u_max, 0x4000, vl); 
  vuint16m8_t base = __riscv_vsub_vv_u16m8(E_bits, __riscv_vsrl_vx_u16m8(M_bit, 3, vl), vl);
  
  // a-subnormal => M=0 => e_in = 29 => sa in [2^-10, 0.5)
  // a < 2  => M = 0 => e_in = 29 - e => sa in [0.5, 1) 
  // a >= 2 => M = 1 => e_in = 31 - e => sa in [2, 4) 
  vuint16m8_t e_out = __riscv_vadd_vx_u16m8(base, 0x0400, vl);
  vuint16m8_t e_in = __riscv_vrsub_vx_u16m8(base, 0x7400, vl);

  vfloat16m8_t scale_in  = __riscv_vreinterpret_v_u16m8_f16m8(e_in);
  vfloat16m8_t scale_out = __riscv_vreinterpret_v_u16m8_f16m8(e_out);

  vfloat16m8_t sa = __riscv_vfmul_vv_f16m8(__riscv_vreinterpret_v_u16m8_f16m8(u_max), scale_in, vl);
  vfloat16m8_t sb = __riscv_vfmul_vv_f16m8(__riscv_vreinterpret_v_u16m8_f16m8(u_min), scale_in, vl);

  vfloat16m8_t sb2 = __riscv_vfmul_vv_f16m8(sb, sb, vl);
  vfloat16m8_t sum2 = __riscv_vfmadd_vv_f16m8(sa, sa, sb2, vl);
  
  vfloat16m8_t r = __riscv_vsqrt_f16m8(sum2, vl);  
  vfloat16m8_t sqrt_value = __riscv_vfmul_vv_f16m8(r, scale_out, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m8(sqrt_value, nan_si, x_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m8(sqrt_value, nan_si, y_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m8(sqrt_value, inf_si, x_inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m8(sqrt_value, inf_si, y_inf_mask, vl);
#endif

  return sqrt_value;
}

vfloat32m1_t __riscv_vhypot_f32m1(vfloat32m1_t x, vfloat32m1_t y, size_t vl)
{
  vuint32m1_t x_int = 
    __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(x), 0x7fffffff, vl);
  vuint32m1_t y_int = 
    __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(y), 0x7fffffff, vl);

#ifndef __FAST_MATH__
  unsigned int nan_ui = 0x7fc00000; // mask for NaN
  float nan_f = *(float*)(&nan_ui); // NaN
  unsigned int inf_ui = 0x7f800000; // mask for +inf
  float inf_f = *(float*)(&inf_ui); // +inf

  vbool32_t x_inf_mask = __riscv_vmseq_vx_u32m1_b32(x_int, 0x7f800000, vl);
  vbool32_t y_inf_mask = __riscv_vmseq_vx_u32m1_b32(y_int, 0x7f800000, vl);
  vbool32_t x_nan_mask = __riscv_vmsgtu_vx_u32m1_b32(x_int, 0x7f800000, vl);
  vbool32_t y_nan_mask = __riscv_vmsgtu_vx_u32m1_b32(y_int, 0x7f800000, vl);

  vbool32_t special_mask = __riscv_vmor_mm_b32(x_inf_mask, y_inf_mask, vl);
  special_mask = __riscv_vmor_mm_b32(special_mask, x_nan_mask, vl);
  special_mask = __riscv_vmor_mm_b32(special_mask, y_nan_mask, vl);
  x_int = __riscv_vmerge_vxm_u32m1(x_int, 1, special_mask, vl);
  y_int = __riscv_vmerge_vxm_u32m1(y_int, 1, special_mask, vl);
#endif

  vuint32m1_t u_max = __riscv_vmaxu_vv_u32m1(x_int, y_int, vl);
  vuint32m1_t u_min = __riscv_vminu_vv_u32m1(x_int, y_int, vl);

  vuint32m1_t diff = __riscv_vsub_vv_u32m1(u_max, u_min, vl);
  vbool32_t cutoff_mask = __riscv_vmsgeu_vx_u32m1_b32(diff, 0x0d000000, vl);
  u_min = __riscv_vmerge_vxm_u32m1(u_min, 0, cutoff_mask, vl);

  vuint32m1_t E_bits = __riscv_vand_vx_u32m1(u_max, 0x7f800000, vl);
  vuint32m1_t M_bit = __riscv_vand_vx_u32m1(u_max, 0x40000000, vl); 
  vuint32m1_t base = __riscv_vsub_vv_u32m1(E_bits, __riscv_vsrl_vx_u32m1(M_bit, 6, vl), vl);
  
  // a-subnormal => M=0 => e_in = 253 => sa in [2^-23, 0.5)
  // a < 2  => M = 0 => e_in = 253 - e => sa in [0.5, 1) 
  // a >= 2 => M = 1 => e_in = 255 - e => sa in [2, 4) 
  vuint32m1_t e_out = __riscv_vadd_vx_u32m1(base, 0x00800000, vl);
  vuint32m1_t e_in = __riscv_vrsub_vx_u32m1(base, 0x7e800000, vl);

  vfloat32m1_t scale_in  = __riscv_vreinterpret_v_u32m1_f32m1(e_in);
  vfloat32m1_t scale_out = __riscv_vreinterpret_v_u32m1_f32m1(e_out);

  vfloat32m1_t sa = __riscv_vfmul_vv_f32m1(__riscv_vreinterpret_v_u32m1_f32m1(u_max), scale_in, vl);
  vfloat32m1_t sb = __riscv_vfmul_vv_f32m1(__riscv_vreinterpret_v_u32m1_f32m1(u_min), scale_in, vl);

  vfloat32m1_t sb2 = __riscv_vfmul_vv_f32m1(sb, sb, vl);
  vfloat32m1_t sum2 = __riscv_vfmadd_vv_f32m1(sa, sa, sb2, vl);
    
  vfloat32m1_t r = __riscv_vsqrt_f32m1(sum2, vl);  
  vfloat32m1_t sqrt_value = __riscv_vfmul_vv_f32m1(r, scale_out, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m1(sqrt_value, nan_f, x_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m1(sqrt_value, nan_f, y_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m1(sqrt_value, inf_f, x_inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m1(sqrt_value, inf_f, y_inf_mask, vl);
#endif

  return sqrt_value;
}

vfloat32m2_t __riscv_vhypot_f32m2(vfloat32m2_t x, vfloat32m2_t y, size_t vl)
{
  vuint32m2_t x_int = 
    __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(x), 0x7fffffff, vl);
  vuint32m2_t y_int = 
    __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(y), 0x7fffffff, vl);

#ifndef __FAST_MATH__
  unsigned int nan_ui = 0x7fc00000; // mask for NaN
  float nan_f = *(float*)(&nan_ui); // NaN
  unsigned int inf_ui = 0x7f800000; // mask for +inf
  float inf_f = *(float*)(&inf_ui); // +inf

  vbool16_t x_inf_mask = __riscv_vmseq_vx_u32m2_b16(x_int, 0x7f800000, vl);
  vbool16_t y_inf_mask = __riscv_vmseq_vx_u32m2_b16(y_int, 0x7f800000, vl);
  vbool16_t x_nan_mask = __riscv_vmsgtu_vx_u32m2_b16(x_int, 0x7f800000, vl);
  vbool16_t y_nan_mask = __riscv_vmsgtu_vx_u32m2_b16(y_int, 0x7f800000, vl);

  vbool16_t special_mask = __riscv_vmor_mm_b16(x_inf_mask, y_inf_mask, vl);
  special_mask = __riscv_vmor_mm_b16(special_mask, x_nan_mask, vl);
  special_mask = __riscv_vmor_mm_b16(special_mask, y_nan_mask, vl);
  x_int = __riscv_vmerge_vxm_u32m2(x_int, 1, special_mask, vl);
  y_int = __riscv_vmerge_vxm_u32m2(y_int, 1, special_mask, vl);
#endif

  vuint32m2_t u_max = __riscv_vmaxu_vv_u32m2(x_int, y_int, vl);
  vuint32m2_t u_min = __riscv_vminu_vv_u32m2(x_int, y_int, vl);

  vuint32m2_t diff = __riscv_vsub_vv_u32m2(u_max, u_min, vl);
  vbool16_t cutoff_mask = __riscv_vmsgeu_vx_u32m2_b16(diff, 0x0d000000, vl);
  u_min = __riscv_vmerge_vxm_u32m2(u_min, 0, cutoff_mask, vl);

  vuint32m2_t E_bits = __riscv_vand_vx_u32m2(u_max, 0x7f800000, vl);
  vuint32m2_t M_bit = __riscv_vand_vx_u32m2(u_max, 0x40000000, vl); 
  vuint32m2_t base = __riscv_vsub_vv_u32m2(E_bits, __riscv_vsrl_vx_u32m2(M_bit, 6, vl), vl);
  
  // a-subnormal => M=0 => e_in = 253 => sa in [2^-23, 0.5)
  // a < 2  => M = 0 => e_in = 253 - e => sa in [0.5, 1) 
  // a >= 2 => M = 1 => e_in = 255 - e => sa in [2, 4) 
  vuint32m2_t e_out = __riscv_vadd_vx_u32m2(base, 0x00800000, vl);
  vuint32m2_t e_in = __riscv_vrsub_vx_u32m2(base, 0x7e800000, vl);

  vfloat32m2_t scale_in  = __riscv_vreinterpret_v_u32m2_f32m2(e_in);
  vfloat32m2_t scale_out = __riscv_vreinterpret_v_u32m2_f32m2(e_out);

  vfloat32m2_t sa = __riscv_vfmul_vv_f32m2(__riscv_vreinterpret_v_u32m2_f32m2(u_max), scale_in, vl);
  vfloat32m2_t sb = __riscv_vfmul_vv_f32m2(__riscv_vreinterpret_v_u32m2_f32m2(u_min), scale_in, vl);

  vfloat32m2_t sb2 = __riscv_vfmul_vv_f32m2(sb, sb, vl);
  vfloat32m2_t sum2 = __riscv_vfmadd_vv_f32m2(sa, sa, sb2, vl);
  
  vfloat32m2_t r = __riscv_vsqrt_f32m2(sum2, vl);  
  vfloat32m2_t sqrt_value = __riscv_vfmul_vv_f32m2(r, scale_out, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m2(sqrt_value, nan_f, x_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m2(sqrt_value, nan_f, y_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m2(sqrt_value, inf_f, x_inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m2(sqrt_value, inf_f, y_inf_mask, vl);
#endif

  return sqrt_value;
}

vfloat32m4_t __riscv_vhypot_f32m4(vfloat32m4_t x, vfloat32m4_t y, size_t vl)
{
  vuint32m4_t x_int = 
    __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(x), 0x7fffffff, vl);
  vuint32m4_t y_int = 
    __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(y), 0x7fffffff, vl);

#ifndef __FAST_MATH__
  unsigned int nan_ui = 0x7fc00000; // mask for NaN
  float nan_f = *(float*)(&nan_ui); // NaN
  unsigned int inf_ui = 0x7f800000; // mask for +inf
  float inf_f = *(float*)(&inf_ui); // +inf

  vbool8_t x_inf_mask = __riscv_vmseq_vx_u32m4_b8(x_int, 0x7f800000, vl);
  vbool8_t y_inf_mask = __riscv_vmseq_vx_u32m4_b8(y_int, 0x7f800000, vl);
  vbool8_t x_nan_mask = __riscv_vmsgtu_vx_u32m4_b8(x_int, 0x7f800000, vl);
  vbool8_t y_nan_mask = __riscv_vmsgtu_vx_u32m4_b8(y_int, 0x7f800000, vl);

  vbool8_t special_mask = __riscv_vmor_mm_b8(x_inf_mask, y_inf_mask, vl);
  special_mask = __riscv_vmor_mm_b8(special_mask, x_nan_mask, vl);
  special_mask = __riscv_vmor_mm_b8(special_mask, y_nan_mask, vl);
  x_int = __riscv_vmerge_vxm_u32m4(x_int, 1, special_mask, vl);
  y_int = __riscv_vmerge_vxm_u32m4(y_int, 1, special_mask, vl);
#endif

  vuint32m4_t u_max = __riscv_vmaxu_vv_u32m4(x_int, y_int, vl);
  vuint32m4_t u_min = __riscv_vminu_vv_u32m4(x_int, y_int, vl);

  vuint32m4_t diff = __riscv_vsub_vv_u32m4(u_max, u_min, vl);
  vbool8_t cutoff_mask = __riscv_vmsgeu_vx_u32m4_b8(diff, 0x0d000000, vl);
  u_min = __riscv_vmerge_vxm_u32m4(u_min, 0, cutoff_mask, vl);

  vuint32m4_t E_bits = __riscv_vand_vx_u32m4(u_max, 0x7f800000, vl);
  vuint32m4_t M_bit = __riscv_vand_vx_u32m4(u_max, 0x40000000, vl); 
  vuint32m4_t base = __riscv_vsub_vv_u32m4(E_bits, __riscv_vsrl_vx_u32m4(M_bit, 6, vl), vl);
  
  // a-subnormal => M=0 => e_in = 253 => sa in [2^-23, 0.5)
  // a < 2  => M = 0 => e_in = 253 - e => sa in [0.5, 1) 
  // a >= 2 => M = 1 => e_in = 255 - e => sa in [2, 4) 
  vuint32m4_t e_out = __riscv_vadd_vx_u32m4(base, 0x00800000, vl);
  vuint32m4_t e_in = __riscv_vrsub_vx_u32m4(base, 0x7e800000, vl);

  vfloat32m4_t scale_in  = __riscv_vreinterpret_v_u32m4_f32m4(e_in);
  vfloat32m4_t scale_out = __riscv_vreinterpret_v_u32m4_f32m4(e_out);

  vfloat32m4_t sa = __riscv_vfmul_vv_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(u_max), scale_in, vl);
  vfloat32m4_t sb = __riscv_vfmul_vv_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(u_min), scale_in, vl);

  vfloat32m4_t sb2 = __riscv_vfmul_vv_f32m4(sb, sb, vl);
  vfloat32m4_t sum2 = __riscv_vfmadd_vv_f32m4(sa, sa, sb2, vl);
  
  vfloat32m4_t r = __riscv_vsqrt_f32m4(sum2, vl);  
  vfloat32m4_t sqrt_value = __riscv_vfmul_vv_f32m4(r, scale_out, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m4(sqrt_value, nan_f, x_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m4(sqrt_value, nan_f, y_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m4(sqrt_value, inf_f, x_inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m4(sqrt_value, inf_f, y_inf_mask, vl);
#endif

  return sqrt_value;
}

vfloat32m8_t __riscv_vhypot_f32m8(vfloat32m8_t x, vfloat32m8_t y, size_t vl)
{
  vuint32m8_t x_int = 
    __riscv_vand_vx_u32m8(__riscv_vreinterpret_v_f32m8_u32m8(x), 0x7fffffff, vl);
  vuint32m8_t y_int = 
    __riscv_vand_vx_u32m8(__riscv_vreinterpret_v_f32m8_u32m8(y), 0x7fffffff, vl);

#ifndef __FAST_MATH__
  unsigned int nan_ui = 0x7fc00000; // mask for NaN
  float nan_f = *(float*)(&nan_ui); // NaN
  unsigned int inf_ui = 0x7f800000; // mask for +inf
  float inf_f = *(float*)(&inf_ui); // +inf

  vbool4_t x_inf_mask = __riscv_vmseq_vx_u32m8_b4(x_int, 0x7f800000, vl);
  vbool4_t y_inf_mask = __riscv_vmseq_vx_u32m8_b4(y_int, 0x7f800000, vl);
  vbool4_t x_nan_mask = __riscv_vmsgtu_vx_u32m8_b4(x_int, 0x7f800000, vl);
  vbool4_t y_nan_mask = __riscv_vmsgtu_vx_u32m8_b4(y_int, 0x7f800000, vl);

  vbool4_t special_mask = __riscv_vmor_mm_b4(x_inf_mask, y_inf_mask, vl);
  special_mask = __riscv_vmor_mm_b4(special_mask, x_nan_mask, vl);
  special_mask = __riscv_vmor_mm_b4(special_mask, y_nan_mask, vl);
  x_int = __riscv_vmerge_vxm_u32m8(x_int, 1, special_mask, vl);
  y_int = __riscv_vmerge_vxm_u32m8(y_int, 1, special_mask, vl);
#endif

  vuint32m8_t u_max = __riscv_vmaxu_vv_u32m8(x_int, y_int, vl);
  vuint32m8_t u_min = __riscv_vminu_vv_u32m8(x_int, y_int, vl);

  vuint32m8_t diff = __riscv_vsub_vv_u32m8(u_max, u_min, vl);
  vbool4_t cutoff_mask = __riscv_vmsgeu_vx_u32m8_b4(diff, 0x0d000000, vl);
  u_min = __riscv_vmerge_vxm_u32m8(u_min, 0, cutoff_mask, vl);

  vuint32m8_t E_bits = __riscv_vand_vx_u32m8(u_max, 0x7f800000, vl);
  vuint32m8_t M_bit = __riscv_vand_vx_u32m8(u_max, 0x40000000, vl); 
  vuint32m8_t base = __riscv_vsub_vv_u32m8(E_bits, __riscv_vsrl_vx_u32m8(M_bit, 6, vl), vl);
  
  // a-subnormal => M=0 => e_in = 253 => sa in [2^-23, 0.5)
  // a < 2  => M = 0 => e_in = 253 - e => sa in [0.5, 1) 
  // a >= 2 => M = 1 => e_in = 254 - e => sa in [2, 4) 
  vuint32m8_t e_out = __riscv_vadd_vx_u32m8(base, 0x00800000, vl);
  vuint32m8_t e_in = __riscv_vrsub_vx_u32m8(base, 0x7e800000, vl);

  vfloat32m8_t scale_in  = __riscv_vreinterpret_v_u32m8_f32m8(e_in);
  vfloat32m8_t scale_out = __riscv_vreinterpret_v_u32m8_f32m8(e_out);

  vfloat32m8_t sa = __riscv_vfmul_vv_f32m8(__riscv_vreinterpret_v_u32m8_f32m8(u_max), scale_in, vl);
  vfloat32m8_t sb = __riscv_vfmul_vv_f32m8(__riscv_vreinterpret_v_u32m8_f32m8(u_min), scale_in, vl);

  vfloat32m8_t sb2 = __riscv_vfmul_vv_f32m8(sb, sb, vl);
  vfloat32m8_t sum2 = __riscv_vfmadd_vv_f32m8(sa, sa, sb2, vl);
  
  vfloat32m8_t r = __riscv_vsqrt_f32m8(sum2, vl);  
  vfloat32m8_t sqrt_value = __riscv_vfmul_vv_f32m8(r, scale_out, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m8(sqrt_value, nan_f, x_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m8(sqrt_value, nan_f, y_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m8(sqrt_value, inf_f, x_inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m8(sqrt_value, inf_f, y_inf_mask, vl);
#endif

  return sqrt_value;
}

vfloat64m1_t __riscv_vhypot_f64m1(vfloat64m1_t x, vfloat64m1_t y, size_t vl)
{
  vuint64m1_t x_int = __riscv_vand_vx_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(x), 0x7fffffffffffffff, vl);
  vuint64m1_t y_int = __riscv_vand_vx_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(y), 0x7fffffffffffffff, vl);

#ifndef __FAST_MATH__
  unsigned long long nan_ull = 0x7ff8000000000000;
  double nan_d = *(double*)(&nan_ull);
  unsigned long long inf_ull = 0x7ff0000000000000;
  double inf_d = *(double*)(&inf_ull);

  vbool64_t x_inf_mask = __riscv_vmseq_vx_u64m1_b64(x_int, 0x7ff0000000000000, vl);
  vbool64_t y_inf_mask = __riscv_vmseq_vx_u64m1_b64(y_int, 0x7ff0000000000000, vl);
  vbool64_t x_nan_mask = __riscv_vmsgtu_vx_u64m1_b64(x_int, 0x7ff0000000000000, vl);
  vbool64_t y_nan_mask = __riscv_vmsgtu_vx_u64m1_b64(y_int, 0x7ff0000000000000, vl);

  vbool64_t special_mask = __riscv_vmor_mm_b64(x_inf_mask, y_inf_mask, vl);
  special_mask = __riscv_vmor_mm_b64(special_mask, x_nan_mask, vl);
  special_mask = __riscv_vmor_mm_b64(special_mask, y_nan_mask, vl);
  
  x_int = __riscv_vmerge_vxm_u64m1(x_int, 1, special_mask, vl);
  y_int = __riscv_vmerge_vxm_u64m1(y_int, 1, special_mask, vl);
#endif

  vuint64m1_t u_max = __riscv_vmaxu_vv_u64m1(x_int, y_int, vl);
  vuint64m1_t u_min = __riscv_vminu_vv_u64m1(x_int, y_int, vl);

  vuint64m1_t diff = __riscv_vsub_vv_u64m1(u_max, u_min, vl);
  vbool64_t cutoff_mask = __riscv_vmsgeu_vx_u64m1_b64(diff, 0x0360000000000000, vl);
  u_min = __riscv_vmerge_vxm_u64m1(u_min, 0, cutoff_mask, vl);

  vuint64m1_t E_bits = __riscv_vand_vx_u64m1(u_max, 0x7ff0000000000000, vl);
  vuint64m1_t M_bit = __riscv_vand_vx_u64m1(u_max, 0x4000000000000000, vl); 
  vuint64m1_t base = __riscv_vsub_vv_u64m1(E_bits, __riscv_vsrl_vx_u64m1(M_bit, 9, vl), vl);
  
  // a-subnormal => M=0 => e_in = 2045 => sa in [2^-52, 0.5)
  // a < 2  => M = 0 => e_in = 2045 - e => sa in [0.5, 1) 
  // a >= 2 => M = 1 => e_in = 2047 - e => sa in [2, 4) 
  vuint64m1_t e_out = __riscv_vadd_vx_u64m1(base, 0x0010000000000000, vl);
  vuint64m1_t e_in = __riscv_vrsub_vx_u64m1(base, 0x7fd0000000000000, vl);

  vfloat64m1_t scale_in  = __riscv_vreinterpret_v_u64m1_f64m1(e_in);
  vfloat64m1_t scale_out = __riscv_vreinterpret_v_u64m1_f64m1(e_out);

  vfloat64m1_t sa = __riscv_vfmul_vv_f64m1(__riscv_vreinterpret_v_u64m1_f64m1(u_max), scale_in, vl);
  vfloat64m1_t sb = __riscv_vfmul_vv_f64m1(__riscv_vreinterpret_v_u64m1_f64m1(u_min), scale_in, vl);

  vfloat64m1_t sb2 = __riscv_vfmul_vv_f64m1(sb, sb, vl);
  vfloat64m1_t sum2 = __riscv_vfmadd_vv_f64m1(sa, sa, sb2, vl);
  
  vfloat64m1_t r = __riscv_vsqrt_f64m1(sum2, vl);  
  vfloat64m1_t sqrt_value = __riscv_vfmul_vv_f64m1(r, scale_out, vl);

#ifndef __FAST_MATH__
  sqrt_value = __riscv_vfmerge_vfm_f64m1(sqrt_value, nan_d, x_nan_mask, vl);
  sqrt_value = __riscv_vfmerge_vfm_f64m1(sqrt_value, nan_d, y_nan_mask, vl);
  sqrt_value = __riscv_vfmerge_vfm_f64m1(sqrt_value, inf_d, x_inf_mask, vl);
  sqrt_value = __riscv_vfmerge_vfm_f64m1(sqrt_value, inf_d, y_inf_mask, vl);
#endif

  return sqrt_value;
}

vfloat64m2_t __riscv_vhypot_f64m2(vfloat64m2_t x, vfloat64m2_t y, size_t vl)
{
  vuint64m2_t x_int = 
    __riscv_vand_vx_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(x), 0x7fffffffffffffff, vl);
  vuint64m2_t y_int = 
    __riscv_vand_vx_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(y), 0x7fffffffffffffff, vl);

#ifndef __FAST_MATH__
  unsigned long long nan_ull = 0x7ff8000000000000; // mask for NaN
  double nan_d = *(double*)(&nan_ull); // NaN
  unsigned long long inf_ull = 0x7ff0000000000000; // mask for +inf
  double inf_d = *(double*)(&inf_ull); // +inf

  vbool32_t x_inf_mask = __riscv_vmseq_vx_u64m2_b32(x_int, 0x7ff0000000000000, vl);
  vbool32_t y_inf_mask = __riscv_vmseq_vx_u64m2_b32(y_int, 0x7ff0000000000000, vl);
  vbool32_t x_nan_mask = __riscv_vmsgtu_vx_u64m2_b32(x_int, 0x7ff0000000000000, vl);
  vbool32_t y_nan_mask = __riscv_vmsgtu_vx_u64m2_b32(y_int, 0x7ff0000000000000, vl);

  vbool32_t special_mask = __riscv_vmor_mm_b32(x_inf_mask, y_inf_mask, vl);
  special_mask = __riscv_vmor_mm_b32(special_mask, x_nan_mask, vl);
  special_mask = __riscv_vmor_mm_b32(special_mask, y_nan_mask, vl);
  x_int = __riscv_vmerge_vxm_u64m2(x_int, 1, special_mask, vl);
  y_int = __riscv_vmerge_vxm_u64m2(y_int, 1, special_mask, vl);
#endif

  vuint64m2_t u_max = __riscv_vmaxu_vv_u64m2(x_int, y_int, vl);
  vuint64m2_t u_min = __riscv_vminu_vv_u64m2(x_int, y_int, vl);

  vuint64m2_t diff = __riscv_vsub_vv_u64m2(u_max, u_min, vl);
  vbool32_t cutoff_mask = __riscv_vmsgeu_vx_u64m2_b32(diff, 0x0360000000000000, vl);
  u_min = __riscv_vmerge_vxm_u64m2(u_min, 0, cutoff_mask, vl);

  vuint64m2_t E_bits = __riscv_vand_vx_u64m2(u_max, 0x7ff0000000000000, vl);
  vuint64m2_t M_bit = __riscv_vand_vx_u64m2(u_max, 0x4000000000000000, vl); 
  vuint64m2_t base = __riscv_vsub_vv_u64m2(E_bits, __riscv_vsrl_vx_u64m2(M_bit, 9, vl), vl);
  
  // a-subnormal => M=0 => e_in = 2045 => sa in [2^-52, 0.5)
  // a < 2  => M = 0 => e_in = 2045 - e => sa in [0.5, 1) 
  // a >= 2 => M = 1 => e_in = 2047 - e => sa in [2, 4) 
  vuint64m2_t e_out = __riscv_vadd_vx_u64m2(base, 0x0010000000000000, vl);
  vuint64m2_t e_in = __riscv_vrsub_vx_u64m2(base, 0x7fd0000000000000, vl);

  vfloat64m2_t scale_in  = __riscv_vreinterpret_v_u64m2_f64m2(e_in);
  vfloat64m2_t scale_out = __riscv_vreinterpret_v_u64m2_f64m2(e_out);

  vfloat64m2_t sa = __riscv_vfmul_vv_f64m2(__riscv_vreinterpret_v_u64m2_f64m2(u_max), scale_in, vl);
  vfloat64m2_t sb = __riscv_vfmul_vv_f64m2(__riscv_vreinterpret_v_u64m2_f64m2(u_min), scale_in, vl);

  vfloat64m2_t sb2 = __riscv_vfmul_vv_f64m2(sb, sb, vl);
  vfloat64m2_t sum2 = __riscv_vfmadd_vv_f64m2(sa, sa, sb2, vl);
  
  vfloat64m2_t r = __riscv_vsqrt_f64m2(sum2, vl);  
  vfloat64m2_t sqrt_value = __riscv_vfmul_vv_f64m2(r, scale_out, vl);
  
#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m2(sqrt_value, nan_d, x_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m2(sqrt_value, nan_d, y_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m2(sqrt_value, inf_d, x_inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m2(sqrt_value, inf_d, y_inf_mask, vl);
#endif

  return sqrt_value;
}

vfloat64m4_t __riscv_vhypot_f64m4(vfloat64m4_t x, vfloat64m4_t y, size_t vl)
{
  vuint64m4_t x_int = 
    __riscv_vand_vx_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(x), 0x7fffffffffffffff, vl);
  vuint64m4_t y_int = 
    __riscv_vand_vx_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(y), 0x7fffffffffffffff, vl);

#ifndef __FAST_MATH__
  unsigned long long nan_ull = 0x7ff8000000000000; // mask for NaN
  double nan_d = *(double*)(&nan_ull); // NaN
  unsigned long long inf_ull = 0x7ff0000000000000; // mask for +inf
  double inf_d = *(double*)(&inf_ull); // +inf

  vbool16_t x_inf_mask = __riscv_vmseq_vx_u64m4_b16(x_int, 0x7ff0000000000000, vl);
  vbool16_t y_inf_mask = __riscv_vmseq_vx_u64m4_b16(y_int, 0x7ff0000000000000, vl);
  vbool16_t x_nan_mask = __riscv_vmsgtu_vx_u64m4_b16(x_int, 0x7ff0000000000000, vl);
  vbool16_t y_nan_mask = __riscv_vmsgtu_vx_u64m4_b16(y_int, 0x7ff0000000000000, vl);

  vbool16_t special_mask = __riscv_vmor_mm_b16(x_inf_mask, y_inf_mask, vl);
  special_mask = __riscv_vmor_mm_b16(special_mask, x_nan_mask, vl);
  special_mask = __riscv_vmor_mm_b16(special_mask, y_nan_mask, vl);
  x_int = __riscv_vmerge_vxm_u64m4(x_int, 1, special_mask, vl);
  y_int = __riscv_vmerge_vxm_u64m4(y_int, 1, special_mask, vl);
#endif

  vuint64m4_t u_max = __riscv_vmaxu_vv_u64m4(x_int, y_int, vl);
  vuint64m4_t u_min = __riscv_vminu_vv_u64m4(x_int, y_int, vl);

  vuint64m4_t diff = __riscv_vsub_vv_u64m4(u_max, u_min, vl);
  vbool16_t cutoff_mask = __riscv_vmsgeu_vx_u64m4_b16(diff, 0x0360000000000000, vl);
  u_min = __riscv_vmerge_vxm_u64m4(u_min, 0, cutoff_mask, vl);

  vuint64m4_t E_bits = __riscv_vand_vx_u64m4(u_max, 0x7ff0000000000000, vl);
  vuint64m4_t M_bit = __riscv_vand_vx_u64m4(u_max, 0x4000000000000000, vl); 
  vuint64m4_t base = __riscv_vsub_vv_u64m4(E_bits, __riscv_vsrl_vx_u64m4(M_bit, 9, vl), vl);
  
  // a-subnormal => M=0 => e_in = 2045 => sa in [2^-52, 0.5)
  // a < 2  => M = 0 => e_in = 2045 - e => sa in [0.5, 1) 
  // a >= 2 => M = 1 => e_in = 2047 - e => sa in [2, 4) 
  vuint64m4_t e_out = __riscv_vadd_vx_u64m4(base, 0x0010000000000000, vl);
  vuint64m4_t e_in = __riscv_vrsub_vx_u64m4(base, 0x7fd0000000000000, vl);

  vfloat64m4_t scale_in  = __riscv_vreinterpret_v_u64m4_f64m4(e_in);
  vfloat64m4_t scale_out = __riscv_vreinterpret_v_u64m4_f64m4(e_out);

  vfloat64m4_t sa = __riscv_vfmul_vv_f64m4(__riscv_vreinterpret_v_u64m4_f64m4(u_max), scale_in, vl);
  vfloat64m4_t sb = __riscv_vfmul_vv_f64m4(__riscv_vreinterpret_v_u64m4_f64m4(u_min), scale_in, vl);

  vfloat64m4_t sb2 = __riscv_vfmul_vv_f64m4(sb, sb, vl);
  vfloat64m4_t sum2 = __riscv_vfmadd_vv_f64m4(sa, sa, sb2, vl);
  
  vfloat64m4_t r = __riscv_vsqrt_f64m4(sum2, vl);  
  vfloat64m4_t sqrt_value = __riscv_vfmul_vv_f64m4(r, scale_out, vl);
  
#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m4(sqrt_value, nan_d, x_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m4(sqrt_value, nan_d, y_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m4(sqrt_value, inf_d, x_inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m4(sqrt_value, inf_d, y_inf_mask, vl);
#endif

  return sqrt_value;
}

vfloat64m8_t __riscv_vhypot_f64m8(vfloat64m8_t x, vfloat64m8_t y, size_t vl)
{
  vuint64m8_t x_int = 
    __riscv_vand_vx_u64m8(__riscv_vreinterpret_v_f64m8_u64m8(x), 0x7fffffffffffffff, vl);
  vuint64m8_t y_int = 
    __riscv_vand_vx_u64m8(__riscv_vreinterpret_v_f64m8_u64m8(y), 0x7fffffffffffffff, vl);

#ifndef __FAST_MATH__
  unsigned long long nan_ull = 0x7ff8000000000000; // mask for NaN
  double nan_d = *(double*)(&nan_ull); // NaN
  unsigned long long inf_ull = 0x7ff0000000000000; // mask for +inf
  double inf_d = *(double*)(&inf_ull); // +inf

  vbool8_t x_inf_mask = __riscv_vmseq_vx_u64m8_b8(x_int, 0x7ff0000000000000, vl);
  vbool8_t y_inf_mask = __riscv_vmseq_vx_u64m8_b8(y_int, 0x7ff0000000000000, vl);
  vbool8_t x_nan_mask = __riscv_vmsgtu_vx_u64m8_b8(x_int, 0x7ff0000000000000, vl);
  vbool8_t y_nan_mask = __riscv_vmsgtu_vx_u64m8_b8(y_int, 0x7ff0000000000000, vl);

  vbool8_t special_mask = __riscv_vmor_mm_b8(x_inf_mask, y_inf_mask, vl);
  special_mask = __riscv_vmor_mm_b8(special_mask, x_nan_mask, vl);
  special_mask = __riscv_vmor_mm_b8(special_mask, y_nan_mask, vl);
  x_int = __riscv_vmerge_vxm_u64m8(x_int, 1, special_mask, vl);
  y_int = __riscv_vmerge_vxm_u64m8(y_int, 1, special_mask, vl);
#endif
  
  vuint64m8_t u_max = __riscv_vmaxu_vv_u64m8(x_int, y_int, vl);
  vuint64m8_t u_min = __riscv_vminu_vv_u64m8(x_int, y_int, vl);

  vuint64m8_t diff = __riscv_vsub_vv_u64m8(u_max, u_min, vl);
  vbool8_t cutoff_mask = __riscv_vmsgeu_vx_u64m8_b8(diff, 0x0360000000000000, vl);
  u_min = __riscv_vmerge_vxm_u64m8(u_min, 0, cutoff_mask, vl);

  vuint64m8_t E_bits = __riscv_vand_vx_u64m8(u_max, 0x7ff0000000000000, vl);
  vuint64m8_t M_bit = __riscv_vand_vx_u64m8(u_max, 0x4000000000000000, vl); 
  vuint64m8_t base = __riscv_vsub_vv_u64m8(E_bits, __riscv_vsrl_vx_u64m8(M_bit, 9, vl), vl);
  
  // a-subnormal => M=0 => e_in = 2045 => sa in [2^-52, 0.5)
  // a < 2  => M = 0 => e_in = 2045 - e => sa in [0.5, 1) 
  // a >= 2 => M = 1 => e_in = 2047 - e => sa in [2, 4) 
  vuint64m8_t e_out = __riscv_vadd_vx_u64m8(base, 0x0010000000000000, vl);
  vuint64m8_t e_in = __riscv_vrsub_vx_u64m8(base, 0x7fd0000000000000, vl);

  vfloat64m8_t scale_in  = __riscv_vreinterpret_v_u64m8_f64m8(e_in);
  vfloat64m8_t scale_out = __riscv_vreinterpret_v_u64m8_f64m8(e_out);

  vfloat64m8_t sa = __riscv_vfmul_vv_f64m8(__riscv_vreinterpret_v_u64m8_f64m8(u_max), scale_in, vl);
  vfloat64m8_t sb = __riscv_vfmul_vv_f64m8(__riscv_vreinterpret_v_u64m8_f64m8(u_min), scale_in, vl);

  vfloat64m8_t sb2 = __riscv_vfmul_vv_f64m8(sb, sb, vl);
  vfloat64m8_t sum2 = __riscv_vfmadd_vv_f64m8(sa, sa, sb2, vl);
  
  vfloat64m8_t r = __riscv_vsqrt_f64m8(sum2, vl);  
  vfloat64m8_t sqrt_value = __riscv_vfmul_vv_f64m8(r, scale_out, vl);
  
#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m8(sqrt_value, nan_d, x_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m8(sqrt_value, nan_d, y_nan_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m8(sqrt_value, inf_d, x_inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m8(sqrt_value, inf_d, y_inf_mask, vl);
#endif

  return sqrt_value;
}
#endif