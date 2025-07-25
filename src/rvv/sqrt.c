/* 
 *========================================================
 * Copyright (c) The Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025 The RVVMF Authors (Alexander Sysoyev)
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

vfloat16m1_t __riscv_vsqrt_f16m1(vfloat16m1_t x, size_t vl)
{
#ifndef __FAST_MATH__
  unsigned short nan_si = 0x7e00; // mask for NaN
  _Float16 nan_f16 = *(_Float16*)(&nan_si); // NaN
  unsigned short inf_si = 0x7c00; // mask for +inf
  _Float16 inf_f16 = *(_Float16*)(&inf_si); // +inf

  vbool16_t zero_less_mask = __riscv_vmflt_vf_f16m1_b16(x, 0.0f16, vl);
  vbool16_t inf_mask = __riscv_vmfeq_vf_f16m1_b16(x, inf_f16, vl);
  vuint16m1_t x_int = 
    __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(x), 0x7fff, vl);
  vbool16_t nan_mask = __riscv_vmsgtu_vx_u16m1_b16(x_int, 0x7e00, vl);

  vbool16_t special_mask = __riscv_vmor_mm_b16(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b16(special_mask, nan_mask, vl);
  vfloat16m1_t x_spec = __riscv_vfmerge_vfm_f16m1(x, 0, special_mask, vl);

  x_int = __riscv_vreinterpret_v_f16m1_u16m1(x_spec);
#else
  vuint16m1_t x_int = __riscv_vreinterpret_v_f16m1_u16m1(x);
#endif

  vuint16m1_t mantissa_in_x = __riscv_vand_vx_u16m1(x_int, 0x03ff, vl);
  vuint16m1_t reduced_x_int = __riscv_vor_vx_u16m1(mantissa_in_x, 0x3c00, vl);
  vfloat16m1_t reduced_x = __riscv_vreinterpret_v_u16m1_f16m1(reduced_x_int);

  vuint16m1_t order_in_x = __riscv_vand_vx_u16m1(x_int, 0x7c00, vl);
  order_in_x = __riscv_vsrl_vx_u16m1(order_in_x, 10, vl);
  vuint16m1_t high_ind = __riscv_vsrl_vx_u16m1(order_in_x, 4, vl);
  vuint16m1_t low_ind = __riscv_vand_vx_u16m1(order_in_x, 0xf, vl);

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
  r = __riscv_vfrsub_vf_f16m1(__riscv_vfmul_vv_f16m1(xx, h, vl), 0.5f16, vl);

  r = __riscv_vfmacc_vv_f16m1(reduced_x, __riscv_vfmul_vf_f16m1(xx, -1.f16, vl), xx, vl);

  vfloat16m1_t zh = __riscv_vfmacc_vv_f16m1(xx, r, h, vl); // high-part of sqrt value
  vfloat16m1_t sh, sl;
  sh = __riscv_vfsub_vv_f16m1(xx, zh, vl);
  sl = __riscv_vfadd_vv_f16m1(sh, zh, vl);
  sl = __riscv_vfsub_vv_f16m1(xx, sl, vl);
  vfloat16m1_t zl = __riscv_vfmacc_vv_f16m1(sh, r, h, vl);
  zl = __riscv_vfadd_vv_f16m1(zl, sl, vl);                 // low-part of sqrt value
  
  high_ind = __riscv_vmul_vx_u16m1(high_ind, 2, vl);
  low_ind = __riscv_vmul_vx_u16m1(low_ind, 2, vl);
  vfloat16m1_t order_high = __riscv_vloxei16_v_f16m1(order_tab_high_f16, high_ind, vl);
  vfloat16m1_t order_low = __riscv_vloxei16_v_f16m1(order_tab_low_f16, low_ind, vl);
  vfloat16m1_t order_low_ = __riscv_vloxei16_v_f16m1(order_tab_low_f16_, low_ind, vl);

  vfloat16m1_t zzh = __riscv_vfmul_vv_f16m1(zh, order_high, vl);
  vfloat16m1_t zzl = __riscv_vfmul_vv_f16m1(zl, order_high, vl);

  sh = __riscv_vfmul_vv_f16m1(zzh, order_low, vl);
  sl = __riscv_vfmacc_vv_f16m1(__riscv_vfmul_vf_f16m1(sh, -1.f16, vl), zzh, order_low, vl);
  vfloat16m1_t part1 = __riscv_vfmul_vv_f16m1(zzl, order_low, vl);
  vfloat16m1_t part2 = __riscv_vfmacc_vv_f16m1(part1, zzh, order_low_, vl);
  sl = __riscv_vfadd_vv_f16m1(sl, part2, vl);

  vfloat16m1_t sqrt_value = __riscv_vfadd_vv_f16m1(sh, sl, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m1(sqrt_value, nan_f16, zero_less_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m1(sqrt_value, inf_f16, inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m1(sqrt_value, nan_f16, nan_mask, vl);
#endif

  return sqrt_value; 
}

vfloat16m2_t __riscv_vsqrt_f16m2(vfloat16m2_t x, size_t vl)
{
#ifndef __FAST_MATH__
  unsigned short nan_si = 0x7e00; // mask for NaN
  _Float16 nan_f16 = *(_Float16*)(&nan_si); // NaN
  unsigned short inf_si = 0x7c00; // mask for +inf
  _Float16 inf_f16 = *(_Float16*)(&inf_si); // +inf

  vbool8_t zero_less_mask = __riscv_vmflt_vf_f16m2_b8(x, 0.0f16, vl);
  vbool8_t inf_mask = __riscv_vmfeq_vf_f16m2_b8(x, inf_f16, vl);
  vuint16m2_t x_int = 
    __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(x), 0x7fff, vl);
  vbool8_t nan_mask = __riscv_vmsgtu_vx_u16m2_b8(x_int, 0x7e00, vl);

  vbool8_t special_mask = __riscv_vmor_mm_b8(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b8(special_mask, nan_mask, vl);
  vfloat16m2_t x_spec = __riscv_vfmerge_vfm_f16m2(x, 0, special_mask, vl);

  x_int = __riscv_vreinterpret_v_f16m2_u16m2(x_spec);
#else
  vuint16m2_t x_int = __riscv_vreinterpret_v_f16m2_u16m2(x);
#endif

  vuint16m2_t mantissa_in_x = __riscv_vand_vx_u16m2(x_int, 0x03ff, vl);
  vuint16m2_t reduced_x_int = __riscv_vor_vx_u16m2(mantissa_in_x, 0x3c00, vl);
  vfloat16m2_t reduced_x = __riscv_vreinterpret_v_u16m2_f16m2(reduced_x_int);

  vuint16m2_t order_in_x = __riscv_vand_vx_u16m2(x_int, 0x7c00, vl);
  order_in_x = __riscv_vsrl_vx_u16m2(order_in_x, 10, vl);
  vuint16m2_t high_ind = __riscv_vsrl_vx_u16m2(order_in_x, 4, vl);
  vuint16m2_t low_ind = __riscv_vand_vx_u16m2(order_in_x, 0xf, vl);

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
  r = __riscv_vfrsub_vf_f16m2(__riscv_vfmul_vv_f16m2(xx, h, vl), 0.5f16, vl);

  r = __riscv_vfmacc_vv_f16m2(reduced_x, __riscv_vfmul_vf_f16m2(xx, -1.f16, vl), xx, vl);

  vfloat16m2_t zh = __riscv_vfmacc_vv_f16m2(xx, r, h, vl); // high-part of sqrt value
  vfloat16m2_t sh, sl;
  sh = __riscv_vfsub_vv_f16m2(xx, zh, vl);
  sl = __riscv_vfadd_vv_f16m2(sh, zh, vl);
  sl = __riscv_vfsub_vv_f16m2(xx, sl, vl);
  vfloat16m2_t zl = __riscv_vfmacc_vv_f16m2(sh, r, h, vl);
  zl = __riscv_vfadd_vv_f16m2(zl, sl, vl);                 // low-part of sqrt value
  
  high_ind = __riscv_vmul_vx_u16m2(high_ind, 2, vl);
  low_ind = __riscv_vmul_vx_u16m2(low_ind, 2, vl);
  vfloat16m2_t order_high = __riscv_vloxei16_v_f16m2(order_tab_high_f16, high_ind, vl);
  vfloat16m2_t order_low = __riscv_vloxei16_v_f16m2(order_tab_low_f16, low_ind, vl);
  vfloat16m2_t order_low_ = __riscv_vloxei16_v_f16m2(order_tab_low_f16_, low_ind, vl);

  vfloat16m2_t zzh = __riscv_vfmul_vv_f16m2(zh, order_high, vl);
  vfloat16m2_t zzl = __riscv_vfmul_vv_f16m2(zl, order_high, vl);

  sh = __riscv_vfmul_vv_f16m2(zzh, order_low, vl);
  sl = __riscv_vfmacc_vv_f16m2(__riscv_vfmul_vf_f16m2(sh, -1.f16, vl), zzh, order_low, vl);
  vfloat16m2_t part1 = __riscv_vfmul_vv_f16m2(zzl, order_low, vl);
  vfloat16m2_t part2 = __riscv_vfmacc_vv_f16m2(part1, zzh, order_low_, vl);
  sl = __riscv_vfadd_vv_f16m2(sl, part2, vl);

  vfloat16m2_t sqrt_value = __riscv_vfadd_vv_f16m2(sh, sl, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m2(sqrt_value, nan_f16, zero_less_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m2(sqrt_value, inf_f16, inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m2(sqrt_value, nan_f16, nan_mask, vl);
#endif

  return sqrt_value;
}

vfloat16m4_t __riscv_vsqrt_f16m4(vfloat16m4_t x, size_t vl)
{
#ifndef __FAST_MATH__
  unsigned short nan_si = 0x7e00; // mask for NaN
  _Float16 nan_f16 = *(_Float16*)(&nan_si); // NaN
  unsigned short inf_si = 0x7c00; // mask for +inf
  _Float16 inf_f16 = *(_Float16*)(&inf_si); // +inf

  vbool4_t zero_less_mask = __riscv_vmflt_vf_f16m4_b4(x, 0.0f16, vl);
  vbool4_t inf_mask = __riscv_vmfeq_vf_f16m4_b4(x, inf_f16, vl);
  vuint16m4_t x_int = 
    __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(x), 0x7fff, vl);
  vbool4_t nan_mask = __riscv_vmsgtu_vx_u16m4_b4(x_int, 0x7e00, vl);

  vbool4_t special_mask = __riscv_vmor_mm_b4(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b4(special_mask, nan_mask, vl);
  vfloat16m4_t x_spec = __riscv_vfmerge_vfm_f16m4(x, 0, special_mask, vl);

  x_int = __riscv_vreinterpret_v_f16m4_u16m4(x_spec);
#else
  vuint16m4_t x_int = __riscv_vreinterpret_v_f16m4_u16m4(x);
#endif

  vuint16m4_t mantissa_in_x = __riscv_vand_vx_u16m4(x_int, 0x03ff, vl);
  vuint16m4_t reduced_x_int = __riscv_vor_vx_u16m4(mantissa_in_x, 0x3c00, vl);
  vfloat16m4_t reduced_x = __riscv_vreinterpret_v_u16m4_f16m4(reduced_x_int);

  vuint16m4_t order_in_x = __riscv_vand_vx_u16m4(x_int, 0x7c00, vl);
  order_in_x = __riscv_vsrl_vx_u16m4(order_in_x, 10, vl);
  vuint16m4_t high_ind = __riscv_vsrl_vx_u16m4(order_in_x, 4, vl);
  vuint16m4_t low_ind = __riscv_vand_vx_u16m4(order_in_x, 0xf, vl);

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
  r = __riscv_vfrsub_vf_f16m4(__riscv_vfmul_vv_f16m4(xx, h, vl), 0.5f16, vl);

  r = __riscv_vfmacc_vv_f16m4(reduced_x, __riscv_vfmul_vf_f16m4(xx, -1.f16, vl), xx, vl);

  vfloat16m4_t zh = __riscv_vfmacc_vv_f16m4(xx, r, h, vl); // high-part of sqrt value
  vfloat16m4_t sh, sl;
  sh = __riscv_vfsub_vv_f16m4(xx, zh, vl);
  sl = __riscv_vfadd_vv_f16m4(sh, zh, vl);
  sl = __riscv_vfsub_vv_f16m4(xx, sl, vl);
  vfloat16m4_t zl = __riscv_vfmacc_vv_f16m4(sh, r, h, vl);
  zl = __riscv_vfadd_vv_f16m4(zl, sl, vl);                 // low-part of sqrt value
  
  high_ind = __riscv_vmul_vx_u16m4(high_ind, 2, vl);
  low_ind = __riscv_vmul_vx_u16m4(low_ind, 2, vl);
  vfloat16m4_t order_high = __riscv_vloxei16_v_f16m4(order_tab_high_f16, high_ind, vl);
  vfloat16m4_t order_low = __riscv_vloxei16_v_f16m4(order_tab_low_f16, low_ind, vl);
  vfloat16m4_t order_low_ = __riscv_vloxei16_v_f16m4(order_tab_low_f16_, low_ind, vl);

  vfloat16m4_t zzh = __riscv_vfmul_vv_f16m4(zh, order_high, vl);
  vfloat16m4_t zzl = __riscv_vfmul_vv_f16m4(zl, order_high, vl);

  sh = __riscv_vfmul_vv_f16m4(zzh, order_low, vl);
  sl = __riscv_vfmacc_vv_f16m4(__riscv_vfmul_vf_f16m4(sh, -1.f16, vl), zzh, order_low, vl);
  vfloat16m4_t part1 = __riscv_vfmul_vv_f16m4(zzl, order_low, vl);
  vfloat16m4_t part2 = __riscv_vfmacc_vv_f16m4(part1, zzh, order_low_, vl);
  sl = __riscv_vfadd_vv_f16m4(sl, part2, vl);

  vfloat16m4_t sqrt_value = __riscv_vfadd_vv_f16m4(sh, sl, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m4(sqrt_value, nan_f16, zero_less_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m4(sqrt_value, inf_f16, inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m4(sqrt_value, nan_f16, nan_mask, vl);
#endif

  return sqrt_value;
}

vfloat16m8_t __riscv_vsqrt_f16m8(vfloat16m8_t x, size_t vl)
{
#ifndef __FAST_MATH__
  unsigned short nan_si = 0x7e00; // mask for NaN
  _Float16 nan_f16 = *(_Float16*)(&nan_si); // NaN
  unsigned short inf_si = 0x7c00; // mask for +inf
  _Float16 inf_f16 = *(_Float16*)(&inf_si); // +inf

  vbool2_t zero_less_mask = __riscv_vmflt_vf_f16m8_b2(x, 0.0f16, vl);
  vbool2_t inf_mask = __riscv_vmfeq_vf_f16m8_b2(x, inf_f16, vl);
  vuint16m8_t x_int = 
    __riscv_vand_vx_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(x), 0x7fff, vl);
  vbool2_t nan_mask = __riscv_vmsgtu_vx_u16m8_b2(x_int, 0x7e00, vl);

  vbool2_t special_mask = __riscv_vmor_mm_b2(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b2(special_mask, nan_mask, vl);
  vfloat16m8_t x_spec = __riscv_vfmerge_vfm_f16m8(x, 0, special_mask, vl);

  x_int = __riscv_vreinterpret_v_f16m8_u16m8(x_spec);
#else
  vuint16m8_t x_int = __riscv_vreinterpret_v_f16m8_u16m8(x);
#endif

  vuint16m8_t mantissa_in_x = __riscv_vand_vx_u16m8(x_int, 0x03ff, vl);
  vuint16m8_t reduced_x_int = __riscv_vor_vx_u16m8(mantissa_in_x, 0x3c00, vl);
  vfloat16m8_t reduced_x = __riscv_vreinterpret_v_u16m8_f16m8(reduced_x_int);

  vuint16m8_t order_in_x = __riscv_vand_vx_u16m8(x_int, 0x7c00, vl);
  order_in_x = __riscv_vsrl_vx_u16m8(order_in_x, 10, vl);
  vuint16m8_t high_ind = __riscv_vsrl_vx_u16m8(order_in_x, 4, vl);
  vuint16m8_t low_ind = __riscv_vand_vx_u16m8(order_in_x, 0xf, vl);

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
  r = __riscv_vfrsub_vf_f16m8(__riscv_vfmul_vv_f16m8(xx, h, vl), 0.5f16, vl);

  r = __riscv_vfmacc_vv_f16m8(reduced_x, __riscv_vfmul_vf_f16m8(xx, -1.f16, vl), xx, vl);

  vfloat16m8_t zh = __riscv_vfmacc_vv_f16m8(xx, r, h, vl); // high-part of sqrt value
  vfloat16m8_t sh, sl;
  sh = __riscv_vfsub_vv_f16m8(xx, zh, vl);
  sl = __riscv_vfadd_vv_f16m8(sh, zh, vl);
  sl = __riscv_vfsub_vv_f16m8(xx, sl, vl);
  vfloat16m8_t zl = __riscv_vfmacc_vv_f16m8(sh, r, h, vl);
  zl = __riscv_vfadd_vv_f16m8(zl, sl, vl);                 // low-part of sqrt value
  
  high_ind = __riscv_vmul_vx_u16m8(high_ind, 2, vl);
  low_ind = __riscv_vmul_vx_u16m8(low_ind, 2, vl);
  vfloat16m8_t order_high = __riscv_vloxei16_v_f16m8(order_tab_high_f16, high_ind, vl);
  vfloat16m8_t order_low = __riscv_vloxei16_v_f16m8(order_tab_low_f16, low_ind, vl);
  vfloat16m8_t order_low_ = __riscv_vloxei16_v_f16m8(order_tab_low_f16_, low_ind, vl);

  vfloat16m8_t zzh = __riscv_vfmul_vv_f16m8(zh, order_high, vl);
  vfloat16m8_t zzl = __riscv_vfmul_vv_f16m8(zl, order_high, vl);

  sh = __riscv_vfmul_vv_f16m8(zzh, order_low, vl);
  sl = __riscv_vfmacc_vv_f16m8(__riscv_vfmul_vf_f16m8(sh, -1.f16, vl), zzh, order_low, vl);
  vfloat16m8_t part1 = __riscv_vfmul_vv_f16m8(zzl, order_low, vl);
  vfloat16m8_t part2 = __riscv_vfmacc_vv_f16m8(part1, zzh, order_low_, vl);
  sl = __riscv_vfadd_vv_f16m8(sl, part2, vl);

  vfloat16m8_t sqrt_value = __riscv_vfadd_vv_f16m8(sh, sl, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m8(sqrt_value, nan_f16, zero_less_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m8(sqrt_value, inf_f16, inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f16m8(sqrt_value, nan_f16, nan_mask, vl);
#endif

  return sqrt_value;
}

vfloat32m1_t __riscv_vsqrt_f32m1(vfloat32m1_t x, size_t vl)
{
#ifndef __FAST_MATH__
  unsigned int inf_ui = 0x7f800000; // mask for +inf
  float inf_f = *(float*)(&inf_ui); // +inf
  unsigned int nan_ui = 0x7fc00000; // mask for NaN
  float nan_f = *(float*)(&nan_ui); // NaN

  vbool32_t zero_less_mask = __riscv_vmflt_vf_f32m1_b32(x, 0, vl);
  vbool32_t inf_mask = __riscv_vmfeq_vf_f32m1_b32(x, inf_f, vl);
  vuint32m1_t x_int = 
    __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(x), 0x7fffffff, vl);
  vbool32_t nan_mask = __riscv_vmsgtu_vx_u32m1_b32(x_int, 0x7f800000, vl);

  vbool32_t special_mask = __riscv_vmor_mm_b32(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b32(special_mask, nan_mask, vl);
  vfloat32m1_t x_spec = __riscv_vfmerge_vfm_f32m1(x, 0, special_mask, vl);

  x_int = __riscv_vreinterpret_v_f32m1_u32m1(x_spec);
#else
  vuint32m1_t x_int = __riscv_vreinterpret_v_f32m1_u32m1(x);
#endif

  vuint32m1_t mantissa_in_x = __riscv_vand_vx_u32m1(x_int, 0x007fffff, vl);
  vuint32m1_t reduced_x_int = __riscv_vor_vx_u32m1(mantissa_in_x, 0x3f800000, vl);
  vfloat32m1_t reduced_x = __riscv_vreinterpret_v_u32m1_f32m1(reduced_x_int);

  vuint32m1_t order_in_x = __riscv_vand_vx_u32m1(x_int, 0x7f800000, vl);
  order_in_x = __riscv_vsrl_vx_u32m1(order_in_x, 23, vl);
  vuint32m1_t high_ind = __riscv_vsrl_vx_u32m1(order_in_x, 4, vl);
  vuint32m1_t low_ind = __riscv_vand_vx_u32m1(order_in_x, 0xf, vl);

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
  r = __riscv_vfrsub_vf_f32m1(__riscv_vfmul_vv_f32m1(xx, h, vl), 0.5f, vl);

  r = __riscv_vfmacc_vv_f32m1(reduced_x, __riscv_vfmul_vf_f32m1(xx, -1.f, vl), xx, vl);
  
  vfloat32m1_t zh = __riscv_vfmacc_vv_f32m1(xx, r, h, vl); // high-part of sqrt value
  vfloat32m1_t sh, sl;
  sh = __riscv_vfsub_vv_f32m1(xx, zh, vl);
  sl = __riscv_vfadd_vv_f32m1(sh, zh, vl);
  sl = __riscv_vfsub_vv_f32m1(xx, sl, vl);
  vfloat32m1_t zl = __riscv_vfmacc_vv_f32m1(sh, r, h, vl);
  zl = __riscv_vfadd_vv_f32m1(zl, sl, vl);                 // low-part of sqrt value
  
  high_ind = __riscv_vmul_vx_u32m1(high_ind, 4, vl);
  low_ind = __riscv_vmul_vx_u32m1(low_ind, 4, vl);
  vfloat32m1_t order_high = __riscv_vloxei32_v_f32m1(order_tab_high_flt, high_ind, vl);
  vfloat32m1_t order_low = __riscv_vloxei32_v_f32m1(order_tab_low_flt, low_ind, vl);
  vfloat32m1_t order_low_ = __riscv_vloxei32_v_f32m1(order_tab_low_flt_, low_ind, vl);

  vfloat32m1_t zzh = __riscv_vfmul_vv_f32m1(zh, order_high, vl);
  vfloat32m1_t zzl = __riscv_vfmul_vv_f32m1(zl, order_high, vl);

  sh = __riscv_vfmul_vv_f32m1(zzh, order_low, vl);
  sl = __riscv_vfmacc_vv_f32m1(__riscv_vfmul_vf_f32m1(sh, -1.f, vl), zzh, order_low, vl);
  vfloat32m1_t part1 = __riscv_vfmul_vv_f32m1(zzl, order_low, vl);
  vfloat32m1_t part2 = __riscv_vfmacc_vv_f32m1(part1, zzh, order_low_, vl);
  sl = __riscv_vfadd_vv_f32m1(sl, part2, vl);

  vfloat32m1_t sqrt_value = __riscv_vfadd_vv_f32m1(sh, sl, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m1(sqrt_value, nan_f, zero_less_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m1(sqrt_value, inf_f, inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m1(sqrt_value, nan_f, nan_mask, vl);
#endif

  return sqrt_value; 
}

vfloat32m2_t __riscv_vsqrt_f32m2(vfloat32m2_t x, size_t vl)
{
#ifndef __FAST_MATH__
  unsigned int inf_ui = 0x7f800000; // mask for +inf
  float inf_f = *(float*)(&inf_ui); // +inf
  unsigned int nan_ui = 0x7fc00000; // mask for NaN
  float nan_f = *(float*)(&nan_ui); // NaN

  vbool16_t zero_less_mask = __riscv_vmflt_vf_f32m2_b16(x, 0, vl);
  vbool16_t inf_mask = __riscv_vmfeq_vf_f32m2_b16(x, inf_f, vl);
  vuint32m2_t x_int = 
    __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(x), 0x7fffffff, vl);
  vbool16_t nan_mask = __riscv_vmsgtu_vx_u32m2_b16(x_int, 0x7f800000, vl);

  vbool16_t special_mask = __riscv_vmor_mm_b16(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b16(special_mask, nan_mask, vl);

  vfloat32m2_t x_spec = __riscv_vfmerge_vfm_f32m2(x, 0, special_mask, vl);

  x_int = __riscv_vreinterpret_v_f32m2_u32m2(x_spec);
#else
  vuint32m2_t x_int = __riscv_vreinterpret_v_f32m2_u32m2(x);
#endif

  vuint32m2_t mantissa_in_x = __riscv_vand_vx_u32m2(x_int, 0x007fffff, vl);
  vuint32m2_t reduced_x_int = __riscv_vor_vx_u32m2(mantissa_in_x, 0x3f800000, vl);
  vfloat32m2_t reduced_x = __riscv_vreinterpret_v_u32m2_f32m2(reduced_x_int);

  vuint32m2_t order_in_x = __riscv_vand_vx_u32m2(x_int, 0x7f800000, vl);
  order_in_x = __riscv_vsrl_vx_u32m2(order_in_x, 23, vl);
  vuint32m2_t high_ind = __riscv_vsrl_vx_u32m2(order_in_x, 4, vl);
  vuint32m2_t low_ind = __riscv_vand_vx_u32m2(order_in_x, 0xf, vl);

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
  r = __riscv_vfrsub_vf_f32m2(__riscv_vfmul_vv_f32m2(xx, h, vl), 0.5f, vl);

  r = __riscv_vfmacc_vv_f32m2(reduced_x, __riscv_vfmul_vf_f32m2(xx, -1.f, vl), xx, vl);

  vfloat32m2_t zh = __riscv_vfmacc_vv_f32m2(xx, r, h, vl); // high-part of sqrt value
  vfloat32m2_t sh, sl;
  sh = __riscv_vfsub_vv_f32m2(xx, zh, vl);
  sl = __riscv_vfadd_vv_f32m2(sh, zh, vl);
  sl = __riscv_vfsub_vv_f32m2(xx, sl, vl);
  vfloat32m2_t zl = __riscv_vfmacc_vv_f32m2(sh, r, h, vl);
  zl = __riscv_vfadd_vv_f32m2(zl, sl, vl);                 // low-part of sqrt value
  
  high_ind = __riscv_vmul_vx_u32m2(high_ind, 4, vl);
  low_ind = __riscv_vmul_vx_u32m2(low_ind, 4, vl);
  vfloat32m2_t order_high = __riscv_vloxei32_v_f32m2(order_tab_high_flt, high_ind, vl);
  vfloat32m2_t order_low = __riscv_vloxei32_v_f32m2(order_tab_low_flt, low_ind, vl);
  vfloat32m2_t order_low_ = __riscv_vloxei32_v_f32m2(order_tab_low_flt_, low_ind, vl);

  vfloat32m2_t zzh = __riscv_vfmul_vv_f32m2(zh, order_high, vl);
  vfloat32m2_t zzl = __riscv_vfmul_vv_f32m2(zl, order_high, vl);

  sh = __riscv_vfmul_vv_f32m2(zzh, order_low, vl);
  sl = __riscv_vfmacc_vv_f32m2(__riscv_vfmul_vf_f32m2(sh, -1.f, vl), zzh, order_low, vl);
  vfloat32m2_t part1 = __riscv_vfmul_vv_f32m2(zzl, order_low, vl);
  vfloat32m2_t part2 = __riscv_vfmacc_vv_f32m2(part1, zzh, order_low_, vl);
  sl = __riscv_vfadd_vv_f32m2(sl, part2, vl);

  vfloat32m2_t sqrt_value = __riscv_vfadd_vv_f32m2(sh, sl, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m2(sqrt_value, nan_f, zero_less_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m2(sqrt_value, inf_f, inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m2(sqrt_value, nan_f, nan_mask, vl);
#endif

  return sqrt_value; 
}

vfloat32m4_t __riscv_vsqrt_f32m4(vfloat32m4_t x, size_t vl)
{
#ifndef __FAST_MATH__
  unsigned int inf_ui = 0x7f800000; // mask for +inf
  float inf_f = *(float*)(&inf_ui); // +inf
  unsigned int nan_ui = 0x7fc00000; // mask for NaN
  float nan_f = *(float*)(&nan_ui); // NaN

  vbool8_t zero_less_mask = __riscv_vmflt_vf_f32m4_b8(x, 0, vl);
  vbool8_t inf_mask = __riscv_vmfeq_vf_f32m4_b8(x, inf_f, vl);
  vuint32m4_t x_int = 
    __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(x), 0x7fffffff, vl);
  vbool8_t nan_mask = __riscv_vmsgtu_vx_u32m4_b8(x_int, 0x7f800000, vl);

  vbool8_t special_mask = __riscv_vmor_mm_b8(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b8(special_mask, nan_mask, vl);
  vfloat32m4_t x_spec = __riscv_vfmerge_vfm_f32m4(x, 0, special_mask, vl);

  x_int = __riscv_vreinterpret_v_f32m4_u32m4(x_spec);
#else
  vuint32m4_t x_int = __riscv_vreinterpret_v_f32m4_u32m4(x);
#endif

  vuint32m4_t mantissa_in_x = __riscv_vand_vx_u32m4(x_int, 0x007fffff, vl);
  vuint32m4_t reduced_x_int = __riscv_vor_vx_u32m4(mantissa_in_x, 0x3f800000, vl);
  vfloat32m4_t reduced_x = __riscv_vreinterpret_v_u32m4_f32m4(reduced_x_int);

  vuint32m4_t order_in_x = __riscv_vand_vx_u32m4(x_int, 0x7f800000, vl);
  order_in_x = __riscv_vsrl_vx_u32m4(order_in_x, 23, vl);
  vuint32m4_t high_ind = __riscv_vsrl_vx_u32m4(order_in_x, 4, vl);
  vuint32m4_t low_ind = __riscv_vand_vx_u32m4(order_in_x, 0xf, vl);

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
  r = __riscv_vfrsub_vf_f32m4(__riscv_vfmul_vv_f32m4(xx, h, vl), 0.5f, vl);

  r = __riscv_vfmacc_vv_f32m4(reduced_x, __riscv_vfmul_vf_f32m4(xx, -1.f, vl), xx, vl);

  vfloat32m4_t zh = __riscv_vfmacc_vv_f32m4(xx, r, h, vl); // high-part of sqrt value
  vfloat32m4_t sh, sl;
  sh = __riscv_vfsub_vv_f32m4(xx, zh, vl);
  sl = __riscv_vfadd_vv_f32m4(sh, zh, vl);
  sl = __riscv_vfsub_vv_f32m4(xx, sl, vl);
  vfloat32m4_t zl = __riscv_vfmacc_vv_f32m4(sh, r, h, vl);
  zl = __riscv_vfadd_vv_f32m4(zl, sl, vl);                 // low-part of sqrt value
  
  high_ind = __riscv_vmul_vx_u32m4(high_ind, 4, vl);
  low_ind = __riscv_vmul_vx_u32m4(low_ind, 4, vl);
  vfloat32m4_t order_high = __riscv_vloxei32_v_f32m4(order_tab_high_flt, high_ind, vl);
  vfloat32m4_t order_low = __riscv_vloxei32_v_f32m4(order_tab_low_flt, low_ind, vl);
  vfloat32m4_t order_low_ = __riscv_vloxei32_v_f32m4(order_tab_low_flt_, low_ind, vl);

  vfloat32m4_t zzh = __riscv_vfmul_vv_f32m4(zh, order_high, vl);
  vfloat32m4_t zzl = __riscv_vfmul_vv_f32m4(zl, order_high, vl);

  sh = __riscv_vfmul_vv_f32m4(zzh, order_low, vl);
  sl = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vf_f32m4(sh, -1.f, vl), zzh, order_low, vl);
  vfloat32m4_t part1 = __riscv_vfmul_vv_f32m4(zzl, order_low, vl);
  vfloat32m4_t part2 = __riscv_vfmacc_vv_f32m4(part1, zzh, order_low_, vl);
  sl = __riscv_vfadd_vv_f32m4(sl, part2, vl);

  vfloat32m4_t sqrt_value = __riscv_vfadd_vv_f32m4(sh, sl, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m4(sqrt_value, nan_f, zero_less_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m4(sqrt_value, inf_f, inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m4(sqrt_value, nan_f, nan_mask, vl);
#endif

  return sqrt_value; 
}

vfloat32m8_t __riscv_vsqrt_f32m8(vfloat32m8_t x, size_t vl)
{
#ifndef __FAST_MATH__
  unsigned int inf_ui = 0x7f800000; // mask for +inf
  float inf_f = *(float*)(&inf_ui); // +inf
  unsigned int nan_ui = 0x7fc00000; // mask for NaN
  float nan_f = *(float*)(&nan_ui); // NaN

  vbool4_t zero_less_mask = __riscv_vmflt_vf_f32m8_b4(x, 0, vl);
  vbool4_t inf_mask = __riscv_vmfeq_vf_f32m8_b4(x, inf_f, vl);
  vuint32m8_t x_int = 
    __riscv_vand_vx_u32m8(__riscv_vreinterpret_v_f32m8_u32m8(x), 0x7fffffff, vl);
  vbool4_t nan_mask = __riscv_vmsgtu_vx_u32m8_b4(x_int, 0x7f800000, vl);

  vbool4_t special_mask = __riscv_vmor_mm_b4(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b4(special_mask, nan_mask, vl);

  vfloat32m8_t x_spec = __riscv_vfmerge_vfm_f32m8(x, 0, special_mask, vl);

  x_int = __riscv_vreinterpret_v_f32m8_u32m8(x_spec);
#else
  vuint32m8_t x_int = __riscv_vreinterpret_v_f32m8_u32m8(x);
#endif

  vuint32m8_t mantissa_in_x = __riscv_vand_vx_u32m8(x_int, 0x007fffff, vl);
  vuint32m8_t reduced_x_int = __riscv_vor_vx_u32m8(mantissa_in_x, 0x3f800000, vl);
  vfloat32m8_t reduced_x = __riscv_vreinterpret_v_u32m8_f32m8(reduced_x_int);

  vuint32m8_t order_in_x = __riscv_vand_vx_u32m8(x_int, 0x7f800000, vl);
  order_in_x = __riscv_vsrl_vx_u32m8(order_in_x, 23, vl);
  vuint32m8_t high_ind = __riscv_vsrl_vx_u32m8(order_in_x, 4, vl);
  vuint32m8_t low_ind = __riscv_vand_vx_u32m8(order_in_x, 0xf, vl);

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
  r = __riscv_vfrsub_vf_f32m8(__riscv_vfmul_vv_f32m8(xx, h, vl), 0.5f, vl);

  r = __riscv_vfmacc_vv_f32m8(reduced_x, __riscv_vfmul_vf_f32m8(xx, -1.f, vl), xx, vl);

  vfloat32m8_t zh = __riscv_vfmacc_vv_f32m8(xx, r, h, vl); // high-part of sqrt value
  vfloat32m8_t sh, sl;
  sh = __riscv_vfsub_vv_f32m8(xx, zh, vl);
  sl = __riscv_vfadd_vv_f32m8(sh, zh, vl);
  sl = __riscv_vfsub_vv_f32m8(xx, sl, vl);
  vfloat32m8_t zl = __riscv_vfmacc_vv_f32m8(sh, r, h, vl);
  zl = __riscv_vfadd_vv_f32m8(zl, sl, vl);                 // low-part of sqrt value
  
  high_ind = __riscv_vmul_vx_u32m8(high_ind, 4, vl);
  low_ind = __riscv_vmul_vx_u32m8(low_ind, 4, vl);
  vfloat32m8_t order_high = __riscv_vloxei32_v_f32m8(order_tab_high_flt, high_ind, vl);
  vfloat32m8_t order_low = __riscv_vloxei32_v_f32m8(order_tab_low_flt, low_ind, vl);
  vfloat32m8_t order_low_ = __riscv_vloxei32_v_f32m8(order_tab_low_flt_, low_ind, vl);

  vfloat32m8_t zzh = __riscv_vfmul_vv_f32m8(zh, order_high, vl);
  vfloat32m8_t zzl = __riscv_vfmul_vv_f32m8(zl, order_high, vl);

  sh = __riscv_vfmul_vv_f32m8(zzh, order_low, vl);
  sl = __riscv_vfmacc_vv_f32m8(__riscv_vfmul_vf_f32m8(sh, -1.f, vl), zzh, order_low, vl);
  vfloat32m8_t part1 = __riscv_vfmul_vv_f32m8(zzl, order_low, vl);
  vfloat32m8_t part2 = __riscv_vfmacc_vv_f32m8(part1, zzh, order_low_, vl);
  sl = __riscv_vfadd_vv_f32m8(sl, part2, vl);

  vfloat32m8_t sqrt_value = __riscv_vfadd_vv_f32m8(sh, sl, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m8(sqrt_value, nan_f, zero_less_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m8(sqrt_value, inf_f, inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f32m8(sqrt_value, nan_f, nan_mask, vl);
#endif

  return sqrt_value; 
}

vfloat64m1_t __riscv_vsqrt_f64m1(vfloat64m1_t x, size_t vl)
{
#ifndef __FAST_MATH__
  unsigned long long nan_ull = 0x7ff8000000000000; // mask for NaN
  double nan_d = *(double*)(&nan_ull); // NaN
  unsigned long long inf_ull = 0x7ff0000000000000; // mask for +inf
  double inf_d = *(double*)(&inf_ull); // +inf

  vbool64_t zero_less_mask = __riscv_vmflt_vf_f64m1_b64(x, 0, vl);
  vbool64_t inf_mask = __riscv_vmfeq_vf_f64m1_b64(x, inf_d, vl);
  vuint64m1_t x_int = 
    __riscv_vand_vx_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(x), 0x7fffffffffffffff, vl);
  vbool64_t nan_mask = __riscv_vmsgtu_vx_u64m1_b64(x_int, 0x7ff0000000000000, vl);

  vbool64_t special_mask = __riscv_vmor_mm_b64(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b64(special_mask, nan_mask, vl);

  vfloat64m1_t x_spec = __riscv_vfmerge_vfm_f64m1(x, 0, special_mask, vl);

  x_int = __riscv_vreinterpret_v_f64m1_u64m1(x_spec);
#else
  vuint64m1_t x_int = __riscv_vreinterpret_v_f64m1_u64m1(x);
#endif

  vuint64m1_t mantissa_in_x = __riscv_vand_vx_u64m1(x_int, 0x000fffffffffffff, vl);
  vuint64m1_t reduced_x_int = __riscv_vor_vx_u64m1(mantissa_in_x, 0x3ff0000000000000, vl);
  vfloat64m1_t reduced_x = __riscv_vreinterpret_v_u64m1_f64m1(reduced_x_int);

  vuint64m1_t order_in_x = __riscv_vand_vx_u64m1(x_int, 0x7ff0000000000000LL, vl);
  order_in_x = __riscv_vsrl_vx_u64m1(order_in_x, 52, vl);
  vuint64m1_t high_ind = __riscv_vsrl_vx_u64m1(order_in_x, 8, vl);
  vuint64m1_t low_ind = __riscv_vand_vx_u64m1(order_in_x, 0xf, vl);
  vuint64m1_t mid_ind = 
    __riscv_vsrl_vx_u64m1(__riscv_vand_vx_u64m1(order_in_x, 0xf << 4, vl), 4, vl);

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
  r = __riscv_vfrsub_vf_f64m1(__riscv_vfmul_vv_f64m1(xx, h, vl), 0.5, vl);

  r = __riscv_vfmacc_vv_f64m1(reduced_x, __riscv_vfmul_vf_f64m1(xx, -1., vl), xx, vl);

  vfloat64m1_t zh = __riscv_vfmacc_vv_f64m1(xx, r, h, vl); // high-part of sqrt value
  vfloat64m1_t sh, sl;
  sh = __riscv_vfsub_vv_f64m1(xx, zh, vl);
  sl = __riscv_vfadd_vv_f64m1(sh, zh, vl);
  sl = __riscv_vfsub_vv_f64m1(xx, sl, vl);
  vfloat64m1_t zl = __riscv_vfmacc_vv_f64m1(sh, r, h, vl);
  zl = __riscv_vfadd_vv_f64m1(zl, sl, vl);                 // low-part of sqrt value

  high_ind = __riscv_vmul_vx_u64m1(high_ind, 8, vl);
  mid_ind = __riscv_vmul_vx_u64m1(mid_ind, 8, vl);
  low_ind = __riscv_vmul_vx_u64m1(low_ind, 8, vl);
  vfloat64m1_t order_high = __riscv_vloxei64_v_f64m1(order_tab_high, high_ind, vl);
  vfloat64m1_t order_mid = __riscv_vloxei64_v_f64m1(order_tab_mid, mid_ind, vl);
  vfloat64m1_t order_low = __riscv_vloxei64_v_f64m1(order_tab_low, low_ind, vl);
  vfloat64m1_t order_low_ = __riscv_vloxei64_v_f64m1(order_tab_low_, low_ind, vl);

  vfloat64m1_t zzh = __riscv_vfmul_vv_f64m1(zh, order_high, vl);
  zzh = __riscv_vfmul_vv_f64m1(zzh, order_mid, vl);
  vfloat64m1_t zzl = __riscv_vfmul_vv_f64m1(zl, order_high, vl);
  zzl = __riscv_vfmul_vv_f64m1(zzl, order_mid, vl);

  sh = __riscv_vfmul_vv_f64m1(zzh, order_low, vl);
  sl = __riscv_vfmacc_vv_f64m1(__riscv_vfmul_vf_f64m1(sh, -1.0, vl), zzh, order_low, vl);
  vfloat64m1_t part1 = __riscv_vfmul_vv_f64m1(zzl, order_low, vl);
  vfloat64m1_t part2 = __riscv_vfmacc_vv_f64m1(part1, zzh, order_low_, vl);
  sl = __riscv_vfadd_vv_f64m1(sl, part2, vl);

  vfloat64m1_t sqrt_value = __riscv_vfadd_vv_f64m1(sh, sl, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m1(sqrt_value, nan_d, zero_less_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m1(sqrt_value, inf_d, inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m1(sqrt_value, nan_d, nan_mask, vl);
#endif

  return sqrt_value; 
}

vfloat64m2_t __riscv_vsqrt_f64m2(vfloat64m2_t x, size_t vl)
{
#ifndef __FAST_MATH__
  unsigned long long nan_ull = 0x7ff8000000000000; // mask for NaN
  double nan_d = *(double*)(&nan_ull); // NaN
  unsigned long long inf_ull = 0x7ff0000000000000; // mask for +inf
  double inf_d = *(double*)(&inf_ull); // +inf

  vbool32_t zero_less_mask = __riscv_vmflt_vf_f64m2_b32(x, 0, vl);
  vbool32_t inf_mask = __riscv_vmfeq_vf_f64m2_b32(x, inf_d, vl);
  vuint64m2_t x_int = 
    __riscv_vand_vx_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(x), 0x7fffffffffffffff, vl);
  vbool32_t nan_mask = __riscv_vmsgtu_vx_u64m2_b32(x_int, 0x7ff0000000000000, vl);

  vbool32_t special_mask = __riscv_vmor_mm_b32(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b32(special_mask, nan_mask, vl);

  vfloat64m2_t x_spec = __riscv_vfmerge_vfm_f64m2(x, 0, special_mask, vl);

  x_int = __riscv_vreinterpret_v_f64m2_u64m2(x_spec);
#else
  vuint64m2_t x_int = __riscv_vreinterpret_v_f64m2_u64m2(x);
#endif

  vuint64m2_t mantissa_in_x = __riscv_vand_vx_u64m2(x_int, 0x000fffffffffffff, vl);
  vuint64m2_t reduced_x_int = __riscv_vor_vx_u64m2(mantissa_in_x, 0x3ff0000000000000, vl);
  vfloat64m2_t reduced_x = __riscv_vreinterpret_v_u64m2_f64m2(reduced_x_int);

  vuint64m2_t order_in_x = __riscv_vand_vx_u64m2(x_int, 0x7ff0000000000000LL, vl);
  order_in_x = __riscv_vsrl_vx_u64m2(order_in_x, 52, vl);
  vuint64m2_t high_ind = __riscv_vsrl_vx_u64m2(order_in_x, 8, vl);
  vuint64m2_t low_ind = __riscv_vand_vx_u64m2(order_in_x, 0xf, vl);
  vuint64m2_t mid_ind = 
    __riscv_vsrl_vx_u64m2(__riscv_vand_vx_u64m2(order_in_x, 0xf << 4, vl), 4, vl);

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
  r = __riscv_vfrsub_vf_f64m2(__riscv_vfmul_vv_f64m2(xx, h, vl), 0.5, vl);

  r = __riscv_vfmacc_vv_f64m2(reduced_x, __riscv_vfmul_vf_f64m2(xx, -1., vl), xx, vl);

  vfloat64m2_t zh = __riscv_vfmacc_vv_f64m2(xx, r, h, vl); // high-part of sqrt value
  vfloat64m2_t sh, sl;
  sh = __riscv_vfsub_vv_f64m2(xx, zh, vl);
  sl = __riscv_vfadd_vv_f64m2(sh, zh, vl);
  sl = __riscv_vfsub_vv_f64m2(xx, sl, vl);
  vfloat64m2_t zl = __riscv_vfmacc_vv_f64m2(sh, r, h, vl);
  zl = __riscv_vfadd_vv_f64m2(zl, sl, vl);                 // low-part of sqrt value

  high_ind = __riscv_vmul_vx_u64m2(high_ind, 8, vl);
  mid_ind = __riscv_vmul_vx_u64m2(mid_ind, 8, vl);
  low_ind = __riscv_vmul_vx_u64m2(low_ind, 8, vl);
  vfloat64m2_t order_high = __riscv_vloxei64_v_f64m2(order_tab_high, high_ind, vl);
  vfloat64m2_t order_mid = __riscv_vloxei64_v_f64m2(order_tab_mid, mid_ind, vl);
  vfloat64m2_t order_low = __riscv_vloxei64_v_f64m2(order_tab_low, low_ind, vl);
  vfloat64m2_t order_low_ = __riscv_vloxei64_v_f64m2(order_tab_low_, low_ind, vl);

  vfloat64m2_t zzh = __riscv_vfmul_vv_f64m2(zh, order_high, vl);
  zzh = __riscv_vfmul_vv_f64m2(zzh, order_mid, vl);
  vfloat64m2_t zzl = __riscv_vfmul_vv_f64m2(zl, order_high, vl);
  zzl = __riscv_vfmul_vv_f64m2(zzl, order_mid, vl);

  sh = __riscv_vfmul_vv_f64m2(zzh, order_low, vl);
  sl = __riscv_vfmacc_vv_f64m2(__riscv_vfmul_vf_f64m2(sh, -1.0, vl), zzh, order_low, vl);
  vfloat64m2_t part1 = __riscv_vfmul_vv_f64m2(zzl, order_low, vl);
  vfloat64m2_t part2 = __riscv_vfmacc_vv_f64m2(part1, zzh, order_low_, vl);
  sl = __riscv_vfadd_vv_f64m2(sl, part2, vl);

  vfloat64m2_t sqrt_value = __riscv_vfadd_vv_f64m2(sh, sl, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m2(sqrt_value, nan_d, zero_less_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m2(sqrt_value, inf_d, inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m2(sqrt_value, nan_d, nan_mask, vl);
#endif

  return sqrt_value; 
}

vfloat64m4_t __riscv_vsqrt_f64m4(vfloat64m4_t x, size_t vl)
{
#ifndef __FAST_MATH__
  unsigned long long nan_ull = 0x7ff8000000000000; // mask for NaN
  double nan_d = *(double*)(&nan_ull); // NaN
  unsigned long long inf_ull = 0x7ff0000000000000; // mask for +inf
  double inf_d = *(double*)(&inf_ull); // +inf

  vbool16_t zero_less_mask = __riscv_vmflt_vf_f64m4_b16(x, 0, vl);
  vbool16_t inf_mask = __riscv_vmfeq_vf_f64m4_b16(x, inf_d, vl);
  vuint64m4_t x_int = 
    __riscv_vand_vx_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(x), 0x7fffffffffffffff, vl);
  vbool16_t nan_mask = __riscv_vmsgtu_vx_u64m4_b16(x_int, 0x7ff0000000000000, vl);

  vbool16_t special_mask = __riscv_vmor_mm_b16(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b16(special_mask, nan_mask, vl);

  vfloat64m4_t x_spec = __riscv_vfmerge_vfm_f64m4(x, 0, special_mask, vl);

  x_int = __riscv_vreinterpret_v_f64m4_u64m4(x_spec);
#else
  vuint64m4_t x_int = __riscv_vreinterpret_v_f64m4_u64m4(x);
#endif

  vuint64m4_t mantissa_in_x = __riscv_vand_vx_u64m4(x_int, 0x000fffffffffffff, vl);
  vuint64m4_t reduced_x_int = __riscv_vor_vx_u64m4(mantissa_in_x, 0x3ff0000000000000, vl);
  vfloat64m4_t reduced_x = __riscv_vreinterpret_v_u64m4_f64m4(reduced_x_int);

  vuint64m4_t order_in_x = __riscv_vand_vx_u64m4(x_int, 0x7ff0000000000000LL, vl);
  order_in_x = __riscv_vsrl_vx_u64m4(order_in_x, 52, vl);
  vuint64m4_t high_ind = __riscv_vsrl_vx_u64m4(order_in_x, 8, vl);
  vuint64m4_t low_ind = __riscv_vand_vx_u64m4(order_in_x, 0xf, vl);
  vuint64m4_t mid_ind = 
    __riscv_vsrl_vx_u64m4(__riscv_vand_vx_u64m4(order_in_x, 0xf << 4, vl), 4, vl);

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
  r = __riscv_vfrsub_vf_f64m4(__riscv_vfmul_vv_f64m4(xx, h, vl), 0.5, vl);

  r = __riscv_vfmacc_vv_f64m4(reduced_x, __riscv_vfmul_vf_f64m4(xx, -1., vl), xx, vl);

  vfloat64m4_t zh = __riscv_vfmacc_vv_f64m4(xx, r, h, vl); // high-part of sqrt value
  vfloat64m4_t sh, sl;
  sh = __riscv_vfsub_vv_f64m4(xx, zh, vl);
  sl = __riscv_vfadd_vv_f64m4(sh, zh, vl);
  sl = __riscv_vfsub_vv_f64m4(xx, sl, vl);
  vfloat64m4_t zl = __riscv_vfmacc_vv_f64m4(sh, r, h, vl);
  zl = __riscv_vfadd_vv_f64m4(zl, sl, vl);                 // low-part of sqrt value

  high_ind = __riscv_vmul_vx_u64m4(high_ind, 8, vl);
  mid_ind = __riscv_vmul_vx_u64m4(mid_ind, 8, vl);
  low_ind = __riscv_vmul_vx_u64m4(low_ind, 8, vl);
  vfloat64m4_t order_high = __riscv_vloxei64_v_f64m4(order_tab_high, high_ind, vl);
  vfloat64m4_t order_mid = __riscv_vloxei64_v_f64m4(order_tab_mid, mid_ind, vl);
  vfloat64m4_t order_low = __riscv_vloxei64_v_f64m4(order_tab_low, low_ind, vl);
  vfloat64m4_t order_low_ = __riscv_vloxei64_v_f64m4(order_tab_low_, low_ind, vl);

  vfloat64m4_t zzh = __riscv_vfmul_vv_f64m4(zh, order_high, vl);
  zzh = __riscv_vfmul_vv_f64m4(zzh, order_mid, vl);
  vfloat64m4_t zzl = __riscv_vfmul_vv_f64m4(zl, order_high, vl);
  zzl = __riscv_vfmul_vv_f64m4(zzl, order_mid, vl);

  sh = __riscv_vfmul_vv_f64m4(zzh, order_low, vl);
  sl = __riscv_vfmacc_vv_f64m4(__riscv_vfmul_vf_f64m4(sh, -1.0, vl), zzh, order_low, vl);
  vfloat64m4_t part1 = __riscv_vfmul_vv_f64m4(zzl, order_low, vl);
  vfloat64m4_t part2 = __riscv_vfmacc_vv_f64m4(part1, zzh, order_low_, vl);
  sl = __riscv_vfadd_vv_f64m4(sl, part2, vl);

  vfloat64m4_t sqrt_value = __riscv_vfadd_vv_f64m4(sh, sl, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m4(sqrt_value, nan_d, zero_less_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m4(sqrt_value, inf_d, inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m4(sqrt_value, nan_d, nan_mask, vl);
#endif

  return sqrt_value; 
}

vfloat64m8_t __riscv_vsqrt_f64m8(vfloat64m8_t x, size_t vl)
{
#ifndef __FAST_MATH__
  unsigned long long nan_ull = 0x7ff8000000000000; // mask for NaN
  double nan_d = *(double*)(&nan_ull); // NaN
  unsigned long long inf_ull = 0x7ff0000000000000; // mask for +inf
  double inf_d = *(double*)(&inf_ull); // +inf

  vbool8_t zero_less_mask = __riscv_vmflt_vf_f64m8_b8(x, 0, vl);
  vbool8_t inf_mask = __riscv_vmfeq_vf_f64m8_b8(x, inf_d, vl);
  vuint64m8_t x_int = 
    __riscv_vand_vx_u64m8(__riscv_vreinterpret_v_f64m8_u64m8(x), 0x7fffffffffffffff, vl);
  vbool8_t nan_mask = __riscv_vmsgtu_vx_u64m8_b8(x_int, 0x7ff0000000000000, vl);

  vbool8_t special_mask = __riscv_vmor_mm_b8(zero_less_mask, inf_mask, vl);
  special_mask = __riscv_vmor_mm_b8(special_mask, nan_mask, vl);

  vfloat64m8_t x_spec = __riscv_vfmerge_vfm_f64m8(x, 0, special_mask, vl);

  x_int = __riscv_vreinterpret_v_f64m8_u64m8(x_spec);
#else
  vuint64m8_t x_int = __riscv_vreinterpret_v_f64m8_u64m8(x);
#endif

  vuint64m8_t mantissa_in_x = __riscv_vand_vx_u64m8(x_int, 0x000fffffffffffff, vl);
  vuint64m8_t reduced_x_int = __riscv_vor_vx_u64m8(mantissa_in_x, 0x3ff0000000000000, vl);
  vfloat64m8_t reduced_x = __riscv_vreinterpret_v_u64m8_f64m8(reduced_x_int);

  vuint64m8_t order_in_x = __riscv_vand_vx_u64m8(x_int, 0x7ff0000000000000LL, vl);
  order_in_x = __riscv_vsrl_vx_u64m8(order_in_x, 52, vl);
  vuint64m8_t high_ind = __riscv_vsrl_vx_u64m8(order_in_x, 8, vl);
  vuint64m8_t low_ind = __riscv_vand_vx_u64m8(order_in_x, 0xf, vl);
  vuint64m8_t mid_ind = 
    __riscv_vsrl_vx_u64m8(__riscv_vand_vx_u64m8(order_in_x, 0xf << 4, vl), 4, vl);

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
  r = __riscv_vfrsub_vf_f64m8(__riscv_vfmul_vv_f64m8(xx, h, vl), 0.5, vl);

  r = __riscv_vfmacc_vv_f64m8(reduced_x, __riscv_vfmul_vf_f64m8(xx, -1., vl), xx, vl);

  vfloat64m8_t zh = __riscv_vfmacc_vv_f64m8(xx, r, h, vl); // high-part of sqrt value
  vfloat64m8_t sh, sl;
  sh = __riscv_vfsub_vv_f64m8(xx, zh, vl);
  sl = __riscv_vfadd_vv_f64m8(sh, zh, vl);
  sl = __riscv_vfsub_vv_f64m8(xx, sl, vl);
  vfloat64m8_t zl = __riscv_vfmacc_vv_f64m8(sh, r, h, vl);
  zl = __riscv_vfadd_vv_f64m8(zl, sl, vl);                 // low-part of sqrt value

  high_ind = __riscv_vmul_vx_u64m8(high_ind, 8, vl);
  mid_ind = __riscv_vmul_vx_u64m8(mid_ind, 8, vl);
  low_ind = __riscv_vmul_vx_u64m8(low_ind, 8, vl);
  vfloat64m8_t order_high = __riscv_vloxei64_v_f64m8(order_tab_high, high_ind, vl);
  vfloat64m8_t order_mid = __riscv_vloxei64_v_f64m8(order_tab_mid, mid_ind, vl);
  vfloat64m8_t order_low = __riscv_vloxei64_v_f64m8(order_tab_low, low_ind, vl);
  vfloat64m8_t order_low_ = __riscv_vloxei64_v_f64m8(order_tab_low_, low_ind, vl);

  vfloat64m8_t zzh = __riscv_vfmul_vv_f64m8(zh, order_high, vl);
  zzh = __riscv_vfmul_vv_f64m8(zzh, order_mid, vl);
  vfloat64m8_t zzl = __riscv_vfmul_vv_f64m8(zl, order_high, vl);
  zzl = __riscv_vfmul_vv_f64m8(zzl, order_mid, vl);

  sh = __riscv_vfmul_vv_f64m8(zzh, order_low, vl);
  sl = __riscv_vfmacc_vv_f64m8(__riscv_vfmul_vf_f64m8(sh, -1.0, vl), zzh, order_low, vl);
  vfloat64m8_t part1 = __riscv_vfmul_vv_f64m8(zzl, order_low, vl);
  vfloat64m8_t part2 = __riscv_vfmacc_vv_f64m8(part1, zzh, order_low_, vl);
  sl = __riscv_vfadd_vv_f64m8(sl, part2, vl);

  vfloat64m8_t sqrt_value = __riscv_vfadd_vv_f64m8(sh, sl, vl);

#ifndef __FAST_MATH__
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m8(sqrt_value, nan_d, zero_less_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m8(sqrt_value, inf_d, inf_mask, vl);
  sqrt_value = 
    __riscv_vfmerge_vfm_f64m8(sqrt_value, nan_d, nan_mask, vl);
#endif

  return sqrt_value; 
}
#endif