/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025-2026 The RVVMF Authors
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
*/

#ifdef __riscv_v_intrinsic
//Exponential
vfloat64m1_t __riscv_vexp_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vexp_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vexp_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vexp_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vexp_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vexp_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vexp_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vexp_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vexp2_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vexp2_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vexp2_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vexp2_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vexp2_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vexp2_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vexp2_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vexp2_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vexpm1_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vexpm1_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vexpm1_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vexpm1_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vexpm1_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vexpm1_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vexpm1_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vexpm1_f32m8(vfloat32m8_t x, size_t avl);

//Logarithmic
vfloat64m1_t __riscv_vlog_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vlog_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vlog_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vlog_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vlog_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vlog_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vlog_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vlog_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vlog2_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vlog2_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vlog2_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vlog2_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vlog2_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vlog2_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vlog2_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vlog2_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vlog10_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vlog10_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vlog10_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vlog10_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vlog10_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vlog10_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vlog10_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vlog10_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vlog1p_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vlog1p_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vlog1p_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vlog1p_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vlog1p_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vlog1p_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vlog1p_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vlog1p_f32m8(vfloat32m8_t x, size_t avl);

//Trigonometric
vfloat64m1_t __riscv_vsin_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vsin_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vsin_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vsin_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vsin_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vsin_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vsin_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vsin_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vcos_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vcos_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vcos_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vcos_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vcos_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vcos_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vcos_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vcos_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vtan_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vtan_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vtan_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vtan_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vtan_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vtan_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vtan_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vtan_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vasin_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vasin_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vasin_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vasin_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vasin_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vasin_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vasin_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vasin_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vacos_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vacos_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vacos_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vacos_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vacos_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vacos_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vacos_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vacos_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vatan_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vatan_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vatan_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vatan_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vatan_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vatan_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vatan_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vatan_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vatan2_f64m1(vfloat64m1_t x, vfloat64m1_t y, size_t avl);
vfloat64m2_t __riscv_vatan2_f64m2(vfloat64m2_t x, vfloat64m2_t y, size_t avl);
vfloat64m4_t __riscv_vatan2_f64m4(vfloat64m4_t x, vfloat64m4_t y, size_t avl);
vfloat64m8_t __riscv_vatan2_f64m8(vfloat64m8_t x, vfloat64m8_t y, size_t avl);
vfloat32m1_t __riscv_vatan2_f32m1(vfloat32m1_t x, vfloat32m1_t y, size_t avl);
vfloat32m2_t __riscv_vatan2_f32m2(vfloat32m2_t x, vfloat32m2_t y, size_t avl);
vfloat32m4_t __riscv_vatan2_f32m4(vfloat32m4_t x, vfloat32m4_t y, size_t avl);
vfloat32m8_t __riscv_vatan2_f32m8(vfloat32m8_t x, vfloat32m8_t y, size_t avl);

//Hyperbolic
vfloat64m1_t __riscv_vsinh_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vsinh_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vsinh_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vsinh_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vsinh_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vsinh_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vsinh_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vsinh_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vcosh_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vcosh_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vcosh_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vcosh_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vcosh_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vcosh_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vcosh_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vcosh_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vtanh_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vtanh_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vtanh_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vtanh_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vtanh_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vtanh_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vtanh_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vtanh_f32m8(vfloat32m8_t x, size_t avl);

//Power & Root
vfloat64m1_t __riscv_vsqrt_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vsqrt_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vsqrt_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vsqrt_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vsqrt_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vsqrt_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vsqrt_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vsqrt_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vcbrt_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vcbrt_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vcbrt_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vcbrt_f64m8(vfloat64m8_t x, size_t avl);
vfloat32m1_t __riscv_vcbrt_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vcbrt_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vcbrt_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vcbrt_f32m8(vfloat32m8_t x, size_t avl);

vfloat64m1_t __riscv_vpow_f64m1(vfloat64m1_t x, vfloat64m1_t y, size_t avl);
vfloat64m2_t __riscv_vpow_f64m2(vfloat64m2_t x, vfloat64m2_t y, size_t avl);
vfloat64m4_t __riscv_vpow_f64m4(vfloat64m4_t x, vfloat64m4_t y, size_t avl);
vfloat64m8_t __riscv_vpow_f64m8(vfloat64m8_t x, vfloat64m8_t y, size_t avl);
vfloat32m1_t __riscv_vpow_f32m1(vfloat32m1_t x, vfloat32m1_t y, size_t avl);
vfloat32m2_t __riscv_vpow_f32m2(vfloat32m2_t x, vfloat32m2_t y, size_t avl);
vfloat32m4_t __riscv_vpow_f32m4(vfloat32m4_t x, vfloat32m4_t y, size_t avl);
vfloat32m8_t __riscv_vpow_f32m8(vfloat32m8_t x, vfloat32m8_t y, size_t avl);

vfloat64m1_t __riscv_vhypot_f64m1(vfloat64m1_t x, vfloat64m1_t y, size_t avl);
vfloat64m2_t __riscv_vhypot_f64m2(vfloat64m2_t x, vfloat64m2_t y, size_t avl);
vfloat64m4_t __riscv_vhypot_f64m4(vfloat64m4_t x, vfloat64m4_t y, size_t avl);
vfloat64m8_t __riscv_vhypot_f64m8(vfloat64m8_t x, vfloat64m8_t y, size_t avl);
vfloat32m1_t __riscv_vhypot_f32m1(vfloat32m1_t x, vfloat32m1_t y, size_t avl);
vfloat32m2_t __riscv_vhypot_f32m2(vfloat32m2_t x, vfloat32m2_t y, size_t avl);
vfloat32m4_t __riscv_vhypot_f32m4(vfloat32m4_t x, vfloat32m4_t y, size_t avl);
vfloat32m8_t __riscv_vhypot_f32m8(vfloat32m8_t x, vfloat32m8_t y, size_t avl);

//Rounding
vfloat64m1_t __riscv_vceil_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vceil_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vceil_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vceil_f64m8(vfloat64m8_t x, size_t avl);
vfloat64m1_t __riscv_vfloor_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vfloor_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vfloor_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vfloor_f64m8(vfloat64m8_t x, size_t avl);
vfloat64m1_t __riscv_vround_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vround_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vround_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vround_f64m8(vfloat64m8_t x, size_t avl);
vfloat64m1_t __riscv_vtrunc_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vtrunc_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vtrunc_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vtrunc_f64m8(vfloat64m8_t x, size_t avl);


vfloat64m1_t __riscv_vrint_f64m1(vfloat64m1_t x, size_t avl);
vfloat64m2_t __riscv_vrint_f64m2(vfloat64m2_t x, size_t avl);
vfloat64m4_t __riscv_vrint_f64m4(vfloat64m4_t x, size_t avl);
vfloat64m8_t __riscv_vrint_f64m8(vfloat64m8_t x, size_t avl);
vint32m1_t __riscv_vlrint_f64_i32m1(vfloat64m2_t x, size_t avl);
vint32m2_t __riscv_vlrint_f64_i32m2(vfloat64m4_t x, size_t avl);
vint32m4_t __riscv_vlrint_f64_i32m4(vfloat64m8_t x, size_t avl);
vint64m1_t __riscv_vllrint_f64_i64m1(vfloat64m1_t x, size_t avl);
vint64m2_t __riscv_vllrint_f64_i64m2(vfloat64m2_t x, size_t avl);
vint64m4_t __riscv_vllrint_f64_i64m4(vfloat64m4_t x, size_t avl);
vint64m8_t __riscv_vllrint_f64_i64m8(vfloat64m8_t x, size_t avl);

vfloat32m1_t __riscv_vceil_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vceil_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vceil_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vceil_f32m8(vfloat32m8_t x, size_t avl);
vfloat32m1_t __riscv_vfloor_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vfloor_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vfloor_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vfloor_f32m8(vfloat32m8_t x, size_t avl);
vfloat32m1_t __riscv_vround_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vround_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vround_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vround_f32m8(vfloat32m8_t x, size_t avl);
vfloat32m1_t __riscv_vtrunc_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vtrunc_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vtrunc_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vtrunc_f32m8(vfloat32m8_t x, size_t avl);

vfloat32m1_t __riscv_vrint_f32m1(vfloat32m1_t x, size_t avl);
vfloat32m2_t __riscv_vrint_f32m2(vfloat32m2_t x, size_t avl);
vfloat32m4_t __riscv_vrint_f32m4(vfloat32m4_t x, size_t avl);
vfloat32m8_t __riscv_vrint_f32m8(vfloat32m8_t x, size_t avl);
vint32m1_t __riscv_vlrint_f32_i32m1(vfloat32m1_t x, size_t avl);
vint32m2_t __riscv_vlrint_f32_i32m2(vfloat32m2_t x, size_t avl);
vint32m4_t __riscv_vlrint_f32_i32m4(vfloat32m4_t x, size_t avl);
vint32m8_t __riscv_vlrint_f32_i32m8(vfloat32m8_t x, size_t avl);
vint64m2_t __riscv_vllrint_f32_i64m2(vfloat32m1_t x, size_t avl);
vint64m4_t __riscv_vllrint_f32_i64m4(vfloat32m2_t x, size_t avl);
vint64m8_t __riscv_vllrint_f32_i64m8(vfloat32m4_t x, size_t avl);

#ifdef __riscv_zvfh
//Exponential
vfloat16m1_t __riscv_vexp_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vexp_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vexp_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vexp_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vexp2_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vexp2_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vexp2_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vexp2_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vexpm1_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vexpm1_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vexpm1_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vexpm1_f16m8(vfloat16m8_t x, size_t avl);

//Logarithmic
vfloat16m1_t __riscv_vlog_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vlog_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vlog_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vlog_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vlog2_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vlog2_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vlog2_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vlog2_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vlog10_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vlog10_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vlog10_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vlog10_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vlog1p_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vlog1p_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vlog1p_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vlog1p_f16m8(vfloat16m8_t x, size_t avl);

//Trigonometric
vfloat16m1_t __riscv_vsin_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vsin_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vsin_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vsin_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vcos_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vcos_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vcos_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vcos_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vtan_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vtan_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vtan_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vtan_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vasin_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vasin_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vasin_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vasin_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vacos_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vacos_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vacos_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vacos_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vatan_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vatan_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vatan_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vatan_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vatan2_f16m1(vfloat16m1_t x, vfloat16m1_t y, size_t avl);
vfloat16m2_t __riscv_vatan2_f16m2(vfloat16m2_t x, vfloat16m2_t y, size_t avl);
vfloat16m4_t __riscv_vatan2_f16m4(vfloat16m4_t x, vfloat16m4_t y, size_t avl);
vfloat16m8_t __riscv_vatan2_f16m8(vfloat16m8_t x, vfloat16m8_t y, size_t avl);

//Hyperbolic
vfloat16m1_t __riscv_vsinh_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vsinh_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vsinh_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vsinh_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vcosh_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vcosh_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vcosh_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vcosh_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vtanh_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vtanh_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vtanh_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vtanh_f16m8(vfloat16m8_t x, size_t avl);

//Power & Root
vfloat16m1_t __riscv_vsqrt_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vsqrt_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vsqrt_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vsqrt_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vcbrt_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vcbrt_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vcbrt_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vcbrt_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vpow_f16m1(vfloat16m1_t x, vfloat16m1_t y, size_t avl);
vfloat16m2_t __riscv_vpow_f16m2(vfloat16m2_t x, vfloat16m2_t y, size_t avl);
vfloat16m4_t __riscv_vpow_f16m4(vfloat16m4_t x, vfloat16m4_t y, size_t avl);
vfloat16m8_t __riscv_vpow_f16m8(vfloat16m8_t x, vfloat16m8_t y, size_t avl);

vfloat16m1_t __riscv_vhypot_f16m1(vfloat16m1_t x, vfloat16m1_t y, size_t avl);
vfloat16m2_t __riscv_vhypot_f16m2(vfloat16m2_t x, vfloat16m2_t y, size_t avl);
vfloat16m4_t __riscv_vhypot_f16m4(vfloat16m4_t x, vfloat16m4_t y, size_t avl);
vfloat16m8_t __riscv_vhypot_f16m8(vfloat16m8_t x, vfloat16m8_t y, size_t avl);

//Rounding
vfloat16m1_t __riscv_vceil_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vceil_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vceil_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vceil_f16m8(vfloat16m8_t x, size_t avl);
vfloat16m1_t __riscv_vfloor_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vfloor_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vfloor_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vfloor_f16m8(vfloat16m8_t x, size_t avl);
vfloat16m1_t __riscv_vround_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vround_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vround_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vround_f16m8(vfloat16m8_t x, size_t avl);
vfloat16m1_t __riscv_vtrunc_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vtrunc_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vtrunc_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vtrunc_f16m8(vfloat16m8_t x, size_t avl);

vfloat16m1_t __riscv_vrint_f16m1(vfloat16m1_t x, size_t avl);
vfloat16m2_t __riscv_vrint_f16m2(vfloat16m2_t x, size_t avl);
vfloat16m4_t __riscv_vrint_f16m4(vfloat16m4_t x, size_t avl);
vfloat16m8_t __riscv_vrint_f16m8(vfloat16m8_t x, size_t avl);
vint32m2_t __riscv_vlrint_f16_i32m2(vfloat16m1_t x, size_t avl);
vint32m4_t __riscv_vlrint_f16_i32m4(vfloat16m2_t x, size_t avl);
vint32m8_t __riscv_vlrint_f16_i32m8(vfloat16m4_t x, size_t avl);
vint64m4_t __riscv_vllrint_f16_i64m4(vfloat16m1_t x, size_t avl);
vint64m8_t __riscv_vllrint_f16_i64m8(vfloat16m2_t x, size_t avl);

#endif /* __riscv_zvfh */

#endif /* __riscv_v_intrinsic */
