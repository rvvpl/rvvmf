/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2026 The RVVMF Authors 
 * (Valentin Volokitin, Anastasia Vodeneeva)
 *
 * Distributed under the BSD 4-Clause License
 * (Seefile LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *  file: asin_acos_f16.cpp                              *
 *  Contains: vector arcsin and arccos for               *
 *            IEEE binary16(f16) via RVV 1.0 intrinsics  *
 *                                                       * 
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the arcsine of input vector V                *
 *                                                       *
 * Algorithm:                                            *
 *    1) Special cases: NaN, |x|>1                       *
 *    2) Reduction to [0, 0.5]:                          *
 *       asin(x) = pi/2-2*asin(sqrt((1-x)/2))            *
 *    3) Table reduction on sin grid T_k:                *
 *       r = x*sqrt(1-T^2) - T*sqrt(1-x^2),              *
 *       result = asin(T_k) + asin_poly(r)               *
 *                                                       *
 * Computes the arccos of input vector V                 *
 *                                                       *
 * Algorithm:                                            *
 *    1) Special cases: NaN, |x|>1                       *
 *    2) Reduction to [0, 0.5]:                          *
 *       acos(x) = 2*asin(sqrt((1-x)/2))                 *
 *                                                       *
 * Note that this implementation provides at most        *
 * 0.501 ulp.                                            *
 *                                                       *
 *********************************************************
*/
 

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#include <stdint.h>

#ifdef __riscv_zvfh
#define PI_2_HI_16 0x1.920p+0f16 
#define PI_2_LO_16 0x1.fb4p-12f16 
#define PI_HI_16   0x1.920p+1f16 
#define PI_LO_16   0x1.fb4p-11f16 

inline void __riscv_vasin_kernel_f16m1(vfloat16m1_t vy, vfloat16m1_t vy_lo,
                                       vfloat16m1_t &vasin_r_hi, vfloat16m1_t &vasin_r_lo, size_t vl) {
    
    vfloat16m1_t vy2    = __riscv_vfmul_vv_f16m1(vy, vy, vl);
    vfloat16m1_t vy2_lo = __riscv_vfmsub_vv_f16m1(vy, vy, vy2, vl);
    vy2_lo              = __riscv_vfmadd_vv_f16m1(__riscv_vfadd_vv_f16m1(vy, vy, vl), vy_lo, vy2_lo, vl);

    vfloat16m1_t vp  = __riscv_vfmadd_vf_f16m1(vy2, 0x1.9e8p-5f16, __riscv_vfmv_v_f_f16m1(0x1.45p-5f16, vl), vl);
    vp               = __riscv_vfmadd_vv_f16m1(vy2, vp, __riscv_vfmv_v_f_f16m1(0x1.354p-4f16, vl), vl);
    vfloat16m1_t vc3 = __riscv_vfmv_v_f_f16m1(0x1.554p-3f16, vl);
    vfloat16m1_t vp1 = __riscv_vfmadd_vv_f16m1(vy2, vp, vc3, vl);
    vp               = __riscv_vfmadd_vv_f16m1(vy2, vp, __riscv_vfsub_vv_f16m1(vc3, vp1, vl), vl);

    vfloat16m1_t vone = __riscv_vfmv_v_f_f16m1(1.0f16, vl);
    vfloat16m1_t vph  = __riscv_vfmadd_vv_f16m1(vy2, vp1, vone, vl);
    
    vfloat16m1_t vpl  = __riscv_vfmadd_vv_f16m1(vy2, vp1, __riscv_vfsub_vv_f16m1(vone, vph, vl), vl);
    vpl               = __riscv_vfmadd_vv_f16m1(vp1, vy2_lo, __riscv_vfmadd_vv_f16m1(vp, vy2, vpl, vl), vl);

    vasin_r_hi = __riscv_vfmul_vv_f16m1(vph, vy, vl);
    vasin_r_lo = __riscv_vfmsub_vv_f16m1(vph, vy, vasin_r_hi, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f16m1(vph, vy_lo, vasin_r_lo, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f16m1(vpl, vy, vasin_r_lo, vl);
                                      
}

vfloat16m1_t __riscv_vasin_f16m1(vfloat16m1_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m1(avl);

    vuint16m1_t vx_u  = __riscv_vreinterpret_v_f16m1_u16m1(vx);
    vuint16m1_t vax_u = __riscv_vand_vx_u16m1(vx_u, 0x7FFF, vl);

#ifndef __FAST_MATH__
    vbool16_t mask_gt1  = __riscv_vmsgtu_vx_u16m1_b16(vax_u, 0x3C00, vl);
    vbool16_t mask_qnan = __riscv_vmsgeu_vx_u16m1_b16(vax_u, 0x7E00, vl);
    vbool16_t mask_inv  = __riscv_vmandn_mm_b16(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u16m1(vax_u, 0, mask_gt1, vl);
#endif

    vfloat16m1_t vax   = __riscv_vreinterpret_v_u16m1_f16m1(vax_u);
    vbool16_t is_large = __riscv_vmfgt_vf_f16m1_b16(vax, 0.5f16, vl);

    vfloat16m1_t vhalf     = __riscv_vfmv_v_f_f16m1(0.5f16, vl);
    vfloat16m1_t vhalf_dif = __riscv_vfnmsub_vf_f16m1(vax, 0.5f16, vhalf, vl);
    vfloat16m1_t vy_lrg_hi = __riscv_vfsqrt_v_f16m1(vhalf_dif, vl);

    vfloat16m1_t vS_res    = __riscv_vfnmsub_vv_f16m1(vy_lrg_hi, vy_lrg_hi, vhalf_dif, vl);
    vbool16_t mask_yzero   = __riscv_vmfeq_vf_f16m1_b16(vy_lrg_hi, 0.0f16, vl);
    vfloat16m1_t vy_safe   = __riscv_vfmerge_vfm_f16m1(vy_lrg_hi, 1.0f16, mask_yzero, vl);
    vfloat16m1_t vy_lrg_lo = __riscv_vfmul_vf_f16m1(__riscv_vfdiv_vv_f16m1(vS_res, vy_safe, vl), 0.5f16, vl);

    vfloat16m1_t vy    = __riscv_vmerge_vvm_f16m1(vax, vy_lrg_hi, is_large, vl);
    vfloat16m1_t vy_lo = __riscv_vmerge_vvm_f16m1(__riscv_vfmv_v_f_f16m1(0.0f16, vl), vy_lrg_lo, is_large, vl);

    vfloat16m1_t vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f16m1( vy, vy_lo, vasin_r_hi, vasin_r_lo,  vl);
    
    vfloat16m1_t vlarge_rh     = __riscv_vfadd_vv_f16m1(vasin_r_hi, vasin_r_hi, vl);
    vfloat16m1_t vlarge_rl     = __riscv_vfadd_vv_f16m1(vasin_r_lo, vasin_r_lo, vl);

    vfloat16m1_t vlarge_res_hi = __riscv_vfrsub_vf_f16m1(vlarge_rh, PI_2_HI_16, vl);
    vfloat16m1_t vlarge_err    = __riscv_vfsub_vv_f16m1(__riscv_vfsub_vv_f16m1(
                                    __riscv_vfrsub_vf_f16m1(vlarge_res_hi, PI_2_HI_16, vl), vlarge_rh, vl), vlarge_rl, vl);
    vfloat16m1_t vlarge_res    = __riscv_vfadd_vv_f16m1(vlarge_res_hi, 
                                   __riscv_vfadd_vf_f16m1(vlarge_err, PI_2_LO_16, vl), vl);

    vfloat16m1_t vsmall_res = __riscv_vfadd_vv_f16m1(vasin_r_hi, vasin_r_lo, vl);
    vfloat16m1_t vres       = __riscv_vmerge_vvm_f16m1(vsmall_res, vlarge_res, is_large, vl);
    vres                    = __riscv_vfsgnj_vv_f16m1(vres, vx, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f16m1(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b16(mask_inv, vl) > 0) {
        volatile _Float16 zero = 0.0f16;
        _Float16 nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f16m1(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

vfloat16m1_t __riscv_vacos_f16m1(vfloat16m1_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m1(avl);

    vuint16m1_t vx_u  = __riscv_vreinterpret_v_f16m1_u16m1(vx);
    vuint16m1_t vax_u = __riscv_vand_vx_u16m1(vx_u, 0x7FFFU, vl);
    vbool16_t is_neg  = __riscv_vmslt_vx_i16m1_b16(__riscv_vreinterpret_v_u16m1_i16m1(vx_u), 0, vl);

#ifndef __FAST_MATH__
    vbool16_t mask_gt1  = __riscv_vmsgtu_vx_u16m1_b16(vax_u, 0x3C00U, vl);
    vbool16_t mask_qnan = __riscv_vmsgeu_vx_u16m1_b16(vax_u, 0x7E00U, vl);
    vbool16_t mask_inv  = __riscv_vmandn_mm_b16(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u16m1(vax_u, 0U, mask_gt1, vl);
#endif

    vfloat16m1_t vax = __riscv_vreinterpret_v_u16m1_f16m1(vax_u);

    vbool16_t is_large = __riscv_vmfgt_vf_f16m1_b16(vax, 0.5f16, vl);

    vfloat16m1_t vhalf     = __riscv_vfmv_v_f_f16m1(0.5f16, vl);
    vfloat16m1_t vhalf_dif = __riscv_vfnmsub_vf_f16m1(vax, 0.5f16, vhalf, vl);
    vfloat16m1_t vy_lrg_hi = __riscv_vfsqrt_v_f16m1(vhalf_dif, vl);

    vfloat16m1_t vS_res    = __riscv_vfnmsub_vv_f16m1(vy_lrg_hi, vy_lrg_hi, vhalf_dif, vl);

    vbool16_t mask_yzero   = __riscv_vmfeq_vf_f16m1_b16(vy_lrg_hi, 0.0f16, vl);
    vfloat16m1_t vy_safe   = __riscv_vfmerge_vfm_f16m1(vy_lrg_hi, 1.0f16, mask_yzero, vl);
    vfloat16m1_t vy_lrg_lo = __riscv_vfmul_vf_f16m1(__riscv_vfdiv_vv_f16m1(vS_res, vy_safe, vl), 0.5f16, vl);

    vfloat16m1_t vzero = __riscv_vfmv_v_f_f16m1(0.0f16, vl);
    vfloat16m1_t vy    = __riscv_vmerge_vvm_f16m1(vax, vy_lrg_hi, is_large, vl);
    vfloat16m1_t vy_lo = __riscv_vmerge_vvm_f16m1(vzero, vy_lrg_lo, is_large, vl);

    vfloat16m1_t vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f16m1(vy, vy_lo, vasin_r_hi, vasin_r_lo, vl);

    vfloat16m1_t vscaled_rh = __riscv_vmerge_vvm_f16m1(vasin_r_hi, __riscv_vfadd_vv_f16m1(vasin_r_hi, vasin_r_hi, vl), is_large, vl);
    vfloat16m1_t vscaled_rl = __riscv_vmerge_vvm_f16m1(vasin_r_lo, __riscv_vfadd_vv_f16m1(vasin_r_lo, vasin_r_lo, vl), is_large, vl);

    vfloat16m1_t vsign        = __riscv_vmerge_vvm_f16m1(__riscv_vfsgnjn_vv_f16m1(vx, vx, vl), vx, is_large, vl);
    vfloat16m1_t vs_arcsin_rh = __riscv_vfsgnjx_vv_f16m1(vscaled_rh, vsign, vl);
    vfloat16m1_t vs_arcsin_rl = __riscv_vfsgnjx_vv_f16m1(vscaled_rl, vsign, vl);

    vfloat16m1_t vbase_lrg_hi = __riscv_vfmerge_vfm_f16m1(vzero, PI_HI_16, is_neg, vl);
    vfloat16m1_t vbase_lrg_lo = __riscv_vfmerge_vfm_f16m1(vzero, PI_LO_16, is_neg, vl);
    
    vbool16_t not_large   = __riscv_vmnot_m_b16(is_large, vl);
    vfloat16m1_t vbase_hi = __riscv_vfmerge_vfm_f16m1(vbase_lrg_hi, PI_2_HI_16, not_large, vl);
    vfloat16m1_t vbase_lo = __riscv_vfmerge_vfm_f16m1(vbase_lrg_lo, PI_2_LO_16, not_large, vl);
    
    vfloat16m1_t vres_hi  = __riscv_vfadd_vv_f16m1(vbase_hi, vs_arcsin_rh, vl);
    vfloat16m1_t vv_sub   = __riscv_vfsub_vv_f16m1(vres_hi, vbase_hi, vl);
    vfloat16m1_t vres_err = __riscv_vfsub_vv_f16m1(vbase_hi, __riscv_vfsub_vv_f16m1(vres_hi, vv_sub, vl), vl);
    vres_err              = __riscv_vfadd_vv_f16m1(vres_err, __riscv_vfsub_vv_f16m1(vs_arcsin_rh, vv_sub, vl), vl);
    vfloat16m1_t vres_lo  = __riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(vbase_lo, vs_arcsin_rl, vl), vres_err, vl);

    vfloat16m1_t vres = __riscv_vfadd_vv_f16m1(vres_hi, vres_lo, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f16m1(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b16(mask_inv, vl) > 0) {
        volatile _Float16 zero = 0.0f16;
        _Float16 nan_val = (zero / zero);
        vres = __riscv_vfmerge_vfm_f16m1(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

inline void __riscv_vasin_kernel_f16m2(vfloat16m2_t vy, vfloat16m2_t vy_lo,
                                       vfloat16m2_t &vasin_r_hi, vfloat16m2_t &vasin_r_lo, size_t vl) {
    
    vfloat16m2_t vy2    = __riscv_vfmul_vv_f16m2(vy, vy, vl);
    vfloat16m2_t vy2_lo = __riscv_vfmsub_vv_f16m2(vy, vy, vy2, vl);
    vy2_lo              = __riscv_vfmadd_vv_f16m2(__riscv_vfadd_vv_f16m2(vy, vy, vl), vy_lo, vy2_lo, vl);

    vfloat16m2_t vp  = __riscv_vfmadd_vf_f16m2(vy2, 0x1.9e8p-5f16, __riscv_vfmv_v_f_f16m2(0x1.45p-5f16, vl), vl);
    vp               = __riscv_vfmadd_vv_f16m2(vy2, vp, __riscv_vfmv_v_f_f16m2(0x1.354p-4f16, vl), vl);
    vfloat16m2_t vc3 = __riscv_vfmv_v_f_f16m2(0x1.554p-3f16, vl);
    vfloat16m2_t vp1 = __riscv_vfmadd_vv_f16m2(vy2, vp, vc3, vl);
    vp               = __riscv_vfmadd_vv_f16m2(vy2, vp, __riscv_vfsub_vv_f16m2(vc3, vp1, vl), vl);

    vfloat16m2_t vone = __riscv_vfmv_v_f_f16m2(1.0f16, vl);
    vfloat16m2_t vph  = __riscv_vfmadd_vv_f16m2(vy2, vp1, vone, vl);
    
    vfloat16m2_t vpl  = __riscv_vfmadd_vv_f16m2(vy2, vp1, __riscv_vfsub_vv_f16m2(vone, vph, vl), vl);
    vpl               = __riscv_vfmadd_vv_f16m2(vp1, vy2_lo, __riscv_vfmadd_vv_f16m2(vp, vy2, vpl, vl), vl);

    vasin_r_hi = __riscv_vfmul_vv_f16m2(vph, vy, vl);
    vasin_r_lo = __riscv_vfmsub_vv_f16m2(vph, vy, vasin_r_hi, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f16m2(vph, vy_lo, vasin_r_lo, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f16m2(vpl, vy, vasin_r_lo, vl);
                                      
}

vfloat16m2_t __riscv_vasin_f16m2(vfloat16m2_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m2(avl);

    vuint16m2_t vx_u  = __riscv_vreinterpret_v_f16m2_u16m2(vx);
    vuint16m2_t vax_u = __riscv_vand_vx_u16m2(vx_u, 0x7FFF, vl);

#ifndef __FAST_MATH__
    vbool8_t mask_gt1  = __riscv_vmsgtu_vx_u16m2_b8(vax_u, 0x3C00, vl);
    vbool8_t mask_qnan = __riscv_vmsgeu_vx_u16m2_b8(vax_u, 0x7E00, vl);
    vbool8_t mask_inv  = __riscv_vmandn_mm_b8(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u16m2(vax_u, 0, mask_gt1, vl);
#endif

    vfloat16m2_t vax   = __riscv_vreinterpret_v_u16m2_f16m2(vax_u);
    vbool8_t is_large = __riscv_vmfgt_vf_f16m2_b8(vax, 0.5f16, vl);

    vfloat16m2_t vhalf     = __riscv_vfmv_v_f_f16m2(0.5f16, vl);
    vfloat16m2_t vhalf_dif = __riscv_vfnmsub_vf_f16m2(vax, 0.5f16, vhalf, vl);
    vfloat16m2_t vy_lrg_hi = __riscv_vfsqrt_v_f16m2(vhalf_dif, vl);

    vfloat16m2_t vS_res    = __riscv_vfnmsub_vv_f16m2(vy_lrg_hi, vy_lrg_hi, vhalf_dif, vl);
    vbool8_t mask_yzero   = __riscv_vmfeq_vf_f16m2_b8(vy_lrg_hi, 0.0f16, vl);
    vfloat16m2_t vy_safe   = __riscv_vfmerge_vfm_f16m2(vy_lrg_hi, 1.0f16, mask_yzero, vl);
    vfloat16m2_t vy_lrg_lo = __riscv_vfmul_vf_f16m2(__riscv_vfdiv_vv_f16m2(vS_res, vy_safe, vl), 0.5f16, vl);

    vfloat16m2_t vy    = __riscv_vmerge_vvm_f16m2(vax, vy_lrg_hi, is_large, vl);
    vfloat16m2_t vy_lo = __riscv_vmerge_vvm_f16m2(__riscv_vfmv_v_f_f16m2(0.0f16, vl), vy_lrg_lo, is_large, vl);

    vfloat16m2_t vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f16m2( vy, vy_lo, vasin_r_hi, vasin_r_lo,  vl);
    
    vfloat16m2_t vlarge_rh     = __riscv_vfadd_vv_f16m2(vasin_r_hi, vasin_r_hi, vl);
    vfloat16m2_t vlarge_rl     = __riscv_vfadd_vv_f16m2(vasin_r_lo, vasin_r_lo, vl);

    vfloat16m2_t vlarge_res_hi = __riscv_vfrsub_vf_f16m2(vlarge_rh, PI_2_HI_16, vl);
    vfloat16m2_t vlarge_err    = __riscv_vfsub_vv_f16m2(__riscv_vfsub_vv_f16m2(
                                    __riscv_vfrsub_vf_f16m2(vlarge_res_hi, PI_2_HI_16, vl), vlarge_rh, vl), vlarge_rl, vl);
    vfloat16m2_t vlarge_res    = __riscv_vfadd_vv_f16m2(vlarge_res_hi, 
                                   __riscv_vfadd_vf_f16m2(vlarge_err, PI_2_LO_16, vl), vl);

    vfloat16m2_t vsmall_res = __riscv_vfadd_vv_f16m2(vasin_r_hi, vasin_r_lo, vl);
    vfloat16m2_t vres       = __riscv_vmerge_vvm_f16m2(vsmall_res, vlarge_res, is_large, vl);
    vres                    = __riscv_vfsgnj_vv_f16m2(vres, vx, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f16m2(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b8(mask_inv, vl) > 0) {
        volatile _Float16 zero = 0.0f16;
        _Float16 nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f16m2(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

vfloat16m2_t __riscv_vacos_f16m2(vfloat16m2_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m2(avl);

    vuint16m2_t vx_u  = __riscv_vreinterpret_v_f16m2_u16m2(vx);
    vuint16m2_t vax_u = __riscv_vand_vx_u16m2(vx_u, 0x7FFFU, vl);
    vbool8_t is_neg  = __riscv_vmslt_vx_i16m2_b8(__riscv_vreinterpret_v_u16m2_i16m2(vx_u), 0, vl);

#ifndef __FAST_MATH__
    vbool8_t mask_gt1  = __riscv_vmsgtu_vx_u16m2_b8(vax_u, 0x3C00U, vl);
    vbool8_t mask_qnan = __riscv_vmsgeu_vx_u16m2_b8(vax_u, 0x7E00U, vl);
    vbool8_t mask_inv  = __riscv_vmandn_mm_b8(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u16m2(vax_u, 0U, mask_gt1, vl);
#endif

    vfloat16m2_t vax = __riscv_vreinterpret_v_u16m2_f16m2(vax_u);

    vbool8_t is_large = __riscv_vmfgt_vf_f16m2_b8(vax, 0.5f16, vl);

    vfloat16m2_t vhalf     = __riscv_vfmv_v_f_f16m2(0.5f16, vl);
    vfloat16m2_t vhalf_dif = __riscv_vfnmsub_vf_f16m2(vax, 0.5f16, vhalf, vl);
    vfloat16m2_t vy_lrg_hi = __riscv_vfsqrt_v_f16m2(vhalf_dif, vl);

    vfloat16m2_t vS_res    = __riscv_vfnmsub_vv_f16m2(vy_lrg_hi, vy_lrg_hi, vhalf_dif, vl);

    vbool8_t mask_yzero   = __riscv_vmfeq_vf_f16m2_b8(vy_lrg_hi, 0.0f16, vl);
    vfloat16m2_t vy_safe   = __riscv_vfmerge_vfm_f16m2(vy_lrg_hi, 1.0f16, mask_yzero, vl);
    vfloat16m2_t vy_lrg_lo = __riscv_vfmul_vf_f16m2(__riscv_vfdiv_vv_f16m2(vS_res, vy_safe, vl), 0.5f16, vl);

    vfloat16m2_t vzero = __riscv_vfmv_v_f_f16m2(0.0f16, vl);
    vfloat16m2_t vy    = __riscv_vmerge_vvm_f16m2(vax, vy_lrg_hi, is_large, vl);
    vfloat16m2_t vy_lo = __riscv_vmerge_vvm_f16m2(vzero, vy_lrg_lo, is_large, vl);

    vfloat16m2_t vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f16m2(vy, vy_lo, vasin_r_hi, vasin_r_lo, vl);

    vfloat16m2_t vscaled_rh = __riscv_vmerge_vvm_f16m2(vasin_r_hi, __riscv_vfadd_vv_f16m2(vasin_r_hi, vasin_r_hi, vl), is_large, vl);
    vfloat16m2_t vscaled_rl = __riscv_vmerge_vvm_f16m2(vasin_r_lo, __riscv_vfadd_vv_f16m2(vasin_r_lo, vasin_r_lo, vl), is_large, vl);

    vfloat16m2_t vsign        = __riscv_vmerge_vvm_f16m2(__riscv_vfsgnjn_vv_f16m2(vx, vx, vl), vx, is_large, vl);
    vfloat16m2_t vs_arcsin_rh = __riscv_vfsgnjx_vv_f16m2(vscaled_rh, vsign, vl);
    vfloat16m2_t vs_arcsin_rl = __riscv_vfsgnjx_vv_f16m2(vscaled_rl, vsign, vl);

    vfloat16m2_t vbase_lrg_hi = __riscv_vfmerge_vfm_f16m2(vzero, PI_HI_16, is_neg, vl);
    vfloat16m2_t vbase_lrg_lo = __riscv_vfmerge_vfm_f16m2(vzero, PI_LO_16, is_neg, vl);
    
    vbool8_t not_large   = __riscv_vmnot_m_b8(is_large, vl);
    vfloat16m2_t vbase_hi = __riscv_vfmerge_vfm_f16m2(vbase_lrg_hi, PI_2_HI_16, not_large, vl);
    vfloat16m2_t vbase_lo = __riscv_vfmerge_vfm_f16m2(vbase_lrg_lo, PI_2_LO_16, not_large, vl);
    
    vfloat16m2_t vres_hi  = __riscv_vfadd_vv_f16m2(vbase_hi, vs_arcsin_rh, vl);
    vfloat16m2_t vv_sub   = __riscv_vfsub_vv_f16m2(vres_hi, vbase_hi, vl);
    vfloat16m2_t vres_err = __riscv_vfsub_vv_f16m2(vbase_hi, __riscv_vfsub_vv_f16m2(vres_hi, vv_sub, vl), vl);
    vres_err              = __riscv_vfadd_vv_f16m2(vres_err, __riscv_vfsub_vv_f16m2(vs_arcsin_rh, vv_sub, vl), vl);
    vfloat16m2_t vres_lo  = __riscv_vfadd_vv_f16m2(__riscv_vfadd_vv_f16m2(vbase_lo, vs_arcsin_rl, vl), vres_err, vl);

    vfloat16m2_t vres = __riscv_vfadd_vv_f16m2(vres_hi, vres_lo, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f16m2(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b8(mask_inv, vl) > 0) {
        volatile _Float16 zero = 0.0f16;
        _Float16 nan_val = (zero / zero);
        vres = __riscv_vfmerge_vfm_f16m2(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

inline void __riscv_vasin_kernel_f16m4(vfloat16m4_t vy, vfloat16m4_t vy_lo,
                                       vfloat16m4_t &vasin_r_hi, vfloat16m4_t &vasin_r_lo, size_t vl) {
    
    vfloat16m4_t vy2    = __riscv_vfmul_vv_f16m4(vy, vy, vl);
    vfloat16m4_t vy2_lo = __riscv_vfmsub_vv_f16m4(vy, vy, vy2, vl);
    vy2_lo              = __riscv_vfmadd_vv_f16m4(__riscv_vfadd_vv_f16m4(vy, vy, vl), vy_lo, vy2_lo, vl);

    vfloat16m4_t vp  = __riscv_vfmadd_vf_f16m4(vy2, 0x1.9e8p-5f16, __riscv_vfmv_v_f_f16m4(0x1.45p-5f16, vl), vl);
    vp               = __riscv_vfmadd_vv_f16m4(vy2, vp, __riscv_vfmv_v_f_f16m4(0x1.354p-4f16, vl), vl);
    vfloat16m4_t vc3 = __riscv_vfmv_v_f_f16m4(0x1.554p-3f16, vl);
    vfloat16m4_t vp1 = __riscv_vfmadd_vv_f16m4(vy2, vp, vc3, vl);
    vp               = __riscv_vfmadd_vv_f16m4(vy2, vp, __riscv_vfsub_vv_f16m4(vc3, vp1, vl), vl);

    vfloat16m4_t vone = __riscv_vfmv_v_f_f16m4(1.0f16, vl);
    vfloat16m4_t vph  = __riscv_vfmadd_vv_f16m4(vy2, vp1, vone, vl);
    
    vfloat16m4_t vpl  = __riscv_vfmadd_vv_f16m4(vy2, vp1, __riscv_vfsub_vv_f16m4(vone, vph, vl), vl);
    vpl               = __riscv_vfmadd_vv_f16m4(vp1, vy2_lo, __riscv_vfmadd_vv_f16m4(vp, vy2, vpl, vl), vl);

    vasin_r_hi = __riscv_vfmul_vv_f16m4(vph, vy, vl);
    vasin_r_lo = __riscv_vfmsub_vv_f16m4(vph, vy, vasin_r_hi, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f16m4(vph, vy_lo, vasin_r_lo, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f16m4(vpl, vy, vasin_r_lo, vl);
                                      
}

vfloat16m4_t __riscv_vasin_f16m4(vfloat16m4_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m4(avl);

    vuint16m4_t vx_u  = __riscv_vreinterpret_v_f16m4_u16m4(vx);
    vuint16m4_t vax_u = __riscv_vand_vx_u16m4(vx_u, 0x7FFF, vl);

#ifndef __FAST_MATH__
    vbool4_t mask_gt1  = __riscv_vmsgtu_vx_u16m4_b4(vax_u, 0x3C00, vl);
    vbool4_t mask_qnan = __riscv_vmsgeu_vx_u16m4_b4(vax_u, 0x7E00, vl);
    vbool4_t mask_inv  = __riscv_vmandn_mm_b4(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u16m4(vax_u, 0, mask_gt1, vl);
#endif

    vfloat16m4_t vax   = __riscv_vreinterpret_v_u16m4_f16m4(vax_u);
    vbool4_t is_large = __riscv_vmfgt_vf_f16m4_b4(vax, 0.5f16, vl);

    vfloat16m4_t vhalf     = __riscv_vfmv_v_f_f16m4(0.5f16, vl);
    vfloat16m4_t vhalf_dif = __riscv_vfnmsub_vf_f16m4(vax, 0.5f16, vhalf, vl);
    vfloat16m4_t vy_lrg_hi = __riscv_vfsqrt_v_f16m4(vhalf_dif, vl);

    vfloat16m4_t vS_res    = __riscv_vfnmsub_vv_f16m4(vy_lrg_hi, vy_lrg_hi, vhalf_dif, vl);
    vbool4_t mask_yzero   = __riscv_vmfeq_vf_f16m4_b4(vy_lrg_hi, 0.0f16, vl);
    vfloat16m4_t vy_safe   = __riscv_vfmerge_vfm_f16m4(vy_lrg_hi, 1.0f16, mask_yzero, vl);
    vfloat16m4_t vy_lrg_lo = __riscv_vfmul_vf_f16m4(__riscv_vfdiv_vv_f16m4(vS_res, vy_safe, vl), 0.5f16, vl);

    vfloat16m4_t vy    = __riscv_vmerge_vvm_f16m4(vax, vy_lrg_hi, is_large, vl);
    vfloat16m4_t vy_lo = __riscv_vmerge_vvm_f16m4(__riscv_vfmv_v_f_f16m4(0.0f16, vl), vy_lrg_lo, is_large, vl);

    vfloat16m4_t vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f16m4( vy, vy_lo, vasin_r_hi, vasin_r_lo,  vl);
    
    vfloat16m4_t vlarge_rh     = __riscv_vfadd_vv_f16m4(vasin_r_hi, vasin_r_hi, vl);
    vfloat16m4_t vlarge_rl     = __riscv_vfadd_vv_f16m4(vasin_r_lo, vasin_r_lo, vl);

    vfloat16m4_t vlarge_res_hi = __riscv_vfrsub_vf_f16m4(vlarge_rh, PI_2_HI_16, vl);
    vfloat16m4_t vlarge_err    = __riscv_vfsub_vv_f16m4(__riscv_vfsub_vv_f16m4(
                                    __riscv_vfrsub_vf_f16m4(vlarge_res_hi, PI_2_HI_16, vl), vlarge_rh, vl), vlarge_rl, vl);
    vfloat16m4_t vlarge_res    = __riscv_vfadd_vv_f16m4(vlarge_res_hi, 
                                   __riscv_vfadd_vf_f16m4(vlarge_err, PI_2_LO_16, vl), vl);

    vfloat16m4_t vsmall_res = __riscv_vfadd_vv_f16m4(vasin_r_hi, vasin_r_lo, vl);
    vfloat16m4_t vres       = __riscv_vmerge_vvm_f16m4(vsmall_res, vlarge_res, is_large, vl);
    vres                    = __riscv_vfsgnj_vv_f16m4(vres, vx, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f16m4(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b4(mask_inv, vl) > 0) {
        volatile _Float16 zero = 0.0f16;
        _Float16 nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f16m4(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

vfloat16m4_t __riscv_vacos_f16m4(vfloat16m4_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m4(avl);

    vuint16m4_t vx_u  = __riscv_vreinterpret_v_f16m4_u16m4(vx);
    vuint16m4_t vax_u = __riscv_vand_vx_u16m4(vx_u, 0x7FFFU, vl);
    vbool4_t is_neg  = __riscv_vmslt_vx_i16m4_b4(__riscv_vreinterpret_v_u16m4_i16m4(vx_u), 0, vl);

#ifndef __FAST_MATH__
    vbool4_t mask_gt1  = __riscv_vmsgtu_vx_u16m4_b4(vax_u, 0x3C00U, vl);
    vbool4_t mask_qnan = __riscv_vmsgeu_vx_u16m4_b4(vax_u, 0x7E00U, vl);
    vbool4_t mask_inv  = __riscv_vmandn_mm_b4(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u16m4(vax_u, 0U, mask_gt1, vl);
#endif

    vfloat16m4_t vax = __riscv_vreinterpret_v_u16m4_f16m4(vax_u);

    vbool4_t is_large = __riscv_vmfgt_vf_f16m4_b4(vax, 0.5f16, vl);

    vfloat16m4_t vhalf     = __riscv_vfmv_v_f_f16m4(0.5f16, vl);
    vfloat16m4_t vhalf_dif = __riscv_vfnmsub_vf_f16m4(vax, 0.5f16, vhalf, vl);
    vfloat16m4_t vy_lrg_hi = __riscv_vfsqrt_v_f16m4(vhalf_dif, vl);

    vfloat16m4_t vS_res    = __riscv_vfnmsub_vv_f16m4(vy_lrg_hi, vy_lrg_hi, vhalf_dif, vl);

    vbool4_t mask_yzero   = __riscv_vmfeq_vf_f16m4_b4(vy_lrg_hi, 0.0f16, vl);
    vfloat16m4_t vy_safe   = __riscv_vfmerge_vfm_f16m4(vy_lrg_hi, 1.0f16, mask_yzero, vl);
    vfloat16m4_t vy_lrg_lo = __riscv_vfmul_vf_f16m4(__riscv_vfdiv_vv_f16m4(vS_res, vy_safe, vl), 0.5f16, vl);

    vfloat16m4_t vzero = __riscv_vfmv_v_f_f16m4(0.0f16, vl);
    vfloat16m4_t vy    = __riscv_vmerge_vvm_f16m4(vax, vy_lrg_hi, is_large, vl);
    vfloat16m4_t vy_lo = __riscv_vmerge_vvm_f16m4(vzero, vy_lrg_lo, is_large, vl);

    vfloat16m4_t vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f16m4(vy, vy_lo, vasin_r_hi, vasin_r_lo, vl);

    vfloat16m4_t vscaled_rh = __riscv_vmerge_vvm_f16m4(vasin_r_hi, __riscv_vfadd_vv_f16m4(vasin_r_hi, vasin_r_hi, vl), is_large, vl);
    vfloat16m4_t vscaled_rl = __riscv_vmerge_vvm_f16m4(vasin_r_lo, __riscv_vfadd_vv_f16m4(vasin_r_lo, vasin_r_lo, vl), is_large, vl);

    vfloat16m4_t vsign        = __riscv_vmerge_vvm_f16m4(__riscv_vfsgnjn_vv_f16m4(vx, vx, vl), vx, is_large, vl);
    vfloat16m4_t vs_arcsin_rh = __riscv_vfsgnjx_vv_f16m4(vscaled_rh, vsign, vl);
    vfloat16m4_t vs_arcsin_rl = __riscv_vfsgnjx_vv_f16m4(vscaled_rl, vsign, vl);

    vfloat16m4_t vbase_lrg_hi = __riscv_vfmerge_vfm_f16m4(vzero, PI_HI_16, is_neg, vl);
    vfloat16m4_t vbase_lrg_lo = __riscv_vfmerge_vfm_f16m4(vzero, PI_LO_16, is_neg, vl);
    
    vbool4_t not_large   = __riscv_vmnot_m_b4(is_large, vl);
    vfloat16m4_t vbase_hi = __riscv_vfmerge_vfm_f16m4(vbase_lrg_hi, PI_2_HI_16, not_large, vl);
    vfloat16m4_t vbase_lo = __riscv_vfmerge_vfm_f16m4(vbase_lrg_lo, PI_2_LO_16, not_large, vl);
    
    vfloat16m4_t vres_hi  = __riscv_vfadd_vv_f16m4(vbase_hi, vs_arcsin_rh, vl);
    vfloat16m4_t vv_sub   = __riscv_vfsub_vv_f16m4(vres_hi, vbase_hi, vl);
    vfloat16m4_t vres_err = __riscv_vfsub_vv_f16m4(vbase_hi, __riscv_vfsub_vv_f16m4(vres_hi, vv_sub, vl), vl);
    vres_err              = __riscv_vfadd_vv_f16m4(vres_err, __riscv_vfsub_vv_f16m4(vs_arcsin_rh, vv_sub, vl), vl);
    vfloat16m4_t vres_lo  = __riscv_vfadd_vv_f16m4(__riscv_vfadd_vv_f16m4(vbase_lo, vs_arcsin_rl, vl), vres_err, vl);

    vfloat16m4_t vres = __riscv_vfadd_vv_f16m4(vres_hi, vres_lo, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f16m4(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b4(mask_inv, vl) > 0) {
        volatile _Float16 zero = 0.0f16;
        _Float16 nan_val = (zero / zero);
        vres = __riscv_vfmerge_vfm_f16m4(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

vfloat16m8_t __riscv_vasin_f16m8(vfloat16m8_t x, size_t avl)
{ 
    vfloat16m8_t res;
    size_t vl       = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1              = __riscv_vasin_f16m4(x1, vl);
    res             = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if(avl > vl){
        vl  = __riscv_vsetvl_e16m4(avl-vl);
        x1  = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1  = __riscv_vasin_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}

vfloat16m8_t __riscv_vacos_f16m8(vfloat16m8_t x, size_t avl)
{ 
    vfloat16m8_t res;
    size_t vl       = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1              = __riscv_vacos_f16m4(x1, vl);
    res             = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if(avl > vl){
        vl  = __riscv_vsetvl_e16m4(avl-vl);
        x1  = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1  = __riscv_vacos_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}

#endif /* __riscv_zvfh */

#endif /* __riscv_v_intrinsic */
