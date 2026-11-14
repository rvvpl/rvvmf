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
 *  file: asin_acos_f32.cpp                              *
 *  Contains: vector arcsin and arccos for               *
 *            IEEE binary32(f32) via RVV 1.0 intrinsics  *
 *                                                       * 
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the arcsine of input vector V                *
 *                                                       *
 * Algorithm:                                            *
 *    1) Special cases: NaN, |x|>1                       *
 *    2) Reduction to [0, sqrt(2)/2]:                    *
 *       asin(x) = pi/2-asin(sqrt(1-x^2))                *
 *    3) Table reduction on sin grid T_k:                *
 *       r = x*sqrt(1-T^2) - T*sqrt(1-x^2),              *
 *       result = asin(T_k) + asin_poly(r)               *
 *                                                       *
 * Computes the arccos of input vector V                 *
 *                                                       *
 * Algorithm:                                            *
 *    1) Special cases: NaN, |x|>1                       *
 *    2) Reduction to [0, sqrt(2)/2]:                    *
 *       acos(x) = pi-asin(sqrt(1-x^2))                  *
 *                                                       *
 * Note that this implementation provides at most        *
 * 0.501 ulp.                                            *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#include <stdint.h>

const int N_f = 16;
// --- Ch (sqrt(1 - T^2)) ---
const float Ch_f32[ 16 ] = {
    0x1p0f,          0x1.ff6fecp-1f,  0x1.fdbebap-1f,  0x1.fae988p-1f,
    0x1.f6eb62p-1f,  0x1.f1bd28p-1f,  0x1.eb5538p-1f,  0x1.e3a71cp-1f,
    0x1.daa2fep-1f,  0x1.d034f4p-1f,  0x1.c443f2p-1f,  0x1.b6b06p-1f,
    0x1.a751fap-1f,  0x1.95f4a6p-1f,  0x1.825388p-1f,  0x1.6c10ep-1f
};
// --- Cl (sqrt(1 - T^2)) ---
const float Cl_f32[ 16 ] = {
    0.0f,            -0x1.16d006p-27f,  0x1.22f87ep-26f, -0x1.57d6dp-26f,
    0x1.a4ee62p-26f, -0x1.1bff4ep-27f, -0x1.3af2cap-27f, -0x1.914e12p-28f,
    0x1.f55c3ap-26f,  0x1.34c10ap-28f, -0x1.596ea8p-28f, -0x1.2d328cp-26f,
   -0x1.77091cp-26f,  0x1.37b36ap-27f, -0x1.d4747ep-27f,  0x1.53cbacp-26f
};
// --- Ah (arcsin(T)) ---
const float Ah_f32[ 16 ] = {
    0.0f,            0x1.80240ap-5f,  0x1.809092p-4f,  0x1.20f53p-3f,
    0x1.82494ep-3f,  0x1.e481cp-3f,   0x1.23f052p-2f,  0x1.565774p-2f,
    0x1.899f4ep-2f,  0x1.bdf6f4p-2f,  0x1.f3958ap-2f,  0x1.155e8cp-1f,
    0x1.31df4p-1f,   0x1.4f7fd2p-1f,  0x1.6e8254p-1f,  0x1.8f3fb2p-1f
};
// --- Al (arcsin(T)) ---
const float Al_f32[ 16 ] = {
    0.0f,             -0x1.c049eap-30f,  0x1.227ca6p-29f,  0x1.84661p-30f,
    0x1.a1cf2p-28f,    0x1.f9ce26p-28f,  0x1.e2ee16p-29f,  0x1.96cdep-27f,
    0x1.b92c5ap-27f,   0x1.eb9a42p-28f,  0x1.d9bbdep-27f, -0x1.abfff6p-26f,
    0x1.f7a63ap-26f,   0x1.785f66p-26f, -0x1.f0cefep-27f, -0x1.636d2ap-26f
};


#define PI_2_HI_F  0x1.921fb6p+0f   
#define PI_2_LO_F -0x1.777a5cp-25f  
#define PI_HI_F    0x1.921fb6p+1f   
#define PI_LO_F   -0x1.777a5cp-24f  
#define THRESHOLD_F 0x1.6a09e6p-1f

inline void __riscv_vasin_kernel_f32m1(vfloat32m1_t vu_hi, vfloat32m1_t vu_lo,
                                       vfloat32m1_t vsqrt_u_hi, vfloat32m1_t vsqrt_u_lo, 
                                       vfloat32m1_t &vA_hi, vfloat32m1_t &vA_lo,
                                       vfloat32m1_t &vasin_r_hi, vfloat32m1_t &vasin_r_lo, size_t vl) {
    
    vfloat32m1_t vscaled_u = __riscv_vfmadd_vf_f32m1(vu_hi, 0x1.555556p+4f, __riscv_vfmv_v_f_f32m1(0.5f, vl), vl);
    vuint32m1_t vidx       = __riscv_vfcvt_rtz_xu_f_v_u32m1(vscaled_u, vl);
    vidx                   = __riscv_vminu_vx_u32m1(vidx, 15, vl);
    
    vfloat32m1_t vT     = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_xu_v_f32m1(vidx, vl), 0x1.8p-5f, vl);
    vuint32m1_t voffset = __riscv_vsll_vx_u32m1(vidx, 2, vl);
    vfloat32m1_t vC_hi  = __riscv_vluxei32_v_f32m1(Ch_f32, voffset, vl);
    vfloat32m1_t vC_lo  = __riscv_vluxei32_v_f32m1(Cl_f32, voffset, vl);
    vA_hi               = __riscv_vluxei32_v_f32m1(Ah_f32, voffset, vl);
    vA_lo               = __riscv_vluxei32_v_f32m1(Al_f32, voffset, vl);

    vfloat32m1_t vuC_hi   = __riscv_vfmul_vv_f32m1(vu_hi, vC_hi, vl);
    vfloat32m1_t vuC_lo   = __riscv_vfmsub_vv_f32m1(vu_hi, vC_hi, vuC_hi, vl);

    vfloat32m1_t vprod_hi = __riscv_vfmul_vv_f32m1(vT, vsqrt_u_hi, vl);
    vfloat32m1_t vprod_lo = __riscv_vfmsub_vv_f32m1(vT, vsqrt_u_hi, vprod_hi, vl);
    vfloat32m1_t vr_base  = __riscv_vfsub_vv_f32m1(vuC_hi, vprod_hi, vl);

    vfloat32m1_t vtot_err = __riscv_vfsub_vv_f32m1(vuC_lo, vprod_lo, vl);
    vtot_err              = __riscv_vfnmsub_vv_f32m1(vT, vsqrt_u_lo, vtot_err, vl);
    vtot_err              = __riscv_vfmadd_vv_f32m1(vC_hi, vu_lo, vtot_err, vl);
    vtot_err              = __riscv_vfmadd_vv_f32m1(vu_hi, vC_lo, vtot_err, vl);

    vfloat32m1_t vr  = __riscv_vfadd_vv_f32m1(vr_base, vtot_err, vl);
    vfloat32m1_t vrl = __riscv_vfsub_vv_f32m1(vtot_err, __riscv_vfsub_vv_f32m1(vr, vr_base, vl), vl);

    vfloat32m1_t vr2 = __riscv_vfmadd_vv_f32m1(vr, vr, __riscv_vfmv_v_f_f32m1(0x1p-50f, vl), vl);
    vfloat32m1_t vp  = __riscv_vfmadd_vf_f32m1(vr2, 0x1.6e799cp-5f, __riscv_vfmv_v_f_f32m1(0x1.33331cp-4f, vl), vl);
    vp               = __riscv_vfmadd_vv_f32m1(vr2, vp, __riscv_vfmv_v_f_f32m1(0x1.555556p-3f, vl), vl);

    vfloat32m1_t vone    = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t vph = __riscv_vfmadd_vv_f32m1(vr2, vp, vone, vl);
    vfloat32m1_t vpl = __riscv_vfmadd_vv_f32m1(vr2, vp, __riscv_vfsub_vv_f32m1(vone, vph, vl), vl);

    vasin_r_hi = __riscv_vfmul_vv_f32m1(vph, vr, vl);
    vasin_r_lo = __riscv_vfmsub_vv_f32m1(vph, vr, vasin_r_hi, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f32m1(vph, vrl, vasin_r_lo, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f32m1(vpl, vr, vasin_r_lo, vl);

}

vfloat32m1_t __riscv_vasin_f32m1(vfloat32m1_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m1(avl);

    vuint32m1_t vx_u  = __riscv_vreinterpret_v_f32m1_u32m1(vx);
    vuint32m1_t vax_u = __riscv_vand_vx_u32m1(vx_u, 0x7FFFFFFFU, vl);

#ifndef __FAST_MATH__
    vbool32_t mask_gt1  = __riscv_vmsgtu_vx_u32m1_b32(vax_u, 0x3F800000U, vl);
    vbool32_t mask_qnan = __riscv_vmsgeu_vx_u32m1_b32(vax_u, 0x7FC00000U, vl);
    vbool32_t mask_inv  = __riscv_vmandn_mm_b32(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u32m1(vax_u, 0U, mask_gt1, vl);
#endif

    vfloat32m1_t vax   = __riscv_vreinterpret_v_u32m1_f32m1(vax_u);
    vbool32_t is_large = __riscv_vmfgt_vf_f32m1_b32(vax, THRESHOLD_F, vl);

    vfloat32m1_t vone    = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t vw_hi   = __riscv_vfnmsub_vv_f32m1(vax, vax, vone, vl);
    vfloat32m1_t vy_hi   = __riscv_vfsqrt_v_f32m1(vw_hi, vl);

    vfloat32m1_t vs_tmp  = __riscv_vfnmsub_vv_f32m1(vy_hi, vy_hi, vone, vl);
    vfloat32m1_t vS_res  = __riscv_vfnmsub_vv_f32m1(vax, vax, vs_tmp, vl);

    vfloat32m1_t vx2_hi  = __riscv_vfmul_vv_f32m1(vax, vax, vl);
    vfloat32m1_t vinv_2y = __riscv_vfmadd_vf_f32m1(vx2_hi, 0.25f, __riscv_vfmv_v_f_f32m1(0.5f, vl), vl);
    vfloat32m1_t vy_lo   = __riscv_vfmul_vv_f32m1(vS_res, vinv_2y, vl);

    vfloat32m1_t vzero      = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vu_hi      = __riscv_vmerge_vvm_f32m1(vax, vy_hi, is_large, vl);
    vfloat32m1_t vu_lo      = __riscv_vmerge_vvm_f32m1(vzero, vy_lo, is_large, vl);
    vfloat32m1_t vsqrt_u_hi = __riscv_vmerge_vvm_f32m1(vy_hi, vax, is_large, vl);
    vfloat32m1_t vsqrt_u_lo = __riscv_vmerge_vvm_f32m1(vy_lo, vzero, is_large, vl);

    vfloat32m1_t vA_hi, vA_lo, vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f32m1( vu_hi, vu_lo, vsqrt_u_hi, vsqrt_u_lo, vA_hi, vA_lo,
                                vasin_r_hi, vasin_r_lo,  vl);
    
    vfloat32m1_t vsign        = __riscv_vfmerge_vfm_f32m1(vone, -1.0f, is_large, vl);
    vfloat32m1_t vsAh         = __riscv_vfsgnjx_vv_f32m1(vA_hi, vsign, vl);
    vfloat32m1_t vsAl         = __riscv_vfsgnjx_vv_f32m1(vA_lo, vsign, vl);
    vfloat32m1_t vs_arcsin_rh = __riscv_vfsgnjx_vv_f32m1(vasin_r_hi, vsign, vl);
    vfloat32m1_t vs_arcsin_rl = __riscv_vfsgnjx_vv_f32m1(vasin_r_lo, vsign, vl);

    vfloat32m1_t vbase_hi = __riscv_vfmerge_vfm_f32m1(vzero, PI_2_HI_F, is_large, vl);
    vfloat32m1_t vbase_lo = __riscv_vfmerge_vfm_f32m1(vzero, PI_2_LO_F, is_large, vl);

    vfloat32m1_t vsum_hi  = __riscv_vfadd_vv_f32m1(vbase_hi, vsAh, vl);
    vfloat32m1_t vsum_err = __riscv_vfadd_vv_f32m1(__riscv_vfsub_vv_f32m1(vbase_hi, vsum_hi, vl), vsAh, vl);
    vfloat32m1_t vsum_lo  = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(vbase_lo, vsAl, vl), vsum_err, vl);

    vfloat32m1_t vres_hi  = __riscv_vfadd_vv_f32m1(vsum_hi, vs_arcsin_rh, vl);
    vfloat32m1_t vres_err = __riscv_vfadd_vv_f32m1(__riscv_vfsub_vv_f32m1(vsum_hi, vres_hi, vl), vs_arcsin_rh, vl);
    vfloat32m1_t vres_lo  = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(vsum_lo, vs_arcsin_rl, vl), vres_err, vl);

    vfloat32m1_t vres = __riscv_vfadd_vv_f32m1(vres_hi, vres_lo, vl);
    vres              = __riscv_vfsgnj_vv_f32m1(vres, vx, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f32m1(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b32(mask_inv, vl) > 0) {
        volatile float zero = 0.0f;
        float nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f32m1(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

vfloat32m1_t __riscv_vacos_f32m1(vfloat32m1_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m1(avl);

    vuint32m1_t vx_u  = __riscv_vreinterpret_v_f32m1_u32m1(vx);
    vuint32m1_t vax_u = __riscv_vand_vx_u32m1(vx_u, 0x7FFFFFFFU, vl);
    vbool32_t is_neg  = __riscv_vmslt_vx_i32m1_b32(__riscv_vreinterpret_v_u32m1_i32m1(vx_u), 0, vl);

#ifndef __FAST_MATH__
    vbool32_t mask_gt1  = __riscv_vmsgtu_vx_u32m1_b32(vax_u, 0x3F800000U, vl);
    vbool32_t mask_qnan = __riscv_vmsgeu_vx_u32m1_b32(vax_u, 0x7FC00000U, vl);
    vbool32_t mask_inv  = __riscv_vmandn_mm_b32(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u32m1(vax_u, 0U, mask_gt1, vl);
#endif

    vfloat32m1_t vax = __riscv_vreinterpret_v_u32m1_f32m1(vax_u);

    vbool32_t is_large = __riscv_vmfgt_vf_f32m1_b32(vax, THRESHOLD_F, vl);

    vfloat32m1_t vx2_hi = __riscv_vfmul_vv_f32m1(vax, vax, vl);
    vfloat32m1_t vx2_lo = __riscv_vfmsub_vv_f32m1(vax, vax, vx2_hi, vl);

    vfloat32m1_t vw_hi  = __riscv_vfrsub_vf_f32m1(vx2_hi, 1.0f, vl);
    vfloat32m1_t vw_err = __riscv_vfsub_vv_f32m1(__riscv_vfrsub_vf_f32m1(vw_hi, 1.0f, vl), vx2_hi, vl);
    vfloat32m1_t vw_lo  = __riscv_vfsub_vv_f32m1(vw_err, vx2_lo, vl);

    vfloat32m1_t vy_hi  = __riscv_vfsqrt_v_f32m1(vw_hi, vl);
    vfloat32m1_t vs_res = __riscv_vfnmsub_vv_f32m1(vy_hi, vy_hi, vw_hi, vl);
    vs_res              = __riscv_vfadd_vv_f32m1(vs_res, vw_lo, vl);

    vbool32_t mask_yzero = __riscv_vmfeq_vf_f32m1_b32(vy_hi, 0.0f, vl);
    vfloat32m1_t vy_safe = __riscv_vfmerge_vfm_f32m1(vy_hi, 1.0f, mask_yzero, vl);
    vfloat32m1_t vinv_2y = __riscv_vfrdiv_vf_f32m1(vy_safe, 0.5f, vl);
    vfloat32m1_t vy_lo   = __riscv_vfmul_vv_f32m1(vs_res, vinv_2y, vl);

    vfloat32m1_t vzero      = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vu_hi      = __riscv_vmerge_vvm_f32m1(vax, vy_hi, is_large, vl);
    vfloat32m1_t vu_lo      = __riscv_vmerge_vvm_f32m1(vzero, vy_lo, is_large, vl);
    vfloat32m1_t vsqrt_u_hi = __riscv_vmerge_vvm_f32m1(vy_hi, vax, is_large, vl);
    vfloat32m1_t vsqrt_u_lo = __riscv_vmerge_vvm_f32m1(vy_lo, vzero, is_large, vl);

    vfloat32m1_t vA_hi, vA_lo, vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f32m1(vu_hi, vu_lo, vsqrt_u_hi, vsqrt_u_lo,
                               vA_hi, vA_lo, vasin_r_hi, vasin_r_lo, vl);

    vfloat32m1_t vsign     = __riscv_vmerge_vvm_f32m1(__riscv_vfsgnjn_vv_f32m1(vx, vx, vl), vx, is_large, vl);
    vfloat32m1_t vsA_hi    = __riscv_vfsgnjx_vv_f32m1(vA_hi, vsign, vl);
    vfloat32m1_t vsA_lo    = __riscv_vfsgnjx_vv_f32m1(vA_lo, vsign, vl);
    vfloat32m1_t vs_asin_h = __riscv_vfsgnjx_vv_f32m1(vasin_r_hi, vsign, vl);
    vfloat32m1_t vs_asin_l = __riscv_vfsgnjx_vv_f32m1(vasin_r_lo, vsign, vl);

    vfloat32m1_t vbase_lrg_hi = __riscv_vfmerge_vfm_f32m1(vzero, PI_HI_F, is_neg, vl);
    vfloat32m1_t vbase_lrg_lo = __riscv_vfmerge_vfm_f32m1(vzero, PI_LO_F, is_neg, vl);
    vfloat32m1_t vbase_hi     = __riscv_vmerge_vvm_f32m1(__riscv_vfmv_v_f_f32m1(PI_2_HI_F, vl), vbase_lrg_hi, is_large, vl);
    vfloat32m1_t vbase_lo     = __riscv_vmerge_vvm_f32m1(__riscv_vfmv_v_f_f32m1(PI_2_LO_F, vl), vbase_lrg_lo, is_large, vl);

    vfloat32m1_t vsum1_hi  = __riscv_vfadd_vv_f32m1(vbase_hi, vsA_hi, vl);
    vfloat32m1_t vv_sum1   = __riscv_vfsub_vv_f32m1(vsum1_hi, vbase_hi, vl);
    vfloat32m1_t vsum1_err = __riscv_vfsub_vv_f32m1(vbase_hi, __riscv_vfsub_vv_f32m1(vsum1_hi, vv_sum1, vl), vl);
    vsum1_err              = __riscv_vfadd_vv_f32m1(vsum1_err, __riscv_vfsub_vv_f32m1(vsA_hi, vv_sum1, vl), vl);
    vfloat32m1_t vsum1_lo  = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(vbase_lo, vsA_lo, vl), vsum1_err, vl);

    vfloat32m1_t vres_hi   = __riscv_vfadd_vv_f32m1(vsum1_hi, vs_asin_h, vl);
    vfloat32m1_t vres_err  = __riscv_vfadd_vv_f32m1(__riscv_vfsub_vv_f32m1(vsum1_hi, vres_hi, vl), vs_asin_h, vl);
    vfloat32m1_t vres_lo   = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(vsum1_lo, vs_asin_l, vl), vres_err, vl);

    vfloat32m1_t vres = __riscv_vfadd_vv_f32m1(vres_hi, vres_lo, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f32m1(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b32(mask_inv, vl) > 0) {
        volatile float zero = 0.0f;
        float nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f32m1(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

inline void __riscv_vasin_kernel_f32m2(vfloat32m2_t vu_hi, vfloat32m2_t vu_lo,
                                       vfloat32m2_t vsqrt_u_hi, vfloat32m2_t vsqrt_u_lo, 
                                       vfloat32m2_t &vA_hi, vfloat32m2_t &vA_lo,
                                       vfloat32m2_t &vasin_r_hi, vfloat32m2_t &vasin_r_lo, size_t vl) {
    
    vfloat32m2_t vscaled_u = __riscv_vfmadd_vf_f32m2(vu_hi, 0x1.555556p+4f, __riscv_vfmv_v_f_f32m2(0.5f, vl), vl);
    vuint32m2_t vidx       = __riscv_vfcvt_rtz_xu_f_v_u32m2(vscaled_u, vl);
    vidx                   = __riscv_vminu_vx_u32m2(vidx, 15, vl);
    
    vfloat32m2_t vT     = __riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_xu_v_f32m2(vidx, vl), 0x1.8p-5f, vl);
    vuint32m2_t voffset = __riscv_vsll_vx_u32m2(vidx, 2, vl);
    vfloat32m2_t vC_hi  = __riscv_vluxei32_v_f32m2(Ch_f32, voffset, vl);
    vfloat32m2_t vC_lo  = __riscv_vluxei32_v_f32m2(Cl_f32, voffset, vl);
    vA_hi               = __riscv_vluxei32_v_f32m2(Ah_f32, voffset, vl);
    vA_lo               = __riscv_vluxei32_v_f32m2(Al_f32, voffset, vl);

    vfloat32m2_t vuC_hi   = __riscv_vfmul_vv_f32m2(vu_hi, vC_hi, vl);
    vfloat32m2_t vuC_lo   = __riscv_vfmsub_vv_f32m2(vu_hi, vC_hi, vuC_hi, vl);

    vfloat32m2_t vprod_hi = __riscv_vfmul_vv_f32m2(vT, vsqrt_u_hi, vl);
    vfloat32m2_t vprod_lo = __riscv_vfmsub_vv_f32m2(vT, vsqrt_u_hi, vprod_hi, vl);
    vfloat32m2_t vr_base  = __riscv_vfsub_vv_f32m2(vuC_hi, vprod_hi, vl);

    vfloat32m2_t vtot_err = __riscv_vfsub_vv_f32m2(vuC_lo, vprod_lo, vl);
    vtot_err              = __riscv_vfnmsub_vv_f32m2(vT, vsqrt_u_lo, vtot_err, vl);
    vtot_err              = __riscv_vfmadd_vv_f32m2(vC_hi, vu_lo, vtot_err, vl);
    vtot_err              = __riscv_vfmadd_vv_f32m2(vu_hi, vC_lo, vtot_err, vl);

    vfloat32m2_t vr  = __riscv_vfadd_vv_f32m2(vr_base, vtot_err, vl);
    vfloat32m2_t vrl = __riscv_vfsub_vv_f32m2(vtot_err, __riscv_vfsub_vv_f32m2(vr, vr_base, vl), vl);

    vfloat32m2_t vr2 = __riscv_vfmadd_vv_f32m2(vr, vr, __riscv_vfmv_v_f_f32m2(0x1p-50f, vl), vl);
    vfloat32m2_t vp  = __riscv_vfmadd_vf_f32m2(vr2, 0x1.6e799cp-5f, __riscv_vfmv_v_f_f32m2(0x1.33331cp-4f, vl), vl);
    vp               = __riscv_vfmadd_vv_f32m2(vr2, vp, __riscv_vfmv_v_f_f32m2(0x1.555556p-3f, vl), vl);

    vfloat32m2_t vone    = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    vfloat32m2_t vph = __riscv_vfmadd_vv_f32m2(vr2, vp, vone, vl);
    vfloat32m2_t vpl = __riscv_vfmadd_vv_f32m2(vr2, vp, __riscv_vfsub_vv_f32m2(vone, vph, vl), vl);

    vasin_r_hi = __riscv_vfmul_vv_f32m2(vph, vr, vl);
    vasin_r_lo = __riscv_vfmsub_vv_f32m2(vph, vr, vasin_r_hi, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f32m2(vph, vrl, vasin_r_lo, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f32m2(vpl, vr, vasin_r_lo, vl);

}

vfloat32m2_t __riscv_vasin_f32m2(vfloat32m2_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m2(avl);

    vuint32m2_t vx_u  = __riscv_vreinterpret_v_f32m2_u32m2(vx);
    vuint32m2_t vax_u = __riscv_vand_vx_u32m2(vx_u, 0x7FFFFFFFU, vl);

#ifndef __FAST_MATH__
    vbool16_t mask_gt1  = __riscv_vmsgtu_vx_u32m2_b16(vax_u, 0x3F800000U, vl);
    vbool16_t mask_qnan = __riscv_vmsgeu_vx_u32m2_b16(vax_u, 0x7FC00000U, vl);
    vbool16_t mask_inv  = __riscv_vmandn_mm_b16(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u32m2(vax_u, 0U, mask_gt1, vl);
#endif

    vfloat32m2_t vax   = __riscv_vreinterpret_v_u32m2_f32m2(vax_u);
    vbool16_t is_large = __riscv_vmfgt_vf_f32m2_b16(vax, THRESHOLD_F, vl);

    vfloat32m2_t vone    = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    vfloat32m2_t vw_hi   = __riscv_vfnmsub_vv_f32m2(vax, vax, vone, vl);
    vfloat32m2_t vy_hi   = __riscv_vfsqrt_v_f32m2(vw_hi, vl);

    vfloat32m2_t vs_tmp  = __riscv_vfnmsub_vv_f32m2(vy_hi, vy_hi, vone, vl);
    vfloat32m2_t vS_res  = __riscv_vfnmsub_vv_f32m2(vax, vax, vs_tmp, vl);

    vfloat32m2_t vx2_hi  = __riscv_vfmul_vv_f32m2(vax, vax, vl);
    vfloat32m2_t vinv_2y = __riscv_vfmadd_vf_f32m2(vx2_hi, 0.25f, __riscv_vfmv_v_f_f32m2(0.5f, vl), vl);
    vfloat32m2_t vy_lo   = __riscv_vfmul_vv_f32m2(vS_res, vinv_2y, vl);

    vfloat32m2_t vzero      = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t vu_hi      = __riscv_vmerge_vvm_f32m2(vax, vy_hi, is_large, vl);
    vfloat32m2_t vu_lo      = __riscv_vmerge_vvm_f32m2(vzero, vy_lo, is_large, vl);
    vfloat32m2_t vsqrt_u_hi = __riscv_vmerge_vvm_f32m2(vy_hi, vax, is_large, vl);
    vfloat32m2_t vsqrt_u_lo = __riscv_vmerge_vvm_f32m2(vy_lo, vzero, is_large, vl);

    vfloat32m2_t vA_hi, vA_lo, vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f32m2( vu_hi, vu_lo, vsqrt_u_hi, vsqrt_u_lo, vA_hi, vA_lo,
                                vasin_r_hi, vasin_r_lo,  vl);
    
    vfloat32m2_t vsign        = __riscv_vfmerge_vfm_f32m2(vone, -1.0f, is_large, vl);
    vfloat32m2_t vsAh         = __riscv_vfsgnjx_vv_f32m2(vA_hi, vsign, vl);
    vfloat32m2_t vsAl         = __riscv_vfsgnjx_vv_f32m2(vA_lo, vsign, vl);
    vfloat32m2_t vs_arcsin_rh = __riscv_vfsgnjx_vv_f32m2(vasin_r_hi, vsign, vl);
    vfloat32m2_t vs_arcsin_rl = __riscv_vfsgnjx_vv_f32m2(vasin_r_lo, vsign, vl);

    vfloat32m2_t vbase_hi = __riscv_vfmerge_vfm_f32m2(vzero, PI_2_HI_F, is_large, vl);
    vfloat32m2_t vbase_lo = __riscv_vfmerge_vfm_f32m2(vzero, PI_2_LO_F, is_large, vl);

    vfloat32m2_t vsum_hi  = __riscv_vfadd_vv_f32m2(vbase_hi, vsAh, vl);
    vfloat32m2_t vsum_err = __riscv_vfadd_vv_f32m2(__riscv_vfsub_vv_f32m2(vbase_hi, vsum_hi, vl), vsAh, vl);
    vfloat32m2_t vsum_lo  = __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(vbase_lo, vsAl, vl), vsum_err, vl);

    vfloat32m2_t vres_hi  = __riscv_vfadd_vv_f32m2(vsum_hi, vs_arcsin_rh, vl);
    vfloat32m2_t vres_err = __riscv_vfadd_vv_f32m2(__riscv_vfsub_vv_f32m2(vsum_hi, vres_hi, vl), vs_arcsin_rh, vl);
    vfloat32m2_t vres_lo  = __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(vsum_lo, vs_arcsin_rl, vl), vres_err, vl);

    vfloat32m2_t vres = __riscv_vfadd_vv_f32m2(vres_hi, vres_lo, vl);
    vres              = __riscv_vfsgnj_vv_f32m2(vres, vx, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f32m2(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b16(mask_inv, vl) > 0) {
        volatile float zero = 0.0f;
        float nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f32m2(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

vfloat32m2_t __riscv_vacos_f32m2(vfloat32m2_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m2(avl);

    vuint32m2_t vx_u  = __riscv_vreinterpret_v_f32m2_u32m2(vx);
    vuint32m2_t vax_u = __riscv_vand_vx_u32m2(vx_u, 0x7FFFFFFFU, vl);
    vbool16_t is_neg  = __riscv_vmslt_vx_i32m2_b16(__riscv_vreinterpret_v_u32m2_i32m2(vx_u), 0, vl);

#ifndef __FAST_MATH__
    vbool16_t mask_gt1  = __riscv_vmsgtu_vx_u32m2_b16(vax_u, 0x3F800000U, vl);
    vbool16_t mask_qnan = __riscv_vmsgeu_vx_u32m2_b16(vax_u, 0x7FC00000U, vl);
    vbool16_t mask_inv  = __riscv_vmandn_mm_b16(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u32m2(vax_u, 0U, mask_gt1, vl);
#endif

    vfloat32m2_t vax = __riscv_vreinterpret_v_u32m2_f32m2(vax_u);

    vbool16_t is_large = __riscv_vmfgt_vf_f32m2_b16(vax, THRESHOLD_F, vl);

    vfloat32m2_t vx2_hi = __riscv_vfmul_vv_f32m2(vax, vax, vl);
    vfloat32m2_t vx2_lo = __riscv_vfmsub_vv_f32m2(vax, vax, vx2_hi, vl);

    vfloat32m2_t vw_hi  = __riscv_vfrsub_vf_f32m2(vx2_hi, 1.0f, vl);
    vfloat32m2_t vw_err = __riscv_vfsub_vv_f32m2(__riscv_vfrsub_vf_f32m2(vw_hi, 1.0f, vl), vx2_hi, vl);
    vfloat32m2_t vw_lo  = __riscv_vfsub_vv_f32m2(vw_err, vx2_lo, vl);

    vfloat32m2_t vy_hi  = __riscv_vfsqrt_v_f32m2(vw_hi, vl);
    vfloat32m2_t vs_res = __riscv_vfnmsub_vv_f32m2(vy_hi, vy_hi, vw_hi, vl);
    vs_res              = __riscv_vfadd_vv_f32m2(vs_res, vw_lo, vl);

    vbool16_t mask_yzero = __riscv_vmfeq_vf_f32m2_b16(vy_hi, 0.0f, vl);
    vfloat32m2_t vy_safe = __riscv_vfmerge_vfm_f32m2(vy_hi, 1.0f, mask_yzero, vl);
    vfloat32m2_t vinv_2y = __riscv_vfrdiv_vf_f32m2(vy_safe, 0.5f, vl);
    vfloat32m2_t vy_lo   = __riscv_vfmul_vv_f32m2(vs_res, vinv_2y, vl);

    vfloat32m2_t vzero      = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t vu_hi      = __riscv_vmerge_vvm_f32m2(vax, vy_hi, is_large, vl);
    vfloat32m2_t vu_lo      = __riscv_vmerge_vvm_f32m2(vzero, vy_lo, is_large, vl);
    vfloat32m2_t vsqrt_u_hi = __riscv_vmerge_vvm_f32m2(vy_hi, vax, is_large, vl);
    vfloat32m2_t vsqrt_u_lo = __riscv_vmerge_vvm_f32m2(vy_lo, vzero, is_large, vl);

    vfloat32m2_t vA_hi, vA_lo, vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f32m2(vu_hi, vu_lo, vsqrt_u_hi, vsqrt_u_lo,
                               vA_hi, vA_lo, vasin_r_hi, vasin_r_lo, vl);

    vfloat32m2_t vsign     = __riscv_vmerge_vvm_f32m2(__riscv_vfsgnjn_vv_f32m2(vx, vx, vl), vx, is_large, vl);
    vfloat32m2_t vsA_hi    = __riscv_vfsgnjx_vv_f32m2(vA_hi, vsign, vl);
    vfloat32m2_t vsA_lo    = __riscv_vfsgnjx_vv_f32m2(vA_lo, vsign, vl);
    vfloat32m2_t vs_asin_h = __riscv_vfsgnjx_vv_f32m2(vasin_r_hi, vsign, vl);
    vfloat32m2_t vs_asin_l = __riscv_vfsgnjx_vv_f32m2(vasin_r_lo, vsign, vl);

    vfloat32m2_t vbase_lrg_hi = __riscv_vfmerge_vfm_f32m2(vzero, PI_HI_F, is_neg, vl);
    vfloat32m2_t vbase_lrg_lo = __riscv_vfmerge_vfm_f32m2(vzero, PI_LO_F, is_neg, vl);
    vfloat32m2_t vbase_hi     = __riscv_vmerge_vvm_f32m2(__riscv_vfmv_v_f_f32m2(PI_2_HI_F, vl), vbase_lrg_hi, is_large, vl);
    vfloat32m2_t vbase_lo     = __riscv_vmerge_vvm_f32m2(__riscv_vfmv_v_f_f32m2(PI_2_LO_F, vl), vbase_lrg_lo, is_large, vl);

    vfloat32m2_t vsum2_hi  = __riscv_vfadd_vv_f32m2(vbase_hi, vsA_hi, vl);
    vfloat32m2_t vv_sum2   = __riscv_vfsub_vv_f32m2(vsum2_hi, vbase_hi, vl);
    vfloat32m2_t vsum2_err = __riscv_vfsub_vv_f32m2(vbase_hi, __riscv_vfsub_vv_f32m2(vsum2_hi, vv_sum2, vl), vl);
    vsum2_err              = __riscv_vfadd_vv_f32m2(vsum2_err, __riscv_vfsub_vv_f32m2(vsA_hi, vv_sum2, vl), vl);
    vfloat32m2_t vsum2_lo  = __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(vbase_lo, vsA_lo, vl), vsum2_err, vl);

    vfloat32m2_t vres_hi   = __riscv_vfadd_vv_f32m2(vsum2_hi, vs_asin_h, vl);
    vfloat32m2_t vres_err  = __riscv_vfadd_vv_f32m2(__riscv_vfsub_vv_f32m2(vsum2_hi, vres_hi, vl), vs_asin_h, vl);
    vfloat32m2_t vres_lo   = __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(vsum2_lo, vs_asin_l, vl), vres_err, vl);

    vfloat32m2_t vres = __riscv_vfadd_vv_f32m2(vres_hi, vres_lo, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f32m2(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b16(mask_inv, vl) > 0) {
        volatile float zero = 0.0f;
        float nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f32m2(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

inline void __riscv_vasin_kernel_f32m4(vfloat32m4_t vu_hi, vfloat32m4_t vu_lo,
                                       vfloat32m4_t vsqrt_u_hi, vfloat32m4_t vsqrt_u_lo, 
                                       vfloat32m4_t &vA_hi, vfloat32m4_t &vA_lo,
                                       vfloat32m4_t &vasin_r_hi, vfloat32m4_t &vasin_r_lo, size_t vl) {
    
    vfloat32m4_t vscaled_u = __riscv_vfmadd_vf_f32m4(vu_hi, 0x1.555556p+4f, __riscv_vfmv_v_f_f32m4(0.5f, vl), vl);
    vuint32m4_t vidx       = __riscv_vfcvt_rtz_xu_f_v_u32m4(vscaled_u, vl);
    vidx                   = __riscv_vminu_vx_u32m4(vidx, 15, vl);
    
    
    vfloat32m4_t vT     = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_xu_v_f32m4(vidx, vl), 0x1.8p-5f, vl);
    vuint32m4_t voffset = __riscv_vsll_vx_u32m4(vidx, 2, vl);
    vfloat32m4_t vC_hi  = __riscv_vluxei32_v_f32m4(Ch_f32, voffset, vl);
    vfloat32m4_t vC_lo  = __riscv_vluxei32_v_f32m4(Cl_f32, voffset, vl);
    vA_hi               = __riscv_vluxei32_v_f32m4(Ah_f32, voffset, vl);
    vA_lo               = __riscv_vluxei32_v_f32m4(Al_f32, voffset, vl);

    vfloat32m4_t vuC_hi   = __riscv_vfmul_vv_f32m4(vu_hi, vC_hi, vl);
    vfloat32m4_t vuC_lo   = __riscv_vfmsub_vv_f32m4(vu_hi, vC_hi, vuC_hi, vl);

    vfloat32m4_t vprod_hi = __riscv_vfmul_vv_f32m4(vT, vsqrt_u_hi, vl);
    vfloat32m4_t vprod_lo = __riscv_vfmsub_vv_f32m4(vT, vsqrt_u_hi, vprod_hi, vl);
    vfloat32m4_t vr_base  = __riscv_vfsub_vv_f32m4(vuC_hi, vprod_hi, vl);

    vfloat32m4_t vtot_err = __riscv_vfsub_vv_f32m4(vuC_lo, vprod_lo, vl);
    vtot_err              = __riscv_vfnmsub_vv_f32m4(vT, vsqrt_u_lo, vtot_err, vl);
    vtot_err              = __riscv_vfmadd_vv_f32m4(vC_hi, vu_lo, vtot_err, vl);
    vtot_err              = __riscv_vfmadd_vv_f32m4(vu_hi, vC_lo, vtot_err, vl);

    vfloat32m4_t vr  = __riscv_vfadd_vv_f32m4(vr_base, vtot_err, vl);
    vfloat32m4_t vrl = __riscv_vfsub_vv_f32m4(vtot_err, __riscv_vfsub_vv_f32m4(vr, vr_base, vl), vl);

    vfloat32m4_t vr2 = __riscv_vfmadd_vv_f32m4(vr, vr, __riscv_vfmv_v_f_f32m4(0x1p-50f, vl), vl);
    vfloat32m4_t vp  = __riscv_vfmadd_vf_f32m4(vr2, 0x1.6e799cp-5f, __riscv_vfmv_v_f_f32m4(0x1.33331cp-4f, vl), vl);
    vp               = __riscv_vfmadd_vv_f32m4(vr2, vp, __riscv_vfmv_v_f_f32m4(0x1.555556p-3f, vl), vl);

    vfloat32m4_t vone = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t vph  = __riscv_vfmadd_vv_f32m4(vr2, vp, vone, vl);
    vfloat32m4_t vpl  = __riscv_vfmadd_vv_f32m4(vr2, vp, __riscv_vfsub_vv_f32m4(vone, vph, vl), vl);

    vasin_r_hi = __riscv_vfmul_vv_f32m4(vph, vr, vl);
    vasin_r_lo = __riscv_vfmsub_vv_f32m4(vph, vr, vasin_r_hi, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f32m4(vph, vrl, vasin_r_lo, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f32m4(vpl, vr, vasin_r_lo, vl);

}

vfloat32m4_t __riscv_vasin_f32m4(vfloat32m4_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m4(avl);

    vuint32m4_t vx_u  = __riscv_vreinterpret_v_f32m4_u32m4(vx);
    vuint32m4_t vax_u = __riscv_vand_vx_u32m4(vx_u, 0x7FFFFFFFU, vl);

#ifndef __FAST_MATH__
    vbool8_t mask_gt1  = __riscv_vmsgtu_vx_u32m4_b8(vax_u, 0x3F800000U, vl);
    vbool8_t mask_qnan = __riscv_vmsgeu_vx_u32m4_b8(vax_u, 0x7FC00000U, vl);
    vbool8_t mask_inv  = __riscv_vmandn_mm_b8(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u32m4(vax_u, 0U, mask_gt1, vl);
#endif

    vfloat32m4_t vax   = __riscv_vreinterpret_v_u32m4_f32m4(vax_u);
    vbool8_t is_large = __riscv_vmfgt_vf_f32m4_b8(vax, THRESHOLD_F, vl);

    vfloat32m4_t vone    = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t vw_hi   = __riscv_vfnmsub_vv_f32m4(vax, vax, vone, vl);
    vfloat32m4_t vy_hi   = __riscv_vfsqrt_v_f32m4(vw_hi, vl);

    vfloat32m4_t vs_tmp  = __riscv_vfnmsub_vv_f32m4(vy_hi, vy_hi, vone, vl);
    vfloat32m4_t vS_res  = __riscv_vfnmsub_vv_f32m4(vax, vax, vs_tmp, vl);

    vfloat32m4_t vx2_hi  = __riscv_vfmul_vv_f32m4(vax, vax, vl);
    vfloat32m4_t vinv_2y = __riscv_vfmadd_vf_f32m4(vx2_hi, 0.25f, __riscv_vfmv_v_f_f32m4(0.5f, vl), vl);
    vfloat32m4_t vy_lo   = __riscv_vfmul_vv_f32m4(vS_res, vinv_2y, vl);

    vfloat32m4_t vzero      = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vu_hi      = __riscv_vmerge_vvm_f32m4(vax, vy_hi, is_large, vl);
    vfloat32m4_t vu_lo      = __riscv_vmerge_vvm_f32m4(vzero, vy_lo, is_large, vl);
    vfloat32m4_t vsqrt_u_hi = __riscv_vmerge_vvm_f32m4(vy_hi, vax, is_large, vl);
    vfloat32m4_t vsqrt_u_lo = __riscv_vmerge_vvm_f32m4(vy_lo, vzero, is_large, vl);

    vfloat32m4_t vA_hi, vA_lo, vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f32m4( vu_hi, vu_lo, vsqrt_u_hi, vsqrt_u_lo, vA_hi, vA_lo,
                                vasin_r_hi, vasin_r_lo,  vl);
    
    vfloat32m4_t vsign        = __riscv_vfmerge_vfm_f32m4(vone, -1.0f, is_large, vl);
    vfloat32m4_t vsAh         = __riscv_vfsgnjx_vv_f32m4(vA_hi, vsign, vl);
    vfloat32m4_t vsAl         = __riscv_vfsgnjx_vv_f32m4(vA_lo, vsign, vl);
    vfloat32m4_t vs_arcsin_rh = __riscv_vfsgnjx_vv_f32m4(vasin_r_hi, vsign, vl);
    vfloat32m4_t vs_arcsin_rl = __riscv_vfsgnjx_vv_f32m4(vasin_r_lo, vsign, vl);

    vfloat32m4_t vbase_hi = __riscv_vfmerge_vfm_f32m4(vzero, PI_2_HI_F, is_large, vl);
    vfloat32m4_t vbase_lo = __riscv_vfmerge_vfm_f32m4(vzero, PI_2_LO_F, is_large, vl);

    vfloat32m4_t vsum_hi  = __riscv_vfadd_vv_f32m4(vbase_hi, vsAh, vl);
    vfloat32m4_t vsum_err = __riscv_vfadd_vv_f32m4(__riscv_vfsub_vv_f32m4(vbase_hi, vsum_hi, vl), vsAh, vl);
    vfloat32m4_t vsum_lo  = __riscv_vfadd_vv_f32m4(__riscv_vfadd_vv_f32m4(vbase_lo, vsAl, vl), vsum_err, vl);

    vfloat32m4_t vres_hi  = __riscv_vfadd_vv_f32m4(vsum_hi, vs_arcsin_rh, vl);
    vfloat32m4_t vres_err = __riscv_vfadd_vv_f32m4(__riscv_vfsub_vv_f32m4(vsum_hi, vres_hi, vl), vs_arcsin_rh, vl);
    vfloat32m4_t vres_lo  = __riscv_vfadd_vv_f32m4(__riscv_vfadd_vv_f32m4(vsum_lo, vs_arcsin_rl, vl), vres_err, vl);

    vfloat32m4_t vres = __riscv_vfadd_vv_f32m4(vres_hi, vres_lo, vl);
    vres              = __riscv_vfsgnj_vv_f32m4(vres, vx, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f32m4(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b8(mask_inv, vl) > 0) {
        volatile float zero = 0.0f;
        float nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f32m4(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

vfloat32m4_t __riscv_vacos_f32m4(vfloat32m4_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m4(avl);

    vuint32m4_t vx_u  = __riscv_vreinterpret_v_f32m4_u32m4(vx);
    vuint32m4_t vax_u = __riscv_vand_vx_u32m4(vx_u, 0x7FFFFFFFU, vl);
    vbool8_t is_neg  = __riscv_vmslt_vx_i32m4_b8(__riscv_vreinterpret_v_u32m4_i32m4(vx_u), 0, vl);

#ifndef __FAST_MATH__
    vbool8_t mask_gt1  = __riscv_vmsgtu_vx_u32m4_b8(vax_u, 0x3F800000U, vl);
    vbool8_t mask_qnan = __riscv_vmsgeu_vx_u32m4_b8(vax_u, 0x7FC00000U, vl);
    vbool8_t mask_inv  = __riscv_vmandn_mm_b8(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u32m4(vax_u, 0U, mask_gt1, vl);
#endif

    vfloat32m4_t vax = __riscv_vreinterpret_v_u32m4_f32m4(vax_u);

    vbool8_t is_large = __riscv_vmfgt_vf_f32m4_b8(vax, THRESHOLD_F, vl);

    vfloat32m4_t vx2_hi = __riscv_vfmul_vv_f32m4(vax, vax, vl);
    vfloat32m4_t vx2_lo = __riscv_vfmsub_vv_f32m4(vax, vax, vx2_hi, vl);

    vfloat32m4_t vw_hi  = __riscv_vfrsub_vf_f32m4(vx2_hi, 1.0f, vl);
    vfloat32m4_t vw_err = __riscv_vfsub_vv_f32m4(__riscv_vfrsub_vf_f32m4(vw_hi, 1.0f, vl), vx2_hi, vl);
    vfloat32m4_t vw_lo  = __riscv_vfsub_vv_f32m4(vw_err, vx2_lo, vl);

    vfloat32m4_t vy_hi  = __riscv_vfsqrt_v_f32m4(vw_hi, vl);
    vfloat32m4_t vs_res = __riscv_vfnmsub_vv_f32m4(vy_hi, vy_hi, vw_hi, vl);
    vs_res              = __riscv_vfadd_vv_f32m4(vs_res, vw_lo, vl);

    vbool8_t mask_yzero = __riscv_vmfeq_vf_f32m4_b8(vy_hi, 0.0f, vl);
    vfloat32m4_t vy_safe = __riscv_vfmerge_vfm_f32m4(vy_hi, 1.0f, mask_yzero, vl);
    vfloat32m4_t vinv_2y = __riscv_vfrdiv_vf_f32m4(vy_safe, 0.5f, vl);
    vfloat32m4_t vy_lo   = __riscv_vfmul_vv_f32m4(vs_res, vinv_2y, vl);

    vfloat32m4_t vzero      = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vu_hi      = __riscv_vmerge_vvm_f32m4(vax, vy_hi, is_large, vl);
    vfloat32m4_t vu_lo      = __riscv_vmerge_vvm_f32m4(vzero, vy_lo, is_large, vl);
    vfloat32m4_t vsqrt_u_hi = __riscv_vmerge_vvm_f32m4(vy_hi, vax, is_large, vl);
    vfloat32m4_t vsqrt_u_lo = __riscv_vmerge_vvm_f32m4(vy_lo, vzero, is_large, vl);

    vfloat32m4_t vA_hi, vA_lo, vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f32m4(vu_hi, vu_lo, vsqrt_u_hi, vsqrt_u_lo,
                               vA_hi, vA_lo, vasin_r_hi, vasin_r_lo, vl);

    vfloat32m4_t vsign     = __riscv_vmerge_vvm_f32m4(__riscv_vfsgnjn_vv_f32m4(vx, vx, vl), vx, is_large, vl);
    vfloat32m4_t vsA_hi    = __riscv_vfsgnjx_vv_f32m4(vA_hi, vsign, vl);
    vfloat32m4_t vsA_lo    = __riscv_vfsgnjx_vv_f32m4(vA_lo, vsign, vl);
    vfloat32m4_t vs_asin_h = __riscv_vfsgnjx_vv_f32m4(vasin_r_hi, vsign, vl);
    vfloat32m4_t vs_asin_l = __riscv_vfsgnjx_vv_f32m4(vasin_r_lo, vsign, vl);

    vfloat32m4_t vbase_lrg_hi = __riscv_vfmerge_vfm_f32m4(vzero, PI_HI_F, is_neg, vl);
    vfloat32m4_t vbase_lrg_lo = __riscv_vfmerge_vfm_f32m4(vzero, PI_LO_F, is_neg, vl);
    vfloat32m4_t vbase_hi     = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(PI_2_HI_F, vl), vbase_lrg_hi, is_large, vl);
    vfloat32m4_t vbase_lo     = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(PI_2_LO_F, vl), vbase_lrg_lo, is_large, vl);

    vfloat32m4_t vsum4_hi  = __riscv_vfadd_vv_f32m4(vbase_hi, vsA_hi, vl);
    vfloat32m4_t vv_sum4   = __riscv_vfsub_vv_f32m4(vsum4_hi, vbase_hi, vl);
    vfloat32m4_t vsum4_err = __riscv_vfsub_vv_f32m4(vbase_hi, __riscv_vfsub_vv_f32m4(vsum4_hi, vv_sum4, vl), vl);
    vsum4_err              = __riscv_vfadd_vv_f32m4(vsum4_err, __riscv_vfsub_vv_f32m4(vsA_hi, vv_sum4, vl), vl);
    vfloat32m4_t vsum4_lo  = __riscv_vfadd_vv_f32m4(__riscv_vfadd_vv_f32m4(vbase_lo, vsA_lo, vl), vsum4_err, vl);

    vfloat32m4_t vres_hi   = __riscv_vfadd_vv_f32m4(vsum4_hi, vs_asin_h, vl);
    vfloat32m4_t vres_err  = __riscv_vfadd_vv_f32m4(__riscv_vfsub_vv_f32m4(vsum4_hi, vres_hi, vl), vs_asin_h, vl);
    vfloat32m4_t vres_lo   = __riscv_vfadd_vv_f32m4(__riscv_vfadd_vv_f32m4(vsum4_lo, vs_asin_l, vl), vres_err, vl);

    vfloat32m4_t vres = __riscv_vfadd_vv_f32m4(vres_hi, vres_lo, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f32m4(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b8(mask_inv, vl) > 0) {
        volatile float zero = 0.0f;
        float nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f32m4(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

vfloat32m8_t __riscv_vasin_f32m8(vfloat32m8_t x, size_t avl)
{ 
    vfloat32m8_t res;
    size_t vl       = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1              = __riscv_vasin_f32m4(x1, vl);
    res             = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if(avl > vl){
        vl  = __riscv_vsetvl_e32m4(avl-vl);
        x1  = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1  = __riscv_vasin_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}

vfloat32m8_t __riscv_vacos_f32m8(vfloat32m8_t x, size_t avl)
{ 
    vfloat32m8_t res;
    size_t vl       = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1              = __riscv_vacos_f32m4(x1, vl);
    res             = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if(avl > vl){
        vl  = __riscv_vsetvl_e32m4(avl-vl);
        x1  = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1  = __riscv_vacos_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}

#endif /* __riscv_v_intrinsic */
