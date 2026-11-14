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
 *  file: asin_acos_f64.cpp                              *
 *  Contains: vector arcsin and arccos for               *
 *            IEEE binary64(f64) via RVV 1.0 intrinsics  *
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

const int N_d = 24;
// --- Ch (sqrt(1 - T^2)) ---
const double Ch_f64[24] = {
    0x1p0,                 0x1.ffbffbff7fecp-1,   0x1.feffbfdfebf1fp-1,  0x1.fdbeba917c3f5p-1,
    0x1.fbfbf7ebc755fp-1,  0x1.f9b61d023725p-1,   0x1.f6eb62d27730dp-1,  0x1.f3998f1b1886cp-1,
    0x1.efbdeb14f4edap-1,  0x1.eb5537b1434dap-1,  0x1.e65b9edeba38ep-1,  0x1.e0cca12e97895p-1,
    0x1.daa2fefaae1d8p-1,  0x1.d3d89be176072p-1,  0x1.cc665b0328622p-1,  0x1.c443f1d4d22afp-1,
    0x1.bb67ae8584caap-1,  0x1.b1c62db2564fep-1,  0x1.a751f9447b724p-1,  0x1.9bfb076d236ebp-1,
    0x1.8fae0c15ad38ap-1,  0x1.8253878ae2e09p-1,  0x1.73ce704fb7b23p-1,  0x1.63fa3f3c02962p-1
};
// --- Cl (sqrt(1 - T^2)) ---
const double Cl_f64[24] = {
     0.0,                   -0x1.c05410835ab2ep-56,  0x1.5dee51994f18bp-55,  0x1.2c0681a46a556p-55,
    -0x1.b2a94084da0b6p-55,  0x1.d3ca3915d1a44p-55, -0x1.401d95ca1ce34p-55,  0x1.6b699b6f7882ep-55,
    -0x1.3a145fe1be078p-55, -0x1.c180d47e8730cp-55, -0x1.bb73251e8c364p-57, -0x1.316d1acdf7b57p-55,
    -0x1.3fe0e03f44594p-58,  0x1.54fe1c5f17367p-56, -0x1.1baa4d369f814p-55, -0x1.dea6ecd25e5e4p-57,
     0x1.cec95d0b5c1e3p-55,  0x1.ed2e2c1a79b97p-55,  0x1.2b909477e9ed1p-56, -0x1.a7391cb03223bp-56,
    -0x1.db7adb6817f6dp-57, -0x1.dd13b570fbe2fp-55,  0x1.1470b816b17a6p-55, -0x1.dfd1505ee685ep-59
};
// --- Ah (arcsin(T)) ---
const double Ah_f64[24] = {
     0.0,                   0x1.000aabde0b9c8p-5,  0x1.002abde953619p-4,  0x1.809092913e52ep-4,
     0x1.00abe0c129e1ep-3,  0x1.41510cb011423p-3,  0x1.82494ed0e78fcp-3,  0x1.c3a6f13aae84bp-3,
     0x1.02be9ce0b87cdp-2,  0x1.23f0523c5dc2bp-2,  0x1.457393b90e2aap-2,  0x1.675441329986ep-2,
     0x1.899f4edc962d3p-2,  0x1.ac62fec0b2a92p-2,  0x1.cfaf27460fe9fp-2,  0x1.f3958aecddef4p-2,
     0x1.0c152382d7366p-1,  0x1.1ec230c714a96p-1,  0x1.31df40fbd31cdp-1,  0x1.457bf318fe517p-1,
     0x1.59aad71ced00fp-1,  0x1.6e825383cc40bp-1,  0x1.841deb5114bb4p-1,  0x1.9aa01babef75ep-1
};
// --- Al (arcsin(T)) ---
const double Al_f64[24] = {
     0.0,                     0x1.d6d94551be3e9p-61,  0x1.182e2dc6ddeedp-58,  0x1.cf6b1f9befb16p-60,
     0x1.7ceb0ee49d42ap-60,  -0x1.15d675180eda8p-58, -0x1.443c2697a7d2fp-57, -0x1.7739d10fe8bc1p-57,
     0x1.e5d09da2e0f04p-58,   0x1.4fc2674a3d6b2p-59,  0x1.b1f64d329fe98p-56,  0x1.d027ed2bb2edap-56,
     0x1.3e919701b7c6dp-60,   0x1.cb9f9a052f11fp-56, -0x1.8bf75f355f723p-57, -0x1.fc135930a7786p-58,
     -0x1.ee6913347c2a6p-55,  0x1.41dc77911b08cp-55,  0x1.10ebcfd1cc29dp-60, -0x1.6189642d67942p-55,
     -0x1.b5b31565e9408p-58,  0x1.1a9ec7321e76ap-56, -0x1.49d1c4e2eba5ep-55, -0x1.df24530d8ffb9p-55
};


#define PI_2_HI   0x1.921fb54442d18p0  
#define PI_2_LO   0x1.1a62633145c07p-54
#define PI_HI     0x1.921fb54442d18p+1
#define PI_LO     0x1.1a62633145c07p-53
#define THRESHOLD 0x1.6a09e667f3bcdp-1

inline void __riscv_vasin_kernel_f64m1(vfloat64m1_t vu_hi, vfloat64m1_t vu_lo,
                                       vfloat64m1_t vsqrt_u_hi, vfloat64m1_t vsqrt_u_lo, 
                                       vfloat64m1_t &vA_hi, vfloat64m1_t &vA_lo,
                                       vfloat64m1_t &vasin_r_hi, vfloat64m1_t &vasin_r_lo, size_t vl) {
    vfloat64m1_t vidx_f = __riscv_vfmadd_vf_f64m1(vu_hi, 32.0, __riscv_vfmv_v_f_f64m1(0.5, vl), vl);
    vuint64m1_t vidx    = __riscv_vfcvt_rtz_xu_f_v_u64m1(vidx_f, vl);
    vidx                = __riscv_vminu_vx_u64m1(vidx, 23, vl);
    
    vfloat64m1_t vT     = __riscv_vfmul_vf_f64m1(__riscv_vfcvt_f_xu_v_f64m1(vidx, vl), 0x1p-5, vl);
    vuint64m1_t voffset = __riscv_vsll_vx_u64m1(vidx, 3, vl);
    vfloat64m1_t vC_hi  = __riscv_vluxei64_v_f64m1(Ch_f64, voffset, vl);
    vfloat64m1_t vC_lo  = __riscv_vluxei64_v_f64m1(Cl_f64, voffset, vl);
    vA_hi               = __riscv_vluxei64_v_f64m1(Ah_f64, voffset, vl);
    vA_lo               = __riscv_vluxei64_v_f64m1(Al_f64, voffset, vl);

    vfloat64m1_t vp2_hi = __riscv_vfmul_vv_f64m1(vT, vsqrt_u_hi, vl);
    vfloat64m1_t vp2_lo = __riscv_vfmsub_vv_f64m1(vT, vsqrt_u_hi, vp2_hi, vl);
    vp2_lo              = __riscv_vfmadd_vv_f64m1(vT, vsqrt_u_lo, vp2_lo, vl);

    vfloat64m1_t vp1_hi = __riscv_vfmul_vv_f64m1(vu_hi, vC_hi, vl);
    vfloat64m1_t vp1_lo = __riscv_vfmsub_vv_f64m1(vu_hi, vC_hi, vp1_hi, vl);
    vp1_lo              = __riscv_vfmadd_vv_f64m1(vu_lo, vC_hi, vp1_lo, vl);
    vp1_lo              = __riscv_vfmadd_vv_f64m1(vu_hi, vC_lo, vp1_lo, vl);
    
    vfloat64m1_t vr_raw_hi = __riscv_vfsub_vv_f64m1(vp1_hi, vp2_hi, vl);
    vfloat64m1_t vr_sub    = __riscv_vfsub_vv_f64m1(vr_raw_hi, vp1_hi, vl);
    vfloat64m1_t vr_err    = __riscv_vfsub_vv_f64m1(vp1_hi, __riscv_vfsub_vv_f64m1(vr_raw_hi, vr_sub, vl), vl);
    vr_err                 = __riscv_vfsub_vv_f64m1(vr_err, __riscv_vfadd_vv_f64m1(vp2_hi, vr_sub, vl), vl);
    vfloat64m1_t vr_raw_lo = __riscv_vfadd_vv_f64m1(vr_err, __riscv_vfsub_vv_f64m1(vp1_lo, vp2_lo, vl), vl);

    vfloat64m1_t vr_hi = __riscv_vfadd_vv_f64m1(vr_raw_hi, vr_raw_lo, vl);
    vfloat64m1_t vr_lo = __riscv_vfadd_vv_f64m1(__riscv_vfsub_vv_f64m1(vr_raw_hi, vr_hi, vl), vr_raw_lo, vl);
    
    vfloat64m1_t vr2 = __riscv_vfmadd_vv_f64m1(vr_hi, vr_hi, __riscv_vfmv_v_f_f64m1(0x1p-120, vl), vl);
    vfloat64m1_t vr4 = __riscv_vfmul_vv_f64m1(vr2, vr2, vl);

    vfloat64m1_t vq0   = __riscv_vfmadd_vf_f64m1(vr2, 0x1.3333333333321p-4, 
                                                 __riscv_vfmv_v_f_f64m1(0x1.5555555555555p-3, vl), vl);
    vfloat64m1_t vq1   = __riscv_vfmadd_vf_f64m1(vr2, 0x1.f1c7109674b33p-6, 
                                                 __riscv_vfmv_v_f_f64m1(0x1.6db6db6e0e4ep-5, vl), vl);
    vfloat64m1_t vq2   = __riscv_vfmadd_vf_f64m1(vr4, 0x1.6eea780121fdfp-6, vq1, vl);
    vfloat64m1_t vpoly = __riscv_vfmadd_vv_f64m1(vr4, vq2, vq0, vl);

    vfloat64m1_t vr2_poly_hi = __riscv_vfmul_vv_f64m1(vr2, vpoly, vl);
    vfloat64m1_t vr2_poly_lo = __riscv_vfmsub_vv_f64m1(vr2, vpoly, vr2_poly_hi, vl);
    vr2_poly_lo              = __riscv_vfmadd_vf_f64m1(vr2, 0x1.5555555555555p-57, vr2_poly_lo, vl);

    vasin_r_hi = __riscv_vfmadd_vv_f64m1(vr_hi, vr2_poly_hi, vr_hi, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f64m1(vr_hi, vr2_poly_hi, __riscv_vfsub_vv_f64m1(vr_hi, vasin_r_hi, vl), vl);
    vasin_r_lo = __riscv_vfmadd_vv_f64m1(vr_hi, vr2_poly_lo, __riscv_vfadd_vv_f64m1(vasin_r_lo, vr_lo, vl), vl);
}

vfloat64m1_t __riscv_vasin_f64m1(vfloat64m1_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m1(avl);

    vuint64m1_t vx_u  = __riscv_vreinterpret_v_f64m1_u64m1(vx);
    vuint64m1_t vax_u = __riscv_vand_vx_u64m1(vx_u, 0x7FFFFFFFFFFFFFFFULL, vl);

#ifndef __FAST_MATH__
    vbool64_t mask_gt1  = __riscv_vmsgtu_vx_u64m1_b64(vax_u, 0x3FF0000000000000ULL, vl);
    vbool64_t mask_qnan = __riscv_vmsgeu_vx_u64m1_b64(vax_u, 0x7FF8000000000000ULL, vl);
    vbool64_t mask_inv  = __riscv_vmandn_mm_b64(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u64m1(vax_u, 0ULL, mask_gt1, vl);
#endif

    vfloat64m1_t vax   = __riscv_vreinterpret_v_u64m1_f64m1(vax_u);
    vbool64_t is_large = __riscv_vmfgt_vf_f64m1_b64(vax, THRESHOLD, vl);

    vfloat64m1_t vx2_hi = __riscv_vfmul_vv_f64m1(vax, vax, vl);
    vfloat64m1_t vx2_lo = __riscv_vfmsub_vv_f64m1(vax, vax, vx2_hi, vl);

    vfloat64m1_t vw_hi  = __riscv_vfrsub_vf_f64m1(vx2_hi, 1.0, vl);
    vfloat64m1_t vw_err = __riscv_vfsub_vv_f64m1(__riscv_vfrsub_vf_f64m1(vw_hi, 1.0, vl), vx2_hi, vl);
    vfloat64m1_t vw_lo  = __riscv_vfsub_vv_f64m1(vw_err, vx2_lo, vl);

    vfloat64m1_t vy_hi  = __riscv_vfsqrt_v_f64m1(vw_hi, vl);
    vfloat64m1_t vs_res = __riscv_vfnmsub_vv_f64m1(vy_hi, vy_hi, vw_hi, vl);
    vs_res              = __riscv_vfadd_vv_f64m1(vs_res, vw_lo, vl);

    //vbool64_t mask_yzero = __riscv_vmfeq_vf_f64m1_b64(vy_hi, 0.0, vl);
    //vfloat64m1_t vy_safe = __riscv_vfmerge_vfm_f64m1(vy_hi, 1.0, mask_yzero, vl);
    //vfloat64m1_t vinv_2y = __riscv_vfrdiv_vf_f64m1(vy_safe, 0.5, vl);
    vbool64_t mask_wzero = __riscv_vmfeq_vf_f64m1_b64(vw_hi, 0.0, vl);
    vfloat64m1_t vw_safe = __riscv_vfmerge_vfm_f64m1(vw_hi, 1.0, mask_wzero, vl);
    vfloat64m1_t vinv_2y = __riscv_vfmul_vf_f64m1(__riscv_vfrsqrt7_v_f64m1(vw_safe, vl), 0.5, vl);
    vfloat64m1_t vy_lo   = __riscv_vfmul_vv_f64m1(vs_res, vinv_2y, vl);

    vfloat64m1_t vzero      = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vu_hi      = __riscv_vmerge_vvm_f64m1(vax, vy_hi, is_large, vl);
    vfloat64m1_t vu_lo      = __riscv_vmerge_vvm_f64m1(vzero, vy_lo, is_large, vl);
    vfloat64m1_t vsqrt_u_hi = __riscv_vmerge_vvm_f64m1(vy_hi, vax, is_large, vl);
    vfloat64m1_t vsqrt_u_lo = __riscv_vmerge_vvm_f64m1(vy_lo, vzero, is_large, vl);
    
    vfloat64m1_t vA_hi, vA_lo, vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f64m1( vu_hi, vu_lo, vsqrt_u_hi, vsqrt_u_lo, vA_hi, vA_lo,
                                vasin_r_hi, vasin_r_lo,  vl);
    
    vfloat64m1_t vsign    = __riscv_vfmerge_vfm_f64m1(__riscv_vfmv_v_f_f64m1(1.0, vl), -1.0, is_large, vl);
    vfloat64m1_t vbase_hi = __riscv_vfmerge_vfm_f64m1(vzero, PI_2_HI, is_large, vl);
    vfloat64m1_t vbase_lo = __riscv_vfmerge_vfm_f64m1(vzero, PI_2_LO, is_large, vl);
    
    vfloat64m1_t vsA_hi       = __riscv_vfsgnjx_vv_f64m1(vA_hi, vsign, vl);
    vfloat64m1_t vsA_lo       = __riscv_vfsgnjx_vv_f64m1(vA_lo, vsign, vl);
    vfloat64m1_t vs_arcsin_rh = __riscv_vfsgnjx_vv_f64m1(vasin_r_hi, vsign, vl);
    vfloat64m1_t vs_arcsin_rl = __riscv_vfsgnjx_vv_f64m1(vasin_r_lo, vsign, vl);

    vfloat64m1_t vsum_hi  = __riscv_vfadd_vv_f64m1(vbase_hi, vsA_hi, vl);
    vfloat64m1_t vsum_err = __riscv_vfadd_vv_f64m1(__riscv_vfsub_vv_f64m1(vbase_hi, vsum_hi, vl), vsA_hi, vl);
    vfloat64m1_t vsum_lo  = __riscv_vfadd_vv_f64m1(__riscv_vfadd_vv_f64m1(vbase_lo, vsA_lo, vl), vsum_err, vl);

    vfloat64m1_t vres_hi  = __riscv_vfadd_vv_f64m1(vsum_hi, vs_arcsin_rh, vl);
    vfloat64m1_t vres_err = __riscv_vfadd_vv_f64m1(__riscv_vfsub_vv_f64m1(vsum_hi, vres_hi, vl), 
                                                     vs_arcsin_rh, vl);
    vfloat64m1_t vres_lo  = __riscv_vfadd_vv_f64m1(__riscv_vfadd_vv_f64m1(vsum_lo, vs_arcsin_rl, vl), 
                                                     vres_err, vl);

    vfloat64m1_t vres = __riscv_vfadd_vv_f64m1(vres_hi, vres_lo, vl);
    vres              = __riscv_vfsgnj_vv_f64m1(vres, vx, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f64m1(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b64(mask_inv, vl) > 0) {
        volatile double zero = 0.0;
        double nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f64m1(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

vfloat64m1_t __riscv_vacos_f64m1(vfloat64m1_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m1(avl);

    vuint64m1_t vx_u  = __riscv_vreinterpret_v_f64m1_u64m1(vx);
    vuint64m1_t vax_u = __riscv_vand_vx_u64m1(vx_u, 0x7FFFFFFFFFFFFFFFULL, vl);
    
    vbool64_t is_neg  = __riscv_vmslt_vx_i64m1_b64(__riscv_vreinterpret_v_u64m1_i64m1(vx_u), 0, vl);

#ifndef __FAST_MATH__
    vbool64_t mask_gt1  = __riscv_vmsgtu_vx_u64m1_b64(vax_u, 0x3FF0000000000000ULL, vl);
    vbool64_t mask_qnan = __riscv_vmsgeu_vx_u64m1_b64(vax_u, 0x7FF8000000000000ULL, vl);
    vbool64_t mask_inv  = __riscv_vmandn_mm_b64(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u64m1(vax_u, 0ULL, mask_gt1, vl);
#endif

    vfloat64m1_t vax = __riscv_vreinterpret_v_u64m1_f64m1(vax_u);

    vbool64_t is_large = __riscv_vmfgt_vf_f64m1_b64(vax, THRESHOLD, vl);

    vfloat64m1_t vx2_hi = __riscv_vfmul_vv_f64m1(vax, vax, vl);
    vfloat64m1_t vx2_lo = __riscv_vfmsub_vv_f64m1(vax, vax, vx2_hi, vl);

    vfloat64m1_t vw_hi  = __riscv_vfrsub_vf_f64m1(vx2_hi, 1.0, vl);
    vfloat64m1_t vw_err = __riscv_vfsub_vv_f64m1(__riscv_vfrsub_vf_f64m1(vw_hi, 1.0, vl), vx2_hi, vl);
    vfloat64m1_t vw_lo  = __riscv_vfsub_vv_f64m1(vw_err, vx2_lo, vl);

    vfloat64m1_t vy_hi  = __riscv_vfsqrt_v_f64m1(vw_hi, vl);
    vfloat64m1_t vs_res = __riscv_vfnmsub_vv_f64m1(vy_hi, vy_hi, vw_hi, vl);
    vs_res              = __riscv_vfadd_vv_f64m1(vs_res, vw_lo, vl);

    vbool64_t mask_yzero = __riscv_vmfeq_vf_f64m1_b64(vy_hi, 0.0, vl);
    vfloat64m1_t vy_safe = __riscv_vfmerge_vfm_f64m1(vy_hi, 1.0, mask_yzero, vl);
    vfloat64m1_t vinv_2y = __riscv_vfrdiv_vf_f64m1(vy_safe, 0.5, vl);
    //vfloat64m1_t vinv_2y = __riscv_vfmul_vf_f64m1(__riscv_vfrsqrt7_v_f64m1(vw_hi, vl), 0.5, vl); ---
    vfloat64m1_t vy_lo   = __riscv_vfmul_vv_f64m1(vs_res, vinv_2y, vl);

    vfloat64m1_t vzero      = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t vu_hi      = __riscv_vmerge_vvm_f64m1(vax, vy_hi, is_large, vl);
    vfloat64m1_t vu_lo      = __riscv_vmerge_vvm_f64m1(vzero, vy_lo, is_large, vl);
    vfloat64m1_t vsqrt_u_hi = __riscv_vmerge_vvm_f64m1(vy_hi, vax, is_large, vl);
    vfloat64m1_t vsqrt_u_lo = __riscv_vmerge_vvm_f64m1(vy_lo, vzero, is_large, vl);

    vfloat64m1_t vA_hi, vA_lo, vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f64m1(vu_hi, vu_lo, vsqrt_u_hi, vsqrt_u_lo,
                               vA_hi, vA_lo, vasin_r_hi, vasin_r_lo, vl);

    vfloat64m1_t vsign = __riscv_vmerge_vvm_f64m1(__riscv_vfsgnjn_vv_f64m1(vx, vx, vl), vx, is_large, vl);

    vfloat64m1_t vsA_hi    = __riscv_vfsgnjx_vv_f64m1(vA_hi, vsign, vl);
    vfloat64m1_t vsA_lo    = __riscv_vfsgnjx_vv_f64m1(vA_lo, vsign, vl);
    vfloat64m1_t vs_asin_h = __riscv_vfsgnjx_vv_f64m1(vasin_r_hi, vsign, vl);
    vfloat64m1_t vs_asin_l = __riscv_vfsgnjx_vv_f64m1(vasin_r_lo, vsign, vl);

    vfloat64m1_t vbase_lrg_hi = __riscv_vfmerge_vfm_f64m1(vzero, PI_HI, is_neg, vl);
    vfloat64m1_t vbase_lrg_lo = __riscv_vfmerge_vfm_f64m1(vzero, PI_LO, is_neg, vl);
    vfloat64m1_t vbase_hi     = __riscv_vmerge_vvm_f64m1(__riscv_vfmv_v_f_f64m1(PI_2_HI, vl), vbase_lrg_hi, is_large, vl);
    vfloat64m1_t vbase_lo     = __riscv_vmerge_vvm_f64m1(__riscv_vfmv_v_f_f64m1(PI_2_LO, vl), vbase_lrg_lo, is_large, vl);

    vfloat64m1_t vsum1_hi  = __riscv_vfadd_vv_f64m1(vbase_hi, vsA_hi, vl);
    vfloat64m1_t vv_sum1   = __riscv_vfsub_vv_f64m1(vsum1_hi, vbase_hi, vl);
    vfloat64m1_t vsum1_err = __riscv_vfsub_vv_f64m1(vbase_hi, __riscv_vfsub_vv_f64m1(vsum1_hi, vv_sum1, vl), vl); 
    vsum1_err              = __riscv_vfadd_vv_f64m1(vsum1_err, __riscv_vfsub_vv_f64m1(vsA_hi, vv_sum1, vl), vl);
    vfloat64m1_t vsum1_lo  = __riscv_vfadd_vv_f64m1(__riscv_vfadd_vv_f64m1(vbase_lo, vsA_lo, vl), vsum1_err, vl);

    vfloat64m1_t vres_hi   = __riscv_vfadd_vv_f64m1(vsum1_hi, vs_asin_h, vl);
    vfloat64m1_t vres_err  = __riscv_vfadd_vv_f64m1(__riscv_vfsub_vv_f64m1(vsum1_hi, vres_hi, vl), vs_asin_h, vl);
    vfloat64m1_t vres_lo   = __riscv_vfadd_vv_f64m1(__riscv_vfadd_vv_f64m1(vsum1_lo, vs_asin_l, vl), vres_err, vl);

    vfloat64m1_t vres = __riscv_vfadd_vv_f64m1(vres_hi, vres_lo, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f64m1(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b64(mask_inv, vl) > 0) {
        volatile double zero = 0.0;
        double nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f64m1(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}


void __riscv_vasin_kernel_f64m2(vfloat64m2_t vu_hi, vfloat64m2_t vu_lo,
                                vfloat64m2_t vsqrt_u_hi, vfloat64m2_t vsqrt_u_lo, 
                                vfloat64m2_t &vA_hi, vfloat64m2_t &vA_lo,
                                vfloat64m2_t &vasin_r_hi, vfloat64m2_t &vasin_r_lo, size_t vl) {
    vfloat64m2_t vidx_f = __riscv_vfmadd_vf_f64m2(vu_hi, 32.0, __riscv_vfmv_v_f_f64m2(0.5, vl), vl);
    vuint64m2_t vidx    = __riscv_vfcvt_rtz_xu_f_v_u64m2(vidx_f, vl);
    vidx                = __riscv_vminu_vx_u64m2(vidx, 23, vl);
    
    vfloat64m2_t vT     = __riscv_vfmul_vf_f64m2(__riscv_vfcvt_f_xu_v_f64m2(vidx, vl), 0x1p-5, vl);
    vuint64m2_t voffset = __riscv_vsll_vx_u64m2(vidx, 3, vl);
    vfloat64m2_t vC_hi  = __riscv_vluxei64_v_f64m2(Ch_f64, voffset, vl);
    vfloat64m2_t vC_lo  = __riscv_vluxei64_v_f64m2(Cl_f64, voffset, vl);
    vA_hi               = __riscv_vluxei64_v_f64m2(Ah_f64, voffset, vl);
    vA_lo               = __riscv_vluxei64_v_f64m2(Al_f64, voffset, vl);

    vfloat64m2_t vp2_hi = __riscv_vfmul_vv_f64m2(vT, vsqrt_u_hi, vl);
    vfloat64m2_t vp2_lo = __riscv_vfmsub_vv_f64m2(vT, vsqrt_u_hi, vp2_hi, vl);
    vp2_lo              = __riscv_vfmadd_vv_f64m2(vT, vsqrt_u_lo, vp2_lo, vl);

    vfloat64m2_t vp1_hi = __riscv_vfmul_vv_f64m2(vu_hi, vC_hi, vl);
    vfloat64m2_t vp1_lo = __riscv_vfmsub_vv_f64m2(vu_hi, vC_hi, vp1_hi, vl);
    vp1_lo              = __riscv_vfmadd_vv_f64m2(vu_lo, vC_hi, vp1_lo, vl);
    vp1_lo              = __riscv_vfmadd_vv_f64m2(vu_hi, vC_lo, vp1_lo, vl);
    
    vfloat64m2_t vr_raw_hi = __riscv_vfsub_vv_f64m2(vp1_hi, vp2_hi, vl);
    vfloat64m2_t vr_sub    = __riscv_vfsub_vv_f64m2(vr_raw_hi, vp1_hi, vl);
    vfloat64m2_t vr_err    = __riscv_vfsub_vv_f64m2(vp1_hi, __riscv_vfsub_vv_f64m2(vr_raw_hi, vr_sub, vl), vl);
    vr_err                 = __riscv_vfsub_vv_f64m2(vr_err, __riscv_vfadd_vv_f64m2(vp2_hi, vr_sub, vl), vl);
    vfloat64m2_t vr_raw_lo = __riscv_vfadd_vv_f64m2(vr_err, __riscv_vfsub_vv_f64m2(vp1_lo, vp2_lo, vl), vl);

    vfloat64m2_t vr_hi = __riscv_vfadd_vv_f64m2(vr_raw_hi, vr_raw_lo, vl);
    vfloat64m2_t vr_lo = __riscv_vfadd_vv_f64m2(__riscv_vfsub_vv_f64m2(vr_raw_hi, vr_hi, vl), vr_raw_lo, vl);
    
    vfloat64m2_t vr2 = __riscv_vfmadd_vv_f64m2(vr_hi, vr_hi, __riscv_vfmv_v_f_f64m2(0x1p-120, vl), vl);
    vfloat64m2_t vr4 = __riscv_vfmul_vv_f64m2(vr2, vr2, vl);

    vfloat64m2_t vq0   = __riscv_vfmadd_vf_f64m2(vr2, 0x1.3333333333321p-4, 
                                                 __riscv_vfmv_v_f_f64m2(0x1.5555555555555p-3, vl), vl);
    vfloat64m2_t vq1   = __riscv_vfmadd_vf_f64m2(vr2, 0x1.f1c7109674b33p-6, 
                                                 __riscv_vfmv_v_f_f64m2(0x1.6db6db6e0e4ep-5, vl), vl);
    vfloat64m2_t vq2   = __riscv_vfmadd_vf_f64m2(vr4, 0x1.6eea780121fdfp-6, vq1, vl);
    vfloat64m2_t vpoly = __riscv_vfmadd_vv_f64m2(vr4, vq2, vq0, vl);

    vfloat64m2_t vr2_poly_hi = __riscv_vfmul_vv_f64m2(vr2, vpoly, vl);
    vfloat64m2_t vr2_poly_lo = __riscv_vfmsub_vv_f64m2(vr2, vpoly, vr2_poly_hi, vl);
    vr2_poly_lo              = __riscv_vfmadd_vf_f64m2(vr2, 0x1.5555555555555p-57, vr2_poly_lo, vl);

    vasin_r_hi = __riscv_vfmadd_vv_f64m2(vr_hi, vr2_poly_hi, vr_hi, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f64m2(vr_hi, vr2_poly_hi, __riscv_vfsub_vv_f64m2(vr_hi, vasin_r_hi, vl), vl);
    vasin_r_lo = __riscv_vfmadd_vv_f64m2(vr_hi, vr2_poly_lo, __riscv_vfadd_vv_f64m2(vasin_r_lo, vr_lo, vl), vl);
}

vfloat64m2_t __riscv_vasin_f64m2(vfloat64m2_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m2(avl);

    vuint64m2_t vx_u  = __riscv_vreinterpret_v_f64m2_u64m2(vx);
    vuint64m2_t vax_u = __riscv_vand_vx_u64m2(vx_u, 0x7FFFFFFFFFFFFFFFULL, vl);

#ifndef __FAST_MATH__
    vbool32_t mask_gt1  = __riscv_vmsgtu_vx_u64m2_b32(vax_u, 0x3FF0000000000000ULL, vl);
    vbool32_t mask_qnan = __riscv_vmsgeu_vx_u64m2_b32(vax_u, 0x7FF8000000000000ULL, vl);
    vbool32_t mask_inv  = __riscv_vmandn_mm_b32(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u64m2(vax_u, 0ULL, mask_gt1, vl);
#endif

    vfloat64m2_t vax   = __riscv_vreinterpret_v_u64m2_f64m2(vax_u);
    vbool32_t is_large = __riscv_vmfgt_vf_f64m2_b32(vax, THRESHOLD, vl);

    vfloat64m2_t vx2_hi = __riscv_vfmul_vv_f64m2(vax, vax, vl);
    vfloat64m2_t vx2_lo = __riscv_vfmsub_vv_f64m2(vax, vax, vx2_hi, vl);

    vfloat64m2_t vw_hi  = __riscv_vfrsub_vf_f64m2(vx2_hi, 1.0, vl);
    vfloat64m2_t vw_err = __riscv_vfsub_vv_f64m2(__riscv_vfrsub_vf_f64m2(vw_hi, 1.0, vl), vx2_hi, vl);
    vfloat64m2_t vw_lo  = __riscv_vfsub_vv_f64m2(vw_err, vx2_lo, vl);

    vfloat64m2_t vy_hi  = __riscv_vfsqrt_v_f64m2(vw_hi, vl);
    vfloat64m2_t vs_res = __riscv_vfnmsub_vv_f64m2(vy_hi, vy_hi, vw_hi, vl);
    vs_res              = __riscv_vfadd_vv_f64m2(vs_res, vw_lo, vl);

    //vbool32_t mask_yzero = __riscv_vmfeq_vf_f64m2_b32(vy_hi, 0.0, vl);
    //vfloat64m2_t vy_safe = __riscv_vfmerge_vfm_f64m2(vy_hi, 1.0, mask_yzero, vl);
    //vfloat64m2_t vinv_2y = __riscv_vfrdiv_vf_f64m2(vy_safe, 0.5, vl);
    vbool32_t mask_wzero = __riscv_vmfeq_vf_f64m2_b32(vw_hi, 0.0, vl);
    vfloat64m2_t vw_safe = __riscv_vfmerge_vfm_f64m2(vw_hi, 1.0, mask_wzero, vl);
    vfloat64m2_t vinv_2y = __riscv_vfmul_vf_f64m2(__riscv_vfrsqrt7_v_f64m2(vw_safe, vl), 0.5, vl);
    vfloat64m2_t vy_lo   = __riscv_vfmul_vv_f64m2(vs_res, vinv_2y, vl);

    vfloat64m2_t vzero      = __riscv_vfmv_v_f_f64m2(0.0, vl);
    vfloat64m2_t vu_hi      = __riscv_vmerge_vvm_f64m2(vax, vy_hi, is_large, vl);
    vfloat64m2_t vu_lo      = __riscv_vmerge_vvm_f64m2(vzero, vy_lo, is_large, vl);
    vfloat64m2_t vsqrt_u_hi = __riscv_vmerge_vvm_f64m2(vy_hi, vax, is_large, vl);
    vfloat64m2_t vsqrt_u_lo = __riscv_vmerge_vvm_f64m2(vy_lo, vzero, is_large, vl);
    
    vfloat64m2_t vA_hi, vA_lo, vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f64m2( vu_hi, vu_lo, vsqrt_u_hi, vsqrt_u_lo, vA_hi, vA_lo,
                                vasin_r_hi, vasin_r_lo,  vl);
    
    vfloat64m2_t vsign    = __riscv_vfmerge_vfm_f64m2(__riscv_vfmv_v_f_f64m2(1.0, vl), -1.0, is_large, vl);
    vfloat64m2_t vbase_hi = __riscv_vfmerge_vfm_f64m2(vzero, PI_2_HI, is_large, vl);
    vfloat64m2_t vbase_lo = __riscv_vfmerge_vfm_f64m2(vzero, PI_2_LO, is_large, vl);
    
    vfloat64m2_t vsA_hi       = __riscv_vfsgnjx_vv_f64m2(vA_hi, vsign, vl);
    vfloat64m2_t vsA_lo       = __riscv_vfsgnjx_vv_f64m2(vA_lo, vsign, vl);
    vfloat64m2_t vs_arcsin_rh = __riscv_vfsgnjx_vv_f64m2(vasin_r_hi, vsign, vl);
    vfloat64m2_t vs_arcsin_rl = __riscv_vfsgnjx_vv_f64m2(vasin_r_lo, vsign, vl);

    vfloat64m2_t vsum_hi  = __riscv_vfadd_vv_f64m2(vbase_hi, vsA_hi, vl);
    vfloat64m2_t vsum_err = __riscv_vfadd_vv_f64m2(__riscv_vfsub_vv_f64m2(vbase_hi, vsum_hi, vl), vsA_hi, vl);
    vfloat64m2_t vsum_lo  = __riscv_vfadd_vv_f64m2(__riscv_vfadd_vv_f64m2(vbase_lo, vsA_lo, vl), vsum_err, vl);

    vfloat64m2_t vres_hi  = __riscv_vfadd_vv_f64m2(vsum_hi, vs_arcsin_rh, vl);
    vfloat64m2_t vres_err = __riscv_vfadd_vv_f64m2(__riscv_vfsub_vv_f64m2(vsum_hi, vres_hi, vl), 
                                                     vs_arcsin_rh, vl);
    vfloat64m2_t vres_lo  = __riscv_vfadd_vv_f64m2(__riscv_vfadd_vv_f64m2(vsum_lo, vs_arcsin_rl, vl), 
                                                     vres_err, vl);

    vfloat64m2_t vres = __riscv_vfadd_vv_f64m2(vres_hi, vres_lo, vl);
    vres              = __riscv_vfsgnj_vv_f64m2(vres, vx, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f64m2(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b32(mask_inv, vl) > 0) {
        volatile double zero = 0.0;
        double nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f64m2(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

vfloat64m2_t __riscv_vacos_f64m2(vfloat64m2_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m2(avl);

    vuint64m2_t vx_u  = __riscv_vreinterpret_v_f64m2_u64m2(vx);
    vuint64m2_t vax_u = __riscv_vand_vx_u64m2(vx_u, 0x7FFFFFFFFFFFFFFFULL, vl);
    
    vbool32_t is_neg  = __riscv_vmslt_vx_i64m2_b32(__riscv_vreinterpret_v_u64m2_i64m2(vx_u), 0, vl);

#ifndef __FAST_MATH__
    vbool32_t mask_gt1  = __riscv_vmsgtu_vx_u64m2_b32(vax_u, 0x3FF0000000000000ULL, vl);
    vbool32_t mask_qnan = __riscv_vmsgeu_vx_u64m2_b32(vax_u, 0x7FF8000000000000ULL, vl);
    vbool32_t mask_inv  = __riscv_vmandn_mm_b32(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u64m2(vax_u, 0ULL, mask_gt1, vl);
#endif

    vfloat64m2_t vax = __riscv_vreinterpret_v_u64m2_f64m2(vax_u);

    vbool32_t is_large = __riscv_vmfgt_vf_f64m2_b32(vax, THRESHOLD, vl);

    vfloat64m2_t vx2_hi = __riscv_vfmul_vv_f64m2(vax, vax, vl);
    vfloat64m2_t vx2_lo = __riscv_vfmsub_vv_f64m2(vax, vax, vx2_hi, vl);

    vfloat64m2_t vw_hi  = __riscv_vfrsub_vf_f64m2(vx2_hi, 1.0, vl);
    vfloat64m2_t vw_err = __riscv_vfsub_vv_f64m2(__riscv_vfrsub_vf_f64m2(vw_hi, 1.0, vl), vx2_hi, vl);
    vfloat64m2_t vw_lo  = __riscv_vfsub_vv_f64m2(vw_err, vx2_lo, vl);

    vfloat64m2_t vy_hi  = __riscv_vfsqrt_v_f64m2(vw_hi, vl);
    vfloat64m2_t vs_res = __riscv_vfnmsub_vv_f64m2(vy_hi, vy_hi, vw_hi, vl);
    vs_res              = __riscv_vfadd_vv_f64m2(vs_res, vw_lo, vl);

    vbool32_t mask_yzero = __riscv_vmfeq_vf_f64m2_b32(vy_hi, 0.0, vl);
    vfloat64m2_t vy_safe = __riscv_vfmerge_vfm_f64m2(vy_hi, 1.0, mask_yzero, vl);
    vfloat64m2_t vinv_2y = __riscv_vfrdiv_vf_f64m2(vy_safe, 0.5, vl);
    //vfloat64m2_t vinv_2y = __riscv_vfmul_vf_f64m2(__riscv_vfrsqrt7_v_f64m2(vw_hi, vl), 0.5, vl); ---
    vfloat64m2_t vy_lo   = __riscv_vfmul_vv_f64m2(vs_res, vinv_2y, vl);

    vfloat64m2_t vzero      = __riscv_vfmv_v_f_f64m2(0.0, vl);
    vfloat64m2_t vu_hi      = __riscv_vmerge_vvm_f64m2(vax, vy_hi, is_large, vl);
    vfloat64m2_t vu_lo      = __riscv_vmerge_vvm_f64m2(vzero, vy_lo, is_large, vl);
    vfloat64m2_t vsqrt_u_hi = __riscv_vmerge_vvm_f64m2(vy_hi, vax, is_large, vl);
    vfloat64m2_t vsqrt_u_lo = __riscv_vmerge_vvm_f64m2(vy_lo, vzero, is_large, vl);

    vfloat64m2_t vA_hi, vA_lo, vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f64m2(vu_hi, vu_lo, vsqrt_u_hi, vsqrt_u_lo,
                               vA_hi, vA_lo, vasin_r_hi, vasin_r_lo, vl);

    vfloat64m2_t vsign = __riscv_vmerge_vvm_f64m2(__riscv_vfsgnjn_vv_f64m2(vx, vx, vl), vx, is_large, vl);

    vfloat64m2_t vsA_hi    = __riscv_vfsgnjx_vv_f64m2(vA_hi, vsign, vl);
    vfloat64m2_t vsA_lo    = __riscv_vfsgnjx_vv_f64m2(vA_lo, vsign, vl);
    vfloat64m2_t vs_asin_h = __riscv_vfsgnjx_vv_f64m2(vasin_r_hi, vsign, vl);
    vfloat64m2_t vs_asin_l = __riscv_vfsgnjx_vv_f64m2(vasin_r_lo, vsign, vl);

    vfloat64m2_t vbase_lrg_hi = __riscv_vfmerge_vfm_f64m2(vzero, PI_HI, is_neg, vl);
    vfloat64m2_t vbase_lrg_lo = __riscv_vfmerge_vfm_f64m2(vzero, PI_LO, is_neg, vl);
    vfloat64m2_t vbase_hi     = __riscv_vmerge_vvm_f64m2(__riscv_vfmv_v_f_f64m2(PI_2_HI, vl), vbase_lrg_hi, is_large, vl);
    vfloat64m2_t vbase_lo     = __riscv_vmerge_vvm_f64m2(__riscv_vfmv_v_f_f64m2(PI_2_LO, vl), vbase_lrg_lo, is_large, vl);

    vfloat64m2_t vsum2_hi  = __riscv_vfadd_vv_f64m2(vbase_hi, vsA_hi, vl);
    vfloat64m2_t vv_sum2   = __riscv_vfsub_vv_f64m2(vsum2_hi, vbase_hi, vl);
    vfloat64m2_t vsum2_err = __riscv_vfsub_vv_f64m2(vbase_hi, __riscv_vfsub_vv_f64m2(vsum2_hi, vv_sum2, vl), vl); 
    vsum2_err              = __riscv_vfadd_vv_f64m2(vsum2_err, __riscv_vfsub_vv_f64m2(vsA_hi, vv_sum2, vl), vl);
    vfloat64m2_t vsum2_lo  = __riscv_vfadd_vv_f64m2(__riscv_vfadd_vv_f64m2(vbase_lo, vsA_lo, vl), vsum2_err, vl);

    vfloat64m2_t vres_hi   = __riscv_vfadd_vv_f64m2(vsum2_hi, vs_asin_h, vl);
    vfloat64m2_t vres_err  = __riscv_vfadd_vv_f64m2(__riscv_vfsub_vv_f64m2(vsum2_hi, vres_hi, vl), vs_asin_h, vl);
    vfloat64m2_t vres_lo   = __riscv_vfadd_vv_f64m2(__riscv_vfadd_vv_f64m2(vsum2_lo, vs_asin_l, vl), vres_err, vl);

    vfloat64m2_t vres = __riscv_vfadd_vv_f64m2(vres_hi, vres_lo, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f64m2(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b32(mask_inv, vl) > 0) {
        volatile double zero = 0.0;
        double nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f64m2(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}


void __riscv_vasin_kernel_f64m4(vfloat64m4_t vu_hi, vfloat64m4_t vu_lo,
                                vfloat64m4_t vsqrt_u_hi, vfloat64m4_t vsqrt_u_lo, 
                                vfloat64m4_t &vA_hi, vfloat64m4_t &vA_lo,
                                vfloat64m4_t &vasin_r_hi, vfloat64m4_t &vasin_r_lo, size_t vl) {
    vfloat64m4_t vidx_f = __riscv_vfmadd_vf_f64m4(vu_hi, 32.0, __riscv_vfmv_v_f_f64m4(0.5, vl), vl);
    vuint64m4_t vidx    = __riscv_vfcvt_rtz_xu_f_v_u64m4(vidx_f, vl);
    vidx                = __riscv_vminu_vx_u64m4(vidx, 23, vl);
    
    vfloat64m4_t vT     = __riscv_vfmul_vf_f64m4(__riscv_vfcvt_f_xu_v_f64m4(vidx, vl), 0x1p-5, vl);
    vuint64m4_t voffset = __riscv_vsll_vx_u64m4(vidx, 3, vl);
    vfloat64m4_t vC_hi  = __riscv_vluxei64_v_f64m4(Ch_f64, voffset, vl);
    vfloat64m4_t vC_lo  = __riscv_vluxei64_v_f64m4(Cl_f64, voffset, vl);
    vA_hi               = __riscv_vluxei64_v_f64m4(Ah_f64, voffset, vl);
    vA_lo               = __riscv_vluxei64_v_f64m4(Al_f64, voffset, vl);

    vfloat64m4_t vp2_hi = __riscv_vfmul_vv_f64m4(vT, vsqrt_u_hi, vl);
    vfloat64m4_t vp2_lo = __riscv_vfmsub_vv_f64m4(vT, vsqrt_u_hi, vp2_hi, vl);
    vp2_lo              = __riscv_vfmadd_vv_f64m4(vT, vsqrt_u_lo, vp2_lo, vl);

    vfloat64m4_t vp1_hi = __riscv_vfmul_vv_f64m4(vu_hi, vC_hi, vl);
    vfloat64m4_t vp1_lo = __riscv_vfmsub_vv_f64m4(vu_hi, vC_hi, vp1_hi, vl);
    vp1_lo              = __riscv_vfmadd_vv_f64m4(vu_lo, vC_hi, vp1_lo, vl);
    vp1_lo              = __riscv_vfmadd_vv_f64m4(vu_hi, vC_lo, vp1_lo, vl);
    
    vfloat64m4_t vr_raw_hi = __riscv_vfsub_vv_f64m4(vp1_hi, vp2_hi, vl);
    vfloat64m4_t vr_sub    = __riscv_vfsub_vv_f64m4(vr_raw_hi, vp1_hi, vl);
    vfloat64m4_t vr_err    = __riscv_vfsub_vv_f64m4(vp1_hi, __riscv_vfsub_vv_f64m4(vr_raw_hi, vr_sub, vl), vl);
    vr_err                 = __riscv_vfsub_vv_f64m4(vr_err, __riscv_vfadd_vv_f64m4(vp2_hi, vr_sub, vl), vl);
    vfloat64m4_t vr_raw_lo = __riscv_vfadd_vv_f64m4(vr_err, __riscv_vfsub_vv_f64m4(vp1_lo, vp2_lo, vl), vl);

    vfloat64m4_t vr_hi = __riscv_vfadd_vv_f64m4(vr_raw_hi, vr_raw_lo, vl);
    vfloat64m4_t vr_lo = __riscv_vfadd_vv_f64m4(__riscv_vfsub_vv_f64m4(vr_raw_hi, vr_hi, vl), vr_raw_lo, vl);
    
    vfloat64m4_t vr2 = __riscv_vfmadd_vv_f64m4(vr_hi, vr_hi, __riscv_vfmv_v_f_f64m4(0x1p-120, vl), vl);
    vfloat64m4_t vr4 = __riscv_vfmul_vv_f64m4(vr2, vr2, vl);

    vfloat64m4_t vq0   = __riscv_vfmadd_vf_f64m4(vr2, 0x1.3333333333321p-4, 
                                                 __riscv_vfmv_v_f_f64m4(0x1.5555555555555p-3, vl), vl);
    vfloat64m4_t vq1   = __riscv_vfmadd_vf_f64m4(vr2, 0x1.f1c7109674b33p-6, 
                                                 __riscv_vfmv_v_f_f64m4(0x1.6db6db6e0e4ep-5, vl), vl);
    vfloat64m4_t vq2   = __riscv_vfmadd_vf_f64m4(vr4, 0x1.6eea780121fdfp-6, vq1, vl);
    vfloat64m4_t vpoly = __riscv_vfmadd_vv_f64m4(vr4, vq2, vq0, vl);

    vfloat64m4_t vr2_poly_hi = __riscv_vfmul_vv_f64m4(vr2, vpoly, vl);
    vfloat64m4_t vr2_poly_lo = __riscv_vfmsub_vv_f64m4(vr2, vpoly, vr2_poly_hi, vl);
    vr2_poly_lo              = __riscv_vfmadd_vf_f64m4(vr2, 0x1.5555555555555p-57, vr2_poly_lo, vl);

    vasin_r_hi = __riscv_vfmadd_vv_f64m4(vr_hi, vr2_poly_hi, vr_hi, vl);
    vasin_r_lo = __riscv_vfmadd_vv_f64m4(vr_hi, vr2_poly_hi, __riscv_vfsub_vv_f64m4(vr_hi, vasin_r_hi, vl), vl);
    vasin_r_lo = __riscv_vfmadd_vv_f64m4(vr_hi, vr2_poly_lo, __riscv_vfadd_vv_f64m4(vasin_r_lo, vr_lo, vl), vl);
}

vfloat64m4_t __riscv_vasin_f64m4(vfloat64m4_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m4(avl);

    vuint64m4_t vx_u  = __riscv_vreinterpret_v_f64m4_u64m4(vx);
    vuint64m4_t vax_u = __riscv_vand_vx_u64m4(vx_u, 0x7FFFFFFFFFFFFFFFULL, vl);

#ifndef __FAST_MATH__
    vbool16_t mask_gt1  = __riscv_vmsgtu_vx_u64m4_b16(vax_u, 0x3FF0000000000000ULL, vl);
    vbool16_t mask_qnan = __riscv_vmsgeu_vx_u64m4_b16(vax_u, 0x7FF8000000000000ULL, vl);
    vbool16_t mask_inv  = __riscv_vmandn_mm_b16(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u64m4(vax_u, 0ULL, mask_gt1, vl);
#endif

    vfloat64m4_t vax   = __riscv_vreinterpret_v_u64m4_f64m4(vax_u);
    vbool16_t is_large = __riscv_vmfgt_vf_f64m4_b16(vax, THRESHOLD, vl);

    vfloat64m4_t vx2_hi = __riscv_vfmul_vv_f64m4(vax, vax, vl);
    vfloat64m4_t vx2_lo = __riscv_vfmsub_vv_f64m4(vax, vax, vx2_hi, vl);

    vfloat64m4_t vw_hi  = __riscv_vfrsub_vf_f64m4(vx2_hi, 1.0, vl);
    vfloat64m4_t vw_err = __riscv_vfsub_vv_f64m4(__riscv_vfrsub_vf_f64m4(vw_hi, 1.0, vl), vx2_hi, vl);
    vfloat64m4_t vw_lo  = __riscv_vfsub_vv_f64m4(vw_err, vx2_lo, vl);

    vfloat64m4_t vy_hi  = __riscv_vfsqrt_v_f64m4(vw_hi, vl);
    vfloat64m4_t vs_res = __riscv_vfnmsub_vv_f64m4(vy_hi, vy_hi, vw_hi, vl);
    vs_res              = __riscv_vfadd_vv_f64m4(vs_res, vw_lo, vl);

    //vbool16_t mask_yzero = __riscv_vmfeq_vf_f64m4_b16(vy_hi, 0.0, vl);
    //vfloat64m4_t vy_safe = __riscv_vfmerge_vfm_f64m4(vy_hi, 1.0, mask_yzero, vl);
    //vfloat64m4_t vinv_2y = __riscv_vfrdiv_vf_f64m4(vy_safe, 0.5, vl);
    vbool16_t mask_wzero = __riscv_vmfeq_vf_f64m4_b16(vw_hi, 0.0, vl);
    vfloat64m4_t vw_safe = __riscv_vfmerge_vfm_f64m4(vw_hi, 1.0, mask_wzero, vl);
    vfloat64m4_t vinv_2y = __riscv_vfmul_vf_f64m4(__riscv_vfrsqrt7_v_f64m4(vw_safe, vl), 0.5, vl);
    vfloat64m4_t vy_lo   = __riscv_vfmul_vv_f64m4(vs_res, vinv_2y, vl);

    vfloat64m4_t vzero      = __riscv_vfmv_v_f_f64m4(0.0, vl);
    vfloat64m4_t vu_hi      = __riscv_vmerge_vvm_f64m4(vax, vy_hi, is_large, vl);
    vfloat64m4_t vu_lo      = __riscv_vmerge_vvm_f64m4(vzero, vy_lo, is_large, vl);
    vfloat64m4_t vsqrt_u_hi = __riscv_vmerge_vvm_f64m4(vy_hi, vax, is_large, vl);
    vfloat64m4_t vsqrt_u_lo = __riscv_vmerge_vvm_f64m4(vy_lo, vzero, is_large, vl);
    
    vfloat64m4_t vA_hi, vA_lo, vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f64m4( vu_hi, vu_lo, vsqrt_u_hi, vsqrt_u_lo, vA_hi, vA_lo,
                                vasin_r_hi, vasin_r_lo,  vl);
    
    vfloat64m4_t vsign    = __riscv_vfmerge_vfm_f64m4(__riscv_vfmv_v_f_f64m4(1.0, vl), -1.0, is_large, vl);
    vfloat64m4_t vbase_hi = __riscv_vfmerge_vfm_f64m4(vzero, PI_2_HI, is_large, vl);
    vfloat64m4_t vbase_lo = __riscv_vfmerge_vfm_f64m4(vzero, PI_2_LO, is_large, vl);
    
    vfloat64m4_t vsA_hi       = __riscv_vfsgnjx_vv_f64m4(vA_hi, vsign, vl);
    vfloat64m4_t vsA_lo       = __riscv_vfsgnjx_vv_f64m4(vA_lo, vsign, vl);
    vfloat64m4_t vs_arcsin_rh = __riscv_vfsgnjx_vv_f64m4(vasin_r_hi, vsign, vl);
    vfloat64m4_t vs_arcsin_rl = __riscv_vfsgnjx_vv_f64m4(vasin_r_lo, vsign, vl);

    vfloat64m4_t vsum_hi  = __riscv_vfadd_vv_f64m4(vbase_hi, vsA_hi, vl);
    vfloat64m4_t vsum_err = __riscv_vfadd_vv_f64m4(__riscv_vfsub_vv_f64m4(vbase_hi, vsum_hi, vl), vsA_hi, vl);
    vfloat64m4_t vsum_lo  = __riscv_vfadd_vv_f64m4(__riscv_vfadd_vv_f64m4(vbase_lo, vsA_lo, vl), vsum_err, vl);

    vfloat64m4_t vres_hi  = __riscv_vfadd_vv_f64m4(vsum_hi, vs_arcsin_rh, vl);
    vfloat64m4_t vres_err = __riscv_vfadd_vv_f64m4(__riscv_vfsub_vv_f64m4(vsum_hi, vres_hi, vl), 
                                                     vs_arcsin_rh, vl);
    vfloat64m4_t vres_lo  = __riscv_vfadd_vv_f64m4(__riscv_vfadd_vv_f64m4(vsum_lo, vs_arcsin_rl, vl), 
                                                     vres_err, vl);

    vfloat64m4_t vres = __riscv_vfadd_vv_f64m4(vres_hi, vres_lo, vl);
    vres              = __riscv_vfsgnj_vv_f64m4(vres, vx, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f64m4(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b16(mask_inv, vl) > 0) {
        volatile double zero = 0.0;
        double nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f64m4(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

vfloat64m4_t __riscv_vacos_f64m4(vfloat64m4_t vx, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m4(avl);

    vuint64m4_t vx_u  = __riscv_vreinterpret_v_f64m4_u64m4(vx);
    vuint64m4_t vax_u = __riscv_vand_vx_u64m4(vx_u, 0x7FFFFFFFFFFFFFFFULL, vl);
    
    vbool16_t is_neg  = __riscv_vmslt_vx_i64m4_b16(__riscv_vreinterpret_v_u64m4_i64m4(vx_u), 0, vl);

#ifndef __FAST_MATH__
    vbool16_t mask_gt1  = __riscv_vmsgtu_vx_u64m4_b16(vax_u, 0x3FF0000000000000ULL, vl);
    vbool16_t mask_qnan = __riscv_vmsgeu_vx_u64m4_b16(vax_u, 0x7FF8000000000000ULL, vl);
    vbool16_t mask_inv  = __riscv_vmandn_mm_b16(mask_gt1, mask_qnan, vl);

    vax_u = __riscv_vmerge_vxm_u64m4(vax_u, 0ULL, mask_gt1, vl);
#endif

    vfloat64m4_t vax = __riscv_vreinterpret_v_u64m4_f64m4(vax_u);

    vbool16_t is_large = __riscv_vmfgt_vf_f64m4_b16(vax, THRESHOLD, vl);

    vfloat64m4_t vx2_hi = __riscv_vfmul_vv_f64m4(vax, vax, vl);
    vfloat64m4_t vx2_lo = __riscv_vfmsub_vv_f64m4(vax, vax, vx2_hi, vl);

    vfloat64m4_t vw_hi  = __riscv_vfrsub_vf_f64m4(vx2_hi, 1.0, vl);
    vfloat64m4_t vw_err = __riscv_vfsub_vv_f64m4(__riscv_vfrsub_vf_f64m4(vw_hi, 1.0, vl), vx2_hi, vl);
    vfloat64m4_t vw_lo  = __riscv_vfsub_vv_f64m4(vw_err, vx2_lo, vl);

    vfloat64m4_t vy_hi  = __riscv_vfsqrt_v_f64m4(vw_hi, vl);
    vfloat64m4_t vs_res = __riscv_vfnmsub_vv_f64m4(vy_hi, vy_hi, vw_hi, vl);
    vs_res              = __riscv_vfadd_vv_f64m4(vs_res, vw_lo, vl);

    vbool16_t mask_yzero = __riscv_vmfeq_vf_f64m4_b16(vy_hi, 0.0, vl);
    vfloat64m4_t vy_safe = __riscv_vfmerge_vfm_f64m4(vy_hi, 1.0, mask_yzero, vl);
    vfloat64m4_t vinv_2y = __riscv_vfrdiv_vf_f64m4(vy_safe, 0.5, vl);
    //vfloat64m4_t vinv_2y = __riscv_vfmul_vf_f64m4(__riscv_vfrsqrt7_v_f64m4(vw_hi, vl), 0.5, vl); ---
    vfloat64m4_t vy_lo   = __riscv_vfmul_vv_f64m4(vs_res, vinv_2y, vl);

    vfloat64m4_t vzero      = __riscv_vfmv_v_f_f64m4(0.0, vl);
    vfloat64m4_t vu_hi      = __riscv_vmerge_vvm_f64m4(vax, vy_hi, is_large, vl);
    vfloat64m4_t vu_lo      = __riscv_vmerge_vvm_f64m4(vzero, vy_lo, is_large, vl);
    vfloat64m4_t vsqrt_u_hi = __riscv_vmerge_vvm_f64m4(vy_hi, vax, is_large, vl);
    vfloat64m4_t vsqrt_u_lo = __riscv_vmerge_vvm_f64m4(vy_lo, vzero, is_large, vl);

    vfloat64m4_t vA_hi, vA_lo, vasin_r_hi, vasin_r_lo;
    __riscv_vasin_kernel_f64m4(vu_hi, vu_lo, vsqrt_u_hi, vsqrt_u_lo,
                               vA_hi, vA_lo, vasin_r_hi, vasin_r_lo, vl);

    vfloat64m4_t vsign = __riscv_vmerge_vvm_f64m4(__riscv_vfsgnjn_vv_f64m4(vx, vx, vl), vx, is_large, vl);

    vfloat64m4_t vsA_hi    = __riscv_vfsgnjx_vv_f64m4(vA_hi, vsign, vl);
    vfloat64m4_t vsA_lo    = __riscv_vfsgnjx_vv_f64m4(vA_lo, vsign, vl);
    vfloat64m4_t vs_asin_h = __riscv_vfsgnjx_vv_f64m4(vasin_r_hi, vsign, vl);
    vfloat64m4_t vs_asin_l = __riscv_vfsgnjx_vv_f64m4(vasin_r_lo, vsign, vl);

    vfloat64m4_t vbase_lrg_hi = __riscv_vfmerge_vfm_f64m4(vzero, PI_HI, is_neg, vl);
    vfloat64m4_t vbase_lrg_lo = __riscv_vfmerge_vfm_f64m4(vzero, PI_LO, is_neg, vl);
    vfloat64m4_t vbase_hi     = __riscv_vmerge_vvm_f64m4(__riscv_vfmv_v_f_f64m4(PI_2_HI, vl), vbase_lrg_hi, is_large, vl);
    vfloat64m4_t vbase_lo     = __riscv_vmerge_vvm_f64m4(__riscv_vfmv_v_f_f64m4(PI_2_LO, vl), vbase_lrg_lo, is_large, vl);

    vfloat64m4_t vsum4_hi  = __riscv_vfadd_vv_f64m4(vbase_hi, vsA_hi, vl);
    vfloat64m4_t vv_sum4   = __riscv_vfsub_vv_f64m4(vsum4_hi, vbase_hi, vl);
    vfloat64m4_t vsum4_err = __riscv_vfsub_vv_f64m4(vbase_hi, __riscv_vfsub_vv_f64m4(vsum4_hi, vv_sum4, vl), vl); 
    vsum4_err              = __riscv_vfadd_vv_f64m4(vsum4_err, __riscv_vfsub_vv_f64m4(vsA_hi, vv_sum4, vl), vl);
    vfloat64m4_t vsum4_lo  = __riscv_vfadd_vv_f64m4(__riscv_vfadd_vv_f64m4(vbase_lo, vsA_lo, vl), vsum4_err, vl);

    vfloat64m4_t vres_hi   = __riscv_vfadd_vv_f64m4(vsum4_hi, vs_asin_h, vl);
    vfloat64m4_t vres_err  = __riscv_vfadd_vv_f64m4(__riscv_vfsub_vv_f64m4(vsum4_hi, vres_hi, vl), vs_asin_h, vl);
    vfloat64m4_t vres_lo   = __riscv_vfadd_vv_f64m4(__riscv_vfadd_vv_f64m4(vsum4_lo, vs_asin_l, vl), vres_err, vl);

    vfloat64m4_t vres = __riscv_vfadd_vv_f64m4(vres_hi, vres_lo, vl);

#ifndef __FAST_MATH__
    vres = __riscv_vmerge_vvm_f64m4(vres, vx, mask_qnan, vl);

    if (__riscv_vcpop_m_b16(mask_inv, vl) > 0) {
        volatile double zero = 0.0;
        double nan_val = zero / zero;
        vres = __riscv_vfmerge_vfm_f64m4(vres, nan_val, mask_inv, vl);
    }
#endif

    return vres;
}

vfloat64m8_t __riscv_vasin_f64m8(vfloat64m8_t x, size_t avl)
{ 
    vfloat64m8_t res;
    size_t vl       = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1              = __riscv_vasin_f64m4(x1, vl);
    res             = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if(avl > vl){
        vl  = __riscv_vsetvl_e64m4(avl-vl);
        x1  = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1  = __riscv_vasin_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}

vfloat64m8_t __riscv_vacos_f64m8(vfloat64m8_t x, size_t avl)
{ 
    vfloat64m8_t res;
    size_t vl       = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1              = __riscv_vacos_f64m4(x1, vl);
    res             = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if(avl > vl){
        vl  = __riscv_vsetvl_e64m4(avl-vl);
        x1  = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1  = __riscv_vacos_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}

#endif /* __riscv_v_intrinsic */
