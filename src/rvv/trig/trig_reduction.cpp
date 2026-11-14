/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025-2026 The RVVMF Authors  
 * (Anastasia Vodeneeva, Valentin Volokitin)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  trig_reduction.cpp                           *
 *   Contains: intrinsic functions for argument          *
 *             reduction for float64, float32, float16   *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Reduction of input vector V to [0, Pi/4]              *
 *                                                       *
 * Algorithm:                                            *
 *    1) Payne-Hanek algorithm to [0, Pi/2] and          * 
 *       Cody-Waite for small argument                   *
 *    2) Modifed on [Pi/4, Pi/2] returns Pi/2-red(V)     *
 *    3) Only for f64m1, f64m2, f64m4, f32m1, f32m2,     *
 *       f32m4, f16m1, f16m2, f16m4                      *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

#include <cstdint>
#include <cfloat>
#include <cmath>

#include "trig_reduction.h"

static constexpr uint32_t TwoOnPI[] = { 0x00000000,
    0x28be60db, 0x24e44152, 0x27f09d5f, 0x11f534dd,
    0x3036d8a5, 0x1993c439, 0x0107f945, 0x23abdebb,
    0x31586dc9, 0x06e3a424, 0x374b8019, 0x092eea09,
    0x3464873f, 0x21deb1cb, 0x04a69cfb, 0x288235f5,
    0x0baed121, 0x0e99c702, 0x1ad17df9, 0x013991d6,
    0x0e60d4ce, 0x1f49c845, 0x3e2ef7e4, 0x283b1ff8,
    0x25fff781, 0x1980fef2, 0x3c462d68, 0x0a6d1f6d,
    0x0d9fb3c9, 0x3cb09b74, 0x3d18fd9a, 0x1e5fea2d,
    0x1d49eeb1, 0x3ebe5f17, 0x2cf41ce7, 0x378a5292,
    0x3a9afed7, 0x3b11f8d5, 0x3421580c, 0x3046fc7b
};

static constexpr uint16_t TwoOnPI_f32[] = { 0x0000, 
     0x517c, 0x60db, 0x49c8, 0x4152, 0x4fe1, 0x1d5f, 0x23ea, 0x34dd, 
     0x606d, 0x58a5, 0x3327, 0x4439, 0x020f, 0x7945, 0x4757, 0x6ebb
};

static constexpr uint8_t TwoOnPI_f16[] = { 
    0x00, 0x51, 0x3e, 0x30, 0x36, 0x72, 0x39, 0x08, 0x15, 0x14
};

inline void __riscv_vred_paynehanek_f64m1(vfloat64m1_t vx, vfloat64m1_t& redx, vfloat64m1_t& redx_lo, 
                                          vuint64m1_t& quadrants, vbool64_t& mask_pi_4, size_t avl) {
    constexpr double PI_DIV_2 = 0x1.921fb54442d18p0;
    constexpr double PI_DIV_2_hi = 0x1.921fb54442d18p0;
    constexpr double PI_DIV_2_lo = 0x1.1a62633145c07p-54;
    constexpr double PI_DIV_4 = 0x1.921fb54442d18p-1;
    size_t vl = __riscv_vsetvl_e64m1(avl);
    
    vbool64_t vfast_mask = __riscv_vmflt_vf_f64m1_b64(vx, PI_DIV_4, vl);
    
    vuint64m1_t vix  = __riscv_vreinterpret_v_f64m1_u64m1(__riscv_vfmerge_vfm_f64m1(vx, 1.75, vfast_mask, vl));
    vuint64m1_t vexp = __riscv_vsrl_vx_u64m1(vix, 52, vl);
    vexp             = __riscv_vand_vx_u64m1(vexp, 0x7ff, vl);
    
    vint64m1_t vshift    = __riscv_vsub_vx_i64m1(__riscv_vreinterpret_v_u64m1_i64m1(vexp), 1023, vl);
    vint64m1_t vshift_pi = __riscv_vdiv_vx_i64m1(__riscv_vsub_vx_i64m1(vshift, 26, vl), 30, vl);
    
    vuint64m1_t vushift_pi_u64 = __riscv_vreinterpret_v_i64m1_u64m1(vshift_pi);
    vuint32mf2_t vushift_pi    = __riscv_vncvt_x_x_w_u32mf2(__riscv_vsll_vx_u64m1(vushift_pi_u64, 2, vl), vl); 
    
    vuint32mf2_t vmant1 = __riscv_vncvt_x_x_w_u32mf2(vix, vl);
    vuint32mf2_t vmant0 = __riscv_vnsrl_wx_u32mf2(vix, 30, vl);
    vmant1 = __riscv_vand_vx_u32mf2(vmant1, 0x3fffffff, vl);
    vmant0 = __riscv_vand_vx_u32mf2(vmant0, 0x3fffff, vl);
    vmant0 = __riscv_vor_vx_u32mf2(vmant0, 0x400000, vl);

    vuint32mf2_t vtwoOnPi0 = __riscv_vloxei32_v_u32mf2(TwoOnPI, vushift_pi, vl);
    vuint32mf2_t vtwoOnPi1 = __riscv_vloxei32_v_u32mf2(TwoOnPI, __riscv_vadd_vx_u32mf2(vushift_pi, 4, vl), vl);
    vuint32mf2_t vtwoOnPi2 = __riscv_vloxei32_v_u32mf2(TwoOnPI, __riscv_vadd_vx_u32mf2(vushift_pi, 8, vl), vl);
    vuint32mf2_t vtwoOnPi3 = __riscv_vloxei32_v_u32mf2(TwoOnPI, __riscv_vadd_vx_u32mf2(vushift_pi, 12, vl), vl);
    vuint32mf2_t vtwoOnPi4 = __riscv_vloxei32_v_u32mf2(TwoOnPI, __riscv_vadd_vx_u32mf2(vushift_pi, 16, vl), vl);
    vuint32mf2_t vtwoOnPi5 = __riscv_vloxei32_v_u32mf2(TwoOnPI, __riscv_vadd_vx_u32mf2(vushift_pi, 20, vl), vl);
    vuint32mf2_t vtwoOnPi6 = __riscv_vloxei32_v_u32mf2(TwoOnPI, __riscv_vadd_vx_u32mf2(vushift_pi, 24, vl), vl);

    vuint64m1_t vr0 = __riscv_vwmaccu_vv_u64m1(__riscv_vwmulu_vv_u64m1(vtwoOnPi1, vmant0, vl), vtwoOnPi0, vmant1, vl);
    vuint64m1_t vr1 = __riscv_vwmaccu_vv_u64m1(__riscv_vwmulu_vv_u64m1(vtwoOnPi2, vmant0, vl), vtwoOnPi1, vmant1, vl);
    vuint64m1_t vr2 = __riscv_vwmaccu_vv_u64m1(__riscv_vwmulu_vv_u64m1(vtwoOnPi3, vmant0, vl), vtwoOnPi2, vmant1, vl);
    vuint64m1_t vr3 = __riscv_vwmaccu_vv_u64m1(__riscv_vwmulu_vv_u64m1(vtwoOnPi4, vmant0, vl), vtwoOnPi3, vmant1, vl);
    vuint64m1_t vr4 = __riscv_vwmaccu_vv_u64m1(__riscv_vwmulu_vv_u64m1(vtwoOnPi5, vmant0, vl), vtwoOnPi4, vmant1, vl);
    vuint64m1_t vr5 = __riscv_vwmaccu_vv_u64m1(__riscv_vwmulu_vv_u64m1(vtwoOnPi6, vmant0, vl), vtwoOnPi5, vmant1, vl);
       
    vuint64m1_t vtmp;
    vr4  = __riscv_vadd_vv_u64m1(vr4, __riscv_vsrl_vx_u64m1(vr5, 30, vl), vl);
    vtmp = __riscv_vsrl_vx_u64m1(vr4, 30, vl);
    vr3  = __riscv_vadd_vv_u64m1(vr3, vtmp, vl);
    vr4  = __riscv_vsub_vv_u64m1(vr4, __riscv_vsll_vx_u64m1(vtmp, 30, vl), vl);
    vtmp = __riscv_vsrl_vx_u64m1(vr3, 30, vl);
    vr2  = __riscv_vadd_vv_u64m1(vr2, vtmp, vl);
	vr3  = __riscv_vsub_vv_u64m1(vr3, __riscv_vsll_vx_u64m1(vtmp, 30, vl), vl);
    vtmp = __riscv_vsrl_vx_u64m1(vr2, 30, vl);
	vr1  = __riscv_vadd_vv_u64m1(vr1, vtmp, vl);
	vr2  = __riscv_vsub_vv_u64m1(vr2, __riscv_vsll_vx_u64m1(vtmp, 30, vl), vl);
    vtmp = __riscv_vsrl_vx_u64m1(vr1, 30, vl);
	vr0  = __riscv_vadd_vv_u64m1(vr0, vtmp, vl);
	vr1  = __riscv_vsub_vv_u64m1(vr1, __riscv_vsll_vx_u64m1(vtmp, 30, vl), vl);
    
    vuint64m1_t vr0_packed = __riscv_vor_vv_u64m1(__riscv_vsll_vx_u64m1(vr0, 4, vl), __riscv_vsrl_vx_u64m1(vr1, 26, vl), vl);
    vuint64m1_t vr1_packed = __riscv_vor_vv_u64m1(__riscv_vor_vv_u64m1(__riscv_vsll_vx_u64m1(vr1, 38, vl), 
                                                                       __riscv_vsll_vx_u64m1(vr2, 8, vl), vl), 
                                                                       __riscv_vsrl_vx_u64m1(vr3, 22, vl), vl);
    vuint64m1_t vr2_packed = __riscv_vor_vv_u64m1(__riscv_vor_vv_u64m1(__riscv_vsll_vx_u64m1(vr3, 42, vl), 
                                                                       __riscv_vsll_vx_u64m1(vr4, 12, vl), vl), 
                                                                       __riscv_vsrl_vx_u64m1(vr5, 18, vl), vl);

    vuint64m1_t vfinal_shift        = __riscv_vreinterpret_v_i64m1_u64m1(__riscv_vsub_vv_i64m1(vshift,
                                                    __riscv_vmul_vx_i64m1(vshift_pi, 30, vl), vl));
    vuint64m1_t vshift_amount_right = __riscv_vrsub_vx_u64m1(vfinal_shift, 56, vl);
    vuint64m1_t vshift_amount_left  = __riscv_vadd_vx_u64m1(vfinal_shift, 8, vl);
    
    vuint64m1_t vr2_final = __riscv_vor_vv_u64m1(__riscv_vsrl_vv_u64m1(vr2_packed, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u64m1(vr1_packed, vshift_amount_left, vl), vl);
    vuint64m1_t vr1_final = __riscv_vor_vv_u64m1(__riscv_vsrl_vv_u64m1(vr1_packed, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u64m1(vr0_packed, vshift_amount_left, vl), vl);
    vuint64m1_t vr0_final = __riscv_vsrl_vv_u64m1(vr0_packed, vshift_amount_right, vl);
    
    vuint64m1_t vr1_final_hi = __riscv_vsrl_vx_u64m1(vr1_final, 32, vl);
    vuint64m1_t vr1_final_lo = __riscv_vand_vx_u64m1(vr1_final, 0xffffffff, vl);
    vuint64m1_t vr2_final_hi = __riscv_vsrl_vx_u64m1(vr2_final, 32, vl);
    vuint64m1_t vr2_final_lo = __riscv_vand_vx_u64m1(vr2_final, 0xffffffff, vl);
    
    vfloat64m1_t vres_r1, vres_r2, vmres_r1, vmres_r2;
    vres_r2 = __riscv_vfcvt_f_xu_v_f64m1(vr1_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f64m1(vr1_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f64m1(vres_r2, 0x1.0p-64, vl);
    vres_r1 = __riscv_vfmul_vf_f64m1(vres_r1, 0x1.0p-32, vl);
    
    mask_pi_4  = __riscv_vmfge_vf_f64m1_b64(vres_r1, 0.5, vl); 
    vmres_r2   = __riscv_vfmul_vf_f64m1(vres_r2, -0x1.0p0, vl);
    vmres_r1   = __riscv_vfmul_vf_f64m1(vres_r1, -0x1.0p0, vl);
    vmres_r1   = __riscv_vfadd_vf_f64m1(vmres_r1, 0x1.0p0, vl);
    vres_r1    = __riscv_vmerge_vvm_f64m1(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2    = __riscv_vmerge_vvm_f64m1(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat64m1_t vr1_htmp, vr1_tmp;
    vr1_htmp = __riscv_vfadd_vv_f64m1(vres_r1, vres_r2, vl);
    vr1_tmp  = __riscv_vfsub_vv_f64m1(vr1_htmp, vres_r1, vl);
    vr1_tmp  = __riscv_vfsub_vv_f64m1(vr1_tmp, vres_r2, vl);
    
    vres_r2  = __riscv_vfcvt_f_xu_v_f64m1(vr2_final_lo, vl);
    vres_r1  = __riscv_vfcvt_f_xu_v_f64m1(vr2_final_hi, vl);
    vres_r2  = __riscv_vfmul_vf_f64m1(vres_r2, 0x1.0p-128, vl);
    vres_r1  = __riscv_vfmul_vf_f64m1(vres_r1, 0x1.0p-96, vl);
    
    vmres_r2 = __riscv_vfmul_vf_f64m1(vres_r2, -0x1.0p0, vl);
    vmres_r1 = __riscv_vfmul_vf_f64m1(vres_r1, -0x1.0p0, vl);
    vres_r1  = __riscv_vmerge_vvm_f64m1(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2  = __riscv_vmerge_vvm_f64m1(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat64m1_t vr1_hi, vr2_hi, vr1_lo;
    vr2_hi  = __riscv_vfadd_vv_f64m1(vres_r1, vres_r2, vl);
    vr1_lo  = __riscv_vfsub_vv_f64m1(vr2_hi, vr1_tmp, vl);
    vr1_tmp = __riscv_vfadd_vv_f64m1(vr1_lo, vr1_tmp, vl);
    vr2_hi  = __riscv_vfsub_vv_f64m1(vr2_hi, vr1_tmp, vl);
    
    vr1_hi  = __riscv_vfadd_vv_f64m1(vr1_htmp, vr1_lo, vl);
    vr1_tmp = __riscv_vfsub_vv_f64m1(vr1_hi, vr1_htmp, vl);
    vr1_lo  = __riscv_vfsub_vv_f64m1(vr1_tmp, vr1_lo, vl);
    
    redx    = __riscv_vfmul_vf_f64m1(vr1_hi, PI_DIV_2_hi, vl);
    redx_lo = __riscv_vfmadd_vf_f64m1(vr1_hi, -PI_DIV_2_hi, redx, vl); 
    redx_lo = __riscv_vfmacc_vf_f64m1(redx_lo, PI_DIV_2_hi, vr1_lo, vl);
    redx_lo = __riscv_vfmsac_vf_f64m1(redx_lo, PI_DIV_2_lo, vr1_hi, vl);
    
    vbool64_t m = __riscv_vmfgt_vf_f64m1_b64(redx, 0x1.921fb54442d18p-1, vl);
    vr1_hi      = __riscv_vfrsub_vf_f64m1(redx, 0x1.921fb54442d18p0, vl);
    vr1_lo      = __riscv_vfrsub_vf_f64m1(redx_lo, 0x1.1a62633145c07p-54, vl);
    vres_r1     = __riscv_vfadd_vv_f64m1(vr1_hi, vr1_lo, vl);
    vres_r2     = __riscv_vfsub_vv_f64m1(vr1_lo, __riscv_vfsub_vv_f64m1(vres_r1, vr1_hi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f64m1(redx, vres_r1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f64m1(redx_lo, vres_r2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b64(mask_pi_4, m, vl);
    
    vuint64m1_t slow_path_quadrant = __riscv_vand_vx_u64m1(vr0_final, 0x3, vl);

    redx    = __riscv_vmerge_vvm_f64m1(redx, vx, vfast_mask, vl);
    redx_lo = __riscv_vfmerge_vfm_f64m1(redx_lo, 0.0, vfast_mask, vl);
    
    mask_pi_4 = __riscv_vmand_mm_b64(mask_pi_4, __riscv_vmnot_m_b64(vfast_mask, vl), vl);
    quadrants = __riscv_vmerge_vxm_u64m1(slow_path_quadrant, 0, vfast_mask, vl);
}

inline void __riscv_vred_codywaite_f64m1(vfloat64m1_t vx, vfloat64m1_t& redx, vfloat64m1_t& redx_lo, 
                                         vuint64m1_t& quadrants, vbool64_t& mask_pi_4, size_t avl) 
{
    constexpr double INV_PI_DIV_2 = 0x1.45f306dc9c883p-1;  
    constexpr double PI_2_HI      = 0x1.921fb54400000p+0;  
    constexpr double PI_2_MID     = 0x1.0b4611a600000p-34; 
    constexpr double PI_2_LO      = 0x1.3198a2e037073p-69;

    size_t vl = __riscv_vsetvl_e64m1(avl);

    vfloat64m1_t vfn  = __riscv_vfmul_vf_f64m1(vx, INV_PI_DIV_2, vl);
    vuint64m1_t  vifn = __riscv_vfcvt_xu_f_v_u64m1(vfn, vl);
    vfn               = __riscv_vfcvt_f_xu_v_f64m1(vifn, vl);
    
    vfloat64m1_t vr2;
    vfloat64m1_t vr1  = __riscv_vfnmsac_vf_f64m1(vx, PI_2_HI, vfn, vl);
    vfloat64m1_t vrhi = __riscv_vfnmsac_vf_f64m1(vr1, PI_2_MID, vfn, vl);
    vfloat64m1_t vrlo = __riscv_vfnmsac_vf_f64m1(__riscv_vfsub_vv_f64m1(vr1, vrhi, vl), PI_2_MID, vfn, vl);
    vrlo              = __riscv_vfnmsac_vf_f64m1(vrlo, PI_2_LO, vfn, vl);
    
    mask_pi_4 = __riscv_vmflt_vf_f64m1_b64(vrhi, 0.0, vl);

    vuint64m1_t vsign = __riscv_vsrl_vx_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(vrhi), 63, vl);
    quadrants         = __riscv_vand_vx_u64m1(__riscv_vsub_vv_u64m1(vifn, vsign, vl), 3, vl);

    vrlo = __riscv_vfsgnjx_vv_f64m1(vrlo, vrhi, vl);
    vrhi = __riscv_vfabs_v_f64m1(vrhi, vl);

    redx    = __riscv_vfadd_vv_f64m1(vrhi, vrlo, vl);
    redx_lo = __riscv_vfsub_vv_f64m1(vrlo, __riscv_vfsub_vv_f64m1(redx, vrhi, vl), vl);  

    vbool64_t m = __riscv_vmfgt_vf_f64m1_b64(redx, 0x1.921fb54442d18p-1, vl);
    vrhi        = __riscv_vfrsub_vf_f64m1(redx, 0x1.921fb54442d18p0, vl);
    vrlo        = __riscv_vfrsub_vf_f64m1(redx_lo, 0x1.1a62633145c07p-54, vl);
    vr1         = __riscv_vfadd_vv_f64m1(vrhi, vrlo, vl);
    vr2         = __riscv_vfsub_vv_f64m1(vrlo, __riscv_vfsub_vv_f64m1(vr1, vrhi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f64m1(redx, vr1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f64m1(redx_lo, vr2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b64(mask_pi_4, m, vl);
} 
 
void __riscv_vreduction_pi4_f64m1(vfloat64m1_t vx, vfloat64m1_t& redx, vfloat64m1_t& redx_lo, 
                                  vuint64m1_t& quadrants, vbool64_t& mask_pi_4, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m1(avl);
    vx        = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vx_u64m1(
                            __riscv_vreinterpret_v_f64m1_u64m1(vx), 0x7fffffffffffffff, vl));
    mask_pi_4 = __riscv_vmfgt_vf_f64m1_b64(vx, 0x1p20, vl);
    if(__riscv_vcpop_m_b64(mask_pi_4, vl))
        __riscv_vred_paynehanek_f64m1(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
    else
        __riscv_vred_codywaite_f64m1(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
}


inline void __riscv_vred_paynehanek_f64m2(vfloat64m2_t vx, vfloat64m2_t& redx, vfloat64m2_t& redx_lo, 
                                          vuint64m2_t& quadrants, vbool32_t& mask_pi_4, size_t avl) {
    constexpr double PI_DIV_2 = 0x1.921fb54442d18p0;
    constexpr double PI_DIV_2_hi = 0x1.921fb54442d18p0;
    constexpr double PI_DIV_2_lo = 0x1.1a62633145c07p-54;
    constexpr double PI_DIV_4 = 0x1.921fb54442d18p-1;
    size_t vl = __riscv_vsetvl_e64m2(avl);
    
    vbool32_t vfast_mask = __riscv_vmflt_vf_f64m2_b32(vx, PI_DIV_4, vl);
    
    vuint64m2_t vix  = __riscv_vreinterpret_v_f64m2_u64m2(__riscv_vfmerge_vfm_f64m2(vx, 1.75, vfast_mask, vl));
    vuint64m2_t vexp = __riscv_vsrl_vx_u64m2(vix, 52, vl);
    vexp             = __riscv_vand_vx_u64m2(vexp, 0x7ff, vl);
    
    vint64m2_t vshift    = __riscv_vsub_vx_i64m2(__riscv_vreinterpret_v_u64m2_i64m2(vexp), 1023, vl);
    vint64m2_t vshift_pi = __riscv_vdiv_vx_i64m2(__riscv_vsub_vx_i64m2(vshift, 26, vl), 30, vl);
    
    vuint64m2_t vushift_pi_u64 = __riscv_vreinterpret_v_i64m2_u64m2(vshift_pi);
    vuint32m1_t vushift_pi     = __riscv_vncvt_x_x_w_u32m1(__riscv_vsll_vx_u64m2(vushift_pi_u64, 2, vl), vl); 
    
    vuint32m1_t vmant1 = __riscv_vncvt_x_x_w_u32m1(vix, vl);
    vuint32m1_t vmant0 = __riscv_vnsrl_wx_u32m1(vix, 30, vl);
    vmant1 = __riscv_vand_vx_u32m1(vmant1, 0x3fffffff, vl);
    vmant0 = __riscv_vand_vx_u32m1(vmant0, 0x3fffff, vl);
    vmant0 = __riscv_vor_vx_u32m1(vmant0, 0x400000, vl);

    vuint32m1_t vtwoOnPi0 = __riscv_vloxei32_v_u32m1(TwoOnPI, vushift_pi, vl);
    vuint32m1_t vtwoOnPi1 = __riscv_vloxei32_v_u32m1(TwoOnPI, __riscv_vadd_vx_u32m1(vushift_pi, 4, vl), vl);
    vuint32m1_t vtwoOnPi2 = __riscv_vloxei32_v_u32m1(TwoOnPI, __riscv_vadd_vx_u32m1(vushift_pi, 8, vl), vl);
    vuint32m1_t vtwoOnPi3 = __riscv_vloxei32_v_u32m1(TwoOnPI, __riscv_vadd_vx_u32m1(vushift_pi, 12, vl), vl);
    vuint32m1_t vtwoOnPi4 = __riscv_vloxei32_v_u32m1(TwoOnPI, __riscv_vadd_vx_u32m1(vushift_pi, 16, vl), vl);
    vuint32m1_t vtwoOnPi5 = __riscv_vloxei32_v_u32m1(TwoOnPI, __riscv_vadd_vx_u32m1(vushift_pi, 20, vl), vl);
    vuint32m1_t vtwoOnPi6 = __riscv_vloxei32_v_u32m1(TwoOnPI, __riscv_vadd_vx_u32m1(vushift_pi, 24, vl), vl);

    vuint64m2_t vr0 = __riscv_vwmaccu_vv_u64m2(__riscv_vwmulu_vv_u64m2(vtwoOnPi1, vmant0, vl), vtwoOnPi0, vmant1, vl);
    vuint64m2_t vr1 = __riscv_vwmaccu_vv_u64m2(__riscv_vwmulu_vv_u64m2(vtwoOnPi2, vmant0, vl), vtwoOnPi1, vmant1, vl);
    vuint64m2_t vr2 = __riscv_vwmaccu_vv_u64m2(__riscv_vwmulu_vv_u64m2(vtwoOnPi3, vmant0, vl), vtwoOnPi2, vmant1, vl);
    vuint64m2_t vr3 = __riscv_vwmaccu_vv_u64m2(__riscv_vwmulu_vv_u64m2(vtwoOnPi4, vmant0, vl), vtwoOnPi3, vmant1, vl);
    vuint64m2_t vr4 = __riscv_vwmaccu_vv_u64m2(__riscv_vwmulu_vv_u64m2(vtwoOnPi5, vmant0, vl), vtwoOnPi4, vmant1, vl);
    vuint64m2_t vr5 = __riscv_vwmaccu_vv_u64m2(__riscv_vwmulu_vv_u64m2(vtwoOnPi6, vmant0, vl), vtwoOnPi5, vmant1, vl);
       
    vuint64m2_t vtmp;
    vr4  = __riscv_vadd_vv_u64m2(vr4, __riscv_vsrl_vx_u64m2(vr5, 30, vl), vl);
    vtmp = __riscv_vsrl_vx_u64m2(vr4, 30, vl);
    vr3  = __riscv_vadd_vv_u64m2(vr3, vtmp, vl);
    vr4  = __riscv_vsub_vv_u64m2(vr4, __riscv_vsll_vx_u64m2(vtmp, 30, vl), vl);
    vtmp = __riscv_vsrl_vx_u64m2(vr3, 30, vl);
    vr2  = __riscv_vadd_vv_u64m2(vr2, vtmp, vl);
	vr3  = __riscv_vsub_vv_u64m2(vr3, __riscv_vsll_vx_u64m2(vtmp, 30, vl), vl);
    vtmp = __riscv_vsrl_vx_u64m2(vr2, 30, vl);
	vr1  = __riscv_vadd_vv_u64m2(vr1, vtmp, vl);
	vr2  = __riscv_vsub_vv_u64m2(vr2, __riscv_vsll_vx_u64m2(vtmp, 30, vl), vl);
    vtmp = __riscv_vsrl_vx_u64m2(vr1, 30, vl);
	vr0  = __riscv_vadd_vv_u64m2(vr0, vtmp, vl);
	vr1  = __riscv_vsub_vv_u64m2(vr1, __riscv_vsll_vx_u64m2(vtmp, 30, vl), vl);
    
    vuint64m2_t vr0_packed = __riscv_vor_vv_u64m2(__riscv_vsll_vx_u64m2(vr0, 4, vl), __riscv_vsrl_vx_u64m2(vr1, 26, vl), vl);
    vuint64m2_t vr1_packed = __riscv_vor_vv_u64m2(__riscv_vor_vv_u64m2(__riscv_vsll_vx_u64m2(vr1, 38, vl), 
                                                                       __riscv_vsll_vx_u64m2(vr2, 8, vl), vl), 
                                                                       __riscv_vsrl_vx_u64m2(vr3, 22, vl), vl);
    vuint64m2_t vr2_packed = __riscv_vor_vv_u64m2(__riscv_vor_vv_u64m2(__riscv_vsll_vx_u64m2(vr3, 42, vl), 
                                                                       __riscv_vsll_vx_u64m2(vr4, 12, vl), vl), 
                                                                       __riscv_vsrl_vx_u64m2(vr5, 18, vl), vl);

    vuint64m2_t vfinal_shift        = __riscv_vreinterpret_v_i64m2_u64m2(__riscv_vsub_vv_i64m2(vshift,
                                                    __riscv_vmul_vx_i64m2(vshift_pi, 30, vl), vl));
    vuint64m2_t vshift_amount_right = __riscv_vrsub_vx_u64m2(vfinal_shift, 56, vl);
    vuint64m2_t vshift_amount_left  = __riscv_vadd_vx_u64m2(vfinal_shift, 8, vl);
    
    vuint64m2_t vr2_final = __riscv_vor_vv_u64m2(__riscv_vsrl_vv_u64m2(vr2_packed, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u64m2(vr1_packed, vshift_amount_left, vl), vl);
    vuint64m2_t vr1_final = __riscv_vor_vv_u64m2(__riscv_vsrl_vv_u64m2(vr1_packed, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u64m2(vr0_packed, vshift_amount_left, vl), vl);
    vuint64m2_t vr0_final = __riscv_vsrl_vv_u64m2(vr0_packed, vshift_amount_right, vl);
    
    vuint64m2_t vr1_final_hi = __riscv_vsrl_vx_u64m2(vr1_final, 32, vl);
    vuint64m2_t vr1_final_lo = __riscv_vand_vx_u64m2(vr1_final, 0xffffffff, vl);
    vuint64m2_t vr2_final_hi = __riscv_vsrl_vx_u64m2(vr2_final, 32, vl);
    vuint64m2_t vr2_final_lo = __riscv_vand_vx_u64m2(vr2_final, 0xffffffff, vl);
    
    vfloat64m2_t vres_r1, vres_r2, vmres_r1, vmres_r2;
    vres_r2 = __riscv_vfcvt_f_xu_v_f64m2(vr1_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f64m2(vr1_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f64m2(vres_r2, 0x1.0p-64, vl);
    vres_r1 = __riscv_vfmul_vf_f64m2(vres_r1, 0x1.0p-32, vl);
    
    mask_pi_4  = __riscv_vmfge_vf_f64m2_b32(vres_r1, 0.5, vl); 
    vmres_r2   = __riscv_vfmul_vf_f64m2(vres_r2, -0x1.0p0, vl);
    vmres_r1   = __riscv_vfmul_vf_f64m2(vres_r1, -0x1.0p0, vl);
    vmres_r1   = __riscv_vfadd_vf_f64m2(vmres_r1, 0x1.0p0, vl);
    vres_r1    = __riscv_vmerge_vvm_f64m2(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2    = __riscv_vmerge_vvm_f64m2(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat64m2_t vr1_htmp, vr1_tmp;
    vr1_htmp = __riscv_vfadd_vv_f64m2(vres_r1, vres_r2, vl);
    vr1_tmp  = __riscv_vfsub_vv_f64m2(vr1_htmp, vres_r1, vl);
    vr1_tmp  = __riscv_vfsub_vv_f64m2(vr1_tmp, vres_r2, vl);
    
    vres_r2  = __riscv_vfcvt_f_xu_v_f64m2(vr2_final_lo, vl);
    vres_r1  = __riscv_vfcvt_f_xu_v_f64m2(vr2_final_hi, vl);
    vres_r2  = __riscv_vfmul_vf_f64m2(vres_r2, 0x1.0p-128, vl);
    vres_r1  = __riscv_vfmul_vf_f64m2(vres_r1, 0x1.0p-96, vl);
    
    vmres_r2 = __riscv_vfmul_vf_f64m2(vres_r2, -0x1.0p0, vl);
    vmres_r1 = __riscv_vfmul_vf_f64m2(vres_r1, -0x1.0p0, vl);
    vres_r1  = __riscv_vmerge_vvm_f64m2(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2  = __riscv_vmerge_vvm_f64m2(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat64m2_t vr1_hi, vr2_hi, vr1_lo;
    vr2_hi  = __riscv_vfadd_vv_f64m2(vres_r1, vres_r2, vl);
    vr1_lo  = __riscv_vfsub_vv_f64m2(vr2_hi, vr1_tmp, vl);
    vr1_tmp = __riscv_vfadd_vv_f64m2(vr1_lo, vr1_tmp, vl);
    vr2_hi  = __riscv_vfsub_vv_f64m2(vr2_hi, vr1_tmp, vl);
    
    vr1_hi  = __riscv_vfadd_vv_f64m2(vr1_htmp, vr1_lo, vl);
    vr1_tmp = __riscv_vfsub_vv_f64m2(vr1_hi, vr1_htmp, vl);
    vr1_lo  = __riscv_vfsub_vv_f64m2(vr1_tmp, vr1_lo, vl);
    
    redx    = __riscv_vfmul_vf_f64m2(vr1_hi, PI_DIV_2_hi, vl);
    redx_lo = __riscv_vfmadd_vf_f64m2(vr1_hi, -PI_DIV_2_hi, redx, vl); 
    redx_lo = __riscv_vfmacc_vf_f64m2(redx_lo, PI_DIV_2_hi, vr1_lo, vl);
    redx_lo = __riscv_vfmsac_vf_f64m2(redx_lo, PI_DIV_2_lo, vr1_hi, vl);
    
    vbool32_t m = __riscv_vmfgt_vf_f64m2_b32(redx, 0x1.921fb54442d18p-1, vl);
    vr1_hi      = __riscv_vfrsub_vf_f64m2(redx, 0x1.921fb54442d18p0, vl);
    vr1_lo      = __riscv_vfrsub_vf_f64m2(redx_lo, 0x1.1a62633145c07p-54, vl);
    vres_r1     = __riscv_vfadd_vv_f64m2(vr1_hi, vr1_lo, vl);
    vres_r2     = __riscv_vfsub_vv_f64m2(vr1_lo, __riscv_vfsub_vv_f64m2(vres_r1, vr1_hi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f64m2(redx, vres_r1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f64m2(redx_lo, vres_r2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b32(mask_pi_4, m, vl);
    
    vuint64m2_t slow_path_quadrant = __riscv_vand_vx_u64m2(vr0_final, 0x3, vl);

    redx    = __riscv_vmerge_vvm_f64m2(redx, vx, vfast_mask, vl);
    redx_lo = __riscv_vfmerge_vfm_f64m2(redx_lo, 0.0, vfast_mask, vl);
    
    mask_pi_4 = __riscv_vmand_mm_b32(mask_pi_4, __riscv_vmnot_m_b32(vfast_mask, vl), vl);
    quadrants = __riscv_vmerge_vxm_u64m2(slow_path_quadrant, 0, vfast_mask, vl);
}

inline void __riscv_vred_codywaite_f64m2(vfloat64m2_t vx, vfloat64m2_t& redx, vfloat64m2_t& redx_lo, 
                                         vuint64m2_t& quadrants, vbool32_t& mask_pi_4, size_t avl) 
{
    constexpr double INV_PI_DIV_2 = 0x1.45f306dc9c883p-1;  
    constexpr double PI_2_HI      = 0x1.921fb54400000p+0;  
    constexpr double PI_2_MID     = 0x1.0b4611a600000p-34; 
    constexpr double PI_2_LO      = 0x1.3198a2e037073p-69;

    size_t vl = __riscv_vsetvl_e64m2(avl);

    vfloat64m2_t vfn  = __riscv_vfmul_vf_f64m2(vx, INV_PI_DIV_2, vl);
    vuint64m2_t  vifn = __riscv_vfcvt_xu_f_v_u64m2(vfn, vl);
    vfn               = __riscv_vfcvt_f_xu_v_f64m2(vifn, vl);

    vfloat64m2_t vr2;
    vfloat64m2_t vr1  = __riscv_vfnmsac_vf_f64m2(vx, PI_2_HI, vfn, vl);
    vfloat64m2_t vrhi = __riscv_vfnmsac_vf_f64m2(vr1, PI_2_MID, vfn, vl);
    vfloat64m2_t vrlo = __riscv_vfnmsac_vf_f64m2(__riscv_vfsub_vv_f64m2(vr1, vrhi, vl), PI_2_MID, vfn, vl);
    vrlo              = __riscv_vfnmsac_vf_f64m2(vrlo, PI_2_LO, vfn, vl);
    
    mask_pi_4 = __riscv_vmflt_vf_f64m2_b32(vrhi, 0.0, vl);

    vuint64m2_t vsign = __riscv_vsrl_vx_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(vrhi), 63, vl);
    quadrants         = __riscv_vand_vx_u64m2(__riscv_vsub_vv_u64m2(vifn, vsign, vl), 3, vl);

    vrlo = __riscv_vfsgnjx_vv_f64m2(vrlo, vrhi, vl);
    vrhi = __riscv_vfabs_v_f64m2(vrhi, vl);

    redx    = __riscv_vfadd_vv_f64m2(vrhi, vrlo, vl);
    redx_lo = __riscv_vfsub_vv_f64m2(vrlo, __riscv_vfsub_vv_f64m2(redx, vrhi, vl), vl);  
    
    vbool32_t m = __riscv_vmfgt_vf_f64m2_b32(redx, 0x1.921fb54442d18p-1, vl);
    vrhi        = __riscv_vfrsub_vf_f64m2(redx, 0x1.921fb54442d18p0, vl);
    vrlo        = __riscv_vfrsub_vf_f64m2(redx_lo, 0x1.1a62633145c07p-54, vl);
    vr1         = __riscv_vfadd_vv_f64m2(vrhi, vrlo, vl);
    vr2         = __riscv_vfsub_vv_f64m2(vrlo, __riscv_vfsub_vv_f64m2(vr1, vrhi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f64m2(redx, vr1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f64m2(redx_lo, vr2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b32(mask_pi_4, m, vl);
} 
 
void __riscv_vreduction_pi4_f64m2(vfloat64m2_t vx, vfloat64m2_t& redx, vfloat64m2_t& redx_lo, 
                                  vuint64m2_t& quadrants, vbool32_t& mask_pi_4, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m2(avl);
    vx        = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vx_u64m2(
                            __riscv_vreinterpret_v_f64m2_u64m2(vx), 0x7fffffffffffffff, vl));
    mask_pi_4 = __riscv_vmfgt_vf_f64m2_b32(vx, 0x1p20, vl);
    if(__riscv_vcpop_m_b32(mask_pi_4, vl))
        __riscv_vred_paynehanek_f64m2(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
    else
        __riscv_vred_codywaite_f64m2(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
}


inline void __riscv_vred_paynehanek_f64m4(vfloat64m4_t vx, vfloat64m4_t& redx, vfloat64m4_t& redx_lo, 
                                          vuint64m4_t& quadrants, vbool16_t& mask_pi_4, size_t avl) {
    constexpr double PI_DIV_2 = 0x1.921fb54442d18p0;
    constexpr double PI_DIV_2_hi = 0x1.921fb54442d18p0;
    constexpr double PI_DIV_2_lo = 0x1.1a62633145c07p-54;
    constexpr double PI_DIV_4 = 0x1.921fb54442d18p-1;
    size_t vl = __riscv_vsetvl_e64m4(avl);
    
    vbool16_t vfast_mask = __riscv_vmflt_vf_f64m4_b16(vx, PI_DIV_4, vl);
    
    vuint64m4_t vix  = __riscv_vreinterpret_v_f64m4_u64m4(__riscv_vfmerge_vfm_f64m4(vx, 1.75, vfast_mask, vl));
    vuint64m4_t vexp = __riscv_vsrl_vx_u64m4(vix, 52, vl);
    vexp             = __riscv_vand_vx_u64m4(vexp, 0x7ff, vl);
    
    vint64m4_t vshift    = __riscv_vsub_vx_i64m4(__riscv_vreinterpret_v_u64m4_i64m4(vexp), 1023, vl);
    vint64m4_t vshift_pi = __riscv_vdiv_vx_i64m4(__riscv_vsub_vx_i64m4(vshift, 26, vl), 30, vl);
    
    vuint64m4_t vushift_pi_u64 = __riscv_vreinterpret_v_i64m4_u64m4(vshift_pi);
    vuint32m2_t vushift_pi     = __riscv_vncvt_x_x_w_u32m2(__riscv_vsll_vx_u64m4(vushift_pi_u64, 2, vl), vl); 
    
    vuint32m2_t vmant1 = __riscv_vncvt_x_x_w_u32m2(vix, vl);
    vuint32m2_t vmant0 = __riscv_vnsrl_wx_u32m2(vix, 30, vl);
    vmant1 = __riscv_vand_vx_u32m2(vmant1, 0x3fffffff, vl);
    vmant0 = __riscv_vand_vx_u32m2(vmant0, 0x3fffff, vl);
    vmant0 = __riscv_vor_vx_u32m2(vmant0, 0x400000, vl);

    vuint32m2_t vtwoOnPi0 = __riscv_vloxei32_v_u32m2(TwoOnPI, vushift_pi, vl);
    vuint32m2_t vtwoOnPi1 = __riscv_vloxei32_v_u32m2(TwoOnPI, __riscv_vadd_vx_u32m2(vushift_pi, 4, vl), vl);
    vuint32m2_t vtwoOnPi2 = __riscv_vloxei32_v_u32m2(TwoOnPI, __riscv_vadd_vx_u32m2(vushift_pi, 8, vl), vl);
    vuint32m2_t vtwoOnPi3 = __riscv_vloxei32_v_u32m2(TwoOnPI, __riscv_vadd_vx_u32m2(vushift_pi, 12, vl), vl);
    vuint32m2_t vtwoOnPi4 = __riscv_vloxei32_v_u32m2(TwoOnPI, __riscv_vadd_vx_u32m2(vushift_pi, 16, vl), vl);
    vuint32m2_t vtwoOnPi5 = __riscv_vloxei32_v_u32m2(TwoOnPI, __riscv_vadd_vx_u32m2(vushift_pi, 20, vl), vl);
    vuint32m2_t vtwoOnPi6 = __riscv_vloxei32_v_u32m2(TwoOnPI, __riscv_vadd_vx_u32m2(vushift_pi, 24, vl), vl);

    vuint64m4_t vr0 = __riscv_vwmaccu_vv_u64m4(__riscv_vwmulu_vv_u64m4(vtwoOnPi1, vmant0, vl), vtwoOnPi0, vmant1, vl);
    vuint64m4_t vr1 = __riscv_vwmaccu_vv_u64m4(__riscv_vwmulu_vv_u64m4(vtwoOnPi2, vmant0, vl), vtwoOnPi1, vmant1, vl);
    vuint64m4_t vr2 = __riscv_vwmaccu_vv_u64m4(__riscv_vwmulu_vv_u64m4(vtwoOnPi3, vmant0, vl), vtwoOnPi2, vmant1, vl);
    vuint64m4_t vr3 = __riscv_vwmaccu_vv_u64m4(__riscv_vwmulu_vv_u64m4(vtwoOnPi4, vmant0, vl), vtwoOnPi3, vmant1, vl);
    vuint64m4_t vr4 = __riscv_vwmaccu_vv_u64m4(__riscv_vwmulu_vv_u64m4(vtwoOnPi5, vmant0, vl), vtwoOnPi4, vmant1, vl);
    vuint64m4_t vr5 = __riscv_vwmaccu_vv_u64m4(__riscv_vwmulu_vv_u64m4(vtwoOnPi6, vmant0, vl), vtwoOnPi5, vmant1, vl);
       
    vuint64m4_t vtmp;
    vr4  = __riscv_vadd_vv_u64m4(vr4, __riscv_vsrl_vx_u64m4(vr5, 30, vl), vl);
    vtmp = __riscv_vsrl_vx_u64m4(vr4, 30, vl);
    vr3  = __riscv_vadd_vv_u64m4(vr3, vtmp, vl);
    vr4  = __riscv_vsub_vv_u64m4(vr4, __riscv_vsll_vx_u64m4(vtmp, 30, vl), vl);
    vtmp = __riscv_vsrl_vx_u64m4(vr3, 30, vl);
    vr2  = __riscv_vadd_vv_u64m4(vr2, vtmp, vl);
	vr3  = __riscv_vsub_vv_u64m4(vr3, __riscv_vsll_vx_u64m4(vtmp, 30, vl), vl);
    vtmp = __riscv_vsrl_vx_u64m4(vr2, 30, vl);
	vr1  = __riscv_vadd_vv_u64m4(vr1, vtmp, vl);
	vr2  = __riscv_vsub_vv_u64m4(vr2, __riscv_vsll_vx_u64m4(vtmp, 30, vl), vl);
    vtmp = __riscv_vsrl_vx_u64m4(vr1, 30, vl);
	vr0  = __riscv_vadd_vv_u64m4(vr0, vtmp, vl);
	vr1  = __riscv_vsub_vv_u64m4(vr1, __riscv_vsll_vx_u64m4(vtmp, 30, vl), vl);
    
    vuint64m4_t vr0_packed = __riscv_vor_vv_u64m4(__riscv_vsll_vx_u64m4(vr0, 4, vl), __riscv_vsrl_vx_u64m4(vr1, 26, vl), vl);
    vuint64m4_t vr1_packed = __riscv_vor_vv_u64m4(__riscv_vor_vv_u64m4(__riscv_vsll_vx_u64m4(vr1, 38, vl), 
                                                                       __riscv_vsll_vx_u64m4(vr2, 8, vl), vl), 
                                                                       __riscv_vsrl_vx_u64m4(vr3, 22, vl), vl);
    vuint64m4_t vr2_packed = __riscv_vor_vv_u64m4(__riscv_vor_vv_u64m4(__riscv_vsll_vx_u64m4(vr3, 42, vl), 
                                                                       __riscv_vsll_vx_u64m4(vr4, 12, vl), vl), 
                                                                       __riscv_vsrl_vx_u64m4(vr5, 18, vl), vl);

    vuint64m4_t vfinal_shift        = __riscv_vreinterpret_v_i64m4_u64m4(__riscv_vsub_vv_i64m4(vshift,
                                                    __riscv_vmul_vx_i64m4(vshift_pi, 30, vl), vl));
    vuint64m4_t vshift_amount_right = __riscv_vrsub_vx_u64m4(vfinal_shift, 56, vl);
    vuint64m4_t vshift_amount_left  = __riscv_vadd_vx_u64m4(vfinal_shift, 8, vl);
    
    vuint64m4_t vr2_final = __riscv_vor_vv_u64m4(__riscv_vsrl_vv_u64m4(vr2_packed, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u64m4(vr1_packed, vshift_amount_left, vl), vl);
    vuint64m4_t vr1_final = __riscv_vor_vv_u64m4(__riscv_vsrl_vv_u64m4(vr1_packed, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u64m4(vr0_packed, vshift_amount_left, vl), vl);
    vuint64m4_t vr0_final = __riscv_vsrl_vv_u64m4(vr0_packed, vshift_amount_right, vl);
    
    vuint64m4_t vr1_final_hi = __riscv_vsrl_vx_u64m4(vr1_final, 32, vl);
    vuint64m4_t vr1_final_lo = __riscv_vand_vx_u64m4(vr1_final, 0xffffffff, vl);
    vuint64m4_t vr2_final_hi = __riscv_vsrl_vx_u64m4(vr2_final, 32, vl);
    vuint64m4_t vr2_final_lo = __riscv_vand_vx_u64m4(vr2_final, 0xffffffff, vl);
    
    vfloat64m4_t vres_r1, vres_r2, vmres_r1, vmres_r2;
    vres_r2 = __riscv_vfcvt_f_xu_v_f64m4(vr1_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f64m4(vr1_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f64m4(vres_r2, 0x1.0p-64, vl);
    vres_r1 = __riscv_vfmul_vf_f64m4(vres_r1, 0x1.0p-32, vl);
    
    mask_pi_4  = __riscv_vmfge_vf_f64m4_b16(vres_r1, 0.5, vl); 
    vmres_r2   = __riscv_vfmul_vf_f64m4(vres_r2, -0x1.0p0, vl);
    vmres_r1   = __riscv_vfmul_vf_f64m4(vres_r1, -0x1.0p0, vl);
    vmres_r1   = __riscv_vfadd_vf_f64m4(vmres_r1, 0x1.0p0, vl);
    vres_r1    = __riscv_vmerge_vvm_f64m4(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2    = __riscv_vmerge_vvm_f64m4(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat64m4_t vr1_htmp, vr1_tmp;
    vr1_htmp = __riscv_vfadd_vv_f64m4(vres_r1, vres_r2, vl);
    vr1_tmp  = __riscv_vfsub_vv_f64m4(vr1_htmp, vres_r1, vl);
    vr1_tmp  = __riscv_vfsub_vv_f64m4(vr1_tmp, vres_r2, vl);
    
    vres_r2  = __riscv_vfcvt_f_xu_v_f64m4(vr2_final_lo, vl);
    vres_r1  = __riscv_vfcvt_f_xu_v_f64m4(vr2_final_hi, vl);
    vres_r2  = __riscv_vfmul_vf_f64m4(vres_r2, 0x1.0p-128, vl);
    vres_r1  = __riscv_vfmul_vf_f64m4(vres_r1, 0x1.0p-96, vl);
    
    vmres_r2 = __riscv_vfmul_vf_f64m4(vres_r2, -0x1.0p0, vl);
    vmres_r1 = __riscv_vfmul_vf_f64m4(vres_r1, -0x1.0p0, vl);
    vres_r1  = __riscv_vmerge_vvm_f64m4(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2  = __riscv_vmerge_vvm_f64m4(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat64m4_t vr1_hi, vr2_hi, vr1_lo;
    vr2_hi  = __riscv_vfadd_vv_f64m4(vres_r1, vres_r2, vl);
    vr1_lo  = __riscv_vfsub_vv_f64m4(vr2_hi, vr1_tmp, vl);
    vr1_tmp = __riscv_vfadd_vv_f64m4(vr1_lo, vr1_tmp, vl);
    vr2_hi  = __riscv_vfsub_vv_f64m4(vr2_hi, vr1_tmp, vl);
    
    vr1_hi  = __riscv_vfadd_vv_f64m4(vr1_htmp, vr1_lo, vl);
    vr1_tmp = __riscv_vfsub_vv_f64m4(vr1_hi, vr1_htmp, vl);
    vr1_lo  = __riscv_vfsub_vv_f64m4(vr1_tmp, vr1_lo, vl);
    
    redx    = __riscv_vfmul_vf_f64m4(vr1_hi, PI_DIV_2_hi, vl);
    redx_lo = __riscv_vfmadd_vf_f64m4(vr1_hi, -PI_DIV_2_hi, redx, vl); 
    redx_lo = __riscv_vfmacc_vf_f64m4(redx_lo, PI_DIV_2_hi, vr1_lo, vl);
    redx_lo = __riscv_vfmsac_vf_f64m4(redx_lo, PI_DIV_2_lo, vr1_hi, vl);
    
    vbool16_t m = __riscv_vmfgt_vf_f64m4_b16(redx, 0x1.921fb54442d18p-1, vl);
    vr1_hi      = __riscv_vfrsub_vf_f64m4(redx, 0x1.921fb54442d18p0, vl);
    vr1_lo      = __riscv_vfrsub_vf_f64m4(redx_lo, 0x1.1a62633145c07p-54, vl);
    vres_r1     = __riscv_vfadd_vv_f64m4(vr1_hi, vr1_lo, vl);
    vres_r2     = __riscv_vfsub_vv_f64m4(vr1_lo, __riscv_vfsub_vv_f64m4(vres_r1, vr1_hi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f64m4(redx, vres_r1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f64m4(redx_lo, vres_r2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b16(mask_pi_4, m, vl);
    
    vuint64m4_t slow_path_quadrant = __riscv_vand_vx_u64m4(vr0_final, 0x3, vl);

    redx    = __riscv_vmerge_vvm_f64m4(redx, vx, vfast_mask, vl);
    redx_lo = __riscv_vfmerge_vfm_f64m4(redx_lo, 0.0, vfast_mask, vl);
    
    mask_pi_4 = __riscv_vmand_mm_b16(mask_pi_4, __riscv_vmnot_m_b16(vfast_mask, vl), vl);
    quadrants = __riscv_vmerge_vxm_u64m4(slow_path_quadrant, 0, vfast_mask, vl);
}

inline void __riscv_vred_codywaite_f64m4(vfloat64m4_t vx, vfloat64m4_t& redx, vfloat64m4_t& redx_lo, 
                                         vuint64m4_t& quadrants, vbool16_t& mask_pi_4, size_t avl) 
{
    constexpr double INV_PI_DIV_2 = 0x1.45f306dc9c883p-1;  
    constexpr double PI_2_HI      = 0x1.921fb54400000p+0;  
    constexpr double PI_2_MID     = 0x1.0b4611a600000p-34; 
    constexpr double PI_2_LO      = 0x1.3198a2e037073p-69;

    size_t vl = __riscv_vsetvl_e64m4(avl);

    vfloat64m4_t vfn  = __riscv_vfmul_vf_f64m4(vx, INV_PI_DIV_2, vl);
    vuint64m4_t  vifn = __riscv_vfcvt_xu_f_v_u64m4(vfn, vl);
    vfn               = __riscv_vfcvt_f_xu_v_f64m4(vifn, vl);

    vfloat64m4_t vr2;
    vfloat64m4_t vr1  = __riscv_vfnmsac_vf_f64m4(vx, PI_2_HI, vfn, vl);
    vfloat64m4_t vrhi = __riscv_vfnmsac_vf_f64m4(vr1, PI_2_MID, vfn, vl);
    vfloat64m4_t vrlo = __riscv_vfnmsac_vf_f64m4(__riscv_vfsub_vv_f64m4(vr1, vrhi, vl), PI_2_MID, vfn, vl);
    vrlo              = __riscv_vfnmsac_vf_f64m4(vrlo, PI_2_LO, vfn, vl);
    
    mask_pi_4 = __riscv_vmflt_vf_f64m4_b16(vrhi, 0.0, vl);

    vuint64m4_t vsign = __riscv_vsrl_vx_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(vrhi), 63, vl);
    quadrants         = __riscv_vand_vx_u64m4(__riscv_vsub_vv_u64m4(vifn, vsign, vl), 3, vl);

    vrlo = __riscv_vfsgnjx_vv_f64m4(vrlo, vrhi, vl);
    vrhi = __riscv_vfabs_v_f64m4(vrhi, vl);

    redx    = __riscv_vfadd_vv_f64m4(vrhi, vrlo, vl);
    redx_lo = __riscv_vfsub_vv_f64m4(vrlo, __riscv_vfsub_vv_f64m4(redx, vrhi, vl), vl);  
    
    vbool16_t m = __riscv_vmfgt_vf_f64m4_b16(redx, 0x1.921fb54442d18p-1, vl);
    vrhi        = __riscv_vfrsub_vf_f64m4(redx, 0x1.921fb54442d18p0, vl);
    vrlo        = __riscv_vfrsub_vf_f64m4(redx_lo, 0x1.1a62633145c07p-54, vl);
    vr1         = __riscv_vfadd_vv_f64m4(vrhi, vrlo, vl);
    vr2         = __riscv_vfsub_vv_f64m4(vrlo, __riscv_vfsub_vv_f64m4(vr1, vrhi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f64m4(redx, vr1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f64m4(redx_lo, vr2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b16(mask_pi_4, m, vl);
} 
 
void __riscv_vreduction_pi4_f64m4(vfloat64m4_t vx, vfloat64m4_t& redx, vfloat64m4_t& redx_lo, 
                                  vuint64m4_t& quadrants, vbool16_t& mask_pi_4, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vx        = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vand_vx_u64m4(
                            __riscv_vreinterpret_v_f64m4_u64m4(vx), 0x7fffffffffffffff, vl));
    mask_pi_4 = __riscv_vmfgt_vf_f64m4_b16(vx, 0x1p20, vl);
    if(__riscv_vcpop_m_b16(mask_pi_4, vl))
        __riscv_vred_paynehanek_f64m4(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
    else
        __riscv_vred_codywaite_f64m4(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
}


inline void __riscv_vred_paynehanek_f32m1(vfloat32m1_t vx, vfloat32m1_t& redx, vfloat32m1_t& redx_lo, 
                                          vuint32m1_t& quadrants, vbool32_t& mask_pi_4, size_t avl) {
    constexpr float PI_DIV_2    = 0x1.921fb6p0f;
    constexpr float PI_DIV_2_lo = -0x1.777a5cp-25f;
    constexpr float PI_DIV_4    = 0x1.921fb6p-1f;
    size_t vl = __riscv_vsetvl_e32m1(avl);
    
    vbool32_t vfast_mask = __riscv_vmflt_vf_f32m1_b32(vx, PI_DIV_4, vl);

    vuint32m1_t vix  = __riscv_vreinterpret_v_f32m1_u32m1(
                            __riscv_vfmerge_vfm_f32m1(vx, 1.75f, vfast_mask, vl));
    vuint32m1_t vexp = __riscv_vsrl_vx_u32m1(vix, 23, vl);
    vexp             = __riscv_vand_vx_u32m1(vexp, 0xff, vl);
    
    vint32m1_t vshift    = __riscv_vsub_vx_i32m1(__riscv_vreinterpret_v_u32m1_i32m1(vexp), 127, vl);
    vint32m1_t vshift_pi = __riscv_vdiv_vx_i32m1(__riscv_vsub_vx_i32m1(vshift, 12, vl), 15, vl);
    
    vuint32m1_t vushift_pi_u32 = __riscv_vreinterpret_v_i32m1_u32m1(vshift_pi);
    vuint16mf2_t vushift_pi    = __riscv_vncvt_x_x_w_u16mf2(__riscv_vsll_vx_u32m1(vushift_pi_u32, 1, vl), vl);
    
    vuint16mf2_t vmant1 = __riscv_vncvt_x_x_w_u16mf2(vix, vl);
    vuint16mf2_t vmant0 = __riscv_vnsrl_wx_u16mf2(vix, 15, vl);
    vmant1 = __riscv_vand_vx_u16mf2(vmant1, 0x7fff, vl);
    vmant0 = __riscv_vand_vx_u16mf2(vmant0, 0xff, vl);
    vmant0 = __riscv_vor_vx_u16mf2(vmant0, 0x100, vl);
    
    vuint16mf2_t vtwoOnPi0 = __riscv_vloxei16_v_u16mf2(TwoOnPI_f32, vushift_pi, vl);
    vuint16mf2_t vtwoOnPi1 = __riscv_vloxei16_v_u16mf2(TwoOnPI_f32, __riscv_vadd_vx_u16mf2(vushift_pi, 2, vl), vl);
    vuint16mf2_t vtwoOnPi2 = __riscv_vloxei16_v_u16mf2(TwoOnPI_f32, __riscv_vadd_vx_u16mf2(vushift_pi, 4, vl), vl);
    vuint16mf2_t vtwoOnPi3 = __riscv_vloxei16_v_u16mf2(TwoOnPI_f32, __riscv_vadd_vx_u16mf2(vushift_pi, 6, vl), vl);
    vuint16mf2_t vtwoOnPi4 = __riscv_vloxei16_v_u16mf2(TwoOnPI_f32, __riscv_vadd_vx_u16mf2(vushift_pi, 8, vl), vl);
    vuint16mf2_t vtwoOnPi5 = __riscv_vloxei16_v_u16mf2(TwoOnPI_f32, __riscv_vadd_vx_u16mf2(vushift_pi, 10, vl), vl);
    vuint16mf2_t vtwoOnPi6 = __riscv_vloxei16_v_u16mf2(TwoOnPI_f32, __riscv_vadd_vx_u16mf2(vushift_pi, 12, vl), vl);
    vuint16mf2_t vtwoOnPi7 = __riscv_vloxei16_v_u16mf2(TwoOnPI_f32, __riscv_vadd_vx_u16mf2(vushift_pi, 14, vl), vl);

    vuint32m1_t vr0 = __riscv_vwmaccu_vv_u32m1(__riscv_vwmulu_vv_u32m1(vtwoOnPi1, vmant0, vl), vtwoOnPi0, vmant1, vl);
    vuint32m1_t vr1 = __riscv_vwmaccu_vv_u32m1(__riscv_vwmulu_vv_u32m1(vtwoOnPi2, vmant0, vl), vtwoOnPi1, vmant1, vl);
    vuint32m1_t vr2 = __riscv_vwmaccu_vv_u32m1(__riscv_vwmulu_vv_u32m1(vtwoOnPi3, vmant0, vl), vtwoOnPi2, vmant1, vl);
    vuint32m1_t vr3 = __riscv_vwmaccu_vv_u32m1(__riscv_vwmulu_vv_u32m1(vtwoOnPi4, vmant0, vl), vtwoOnPi3, vmant1, vl);
    vuint32m1_t vr4 = __riscv_vwmaccu_vv_u32m1(__riscv_vwmulu_vv_u32m1(vtwoOnPi5, vmant0, vl), vtwoOnPi4, vmant1, vl);
    vuint32m1_t vr5 = __riscv_vwmaccu_vv_u32m1(__riscv_vwmulu_vv_u32m1(vtwoOnPi6, vmant0, vl), vtwoOnPi5, vmant1, vl);
    vuint32m1_t vr6 = __riscv_vwmaccu_vv_u32m1(__riscv_vwmulu_vv_u32m1(vtwoOnPi7, vmant0, vl), vtwoOnPi6, vmant1, vl);
    
    vuint32m1_t tmp;
    vr5 = __riscv_vadd_vv_u32m1(vr5, __riscv_vsrl_vx_u32m1(vr6, 15, vl), vl);
    tmp = __riscv_vsrl_vx_u32m1(vr5, 15, vl);
    vr4 = __riscv_vadd_vv_u32m1(vr4, tmp, vl);
    vr5 = __riscv_vsub_vv_u32m1(vr5, __riscv_vsll_vx_u32m1(tmp, 15, vl),vl);
    tmp = __riscv_vsrl_vx_u32m1(vr4, 15, vl);
    vr3 = __riscv_vadd_vv_u32m1(vr3, tmp, vl);
    vr4 = __riscv_vsub_vv_u32m1(vr4, __riscv_vsll_vx_u32m1(tmp, 15, vl),vl);
    tmp = __riscv_vsrl_vx_u32m1(vr3, 15, vl);
    vr2 = __riscv_vadd_vv_u32m1(vr2, tmp, vl);
    vr3 = __riscv_vsub_vv_u32m1(vr3, __riscv_vsll_vx_u32m1(tmp, 15, vl),vl);
    tmp = __riscv_vsrl_vx_u32m1(vr2, 15, vl);
    vr1 = __riscv_vadd_vv_u32m1(vr1, tmp, vl);
    vr2 = __riscv_vsub_vv_u32m1(vr2, __riscv_vsll_vx_u32m1(tmp, 15, vl),vl);
    tmp = __riscv_vsrl_vx_u32m1(vr1, 15, vl);
    vr0 = __riscv_vadd_vv_u32m1(vr0, tmp, vl);
    vr1 = __riscv_vsub_vv_u32m1(vr1, __riscv_vsll_vx_u32m1(tmp, 15, vl),vl);
    
    vr0 = __riscv_vor_vv_u32m1(__riscv_vsll_vx_u32m1(vr0, 4, vl), __riscv_vsrl_vx_u32m1(vr1, 11, vl), vl);
    vr1 = __riscv_vor_vv_u32m1(__riscv_vor_vv_u32m1(__riscv_vsll_vx_u32m1(vr1, 21, vl), 
                                                    __riscv_vsll_vx_u32m1(vr2, 6, vl), vl), 
                                                    __riscv_vsrl_vx_u32m1(vr3, 9, vl), vl);
    vr2 = __riscv_vor_vv_u32m1(__riscv_vor_vv_u32m1(__riscv_vsll_vx_u32m1(vr3, 23, vl), 
                                                    __riscv_vsll_vx_u32m1(vr4, 8, vl), vl), 
                                                    __riscv_vsrl_vx_u32m1(vr5, 7, vl), vl);
    
    vuint32m1_t vfinal_shift        = __riscv_vreinterpret_v_i32m1_u32m1(__riscv_vsub_vv_i32m1(
                                               vshift, __riscv_vmul_vx_i32m1(vshift_pi, 15, vl), vl));
    vuint32m1_t vshift_amount_right = __riscv_vrsub_vx_u32m1(vfinal_shift, 27, vl);
    vuint32m1_t vshift_amount_left  = __riscv_vadd_vx_u32m1(vfinal_shift, 5, vl);
    
    vuint32m1_t vr2_final = __riscv_vor_vv_u32m1(__riscv_vsrl_vv_u32m1(vr2, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u32m1(vr1, vshift_amount_left, vl), vl);
    vuint32m1_t vr1_final = __riscv_vor_vv_u32m1(__riscv_vsrl_vv_u32m1(vr1, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u32m1(vr0, vshift_amount_left, vl), vl);
    vuint32m1_t vr0_final = __riscv_vsrl_vv_u32m1(vr0, vshift_amount_right, vl);
    
    vuint32m1_t vr1_final_hi = __riscv_vsrl_vx_u32m1(vr1_final, 16, vl);
    vuint32m1_t vr1_final_lo = __riscv_vand_vx_u32m1(vr1_final, 0xffff, vl);
    vuint32m1_t vr2_final_hi = __riscv_vsrl_vx_u32m1(vr2_final, 16, vl);
    vuint32m1_t vr2_final_lo = __riscv_vand_vx_u32m1(vr2_final, 0xffff, vl);
    
    vfloat32m1_t vres_r1, vres_r2, vmres_r1, vmres_r2;
    vres_r2 = __riscv_vfcvt_f_xu_v_f32m1(vr1_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f32m1(vr1_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f32m1(vres_r2, 0x1.0p-32f, vl);
    vres_r1 = __riscv_vfmul_vf_f32m1(vres_r1, 0x1.0p-16f, vl);
    
    mask_pi_4 = __riscv_vmfge_vf_f32m1_b32(vres_r1, 0.5f, vl); 
    vmres_r2  = __riscv_vfmul_vf_f32m1(vres_r2, -0x1.0p0f, vl);
    vmres_r1  = __riscv_vfmul_vf_f32m1(vres_r1, -0x1.0p0f, vl);
    vmres_r1  = __riscv_vfadd_vf_f32m1(vmres_r1, 0x1.0p0f, vl);
    vres_r1   = __riscv_vmerge_vvm_f32m1(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2   = __riscv_vmerge_vvm_f32m1(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat32m1_t vr1_htmp, vr1_tmp;
    vr1_htmp = __riscv_vfadd_vv_f32m1(vres_r1, vres_r2, vl);
    vr1_tmp  = __riscv_vfsub_vv_f32m1(vr1_htmp, vres_r1, vl);
    vr1_tmp  = __riscv_vfsub_vv_f32m1(vr1_tmp, vres_r2, vl);
    
    vres_r2 = __riscv_vfcvt_f_xu_v_f32m1(vr2_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f32m1(vr2_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f32m1(vres_r2, 0x1.0p-64f, vl);
    vres_r1 = __riscv_vfmul_vf_f32m1(vres_r1, 0x1.0p-48f, vl);
    
    vmres_r2 = __riscv_vfmul_vf_f32m1(vres_r2, -0x1.0p0f, vl);
    vmres_r1 = __riscv_vfmul_vf_f32m1(vres_r1, -0x1.0p0f, vl);
    vres_r1  = __riscv_vmerge_vvm_f32m1(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2  = __riscv_vmerge_vvm_f32m1(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat32m1_t vr1_hi, vr2_hi, vr1_lo;
    vr2_hi  = __riscv_vfadd_vv_f32m1(vres_r1, vres_r2, vl);
    vr1_lo  = __riscv_vfsub_vv_f32m1(vr2_hi, vr1_tmp, vl);
    vr1_tmp = __riscv_vfadd_vv_f32m1(vr1_lo, vr1_tmp, vl);
    vr2_hi  = __riscv_vfsub_vv_f32m1(vr2_hi, vr1_tmp, vl);
    
    vr1_hi  = __riscv_vfadd_vv_f32m1(vr1_htmp, vr1_lo, vl);
    vr1_tmp = __riscv_vfsub_vv_f32m1(vr1_hi, vr1_htmp, vl);
    vr1_lo  = __riscv_vfsub_vv_f32m1(vr1_tmp, vr1_lo, vl);
    
    redx    = __riscv_vfmul_vf_f32m1(vr1_hi, PI_DIV_2, vl);
    redx_lo = __riscv_vfmadd_vf_f32m1(vr1_hi, -PI_DIV_2, redx, vl);    
    redx_lo = __riscv_vfmacc_vf_f32m1(redx_lo, PI_DIV_2, vr1_lo, vl);
    redx_lo = __riscv_vfmsac_vf_f32m1(redx_lo, PI_DIV_2_lo, vr1_hi, vl);
    
    vbool32_t m = __riscv_vmfgt_vf_f32m1_b32(redx, 0x1.921fb6p-1f, vl);
    vr1_hi      = __riscv_vfrsub_vf_f32m1(redx, 0x1.921fb6p0f, vl);
    vr1_lo      = __riscv_vfrsub_vf_f32m1(redx_lo, -0x1.777a5cp-25f, vl);
    vres_r1     = __riscv_vfadd_vv_f32m1(vr1_hi, vr1_lo, vl);
    vres_r2     = __riscv_vfsub_vv_f32m1(vr1_lo, __riscv_vfsub_vv_f32m1(vres_r1, vr1_hi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f32m1(redx, vres_r1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f32m1(redx_lo, vres_r2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b32(mask_pi_4, m, vl);
    
    vuint32m1_t slow_path_quadrant = __riscv_vand_vx_u32m1(vr0_final, 0x3, vl);

    redx = __riscv_vmerge_vvm_f32m1(redx, vx, vfast_mask, vl);
    redx_lo = __riscv_vfmerge_vfm_f32m1(redx_lo, 0.0f, vfast_mask, vl);
    
    mask_pi_4 = __riscv_vmand_mm_b32(mask_pi_4, __riscv_vmnot_m_b32(vfast_mask, vl), vl); 
    quadrants = __riscv_vmerge_vxm_u32m1(slow_path_quadrant, 0, vfast_mask, vl);
}

inline void __riscv_vred_codywaite_f32m1(vfloat32m1_t vx, vfloat32m1_t& redx, vfloat32m1_t& redx_lo, 
                                         vuint32m1_t& quadrants, vbool32_t& mask_pi_4, size_t avl) 
{
    constexpr float INV_PI_DIV_2 = 0x1.45f308p-1f; 
    constexpr float PI_2_HI      = 0x1.920000p+0f; 
    constexpr float PI_2_MID     = 0x1.fb4000p-12f;
    constexpr float PI_2_LO      = 0x1.444000p-24f;
    constexpr float PI_2_TAIL    = 0x1.68c235p-39f;

    size_t vl = __riscv_vsetvl_e32m1(avl);

    vfloat32m1_t vfn  = __riscv_vfmul_vf_f32m1(vx, INV_PI_DIV_2, vl);
    vuint32m1_t  vifn = __riscv_vfcvt_xu_f_v_u32m1(vfn, vl);
    vfn = __riscv_vfcvt_f_xu_v_f32m1(vifn, vl);

    vfloat32m1_t vr1  = __riscv_vfnmsac_vf_f32m1(vx, PI_2_HI, vfn, vl);
    vfloat32m1_t vr2  = __riscv_vfnmsac_vf_f32m1(vr1, PI_2_MID, vfn, vl);
    vfloat32m1_t err2 = __riscv_vfnmsac_vf_f32m1(__riscv_vfsub_vv_f32m1(vr1, vr2, vl), PI_2_MID, vfn, vl);
    vfloat32m1_t vrhi = __riscv_vfnmsac_vf_f32m1(vr2, PI_2_LO, vfn, vl);
    vfloat32m1_t err3 = __riscv_vfnmsac_vf_f32m1(__riscv_vfsub_vv_f32m1(vr2, vrhi, vl), PI_2_LO, vfn, vl);
    vfloat32m1_t vrlo = __riscv_vfadd_vv_f32m1(err2, err3, vl);
    vrlo              = __riscv_vfnmsac_vf_f32m1(vrlo, PI_2_TAIL, vfn, vl);

    mask_pi_4 = __riscv_vmflt_vf_f32m1_b32(vrhi, 0.0f, vl);

    vuint32m1_t vsign = __riscv_vsrl_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(vrhi), 31, vl);
    quadrants         = __riscv_vand_vx_u32m1(__riscv_vsub_vv_u32m1(vifn, vsign, vl), 3, vl);

    vrlo = __riscv_vfsgnjx_vv_f32m1(vrlo, vrhi, vl);
    vrhi = __riscv_vfabs_v_f32m1(vrhi, vl);

    redx    = __riscv_vfadd_vv_f32m1(vrhi, vrlo, vl);
    redx_lo = __riscv_vfsub_vv_f32m1(vrlo, __riscv_vfsub_vv_f32m1(redx, vrhi, vl), vl);

    vbool32_t m = __riscv_vmfgt_vf_f32m1_b32(redx, 0x1.921fb6p-1f, vl);
    vrhi        = __riscv_vfrsub_vf_f32m1(redx, 0x1.921fb6p0f, vl);
    vrlo        = __riscv_vfrsub_vf_f32m1(redx_lo, -0x1.777a5cp-25f, vl);
    vr1         = __riscv_vfadd_vv_f32m1(vrhi, vrlo, vl);
    vr2         = __riscv_vfsub_vv_f32m1(vrlo, __riscv_vfsub_vv_f32m1(vr1, vrhi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f32m1(redx, vr1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f32m1(redx_lo, vr2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b32(mask_pi_4, m, vl);
}

void __riscv_vreduction_pi4_f32m1(vfloat32m1_t vx, vfloat32m1_t& redx, vfloat32m1_t& redx_lo, 
                                  vuint32m1_t& quadrants, vbool32_t& mask_pi_4, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m1(avl);
    
    vx = __riscv_vreinterpret_v_u32m1_f32m1(
                __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(vx), 0x7fffffff, vl));
    
    mask_pi_4 = __riscv_vmfgt_vf_f32m1_b32(vx, 0x1p12f, vl);
    if (__riscv_vcpop_m_b32( mask_pi_4, vl))
        __riscv_vred_paynehanek_f32m1(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
    else
        __riscv_vred_codywaite_f32m1(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
}


inline void __riscv_vred_paynehanek_f32m2(vfloat32m2_t vx, vfloat32m2_t& redx, vfloat32m2_t& redx_lo, 
                                          vuint32m2_t& quadrants, vbool16_t& mask_pi_4, size_t avl) {
    constexpr float PI_DIV_2    = 0x1.921fb6p0f;
    constexpr float PI_DIV_2_lo = -0x1.777a5cp-25f;
    constexpr float PI_DIV_4    = 0x1.921fb6p-1f;
    size_t vl = __riscv_vsetvl_e32m2(avl);
    
    vbool16_t vfast_mask = __riscv_vmflt_vf_f32m2_b16(vx, PI_DIV_4, vl);

    vuint32m2_t vix  = __riscv_vreinterpret_v_f32m2_u32m2(
                            __riscv_vfmerge_vfm_f32m2(vx, 1.75f, vfast_mask, vl));
    vuint32m2_t vexp = __riscv_vsrl_vx_u32m2(vix, 23, vl);
    vexp             = __riscv_vand_vx_u32m2(vexp, 0xff, vl);
    
    vint32m2_t vshift    = __riscv_vsub_vx_i32m2(__riscv_vreinterpret_v_u32m2_i32m2(vexp), 127, vl);
    vint32m2_t vshift_pi = __riscv_vdiv_vx_i32m2(__riscv_vsub_vx_i32m2(vshift, 12, vl), 15, vl);
    
    vuint32m2_t vushift_pi_u32 = __riscv_vreinterpret_v_i32m2_u32m2(vshift_pi);
    vuint16m1_t vushift_pi     = __riscv_vncvt_x_x_w_u16m1(__riscv_vsll_vx_u32m2(vushift_pi_u32, 1, vl), vl);
    
    vuint16m1_t vmant1 = __riscv_vncvt_x_x_w_u16m1(vix, vl);
    vuint16m1_t vmant0 = __riscv_vnsrl_wx_u16m1(vix, 15, vl);
    vmant1 = __riscv_vand_vx_u16m1(vmant1, 0x7fff, vl);
    vmant0 = __riscv_vand_vx_u16m1(vmant0, 0xff, vl);
    vmant0 = __riscv_vor_vx_u16m1(vmant0, 0x100, vl);
    
    vuint16m1_t vtwoOnPi0 = __riscv_vloxei16_v_u16m1(TwoOnPI_f32, vushift_pi, vl);
    vuint16m1_t vtwoOnPi1 = __riscv_vloxei16_v_u16m1(TwoOnPI_f32, __riscv_vadd_vx_u16m1(vushift_pi, 2, vl), vl);
    vuint16m1_t vtwoOnPi2 = __riscv_vloxei16_v_u16m1(TwoOnPI_f32, __riscv_vadd_vx_u16m1(vushift_pi, 4, vl), vl);
    vuint16m1_t vtwoOnPi3 = __riscv_vloxei16_v_u16m1(TwoOnPI_f32, __riscv_vadd_vx_u16m1(vushift_pi, 6, vl), vl);
    vuint16m1_t vtwoOnPi4 = __riscv_vloxei16_v_u16m1(TwoOnPI_f32, __riscv_vadd_vx_u16m1(vushift_pi, 8, vl), vl);
    vuint16m1_t vtwoOnPi5 = __riscv_vloxei16_v_u16m1(TwoOnPI_f32, __riscv_vadd_vx_u16m1(vushift_pi, 10, vl), vl);
    vuint16m1_t vtwoOnPi6 = __riscv_vloxei16_v_u16m1(TwoOnPI_f32, __riscv_vadd_vx_u16m1(vushift_pi, 12, vl), vl);
    vuint16m1_t vtwoOnPi7 = __riscv_vloxei16_v_u16m1(TwoOnPI_f32, __riscv_vadd_vx_u16m1(vushift_pi, 14, vl), vl);

    vuint32m2_t vr0 = __riscv_vwmaccu_vv_u32m2(__riscv_vwmulu_vv_u32m2(vtwoOnPi1, vmant0, vl), vtwoOnPi0, vmant1, vl);
    vuint32m2_t vr1 = __riscv_vwmaccu_vv_u32m2(__riscv_vwmulu_vv_u32m2(vtwoOnPi2, vmant0, vl), vtwoOnPi1, vmant1, vl);
    vuint32m2_t vr2 = __riscv_vwmaccu_vv_u32m2(__riscv_vwmulu_vv_u32m2(vtwoOnPi3, vmant0, vl), vtwoOnPi2, vmant1, vl);
    vuint32m2_t vr3 = __riscv_vwmaccu_vv_u32m2(__riscv_vwmulu_vv_u32m2(vtwoOnPi4, vmant0, vl), vtwoOnPi3, vmant1, vl);
    vuint32m2_t vr4 = __riscv_vwmaccu_vv_u32m2(__riscv_vwmulu_vv_u32m2(vtwoOnPi5, vmant0, vl), vtwoOnPi4, vmant1, vl);
    vuint32m2_t vr5 = __riscv_vwmaccu_vv_u32m2(__riscv_vwmulu_vv_u32m2(vtwoOnPi6, vmant0, vl), vtwoOnPi5, vmant1, vl);
    vuint32m2_t vr6 = __riscv_vwmaccu_vv_u32m2(__riscv_vwmulu_vv_u32m2(vtwoOnPi7, vmant0, vl), vtwoOnPi6, vmant1, vl);
    
    vuint32m2_t tmp;
    vr5 = __riscv_vadd_vv_u32m2(vr5, __riscv_vsrl_vx_u32m2(vr6, 15, vl), vl);
    tmp = __riscv_vsrl_vx_u32m2(vr5, 15, vl);
    vr4 = __riscv_vadd_vv_u32m2(vr4, tmp, vl);
    vr5 = __riscv_vsub_vv_u32m2(vr5, __riscv_vsll_vx_u32m2(tmp, 15, vl),vl);
    tmp = __riscv_vsrl_vx_u32m2(vr4, 15, vl);
    vr3 = __riscv_vadd_vv_u32m2(vr3, tmp, vl);
    vr4 = __riscv_vsub_vv_u32m2(vr4, __riscv_vsll_vx_u32m2(tmp, 15, vl),vl);
    tmp = __riscv_vsrl_vx_u32m2(vr3, 15, vl);
    vr2 = __riscv_vadd_vv_u32m2(vr2, tmp, vl);
    vr3 = __riscv_vsub_vv_u32m2(vr3, __riscv_vsll_vx_u32m2(tmp, 15, vl),vl);
    tmp = __riscv_vsrl_vx_u32m2(vr2, 15, vl);
    vr1 = __riscv_vadd_vv_u32m2(vr1, tmp, vl);
    vr2 = __riscv_vsub_vv_u32m2(vr2, __riscv_vsll_vx_u32m2(tmp, 15, vl),vl);
    tmp = __riscv_vsrl_vx_u32m2(vr1, 15, vl);
    vr0 = __riscv_vadd_vv_u32m2(vr0, tmp, vl);
    vr1 = __riscv_vsub_vv_u32m2(vr1, __riscv_vsll_vx_u32m2(tmp, 15, vl),vl);
    
    vr0 = __riscv_vor_vv_u32m2(__riscv_vsll_vx_u32m2(vr0, 4, vl), __riscv_vsrl_vx_u32m2(vr1, 11, vl), vl);
    vr1 = __riscv_vor_vv_u32m2(__riscv_vor_vv_u32m2(__riscv_vsll_vx_u32m2(vr1, 21, vl), 
                                                    __riscv_vsll_vx_u32m2(vr2, 6, vl), vl), 
                                                    __riscv_vsrl_vx_u32m2(vr3, 9, vl), vl);
    vr2 = __riscv_vor_vv_u32m2(__riscv_vor_vv_u32m2(__riscv_vsll_vx_u32m2(vr3, 23, vl), 
                                                    __riscv_vsll_vx_u32m2(vr4, 8, vl), vl), 
                                                    __riscv_vsrl_vx_u32m2(vr5, 7, vl), vl);
    
    vuint32m2_t vfinal_shift        = __riscv_vreinterpret_v_i32m2_u32m2(__riscv_vsub_vv_i32m2(
                                               vshift, __riscv_vmul_vx_i32m2(vshift_pi, 15, vl), vl));
    vuint32m2_t vshift_amount_right = __riscv_vrsub_vx_u32m2(vfinal_shift, 27, vl);
    vuint32m2_t vshift_amount_left  = __riscv_vadd_vx_u32m2(vfinal_shift, 5, vl);
    
    vuint32m2_t vr2_final = __riscv_vor_vv_u32m2(__riscv_vsrl_vv_u32m2(vr2, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u32m2(vr1, vshift_amount_left, vl), vl);
    vuint32m2_t vr1_final = __riscv_vor_vv_u32m2(__riscv_vsrl_vv_u32m2(vr1, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u32m2(vr0, vshift_amount_left, vl), vl);
    vuint32m2_t vr0_final = __riscv_vsrl_vv_u32m2(vr0, vshift_amount_right, vl);
    
    vuint32m2_t vr1_final_hi = __riscv_vsrl_vx_u32m2(vr1_final, 16, vl);
    vuint32m2_t vr1_final_lo = __riscv_vand_vx_u32m2(vr1_final, 0xffff, vl);
    vuint32m2_t vr2_final_hi = __riscv_vsrl_vx_u32m2(vr2_final, 16, vl);
    vuint32m2_t vr2_final_lo = __riscv_vand_vx_u32m2(vr2_final, 0xffff, vl);
    
    vfloat32m2_t vres_r1, vres_r2, vmres_r1, vmres_r2;
    vres_r2 = __riscv_vfcvt_f_xu_v_f32m2(vr1_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f32m2(vr1_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f32m2(vres_r2, 0x1.0p-32f, vl);
    vres_r1 = __riscv_vfmul_vf_f32m2(vres_r1, 0x1.0p-16f, vl);
    
    mask_pi_4 = __riscv_vmfge_vf_f32m2_b16(vres_r1, 0.5f, vl); 
    vmres_r2  = __riscv_vfmul_vf_f32m2(vres_r2, -0x1.0p0f, vl);
    vmres_r1  = __riscv_vfmul_vf_f32m2(vres_r1, -0x1.0p0f, vl);
    vmres_r1  = __riscv_vfadd_vf_f32m2(vmres_r1, 0x1.0p0f, vl);
    vres_r1   = __riscv_vmerge_vvm_f32m2(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2   = __riscv_vmerge_vvm_f32m2(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat32m2_t vr1_htmp, vr1_tmp;
    vr1_htmp = __riscv_vfadd_vv_f32m2(vres_r1, vres_r2, vl);
    vr1_tmp  = __riscv_vfsub_vv_f32m2(vr1_htmp, vres_r1, vl);
    vr1_tmp  = __riscv_vfsub_vv_f32m2(vr1_tmp, vres_r2, vl);
    
    vres_r2 = __riscv_vfcvt_f_xu_v_f32m2(vr2_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f32m2(vr2_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f32m2(vres_r2, 0x1.0p-64f, vl);
    vres_r1 = __riscv_vfmul_vf_f32m2(vres_r1, 0x1.0p-48f, vl);
    
    vmres_r2 = __riscv_vfmul_vf_f32m2(vres_r2, -0x1.0p0f, vl);
    vmres_r1 = __riscv_vfmul_vf_f32m2(vres_r1, -0x1.0p0f, vl);
    vres_r1  = __riscv_vmerge_vvm_f32m2(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2  = __riscv_vmerge_vvm_f32m2(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat32m2_t vr1_hi, vr2_hi, vr1_lo;
    vr2_hi  = __riscv_vfadd_vv_f32m2(vres_r1, vres_r2, vl);
    vr1_lo  = __riscv_vfsub_vv_f32m2(vr2_hi, vr1_tmp, vl);
    vr1_tmp = __riscv_vfadd_vv_f32m2(vr1_lo, vr1_tmp, vl);
    vr2_hi  = __riscv_vfsub_vv_f32m2(vr2_hi, vr1_tmp, vl);
    
    vr1_hi  = __riscv_vfadd_vv_f32m2(vr1_htmp, vr1_lo, vl);
    vr1_tmp = __riscv_vfsub_vv_f32m2(vr1_hi, vr1_htmp, vl);
    vr1_lo  = __riscv_vfsub_vv_f32m2(vr1_tmp, vr1_lo, vl);
    
    redx    = __riscv_vfmul_vf_f32m2(vr1_hi, PI_DIV_2, vl);
    redx_lo = __riscv_vfmadd_vf_f32m2(vr1_hi, -PI_DIV_2, redx, vl);    
    redx_lo = __riscv_vfmacc_vf_f32m2(redx_lo, PI_DIV_2, vr1_lo, vl);
    redx_lo = __riscv_vfmsac_vf_f32m2(redx_lo, PI_DIV_2_lo, vr1_hi, vl); 
    
    vbool16_t m = __riscv_vmfgt_vf_f32m2_b16(redx, 0x1.921fb6p-1f, vl);
    vr1_hi      = __riscv_vfrsub_vf_f32m2(redx, 0x1.921fb6p0f, vl);
    vr1_lo      = __riscv_vfrsub_vf_f32m2(redx_lo, -0x1.777a5cp-25f, vl);
    vres_r1     = __riscv_vfadd_vv_f32m2(vr1_hi, vr1_lo, vl);
    vres_r2     = __riscv_vfsub_vv_f32m2(vr1_lo, __riscv_vfsub_vv_f32m2(vres_r1, vr1_hi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f32m2(redx, vres_r1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f32m2(redx_lo, vres_r2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b16(mask_pi_4, m, vl);
    
    vuint32m2_t slow_path_quadrant = __riscv_vand_vx_u32m2(vr0_final, 0x3, vl);

    redx = __riscv_vmerge_vvm_f32m2(redx, vx, vfast_mask, vl);
    redx_lo = __riscv_vfmerge_vfm_f32m2(redx_lo, 0.0f, vfast_mask, vl);
    
    mask_pi_4 = __riscv_vmand_mm_b16(mask_pi_4, __riscv_vmnot_m_b16(vfast_mask, vl), vl); 
    quadrants = __riscv_vmerge_vxm_u32m2(slow_path_quadrant, 0, vfast_mask, vl);
}

inline void __riscv_vred_codywaite_f32m2(vfloat32m2_t vx, vfloat32m2_t& redx, vfloat32m2_t& redx_lo, 
                                         vuint32m2_t& quadrants, vbool16_t& mask_pi_4, size_t avl) 
{
    constexpr float INV_PI_DIV_2 = 0x1.45f308p-1f; 
    constexpr float PI_2_HI      = 0x1.920000p+0f; 
    constexpr float PI_2_MID     = 0x1.fb4000p-12f;
    constexpr float PI_2_LO      = 0x1.444000p-24f;
    constexpr float PI_2_TAIL    = 0x1.68c235p-39f;

    size_t vl = __riscv_vsetvl_e32m2(avl);

    vfloat32m2_t vfn  = __riscv_vfmul_vf_f32m2(vx, INV_PI_DIV_2, vl);
    vuint32m2_t  vifn = __riscv_vfcvt_xu_f_v_u32m2(vfn, vl);
    vfn = __riscv_vfcvt_f_xu_v_f32m2(vifn, vl);

    vfloat32m2_t vr1  = __riscv_vfnmsac_vf_f32m2(vx, PI_2_HI, vfn, vl);
    vfloat32m2_t vr2  = __riscv_vfnmsac_vf_f32m2(vr1, PI_2_MID, vfn, vl);
    vfloat32m2_t err2 = __riscv_vfnmsac_vf_f32m2(__riscv_vfsub_vv_f32m2(vr1, vr2, vl), PI_2_MID, vfn, vl);
    vfloat32m2_t vrhi = __riscv_vfnmsac_vf_f32m2(vr2, PI_2_LO, vfn, vl);
    vfloat32m2_t err3 = __riscv_vfnmsac_vf_f32m2(__riscv_vfsub_vv_f32m2(vr2, vrhi, vl), PI_2_LO, vfn, vl);
    vfloat32m2_t vrlo = __riscv_vfadd_vv_f32m2(err2, err3, vl);
    vrlo              = __riscv_vfnmsac_vf_f32m2(vrlo, PI_2_TAIL, vfn, vl);

    mask_pi_4 = __riscv_vmflt_vf_f32m2_b16(vrhi, 0.0f, vl);

    vuint32m2_t vsign = __riscv_vsrl_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(vrhi), 31, vl);
    quadrants         = __riscv_vand_vx_u32m2(__riscv_vsub_vv_u32m2(vifn, vsign, vl), 3, vl);

    vrlo = __riscv_vfsgnjx_vv_f32m2(vrlo, vrhi, vl);
    vrhi = __riscv_vfabs_v_f32m2(vrhi, vl);

    redx    = __riscv_vfadd_vv_f32m2(vrhi, vrlo, vl);
    redx_lo = __riscv_vfsub_vv_f32m2(vrlo, __riscv_vfsub_vv_f32m2(redx, vrhi, vl), vl);

    vbool16_t m = __riscv_vmfgt_vf_f32m2_b16(redx, 0x1.921fb6p-1f, vl);
    vrhi        = __riscv_vfrsub_vf_f32m2(redx, 0x1.921fb6p0f, vl);
    vrlo        = __riscv_vfrsub_vf_f32m2(redx_lo, -0x1.777a5cp-25f, vl);
    vr1         = __riscv_vfadd_vv_f32m2(vrhi, vrlo, vl);
    vr2         = __riscv_vfsub_vv_f32m2(vrlo, __riscv_vfsub_vv_f32m2(vr1, vrhi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f32m2(redx, vr1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f32m2(redx_lo, vr2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b16(mask_pi_4, m, vl);
}

void __riscv_vreduction_pi4_f32m2(vfloat32m2_t vx, vfloat32m2_t& redx, vfloat32m2_t& redx_lo, 
                                  vuint32m2_t& quadrants, vbool16_t& mask_pi_4, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m2(avl);
    
    vx = __riscv_vreinterpret_v_u32m2_f32m2(
                __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(vx), 0x7fffffff, vl));
    
    mask_pi_4 = __riscv_vmfgt_vf_f32m2_b16(vx, 0x1p12f, vl);
    if (__riscv_vcpop_m_b16( mask_pi_4, vl))
        __riscv_vred_paynehanek_f32m2(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
    else
        __riscv_vred_codywaite_f32m2(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
}


inline void __riscv_vred_paynehanek_f32m4(vfloat32m4_t vx, vfloat32m4_t& redx, vfloat32m4_t& redx_lo, 
                                          vuint32m4_t& quadrants, vbool8_t& mask_pi_4, size_t avl) {
    constexpr float PI_DIV_2    =  0x1.921fb6p0f;
    constexpr float PI_DIV_2_lo = -0x1.777a5cp-25f;
    constexpr float PI_DIV_4    =  0x1.921fb6p-1f;
    size_t vl = __riscv_vsetvl_e32m4(avl);
    
    vbool8_t vfast_mask = __riscv_vmflt_vf_f32m4_b8(vx, PI_DIV_4, vl);

    vuint32m4_t vix  = __riscv_vreinterpret_v_f32m4_u32m4(
                            __riscv_vfmerge_vfm_f32m4(vx, 1.75f, vfast_mask, vl));
    vuint32m4_t vexp = __riscv_vsrl_vx_u32m4(vix, 23, vl);
    vexp             = __riscv_vand_vx_u32m4(vexp, 0xff, vl);
    
    vint32m4_t vshift    = __riscv_vsub_vx_i32m4(__riscv_vreinterpret_v_u32m4_i32m4(vexp), 127, vl);
    vint32m4_t vshift_pi = __riscv_vdiv_vx_i32m4(__riscv_vsub_vx_i32m4(vshift, 12, vl), 15, vl);
    
    vuint32m4_t vushift_pi_u32 = __riscv_vreinterpret_v_i32m4_u32m4(vshift_pi);
    vuint16m2_t vushift_pi     = __riscv_vncvt_x_x_w_u16m2(__riscv_vsll_vx_u32m4(vushift_pi_u32, 1, vl), vl);
    
    vuint16m2_t vmant1 = __riscv_vncvt_x_x_w_u16m2(vix, vl);
    vuint16m2_t vmant0 = __riscv_vnsrl_wx_u16m2(vix, 15, vl);
    vmant1 = __riscv_vand_vx_u16m2(vmant1, 0x7fff, vl);
    vmant0 = __riscv_vand_vx_u16m2(vmant0, 0xff, vl);
    vmant0 = __riscv_vor_vx_u16m2(vmant0, 0x100, vl);
    
    vuint16m2_t vtwoOnPi0 = __riscv_vloxei16_v_u16m2(TwoOnPI_f32, vushift_pi, vl);
    vuint16m2_t vtwoOnPi1 = __riscv_vloxei16_v_u16m2(TwoOnPI_f32, __riscv_vadd_vx_u16m2(vushift_pi, 2, vl), vl);
    vuint16m2_t vtwoOnPi2 = __riscv_vloxei16_v_u16m2(TwoOnPI_f32, __riscv_vadd_vx_u16m2(vushift_pi, 4, vl), vl);
    vuint16m2_t vtwoOnPi3 = __riscv_vloxei16_v_u16m2(TwoOnPI_f32, __riscv_vadd_vx_u16m2(vushift_pi, 6, vl), vl);
    vuint16m2_t vtwoOnPi4 = __riscv_vloxei16_v_u16m2(TwoOnPI_f32, __riscv_vadd_vx_u16m2(vushift_pi, 8, vl), vl);
    vuint16m2_t vtwoOnPi5 = __riscv_vloxei16_v_u16m2(TwoOnPI_f32, __riscv_vadd_vx_u16m2(vushift_pi, 10, vl), vl);
    vuint16m2_t vtwoOnPi6 = __riscv_vloxei16_v_u16m2(TwoOnPI_f32, __riscv_vadd_vx_u16m2(vushift_pi, 12, vl), vl);
    vuint16m2_t vtwoOnPi7 = __riscv_vloxei16_v_u16m2(TwoOnPI_f32, __riscv_vadd_vx_u16m2(vushift_pi, 14, vl), vl);

    vuint32m4_t vr0 = __riscv_vwmaccu_vv_u32m4(__riscv_vwmulu_vv_u32m4(vtwoOnPi1, vmant0, vl), vtwoOnPi0, vmant1, vl);
    vuint32m4_t vr1 = __riscv_vwmaccu_vv_u32m4(__riscv_vwmulu_vv_u32m4(vtwoOnPi2, vmant0, vl), vtwoOnPi1, vmant1, vl);
    vuint32m4_t vr2 = __riscv_vwmaccu_vv_u32m4(__riscv_vwmulu_vv_u32m4(vtwoOnPi3, vmant0, vl), vtwoOnPi2, vmant1, vl);
    vuint32m4_t vr3 = __riscv_vwmaccu_vv_u32m4(__riscv_vwmulu_vv_u32m4(vtwoOnPi4, vmant0, vl), vtwoOnPi3, vmant1, vl);
    vuint32m4_t vr4 = __riscv_vwmaccu_vv_u32m4(__riscv_vwmulu_vv_u32m4(vtwoOnPi5, vmant0, vl), vtwoOnPi4, vmant1, vl);
    vuint32m4_t vr5 = __riscv_vwmaccu_vv_u32m4(__riscv_vwmulu_vv_u32m4(vtwoOnPi6, vmant0, vl), vtwoOnPi5, vmant1, vl);
    vuint32m4_t vr6 = __riscv_vwmaccu_vv_u32m4(__riscv_vwmulu_vv_u32m4(vtwoOnPi7, vmant0, vl), vtwoOnPi6, vmant1, vl);
    
    vuint32m4_t tmp;
    vr5 = __riscv_vadd_vv_u32m4(vr5, __riscv_vsrl_vx_u32m4(vr6, 15, vl), vl);
    tmp = __riscv_vsrl_vx_u32m4(vr5, 15, vl);
    vr4 = __riscv_vadd_vv_u32m4(vr4, tmp, vl);
    vr5 = __riscv_vsub_vv_u32m4(vr5, __riscv_vsll_vx_u32m4(tmp, 15, vl),vl);
    tmp = __riscv_vsrl_vx_u32m4(vr4, 15, vl);
    vr3 = __riscv_vadd_vv_u32m4(vr3, tmp, vl);
    vr4 = __riscv_vsub_vv_u32m4(vr4, __riscv_vsll_vx_u32m4(tmp, 15, vl),vl);
    tmp = __riscv_vsrl_vx_u32m4(vr3, 15, vl);
    vr2 = __riscv_vadd_vv_u32m4(vr2, tmp, vl);
    vr3 = __riscv_vsub_vv_u32m4(vr3, __riscv_vsll_vx_u32m4(tmp, 15, vl),vl);
    tmp = __riscv_vsrl_vx_u32m4(vr2, 15, vl);
    vr1 = __riscv_vadd_vv_u32m4(vr1, tmp, vl);
    vr2 = __riscv_vsub_vv_u32m4(vr2, __riscv_vsll_vx_u32m4(tmp, 15, vl),vl);
    tmp = __riscv_vsrl_vx_u32m4(vr1, 15, vl);
    vr0 = __riscv_vadd_vv_u32m4(vr0, tmp, vl);
    vr1 = __riscv_vsub_vv_u32m4(vr1, __riscv_vsll_vx_u32m4(tmp, 15, vl),vl);
    
    vr0 = __riscv_vor_vv_u32m4(__riscv_vsll_vx_u32m4(vr0, 4, vl), __riscv_vsrl_vx_u32m4(vr1, 11, vl), vl);
    vr1 = __riscv_vor_vv_u32m4(__riscv_vor_vv_u32m4(__riscv_vsll_vx_u32m4(vr1, 21, vl), 
                                                    __riscv_vsll_vx_u32m4(vr2, 6, vl), vl), 
                                                    __riscv_vsrl_vx_u32m4(vr3, 9, vl), vl);
    vr2 = __riscv_vor_vv_u32m4(__riscv_vor_vv_u32m4(__riscv_vsll_vx_u32m4(vr3, 23, vl), 
                                                    __riscv_vsll_vx_u32m4(vr4, 8, vl), vl), 
                                                    __riscv_vsrl_vx_u32m4(vr5, 7, vl), vl);
    
    vuint32m4_t vfinal_shift        = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vsub_vv_i32m4(
                                               vshift, __riscv_vmul_vx_i32m4(vshift_pi, 15, vl), vl));
    vuint32m4_t vshift_amount_right = __riscv_vrsub_vx_u32m4(vfinal_shift, 27, vl);
    vuint32m4_t vshift_amount_left  = __riscv_vadd_vx_u32m4(vfinal_shift, 5, vl);
    
    vuint32m4_t vr2_final = __riscv_vor_vv_u32m4(__riscv_vsrl_vv_u32m4(vr2, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u32m4(vr1, vshift_amount_left, vl), vl);
    vuint32m4_t vr1_final = __riscv_vor_vv_u32m4(__riscv_vsrl_vv_u32m4(vr1, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u32m4(vr0, vshift_amount_left, vl), vl);
    vuint32m4_t vr0_final = __riscv_vsrl_vv_u32m4(vr0, vshift_amount_right, vl);
    
    vuint32m4_t vr1_final_hi = __riscv_vsrl_vx_u32m4(vr1_final, 16, vl);
    vuint32m4_t vr1_final_lo = __riscv_vand_vx_u32m4(vr1_final, 0xffff, vl);
    vuint32m4_t vr2_final_hi = __riscv_vsrl_vx_u32m4(vr2_final, 16, vl);
    vuint32m4_t vr2_final_lo = __riscv_vand_vx_u32m4(vr2_final, 0xffff, vl);
    
    vfloat32m4_t vres_r1, vres_r2, vmres_r1, vmres_r2;
    vres_r2 = __riscv_vfcvt_f_xu_v_f32m4(vr1_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f32m4(vr1_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f32m4(vres_r2, 0x1.0p-32f, vl);
    vres_r1 = __riscv_vfmul_vf_f32m4(vres_r1, 0x1.0p-16f, vl);
    
    mask_pi_4 = __riscv_vmfge_vf_f32m4_b8(vres_r1, 0.5f, vl); 
    vmres_r2  = __riscv_vfmul_vf_f32m4(vres_r2, -0x1.0p0f, vl);
    vmres_r1  = __riscv_vfmul_vf_f32m4(vres_r1, -0x1.0p0f, vl);
    vmres_r1  = __riscv_vfadd_vf_f32m4(vmres_r1, 0x1.0p0f, vl);
    vres_r1   = __riscv_vmerge_vvm_f32m4(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2   = __riscv_vmerge_vvm_f32m4(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat32m4_t vr1_htmp, vr1_tmp;
    vr1_htmp = __riscv_vfadd_vv_f32m4(vres_r1, vres_r2, vl);
    vr1_tmp  = __riscv_vfsub_vv_f32m4(vr1_htmp, vres_r1, vl);
    vr1_tmp  = __riscv_vfsub_vv_f32m4(vr1_tmp, vres_r2, vl);
    
    vres_r2 = __riscv_vfcvt_f_xu_v_f32m4(vr2_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f32m4(vr2_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f32m4(vres_r2, 0x1.0p-64f, vl);
    vres_r1 = __riscv_vfmul_vf_f32m4(vres_r1, 0x1.0p-48f, vl);
    
    vmres_r2 = __riscv_vfmul_vf_f32m4(vres_r2, -0x1.0p0f, vl);
    vmres_r1 = __riscv_vfmul_vf_f32m4(vres_r1, -0x1.0p0f, vl);
    vres_r1  = __riscv_vmerge_vvm_f32m4(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2  = __riscv_vmerge_vvm_f32m4(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat32m4_t vr1_hi, vr2_hi, vr1_lo;
    vr2_hi  = __riscv_vfadd_vv_f32m4(vres_r1, vres_r2, vl);
    vr1_lo  = __riscv_vfsub_vv_f32m4(vr2_hi, vr1_tmp, vl);
    vr1_tmp = __riscv_vfadd_vv_f32m4(vr1_lo, vr1_tmp, vl);
    vr2_hi  = __riscv_vfsub_vv_f32m4(vr2_hi, vr1_tmp, vl);
    
    vr1_hi  = __riscv_vfadd_vv_f32m4(vr1_htmp, vr1_lo, vl);
    vr1_tmp = __riscv_vfsub_vv_f32m4(vr1_hi, vr1_htmp, vl);
    vr1_lo  = __riscv_vfsub_vv_f32m4(vr1_tmp, vr1_lo, vl);
    
    redx    = __riscv_vfmul_vf_f32m4(vr1_hi, PI_DIV_2, vl);
    redx_lo = __riscv_vfmadd_vf_f32m4(vr1_hi, -PI_DIV_2, redx, vl);    
    redx_lo = __riscv_vfmacc_vf_f32m4(redx_lo, PI_DIV_2, vr1_lo, vl);
    redx_lo = __riscv_vfmsac_vf_f32m4(redx_lo, PI_DIV_2_lo, vr1_hi, vl); 
    
    vbool8_t m  = __riscv_vmfgt_vf_f32m4_b8(redx, 0x1.921fb6p-1f, vl);
    vr1_hi      = __riscv_vfrsub_vf_f32m4(redx, 0x1.921fb6p0f, vl);
    vr1_lo      = __riscv_vfrsub_vf_f32m4(redx_lo, -0x1.777a5cp-25f, vl);
    vres_r1     = __riscv_vfadd_vv_f32m4(vr1_hi, vr1_lo, vl);
    vres_r2     = __riscv_vfsub_vv_f32m4(vr1_lo, __riscv_vfsub_vv_f32m4(vres_r1, vr1_hi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f32m4(redx, vres_r1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f32m4(redx_lo, vres_r2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b8(mask_pi_4, m, vl);
    
    vuint32m4_t slow_path_quadrant = __riscv_vand_vx_u32m4(vr0_final, 0x3, vl);

    redx = __riscv_vmerge_vvm_f32m4(redx, vx, vfast_mask, vl);
    redx_lo = __riscv_vfmerge_vfm_f32m4(redx_lo, 0.0f, vfast_mask, vl);
    
    mask_pi_4 = __riscv_vmand_mm_b8(mask_pi_4, __riscv_vmnot_m_b8(vfast_mask, vl), vl); 
    quadrants = __riscv_vmerge_vxm_u32m4(slow_path_quadrant, 0, vfast_mask, vl);
}

inline void __riscv_vred_codywaite_f32m4(vfloat32m4_t vx, vfloat32m4_t& redx, vfloat32m4_t& redx_lo, 
                                         vuint32m4_t& quadrants, vbool8_t& mask_pi_4, size_t avl) 
{
    constexpr float INV_PI_DIV_2 = 0x1.45f308p-1f; 
    constexpr float PI_2_HI      = 0x1.920000p+0f; 
    constexpr float PI_2_MID     = 0x1.fb4000p-12f;
    constexpr float PI_2_LO      = 0x1.444000p-24f;
    constexpr float PI_2_TAIL    = 0x1.68c235p-39f;

    size_t vl = __riscv_vsetvl_e32m4(avl);

    vfloat32m4_t vfn  = __riscv_vfmul_vf_f32m4(vx, INV_PI_DIV_2, vl);
    vuint32m4_t  vifn = __riscv_vfcvt_xu_f_v_u32m4(vfn, vl);
    vfn = __riscv_vfcvt_f_xu_v_f32m4(vifn, vl);

    vfloat32m4_t vr1  = __riscv_vfnmsac_vf_f32m4(vx, PI_2_HI, vfn, vl);
    vfloat32m4_t vr2  = __riscv_vfnmsac_vf_f32m4(vr1, PI_2_MID, vfn, vl);
    vfloat32m4_t err2 = __riscv_vfnmsac_vf_f32m4(__riscv_vfsub_vv_f32m4(vr1, vr2, vl), PI_2_MID, vfn, vl);
    vfloat32m4_t vrhi = __riscv_vfnmsac_vf_f32m4(vr2, PI_2_LO, vfn, vl);
    vfloat32m4_t err3 = __riscv_vfnmsac_vf_f32m4(__riscv_vfsub_vv_f32m4(vr2, vrhi, vl), PI_2_LO, vfn, vl);
    vfloat32m4_t vrlo = __riscv_vfadd_vv_f32m4(err2, err3, vl);
    vrlo              = __riscv_vfnmsac_vf_f32m4(vrlo, PI_2_TAIL, vfn, vl);

    mask_pi_4 = __riscv_vmflt_vf_f32m4_b8(vrhi, 0.0f, vl);

    vuint32m4_t vsign = __riscv_vsrl_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(vrhi), 31, vl);
    quadrants         = __riscv_vand_vx_u32m4(__riscv_vsub_vv_u32m4(vifn, vsign, vl), 3, vl);

    vrlo = __riscv_vfsgnjx_vv_f32m4(vrlo, vrhi, vl);
    vrhi = __riscv_vfabs_v_f32m4(vrhi, vl);

    redx    = __riscv_vfadd_vv_f32m4(vrhi, vrlo, vl);
    redx_lo = __riscv_vfsub_vv_f32m4(vrlo, __riscv_vfsub_vv_f32m4(redx, vrhi, vl), vl);
    
    vbool8_t m = __riscv_vmfgt_vf_f32m4_b8(redx, 0x1.921fb6p-1f, vl);
    vrhi        = __riscv_vfrsub_vf_f32m4(redx, 0x1.921fb6p0f, vl);
    vrlo        = __riscv_vfrsub_vf_f32m4(redx_lo, -0x1.777a5cp-25f, vl);
    vr1         = __riscv_vfadd_vv_f32m4(vrhi, vrlo, vl);
    vr2         = __riscv_vfsub_vv_f32m4(vrlo, __riscv_vfsub_vv_f32m4(vr1, vrhi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f32m4(redx, vr1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f32m4(redx_lo, vr2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b8(mask_pi_4, m, vl);
}

void __riscv_vreduction_pi4_f32m4(vfloat32m4_t vx, vfloat32m4_t& redx, vfloat32m4_t& redx_lo, 
                                  vuint32m4_t& quadrants, vbool8_t& mask_pi_4, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m4(avl);
    
    vx = __riscv_vreinterpret_v_u32m4_f32m4(
                __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(vx), 0x7fffffff, vl));
    
    mask_pi_4 = __riscv_vmfgt_vf_f32m4_b8(vx, 0x1p12f, vl);
    if (__riscv_vcpop_m_b8( mask_pi_4, vl))
        __riscv_vred_paynehanek_f32m4(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
    else
        __riscv_vred_codywaite_f32m4(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
}

#ifdef __riscv_zvfh

inline void __riscv_vred_paynehanek_f16m1(vfloat16m1_t vx, vfloat16m1_t& redx, vfloat16m1_t& redx_lo, 
                                          vuint16m1_t& quadrants, vbool16_t& mask_pi_4, size_t avl) {
    constexpr _Float16 PI_DIV_2    = 0x1.92p0f16;
    constexpr _Float16 PI_DIV_2_lo = 0x1.fb4p-12f16;
    constexpr _Float16 PI_DIV_4    = 0x1.92p-1f16;
    size_t vl = __riscv_vsetvl_e16m1(avl);
    
    vbool16_t vfast_mask = __riscv_vmfle_vf_f16m1_b16(vx, PI_DIV_4, vl);

    vuint16m1_t vix  = __riscv_vreinterpret_v_f16m1_u16m1(
                           __riscv_vfmerge_vfm_f16m1(vx, 1.75f16, vfast_mask, vl));
    vuint16m1_t vexp = __riscv_vsrl_vx_u16m1(vix, 10, vl);
    vexp             = __riscv_vand_vx_u16m1(vexp, 0x1f, vl);
    
    vint16m1_t vshift    = __riscv_vsub_vx_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(vexp), 15, vl);
    vint16m1_t vshift_pi = __riscv_vdiv_vx_i16m1(__riscv_vsub_vx_i16m1(vshift, 5, vl), 7, vl);
    
    vuint16m1_t vushift_pi_u16 = __riscv_vreinterpret_v_i16m1_u16m1(vshift_pi);
    vuint8mf2_t vushift_pi     = __riscv_vncvt_x_x_w_u8mf2(vushift_pi_u16, vl);
    
    
    vuint8mf2_t vmant1 = __riscv_vncvt_x_x_w_u8mf2(vix, vl);
    vuint8mf2_t vmant0 = __riscv_vnsrl_wx_u8mf2(vix, 7, vl);
    vmant1 = __riscv_vand_vx_u8mf2(vmant1, 0x7f, vl);
    vmant0 = __riscv_vand_vx_u8mf2(vmant0, 0x7, vl);
    vmant0 = __riscv_vor_vx_u8mf2(vmant0, 0x8, vl);
    
    vuint8mf2_t vtwoOnPi0 = __riscv_vloxei8_v_u8mf2(TwoOnPI_f16, vushift_pi, vl);
    vuint8mf2_t vtwoOnPi1 = __riscv_vloxei8_v_u8mf2(TwoOnPI_f16, __riscv_vadd_vx_u8mf2(vushift_pi, 1, vl), vl);
    vuint8mf2_t vtwoOnPi2 = __riscv_vloxei8_v_u8mf2(TwoOnPI_f16, __riscv_vadd_vx_u8mf2(vushift_pi, 2, vl), vl);
    vuint8mf2_t vtwoOnPi3 = __riscv_vloxei8_v_u8mf2(TwoOnPI_f16, __riscv_vadd_vx_u8mf2(vushift_pi, 3, vl), vl);
    vuint8mf2_t vtwoOnPi4 = __riscv_vloxei8_v_u8mf2(TwoOnPI_f16, __riscv_vadd_vx_u8mf2(vushift_pi, 4, vl), vl);
    vuint8mf2_t vtwoOnPi5 = __riscv_vloxei8_v_u8mf2(TwoOnPI_f16, __riscv_vadd_vx_u8mf2(vushift_pi, 5, vl), vl);
    vuint8mf2_t vtwoOnPi6 = __riscv_vloxei8_v_u8mf2(TwoOnPI_f16, __riscv_vadd_vx_u8mf2(vushift_pi, 6, vl), vl);
    
    vuint16m1_t vr0 = __riscv_vwmaccu_vv_u16m1(__riscv_vwmulu_vv_u16m1(vtwoOnPi1, vmant0, vl), vtwoOnPi0, vmant1, vl);
    vuint16m1_t vr1 = __riscv_vwmaccu_vv_u16m1(__riscv_vwmulu_vv_u16m1(vtwoOnPi2, vmant0, vl), vtwoOnPi1, vmant1, vl);
    vuint16m1_t vr2 = __riscv_vwmaccu_vv_u16m1(__riscv_vwmulu_vv_u16m1(vtwoOnPi3, vmant0, vl), vtwoOnPi2, vmant1, vl);
    vuint16m1_t vr3 = __riscv_vwmaccu_vv_u16m1(__riscv_vwmulu_vv_u16m1(vtwoOnPi4, vmant0, vl), vtwoOnPi3, vmant1, vl);
    vuint16m1_t vr4 = __riscv_vwmaccu_vv_u16m1(__riscv_vwmulu_vv_u16m1(vtwoOnPi5, vmant0, vl), vtwoOnPi4, vmant1, vl);
    vuint16m1_t vr5 = __riscv_vwmaccu_vv_u16m1(__riscv_vwmulu_vv_u16m1(vtwoOnPi6, vmant0, vl), vtwoOnPi5, vmant1, vl);
    
    vuint16m1_t tmp;
    vr4 = __riscv_vadd_vv_u16m1(vr4, __riscv_vsrl_vx_u16m1(vr5, 7, vl), vl);
    tmp =  __riscv_vsrl_vx_u16m1(vr4, 7, vl);
    vr3 = __riscv_vadd_vv_u16m1(vr3, tmp, vl);
    vr4 = __riscv_vsub_vv_u16m1(vr4, __riscv_vsll_vx_u16m1(tmp, 7, vl),vl);
    tmp =  __riscv_vsrl_vx_u16m1(vr3, 7, vl);
    vr2 = __riscv_vadd_vv_u16m1(vr2, tmp, vl);
    vr3 = __riscv_vsub_vv_u16m1(vr3, __riscv_vsll_vx_u16m1(tmp, 7, vl),vl);
    tmp =  __riscv_vsrl_vx_u16m1(vr2, 7, vl);
    vr1 = __riscv_vadd_vv_u16m1(vr1, tmp, vl);
    vr2 = __riscv_vsub_vv_u16m1(vr2, __riscv_vsll_vx_u16m1(tmp, 7, vl),vl);
    tmp =  __riscv_vsrl_vx_u16m1(vr1, 7, vl);
    vr0 = __riscv_vadd_vv_u16m1(vr0, tmp, vl);
    vr1 = __riscv_vsub_vv_u16m1(vr1, __riscv_vsll_vx_u16m1(tmp, 7, vl),vl);
    
    vr0 = __riscv_vor_vv_u16m1(__riscv_vsll_vx_u16m1(vr0, 4, vl), __riscv_vsrl_vx_u16m1(vr1, 3, vl), vl);
    vr1 = __riscv_vor_vv_u16m1(__riscv_vor_vv_u16m1(__riscv_vsll_vx_u16m1(vr1, 13, vl), 
                                                    __riscv_vsll_vx_u16m1(vr2, 6, vl), vl), 
                                                    __riscv_vsrl_vx_u16m1(vr3, 1, vl), vl);
    vr2 = __riscv_vor_vv_u16m1(__riscv_vsll_vx_u16m1(vr3, 15, vl), __riscv_vsll_vx_u16m1(vr4, 8, vl), vl);
    
    vuint16m1_t vfinal_shift        = __riscv_vreinterpret_v_i16m1_u16m1(
                                            __riscv_vsub_vv_i16m1(vshift, __riscv_vmul_vx_i16m1(vshift_pi, 7, vl), vl));
    vuint16m1_t vshift_amount_right = __riscv_vrsub_vx_u16m1(vfinal_shift, 14, vl);
    vuint16m1_t vshift_amount_left  = __riscv_vadd_vx_u16m1(vfinal_shift, 2, vl);
    
    vuint16m1_t vr2_final = __riscv_vor_vv_u16m1(__riscv_vsrl_vv_u16m1(vr2, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u16m1(vr1, vshift_amount_left, vl), vl);
    vuint16m1_t vr1_final = __riscv_vor_vv_u16m1(__riscv_vsrl_vv_u16m1(vr1, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u16m1(vr0, vshift_amount_left, vl), vl);
    vuint16m1_t vr0_final = __riscv_vsrl_vv_u16m1(vr0, vshift_amount_right, vl);
    
    vuint16m1_t vr1_final_hi = __riscv_vsrl_vx_u16m1(vr1_final, 8, vl);
    vuint16m1_t vr1_final_lo = __riscv_vand_vx_u16m1(vr1_final, 0xff, vl);
    vuint16m1_t vr2_final_hi = __riscv_vsrl_vx_u16m1(vr2_final, 8, vl);
    vuint16m1_t vr2_final_lo = __riscv_vand_vx_u16m1(vr2_final, 0xff, vl);
    
    vfloat16m1_t vres_r1, vres_r2, vmres_r1, vmres_r2;
    vres_r2 = __riscv_vfcvt_f_xu_v_f16m1(vr1_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f16m1(vr1_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f16m1(vres_r2, 0x1.0p-4f16, vl);
    vres_r1 = __riscv_vfmul_vf_f16m1(vres_r1, 0x1.0p4f16, vl);
    
    mask_pi_4 = __riscv_vmfge_vf_f16m1_b16(vres_r1, 0x1.0p11f16, vl);
    vmres_r2  = __riscv_vfmul_vf_f16m1(vres_r2, -0x1.0p0f16, vl);
    vmres_r1  = __riscv_vfmul_vf_f16m1(vres_r1, -0x1.0p0f16, vl);
    vmres_r1  = __riscv_vfadd_vf_f16m1(vmres_r1, 0x1.0p12f16, vl);
    vres_r1   = __riscv_vmerge_vvm_f16m1(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2   = __riscv_vmerge_vvm_f16m1(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat16m1_t vr1_htmp, vr1_tmp;
    vr1_htmp = __riscv_vfadd_vv_f16m1(vres_r1, vres_r2, vl);
    vr1_tmp  = __riscv_vfsub_vv_f16m1(vres_r1, vr1_htmp, vl);
    vr1_tmp  = __riscv_vfadd_vv_f16m1(vr1_tmp, vres_r2, vl);
    
    vres_r2 = __riscv_vfcvt_f_xu_v_f16m1(vr2_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f16m1(vr2_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f16m1(vres_r2, 0x1.0p-20f16, vl);
    vres_r1 = __riscv_vfmul_vf_f16m1(vres_r1, 0x1.0p-12f16, vl);
    
    vmres_r2 = __riscv_vfmul_vf_f16m1(vres_r2, -0x1.0p0f16, vl);
    vmres_r1 = __riscv_vfmul_vf_f16m1(vres_r1, -0x1.0p0f16, vl);
    vres_r2  = __riscv_vmerge_vvm_f16m1(vres_r2, vmres_r2, mask_pi_4, vl);
    vres_r1  = __riscv_vmerge_vvm_f16m1(vres_r1, vmres_r1, mask_pi_4, vl);
    
    vfloat16m1_t vr1_hi, vr2_hi, vr1_lo, vr2_lo;
    vr2_hi = __riscv_vfadd_vv_f16m1(vres_r1, vres_r2, vl);
    vr2_lo = __riscv_vfsub_vv_f16m1(vres_r1, vr2_hi, vl);
    vr2_lo = __riscv_vfadd_vv_f16m1(vr2_lo, vres_r2, vl);
    
    vr1_lo  = __riscv_vfadd_vv_f16m1(vr2_hi, vr1_tmp, vl);
    vr1_tmp = __riscv_vfsub_vv_f16m1(vr1_tmp, vr1_lo,  vl);
    vr1_tmp = __riscv_vfadd_vv_f16m1(vr2_hi, vr1_tmp, vl);
    
    vr1_hi   = __riscv_vfadd_vv_f16m1(vr1_htmp, vr1_lo, vl);
    vr1_htmp = __riscv_vfsub_vv_f16m1(vr1_htmp, vr1_hi, vl);
    vr1_lo = __riscv_vfadd_vv_f16m1(vr1_htmp, vr1_lo, vl);
    vr1_lo = __riscv_vfadd_vv_f16m1(vr1_lo, vr2_lo, vl);
    vr1_lo = __riscv_vfadd_vv_f16m1(vr1_lo, vr1_tmp, vl);
    
    vres_r1 = __riscv_vfmul_vf_f16m1(vr1_hi, PI_DIV_2, vl);
    redx_lo = __riscv_vfmsub_vf_f16m1(vr1_hi, PI_DIV_2, vres_r1, vl); 
    redx_lo = __riscv_vfmacc_vf_f16m1(redx_lo, PI_DIV_2_lo, vr1_hi, vl);   
    redx_lo = __riscv_vfmacc_vf_f16m1(redx_lo, PI_DIV_2, vr1_lo, vl);
    
    redx    = __riscv_vfmul_vf_f16m1(vres_r1, 0x1p-12f16, vl);
    vres_r1 = __riscv_vfmsub_vf_f16m1(redx, 0x1p12f16, vres_r1, vl);
    redx_lo = __riscv_vfsub_vv_f16m1(redx_lo, vres_r1, vl);
    redx_lo = __riscv_vfmul_vf_f16m1(redx_lo, 0x1p-12f16, vl);
    
    vuint16m1_t slow_path_quadrant = __riscv_vand_vx_u16m1(vr0_final, 0x3, vl);

    redx    = __riscv_vmerge_vvm_f16m1(redx, vx, vfast_mask, vl);
    redx_lo = __riscv_vfmerge_vfm_f16m1(redx_lo, 0.0f16, vfast_mask, vl);
    
    mask_pi_4 = __riscv_vmand_mm_b16(mask_pi_4, __riscv_vmnot_m_b16(vfast_mask, vl), vl);
    quadrants = __riscv_vmerge_vxm_u16m1(slow_path_quadrant, 0, vfast_mask, vl);   
}

inline void __riscv_vred_codywaite_f16m1(vfloat16m1_t vx, vfloat16m1_t& redx, vfloat16m1_t& redx_lo, 
                                         vuint16m1_t& quadrants, vbool16_t& mask_pi_4, size_t avl) 
{
    constexpr _Float16 INV_PI_DIV_2 = 0x1.45cp-1f16; 
    constexpr _Float16 PI_2_HI      = 0x1.900p+0f16; 
    constexpr _Float16 PI_2_MID     = 0x1.0c0p-7f16; 
    constexpr _Float16 PI_2_LO      = 0x1.ed4p-14f16;

    size_t vl = __riscv_vsetvl_e16m1(avl);

    vfloat16m1_t vfn  = __riscv_vfmul_vf_f16m1(vx, INV_PI_DIV_2, vl);
    vuint16m1_t  vifn = __riscv_vfcvt_xu_f_v_u16m1(vfn, vl);
    vfn               = __riscv_vfcvt_f_xu_v_f16m1(vifn, vl);

    vfloat16m1_t vr2;
    vfloat16m1_t vr1  = __riscv_vfnmsac_vf_f16m1(vx, PI_2_HI, vfn, vl);
    vfloat16m1_t vrhi = __riscv_vfnmsac_vf_f16m1(vr1, PI_2_MID, vfn, vl);
    vfloat16m1_t vrlo = __riscv_vfnmsac_vf_f16m1(__riscv_vfsub_vv_f16m1(vr1, vrhi, vl), PI_2_MID, vfn, vl);
    vrlo              = __riscv_vfnmsac_vf_f16m1(vrlo, PI_2_LO, vfn, vl);

    mask_pi_4 = __riscv_vmflt_vf_f16m1_b16(vrhi, 0.0f16, vl);

    vuint16m1_t vsign = __riscv_vsrl_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(vrhi), 15, vl);
    quadrants         = __riscv_vand_vx_u16m1(__riscv_vsub_vv_u16m1(vifn, vsign, vl), 3, vl);

    vrlo = __riscv_vfsgnjx_vv_f16m1(vrlo, vrhi, vl);
    vrhi = __riscv_vfabs_v_f16m1(vrhi, vl);

    redx    = __riscv_vfadd_vv_f16m1(vrhi, vrlo, vl);
    redx_lo = __riscv_vfsub_vv_f16m1(vrlo, __riscv_vfsub_vv_f16m1(redx, vrhi, vl), vl);

    vbool16_t m = __riscv_vmfgt_vf_f16m1_b16(redx, 0x1.92p-1f16, vl);
    vrhi        = __riscv_vfrsub_vf_f16m1(redx, 0x1.92p0f16, vl);
    vrlo        = __riscv_vfrsub_vf_f16m1(redx_lo, 0x1.fb4p-12f16, vl);
    vr1         = __riscv_vfadd_vv_f16m1(vrhi, vrlo, vl);
    vr2         = __riscv_vfsub_vv_f16m1(vrlo, __riscv_vfsub_vv_f16m1(vr1, vrhi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f16m1(redx, vr1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f16m1(redx_lo, vr2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b16(mask_pi_4, m, vl);
}

void __riscv_vreduction_pi4_f16m1(vfloat16m1_t vx, vfloat16m1_t& redx, vfloat16m1_t& redx_lo, 
                                  vuint16m1_t& quadrants, vbool16_t& mask_pi_4, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m1(avl);
    vx = __riscv_vreinterpret_v_u16m1_f16m1(
              __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(vx), 0x7fff, vl));
    
    mask_pi_4 = __riscv_vmfgt_vf_f16m1_b16(vx, 0x1p4f16, vl);
    if (__riscv_vcpop_m_b16(mask_pi_4, vl))
        __riscv_vred_paynehanek_f16m1(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
    else
        __riscv_vred_codywaite_f16m1(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
}


inline void __riscv_vred_paynehanek_f16m2(vfloat16m2_t vx, vfloat16m2_t& redx, vfloat16m2_t& redx_lo, 
                                          vuint16m2_t& quadrants, vbool8_t& mask_pi_4, size_t avl) {
    constexpr _Float16 PI_DIV_2    = 0x1.92p0f16;
    constexpr _Float16 PI_DIV_2_lo = 0x1.fb4p-12f16;
    constexpr _Float16 PI_DIV_4    = 0x1.92p-1f16;
    size_t vl = __riscv_vsetvl_e16m2(avl);
    
    vbool8_t vfast_mask = __riscv_vmfle_vf_f16m2_b8(vx, PI_DIV_4, vl);

    vuint16m2_t vix  = __riscv_vreinterpret_v_f16m2_u16m2(
                           __riscv_vfmerge_vfm_f16m2(vx, 1.75f16, vfast_mask, vl));
    vuint16m2_t vexp = __riscv_vsrl_vx_u16m2(vix, 10, vl);
    vexp             = __riscv_vand_vx_u16m2(vexp, 0x1f, vl);
    
    vint16m2_t vshift    = __riscv_vsub_vx_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(vexp), 15, vl);
    vint16m2_t vshift_pi = __riscv_vdiv_vx_i16m2(__riscv_vsub_vx_i16m2(vshift, 5, vl), 7, vl);
    
    vuint16m2_t vushift_pi_u16 = __riscv_vreinterpret_v_i16m2_u16m2(vshift_pi);
    vuint8m1_t vushift_pi      = __riscv_vncvt_x_x_w_u8m1(vushift_pi_u16, vl);
    
    
    vuint8m1_t vmant1 = __riscv_vncvt_x_x_w_u8m1(vix, vl);
    vuint8m1_t vmant0 = __riscv_vnsrl_wx_u8m1(vix, 7, vl);
    vmant1 = __riscv_vand_vx_u8m1(vmant1, 0x7f, vl);
    vmant0 = __riscv_vand_vx_u8m1(vmant0, 0x7, vl);
    vmant0 = __riscv_vor_vx_u8m1(vmant0, 0x8, vl);
    
    vuint8m1_t vtwoOnPi0 = __riscv_vloxei8_v_u8m1(TwoOnPI_f16, vushift_pi, vl);
    vuint8m1_t vtwoOnPi1 = __riscv_vloxei8_v_u8m1(TwoOnPI_f16, __riscv_vadd_vx_u8m1(vushift_pi, 1, vl), vl);
    vuint8m1_t vtwoOnPi2 = __riscv_vloxei8_v_u8m1(TwoOnPI_f16, __riscv_vadd_vx_u8m1(vushift_pi, 2, vl), vl);
    vuint8m1_t vtwoOnPi3 = __riscv_vloxei8_v_u8m1(TwoOnPI_f16, __riscv_vadd_vx_u8m1(vushift_pi, 3, vl), vl);
    vuint8m1_t vtwoOnPi4 = __riscv_vloxei8_v_u8m1(TwoOnPI_f16, __riscv_vadd_vx_u8m1(vushift_pi, 4, vl), vl);
    vuint8m1_t vtwoOnPi5 = __riscv_vloxei8_v_u8m1(TwoOnPI_f16, __riscv_vadd_vx_u8m1(vushift_pi, 5, vl), vl);
    vuint8m1_t vtwoOnPi6 = __riscv_vloxei8_v_u8m1(TwoOnPI_f16, __riscv_vadd_vx_u8m1(vushift_pi, 6, vl), vl);
    
    vuint16m2_t vr0 = __riscv_vwmaccu_vv_u16m2(__riscv_vwmulu_vv_u16m2(vtwoOnPi1, vmant0, vl), vtwoOnPi0, vmant1, vl);
    vuint16m2_t vr1 = __riscv_vwmaccu_vv_u16m2(__riscv_vwmulu_vv_u16m2(vtwoOnPi2, vmant0, vl), vtwoOnPi1, vmant1, vl);
    vuint16m2_t vr2 = __riscv_vwmaccu_vv_u16m2(__riscv_vwmulu_vv_u16m2(vtwoOnPi3, vmant0, vl), vtwoOnPi2, vmant1, vl);
    vuint16m2_t vr3 = __riscv_vwmaccu_vv_u16m2(__riscv_vwmulu_vv_u16m2(vtwoOnPi4, vmant0, vl), vtwoOnPi3, vmant1, vl);
    vuint16m2_t vr4 = __riscv_vwmaccu_vv_u16m2(__riscv_vwmulu_vv_u16m2(vtwoOnPi5, vmant0, vl), vtwoOnPi4, vmant1, vl);
    vuint16m2_t vr5 = __riscv_vwmaccu_vv_u16m2(__riscv_vwmulu_vv_u16m2(vtwoOnPi6, vmant0, vl), vtwoOnPi5, vmant1, vl);
    
    vuint16m2_t tmp;
    vr4 = __riscv_vadd_vv_u16m2(vr4, __riscv_vsrl_vx_u16m2(vr5, 7, vl), vl);
    tmp =  __riscv_vsrl_vx_u16m2(vr4, 7, vl);
    vr3 = __riscv_vadd_vv_u16m2(vr3, tmp, vl);
    vr4 = __riscv_vsub_vv_u16m2(vr4, __riscv_vsll_vx_u16m2(tmp, 7, vl),vl);
    tmp =  __riscv_vsrl_vx_u16m2(vr3, 7, vl);
    vr2 = __riscv_vadd_vv_u16m2(vr2, tmp, vl);
    vr3 = __riscv_vsub_vv_u16m2(vr3, __riscv_vsll_vx_u16m2(tmp, 7, vl),vl);
    tmp =  __riscv_vsrl_vx_u16m2(vr2, 7, vl);
    vr1 = __riscv_vadd_vv_u16m2(vr1, tmp, vl);
    vr2 = __riscv_vsub_vv_u16m2(vr2, __riscv_vsll_vx_u16m2(tmp, 7, vl),vl);
    tmp =  __riscv_vsrl_vx_u16m2(vr1, 7, vl);
    vr0 = __riscv_vadd_vv_u16m2(vr0, tmp, vl);
    vr1 = __riscv_vsub_vv_u16m2(vr1, __riscv_vsll_vx_u16m2(tmp, 7, vl),vl);
    
    vr0 = __riscv_vor_vv_u16m2(__riscv_vsll_vx_u16m2(vr0, 4, vl), __riscv_vsrl_vx_u16m2(vr1, 3, vl), vl);
    vr1 = __riscv_vor_vv_u16m2(__riscv_vor_vv_u16m2(__riscv_vsll_vx_u16m2(vr1, 13, vl), 
                                                    __riscv_vsll_vx_u16m2(vr2, 6, vl), vl), 
                                                    __riscv_vsrl_vx_u16m2(vr3, 1, vl), vl);
    vr2 = __riscv_vor_vv_u16m2(__riscv_vsll_vx_u16m2(vr3, 15, vl), __riscv_vsll_vx_u16m2(vr4, 8, vl), vl);
    
    vuint16m2_t vfinal_shift        = __riscv_vreinterpret_v_i16m2_u16m2(
                                            __riscv_vsub_vv_i16m2(vshift, __riscv_vmul_vx_i16m2(vshift_pi, 7, vl), vl));
    vuint16m2_t vshift_amount_right = __riscv_vrsub_vx_u16m2(vfinal_shift, 14, vl);
    vuint16m2_t vshift_amount_left  = __riscv_vadd_vx_u16m2(vfinal_shift, 2, vl);
    
    vuint16m2_t vr2_final = __riscv_vor_vv_u16m2(__riscv_vsrl_vv_u16m2(vr2, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u16m2(vr1, vshift_amount_left, vl), vl);
    vuint16m2_t vr1_final = __riscv_vor_vv_u16m2(__riscv_vsrl_vv_u16m2(vr1, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u16m2(vr0, vshift_amount_left, vl), vl);
    vuint16m2_t vr0_final = __riscv_vsrl_vv_u16m2(vr0, vshift_amount_right, vl);
    
    vuint16m2_t vr1_final_hi = __riscv_vsrl_vx_u16m2(vr1_final, 8, vl);
    vuint16m2_t vr1_final_lo = __riscv_vand_vx_u16m2(vr1_final, 0xff, vl);
    vuint16m2_t vr2_final_hi = __riscv_vsrl_vx_u16m2(vr2_final, 8, vl);
    vuint16m2_t vr2_final_lo = __riscv_vand_vx_u16m2(vr2_final, 0xff, vl);
    
    vfloat16m2_t vres_r1, vres_r2, vmres_r1, vmres_r2;
    vres_r2 = __riscv_vfcvt_f_xu_v_f16m2(vr1_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f16m2(vr1_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f16m2(vres_r2, 0x1.0p-4f16, vl);
    vres_r1 = __riscv_vfmul_vf_f16m2(vres_r1, 0x1.0p4f16, vl);
    
    mask_pi_4 = __riscv_vmfge_vf_f16m2_b8(vres_r1, 0x1.0p11f16, vl);
    vmres_r2  = __riscv_vfmul_vf_f16m2(vres_r2, -0x1.0p0f16, vl);
    vmres_r1  = __riscv_vfmul_vf_f16m2(vres_r1, -0x1.0p0f16, vl);
    vmres_r1  = __riscv_vfadd_vf_f16m2(vmres_r1, 0x1.0p12f16, vl);
    vres_r1   = __riscv_vmerge_vvm_f16m2(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2   = __riscv_vmerge_vvm_f16m2(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat16m2_t vr1_htmp, vr1_tmp;
    vr1_htmp = __riscv_vfadd_vv_f16m2(vres_r1, vres_r2, vl);
    vr1_tmp  = __riscv_vfsub_vv_f16m2(vres_r1, vr1_htmp, vl);
    vr1_tmp  = __riscv_vfadd_vv_f16m2(vr1_tmp, vres_r2, vl);
    
    vres_r2 = __riscv_vfcvt_f_xu_v_f16m2(vr2_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f16m2(vr2_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f16m2(vres_r2, 0x1.0p-20f16, vl);
    vres_r1 = __riscv_vfmul_vf_f16m2(vres_r1, 0x1.0p-12f16, vl);
    
    vmres_r2 = __riscv_vfmul_vf_f16m2(vres_r2, -0x1.0p0f16, vl);
    vmres_r1 = __riscv_vfmul_vf_f16m2(vres_r1, -0x1.0p0f16, vl);
    vres_r2  = __riscv_vmerge_vvm_f16m2(vres_r2, vmres_r2, mask_pi_4, vl);
    vres_r1  = __riscv_vmerge_vvm_f16m2(vres_r1, vmres_r1, mask_pi_4, vl);
    
    vfloat16m2_t vr1_hi, vr2_hi, vr1_lo, vr2_lo;
    vr2_hi = __riscv_vfadd_vv_f16m2(vres_r1, vres_r2, vl);
    vr2_lo = __riscv_vfsub_vv_f16m2(vres_r1, vr2_hi, vl);
    vr2_lo = __riscv_vfadd_vv_f16m2(vr2_lo, vres_r2, vl);
    
    vr1_lo  = __riscv_vfadd_vv_f16m2(vr2_hi, vr1_tmp, vl);
    vr1_tmp = __riscv_vfsub_vv_f16m2(vr1_tmp, vr1_lo,  vl);
    vr1_tmp = __riscv_vfadd_vv_f16m2(vr2_hi, vr1_tmp, vl);
    
    vr1_hi   = __riscv_vfadd_vv_f16m2(vr1_htmp, vr1_lo, vl);
    vr1_htmp = __riscv_vfsub_vv_f16m2(vr1_htmp, vr1_hi, vl);
    vr1_lo = __riscv_vfadd_vv_f16m2(vr1_htmp, vr1_lo, vl);
    vr1_lo = __riscv_vfadd_vv_f16m2(vr1_lo, vr2_lo, vl);
    vr1_lo = __riscv_vfadd_vv_f16m2(vr1_lo, vr1_tmp, vl);
    
    vres_r1 = __riscv_vfmul_vf_f16m2(vr1_hi, PI_DIV_2, vl);
    redx_lo = __riscv_vfmsub_vf_f16m2(vr1_hi, PI_DIV_2, vres_r1, vl); 
    redx_lo = __riscv_vfmacc_vf_f16m2(redx_lo, PI_DIV_2_lo, vr1_hi, vl);   
    redx_lo = __riscv_vfmacc_vf_f16m2(redx_lo, PI_DIV_2, vr1_lo, vl);
    
    redx    = __riscv_vfmul_vf_f16m2(vres_r1, 0x1p-12f16, vl);
    vres_r1 = __riscv_vfmsub_vf_f16m2(redx, 0x1p12f16, vres_r1, vl);
    redx_lo = __riscv_vfsub_vv_f16m2(redx_lo, vres_r1, vl);
    redx_lo = __riscv_vfmul_vf_f16m2(redx_lo, 0x1p-12f16, vl);
        
    vuint16m2_t slow_path_quadrant = __riscv_vand_vx_u16m2(vr0_final, 0x3, vl);

    redx    = __riscv_vmerge_vvm_f16m2(redx, vx, vfast_mask, vl);
    redx_lo = __riscv_vfmerge_vfm_f16m2(redx_lo, 0.0f16, vfast_mask, vl);
    
    mask_pi_4 = __riscv_vmand_mm_b8(mask_pi_4, __riscv_vmnot_m_b8(vfast_mask, vl), vl);
    quadrants = __riscv_vmerge_vxm_u16m2(slow_path_quadrant, 0, vfast_mask, vl);   
}

inline void __riscv_vred_codywaite_f16m2(vfloat16m2_t vx, vfloat16m2_t& redx, vfloat16m2_t& redx_lo, 
                                         vuint16m2_t& quadrants, vbool8_t& mask_pi_4, size_t avl) 
{
    constexpr _Float16 INV_PI_DIV_2 = 0x1.45cp-1f16; 
    constexpr _Float16 PI_2_HI      = 0x1.900p+0f16; 
    constexpr _Float16 PI_2_MID     = 0x1.0c0p-7f16; 
    constexpr _Float16 PI_2_LO      = 0x1.ed4p-14f16;

    size_t vl = __riscv_vsetvl_e16m2(avl);

    vfloat16m2_t vfn  = __riscv_vfmul_vf_f16m2(vx, INV_PI_DIV_2, vl);
    vuint16m2_t  vifn = __riscv_vfcvt_xu_f_v_u16m2(vfn, vl);
    vfn               = __riscv_vfcvt_f_xu_v_f16m2(vifn, vl);

    vfloat16m2_t vr2;
    vfloat16m2_t vr1  = __riscv_vfnmsac_vf_f16m2(vx, PI_2_HI, vfn, vl);
    vfloat16m2_t vrhi = __riscv_vfnmsac_vf_f16m2(vr1, PI_2_MID, vfn, vl);
    vfloat16m2_t vrlo = __riscv_vfnmsac_vf_f16m2(__riscv_vfsub_vv_f16m2(vr1, vrhi, vl), PI_2_MID, vfn, vl);
    vrlo              = __riscv_vfnmsac_vf_f16m2(vrlo, PI_2_LO, vfn, vl);

    mask_pi_4 = __riscv_vmflt_vf_f16m2_b8(vrhi, 0.0f16, vl);

    vuint16m2_t vsign = __riscv_vsrl_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(vrhi), 15, vl);
    quadrants         = __riscv_vand_vx_u16m2(__riscv_vsub_vv_u16m2(vifn, vsign, vl), 3, vl);

    vrlo = __riscv_vfsgnjx_vv_f16m2(vrlo, vrhi, vl);
    vrhi = __riscv_vfabs_v_f16m2(vrhi, vl);

    redx    = __riscv_vfadd_vv_f16m2(vrhi, vrlo, vl);
    redx_lo = __riscv_vfsub_vv_f16m2(vrlo, __riscv_vfsub_vv_f16m2(redx, vrhi, vl), vl);

    vbool8_t m = __riscv_vmfgt_vf_f16m2_b8(redx, 0x1.92p-1f16, vl);
    vrhi        = __riscv_vfrsub_vf_f16m2(redx, 0x1.92p0f16, vl);
    vrlo        = __riscv_vfrsub_vf_f16m2(redx_lo, 0x1.fb4p-12f16, vl);
    vr1         = __riscv_vfadd_vv_f16m2(vrhi, vrlo, vl);
    vr2         = __riscv_vfsub_vv_f16m2(vrlo, __riscv_vfsub_vv_f16m2(vr1, vrhi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f16m2(redx, vr1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f16m2(redx_lo, vr2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b8(mask_pi_4, m, vl);
}

void __riscv_vreduction_pi4_f16m2(vfloat16m2_t vx, vfloat16m2_t& redx, vfloat16m2_t& redx_lo, 
                                  vuint16m2_t& quadrants, vbool8_t& mask_pi_4, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m2(avl);
    vx = __riscv_vreinterpret_v_u16m2_f16m2(
              __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(vx), 0x7fff, vl));
    
    mask_pi_4 = __riscv_vmfgt_vf_f16m2_b8(vx, 0x1p4f16, vl);
    if (__riscv_vcpop_m_b8(mask_pi_4, vl))
        __riscv_vred_paynehanek_f16m2(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
    else
        __riscv_vred_codywaite_f16m2(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
}


inline void __riscv_vred_paynehanek_f16m4(vfloat16m4_t vx, vfloat16m4_t& redx, vfloat16m4_t& redx_lo, 
                                          vuint16m4_t& quadrants, vbool4_t& mask_pi_4, size_t avl) {
    constexpr _Float16 PI_DIV_2    = 0x1.92p0f16;
    constexpr _Float16 PI_DIV_2_lo = 0x1.fb4p-12f16;
    constexpr _Float16 PI_DIV_4    = 0x1.92p-1f16;
    size_t vl = __riscv_vsetvl_e16m4(avl);
    
    vbool4_t vfast_mask = __riscv_vmfle_vf_f16m4_b4(vx, PI_DIV_4, vl);

    vuint16m4_t vix  = __riscv_vreinterpret_v_f16m4_u16m4(
                           __riscv_vfmerge_vfm_f16m4(vx, 1.75f16, vfast_mask, vl));
    vuint16m4_t vexp = __riscv_vsrl_vx_u16m4(vix, 10, vl);
    vexp             = __riscv_vand_vx_u16m4(vexp, 0x1f, vl);
    
    vint16m4_t vshift    = __riscv_vsub_vx_i16m4(__riscv_vreinterpret_v_u16m4_i16m4(vexp), 15, vl);
    vint16m4_t vshift_pi = __riscv_vdiv_vx_i16m4(__riscv_vsub_vx_i16m4(vshift, 5, vl), 7, vl);
    
    vuint16m4_t vushift_pi_u16 = __riscv_vreinterpret_v_i16m4_u16m4(vshift_pi);
    vuint8m2_t vushift_pi      = __riscv_vncvt_x_x_w_u8m2(vushift_pi_u16, vl);
    
    
    vuint8m2_t vmant1 = __riscv_vncvt_x_x_w_u8m2(vix, vl);
    vuint8m2_t vmant0 = __riscv_vnsrl_wx_u8m2(vix, 7, vl);
    vmant1 = __riscv_vand_vx_u8m2(vmant1, 0x7f, vl);
    vmant0 = __riscv_vand_vx_u8m2(vmant0, 0x7, vl);
    vmant0 = __riscv_vor_vx_u8m2(vmant0, 0x8, vl);
    
    vuint8m2_t vtwoOnPi0 = __riscv_vloxei8_v_u8m2(TwoOnPI_f16, vushift_pi, vl);
    vuint8m2_t vtwoOnPi1 = __riscv_vloxei8_v_u8m2(TwoOnPI_f16, __riscv_vadd_vx_u8m2(vushift_pi, 1, vl), vl);
    vuint8m2_t vtwoOnPi2 = __riscv_vloxei8_v_u8m2(TwoOnPI_f16, __riscv_vadd_vx_u8m2(vushift_pi, 2, vl), vl);
    vuint8m2_t vtwoOnPi3 = __riscv_vloxei8_v_u8m2(TwoOnPI_f16, __riscv_vadd_vx_u8m2(vushift_pi, 3, vl), vl);
    vuint8m2_t vtwoOnPi4 = __riscv_vloxei8_v_u8m2(TwoOnPI_f16, __riscv_vadd_vx_u8m2(vushift_pi, 4, vl), vl);
    vuint8m2_t vtwoOnPi5 = __riscv_vloxei8_v_u8m2(TwoOnPI_f16, __riscv_vadd_vx_u8m2(vushift_pi, 5, vl), vl);
    vuint8m2_t vtwoOnPi6 = __riscv_vloxei8_v_u8m2(TwoOnPI_f16, __riscv_vadd_vx_u8m2(vushift_pi, 6, vl), vl);
    
    vuint16m4_t vr0 = __riscv_vwmaccu_vv_u16m4(__riscv_vwmulu_vv_u16m4(vtwoOnPi1, vmant0, vl), vtwoOnPi0, vmant1, vl);
    vuint16m4_t vr1 = __riscv_vwmaccu_vv_u16m4(__riscv_vwmulu_vv_u16m4(vtwoOnPi2, vmant0, vl), vtwoOnPi1, vmant1, vl);
    vuint16m4_t vr2 = __riscv_vwmaccu_vv_u16m4(__riscv_vwmulu_vv_u16m4(vtwoOnPi3, vmant0, vl), vtwoOnPi2, vmant1, vl);
    vuint16m4_t vr3 = __riscv_vwmaccu_vv_u16m4(__riscv_vwmulu_vv_u16m4(vtwoOnPi4, vmant0, vl), vtwoOnPi3, vmant1, vl);
    vuint16m4_t vr4 = __riscv_vwmaccu_vv_u16m4(__riscv_vwmulu_vv_u16m4(vtwoOnPi5, vmant0, vl), vtwoOnPi4, vmant1, vl);
    vuint16m4_t vr5 = __riscv_vwmaccu_vv_u16m4(__riscv_vwmulu_vv_u16m4(vtwoOnPi6, vmant0, vl), vtwoOnPi5, vmant1, vl);
    
    vuint16m4_t tmp;
    vr4 = __riscv_vadd_vv_u16m4(vr4, __riscv_vsrl_vx_u16m4(vr5, 7, vl), vl);
    tmp =  __riscv_vsrl_vx_u16m4(vr4, 7, vl);
    vr3 = __riscv_vadd_vv_u16m4(vr3, tmp, vl);
    vr4 = __riscv_vsub_vv_u16m4(vr4, __riscv_vsll_vx_u16m4(tmp, 7, vl),vl);
    tmp =  __riscv_vsrl_vx_u16m4(vr3, 7, vl);
    vr2 = __riscv_vadd_vv_u16m4(vr2, tmp, vl);
    vr3 = __riscv_vsub_vv_u16m4(vr3, __riscv_vsll_vx_u16m4(tmp, 7, vl),vl);
    tmp =  __riscv_vsrl_vx_u16m4(vr2, 7, vl);
    vr1 = __riscv_vadd_vv_u16m4(vr1, tmp, vl);
    vr2 = __riscv_vsub_vv_u16m4(vr2, __riscv_vsll_vx_u16m4(tmp, 7, vl),vl);
    tmp =  __riscv_vsrl_vx_u16m4(vr1, 7, vl);
    vr0 = __riscv_vadd_vv_u16m4(vr0, tmp, vl);
    vr1 = __riscv_vsub_vv_u16m4(vr1, __riscv_vsll_vx_u16m4(tmp, 7, vl),vl);
    
    vr0 = __riscv_vor_vv_u16m4(__riscv_vsll_vx_u16m4(vr0, 4, vl), __riscv_vsrl_vx_u16m4(vr1, 3, vl), vl);
    vr1 = __riscv_vor_vv_u16m4(__riscv_vor_vv_u16m4(__riscv_vsll_vx_u16m4(vr1, 13, vl), 
                                                    __riscv_vsll_vx_u16m4(vr2, 6, vl), vl), 
                                                    __riscv_vsrl_vx_u16m4(vr3, 1, vl), vl);
    vr2 = __riscv_vor_vv_u16m4(__riscv_vsll_vx_u16m4(vr3, 15, vl), __riscv_vsll_vx_u16m4(vr4, 8, vl), vl);
    
    vuint16m4_t vfinal_shift        = __riscv_vreinterpret_v_i16m4_u16m4(
                                            __riscv_vsub_vv_i16m4(vshift, __riscv_vmul_vx_i16m4(vshift_pi, 7, vl), vl));
    vuint16m4_t vshift_amount_right = __riscv_vrsub_vx_u16m4(vfinal_shift, 14, vl);
    vuint16m4_t vshift_amount_left  = __riscv_vadd_vx_u16m4(vfinal_shift, 2, vl);
    
    vuint16m4_t vr2_final = __riscv_vor_vv_u16m4(__riscv_vsrl_vv_u16m4(vr2, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u16m4(vr1, vshift_amount_left, vl), vl);
    vuint16m4_t vr1_final = __riscv_vor_vv_u16m4(__riscv_vsrl_vv_u16m4(vr1, vshift_amount_right, vl), 
                                                 __riscv_vsll_vv_u16m4(vr0, vshift_amount_left, vl), vl);
    vuint16m4_t vr0_final = __riscv_vsrl_vv_u16m4(vr0, vshift_amount_right, vl);
    
    vuint16m4_t vr1_final_hi = __riscv_vsrl_vx_u16m4(vr1_final, 8, vl);
    vuint16m4_t vr1_final_lo = __riscv_vand_vx_u16m4(vr1_final, 0xff, vl);
    vuint16m4_t vr2_final_hi = __riscv_vsrl_vx_u16m4(vr2_final, 8, vl);
    vuint16m4_t vr2_final_lo = __riscv_vand_vx_u16m4(vr2_final, 0xff, vl);
    
    vfloat16m4_t vres_r1, vres_r2, vmres_r1, vmres_r2;
    vres_r2 = __riscv_vfcvt_f_xu_v_f16m4(vr1_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f16m4(vr1_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f16m4(vres_r2, 0x1.0p-4f16, vl);
    vres_r1 = __riscv_vfmul_vf_f16m4(vres_r1, 0x1.0p4f16, vl);
    
    mask_pi_4 = __riscv_vmfge_vf_f16m4_b4(vres_r1, 0x1.0p11f16, vl);
    vmres_r2  = __riscv_vfmul_vf_f16m4(vres_r2, -0x1.0p0f16, vl);
    vmres_r1  = __riscv_vfmul_vf_f16m4(vres_r1, -0x1.0p0f16, vl);
    vmres_r1  = __riscv_vfadd_vf_f16m4(vmres_r1, 0x1.0p12f16, vl);
    vres_r1   = __riscv_vmerge_vvm_f16m4(vres_r1, vmres_r1, mask_pi_4, vl);
    vres_r2   = __riscv_vmerge_vvm_f16m4(vres_r2, vmres_r2, mask_pi_4, vl);
    
    vfloat16m4_t vr1_htmp, vr1_tmp;
    vr1_htmp = __riscv_vfadd_vv_f16m4(vres_r1, vres_r2, vl);
    vr1_tmp  = __riscv_vfsub_vv_f16m4(vres_r1, vr1_htmp, vl);
    vr1_tmp  = __riscv_vfadd_vv_f16m4(vr1_tmp, vres_r2, vl);
    
    vres_r2 = __riscv_vfcvt_f_xu_v_f16m4(vr2_final_lo, vl);
    vres_r1 = __riscv_vfcvt_f_xu_v_f16m4(vr2_final_hi, vl);
    vres_r2 = __riscv_vfmul_vf_f16m4(vres_r2, 0x1.0p-20f16, vl);
    vres_r1 = __riscv_vfmul_vf_f16m4(vres_r1, 0x1.0p-12f16, vl);
    
    vmres_r2 = __riscv_vfmul_vf_f16m4(vres_r2, -0x1.0p0f16, vl);
    vmres_r1 = __riscv_vfmul_vf_f16m4(vres_r1, -0x1.0p0f16, vl);
    vres_r2  = __riscv_vmerge_vvm_f16m4(vres_r2, vmres_r2, mask_pi_4, vl);
    vres_r1  = __riscv_vmerge_vvm_f16m4(vres_r1, vmres_r1, mask_pi_4, vl);
    
    vfloat16m4_t vr1_hi, vr2_hi, vr1_lo, vr2_lo;
    vr2_hi = __riscv_vfadd_vv_f16m4(vres_r1, vres_r2, vl);
    vr2_lo = __riscv_vfsub_vv_f16m4(vres_r1, vr2_hi, vl);
    vr2_lo = __riscv_vfadd_vv_f16m4(vr2_lo, vres_r2, vl);
    
    vr1_lo  = __riscv_vfadd_vv_f16m4(vr2_hi, vr1_tmp, vl);
    vr1_tmp = __riscv_vfsub_vv_f16m4(vr1_tmp, vr1_lo,  vl);
    vr1_tmp = __riscv_vfadd_vv_f16m4(vr2_hi, vr1_tmp, vl);
    
    vr1_hi   = __riscv_vfadd_vv_f16m4(vr1_htmp, vr1_lo, vl);
    vr1_htmp = __riscv_vfsub_vv_f16m4(vr1_htmp, vr1_hi, vl);
    vr1_lo = __riscv_vfadd_vv_f16m4(vr1_htmp, vr1_lo, vl);
    vr1_lo = __riscv_vfadd_vv_f16m4(vr1_lo, vr2_lo, vl);
    vr1_lo = __riscv_vfadd_vv_f16m4(vr1_lo, vr1_tmp, vl);
    
    vres_r1 = __riscv_vfmul_vf_f16m4(vr1_hi, PI_DIV_2, vl);
    redx_lo = __riscv_vfmsub_vf_f16m4(vr1_hi, PI_DIV_2, vres_r1, vl); 
    redx_lo = __riscv_vfmacc_vf_f16m4(redx_lo, PI_DIV_2_lo, vr1_hi, vl);   
    redx_lo = __riscv_vfmacc_vf_f16m4(redx_lo, PI_DIV_2, vr1_lo, vl);
    
    redx    = __riscv_vfmul_vf_f16m4(vres_r1, 0x1p-12f16, vl);
    vres_r1 = __riscv_vfmsub_vf_f16m4(redx, 0x1p12f16, vres_r1, vl);
    redx_lo = __riscv_vfsub_vv_f16m4(redx_lo, vres_r1, vl);
    redx_lo = __riscv_vfmul_vf_f16m4(redx_lo, 0x1p-12f16, vl);
    
    vuint16m4_t slow_path_quadrant = __riscv_vand_vx_u16m4(vr0_final, 0x3, vl);

    redx    = __riscv_vmerge_vvm_f16m4(redx, vx, vfast_mask, vl);
    redx_lo = __riscv_vfmerge_vfm_f16m4(redx_lo, 0.0f16, vfast_mask, vl);
    
    mask_pi_4 = __riscv_vmand_mm_b4(mask_pi_4, __riscv_vmnot_m_b4(vfast_mask, vl), vl);
    quadrants = __riscv_vmerge_vxm_u16m4(slow_path_quadrant, 0, vfast_mask, vl);   
}

inline void __riscv_vred_codywaite_f16m4(vfloat16m4_t vx, vfloat16m4_t& redx, vfloat16m4_t& redx_lo, 
                                         vuint16m4_t& quadrants, vbool4_t& mask_pi_4, size_t avl) 
{
    constexpr _Float16 INV_PI_DIV_2 = 0x1.45cp-1f16; 
    constexpr _Float16 PI_2_HI      = 0x1.900p+0f16; 
    constexpr _Float16 PI_2_MID     = 0x1.0c0p-7f16; 
    constexpr _Float16 PI_2_LO      = 0x1.ed4p-14f16;

    size_t vl = __riscv_vsetvl_e16m4(avl);

    vfloat16m4_t vfn  = __riscv_vfmul_vf_f16m4(vx, INV_PI_DIV_2, vl);
    vuint16m4_t  vifn = __riscv_vfcvt_xu_f_v_u16m4(vfn, vl);
    vfn               = __riscv_vfcvt_f_xu_v_f16m4(vifn, vl);

    vfloat16m4_t vr2;
    vfloat16m4_t vr1  = __riscv_vfnmsac_vf_f16m4(vx, PI_2_HI, vfn, vl);
    vfloat16m4_t vrhi = __riscv_vfnmsac_vf_f16m4(vr1, PI_2_MID, vfn, vl);
    vfloat16m4_t vrlo = __riscv_vfnmsac_vf_f16m4(__riscv_vfsub_vv_f16m4(vr1, vrhi, vl), PI_2_MID, vfn, vl);
    vrlo              = __riscv_vfnmsac_vf_f16m4(vrlo, PI_2_LO, vfn, vl);

    mask_pi_4 = __riscv_vmflt_vf_f16m4_b4(vrhi, 0.0f16, vl);

    vuint16m4_t vsign = __riscv_vsrl_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(vrhi), 15, vl);
    quadrants         = __riscv_vand_vx_u16m4(__riscv_vsub_vv_u16m4(vifn, vsign, vl), 3, vl);

    vrlo = __riscv_vfsgnjx_vv_f16m4(vrlo, vrhi, vl);
    vrhi = __riscv_vfabs_v_f16m4(vrhi, vl);

    redx    = __riscv_vfadd_vv_f16m4(vrhi, vrlo, vl);
    redx_lo = __riscv_vfsub_vv_f16m4(vrlo, __riscv_vfsub_vv_f16m4(redx, vrhi, vl), vl);

    vbool4_t m = __riscv_vmfgt_vf_f16m4_b4(redx, 0x1.92p-1f16, vl);
    vrhi        = __riscv_vfrsub_vf_f16m4(redx, 0x1.92p0f16, vl);
    vrlo        = __riscv_vfrsub_vf_f16m4(redx_lo, 0x1.fb4p-12f16, vl);
    vr1         = __riscv_vfadd_vv_f16m4(vrhi, vrlo, vl);
    vr2         = __riscv_vfsub_vv_f16m4(vrlo, __riscv_vfsub_vv_f16m4(vr1, vrhi, vl), vl);
    
    redx      = __riscv_vmerge_vvm_f16m4(redx, vr1, m, vl);
    redx_lo   = __riscv_vmerge_vvm_f16m4(redx_lo, vr2, m, vl);
    mask_pi_4 = __riscv_vmxor_mm_b4(mask_pi_4, m, vl);
}

void __riscv_vreduction_pi4_f16m4(vfloat16m4_t vx, vfloat16m4_t& redx, vfloat16m4_t& redx_lo, 
                                  vuint16m4_t& quadrants, vbool4_t& mask_pi_4, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vx = __riscv_vreinterpret_v_u16m4_f16m4(
              __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(vx), 0x7fff, vl));
    
    mask_pi_4 = __riscv_vmfgt_vf_f16m4_b4(vx, 0x1p4f16, vl);
    if (__riscv_vcpop_m_b4(mask_pi_4, vl))
        __riscv_vred_paynehanek_f16m4(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
    else
        __riscv_vred_codywaite_f16m4(vx, redx, redx_lo, quadrants, mask_pi_4, avl);
}


#endif /* __riscv_zvfh */

#endif /* __riscv_v_intrinsic */

