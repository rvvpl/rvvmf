/*
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of
 * Nizhny Novgorod and its affiliates. All rights reserved.
 *
 * Copyright 2026 The RVVMF Authors (Dmitrii Suvorov,
 * Valentin Volokitin, Elena Panova)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  dtan.cpp                                     *
 *   Contains: intrinsic function tan for float64        *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the tangent of input vector V                *
 *                                                       *
 * Algorithm:                                            *
 *    1) Argument reduction by modified                  *
 *       Payne-Hanek algorithm to [0, pi/4]              *
 *    2) Second reduction using a pre-calculated table   *
 *    3) Calculating tan                                 *
 *    4) Restoring the value of a function using         *
 *       formulas:                                       *
 *       tan(x+y) = (tan(x)+tan(y))/(1-tan(x)*tan(y))    *
 *       tan(x) = 1/tan(pi/2-x)                          *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

#include <cstdint>
#include <cfloat>

#include "trig_reduction.h"


constexpr double FACTOR = 0x1p60;
constexpr double SHIFT_AVOID_UNDERFFLOW = 0x1p-100;
constexpr double SHIFT_REDUCED_RANGE = -0x1p-40;

constexpr double TAND_TABLE_SIZE = 0x1p4;
constexpr double TAND_INV_TABLE_SIZE = 0x1p-4;

double TAND_TABLE_hi[16] = {
    0.0, 0x1.927278a3b1162p-5, 0x1.936bb8c5b2da2p-4, 0x1.2fcac73a6064p-3,
    0x1.975f5e0553158p-3, 0x1.007fa758626aep-2, 0x1.36a08355c63dcp-2, 0x1.6e649f7d78649p-2,
    0x1.a827999fcef32p-2, 0x1.e450e0d273e7ap-2, 0x1.11ab7190834ecp-1, 0x1.32e1889047ffdp-1,
    0x1.561b82ab7f99p-1, 0x1.7bb99ed2990cfp-1, 0x1.a43002ae4285p-1, 0x1.d00cbc7384d2ep-1
};
double TAND_TABLE_lo[16] = {
    0.0, 0x1.ada13ceebab9dp-64, 0x1.87d054f689d7ap-58, 0x1.2cfb5a746f62cp-58,
    0x1.ef5d367441946p-61, 0x1.5a6d6c3c8b6a7p-57, -0x1.d704d1bfdb6e8p-57, 0x1.78e945dc3913cp-57,
    0x1.08b2fb1366ea9p-56, 0x1.41522f15e53c5p-56, -0x1.e564bcd1e635dp-55, -0x1.10b4421e6a4f8p-55,
    0x1.7a8c52172b675p-55, -0x1.aa7538e44e996p-55, -0x1.a6db80fe796fep-56, 0x1.8dcad85e60fbep-56
};

constexpr double PI_DIV_4_hi = 0x1.921fb54442d18p-1;
constexpr double PI_DIV_4_lo = 0x1.1a62633145c07p-55;
constexpr double INV_PI_DIV_4 = 0x1.45f306dc9c883p0;

constexpr double REDUCTION_COEFF = INV_PI_DIV_4*TAND_TABLE_SIZE;
constexpr double INV_REDUCTION_COEFF_hi = PI_DIV_4_hi*TAND_INV_TABLE_SIZE;
constexpr double INV_REDUCTION_COEFF_lo = PI_DIV_4_lo*TAND_INV_TABLE_SIZE;

constexpr double POL_COEFF_9  = 0x1.672f94c9a7411p-6*FACTOR;
constexpr double POL_COEFF_7  = 0x1.ba1b6529f88dcp-5*FACTOR;
constexpr double POL_COEFF_5  = 0x1.11111114672d7p-3*FACTOR;
constexpr double POL_COEFF_3  = 0x1.5555555555346p-2*FACTOR;


vfloat64m1_t __riscv_vtan_kernel_f64m1(vfloat64m1_t vxh, vfloat64m1_t vxl, vbool64_t mask_div, size_t vl)
{
    vfloat64m1_t vzh, vzl, vyh, vyl, vx2h, vx2l, vr;
      
    vfloat64m1_t vfindex = __riscv_vfmadd_vf_f64m1(vxh, REDUCTION_COEFF,
        __riscv_vfmv_v_f_f64m1(SHIFT_REDUCED_RANGE, vl), vl);
    vuint64m1_t	index = __riscv_vfcvt_rtz_xu_f_v_u64m1(vfindex, vl);
    
    vfindex = __riscv_vfcvt_f_xu_v_f64m1(index, vl);
    
    index = __riscv_vsll_vx_u64m1(index, 3, vl);
    
    vfloat64m1_t vch = __riscv_vloxei64_v_f64m1(TAND_TABLE_hi, index, vl);
    vfloat64m1_t vcl = __riscv_vloxei64_v_f64m1(TAND_TABLE_lo, index, vl); 
    
    vzh = __riscv_vfnmsub_vf_f64m1(vfindex, INV_REDUCTION_COEFF_hi, vxh, vl);
    vzl = __riscv_vfnmsub_vf_f64m1(vfindex, INV_REDUCTION_COEFF_lo, vxl, vl);
    vxh = __riscv_vfadd_vv_f64m1(vzh, vzl, vl);
    vxl = __riscv_vfsub_vv_f64m1(vzl, __riscv_vfsub_vv_f64m1(vxh, vzh, vl), vl);
    
    vx2h = __riscv_vfmadd_vv_f64m1(vxh, vxh, __riscv_vfmv_v_f_f64m1(SHIFT_AVOID_UNDERFFLOW, vl), vl);
    vx2l = __riscv_vfmsub_vv_f64m1(vxh, vxh, vx2h, vl);
    vx2l = __riscv_vfmacc_vv_f64m1(vx2l, vxh, __riscv_vfmul_vf_f64m1(vxl, 2.0, vl), vl);
    
    vr = __riscv_vfmadd_vf_f64m1(vx2h, POL_COEFF_9, __riscv_vfmv_v_f_f64m1(POL_COEFF_7, vl), vl);
    vr = __riscv_vfmadd_vv_f64m1(vr, vx2h, __riscv_vfmv_v_f_f64m1(POL_COEFF_5, vl), vl);
    vr = __riscv_vfmadd_vv_f64m1(vr, vx2h, __riscv_vfmv_v_f_f64m1(POL_COEFF_3, vl), vl);
    vzh = __riscv_vfmadd_vv_f64m1(vr, vx2h, __riscv_vfmv_v_f_f64m1(FACTOR, vl), vl);
    vzl = __riscv_vfmadd_vv_f64m1(vr, vx2h, __riscv_vfrsub_vf_f64m1(vzh, FACTOR, vl), vl);
    vzl = __riscv_vfmacc_vv_f64m1(vzl, vr, vx2l, vl); 
    
    vfloat64m1_t vrh, vrl, vth, vtl;
    vrh = __riscv_vfmul_vv_f64m1(vxh, vzh, vl); 
    vrl = __riscv_vfmsub_vv_f64m1(vxh, vzh, vrh, vl);
    vrl = __riscv_vfmacc_vv_f64m1(vrl, vzh, vxl, vl);
    vrl = __riscv_vfmacc_vv_f64m1(vrl, vzl, vxh, vl);
    
    vfloat64m1_t vnumh, vnuml, vdenh, vdenl, vd;
    vzh = __riscv_vfmadd_vf_f64m1(vch, FACTOR, vrh, vl);
    vzl = __riscv_vfsub_vv_f64m1(vrh, __riscv_vfnmsub_vf_f64m1(vch, FACTOR, vzh, vl), vl);
    vzl = __riscv_vfmadd_vf_f64m1(vcl, FACTOR, vzl, vl);
    vzl = __riscv_vfadd_vv_f64m1(vzl, vrl, vl);
    
    vth = __riscv_vfmv_v_f_f64m1(FACTOR, vl);
    vth = __riscv_vfnmsac_vv_f64m1(vth, vrh, vch, vl);
    vtl = __riscv_vfrsub_vf_f64m1(vth, FACTOR, vl);
    vtl = __riscv_vfnmsac_vv_f64m1(vtl, vrh, vch, vl);
    vtl = __riscv_vfnmsac_vv_f64m1(vtl, vrl, vch, vl);
    vtl = __riscv_vfnmsac_vv_f64m1(vtl, vrh, vcl, vl);
    
    vnumh = __riscv_vmerge_vvm_f64m1(vzh, vth, mask_div, vl);
    vnuml = __riscv_vmerge_vvm_f64m1(vzl, vtl, mask_div, vl);
    vdenh = __riscv_vmerge_vvm_f64m1(vth, vzh, mask_div, vl);
    vdenl = __riscv_vmerge_vvm_f64m1(vtl, vzl, mask_div, vl);
    
    vd = __riscv_vfrdiv_vf_f64m1(vdenh, 1.0, vl);
    vrh = __riscv_vfmul_vv_f64m1(vnumh, vd, vl);
    vr = __riscv_vfnmsac_vv_f64m1(vnumh, vrh, vdenh, vl);
    vrl = __riscv_vfnmsac_vv_f64m1(vnuml, vrh, vdenl, vl);
    
    vr = __riscv_vfadd_vv_f64m1(vr, vrl, vl);
    vr = __riscv_vfmadd_vv_f64m1(vr, vd, vrh, vl);
    return vr;
}

vfloat64m1_t __riscv_vtan_f64m1(vfloat64m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m1(avl);

    vfloat64m1_t redx_hi, redx_lo, vtan, vntan;
    vbool64_t mask_pi_4;
    vuint64m1_t quadrants;

    __riscv_vred_paynehanek_f64m1(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);

    vbool64_t mask_div = __riscv_vmxor_mm_b64(mask_pi_4,
        __riscv_vmseq_vx_u64m1_b64(__riscv_vand_vx_u64m1(quadrants, 1, vl), 1, vl), vl);

    vtan = __riscv_vtan_kernel_f64m1(redx_hi, redx_lo,  mask_div, vl);
    vntan = __riscv_vfmul_vf_f64m1(vtan, -1.0, vl);

    vuint64m1_t signx = __riscv_vand_vx_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(x),
        0x8000000000000000, vl);
    vbool64_t mask = __riscv_vmsne_vx_u64m1_b64(signx, 0, vl);
    quadrants = __riscv_vrsub_vx_u64m1_mu(mask, quadrants, quadrants, 3, vl);
    quadrants = __riscv_vand_vx_u64m1(quadrants, 1, vl);
    mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 1, vl);
    vtan = __riscv_vmerge_vvm_f64m1(vtan, vntan, mask, vl);

#ifndef __FAST_MATH__
    vuint64m1_t ix = __riscv_vand_vx_u64m1(
                __riscv_vreinterpret_v_f64m1_u64m1(x), 0x7fffffffffffffff, vl);
    vbool64_t mask_NaN = __riscv_vmsgeu_vx_u64m1_b64(ix, 0x7ff0000000000000, vl);
    vtan = __riscv_vmerge_vvm_f64m1(vtan, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b64(mask_NaN,
        __riscv_vmsltu_vx_u64m1_b64(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b64(mask_NaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        vtan = __riscv_vfmerge_vfm_f64m1(vtan, x1, mask_NaN, vl);
    }
    volatile vfloat64m1_t tmp = __riscv_vfmadd_vf_f64m1(vtan, 0x1p-60, vtan, vl);
#endif

    return vtan;
}


vfloat64m2_t __riscv_vtan_kernel_f64m2(vfloat64m2_t vxh, vfloat64m2_t vxl, vbool32_t mask_div, size_t vl)
{
    vfloat64m2_t vzh, vzl, vyh, vyl, vx2h, vx2l, vr;
      
    vfloat64m2_t vfindex = __riscv_vfmadd_vf_f64m2(vxh, REDUCTION_COEFF,
        __riscv_vfmv_v_f_f64m2(SHIFT_REDUCED_RANGE, vl), vl);
    vuint64m2_t	index = __riscv_vfcvt_rtz_xu_f_v_u64m2(vfindex, vl);
    
    vfindex = __riscv_vfcvt_f_xu_v_f64m2(index, vl);
    
    index = __riscv_vsll_vx_u64m2(index, 3, vl);
    
    vfloat64m2_t vch = __riscv_vloxei64_v_f64m2(TAND_TABLE_hi, index, vl);
    vfloat64m2_t vcl = __riscv_vloxei64_v_f64m2(TAND_TABLE_lo, index, vl); 
    
    vzh = __riscv_vfnmsub_vf_f64m2(vfindex, INV_REDUCTION_COEFF_hi, vxh, vl);
    vzl = __riscv_vfnmsub_vf_f64m2(vfindex, INV_REDUCTION_COEFF_lo, vxl, vl);
    vxh = __riscv_vfadd_vv_f64m2(vzh, vzl, vl);
    vxl = __riscv_vfsub_vv_f64m2(vzl, __riscv_vfsub_vv_f64m2(vxh, vzh, vl), vl);
    
    vx2h = __riscv_vfmadd_vv_f64m2(vxh, vxh, __riscv_vfmv_v_f_f64m2(SHIFT_AVOID_UNDERFFLOW, vl), vl);
    vx2l = __riscv_vfmsub_vv_f64m2(vxh, vxh, vx2h, vl);
    vx2l = __riscv_vfmacc_vv_f64m2(vx2l, vxh, __riscv_vfmul_vf_f64m2(vxl, 2.0, vl), vl);
    
    vr = __riscv_vfmadd_vf_f64m2(vx2h, POL_COEFF_9, __riscv_vfmv_v_f_f64m2(POL_COEFF_7, vl), vl);
    vr = __riscv_vfmadd_vv_f64m2(vr, vx2h, __riscv_vfmv_v_f_f64m2(POL_COEFF_5, vl), vl);
    vr = __riscv_vfmadd_vv_f64m2(vr, vx2h, __riscv_vfmv_v_f_f64m2(POL_COEFF_3, vl), vl);
    vzh = __riscv_vfmadd_vv_f64m2(vr, vx2h, __riscv_vfmv_v_f_f64m2(FACTOR, vl), vl);
    vzl = __riscv_vfmadd_vv_f64m2(vr, vx2h, __riscv_vfrsub_vf_f64m2(vzh, FACTOR, vl), vl);
    vzl = __riscv_vfmacc_vv_f64m2(vzl, vr, vx2l, vl); 
    
    vfloat64m2_t vrh, vrl, vth, vtl;
    vrh = __riscv_vfmul_vv_f64m2(vxh, vzh, vl); 
    vrl = __riscv_vfmsub_vv_f64m2(vxh, vzh, vrh, vl);
    vrl = __riscv_vfmacc_vv_f64m2(vrl, vzh, vxl, vl);
    vrl = __riscv_vfmacc_vv_f64m2(vrl, vzl, vxh, vl);
    
    vfloat64m2_t vnumh, vnuml, vdenh, vdenl, vd;
    vzh = __riscv_vfmadd_vf_f64m2(vch, FACTOR, vrh, vl);
    vzl = __riscv_vfsub_vv_f64m2(vrh, __riscv_vfnmsub_vf_f64m2(vch, FACTOR, vzh, vl), vl);
    vzl = __riscv_vfmadd_vf_f64m2(vcl, FACTOR, vzl, vl);
    vzl = __riscv_vfadd_vv_f64m2(vzl, vrl, vl);
    
    vth = __riscv_vfmv_v_f_f64m2(FACTOR, vl);
    vth = __riscv_vfnmsac_vv_f64m2(vth, vrh, vch, vl);
    vtl = __riscv_vfrsub_vf_f64m2(vth, FACTOR, vl);
    vtl = __riscv_vfnmsac_vv_f64m2(vtl, vrh, vch, vl);
    vtl = __riscv_vfnmsac_vv_f64m2(vtl, vrl, vch, vl);
    vtl = __riscv_vfnmsac_vv_f64m2(vtl, vrh, vcl, vl);
    
    vnumh = __riscv_vmerge_vvm_f64m2(vzh, vth, mask_div, vl);
    vnuml = __riscv_vmerge_vvm_f64m2(vzl, vtl, mask_div, vl);
    vdenh = __riscv_vmerge_vvm_f64m2(vth, vzh, mask_div, vl);
    vdenl = __riscv_vmerge_vvm_f64m2(vtl, vzl, mask_div, vl);
    
    vd = __riscv_vfrdiv_vf_f64m2(vdenh, 1.0, vl);
    vrh = __riscv_vfmul_vv_f64m2(vnumh, vd, vl);
    vr = __riscv_vfnmsac_vv_f64m2(vnumh, vrh, vdenh, vl);
    vrl = __riscv_vfnmsac_vv_f64m2(vnuml, vrh, vdenl, vl);
    
    vr = __riscv_vfadd_vv_f64m2(vr, vrl, vl);
    vr = __riscv_vfmadd_vv_f64m2(vr, vd, vrh, vl);
    return vr;
}

vfloat64m2_t __riscv_vtan_f64m2(vfloat64m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m2(avl);

    vfloat64m2_t redx_hi, redx_lo, vtan, vntan;
    vbool32_t mask_pi_4;
    vuint64m2_t quadrants;

    __riscv_vred_paynehanek_f64m2(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);

    vbool32_t mask_div = __riscv_vmxor_mm_b32(mask_pi_4,
        __riscv_vmseq_vx_u64m2_b32(__riscv_vand_vx_u64m2(quadrants, 1, vl), 1, vl), vl);

    vtan = __riscv_vtan_kernel_f64m2(redx_hi, redx_lo,  mask_div, vl);
    vntan = __riscv_vfmul_vf_f64m2(vtan, -1.0, vl);

    vuint64m2_t signx = __riscv_vand_vx_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(x),
        0x8000000000000000, vl);
    vbool32_t mask = __riscv_vmsne_vx_u64m2_b32(signx, 0, vl);
    quadrants = __riscv_vrsub_vx_u64m2_mu(mask, quadrants, quadrants, 3, vl);
    quadrants = __riscv_vand_vx_u64m2(quadrants, 1, vl);
    mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 1, vl);
    vtan = __riscv_vmerge_vvm_f64m2(vtan, vntan, mask, vl);

#ifndef __FAST_MATH__
    vuint64m2_t ix = __riscv_vand_vx_u64m2(
                __riscv_vreinterpret_v_f64m2_u64m2(x), 0x7fffffffffffffff, vl);
    vbool32_t mask_NaN = __riscv_vmsgeu_vx_u64m2_b32(ix, 0x7ff0000000000000, vl);
    vtan = __riscv_vmerge_vvm_f64m2(vtan, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b32(mask_NaN,
        __riscv_vmsltu_vx_u64m2_b32(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b32(mask_NaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        vtan = __riscv_vfmerge_vfm_f64m2(vtan, x1, mask_NaN, vl);
    }
    volatile vfloat64m2_t tmp = __riscv_vfmadd_vf_f64m2(vtan, 0x1p-60, vtan, vl);
#endif

    return vtan;
}


vfloat64m4_t __riscv_vtan_kernel_f64m4(vfloat64m4_t vxh, vfloat64m4_t vxl, vbool16_t mask_div, size_t vl)
{
    vfloat64m4_t vzh, vzl, vyh, vyl, vx2h, vx2l, vr;
      
    vfloat64m4_t vfindex = __riscv_vfmadd_vf_f64m4(vxh, REDUCTION_COEFF,
        __riscv_vfmv_v_f_f64m4(SHIFT_REDUCED_RANGE, vl), vl);
    vuint64m4_t	index = __riscv_vfcvt_rtz_xu_f_v_u64m4(vfindex, vl);
    
    vfindex = __riscv_vfcvt_f_xu_v_f64m4(index, vl);
    
    index = __riscv_vsll_vx_u64m4(index, 3, vl);
    
    vfloat64m4_t vch = __riscv_vloxei64_v_f64m4(TAND_TABLE_hi, index, vl);
    vfloat64m4_t vcl = __riscv_vloxei64_v_f64m4(TAND_TABLE_lo, index, vl); 
    
    vzh = __riscv_vfnmsub_vf_f64m4(vfindex, INV_REDUCTION_COEFF_hi, vxh, vl);
    vzl = __riscv_vfnmsub_vf_f64m4(vfindex, INV_REDUCTION_COEFF_lo, vxl, vl);
    vxh = __riscv_vfadd_vv_f64m4(vzh, vzl, vl);
    vxl = __riscv_vfsub_vv_f64m4(vzl, __riscv_vfsub_vv_f64m4(vxh, vzh, vl), vl);
    
    vx2h = __riscv_vfmadd_vv_f64m4(vxh, vxh, __riscv_vfmv_v_f_f64m4(SHIFT_AVOID_UNDERFFLOW, vl), vl);
    vx2l = __riscv_vfmsub_vv_f64m4(vxh, vxh, vx2h, vl);
    vx2l = __riscv_vfmacc_vv_f64m4(vx2l, vxh, __riscv_vfmul_vf_f64m4(vxl, 2.0, vl), vl);
    
    vr = __riscv_vfmadd_vf_f64m4(vx2h, POL_COEFF_9, __riscv_vfmv_v_f_f64m4(POL_COEFF_7, vl), vl);
    vr = __riscv_vfmadd_vv_f64m4(vr, vx2h, __riscv_vfmv_v_f_f64m4(POL_COEFF_5, vl), vl);
    vr = __riscv_vfmadd_vv_f64m4(vr, vx2h, __riscv_vfmv_v_f_f64m4(POL_COEFF_3, vl), vl);
    vzh = __riscv_vfmadd_vv_f64m4(vr, vx2h, __riscv_vfmv_v_f_f64m4(FACTOR, vl), vl);
    vzl = __riscv_vfmadd_vv_f64m4(vr, vx2h, __riscv_vfrsub_vf_f64m4(vzh, FACTOR, vl), vl);
    vzl = __riscv_vfmacc_vv_f64m4(vzl, vr, vx2l, vl); 
    
    vfloat64m4_t vrh, vrl, vth, vtl;
    vrh = __riscv_vfmul_vv_f64m4(vxh, vzh, vl); 
    vrl = __riscv_vfmsub_vv_f64m4(vxh, vzh, vrh, vl);
    vrl = __riscv_vfmacc_vv_f64m4(vrl, vzh, vxl, vl);
    vrl = __riscv_vfmacc_vv_f64m4(vrl, vzl, vxh, vl);
    
    vfloat64m4_t vnumh, vnuml, vdenh, vdenl, vd;
    vzh = __riscv_vfmadd_vf_f64m4(vch, FACTOR, vrh, vl);
    vzl = __riscv_vfsub_vv_f64m4(vrh, __riscv_vfnmsub_vf_f64m4(vch, FACTOR, vzh, vl), vl);
    vzl = __riscv_vfmadd_vf_f64m4(vcl, FACTOR, vzl, vl);
    vzl = __riscv_vfadd_vv_f64m4(vzl, vrl, vl);
    
    vth = __riscv_vfmv_v_f_f64m4(FACTOR, vl);
    vth = __riscv_vfnmsac_vv_f64m4(vth, vrh, vch, vl);
    vtl = __riscv_vfrsub_vf_f64m4(vth, FACTOR, vl);
    vtl = __riscv_vfnmsac_vv_f64m4(vtl, vrh, vch, vl);
    vtl = __riscv_vfnmsac_vv_f64m4(vtl, vrl, vch, vl);
    vtl = __riscv_vfnmsac_vv_f64m4(vtl, vrh, vcl, vl);
    
    vnumh = __riscv_vmerge_vvm_f64m4(vzh, vth, mask_div, vl);
    vnuml = __riscv_vmerge_vvm_f64m4(vzl, vtl, mask_div, vl);
    vdenh = __riscv_vmerge_vvm_f64m4(vth, vzh, mask_div, vl);
    vdenl = __riscv_vmerge_vvm_f64m4(vtl, vzl, mask_div, vl);
    
    vd = __riscv_vfrdiv_vf_f64m4(vdenh, 1.0, vl);
    vrh = __riscv_vfmul_vv_f64m4(vnumh, vd, vl);
    vr = __riscv_vfnmsac_vv_f64m4(vnumh, vrh, vdenh, vl);
    vrl = __riscv_vfnmsac_vv_f64m4(vnuml, vrh, vdenl, vl);
    
    vr = __riscv_vfadd_vv_f64m4(vr, vrl, vl);
    vr = __riscv_vfmadd_vv_f64m4(vr, vd, vrh, vl);
    return vr;
}

vfloat64m4_t __riscv_vtan_f64m4(vfloat64m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m4(avl);

    vfloat64m4_t redx_hi, redx_lo, vtan, vntan;
    vbool16_t mask_pi_4;
    vuint64m4_t quadrants;

    __riscv_vred_paynehanek_f64m4(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);

    vbool16_t mask_div = __riscv_vmxor_mm_b16(mask_pi_4,
        __riscv_vmseq_vx_u64m4_b16(__riscv_vand_vx_u64m4(quadrants, 1, vl), 1, vl), vl);

    vtan = __riscv_vtan_kernel_f64m4(redx_hi, redx_lo,  mask_div, vl);
    vntan = __riscv_vfmul_vf_f64m4(vtan, -1.0, vl);

    vuint64m4_t signx = __riscv_vand_vx_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(x),
        0x8000000000000000, vl);
    vbool16_t mask = __riscv_vmsne_vx_u64m4_b16(signx, 0, vl);
    quadrants = __riscv_vrsub_vx_u64m4_mu(mask, quadrants, quadrants, 3, vl);
    quadrants = __riscv_vand_vx_u64m4(quadrants, 1, vl);
    mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 1, vl);
    vtan = __riscv_vmerge_vvm_f64m4(vtan, vntan, mask, vl);

#ifndef __FAST_MATH__
    vuint64m4_t ix = __riscv_vand_vx_u64m4(
                __riscv_vreinterpret_v_f64m4_u64m4(x), 0x7fffffffffffffff, vl);
    vbool16_t mask_NaN = __riscv_vmsgeu_vx_u64m4_b16(ix, 0x7ff0000000000000, vl);
    vtan = __riscv_vmerge_vvm_f64m4(vtan, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b16(mask_NaN,
        __riscv_vmsltu_vx_u64m4_b16(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_NaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        vtan = __riscv_vfmerge_vfm_f64m4(vtan, x1, mask_NaN, vl);
    }
    volatile vfloat64m4_t tmp = __riscv_vfmadd_vf_f64m4(vtan, 0x1p-60, vtan, vl);
#endif

    return vtan;
}


vfloat64m8_t __riscv_vtan_f64m8(vfloat64m8_t x, size_t avl)
{
    vfloat64m8_t res;
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1 = __riscv_vtan_f64m4(x1, vl);
    res = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e64m4(avl-vl);
        x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1 = __riscv_vtan_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}


#endif /* __riscv_v_intrinsic */
