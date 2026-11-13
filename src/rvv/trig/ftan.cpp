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
 *   File:  ftan.cpp                                     *
 *   Contains: intrinsic function tan for float32        *
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


constexpr float FACTOR = 0x1p40f;
constexpr float SHIFT_AVOID_UNDERFFLOW = 0x1p-60f;
constexpr float SHIFT_REDUCED_RANGE = -0x1p-20f;

constexpr float TANF_TABLE_SIZE = 0x1p3f;
constexpr float TANF_INV_TABLE_SIZE = 0x1p-3f;

float TANF_TABLE_hi[8] = {
    0.0f, 0x1.936bb8p-4f, 0x1.975f5ep-3f, 0x1.36a084p-2f,
    0x1.a8279ap-2f, 0x1.11ab72p-1f, 0x1.561b82p-1f, 0x1.a43002p-1f
};
float TANF_TABLE_lo[8] = {
    0.0f, 0x1.8b65b4p-29f, 0x1.54c56p-33f, -0x1.547384p-27f,
    -0x1.80c434p-28f, -0x1.bdf2c6p-27f, 0x1.56ff32p-26f, 0x1.5c850ap-26f
};

constexpr float PI_DIV_4_hi = 0x1.921fb6p-1f;
constexpr float PI_DIV_4_lo = -0x1.7778c6p-26f;
constexpr float INV_PI_DIV_4 = 0x1.45f306p0f;

constexpr float REDUCTION_COEFF = INV_PI_DIV_4*TANF_TABLE_SIZE;
constexpr float INV_REDUCTION_COEFF_hi = PI_DIV_4_hi*TANF_INV_TABLE_SIZE;
constexpr float INV_REDUCTION_COEFF_lo = PI_DIV_4_lo*TANF_INV_TABLE_SIZE;

constexpr float POL_COEFF_7  = 0x1.bd9488p-5f*FACTOR;
constexpr float POL_COEFF_5  = 0x1.110fbap-3f*FACTOR;
constexpr float POL_COEFF_3  = 0x1.555556p-2f*FACTOR;


vfloat32m1_t __riscv_vtan_kernel_f32m1(vfloat32m1_t vxh, vfloat32m1_t vxl, vbool32_t mask_div, size_t vl)
{
    vfloat32m1_t vzh, vzl, vyh, vyl, vx2h, vx2l, vr;
      
    vfloat32m1_t vfindex = __riscv_vfmadd_vf_f32m1(vxh, REDUCTION_COEFF,
        __riscv_vfmv_v_f_f32m1(SHIFT_REDUCED_RANGE, vl), vl);
    vuint32m1_t	index = __riscv_vfcvt_rtz_xu_f_v_u32m1(vfindex, vl);
    
    vfindex = __riscv_vfcvt_f_xu_v_f32m1(index, vl);
    
    index = __riscv_vsll_vx_u32m1(index, 2, vl);
    
    vfloat32m1_t vch = __riscv_vloxei32_v_f32m1(TANF_TABLE_hi, index, vl);
    vfloat32m1_t vcl = __riscv_vloxei32_v_f32m1(TANF_TABLE_lo, index, vl); 
    
    vzh = __riscv_vfnmsub_vf_f32m1(vfindex, INV_REDUCTION_COEFF_hi, vxh, vl);
    vzl = __riscv_vfnmsub_vf_f32m1(vfindex, INV_REDUCTION_COEFF_lo, vxl, vl);
    vxh = __riscv_vfadd_vv_f32m1(vzh, vzl, vl);
    vxl = __riscv_vfsub_vv_f32m1(vzl, __riscv_vfsub_vv_f32m1(vxh, vzh, vl), vl);
    
    vx2h = __riscv_vfmadd_vv_f32m1(vxh, vxh, __riscv_vfmv_v_f_f32m1(SHIFT_AVOID_UNDERFFLOW, vl), vl);
    vx2l = __riscv_vfmsub_vv_f32m1(vxh, vxh, vx2h, vl);
    vx2l = __riscv_vfmacc_vv_f32m1(vx2l, vxh, __riscv_vfmul_vf_f32m1(vxl, 2.0f, vl), vl);
    
    vr = __riscv_vfmadd_vf_f32m1(vx2h, POL_COEFF_7, __riscv_vfmv_v_f_f32m1(POL_COEFF_5, vl), vl);
    vr = __riscv_vfmadd_vv_f32m1(vr, vx2h, __riscv_vfmv_v_f_f32m1(POL_COEFF_3, vl), vl);
    vzh = __riscv_vfmadd_vv_f32m1(vr, vx2h, __riscv_vfmv_v_f_f32m1(FACTOR, vl), vl);
    vzl = __riscv_vfmadd_vv_f32m1(vr, vx2h, __riscv_vfrsub_vf_f32m1(vzh, FACTOR, vl), vl);
    vzl = __riscv_vfmacc_vv_f32m1(vzl, vr, vx2l, vl); 
    
    vfloat32m1_t vrh, vrl, vth, vtl;
    vrh = __riscv_vfmul_vv_f32m1(vxh, vzh, vl); 
    vrl = __riscv_vfmsub_vv_f32m1(vxh, vzh, vrh, vl);
    vrl = __riscv_vfmacc_vv_f32m1(vrl, vzh, vxl, vl);
    vrl = __riscv_vfmacc_vv_f32m1(vrl, vzl, vxh, vl);
    
    vfloat32m1_t vnumh, vnuml, vdenh, vdenl, vd;
    vzh = __riscv_vfmadd_vf_f32m1(vch, FACTOR, vrh, vl);
    vzl = __riscv_vfsub_vv_f32m1(vrh, __riscv_vfnmsub_vf_f32m1(vch, FACTOR, vzh, vl), vl);
    vzl = __riscv_vfmadd_vf_f32m1(vcl, FACTOR, vzl, vl);
    vzl = __riscv_vfadd_vv_f32m1(vzl, vrl, vl);
    
    vth = __riscv_vfmv_v_f_f32m1(FACTOR, vl);
    vth = __riscv_vfnmsac_vv_f32m1(vth, vrh, vch, vl);
    vtl = __riscv_vfrsub_vf_f32m1(vth, FACTOR, vl);
    vtl = __riscv_vfnmsac_vv_f32m1(vtl, vrh, vch, vl);
    vtl = __riscv_vfnmsac_vv_f32m1(vtl, vrl, vch, vl);
    vtl = __riscv_vfnmsac_vv_f32m1(vtl, vrh, vcl, vl);
    
    vnumh = __riscv_vmerge_vvm_f32m1(vzh, vth, mask_div, vl);
    vnuml = __riscv_vmerge_vvm_f32m1(vzl, vtl, mask_div, vl);
    vdenh = __riscv_vmerge_vvm_f32m1(vth, vzh, mask_div, vl);
    vdenl = __riscv_vmerge_vvm_f32m1(vtl, vzl, mask_div, vl);
    
    vd = __riscv_vfrdiv_vf_f32m1(vdenh, 1.0f, vl);
    vrh = __riscv_vfmul_vv_f32m1(vnumh, vd, vl);
    vr = __riscv_vfnmsac_vv_f32m1(vnumh, vrh, vdenh, vl);
    vrl = __riscv_vfnmsac_vv_f32m1(vnuml, vrh, vdenl, vl);
    
    vr = __riscv_vfadd_vv_f32m1(vr, vrl, vl);
    vr = __riscv_vfmadd_vv_f32m1(vr, vd, vrh, vl);
    return vr;
}

vfloat32m1_t __riscv_vtan_f32m1(vfloat32m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m1(avl);

    vfloat32m1_t redx_hi, redx_lo, vtan, vntan;
    vbool32_t mask_pi_4;
    vuint32m1_t quadrants;

    __riscv_vred_paynehanek_f32m1(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);

    vbool32_t mask_div = __riscv_vmxor_mm_b32(mask_pi_4,
        __riscv_vmseq_vx_u32m1_b32(__riscv_vand_vx_u32m1(quadrants, 1, vl), 1, vl), vl);

    vtan = __riscv_vtan_kernel_f32m1(redx_hi, redx_lo,  mask_div, vl);
    vntan = __riscv_vfmul_vf_f32m1(vtan, -1.0f, vl);

    vuint32m1_t signx = __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(x), 0x80000000, vl);
    vbool32_t mask = __riscv_vmsne_vx_u32m1_b32(signx, 0, vl);
    quadrants = __riscv_vrsub_vx_u32m1_mu(mask, quadrants, quadrants, 3, vl);
    quadrants = __riscv_vand_vx_u32m1(quadrants, 1, vl);
    mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 1, vl);
    vtan = __riscv_vmerge_vvm_f32m1(vtan, vntan, mask, vl);

#ifndef __FAST_MATH__
    vuint32m1_t ix = __riscv_vand_vx_u32m1(
                __riscv_vreinterpret_v_f32m1_u32m1(x), 0x7fffffff, vl);
    vbool32_t mask_NaN = __riscv_vmsgeu_vx_u32m1_b32(ix, 0x7f800000, vl);
    vtan = __riscv_vmerge_vvm_f32m1(vtan, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b32(mask_NaN, __riscv_vmsltu_vx_u32m1_b32(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b32(mask_NaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        vtan = __riscv_vfmerge_vfm_f32m1(vtan, x1, mask_NaN, vl);
    }
    volatile vfloat32m1_t tmp = __riscv_vfmadd_vf_f32m1(vtan, 0x1p-26f, vtan, vl);
#endif

    return vtan;
}


vfloat32m2_t __riscv_vtan_kernel_f32m2(vfloat32m2_t vxh, vfloat32m2_t vxl, vbool16_t mask_div, size_t vl)
{
    vfloat32m2_t vzh, vzl, vyh, vyl, vx2h, vx2l, vr;
      
    vfloat32m2_t vfindex = __riscv_vfmadd_vf_f32m2(vxh, REDUCTION_COEFF,
        __riscv_vfmv_v_f_f32m2(SHIFT_REDUCED_RANGE, vl), vl);
    vuint32m2_t	index = __riscv_vfcvt_rtz_xu_f_v_u32m2(vfindex, vl);
    
    vfindex = __riscv_vfcvt_f_xu_v_f32m2(index, vl);
    
    index = __riscv_vsll_vx_u32m2(index, 2, vl);
    
    vfloat32m2_t vch = __riscv_vloxei32_v_f32m2(TANF_TABLE_hi, index, vl);
    vfloat32m2_t vcl = __riscv_vloxei32_v_f32m2(TANF_TABLE_lo, index, vl); 
    
    vzh = __riscv_vfnmsub_vf_f32m2(vfindex, INV_REDUCTION_COEFF_hi, vxh, vl);
    vzl = __riscv_vfnmsub_vf_f32m2(vfindex, INV_REDUCTION_COEFF_lo, vxl, vl);
    vxh = __riscv_vfadd_vv_f32m2(vzh, vzl, vl);
    vxl = __riscv_vfsub_vv_f32m2(vzl, __riscv_vfsub_vv_f32m2(vxh, vzh, vl), vl);
    
    vx2h = __riscv_vfmadd_vv_f32m2(vxh, vxh, __riscv_vfmv_v_f_f32m2(SHIFT_AVOID_UNDERFFLOW, vl), vl);
    vx2l = __riscv_vfmsub_vv_f32m2(vxh, vxh, vx2h, vl);
    vx2l = __riscv_vfmacc_vv_f32m2(vx2l, vxh, __riscv_vfmul_vf_f32m2(vxl, 2.0f, vl), vl);
    
    vr = __riscv_vfmadd_vf_f32m2(vx2h, POL_COEFF_7, __riscv_vfmv_v_f_f32m2(POL_COEFF_5, vl), vl);
    vr = __riscv_vfmadd_vv_f32m2(vr, vx2h, __riscv_vfmv_v_f_f32m2(POL_COEFF_3, vl), vl);
    vzh = __riscv_vfmadd_vv_f32m2(vr, vx2h, __riscv_vfmv_v_f_f32m2(FACTOR, vl), vl);
    vzl = __riscv_vfmadd_vv_f32m2(vr, vx2h, __riscv_vfrsub_vf_f32m2(vzh, FACTOR, vl), vl);
    vzl = __riscv_vfmacc_vv_f32m2(vzl, vr, vx2l, vl); 
    
    vfloat32m2_t vrh, vrl, vth, vtl;
    vrh = __riscv_vfmul_vv_f32m2(vxh, vzh, vl); 
    vrl = __riscv_vfmsub_vv_f32m2(vxh, vzh, vrh, vl);
    vrl = __riscv_vfmacc_vv_f32m2(vrl, vzh, vxl, vl);
    vrl = __riscv_vfmacc_vv_f32m2(vrl, vzl, vxh, vl);
    
    vfloat32m2_t vnumh, vnuml, vdenh, vdenl, vd;
    vzh = __riscv_vfmadd_vf_f32m2(vch, FACTOR, vrh, vl);
    vzl = __riscv_vfsub_vv_f32m2(vrh, __riscv_vfnmsub_vf_f32m2(vch, FACTOR, vzh, vl), vl);
    vzl = __riscv_vfmadd_vf_f32m2(vcl, FACTOR, vzl, vl);
    vzl = __riscv_vfadd_vv_f32m2(vzl, vrl, vl);
    
    vth = __riscv_vfmv_v_f_f32m2(FACTOR, vl);
    vth = __riscv_vfnmsac_vv_f32m2(vth, vrh, vch, vl);
    vtl = __riscv_vfrsub_vf_f32m2(vth, FACTOR, vl);
    vtl = __riscv_vfnmsac_vv_f32m2(vtl, vrh, vch, vl);
    vtl = __riscv_vfnmsac_vv_f32m2(vtl, vrl, vch, vl);
    vtl = __riscv_vfnmsac_vv_f32m2(vtl, vrh, vcl, vl);
    
    vnumh = __riscv_vmerge_vvm_f32m2(vzh, vth, mask_div, vl);
    vnuml = __riscv_vmerge_vvm_f32m2(vzl, vtl, mask_div, vl);
    vdenh = __riscv_vmerge_vvm_f32m2(vth, vzh, mask_div, vl);
    vdenl = __riscv_vmerge_vvm_f32m2(vtl, vzl, mask_div, vl);
    
    vd = __riscv_vfrdiv_vf_f32m2(vdenh, 1.0f, vl);
    vrh = __riscv_vfmul_vv_f32m2(vnumh, vd, vl);
    vr = __riscv_vfnmsac_vv_f32m2(vnumh, vrh, vdenh, vl);
    vrl = __riscv_vfnmsac_vv_f32m2(vnuml, vrh, vdenl, vl);
    
    vr = __riscv_vfadd_vv_f32m2(vr, vrl, vl);
    vr = __riscv_vfmadd_vv_f32m2(vr, vd, vrh, vl);
    return vr;
}

vfloat32m2_t __riscv_vtan_f32m2(vfloat32m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m2(avl);

    vfloat32m2_t redx_hi, redx_lo, vtan, vntan;
    vbool16_t mask_pi_4;
    vuint32m2_t quadrants;

    __riscv_vred_paynehanek_f32m2(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);

    vbool16_t mask_div = __riscv_vmxor_mm_b16(mask_pi_4,
        __riscv_vmseq_vx_u32m2_b16(__riscv_vand_vx_u32m2(quadrants, 1, vl), 1, vl), vl);

    vtan = __riscv_vtan_kernel_f32m2(redx_hi, redx_lo,  mask_div, vl);
    vntan = __riscv_vfmul_vf_f32m2(vtan, -1.0f, vl);

    vuint32m2_t signx = __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(x), 0x80000000, vl);
    vbool16_t mask = __riscv_vmsne_vx_u32m2_b16(signx, 0, vl);
    quadrants = __riscv_vrsub_vx_u32m2_mu(mask, quadrants, quadrants, 3, vl);
    quadrants = __riscv_vand_vx_u32m2(quadrants, 1, vl);
    mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 1, vl);
    vtan = __riscv_vmerge_vvm_f32m2(vtan, vntan, mask, vl);

#ifndef __FAST_MATH__
    vuint32m2_t ix = __riscv_vand_vx_u32m2(
                __riscv_vreinterpret_v_f32m2_u32m2(x), 0x7fffffff, vl);
    vbool16_t mask_NaN = __riscv_vmsgeu_vx_u32m2_b16(ix, 0x7f800000, vl);
    vtan = __riscv_vmerge_vvm_f32m2(vtan, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b16(mask_NaN, __riscv_vmsltu_vx_u32m2_b16(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_NaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        vtan = __riscv_vfmerge_vfm_f32m2(vtan, x1, mask_NaN, vl);
    }
    volatile vfloat32m2_t tmp = __riscv_vfmadd_vf_f32m2(vtan, 0x1p-26f, vtan, vl);
#endif

    return vtan;
}


vfloat32m4_t __riscv_vtan_kernel_f32m4(vfloat32m4_t vxh, vfloat32m4_t vxl, vbool8_t mask_div, size_t vl)
{
    vfloat32m4_t vzh, vzl, vyh, vyl, vx2h, vx2l, vr;
      
    vfloat32m4_t vfindex = __riscv_vfmadd_vf_f32m4(vxh, REDUCTION_COEFF,
        __riscv_vfmv_v_f_f32m4(SHIFT_REDUCED_RANGE, vl), vl);
    vuint32m4_t	index = __riscv_vfcvt_rtz_xu_f_v_u32m4(vfindex, vl);
    
    vfindex = __riscv_vfcvt_f_xu_v_f32m4(index, vl);
    
    index = __riscv_vsll_vx_u32m4(index, 2, vl);
    
    vfloat32m4_t vch = __riscv_vloxei32_v_f32m4(TANF_TABLE_hi, index, vl);
    vfloat32m4_t vcl = __riscv_vloxei32_v_f32m4(TANF_TABLE_lo, index, vl); 
    
    vzh = __riscv_vfnmsub_vf_f32m4(vfindex, INV_REDUCTION_COEFF_hi, vxh, vl);
    vzl = __riscv_vfnmsub_vf_f32m4(vfindex, INV_REDUCTION_COEFF_lo, vxl, vl);
    vxh = __riscv_vfadd_vv_f32m4(vzh, vzl, vl);
    vxl = __riscv_vfsub_vv_f32m4(vzl, __riscv_vfsub_vv_f32m4(vxh, vzh, vl), vl);
    
    vx2h = __riscv_vfmadd_vv_f32m4(vxh, vxh, __riscv_vfmv_v_f_f32m4(SHIFT_AVOID_UNDERFFLOW, vl), vl);
    vx2l = __riscv_vfmsub_vv_f32m4(vxh, vxh, vx2h, vl);
    vx2l = __riscv_vfmacc_vv_f32m4(vx2l, vxh, __riscv_vfmul_vf_f32m4(vxl, 2.0f, vl), vl);
    
    vr = __riscv_vfmadd_vf_f32m4(vx2h, POL_COEFF_7, __riscv_vfmv_v_f_f32m4(POL_COEFF_5, vl), vl);
    vr = __riscv_vfmadd_vv_f32m4(vr, vx2h, __riscv_vfmv_v_f_f32m4(POL_COEFF_3, vl), vl);
    vzh = __riscv_vfmadd_vv_f32m4(vr, vx2h, __riscv_vfmv_v_f_f32m4(FACTOR, vl), vl);
    vzl = __riscv_vfmadd_vv_f32m4(vr, vx2h, __riscv_vfrsub_vf_f32m4(vzh, FACTOR, vl), vl);
    vzl = __riscv_vfmacc_vv_f32m4(vzl, vr, vx2l, vl); 
    
    vfloat32m4_t vrh, vrl, vth, vtl;
    vrh = __riscv_vfmul_vv_f32m4(vxh, vzh, vl); 
    vrl = __riscv_vfmsub_vv_f32m4(vxh, vzh, vrh, vl);
    vrl = __riscv_vfmacc_vv_f32m4(vrl, vzh, vxl, vl);
    vrl = __riscv_vfmacc_vv_f32m4(vrl, vzl, vxh, vl);
    
    vfloat32m4_t vnumh, vnuml, vdenh, vdenl, vd;
    vzh = __riscv_vfmadd_vf_f32m4(vch, FACTOR, vrh, vl);
    vzl = __riscv_vfsub_vv_f32m4(vrh, __riscv_vfnmsub_vf_f32m4(vch, FACTOR, vzh, vl), vl);
    vzl = __riscv_vfmadd_vf_f32m4(vcl, FACTOR, vzl, vl);
    vzl = __riscv_vfadd_vv_f32m4(vzl, vrl, vl);
    
    vth = __riscv_vfmv_v_f_f32m4(FACTOR, vl);
    vth = __riscv_vfnmsac_vv_f32m4(vth, vrh, vch, vl);
    vtl = __riscv_vfrsub_vf_f32m4(vth, FACTOR, vl);
    vtl = __riscv_vfnmsac_vv_f32m4(vtl, vrh, vch, vl);
    vtl = __riscv_vfnmsac_vv_f32m4(vtl, vrl, vch, vl);
    vtl = __riscv_vfnmsac_vv_f32m4(vtl, vrh, vcl, vl);
    
    vnumh = __riscv_vmerge_vvm_f32m4(vzh, vth, mask_div, vl);
    vnuml = __riscv_vmerge_vvm_f32m4(vzl, vtl, mask_div, vl);
    vdenh = __riscv_vmerge_vvm_f32m4(vth, vzh, mask_div, vl);
    vdenl = __riscv_vmerge_vvm_f32m4(vtl, vzl, mask_div, vl);
    
    vd = __riscv_vfrdiv_vf_f32m4(vdenh, 1.0f, vl);
    vrh = __riscv_vfmul_vv_f32m4(vnumh, vd, vl);
    vr = __riscv_vfnmsac_vv_f32m4(vnumh, vrh, vdenh, vl);
    vrl = __riscv_vfnmsac_vv_f32m4(vnuml, vrh, vdenl, vl);
    
    vr = __riscv_vfadd_vv_f32m4(vr, vrl, vl);
    vr = __riscv_vfmadd_vv_f32m4(vr, vd, vrh, vl);
    return vr;
}

vfloat32m4_t __riscv_vtan_f32m4(vfloat32m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m4(avl);

    vfloat32m4_t redx_hi, redx_lo, vtan, vntan;
    vbool8_t mask_pi_4;
    vuint32m4_t quadrants;

    __riscv_vred_paynehanek_f32m4(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);

    vbool8_t mask_div = __riscv_vmxor_mm_b8(mask_pi_4,
        __riscv_vmseq_vx_u32m4_b8(__riscv_vand_vx_u32m4(quadrants, 1, vl), 1, vl), vl);

    vtan = __riscv_vtan_kernel_f32m4(redx_hi, redx_lo,  mask_div, vl);
    vntan = __riscv_vfmul_vf_f32m4(vtan, -1.0f, vl);

    vuint32m4_t signx = __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(x), 0x80000000, vl);
    vbool8_t mask = __riscv_vmsne_vx_u32m4_b8(signx, 0, vl);
    quadrants = __riscv_vrsub_vx_u32m4_mu(mask, quadrants, quadrants, 3, vl);
    quadrants = __riscv_vand_vx_u32m4(quadrants, 1, vl);
    mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 1, vl);
    vtan = __riscv_vmerge_vvm_f32m4(vtan, vntan, mask, vl);

#ifndef __FAST_MATH__
    vuint32m4_t ix = __riscv_vand_vx_u32m4(
                __riscv_vreinterpret_v_f32m4_u32m4(x), 0x7fffffff, vl);
    vbool8_t mask_NaN = __riscv_vmsgeu_vx_u32m4_b8(ix, 0x7f800000, vl);
    vtan = __riscv_vmerge_vvm_f32m4(vtan, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b8(mask_NaN, __riscv_vmsltu_vx_u32m4_b8(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_NaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        vtan = __riscv_vfmerge_vfm_f32m4(vtan, x1, mask_NaN, vl);
    }
    volatile vfloat32m4_t tmp = __riscv_vfmadd_vf_f32m4(vtan, 0x1p-26f, vtan, vl);
#endif

    return vtan;
}


vfloat32m8_t __riscv_vtan_f32m8(vfloat32m8_t x, size_t avl)
{
    vfloat32m8_t res;
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1 = __riscv_vtan_f32m4(x1, vl);
    res = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e32m4(avl-vl);
        x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1 = __riscv_vtan_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}


#endif /* __riscv_v_intrinsic */
