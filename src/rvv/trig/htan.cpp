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
 *   File:  htan.cpp                                     *
 *   Contains: intrinsic function tan for float16        *
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

#ifdef __riscv_zvfh


constexpr size_t FACTOR_DEG = 10;
constexpr _Float16 FACTOR = 0x1p10f16;
constexpr _Float16 INV_FACTOR = 0x1p-10f16;
constexpr _Float16 SHIFT_AVOID_UNDERFFLOW = 0x1p-11f16;
constexpr _Float16 SHIFT_REDUCED_RANGE = -0x1p-11f16*FACTOR;

constexpr _Float16 TANH_TABLE_SIZE = 0x1p2f16;
constexpr _Float16 TANH_INV_TABLE_SIZE = 0x1p-2f16;

static _Float16 TANH_TABLE_hi[4] = {
    0.0f16, 0x1.974p-3f16, 0x1.a84p-2f16, 0x1.56p-1f16
};
static _Float16 TANH_TABLE_lo[4] = {
    0.0f16*FACTOR, 0x1.f6p-15f16*FACTOR, -0x1.868p-14f16*FACTOR, 0x1.b84p-13f16*FACTOR
};

constexpr _Float16 PI_DIV_4_hi = 0x1.92p-1f16;
constexpr _Float16 PI_DIV_4_lo = 0x1.fb4p-13f16;
constexpr _Float16 INV_PI_DIV_4_hi = 0x1.46p0f16;
constexpr _Float16 INV_PI_DIV_4_lo = -0x1.9f4p-13f16;

constexpr _Float16 REDUCTION_COEFF = INV_PI_DIV_4_hi*TANH_TABLE_SIZE*FACTOR;
constexpr _Float16 INV_REDUCTION_COEFF_hi = PI_DIV_4_hi*TANH_INV_TABLE_SIZE;
constexpr _Float16 INV_REDUCTION_COEFF_lo = PI_DIV_4_lo*TANH_INV_TABLE_SIZE;

constexpr _Float16 POL_COEFF_5 = 0x1.178p-3f16*FACTOR;
constexpr _Float16 POL_COEFF_3 = 0x1.554p-2f16*FACTOR;


vfloat16m1_t __riscv_vtan_kernel_f16m1(vfloat16m1_t vxh, vfloat16m1_t vxl, vbool16_t mask_div, size_t vl)
{
    vfloat16m1_t vzh, vzl, vyh, vyl, vx2h, vx2l, vr;
    
    vfloat16m1_t vfindex = __riscv_vfmadd_vf_f16m1(vxh, REDUCTION_COEFF,
        __riscv_vfmv_v_f_f16m1(SHIFT_REDUCED_RANGE, vl), vl);
    vuint16m1_t	index = __riscv_vfcvt_rtz_xu_f_v_u16m1(vfindex, vl);
    index = __riscv_vsrl_vx_u16m1(index, FACTOR_DEG, vl);
    
    vfindex = __riscv_vfcvt_f_xu_v_f16m1(index, vl);
    
    index = __riscv_vsll_vx_u16m1(index, 1, vl);
    
    vfloat16m1_t vch = __riscv_vloxei16_v_f16m1(TANH_TABLE_hi, index, vl);
    vfloat16m1_t vcl = __riscv_vloxei16_v_f16m1(TANH_TABLE_lo, index, vl);
    
    vzh = __riscv_vfnmsub_vf_f16m1(vfindex, INV_REDUCTION_COEFF_hi, vxh, vl);
    vzl = __riscv_vfnmsub_vf_f16m1(vfindex, INV_REDUCTION_COEFF_lo, vxl, vl);
    vxh = __riscv_vfadd_vv_f16m1(vzh, vzl, vl);
    vxl = __riscv_vfsub_vv_f16m1(vzl, __riscv_vfsub_vv_f16m1(vxh, vzh, vl), vl);
    
    vx2h = __riscv_vfmadd_vv_f16m1(vxh, vxh, __riscv_vfmv_v_f_f16m1(SHIFT_AVOID_UNDERFFLOW, vl), vl);
    vx2l = __riscv_vfmsub_vv_f16m1(vxh, vxh, vx2h, vl);
    vx2l = __riscv_vfmacc_vv_f16m1(vx2l, vxh, __riscv_vfmul_vf_f16m1(vxl, 2.0f16, vl), vl);
    
    vr = __riscv_vfmadd_vf_f16m1(vx2h, POL_COEFF_5, __riscv_vfmv_v_f_f16m1(POL_COEFF_3, vl), vl);
    vzh = __riscv_vfmadd_vv_f16m1(vr, vx2h, __riscv_vfmv_v_f_f16m1(FACTOR, vl), vl);
    vzl = __riscv_vfmadd_vv_f16m1(vr, vx2h, __riscv_vfrsub_vf_f16m1(vzh, FACTOR, vl), vl);
    vzl = __riscv_vfmacc_vv_f16m1(__riscv_vfmul_vf_f16m1(vzl, FACTOR, vl), vr,
        __riscv_vfmul_vf_f16m1(vx2l, FACTOR, vl), vl);
    
#ifndef __FAST_MATH__
    // avoid underflow, does not affect the accuracy
    vuint16m1_t vzli = __riscv_vreinterpret_v_f16m1_u16m1(vzl);
    vzl = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vmul_vv_u16m1(vzli,
        __riscv_vsrl_vx_u16m1(__riscv_vand_vx_u16m1(vzli, 0x4000, vl), 14, vl), vl));
#endif
    
    vfloat16m1_t vrh, vrl, vth, vtl;
    vrh = __riscv_vfmul_vv_f16m1(vxh, vzh, vl);
    vrl = __riscv_vfmsub_vv_f16m1(vxh, vzh, vrh, vl);
    vrl = __riscv_vfmacc_vv_f16m1(vrl, vzh, vxl, vl);
    vrl = __riscv_vfmacc_vv_f16m1(__riscv_vfmul_vf_f16m1(vrl, FACTOR, vl), vzl, vxh, vl);
    
    vfloat16m1_t vnumh, vnuml, vdenh, vdenl, vd;
    vzh = __riscv_vfmadd_vf_f16m1(vch, FACTOR, vrh, vl);
    vzl = __riscv_vfsub_vv_f16m1(vrh, __riscv_vfnmsub_vf_f16m1(vch, FACTOR, vzh, vl), vl);
    vzl = __riscv_vfadd_vv_f16m1(vcl, vzl, vl);
    vzl = __riscv_vfmadd_vf_f16m1(vzl, FACTOR, vrl, vl);
    
    vth = __riscv_vfmv_v_f_f16m1(FACTOR, vl);
    vth = __riscv_vfnmsac_vv_f16m1(vth, vrh, vch, vl);
    vtl = __riscv_vfrsub_vf_f16m1(vth, FACTOR, vl);
    vtl = __riscv_vfnmsac_vv_f16m1(vtl, vrh, vch, vl);
    vtl = __riscv_vfnmsac_vv_f16m1(__riscv_vfmul_vf_f16m1(vtl, FACTOR, vl), vrl, vch, vl);
    vtl = __riscv_vfnmsac_vv_f16m1(vtl, vrh, vcl, vl);
    
    vnumh = __riscv_vmerge_vvm_f16m1(vzh, vth, mask_div, vl);
    vnuml = __riscv_vmerge_vvm_f16m1(vzl, vtl, mask_div, vl);
    vdenh = __riscv_vmerge_vvm_f16m1(vth, vzh, mask_div, vl);
    vdenl = __riscv_vmerge_vvm_f16m1(vtl, vzl, mask_div, vl);
    
    vd = __riscv_vfrdiv_vf_f16m1(vdenh, 1.0f16, vl);
    vrh = __riscv_vfmul_vv_f16m1(vnumh, vd, vl);
    vrh = __riscv_vfmerge_vfm_f16m1(vrh, 0.0f16, __riscv_vmfgt_vf_f16m1_b16(vrh, 0x1.ffcp15f16, vl), vl);
    vr = __riscv_vfnmsac_vv_f16m1(vnumh, vrh, vdenh, vl);
    vrl = __riscv_vfnmsac_vv_f16m1(vnuml, vrh, vdenl, vl);
    
#ifndef __FAST_MATH__
    // avoid underflow, does not affect the accuracy
    vuint16m1_t vrli = __riscv_vreinterpret_v_f16m1_u16m1(vrl);
    vrl = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vmul_vv_u16m1(vrli,
        __riscv_vsrl_vx_u16m1(__riscv_vand_vx_u16m1(vrli, 0x4000, vl), 14, vl), vl));
#endif
    
    vr = __riscv_vfmadd_vf_f16m1(vr, FACTOR, vrl, vl);
    vr = __riscv_vfmadd_vf_f16m1(__riscv_vfmul_vv_f16m1(vr, vd, vl), INV_FACTOR, vrh, vl);
    
    return vr;
}

vfloat16m1_t __riscv_vtan_f16m1(vfloat16m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m1(avl);

    vfloat16m1_t redx_hi, redx_lo, vtan, vntan;
    vbool16_t mask_pi_4;
    vuint16m1_t quadrants;
    
    __riscv_vred_paynehanek_f16m1(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    
    vbool16_t mask_div = __riscv_vmxor_mm_b16(mask_pi_4,
        __riscv_vmseq_vx_u16m1_b16(__riscv_vand_vx_u16m1(quadrants, 1, vl), 1, vl), vl);

    vtan = __riscv_vtan_kernel_f16m1(redx_hi, redx_lo,  mask_div, vl);
    vntan = __riscv_vfmul_vf_f16m1(vtan, -1.0f16, vl);

    vtan = __riscv_vtan_kernel_f16m1(redx_hi, redx_lo,  mask_div, vl);
    vntan = __riscv_vfmul_vf_f16m1(vtan, -1.0f16, vl);

    vuint16m1_t signx = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(x), 0x8000, vl);
    vbool16_t mask = __riscv_vmsne_vx_u16m1_b16(signx, 0, vl);
    quadrants = __riscv_vrsub_vx_u16m1_mu(mask, quadrants, quadrants, 3, vl);
    quadrants = __riscv_vand_vx_u16m1(quadrants, 1, vl);
    mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 1, vl);
    vtan = __riscv_vmerge_vvm_f16m1(vtan, vntan, mask, vl);

#ifndef __FAST_MATH__
    vuint16m1_t ix = __riscv_vand_vx_u16m1(
                __riscv_vreinterpret_v_f16m1_u16m1(x), 0x7fff, vl);
    vbool16_t mask_NaN = __riscv_vmsgeu_vx_u16m1_b16(ix, 0x7c00, vl);
    vtan = __riscv_vmerge_vvm_f16m1(vtan, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b16(mask_NaN, __riscv_vmsltu_vx_u16m1_b16(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_NaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        vtan = __riscv_vfmerge_vfm_f16m1(vtan, x1, mask_NaN, vl);
    }
    volatile vfloat16m1_t tmp = __riscv_vfmadd_vf_f16m1(vtan, 0x1p-14f16, vtan, vl);
#endif

    return vtan;
}


vfloat16m2_t __riscv_vtan_kernel_f16m2(vfloat16m2_t vxh, vfloat16m2_t vxl, vbool8_t mask_div, size_t vl)
{
    vfloat16m2_t vzh, vzl, vyh, vyl, vx2h, vx2l, vr;
    
    vfloat16m2_t vfindex = __riscv_vfmadd_vf_f16m2(vxh, REDUCTION_COEFF,
        __riscv_vfmv_v_f_f16m2(SHIFT_REDUCED_RANGE, vl), vl);
    vuint16m2_t	index = __riscv_vfcvt_rtz_xu_f_v_u16m2(vfindex, vl);
    index = __riscv_vsrl_vx_u16m2(index, FACTOR_DEG, vl);
    
    vfindex = __riscv_vfcvt_f_xu_v_f16m2(index, vl);
    
    index = __riscv_vsll_vx_u16m2(index, 1, vl);
    
    vfloat16m2_t vch = __riscv_vloxei16_v_f16m2(TANH_TABLE_hi, index, vl);
    vfloat16m2_t vcl = __riscv_vloxei16_v_f16m2(TANH_TABLE_lo, index, vl);
    
    vzh = __riscv_vfnmsub_vf_f16m2(vfindex, INV_REDUCTION_COEFF_hi, vxh, vl);
    vzl = __riscv_vfnmsub_vf_f16m2(vfindex, INV_REDUCTION_COEFF_lo, vxl, vl);
    vxh = __riscv_vfadd_vv_f16m2(vzh, vzl, vl);
    vxl = __riscv_vfsub_vv_f16m2(vzl, __riscv_vfsub_vv_f16m2(vxh, vzh, vl), vl);
    
    vx2h = __riscv_vfmadd_vv_f16m2(vxh, vxh, __riscv_vfmv_v_f_f16m2(SHIFT_AVOID_UNDERFFLOW, vl), vl);
    vx2l = __riscv_vfmsub_vv_f16m2(vxh, vxh, vx2h, vl);
    vx2l = __riscv_vfmacc_vv_f16m2(vx2l, vxh, __riscv_vfmul_vf_f16m2(vxl, 2.0f16, vl), vl);
    
    vr = __riscv_vfmadd_vf_f16m2(vx2h, POL_COEFF_5, __riscv_vfmv_v_f_f16m2(POL_COEFF_3, vl), vl);
    vzh = __riscv_vfmadd_vv_f16m2(vr, vx2h, __riscv_vfmv_v_f_f16m2(FACTOR, vl), vl);
    vzl = __riscv_vfmadd_vv_f16m2(vr, vx2h, __riscv_vfrsub_vf_f16m2(vzh, FACTOR, vl), vl);
    vzl = __riscv_vfmacc_vv_f16m2(__riscv_vfmul_vf_f16m2(vzl, FACTOR, vl), vr,
        __riscv_vfmul_vf_f16m2(vx2l, FACTOR, vl), vl);
    
#ifndef __FAST_MATH__
    // avoid underflow, does not affect the accuracy
    vuint16m2_t vzli = __riscv_vreinterpret_v_f16m2_u16m2(vzl);
    vzl = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vmul_vv_u16m2(vzli,
        __riscv_vsrl_vx_u16m2(__riscv_vand_vx_u16m2(vzli, 0x4000, vl), 14, vl), vl));
#endif
    
    vfloat16m2_t vrh, vrl, vth, vtl;
    vrh = __riscv_vfmul_vv_f16m2(vxh, vzh, vl);
    vrl = __riscv_vfmsub_vv_f16m2(vxh, vzh, vrh, vl);
    vrl = __riscv_vfmacc_vv_f16m2(vrl, vzh, vxl, vl);
    vrl = __riscv_vfmacc_vv_f16m2(__riscv_vfmul_vf_f16m2(vrl, FACTOR, vl), vzl, vxh, vl);
    
    vfloat16m2_t vnumh, vnuml, vdenh, vdenl, vd;
    vzh = __riscv_vfmadd_vf_f16m2(vch, FACTOR, vrh, vl);
    vzl = __riscv_vfsub_vv_f16m2(vrh, __riscv_vfnmsub_vf_f16m2(vch, FACTOR, vzh, vl), vl);
    vzl = __riscv_vfadd_vv_f16m2(vcl, vzl, vl);
    vzl = __riscv_vfmadd_vf_f16m2(vzl, FACTOR, vrl, vl);
    
    vth = __riscv_vfmv_v_f_f16m2(FACTOR, vl);
    vth = __riscv_vfnmsac_vv_f16m2(vth, vrh, vch, vl);
    vtl = __riscv_vfrsub_vf_f16m2(vth, FACTOR, vl);
    vtl = __riscv_vfnmsac_vv_f16m2(vtl, vrh, vch, vl);
    vtl = __riscv_vfnmsac_vv_f16m2(__riscv_vfmul_vf_f16m2(vtl, FACTOR, vl), vrl, vch, vl);
    vtl = __riscv_vfnmsac_vv_f16m2(vtl, vrh, vcl, vl);
    
    vnumh = __riscv_vmerge_vvm_f16m2(vzh, vth, mask_div, vl);
    vnuml = __riscv_vmerge_vvm_f16m2(vzl, vtl, mask_div, vl);
    vdenh = __riscv_vmerge_vvm_f16m2(vth, vzh, mask_div, vl);
    vdenl = __riscv_vmerge_vvm_f16m2(vtl, vzl, mask_div, vl);
    
    vd = __riscv_vfrdiv_vf_f16m2(vdenh, 1.0f16, vl);
    vrh = __riscv_vfmul_vv_f16m2(vnumh, vd, vl);
    vrh = __riscv_vfmerge_vfm_f16m2(vrh, 0.0f16, __riscv_vmfgt_vf_f16m2_b8(vrh, 0x1.ffcp15f16, vl), vl);
    vr = __riscv_vfnmsac_vv_f16m2(vnumh, vrh, vdenh, vl);
    vrl = __riscv_vfnmsac_vv_f16m2(vnuml, vrh, vdenl, vl);
    
#ifndef __FAST_MATH__
    // avoid underflow, does not affect the accuracy
    vuint16m2_t vrli = __riscv_vreinterpret_v_f16m2_u16m2(vrl);
    vrl = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vmul_vv_u16m2(vrli,
        __riscv_vsrl_vx_u16m2(__riscv_vand_vx_u16m2(vrli, 0x4000, vl), 14, vl), vl));
#endif
    
    vr = __riscv_vfmadd_vf_f16m2(vr, FACTOR, vrl, vl);
    vr = __riscv_vfmadd_vf_f16m2(__riscv_vfmul_vv_f16m2(vr, vd, vl), INV_FACTOR, vrh, vl);
    
    return vr;
}

vfloat16m2_t __riscv_vtan_f16m2(vfloat16m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m2(avl);

    vfloat16m2_t redx_hi, redx_lo, vtan, vntan;
    vbool8_t mask_pi_4;
    vuint16m2_t quadrants;
    
    __riscv_vred_paynehanek_f16m2(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    
    vbool8_t mask_div = __riscv_vmxor_mm_b8(mask_pi_4,
        __riscv_vmseq_vx_u16m2_b8(__riscv_vand_vx_u16m2(quadrants, 1, vl), 1, vl), vl);

    vtan = __riscv_vtan_kernel_f16m2(redx_hi, redx_lo,  mask_div, vl);
    vntan = __riscv_vfmul_vf_f16m2(vtan, -1.0f16, vl);

    vtan = __riscv_vtan_kernel_f16m2(redx_hi, redx_lo,  mask_div, vl);
    vntan = __riscv_vfmul_vf_f16m2(vtan, -1.0f16, vl);

    vuint16m2_t signx = __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(x), 0x8000, vl);
    vbool8_t mask = __riscv_vmsne_vx_u16m2_b8(signx, 0, vl);
    quadrants = __riscv_vrsub_vx_u16m2_mu(mask, quadrants, quadrants, 3, vl);
    quadrants = __riscv_vand_vx_u16m2(quadrants, 1, vl);
    mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 1, vl);
    vtan = __riscv_vmerge_vvm_f16m2(vtan, vntan, mask, vl);

#ifndef __FAST_MATH__
    vuint16m2_t ix = __riscv_vand_vx_u16m2(
                __riscv_vreinterpret_v_f16m2_u16m2(x), 0x7fff, vl);
    vbool8_t mask_NaN = __riscv_vmsgeu_vx_u16m2_b8(ix, 0x7c00, vl);
    vtan = __riscv_vmerge_vvm_f16m2(vtan, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b8(mask_NaN, __riscv_vmsltu_vx_u16m2_b8(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_NaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        vtan = __riscv_vfmerge_vfm_f16m2(vtan, x1, mask_NaN, vl);
    }
    volatile vfloat16m2_t tmp = __riscv_vfmadd_vf_f16m2(vtan, 0x1p-14f16, vtan, vl);
#endif

    return vtan;
}


vfloat16m4_t __riscv_vtan_kernel_f16m4(vfloat16m4_t vxh, vfloat16m4_t vxl, vbool4_t mask_div, size_t vl)
{
    vfloat16m4_t vzh, vzl, vyh, vyl, vx2h, vx2l, vr;
    
    vfloat16m4_t vfindex = __riscv_vfmadd_vf_f16m4(vxh, REDUCTION_COEFF,
        __riscv_vfmv_v_f_f16m4(SHIFT_REDUCED_RANGE, vl), vl);
    vuint16m4_t	index = __riscv_vfcvt_rtz_xu_f_v_u16m4(vfindex, vl);
    index = __riscv_vsrl_vx_u16m4(index, FACTOR_DEG, vl);
    
    vfindex = __riscv_vfcvt_f_xu_v_f16m4(index, vl);
    
    index = __riscv_vsll_vx_u16m4(index, 1, vl);
    
    vfloat16m4_t vch = __riscv_vloxei16_v_f16m4(TANH_TABLE_hi, index, vl);
    vfloat16m4_t vcl = __riscv_vloxei16_v_f16m4(TANH_TABLE_lo, index, vl);
    
    vzh = __riscv_vfnmsub_vf_f16m4(vfindex, INV_REDUCTION_COEFF_hi, vxh, vl);
    vzl = __riscv_vfnmsub_vf_f16m4(vfindex, INV_REDUCTION_COEFF_lo, vxl, vl);
    vxh = __riscv_vfadd_vv_f16m4(vzh, vzl, vl);
    vxl = __riscv_vfsub_vv_f16m4(vzl, __riscv_vfsub_vv_f16m4(vxh, vzh, vl), vl);
    
    vx2h = __riscv_vfmadd_vv_f16m4(vxh, vxh, __riscv_vfmv_v_f_f16m4(SHIFT_AVOID_UNDERFFLOW, vl), vl);
    vx2l = __riscv_vfmsub_vv_f16m4(vxh, vxh, vx2h, vl);
    vx2l = __riscv_vfmacc_vv_f16m4(vx2l, vxh, __riscv_vfmul_vf_f16m4(vxl, 2.0f16, vl), vl);
    
    vr = __riscv_vfmadd_vf_f16m4(vx2h, POL_COEFF_5, __riscv_vfmv_v_f_f16m4(POL_COEFF_3, vl), vl);
    vzh = __riscv_vfmadd_vv_f16m4(vr, vx2h, __riscv_vfmv_v_f_f16m4(FACTOR, vl), vl);
    vzl = __riscv_vfmadd_vv_f16m4(vr, vx2h, __riscv_vfrsub_vf_f16m4(vzh, FACTOR, vl), vl);
    vzl = __riscv_vfmacc_vv_f16m4(__riscv_vfmul_vf_f16m4(vzl, FACTOR, vl), vr,
        __riscv_vfmul_vf_f16m4(vx2l, FACTOR, vl), vl);
    
#ifndef __FAST_MATH__
    // avoid underflow, does not affect the accuracy
    vuint16m4_t vzli = __riscv_vreinterpret_v_f16m4_u16m4(vzl);
    vzl = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vmul_vv_u16m4(vzli,
        __riscv_vsrl_vx_u16m4(__riscv_vand_vx_u16m4(vzli, 0x4000, vl), 14, vl), vl));
#endif
    
    vfloat16m4_t vrh, vrl, vth, vtl;
    vrh = __riscv_vfmul_vv_f16m4(vxh, vzh, vl);
    vrl = __riscv_vfmsub_vv_f16m4(vxh, vzh, vrh, vl);
    vrl = __riscv_vfmacc_vv_f16m4(vrl, vzh, vxl, vl);
    vrl = __riscv_vfmacc_vv_f16m4(__riscv_vfmul_vf_f16m4(vrl, FACTOR, vl), vzl, vxh, vl);
    
    vfloat16m4_t vnumh, vnuml, vdenh, vdenl, vd;
    vzh = __riscv_vfmadd_vf_f16m4(vch, FACTOR, vrh, vl);
    vzl = __riscv_vfsub_vv_f16m4(vrh, __riscv_vfnmsub_vf_f16m4(vch, FACTOR, vzh, vl), vl);
    vzl = __riscv_vfadd_vv_f16m4(vcl, vzl, vl);
    vzl = __riscv_vfmadd_vf_f16m4(vzl, FACTOR, vrl, vl);
    
    vth = __riscv_vfmv_v_f_f16m4(FACTOR, vl);
    vth = __riscv_vfnmsac_vv_f16m4(vth, vrh, vch, vl);
    vtl = __riscv_vfrsub_vf_f16m4(vth, FACTOR, vl);
    vtl = __riscv_vfnmsac_vv_f16m4(vtl, vrh, vch, vl);
    vtl = __riscv_vfnmsac_vv_f16m4(__riscv_vfmul_vf_f16m4(vtl, FACTOR, vl), vrl, vch, vl);
    vtl = __riscv_vfnmsac_vv_f16m4(vtl, vrh, vcl, vl);
    
    vnumh = __riscv_vmerge_vvm_f16m4(vzh, vth, mask_div, vl);
    vnuml = __riscv_vmerge_vvm_f16m4(vzl, vtl, mask_div, vl);
    vdenh = __riscv_vmerge_vvm_f16m4(vth, vzh, mask_div, vl);
    vdenl = __riscv_vmerge_vvm_f16m4(vtl, vzl, mask_div, vl);
    
    vd = __riscv_vfrdiv_vf_f16m4(vdenh, 1.0f16, vl);
    vrh = __riscv_vfmul_vv_f16m4(vnumh, vd, vl);
    vrh = __riscv_vfmerge_vfm_f16m4(vrh, 0.0f16, __riscv_vmfgt_vf_f16m4_b4(vrh, 0x1.ffcp15f16, vl), vl);
    vr = __riscv_vfnmsac_vv_f16m4(vnumh, vrh, vdenh, vl);
    vrl = __riscv_vfnmsac_vv_f16m4(vnuml, vrh, vdenl, vl);
    
#ifndef __FAST_MATH__
    // avoid underflow, does not affect the accuracy
    vuint16m4_t vrli = __riscv_vreinterpret_v_f16m4_u16m4(vrl);
    vrl = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vmul_vv_u16m4(vrli,
        __riscv_vsrl_vx_u16m4(__riscv_vand_vx_u16m4(vrli, 0x4000, vl), 14, vl), vl));
#endif
    
    vr = __riscv_vfmadd_vf_f16m4(vr, FACTOR, vrl, vl);
    vr = __riscv_vfmadd_vf_f16m4(__riscv_vfmul_vv_f16m4(vr, vd, vl), INV_FACTOR, vrh, vl);
    
    return vr;
}

vfloat16m4_t __riscv_vtan_f16m4(vfloat16m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m4(avl);

    vfloat16m4_t redx_hi, redx_lo, vtan, vntan;
    vbool4_t mask_pi_4;
    vuint16m4_t quadrants;
    
    __riscv_vred_paynehanek_f16m4(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    
    vbool4_t mask_div = __riscv_vmxor_mm_b4(mask_pi_4,
        __riscv_vmseq_vx_u16m4_b4(__riscv_vand_vx_u16m4(quadrants, 1, vl), 1, vl), vl);

    vtan = __riscv_vtan_kernel_f16m4(redx_hi, redx_lo,  mask_div, vl);
    vntan = __riscv_vfmul_vf_f16m4(vtan, -1.0f16, vl);

    vtan = __riscv_vtan_kernel_f16m4(redx_hi, redx_lo,  mask_div, vl);
    vntan = __riscv_vfmul_vf_f16m4(vtan, -1.0f16, vl);

    vuint16m4_t signx = __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(x), 0x8000, vl);
    vbool4_t mask = __riscv_vmsne_vx_u16m4_b4(signx, 0, vl);
    quadrants = __riscv_vrsub_vx_u16m4_mu(mask, quadrants, quadrants, 3, vl);
    quadrants = __riscv_vand_vx_u16m4(quadrants, 1, vl);
    mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 1, vl);
    vtan = __riscv_vmerge_vvm_f16m4(vtan, vntan, mask, vl);

#ifndef __FAST_MATH__
    vuint16m4_t ix = __riscv_vand_vx_u16m4(
                __riscv_vreinterpret_v_f16m4_u16m4(x), 0x7fff, vl);
    vbool4_t mask_NaN = __riscv_vmsgeu_vx_u16m4_b4(ix, 0x7c00, vl);
    vtan = __riscv_vmerge_vvm_f16m4(vtan, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b4(mask_NaN, __riscv_vmsltu_vx_u16m4_b4(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b4(mask_NaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        vtan = __riscv_vfmerge_vfm_f16m4(vtan, x1, mask_NaN, vl);
    }
    volatile vfloat16m4_t tmp = __riscv_vfmadd_vf_f16m4(vtan, 0x1p-14f16, vtan, vl);
#endif

    return vtan;
}


vfloat16m8_t __riscv_vtan_f16m8(vfloat16m8_t x, size_t avl)
{
    vfloat16m8_t res;
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1 = __riscv_vtan_f16m4(x1, vl);
    res = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e16m4(avl-vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1 = __riscv_vtan_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}



#endif /* __riscv_zvfh */

#endif /* __riscv_v_intrinsic */
