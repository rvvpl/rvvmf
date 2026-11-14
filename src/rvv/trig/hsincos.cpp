/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025-2026 The RVVMF Authors  
 * (Valentin Volokitin, Dmitrii Suvorov)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  hsincos.cpp                                  *
 *   Contains: intrinsic functions sin, cos, sincos      *
 *             for float16                               *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Returns the value of a trigonometric function         *
 * of vector V                                           *
 *                                                       *
 * Algorithm:                                            *
 *    1) Argument reduction by modified                  *
 *       Payne-Hanek algorithm to [0, Pi/4]              *
 *    2) Calculating sincos                              *
 *                                                       *
 * Note that this implementation provides 0.501 ulp      *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

#include <cstdint>
#include <cfloat>
#include <cmath>

#include "trig_reduction.h"

#ifdef __riscv_zvfh

void __riscv_vsincos_kernel_f16m1(vfloat16m1_t x, vfloat16m1_t x_l, vfloat16m1_t& sinrx, vfloat16m1_t& cosrx, size_t avl)
{    
    size_t vl = __riscv_vsetvl_e16m1(avl);
    vfloat16m1_t tmp, tmp2, tl;
    
    tmp = __riscv_vfmul_vv_f16m1(x_l, x, vl);
    tmp = __riscv_vfmul_vf_f16m1(tmp, 2.0f16, vl);
    vfloat16m1_t x2 = __riscv_vfmul_vv_f16m1(x, x, vl);
    vfloat16m1_t x2_lo = __riscv_vfmsub_vv_f16m1(x, x, x2, vl);
    x2_lo = __riscv_vfadd_vv_f16m1(tmp, x2_lo, vl);

    tmp2 = __riscv_vfmadd_vf_f16m1(x2, -0x1.eacp-13f16, 
                    __riscv_vfmv_v_f_f16m1(0x1.0bp1f16, vl), vl);
    tl = __riscv_vfmadd_vf_f16m1(x2, -0x1.eacp-13f16, 
                                    __riscv_vfrsub_vf_f16m1(tmp2, 0x1.0bp1f16, vl), vl);
    sinrx = __riscv_vfmadd_vv_f16m1(x2, tmp2, 
                    __riscv_vfmv_v_f_f16m1(-0x1.554p5f16, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m1(x2, tmp2, __riscv_vfrsub_vf_f16m1(sinrx, -0x1.554p5f16, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m1(x2_lo, tmp2, tmp, vl);
    tmp = __riscv_vfmadd_vv_f16m1(tmp, x2, tl, vl);
    tmp = __riscv_vfmadd_vv_f16m1(sinrx, x2_lo, tmp, vl);
    tmp2 = __riscv_vfmul_vv_f16m1(sinrx, x2, vl);
    tl = __riscv_vfmsub_vv_f16m1(sinrx, x2, tmp2, vl);
    tl = __riscv_vfadd_vv_f16m1(tl, tmp, vl);   
    tl = __riscv_vfmadd_vv_f16m1(x, tl, __riscv_vfmul_vf_f16m1(x_l, 0x1p8f16, vl), vl);
    sinrx = __riscv_vfmadd_vv_f16m1(tmp2, x, __riscv_vfmul_vf_f16m1(x, 0x1p8f16, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m1(tmp2, x, 
                            __riscv_vfsub_vv_f16m1(__riscv_vfmul_vf_f16m1(x, 0x1p8f16, vl), sinrx, vl), vl); 
    tmp = __riscv_vfmadd_vv_f16m1(tmp2, x_l, tmp, vl); 
    tmp = __riscv_vfadd_vv_f16m1(tmp, tl, vl);
    sinrx = __riscv_vfadd_vv_f16m1(sinrx, tmp, vl);
    sinrx = __riscv_vfmul_vf_f16m1(sinrx, 0x1p-8f16, vl);
    
    cosrx = __riscv_vfmadd_vf_f16m1(x2, -0x1.63cp-10f16,
                    __riscv_vfmv_v_f_f16m1(0x1.554p-5f16, vl), vl);
    tmp2 = __riscv_vfmadd_vf_f16m1(x2, -0x1.63cp-10f16, 
                                        __riscv_vfrsub_vf_f16m1(cosrx, 0x1.554p-5f16, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m1(cosrx, x2, 
                    __riscv_vfmv_v_f_f16m1(-0x1p-1f16, vl), vl);
    tl = __riscv_vfmadd_vv_f16m1(cosrx, x2, __riscv_vfrsub_vf_f16m1(tmp, -0x1p-1f16, vl), vl);
    tl = __riscv_vfmadd_vv_f16m1(cosrx, x2_lo, tl, vl);
    tl = __riscv_vfmadd_vv_f16m1(tmp2, x2, tl, vl);
    cosrx = __riscv_vfmadd_vv_f16m1(tmp, x2, 
                    __riscv_vfmv_v_f_f16m1(0x1p0f16, vl), vl);
    tmp2 = __riscv_vfmadd_vv_f16m1(tmp, x2, __riscv_vfrsub_vf_f16m1(cosrx, 0x1p0f16, vl), vl);     
    tmp2 = __riscv_vfmacc_vv_f16m1(tmp2, tl, x2_lo, vl);
    tmp2 = __riscv_vfmacc_vv_f16m1(tmp2, tmp, x2_lo, vl);
    tmp2 = __riscv_vfmacc_vv_f16m1(tmp2, tl, x2, vl);
    cosrx = __riscv_vfadd_vv_f16m1(cosrx, tmp2, vl);
}

void __riscv_vsincos_kernel_f16m2(vfloat16m2_t x, vfloat16m2_t x_l, vfloat16m2_t& sinrx, vfloat16m2_t& cosrx, size_t avl)
{    
    size_t vl = __riscv_vsetvl_e16m2(avl);
    vfloat16m2_t tmp, tmp2, tl;
    
    tmp = __riscv_vfmul_vv_f16m2(x_l, x, vl);
    tmp = __riscv_vfmul_vf_f16m2(tmp, 2.0f16, vl);
    vfloat16m2_t x2 = __riscv_vfmul_vv_f16m2(x, x, vl);
    vfloat16m2_t x2_lo = __riscv_vfmsub_vv_f16m2(x, x, x2, vl);
    x2_lo = __riscv_vfadd_vv_f16m2(tmp, x2_lo, vl);

    tmp2 = __riscv_vfmadd_vf_f16m2(x2, -0x1.eacp-13f16, 
                    __riscv_vfmv_v_f_f16m2(0x1.0bp1f16, vl), vl);
    tl = __riscv_vfmadd_vf_f16m2(x2, -0x1.eacp-13f16, 
                                    __riscv_vfrsub_vf_f16m2(tmp2, 0x1.0bp1f16, vl), vl);
    sinrx = __riscv_vfmadd_vv_f16m2(x2, tmp2, 
                    __riscv_vfmv_v_f_f16m2(-0x1.554p5f16, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m2(x2, tmp2, __riscv_vfrsub_vf_f16m2(sinrx, -0x1.554p5f16, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m2(x2_lo, tmp2, tmp, vl);
    tmp = __riscv_vfmadd_vv_f16m2(tmp, x2, tl, vl);
    tmp = __riscv_vfmadd_vv_f16m2(sinrx, x2_lo, tmp, vl);
    tmp2 = __riscv_vfmul_vv_f16m2(sinrx, x2, vl);
    tl = __riscv_vfmsub_vv_f16m2(sinrx, x2, tmp2, vl);
    tl = __riscv_vfadd_vv_f16m2(tl, tmp, vl);   
    tl = __riscv_vfmadd_vv_f16m2(x, tl, __riscv_vfmul_vf_f16m2(x_l, 0x1p8f16, vl), vl);
    sinrx = __riscv_vfmadd_vv_f16m2(tmp2, x, __riscv_vfmul_vf_f16m2(x, 0x1p8f16, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m2(tmp2, x, 
                            __riscv_vfsub_vv_f16m2(__riscv_vfmul_vf_f16m2(x, 0x1p8f16, vl), sinrx, vl), vl); 
    tmp = __riscv_vfmadd_vv_f16m2(tmp2, x_l, tmp, vl); 
    tmp = __riscv_vfadd_vv_f16m2(tmp, tl, vl);
    sinrx = __riscv_vfadd_vv_f16m2(sinrx, tmp, vl);
    sinrx = __riscv_vfmul_vf_f16m2(sinrx, 0x1p-8f16, vl);
    
    cosrx = __riscv_vfmadd_vf_f16m2(x2, -0x1.63cp-10f16,
                    __riscv_vfmv_v_f_f16m2(0x1.554p-5f16, vl), vl);
    tmp2 = __riscv_vfmadd_vf_f16m2(x2, -0x1.63cp-10f16, 
                                        __riscv_vfrsub_vf_f16m2(cosrx, 0x1.554p-5f16, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m2(cosrx, x2, 
                    __riscv_vfmv_v_f_f16m2(-0x1p-1f16, vl), vl);
    tl = __riscv_vfmadd_vv_f16m2(cosrx, x2, __riscv_vfrsub_vf_f16m2(tmp, -0x1p-1f16, vl), vl);
    tl = __riscv_vfmadd_vv_f16m2(cosrx, x2_lo, tl, vl);
    tl = __riscv_vfmadd_vv_f16m2(tmp2, x2, tl, vl);
    cosrx = __riscv_vfmadd_vv_f16m2(tmp, x2, 
                    __riscv_vfmv_v_f_f16m2(0x1p0f16, vl), vl);
    tmp2 = __riscv_vfmadd_vv_f16m2(tmp, x2, __riscv_vfrsub_vf_f16m2(cosrx, 0x1p0f16, vl), vl);     
    tmp2 = __riscv_vfmacc_vv_f16m2(tmp2, tl, x2_lo, vl);
    tmp2 = __riscv_vfmacc_vv_f16m2(tmp2, tmp, x2_lo, vl);
    tmp2 = __riscv_vfmacc_vv_f16m2(tmp2, tl, x2, vl);
    cosrx = __riscv_vfadd_vv_f16m2(cosrx, tmp2, vl);
}

void __riscv_vsincos_kernel_f16m4(vfloat16m4_t x, vfloat16m4_t x_l, vfloat16m4_t& sinrx, vfloat16m4_t& cosrx, size_t avl)
{    
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t tmp, tmp2, tl;
    
    tmp = __riscv_vfmul_vv_f16m4(x_l, x, vl);
    tmp = __riscv_vfmul_vf_f16m4(tmp, 2.0f16, vl);
    vfloat16m4_t x2 = __riscv_vfmul_vv_f16m4(x, x, vl);
    vfloat16m4_t x2_lo = __riscv_vfmsub_vv_f16m4(x, x, x2, vl);
    x2_lo = __riscv_vfadd_vv_f16m4(tmp, x2_lo, vl);

    tmp2 = __riscv_vfmadd_vf_f16m4(x2, -0x1.eacp-13f16, 
                    __riscv_vfmv_v_f_f16m4(0x1.0bp1f16, vl), vl);
    tl = __riscv_vfmadd_vf_f16m4(x2, -0x1.eacp-13f16, 
                                    __riscv_vfrsub_vf_f16m4(tmp2, 0x1.0bp1f16, vl), vl);
    sinrx = __riscv_vfmadd_vv_f16m4(x2, tmp2, 
                    __riscv_vfmv_v_f_f16m4(-0x1.554p5f16, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m4(x2, tmp2, __riscv_vfrsub_vf_f16m4(sinrx, -0x1.554p5f16, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m4(x2_lo, tmp2, tmp, vl);
    tmp = __riscv_vfmadd_vv_f16m4(tmp, x2, tl, vl);
    tmp = __riscv_vfmadd_vv_f16m4(sinrx, x2_lo, tmp, vl);
    tmp2 = __riscv_vfmul_vv_f16m4(sinrx, x2, vl);
    tl = __riscv_vfmsub_vv_f16m4(sinrx, x2, tmp2, vl);
    tl = __riscv_vfadd_vv_f16m4(tl, tmp, vl);   
    tl = __riscv_vfmadd_vv_f16m4(x, tl, __riscv_vfmul_vf_f16m4(x_l, 0x1p8f16, vl), vl);
    sinrx = __riscv_vfmadd_vv_f16m4(tmp2, x, __riscv_vfmul_vf_f16m4(x, 0x1p8f16, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m4(tmp2, x, 
                            __riscv_vfsub_vv_f16m4(__riscv_vfmul_vf_f16m4(x, 0x1p8f16, vl), sinrx, vl), vl); 
    tmp = __riscv_vfmadd_vv_f16m4(tmp2, x_l, tmp, vl); 
    tmp = __riscv_vfadd_vv_f16m4(tmp, tl, vl);
    sinrx = __riscv_vfadd_vv_f16m4(sinrx, tmp, vl);
    sinrx = __riscv_vfmul_vf_f16m4(sinrx, 0x1p-8f16, vl);
    
    cosrx = __riscv_vfmadd_vf_f16m4(x2, -0x1.63cp-10f16,
                    __riscv_vfmv_v_f_f16m4(0x1.554p-5f16, vl), vl);
    tmp2 = __riscv_vfmadd_vf_f16m4(x2, -0x1.63cp-10f16, 
                                        __riscv_vfrsub_vf_f16m4(cosrx, 0x1.554p-5f16, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m4(cosrx, x2, 
                    __riscv_vfmv_v_f_f16m4(-0x1p-1f16, vl), vl);
    tl = __riscv_vfmadd_vv_f16m4(cosrx, x2, __riscv_vfrsub_vf_f16m4(tmp, -0x1p-1f16, vl), vl);
    tl = __riscv_vfmadd_vv_f16m4(cosrx, x2_lo, tl, vl);
    tl = __riscv_vfmadd_vv_f16m4(tmp2, x2, tl, vl);
    cosrx = __riscv_vfmadd_vv_f16m4(tmp, x2, 
                    __riscv_vfmv_v_f_f16m4(0x1p0f16, vl), vl);
    tmp2 = __riscv_vfmadd_vv_f16m4(tmp, x2, __riscv_vfrsub_vf_f16m4(cosrx, 0x1p0f16, vl), vl);     
    tmp2 = __riscv_vfmacc_vv_f16m4(tmp2, tl, x2_lo, vl);
    tmp2 = __riscv_vfmacc_vv_f16m4(tmp2, tmp, x2_lo, vl);
    tmp2 = __riscv_vfmacc_vv_f16m4(tmp2, tl, x2, vl);
    cosrx = __riscv_vfadd_vv_f16m4(cosrx, tmp2, vl);
}

vfloat16m1_t __riscv_vsin_f16m1(vfloat16m1_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m1(avl);
    vfloat16m1_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool16_t mask_pi_4;
    vuint16m1_t quadrants;
    
    __riscv_vreduction_pi4_f16m1(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f16m1(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f16m1(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f16m1(cosr, sinr, mask_pi_4, vl);
    
    vfloat16m1_t res;
    vfloat16m1_t nsinrx = __riscv_vfmul_vf_f16m1(sinrx, -1.0f16, vl);
    vfloat16m1_t ncosrx = __riscv_vfmul_vf_f16m1(cosrx, -1.0f16, vl);
    
    vuint16m1_t signx = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(x), 
                                            0x8000, vl);  
    vbool16_t mask = __riscv_vmseq_vx_u16m1_b16(signx, 0, vl);
    res = __riscv_vmv_v_v_f16m1(sinrx, vl);
    sinrx = __riscv_vmerge_vvm_f16m1(nsinrx, sinrx, mask, vl);
    nsinrx = __riscv_vmerge_vvm_f16m1(res, nsinrx, mask, vl);
    res = __riscv_vmv_v_v_f16m1(cosrx, vl);
    cosrx = __riscv_vmerge_vvm_f16m1(ncosrx, cosrx, mask, vl);
    ncosrx = __riscv_vmerge_vvm_f16m1(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f16m1(res, sinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f16m1(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f16m1(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f16m1(res, ncosrx, mask, vl);

#ifndef __FAST_MATH__
    vuint16m1_t ix = __riscv_vand_vx_u16m1(
                 __riscv_vreinterpret_v_f16m1_u16m1(x), 0x7fff, vl);
    vbool16_t mask_NaN = __riscv_vmsgeu_vx_u16m1_b16 (ix, 0x7c00, vl);
    res = __riscv_vmerge_vvm_f16m1(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b16(mask_NaN,
                  __riscv_vmsltu_vx_u16m1_b16(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_NaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        res = __riscv_vfmerge_vfm_f16m1(res, x1, mask_NaN, vl);
    }
#endif

    return res;            
}

vfloat16m2_t __riscv_vsin_f16m2(vfloat16m2_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m2(avl);
    vfloat16m2_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool8_t mask_pi_4;
    vuint16m2_t quadrants;
    
    __riscv_vreduction_pi4_f16m2(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f16m2(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f16m2(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f16m2(cosr, sinr, mask_pi_4, vl);
    
    vfloat16m2_t res;
    vfloat16m2_t nsinrx = __riscv_vfmul_vf_f16m2(sinrx, -1.0f16, vl);
    vfloat16m2_t ncosrx = __riscv_vfmul_vf_f16m2(cosrx, -1.0f16, vl);
    
    vuint16m2_t signx = __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(x), 
                                            0x8000, vl);  
    vbool8_t mask = __riscv_vmseq_vx_u16m2_b8(signx, 0, vl);
    res = __riscv_vmv_v_v_f16m2(sinrx, vl);
    sinrx = __riscv_vmerge_vvm_f16m2(nsinrx, sinrx, mask, vl);
    nsinrx = __riscv_vmerge_vvm_f16m2(res, nsinrx, mask, vl);
    res = __riscv_vmv_v_v_f16m2(cosrx, vl);
    cosrx = __riscv_vmerge_vvm_f16m2(ncosrx, cosrx, mask, vl);
    ncosrx = __riscv_vmerge_vvm_f16m2(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f16m2(res, sinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f16m2(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f16m2(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f16m2(res, ncosrx, mask, vl);

#ifndef __FAST_MATH__
    vuint16m2_t ix = __riscv_vand_vx_u16m2(
                 __riscv_vreinterpret_v_f16m2_u16m2(x), 0x7fff, vl);
    vbool8_t mask_NaN = __riscv_vmsgeu_vx_u16m2_b8 (ix, 0x7c00, vl);
    res = __riscv_vmerge_vvm_f16m2(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b8(mask_NaN,
                  __riscv_vmsltu_vx_u16m2_b8(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_NaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        res = __riscv_vfmerge_vfm_f16m2(res, x1, mask_NaN, vl);
    }
#endif

    return res;            
}

vfloat16m4_t __riscv_vsin_f16m4(vfloat16m4_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool4_t mask_pi_4;
    vuint16m4_t quadrants;
    
    __riscv_vreduction_pi4_f16m4(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f16m4(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f16m4(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f16m4(cosr, sinr, mask_pi_4, vl);
    
    vfloat16m4_t res;
    vfloat16m4_t nsinrx = __riscv_vfmul_vf_f16m4(sinrx, -1.0f16, vl);
    vfloat16m4_t ncosrx = __riscv_vfmul_vf_f16m4(cosrx, -1.0f16, vl);
    
    vuint16m4_t signx = __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(x), 
                                            0x8000, vl);  
    vbool4_t mask = __riscv_vmseq_vx_u16m4_b4(signx, 0, vl);
    res = __riscv_vmv_v_v_f16m4(sinrx, vl);
    sinrx = __riscv_vmerge_vvm_f16m4(nsinrx, sinrx, mask, vl);
    nsinrx = __riscv_vmerge_vvm_f16m4(res, nsinrx, mask, vl);
    res = __riscv_vmv_v_v_f16m4(cosrx, vl);
    cosrx = __riscv_vmerge_vvm_f16m4(ncosrx, cosrx, mask, vl);
    ncosrx = __riscv_vmerge_vvm_f16m4(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f16m4(res, sinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f16m4(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f16m4(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f16m4(res, ncosrx, mask, vl);

#ifndef __FAST_MATH__
    vuint16m4_t ix = __riscv_vand_vx_u16m4(
                 __riscv_vreinterpret_v_f16m4_u16m4(x), 0x7fff, vl);
    vbool4_t mask_NaN = __riscv_vmsgeu_vx_u16m4_b4 (ix, 0x7c00, vl);
    res = __riscv_vmerge_vvm_f16m4(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b4(mask_NaN,
                  __riscv_vmsltu_vx_u16m4_b4(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b4(mask_NaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        res = __riscv_vfmerge_vfm_f16m4(res, x1, mask_NaN, vl);
    }
#endif

    return res;            
}

vfloat16m8_t __riscv_vsin_f16m8(vfloat16m8_t x, size_t avl)
{ 
    vfloat16m8_t res;
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1 = __riscv_vsin_f16m4(x1, vl);
    res = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e16m4(avl-vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1 = __riscv_vsin_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}

vfloat16m1_t __riscv_vcos_f16m1(vfloat16m1_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m1(avl);
    vfloat16m1_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool16_t mask_pi_4;
    vuint16m1_t quadrants;
    
    __riscv_vreduction_pi4_f16m1(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f16m1(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f16m1(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f16m1(cosr, sinr, mask_pi_4, vl);
    
    vfloat16m1_t res = __riscv_vfmv_v_f_f16m1(0.0f16, vl);
    vfloat16m1_t nsinrx = __riscv_vfmul_vf_f16m1(sinrx, -1.0f16, vl);
    vfloat16m1_t ncosrx = __riscv_vfmul_vf_f16m1(cosrx, -1.0f16, vl);
    
    vbool16_t mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f16m1(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f16m1(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f16m1(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f16m1(res, sinrx, mask, vl);

#ifndef __FAST_MATH__
    vuint16m1_t ix = __riscv_vand_vx_u16m1(
                 __riscv_vreinterpret_v_f16m1_u16m1(x), 0x7fff, vl);
    vbool16_t mask_NaN = __riscv_vmsgeu_vx_u16m1_b16 (ix, 0x7c00, vl);
    res = __riscv_vmerge_vvm_f16m1(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b16(mask_NaN,
                  __riscv_vmsltu_vx_u16m1_b16(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_NaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        res = __riscv_vfmerge_vfm_f16m1(res, x1, mask_NaN, vl);
    }
#endif
            
    return res;
}

vfloat16m2_t __riscv_vcos_f16m2(vfloat16m2_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m2(avl);
    vfloat16m2_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool8_t mask_pi_4;
    vuint16m2_t quadrants;
    
    __riscv_vreduction_pi4_f16m2(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f16m2(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f16m2(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f16m2(cosr, sinr, mask_pi_4, vl);
    
    vfloat16m2_t res = __riscv_vfmv_v_f_f16m2(0.0f16, vl);
    vfloat16m2_t nsinrx = __riscv_vfmul_vf_f16m2(sinrx, -1.0f16, vl);
    vfloat16m2_t ncosrx = __riscv_vfmul_vf_f16m2(cosrx, -1.0f16, vl);
    
    vbool8_t mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f16m2(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f16m2(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f16m2(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f16m2(res, sinrx, mask, vl);

#ifndef __FAST_MATH__
    vuint16m2_t ix = __riscv_vand_vx_u16m2(
                 __riscv_vreinterpret_v_f16m2_u16m2(x), 0x7fff, vl);
    vbool8_t mask_NaN = __riscv_vmsgeu_vx_u16m2_b8 (ix, 0x7c00, vl);
    res = __riscv_vmerge_vvm_f16m2(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b8(mask_NaN,
                  __riscv_vmsltu_vx_u16m2_b8(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_NaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        res = __riscv_vfmerge_vfm_f16m2(res, x1, mask_NaN, vl);
    }
#endif
            
    return res;
}

vfloat16m4_t __riscv_vcos_f16m4(vfloat16m4_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool4_t mask_pi_4;
    vuint16m4_t quadrants;
    
    __riscv_vreduction_pi4_f16m4(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f16m4(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f16m4(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f16m4(cosr, sinr, mask_pi_4, vl);
    
    vfloat16m4_t res = __riscv_vfmv_v_f_f16m4(0.0f16, vl);
    vfloat16m4_t nsinrx = __riscv_vfmul_vf_f16m4(sinrx, -1.0f16, vl);
    vfloat16m4_t ncosrx = __riscv_vfmul_vf_f16m4(cosrx, -1.0f16, vl);
    
    vbool4_t mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f16m4(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f16m4(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f16m4(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f16m4(res, sinrx, mask, vl);

#ifndef __FAST_MATH__
    vuint16m4_t ix = __riscv_vand_vx_u16m4(
                 __riscv_vreinterpret_v_f16m4_u16m4(x), 0x7fff, vl);
    vbool4_t mask_NaN = __riscv_vmsgeu_vx_u16m4_b4 (ix, 0x7c00, vl);
    res = __riscv_vmerge_vvm_f16m4(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b4(mask_NaN,
                  __riscv_vmsltu_vx_u16m4_b4(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b4(mask_NaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        res = __riscv_vfmerge_vfm_f16m4(res, x1, mask_NaN, vl);
    }
#endif

    return res;
}

vfloat16m8_t __riscv_vcos_f16m8(vfloat16m8_t x, size_t avl)
{ 
    vfloat16m8_t res;
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1 = __riscv_vcos_f16m4(x1, vl);
    res = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e16m4(avl-vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1 = __riscv_vcos_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}

void __riscv_vsincos_f16m1(vfloat16m1_t x, vfloat16m1_t& sinx, vfloat16m1_t& cosx, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m1(avl);
    vfloat16m1_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool16_t mask_pi_4;
    vuint16m1_t quadrants;
    
    __riscv_vreduction_pi4_f16m1(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f16m1(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f16m1(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f16m1(cosr, sinr, mask_pi_4, vl);
         
    vfloat16m1_t nsinrx = __riscv_vfmul_vf_f16m1(sinrx, -1.0f16, vl);
    vfloat16m1_t ncosrx = __riscv_vfmul_vf_f16m1(cosrx, -1.0f16, vl);
     
    vuint16m1_t signx = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(x), 
                                            0x8000, vl);  
    vbool16_t mask = __riscv_vmseq_vx_u16m1_b16(signx, 0, vl);
    vfloat16m1_t ssinrx = __riscv_vmerge_vvm_f16m1(nsinrx, sinrx, mask, vl);
    vfloat16m1_t snsinrx = __riscv_vmerge_vvm_f16m1(sinrx, nsinrx, mask, vl);
    vfloat16m1_t scosrx = __riscv_vmerge_vvm_f16m1(ncosrx, cosrx, mask, vl);
    vfloat16m1_t sncosrx = __riscv_vmerge_vvm_f16m1(cosrx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 0, vl);
    sinx = __riscv_vmerge_vvm_f16m1(sinx, ssinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f16m1(cosx, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 1, vl);
    sinx = __riscv_vmerge_vvm_f16m1(sinx, scosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f16m1(cosx, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 2, vl);
    sinx = __riscv_vmerge_vvm_f16m1(sinx, snsinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f16m1(cosx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m1_b16(quadrants, 3, vl);
    sinx = __riscv_vmerge_vvm_f16m1(sinx, sncosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f16m1(cosx, sinrx, mask, vl);

    
#ifndef __FAST_MATH__
    vuint16m1_t ix = __riscv_vand_vx_u16m1(
                 __riscv_vreinterpret_v_f16m1_u16m1(x), 0x7fff, vl);
    vbool16_t mask_NaN = __riscv_vmsgeu_vx_u16m1_b16 (ix, 0x7c00, vl);
    sinx = __riscv_vmerge_vvm_f16m1(sinx, x, mask_NaN, vl);
    cosx = __riscv_vmerge_vvm_f16m1(cosx, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b16(mask_NaN,
                  __riscv_vmsltu_vx_u16m1_b16(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_NaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        sinx = __riscv_vfmerge_vfm_f16m1(sinx, x1, mask_NaN, vl);
        cosx = __riscv_vfmerge_vfm_f16m1(cosx, x1, mask_NaN, vl);
    }
#endif
}

void __riscv_vsincos_f16m2(vfloat16m2_t x, vfloat16m2_t& sinx, vfloat16m2_t& cosx, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m2(avl);
    vfloat16m2_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool8_t mask_pi_4;
    vuint16m2_t quadrants;
    
    __riscv_vreduction_pi4_f16m2(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f16m2(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f16m2(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f16m2(cosr, sinr, mask_pi_4, vl);
         
    vfloat16m2_t nsinrx = __riscv_vfmul_vf_f16m2(sinrx, -1.0f16, vl);
    vfloat16m2_t ncosrx = __riscv_vfmul_vf_f16m2(cosrx, -1.0f16, vl);
     
    vuint16m2_t signx = __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(x), 
                                            0x8000, vl);  
    vbool8_t mask = __riscv_vmseq_vx_u16m2_b8(signx, 0, vl);
    vfloat16m2_t ssinrx = __riscv_vmerge_vvm_f16m2(nsinrx, sinrx, mask, vl);
    vfloat16m2_t snsinrx = __riscv_vmerge_vvm_f16m2(sinrx, nsinrx, mask, vl);
    vfloat16m2_t scosrx = __riscv_vmerge_vvm_f16m2(ncosrx, cosrx, mask, vl);
    vfloat16m2_t sncosrx = __riscv_vmerge_vvm_f16m2(cosrx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 0, vl);
    sinx = __riscv_vmerge_vvm_f16m2(sinx, ssinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f16m2(cosx, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 1, vl);
    sinx = __riscv_vmerge_vvm_f16m2(sinx, scosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f16m2(cosx, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 2, vl);
    sinx = __riscv_vmerge_vvm_f16m2(sinx, snsinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f16m2(cosx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m2_b8(quadrants, 3, vl);
    sinx = __riscv_vmerge_vvm_f16m2(sinx, sncosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f16m2(cosx, sinrx, mask, vl);

#ifndef __FAST_MATH__
    vuint16m2_t ix = __riscv_vand_vx_u16m2(
                 __riscv_vreinterpret_v_f16m2_u16m2(x), 0x7fff, vl);
    vbool8_t mask_NaN = __riscv_vmsgeu_vx_u16m2_b8 (ix, 0x7c00, vl);
    sinx = __riscv_vmerge_vvm_f16m2(sinx, x, mask_NaN, vl);
    cosx = __riscv_vmerge_vvm_f16m2(cosx, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b8(mask_NaN,
                  __riscv_vmsltu_vx_u16m2_b8(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_NaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        sinx = __riscv_vfmerge_vfm_f16m2(sinx, x1, mask_NaN, vl);
        cosx = __riscv_vfmerge_vfm_f16m2(cosx, x1, mask_NaN, vl);
    }
#endif
}

void __riscv_vsincos_f16m4(vfloat16m4_t x, vfloat16m4_t& sinx, vfloat16m4_t& cosx, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool4_t mask_pi_4;
    vuint16m4_t quadrants;
    
    __riscv_vreduction_pi4_f16m4(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f16m4(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f16m4(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f16m4(cosr, sinr, mask_pi_4, vl);
         
    vfloat16m4_t nsinrx = __riscv_vfmul_vf_f16m4(sinrx, -1.0f16, vl);
    vfloat16m4_t ncosrx = __riscv_vfmul_vf_f16m4(cosrx, -1.0f16, vl);
     
    vuint16m4_t signx = __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(x), 
                                            0x8000, vl);  
    vbool4_t mask = __riscv_vmseq_vx_u16m4_b4(signx, 0, vl);
    vfloat16m4_t ssinrx = __riscv_vmerge_vvm_f16m4(nsinrx, sinrx, mask, vl);
    vfloat16m4_t snsinrx = __riscv_vmerge_vvm_f16m4(sinrx, nsinrx, mask, vl);
    vfloat16m4_t scosrx = __riscv_vmerge_vvm_f16m4(ncosrx, cosrx, mask, vl);
    vfloat16m4_t sncosrx = __riscv_vmerge_vvm_f16m4(cosrx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 0, vl);
    sinx = __riscv_vmerge_vvm_f16m4(sinx, ssinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f16m4(cosx, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 1, vl);
    sinx = __riscv_vmerge_vvm_f16m4(sinx, scosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f16m4(cosx, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 2, vl);
    sinx = __riscv_vmerge_vvm_f16m4(sinx, snsinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f16m4(cosx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u16m4_b4(quadrants, 3, vl);
    sinx = __riscv_vmerge_vvm_f16m4(sinx, sncosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f16m4(cosx, sinrx, mask, vl);
    
#ifndef __FAST_MATH__
    vuint16m4_t ix = __riscv_vand_vx_u16m4(
                 __riscv_vreinterpret_v_f16m4_u16m4(x), 0x7fff, vl);
    vbool4_t mask_NaN = __riscv_vmsgeu_vx_u16m4_b4 (ix, 0x7c00, vl);
    sinx = __riscv_vmerge_vvm_f16m4(sinx, x, mask_NaN, vl);
    cosx = __riscv_vmerge_vvm_f16m4(cosx, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b4(mask_NaN,
                  __riscv_vmsltu_vx_u16m4_b4(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b4(mask_NaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        sinx = __riscv_vfmerge_vfm_f16m4(sinx, x1, mask_NaN, vl);
        cosx = __riscv_vfmerge_vfm_f16m4(cosx, x1, mask_NaN, vl);
    }
#endif
}

void __riscv_vsincos_f16m8(vfloat16m8_t x, vfloat16m8_t& sinx, vfloat16m8_t& cosx, size_t avl)
{ 
    vfloat16m8_t res;
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    vfloat16m4_t sinx1 = __riscv_vget_v_f16m8_f16m4(sinx, 0);
    vfloat16m4_t cosx1 = __riscv_vget_v_f16m8_f16m4(cosx, 0);
    __riscv_vsincos_f16m4(x1, sinx1, cosx1, vl);
    sinx = __riscv_vset_v_f16m4_f16m8(sinx, 0, sinx1);
    cosx = __riscv_vset_v_f16m4_f16m8(cosx, 0, cosx1);
    if(avl > vl){
        vl = __riscv_vsetvl_e16m4(avl-vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        sinx1 = __riscv_vget_v_f16m8_f16m4(sinx, 1);
        cosx1 = __riscv_vget_v_f16m8_f16m4(cosx, 1);
        __riscv_vsincos_f16m4(x1, sinx1, cosx1, vl);
        sinx = __riscv_vset_v_f16m4_f16m8(sinx, 1, sinx1);
        cosx = __riscv_vset_v_f16m4_f16m8(cosx, 1, cosx1);
    }
}

#endif /* __riscv_zvfh */

#endif /* __riscv_v_intrinsic */