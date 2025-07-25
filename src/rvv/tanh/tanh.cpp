/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025 The RVVMF Authors (Valentin Volokitin)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  tanh.c                                       *
 *   Contains: intrinsic function tanh for f64, f32, f16 *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the hyperbolic tangent of input vector V     *
 *                                                       *
 * Algorithm:                                            *
 *    1) Piecewise polynomial approximation on segments: *
 *       f64 [0, 0x1.30fc1931f09c9p+4] - 94,             *
 *       f32 [0, 0x1.205966p+3] - 83,                    *
 *       f16 [0, 0x1.0a4p+2] - 10                        *
 *    2) For efficiency, some sections are divided into  *
 *       2 (fp16), 4 (fp64) or 8 (fp32) equal sections   *
 *    3) Polynomial degrees: f64 - 13, f32 - 5, f16 - 5  *
 *                                                       *
 *                                                       *
 *********************************************************
*/
 
#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"


//static double tanhdp [1520];
#include "dtanh.data"
//static float tanhsp [672];
#include "stanh.data"

vfloat64m1_t __riscv_vtanh_f64m1(vfloat64m1_t x, size_t avl)
{ 
    size_t vl = __riscv_vsetvl_e64m1(avl);
    vuint64m1_t ix = __riscv_vand_vx_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(x), 0x7fffffffffffffff, vl);
    
    vuint64m1_t index = __riscv_vsrl_vx_u64m1(ix, 50, vl);
    index = __riscv_vsub_vx_u64m1(index, 4015ULL, vl);

    vbool64_t mask = __riscv_vmsltu_vx_u64m1_b64(ix, 0x3ec0000000000000, vl);
    index = __riscv_vmerge_vxm_u64m1(index, 0x0000000000000000, mask, vl);
     
    // 0x1.30fc1931f09c9p+4
    mask = __riscv_vmsgtu_vx_u64m1_b64(ix, 0x40330fc1931f09c9, vl);
    vfloat64m1_t y = __riscv_vreinterpret_v_u64m1_f64m1(
                __riscv_vmerge_vxm_u64m1(ix, 0x0000000000000000, mask, vl));
    index = __riscv_vmerge_vxm_u64m1(index, 94, mask, vl);
    
    index = __riscv_vsll_vx_u64m1(index, 7, vl);
    
    
    vfloat64m1_t p0H = __riscv_vloxei64_v_f64m1(tanhdp, index, vl);
    vfloat64m1_t p0L = __riscv_vloxei64_v_f64m1(tanhdp + 1, index, vl);
    vfloat64m1_t p1 = __riscv_vloxei64_v_f64m1(tanhdp + 2, index, vl);
    vfloat64m1_t p2 = __riscv_vloxei64_v_f64m1(tanhdp + 3, index, vl);
    vfloat64m1_t p3 = __riscv_vloxei64_v_f64m1(tanhdp + 4, index, vl);
    vfloat64m1_t p4 = __riscv_vloxei64_v_f64m1(tanhdp + 5, index, vl);
    vfloat64m1_t p5 = __riscv_vloxei64_v_f64m1(tanhdp + 6, index, vl);
    vfloat64m1_t p6 = __riscv_vloxei64_v_f64m1(tanhdp + 7, index, vl);
    vfloat64m1_t p7 = __riscv_vloxei64_v_f64m1(tanhdp + 8, index, vl);
    vfloat64m1_t p8 = __riscv_vloxei64_v_f64m1(tanhdp + 9, index, vl);
    vfloat64m1_t p9 = __riscv_vloxei64_v_f64m1(tanhdp + 10, index, vl);
    vfloat64m1_t p10 = __riscv_vloxei64_v_f64m1(tanhdp + 11, index, vl);
    vfloat64m1_t p11 = __riscv_vloxei64_v_f64m1(tanhdp + 12, index, vl);
    vfloat64m1_t p12 = __riscv_vloxei64_v_f64m1(tanhdp + 13, index, vl);
    vfloat64m1_t p13 = __riscv_vloxei64_v_f64m1(tanhdp + 14, index, vl);
    vfloat64m1_t x_m = __riscv_vloxei64_v_f64m1(tanhdp + 15, index, vl);
    
    y = __riscv_vfadd_vv_f64m1(y, x_m, vl);
    
    vfloat64m1_t px = __riscv_vfmadd_vv_f64m1(y, p13, p12, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y, p11, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y, p10, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y, p9, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y, p8, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y, p7, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y, p6, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y, p5, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y, p4, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y, p3, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y, p2, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y, p1, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y, p0L, vl);
    px = __riscv_vfadd_vv_f64m1(px, p0H, vl);
    
    
    
    vuint64m1_t signx = __riscv_vand_vx_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(x), 0x8000000000000000, vl);
    px = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vor_vv_u64m1(
                __riscv_vreinterpret_v_f64m1_u64m1(px), signx, vl));
    
#ifndef __FAST_MATH__
    vbool64_t mask_sNaN = 
                __riscv_vmsgtu_vx_u64m1_b64 (ix, 0x7ff0000000000000, vl);
    px = __riscv_vmerge_vvm_f64m1(px, x, mask_sNaN, vl);
    mask_sNaN = __riscv_vmand_mm_b64(mask_sNaN,
                  __riscv_vmsltu_vx_u64m1_b64(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b64(mask_sNaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        px = __riscv_vfmerge_vfm_f64m1(px, x1, mask_sNaN, vl);
    }
#endif

    return px;
}

vfloat64m2_t __riscv_vtanh_f64m2(vfloat64m2_t x, size_t avl)
{ 
    size_t vl = __riscv_vsetvl_e64m2(avl);
    vuint64m2_t ix = __riscv_vand_vx_u64m2(
            __riscv_vreinterpret_v_f64m2_u64m2(x), 0x7fffffffffffffff, vl);
    
    vuint64m2_t index = __riscv_vsrl_vx_u64m2(ix, 50, vl);
    index = __riscv_vsub_vx_u64m2(index, 4015ULL, vl);

    vbool32_t mask = __riscv_vmsltu_vx_u64m2_b32(ix, 0x3ec0000000000000, vl);
    index = __riscv_vmerge_vxm_u64m2(index, 0x0000000000000000, mask, vl);
     
    // 0x1.30fc1931f09c9p+4
    mask = __riscv_vmsgtu_vx_u64m2_b32(ix, 0x40330fc1931f09c9, vl);
    vfloat64m2_t y = __riscv_vreinterpret_v_u64m2_f64m2(
                __riscv_vmerge_vxm_u64m2(ix, 0x0000000000000000, mask, vl));
    index = __riscv_vmerge_vxm_u64m2(index, 94, mask, vl);
    
    index = __riscv_vsll_vx_u64m2(index, 7, vl);
    
    
    vfloat64m2_t p0H = __riscv_vloxei64_v_f64m2(tanhdp, index, vl);
    vfloat64m2_t p0L = __riscv_vloxei64_v_f64m2(tanhdp + 1, index, vl);
    vfloat64m2_t p1 = __riscv_vloxei64_v_f64m2(tanhdp + 2, index, vl);
    vfloat64m2_t p2 = __riscv_vloxei64_v_f64m2(tanhdp + 3, index, vl);
    vfloat64m2_t p3 = __riscv_vloxei64_v_f64m2(tanhdp + 4, index, vl);
    vfloat64m2_t p4 = __riscv_vloxei64_v_f64m2(tanhdp + 5, index, vl);
    vfloat64m2_t p5 = __riscv_vloxei64_v_f64m2(tanhdp + 6, index, vl);
    vfloat64m2_t p6 = __riscv_vloxei64_v_f64m2(tanhdp + 7, index, vl);
    vfloat64m2_t p7 = __riscv_vloxei64_v_f64m2(tanhdp + 8, index, vl);
    vfloat64m2_t p8 = __riscv_vloxei64_v_f64m2(tanhdp + 9, index, vl);
    vfloat64m2_t p9 = __riscv_vloxei64_v_f64m2(tanhdp + 10, index, vl);
    vfloat64m2_t p10 = __riscv_vloxei64_v_f64m2(tanhdp + 11, index, vl);
    vfloat64m2_t p11 = __riscv_vloxei64_v_f64m2(tanhdp + 12, index, vl);
    vfloat64m2_t p12 = __riscv_vloxei64_v_f64m2(tanhdp + 13, index, vl);
    vfloat64m2_t p13 = __riscv_vloxei64_v_f64m2(tanhdp + 14, index, vl);
    vfloat64m2_t x_m = __riscv_vloxei64_v_f64m2(tanhdp + 15, index, vl);
    
    y = __riscv_vfadd_vv_f64m2(y, x_m, vl);
    
    vfloat64m2_t px = __riscv_vfmadd_vv_f64m2(y, p13, p12, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y, p11, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y, p10, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y, p9, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y, p8, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y, p7, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y, p6, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y, p5, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y, p4, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y, p3, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y, p2, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y, p1, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y, p0L, vl);
    px = __riscv_vfadd_vv_f64m2(px, p0H, vl);
    
    
    
    vuint64m2_t signx = __riscv_vand_vx_u64m2(
            __riscv_vreinterpret_v_f64m2_u64m2(x), 0x8000000000000000, vl);
    px = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vor_vv_u64m2(
                __riscv_vreinterpret_v_f64m2_u64m2(px), signx, vl));
    
#ifndef __FAST_MATH__
    vbool32_t mask_sNaN = 
                __riscv_vmsgtu_vx_u64m2_b32 (ix, 0x7ff0000000000000, vl);
    px = __riscv_vmerge_vvm_f64m2(px, x, mask_sNaN, vl);
    mask_sNaN = __riscv_vmand_mm_b32(mask_sNaN,
                  __riscv_vmsltu_vx_u64m2_b32(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b32(mask_sNaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        px = __riscv_vfmerge_vfm_f64m2(px, x1, mask_sNaN, vl);
    }
#endif

    return px;
}

vfloat64m4_t __riscv_vtanh_f64m4(vfloat64m4_t x, size_t avl)
{ 
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vuint64m4_t ix = __riscv_vand_vx_u64m4(
            __riscv_vreinterpret_v_f64m4_u64m4(x), 0x7fffffffffffffff, vl);
    
    vuint64m4_t index = __riscv_vsrl_vx_u64m4(ix, 50, vl);
    index = __riscv_vsub_vx_u64m4(index, 4015ULL, vl);

    vbool16_t mask = __riscv_vmsltu_vx_u64m4_b16(ix, 0x3ec0000000000000, vl);
    index = __riscv_vmerge_vxm_u64m4(index, 0x0000000000000000, mask, vl);
     
    // 0x1.30fc1931f09c9p+4
    mask = __riscv_vmsgtu_vx_u64m4_b16(ix, 0x40330fc1931f09c9, vl);
    vfloat64m4_t y = __riscv_vreinterpret_v_u64m4_f64m4(
                __riscv_vmerge_vxm_u64m4(ix, 0x0000000000000000, mask, vl));
    index = __riscv_vmerge_vxm_u64m4(index, 94, mask, vl);
    
    index = __riscv_vsll_vx_u64m4(index, 7, vl);
    
    
    vfloat64m4_t p0H = __riscv_vloxei64_v_f64m4(tanhdp, index, vl);
    vfloat64m4_t p0L = __riscv_vloxei64_v_f64m4(tanhdp + 1, index, vl);
    vfloat64m4_t p1 = __riscv_vloxei64_v_f64m4(tanhdp + 2, index, vl);
    vfloat64m4_t p2 = __riscv_vloxei64_v_f64m4(tanhdp + 3, index, vl);
    vfloat64m4_t p3 = __riscv_vloxei64_v_f64m4(tanhdp + 4, index, vl);
    vfloat64m4_t p4 = __riscv_vloxei64_v_f64m4(tanhdp + 5, index, vl);
    vfloat64m4_t p5 = __riscv_vloxei64_v_f64m4(tanhdp + 6, index, vl);
    vfloat64m4_t p6 = __riscv_vloxei64_v_f64m4(tanhdp + 7, index, vl);
    vfloat64m4_t p7 = __riscv_vloxei64_v_f64m4(tanhdp + 8, index, vl);
    vfloat64m4_t p8 = __riscv_vloxei64_v_f64m4(tanhdp + 9, index, vl);
    vfloat64m4_t p9 = __riscv_vloxei64_v_f64m4(tanhdp + 10, index, vl);
    vfloat64m4_t p10 = __riscv_vloxei64_v_f64m4(tanhdp + 11, index, vl);
    vfloat64m4_t p11 = __riscv_vloxei64_v_f64m4(tanhdp + 12, index, vl);
    vfloat64m4_t p12 = __riscv_vloxei64_v_f64m4(tanhdp + 13, index, vl);
    vfloat64m4_t p13 = __riscv_vloxei64_v_f64m4(tanhdp + 14, index, vl);
    vfloat64m4_t x_m = __riscv_vloxei64_v_f64m4(tanhdp + 15, index, vl);
    
    y = __riscv_vfadd_vv_f64m4(y, x_m, vl);
    
    vfloat64m4_t px = __riscv_vfmadd_vv_f64m4(y, p13, p12, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y, p11, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y, p10, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y, p9, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y, p8, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y, p7, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y, p6, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y, p5, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y, p4, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y, p3, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y, p2, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y, p1, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y, p0L, vl);
    px = __riscv_vfadd_vv_f64m4(px, p0H, vl);
    
    
    
    vuint64m4_t signx = __riscv_vand_vx_u64m4(
            __riscv_vreinterpret_v_f64m4_u64m4(x), 0x8000000000000000, vl);
    px = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vor_vv_u64m4(
                __riscv_vreinterpret_v_f64m4_u64m4(px), signx, vl));
    
#ifndef __FAST_MATH__
    vbool16_t mask_sNaN = 
                __riscv_vmsgtu_vx_u64m4_b16 (ix, 0x7ff0000000000000, vl);
    px = __riscv_vmerge_vvm_f64m4(px, x, mask_sNaN, vl);
    mask_sNaN = __riscv_vmand_mm_b16(mask_sNaN,
                  __riscv_vmsltu_vx_u64m4_b16(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        px = __riscv_vfmerge_vfm_f64m4(px, x1, mask_sNaN, vl);
    }
#endif

    return px;
}

vfloat64m8_t __riscv_vtanh_f64m8(vfloat64m8_t x, size_t avl)
{ 
    vfloat64m8_t res;
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1 = __riscv_vtanh_f64m4(x1, vl);
    res = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e64m4(avl-vl);
        x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1 = __riscv_vtanh_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}

vfloat32m1_t __riscv_vtanh_f32m1(vfloat32m1_t x, size_t avl)
{ 
    size_t vl = __riscv_vsetvl_e32m1(avl);
    vuint32m1_t ix = __riscv_vand_vx_u32m1(
                 __riscv_vreinterpret_v_f32m1_u32m1(x), 0x7fffffff, vl);
    
    vuint32m1_t index = __riscv_vsrl_vx_u32m1(ix, 20, vl);
    index = __riscv_vsub_vx_u32m1(index, 959, vl);

    vbool32_t mask = __riscv_vmsltu_vx_u32m1_b32(ix, 0x3c000000, vl);
    index = __riscv_vmerge_vxm_u32m1(index, 0x00000000, mask, vl);
     
    // 0x1.205966p+3f
    mask = __riscv_vmsgtu_vx_u32m1_b32(ix, 0x41102cb3, vl);
    vfloat32m1_t y = __riscv_vreinterpret_v_u32m1_f32m1(
                        __riscv_vmerge_vxm_u32m1(ix, 0x00000000, mask, vl));
    index = __riscv_vmerge_vxm_u32m1(index, 83, mask, vl);
    
    index = __riscv_vsll_vx_u32m1(index, 5, vl);
            
    vfloat32m1_t p0H = __riscv_vloxei32_v_f32m1(tanhsp, index, vl);
    vfloat32m1_t p0L = __riscv_vloxei32_v_f32m1(tanhsp + 1, index, vl);
    vfloat32m1_t p1 = __riscv_vloxei32_v_f32m1(tanhsp + 2, index, vl);
    vfloat32m1_t p2 = __riscv_vloxei32_v_f32m1(tanhsp + 3, index, vl);
    vfloat32m1_t p3 = __riscv_vloxei32_v_f32m1(tanhsp + 4, index, vl);
    vfloat32m1_t p4 = __riscv_vloxei32_v_f32m1(tanhsp + 5, index, vl);
    vfloat32m1_t p5 = __riscv_vloxei32_v_f32m1(tanhsp + 6, index, vl);
    vfloat32m1_t x_m = __riscv_vloxei32_v_f32m1(tanhsp + 7, index, vl);
    
    y = __riscv_vfadd_vv_f32m1(y, x_m, vl);
    
    vfloat32m1_t px = __riscv_vfmadd_vv_f32m1(y, p5, p4, vl);
    px = __riscv_vfmadd_vv_f32m1(px, y, p3, vl);
    px = __riscv_vfmadd_vv_f32m1(px, y, p2, vl);
    px = __riscv_vfmadd_vv_f32m1(px, y, p1, vl);
    px = __riscv_vfmadd_vv_f32m1(px, y, p0L, vl);
    px = __riscv_vfadd_vv_f32m1(px, p0H, vl);
    
    vuint32m1_t signx = __riscv_vand_vx_u32m1(
                __riscv_vreinterpret_v_f32m1_u32m1(x), 0x80000000, vl);
    px = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vor_vv_u32m1(
                __riscv_vreinterpret_v_f32m1_u32m1(px), signx, vl));

#ifndef __FAST_MATH__
    vbool32_t mask_sNaN = __riscv_vmsgtu_vx_u32m1_b32 (ix, 0x7f800000, vl);
    px = __riscv_vmerge_vvm_f32m1(px, x, mask_sNaN, vl);
    mask_sNaN = __riscv_vmand_mm_b32(mask_sNaN,
                  __riscv_vmsltu_vx_u32m1_b32(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b32(mask_sNaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        px = __riscv_vfmerge_vfm_f32m1(px, x1, mask_sNaN, vl);
    }
#endif

    return px;
}

vfloat32m2_t __riscv_vtanh_f32m2(vfloat32m2_t x, size_t avl)
{ 
    size_t vl = __riscv_vsetvl_e32m2(avl);
    vuint32m2_t ix = __riscv_vand_vx_u32m2(
                 __riscv_vreinterpret_v_f32m2_u32m2(x), 0x7fffffff, vl);
    
    vuint32m2_t index = __riscv_vsrl_vx_u32m2(ix, 20, vl);
    index = __riscv_vsub_vx_u32m2(index, 959, vl);

    vbool16_t mask = __riscv_vmsltu_vx_u32m2_b16(ix, 0x3c000000, vl);
    index = __riscv_vmerge_vxm_u32m2(index, 0x00000000, mask, vl);
     
    // 0x1.205966p+3f
    mask = __riscv_vmsgtu_vx_u32m2_b16(ix, 0x41102cb3, vl);
    vfloat32m2_t y = __riscv_vreinterpret_v_u32m2_f32m2(
                        __riscv_vmerge_vxm_u32m2(ix, 0x00000000, mask, vl));
    index = __riscv_vmerge_vxm_u32m2(index, 83, mask, vl);
    
    index = __riscv_vsll_vx_u32m2(index, 5, vl);
            
    vfloat32m2_t p0H = __riscv_vloxei32_v_f32m2(tanhsp, index, vl);
    vfloat32m2_t p0L = __riscv_vloxei32_v_f32m2(tanhsp + 1, index, vl);
    vfloat32m2_t p1 = __riscv_vloxei32_v_f32m2(tanhsp + 2, index, vl);
    vfloat32m2_t p2 = __riscv_vloxei32_v_f32m2(tanhsp + 3, index, vl);
    vfloat32m2_t p3 = __riscv_vloxei32_v_f32m2(tanhsp + 4, index, vl);
    vfloat32m2_t p4 = __riscv_vloxei32_v_f32m2(tanhsp + 5, index, vl);
    vfloat32m2_t p5 = __riscv_vloxei32_v_f32m2(tanhsp + 6, index, vl);
    vfloat32m2_t x_m = __riscv_vloxei32_v_f32m2(tanhsp + 7, index, vl);
    
    y = __riscv_vfadd_vv_f32m2(y, x_m, vl);
    
    vfloat32m2_t px = __riscv_vfmadd_vv_f32m2(y, p5, p4, vl);
    px = __riscv_vfmadd_vv_f32m2(px, y, p3, vl);
    px = __riscv_vfmadd_vv_f32m2(px, y, p2, vl);
    px = __riscv_vfmadd_vv_f32m2(px, y, p1, vl);
    px = __riscv_vfmadd_vv_f32m2(px, y, p0L, vl);
    px = __riscv_vfadd_vv_f32m2(px, p0H, vl);
    
    vuint32m2_t signx = __riscv_vand_vx_u32m2(
                __riscv_vreinterpret_v_f32m2_u32m2(x), 0x80000000, vl);
    px = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vor_vv_u32m2(
                __riscv_vreinterpret_v_f32m2_u32m2(px), signx, vl));

#ifndef __FAST_MATH__
    vbool16_t mask_sNaN = __riscv_vmsgtu_vx_u32m2_b16 (ix, 0x7f800000, vl);
    px = __riscv_vmerge_vvm_f32m2(px, x, mask_sNaN, vl);
    mask_sNaN = __riscv_vmand_mm_b16(mask_sNaN,
                  __riscv_vmsltu_vx_u32m2_b16(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        px = __riscv_vfmerge_vfm_f32m2(px, x1, mask_sNaN, vl);
    }
#endif

    return px;
}

vfloat32m4_t __riscv_vtanh_f32m4(vfloat32m4_t x, size_t avl)
{ 
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vuint32m4_t ix = __riscv_vand_vx_u32m4(
                 __riscv_vreinterpret_v_f32m4_u32m4(x), 0x7fffffff, vl);
    
    vuint32m4_t index = __riscv_vsrl_vx_u32m4(ix, 20, vl);
    index = __riscv_vsub_vx_u32m4(index, 959, vl);

    vbool8_t mask = __riscv_vmsltu_vx_u32m4_b8(ix, 0x3c000000, vl);
    index = __riscv_vmerge_vxm_u32m4(index, 0x00000000, mask, vl);
     
    // 0x1.205966p+3f
    mask = __riscv_vmsgtu_vx_u32m4_b8(ix, 0x41102cb3, vl);
    vfloat32m4_t y = __riscv_vreinterpret_v_u32m4_f32m4(
                        __riscv_vmerge_vxm_u32m4(ix, 0x00000000, mask, vl));
    index = __riscv_vmerge_vxm_u32m4(index, 83, mask, vl);
    
    index = __riscv_vsll_vx_u32m4(index, 5, vl);
            
    vfloat32m4_t p0H = __riscv_vloxei32_v_f32m4(tanhsp, index, vl);
    vfloat32m4_t p0L = __riscv_vloxei32_v_f32m4(tanhsp + 1, index, vl);
    vfloat32m4_t p1 = __riscv_vloxei32_v_f32m4(tanhsp + 2, index, vl);
    vfloat32m4_t p2 = __riscv_vloxei32_v_f32m4(tanhsp + 3, index, vl);
    vfloat32m4_t p3 = __riscv_vloxei32_v_f32m4(tanhsp + 4, index, vl);
    vfloat32m4_t p4 = __riscv_vloxei32_v_f32m4(tanhsp + 5, index, vl);
    vfloat32m4_t p5 = __riscv_vloxei32_v_f32m4(tanhsp + 6, index, vl);
    vfloat32m4_t x_m = __riscv_vloxei32_v_f32m4(tanhsp + 7, index, vl);
    
    y = __riscv_vfadd_vv_f32m4(y, x_m, vl);
    
    vfloat32m4_t px = __riscv_vfmadd_vv_f32m4(y, p5, p4, vl);
    px = __riscv_vfmadd_vv_f32m4(px, y, p3, vl);
    px = __riscv_vfmadd_vv_f32m4(px, y, p2, vl);
    px = __riscv_vfmadd_vv_f32m4(px, y, p1, vl);
    px = __riscv_vfmadd_vv_f32m4(px, y, p0L, vl);
    px = __riscv_vfadd_vv_f32m4(px, p0H, vl);
    
    vuint32m4_t signx = __riscv_vand_vx_u32m4(
                __riscv_vreinterpret_v_f32m4_u32m4(x), 0x80000000, vl);
    px = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vv_u32m4(
                __riscv_vreinterpret_v_f32m4_u32m4(px), signx, vl));

#ifndef __FAST_MATH__
    vbool8_t mask_sNaN = __riscv_vmsgtu_vx_u32m4_b8 (ix, 0x7f800000, vl);
    px = __riscv_vmerge_vvm_f32m4(px, x, mask_sNaN, vl);
    mask_sNaN = __riscv_vmand_mm_b8(mask_sNaN,
                  __riscv_vmsltu_vx_u32m4_b8(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_sNaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        px = __riscv_vfmerge_vfm_f32m4(px, x1, mask_sNaN, vl);
    }
#endif

    return px;
}

vfloat32m8_t __riscv_vtanh_f32m8(vfloat32m8_t x, size_t avl)
{  
    vfloat32m8_t res;
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1 = __riscv_vtanh_f32m4(x1, vl);
    res = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e32m4(avl-vl);
        x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1 = __riscv_vtanh_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}


#ifdef __riscv_zvfh
//static _Float16 tanhhp [88];
#include "htanh.data"

vfloat16m1_t __riscv_vtanh_f16m1(vfloat16m1_t x, size_t avl)
{ 
    size_t vl = __riscv_vsetvl_e16m1(avl);
    vuint16m1_t ix = __riscv_vand_vx_u16m1(
                 __riscv_vreinterpret_v_f16m1_u16m1(x), 0x7fff, vl);
    
    vuint16m1_t index = __riscv_vsrl_vx_u16m1(ix, 9, vl);
    index = __riscv_vsub_vx_u16m1(index, 25, vl);

    vbool16_t mask = __riscv_vmsltu_vx_u16m1_b16(ix, 0x3400, vl);
    index = __riscv_vmerge_vxm_u16m1(index, 0x0000, mask, vl);
     
    // 0x1.0a4p+2f16
    mask = __riscv_vmsgtu_vx_u16m1_b16(ix, 0x4429, vl);
    vfloat16m1_t y = __riscv_vreinterpret_v_u16m1_f16m1(
                        __riscv_vmerge_vxm_u16m1(ix, 0x0000, mask, vl));
    index = __riscv_vmerge_vxm_u16m1(index, 10, mask, vl);
    
    index = __riscv_vsll_vx_u16m1(index, 4, vl);
            
    vfloat16m1_t p0H = __riscv_vloxei16_v_f16m1(tanhhp, index, vl);
    vfloat16m1_t p0L = __riscv_vloxei16_v_f16m1(tanhhp + 1, index, vl);
    vfloat16m1_t p1 = __riscv_vloxei16_v_f16m1(tanhhp + 2, index, vl);
    vfloat16m1_t p2 = __riscv_vloxei16_v_f16m1(tanhhp + 3, index, vl);
    vfloat16m1_t p3 = __riscv_vloxei16_v_f16m1(tanhhp + 4, index, vl);
    vfloat16m1_t p4 = __riscv_vloxei16_v_f16m1(tanhhp + 5, index, vl);
    vfloat16m1_t p5 = __riscv_vloxei16_v_f16m1(tanhhp + 6, index, vl);
    vfloat16m1_t x_m = __riscv_vloxei16_v_f16m1(tanhhp + 7, index, vl);
    
    y = __riscv_vfadd_vv_f16m1(y, x_m, vl);
    
    vfloat16m1_t px = __riscv_vfmadd_vv_f16m1(y, p5, p4, vl);
    px = __riscv_vfmadd_vv_f16m1(px, y, p3, vl);
    px = __riscv_vfmadd_vv_f16m1(px, y, p2, vl);
    px = __riscv_vfmadd_vv_f16m1(px, y, p1, vl);
    px = __riscv_vfmadd_vv_f16m1(px, y, p0L, vl);
    px = __riscv_vfadd_vv_f16m1(px, p0H, vl);
    
    vuint16m1_t signx = __riscv_vand_vx_u16m1(
                __riscv_vreinterpret_v_f16m1_u16m1(x), 0x8000, vl);
    px = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vor_vv_u16m1(
                __riscv_vreinterpret_v_f16m1_u16m1(px), signx, vl));

#ifndef __FAST_MATH__
    vbool16_t mask_sNaN = __riscv_vmsgtu_vx_u16m1_b16 (ix, 0x7c00, vl);
    px = __riscv_vmerge_vvm_f16m1(px, x, mask_sNaN, vl);
    mask_sNaN = __riscv_vmand_mm_b16(mask_sNaN,
                  __riscv_vmsltu_vx_u16m1_b16(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        px = __riscv_vfmerge_vfm_f16m1(px, x1, mask_sNaN, vl);
    }
#endif

    return px;
}

vfloat16m2_t __riscv_vtanh_f16m2(vfloat16m2_t x, size_t avl)
{ 
    size_t vl = __riscv_vsetvl_e16m2(avl);
    vuint16m2_t ix = __riscv_vand_vx_u16m2(
                 __riscv_vreinterpret_v_f16m2_u16m2(x), 0x7fff, vl);
    
    vuint16m2_t index = __riscv_vsrl_vx_u16m2(ix, 9, vl);
    index = __riscv_vsub_vx_u16m2(index, 25, vl);

    vbool8_t mask = __riscv_vmsltu_vx_u16m2_b8(ix, 0x3400, vl);
    index = __riscv_vmerge_vxm_u16m2(index, 0x0000, mask, vl);
     
    // 0x1.0a4p+2f16
    mask = __riscv_vmsgtu_vx_u16m2_b8(ix, 0x4429, vl);
    vfloat16m2_t y = __riscv_vreinterpret_v_u16m2_f16m2(
                        __riscv_vmerge_vxm_u16m2(ix, 0x0000, mask, vl));
    index = __riscv_vmerge_vxm_u16m2(index, 10, mask, vl);
    
    index = __riscv_vsll_vx_u16m2(index, 4, vl);
            
    vfloat16m2_t p0H = __riscv_vloxei16_v_f16m2(tanhhp, index, vl);
    vfloat16m2_t p0L = __riscv_vloxei16_v_f16m2(tanhhp + 1, index, vl);
    vfloat16m2_t p1 = __riscv_vloxei16_v_f16m2(tanhhp + 2, index, vl);
    vfloat16m2_t p2 = __riscv_vloxei16_v_f16m2(tanhhp + 3, index, vl);
    vfloat16m2_t p3 = __riscv_vloxei16_v_f16m2(tanhhp + 4, index, vl);
    vfloat16m2_t p4 = __riscv_vloxei16_v_f16m2(tanhhp + 5, index, vl);
    vfloat16m2_t p5 = __riscv_vloxei16_v_f16m2(tanhhp + 6, index, vl);
    vfloat16m2_t x_m = __riscv_vloxei16_v_f16m2(tanhhp + 7, index, vl);
    
    y = __riscv_vfadd_vv_f16m2(y, x_m, vl);
    
    vfloat16m2_t px = __riscv_vfmadd_vv_f16m2(y, p5, p4, vl);
    px = __riscv_vfmadd_vv_f16m2(px, y, p3, vl);
    px = __riscv_vfmadd_vv_f16m2(px, y, p2, vl);
    px = __riscv_vfmadd_vv_f16m2(px, y, p1, vl);
    px = __riscv_vfmadd_vv_f16m2(px, y, p0L, vl);
    px = __riscv_vfadd_vv_f16m2(px, p0H, vl);
    
    vuint16m2_t signx = __riscv_vand_vx_u16m2(
                __riscv_vreinterpret_v_f16m2_u16m2(x), 0x8000, vl);
    px = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vor_vv_u16m2(
                __riscv_vreinterpret_v_f16m2_u16m2(px), signx, vl));

#ifndef __FAST_MATH__
    vbool8_t mask_sNaN = __riscv_vmsgtu_vx_u16m2_b8 (ix, 0x7c00, vl);
    px = __riscv_vmerge_vvm_f16m2(px, x, mask_sNaN, vl);
    mask_sNaN = __riscv_vmand_mm_b8(mask_sNaN,
                  __riscv_vmsltu_vx_u16m2_b8(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_sNaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        px = __riscv_vfmerge_vfm_f16m2(px, x1, mask_sNaN, vl);
    }
#endif

    return px;
}

vfloat16m4_t __riscv_vtanh_f16m4(vfloat16m4_t x, size_t avl)
{ 
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vuint16m4_t ix = __riscv_vand_vx_u16m4(
                 __riscv_vreinterpret_v_f16m4_u16m4(x), 0x7fff, vl);
    
    vuint16m4_t index = __riscv_vsrl_vx_u16m4(ix, 9, vl);
    index = __riscv_vsub_vx_u16m4(index, 25, vl);

    vbool4_t mask = __riscv_vmsltu_vx_u16m4_b4(ix, 0x3400, vl);
    index = __riscv_vmerge_vxm_u16m4(index, 0x0000, mask, vl);
     
    // 0x1.0a4p+2f16
    mask = __riscv_vmsgtu_vx_u16m4_b4(ix, 0x4429, vl);
    vfloat16m4_t y = __riscv_vreinterpret_v_u16m4_f16m4(
                        __riscv_vmerge_vxm_u16m4(ix, 0x0000, mask, vl));
    index = __riscv_vmerge_vxm_u16m4(index, 10, mask, vl);
    
    index = __riscv_vsll_vx_u16m4(index, 4, vl);
            
    vfloat16m4_t p0H = __riscv_vloxei16_v_f16m4(tanhhp, index, vl);
    vfloat16m4_t p0L = __riscv_vloxei16_v_f16m4(tanhhp + 1, index, vl);
    vfloat16m4_t p1 = __riscv_vloxei16_v_f16m4(tanhhp + 2, index, vl);
    vfloat16m4_t p2 = __riscv_vloxei16_v_f16m4(tanhhp + 3, index, vl);
    vfloat16m4_t p3 = __riscv_vloxei16_v_f16m4(tanhhp + 4, index, vl);
    vfloat16m4_t p4 = __riscv_vloxei16_v_f16m4(tanhhp + 5, index, vl);
    vfloat16m4_t p5 = __riscv_vloxei16_v_f16m4(tanhhp + 6, index, vl);
    vfloat16m4_t x_m = __riscv_vloxei16_v_f16m4(tanhhp + 7, index, vl);
    
    y = __riscv_vfadd_vv_f16m4(y, x_m, vl);
    
    vfloat16m4_t px = __riscv_vfmadd_vv_f16m4(y, p5, p4, vl);
    px = __riscv_vfmadd_vv_f16m4(px, y, p3, vl);
    px = __riscv_vfmadd_vv_f16m4(px, y, p2, vl);
    px = __riscv_vfmadd_vv_f16m4(px, y, p1, vl);
    px = __riscv_vfmadd_vv_f16m4(px, y, p0L, vl);
    px = __riscv_vfadd_vv_f16m4(px, p0H, vl);
    
    vuint16m4_t signx = __riscv_vand_vx_u16m4(
                __riscv_vreinterpret_v_f16m4_u16m4(x), 0x8000, vl);
    px = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vor_vv_u16m4(
                __riscv_vreinterpret_v_f16m4_u16m4(px), signx, vl));

#ifndef __FAST_MATH__
    vbool4_t mask_sNaN = __riscv_vmsgtu_vx_u16m4_b4 (ix, 0x7c00, vl);
    px = __riscv_vmerge_vvm_f16m4(px, x, mask_sNaN, vl);
    mask_sNaN = __riscv_vmand_mm_b4(mask_sNaN,
                  __riscv_vmsltu_vx_u16m4_b4(ix, 0x7e00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b4(mask_sNaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
        px = __riscv_vfmerge_vfm_f16m4(px, x1, mask_sNaN, vl);
    }
#endif

    return px;
}

vfloat16m8_t __riscv_vtanh_f16m8(vfloat16m8_t x, size_t avl)
{  
    vfloat16m8_t res;
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1 = __riscv_vtanh_f16m4(x1, vl);
    res = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e16m4(avl-vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1 = __riscv_vtanh_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}

#endif /* __riscv_zvfh */

#endif /* __riscv_v_intrinsic */

