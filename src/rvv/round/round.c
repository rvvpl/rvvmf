/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2024 The RVVMF Authors (Valentin Volokitin)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *  File:  round.c                                       *
 *  Contains: intrinsic function round for f64, f32, f16 *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Return value the nearest integer value to V,          *
 * rounding halfway cases away from zero                 *
 *                                                       *
 * Algorithm:                                            *
 *    1) Right-shifter (+num -num)                       *
 *                                                       *
 *                                                       *
 * Note that this intrinsic is less efficient than       *
 * 2x __riscv_vfcvt                                      *
 *********************************************************
*/
 
#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

vfloat64m1_t __riscv_vround_f64m1(vfloat64m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m1(avl);
    vuint64m1_t ix = __riscv_vand_vx_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(x), 0x7fffffffffffffff, vl);
    vbool64_t mask = __riscv_vmsgeu_vx_u64m1_b64(ix, 0x4330000000000000, vl);
    
    vfloat64m1_t maskedx = __riscv_vfmerge_vfm_f64m1(
                __riscv_vreinterpret_v_u64m1_f64m1(ix), 0.0, mask, vl);
    vfloat64m1_t mx = maskedx;
    maskedx = __riscv_vfadd_vf_f64m1(maskedx, 0x1p52, vl);
    maskedx = __riscv_vfsub_vf_f64m1(maskedx, 0x1p52, vl);

    vbool64_t mask2 = __riscv_vmsltu_vv_u64m1_b64 (ix, 
                __riscv_vreinterpret_v_f64m1_u64m1(maskedx), vl);
    maskedx = __riscv_vmerge_vvm_f64m1(maskedx, 
                __riscv_vfsub_vf_f64m1(maskedx, 1.0, vl), mask2, vl);;
   
    mask2 = __riscv_vmfgt_vf_f64m1_b64(
                __riscv_vfsub_vv_f64m1(mx, maskedx, vl), 
                0x1.fffffffffffffp-2, vl);
    maskedx = __riscv_vmerge_vvm_f64m1(maskedx, 
                __riscv_vfadd_vf_f64m1(maskedx, 1.0, vl), mask2, vl);
    
    vuint64m1_t signx = __riscv_vand_vx_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(x), 0x8000000000000000, vl);
    maskedx = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vor_vv_u64m1(
                __riscv_vreinterpret_v_f64m1_u64m1(maskedx), signx, vl));
                
#ifndef __FAST_MATH__
    vbool64_t mask_sNaN = __riscv_vmsltu_vx_u64m1_b64 (ix, 
                                    0x7ff8000000000000, vl);
    mask_sNaN = __riscv_vmand_mm_b64(mask_sNaN,
                  __riscv_vmsgtu_vx_u64m1_b64(ix, 0x7ff0000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b64(mask_sNaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
    }
#endif

    return __riscv_vmerge_vvm_f64m1(maskedx, x, mask, vl);
}

vfloat64m2_t __riscv_vround_f64m2(vfloat64m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m2(avl);
    vuint64m2_t ix = __riscv_vand_vx_u64m2(
            __riscv_vreinterpret_v_f64m2_u64m2(x), 0x7fffffffffffffff, vl);
    vbool32_t mask = __riscv_vmsgeu_vx_u64m2_b32(ix, 0x4330000000000000, vl);
    
    vfloat64m2_t maskedx = __riscv_vfmerge_vfm_f64m2(
                __riscv_vreinterpret_v_u64m2_f64m2(ix), 0.0, mask, vl);
    vfloat64m2_t mx = maskedx;
    maskedx = __riscv_vfadd_vf_f64m2(maskedx, 0x1p52, vl);
    maskedx = __riscv_vfsub_vf_f64m2(maskedx, 0x1p52, vl);

    vbool32_t mask2 = __riscv_vmsltu_vv_u64m2_b32 (ix, 
                __riscv_vreinterpret_v_f64m2_u64m2(maskedx), vl);
    maskedx = __riscv_vmerge_vvm_f64m2(maskedx, 
                __riscv_vfsub_vf_f64m2(maskedx, 1.0, vl), mask2, vl);;
   
    mask2 = __riscv_vmfgt_vf_f64m2_b32(
                __riscv_vfsub_vv_f64m2(mx, maskedx, vl), 
                0x1.fffffffffffffp-2, vl);
    maskedx = __riscv_vmerge_vvm_f64m2(maskedx, 
                __riscv_vfadd_vf_f64m2(maskedx, 1.0, vl), mask2, vl);
    
    vuint64m2_t signx = __riscv_vand_vx_u64m2(
            __riscv_vreinterpret_v_f64m2_u64m2(x), 0x8000000000000000, vl);
    maskedx = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vor_vv_u64m2(
                __riscv_vreinterpret_v_f64m2_u64m2(maskedx), signx, vl));
                
#ifndef __FAST_MATH__
    vbool32_t mask_sNaN = __riscv_vmsltu_vx_u64m2_b32 (ix, 
                                    0x7ff8000000000000, vl);
    mask_sNaN = __riscv_vmand_mm_b32(mask_sNaN,
                  __riscv_vmsgtu_vx_u64m2_b32(ix, 0x7ff0000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b32(mask_sNaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
    }
#endif

    return __riscv_vmerge_vvm_f64m2(maskedx, x, mask, vl);
}

vfloat64m4_t __riscv_vround_f64m4(vfloat64m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vuint64m4_t ix = __riscv_vand_vx_u64m4(
            __riscv_vreinterpret_v_f64m4_u64m4(x), 0x7fffffffffffffff, vl);
    vbool16_t mask = __riscv_vmsgeu_vx_u64m4_b16(ix, 0x4330000000000000, vl);
    
    vfloat64m4_t maskedx = __riscv_vfmerge_vfm_f64m4(
                __riscv_vreinterpret_v_u64m4_f64m4(ix), 0.0, mask, vl);
    vfloat64m4_t mx = maskedx;
    maskedx = __riscv_vfadd_vf_f64m4(maskedx, 0x1p52, vl);
    maskedx = __riscv_vfsub_vf_f64m4(maskedx, 0x1p52, vl);

    vbool16_t mask2 = __riscv_vmsltu_vv_u64m4_b16 (ix, 
                __riscv_vreinterpret_v_f64m4_u64m4(maskedx), vl);
    maskedx = __riscv_vmerge_vvm_f64m4(maskedx, 
                __riscv_vfsub_vf_f64m4(maskedx, 1.0, vl), mask2, vl);;
   
    mask2 = __riscv_vmfgt_vf_f64m4_b16(
                __riscv_vfsub_vv_f64m4(mx, maskedx, vl), 
                0x1.fffffffffffffp-2, vl);
    maskedx = __riscv_vmerge_vvm_f64m4(maskedx, 
                __riscv_vfadd_vf_f64m4(maskedx, 1.0, vl), mask2, vl);
    
    vuint64m4_t signx = __riscv_vand_vx_u64m4(
            __riscv_vreinterpret_v_f64m4_u64m4(x), 0x8000000000000000, vl);
    maskedx = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vor_vv_u64m4(
                __riscv_vreinterpret_v_f64m4_u64m4(maskedx), signx, vl));
                
#ifndef __FAST_MATH__
    vbool16_t mask_sNaN = __riscv_vmsltu_vx_u64m4_b16 (ix, 
                                    0x7ff8000000000000, vl);
    mask_sNaN = __riscv_vmand_mm_b16(mask_sNaN,
                  __riscv_vmsgtu_vx_u64m4_b16(ix, 0x7ff0000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
    }
#endif

    return __riscv_vmerge_vvm_f64m4(maskedx, x, mask, vl);
}

vfloat64m8_t __riscv_vround_f64m8(vfloat64m8_t x, size_t avl)
{
    vfloat64m8_t res;
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1 = __riscv_vround_f64m4(x1, vl);
    res = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e64m4(avl-vl);
        x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1 = __riscv_vround_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}

vfloat32m1_t __riscv_vround_f32m1(vfloat32m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m1(avl);
    vuint32m1_t ix = __riscv_vand_vx_u32m1(
                 __riscv_vreinterpret_v_f32m1_u32m1(x), 0x7fffffff, vl);
    vbool32_t mask = __riscv_vmsgeu_vx_u32m1_b32(ix, 0x4b000000, vl);
    
    vfloat32m1_t maskedx = __riscv_vfmerge_vfm_f32m1(
                __riscv_vreinterpret_v_u32m1_f32m1(ix), 0.0f, mask, vl);
    vfloat32m1_t mx = maskedx;
    maskedx = __riscv_vfadd_vf_f32m1(maskedx, 0x1p23f, vl);
    maskedx = __riscv_vfsub_vf_f32m1(maskedx, 0x1p23f, vl);

    vbool32_t mask2 = __riscv_vmsltu_vv_u32m1_b32 (ix, 
                __riscv_vreinterpret_v_f32m1_u32m1(maskedx), vl);
    maskedx = __riscv_vmerge_vvm_f32m1(maskedx, 
                __riscv_vfsub_vf_f32m1(maskedx, 1.0f, vl), mask2, vl);
    
    mask2 = __riscv_vmfgt_vf_f32m1_b32(
                __riscv_vfsub_vv_f32m1(mx, maskedx, vl), 0x1.fffffep-2f, vl);
    maskedx = __riscv_vmerge_vvm_f32m1(maskedx,
                __riscv_vfadd_vf_f32m1(maskedx, 1.0f, vl), mask2, vl);
    
    vuint32m1_t signx = __riscv_vand_vx_u32m1(
                __riscv_vreinterpret_v_f32m1_u32m1(x), 0x80000000, vl);
    maskedx = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vor_vv_u32m1(
                __riscv_vreinterpret_v_f32m1_u32m1(maskedx), signx, vl));
                
#ifndef __FAST_MATH__
    vbool32_t mask_sNaN = __riscv_vmsltu_vx_u32m1_b32 (ix, 0x7fc00000, vl);
    mask_sNaN = __riscv_vmand_mm_b32(mask_sNaN,
                  __riscv_vmsgtu_vx_u32m1_b32(ix, 0x7f800000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b32(mask_sNaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
    }
#endif

    return __riscv_vmerge_vvm_f32m1(maskedx, x, mask, vl);
}

vfloat32m2_t __riscv_vround_f32m2(vfloat32m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m2(avl);
    vuint32m2_t ix = __riscv_vand_vx_u32m2(
                 __riscv_vreinterpret_v_f32m2_u32m2(x), 0x7fffffff, vl);
    vbool16_t mask = __riscv_vmsgeu_vx_u32m2_b16(ix, 0x4b000000, vl);
    
    vfloat32m2_t maskedx = __riscv_vfmerge_vfm_f32m2(
                __riscv_vreinterpret_v_u32m2_f32m2(ix), 0.0f, mask, vl);
    vfloat32m2_t mx = maskedx;
    maskedx = __riscv_vfadd_vf_f32m2(maskedx, 0x1p23f, vl);
    maskedx = __riscv_vfsub_vf_f32m2(maskedx, 0x1p23f, vl);

    vbool16_t mask2 = __riscv_vmsltu_vv_u32m2_b16 (ix, 
                __riscv_vreinterpret_v_f32m2_u32m2(maskedx), vl);
    maskedx = __riscv_vmerge_vvm_f32m2(maskedx, 
                __riscv_vfsub_vf_f32m2(maskedx, 1.0f, vl), mask2, vl);
    
    mask2 = __riscv_vmfgt_vf_f32m2_b16(
                __riscv_vfsub_vv_f32m2(mx, maskedx, vl), 0x1.fffffep-2f, vl);
    maskedx = __riscv_vmerge_vvm_f32m2(maskedx,
                __riscv_vfadd_vf_f32m2(maskedx, 1.0f, vl), mask2, vl);
    
    vuint32m2_t signx = __riscv_vand_vx_u32m2(
                __riscv_vreinterpret_v_f32m2_u32m2(x), 0x80000000, vl);
    maskedx = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vor_vv_u32m2(
                __riscv_vreinterpret_v_f32m2_u32m2(maskedx), signx, vl));
                
#ifndef __FAST_MATH__
    vbool16_t mask_sNaN = __riscv_vmsltu_vx_u32m2_b16 (ix, 0x7fc00000, vl);
    mask_sNaN = __riscv_vmand_mm_b16(mask_sNaN,
                  __riscv_vmsgtu_vx_u32m2_b16(ix, 0x7f800000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
    }
#endif

    return __riscv_vmerge_vvm_f32m2(maskedx, x, mask, vl);
}

vfloat32m4_t __riscv_vround_f32m4(vfloat32m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vuint32m4_t ix = __riscv_vand_vx_u32m4(
                 __riscv_vreinterpret_v_f32m4_u32m4(x), 0x7fffffff, vl);
    vbool8_t mask = __riscv_vmsgeu_vx_u32m4_b8(ix, 0x4b000000, vl);
    
    vfloat32m4_t maskedx = __riscv_vfmerge_vfm_f32m4(
                __riscv_vreinterpret_v_u32m4_f32m4(ix), 0.0f, mask, vl);
    vfloat32m4_t mx = maskedx;
    maskedx = __riscv_vfadd_vf_f32m4(maskedx, 0x1p23f, vl);
    maskedx = __riscv_vfsub_vf_f32m4(maskedx, 0x1p23f, vl);

    vbool8_t mask2 = __riscv_vmsltu_vv_u32m4_b8 (ix, 
                __riscv_vreinterpret_v_f32m4_u32m4(maskedx), vl);
    maskedx = __riscv_vmerge_vvm_f32m4(maskedx, 
                __riscv_vfsub_vf_f32m4(maskedx, 1.0f, vl), mask2, vl);
    
    mask2 = __riscv_vmfgt_vf_f32m4_b8(
                __riscv_vfsub_vv_f32m4(mx, maskedx, vl), 0x1.fffffep-2f, vl);
    maskedx = __riscv_vmerge_vvm_f32m4(maskedx,
                __riscv_vfadd_vf_f32m4(maskedx, 1.0f, vl), mask2, vl);
    
    vuint32m4_t signx = __riscv_vand_vx_u32m4(
                __riscv_vreinterpret_v_f32m4_u32m4(x), 0x80000000, vl);
    maskedx = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vv_u32m4(
                __riscv_vreinterpret_v_f32m4_u32m4(maskedx), signx, vl));
                
#ifndef __FAST_MATH__
    vbool8_t mask_sNaN = __riscv_vmsltu_vx_u32m4_b8 (ix, 0x7fc00000, vl);
    mask_sNaN = __riscv_vmand_mm_b8(mask_sNaN,
                  __riscv_vmsgtu_vx_u32m4_b8(ix, 0x7f800000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_sNaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
    }
#endif

    return __riscv_vmerge_vvm_f32m4(maskedx, x, mask, vl);
}

vfloat32m8_t __riscv_vround_f32m8(vfloat32m8_t x, size_t avl)
{
    vfloat32m8_t res;
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1 = __riscv_vround_f32m4(x1, vl);
    res = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e32m4(avl-vl);
        x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1 = __riscv_vround_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}


#if (defined(__riscv_zvfh) || defined(__riscv_zvfhmin))

vfloat16m1_t __riscv_vround_f16m1(vfloat16m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m1(avl);
    vuint16m1_t ix = __riscv_vand_vx_u16m1(
                 __riscv_vreinterpret_v_f16m1_u16m1(x), 0x7fff, vl);
    vbool16_t mask = __riscv_vmsgeu_vx_u16m1_b16(ix, 0x6400, vl);
    
    vfloat16m1_t maskedx = __riscv_vfmerge_vfm_f16m1(
                __riscv_vreinterpret_v_u16m1_f16m1(ix), 0.0f16, mask, vl);
    vfloat16m1_t mx = maskedx;
    maskedx = __riscv_vfadd_vf_f16m1(maskedx, 0x1p10f16, vl);
    maskedx = __riscv_vfsub_vf_f16m1(maskedx, 0x1p10f16, vl);

    vbool16_t mask2 = __riscv_vmsltu_vv_u16m1_b16 (ix, 
                __riscv_vreinterpret_v_f16m1_u16m1(maskedx), vl);
    maskedx = __riscv_vmerge_vvm_f16m1(maskedx, 
                __riscv_vfsub_vf_f16m1(maskedx, 1.0f16, vl), mask2, vl);
    
    mask2 = __riscv_vmfgt_vf_f16m1_b16(
                __riscv_vfsub_vv_f16m1(mx, maskedx, vl), 0x1.ffcp-2f16, vl);
    maskedx = __riscv_vmerge_vvm_f16m1(maskedx,
                __riscv_vfadd_vf_f16m1(maskedx, 1.0f16, vl), mask2, vl);
    
    vuint16m1_t signx = __riscv_vand_vx_u16m1(
                __riscv_vreinterpret_v_f16m1_u16m1(x), 0x8000, vl);
    maskedx = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vor_vv_u16m1(
                __riscv_vreinterpret_v_f16m1_u16m1(maskedx), signx, vl));
                
#ifndef __FAST_MATH__
    vbool16_t mask_sNaN = __riscv_vmsltu_vx_u16m1_b16 (ix, 0x7e00, vl);
    mask_sNaN = __riscv_vmand_mm_b16(mask_sNaN,
                  __riscv_vmsgtu_vx_u16m1_b16(ix, 0x7c00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_sNaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
    }
#endif

    return __riscv_vmerge_vvm_f16m1(maskedx, x, mask, vl);
}

vfloat16m2_t __riscv_vround_f16m2(vfloat16m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m2(avl);
    vuint16m2_t ix = __riscv_vand_vx_u16m2(
                 __riscv_vreinterpret_v_f16m2_u16m2(x), 0x7fff, vl);
    vbool8_t mask = __riscv_vmsgeu_vx_u16m2_b8(ix, 0x6400, vl);
    
    vfloat16m2_t maskedx = __riscv_vfmerge_vfm_f16m2(
                __riscv_vreinterpret_v_u16m2_f16m2(ix), 0.0f16, mask, vl);
    vfloat16m2_t mx = maskedx;
    maskedx = __riscv_vfadd_vf_f16m2(maskedx, 0x1p10f16, vl);
    maskedx = __riscv_vfsub_vf_f16m2(maskedx, 0x1p10f16, vl);

    vbool8_t mask2 = __riscv_vmsltu_vv_u16m2_b8 (ix, 
                __riscv_vreinterpret_v_f16m2_u16m2(maskedx), vl);
    maskedx = __riscv_vmerge_vvm_f16m2(maskedx, 
                __riscv_vfsub_vf_f16m2(maskedx, 1.0f16, vl), mask2, vl);
    
    mask2 = __riscv_vmfgt_vf_f16m2_b8(
                __riscv_vfsub_vv_f16m2(mx, maskedx, vl), 0x1.ffcp-2f16, vl);
    maskedx = __riscv_vmerge_vvm_f16m2(maskedx,
                __riscv_vfadd_vf_f16m2(maskedx, 1.0f16, vl), mask2, vl);
    
    vuint16m2_t signx = __riscv_vand_vx_u16m2(
                __riscv_vreinterpret_v_f16m2_u16m2(x), 0x8000, vl);
    maskedx = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vor_vv_u16m2(
                __riscv_vreinterpret_v_f16m2_u16m2(maskedx), signx, vl));
                
#ifndef __FAST_MATH__
    vbool8_t mask_sNaN = __riscv_vmsltu_vx_u16m2_b8 (ix, 0x7e00, vl);
    mask_sNaN = __riscv_vmand_mm_b8(mask_sNaN,
                  __riscv_vmsgtu_vx_u16m2_b8(ix, 0x7c00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_sNaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
    }
#endif

    return __riscv_vmerge_vvm_f16m2(maskedx, x, mask, vl);
}

vfloat16m4_t __riscv_vround_f16m4(vfloat16m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vuint16m4_t ix = __riscv_vand_vx_u16m4(
                 __riscv_vreinterpret_v_f16m4_u16m4(x), 0x7fff, vl);
    vbool4_t mask = __riscv_vmsgeu_vx_u16m4_b4(ix, 0x6400, vl);
    
    vfloat16m4_t maskedx = __riscv_vfmerge_vfm_f16m4(
                __riscv_vreinterpret_v_u16m4_f16m4(ix), 0.0f16, mask, vl);
    vfloat16m4_t mx = maskedx;
    maskedx = __riscv_vfadd_vf_f16m4(maskedx, 0x1p10f16, vl);
    maskedx = __riscv_vfsub_vf_f16m4(maskedx, 0x1p10f16, vl);

    vbool4_t mask2 = __riscv_vmsltu_vv_u16m4_b4 (ix, 
                __riscv_vreinterpret_v_f16m4_u16m4(maskedx), vl);
    maskedx = __riscv_vmerge_vvm_f16m4(maskedx, 
                __riscv_vfsub_vf_f16m4(maskedx, 1.0f16, vl), mask2, vl);
    
    mask2 = __riscv_vmfgt_vf_f16m4_b4(
                __riscv_vfsub_vv_f16m4(mx, maskedx, vl), 0x1.ffcp-2f16, vl);
    maskedx = __riscv_vmerge_vvm_f16m4(maskedx,
                __riscv_vfadd_vf_f16m4(maskedx, 1.0f16, vl), mask2, vl);
    
    vuint16m4_t signx = __riscv_vand_vx_u16m4(
                __riscv_vreinterpret_v_f16m4_u16m4(x), 0x8000, vl);
    maskedx = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vor_vv_u16m4(
                __riscv_vreinterpret_v_f16m4_u16m4(maskedx), signx, vl));
                
#ifndef __FAST_MATH__
    vbool4_t mask_sNaN = __riscv_vmsltu_vx_u16m4_b4 (ix, 0x7e00, vl);
    mask_sNaN = __riscv_vmand_mm_b4(mask_sNaN,
                  __riscv_vmsgtu_vx_u16m4_b4(ix, 0x7c00, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b4(mask_sNaN, vl);
    if (issNaN) {
        volatile _Float16 x1 = 0.0f16/0.0f16;
    }
#endif

    return __riscv_vmerge_vvm_f16m4(maskedx, x, mask, vl);
}
   
vfloat16m8_t __riscv_vround_f16m8(vfloat16m8_t x, size_t avl)
{
    vfloat16m8_t res;
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1 = __riscv_vround_f16m4(x1, vl);
    res = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e16m4(avl-vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1 = __riscv_vround_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}

#endif /* __riscv_zvfh || __riscv_zvfhmin */

#endif /* __riscv_v_intrinsic */