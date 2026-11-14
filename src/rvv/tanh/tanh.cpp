/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025-2026 The RVVMF Authors 
 *                     (Valentin Volokitin)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  tanh.cpp                                     *
 *   Contains: intrinsic function tanh for f64, f32, f16 *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the hyperbolic tangent of input vector V     *
 *                                                       *
 * Algorithm:                                            *
 *    1) Argument reduction for f32&f16:               *
 *                       tx + ty                         *
 *       tanh(x + y) = ------------                      *
 *                     1 + tx * ty                       *
 *      Argument reduction for f64:                     *
 *                         tx*(1 + ty * tz) + (ty + tz)  *
 *       tanh(x + y + z) = ----------------------------  *
 *                        (1 + ty * tz) + tx * (ty + tz) *
 *       where tx - tanh(x), ty - tanh(y), tz - tanh(z)  * 
 *       tanh(y) and tanh(z) - table values              *
 *    2) Polynomial degrees for tanh(x):                 *
 *       f64 - 13([0, 0.125]), f32 - 9([0, 0.5]),        *
 *       f16 - 3([0, 0.25])                              *
 *                                                       *
 *                                                       *
 *********************************************************
*/
 
#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"


//static double tanhyf64_hi [21];
//static double tanhyf64_lo [21];
//static double tanhzf64_hi [8];
//static double tanhzf64_lo [8];
#include "tanhf64.data"
//static float tanhyf32_hi [20];
//static float tanhyf32_lo [20];
#include "tanhf32.data"

vfloat64m1_t __riscv_vtanh_f64m1(vfloat64m1_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m1(avl);
    vuint64m1_t ix = __riscv_vand_vx_u64m1(
                 __riscv_vreinterpret_v_f64m1_u64m1(x), 0x7fffffffffffffff, vl);
    
    vfloat64m1_t ax = __riscv_vfmul_vf_f64m1(__riscv_vreinterpret_v_u64m1_f64m1(ix), 8.0, vl); 
    
    vuint64m1_t index = __riscv_vfcvt_rtz_xu_f_v_u64m1(ax, vl);
    
    vbool64_t mask = __riscv_vmsltu_vx_u64m1_b64(ix, 0x3fc0000000000000, vl);
    index = __riscv_vmerge_vxm_u64m1(index, 0, mask, vl);
     
    
    mask = __riscv_vmsgtu_vx_u64m1_b64(ix, 0x4034000000000000, vl);
    vfloat64m1_t y = __riscv_vfmerge_vfm_f64m1(__riscv_vreinterpret_v_u64m1_f64m1(ix), 20.0, mask, vl);
    
    index = __riscv_vmerge_vxm_u64m1(index, 160, mask, vl);
    y = __riscv_vfmacc_vf_f64m1(y, -0.125, __riscv_vfcvt_f_xu_v_f64m1(index, vl), vl);
    
    vuint64m1_t index_y = __riscv_vsrl_vx_u64m1(index, 3, vl);
    vuint64m1_t index_z = __riscv_vand_vx_u64m1(index, 0x7, vl);
    
    
    index_y = __riscv_vsll_vx_u64m1(index_y, 3, vl);
    index_z = __riscv_vsll_vx_u64m1(index_z, 3, vl);
            
    vfloat64m1_t tanhy_hi = __riscv_vloxei64_v_f64m1(tanhyf64_hi, index_y, vl);
    vfloat64m1_t tanhy_lo = __riscv_vloxei64_v_f64m1(tanhyf64_lo, index_y, vl);
    vfloat64m1_t tanhz_hi = __riscv_vloxei64_v_f64m1(tanhzf64_hi, index_z, vl);
    vfloat64m1_t tanhz_lo = __riscv_vloxei64_v_f64m1(tanhzf64_lo, index_z, vl);
    
    vfloat64m1_t tanhhi = __riscv_vfadd_vv_f64m1(tanhy_hi, tanhz_hi, vl);
    vfloat64m1_t tanhlo = __riscv_vfsub_vv_f64m1(tanhhi, tanhy_hi, vl);
    tanhlo = __riscv_vfsub_vv_f64m1(tanhz_hi, tanhlo, vl);
    tanhlo = __riscv_vfadd_vv_f64m1(tanhy_lo, tanhlo, vl);
    tanhlo = __riscv_vfadd_vv_f64m1(tanhz_lo, tanhlo, vl);
    
    vfloat64m1_t tanh1hi = __riscv_vfmadd_vv_f64m1(tanhy_hi, tanhz_hi, __riscv_vfmv_v_f_f64m1(1.0, vl), vl);
    vfloat64m1_t tanh1lo = __riscv_vfmadd_vv_f64m1(tanhy_hi, tanhz_hi, __riscv_vfrsub_vf_f64m1(tanh1hi, 1.0, vl), vl);
    tanh1lo = __riscv_vfmacc_vv_f64m1(tanh1lo, tanhy_hi, tanhz_lo, vl);
    tanh1lo = __riscv_vfmacc_vv_f64m1(tanh1lo, tanhy_lo, tanhz_hi, vl);
    
    vfloat64m1_t tl, th; 
    vfloat64m1_t y2h;
    y2h = __riscv_vfmadd_vv_f64m1(y, y, __riscv_vfmv_v_f_f64m1(0x1p-150, vl), vl);
       
    
    vfloat64m1_t pxl, px = __riscv_vfmv_v_f_f64m1(-0x1.22601d326d603p-7, vl);
    px = __riscv_vfmacc_vf_f64m1(px, 0x1.cc88061274d34p-9, y2h, vl);
    px = __riscv_vfmadd_vv_f64m1(px, y2h, __riscv_vfmv_v_f_f64m1(0x1.664f355520d9ap-6, vl), vl);
    px = __riscv_vfmadd_vv_f64m1(px, y2h, __riscv_vfmv_v_f_f64m1(-0x1.ba1ba1acb31ebp-5, vl), vl);
    px = __riscv_vfmadd_vv_f64m1(px, y2h, __riscv_vfmv_v_f_f64m1(0x1.111111110ed57p-3, vl), vl);
    px = __riscv_vfmadd_vv_f64m1(px, y2h, __riscv_vfmv_v_f_f64m1(-0x1.5555555555551p-2, vl), vl);
    th = __riscv_vfmadd_vv_f64m1(px, y2h, __riscv_vfmv_v_f_f64m1(1.0, vl), vl);
    tl = __riscv_vfmsub_vv_f64m1(px, y2h, __riscv_vfsub_vf_f64m1(th, 1.0, vl), vl);
    
    tl = __riscv_vfmul_vv_f64m1(y, tl, vl);
    px = __riscv_vfmadd_vv_f64m1(th, y, tl, vl);
    pxl = __riscv_vfsub_vv_f64m1(tl, __riscv_vfnmsub_vv_f64m1(th, y, px, vl), vl);
    
    vfloat64m1_t vnumh, vnuml, vdenh, vdenl, vden;
  
    th = __riscv_vfmadd_vv_f64m1(px, tanh1hi, tanhlo, vl);
    tl = __riscv_vfmadd_vv_f64m1(px, tanh1hi, __riscv_vfsub_vv_f64m1 (tanhlo, th, vl), vl);
    tl = __riscv_vfmacc_vv_f64m1(tl, pxl, tanh1hi, vl);
    tl = __riscv_vfmacc_vv_f64m1(tl, px, tanh1lo, vl);
    
    vnumh = __riscv_vfadd_vv_f64m1(tanhhi, th, vl);
    vnuml = __riscv_vfsub_vv_f64m1(th, __riscv_vfsub_vv_f64m1(vnumh, tanhhi, vl), vl);
    vnuml = __riscv_vfadd_vv_f64m1(vnuml, tl, vl);
    
    th = __riscv_vfmadd_vv_f64m1(px, tanhhi, tanh1lo, vl);
    tl = __riscv_vfmadd_vv_f64m1(px, tanhhi, __riscv_vfsub_vv_f64m1 (tanh1lo, th, vl), vl);
    tl = __riscv_vfmacc_vv_f64m1(tl, pxl, tanhhi, vl);
    tl = __riscv_vfmacc_vv_f64m1(tl, px, tanhlo, vl);
    
    vdenh = __riscv_vfadd_vv_f64m1(tanh1hi, th, vl);
    vdenl = __riscv_vfsub_vv_f64m1(th, __riscv_vfsub_vv_f64m1(vdenh, tanh1hi, vl), vl);
    vdenl = __riscv_vfadd_vv_f64m1(vdenl, tl, vl);
    
    vden = __riscv_vfrdiv_vf_f64m1(vdenh, 1.0, vl);
    th = __riscv_vfmul_vv_f64m1(vnumh, vden, vl);
    px = __riscv_vfnmsac_vv_f64m1(vnumh, th, vdenh, vl);
    tl = __riscv_vfnmsac_vv_f64m1(vnuml, th, vdenl, vl);
    
    px = __riscv_vfadd_vv_f64m1(px, tl, vl);
    px = __riscv_vfmadd_vv_f64m1(px, vden, th, vl);
  
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

vfloat64m2_t __riscv_vtanh_f64m2(vfloat64m2_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m2(avl);
    vuint64m2_t ix = __riscv_vand_vx_u64m2(
                 __riscv_vreinterpret_v_f64m2_u64m2(x), 0x7fffffffffffffff, vl);
    
    vfloat64m2_t ax = __riscv_vfmul_vf_f64m2(__riscv_vreinterpret_v_u64m2_f64m2(ix), 8.0, vl); 
    
    vuint64m2_t index = __riscv_vfcvt_rtz_xu_f_v_u64m2(ax, vl);
    
    vbool32_t mask = __riscv_vmsltu_vx_u64m2_b32(ix, 0x3fc0000000000000, vl);
    index = __riscv_vmerge_vxm_u64m2(index, 0, mask, vl);
     
    
    mask = __riscv_vmsgtu_vx_u64m2_b32(ix, 0x4034000000000000, vl);
    vfloat64m2_t y = __riscv_vfmerge_vfm_f64m2(__riscv_vreinterpret_v_u64m2_f64m2(ix), 20.0, mask, vl);
    
    index = __riscv_vmerge_vxm_u64m2(index, 160, mask, vl);
    y = __riscv_vfmacc_vf_f64m2(y, -0.125, __riscv_vfcvt_f_xu_v_f64m2(index, vl), vl);
    
    vuint64m2_t index_y = __riscv_vsrl_vx_u64m2(index, 3, vl);
    vuint64m2_t index_z = __riscv_vand_vx_u64m2(index, 0x7, vl);
    
    
    index_y = __riscv_vsll_vx_u64m2(index_y, 3, vl);
    index_z = __riscv_vsll_vx_u64m2(index_z, 3, vl);
            
    vfloat64m2_t tanhy_hi = __riscv_vloxei64_v_f64m2(tanhyf64_hi, index_y, vl);
    vfloat64m2_t tanhy_lo = __riscv_vloxei64_v_f64m2(tanhyf64_lo, index_y, vl);
    vfloat64m2_t tanhz_hi = __riscv_vloxei64_v_f64m2(tanhzf64_hi, index_z, vl);
    vfloat64m2_t tanhz_lo = __riscv_vloxei64_v_f64m2(tanhzf64_lo, index_z, vl);
    
    vfloat64m2_t tanhhi = __riscv_vfadd_vv_f64m2(tanhy_hi, tanhz_hi, vl);
    vfloat64m2_t tanhlo = __riscv_vfsub_vv_f64m2(tanhhi, tanhy_hi, vl);
    tanhlo = __riscv_vfsub_vv_f64m2(tanhz_hi, tanhlo, vl);
    tanhlo = __riscv_vfadd_vv_f64m2(tanhy_lo, tanhlo, vl);
    tanhlo = __riscv_vfadd_vv_f64m2(tanhz_lo, tanhlo, vl);
    
    vfloat64m2_t tanh1hi = __riscv_vfmadd_vv_f64m2(tanhy_hi, tanhz_hi, __riscv_vfmv_v_f_f64m2(1.0, vl), vl);
    vfloat64m2_t tanh1lo = __riscv_vfmadd_vv_f64m2(tanhy_hi, tanhz_hi, __riscv_vfrsub_vf_f64m2(tanh1hi, 1.0, vl), vl);
    tanh1lo = __riscv_vfmacc_vv_f64m2(tanh1lo, tanhy_hi, tanhz_lo, vl);
    tanh1lo = __riscv_vfmacc_vv_f64m2(tanh1lo, tanhy_lo, tanhz_hi, vl);
    
    vfloat64m2_t tl, th; 
    vfloat64m2_t y2h, y2l;
    y2h = __riscv_vfmadd_vv_f64m2(y, y, __riscv_vfmv_v_f_f64m2(0x1p-150, vl), vl);
    
       
    
    vfloat64m2_t pxl, px = __riscv_vfmv_v_f_f64m2(-0x1.22601d326d603p-7, vl);
    px = __riscv_vfmacc_vf_f64m2(px, 0x1.cc88061274d34p-9, y2h, vl);
    px = __riscv_vfmadd_vv_f64m2(px, y2h, __riscv_vfmv_v_f_f64m2(0x1.664f355520d9ap-6, vl), vl);
    px = __riscv_vfmadd_vv_f64m2(px, y2h, __riscv_vfmv_v_f_f64m2(-0x1.ba1ba1acb31ebp-5, vl), vl);
    px = __riscv_vfmadd_vv_f64m2(px, y2h, __riscv_vfmv_v_f_f64m2(0x1.111111110ed57p-3, vl), vl);
    px = __riscv_vfmadd_vv_f64m2(px, y2h, __riscv_vfmv_v_f_f64m2(-0x1.5555555555551p-2, vl), vl);
    th = __riscv_vfmadd_vv_f64m2(px, y2h, __riscv_vfmv_v_f_f64m2(1.0, vl), vl);
    tl = __riscv_vfmsub_vv_f64m2(px, y2h, __riscv_vfsub_vf_f64m2(th, 1.0, vl), vl);
    
    tl = __riscv_vfmul_vv_f64m2(y, tl, vl);
    px = __riscv_vfmadd_vv_f64m2(th, y, tl, vl);
    pxl = __riscv_vfsub_vv_f64m2(tl, __riscv_vfnmsub_vv_f64m2(th, y, px, vl), vl);
    
    vfloat64m2_t vnumh, vnuml, vdenh, vdenl, vden;
  
    th = __riscv_vfmadd_vv_f64m2(px, tanh1hi, tanhlo, vl);
    tl = __riscv_vfmadd_vv_f64m2(px, tanh1hi, __riscv_vfsub_vv_f64m2 (tanhlo, th, vl), vl);
    tl = __riscv_vfmacc_vv_f64m2(tl, pxl, tanh1hi, vl);
    tl = __riscv_vfmacc_vv_f64m2(tl, px, tanh1lo, vl);
    
    vnumh = __riscv_vfadd_vv_f64m2(tanhhi, th, vl);
    vnuml = __riscv_vfsub_vv_f64m2(th, __riscv_vfsub_vv_f64m2(vnumh, tanhhi, vl), vl);
    vnuml = __riscv_vfadd_vv_f64m2(vnuml, tl, vl);
    
    th = __riscv_vfmadd_vv_f64m2(px, tanhhi, tanh1lo, vl);
    tl = __riscv_vfmadd_vv_f64m2(px, tanhhi, __riscv_vfsub_vv_f64m2 (tanh1lo, th, vl), vl);
    tl = __riscv_vfmacc_vv_f64m2(tl, pxl, tanhhi, vl);
    tl = __riscv_vfmacc_vv_f64m2(tl, px, tanhlo, vl);
    
    vdenh = __riscv_vfadd_vv_f64m2(tanh1hi, th, vl);
    vdenl = __riscv_vfsub_vv_f64m2(th, __riscv_vfsub_vv_f64m2(vdenh, tanh1hi, vl), vl);
    vdenl = __riscv_vfadd_vv_f64m2(vdenl, tl, vl);
    
    vden = __riscv_vfrdiv_vf_f64m2(vdenh, 1.0, vl);
    th = __riscv_vfmul_vv_f64m2(vnumh, vden, vl);
    px = __riscv_vfnmsac_vv_f64m2(vnumh, th, vdenh, vl);
    tl = __riscv_vfnmsac_vv_f64m2(vnuml, th, vdenl, vl);
    
    px = __riscv_vfadd_vv_f64m2(px, tl, vl);
    px = __riscv_vfmadd_vv_f64m2(px, vden, th, vl);
  
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

vfloat64m4_t __riscv_vtanh_f64m4(vfloat64m4_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vuint64m4_t ix = __riscv_vand_vx_u64m4(
                 __riscv_vreinterpret_v_f64m4_u64m4(x), 0x7fffffffffffffff, vl);
    
    vfloat64m4_t ax = __riscv_vfmul_vf_f64m4(__riscv_vreinterpret_v_u64m4_f64m4(ix), 8.0, vl); 
    
    vuint64m4_t index = __riscv_vfcvt_rtz_xu_f_v_u64m4(ax, vl);
    
    vbool16_t mask = __riscv_vmsltu_vx_u64m4_b16(ix, 0x3fc0000000000000, vl);
    index = __riscv_vmerge_vxm_u64m4(index, 0, mask, vl);
     
    
    mask = __riscv_vmsgtu_vx_u64m4_b16(ix, 0x4034000000000000, vl);
    vfloat64m4_t y = __riscv_vfmerge_vfm_f64m4(__riscv_vreinterpret_v_u64m4_f64m4(ix), 20.0, mask, vl);
    
    index = __riscv_vmerge_vxm_u64m4(index, 160, mask, vl);
    y = __riscv_vfmacc_vf_f64m4(y, -0.125, __riscv_vfcvt_f_xu_v_f64m4(index, vl), vl);
    
    vuint64m4_t index_y = __riscv_vsrl_vx_u64m4(index, 3, vl);
    vuint64m4_t index_z = __riscv_vand_vx_u64m4(index, 0x7, vl);
    
    
    index_y = __riscv_vsll_vx_u64m4(index_y, 3, vl);
    index_z = __riscv_vsll_vx_u64m4(index_z, 3, vl);
            
    vfloat64m4_t tanhy_hi = __riscv_vloxei64_v_f64m4(tanhyf64_hi, index_y, vl);
    vfloat64m4_t tanhy_lo = __riscv_vloxei64_v_f64m4(tanhyf64_lo, index_y, vl);
    vfloat64m4_t tanhz_hi = __riscv_vloxei64_v_f64m4(tanhzf64_hi, index_z, vl);
    vfloat64m4_t tanhz_lo = __riscv_vloxei64_v_f64m4(tanhzf64_lo, index_z, vl);
    
    vfloat64m4_t tanhhi = __riscv_vfadd_vv_f64m4(tanhy_hi, tanhz_hi, vl);
    vfloat64m4_t tanhlo = __riscv_vfsub_vv_f64m4(tanhhi, tanhy_hi, vl);
    tanhlo = __riscv_vfsub_vv_f64m4(tanhz_hi, tanhlo, vl);
    tanhlo = __riscv_vfadd_vv_f64m4(tanhy_lo, tanhlo, vl);
    tanhlo = __riscv_vfadd_vv_f64m4(tanhz_lo, tanhlo, vl);
    
    vfloat64m4_t tanh1hi = __riscv_vfmadd_vv_f64m4(tanhy_hi, tanhz_hi, __riscv_vfmv_v_f_f64m4(1.0, vl), vl);
    vfloat64m4_t tanh1lo = __riscv_vfmadd_vv_f64m4(tanhy_hi, tanhz_hi, __riscv_vfrsub_vf_f64m4(tanh1hi, 1.0, vl), vl);
    tanh1lo = __riscv_vfmacc_vv_f64m4(tanh1lo, tanhy_hi, tanhz_lo, vl);
    tanh1lo = __riscv_vfmacc_vv_f64m4(tanh1lo, tanhy_lo, tanhz_hi, vl);
    
    vfloat64m4_t tl, th; 
    vfloat64m4_t y2h, y2l;
    y2h = __riscv_vfmadd_vv_f64m4(y, y, __riscv_vfmv_v_f_f64m4(0x1p-150, vl), vl);
       
    
    vfloat64m4_t pxl, px = __riscv_vfmv_v_f_f64m4(-0x1.22601d326d603p-7, vl);
    px = __riscv_vfmacc_vf_f64m4(px, 0x1.cc88061274d34p-9, y2h, vl);
    px = __riscv_vfmadd_vv_f64m4(px, y2h, __riscv_vfmv_v_f_f64m4(0x1.664f355520d9ap-6, vl), vl);
    px = __riscv_vfmadd_vv_f64m4(px, y2h, __riscv_vfmv_v_f_f64m4(-0x1.ba1ba1acb31ebp-5, vl), vl);
    px = __riscv_vfmadd_vv_f64m4(px, y2h, __riscv_vfmv_v_f_f64m4(0x1.111111110ed57p-3, vl), vl);
    px = __riscv_vfmadd_vv_f64m4(px, y2h, __riscv_vfmv_v_f_f64m4(-0x1.5555555555551p-2, vl), vl);
    th = __riscv_vfmadd_vv_f64m4(px, y2h, __riscv_vfmv_v_f_f64m4(1.0, vl), vl);
    tl = __riscv_vfmsub_vv_f64m4(px, y2h, __riscv_vfsub_vf_f64m4(th, 1.0, vl), vl);
    
    tl = __riscv_vfmul_vv_f64m4(y, tl, vl);
    px = __riscv_vfmadd_vv_f64m4(th, y, tl, vl);
    pxl = __riscv_vfsub_vv_f64m4(tl, __riscv_vfnmsub_vv_f64m4(th, y, px, vl), vl);
    
    vfloat64m4_t vnumh, vnuml, vdenh, vdenl, vden;
  
    th = __riscv_vfmadd_vv_f64m4(px, tanh1hi, tanhlo, vl);
    tl = __riscv_vfmadd_vv_f64m4(px, tanh1hi, __riscv_vfsub_vv_f64m4 (tanhlo, th, vl), vl);
    tl = __riscv_vfmacc_vv_f64m4(tl, pxl, tanh1hi, vl);
    tl = __riscv_vfmacc_vv_f64m4(tl, px, tanh1lo, vl);
    
    vnumh = __riscv_vfadd_vv_f64m4(tanhhi, th, vl);
    vnuml = __riscv_vfsub_vv_f64m4(th, __riscv_vfsub_vv_f64m4(vnumh, tanhhi, vl), vl);
    vnuml = __riscv_vfadd_vv_f64m4(vnuml, tl, vl);
    
    th = __riscv_vfmadd_vv_f64m4(px, tanhhi, tanh1lo, vl);
    tl = __riscv_vfmadd_vv_f64m4(px, tanhhi, __riscv_vfsub_vv_f64m4 (tanh1lo, th, vl), vl);
    tl = __riscv_vfmacc_vv_f64m4(tl, pxl, tanhhi, vl);
    tl = __riscv_vfmacc_vv_f64m4(tl, px, tanhlo, vl);
    
    vdenh = __riscv_vfadd_vv_f64m4(tanh1hi, th, vl);
    vdenl = __riscv_vfsub_vv_f64m4(th, __riscv_vfsub_vv_f64m4(vdenh, tanh1hi, vl), vl);
    vdenl = __riscv_vfadd_vv_f64m4(vdenl, tl, vl);
    
    vden = __riscv_vfrdiv_vf_f64m4(vdenh, 1.0, vl);
    th = __riscv_vfmul_vv_f64m4(vnumh, vden, vl);
    px = __riscv_vfnmsac_vv_f64m4(vnumh, th, vdenh, vl);
    tl = __riscv_vfnmsac_vv_f64m4(vnuml, th, vdenl, vl);
    
    px = __riscv_vfadd_vv_f64m4(px, tl, vl);
    px = __riscv_vfmadd_vv_f64m4(px, vden, th, vl);
  
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
    
    vfloat32m1_t ax = __riscv_vfmul_vf_f32m1(__riscv_vreinterpret_v_u32m1_f32m1(ix), 2.0f, vl); 
    
    vuint32m1_t index = __riscv_vfcvt_rtz_xu_f_v_u32m1(ax, vl);
    
    vbool32_t mask = __riscv_vmsltu_vx_u32m1_b32(ix, 0x3f000000, vl);
    index = __riscv_vmerge_vxm_u32m1(index, 0x00000000, mask, vl);
     
    mask = __riscv_vmsgtu_vx_u32m1_b32(ix, 0x41180000, vl);
    vfloat32m1_t y = __riscv_vfmerge_vfm_f32m1(__riscv_vreinterpret_v_u32m1_f32m1(ix), 9.5f, mask, vl);
    
    index = __riscv_vmerge_vxm_u32m1(index, 19, mask, vl);
    y = __riscv_vfmacc_vf_f32m1(y, -0.5f, __riscv_vfcvt_f_xu_v_f32m1(index, vl), vl);
    
    index = __riscv_vsll_vx_u32m1(index, 2, vl);
            
    vfloat32m1_t tanhy_hi = __riscv_vloxei32_v_f32m1(tanhyf32_hi, index, vl);
    vfloat32m1_t tanhy_lo = __riscv_vloxei32_v_f32m1(tanhyf32_lo, index, vl);
    
    
    vfloat32m1_t tl, th; 
    vfloat32m1_t y2h;
    y2h = __riscv_vfmadd_vv_f32m1(y, y, __riscv_vfmv_v_f_f32m1(0x1p-50f, vl), vl);
     
    
    vfloat32m1_t px = __riscv_vfmv_v_f_f32m1(0x1.5c9098p-6f, vl);
    px = __riscv_vfmacc_vf_f32m1(px, -0x1.b16ae8p-8f, y2h, vl);
    px = __riscv_vfmadd_vv_f32m1(px, y2h, __riscv_vfmv_v_f_f32m1(-0x1.b97ebp-5f, vl), vl);
    px = __riscv_vfmadd_vv_f32m1(px, y2h, __riscv_vfmv_v_f_f32m1(0x1.110ecep-3f, vl), vl);
    px = __riscv_vfmadd_vv_f32m1(px, y2h, __riscv_vfmv_v_f_f32m1(-0x1.55555p-2f, vl), vl);
    th = __riscv_vfmadd_vv_f32m1(px, y2h, __riscv_vfmv_v_f_f32m1(1.0f, vl), vl);
    tl = __riscv_vfmsub_vv_f32m1(px, y2h, __riscv_vfsub_vf_f32m1(th, 1.0f, vl), vl);
    
    tl = __riscv_vfmul_vv_f32m1(y, tl, vl);
    px = __riscv_vfmadd_vv_f32m1(th, y, tl, vl);
    tl = __riscv_vfsub_vv_f32m1(tl, __riscv_vfnmsac_vv_f32m1(px, th, y, vl), vl);
    
    vfloat32m1_t vnumh, vnuml, vdenh, vdenl, vden;
  
    vnumh = __riscv_vfadd_vv_f32m1(px, tanhy_hi, vl);
    vnuml = __riscv_vfsub_vv_f32m1(px, __riscv_vfsub_vv_f32m1(vnumh, tanhy_hi, vl), vl);
    vnuml = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(vnuml, tanhy_lo, vl), tl, vl);
    
    vdenh = __riscv_vfmadd_vv_f32m1(tanhy_hi, px, __riscv_vfmv_v_f_f32m1(1.0f, vl), vl);
    vdenl = __riscv_vfmadd_vv_f32m1(tanhy_hi, px, __riscv_vfrsub_vf_f32m1(vdenh, 1.0f, vl), vl);
    vdenl = __riscv_vfmacc_vv_f32m1(vdenl, tanhy_hi, tl, vl);
    vdenl = __riscv_vfmacc_vv_f32m1(vdenl, tanhy_lo, px, vl);

    vden = __riscv_vfrdiv_vf_f32m1(vdenh, 1.0f, vl);
    th = __riscv_vfmul_vv_f32m1(vnumh, vden, vl);
    px = __riscv_vfnmsac_vv_f32m1(vnumh, th, vdenh, vl);
    tl = __riscv_vfnmsac_vv_f32m1(vnuml, th, vdenl, vl);
    
    px = __riscv_vfadd_vv_f32m1(px, tl, vl);
    px = __riscv_vfmadd_vv_f32m1(px, vden, th, vl);
  
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
    
    vfloat32m2_t ax = __riscv_vfmul_vf_f32m2(__riscv_vreinterpret_v_u32m2_f32m2(ix), 2.0f, vl); 
    
    vuint32m2_t index = __riscv_vfcvt_rtz_xu_f_v_u32m2(ax, vl);
    
    vbool16_t mask = __riscv_vmsltu_vx_u32m2_b16(ix, 0x3f000000, vl);
    index = __riscv_vmerge_vxm_u32m2(index, 0x00000000, mask, vl);
     
    mask = __riscv_vmsgtu_vx_u32m2_b16(ix, 0x41180000, vl);
    vfloat32m2_t y = __riscv_vfmerge_vfm_f32m2(__riscv_vreinterpret_v_u32m2_f32m2(ix), 9.5f, mask, vl);
    
    index = __riscv_vmerge_vxm_u32m2(index, 19, mask, vl);
    y = __riscv_vfmacc_vf_f32m2(y, -0.5f, __riscv_vfcvt_f_xu_v_f32m2(index, vl), vl);
    
    index = __riscv_vsll_vx_u32m2(index, 2, vl);
            
    vfloat32m2_t tanhy_hi = __riscv_vloxei32_v_f32m2(tanhyf32_hi, index, vl);
    vfloat32m2_t tanhy_lo = __riscv_vloxei32_v_f32m2(tanhyf32_lo, index, vl);
    
    
    vfloat32m2_t tl, th; 
    vfloat32m2_t y2h;
    y2h = __riscv_vfmadd_vv_f32m2(y, y, __riscv_vfmv_v_f_f32m2(0x1p-50f, vl), vl);
       
    
    vfloat32m2_t px = __riscv_vfmv_v_f_f32m2(0x1.5c9098p-6f, vl);
    px = __riscv_vfmacc_vf_f32m2(px, -0x1.b16ae8p-8f, y2h, vl);
    px = __riscv_vfmadd_vv_f32m2(px, y2h, __riscv_vfmv_v_f_f32m2(-0x1.b97ebp-5f, vl), vl);
    px = __riscv_vfmadd_vv_f32m2(px, y2h, __riscv_vfmv_v_f_f32m2(0x1.110ecep-3f, vl), vl);
    px = __riscv_vfmadd_vv_f32m2(px, y2h, __riscv_vfmv_v_f_f32m2(-0x1.55555p-2f, vl), vl);
    th = __riscv_vfmadd_vv_f32m2(px, y2h, __riscv_vfmv_v_f_f32m2(1.0f, vl), vl);
    tl = __riscv_vfmsub_vv_f32m2(px, y2h, __riscv_vfsub_vf_f32m2(th, 1.0f, vl), vl);
    
    tl = __riscv_vfmul_vv_f32m2(y, tl, vl);
    px = __riscv_vfmadd_vv_f32m2(th, y, tl, vl);
    tl = __riscv_vfsub_vv_f32m2(tl, __riscv_vfnmsac_vv_f32m2(px, th, y, vl), vl);
    
    vfloat32m2_t vnumh, vnuml, vdenh, vdenl, vden;
  
    vnumh = __riscv_vfadd_vv_f32m2(px, tanhy_hi, vl);
    vnuml = __riscv_vfsub_vv_f32m2(px, __riscv_vfsub_vv_f32m2(vnumh, tanhy_hi, vl), vl);
    vnuml = __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(vnuml, tanhy_lo, vl), tl, vl);
    
    vdenh = __riscv_vfmadd_vv_f32m2(tanhy_hi, px, __riscv_vfmv_v_f_f32m2(1.0f, vl), vl);
    vdenl = __riscv_vfmadd_vv_f32m2(tanhy_hi, px, __riscv_vfrsub_vf_f32m2(vdenh, 1.0f, vl), vl);
    vdenl = __riscv_vfmacc_vv_f32m2(vdenl, tanhy_hi, tl, vl);
    vdenl = __riscv_vfmacc_vv_f32m2(vdenl, tanhy_lo, px, vl);
    
    vden = __riscv_vfrdiv_vf_f32m2(vdenh, 1.0f, vl);
    th = __riscv_vfmul_vv_f32m2(vnumh, vden, vl);
    px = __riscv_vfnmsac_vv_f32m2(vnumh, th, vdenh, vl);
    tl = __riscv_vfnmsac_vv_f32m2(vnuml, th, vdenl, vl);
    
    px = __riscv_vfadd_vv_f32m2(px, tl, vl);
    px = __riscv_vfmadd_vv_f32m2(px, vden, th, vl);
  
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
    
    vfloat32m4_t ax = __riscv_vfmul_vf_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(ix), 2.0f, vl); 
    
    vuint32m4_t index = __riscv_vfcvt_rtz_xu_f_v_u32m4(ax, vl);
    
    vbool8_t mask = __riscv_vmsltu_vx_u32m4_b8(ix, 0x3f000000, vl);
    index = __riscv_vmerge_vxm_u32m4(index, 0x00000000, mask, vl);
     
    mask = __riscv_vmsgtu_vx_u32m4_b8(ix, 0x41180000, vl);
    vfloat32m4_t y = __riscv_vfmerge_vfm_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(ix), 9.5f, mask, vl);
    
    index = __riscv_vmerge_vxm_u32m4(index, 19, mask, vl);
    y = __riscv_vfmacc_vf_f32m4(y, -0.5f, __riscv_vfcvt_f_xu_v_f32m4(index, vl), vl);
    
    index = __riscv_vsll_vx_u32m4(index, 2, vl);
            
    vfloat32m4_t tanhy_hi = __riscv_vloxei32_v_f32m4(tanhyf32_hi, index, vl);
    vfloat32m4_t tanhy_lo = __riscv_vloxei32_v_f32m4(tanhyf32_lo, index, vl);
    
    
    vfloat32m4_t tl, th; 
    vfloat32m4_t y2h;
    y2h = __riscv_vfmadd_vv_f32m4(y, y, __riscv_vfmv_v_f_f32m4(0x1p-50f, vl), vl);
       
    
    vfloat32m4_t px = __riscv_vfmv_v_f_f32m4(0x1.5c9098p-6f, vl);
    px = __riscv_vfmacc_vf_f32m4(px, -0x1.b16ae8p-8f, y2h, vl);
    px = __riscv_vfmadd_vv_f32m4(px, y2h, __riscv_vfmv_v_f_f32m4(-0x1.b97ebp-5f, vl), vl);
    px = __riscv_vfmadd_vv_f32m4(px, y2h, __riscv_vfmv_v_f_f32m4(0x1.110ecep-3f, vl), vl);
    px = __riscv_vfmadd_vv_f32m4(px, y2h, __riscv_vfmv_v_f_f32m4(-0x1.55555p-2f, vl), vl);
    th = __riscv_vfmadd_vv_f32m4(px, y2h, __riscv_vfmv_v_f_f32m4(1.0f, vl), vl);
    tl = __riscv_vfmsub_vv_f32m4(px, y2h, __riscv_vfsub_vf_f32m4(th, 1.0f, vl), vl);
    
    tl = __riscv_vfmul_vv_f32m4(y, tl, vl);
    px = __riscv_vfmadd_vv_f32m4(th, y, tl, vl);
    tl = __riscv_vfsub_vv_f32m4(tl, __riscv_vfnmsac_vv_f32m4(px, th, y, vl), vl);
    
    vfloat32m4_t vnumh, vnuml, vdenh, vdenl, vden;
  
    vnumh = __riscv_vfadd_vv_f32m4(px, tanhy_hi, vl);
    vnuml = __riscv_vfsub_vv_f32m4(px, __riscv_vfsub_vv_f32m4(vnumh, tanhy_hi, vl), vl);
    vnuml = __riscv_vfadd_vv_f32m4(__riscv_vfadd_vv_f32m4(vnuml, tanhy_lo, vl), tl, vl);
    
    vdenh = __riscv_vfmadd_vv_f32m4(tanhy_hi, px, __riscv_vfmv_v_f_f32m4(1.0f, vl), vl);
    vdenl = __riscv_vfmadd_vv_f32m4(tanhy_hi, px, __riscv_vfrsub_vf_f32m4(vdenh, 1.0f, vl), vl);
    vdenl = __riscv_vfmacc_vv_f32m4(vdenl, tanhy_hi, tl, vl);
    vdenl = __riscv_vfmacc_vv_f32m4(vdenl, tanhy_lo, px, vl);
    
    vden = __riscv_vfrdiv_vf_f32m4(vdenh, 1.0f, vl);
    th = __riscv_vfmul_vv_f32m4(vnumh, vden, vl);
    px = __riscv_vfnmsac_vv_f32m4(vnumh, th, vdenh, vl);
    tl = __riscv_vfnmsac_vv_f32m4(vnuml, th, vdenl, vl);
    
    px = __riscv_vfadd_vv_f32m4(px, tl, vl);
    px = __riscv_vfmadd_vv_f32m4(px, vden, th, vl);
  
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
//static _Float16 tanhyf16_hi [20];
//static _Float16 tanhyf16_lo [20];
#include "tanhf16.data"

vfloat16m1_t __riscv_vtanh_f16m1(vfloat16m1_t x, size_t avl)
{ 
    size_t vl = __riscv_vsetvl_e16m1(avl);
    vuint16m1_t ix = __riscv_vand_vx_u16m1(
                 __riscv_vreinterpret_v_f16m1_u16m1(x), 0x7fff, vl);
    
    vfloat16m1_t ax = __riscv_vfmul_vf_f16m1(__riscv_vreinterpret_v_u16m1_f16m1(ix), 4.0f16, vl); 
    
    vuint16m1_t index = __riscv_vfcvt_rtz_xu_f_v_u16m1(ax, vl);
    
    vbool16_t mask = __riscv_vmsltu_vx_u16m1_b16(ix, 0x3400, vl);
    index = __riscv_vmerge_vxm_u16m1(index, 0, mask, vl);
     
    mask = __riscv_vmsgtu_vx_u16m1_b16(ix, 0x44c0, vl);
    vfloat16m1_t y = __riscv_vfmerge_vfm_f16m1(__riscv_vreinterpret_v_u16m1_f16m1(ix), 4.75f16, mask, vl);
    
    index = __riscv_vmerge_vxm_u16m1(index, 19, mask, vl);
    y = __riscv_vfmacc_vf_f16m1(y, -0.25f16, __riscv_vfcvt_f_xu_v_f16m1(index, vl), vl);
    
    index = __riscv_vsll_vx_u16m1(index, 1, vl);
            
    vfloat16m1_t tanh_hi = __riscv_vloxei16_v_f16m1(tanhyf16_hi, index, vl);
    vfloat16m1_t tanh_lo = __riscv_vloxei16_v_f16m1(tanhyf16_lo, index, vl);
    
    
    vfloat16m1_t tl, th; 
    vfloat16m1_t y2h;
    y2h = __riscv_vfmul_vv_f16m1(y, y, vl);   
    
    vfloat16m1_t px = __riscv_vfmv_v_f_f16m1(-0x1.554p-2f16, vl);
    px = __riscv_vfmacc_vf_f16m1(px, 0x1.07p-3f16, y2h, vl);
    th = __riscv_vfmadd_vv_f16m1(px, y2h, __riscv_vfmv_v_f_f16m1(1.0f16, vl), vl);
    tl = __riscv_vfmsub_vv_f16m1(px, y2h, __riscv_vfsub_vf_f16m1(th, 1.0f16, vl), vl);
    
    tl = __riscv_vfmul_vv_f16m1(y, tl, vl);
    px = __riscv_vfmadd_vv_f16m1(th, y, tl, vl);
    tl = __riscv_vfsub_vv_f16m1(tl, __riscv_vfnmsac_vv_f16m1(px, th, y, vl), vl);
    
    vfloat16m1_t vnumh, vnuml, vdenh, vdenl, vden;
  
    vnumh = __riscv_vfadd_vv_f16m1(px, tanh_hi, vl);
    vnuml = __riscv_vfsub_vv_f16m1(px, __riscv_vfsub_vv_f16m1(vnumh, tanh_hi, vl), vl);
    vnuml = __riscv_vfadd_vv_f16m1(__riscv_vfadd_vv_f16m1(vnuml, tanh_lo, vl), tl, vl);
    
    vdenh = __riscv_vfmadd_vv_f16m1(tanh_hi, px, __riscv_vfmv_v_f_f16m1(1.0f16, vl), vl);
    vdenl = __riscv_vfmadd_vv_f16m1(tanh_hi, px, __riscv_vfrsub_vf_f16m1(vdenh, 1.0f16, vl), vl);
    vdenl = __riscv_vfmacc_vv_f16m1(vdenl, tanh_hi, tl, vl);
    vdenl = __riscv_vfmacc_vv_f16m1(vdenl, tanh_lo, px, vl);
    
    vden = __riscv_vfrdiv_vf_f16m1(vdenh, 1.0f16, vl);
    th = __riscv_vfmul_vv_f16m1(vnumh, vden, vl);
    px = __riscv_vfnmsac_vv_f16m1(vnumh, th, vdenh, vl);
    tl = __riscv_vfnmsac_vv_f16m1(vnuml, th, vdenl, vl);
    
    px = __riscv_vfadd_vv_f16m1(px, tl, vl);
    px = __riscv_vfmadd_vv_f16m1(px, vden, th, vl);
  
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
    
    vfloat16m2_t ax = __riscv_vfmul_vf_f16m2(__riscv_vreinterpret_v_u16m2_f16m2(ix), 4.0f16, vl); 
    
    vuint16m2_t index = __riscv_vfcvt_rtz_xu_f_v_u16m2(ax, vl);
    
    vbool8_t mask = __riscv_vmsltu_vx_u16m2_b8(ix, 0x3400, vl);
    index = __riscv_vmerge_vxm_u16m2(index, 0, mask, vl);
     
    mask = __riscv_vmsgtu_vx_u16m2_b8(ix, 0x44c0, vl);
    vfloat16m2_t y = __riscv_vfmerge_vfm_f16m2(__riscv_vreinterpret_v_u16m2_f16m2(ix), 4.75f16, mask, vl);
    
    index = __riscv_vmerge_vxm_u16m2(index, 19, mask, vl);
    y = __riscv_vfmacc_vf_f16m2(y, -0.25f16, __riscv_vfcvt_f_xu_v_f16m2(index, vl), vl);
    
    index = __riscv_vsll_vx_u16m2(index, 1, vl);
            
    vfloat16m2_t tanh_hi = __riscv_vloxei16_v_f16m2(tanhyf16_hi, index, vl);
    vfloat16m2_t tanh_lo = __riscv_vloxei16_v_f16m2(tanhyf16_lo, index, vl);
    
    
    vfloat16m2_t tl, th; 
    vfloat16m2_t y2h;
    y2h = __riscv_vfmul_vv_f16m2(y, y, vl);
    
    vfloat16m2_t px = __riscv_vfmv_v_f_f16m2(-0x1.554p-2f16, vl);
    px = __riscv_vfmacc_vf_f16m2(px, 0x1.07p-3f16, y2h, vl);
    th = __riscv_vfmadd_vv_f16m2(px, y2h, __riscv_vfmv_v_f_f16m2(1.0f16, vl), vl);
    tl = __riscv_vfmsub_vv_f16m2(px, y2h, __riscv_vfsub_vf_f16m2(th, 1.0f16, vl), vl);
    
    tl = __riscv_vfmul_vv_f16m2(y, tl, vl);
    px = __riscv_vfmadd_vv_f16m2(th, y, tl, vl);
    tl = __riscv_vfsub_vv_f16m2(tl, __riscv_vfnmsac_vv_f16m2(px, th, y, vl), vl);
    
    vfloat16m2_t vnumh, vnuml, vdenh, vdenl, vden;
  
    vnumh = __riscv_vfadd_vv_f16m2(px, tanh_hi, vl);
    vnuml = __riscv_vfsub_vv_f16m2(px, __riscv_vfsub_vv_f16m2(vnumh, tanh_hi, vl), vl);
    vnuml = __riscv_vfadd_vv_f16m2(__riscv_vfadd_vv_f16m2(vnuml, tanh_lo, vl), tl, vl);
    
    vdenh = __riscv_vfmadd_vv_f16m2(tanh_hi, px, __riscv_vfmv_v_f_f16m2(1.0f16, vl), vl);
    vdenl = __riscv_vfmadd_vv_f16m2(tanh_hi, px, __riscv_vfrsub_vf_f16m2(vdenh, 1.0f16, vl), vl);
    vdenl = __riscv_vfmacc_vv_f16m2(vdenl, tanh_hi, tl, vl);
    vdenl = __riscv_vfmacc_vv_f16m2(vdenl, tanh_lo, px, vl);
    
    vden = __riscv_vfrdiv_vf_f16m2(vdenh, 1.0f16, vl);
    th = __riscv_vfmul_vv_f16m2(vnumh, vden, vl);
    px = __riscv_vfnmsac_vv_f16m2(vnumh, th, vdenh, vl);
    tl = __riscv_vfnmsac_vv_f16m2(vnuml, th, vdenl, vl);
    
    px = __riscv_vfadd_vv_f16m2(px, tl, vl);
    px = __riscv_vfmadd_vv_f16m2(px, vden, th, vl);
  
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
    
    vfloat16m4_t ax = __riscv_vfmul_vf_f16m4(__riscv_vreinterpret_v_u16m4_f16m4(ix), 4.0f16, vl); 
    
    vuint16m4_t index = __riscv_vfcvt_rtz_xu_f_v_u16m4(ax, vl);
    
    vbool4_t mask = __riscv_vmsltu_vx_u16m4_b4(ix, 0x3400, vl);
    index = __riscv_vmerge_vxm_u16m4(index, 0, mask, vl);
     
    mask = __riscv_vmsgtu_vx_u16m4_b4(ix, 0x44c0, vl);
    vfloat16m4_t y = __riscv_vfmerge_vfm_f16m4(__riscv_vreinterpret_v_u16m4_f16m4(ix), 4.75f16, mask, vl);
    
    index = __riscv_vmerge_vxm_u16m4(index, 19, mask, vl);
    y = __riscv_vfmacc_vf_f16m4(y, -0.25f16, __riscv_vfcvt_f_xu_v_f16m4(index, vl), vl);
    
    index = __riscv_vsll_vx_u16m4(index, 1, vl);
            
    vfloat16m4_t tanh_hi = __riscv_vloxei16_v_f16m4(tanhyf16_hi, index, vl);
    vfloat16m4_t tanh_lo = __riscv_vloxei16_v_f16m4(tanhyf16_lo, index, vl);
    
    
    vfloat16m4_t tl, th; 
    vfloat16m4_t y2h;
    y2h = __riscv_vfmul_vv_f16m4(y, y, vl);
    
    vfloat16m4_t px = __riscv_vfmv_v_f_f16m4(-0x1.554p-2f16, vl);
    px = __riscv_vfmacc_vf_f16m4(px, 0x1.07p-3f16, y2h, vl);
    th = __riscv_vfmadd_vv_f16m4(px, y2h, __riscv_vfmv_v_f_f16m4(1.0f16, vl), vl);
    tl = __riscv_vfmsub_vv_f16m4(px, y2h, __riscv_vfsub_vf_f16m4(th, 1.0f16, vl), vl);
    
    tl = __riscv_vfmul_vv_f16m4(y, tl, vl);
    px = __riscv_vfmadd_vv_f16m4(th, y, tl, vl);
    tl = __riscv_vfsub_vv_f16m4(tl, __riscv_vfnmsac_vv_f16m4(px, th, y, vl), vl);
    
    vfloat16m4_t vnumh, vnuml, vdenh, vdenl, vden;
  
    vnumh = __riscv_vfadd_vv_f16m4(px, tanh_hi, vl);
    vnuml = __riscv_vfsub_vv_f16m4(px, __riscv_vfsub_vv_f16m4(vnumh, tanh_hi, vl), vl);
    vnuml = __riscv_vfadd_vv_f16m4(__riscv_vfadd_vv_f16m4(vnuml, tanh_lo, vl), tl, vl);
    
    vdenh = __riscv_vfmadd_vv_f16m4(tanh_hi, px, __riscv_vfmv_v_f_f16m4(1.0f16, vl), vl);
    vdenl = __riscv_vfmadd_vv_f16m4(tanh_hi, px, __riscv_vfrsub_vf_f16m4(vdenh, 1.0f16, vl), vl);
    vdenl = __riscv_vfmacc_vv_f16m4(vdenl, tanh_hi, tl, vl);
    vdenl = __riscv_vfmacc_vv_f16m4(vdenl, tanh_lo, px, vl);
    
    vden = __riscv_vfrdiv_vf_f16m4(vdenh, 1.0f16, vl);
    th = __riscv_vfmul_vv_f16m4(vnumh, vden, vl);
    px = __riscv_vfnmsac_vv_f16m4(vnumh, th, vdenh, vl);
    tl = __riscv_vfnmsac_vv_f16m4(vnuml, th, vdenl, vl);
    
    px = __riscv_vfadd_vv_f16m4(px, tl, vl);
    px = __riscv_vfmadd_vv_f16m4(px, vden, th, vl);
  
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

