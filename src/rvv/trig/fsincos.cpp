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
 *   File:  fsincos.cpp                                  *
 *   Contains: intrinsic functions sin, cos, sincos      *
 *             for float32                               *
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
 *    2) Reduction to [-0x1p-3, 0x1p-3]                  *
 *    3) Calculating sincos                              *
 *    4) Restoring the value of a function using         *
 *       formulas and table values:                      *
 *       sin(x+a)=sin(x)*cos(a)+cos(x)*sin(a)            *
 *       cos(x+a)=cos(x)*cos(a)-sin(x)*sin(a)            *
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

float table_cos_hi_f [] = {
    0x1p+0,
    0x1.fc0156p-1, 0x1.faf222p-1, 0x1.f9c34p-1, 0x1.f874c2p-1,
    0x1.f706bep-1, 0x1.f57948p-1, 0x1.f3cc7cp-1, 0x1.f20074p-1,
    0x1.f0154ap-1, 0x1.ebe214p-1, 0x1.e733eap-1, 0x1.e20bf4p-1,
    0x1.dc6b7ep-1, 0x1.d653fp-1, 0x1.cfc6dp-1, 0x1.c8c5cp-1,
    0x1.c1528p-1, 0x1.b11d04p-1, 0x1.9f368ep-1, 0x1.8bb106p-1,
    0x1.769fecp-1, 0x1.601852p-1
};
    
float table_cos_lo_f [] = {
    0x0p+0,
    -0x1.b05486p-26, 0x1.8f12f4p-27, 0x1.4f9886p-26, 0x1.c3dd9ep-26,
    -0x1.84c792p-31, 0x1.9fecf2p-26, 0x1.d9e8b6p-28, -0x1.ef336cp-26,
    -0x1.0422bep-30, 0x1.eeddf4p-26, 0x1.93d3fap-33, 0x1.359ad8p-26,
    0x1.732b22p-26, 0x1.cf901p-27, -0x1.6b5498p-27, -0x1.cc795ep-27,
    0x1.96df54p-27, 0x1.62a4c6p-29, 0x1.b225fp-26, -0x1.688dcp-27,
    0x1.954848p-27, 0x1.bd58f8p-27
};
    
float table_sin_hi_f [] = {
    0x0p+0,
    0x1.feaaeep-4, 0x1.1f0d3ep-3, 0x1.3eb312p-3, 0x1.5e44fcp-3,
    0x1.7dc102p-3, 0x1.9d252ep-3, 0x1.bc6f84p-3, 0x1.db9e16p-3,
    0x1.faaeeep-3, 0x1.1c37d6p-2, 0x1.3ad12ap-2, 0x1.591bcap-2,
    0x1.771026p-2, 0x1.94a6bep-2, 0x1.b1d83p-2, 0x1.ce9d2ep-2,
    0x1.eaee88p-2, 0x1.110d0cp-1, 0x1.2b91dep-1, 0x1.44eb38p-1,
    0x1.5cffc2p-1, 0x1.73b768p-1
};
    
float table_sin_lo_f [] = {
    0x0p+0,
    0x1.d0ddc6p-29, -0x1.0a062ap-28, 0x1.8bacdap-28, 0x1.f424dep-28,
    0x1.f75e56p-28, -0x1.e6279ep-28, 0x1.db8c34p-28, -0x1.2968c2p-33,
    -0x1.619d52p-28, 0x1.31ae1ep-28, -0x1.12c584p-27, -0x1.7429a4p-32,
    -0x1.5137bep-27, 0x1.3ea8d8p-27, 0x1.4c8586p-28, 0x1.ea529p-29,
    -0x1.769f42p-27, 0x1.2da70ep-27, 0x1.510844p-26, 0x1.cf386ap-29,
    -0x1.280e1ep-26, 0x1.bd4afp-30
};

void __riscv_vsincos_kernel_f32m1(vfloat32m1_t x, vfloat32m1_t x_l, vfloat32m1_t& sinrx, vfloat32m1_t& cosrx, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m1(avl);
    vfloat32m1_t x_lo, x_hi, tmp, tmp2;
    x_hi = __riscv_vmv_v_v_f32m1(x, vl);
    x_lo = __riscv_vmv_v_v_f32m1(x_l, vl);

    vuint32m1_t ix = __riscv_vreinterpret_v_f32m1_u32m1(x_hi);
    vuint32m1_t c_u32 = __riscv_vadd_vx_u32m1(ix, 0x00080000, vl);
    c_u32 = __riscv_vand_vx_u32m1(c_u32, 0x7ff00000, vl);
    vuint32m1_t ind = __riscv_vsrl_vx_u32m1(c_u32, 20, vl);
    ind = __riscv_vsub_vx_u32m1(ind, 991, vl);
    
    vbool32_t mask = __riscv_vmsltu_vx_u32m1_b32(c_u32, 0x3e000000, vl);
    vfloat32m1_t c = __riscv_vfmerge_vfm_f32m1(
               __riscv_vreinterpret_v_u32m1_f32m1(c_u32), 0.0f, mask, vl);
    ind = __riscv_vmerge_vxm_u32m1(ind, 0, mask, vl);
    ind = __riscv_vsll_vx_u32m1(ind, 2, vl);

    tmp = __riscv_vfsub_vv_f32m1(x_hi, c, vl);
    x_hi = __riscv_vfadd_vv_f32m1(tmp, x_lo, vl);
    tmp = __riscv_vfsub_vv_f32m1(x_hi, tmp, vl);
    x_lo = __riscv_vfsub_vv_f32m1(x_lo, tmp, vl);

    tmp = __riscv_vfmul_vv_f32m1(x_lo, x_hi, vl);
    tmp = __riscv_vfmul_vf_f32m1(tmp, 2.0f, vl);
    vfloat32m1_t x2 = __riscv_vfmul_vv_f32m1(x_hi, x_hi, vl);
    vfloat32m1_t x2_lo = __riscv_vfmsub_vv_f32m1(x_hi, x_hi, x2, vl);
    x2_lo = __riscv_vfadd_vv_f32m1(tmp, x2_lo, vl);

    vfloat32m1_t sin_r_h = __riscv_vfmadd_vf_f32m1(x2, 0x1.10ea6ep-7f, 
                            __riscv_vfmv_v_f_f32m1(-0x1.555552p-3f, vl), vl);
    tmp = __riscv_vfmul_vv_f32m1(sin_r_h, x_lo, vl);
    tmp = __riscv_vfmadd_vv_f32m1(sin_r_h, x_hi, tmp, vl);
    sin_r_h = __riscv_vfmadd_vv_f32m1(tmp, x2, x_hi, vl);
    vfloat32m1_t sin_r_l = __riscv_vfsub_vv_f32m1(x_hi, sin_r_h, vl);
    sin_r_l = __riscv_vfmadd_vv_f32m1(tmp, x2, sin_r_l, vl);
    sin_r_l = __riscv_vfadd_vv_f32m1(sin_r_l, x_lo, vl);
    
    vfloat32m1_t cos_r_l = __riscv_vfmadd_vf_f32m1(x2, 0x1.551586p-5f,
                            __riscv_vfmv_v_f_f32m1(-0x1.fffff6p-2f, vl), vl);
    vfloat32m1_t cos_r_h = __riscv_vfmadd_vv_f32m1(cos_r_l, x2, 
                            __riscv_vfmv_v_f_f32m1(0x1p0f, vl), vl);
    tmp = __riscv_vfrsub_vf_f32m1(cos_r_h, 0x1p0f, vl);
    tmp = __riscv_vfmadd_vv_f32m1(cos_r_l, x2, tmp, vl);
    cos_r_l = __riscv_vfmadd_vv_f32m1(cos_r_l, x2_lo, tmp, vl);
    
    vfloat32m1_t cos_hi = __riscv_vloxei32_v_f32m1(table_cos_hi_f, ind, vl);
    vfloat32m1_t cos_lo = __riscv_vloxei32_v_f32m1(table_cos_lo_f, ind, vl);
    vfloat32m1_t sin_hi = __riscv_vloxei32_v_f32m1(table_sin_hi_f, ind, vl);
    vfloat32m1_t sin_lo = __riscv_vloxei32_v_f32m1(table_sin_lo_f, ind, vl);
    
    sinrx = __riscv_vfmul_vv_f32m1(cos_lo, sin_r_h, vl);
    sinrx = __riscv_vfmacc_vv_f32m1(sinrx, cos_hi, sin_r_l, vl);
    sinrx = __riscv_vfmacc_vv_f32m1(sinrx, sin_hi, cos_r_l, vl);
    sinrx = __riscv_vfmacc_vv_f32m1(sinrx, sin_lo, cos_r_h, vl);    
    vfloat32m1_t t1h = __riscv_vfmadd_vv_f32m1(cos_hi, sin_r_h, sinrx, vl);
    tmp = __riscv_vfmul_vv_f32m1(cos_hi, sin_r_h, vl);
    tmp2 = __riscv_vfmsub_vv_f32m1(cos_hi, sin_r_h, tmp, vl);
    vfloat32m1_t t1l = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(
                            __riscv_vfsub_vv_f32m1(tmp, t1h, vl), sinrx, vl), tmp2, vl);
    vfloat32m1_t t2h = __riscv_vfmadd_vv_f32m1(sin_hi, cos_r_h, t1h, vl);
    tmp = __riscv_vfmul_vv_f32m1(sin_hi, cos_r_h, vl);
    tmp2 = __riscv_vfmsub_vv_f32m1(sin_hi, cos_r_h, tmp, vl);
    vfloat32m1_t t2l = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(
                            __riscv_vfsub_vv_f32m1(tmp, t2h, vl), t1h, vl), tmp2, vl);
    sinrx = __riscv_vfadd_vv_f32m1(t2l, t1l, vl);
    sinrx = __riscv_vfadd_vv_f32m1(sinrx, t2h, vl);
    
    cosrx = __riscv_vfmul_vv_f32m1(sin_hi, sin_r_l, vl);
    cosrx = __riscv_vfmacc_vv_f32m1(cosrx, sin_lo, sin_r_h, vl);
    cosrx = __riscv_vfmsac_vv_f32m1(cosrx, cos_hi, cos_r_l, vl);
    cosrx = __riscv_vfmacc_vv_f32m1(cosrx, cos_lo, cos_r_h, vl);
    cosrx = __riscv_vfnmsac_vv_f32m1(cosrx, sin_hi, sin_r_h, vl);
    cosrx = __riscv_vfmacc_vv_f32m1(cosrx, cos_hi, cos_r_h, vl);   
}

void __riscv_vsincos_kernel_f32m2(vfloat32m2_t x, vfloat32m2_t x_l, vfloat32m2_t& sinrx, vfloat32m2_t& cosrx, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m2(avl);
    vfloat32m2_t x_lo, x_hi, tmp, tmp2;
    x_hi = __riscv_vmv_v_v_f32m2(x, vl);
    x_lo = __riscv_vmv_v_v_f32m2(x_l, vl);

    vuint32m2_t ix = __riscv_vreinterpret_v_f32m2_u32m2(x_hi);
    vuint32m2_t c_u32 = __riscv_vadd_vx_u32m2(ix, 0x00080000, vl);
    c_u32 = __riscv_vand_vx_u32m2(c_u32, 0x7ff00000, vl);
    vuint32m2_t ind = __riscv_vsrl_vx_u32m2(c_u32, 20, vl);
    ind = __riscv_vsub_vx_u32m2(ind, 991, vl);
    
    vbool16_t mask = __riscv_vmsltu_vx_u32m2_b16(c_u32, 0x3e000000, vl);
    vfloat32m2_t c = __riscv_vfmerge_vfm_f32m2(
               __riscv_vreinterpret_v_u32m2_f32m2(c_u32), 0.0f, mask, vl);
    ind = __riscv_vmerge_vxm_u32m2(ind, 0, mask, vl);
    ind = __riscv_vsll_vx_u32m2(ind, 2, vl);

    tmp = __riscv_vfsub_vv_f32m2(x_hi, c, vl);
    x_hi = __riscv_vfadd_vv_f32m2(tmp, x_lo, vl);
    tmp = __riscv_vfsub_vv_f32m2(x_hi, tmp, vl);
    x_lo = __riscv_vfsub_vv_f32m2(x_lo, tmp, vl);

    tmp = __riscv_vfmul_vv_f32m2(x_lo, x_hi, vl);
    tmp = __riscv_vfmul_vf_f32m2(tmp, 2.0f, vl);
    vfloat32m2_t x2 = __riscv_vfmul_vv_f32m2(x_hi, x_hi, vl);
    vfloat32m2_t x2_lo = __riscv_vfmsub_vv_f32m2(x_hi, x_hi, x2, vl);
    x2_lo = __riscv_vfadd_vv_f32m2(tmp, x2_lo, vl);

    vfloat32m2_t sin_r_h = __riscv_vfmadd_vf_f32m2(x2, 0x1.10ea6ep-7f, 
                            __riscv_vfmv_v_f_f32m2(-0x1.555552p-3f, vl), vl);
    tmp = __riscv_vfmul_vv_f32m2(sin_r_h, x_lo, vl);
    tmp = __riscv_vfmadd_vv_f32m2(sin_r_h, x_hi, tmp, vl);
    sin_r_h = __riscv_vfmadd_vv_f32m2(tmp, x2, x_hi, vl);
    vfloat32m2_t sin_r_l = __riscv_vfsub_vv_f32m2(x_hi, sin_r_h, vl);
    sin_r_l = __riscv_vfmadd_vv_f32m2(tmp, x2, sin_r_l, vl);
    sin_r_l = __riscv_vfadd_vv_f32m2(sin_r_l, x_lo, vl);
    
    vfloat32m2_t cos_r_l = __riscv_vfmadd_vf_f32m2(x2, 0x1.551586p-5f,
                            __riscv_vfmv_v_f_f32m2(-0x1.fffff6p-2f, vl), vl);
    vfloat32m2_t cos_r_h = __riscv_vfmadd_vv_f32m2(cos_r_l, x2, 
                            __riscv_vfmv_v_f_f32m2(0x1p0f, vl), vl);
    tmp = __riscv_vfrsub_vf_f32m2(cos_r_h, 0x1p0f, vl);
    tmp = __riscv_vfmadd_vv_f32m2(cos_r_l, x2, tmp, vl);
    cos_r_l = __riscv_vfmadd_vv_f32m2(cos_r_l, x2_lo, tmp, vl);
    
    vfloat32m2_t cos_hi = __riscv_vloxei32_v_f32m2(table_cos_hi_f, ind, vl);
    vfloat32m2_t cos_lo = __riscv_vloxei32_v_f32m2(table_cos_lo_f, ind, vl);
    vfloat32m2_t sin_hi = __riscv_vloxei32_v_f32m2(table_sin_hi_f, ind, vl);
    vfloat32m2_t sin_lo = __riscv_vloxei32_v_f32m2(table_sin_lo_f, ind, vl);
    
    sinrx = __riscv_vfmul_vv_f32m2(cos_lo, sin_r_h, vl);
    sinrx = __riscv_vfmacc_vv_f32m2(sinrx, cos_hi, sin_r_l, vl);
    sinrx = __riscv_vfmacc_vv_f32m2(sinrx, sin_hi, cos_r_l, vl);
    sinrx = __riscv_vfmacc_vv_f32m2(sinrx, sin_lo, cos_r_h, vl);    
    vfloat32m2_t t1h = __riscv_vfmadd_vv_f32m2(cos_hi, sin_r_h, sinrx, vl);
    tmp = __riscv_vfmul_vv_f32m2(cos_hi, sin_r_h, vl);
    tmp2 = __riscv_vfmsub_vv_f32m2(cos_hi, sin_r_h, tmp, vl);
    vfloat32m2_t t1l = __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(
                            __riscv_vfsub_vv_f32m2(tmp, t1h, vl), sinrx, vl), tmp2, vl);
    vfloat32m2_t t2h = __riscv_vfmadd_vv_f32m2(sin_hi, cos_r_h, t1h, vl);
    tmp = __riscv_vfmul_vv_f32m2(sin_hi, cos_r_h, vl);
    tmp2 = __riscv_vfmsub_vv_f32m2(sin_hi, cos_r_h, tmp, vl);
    vfloat32m2_t t2l = __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(
                            __riscv_vfsub_vv_f32m2(tmp, t2h, vl), t1h, vl), tmp2, vl);
    sinrx = __riscv_vfadd_vv_f32m2(t2l, t1l, vl);
    sinrx = __riscv_vfadd_vv_f32m2(sinrx, t2h, vl);
    
    cosrx = __riscv_vfmul_vv_f32m2(sin_hi, sin_r_l, vl);
    cosrx = __riscv_vfmacc_vv_f32m2(cosrx, sin_lo, sin_r_h, vl);
    cosrx = __riscv_vfmsac_vv_f32m2(cosrx, cos_hi, cos_r_l, vl);
    cosrx = __riscv_vfmacc_vv_f32m2(cosrx, cos_lo, cos_r_h, vl);
    cosrx = __riscv_vfnmsac_vv_f32m2(cosrx, sin_hi, sin_r_h, vl);
    cosrx = __riscv_vfmacc_vv_f32m2(cosrx, cos_hi, cos_r_h, vl);   
}

void __riscv_vsincos_kernel_f32m4(vfloat32m4_t x, vfloat32m4_t x_l, vfloat32m4_t& sinrx, vfloat32m4_t& cosrx, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x_lo, x_hi, tmp, tmp2;
    x_hi = __riscv_vmv_v_v_f32m4(x, vl);
    x_lo = __riscv_vmv_v_v_f32m4(x_l, vl);

    vuint32m4_t ix = __riscv_vreinterpret_v_f32m4_u32m4(x_hi);
    vuint32m4_t c_u32 = __riscv_vadd_vx_u32m4(ix, 0x00080000, vl);
    c_u32 = __riscv_vand_vx_u32m4(c_u32, 0x7ff00000, vl);
    vuint32m4_t ind = __riscv_vsrl_vx_u32m4(c_u32, 20, vl);
    ind = __riscv_vsub_vx_u32m4(ind, 991, vl);
    
    vbool8_t mask = __riscv_vmsltu_vx_u32m4_b8(c_u32, 0x3e000000, vl);
    vfloat32m4_t c = __riscv_vfmerge_vfm_f32m4(
               __riscv_vreinterpret_v_u32m4_f32m4(c_u32), 0.0f, mask, vl);
    ind = __riscv_vmerge_vxm_u32m4(ind, 0, mask, vl);
    ind = __riscv_vsll_vx_u32m4(ind, 2, vl);

    tmp = __riscv_vfsub_vv_f32m4(x_hi, c, vl);
    x_hi = __riscv_vfadd_vv_f32m4(tmp, x_lo, vl);
    tmp = __riscv_vfsub_vv_f32m4(x_hi, tmp, vl);
    x_lo = __riscv_vfsub_vv_f32m4(x_lo, tmp, vl);

    tmp = __riscv_vfmul_vv_f32m4(x_lo, x_hi, vl);
    tmp = __riscv_vfmul_vf_f32m4(tmp, 2.0f, vl);
    vfloat32m4_t x2 = __riscv_vfmul_vv_f32m4(x_hi, x_hi, vl);
    vfloat32m4_t x2_lo = __riscv_vfmsub_vv_f32m4(x_hi, x_hi, x2, vl);
    x2_lo = __riscv_vfadd_vv_f32m4(tmp, x2_lo, vl);

    vfloat32m4_t sin_r_h = __riscv_vfmadd_vf_f32m4(x2, 0x1.10ea6ep-7f, 
                            __riscv_vfmv_v_f_f32m4(-0x1.555552p-3f, vl), vl);
    tmp = __riscv_vfmul_vv_f32m4(sin_r_h, x_lo, vl);
    tmp = __riscv_vfmadd_vv_f32m4(sin_r_h, x_hi, tmp, vl);
    sin_r_h = __riscv_vfmadd_vv_f32m4(tmp, x2, x_hi, vl);
    vfloat32m4_t sin_r_l = __riscv_vfsub_vv_f32m4(x_hi, sin_r_h, vl);
    sin_r_l = __riscv_vfmadd_vv_f32m4(tmp, x2, sin_r_l, vl);
    sin_r_l = __riscv_vfadd_vv_f32m4(sin_r_l, x_lo, vl);
    
    vfloat32m4_t cos_r_l = __riscv_vfmadd_vf_f32m4(x2, 0x1.551586p-5f,
                            __riscv_vfmv_v_f_f32m4(-0x1.fffff6p-2f, vl), vl);
    vfloat32m4_t cos_r_h = __riscv_vfmadd_vv_f32m4(cos_r_l, x2, 
                            __riscv_vfmv_v_f_f32m4(0x1p0f, vl), vl);
    tmp = __riscv_vfrsub_vf_f32m4(cos_r_h, 0x1p0f, vl);
    tmp = __riscv_vfmadd_vv_f32m4(cos_r_l, x2, tmp, vl);
    cos_r_l = __riscv_vfmadd_vv_f32m4(cos_r_l, x2_lo, tmp, vl);
    
    vfloat32m4_t cos_hi = __riscv_vloxei32_v_f32m4(table_cos_hi_f, ind, vl);
    vfloat32m4_t cos_lo = __riscv_vloxei32_v_f32m4(table_cos_lo_f, ind, vl);
    vfloat32m4_t sin_hi = __riscv_vloxei32_v_f32m4(table_sin_hi_f, ind, vl);
    vfloat32m4_t sin_lo = __riscv_vloxei32_v_f32m4(table_sin_lo_f, ind, vl);
    
    sinrx = __riscv_vfmul_vv_f32m4(cos_lo, sin_r_h, vl);
    sinrx = __riscv_vfmacc_vv_f32m4(sinrx, cos_hi, sin_r_l, vl);
    sinrx = __riscv_vfmacc_vv_f32m4(sinrx, sin_hi, cos_r_l, vl);
    sinrx = __riscv_vfmacc_vv_f32m4(sinrx, sin_lo, cos_r_h, vl);    
    vfloat32m4_t t1h = __riscv_vfmadd_vv_f32m4(cos_hi, sin_r_h, sinrx, vl);
    tmp = __riscv_vfmul_vv_f32m4(cos_hi, sin_r_h, vl);
    tmp2 = __riscv_vfmsub_vv_f32m4(cos_hi, sin_r_h, tmp, vl);
    vfloat32m4_t t1l = __riscv_vfadd_vv_f32m4(__riscv_vfadd_vv_f32m4(
                            __riscv_vfsub_vv_f32m4(tmp, t1h, vl), sinrx, vl), tmp2, vl);
    vfloat32m4_t t2h = __riscv_vfmadd_vv_f32m4(sin_hi, cos_r_h, t1h, vl);
    tmp = __riscv_vfmul_vv_f32m4(sin_hi, cos_r_h, vl);
    tmp2 = __riscv_vfmsub_vv_f32m4(sin_hi, cos_r_h, tmp, vl);
    vfloat32m4_t t2l = __riscv_vfadd_vv_f32m4(__riscv_vfadd_vv_f32m4(
                            __riscv_vfsub_vv_f32m4(tmp, t2h, vl), t1h, vl), tmp2, vl);
    sinrx = __riscv_vfadd_vv_f32m4(t2l, t1l, vl);
    sinrx = __riscv_vfadd_vv_f32m4(sinrx, t2h, vl);
    
    cosrx = __riscv_vfmul_vv_f32m4(sin_hi, sin_r_l, vl);
    cosrx = __riscv_vfmacc_vv_f32m4(cosrx, sin_lo, sin_r_h, vl);
    cosrx = __riscv_vfmsac_vv_f32m4(cosrx, cos_hi, cos_r_l, vl);
    cosrx = __riscv_vfmacc_vv_f32m4(cosrx, cos_lo, cos_r_h, vl);
    cosrx = __riscv_vfnmsac_vv_f32m4(cosrx, sin_hi, sin_r_h, vl);
    cosrx = __riscv_vfmacc_vv_f32m4(cosrx, cos_hi, cos_r_h, vl);   
}

vfloat32m1_t __riscv_vsin_f32m1(vfloat32m1_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m1(avl);
    vfloat32m1_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool32_t mask_pi_4;
    vuint32m1_t quadrants;
    
    __riscv_vreduction_pi4_f32m1(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f32m1(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f32m1(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f32m1(cosr, sinr, mask_pi_4, vl);
    
    vfloat32m1_t res;
    vfloat32m1_t nsinrx = __riscv_vfmul_vf_f32m1(sinrx, -1.0f, vl);
    vfloat32m1_t ncosrx = __riscv_vfmul_vf_f32m1(cosrx, -1.0f, vl);
    
    vuint32m1_t signx = __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(x), 
                                            0x80000000, vl);  
    vbool32_t mask = __riscv_vmseq_vx_u32m1_b32(signx, 0, vl);
    res = __riscv_vmv_v_v_f32m1(sinrx, vl);
    sinrx = __riscv_vmerge_vvm_f32m1(nsinrx, sinrx, mask, vl);
    nsinrx = __riscv_vmerge_vvm_f32m1(res, nsinrx, mask, vl);
    res = __riscv_vmv_v_v_f32m1(cosrx, vl);
    cosrx = __riscv_vmerge_vvm_f32m1(ncosrx, cosrx, mask, vl);
    ncosrx = __riscv_vmerge_vvm_f32m1(res, ncosrx, mask, vl);
    
    
    mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f32m1(res, sinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f32m1(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f32m1(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f32m1(res, ncosrx, mask, vl);
    
#ifndef __FAST_MATH__
    vuint32m1_t ix = __riscv_vand_vx_u32m1(
                 __riscv_vreinterpret_v_f32m1_u32m1(x), 0x7fffffff, vl);
    vbool32_t mask_NaN = __riscv_vmsgeu_vx_u32m1_b32 (ix, 0x7f800000, vl);
    res = __riscv_vmerge_vvm_f32m1(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b32(mask_NaN,
                  __riscv_vmsltu_vx_u32m1_b32(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b32(mask_NaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        res = __riscv_vfmerge_vfm_f32m1(res, x1, mask_NaN, vl);
    }
#endif

    return res;            
}

vfloat32m2_t __riscv_vsin_f32m2(vfloat32m2_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m2(avl);
    vfloat32m2_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool16_t mask_pi_4;
    vuint32m2_t quadrants;
    
    __riscv_vreduction_pi4_f32m2(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f32m2(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f32m2(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f32m2(cosr, sinr, mask_pi_4, vl);
    
    vfloat32m2_t res;
    vfloat32m2_t nsinrx = __riscv_vfmul_vf_f32m2(sinrx, -1.0f, vl);
    vfloat32m2_t ncosrx = __riscv_vfmul_vf_f32m2(cosrx, -1.0f, vl);
    
    vuint32m2_t signx = __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(x), 
                                            0x80000000, vl);  
    vbool16_t mask = __riscv_vmseq_vx_u32m2_b16(signx, 0, vl);
    res = __riscv_vmv_v_v_f32m2(sinrx, vl);
    sinrx = __riscv_vmerge_vvm_f32m2(nsinrx, sinrx, mask, vl);
    nsinrx = __riscv_vmerge_vvm_f32m2(res, nsinrx, mask, vl);
    res = __riscv_vmv_v_v_f32m2(cosrx, vl);
    cosrx = __riscv_vmerge_vvm_f32m2(ncosrx, cosrx, mask, vl);
    ncosrx = __riscv_vmerge_vvm_f32m2(res, ncosrx, mask, vl);
    
    
    mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f32m2(res, sinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f32m2(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f32m2(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f32m2(res, ncosrx, mask, vl);

#ifndef __FAST_MATH__
    vuint32m2_t ix = __riscv_vand_vx_u32m2(
                 __riscv_vreinterpret_v_f32m2_u32m2(x), 0x7fffffff, vl);
    vbool16_t mask_NaN = __riscv_vmsgeu_vx_u32m2_b16 (ix, 0x7f800000, vl);
    res = __riscv_vmerge_vvm_f32m2(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b16(mask_NaN,
                  __riscv_vmsltu_vx_u32m2_b16(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_NaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        res = __riscv_vfmerge_vfm_f32m2(res, x1, mask_NaN, vl);
    }
#endif

    return res;            
}

vfloat32m4_t __riscv_vsin_f32m4(vfloat32m4_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool8_t mask_pi_4;
    vuint32m4_t quadrants;
    
    __riscv_vreduction_pi4_f32m4(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f32m4(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f32m4(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f32m4(cosr, sinr, mask_pi_4, vl);
    
    vfloat32m4_t res;
    vfloat32m4_t nsinrx = __riscv_vfmul_vf_f32m4(sinrx, -1.0f, vl);
    vfloat32m4_t ncosrx = __riscv_vfmul_vf_f32m4(cosrx, -1.0f, vl);
    
    vuint32m4_t signx = __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(x), 
                                            0x80000000, vl);  
    vbool8_t mask = __riscv_vmseq_vx_u32m4_b8(signx, 0, vl);
    res = __riscv_vmv_v_v_f32m4(sinrx, vl);
    sinrx = __riscv_vmerge_vvm_f32m4(nsinrx, sinrx, mask, vl);
    nsinrx = __riscv_vmerge_vvm_f32m4(res, nsinrx, mask, vl);
    res = __riscv_vmv_v_v_f32m4(cosrx, vl);
    cosrx = __riscv_vmerge_vvm_f32m4(ncosrx, cosrx, mask, vl);
    ncosrx = __riscv_vmerge_vvm_f32m4(res, ncosrx, mask, vl);
    
    
    mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f32m4(res, sinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f32m4(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f32m4(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f32m4(res, ncosrx, mask, vl);

#ifndef __FAST_MATH__
    vuint32m4_t ix = __riscv_vand_vx_u32m4(
                 __riscv_vreinterpret_v_f32m4_u32m4(x), 0x7fffffff, vl);
    vbool8_t mask_NaN = __riscv_vmsgeu_vx_u32m4_b8 (ix, 0x7f800000, vl);
    res = __riscv_vmerge_vvm_f32m4(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b8(mask_NaN,
                  __riscv_vmsltu_vx_u32m4_b8(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_NaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        res = __riscv_vfmerge_vfm_f32m4(res, x1, mask_NaN, vl);
    }
#endif

    return res;            
}

vfloat32m8_t __riscv_vsin_f32m8(vfloat32m8_t x, size_t avl)
{ 
    vfloat32m8_t res;
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1 = __riscv_vsin_f32m4(x1, vl);
    res = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e32m4(avl-vl);
        x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1 = __riscv_vsin_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}

vfloat32m1_t __riscv_vcos_f32m1(vfloat32m1_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m1(avl);
    vfloat32m1_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool32_t mask_pi_4;
    vuint32m1_t quadrants;
    
    __riscv_vreduction_pi4_f32m1(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f32m1(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f32m1(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f32m1(cosr, sinr, mask_pi_4, vl);
    
    vfloat32m1_t res = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t nsinrx = __riscv_vfmul_vf_f32m1(sinrx, -1.0f, vl);
    vfloat32m1_t ncosrx = __riscv_vfmul_vf_f32m1(cosrx, -1.0f, vl);
    
    vbool32_t mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f32m1(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f32m1(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f32m1(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f32m1(res, sinrx, mask, vl);
    
#ifndef __FAST_MATH__
    vuint32m1_t ix = __riscv_vand_vx_u32m1(
                 __riscv_vreinterpret_v_f32m1_u32m1(x), 0x7fffffff, vl);
    vbool32_t mask_NaN = __riscv_vmsgeu_vx_u32m1_b32 (ix, 0x7f800000, vl);
    res = __riscv_vmerge_vvm_f32m1(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b32(mask_NaN,
                  __riscv_vmsltu_vx_u32m1_b32(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b32(mask_NaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        res = __riscv_vfmerge_vfm_f32m1(res, x1, mask_NaN, vl);
    }
#endif
            
    return res;
}

vfloat32m2_t __riscv_vcos_f32m2(vfloat32m2_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m2(avl);
    vfloat32m2_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool16_t mask_pi_4;
    vuint32m2_t quadrants;
    
    __riscv_vreduction_pi4_f32m2(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f32m2(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f32m2(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f32m2(cosr, sinr, mask_pi_4, vl);
    
    vfloat32m2_t res = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t nsinrx = __riscv_vfmul_vf_f32m2(sinrx, -1.0f, vl);
    vfloat32m2_t ncosrx = __riscv_vfmul_vf_f32m2(cosrx, -1.0f, vl);
    
    vbool16_t mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f32m2(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f32m2(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f32m2(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f32m2(res, sinrx, mask, vl);
    
#ifndef __FAST_MATH__
    vuint32m2_t ix = __riscv_vand_vx_u32m2(
                 __riscv_vreinterpret_v_f32m2_u32m2(x), 0x7fffffff, vl);
    vbool16_t mask_NaN = __riscv_vmsgeu_vx_u32m2_b16 (ix, 0x7f800000, vl);
    res = __riscv_vmerge_vvm_f32m2(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b16(mask_NaN,
                  __riscv_vmsltu_vx_u32m2_b16(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_NaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        res = __riscv_vfmerge_vfm_f32m2(res, x1, mask_NaN, vl);
    }
#endif

    return res;
}

vfloat32m4_t __riscv_vcos_f32m4(vfloat32m4_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool8_t mask_pi_4;
    vuint32m4_t quadrants;
    
    __riscv_vreduction_pi4_f32m4(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f32m4(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f32m4(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f32m4(cosr, sinr, mask_pi_4, vl);
    
    vfloat32m4_t res = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t nsinrx = __riscv_vfmul_vf_f32m4(sinrx, -1.0f, vl);
    vfloat32m4_t ncosrx = __riscv_vfmul_vf_f32m4(cosrx, -1.0f, vl);
    
    vbool8_t mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f32m4(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f32m4(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f32m4(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f32m4(res, sinrx, mask, vl);

#ifndef __FAST_MATH__
    vuint32m4_t ix = __riscv_vand_vx_u32m4(
                 __riscv_vreinterpret_v_f32m4_u32m4(x), 0x7fffffff, vl);
    vbool8_t mask_NaN = __riscv_vmsgeu_vx_u32m4_b8 (ix, 0x7f800000, vl);
    res = __riscv_vmerge_vvm_f32m4(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b8(mask_NaN,
                  __riscv_vmsltu_vx_u32m4_b8(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_NaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        res = __riscv_vfmerge_vfm_f32m4(res, x1, mask_NaN, vl);
    }
#endif

    return res;
}

vfloat32m8_t __riscv_vcos_f32m8(vfloat32m8_t x, size_t avl)
{ 
    vfloat32m8_t res;
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1 = __riscv_vcos_f32m4(x1, vl);
    res = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e32m4(avl-vl);
        x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1 = __riscv_vcos_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}

void __riscv_vsincos_f32m1(vfloat32m1_t x, vfloat32m1_t& sinx, vfloat32m1_t& cosx, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m1(avl);
    vfloat32m1_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool32_t mask_pi_4;
    vuint32m1_t quadrants;
    
    __riscv_vreduction_pi4_f32m1(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f32m1(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f32m1(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f32m1(cosr, sinr, mask_pi_4, vl);
         
    vfloat32m1_t nsinrx = __riscv_vfmul_vf_f32m1(sinrx, -1.0f, vl);
    vfloat32m1_t ncosrx = __riscv_vfmul_vf_f32m1(cosrx, -1.0f, vl);
     
    vuint32m1_t signx = __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(x), 
                                            0x80000000, vl);  
    vbool32_t mask = __riscv_vmseq_vx_u32m1_b32(signx, 0, vl);
    vfloat32m1_t ssinrx = __riscv_vmerge_vvm_f32m1(nsinrx, sinrx, mask, vl);
    vfloat32m1_t snsinrx = __riscv_vmerge_vvm_f32m1(sinrx, nsinrx, mask, vl);
    vfloat32m1_t scosrx = __riscv_vmerge_vvm_f32m1(ncosrx, cosrx, mask, vl);
    vfloat32m1_t sncosrx = __riscv_vmerge_vvm_f32m1(cosrx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 0, vl);
    sinx = __riscv_vmerge_vvm_f32m1(sinx, ssinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f32m1(cosx, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 1, vl);
    sinx = __riscv_vmerge_vvm_f32m1(sinx, scosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f32m1(cosx, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 2, vl);
    sinx = __riscv_vmerge_vvm_f32m1(sinx, snsinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f32m1(cosx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m1_b32(quadrants, 3, vl);
    sinx = __riscv_vmerge_vvm_f32m1(sinx, sncosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f32m1(cosx, sinrx, mask, vl);
    
#ifndef __FAST_MATH__
    vuint32m1_t ix = __riscv_vand_vx_u32m1(
                 __riscv_vreinterpret_v_f32m1_u32m1(x), 0x7fffffff, vl);
    vbool32_t mask_NaN = __riscv_vmsgeu_vx_u32m1_b32 (ix, 0x7f800000, vl);
    sinx = __riscv_vmerge_vvm_f32m1(sinx, x, mask_NaN, vl);
    cosx = __riscv_vmerge_vvm_f32m1(cosx, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b32(mask_NaN,
                  __riscv_vmsltu_vx_u32m1_b32(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b32(mask_NaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        sinx = __riscv_vfmerge_vfm_f32m1(sinx, x1, mask_NaN, vl);
        cosx = __riscv_vfmerge_vfm_f32m1(cosx, x1, mask_NaN, vl);
    }
#endif
}

void __riscv_vsincos_f32m2(vfloat32m2_t x, vfloat32m2_t& sinx, vfloat32m2_t& cosx, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m2(avl);
    vfloat32m2_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool16_t mask_pi_4;
    vuint32m2_t quadrants;
    
    __riscv_vreduction_pi4_f32m2(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f32m2(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f32m2(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f32m2(cosr, sinr, mask_pi_4, vl);
         
    vfloat32m2_t nsinrx = __riscv_vfmul_vf_f32m2(sinrx, -1.0f, vl);
    vfloat32m2_t ncosrx = __riscv_vfmul_vf_f32m2(cosrx, -1.0f, vl);
    
    vuint32m2_t signx = __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(x), 
                                            0x80000000, vl);  
    vbool16_t mask = __riscv_vmseq_vx_u32m2_b16(signx, 0, vl);
    vfloat32m2_t ssinrx = __riscv_vmerge_vvm_f32m2(nsinrx, sinrx, mask, vl);
    vfloat32m2_t snsinrx = __riscv_vmerge_vvm_f32m2(sinrx, nsinrx, mask, vl);
    vfloat32m2_t scosrx = __riscv_vmerge_vvm_f32m2(ncosrx, cosrx, mask, vl);
    vfloat32m2_t sncosrx = __riscv_vmerge_vvm_f32m2(cosrx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 0, vl);
    sinx = __riscv_vmerge_vvm_f32m2(sinx, ssinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f32m2(cosx, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 1, vl);
    sinx = __riscv_vmerge_vvm_f32m2(sinx, scosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f32m2(cosx, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 2, vl);
    sinx = __riscv_vmerge_vvm_f32m2(sinx, snsinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f32m2(cosx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m2_b16(quadrants, 3, vl);
    sinx = __riscv_vmerge_vvm_f32m2(sinx, sncosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f32m2(cosx, sinrx, mask, vl);
    
#ifndef __FAST_MATH__
    vuint32m2_t ix = __riscv_vand_vx_u32m2(
                 __riscv_vreinterpret_v_f32m2_u32m2(x), 0x7fffffff, vl);
    vbool16_t mask_NaN = __riscv_vmsgeu_vx_u32m2_b16 (ix, 0x7f800000, vl);
    sinx = __riscv_vmerge_vvm_f32m2(sinx, x, mask_NaN, vl);
    cosx = __riscv_vmerge_vvm_f32m2(cosx, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b16(mask_NaN,
                  __riscv_vmsltu_vx_u32m2_b16(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_NaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        sinx = __riscv_vfmerge_vfm_f32m2(sinx, x1, mask_NaN, vl);
        cosx = __riscv_vfmerge_vfm_f32m2(cosx, x1, mask_NaN, vl);
    }
#endif
}

void __riscv_vsincos_f32m4(vfloat32m4_t x, vfloat32m4_t& sinx, vfloat32m4_t& cosx, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool8_t mask_pi_4;
    vuint32m4_t quadrants;
    
    __riscv_vreduction_pi4_f32m4(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f32m4(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f32m4(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f32m4(cosr, sinr, mask_pi_4, vl);
         
    vfloat32m4_t nsinrx = __riscv_vfmul_vf_f32m4(sinrx, -1.0f, vl);
    vfloat32m4_t ncosrx = __riscv_vfmul_vf_f32m4(cosrx, -1.0f, vl);
    
    vuint32m4_t signx = __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(x), 
                                            0x80000000, vl);  
    vbool8_t mask = __riscv_vmseq_vx_u32m4_b8(signx, 0, vl);
    vfloat32m4_t ssinrx = __riscv_vmerge_vvm_f32m4(nsinrx, sinrx, mask, vl);
    vfloat32m4_t snsinrx = __riscv_vmerge_vvm_f32m4(sinrx, nsinrx, mask, vl);
    vfloat32m4_t scosrx = __riscv_vmerge_vvm_f32m4(ncosrx, cosrx, mask, vl);
    vfloat32m4_t sncosrx = __riscv_vmerge_vvm_f32m4(cosrx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 0, vl);
    sinx = __riscv_vmerge_vvm_f32m4(sinx, ssinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f32m4(cosx, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 1, vl);
    sinx = __riscv_vmerge_vvm_f32m4(sinx, scosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f32m4(cosx, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 2, vl);
    sinx = __riscv_vmerge_vvm_f32m4(sinx, snsinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f32m4(cosx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u32m4_b8(quadrants, 3, vl);
    sinx = __riscv_vmerge_vvm_f32m4(sinx, sncosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f32m4(cosx, sinrx, mask, vl);
    
#ifndef __FAST_MATH__
    vuint32m4_t ix = __riscv_vand_vx_u32m4(
                 __riscv_vreinterpret_v_f32m4_u32m4(x), 0x7fffffff, vl);
    vbool8_t mask_NaN = __riscv_vmsgeu_vx_u32m4_b8 (ix, 0x7f800000, vl);
    sinx = __riscv_vmerge_vvm_f32m4(sinx, x, mask_NaN, vl);
    cosx = __riscv_vmerge_vvm_f32m4(cosx, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b8(mask_NaN,
                  __riscv_vmsltu_vx_u32m4_b8(ix, 0x7fc00000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b8(mask_NaN, vl);
    if (issNaN) {
        volatile float x1 = 0.0f/0.0f;
        sinx = __riscv_vfmerge_vfm_f32m4(sinx, x1, mask_NaN, vl);
        cosx = __riscv_vfmerge_vfm_f32m4(cosx, x1, mask_NaN, vl);
    }
#endif
}

void __riscv_vsincos_f32m8(vfloat32m8_t x, vfloat32m8_t& sinx, vfloat32m8_t& cosx, size_t avl)
{ 
    vfloat32m8_t res;
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    vfloat32m4_t sinx1 = __riscv_vget_v_f32m8_f32m4(sinx, 0);
    vfloat32m4_t cosx1 = __riscv_vget_v_f32m8_f32m4(cosx, 0);
    __riscv_vsincos_f32m4(x1, sinx1, cosx1, vl);
    sinx = __riscv_vset_v_f32m4_f32m8(sinx, 0, sinx1);
    cosx = __riscv_vset_v_f32m4_f32m8(cosx, 0, cosx1);
    if(avl > vl){
        vl = __riscv_vsetvl_e32m4(avl-vl);
        x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
        sinx1 = __riscv_vget_v_f32m8_f32m4(sinx, 1);
        cosx1 = __riscv_vget_v_f32m8_f32m4(cosx, 1);
        __riscv_vsincos_f32m4(x1, sinx1, cosx1, vl);
        sinx = __riscv_vset_v_f32m4_f32m8(sinx, 1, sinx1);
        cosx = __riscv_vset_v_f32m4_f32m8(cosx, 1, cosx1);
    }
}

#endif /* __riscv_v_intrinsic */