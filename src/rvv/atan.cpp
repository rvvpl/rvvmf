/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025-2026 The RVVMF Authors (Egor Korotin)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  atan2.cpp                                    *
 *   Contains: intrinsic functions atan2 for f64, f32    *
 *             and f16                                   *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input VL number of elements in vector register        *
 *                                                       *
 * Computes the atan2 of input vector V                  *
 *                                                       *
 * Algorithm:                                            *
 *    1) Argument reduction to (-1/8, 1/8) using         *
 *       look-up tables of size 16                       *
 *    2) Polynomial degrees: f64 - 15, f32 - 7, f16 - 3  *
 *    3) Reconstruction of the result                    *
 *                                                       *
 *********************************************************
*/

#include <riscv_vector.h>

static const double borders1[4] = {-0x0p0, 0x1p-3, 0x1.8p-2, 0x1.5p-1};
static const double borders2[4] = {0x1p0, 0x1.9p0, 0x1.6p1, 0x1.04p3};
static const double magic_consts1[4] = {-0x1p-86, 0x1p-86, -0x1p0, 0x1p0};
static const uint64_t magic_consts2[4] = {0x7ff8000000000000, 0x7ff0000000000000, 0, 0};
static const double pol_coef1[4] = {-0x1.02e52393cf313p-4, 0x1.3abbe13593d39p-4, -0x1.745bfe2487e06p-4, 0x1.c71c6fde46a39p-4};
static const double pol_coef2[4] = {-0x1.24924923b4715p-3, 0x1.99999999993f2p-3, -0x1.5555555555555p-2, 0.0};
static const double d_table_hi[8] = {0, 		    0x1.f5b75f92c80ddp-3, 0x1.dac670561bb4fp-2, 0x1.5d58987169b18p-1,
                        	     0x1.cac7c57846f9ep-1,  0x1.1b6e192ebbe44p0, 0x1.5368c951e9cfdp0, 0x1.921fb54442d18p0};
static const double d_table_lo[8] = {-0x1p-1022,            0x1.8ab6e3cf7afbdp-57, 0x1.a2b7f222f65e2p-56, 0x1.0028e4bc5e7cap-57,
                        	     0x1.0dae13ad18a6bp-55, 0x1.b1b466a88828ep-54, -0x1.96f47948a99f1p-54, 0x1.1a62633145c07p-54};
static const double d_k[8]  = {0, 0x1p-2, 0x1p-1, 0x1.ap-1, 0x1.4p0, 2.0, 4.0, -0x1p0};

static const float f_borders[8] = {-0x0p0f, 0x1.cp-2f, 0x1.6p-1f, 0x1.3p0f, 0x1.38p1f, -0x1p0f, 0x1p0f, -0x1.dba1bap-5f};

static const float f_pol_coef[4] = {0x1.ad7058p-4f, -0x1.236616p-3f, 0x1.998d52p-3f, -0x1.55554p-2f};

static const int32_t magic_consts[2] = {0x7f800000, 0x7fc00000};

static const float f_table_hi[5] = {0x0p0f, 0x1.dac67p-2f,    0x1.921fb6p-1f,   0x1.f730bep-1f,   0x1.921fb6p0f},
		   f_table_lo[5] = {-0x1p-126f, 0x1.586ed4p-28f, -0x1.777a5cp-26f, -0x1.afc12cp-26f, -0x1.777a5cp-25f},
		   f_k[5]        = {0x0p0f, 0x1p-1f,          0x1p0f,           0x1.8p0f,         -0x1p0f};

vfloat64m1_t __riscv_vatan_f64m1(vfloat64m1_t x, size_t vl) {
    vfloat64m1_t w, wc, w_l, y, z, k, a, b, pol1, pol2, minus, bl, _b, ch, uh, ul, cl, t;
    vuint64m1_t index;
    vfloat64m1_t sign;
    vbool64_t mask1, mask2, mask3, mask4, mask5, mask6, mask7;
#ifndef __FAST_MATH__
    vbool64_t mask_nan, mask_big;
#endif

    minus = __riscv_vfmv_v_f_f64m1(borders1[0], vl);

    sign = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vv_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(minus), __riscv_vreinterpret_v_f64m1_u64m1(x), vl));
    x = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vv_u64m1(__riscv_vnot_v_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(minus), vl), __riscv_vreinterpret_v_f64m1_u64m1(x), vl));

#ifndef __FAST_MATH__
    mask_nan = __riscv_vmfne_vv_f64m1_b64(x, x, vl);
    x = __riscv_vreinterpret_v_i64m1_f64m1(__riscv_vxor_vv_i64m1_mu(mask_nan, __riscv_vreinterpret_v_f64m1_i64m1(x), __riscv_vreinterpret_v_f64m1_i64m1(x), __riscv_vreinterpret_v_f64m1_i64m1(x), vl));

#endif

    mask1 = __riscv_vmfge_vf_f64m1_b64(x, borders1[1], vl);
    mask2 = __riscv_vmfge_vf_f64m1_b64(x, borders1[2], vl);
    mask3 = __riscv_vmfge_vf_f64m1_b64(x, borders1[3], vl);
    mask4 = __riscv_vmfge_vf_f64m1_b64(x, borders2[0], vl);
    mask5 = __riscv_vmfge_vf_f64m1_b64(x, borders2[1], vl);
    mask6 = __riscv_vmfge_vf_f64m1_b64(x, borders2[2], vl);
    mask7 = __riscv_vmfge_vf_f64m1_b64(x, borders2[3], vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmflt_vf_f64m1_b64(x, 0x1p500, vl);
#endif

    index = __riscv_vmv_v_x_u64m1(0, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask1, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask2, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask3, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask4, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask5, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask6, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask7, index, index, 8, vl);

    k = __riscv_vloxei64_v_f64m1(d_k, index, vl);
    mask7 = __riscv_vmnot_m_b64(mask7, vl);

    a  = __riscv_vfsub_vv_f64m1_mu(mask7, k, x, k, vl);
    b  = __riscv_vfmadd_vv_f64m1_mu(mask7, x, k, __riscv_vfmv_v_f_f64m1(magic_consts1[3], vl), vl);

    bl = __riscv_vfsub_vf_f64m1_mu(mask7, minus, b, magic_consts1[3], vl);

    bl = __riscv_vfmsac_vv_f64m1_mu(mask7, bl, x, k, vl);

#ifndef __FAST_MATH__
    _b = __riscv_vfrdiv_vf_f64m1_mu(mask_big, minus, b, magic_consts1[3], vl);
#else
    _b = __riscv_vfrdiv_vf_f64m1(b, magic_consts1[3], vl);
#endif

    ch = __riscv_vfmul_vv_f64m1(a, _b, vl);

#ifndef __FAST_MATH__
    uh = __riscv_vfmul_vv_f64m1_mu(mask_big, ch, ch, b, vl);
    ul = __riscv_vfmsub_vv_f64m1_mu(mask_big, ch, b, uh, vl);
#else
    uh = __riscv_vfmul_vv_f64m1(ch, b, vl);
    ul = __riscv_vfmsub_vv_f64m1(ch, b, uh, vl);
#endif

    cl = __riscv_vfmul_vv_f64m1(__riscv_vfsub_vv_f64m1(__riscv_vfsub_vv_f64m1(__riscv_vfsub_vv_f64m1(a, uh, vl), ul, vl), __riscv_vfmul_vv_f64m1(ch, bl, vl), vl), _b, vl);

    w = __riscv_vfadd_vv_f64m1(ch, cl, vl);

#ifndef __FAST_MATH__
    wc = __riscv_vfadd_vf_f64m1(w, magic_consts1[1], vl);
#else
    wc = w;
#endif

    a = __riscv_vloxei64_v_f64m1(d_table_hi, index, vl);
    b = __riscv_vloxei64_v_f64m1(d_table_lo, index, vl);

    x = __riscv_vfadd_vv_f64m1(a, w, vl);
    y = __riscv_vfmul_vv_f64m1(wc, wc, vl);

    w_l = __riscv_vfadd_vv_f64m1(__riscv_vfsub_vv_f64m1(ch, w, vl), cl, vl);

    z = __riscv_vfmul_vv_f64m1(y, y, vl);
    t = __riscv_vfsub_vv_f64m1(x, a, vl);

    pol1 = __riscv_vfmadd_vf_f64m1(z, pol_coef1[0], __riscv_vfmv_v_f_f64m1(pol_coef1[2], vl), vl);
    pol2 = __riscv_vfmadd_vf_f64m1(z, pol_coef1[1], __riscv_vfmv_v_f_f64m1(pol_coef1[3], vl), vl);

    w_l = __riscv_vfadd_vv_f64m1(__riscv_vfsub_vv_f64m1(w, t, vl), w_l, vl);

    pol1 = __riscv_vfmadd_vv_f64m1(pol1, z, __riscv_vfmv_v_f_f64m1(pol_coef2[0], vl), vl);
    pol2 = __riscv_vfmadd_vv_f64m1(pol2, z, __riscv_vfmv_v_f_f64m1(pol_coef2[1], vl), vl);

#ifndef __FAST_MATH__
    x = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vor_vx_u64m1_mu(mask_nan, __riscv_vreinterpret_v_f64m1_u64m1(x), __riscv_vreinterpret_v_f64m1_u64m1(x), magic_consts2[0], vl));
#endif

    pol1 = __riscv_vfmadd_vv_f64m1(pol1, z, __riscv_vfmv_v_f_f64m1(pol_coef2[2], vl), vl);
    pol2 = __riscv_vfmul_vv_f64m1(pol2, z, vl);

    x = __riscv_vfadd_vv_f64m1(
        x, 
        __riscv_vfadd_vv_f64m1(
            w_l, 
            __riscv_vfsub_vf_f64m1(
                __riscv_vfmacc_vv_f64m1(
                    b, 
                    __riscv_vfmacc_vv_f64m1(pol2, pol1, y, vl), 
                    w, vl), 
                d_table_lo[0], vl), 
            vl), 
        vl);

#ifndef __FAST_MATH__
     x = __riscv_vfmadd_vf_f64m1(x, 0x1p-60, x, vl);
#endif

    return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vor_vv_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(sign), __riscv_vreinterpret_v_f64m1_u64m1(x), vl));
}

vfloat64m2_t __riscv_vatan_f64m2(vfloat64m2_t x, size_t vl) {
    vfloat64m2_t w, wc, w_l, y, z, k, a, b, pol1, pol2, minus, bl, _b, ch, uh, ul, cl, t;
    vuint64m2_t index;
    vfloat64m2_t sign;
    vbool32_t mask1, mask2, mask3, mask4, mask5, mask6, mask7;
#ifndef __FAST_MATH__
    vbool32_t mask_nan, mask_big;
#endif

    minus = __riscv_vfmv_v_f_f64m2(borders1[0], vl);

    sign = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vv_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(minus), __riscv_vreinterpret_v_f64m2_u64m2(x), vl));
    x = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vv_u64m2(__riscv_vnot_v_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(minus), vl), __riscv_vreinterpret_v_f64m2_u64m2(x), vl));

#ifndef __FAST_MATH__
    mask_nan = __riscv_vmfne_vv_f64m2_b32(x, x, vl);
    x = __riscv_vreinterpret_v_i64m2_f64m2(__riscv_vxor_vv_i64m2_mu(mask_nan, __riscv_vreinterpret_v_f64m2_i64m2(x), __riscv_vreinterpret_v_f64m2_i64m2(x), __riscv_vreinterpret_v_f64m2_i64m2(x), vl));

#endif

    mask1 = __riscv_vmfge_vf_f64m2_b32(x, borders1[1], vl);
    mask2 = __riscv_vmfge_vf_f64m2_b32(x, borders1[2], vl);
    mask3 = __riscv_vmfge_vf_f64m2_b32(x, borders1[3], vl);
    mask4 = __riscv_vmfge_vf_f64m2_b32(x, borders2[0], vl);
    mask5 = __riscv_vmfge_vf_f64m2_b32(x, borders2[1], vl);
    mask6 = __riscv_vmfge_vf_f64m2_b32(x, borders2[2], vl);
    mask7 = __riscv_vmfge_vf_f64m2_b32(x, borders2[3], vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmflt_vf_f64m2_b32(x, 0x1p500, vl);
#endif

    index = __riscv_vmv_v_x_u64m2(0, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask1, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask2, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask3, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask4, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask5, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask6, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask7, index, index, 8, vl);

    k = __riscv_vloxei64_v_f64m2(d_k, index, vl);
    mask7 = __riscv_vmnot_m_b32(mask7, vl);

    a  = __riscv_vfsub_vv_f64m2_mu(mask7, k, x, k, vl);
    b  = __riscv_vfmadd_vv_f64m2_mu(mask7, x, k, __riscv_vfmv_v_f_f64m2(magic_consts1[3], vl), vl);

    bl = __riscv_vfsub_vf_f64m2_mu(mask7, minus, b, magic_consts1[3], vl);

    bl = __riscv_vfmsac_vv_f64m2_mu(mask7, bl, x, k, vl);

#ifndef __FAST_MATH__
    _b = __riscv_vfrdiv_vf_f64m2_mu(mask_big, minus, b, magic_consts1[3], vl);
#else
    _b = __riscv_vfrdiv_vf_f64m2(b, magic_consts1[3], vl);
#endif

    ch = __riscv_vfmul_vv_f64m2(a, _b, vl);

#ifndef __FAST_MATH__
    uh = __riscv_vfmul_vv_f64m2_mu(mask_big, ch, ch, b, vl);
    ul = __riscv_vfmsub_vv_f64m2_mu(mask_big, ch, b, uh, vl);
#else
    uh = __riscv_vfmul_vv_f64m2(ch, b, vl);
    ul = __riscv_vfmsub_vv_f64m2(ch, b, uh, vl);
#endif

    cl = __riscv_vfmul_vv_f64m2(__riscv_vfsub_vv_f64m2(__riscv_vfsub_vv_f64m2(__riscv_vfsub_vv_f64m2(a, uh, vl), ul, vl), __riscv_vfmul_vv_f64m2(ch, bl, vl), vl), _b, vl);

    w = __riscv_vfadd_vv_f64m2(ch, cl, vl);

#ifndef __FAST_MATH__
    wc = __riscv_vfadd_vf_f64m2(w, magic_consts1[1], vl);
#else
    wc = w;
#endif

    a = __riscv_vloxei64_v_f64m2(d_table_hi, index, vl);
    b = __riscv_vloxei64_v_f64m2(d_table_lo, index, vl);

    x = __riscv_vfadd_vv_f64m2(a, w, vl);
    y = __riscv_vfmul_vv_f64m2(wc, wc, vl);

    w_l = __riscv_vfadd_vv_f64m2(__riscv_vfsub_vv_f64m2(ch, w, vl), cl, vl);

    z = __riscv_vfmul_vv_f64m2(y, y, vl);
    t = __riscv_vfsub_vv_f64m2(x, a, vl);

    pol1 = __riscv_vfmadd_vf_f64m2(z, pol_coef1[0], __riscv_vfmv_v_f_f64m2(pol_coef1[2], vl), vl);
    pol2 = __riscv_vfmadd_vf_f64m2(z, pol_coef1[1], __riscv_vfmv_v_f_f64m2(pol_coef1[3], vl), vl);

    w_l = __riscv_vfadd_vv_f64m2(__riscv_vfsub_vv_f64m2(w, t, vl), w_l, vl);

    pol1 = __riscv_vfmadd_vv_f64m2(pol1, z, __riscv_vfmv_v_f_f64m2(pol_coef2[0], vl), vl);
    pol2 = __riscv_vfmadd_vv_f64m2(pol2, z, __riscv_vfmv_v_f_f64m2(pol_coef2[1], vl), vl);

#ifndef __FAST_MATH__
    x = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vor_vx_u64m2_mu(mask_nan, __riscv_vreinterpret_v_f64m2_u64m2(x), __riscv_vreinterpret_v_f64m2_u64m2(x), magic_consts2[0], vl));
#endif

    pol1 = __riscv_vfmadd_vv_f64m2(pol1, z, __riscv_vfmv_v_f_f64m2(pol_coef2[2], vl), vl);
    pol2 = __riscv_vfmul_vv_f64m2(pol2, z, vl);

    x = __riscv_vfadd_vv_f64m2(
        x, 
        __riscv_vfadd_vv_f64m2(
            w_l, 
            __riscv_vfsub_vf_f64m2(
                __riscv_vfmacc_vv_f64m2(
                    b, 
                    __riscv_vfmacc_vv_f64m2(pol2, pol1, y, vl), 
                    w, vl), 
                d_table_lo[0], vl), 
            vl), 
        vl);

#ifndef __FAST_MATH__
     x = __riscv_vfmadd_vf_f64m2(x, 0x1p-60, x, vl);
#endif

    return __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vor_vv_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(sign), __riscv_vreinterpret_v_f64m2_u64m2(x), vl));
}

vfloat64m4_t __riscv_vatan_f64m4(vfloat64m4_t x, size_t vl) {
    vfloat64m4_t w, wc, w_l, y, z, k, a, b, pol1, pol2, minus, bl, _b, ch, uh, ul, cl, t;
    vuint64m4_t index;
    vfloat64m4_t sign;
    vbool16_t mask1, mask2, mask3, mask4, mask5, mask6, mask7;
#ifndef __FAST_MATH__
    vbool16_t mask_nan, mask_big;
#endif

    minus = __riscv_vfmv_v_f_f64m4(borders1[0], vl);

    sign = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vand_vv_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(minus), __riscv_vreinterpret_v_f64m4_u64m4(x), vl));
    x = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vand_vv_u64m4(__riscv_vnot_v_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(minus), vl), __riscv_vreinterpret_v_f64m4_u64m4(x), vl));

#ifndef __FAST_MATH__
    mask_nan = __riscv_vmfne_vv_f64m4_b16(x, x, vl);
    x = __riscv_vreinterpret_v_i64m4_f64m4(__riscv_vxor_vv_i64m4_mu(mask_nan, __riscv_vreinterpret_v_f64m4_i64m4(x), __riscv_vreinterpret_v_f64m4_i64m4(x), __riscv_vreinterpret_v_f64m4_i64m4(x), vl));

#endif

    mask1 = __riscv_vmfge_vf_f64m4_b16(x, borders1[1], vl);
    mask2 = __riscv_vmfge_vf_f64m4_b16(x, borders1[2], vl);
    mask3 = __riscv_vmfge_vf_f64m4_b16(x, borders1[3], vl);
    mask4 = __riscv_vmfge_vf_f64m4_b16(x, borders2[0], vl);
    mask5 = __riscv_vmfge_vf_f64m4_b16(x, borders2[1], vl);
    mask6 = __riscv_vmfge_vf_f64m4_b16(x, borders2[2], vl);
    mask7 = __riscv_vmfge_vf_f64m4_b16(x, borders2[3], vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmflt_vf_f64m4_b16(x, 0x1p500, vl);
#endif

    index = __riscv_vmv_v_x_u64m4(0, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask1, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask2, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask3, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask4, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask5, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask6, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask7, index, index, 8, vl);

    k = __riscv_vloxei64_v_f64m4(d_k, index, vl);
    mask7 = __riscv_vmnot_m_b16(mask7, vl);

    a  = __riscv_vfsub_vv_f64m4_mu(mask7, k, x, k, vl);
    b  = __riscv_vfmadd_vv_f64m4_mu(mask7, x, k, __riscv_vfmv_v_f_f64m4(magic_consts1[3], vl), vl);

    bl = __riscv_vfsub_vf_f64m4_mu(mask7, minus, b, magic_consts1[3], vl);

    bl = __riscv_vfmsac_vv_f64m4_mu(mask7, bl, x, k, vl);

#ifndef __FAST_MATH__
    _b = __riscv_vfrdiv_vf_f64m4_mu(mask_big, minus, b, magic_consts1[3], vl);
#else
    _b = __riscv_vfrdiv_vf_f64m4(b, magic_consts1[3], vl);
#endif

    ch = __riscv_vfmul_vv_f64m4(a, _b, vl);

#ifndef __FAST_MATH__
    uh = __riscv_vfmul_vv_f64m4_mu(mask_big, ch, ch, b, vl);
    ul = __riscv_vfmsub_vv_f64m4_mu(mask_big, ch, b, uh, vl);
#else
    uh = __riscv_vfmul_vv_f64m4(ch, b, vl);
    ul = __riscv_vfmsub_vv_f64m4(ch, b, uh, vl);
#endif

    cl = __riscv_vfmul_vv_f64m4(__riscv_vfsub_vv_f64m4(__riscv_vfsub_vv_f64m4(__riscv_vfsub_vv_f64m4(a, uh, vl), ul, vl), __riscv_vfmul_vv_f64m4(ch, bl, vl), vl), _b, vl);

    w = __riscv_vfadd_vv_f64m4(ch, cl, vl);

#ifndef __FAST_MATH__
    wc = __riscv_vfadd_vf_f64m4(w, magic_consts1[1], vl);
#else
    wc = w;
#endif

    a = __riscv_vloxei64_v_f64m4(d_table_hi, index, vl);
    b = __riscv_vloxei64_v_f64m4(d_table_lo, index, vl);

    x = __riscv_vfadd_vv_f64m4(a, w, vl);
    y = __riscv_vfmul_vv_f64m4(wc, wc, vl);

    w_l = __riscv_vfadd_vv_f64m4(__riscv_vfsub_vv_f64m4(ch, w, vl), cl, vl);

    z = __riscv_vfmul_vv_f64m4(y, y, vl);
    t = __riscv_vfsub_vv_f64m4(x, a, vl);

    pol1 = __riscv_vfmadd_vf_f64m4(z, pol_coef1[0], __riscv_vfmv_v_f_f64m4(pol_coef1[2], vl), vl);
    pol2 = __riscv_vfmadd_vf_f64m4(z, pol_coef1[1], __riscv_vfmv_v_f_f64m4(pol_coef1[3], vl), vl);

    w_l = __riscv_vfadd_vv_f64m4(__riscv_vfsub_vv_f64m4(w, t, vl), w_l, vl);

    pol1 = __riscv_vfmadd_vv_f64m4(pol1, z, __riscv_vfmv_v_f_f64m4(pol_coef2[0], vl), vl);
    pol2 = __riscv_vfmadd_vv_f64m4(pol2, z, __riscv_vfmv_v_f_f64m4(pol_coef2[1], vl), vl);

#ifndef __FAST_MATH__
    x = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vor_vx_u64m4_mu(mask_nan, __riscv_vreinterpret_v_f64m4_u64m4(x), __riscv_vreinterpret_v_f64m4_u64m4(x), magic_consts2[0], vl));
#endif

    pol1 = __riscv_vfmadd_vv_f64m4(pol1, z, __riscv_vfmv_v_f_f64m4(pol_coef2[2], vl), vl);
    pol2 = __riscv_vfmul_vv_f64m4(pol2, z, vl);

    x = __riscv_vfadd_vv_f64m4(
        x, 
        __riscv_vfadd_vv_f64m4(
            w_l, 
            __riscv_vfsub_vf_f64m4(
                __riscv_vfmacc_vv_f64m4(
                    b, 
                    __riscv_vfmacc_vv_f64m4(pol2, pol1, y, vl), 
                    w, vl), 
                d_table_lo[0], vl), 
            vl), 
        vl);

#ifndef __FAST_MATH__
     x = __riscv_vfmadd_vf_f64m4(x, 0x1p-60, x, vl);
#endif

    return __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vor_vv_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(sign), __riscv_vreinterpret_v_f64m4_u64m4(x), vl));
}

vfloat64m8_t __riscv_vatan_f64m8(vfloat64m8_t x, size_t vl) {
    vfloat64m8_t w, wc, w_l, y, z, k, a, b, pol1, pol2, minus, bl, _b, ch, uh, ul, cl, t;
    vuint64m8_t index;
    vfloat64m8_t sign;
    vbool8_t mask1, mask2, mask3, mask4, mask5, mask6, mask7;
#ifndef __FAST_MATH__
    vbool8_t mask_nan, mask_big;
#endif

    minus = __riscv_vfmv_v_f_f64m8(borders1[0], vl);

    sign = __riscv_vreinterpret_v_u64m8_f64m8(__riscv_vand_vv_u64m8(__riscv_vreinterpret_v_f64m8_u64m8(minus), __riscv_vreinterpret_v_f64m8_u64m8(x), vl));
    x = __riscv_vreinterpret_v_u64m8_f64m8(__riscv_vand_vv_u64m8(__riscv_vnot_v_u64m8(__riscv_vreinterpret_v_f64m8_u64m8(minus), vl), __riscv_vreinterpret_v_f64m8_u64m8(x), vl));

#ifndef __FAST_MATH__
    mask_nan = __riscv_vmfne_vv_f64m8_b8(x, x, vl);
    x = __riscv_vreinterpret_v_i64m8_f64m8(__riscv_vxor_vv_i64m8_mu(mask_nan, __riscv_vreinterpret_v_f64m8_i64m8(x), __riscv_vreinterpret_v_f64m8_i64m8(x), __riscv_vreinterpret_v_f64m8_i64m8(x), vl));

#endif

    mask1 = __riscv_vmfge_vf_f64m8_b8(x, borders1[1], vl);
    mask2 = __riscv_vmfge_vf_f64m8_b8(x, borders1[2], vl);
    mask3 = __riscv_vmfge_vf_f64m8_b8(x, borders1[3], vl);
    mask4 = __riscv_vmfge_vf_f64m8_b8(x, borders2[0], vl);
    mask5 = __riscv_vmfge_vf_f64m8_b8(x, borders2[1], vl);
    mask6 = __riscv_vmfge_vf_f64m8_b8(x, borders2[2], vl);
    mask7 = __riscv_vmfge_vf_f64m8_b8(x, borders2[3], vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmflt_vf_f64m8_b8(x, 0x1p500, vl);
#endif

    index = __riscv_vmv_v_x_u64m8(0, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask1, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask2, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask3, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask4, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask5, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask6, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask7, index, index, 8, vl);

    k = __riscv_vloxei64_v_f64m8(d_k, index, vl);
    mask7 = __riscv_vmnot_m_b8(mask7, vl);

    a  = __riscv_vfsub_vv_f64m8_mu(mask7, k, x, k, vl);
    b  = __riscv_vfmadd_vv_f64m8_mu(mask7, x, k, __riscv_vfmv_v_f_f64m8(magic_consts1[3], vl), vl);

    bl = __riscv_vfsub_vf_f64m8_mu(mask7, minus, b, magic_consts1[3], vl);
    bl = __riscv_vfmsac_vv_f64m8_mu(mask7, bl, x, k, vl);

#ifndef __FAST_MATH__
    _b = __riscv_vfrdiv_vf_f64m8_mu(mask_big, minus, b, magic_consts1[3], vl);
#else
    _b = __riscv_vfrdiv_vf_f64m8(b, magic_consts1[3], vl);
#endif

    ch = __riscv_vfmul_vv_f64m8(a, _b, vl);

#ifndef __FAST_MATH__
    uh = __riscv_vfmul_vv_f64m8_mu(mask_big, ch, ch, b, vl);
    ul = __riscv_vfmsub_vv_f64m8_mu(mask_big, ch, b, uh, vl);
#else
    uh = __riscv_vfmul_vv_f64m8(ch, b, vl);
    ul = __riscv_vfmsub_vv_f64m8(ch, b, uh, vl);
#endif

    cl = __riscv_vfmul_vv_f64m8(__riscv_vfsub_vv_f64m8(__riscv_vfsub_vv_f64m8(__riscv_vfsub_vv_f64m8(a, uh, vl), ul, vl), __riscv_vfmul_vv_f64m8(ch, bl, vl), vl), _b, vl);

    w = __riscv_vfadd_vv_f64m8(ch, cl, vl);

#ifndef __FAST_MATH__
    wc = __riscv_vfadd_vf_f64m8(w, magic_consts1[1], vl);
#else
    wc = w;
#endif

    a = __riscv_vloxei64_v_f64m8(d_table_hi, index, vl);
    b = __riscv_vloxei64_v_f64m8(d_table_lo, index, vl);

    x = __riscv_vfadd_vv_f64m8(a, w, vl);
    y = __riscv_vfmul_vv_f64m8(wc, wc, vl);

    w_l = __riscv_vfadd_vv_f64m8(__riscv_vfsub_vv_f64m8(ch, w, vl), cl, vl);

    z = __riscv_vfmul_vv_f64m8(y, y, vl);
    t = __riscv_vfsub_vv_f64m8(x, a, vl);

    pol1 = __riscv_vfmadd_vf_f64m8(z, pol_coef1[0], __riscv_vfmv_v_f_f64m8(pol_coef1[2], vl), vl);
    pol2 = __riscv_vfmadd_vf_f64m8(z, pol_coef1[1], __riscv_vfmv_v_f_f64m8(pol_coef1[3], vl), vl);

    w_l = __riscv_vfadd_vv_f64m8(__riscv_vfsub_vv_f64m8(w, t, vl), w_l, vl);

    pol1 = __riscv_vfmadd_vv_f64m8(pol1, z, __riscv_vfmv_v_f_f64m8(pol_coef2[0], vl), vl);
    pol2 = __riscv_vfmadd_vv_f64m8(pol2, z, __riscv_vfmv_v_f_f64m8(pol_coef2[1], vl), vl);

#ifndef __FAST_MATH__
    x = __riscv_vreinterpret_v_u64m8_f64m8(__riscv_vor_vx_u64m8_mu(mask_nan, __riscv_vreinterpret_v_f64m8_u64m8(x), __riscv_vreinterpret_v_f64m8_u64m8(x), magic_consts2[0], vl));
#endif

    pol1 = __riscv_vfmadd_vv_f64m8(pol1, z, __riscv_vfmv_v_f_f64m8(pol_coef2[2], vl), vl);
    pol2 = __riscv_vfmul_vv_f64m8(pol2, z, vl);

    x = __riscv_vfadd_vv_f64m8(
        x, 
        __riscv_vfadd_vv_f64m8(
            w_l, 
            __riscv_vfsub_vf_f64m8(
                __riscv_vfmacc_vv_f64m8(
                    b, 
                    __riscv_vfmacc_vv_f64m8(pol2, pol1, y, vl), 
                    w, vl), 
                d_table_lo[0], vl), 
            vl), 
        vl);

#ifndef __FAST_MATH__
     x = __riscv_vfmadd_vf_f64m8(x, 0x1p-60, x, vl);
#endif

    return __riscv_vreinterpret_v_u64m8_f64m8(__riscv_vor_vv_u64m8(__riscv_vreinterpret_v_f64m8_u64m8(sign), __riscv_vreinterpret_v_f64m8_u64m8(x), vl));
}

vfloat32m1_t __riscv_vatan_f32m1(vfloat32m1_t x, size_t vl) {
  vfloat32m1_t w, w_l, y, z, k, a, b, pol1, pol2, minus, bl, _b, t, tab_h, tab_l;
  vuint32m1_t index;
  vfloat32m1_t sign;
  vbool32_t mask1, mask2, mask3, mask4;
#ifndef __FAST_MATH__
    vbool32_t mask_nan, mask_big;
#endif

  minus = __riscv_vfmv_v_f_f32m1(f_borders[0], vl);

  sign = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(minus), __riscv_vreinterpret_v_f32m1_u32m1(x), vl));
  x = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(__riscv_vnot_v_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(minus), vl), __riscv_vreinterpret_v_f32m1_u32m1(x), vl));
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmfne_vv_f32m1_b32(x, x, vl);
    x = __riscv_vreinterpret_v_i32m1_f32m1(__riscv_vxor_vv_i32m1_mu(mask_nan, __riscv_vreinterpret_v_f32m1_i32m1(x), __riscv_vreinterpret_v_f32m1_i32m1(x), __riscv_vreinterpret_v_f32m1_i32m1(x), vl));
#endif

  mask1 = __riscv_vmfge_vf_f32m1_b32(x, f_borders[1], vl);
  mask2 = __riscv_vmfge_vf_f32m1_b32(x, f_borders[2], vl);
  mask3 = __riscv_vmfge_vf_f32m1_b32(x, f_borders[3], vl);
  mask4 = __riscv_vmfge_vf_f32m1_b32(x, f_borders[4], vl);
#ifndef __FAST_MATH__
  mask_big = __riscv_vmflt_vf_f32m1_b32(x, 0x1p60f, vl);
#endif

  index = __riscv_vmv_v_x_u32m1(0, vl);
  index = __riscv_vadd_vx_u32m1_mu(mask1, index, index, 4, vl);
  index = __riscv_vadd_vx_u32m1_mu(mask2, index, index, 4, vl);
  index = __riscv_vadd_vx_u32m1_mu(mask3, index, index, 4, vl);
  index = __riscv_vadd_vx_u32m1_mu(mask4, index, index, 4, vl);

  k = __riscv_vloxei32_v_f32m1(f_k, index, vl);
  mask4 = __riscv_vmnot_m_b32(mask4, vl);

  a  = __riscv_vfsub_vv_f32m1_mu(mask4, k, x, k, vl);
  b  = __riscv_vfmadd_vv_f32m1_mu(mask4, x, k, __riscv_vfmv_v_f_f32m1(f_borders[6], vl), vl);

#ifndef __FAST_MATH__
  _b = __riscv_vfrdiv_vf_f32m1_mu(mask_big, minus, b, f_borders[6], vl);
  bl = __riscv_vfnmsub_vv_f32m1_mu(mask_big, _b, b, __riscv_vfmv_v_f_f32m1(f_borders[6], vl), vl);
#else
  _b = __riscv_vfrdiv_vf_f32m1(b, f_borders[6], vl);
  bl = __riscv_vfnmsub_vv_f32m1(_b, b, __riscv_vfmv_v_f_f32m1(f_borders[6], vl), vl);
#endif

  bl = __riscv_vfmul_vv_f32m1(_b, bl, vl);
  w = __riscv_vfmul_vv_f32m1(a, _b, vl);
  w_l = __riscv_vfmadd_vv_f32m1(a, bl, __riscv_vfmsub_vv_f32m1(a, _b, w, vl), vl);

  tab_h = __riscv_vloxei32_v_f32m1(f_table_hi, index, vl);
  tab_l = __riscv_vloxei32_v_f32m1(f_table_lo, index, vl);

  y = __riscv_vfmadd_vv_f32m1(w, w, __riscv_vfmv_v_f_f32m1(0x1p-40, vl), vl);
  z = __riscv_vfmul_vv_f32m1(y, y, vl);

  pol1 = __riscv_vfmadd_vf_f32m1(z, f_borders[7], __riscv_vfmv_v_f_f32m1(f_pol_coef[1], vl), vl);
  pol2 = __riscv_vfmadd_vf_f32m1(z, f_pol_coef[0], __riscv_vfmv_v_f_f32m1(f_pol_coef[2], vl), vl);
  pol1 = __riscv_vfmadd_vv_f32m1(pol1, z, __riscv_vfmv_v_f_f32m1(f_pol_coef[3], vl), vl);
  pol2 = __riscv_vfmul_vv_f32m1(pol2, z, vl);

  x = __riscv_vfadd_vv_f32m1(tab_h, w, vl);
  t = __riscv_vfsub_vv_f32m1(x, tab_h, vl);
  w_l = __riscv_vfadd_vv_f32m1(__riscv_vfsub_vv_f32m1(w, t, vl), w_l, vl);

#ifndef __FAST_MATH__
    x = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vor_vx_u32m1_mu(mask_nan, __riscv_vreinterpret_v_f32m1_u32m1(x), __riscv_vreinterpret_v_f32m1_u32m1(x), magic_consts[1], vl));
#endif

    x = __riscv_vfadd_vv_f32m1(
		x, __riscv_vfadd_vv_f32m1(
			w_l, 
			__riscv_vfsub_vf_f32m1(
				__riscv_vfmacc_vv_f32m1(
					tab_l, 
					__riscv_vfmacc_vv_f32m1(pol2, pol1, y, vl), 
					w, 
					vl), 
				f_table_lo[0],
				vl),
			vl), 
		vl);
#ifndef __FAST_MATH__
     x = __riscv_vfmadd_vf_f32m1(x, 0x1p-30f, x, vl);
#endif
  return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vor_vv_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(sign), __riscv_vreinterpret_v_f32m1_u32m1(x), vl));
}

vfloat32m2_t __riscv_vatan_f32m2(vfloat32m2_t x, size_t vl) {
  vfloat32m2_t w, w_l, y, z, k, a, b, pol1, pol2, minus, bl, _b, t, tab_h, tab_l;
  vuint32m2_t index;
  vfloat32m2_t sign;
  vbool16_t mask1, mask2, mask3, mask4;
#ifndef __FAST_MATH__
    vbool16_t mask_nan, mask_big;
#endif

  minus = __riscv_vfmv_v_f_f32m2(f_borders[0], vl);

  sign = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vand_vv_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(minus), __riscv_vreinterpret_v_f32m2_u32m2(x), vl));
  x = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vand_vv_u32m2(__riscv_vnot_v_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(minus), vl), __riscv_vreinterpret_v_f32m2_u32m2(x), vl));
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmfne_vv_f32m2_b16(x, x, vl);
    x = __riscv_vreinterpret_v_i32m2_f32m2(__riscv_vxor_vv_i32m2_mu(mask_nan, __riscv_vreinterpret_v_f32m2_i32m2(x), __riscv_vreinterpret_v_f32m2_i32m2(x), __riscv_vreinterpret_v_f32m2_i32m2(x), vl));
#endif

  mask1 = __riscv_vmfge_vf_f32m2_b16(x, f_borders[1], vl);
  mask2 = __riscv_vmfge_vf_f32m2_b16(x, f_borders[2], vl);
  mask3 = __riscv_vmfge_vf_f32m2_b16(x, f_borders[3], vl);
  mask4 = __riscv_vmfge_vf_f32m2_b16(x, f_borders[4], vl);
#ifndef __FAST_MATH__
  mask_big = __riscv_vmflt_vf_f32m2_b16(x, 0x1p60f, vl);
#endif

  index = __riscv_vmv_v_x_u32m2(0, vl);
  index = __riscv_vadd_vx_u32m2_mu(mask1, index, index, 4, vl);
  index = __riscv_vadd_vx_u32m2_mu(mask2, index, index, 4, vl);
  index = __riscv_vadd_vx_u32m2_mu(mask3, index, index, 4, vl);
  index = __riscv_vadd_vx_u32m2_mu(mask4, index, index, 4, vl);

  k = __riscv_vloxei32_v_f32m2(f_k, index, vl);
  mask4 = __riscv_vmnot_m_b16(mask4, vl);

  a  = __riscv_vfsub_vv_f32m2_mu(mask4, k, x, k, vl);
  b  = __riscv_vfmadd_vv_f32m2_mu(mask4, x, k, __riscv_vfmv_v_f_f32m2(f_borders[6], vl), vl);

#ifndef __FAST_MATH__
  _b = __riscv_vfrdiv_vf_f32m2_mu(mask_big, minus, b, f_borders[6], vl);
  bl = __riscv_vfnmsub_vv_f32m2_mu(mask_big, _b, b, __riscv_vfmv_v_f_f32m2(f_borders[6], vl), vl);
#else
  _b = __riscv_vfrdiv_vf_f32m2(b, f_borders[6], vl);
  bl = __riscv_vfnmsub_vv_f32m2(_b, b, __riscv_vfmv_v_f_f32m2(f_borders[6], vl), vl);
#endif

  bl = __riscv_vfmul_vv_f32m2(_b, bl, vl);
  w = __riscv_vfmul_vv_f32m2(a, _b, vl);
  w_l = __riscv_vfmadd_vv_f32m2(a, bl, __riscv_vfmsub_vv_f32m2(a, _b, w, vl), vl);

  tab_h = __riscv_vloxei32_v_f32m2(f_table_hi, index, vl);
  tab_l = __riscv_vloxei32_v_f32m2(f_table_lo, index, vl);

  y = __riscv_vfmadd_vv_f32m2(w, w, __riscv_vfmv_v_f_f32m2(0x1p-40, vl), vl);
  z = __riscv_vfmul_vv_f32m2(y, y, vl);

  pol1 = __riscv_vfmadd_vf_f32m2(z, f_borders[7], __riscv_vfmv_v_f_f32m2(f_pol_coef[1], vl), vl);
  pol2 = __riscv_vfmadd_vf_f32m2(z, f_pol_coef[0], __riscv_vfmv_v_f_f32m2(f_pol_coef[2], vl), vl);
  pol1 = __riscv_vfmadd_vv_f32m2(pol1, z, __riscv_vfmv_v_f_f32m2(f_pol_coef[3], vl), vl);
  pol2 = __riscv_vfmul_vv_f32m2(pol2, z, vl);

  x = __riscv_vfadd_vv_f32m2(tab_h, w, vl);
  t = __riscv_vfsub_vv_f32m2(x, tab_h, vl);
  w_l = __riscv_vfadd_vv_f32m2(__riscv_vfsub_vv_f32m2(w, t, vl), w_l, vl);

#ifndef __FAST_MATH__
    x = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vor_vx_u32m2_mu(mask_nan, __riscv_vreinterpret_v_f32m2_u32m2(x), __riscv_vreinterpret_v_f32m2_u32m2(x), magic_consts[1], vl));
#endif

    x = __riscv_vfadd_vv_f32m2(
		x, __riscv_vfadd_vv_f32m2(
			w_l, 
			__riscv_vfsub_vf_f32m2(
				__riscv_vfmacc_vv_f32m2(
					tab_l, 
					__riscv_vfmacc_vv_f32m2(pol2, pol1, y, vl), 
					w, 
					vl), 
				f_table_lo[0],
				vl),
			vl), 
		vl);
#ifndef __FAST_MATH__
     x = __riscv_vfmadd_vf_f32m2(x, 0x1p-30f, x, vl);
#endif
  return __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vor_vv_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(sign), __riscv_vreinterpret_v_f32m2_u32m2(x), vl));
}

vfloat32m4_t __riscv_vatan_f32m4(vfloat32m4_t x, size_t vl) {
  vfloat32m4_t w, w_l, y, z, k, a, b, pol1, pol2, minus, bl, _b, t, tab_h, tab_l;
  vuint32m4_t index;
  vfloat32m4_t sign;
  vbool8_t mask1, mask2, mask3, mask4;
#ifndef __FAST_MATH__
    vbool8_t mask_nan, mask_big;
#endif

  minus = __riscv_vfmv_v_f_f32m4(f_borders[0], vl);

  sign = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vand_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(minus), __riscv_vreinterpret_v_f32m4_u32m4(x), vl));
  x = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vand_vv_u32m4(__riscv_vnot_v_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(minus), vl), __riscv_vreinterpret_v_f32m4_u32m4(x), vl));
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmfne_vv_f32m4_b8(x, x, vl);
    x = __riscv_vreinterpret_v_i32m4_f32m4(__riscv_vxor_vv_i32m4_mu(mask_nan, __riscv_vreinterpret_v_f32m4_i32m4(x), __riscv_vreinterpret_v_f32m4_i32m4(x), __riscv_vreinterpret_v_f32m4_i32m4(x), vl));
#endif

  mask1 = __riscv_vmfge_vf_f32m4_b8(x, f_borders[1], vl);
  mask2 = __riscv_vmfge_vf_f32m4_b8(x, f_borders[2], vl);
  mask3 = __riscv_vmfge_vf_f32m4_b8(x, f_borders[3], vl);
  mask4 = __riscv_vmfge_vf_f32m4_b8(x, f_borders[4], vl);
#ifndef __FAST_MATH__
  mask_big = __riscv_vmflt_vf_f32m4_b8(x, 0x1p60f, vl);
#endif

  index = __riscv_vmv_v_x_u32m4(0, vl);
  index = __riscv_vadd_vx_u32m4_mu(mask1, index, index, 4, vl);
  index = __riscv_vadd_vx_u32m4_mu(mask2, index, index, 4, vl);
  index = __riscv_vadd_vx_u32m4_mu(mask3, index, index, 4, vl);
  index = __riscv_vadd_vx_u32m4_mu(mask4, index, index, 4, vl);

  k = __riscv_vloxei32_v_f32m4(f_k, index, vl);
  mask4 = __riscv_vmnot_m_b8(mask4, vl);

  a  = __riscv_vfsub_vv_f32m4_mu(mask4, k, x, k, vl);
  b  = __riscv_vfmadd_vv_f32m4_mu(mask4, x, k, __riscv_vfmv_v_f_f32m4(f_borders[6], vl), vl);

#ifndef __FAST_MATH__
  _b = __riscv_vfrdiv_vf_f32m4_mu(mask_big, minus, b, f_borders[6], vl);
  bl = __riscv_vfnmsub_vv_f32m4_mu(mask_big, _b, b, __riscv_vfmv_v_f_f32m4(f_borders[6], vl), vl);
#else
  _b = __riscv_vfrdiv_vf_f32m4(b, f_borders[6], vl);
  bl = __riscv_vfnmsub_vv_f32m4(_b, b, __riscv_vfmv_v_f_f32m4(f_borders[6], vl), vl);
#endif

  bl = __riscv_vfmul_vv_f32m4(_b, bl, vl);
  w = __riscv_vfmul_vv_f32m4(a, _b, vl);
  w_l = __riscv_vfmadd_vv_f32m4(a, bl, __riscv_vfmsub_vv_f32m4(a, _b, w, vl), vl);

  tab_h = __riscv_vloxei32_v_f32m4(f_table_hi, index, vl);
  tab_l = __riscv_vloxei32_v_f32m4(f_table_lo, index, vl);

  y = __riscv_vfmadd_vv_f32m4(w, w, __riscv_vfmv_v_f_f32m4(0x1p-40, vl), vl);
  z = __riscv_vfmul_vv_f32m4(y, y, vl);

  pol1 = __riscv_vfmadd_vf_f32m4(z, f_borders[7], __riscv_vfmv_v_f_f32m4(f_pol_coef[1], vl), vl);
  pol2 = __riscv_vfmadd_vf_f32m4(z, f_pol_coef[0], __riscv_vfmv_v_f_f32m4(f_pol_coef[2], vl), vl);
  pol1 = __riscv_vfmadd_vv_f32m4(pol1, z, __riscv_vfmv_v_f_f32m4(f_pol_coef[3], vl), vl);
  pol2 = __riscv_vfmul_vv_f32m4(pol2, z, vl);

  x = __riscv_vfadd_vv_f32m4(tab_h, w, vl);
  t = __riscv_vfsub_vv_f32m4(x, tab_h, vl);
  w_l = __riscv_vfadd_vv_f32m4(__riscv_vfsub_vv_f32m4(w, t, vl), w_l, vl);

#ifndef __FAST_MATH__
    x = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vx_u32m4_mu(mask_nan, __riscv_vreinterpret_v_f32m4_u32m4(x), __riscv_vreinterpret_v_f32m4_u32m4(x), magic_consts[1], vl));
#endif

    x = __riscv_vfadd_vv_f32m4(
		x, __riscv_vfadd_vv_f32m4(
			w_l, 
			__riscv_vfsub_vf_f32m4(
				__riscv_vfmacc_vv_f32m4(
					tab_l, 
					__riscv_vfmacc_vv_f32m4(pol2, pol1, y, vl), 
					w, 
					vl), 
				f_table_lo[0],
				vl),
			vl), 
		vl);
#ifndef __FAST_MATH__
     x = __riscv_vfmadd_vf_f32m4(x, 0x1p-30f, x, vl);
#endif
  return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(sign), __riscv_vreinterpret_v_f32m4_u32m4(x), vl));
}

vfloat32m8_t __riscv_vatan_f32m8(vfloat32m8_t x, size_t vl) {
  vfloat32m8_t w, w_l, y, z, k, a, b, pol1, pol2, minus, bl, _b, t, tab_h, tab_l;
  vuint32m8_t index;
  vfloat32m8_t sign;
  vbool4_t mask1, mask2, mask3, mask4;
#ifndef __FAST_MATH__
    vbool4_t mask_nan, mask_big;
#endif

  minus = __riscv_vfmv_v_f_f32m8(f_borders[0], vl);

  sign = __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vand_vv_u32m8(__riscv_vreinterpret_v_f32m8_u32m8(minus), __riscv_vreinterpret_v_f32m8_u32m8(x), vl));
  x = __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vand_vv_u32m8(__riscv_vnot_v_u32m8(__riscv_vreinterpret_v_f32m8_u32m8(minus), vl), __riscv_vreinterpret_v_f32m8_u32m8(x), vl));
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmfne_vv_f32m8_b4(x, x, vl);
    x = __riscv_vreinterpret_v_i32m8_f32m8(__riscv_vxor_vv_i32m8_mu(mask_nan, __riscv_vreinterpret_v_f32m8_i32m8(x), __riscv_vreinterpret_v_f32m8_i32m8(x), __riscv_vreinterpret_v_f32m8_i32m8(x), vl));
#endif

  mask1 = __riscv_vmfge_vf_f32m8_b4(x, f_borders[1], vl);
  mask2 = __riscv_vmfge_vf_f32m8_b4(x, f_borders[2], vl);
  mask3 = __riscv_vmfge_vf_f32m8_b4(x, f_borders[3], vl);
  mask4 = __riscv_vmfge_vf_f32m8_b4(x, f_borders[4], vl);
#ifndef __FAST_MATH__
  mask_big = __riscv_vmflt_vf_f32m8_b4(x, 0x1p60f, vl);
#endif

  index = __riscv_vmv_v_x_u32m8(0, vl);
  index = __riscv_vadd_vx_u32m8_mu(mask1, index, index, 4, vl);
  index = __riscv_vadd_vx_u32m8_mu(mask2, index, index, 4, vl);
  index = __riscv_vadd_vx_u32m8_mu(mask3, index, index, 4, vl);
  index = __riscv_vadd_vx_u32m8_mu(mask4, index, index, 4, vl);

  k = __riscv_vloxei32_v_f32m8(f_k, index, vl);
  mask4 = __riscv_vmnot_m_b4(mask4, vl);

  a  = __riscv_vfsub_vv_f32m8_mu(mask4, k, x, k, vl);
  b  = __riscv_vfmadd_vv_f32m8_mu(mask4, x, k, __riscv_vfmv_v_f_f32m8(f_borders[6], vl), vl);

#ifndef __FAST_MATH__
  _b = __riscv_vfrdiv_vf_f32m8_mu(mask_big, minus, b, f_borders[6], vl);
  bl = __riscv_vfnmsub_vv_f32m8_mu(mask_big, _b, b, __riscv_vfmv_v_f_f32m8(f_borders[6], vl), vl);
#else
  _b = __riscv_vfrdiv_vf_f32m8(b, f_borders[6], vl);
  bl = __riscv_vfnmsub_vv_f32m8(_b, b, __riscv_vfmv_v_f_f32m8(f_borders[6], vl), vl);
#endif

  bl = __riscv_vfmul_vv_f32m8(_b, bl, vl);
  w = __riscv_vfmul_vv_f32m8(a, _b, vl);
  w_l = __riscv_vfmadd_vv_f32m8(a, bl, __riscv_vfmsub_vv_f32m8(a, _b, w, vl), vl);

  tab_h = __riscv_vloxei32_v_f32m8(f_table_hi, index, vl);
  tab_l = __riscv_vloxei32_v_f32m8(f_table_lo, index, vl);

  y = __riscv_vfmadd_vv_f32m8(w, w, __riscv_vfmv_v_f_f32m8(0x1p-40, vl), vl);
  z = __riscv_vfmul_vv_f32m8(y, y, vl);

  pol1 = __riscv_vfmadd_vf_f32m8(z, f_borders[7], __riscv_vfmv_v_f_f32m8(f_pol_coef[1], vl), vl);
  pol2 = __riscv_vfmadd_vf_f32m8(z, f_pol_coef[0], __riscv_vfmv_v_f_f32m8(f_pol_coef[2], vl), vl);
  pol1 = __riscv_vfmadd_vv_f32m8(pol1, z, __riscv_vfmv_v_f_f32m8(f_pol_coef[3], vl), vl);
  pol2 = __riscv_vfmul_vv_f32m8(pol2, z, vl);

  x = __riscv_vfadd_vv_f32m8(tab_h, w, vl);
  t = __riscv_vfsub_vv_f32m8(x, tab_h, vl);
  w_l = __riscv_vfadd_vv_f32m8(__riscv_vfsub_vv_f32m8(w, t, vl), w_l, vl);

#ifndef __FAST_MATH__
    x = __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vor_vx_u32m8_mu(mask_nan, __riscv_vreinterpret_v_f32m8_u32m8(x), __riscv_vreinterpret_v_f32m8_u32m8(x), magic_consts[1], vl));
#endif

    x = __riscv_vfadd_vv_f32m8(
		x, __riscv_vfadd_vv_f32m8(
			w_l, 
			__riscv_vfsub_vf_f32m8(
				__riscv_vfmacc_vv_f32m8(
					tab_l, 
					__riscv_vfmacc_vv_f32m8(pol2, pol1, y, vl), 
					w, 
					vl), 
				f_table_lo[0],
				vl),
			vl), 
		vl);
#ifndef __FAST_MATH__
     x = __riscv_vfmadd_vf_f32m8(x, 0x1p-30f, x, vl);
#endif
  return __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vor_vv_u32m8(__riscv_vreinterpret_v_f32m8_u32m8(sign), __riscv_vreinterpret_v_f32m8_u32m8(x), vl));
}

#ifdef __riscv_zvfh

static const _Float16 h_borders1[4] = {-0x0p0f16, 0x1p-3f16,  0x1.8p-2f16, 0x1.5p-1f16};
static const _Float16 h_borders2[4] = {0x1p0f16,  0x1.9p0f16, 0x1.6p1f16,  0x1.04p3f16};

static const int16_t h_magic_consts[2] = {static_cast<int16_t>(0x7c00), static_cast<int16_t>(0x7e00)};

static const _Float16   h_table_hi[16] = {0.0f16,  0x1.f5cp-3f16, 0x1.dacp-2f16, 0x1.5d4p-1f16,  0x1.cacp-1f16,   0x1.1b8p0f16,    0x1.538p0f16,   0x1.92p0f16},
	                h_table_lo[16] = {0.0f16, -0x1.14p8f16, 0x1.9cp8f16, 0x1.898p11f16, 0x1.f18p9f16, -0x1.1e8p12f16, -0x1.738p12f16, 0x1.fb4p12f16},
			h_k[8] = {0.0f16,          0x1p-2f16,     0x1p-1f16,     0x1.ap-1f16,    0x1.4p0f16,      2.0f16,          4.0f16,         -1.0f16};


vfloat16m1_t __riscv_vatan_f16m1(vfloat16m1_t x, size_t vl) {
    vfloat16m1_t w, wc, w12, w_l, y, z, k, a, b, pol1, pol2, minus, bl, _b, ch, uh, ul, cl, t;
    vuint16m1_t index;
    vfloat16m1_t sign;
    vbool16_t mask1, mask2, mask3, mask4, mask5, mask6, mask7;
#ifndef __FAST_MATH__
    vbool16_t mask_nan, mask_inf, mask_uf, mask_big;
#endif

    minus = __riscv_vfmv_v_f_f16m1(h_borders1[0], vl);

    sign = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vv_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(minus), __riscv_vreinterpret_v_f16m1_u16m1(x), vl));
    x = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vv_u16m1(__riscv_vnot_v_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(minus), vl), __riscv_vreinterpret_v_f16m1_u16m1(x), vl));

#ifndef __FAST_MATH__
    mask_nan = __riscv_vmfne_vv_f16m1_b16(x, x, vl);
    x = __riscv_vreinterpret_v_i16m1_f16m1(__riscv_vxor_vv_i16m1_mu(mask_nan, __riscv_vreinterpret_v_f16m1_i16m1(x), __riscv_vreinterpret_v_f16m1_i16m1(x), __riscv_vreinterpret_v_f16m1_i16m1(x), vl));
#endif

    mask1 = __riscv_vmfge_vf_f16m1_b16(x, h_borders1[1], vl);
    mask2 = __riscv_vmfge_vf_f16m1_b16(x, h_borders1[2], vl);
    mask3 = __riscv_vmfge_vf_f16m1_b16(x, h_borders1[3], vl);
    mask4 = __riscv_vmfge_vf_f16m1_b16(x, h_borders2[0], vl);
    mask5 = __riscv_vmfge_vf_f16m1_b16(x, h_borders2[1], vl);
    mask6 = __riscv_vmfge_vf_f16m1_b16(x, h_borders2[2], vl);
    mask7 = __riscv_vmfge_vf_f16m1_b16(x, h_borders2[3], vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmflt_vf_f16m1_b16(x, 0x1p13f16, vl);
#endif

    index = __riscv_vmv_v_x_u16m1(0, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask1, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask2, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask3, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask4, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask5, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask6, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask7, index, index, 2, vl);

    k = __riscv_vloxei16_v_f16m1(h_k, index, vl);
    mask7 = __riscv_vmnot_m_b16(mask7, vl);

    a  = __riscv_vfsub_vv_f16m1_mu(mask7, k, x, k, vl);
    b  = __riscv_vfmadd_vv_f16m1_mu(mask7, x, k, __riscv_vfmv_v_f_f16m1(h_borders2[0], vl), vl);

    bl = __riscv_vfsub_vf_f16m1_mu(mask7, minus, b, h_borders2[0], vl);
    bl = __riscv_vfmsac_vv_f16m1_mu(mask7, bl, x, k, vl);

#ifndef __FAST_MATH__
    _b = __riscv_vfrdiv_vf_f16m1_mu(mask_big, minus, b, 0x1p12f16, vl);
#else
    _b = __riscv_vfrdiv_vf_f16m1(b, 0x1p12f16, vl);
#endif
    ch = __riscv_vfmul_vv_f16m1(a, _b, vl);
#ifndef __FAST_MATH__
    mask_uf = __riscv_vmflt_vf_f16m1_b16(__riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vv_u16m1(__riscv_vnot_v_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(minus), vl), __riscv_vreinterpret_v_f16m1_u16m1(ch), vl)), 0x1p6f16, vl);
    uh = __riscv_vfmul_vv_f16m1_mu(mask_big, ch, ch, b, vl);
    ul = __riscv_vfmsub_vv_f16m1_mu(mask_big, ch, b, uh, vl);
#else
    uh = __riscv_vfmul_vv_f16m1(ch, b, vl);
    ul = __riscv_vfmsub_vv_f16m1(ch, b, uh, vl);
#endif

    cl = __riscv_vfmul_vv_f16m1(
		__riscv_vfnmsub_vf_f16m1(
			__riscv_vfmul_vv_f16m1(ch, __riscv_vfmul_vf_f16m1(bl, 0x1p12f16, vl), vl), 
			0x1p-12f16, 
			__riscv_vfsub_vv_f16m1(__riscv_vfmsub_vf_f16m1(a, 0x1p12f16, uh, vl), ul, vl), 
			vl), 
		_b, vl);

    w12 = __riscv_vfmadd_vf_f16m1(cl, 0x1p-12f16, ch, vl);

    w = __riscv_vfmul_vf_f16m1(w12, 0x1p-12f16, vl);
#ifndef __FAST_MATH__
    wc = __riscv_vreinterpret_v_i16m1_f16m1(__riscv_vxor_vv_i16m1_mu(mask_uf, __riscv_vreinterpret_v_f16m1_i16m1(w), __riscv_vreinterpret_v_f16m1_i16m1(w), __riscv_vreinterpret_v_f16m1_i16m1(w), vl));
#else
    wc = w;
#endif
    a = __riscv_vloxei16_v_f16m1(h_table_hi, index, vl);
    b = __riscv_vloxei16_v_f16m1(h_table_lo, index, vl);

    x = __riscv_vfadd_vv_f16m1(a, w, vl);
    y = __riscv_vfmul_vv_f16m1(wc, wc, vl);

    w_l = __riscv_vfmadd_vf_f16m1(__riscv_vfsub_vv_f16m1(ch, w12, vl), 0x1p12f16, cl, vl);

    t = __riscv_vfsub_vv_f16m1(x, a, vl);

    pol1 = __riscv_vfmul_vv_f16m1(y, __riscv_vfmv_v_f_f16m1(0x1.894p9f16, vl), vl);
    pol2 = __riscv_vfmul_vv_f16m1(y, __riscv_vfmv_v_f_f16m1(-0x1.554p10f16, vl), vl);

    w_l = __riscv_vfmadd_vf_f16m1(__riscv_vfnmsub_vf_f16m1(t, 0x1p12f16, w12, vl), 0x1p12f16, w_l, vl);

#ifndef __FAST_MATH__
    x = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vor_vx_u16m1_mu(mask_nan, __riscv_vreinterpret_v_f16m1_u16m1(x), __riscv_vreinterpret_v_f16m1_u16m1(x), h_magic_consts[1], vl));
#endif

    x = __riscv_vfmadd_vf_f16m1(__riscv_vfadd_vv_f16m1(w_l, __riscv_vfmadd_vv_f16m1(__riscv_vfmadd_vv_f16m1(pol1, y, pol2, vl), w12, b, vl), vl), 0x1p-24f16, x, vl);
#ifndef __FAST_MATH__
     x = __riscv_vfmadd_vf_f16m1(x, 0x1p-14f16, x, vl);
#endif
    return __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vor_vv_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(sign), __riscv_vreinterpret_v_f16m1_u16m1(x), vl));
}

vfloat16m2_t __riscv_vatan_f16m2(vfloat16m2_t x, size_t vl) {
    vfloat16m2_t w, wc, w12, w_l, y, z, k, a, b, pol1, pol2, minus, bl, _b, ch, uh, ul, cl, t;
    vuint16m2_t index;
    vfloat16m2_t sign;
    vbool8_t mask1, mask2, mask3, mask4, mask5, mask6, mask7;
#ifndef __FAST_MATH__
    vbool8_t mask_nan, mask_inf, mask_uf, mask_big;
#endif

    minus = __riscv_vfmv_v_f_f16m2(h_borders1[0], vl);

    sign = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vv_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(minus), __riscv_vreinterpret_v_f16m2_u16m2(x), vl));
    x = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vv_u16m2(__riscv_vnot_v_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(minus), vl), __riscv_vreinterpret_v_f16m2_u16m2(x), vl));

#ifndef __FAST_MATH__
    mask_nan = __riscv_vmfne_vv_f16m2_b8(x, x, vl);
    x = __riscv_vreinterpret_v_i16m2_f16m2(__riscv_vxor_vv_i16m2_mu(mask_nan, __riscv_vreinterpret_v_f16m2_i16m2(x), __riscv_vreinterpret_v_f16m2_i16m2(x), __riscv_vreinterpret_v_f16m2_i16m2(x), vl));
#endif

    mask1 = __riscv_vmfge_vf_f16m2_b8(x, h_borders1[1], vl);
    mask2 = __riscv_vmfge_vf_f16m2_b8(x, h_borders1[2], vl);
    mask3 = __riscv_vmfge_vf_f16m2_b8(x, h_borders1[3], vl);
    mask4 = __riscv_vmfge_vf_f16m2_b8(x, h_borders2[0], vl);
    mask5 = __riscv_vmfge_vf_f16m2_b8(x, h_borders2[1], vl);
    mask6 = __riscv_vmfge_vf_f16m2_b8(x, h_borders2[2], vl);
    mask7 = __riscv_vmfge_vf_f16m2_b8(x, h_borders2[3], vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmflt_vf_f16m2_b8(x, 0x1p13f16, vl);
#endif

    index = __riscv_vmv_v_x_u16m2(0, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask1, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask2, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask3, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask4, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask5, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask6, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask7, index, index, 2, vl);

    k = __riscv_vloxei16_v_f16m2(h_k, index, vl);
    mask7 = __riscv_vmnot_m_b8(mask7, vl);

    a  = __riscv_vfsub_vv_f16m2_mu(mask7, k, x, k, vl);
    b  = __riscv_vfmadd_vv_f16m2_mu(mask7, x, k, __riscv_vfmv_v_f_f16m2(h_borders2[0], vl), vl);

    bl = __riscv_vfsub_vf_f16m2_mu(mask7, minus, b, h_borders2[0], vl);
    bl = __riscv_vfmsac_vv_f16m2_mu(mask7, bl, x, k, vl);

#ifndef __FAST_MATH__
    _b = __riscv_vfrdiv_vf_f16m2_mu(mask_big, minus, b, 0x1p12f16, vl);
#else
    _b = __riscv_vfrdiv_vf_f16m2(b, 0x1p12f16, vl);
#endif
    ch = __riscv_vfmul_vv_f16m2(a, _b, vl);
#ifndef __FAST_MATH__
    mask_uf = __riscv_vmflt_vf_f16m2_b8(__riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vv_u16m2(__riscv_vnot_v_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(minus), vl), __riscv_vreinterpret_v_f16m2_u16m2(ch), vl)), 0x1p6f16, vl);
    uh = __riscv_vfmul_vv_f16m2_mu(mask_big, ch, ch, b, vl);
    ul = __riscv_vfmsub_vv_f16m2_mu(mask_big, ch, b, uh, vl);
#else
    uh = __riscv_vfmul_vv_f16m2(ch, b, vl);
    ul = __riscv_vfmsub_vv_f16m2(ch, b, uh, vl);
#endif

    cl = __riscv_vfmul_vv_f16m2(
		__riscv_vfnmsub_vf_f16m2(
			__riscv_vfmul_vv_f16m2(ch, __riscv_vfmul_vf_f16m2(bl, 0x1p12f16, vl), vl), 
			0x1p-12f16, 
			__riscv_vfsub_vv_f16m2(__riscv_vfmsub_vf_f16m2(a, 0x1p12f16, uh, vl), ul, vl), 
			vl), 
		_b, vl);

    w12 = __riscv_vfmadd_vf_f16m2(cl, 0x1p-12f16, ch, vl);

    w = __riscv_vfmul_vf_f16m2(w12, 0x1p-12f16, vl);
#ifndef __FAST_MATH__
    wc = __riscv_vreinterpret_v_i16m2_f16m2(__riscv_vxor_vv_i16m2_mu(mask_uf, __riscv_vreinterpret_v_f16m2_i16m2(w), __riscv_vreinterpret_v_f16m2_i16m2(w), __riscv_vreinterpret_v_f16m2_i16m2(w), vl));
#else
    wc = w;
#endif
    a = __riscv_vloxei16_v_f16m2(h_table_hi, index, vl);
    b = __riscv_vloxei16_v_f16m2(h_table_lo, index, vl);

    x = __riscv_vfadd_vv_f16m2(a, w, vl);
    y = __riscv_vfmul_vv_f16m2(wc, wc, vl);

    w_l = __riscv_vfmadd_vf_f16m2(__riscv_vfsub_vv_f16m2(ch, w12, vl), 0x1p12f16, cl, vl);

    t = __riscv_vfsub_vv_f16m2(x, a, vl);

    pol1 = __riscv_vfmul_vv_f16m2(y, __riscv_vfmv_v_f_f16m2(0x1.894p9f16, vl), vl);
    pol2 = __riscv_vfmul_vv_f16m2(y, __riscv_vfmv_v_f_f16m2(-0x1.554p10f16, vl), vl);

    w_l = __riscv_vfmadd_vf_f16m2(__riscv_vfnmsub_vf_f16m2(t, 0x1p12f16, w12, vl), 0x1p12f16, w_l, vl);

#ifndef __FAST_MATH__
    x = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vor_vx_u16m2_mu(mask_nan, __riscv_vreinterpret_v_f16m2_u16m2(x), __riscv_vreinterpret_v_f16m2_u16m2(x), h_magic_consts[1], vl));
#endif

    x = __riscv_vfmadd_vf_f16m2(__riscv_vfadd_vv_f16m2(w_l, __riscv_vfmadd_vv_f16m2(__riscv_vfmadd_vv_f16m2(pol1, y, pol2, vl), w12, b, vl), vl), 0x1p-24f16, x, vl);
#ifndef __FAST_MATH__
     x = __riscv_vfmadd_vf_f16m2(x, 0x1p-14f16, x, vl);
#endif
    return __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vor_vv_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(sign), __riscv_vreinterpret_v_f16m2_u16m2(x), vl));
}

vfloat16m4_t __riscv_vatan_f16m4(vfloat16m4_t x, size_t vl) {
    vfloat16m4_t w, wc, w12, w_l, y, z, k, a, b, pol1, pol2, minus, bl, _b, ch, uh, ul, cl, t;
    vuint16m4_t index;
    vfloat16m4_t sign;
    vbool4_t mask1, mask2, mask3, mask4, mask5, mask6, mask7;
#ifndef __FAST_MATH__
    vbool4_t mask_nan, mask_inf, mask_uf, mask_big;
#endif

    minus = __riscv_vfmv_v_f_f16m4(h_borders1[0], vl);

    sign = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vv_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(minus), __riscv_vreinterpret_v_f16m4_u16m4(x), vl));
    x = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vv_u16m4(__riscv_vnot_v_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(minus), vl), __riscv_vreinterpret_v_f16m4_u16m4(x), vl));

#ifndef __FAST_MATH__
    mask_nan = __riscv_vmfne_vv_f16m4_b4(x, x, vl);
    x = __riscv_vreinterpret_v_i16m4_f16m4(__riscv_vxor_vv_i16m4_mu(mask_nan, __riscv_vreinterpret_v_f16m4_i16m4(x), __riscv_vreinterpret_v_f16m4_i16m4(x), __riscv_vreinterpret_v_f16m4_i16m4(x), vl));
#endif

    mask1 = __riscv_vmfge_vf_f16m4_b4(x, h_borders1[1], vl);
    mask2 = __riscv_vmfge_vf_f16m4_b4(x, h_borders1[2], vl);
    mask3 = __riscv_vmfge_vf_f16m4_b4(x, h_borders1[3], vl);
    mask4 = __riscv_vmfge_vf_f16m4_b4(x, h_borders2[0], vl);
    mask5 = __riscv_vmfge_vf_f16m4_b4(x, h_borders2[1], vl);
    mask6 = __riscv_vmfge_vf_f16m4_b4(x, h_borders2[2], vl);
    mask7 = __riscv_vmfge_vf_f16m4_b4(x, h_borders2[3], vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmflt_vf_f16m4_b4(x, 0x1p13f16, vl);
#endif

    index = __riscv_vmv_v_x_u16m4(0, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask1, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask2, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask3, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask4, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask5, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask6, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask7, index, index, 2, vl);

    k = __riscv_vloxei16_v_f16m4(h_k, index, vl);
    mask7 = __riscv_vmnot_m_b4(mask7, vl);

    a  = __riscv_vfsub_vv_f16m4_mu(mask7, k, x, k, vl);
    b  = __riscv_vfmadd_vv_f16m4_mu(mask7, x, k, __riscv_vfmv_v_f_f16m4(h_borders2[0], vl), vl);

    bl = __riscv_vfsub_vf_f16m4_mu(mask7, minus, b, h_borders2[0], vl);
    bl = __riscv_vfmsac_vv_f16m4_mu(mask7, bl, x, k, vl);

#ifndef __FAST_MATH__
    _b = __riscv_vfrdiv_vf_f16m4_mu(mask_big, minus, b, 0x1p12f16, vl);
#else
    _b = __riscv_vfrdiv_vf_f16m4(b, 0x1p12f16, vl);
#endif
    ch = __riscv_vfmul_vv_f16m4(a, _b, vl);
#ifndef __FAST_MATH__
    mask_uf = __riscv_vmflt_vf_f16m4_b4(__riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vv_u16m4(__riscv_vnot_v_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(minus), vl), __riscv_vreinterpret_v_f16m4_u16m4(ch), vl)), 0x1p6f16, vl);
    uh = __riscv_vfmul_vv_f16m4_mu(mask_big, ch, ch, b, vl);
    ul = __riscv_vfmsub_vv_f16m4_mu(mask_big, ch, b, uh, vl);
#else
    uh = __riscv_vfmul_vv_f16m4(ch, b, vl);
    ul = __riscv_vfmsub_vv_f16m4(ch, b, uh, vl);
#endif

    cl = __riscv_vfmul_vv_f16m4(
		__riscv_vfnmsub_vf_f16m4(
			__riscv_vfmul_vv_f16m4(ch, __riscv_vfmul_vf_f16m4(bl, 0x1p12f16, vl), vl), 
			0x1p-12f16, 
			__riscv_vfsub_vv_f16m4(__riscv_vfmsub_vf_f16m4(a, 0x1p12f16, uh, vl), ul, vl), 
			vl), 
		_b, vl);

    w12 = __riscv_vfmadd_vf_f16m4(cl, 0x1p-12f16, ch, vl);

    w = __riscv_vfmul_vf_f16m4(w12, 0x1p-12f16, vl);
#ifndef __FAST_MATH__
    wc = __riscv_vreinterpret_v_i16m4_f16m4(__riscv_vxor_vv_i16m4_mu(mask_uf, __riscv_vreinterpret_v_f16m4_i16m4(w), __riscv_vreinterpret_v_f16m4_i16m4(w), __riscv_vreinterpret_v_f16m4_i16m4(w), vl));
#else
    wc = w;
#endif
    a = __riscv_vloxei16_v_f16m4(h_table_hi, index, vl);
    b = __riscv_vloxei16_v_f16m4(h_table_lo, index, vl);

    x = __riscv_vfadd_vv_f16m4(a, w, vl);
    y = __riscv_vfmul_vv_f16m4(wc, wc, vl);

    w_l = __riscv_vfmadd_vf_f16m4(__riscv_vfsub_vv_f16m4(ch, w12, vl), 0x1p12f16, cl, vl);

    t = __riscv_vfsub_vv_f16m4(x, a, vl);

    pol1 = __riscv_vfmul_vv_f16m4(y, __riscv_vfmv_v_f_f16m4(0x1.894p9f16, vl), vl);
    pol2 = __riscv_vfmul_vv_f16m4(y, __riscv_vfmv_v_f_f16m4(-0x1.554p10f16, vl), vl);

    w_l = __riscv_vfmadd_vf_f16m4(__riscv_vfnmsub_vf_f16m4(t, 0x1p12f16, w12, vl), 0x1p12f16, w_l, vl);

#ifndef __FAST_MATH__
    x = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vor_vx_u16m4_mu(mask_nan, __riscv_vreinterpret_v_f16m4_u16m4(x), __riscv_vreinterpret_v_f16m4_u16m4(x), h_magic_consts[1], vl));
#endif

    x = __riscv_vfmadd_vf_f16m4(__riscv_vfadd_vv_f16m4(w_l, __riscv_vfmadd_vv_f16m4(__riscv_vfmadd_vv_f16m4(pol1, y, pol2, vl), w12, b, vl), vl), 0x1p-24f16, x, vl);
#ifndef __FAST_MATH__
     x = __riscv_vfmadd_vf_f16m4(x, 0x1p-14f16, x, vl);
#endif
    return __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vor_vv_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(sign), __riscv_vreinterpret_v_f16m4_u16m4(x), vl));
}

vfloat16m8_t __riscv_vatan_f16m8(vfloat16m8_t x, size_t vl) {
    vfloat16m8_t w, wc, w12, w_l, y, z, k, a, b, pol1, pol2, minus, bl, _b, ch, uh, ul, cl, t;
    vuint16m8_t index;
    vfloat16m8_t sign;
    vbool2_t mask1, mask2, mask3, mask4, mask5, mask6, mask7;
#ifndef __FAST_MATH__
    vbool2_t mask_nan, mask_inf, mask_uf, mask_big;
#endif

    minus = __riscv_vfmv_v_f_f16m8(h_borders1[0], vl);

    sign = __riscv_vreinterpret_v_u16m8_f16m8(__riscv_vand_vv_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(minus), __riscv_vreinterpret_v_f16m8_u16m8(x), vl));
    x = __riscv_vreinterpret_v_u16m8_f16m8(__riscv_vand_vv_u16m8(__riscv_vnot_v_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(minus), vl), __riscv_vreinterpret_v_f16m8_u16m8(x), vl));

#ifndef __FAST_MATH__
    mask_nan = __riscv_vmfne_vv_f16m8_b2(x, x, vl);
    x = __riscv_vreinterpret_v_i16m8_f16m8(__riscv_vxor_vv_i16m8_mu(mask_nan, __riscv_vreinterpret_v_f16m8_i16m8(x), __riscv_vreinterpret_v_f16m8_i16m8(x), __riscv_vreinterpret_v_f16m8_i16m8(x), vl));
#endif

    mask1 = __riscv_vmfge_vf_f16m8_b2(x, h_borders1[1], vl);
    mask2 = __riscv_vmfge_vf_f16m8_b2(x, h_borders1[2], vl);
    mask3 = __riscv_vmfge_vf_f16m8_b2(x, h_borders1[3], vl);
    mask4 = __riscv_vmfge_vf_f16m8_b2(x, h_borders2[0], vl);
    mask5 = __riscv_vmfge_vf_f16m8_b2(x, h_borders2[1], vl);
    mask6 = __riscv_vmfge_vf_f16m8_b2(x, h_borders2[2], vl);
    mask7 = __riscv_vmfge_vf_f16m8_b2(x, h_borders2[3], vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmflt_vf_f16m8_b2(x, 0x1p13f16, vl);
#endif

    index = __riscv_vmv_v_x_u16m8(0, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask1, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask2, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask3, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask4, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask5, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask6, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask7, index, index, 2, vl);

    k = __riscv_vloxei16_v_f16m8(h_k, index, vl);
    mask7 = __riscv_vmnot_m_b2(mask7, vl);

    a  = __riscv_vfsub_vv_f16m8_mu(mask7, k, x, k, vl);
    b  = __riscv_vfmadd_vv_f16m8_mu(mask7, x, k, __riscv_vfmv_v_f_f16m8(h_borders2[0], vl), vl);

    bl = __riscv_vfsub_vf_f16m8_mu(mask7, minus, b, h_borders2[0], vl);
    bl = __riscv_vfmsac_vv_f16m8_mu(mask7, bl, x, k, vl);

#ifndef __FAST_MATH__
    _b = __riscv_vfrdiv_vf_f16m8_mu(mask_big, minus, b, 0x1p12f16, vl);
#else
    _b = __riscv_vfrdiv_vf_f16m8(b, 0x1p12f16, vl);
#endif
    ch = __riscv_vfmul_vv_f16m8(a, _b, vl);
#ifndef __FAST_MATH__
    mask_uf = __riscv_vmflt_vf_f16m8_b2(__riscv_vreinterpret_v_u16m8_f16m8(__riscv_vand_vv_u16m8(__riscv_vnot_v_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(minus), vl), __riscv_vreinterpret_v_f16m8_u16m8(ch), vl)), 0x1p6f16, vl);
    uh = __riscv_vfmul_vv_f16m8_mu(mask_big, ch, ch, b, vl);
    ul = __riscv_vfmsub_vv_f16m8_mu(mask_big, ch, b, uh, vl);
#else
    uh = __riscv_vfmul_vv_f16m8(ch, b, vl);
    ul = __riscv_vfmsub_vv_f16m8(ch, b, uh, vl);
#endif

    cl = __riscv_vfmul_vv_f16m8(
		__riscv_vfnmsub_vf_f16m8(
			__riscv_vfmul_vv_f16m8(ch, __riscv_vfmul_vf_f16m8(bl, 0x1p12f16, vl), vl), 
			0x1p-12f16, 
			__riscv_vfsub_vv_f16m8(__riscv_vfmsub_vf_f16m8(a, 0x1p12f16, uh, vl), ul, vl), 
			vl), 
		_b, vl);

    w12 = __riscv_vfmadd_vf_f16m8(cl, 0x1p-12f16, ch, vl);

    w = __riscv_vfmul_vf_f16m8(w12, 0x1p-12f16, vl);
#ifndef __FAST_MATH__
    wc = __riscv_vreinterpret_v_i16m8_f16m8(__riscv_vxor_vv_i16m8_mu(mask_uf, __riscv_vreinterpret_v_f16m8_i16m8(w), __riscv_vreinterpret_v_f16m8_i16m8(w), __riscv_vreinterpret_v_f16m8_i16m8(w), vl));
#else
    wc = w;
#endif
    a = __riscv_vloxei16_v_f16m8(h_table_hi, index, vl);
    b = __riscv_vloxei16_v_f16m8(h_table_lo, index, vl);

    x = __riscv_vfadd_vv_f16m8(a, w, vl);
    y = __riscv_vfmul_vv_f16m8(wc, wc, vl);

    w_l = __riscv_vfmadd_vf_f16m8(__riscv_vfsub_vv_f16m8(ch, w12, vl), 0x1p12f16, cl, vl);

    t = __riscv_vfsub_vv_f16m8(x, a, vl);

    pol1 = __riscv_vfmul_vv_f16m8(y, __riscv_vfmv_v_f_f16m8(0x1.894p9f16, vl), vl);
    pol2 = __riscv_vfmul_vv_f16m8(y, __riscv_vfmv_v_f_f16m8(-0x1.554p10f16, vl), vl);

    w_l = __riscv_vfmadd_vf_f16m8(__riscv_vfnmsub_vf_f16m8(t, 0x1p12f16, w12, vl), 0x1p12f16, w_l, vl);

#ifndef __FAST_MATH__
    x = __riscv_vreinterpret_v_u16m8_f16m8(__riscv_vor_vx_u16m8_mu(mask_nan, __riscv_vreinterpret_v_f16m8_u16m8(x), __riscv_vreinterpret_v_f16m8_u16m8(x), h_magic_consts[1], vl));
#endif

    x = __riscv_vfmadd_vf_f16m8(__riscv_vfadd_vv_f16m8(w_l, __riscv_vfmadd_vv_f16m8(__riscv_vfmadd_vv_f16m8(pol1, y, pol2, vl), w12, b, vl), vl), 0x1p-24f16, x, vl);
#ifndef __FAST_MATH__
     x = __riscv_vfmadd_vf_f16m8(x, 0x1p-14f16, x, vl);
#endif
    return __riscv_vreinterpret_v_u16m8_f16m8(__riscv_vor_vv_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(sign), __riscv_vreinterpret_v_f16m8_u16m8(x), vl));
}

#endif