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

static const double borders1[4] = {-0x0p0, 0x1p-3,  0x1.8p-2, 0x1.5p-1};
static const double borders2[4] = { 0x1p0, 0x1.47ae147ae147bp-1, 0x1.745d1745d1746p-2, 0x1.f81f81f81f82p-4};
static const double magic_consts1[4] = { 0.0, 0x0.8p-1022, 0x1p0, 0x1p0};

static const int64_t magic_consts2[4] = { 0x7ff0000000000000, 64, 0x7fe, 0x7ff8000000000000};

static const double pol_coef1[4] = {-0x1.02e52393cf313p-4, 0x1.3abbe13593d39p-4, -0x1.745bfe2487e06p-4, 0x1.c71c6fde46a39p-4},
	            pol_coef2[4] = {-0x1.24924923b4715p-3, 0x1.99999999993f2p-3, -0x1.5555555555555p-2, 0.0};

static const double d_table_hi[16] = {0,                     0x1.f5b75f92c80ddp-3,   0x1.dac670561bb4fp-2,  0x1.5d58987169b18p-1,
			              0x1.cac7c57846f9ep-1,  0x1.1b6e192ebbe44p0,    0x1.5368c951e9cfdp0,   0x1.921fb54442d18p0,
			              0x1.921fb54442d18p1,   0x1.72c43f4b1650ap1,    0x1.56c6e7397f5aep1,   0x1.3ac98f27e8652p1,
			              0x1.1f6dc3e631131p1,   0x1.0468a8ace4df6p1,    0x1.d0d6a1369bd34p0,   0x1.921fb54442d18p0},
                    d_table_lo[16] = {-0x1p-1022,            0x1.8ab6e3cf7afbdp-57,  0x1.a2b7f222f65e2p-56, 0x1.0028e4bc5e7cap-57, 
                                      0x1.0dae13ad18a6bp-55, 0x1.b1b466a88828ep-54, -0x1.96f47948a99f1p-54, 0x1.1a62633145c07p-54,
			              0x1.1a62633145c07p-53, 0x1.c1b6f4f44e10bp-53,  0x1.660b64ece6f4bp-53, 0x1.0a5fd4e57fd8ap-53,
		                     -0x1.290921ba00694p-53, 0x1.0620bf7406affp-55, -0x1.a23602a65700cp-57, 0x1.1a62633145c07p-54},
                    d_k[8]        = { 0,                     0x1p-2,                 0x1p-1,                0x1.ap-1,
	                              0x1.4p0,               2.0,                    4.0,                   1.0};

static const float f_borders1[4] = {-0x0p0, 0x1p-3,  0x1.8p-2, 0x1.5p-1};
static const float f_borders2[4] = { 0x1p0, 0x1.47ae14p-1, 0x1.745d18p-2, 0x1.f81f82p-4};

static const float f_magic_consts1[4] = { static_cast<float>(0x1p128), 0x0.8p-126, 0x1p23, 0x1p-23};

static const int32_t f_magic_consts2[4] = { 1, 32, static_cast<int32_t>(0xfe), 0x7fc00000};

static const float   f_pol_coef[4] = {-0x1.55555p-2, 0x1.9989c6p-3, -0x1.1dccf8p-3, 0.0};
 
static const float   f_table_hi[16] = {0,     	         0x1.f5b76p-3,    0x1.dac67p-2,    0x1.5d5898p-1,   0x1.cac7c6p-1,   0x1.1b6e1ap0,    0x1.5368cap0,    0x1.921fb6p0,
				       0x1.921fb6p1,     0x1.72c44p1,     0x1.56c6e8p1,    0x1.3ac99p1,     0x1.1f6dc4p1,    0x1.0468a8p1,    0x1.d0d6a2p0,    0x1.921fb6p0},
		     f_table_lo[16] = {-0x1p-126,       -0x1.b4dfc8p-6,   0x1.586ed4p-5,   0x1.c5a6c6p-4,  -0x1.0f720cp-3,  -0x1.a28838p-2,  -0x1.5c2c6p-2,   -0x1.777a5cp-2,
		         	       -0x1.777a5cp-1,  -0x1.69d35ep-1,  -0x1.8d014ap-1,  -0x1.b02f36p-1,  -0x1.9ceedp-4,    0x1.59c9bep-1,  -0x1.92c85ap-2,  -0x1.777a5cp-2},
		     f_k[8]         = {0,  		0x1p-2,          0x1p-1,          0x1.ap-1,        0x1.4p0,         2.0,             4.0,             1.0};

inline vfloat64m1_t mask_null_f64m1(vbool64_t mask, vfloat64m1_t arg, size_t vl) {
  return __riscv_vreinterpret_v_i64m1_f64m1(__riscv_vxor_vv_i64m1_mu(mask, __riscv_vreinterpret_v_f64m1_i64m1(arg), __riscv_vreinterpret_v_f64m1_i64m1(arg), __riscv_vreinterpret_v_f64m1_i64m1(arg), vl));
}
inline vfloat64m2_t mask_null_f64m2(vbool32_t mask, vfloat64m2_t arg, size_t vl) {
  return __riscv_vreinterpret_v_i64m2_f64m2(__riscv_vxor_vv_i64m2_mu(mask, __riscv_vreinterpret_v_f64m2_i64m2(arg), __riscv_vreinterpret_v_f64m2_i64m2(arg), __riscv_vreinterpret_v_f64m2_i64m2(arg), vl));
}
inline vfloat64m4_t mask_null_f64m4(vbool16_t mask, vfloat64m4_t arg, size_t vl) {
  return __riscv_vreinterpret_v_i64m4_f64m4(__riscv_vxor_vv_i64m4_mu(mask, __riscv_vreinterpret_v_f64m4_i64m4(arg), __riscv_vreinterpret_v_f64m4_i64m4(arg), __riscv_vreinterpret_v_f64m4_i64m4(arg), vl));
}
inline vfloat64m8_t mask_null_f64m8(vbool8_t mask, vfloat64m8_t arg, size_t vl) {
  return __riscv_vreinterpret_v_i64m8_f64m8(__riscv_vxor_vv_i64m8_mu(mask, __riscv_vreinterpret_v_f64m8_i64m8(arg), __riscv_vreinterpret_v_f64m8_i64m8(arg), __riscv_vreinterpret_v_f64m8_i64m8(arg), vl));
}

inline vfloat32m1_t mask_null_f32m1(vbool32_t mask, vfloat32m1_t arg, size_t vl) {
  return __riscv_vreinterpret_v_i32m1_f32m1(__riscv_vxor_vv_i32m1_mu(mask, __riscv_vreinterpret_v_f32m1_i32m1(arg), __riscv_vreinterpret_v_f32m1_i32m1(arg), __riscv_vreinterpret_v_f32m1_i32m1(arg), vl));
}
inline vfloat32m2_t mask_null_f32m2(vbool16_t mask, vfloat32m2_t arg, size_t vl) {
  return __riscv_vreinterpret_v_i32m2_f32m2(__riscv_vxor_vv_i32m2_mu(mask, __riscv_vreinterpret_v_f32m2_i32m2(arg), __riscv_vreinterpret_v_f32m2_i32m2(arg), __riscv_vreinterpret_v_f32m2_i32m2(arg), vl));
}
inline vfloat32m4_t mask_null_f32m4(vbool8_t mask, vfloat32m4_t arg, size_t vl) {
  return __riscv_vreinterpret_v_i32m4_f32m4(__riscv_vxor_vv_i32m4_mu(mask, __riscv_vreinterpret_v_f32m4_i32m4(arg), __riscv_vreinterpret_v_f32m4_i32m4(arg), __riscv_vreinterpret_v_f32m4_i32m4(arg), vl));
}
inline vfloat32m8_t mask_null_f32m8(vbool4_t mask, vfloat32m8_t arg, size_t vl) {
  return __riscv_vreinterpret_v_i32m8_f32m8(__riscv_vxor_vv_i32m8_mu(mask, __riscv_vreinterpret_v_f32m8_i32m8(arg), __riscv_vreinterpret_v_f32m8_i32m8(arg), __riscv_vreinterpret_v_f32m8_i32m8(arg), vl));
}

vfloat64m1_t __riscv_vatan2_f64m1(vfloat64m1_t y, vfloat64m1_t x, size_t vl) {
    vfloat64m1_t w, wc, w_l, z, k1, k2, k, a, b, al, bl, A, Al, B, Bl, pol1, pol2, minus, t, c1, c2, c, blend_x, blend_y, ch, cl, rh, rl, r, multiplicator, correction, xc, yc;
    vuint64m1_t index;
    vfloat64m1_t sign_y;
    vbool64_t masky0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, sign_x;
#ifndef __FAST_MATH__
    vbool64_t mask_inf_x, mask_inf_y, mask_nan;
    vbool64_t mask_small, mask_big, magic_mask, masky_uf, maskx_uf;
    vint64m1_t p_diff;
#endif
    minus = __riscv_vfmv_v_f_f64m1(borders1[0], vl);

    sign_x = __riscv_vmsgeu_vv_u64m1_b64(__riscv_vreinterpret_v_f64m1_u64m1(x), __riscv_vreinterpret_v_f64m1_u64m1(minus), vl);
    masky0 = __riscv_vmfeq_vv_f64m1_b64(minus, y, vl);
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmor_mm_b64(__riscv_vmfne_vv_f64m1_b64(x, x, vl), __riscv_vmfne_vv_f64m1_b64(y, y, vl), vl);
#endif
    sign_y = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vv_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(minus), __riscv_vreinterpret_v_f64m1_u64m1(y), vl));
    x = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vv_u64m1(__riscv_vnot_v_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(minus), vl), __riscv_vreinterpret_v_f64m1_u64m1(x), vl));
    y = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vv_u64m1(__riscv_vnot_v_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(minus), vl), __riscv_vreinterpret_v_f64m1_u64m1(y), vl));
#ifndef __FAST_MATH__
    mask_inf_y = __riscv_vmfeq_vf_f64m1_b64(y, *reinterpret_cast<const double*>(&magic_consts2[0]), vl);
    mask_inf_x = __riscv_vmfeq_vf_f64m1_b64(x, *reinterpret_cast<const double*>(&magic_consts2[0]), vl);
    y = __riscv_vfmerge_vfm_f64m1(y, borders2[0], mask_inf_y, vl);
    x = mask_null_f64m1(mask_inf_y, x, vl);

    y = mask_null_f64m1(__riscv_vmor_mm_b64(__riscv_vmandn_mm_b64(mask_inf_x, mask_inf_y, vl), mask_nan, vl), y, vl);
    x = __riscv_vfmerge_vfm_f64m1(x, borders2[0],  __riscv_vmor_mm_b64(__riscv_vmor_mm_b64(mask_inf_x, masky0, vl), mask_nan, vl), vl);
#else
    x = __riscv_vfmerge_vfm_f64m1(x, borders2[0], masky0, vl);
#endif

    multiplicator = __riscv_vreinterpret_v_i64m1_f64m1(
			__riscv_vsll_vx_i64m1(
				__riscv_vsub_vv_i64m1(
					__riscv_vmv_v_x_i64m1(magic_consts2[2], vl), 
					__riscv_vreinterpret_v_u64m1_i64m1(__riscv_vsrl_vx_u64m1(
						__riscv_vadd_vv_u64m1(
								      __riscv_vand_vx_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(y), magic_consts2[0], vl), 
								      __riscv_vreinterpret_v_f64m1_u64m1(x), vl), 
						53, vl)), 
					vl),
				52, vl)
			);
#ifndef __FAST_MATH__
    magic_mask = __riscv_vmandn_mm_b64(__riscv_vmandn_mm_b64(sign_x, mask_nan, vl), mask_inf_x, vl);
#endif
    multiplicator = __riscv_vfmerge_vfm_f64m1(multiplicator, magic_consts1[1], __riscv_vmfle_vv_f64m1_b64(multiplicator, minus, vl), vl);
#ifndef __FAST_MATH__
    p_diff = __riscv_vsra_vx_i64m1(__riscv_vsub_vv_i64m1(__riscv_vreinterpret_v_f64m1_i64m1(y), __riscv_vreinterpret_v_f64m1_i64m1(x), vl), 52, vl);
#endif

    x = __riscv_vfmul_vv_f64m1(x, multiplicator, vl);
    y = __riscv_vfmul_vv_f64m1(y, multiplicator, vl);

#ifndef __FAST_MATH__
    maskx_uf = __riscv_vmflt_vf_f64m1_b64(x, 0x1p-1010, vl);
    masky_uf = __riscv_vmflt_vf_f64m1_b64(y, 0x1p-1010, vl);

    xc = mask_null_f64m1(maskx_uf, x, vl);
    magic_mask = __riscv_vmandn_mm_b64(magic_mask, masky0, vl);
    yc = mask_null_f64m1(masky_uf, y, vl);
    mask_small = __riscv_vmsgt_vx_i64m1_b64(p_diff, -0x3b, vl);
    magic_mask =  __riscv_vmandn_mm_b64(magic_mask, mask_small, vl);
#else
    xc = x;
    yc = y;
#endif

    mask1 = __riscv_vmflt_vv_f64m1_b64(__riscv_vfmul_vf_f64m1(xc, borders1[1], vl), y, vl);
    mask2 = __riscv_vmflt_vv_f64m1_b64(__riscv_vfmul_vf_f64m1(xc, borders1[2], vl), y, vl);
    mask3 = __riscv_vmflt_vv_f64m1_b64(__riscv_vfmul_vf_f64m1(xc, borders1[3], vl), y, vl);
    mask4 = __riscv_vmflt_vv_f64m1_b64(x, y, vl);
    mask5 = __riscv_vmfle_vv_f64m1_b64(x, __riscv_vfmul_vf_f64m1(yc, borders2[1], vl), vl);
    mask6 = __riscv_vmfle_vv_f64m1_b64(x, __riscv_vfmul_vf_f64m1(yc, borders2[2], vl), vl);
    mask7 = __riscv_vmfle_vv_f64m1_b64(x, __riscv_vfmul_vf_f64m1(yc, borders2[3], vl), vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmsge_vx_i64m1_b64(p_diff, 0x1f4, vl);
#endif

    index = __riscv_vmv_v_x_u64m1(0, vl);
#ifndef __FAST_MATH__
    correction = __riscv_vfmerge_vfm_f64m1(__riscv_vfmv_v_f_f64m1(0x1p60, vl), 0x1p-500, mask_big, vl);
#endif
    index = __riscv_vadd_vx_u64m1_mu(mask1, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask2, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask3, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask4, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask5, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask6, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m1_mu(mask7, index, index, 8, vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmul_vv_f64m1_mu(__riscv_vmor_mm_b64(magic_mask, mask_big, vl), x, y, correction, vl);
#endif

    A = mask_null_f64m1(mask7, y, vl);
    k = __riscv_vloxei64_v_f64m1(d_k, index, vl);

    c = __riscv_vfmul_vv_f64m1(k, x, vl);
    al = __riscv_vfmsub_vv_f64m1(k, x, c, vl);
    a = __riscv_vfsub_vv_f64m1(A, c, vl);
    mask7 = __riscv_vmnot_m_b64(mask7, vl);
    Al = __riscv_vfsub_vv_f64m1(a, A, vl);
    al = __riscv_vfsub_vv_f64m1(__riscv_vfsub_vv_f64m1(__riscv_vfsub_vv_f64m1(A, __riscv_vfsub_vv_f64m1(a, Al, vl), vl), __riscv_vfadd_vv_f64m1(Al, c, vl), vl), al, vl);

    b = __riscv_vfmadd_vv_f64m1_mu(mask7, y, k, x, vl);
    Bl = __riscv_vfsub_vv_f64m1_mu(mask7, minus, b, x, vl);
    bl = __riscv_vfadd_vv_f64m1_mu(mask7, minus, __riscv_vfsub_vv_f64m1_mu(mask7, minus, x, __riscv_vfsub_vv_f64m1(b, Bl, vl), vl), __riscv_vfmsac_vv_f64m1(Bl, k, y, vl), vl);

    ch =  __riscv_vfdiv_vv_f64m1(a, b, vl);
    rh = __riscv_vfnmsub_vv_f64m1(ch, b, a, vl);
    rl = __riscv_vfnmsub_vv_f64m1(ch, bl, al, vl);
#ifndef __FAST_MATH__
    r = __riscv_vfadd_vv_f64m1_mu(mask_small, minus, rh, rl, vl);
#else
    r = __riscv_vfadd_vv_f64m1(rh, rl, vl);
#endif
    cl = __riscv_vfdiv_vv_f64m1(r, b, vl);

    w  =  __riscv_vfadd_vv_f64m1(cl, ch, vl);
#ifndef __FAST_MATH__
    wc =  __riscv_vfadd_vf_f64m1(w, 0x1p-86, vl);
#else
    wc = w;
#endif
    w_l = __riscv_vfadd_vv_f64m1(__riscv_vfsub_vv_f64m1(ch, w, vl), cl, vl);

#ifndef __FAST_MATH__
    index = __riscv_vadd_vx_u64m1_mu(__riscv_vmandn_mm_b64(sign_x, mask_nan, vl), index, index, magic_consts2[1], vl);
#else
    index = __riscv_vadd_vx_u64m1_mu(sign_x, index, index, magic_consts2[1], vl);
#endif

    a = __riscv_vloxei64_v_f64m1(d_table_hi, index, vl);
    b = __riscv_vloxei64_v_f64m1(d_table_lo, index, vl);

    y = __riscv_vfmul_vv_f64m1(wc, wc, vl);

    z = __riscv_vfmul_vv_f64m1(y, y, vl);

    pol1 = __riscv_vfmadd_vf_f64m1(z, pol_coef1[0], __riscv_vfmv_v_f_f64m1(pol_coef1[2], vl), vl);
    pol2 = __riscv_vfmadd_vf_f64m1(z, pol_coef1[1], __riscv_vfmv_v_f_f64m1(pol_coef1[3], vl), vl);
    pol1 = __riscv_vfmadd_vv_f64m1(pol1, z, __riscv_vfmv_v_f_f64m1(pol_coef2[0], vl), vl);
    pol2 = __riscv_vfmadd_vv_f64m1(pol2, z, __riscv_vfmv_v_f_f64m1(pol_coef2[1], vl), vl);
    pol1 = __riscv_vfmadd_vv_f64m1(pol1, z, __riscv_vfmv_v_f_f64m1(pol_coef2[2], vl), vl);
    pol2 = __riscv_vfmul_vv_f64m1(pol2,  z, vl);

    w  =  __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vxor_vv_u64m1_mu(sign_x, __riscv_vreinterpret_v_f64m1_u64m1(w),   __riscv_vreinterpret_v_f64m1_u64m1(minus), __riscv_vreinterpret_v_f64m1_u64m1(w),   vl));
    w_l = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vxor_vv_u64m1_mu(sign_x, __riscv_vreinterpret_v_f64m1_u64m1(w_l), __riscv_vreinterpret_v_f64m1_u64m1(minus), __riscv_vreinterpret_v_f64m1_u64m1(w_l), vl));

    x = __riscv_vfadd_vv_f64m1(a, w, vl);
    t = __riscv_vfsub_vv_f64m1(x, a, vl);

    w_l = __riscv_vfadd_vv_f64m1(__riscv_vfsub_vv_f64m1(w, t, vl), w_l, vl);

    x = __riscv_vfadd_vv_f64m1(
			       __riscv_vfadd_vv_f64m1(__riscv_vfsub_vf_f64m1(__riscv_vfmadd_vv_f64m1(__riscv_vfmadd_vv_f64m1(pol1, y, pol2, vl), w, b, vl), 
									      d_table_lo[0], 
									      vl),
						       w_l,
						       vl), 
			       x, 
			       vl);
#ifndef __FAST_MATH__
    x = __riscv_vfmacc_vf_f64m1_mu(mask_inf_x, x, 0x1p-1022, __riscv_vfmv_v_f_f64m1(0x1p-1022, vl), vl);
    x = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vor_vx_u64m1_mu(mask_nan, __riscv_vreinterpret_v_f64m1_u64m1(x), __riscv_vreinterpret_v_f64m1_u64m1(x), magic_consts2[3], vl));
    x = __riscv_vfmadd_vf_f64m1(x, 0x1p-60, x, vl);
#endif
    return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vor_vv_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(sign_y), __riscv_vreinterpret_v_f64m1_u64m1(x), vl));
}



vfloat64m2_t __riscv_vatan2_f64m2(vfloat64m2_t y, vfloat64m2_t x, size_t vl) {
    vfloat64m2_t w, wc, w_l, z, k1, k2, k, a, b, al, bl, A, Al, B, Bl, pol1, pol2, minus, t, c1, c2, c, blend_x, blend_y, ch, cl, rh, rl, r, multiplicator, correction, xc, yc;
    vuint64m2_t index;
    vfloat64m2_t sign_y;
    vbool32_t masky0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, sign_x;
#ifndef __FAST_MATH__
    vbool32_t mask_inf_x, mask_inf_y, mask_nan;
    vbool32_t mask_small, mask_big, magic_mask, masky_uf, maskx_uf;
    vint64m2_t p_diff;
#endif
    minus = __riscv_vfmv_v_f_f64m2(borders1[0], vl);

    sign_x = __riscv_vmsgeu_vv_u64m2_b32(__riscv_vreinterpret_v_f64m2_u64m2(x), __riscv_vreinterpret_v_f64m2_u64m2(minus), vl);
    masky0 = __riscv_vmfeq_vv_f64m2_b32(minus, y, vl);
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmor_mm_b32(__riscv_vmfne_vv_f64m2_b32(x, x, vl), __riscv_vmfne_vv_f64m2_b32(y, y, vl), vl);
#endif
    sign_y = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vv_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(minus), __riscv_vreinterpret_v_f64m2_u64m2(y), vl));
    x = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vv_u64m2(__riscv_vnot_v_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(minus), vl), __riscv_vreinterpret_v_f64m2_u64m2(x), vl));
    y = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vv_u64m2(__riscv_vnot_v_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(minus), vl), __riscv_vreinterpret_v_f64m2_u64m2(y), vl));
#ifndef __FAST_MATH__
    mask_inf_y = __riscv_vmfeq_vf_f64m2_b32(y, *reinterpret_cast<const double*>(&magic_consts2[0]), vl);
    mask_inf_x = __riscv_vmfeq_vf_f64m2_b32(x, *reinterpret_cast<const double*>(&magic_consts2[0]), vl);
    y = __riscv_vfmerge_vfm_f64m2(y, borders2[0], mask_inf_y, vl);
    x = mask_null_f64m2(mask_inf_y, x, vl);

    y = mask_null_f64m2(__riscv_vmor_mm_b32(__riscv_vmandn_mm_b32(mask_inf_x, mask_inf_y, vl), mask_nan, vl), y, vl);
    x = __riscv_vfmerge_vfm_f64m2(x, borders2[0],  __riscv_vmor_mm_b32(__riscv_vmor_mm_b32(mask_inf_x, masky0, vl), mask_nan, vl), vl);
#else
    x = __riscv_vfmerge_vfm_f64m2(x, borders2[0], masky0, vl);
#endif

    multiplicator = __riscv_vreinterpret_v_i64m2_f64m2(
			__riscv_vsll_vx_i64m2(
				__riscv_vsub_vv_i64m2(
					__riscv_vmv_v_x_i64m2(magic_consts2[2], vl), 
					__riscv_vreinterpret_v_u64m2_i64m2(__riscv_vsrl_vx_u64m2(
						__riscv_vadd_vv_u64m2(
								      __riscv_vand_vx_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(y), magic_consts2[0], vl), 
								      __riscv_vreinterpret_v_f64m2_u64m2(x), vl), 
						53, vl)), 
					vl),
				52, vl)
			);
#ifndef __FAST_MATH__
    magic_mask = __riscv_vmandn_mm_b32(__riscv_vmandn_mm_b32(sign_x, mask_nan, vl), mask_inf_x, vl);
#endif
    multiplicator = __riscv_vfmerge_vfm_f64m2(multiplicator, magic_consts1[1], __riscv_vmfle_vv_f64m2_b32(multiplicator, minus, vl), vl);
#ifndef __FAST_MATH__
    p_diff = __riscv_vsra_vx_i64m2(__riscv_vsub_vv_i64m2(__riscv_vreinterpret_v_f64m2_i64m2(y), __riscv_vreinterpret_v_f64m2_i64m2(x), vl), 52, vl);
#endif

    x = __riscv_vfmul_vv_f64m2(x, multiplicator, vl);
    y = __riscv_vfmul_vv_f64m2(y, multiplicator, vl);

#ifndef __FAST_MATH__
    maskx_uf = __riscv_vmflt_vf_f64m2_b32(x, 0x1p-1010, vl);
    masky_uf = __riscv_vmflt_vf_f64m2_b32(y, 0x1p-1010, vl);

    xc = mask_null_f64m2(maskx_uf, x, vl);
    magic_mask = __riscv_vmandn_mm_b32(magic_mask, masky0, vl);
    yc = mask_null_f64m2(masky_uf, y, vl);
    mask_small = __riscv_vmsgt_vx_i64m2_b32(p_diff, -0x3b, vl);
    magic_mask =  __riscv_vmandn_mm_b32(magic_mask, mask_small, vl);
#else
    xc = x;
    yc = y;
#endif

    mask1 = __riscv_vmflt_vv_f64m2_b32(__riscv_vfmul_vf_f64m2(xc, borders1[1], vl), y, vl);
    mask2 = __riscv_vmflt_vv_f64m2_b32(__riscv_vfmul_vf_f64m2(xc, borders1[2], vl), y, vl);
    mask3 = __riscv_vmflt_vv_f64m2_b32(__riscv_vfmul_vf_f64m2(xc, borders1[3], vl), y, vl);
    mask4 = __riscv_vmflt_vv_f64m2_b32(x, y, vl);
    mask5 = __riscv_vmfle_vv_f64m2_b32(x, __riscv_vfmul_vf_f64m2(yc, borders2[1], vl), vl);
    mask6 = __riscv_vmfle_vv_f64m2_b32(x, __riscv_vfmul_vf_f64m2(yc, borders2[2], vl), vl);
    mask7 = __riscv_vmfle_vv_f64m2_b32(x, __riscv_vfmul_vf_f64m2(yc, borders2[3], vl), vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmsge_vx_i64m2_b32(p_diff, 0x1f4, vl);
#endif

    index = __riscv_vmv_v_x_u64m2(0, vl);
#ifndef __FAST_MATH__
    correction = __riscv_vfmerge_vfm_f64m2(__riscv_vfmv_v_f_f64m2(0x1p60, vl), 0x1p-500, mask_big, vl);
#endif
    index = __riscv_vadd_vx_u64m2_mu(mask1, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask2, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask3, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask4, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask5, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask6, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m2_mu(mask7, index, index, 8, vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmul_vv_f64m2_mu(__riscv_vmor_mm_b32(magic_mask, mask_big, vl), x, y, correction, vl);
#endif

    A = mask_null_f64m2(mask7, y, vl);
    k = __riscv_vloxei64_v_f64m2(d_k, index, vl);

    c = __riscv_vfmul_vv_f64m2(k, x, vl);
    al = __riscv_vfmsub_vv_f64m2(k, x, c, vl);
    a = __riscv_vfsub_vv_f64m2(A, c, vl);
    mask7 = __riscv_vmnot_m_b32(mask7, vl);
    Al = __riscv_vfsub_vv_f64m2(a, A, vl);
    al = __riscv_vfsub_vv_f64m2(__riscv_vfsub_vv_f64m2(__riscv_vfsub_vv_f64m2(A, __riscv_vfsub_vv_f64m2(a, Al, vl), vl), __riscv_vfadd_vv_f64m2(Al, c, vl), vl), al, vl);

    b = __riscv_vfmadd_vv_f64m2_mu(mask7, y, k, x, vl);
    Bl = __riscv_vfsub_vv_f64m2_mu(mask7, minus, b, x, vl);
    bl = __riscv_vfadd_vv_f64m2_mu(mask7, minus, __riscv_vfsub_vv_f64m2_mu(mask7, minus, x, __riscv_vfsub_vv_f64m2(b, Bl, vl), vl), __riscv_vfmsac_vv_f64m2(Bl, k, y, vl), vl);

    ch =  __riscv_vfdiv_vv_f64m2(a, b, vl);
    rh = __riscv_vfnmsub_vv_f64m2(ch, b, a, vl);
    rl = __riscv_vfnmsub_vv_f64m2(ch, bl, al, vl);
#ifndef __FAST_MATH__
    r = __riscv_vfadd_vv_f64m2_mu(mask_small, minus, rh, rl, vl);
#else
    r = __riscv_vfadd_vv_f64m2(rh, rl, vl);
#endif
    cl = __riscv_vfdiv_vv_f64m2(r, b, vl);

    w  =  __riscv_vfadd_vv_f64m2(cl, ch, vl);
#ifndef __FAST_MATH__
    wc =  __riscv_vfadd_vf_f64m2(w, 0x1p-86, vl);
#else
    wc = w;
#endif
    w_l = __riscv_vfadd_vv_f64m2(__riscv_vfsub_vv_f64m2(ch, w, vl), cl, vl);

#ifndef __FAST_MATH__
    index = __riscv_vadd_vx_u64m2_mu(__riscv_vmandn_mm_b32(sign_x, mask_nan, vl), index, index, magic_consts2[1], vl);
#else
    index = __riscv_vadd_vx_u64m2_mu(sign_x, index, index, magic_consts2[1], vl);
#endif

    a = __riscv_vloxei64_v_f64m2(d_table_hi, index, vl);
    b = __riscv_vloxei64_v_f64m2(d_table_lo, index, vl);

    y = __riscv_vfmul_vv_f64m2(wc, wc, vl);

    z = __riscv_vfmul_vv_f64m2(y, y, vl);

    pol1 = __riscv_vfmadd_vf_f64m2(z, pol_coef1[0], __riscv_vfmv_v_f_f64m2(pol_coef1[2], vl), vl);
    pol2 = __riscv_vfmadd_vf_f64m2(z, pol_coef1[1], __riscv_vfmv_v_f_f64m2(pol_coef1[3], vl), vl);
    pol1 = __riscv_vfmadd_vv_f64m2(pol1, z, __riscv_vfmv_v_f_f64m2(pol_coef2[0], vl), vl);
    pol2 = __riscv_vfmadd_vv_f64m2(pol2, z, __riscv_vfmv_v_f_f64m2(pol_coef2[1], vl), vl);
    pol1 = __riscv_vfmadd_vv_f64m2(pol1, z, __riscv_vfmv_v_f_f64m2(pol_coef2[2], vl), vl);
    pol2 = __riscv_vfmul_vv_f64m2(pol2,  z, vl);

    w  =  __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vxor_vv_u64m2_mu(sign_x, __riscv_vreinterpret_v_f64m2_u64m2(w),   __riscv_vreinterpret_v_f64m2_u64m2(minus), __riscv_vreinterpret_v_f64m2_u64m2(w),   vl));
    w_l = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vxor_vv_u64m2_mu(sign_x, __riscv_vreinterpret_v_f64m2_u64m2(w_l), __riscv_vreinterpret_v_f64m2_u64m2(minus), __riscv_vreinterpret_v_f64m2_u64m2(w_l), vl));

    x = __riscv_vfadd_vv_f64m2(a, w, vl);
    t = __riscv_vfsub_vv_f64m2(x, a, vl);

    w_l = __riscv_vfadd_vv_f64m2(__riscv_vfsub_vv_f64m2(w, t, vl), w_l, vl);

    x = __riscv_vfadd_vv_f64m2(
			       __riscv_vfadd_vv_f64m2(__riscv_vfsub_vf_f64m2(__riscv_vfmadd_vv_f64m2(__riscv_vfmadd_vv_f64m2(pol1, y, pol2, vl), w, b, vl), 
									      d_table_lo[0], 
									      vl),
						       w_l,
						       vl), 
			       x, 
			       vl);
#ifndef __FAST_MATH__
    x = __riscv_vfmacc_vf_f64m2_mu(mask_inf_x, x, 0x1p-1022, __riscv_vfmv_v_f_f64m2(0x1p-1022, vl), vl);
    x = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vor_vx_u64m2_mu(mask_nan, __riscv_vreinterpret_v_f64m2_u64m2(x), __riscv_vreinterpret_v_f64m2_u64m2(x), magic_consts2[3], vl));
    x = __riscv_vfmadd_vf_f64m2(x, 0x1p-60, x, vl);
#endif
    return __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vor_vv_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(sign_y), __riscv_vreinterpret_v_f64m2_u64m2(x), vl));
}





vfloat64m4_t __riscv_vatan2_f64m4(vfloat64m4_t y, vfloat64m4_t x, size_t vl) {
    vfloat64m4_t w, wc, w_l, z, k1, k2, k, a, b, al, bl, A, Al, B, Bl, pol1, pol2, minus, t, c1, c2, c, blend_x, blend_y, ch, cl, rh, rl, r, multiplicator, correction, xc, yc;
    vuint64m4_t index;
    vfloat64m4_t sign_y;
    vbool16_t masky0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, sign_x;
#ifndef __FAST_MATH__
    vbool16_t mask_inf_x, mask_inf_y, mask_nan;
    vbool16_t mask_small, mask_big, magic_mask, masky_uf, maskx_uf;
    vint64m4_t p_diff;
#endif
    minus = __riscv_vfmv_v_f_f64m4(borders1[0], vl);

    sign_x = __riscv_vmsgeu_vv_u64m4_b16(__riscv_vreinterpret_v_f64m4_u64m4(x), __riscv_vreinterpret_v_f64m4_u64m4(minus), vl);
    masky0 = __riscv_vmfeq_vv_f64m4_b16(minus, y, vl);
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmor_mm_b16(__riscv_vmfne_vv_f64m4_b16(x, x, vl), __riscv_vmfne_vv_f64m4_b16(y, y, vl), vl);
#endif
    sign_y = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vand_vv_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(minus), __riscv_vreinterpret_v_f64m4_u64m4(y), vl));
    x = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vand_vv_u64m4(__riscv_vnot_v_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(minus), vl), __riscv_vreinterpret_v_f64m4_u64m4(x), vl));
    y = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vand_vv_u64m4(__riscv_vnot_v_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(minus), vl), __riscv_vreinterpret_v_f64m4_u64m4(y), vl));
#ifndef __FAST_MATH__
    mask_inf_y = __riscv_vmfeq_vf_f64m4_b16(y, *reinterpret_cast<const double*>(&magic_consts2[0]), vl);
    mask_inf_x = __riscv_vmfeq_vf_f64m4_b16(x, *reinterpret_cast<const double*>(&magic_consts2[0]), vl);
    y = __riscv_vfmerge_vfm_f64m4(y, borders2[0], mask_inf_y, vl);
    x = mask_null_f64m4(mask_inf_y, x, vl);

    y = mask_null_f64m4(__riscv_vmor_mm_b16(__riscv_vmandn_mm_b16(mask_inf_x, mask_inf_y, vl), mask_nan, vl), y, vl);
    x = __riscv_vfmerge_vfm_f64m4(x, borders2[0],  __riscv_vmor_mm_b16(__riscv_vmor_mm_b16(mask_inf_x, masky0, vl), mask_nan, vl), vl);
#else
    x = __riscv_vfmerge_vfm_f64m4(x, borders2[0], masky0, vl);
#endif

    multiplicator = __riscv_vreinterpret_v_i64m4_f64m4(
			__riscv_vsll_vx_i64m4(
				__riscv_vsub_vv_i64m4(
					__riscv_vmv_v_x_i64m4(magic_consts2[2], vl), 
					__riscv_vreinterpret_v_u64m4_i64m4(__riscv_vsrl_vx_u64m4(
						__riscv_vadd_vv_u64m4(
								      __riscv_vand_vx_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(y), magic_consts2[0], vl), 
								      __riscv_vreinterpret_v_f64m4_u64m4(x), vl), 
						53, vl)), 
					vl),
				52, vl)
			);
#ifndef __FAST_MATH__
    magic_mask = __riscv_vmandn_mm_b16(__riscv_vmandn_mm_b16(sign_x, mask_nan, vl), mask_inf_x, vl);
#endif
    multiplicator = __riscv_vfmerge_vfm_f64m4(multiplicator, magic_consts1[1], __riscv_vmfle_vv_f64m4_b16(multiplicator, minus, vl), vl);
#ifndef __FAST_MATH__
    p_diff = __riscv_vsra_vx_i64m4(__riscv_vsub_vv_i64m4(__riscv_vreinterpret_v_f64m4_i64m4(y), __riscv_vreinterpret_v_f64m4_i64m4(x), vl), 52, vl);
#endif

    x = __riscv_vfmul_vv_f64m4(x, multiplicator, vl);
    y = __riscv_vfmul_vv_f64m4(y, multiplicator, vl);

#ifndef __FAST_MATH__
    maskx_uf = __riscv_vmflt_vf_f64m4_b16(x, 0x1p-1010, vl);
    masky_uf = __riscv_vmflt_vf_f64m4_b16(y, 0x1p-1010, vl);

    xc = mask_null_f64m4(maskx_uf, x, vl);
    magic_mask = __riscv_vmandn_mm_b16(magic_mask, masky0, vl);
    yc = mask_null_f64m4(masky_uf, y, vl);
    mask_small = __riscv_vmsgt_vx_i64m4_b16(p_diff, -0x3b, vl);
    magic_mask =  __riscv_vmandn_mm_b16(magic_mask, mask_small, vl);
#else
    xc = x;
    yc = y;
#endif

    mask1 = __riscv_vmflt_vv_f64m4_b16(__riscv_vfmul_vf_f64m4(xc, borders1[1], vl), y, vl);
    mask2 = __riscv_vmflt_vv_f64m4_b16(__riscv_vfmul_vf_f64m4(xc, borders1[2], vl), y, vl);
    mask3 = __riscv_vmflt_vv_f64m4_b16(__riscv_vfmul_vf_f64m4(xc, borders1[3], vl), y, vl);
    mask4 = __riscv_vmflt_vv_f64m4_b16(x, y, vl);
    mask5 = __riscv_vmfle_vv_f64m4_b16(x, __riscv_vfmul_vf_f64m4(yc, borders2[1], vl), vl);
    mask6 = __riscv_vmfle_vv_f64m4_b16(x, __riscv_vfmul_vf_f64m4(yc, borders2[2], vl), vl);
    mask7 = __riscv_vmfle_vv_f64m4_b16(x, __riscv_vfmul_vf_f64m4(yc, borders2[3], vl), vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmsge_vx_i64m4_b16(p_diff, 0x1f4, vl);
#endif

    index = __riscv_vmv_v_x_u64m4(0, vl);
#ifndef __FAST_MATH__
    correction = __riscv_vfmerge_vfm_f64m4(__riscv_vfmv_v_f_f64m4(0x1p60, vl), 0x1p-500, mask_big, vl);
#endif
    index = __riscv_vadd_vx_u64m4_mu(mask1, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask2, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask3, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask4, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask5, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask6, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m4_mu(mask7, index, index, 8, vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmul_vv_f64m4_mu(__riscv_vmor_mm_b16(magic_mask, mask_big, vl), x, y, correction, vl);
#endif

    A = mask_null_f64m4(mask7, y, vl);
    k = __riscv_vloxei64_v_f64m4(d_k, index, vl);

    c = __riscv_vfmul_vv_f64m4(k, x, vl);
    al = __riscv_vfmsub_vv_f64m4(k, x, c, vl);
    a = __riscv_vfsub_vv_f64m4(A, c, vl);
    mask7 = __riscv_vmnot_m_b16(mask7, vl);
    Al = __riscv_vfsub_vv_f64m4(a, A, vl);
    al = __riscv_vfsub_vv_f64m4(__riscv_vfsub_vv_f64m4(__riscv_vfsub_vv_f64m4(A, __riscv_vfsub_vv_f64m4(a, Al, vl), vl), __riscv_vfadd_vv_f64m4(Al, c, vl), vl), al, vl);

    b = __riscv_vfmadd_vv_f64m4_mu(mask7, y, k, x, vl);
    Bl = __riscv_vfsub_vv_f64m4_mu(mask7, minus, b, x, vl);
    bl = __riscv_vfadd_vv_f64m4_mu(mask7, minus, __riscv_vfsub_vv_f64m4_mu(mask7, minus, x, __riscv_vfsub_vv_f64m4(b, Bl, vl), vl), __riscv_vfmsac_vv_f64m4(Bl, k, y, vl), vl);

    ch =  __riscv_vfdiv_vv_f64m4(a, b, vl);
    rh = __riscv_vfnmsub_vv_f64m4(ch, b, a, vl);
    rl = __riscv_vfnmsub_vv_f64m4(ch, bl, al, vl);
#ifndef __FAST_MATH__
    r = __riscv_vfadd_vv_f64m4_mu(mask_small, minus, rh, rl, vl);
#else
    r = __riscv_vfadd_vv_f64m4(rh, rl, vl);
#endif
    cl = __riscv_vfdiv_vv_f64m4(r, b, vl);

    w  =  __riscv_vfadd_vv_f64m4(cl, ch, vl);
#ifndef __FAST_MATH__
    wc =  __riscv_vfadd_vf_f64m4(w, 0x1p-86, vl);
#else
    wc = w;
#endif
    w_l = __riscv_vfadd_vv_f64m4(__riscv_vfsub_vv_f64m4(ch, w, vl), cl, vl);

#ifndef __FAST_MATH__
    index = __riscv_vadd_vx_u64m4_mu(__riscv_vmandn_mm_b16(sign_x, mask_nan, vl), index, index, magic_consts2[1], vl);
#else
    index = __riscv_vadd_vx_u64m4_mu(sign_x, index, index, magic_consts2[1], vl);
#endif

    a = __riscv_vloxei64_v_f64m4(d_table_hi, index, vl);
    b = __riscv_vloxei64_v_f64m4(d_table_lo, index, vl);

    y = __riscv_vfmul_vv_f64m4(wc, wc, vl);

    z = __riscv_vfmul_vv_f64m4(y, y, vl);

    pol1 = __riscv_vfmadd_vf_f64m4(z, pol_coef1[0], __riscv_vfmv_v_f_f64m4(pol_coef1[2], vl), vl);
    pol2 = __riscv_vfmadd_vf_f64m4(z, pol_coef1[1], __riscv_vfmv_v_f_f64m4(pol_coef1[3], vl), vl);
    pol1 = __riscv_vfmadd_vv_f64m4(pol1, z, __riscv_vfmv_v_f_f64m4(pol_coef2[0], vl), vl);
    pol2 = __riscv_vfmadd_vv_f64m4(pol2, z, __riscv_vfmv_v_f_f64m4(pol_coef2[1], vl), vl);
    pol1 = __riscv_vfmadd_vv_f64m4(pol1, z, __riscv_vfmv_v_f_f64m4(pol_coef2[2], vl), vl);
    pol2 = __riscv_vfmul_vv_f64m4(pol2,  z, vl);

    w  =  __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vxor_vv_u64m4_mu(sign_x, __riscv_vreinterpret_v_f64m4_u64m4(w),   __riscv_vreinterpret_v_f64m4_u64m4(minus), __riscv_vreinterpret_v_f64m4_u64m4(w),   vl));
    w_l = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vxor_vv_u64m4_mu(sign_x, __riscv_vreinterpret_v_f64m4_u64m4(w_l), __riscv_vreinterpret_v_f64m4_u64m4(minus), __riscv_vreinterpret_v_f64m4_u64m4(w_l), vl));

    x = __riscv_vfadd_vv_f64m4(a, w, vl);
    t = __riscv_vfsub_vv_f64m4(x, a, vl);

    w_l = __riscv_vfadd_vv_f64m4(__riscv_vfsub_vv_f64m4(w, t, vl), w_l, vl);

    x = __riscv_vfadd_vv_f64m4(
			       __riscv_vfadd_vv_f64m4(__riscv_vfsub_vf_f64m4(__riscv_vfmadd_vv_f64m4(__riscv_vfmadd_vv_f64m4(pol1, y, pol2, vl), w, b, vl), 
									      d_table_lo[0], 
									      vl),
						       w_l,
						       vl), 
			       x, 
			       vl);
#ifndef __FAST_MATH__
    x = __riscv_vfmacc_vf_f64m4_mu(mask_inf_x, x, 0x1p-1022, __riscv_vfmv_v_f_f64m4(0x1p-1022, vl), vl);
    x = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vor_vx_u64m4_mu(mask_nan, __riscv_vreinterpret_v_f64m4_u64m4(x), __riscv_vreinterpret_v_f64m4_u64m4(x), magic_consts2[3], vl));
    x = __riscv_vfmadd_vf_f64m4(x, 0x1p-60, x, vl);
#endif
    return __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vor_vv_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(sign_y), __riscv_vreinterpret_v_f64m4_u64m4(x), vl));
}







vfloat64m8_t __riscv_vatan2_f64m8(vfloat64m8_t y, vfloat64m8_t x, size_t vl) {
    vfloat64m8_t w, wc, w_l, z, k1, k2, k, a, b, al, bl, A, Al, B, Bl, pol1, pol2, minus, t, c1, c2, c, blend_x, blend_y, ch, cl, rh, rl, r, multiplicator, correction, xc, yc;
    vuint64m8_t index;
    vfloat64m8_t sign_y;
    vbool8_t masky0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, sign_x;
#ifndef __FAST_MATH__
    vbool8_t mask_inf_x, mask_inf_y, mask_nan;
    vbool8_t mask_small, mask_big, magic_mask, masky_uf, maskx_uf;
    vint64m8_t p_diff;
#endif
    minus = __riscv_vfmv_v_f_f64m8(borders1[0], vl);

    sign_x = __riscv_vmsgeu_vv_u64m8_b8(__riscv_vreinterpret_v_f64m8_u64m8(x), __riscv_vreinterpret_v_f64m8_u64m8(minus), vl);
    masky0 = __riscv_vmfeq_vv_f64m8_b8(minus, y, vl);
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmor_mm_b8(__riscv_vmfne_vv_f64m8_b8(x, x, vl), __riscv_vmfne_vv_f64m8_b8(y, y, vl), vl);
#endif
    sign_y = __riscv_vreinterpret_v_u64m8_f64m8(__riscv_vand_vv_u64m8(__riscv_vreinterpret_v_f64m8_u64m8(minus), __riscv_vreinterpret_v_f64m8_u64m8(y), vl));
    x = __riscv_vreinterpret_v_u64m8_f64m8(__riscv_vand_vv_u64m8(__riscv_vnot_v_u64m8(__riscv_vreinterpret_v_f64m8_u64m8(minus), vl), __riscv_vreinterpret_v_f64m8_u64m8(x), vl));
    y = __riscv_vreinterpret_v_u64m8_f64m8(__riscv_vand_vv_u64m8(__riscv_vnot_v_u64m8(__riscv_vreinterpret_v_f64m8_u64m8(minus), vl), __riscv_vreinterpret_v_f64m8_u64m8(y), vl));
#ifndef __FAST_MATH__
    mask_inf_y = __riscv_vmfeq_vf_f64m8_b8(y, *reinterpret_cast<const double*>(&magic_consts2[0]), vl);
    mask_inf_x = __riscv_vmfeq_vf_f64m8_b8(x, *reinterpret_cast<const double*>(&magic_consts2[0]), vl);
    y = __riscv_vfmerge_vfm_f64m8(y, borders2[0], mask_inf_y, vl);
    x = mask_null_f64m8(mask_inf_y, x, vl);

    y = mask_null_f64m8(__riscv_vmor_mm_b8(__riscv_vmandn_mm_b8(mask_inf_x, mask_inf_y, vl), mask_nan, vl), y, vl);
    x = __riscv_vfmerge_vfm_f64m8(x, borders2[0],  __riscv_vmor_mm_b8(__riscv_vmor_mm_b8(mask_inf_x, masky0, vl), mask_nan, vl), vl);
#else
    x = __riscv_vfmerge_vfm_f64m8(x, borders2[0], masky0, vl);
#endif

    multiplicator = __riscv_vreinterpret_v_i64m8_f64m8(
			__riscv_vsll_vx_i64m8(
				__riscv_vsub_vv_i64m8(
					__riscv_vmv_v_x_i64m8(magic_consts2[2], vl), 
					__riscv_vreinterpret_v_u64m8_i64m8(__riscv_vsrl_vx_u64m8(
						__riscv_vadd_vv_u64m8(
								      __riscv_vand_vx_u64m8(__riscv_vreinterpret_v_f64m8_u64m8(y), magic_consts2[0], vl), 
								      __riscv_vreinterpret_v_f64m8_u64m8(x), vl), 
						53, vl)), 
					vl),
				52, vl)
			);
#ifndef __FAST_MATH__
    magic_mask = __riscv_vmandn_mm_b8(__riscv_vmandn_mm_b8(sign_x, mask_nan, vl), mask_inf_x, vl);
#endif
    multiplicator = __riscv_vfmerge_vfm_f64m8(multiplicator, magic_consts1[1], __riscv_vmfle_vv_f64m8_b8(multiplicator, minus, vl), vl);
#ifndef __FAST_MATH__
    p_diff = __riscv_vsra_vx_i64m8(__riscv_vsub_vv_i64m8(__riscv_vreinterpret_v_f64m8_i64m8(y), __riscv_vreinterpret_v_f64m8_i64m8(x), vl), 52, vl);
#endif

    x = __riscv_vfmul_vv_f64m8(x, multiplicator, vl);
    y = __riscv_vfmul_vv_f64m8(y, multiplicator, vl);

#ifndef __FAST_MATH__
    maskx_uf = __riscv_vmflt_vf_f64m8_b8(x, 0x1p-1010, vl);
    masky_uf = __riscv_vmflt_vf_f64m8_b8(y, 0x1p-1010, vl);

    xc = mask_null_f64m8(maskx_uf, x, vl);
    magic_mask = __riscv_vmandn_mm_b8(magic_mask, masky0, vl);
    yc = mask_null_f64m8(masky_uf, y, vl);
    mask_small = __riscv_vmsgt_vx_i64m8_b8(p_diff, -0x3b, vl);
    magic_mask =  __riscv_vmandn_mm_b8(magic_mask, mask_small, vl);
#else
    xc = x;
    yc = y;
#endif

    mask1 = __riscv_vmflt_vv_f64m8_b8(__riscv_vfmul_vf_f64m8(xc, borders1[1], vl), y, vl);
    mask2 = __riscv_vmflt_vv_f64m8_b8(__riscv_vfmul_vf_f64m8(xc, borders1[2], vl), y, vl);
    mask3 = __riscv_vmflt_vv_f64m8_b8(__riscv_vfmul_vf_f64m8(xc, borders1[3], vl), y, vl);
    mask4 = __riscv_vmflt_vv_f64m8_b8(x, y, vl);
    mask5 = __riscv_vmfle_vv_f64m8_b8(x, __riscv_vfmul_vf_f64m8(yc, borders2[1], vl), vl);
    mask6 = __riscv_vmfle_vv_f64m8_b8(x, __riscv_vfmul_vf_f64m8(yc, borders2[2], vl), vl);
    mask7 = __riscv_vmfle_vv_f64m8_b8(x, __riscv_vfmul_vf_f64m8(yc, borders2[3], vl), vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmsge_vx_i64m8_b8(p_diff, 0x1f4, vl);
#endif

    index = __riscv_vmv_v_x_u64m8(0, vl);
#ifndef __FAST_MATH__
    correction = __riscv_vfmerge_vfm_f64m8(__riscv_vfmv_v_f_f64m8(0x1p60, vl), 0x1p-500, mask_big, vl);
#endif
    index = __riscv_vadd_vx_u64m8_mu(mask1, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask2, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask3, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask4, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask5, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask6, index, index, 8, vl);
    index = __riscv_vadd_vx_u64m8_mu(mask7, index, index, 8, vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmul_vv_f64m8_mu(__riscv_vmor_mm_b8(magic_mask, mask_big, vl), x, y, correction, vl);
#endif

    A = mask_null_f64m8(mask7, y, vl);
    k = __riscv_vloxei64_v_f64m8(d_k, index, vl);

    c = __riscv_vfmul_vv_f64m8(k, x, vl);
    al = __riscv_vfmsub_vv_f64m8(k, x, c, vl);
    a = __riscv_vfsub_vv_f64m8(A, c, vl);
    mask7 = __riscv_vmnot_m_b8(mask7, vl);
    Al = __riscv_vfsub_vv_f64m8(a, A, vl);
    al = __riscv_vfsub_vv_f64m8(__riscv_vfsub_vv_f64m8(__riscv_vfsub_vv_f64m8(A, __riscv_vfsub_vv_f64m8(a, Al, vl), vl), __riscv_vfadd_vv_f64m8(Al, c, vl), vl), al, vl);

    b = __riscv_vfmadd_vv_f64m8_mu(mask7, y, k, x, vl);
    Bl = __riscv_vfsub_vv_f64m8_mu(mask7, minus, b, x, vl);
    bl = __riscv_vfadd_vv_f64m8_mu(mask7, minus, __riscv_vfsub_vv_f64m8_mu(mask7, minus, x, __riscv_vfsub_vv_f64m8(b, Bl, vl), vl), __riscv_vfmsac_vv_f64m8(Bl, k, y, vl), vl);

    ch =  __riscv_vfdiv_vv_f64m8(a, b, vl);
    rh = __riscv_vfnmsub_vv_f64m8(ch, b, a, vl);
    rl = __riscv_vfnmsub_vv_f64m8(ch, bl, al, vl);
#ifndef __FAST_MATH__
    r = __riscv_vfadd_vv_f64m8_mu(mask_small, minus, rh, rl, vl);
#else
    r = __riscv_vfadd_vv_f64m8(rh, rl, vl);
#endif
    cl = __riscv_vfdiv_vv_f64m8(r, b, vl);

    w  =  __riscv_vfadd_vv_f64m8(cl, ch, vl);
#ifndef __FAST_MATH__
    wc =  __riscv_vfadd_vf_f64m8(w, 0x1p-86, vl);
#else
    wc = w;
#endif
    w_l = __riscv_vfadd_vv_f64m8(__riscv_vfsub_vv_f64m8(ch, w, vl), cl, vl);

#ifndef __FAST_MATH__
    index = __riscv_vadd_vx_u64m8_mu(__riscv_vmandn_mm_b8(sign_x, mask_nan, vl), index, index, magic_consts2[1], vl);
#else
    index = __riscv_vadd_vx_u64m8_mu(sign_x, index, index, magic_consts2[1], vl);
#endif

    a = __riscv_vloxei64_v_f64m8(d_table_hi, index, vl);
    b = __riscv_vloxei64_v_f64m8(d_table_lo, index, vl);

    y = __riscv_vfmul_vv_f64m8(wc, wc, vl);

    z = __riscv_vfmul_vv_f64m8(y, y, vl);

    pol1 = __riscv_vfmadd_vf_f64m8(z, pol_coef1[0], __riscv_vfmv_v_f_f64m8(pol_coef1[2], vl), vl);
    pol2 = __riscv_vfmadd_vf_f64m8(z, pol_coef1[1], __riscv_vfmv_v_f_f64m8(pol_coef1[3], vl), vl);
    pol1 = __riscv_vfmadd_vv_f64m8(pol1, z, __riscv_vfmv_v_f_f64m8(pol_coef2[0], vl), vl);
    pol2 = __riscv_vfmadd_vv_f64m8(pol2, z, __riscv_vfmv_v_f_f64m8(pol_coef2[1], vl), vl);
    pol1 = __riscv_vfmadd_vv_f64m8(pol1, z, __riscv_vfmv_v_f_f64m8(pol_coef2[2], vl), vl);
    pol2 = __riscv_vfmul_vv_f64m8(pol2,  z, vl);

    w  =  __riscv_vreinterpret_v_u64m8_f64m8(__riscv_vxor_vv_u64m8_mu(sign_x, __riscv_vreinterpret_v_f64m8_u64m8(w),   __riscv_vreinterpret_v_f64m8_u64m8(minus), __riscv_vreinterpret_v_f64m8_u64m8(w),   vl));
    w_l = __riscv_vreinterpret_v_u64m8_f64m8(__riscv_vxor_vv_u64m8_mu(sign_x, __riscv_vreinterpret_v_f64m8_u64m8(w_l), __riscv_vreinterpret_v_f64m8_u64m8(minus), __riscv_vreinterpret_v_f64m8_u64m8(w_l), vl));

    x = __riscv_vfadd_vv_f64m8(a, w, vl);
    t = __riscv_vfsub_vv_f64m8(x, a, vl);

    w_l = __riscv_vfadd_vv_f64m8(__riscv_vfsub_vv_f64m8(w, t, vl), w_l, vl);

    x = __riscv_vfadd_vv_f64m8(
			       __riscv_vfadd_vv_f64m8(__riscv_vfsub_vf_f64m8(__riscv_vfmadd_vv_f64m8(__riscv_vfmadd_vv_f64m8(pol1, y, pol2, vl), w, b, vl), 
									      d_table_lo[0], 
									      vl),
						       w_l,
						       vl), 
			       x, 
			       vl);
#ifndef __FAST_MATH__
    x = __riscv_vfmacc_vf_f64m8_mu(mask_inf_x, x, 0x1p-1022, __riscv_vfmv_v_f_f64m8(0x1p-1022, vl), vl);
    x = __riscv_vreinterpret_v_u64m8_f64m8(__riscv_vor_vx_u64m8_mu(mask_nan, __riscv_vreinterpret_v_f64m8_u64m8(x), __riscv_vreinterpret_v_f64m8_u64m8(x), magic_consts2[3], vl));
    x = __riscv_vfmadd_vf_f64m8(x, 0x1p-60, x, vl);
#endif
    return __riscv_vreinterpret_v_u64m8_f64m8(__riscv_vor_vv_u64m8(__riscv_vreinterpret_v_f64m8_u64m8(sign_y), __riscv_vreinterpret_v_f64m8_u64m8(x), vl));
}

vfloat32m1_t __riscv_vatan2_f32m1(vfloat32m1_t y, vfloat32m1_t x, size_t vl) {
    vfloat32m1_t w, wc, w_l, z, k, a, b, al, bl, A, Al, B, Bl, pol1, pol2, minus, t, c, blend_x, blend_y, ch, cl, rh, rl, r, multiplicator, correction, xc, yc;
    vuint32m1_t index;
    vfloat32m1_t sign_y;
    vbool32_t masky0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, sign_x;
#ifndef __FAST_MATH__
    vbool32_t mask_inf_x, mask_inf_y, mask_nan;
    vbool32_t mask_small, mask_big, magic_mask, masky_uf, maskx_uf;
    vint32m1_t p_diff;
#endif
    minus = __riscv_vfmv_v_f_f32m1(f_borders1[0], vl);

    sign_x = __riscv_vmsgeu_vv_u32m1_b32(__riscv_vreinterpret_v_f32m1_u32m1(x), __riscv_vreinterpret_v_f32m1_u32m1(minus), vl);
    masky0 = __riscv_vmfeq_vv_f32m1_b32(minus, y, vl);
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmor_mm_b32(__riscv_vmfne_vv_f32m1_b32(x, x, vl), __riscv_vmfne_vv_f32m1_b32(y, y, vl), vl);
#endif
    sign_y = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(minus), __riscv_vreinterpret_v_f32m1_u32m1(y), vl));
    x = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(__riscv_vnot_v_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(minus), vl), __riscv_vreinterpret_v_f32m1_u32m1(x), vl));
    y = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1(__riscv_vnot_v_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(minus), vl), __riscv_vreinterpret_v_f32m1_u32m1(y), vl));
#ifndef __FAST_MATH__
    mask_inf_y = __riscv_vmfeq_vf_f32m1_b32(y, f_magic_consts1[0], vl);
    mask_inf_x = __riscv_vmfeq_vf_f32m1_b32(x, f_magic_consts1[0], vl);
    y = __riscv_vfmerge_vfm_f32m1(y, f_borders2[0], mask_inf_y, vl);
    x = mask_null_f32m1(mask_inf_y, x, vl);

    y = mask_null_f32m1(__riscv_vmor_mm_b32(__riscv_vmandn_mm_b32(mask_inf_x, mask_inf_y, vl), mask_nan, vl), y, vl);
    x = __riscv_vfmerge_vfm_f32m1(x, f_borders2[0],  __riscv_vmor_mm_b32(__riscv_vmor_mm_b32(mask_inf_x, masky0, vl), mask_nan, vl), vl);
#else
    x = __riscv_vfmerge_vfm_f32m1(x, f_borders2[0], masky0, vl);
#endif

    multiplicator = __riscv_vreinterpret_v_i32m1_f32m1(
			__riscv_vsll_vx_i32m1(
				__riscv_vsub_vv_i32m1(
					__riscv_vmv_v_x_i32m1(f_magic_consts2[2], vl), 
					__riscv_vreinterpret_v_u32m1_i32m1(__riscv_vsrl_vx_u32m1(
						__riscv_vadd_vv_u32m1(
								      __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(y), 0x7f800000, vl), 
								      __riscv_vreinterpret_v_f32m1_u32m1(x), 
								      vl), 
						24, vl)), 
					vl),
				23, vl)
			);
#ifndef __FAST_MATH__
    magic_mask = __riscv_vmandn_mm_b32(__riscv_vmandn_mm_b32(sign_x, mask_nan, vl), mask_inf_x, vl);
#endif

    multiplicator = __riscv_vfmerge_vfm_f32m1(multiplicator, f_magic_consts1[1], __riscv_vmfle_vv_f32m1_b32(multiplicator, minus, vl), vl);
#ifndef __FAST_MATH__
    p_diff = __riscv_vsra_vx_i32m1(__riscv_vsub_vv_i32m1(__riscv_vreinterpret_v_f32m1_i32m1(y), __riscv_vreinterpret_v_f32m1_i32m1(x), vl), 23, vl);
#endif

    x = __riscv_vfmul_vv_f32m1(x, multiplicator, vl);
    y = __riscv_vfmul_vv_f32m1(y, multiplicator, vl);

#ifndef __FAST_MATH__
    maskx_uf = __riscv_vmflt_vf_f32m1_b32(x, 0x1p-122f, vl);
    masky_uf = __riscv_vmflt_vf_f32m1_b32(y, 0x1p-122f, vl);

    xc = mask_null_f32m1(maskx_uf, x, vl);
    magic_mask = __riscv_vmandn_mm_b32(magic_mask, masky0, vl);
    yc = mask_null_f32m1(masky_uf, y, vl);
    mask_small = __riscv_vmsgt_vx_i32m1_b32(p_diff, -0x20, vl);
    magic_mask =  __riscv_vmandn_mm_b32(magic_mask, mask_small, vl);
#else
    xc = x;
    yc = y;
#endif
    mask1 = __riscv_vmflt_vv_f32m1_b32(__riscv_vfmul_vf_f32m1(xc, f_borders1[1], vl), y, vl);
    mask2 = __riscv_vmflt_vv_f32m1_b32(__riscv_vfmul_vf_f32m1(xc, f_borders1[2], vl), y, vl);
    mask3 = __riscv_vmflt_vv_f32m1_b32(__riscv_vfmul_vf_f32m1(xc, f_borders1[3], vl), y, vl);
    mask4 = __riscv_vmflt_vv_f32m1_b32(x, y, vl);
    mask5 = __riscv_vmfle_vv_f32m1_b32(x, __riscv_vfmul_vf_f32m1(yc, f_borders2[1], vl), vl);
    mask6 = __riscv_vmfle_vv_f32m1_b32(x, __riscv_vfmul_vf_f32m1(yc, f_borders2[2], vl), vl);
    mask7 = __riscv_vmfle_vv_f32m1_b32(x, __riscv_vfmul_vf_f32m1(yc, f_borders2[3], vl), vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmsge_vx_i32m1_b32(p_diff, 0x3b, vl);
#endif
    index = __riscv_vmv_v_x_u32m1(0, vl);
#ifndef __FAST_MATH__
    correction = __riscv_vfmerge_vfm_f32m1(__riscv_vfmv_v_f_f32m1(0x1p32f, vl), 0x1p-60, mask_big, vl);
#endif
    index = __riscv_vadd_vx_u32m1_mu(mask1, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m1_mu(mask2, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m1_mu(mask3, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m1_mu(mask4, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m1_mu(mask5, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m1_mu(mask6, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m1_mu(mask7, index, index, 4, vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmul_vv_f32m1_mu(__riscv_vmor_mm_b32(magic_mask, mask_big, vl), x, y, correction, vl);
#endif

    A = mask_null_f32m1(mask7, y, vl);
    k = __riscv_vloxei32_v_f32m1(f_k, index, vl);

    c = __riscv_vfmul_vv_f32m1(k, x, vl);
    al = __riscv_vfmsub_vv_f32m1(k, x, c, vl);
    a = __riscv_vfsub_vv_f32m1(A, c, vl);
    mask7 = __riscv_vmnot_m_b32(mask7, vl);
    Al = __riscv_vfsub_vv_f32m1(a, A, vl);
    al = __riscv_vfsub_vv_f32m1(__riscv_vfsub_vv_f32m1(__riscv_vfsub_vv_f32m1(A, __riscv_vfsub_vv_f32m1(a, Al, vl), vl), __riscv_vfadd_vv_f32m1(Al, c, vl), vl), al, vl);

    b = __riscv_vfmadd_vv_f32m1_mu(mask7, y, k, x, vl);
    Bl = __riscv_vfsub_vv_f32m1_mu(mask7, minus, b, x, vl);
    bl = __riscv_vfadd_vv_f32m1_mu(mask7, minus, __riscv_vfsub_vv_f32m1_mu(mask7, minus, x, __riscv_vfsub_vv_f32m1(b, Bl, vl), vl), __riscv_vfmsac_vv_f32m1(Bl, k, y, vl), vl);

    ch =  __riscv_vfdiv_vv_f32m1(a, b, vl);
    rh = __riscv_vfnmsub_vv_f32m1(ch, b, a, vl);
    rl = __riscv_vfnmsub_vv_f32m1(ch, bl, al, vl);

#ifndef __FAST_MATH__
    r = __riscv_vfadd_vv_f32m1_mu(mask_small, minus, rh, rl, vl);
#else
    r = __riscv_vfadd_vv_f32m1(rh, rl, vl);
#endif

    r =  __riscv_vfmul_vf_f32m1(r, f_magic_consts1[2], vl);

    cl = __riscv_vfdiv_vv_f32m1(r, b, vl);

    w = __riscv_vfmadd_vf_f32m1(ch, f_magic_consts1[2], cl, vl);
    w_l = __riscv_vfadd_vv_f32m1(__riscv_vfmsub_vf_f32m1(ch, f_magic_consts1[2], w, vl), cl, vl);
    wc = __riscv_vfmadd_vf_f32m1(w, f_magic_consts1[3], __riscv_vfmv_v_f_f32m1(0x1p-40f, vl), vl);

#ifndef __FAST_MATH__
    index = __riscv_vadd_vx_u32m1_mu(__riscv_vmandn_mm_b32(sign_x, mask_nan, vl), index, index, f_magic_consts2[1], vl);
#else
    index = __riscv_vadd_vx_u32m1_mu(sign_x, index, index, f_magic_consts2[1], vl);
#endif

    a = __riscv_vloxei32_v_f32m1(f_table_hi, index, vl);
    b = __riscv_vloxei32_v_f32m1(f_table_lo, index, vl);

    y = __riscv_vfmadd_vv_f32m1(wc, wc, __riscv_vfmv_v_f_f32m1(0x1p-40f, vl), vl);

    z = __riscv_vfmul_vv_f32m1(y, y, vl);

    pol1 = __riscv_vfmadd_vf_f32m1(z, f_pol_coef[2], __riscv_vfmv_v_f_f32m1(f_pol_coef[0], vl), vl);
    pol2 = __riscv_vfmul_vf_f32m1(z, f_pol_coef[1], vl);

    w  =  __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vxor_vv_u32m1_mu(sign_x, __riscv_vreinterpret_v_f32m1_u32m1(w),   __riscv_vreinterpret_v_f32m1_u32m1(minus), __riscv_vreinterpret_v_f32m1_u32m1(w),   vl));
    w_l = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vxor_vv_u32m1_mu(sign_x, __riscv_vreinterpret_v_f32m1_u32m1(w_l), __riscv_vreinterpret_v_f32m1_u32m1(minus), __riscv_vreinterpret_v_f32m1_u32m1(w_l), vl));

    x = __riscv_vfmadd_vf_f32m1(w, f_magic_consts1[3], a, vl);
    t = __riscv_vfsub_vv_f32m1(x, a, vl);
    w_l = __riscv_vfadd_vv_f32m1(__riscv_vfnmsub_vf_f32m1(t, f_magic_consts1[2], w, vl), w_l, vl);

    x = __riscv_vfmadd_vf_f32m1(__riscv_vfadd_vv_f32m1(__riscv_vfsub_vf_f32m1(__riscv_vfmadd_vv_f32m1(__riscv_vfmadd_vv_f32m1(pol1, y, pol2, vl), w, b, vl), f_table_lo[0], vl),  w_l, vl), f_magic_consts1[3], x, vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmacc_vf_f32m1_mu(mask_inf_x, x, 0x1p-126f, __riscv_vfmv_v_f_f32m1(0x1p-126f, vl), vl);
    x = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vor_vx_u32m1_mu(mask_nan, __riscv_vreinterpret_v_f32m1_u32m1(x), __riscv_vreinterpret_v_f32m1_u32m1(x), f_magic_consts2[3], vl));
    x = __riscv_vfmadd_vf_f32m1(x, 0x1p-30f, x, vl);
#endif

    return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vor_vv_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(sign_y), __riscv_vreinterpret_v_f32m1_u32m1(x), vl));
}






vfloat32m2_t __riscv_vatan2_f32m2(vfloat32m2_t y, vfloat32m2_t x, size_t vl) {
    vfloat32m2_t w, wc, w_l, z, k, a, b, al, bl, A, Al, B, Bl, pol1, pol2, minus, t, c, blend_x, blend_y, ch, cl, rh, rl, r, multiplicator, correction, xc, yc;
    vuint32m2_t index;
    vfloat32m2_t sign_y;
    vbool16_t masky0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, sign_x;
#ifndef __FAST_MATH__
    vbool16_t mask_inf_x, mask_inf_y, mask_nan;
    vbool16_t mask_small, mask_big, magic_mask, masky_uf, maskx_uf;
    vint32m2_t p_diff;
#endif
    minus = __riscv_vfmv_v_f_f32m2(f_borders1[0], vl);

    sign_x = __riscv_vmsgeu_vv_u32m2_b16(__riscv_vreinterpret_v_f32m2_u32m2(x), __riscv_vreinterpret_v_f32m2_u32m2(minus), vl);
    masky0 = __riscv_vmfeq_vv_f32m2_b16(minus, y, vl);
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmor_mm_b16(__riscv_vmfne_vv_f32m2_b16(x, x, vl), __riscv_vmfne_vv_f32m2_b16(y, y, vl), vl);
#endif
    sign_y = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vand_vv_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(minus), __riscv_vreinterpret_v_f32m2_u32m2(y), vl));
    x = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vand_vv_u32m2(__riscv_vnot_v_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(minus), vl), __riscv_vreinterpret_v_f32m2_u32m2(x), vl));
    y = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vand_vv_u32m2(__riscv_vnot_v_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(minus), vl), __riscv_vreinterpret_v_f32m2_u32m2(y), vl));
#ifndef __FAST_MATH__
    mask_inf_y = __riscv_vmfeq_vf_f32m2_b16(y, f_magic_consts1[0], vl);
    mask_inf_x = __riscv_vmfeq_vf_f32m2_b16(x, f_magic_consts1[0], vl);
    y = __riscv_vfmerge_vfm_f32m2(y, f_borders2[0], mask_inf_y, vl);
    x = mask_null_f32m2(mask_inf_y, x, vl);

    y = mask_null_f32m2(__riscv_vmor_mm_b16(__riscv_vmandn_mm_b16(mask_inf_x, mask_inf_y, vl), mask_nan, vl), y, vl);
    x = __riscv_vfmerge_vfm_f32m2(x, f_borders2[0],  __riscv_vmor_mm_b16(__riscv_vmor_mm_b16(mask_inf_x, masky0, vl), mask_nan, vl), vl);
#else
    x = __riscv_vfmerge_vfm_f32m2(x, f_borders2[0], masky0, vl);
#endif

    multiplicator = __riscv_vreinterpret_v_i32m2_f32m2(
			__riscv_vsll_vx_i32m2(
				__riscv_vsub_vv_i32m2(
					__riscv_vmv_v_x_i32m2(f_magic_consts2[2], vl), 
					__riscv_vreinterpret_v_u32m2_i32m2(__riscv_vsrl_vx_u32m2(
						__riscv_vadd_vv_u32m2(
								      __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(y), 0x7f800000, vl), 
								      __riscv_vreinterpret_v_f32m2_u32m2(x), 
								      vl), 
						24, vl)), 
					vl),
				23, vl)
			);
#ifndef __FAST_MATH__
    magic_mask = __riscv_vmandn_mm_b16(__riscv_vmandn_mm_b16(sign_x, mask_nan, vl), mask_inf_x, vl);
#endif

    multiplicator = __riscv_vfmerge_vfm_f32m2(multiplicator, f_magic_consts1[1], __riscv_vmfle_vv_f32m2_b16(multiplicator, minus, vl), vl);
#ifndef __FAST_MATH__
    p_diff = __riscv_vsra_vx_i32m2(__riscv_vsub_vv_i32m2(__riscv_vreinterpret_v_f32m2_i32m2(y), __riscv_vreinterpret_v_f32m2_i32m2(x), vl), 23, vl);
#endif

    x = __riscv_vfmul_vv_f32m2(x, multiplicator, vl);
    y = __riscv_vfmul_vv_f32m2(y, multiplicator, vl);

#ifndef __FAST_MATH__
    maskx_uf = __riscv_vmflt_vf_f32m2_b16(x, 0x1p-122f, vl);
    masky_uf = __riscv_vmflt_vf_f32m2_b16(y, 0x1p-122f, vl);

    xc = mask_null_f32m2(maskx_uf, x, vl);
    magic_mask = __riscv_vmandn_mm_b16(magic_mask, masky0, vl);
    yc = mask_null_f32m2(masky_uf, y, vl);
    mask_small = __riscv_vmsgt_vx_i32m2_b16(p_diff, -0x20, vl);
    magic_mask =  __riscv_vmandn_mm_b16(magic_mask, mask_small, vl);
#else
    xc = x;
    yc = y;
#endif
    mask1 = __riscv_vmflt_vv_f32m2_b16(__riscv_vfmul_vf_f32m2(xc, f_borders1[1], vl), y, vl);
    mask2 = __riscv_vmflt_vv_f32m2_b16(__riscv_vfmul_vf_f32m2(xc, f_borders1[2], vl), y, vl);
    mask3 = __riscv_vmflt_vv_f32m2_b16(__riscv_vfmul_vf_f32m2(xc, f_borders1[3], vl), y, vl);
    mask4 = __riscv_vmflt_vv_f32m2_b16(x, y, vl);
    mask5 = __riscv_vmfle_vv_f32m2_b16(x, __riscv_vfmul_vf_f32m2(yc, f_borders2[1], vl), vl);
    mask6 = __riscv_vmfle_vv_f32m2_b16(x, __riscv_vfmul_vf_f32m2(yc, f_borders2[2], vl), vl);
    mask7 = __riscv_vmfle_vv_f32m2_b16(x, __riscv_vfmul_vf_f32m2(yc, f_borders2[3], vl), vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmsge_vx_i32m2_b16(p_diff, 0x3b, vl);
#endif
    index = __riscv_vmv_v_x_u32m2(0, vl);
#ifndef __FAST_MATH__
    correction = __riscv_vfmerge_vfm_f32m2(__riscv_vfmv_v_f_f32m2(0x1p32f, vl), 0x1p-60, mask_big, vl);
#endif
    index = __riscv_vadd_vx_u32m2_mu(mask1, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m2_mu(mask2, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m2_mu(mask3, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m2_mu(mask4, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m2_mu(mask5, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m2_mu(mask6, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m2_mu(mask7, index, index, 4, vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmul_vv_f32m2_mu(__riscv_vmor_mm_b16(magic_mask, mask_big, vl), x, y, correction, vl);
#endif

    A = mask_null_f32m2(mask7, y, vl);
    k = __riscv_vloxei32_v_f32m2(f_k, index, vl);

    c = __riscv_vfmul_vv_f32m2(k, x, vl);
    al = __riscv_vfmsub_vv_f32m2(k, x, c, vl);
    a = __riscv_vfsub_vv_f32m2(A, c, vl);
    mask7 = __riscv_vmnot_m_b16(mask7, vl);
    Al = __riscv_vfsub_vv_f32m2(a, A, vl);
    al = __riscv_vfsub_vv_f32m2(__riscv_vfsub_vv_f32m2(__riscv_vfsub_vv_f32m2(A, __riscv_vfsub_vv_f32m2(a, Al, vl), vl), __riscv_vfadd_vv_f32m2(Al, c, vl), vl), al, vl);

    b = __riscv_vfmadd_vv_f32m2_mu(mask7, y, k, x, vl);
    Bl = __riscv_vfsub_vv_f32m2_mu(mask7, minus, b, x, vl);
    bl = __riscv_vfadd_vv_f32m2_mu(mask7, minus, __riscv_vfsub_vv_f32m2_mu(mask7, minus, x, __riscv_vfsub_vv_f32m2(b, Bl, vl), vl), __riscv_vfmsac_vv_f32m2(Bl, k, y, vl), vl);

    ch =  __riscv_vfdiv_vv_f32m2(a, b, vl);
    rh = __riscv_vfnmsub_vv_f32m2(ch, b, a, vl);
    rl = __riscv_vfnmsub_vv_f32m2(ch, bl, al, vl);

#ifndef __FAST_MATH__
    r = __riscv_vfadd_vv_f32m2_mu(mask_small, minus, rh, rl, vl);
#else
    r = __riscv_vfadd_vv_f32m2(rh, rl, vl);
#endif

    r =  __riscv_vfmul_vf_f32m2(r, f_magic_consts1[2], vl);

    cl = __riscv_vfdiv_vv_f32m2(r, b, vl);

    w = __riscv_vfmadd_vf_f32m2(ch, f_magic_consts1[2], cl, vl);
    w_l = __riscv_vfadd_vv_f32m2(__riscv_vfmsub_vf_f32m2(ch, f_magic_consts1[2], w, vl), cl, vl);
    wc = __riscv_vfmadd_vf_f32m2(w, f_magic_consts1[3], __riscv_vfmv_v_f_f32m2(0x1p-40f, vl), vl);

#ifndef __FAST_MATH__
    index = __riscv_vadd_vx_u32m2_mu(__riscv_vmandn_mm_b16(sign_x, mask_nan, vl), index, index, f_magic_consts2[1], vl);
#else
    index = __riscv_vadd_vx_u32m2_mu(sign_x, index, index, f_magic_consts2[1], vl);
#endif

    a = __riscv_vloxei32_v_f32m2(f_table_hi, index, vl);
    b = __riscv_vloxei32_v_f32m2(f_table_lo, index, vl);

    y = __riscv_vfmadd_vv_f32m2(wc, wc, __riscv_vfmv_v_f_f32m2(0x1p-40f, vl), vl);

    z = __riscv_vfmul_vv_f32m2(y, y, vl);

    pol1 = __riscv_vfmadd_vf_f32m2(z, f_pol_coef[2], __riscv_vfmv_v_f_f32m2(f_pol_coef[0], vl), vl);
    pol2 = __riscv_vfmul_vf_f32m2(z, f_pol_coef[1], vl);

    w  =  __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vxor_vv_u32m2_mu(sign_x, __riscv_vreinterpret_v_f32m2_u32m2(w),   __riscv_vreinterpret_v_f32m2_u32m2(minus), __riscv_vreinterpret_v_f32m2_u32m2(w),   vl));
    w_l = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vxor_vv_u32m2_mu(sign_x, __riscv_vreinterpret_v_f32m2_u32m2(w_l), __riscv_vreinterpret_v_f32m2_u32m2(minus), __riscv_vreinterpret_v_f32m2_u32m2(w_l), vl));

    x = __riscv_vfmadd_vf_f32m2(w, f_magic_consts1[3], a, vl);
    t = __riscv_vfsub_vv_f32m2(x, a, vl);
    w_l = __riscv_vfadd_vv_f32m2(__riscv_vfnmsub_vf_f32m2(t, f_magic_consts1[2], w, vl), w_l, vl);

    x = __riscv_vfmadd_vf_f32m2(__riscv_vfadd_vv_f32m2(__riscv_vfsub_vf_f32m2(__riscv_vfmadd_vv_f32m2(__riscv_vfmadd_vv_f32m2(pol1, y, pol2, vl), w, b, vl), f_table_lo[0], vl),  w_l, vl), f_magic_consts1[3], x, vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmacc_vf_f32m2_mu(mask_inf_x, x, 0x1p-126f, __riscv_vfmv_v_f_f32m2(0x1p-126f, vl), vl);
    x = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vor_vx_u32m2_mu(mask_nan, __riscv_vreinterpret_v_f32m2_u32m2(x), __riscv_vreinterpret_v_f32m2_u32m2(x), f_magic_consts2[3], vl));
    x = __riscv_vfmadd_vf_f32m2(x, 0x1p-30f, x, vl);
#endif

    return __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vor_vv_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(sign_y), __riscv_vreinterpret_v_f32m2_u32m2(x), vl));
}







vfloat32m4_t __riscv_vatan2_f32m4(vfloat32m4_t y, vfloat32m4_t x, size_t vl) {
    vfloat32m4_t w, wc, w_l, z, k, a, b, al, bl, A, Al, B, Bl, pol1, pol2, minus, t, c, blend_x, blend_y, ch, cl, rh, rl, r, multiplicator, correction, xc, yc;
    vuint32m4_t index;
    vfloat32m4_t sign_y;
    vbool8_t masky0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, sign_x;
#ifndef __FAST_MATH__
    vbool8_t mask_inf_x, mask_inf_y, mask_nan;
    vbool8_t mask_small, mask_big, magic_mask, masky_uf, maskx_uf;
    vint32m4_t p_diff;
#endif
    minus = __riscv_vfmv_v_f_f32m4(f_borders1[0], vl);

    sign_x = __riscv_vmsgeu_vv_u32m4_b8(__riscv_vreinterpret_v_f32m4_u32m4(x), __riscv_vreinterpret_v_f32m4_u32m4(minus), vl);
    masky0 = __riscv_vmfeq_vv_f32m4_b8(minus, y, vl);
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmor_mm_b8(__riscv_vmfne_vv_f32m4_b8(x, x, vl), __riscv_vmfne_vv_f32m4_b8(y, y, vl), vl);
#endif
    sign_y = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vand_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(minus), __riscv_vreinterpret_v_f32m4_u32m4(y), vl));
    x = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vand_vv_u32m4(__riscv_vnot_v_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(minus), vl), __riscv_vreinterpret_v_f32m4_u32m4(x), vl));
    y = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vand_vv_u32m4(__riscv_vnot_v_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(minus), vl), __riscv_vreinterpret_v_f32m4_u32m4(y), vl));
#ifndef __FAST_MATH__
    mask_inf_y = __riscv_vmfeq_vf_f32m4_b8(y, f_magic_consts1[0], vl);
    mask_inf_x = __riscv_vmfeq_vf_f32m4_b8(x, f_magic_consts1[0], vl);
    y = __riscv_vfmerge_vfm_f32m4(y, f_borders2[0], mask_inf_y, vl);
    x = mask_null_f32m4(mask_inf_y, x, vl);

    y = mask_null_f32m4(__riscv_vmor_mm_b8(__riscv_vmandn_mm_b8(mask_inf_x, mask_inf_y, vl), mask_nan, vl), y, vl);
    x = __riscv_vfmerge_vfm_f32m4(x, f_borders2[0],  __riscv_vmor_mm_b8(__riscv_vmor_mm_b8(mask_inf_x, masky0, vl), mask_nan, vl), vl);
#else
    x = __riscv_vfmerge_vfm_f32m4(x, f_borders2[0], masky0, vl);
#endif

    multiplicator = __riscv_vreinterpret_v_i32m4_f32m4(
			__riscv_vsll_vx_i32m4(
				__riscv_vsub_vv_i32m4(
					__riscv_vmv_v_x_i32m4(f_magic_consts2[2], vl), 
					__riscv_vreinterpret_v_u32m4_i32m4(__riscv_vsrl_vx_u32m4(
						__riscv_vadd_vv_u32m4(
								      __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(y), 0x7f800000, vl), 
								      __riscv_vreinterpret_v_f32m4_u32m4(x), 
								      vl), 
						24, vl)), 
					vl),
				23, vl)
			);
#ifndef __FAST_MATH__
    magic_mask = __riscv_vmandn_mm_b8(__riscv_vmandn_mm_b8(sign_x, mask_nan, vl), mask_inf_x, vl);
#endif

    multiplicator = __riscv_vfmerge_vfm_f32m4(multiplicator, f_magic_consts1[1], __riscv_vmfle_vv_f32m4_b8(multiplicator, minus, vl), vl);
#ifndef __FAST_MATH__
    p_diff = __riscv_vsra_vx_i32m4(__riscv_vsub_vv_i32m4(__riscv_vreinterpret_v_f32m4_i32m4(y), __riscv_vreinterpret_v_f32m4_i32m4(x), vl), 23, vl);
#endif

    x = __riscv_vfmul_vv_f32m4(x, multiplicator, vl);
    y = __riscv_vfmul_vv_f32m4(y, multiplicator, vl);

#ifndef __FAST_MATH__
    maskx_uf = __riscv_vmflt_vf_f32m4_b8(x, 0x1p-122f, vl);
    masky_uf = __riscv_vmflt_vf_f32m4_b8(y, 0x1p-122f, vl);

    xc = mask_null_f32m4(maskx_uf, x, vl);
    magic_mask = __riscv_vmandn_mm_b8(magic_mask, masky0, vl);
    yc = mask_null_f32m4(masky_uf, y, vl);
    mask_small = __riscv_vmsgt_vx_i32m4_b8(p_diff, -0x20, vl);
    magic_mask =  __riscv_vmandn_mm_b8(magic_mask, mask_small, vl);
#else
    xc = x;
    yc = y;
#endif
    mask1 = __riscv_vmflt_vv_f32m4_b8(__riscv_vfmul_vf_f32m4(xc, f_borders1[1], vl), y, vl);
    mask2 = __riscv_vmflt_vv_f32m4_b8(__riscv_vfmul_vf_f32m4(xc, f_borders1[2], vl), y, vl);
    mask3 = __riscv_vmflt_vv_f32m4_b8(__riscv_vfmul_vf_f32m4(xc, f_borders1[3], vl), y, vl);
    mask4 = __riscv_vmflt_vv_f32m4_b8(x, y, vl);
    mask5 = __riscv_vmfle_vv_f32m4_b8(x, __riscv_vfmul_vf_f32m4(yc, f_borders2[1], vl), vl);
    mask6 = __riscv_vmfle_vv_f32m4_b8(x, __riscv_vfmul_vf_f32m4(yc, f_borders2[2], vl), vl);
    mask7 = __riscv_vmfle_vv_f32m4_b8(x, __riscv_vfmul_vf_f32m4(yc, f_borders2[3], vl), vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmsge_vx_i32m4_b8(p_diff, 0x3b, vl);
#endif
    index = __riscv_vmv_v_x_u32m4(0, vl);
#ifndef __FAST_MATH__
    correction = __riscv_vfmerge_vfm_f32m4(__riscv_vfmv_v_f_f32m4(0x1p32f, vl), 0x1p-60, mask_big, vl);
#endif
    index = __riscv_vadd_vx_u32m4_mu(mask1, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m4_mu(mask2, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m4_mu(mask3, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m4_mu(mask4, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m4_mu(mask5, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m4_mu(mask6, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m4_mu(mask7, index, index, 4, vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmul_vv_f32m4_mu(__riscv_vmor_mm_b8(magic_mask, mask_big, vl), x, y, correction, vl);
#endif

    A = mask_null_f32m4(mask7, y, vl);
    k = __riscv_vloxei32_v_f32m4(f_k, index, vl);

    c = __riscv_vfmul_vv_f32m4(k, x, vl);
    al = __riscv_vfmsub_vv_f32m4(k, x, c, vl);
    a = __riscv_vfsub_vv_f32m4(A, c, vl);
    mask7 = __riscv_vmnot_m_b8(mask7, vl);
    Al = __riscv_vfsub_vv_f32m4(a, A, vl);
    al = __riscv_vfsub_vv_f32m4(__riscv_vfsub_vv_f32m4(__riscv_vfsub_vv_f32m4(A, __riscv_vfsub_vv_f32m4(a, Al, vl), vl), __riscv_vfadd_vv_f32m4(Al, c, vl), vl), al, vl);

    b = __riscv_vfmadd_vv_f32m4_mu(mask7, y, k, x, vl);
    Bl = __riscv_vfsub_vv_f32m4_mu(mask7, minus, b, x, vl);
    bl = __riscv_vfadd_vv_f32m4_mu(mask7, minus, __riscv_vfsub_vv_f32m4_mu(mask7, minus, x, __riscv_vfsub_vv_f32m4(b, Bl, vl), vl), __riscv_vfmsac_vv_f32m4(Bl, k, y, vl), vl);

    ch =  __riscv_vfdiv_vv_f32m4(a, b, vl);
    rh = __riscv_vfnmsub_vv_f32m4(ch, b, a, vl);
    rl = __riscv_vfnmsub_vv_f32m4(ch, bl, al, vl);

#ifndef __FAST_MATH__
    r = __riscv_vfadd_vv_f32m4_mu(mask_small, minus, rh, rl, vl);
#else
    r = __riscv_vfadd_vv_f32m4(rh, rl, vl);
#endif

    r =  __riscv_vfmul_vf_f32m4(r, f_magic_consts1[2], vl);

    cl = __riscv_vfdiv_vv_f32m4(r, b, vl);

    w = __riscv_vfmadd_vf_f32m4(ch, f_magic_consts1[2], cl, vl);
    w_l = __riscv_vfadd_vv_f32m4(__riscv_vfmsub_vf_f32m4(ch, f_magic_consts1[2], w, vl), cl, vl);
    wc = __riscv_vfmadd_vf_f32m4(w, f_magic_consts1[3], __riscv_vfmv_v_f_f32m4(0x1p-40f, vl), vl);

#ifndef __FAST_MATH__
    index = __riscv_vadd_vx_u32m4_mu(__riscv_vmandn_mm_b8(sign_x, mask_nan, vl), index, index, f_magic_consts2[1], vl);
#else
    index = __riscv_vadd_vx_u32m4_mu(sign_x, index, index, f_magic_consts2[1], vl);
#endif

    a = __riscv_vloxei32_v_f32m4(f_table_hi, index, vl);
    b = __riscv_vloxei32_v_f32m4(f_table_lo, index, vl);

    y = __riscv_vfmadd_vv_f32m4(wc, wc, __riscv_vfmv_v_f_f32m4(0x1p-40f, vl), vl);

    z = __riscv_vfmul_vv_f32m4(y, y, vl);

    pol1 = __riscv_vfmadd_vf_f32m4(z, f_pol_coef[2], __riscv_vfmv_v_f_f32m4(f_pol_coef[0], vl), vl);
    pol2 = __riscv_vfmul_vf_f32m4(z, f_pol_coef[1], vl);

    w  =  __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vxor_vv_u32m4_mu(sign_x, __riscv_vreinterpret_v_f32m4_u32m4(w),   __riscv_vreinterpret_v_f32m4_u32m4(minus), __riscv_vreinterpret_v_f32m4_u32m4(w),   vl));
    w_l = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vxor_vv_u32m4_mu(sign_x, __riscv_vreinterpret_v_f32m4_u32m4(w_l), __riscv_vreinterpret_v_f32m4_u32m4(minus), __riscv_vreinterpret_v_f32m4_u32m4(w_l), vl));

    x = __riscv_vfmadd_vf_f32m4(w, f_magic_consts1[3], a, vl);
    t = __riscv_vfsub_vv_f32m4(x, a, vl);
    w_l = __riscv_vfadd_vv_f32m4(__riscv_vfnmsub_vf_f32m4(t, f_magic_consts1[2], w, vl), w_l, vl);

    x = __riscv_vfmadd_vf_f32m4(__riscv_vfadd_vv_f32m4(__riscv_vfsub_vf_f32m4(__riscv_vfmadd_vv_f32m4(__riscv_vfmadd_vv_f32m4(pol1, y, pol2, vl), w, b, vl), f_table_lo[0], vl),  w_l, vl), f_magic_consts1[3], x, vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmacc_vf_f32m4_mu(mask_inf_x, x, 0x1p-126f, __riscv_vfmv_v_f_f32m4(0x1p-126f, vl), vl);
    x = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vx_u32m4_mu(mask_nan, __riscv_vreinterpret_v_f32m4_u32m4(x), __riscv_vreinterpret_v_f32m4_u32m4(x), f_magic_consts2[3], vl));
    x = __riscv_vfmadd_vf_f32m4(x, 0x1p-30f, x, vl);
#endif

    return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(sign_y), __riscv_vreinterpret_v_f32m4_u32m4(x), vl));
}








vfloat32m8_t __riscv_vatan2_f32m8(vfloat32m8_t y, vfloat32m8_t x, size_t vl) {
    vfloat32m8_t w, wc, w_l, z, k, a, b, al, bl, A, Al, B, Bl, pol1, pol2, minus, t, c, blend_x, blend_y, ch, cl, rh, rl, r, multiplicator, correction, xc, yc;
    vuint32m8_t index;
    vfloat32m8_t sign_y;
    vbool4_t masky0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, sign_x;
#ifndef __FAST_MATH__
    vbool4_t mask_inf_x, mask_inf_y, mask_nan;
    vbool4_t mask_small, mask_big, magic_mask, masky_uf, maskx_uf;
    vint32m8_t p_diff;
#endif
    minus = __riscv_vfmv_v_f_f32m8(f_borders1[0], vl);

    sign_x = __riscv_vmsgeu_vv_u32m8_b4(__riscv_vreinterpret_v_f32m8_u32m8(x), __riscv_vreinterpret_v_f32m8_u32m8(minus), vl);
    masky0 = __riscv_vmfeq_vv_f32m8_b4(minus, y, vl);
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmor_mm_b4(__riscv_vmfne_vv_f32m8_b4(x, x, vl), __riscv_vmfne_vv_f32m8_b4(y, y, vl), vl);
#endif
    sign_y = __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vand_vv_u32m8(__riscv_vreinterpret_v_f32m8_u32m8(minus), __riscv_vreinterpret_v_f32m8_u32m8(y), vl));
    x = __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vand_vv_u32m8(__riscv_vnot_v_u32m8(__riscv_vreinterpret_v_f32m8_u32m8(minus), vl), __riscv_vreinterpret_v_f32m8_u32m8(x), vl));
    y = __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vand_vv_u32m8(__riscv_vnot_v_u32m8(__riscv_vreinterpret_v_f32m8_u32m8(minus), vl), __riscv_vreinterpret_v_f32m8_u32m8(y), vl));
#ifndef __FAST_MATH__
    mask_inf_y = __riscv_vmfeq_vf_f32m8_b4(y, f_magic_consts1[0], vl);
    mask_inf_x = __riscv_vmfeq_vf_f32m8_b4(x, f_magic_consts1[0], vl);
    y = __riscv_vfmerge_vfm_f32m8(y, f_borders2[0], mask_inf_y, vl);
    x = mask_null_f32m8(mask_inf_y, x, vl);

    y = mask_null_f32m8(__riscv_vmor_mm_b4(__riscv_vmandn_mm_b4(mask_inf_x, mask_inf_y, vl), mask_nan, vl), y, vl);
    x = __riscv_vfmerge_vfm_f32m8(x, f_borders2[0],  __riscv_vmor_mm_b4(__riscv_vmor_mm_b4(mask_inf_x, masky0, vl), mask_nan, vl), vl);
#else
    x = __riscv_vfmerge_vfm_f32m8(x, f_borders2[0], masky0, vl);
#endif

    multiplicator = __riscv_vreinterpret_v_i32m8_f32m8(
			__riscv_vsll_vx_i32m8(
				__riscv_vsub_vv_i32m8(
					__riscv_vmv_v_x_i32m8(f_magic_consts2[2], vl), 
					__riscv_vreinterpret_v_u32m8_i32m8(__riscv_vsrl_vx_u32m8(
						__riscv_vadd_vv_u32m8(
								      __riscv_vand_vx_u32m8(__riscv_vreinterpret_v_f32m8_u32m8(y), 0x7f800000, vl), 
								      __riscv_vreinterpret_v_f32m8_u32m8(x), 
								      vl), 
						24, vl)), 
					vl),
				23, vl)
			);
#ifndef __FAST_MATH__
    magic_mask = __riscv_vmandn_mm_b4(__riscv_vmandn_mm_b4(sign_x, mask_nan, vl), mask_inf_x, vl);
#endif

    multiplicator = __riscv_vfmerge_vfm_f32m8(multiplicator, f_magic_consts1[1], __riscv_vmfle_vv_f32m8_b4(multiplicator, minus, vl), vl);
#ifndef __FAST_MATH__
    p_diff = __riscv_vsra_vx_i32m8(__riscv_vsub_vv_i32m8(__riscv_vreinterpret_v_f32m8_i32m8(y), __riscv_vreinterpret_v_f32m8_i32m8(x), vl), 23, vl);
#endif

    x = __riscv_vfmul_vv_f32m8(x, multiplicator, vl);
    y = __riscv_vfmul_vv_f32m8(y, multiplicator, vl);

#ifndef __FAST_MATH__
    maskx_uf = __riscv_vmflt_vf_f32m8_b4(x, 0x1p-122f, vl);
    masky_uf = __riscv_vmflt_vf_f32m8_b4(y, 0x1p-122f, vl);

    xc = mask_null_f32m8(maskx_uf, x, vl);
    magic_mask = __riscv_vmandn_mm_b4(magic_mask, masky0, vl);
    yc = mask_null_f32m8(masky_uf, y, vl);
    mask_small = __riscv_vmsgt_vx_i32m8_b4(p_diff, -0x20, vl);
    magic_mask =  __riscv_vmandn_mm_b4(magic_mask, mask_small, vl);
#else
    xc = x;
    yc = y;
#endif
    mask1 = __riscv_vmflt_vv_f32m8_b4(__riscv_vfmul_vf_f32m8(xc, f_borders1[1], vl), y, vl);
    mask2 = __riscv_vmflt_vv_f32m8_b4(__riscv_vfmul_vf_f32m8(xc, f_borders1[2], vl), y, vl);
    mask3 = __riscv_vmflt_vv_f32m8_b4(__riscv_vfmul_vf_f32m8(xc, f_borders1[3], vl), y, vl);
    mask4 = __riscv_vmflt_vv_f32m8_b4(x, y, vl);
    mask5 = __riscv_vmfle_vv_f32m8_b4(x, __riscv_vfmul_vf_f32m8(yc, f_borders2[1], vl), vl);
    mask6 = __riscv_vmfle_vv_f32m8_b4(x, __riscv_vfmul_vf_f32m8(yc, f_borders2[2], vl), vl);
    mask7 = __riscv_vmfle_vv_f32m8_b4(x, __riscv_vfmul_vf_f32m8(yc, f_borders2[3], vl), vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmsge_vx_i32m8_b4(p_diff, 0x3b, vl);
#endif
    index = __riscv_vmv_v_x_u32m8(0, vl);
#ifndef __FAST_MATH__
    correction = __riscv_vfmerge_vfm_f32m8(__riscv_vfmv_v_f_f32m8(0x1p32f, vl), 0x1p-60, mask_big, vl);
#endif
    index = __riscv_vadd_vx_u32m8_mu(mask1, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m8_mu(mask2, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m8_mu(mask3, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m8_mu(mask4, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m8_mu(mask5, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m8_mu(mask6, index, index, 4, vl);
    index = __riscv_vadd_vx_u32m8_mu(mask7, index, index, 4, vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmul_vv_f32m8_mu(__riscv_vmor_mm_b4(magic_mask, mask_big, vl), x, y, correction, vl);
#endif

    A = mask_null_f32m8(mask7, y, vl);
    k = __riscv_vloxei32_v_f32m8(f_k, index, vl);

    c = __riscv_vfmul_vv_f32m8(k, x, vl);
    al = __riscv_vfmsub_vv_f32m8(k, x, c, vl);
    a = __riscv_vfsub_vv_f32m8(A, c, vl);
    mask7 = __riscv_vmnot_m_b4(mask7, vl);
    Al = __riscv_vfsub_vv_f32m8(a, A, vl);
    al = __riscv_vfsub_vv_f32m8(__riscv_vfsub_vv_f32m8(__riscv_vfsub_vv_f32m8(A, __riscv_vfsub_vv_f32m8(a, Al, vl), vl), __riscv_vfadd_vv_f32m8(Al, c, vl), vl), al, vl);

    b = __riscv_vfmadd_vv_f32m8_mu(mask7, y, k, x, vl);
    Bl = __riscv_vfsub_vv_f32m8_mu(mask7, minus, b, x, vl);
    bl = __riscv_vfadd_vv_f32m8_mu(mask7, minus, __riscv_vfsub_vv_f32m8_mu(mask7, minus, x, __riscv_vfsub_vv_f32m8(b, Bl, vl), vl), __riscv_vfmsac_vv_f32m8(Bl, k, y, vl), vl);

    ch =  __riscv_vfdiv_vv_f32m8(a, b, vl);
    rh = __riscv_vfnmsub_vv_f32m8(ch, b, a, vl);
    rl = __riscv_vfnmsub_vv_f32m8(ch, bl, al, vl);

#ifndef __FAST_MATH__
    r = __riscv_vfadd_vv_f32m8_mu(mask_small, minus, rh, rl, vl);
#else
    r = __riscv_vfadd_vv_f32m8(rh, rl, vl);
#endif

    r =  __riscv_vfmul_vf_f32m8(r, f_magic_consts1[2], vl);

    cl = __riscv_vfdiv_vv_f32m8(r, b, vl);

    w = __riscv_vfmadd_vf_f32m8(ch, f_magic_consts1[2], cl, vl);
    w_l = __riscv_vfadd_vv_f32m8(__riscv_vfmsub_vf_f32m8(ch, f_magic_consts1[2], w, vl), cl, vl);
    wc = __riscv_vfmadd_vf_f32m8(w, f_magic_consts1[3], __riscv_vfmv_v_f_f32m8(0x1p-40f, vl), vl);

#ifndef __FAST_MATH__
    index = __riscv_vadd_vx_u32m8_mu(__riscv_vmandn_mm_b4(sign_x, mask_nan, vl), index, index, f_magic_consts2[1], vl);
#else
    index = __riscv_vadd_vx_u32m8_mu(sign_x, index, index, f_magic_consts2[1], vl);
#endif

    a = __riscv_vloxei32_v_f32m8(f_table_hi, index, vl);
    b = __riscv_vloxei32_v_f32m8(f_table_lo, index, vl);

    y = __riscv_vfmadd_vv_f32m8(wc, wc, __riscv_vfmv_v_f_f32m8(0x1p-40f, vl), vl);

    z = __riscv_vfmul_vv_f32m8(y, y, vl);

    pol1 = __riscv_vfmadd_vf_f32m8(z, f_pol_coef[2], __riscv_vfmv_v_f_f32m8(f_pol_coef[0], vl), vl);
    pol2 = __riscv_vfmul_vf_f32m8(z, f_pol_coef[1], vl);

    w  =  __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vxor_vv_u32m8_mu(sign_x, __riscv_vreinterpret_v_f32m8_u32m8(w),   __riscv_vreinterpret_v_f32m8_u32m8(minus), __riscv_vreinterpret_v_f32m8_u32m8(w),   vl));
    w_l = __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vxor_vv_u32m8_mu(sign_x, __riscv_vreinterpret_v_f32m8_u32m8(w_l), __riscv_vreinterpret_v_f32m8_u32m8(minus), __riscv_vreinterpret_v_f32m8_u32m8(w_l), vl));

    x = __riscv_vfmadd_vf_f32m8(w, f_magic_consts1[3], a, vl);
    t = __riscv_vfsub_vv_f32m8(x, a, vl);
    w_l = __riscv_vfadd_vv_f32m8(__riscv_vfnmsub_vf_f32m8(t, f_magic_consts1[2], w, vl), w_l, vl);

    x = __riscv_vfmadd_vf_f32m8(__riscv_vfadd_vv_f32m8(__riscv_vfsub_vf_f32m8(__riscv_vfmadd_vv_f32m8(__riscv_vfmadd_vv_f32m8(pol1, y, pol2, vl), w, b, vl), f_table_lo[0], vl),  w_l, vl), f_magic_consts1[3], x, vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmacc_vf_f32m8_mu(mask_inf_x, x, 0x1p-126f, __riscv_vfmv_v_f_f32m8(0x1p-126f, vl), vl);
    x = __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vor_vx_u32m8_mu(mask_nan, __riscv_vreinterpret_v_f32m8_u32m8(x), __riscv_vreinterpret_v_f32m8_u32m8(x), f_magic_consts2[3], vl));
    x = __riscv_vfmadd_vf_f32m8(x, 0x1p-30f, x, vl);
#endif

    return __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vor_vv_u32m8(__riscv_vreinterpret_v_f32m8_u32m8(sign_y), __riscv_vreinterpret_v_f32m8_u32m8(x), vl));
}

#ifdef __riscv_zvfh

static const _Float16 h_borders1[4] = {-0x0p0f16, 0x1p-3f16,  0x1.8p-2f16, 0x1.5p-1f16};
static const _Float16 h_borders2[4] = { 0x1p0f16, 0x1.47cp-1f16, 0x1.744p-2f16,  0x1.f8p-4f16};

static const _Float16 h_magic_consts1[4] = { static_cast<_Float16>(0x1p16), 0x0.8p-14f16, 0x1p11f16, 0x1p-11f16};

static const int16_t h_magic_consts2[4] = { 1, 16, static_cast<int16_t>(0x1e), static_cast<int16_t>(0x7e00)};

static const _Float16   h_table_hi[16] = {0.0f16,        0x1.f5cp8f16,  0x1.dacp9f16,  0x1.5d4p10f16,  0x1.cacp10f16,  0x1.1b8p11f16,  0x1.538p11f16,  0x1.92p11f16,
					  0x1.92p12f16,  0x1.72cp12f16, 0x1.56cp12f16, 0x1.3acp12f16,  0x1.1f8p12f16,  0x1.048p12f16,  0x1.d0cp11f16,  0x1.924p11f16},
			h_table_lo[16] = {-0x1p-14f16,  -0x1.14p6f16,   0x1.9cp6f16,   0x1.898p9f16,   0x1.f14p7f16,  -0x1.1e8p10f16, -0x1.738p10f16,  0x1.fb4p10f16,
					  0x1.fb4p11f16, 0x1.0fcp9f16,  0x1.b9cp9f16,  0x1.32p10f16,  -0x1.23cp11f16, -0x1.758p11f16,  0x1.6ap10f16,  -0x1.024p11f16},
			h_k[8]         = {0.0f16,        0x1p-2f16,     0x1p-1f16,     0x1.ap-1f16,    0x1.4p0f16,     2.0f16,         4.0f16,         1.0f16};

inline vfloat16m1_t mask_null_f16m1(vbool16_t mask, vfloat16m1_t arg, size_t vl) {
  return __riscv_vreinterpret_v_i16m1_f16m1(__riscv_vxor_vv_i16m1_mu(mask, __riscv_vreinterpret_v_f16m1_i16m1(arg), __riscv_vreinterpret_v_f16m1_i16m1(arg), __riscv_vreinterpret_v_f16m1_i16m1(arg), vl));
}
inline vfloat16m2_t mask_null_f16m2(vbool8_t mask, vfloat16m2_t arg, size_t vl) {
  return __riscv_vreinterpret_v_i16m2_f16m2(__riscv_vxor_vv_i16m2_mu(mask, __riscv_vreinterpret_v_f16m2_i16m2(arg), __riscv_vreinterpret_v_f16m2_i16m2(arg), __riscv_vreinterpret_v_f16m2_i16m2(arg), vl));
}
inline vfloat16m4_t mask_null_f16m4(vbool4_t mask, vfloat16m4_t arg, size_t vl) {
  return __riscv_vreinterpret_v_i16m4_f16m4(__riscv_vxor_vv_i16m4_mu(mask, __riscv_vreinterpret_v_f16m4_i16m4(arg), __riscv_vreinterpret_v_f16m4_i16m4(arg), __riscv_vreinterpret_v_f16m4_i16m4(arg), vl));
}
inline vfloat16m8_t mask_null_f16m8(vbool2_t mask, vfloat16m8_t arg, size_t vl) {
  return __riscv_vreinterpret_v_i16m8_f16m8(__riscv_vxor_vv_i16m8_mu(mask, __riscv_vreinterpret_v_f16m8_i16m8(arg), __riscv_vreinterpret_v_f16m8_i16m8(arg), __riscv_vreinterpret_v_f16m8_i16m8(arg), vl));
}

vfloat16m1_t __riscv_vatan2_f16m1(vfloat16m1_t y, vfloat16m1_t x, size_t vl) {
    vfloat16m1_t w, xc, yc, wc, w_l, z, k, a, b, al, bl, A, Al, B, Bl, pol, minus, t, c, blend_x, blend_y, ch, cl, rh, rl, r, multiplicator;
    vuint16m1_t index;
    vfloat16m1_t sign_y;
    vbool16_t masky0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, sign_x;
#ifndef __FAST_MATH__
    vbool16_t mask_inf_x, mask_inf_y, mask_nan;
    vbool16_t lo_w_mask, hi_w_mask, mask_small, mask_big, maskx_uf, masky_uf;
    vint16m1_t p_diff;
#endif
    minus = __riscv_vfmv_v_f_f16m1(h_borders1[0], vl);

    sign_x = __riscv_vmsgeu_vv_u16m1_b16(__riscv_vreinterpret_v_f16m1_u16m1(x), __riscv_vreinterpret_v_f16m1_u16m1(minus), vl);
    masky0 = __riscv_vmfeq_vv_f16m1_b16(minus, y, vl);
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmor_mm_b16(__riscv_vmfne_vv_f16m1_b16(x, x, vl), __riscv_vmfne_vv_f16m1_b16(y, y, vl), vl);
#endif
    sign_y = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vv_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(minus), __riscv_vreinterpret_v_f16m1_u16m1(y), vl));
    x = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vv_u16m1(__riscv_vnot_v_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(minus), vl), __riscv_vreinterpret_v_f16m1_u16m1(x), vl));
    y = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vv_u16m1(__riscv_vnot_v_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(minus), vl), __riscv_vreinterpret_v_f16m1_u16m1(y), vl));
#ifndef __FAST_MATH__
    mask_inf_y = __riscv_vmfeq_vf_f16m1_b16(y, h_magic_consts1[0], vl);
    mask_inf_x = __riscv_vmfeq_vf_f16m1_b16(x, h_magic_consts1[0], vl);
    y = __riscv_vfmerge_vfm_f16m1(y, h_borders2[0], mask_inf_y, vl);
    x = mask_null_f16m1(mask_inf_y, x, vl);

    y = mask_null_f16m1(__riscv_vmor_mm_b16(__riscv_vmandn_mm_b16(mask_inf_x, mask_inf_y, vl), mask_nan, vl), y, vl);
    x = __riscv_vfmerge_vfm_f16m1(x, h_borders2[0],  __riscv_vmor_mm_b16(__riscv_vmor_mm_b16(mask_inf_x, masky0, vl), mask_nan, vl), vl);
#else
    x = __riscv_vfmerge_vfm_f16m1(x, h_borders2[0], masky0, vl);
#endif

    multiplicator = __riscv_vreinterpret_v_i16m1_f16m1(
			__riscv_vsll_vx_i16m1(
				__riscv_vsub_vv_i16m1(
					__riscv_vmv_v_x_i16m1(h_magic_consts2[2], vl), 
					__riscv_vreinterpret_v_u16m1_i16m1(__riscv_vsrl_vx_u16m1(
						__riscv_vadd_vv_u16m1(
								      __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(y), 0x7c00, vl), 
								      __riscv_vreinterpret_v_f16m1_u16m1(x), 
								      vl), 
						11, vl)), 
					vl),
				10, vl)
			);
#ifndef __FAST_MATH__
    hi_w_mask = __riscv_vmandn_mm_b16(__riscv_vmandn_mm_b16(sign_x, mask_nan, vl), mask_inf_x, vl);
#endif

    multiplicator = __riscv_vfmerge_vfm_f16m1(multiplicator, h_magic_consts1[1], __riscv_vmfle_vv_f16m1_b16(multiplicator, minus, vl), vl);

    x = __riscv_vfmul_vv_f16m1(x, multiplicator, vl);
    y = __riscv_vfmul_vv_f16m1(y, multiplicator, vl);

#ifndef __FAST_MATH__
    p_diff = __riscv_vsra_vx_i16m1(__riscv_vsub_vv_i16m1(__riscv_vreinterpret_v_f16m1_i16m1(y), __riscv_vreinterpret_v_f16m1_i16m1(x), vl), 10, vl);

    maskx_uf = __riscv_vmflt_vf_f16m1_b16(x, 0x1p-10f16, vl);
    masky_uf = __riscv_vmflt_vf_f16m1_b16(y, 0x1p-10f16, vl);
    mask_small = __riscv_vmsle_vx_i16m1_b16(p_diff, -0xb, vl);
    mask_big = __riscv_vmsge_vx_i16m1_b16(p_diff, 0x12, vl);
    xc = mask_null_f16m1(maskx_uf, x, vl);
    yc = mask_null_f16m1(masky_uf, y, vl);

    hi_w_mask = __riscv_vmand_mm_b16(hi_w_mask, __riscv_vmor_mm_b16(mask_small, mask_big, vl), vl);
#else
    xc = x;
    yc = y;
#endif

    mask1 = __riscv_vmfle_vv_f16m1_b16(__riscv_vfmul_vf_f16m1(xc, h_borders1[1], vl), y, vl);
    mask2 = __riscv_vmflt_vv_f16m1_b16(__riscv_vfmul_vf_f16m1(xc, h_borders1[2], vl), y, vl);
    mask3 = __riscv_vmflt_vv_f16m1_b16(__riscv_vfmul_vf_f16m1(xc, h_borders1[3], vl), y, vl);
    mask4 = __riscv_vmflt_vv_f16m1_b16(x, y, vl);
    mask5 = __riscv_vmfle_vv_f16m1_b16(x, __riscv_vfmul_vf_f16m1(yc, h_borders2[1], vl), vl);
    mask6 = __riscv_vmfle_vv_f16m1_b16(x, __riscv_vfmul_vf_f16m1(yc, h_borders2[2], vl), vl);
    mask7 = __riscv_vmfle_vv_f16m1_b16(x, __riscv_vfmul_vf_f16m1(yc, h_borders2[3], vl), vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmsge_vx_i16m1_b16(p_diff, 0xb, vl);
#endif

    index = __riscv_vmv_v_x_u16m1(0, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask1, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask2, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask3, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask4, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask5, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask6, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m1_mu(mask7, index, index, 2, vl);

#ifndef __FAST_MATH__
    lo_w_mask = __riscv_vmor_mm_b16(mask_small, mask_big, vl);
    hi_w_mask = __riscv_vmor_mm_b16(hi_w_mask, __riscv_vmandn_mm_b16(mask_big, sign_x, vl), vl);
#endif

    A = mask_null_f16m1(mask7, y, vl);
    k = __riscv_vloxei16_v_f16m1(h_k, index, vl);

    c = __riscv_vfmul_vv_f16m1(k, x, vl);
    al = __riscv_vfmsub_vv_f16m1(k, x, c, vl);
    a = __riscv_vfsub_vv_f16m1(A, c, vl);
    mask7 = __riscv_vmnot_m_b16(mask7, vl);
    Al = __riscv_vfsub_vv_f16m1(a, A, vl);
    al = __riscv_vfsub_vv_f16m1(__riscv_vfsub_vv_f16m1(__riscv_vfsub_vv_f16m1(A, __riscv_vfsub_vv_f16m1(a, Al, vl), vl), __riscv_vfadd_vv_f16m1(Al, c, vl), vl), al, vl);
    a = __riscv_vfmul_vf_f16m1(a, h_magic_consts1[2], vl);
#ifndef __FAST_MATH__
    a = mask_null_f16m1(hi_w_mask, a, vl);
#endif
    al = __riscv_vfmul_vf_f16m1(al, h_magic_consts1[2], vl);

    b = __riscv_vfmadd_vv_f16m1_mu(mask7, y, k, x, vl);
    Bl = __riscv_vfsub_vv_f16m1_mu(mask7, minus, b, x, vl);
    bl = __riscv_vfadd_vv_f16m1_mu(mask7, minus, __riscv_vfsub_vv_f16m1_mu(mask7, minus, x, __riscv_vfsub_vv_f16m1(b, Bl, vl), vl), __riscv_vfmsac_vv_f16m1(Bl, k, y, vl), vl);
    bl = __riscv_vfmul_vf_f16m1(bl, h_magic_consts1[2], vl);

    ch =  __riscv_vfdiv_vv_f16m1(a, b, vl);
    al = __riscv_vfmul_vf_f16m1(al, h_magic_consts1[2], vl);

    rh = __riscv_vfnmsub_vv_f16m1(ch, b, a, vl);
    rl = __riscv_vfnmsub_vv_f16m1(ch, bl, al, vl);
    r =  __riscv_vfmadd_vf_f16m1(rh, h_magic_consts1[2], rl, vl);
#ifndef __FAST_MATH__
    r = mask_null_f16m1(lo_w_mask, r, vl);
#endif
    cl = __riscv_vfdiv_vv_f16m1(r, b, vl);

    w = __riscv_vfmadd_vf_f16m1(cl, h_magic_consts1[3], ch, vl);
    w_l = __riscv_vfmadd_vf_f16m1(__riscv_vfsub_vv_f16m1(ch, w, vl), h_magic_consts1[2], cl, vl);

#ifndef __FAST_MATH__
    index = __riscv_vadd_vx_u16m1_mu(__riscv_vmandn_mm_b16(sign_x, mask_nan, vl), index, index, h_magic_consts2[1], vl);
#else
    index = __riscv_vadd_vx_u16m1_mu(sign_x, index, index, h_magic_consts2[1], vl);
#endif

    a = __riscv_vloxei16_v_f16m1(h_table_hi, index, vl);
    b = __riscv_vloxei16_v_f16m1(h_table_lo, index, vl);

#ifndef __FAST_MATH__
    wc = mask_null_f16m1(lo_w_mask, w, vl);
#else
    wc = w;
#endif

    pol = __riscv_vfmul_vv_f16m1(wc, wc, vl);
    pol = __riscv_vfmadd_vf_f16m1(pol, -0x1.52cp-13f16, __riscv_vfmv_v_f_f16m1(-0x1p-14f16, vl), vl);

    w  =  __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vxor_vv_u16m1_mu(sign_x, __riscv_vreinterpret_v_f16m1_u16m1(w),   __riscv_vreinterpret_v_f16m1_u16m1(minus), __riscv_vreinterpret_v_f16m1_u16m1(w),   vl));
    w_l = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vxor_vv_u16m1_mu(sign_x, __riscv_vreinterpret_v_f16m1_u16m1(w_l), __riscv_vreinterpret_v_f16m1_u16m1(minus), __riscv_vreinterpret_v_f16m1_u16m1(w_l), vl));

    x = __riscv_vfadd_vv_f16m1(a, w, vl);
    t = __riscv_vfsub_vv_f16m1(x, a, vl);
    w_l = __riscv_vfmadd_vf_f16m1(__riscv_vfsub_vv_f16m1(w, t, vl), h_magic_consts1[2], w_l, vl);
    x = __riscv_vfmadd_vf_f16m1(__riscv_vfsub_vf_f16m1(__riscv_vfadd_vv_f16m1(__riscv_vfmadd_vv_f16m1(pol, w, b, vl), w_l, vl), h_table_lo[0], vl), h_magic_consts1[3], x, vl);
    x = __riscv_vfmul_vf_f16m1(x, h_magic_consts1[3], vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmacc_vf_f16m1_mu(mask_inf_x, x, 0x1p-14f16, __riscv_vfmv_v_f_f16m1(0x1p-14f16, vl), vl);
    x = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vor_vx_u16m1_mu(mask_nan, __riscv_vreinterpret_v_f16m1_u16m1(x), __riscv_vreinterpret_v_f16m1_u16m1(x), h_magic_consts2[3], vl));
    x = __riscv_vfmadd_vf_f16m1(x, 0x1p-14f16, x, vl);
#endif

    return __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vor_vv_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(sign_y), __riscv_vreinterpret_v_f16m1_u16m1(x), vl));
}




vfloat16m2_t __riscv_vatan2_f16m2(vfloat16m2_t y, vfloat16m2_t x, size_t vl) {
    vfloat16m2_t w, xc, yc, wc, w_l, z, k, a, b, al, bl, A, Al, B, Bl, pol, minus, t, c, blend_x, blend_y, ch, cl, rh, rl, r, multiplicator;
    vuint16m2_t index;
    vfloat16m2_t sign_y;
    vbool8_t masky0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, sign_x;
#ifndef __FAST_MATH__
    vbool8_t mask_inf_x, mask_inf_y, mask_nan;
    vbool8_t lo_w_mask, hi_w_mask, mask_small, mask_big, maskx_uf, masky_uf;
    vint16m2_t p_diff;
#endif
    minus = __riscv_vfmv_v_f_f16m2(h_borders1[0], vl);

    sign_x = __riscv_vmsgeu_vv_u16m2_b8(__riscv_vreinterpret_v_f16m2_u16m2(x), __riscv_vreinterpret_v_f16m2_u16m2(minus), vl);
    masky0 = __riscv_vmfeq_vv_f16m2_b8(minus, y, vl);
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmor_mm_b8(__riscv_vmfne_vv_f16m2_b8(x, x, vl), __riscv_vmfne_vv_f16m2_b8(y, y, vl), vl);
#endif
    sign_y = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vv_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(minus), __riscv_vreinterpret_v_f16m2_u16m2(y), vl));
    x = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vv_u16m2(__riscv_vnot_v_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(minus), vl), __riscv_vreinterpret_v_f16m2_u16m2(x), vl));
    y = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vv_u16m2(__riscv_vnot_v_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(minus), vl), __riscv_vreinterpret_v_f16m2_u16m2(y), vl));
#ifndef __FAST_MATH__
    mask_inf_y = __riscv_vmfeq_vf_f16m2_b8(y, h_magic_consts1[0], vl);
    mask_inf_x = __riscv_vmfeq_vf_f16m2_b8(x, h_magic_consts1[0], vl);
    y = __riscv_vfmerge_vfm_f16m2(y, h_borders2[0], mask_inf_y, vl);
    x = mask_null_f16m2(mask_inf_y, x, vl);

    y = mask_null_f16m2(__riscv_vmor_mm_b8(__riscv_vmandn_mm_b8(mask_inf_x, mask_inf_y, vl), mask_nan, vl), y, vl);
    x = __riscv_vfmerge_vfm_f16m2(x, h_borders2[0],  __riscv_vmor_mm_b8(__riscv_vmor_mm_b8(mask_inf_x, masky0, vl), mask_nan, vl), vl);
#else
    x = __riscv_vfmerge_vfm_f16m2(x, h_borders2[0], masky0, vl);
#endif

    multiplicator = __riscv_vreinterpret_v_i16m2_f16m2(
			__riscv_vsll_vx_i16m2(
				__riscv_vsub_vv_i16m2(
					__riscv_vmv_v_x_i16m2(h_magic_consts2[2], vl), 
					__riscv_vreinterpret_v_u16m2_i16m2(__riscv_vsrl_vx_u16m2(
						__riscv_vadd_vv_u16m2(
								      __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(y), 0x7c00, vl), 
								      __riscv_vreinterpret_v_f16m2_u16m2(x), 
								      vl), 
						11, vl)), 
					vl),
				10, vl)
			);
#ifndef __FAST_MATH__
    hi_w_mask = __riscv_vmandn_mm_b8(__riscv_vmandn_mm_b8(sign_x, mask_nan, vl), mask_inf_x, vl);
#endif

    multiplicator = __riscv_vfmerge_vfm_f16m2(multiplicator, h_magic_consts1[1], __riscv_vmfle_vv_f16m2_b8(multiplicator, minus, vl), vl);

    x = __riscv_vfmul_vv_f16m2(x, multiplicator, vl);
    y = __riscv_vfmul_vv_f16m2(y, multiplicator, vl);

#ifndef __FAST_MATH__
    p_diff = __riscv_vsra_vx_i16m2(__riscv_vsub_vv_i16m2(__riscv_vreinterpret_v_f16m2_i16m2(y), __riscv_vreinterpret_v_f16m2_i16m2(x), vl), 10, vl);

    maskx_uf = __riscv_vmflt_vf_f16m2_b8(x, 0x1p-10f16, vl);
    masky_uf = __riscv_vmflt_vf_f16m2_b8(y, 0x1p-10f16, vl);
    mask_small = __riscv_vmsle_vx_i16m2_b8(p_diff, -0xb, vl);
    mask_big = __riscv_vmsge_vx_i16m2_b8(p_diff, 0x12, vl);
    xc = mask_null_f16m2(maskx_uf, x, vl);
    yc = mask_null_f16m2(masky_uf, y, vl);

    hi_w_mask = __riscv_vmand_mm_b8(hi_w_mask, __riscv_vmor_mm_b8(mask_small, mask_big, vl), vl);
#else
    xc = x;
    yc = y;
#endif

    mask1 = __riscv_vmfle_vv_f16m2_b8(__riscv_vfmul_vf_f16m2(xc, h_borders1[1], vl), y, vl);
    mask2 = __riscv_vmflt_vv_f16m2_b8(__riscv_vfmul_vf_f16m2(xc, h_borders1[2], vl), y, vl);
    mask3 = __riscv_vmflt_vv_f16m2_b8(__riscv_vfmul_vf_f16m2(xc, h_borders1[3], vl), y, vl);
    mask4 = __riscv_vmflt_vv_f16m2_b8(x, y, vl);
    mask5 = __riscv_vmfle_vv_f16m2_b8(x, __riscv_vfmul_vf_f16m2(yc, h_borders2[1], vl), vl);
    mask6 = __riscv_vmfle_vv_f16m2_b8(x, __riscv_vfmul_vf_f16m2(yc, h_borders2[2], vl), vl);
    mask7 = __riscv_vmfle_vv_f16m2_b8(x, __riscv_vfmul_vf_f16m2(yc, h_borders2[3], vl), vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmsge_vx_i16m2_b8(p_diff, 0xb, vl);
#endif

    index = __riscv_vmv_v_x_u16m2(0, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask1, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask2, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask3, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask4, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask5, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask6, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m2_mu(mask7, index, index, 2, vl);

#ifndef __FAST_MATH__
    lo_w_mask = __riscv_vmor_mm_b8(mask_small, mask_big, vl);
    hi_w_mask = __riscv_vmor_mm_b8(hi_w_mask, __riscv_vmandn_mm_b8(mask_big, sign_x, vl), vl);
#endif

    A = mask_null_f16m2(mask7, y, vl);
    k = __riscv_vloxei16_v_f16m2(h_k, index, vl);

    c = __riscv_vfmul_vv_f16m2(k, x, vl);
    al = __riscv_vfmsub_vv_f16m2(k, x, c, vl);
    a = __riscv_vfsub_vv_f16m2(A, c, vl);
    mask7 = __riscv_vmnot_m_b8(mask7, vl);
    Al = __riscv_vfsub_vv_f16m2(a, A, vl);
    al = __riscv_vfsub_vv_f16m2(__riscv_vfsub_vv_f16m2(__riscv_vfsub_vv_f16m2(A, __riscv_vfsub_vv_f16m2(a, Al, vl), vl), __riscv_vfadd_vv_f16m2(Al, c, vl), vl), al, vl);
    a = __riscv_vfmul_vf_f16m2(a, h_magic_consts1[2], vl);
#ifndef __FAST_MATH__
    a = mask_null_f16m2(hi_w_mask, a, vl);
#endif
    al = __riscv_vfmul_vf_f16m2(al, h_magic_consts1[2], vl);

    b = __riscv_vfmadd_vv_f16m2_mu(mask7, y, k, x, vl);
    Bl = __riscv_vfsub_vv_f16m2_mu(mask7, minus, b, x, vl);
    bl = __riscv_vfadd_vv_f16m2_mu(mask7, minus, __riscv_vfsub_vv_f16m2_mu(mask7, minus, x, __riscv_vfsub_vv_f16m2(b, Bl, vl), vl), __riscv_vfmsac_vv_f16m2(Bl, k, y, vl), vl);
    bl = __riscv_vfmul_vf_f16m2(bl, h_magic_consts1[2], vl);

    ch =  __riscv_vfdiv_vv_f16m2(a, b, vl);
    al = __riscv_vfmul_vf_f16m2(al, h_magic_consts1[2], vl);

    rh = __riscv_vfnmsub_vv_f16m2(ch, b, a, vl);
    rl = __riscv_vfnmsub_vv_f16m2(ch, bl, al, vl);
    r =  __riscv_vfmadd_vf_f16m2(rh, h_magic_consts1[2], rl, vl);
#ifndef __FAST_MATH__
    r = mask_null_f16m2(lo_w_mask, r, vl);
#endif
    cl = __riscv_vfdiv_vv_f16m2(r, b, vl);

    w = __riscv_vfmadd_vf_f16m2(cl, h_magic_consts1[3], ch, vl);
    w_l = __riscv_vfmadd_vf_f16m2(__riscv_vfsub_vv_f16m2(ch, w, vl), h_magic_consts1[2], cl, vl);

#ifndef __FAST_MATH__
    index = __riscv_vadd_vx_u16m2_mu(__riscv_vmandn_mm_b8(sign_x, mask_nan, vl), index, index, h_magic_consts2[1], vl);
#else
    index = __riscv_vadd_vx_u16m2_mu(sign_x, index, index, h_magic_consts2[1], vl);
#endif

    a = __riscv_vloxei16_v_f16m2(h_table_hi, index, vl);
    b = __riscv_vloxei16_v_f16m2(h_table_lo, index, vl);

#ifndef __FAST_MATH__
    wc = mask_null_f16m2(lo_w_mask, w, vl);
#else
    wc = w;
#endif

    pol = __riscv_vfmul_vv_f16m2(wc, wc, vl);
    pol = __riscv_vfmadd_vf_f16m2(pol, -0x1.52cp-13f16, __riscv_vfmv_v_f_f16m2(-0x1p-14f16, vl), vl);

    w  =  __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vxor_vv_u16m2_mu(sign_x, __riscv_vreinterpret_v_f16m2_u16m2(w),   __riscv_vreinterpret_v_f16m2_u16m2(minus), __riscv_vreinterpret_v_f16m2_u16m2(w),   vl));
    w_l = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vxor_vv_u16m2_mu(sign_x, __riscv_vreinterpret_v_f16m2_u16m2(w_l), __riscv_vreinterpret_v_f16m2_u16m2(minus), __riscv_vreinterpret_v_f16m2_u16m2(w_l), vl));

    x = __riscv_vfadd_vv_f16m2(a, w, vl);
    t = __riscv_vfsub_vv_f16m2(x, a, vl);
    w_l = __riscv_vfmadd_vf_f16m2(__riscv_vfsub_vv_f16m2(w, t, vl), h_magic_consts1[2], w_l, vl);
    x = __riscv_vfmadd_vf_f16m2(__riscv_vfsub_vf_f16m2(__riscv_vfadd_vv_f16m2(__riscv_vfmadd_vv_f16m2(pol, w, b, vl), w_l, vl), h_table_lo[0], vl), h_magic_consts1[3], x, vl);
    x = __riscv_vfmul_vf_f16m2(x, h_magic_consts1[3], vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmacc_vf_f16m2_mu(mask_inf_x, x, 0x1p-14f16, __riscv_vfmv_v_f_f16m2(0x1p-14f16, vl), vl);
    x = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vor_vx_u16m2_mu(mask_nan, __riscv_vreinterpret_v_f16m2_u16m2(x), __riscv_vreinterpret_v_f16m2_u16m2(x), h_magic_consts2[3], vl));
    x = __riscv_vfmadd_vf_f16m2(x, 0x1p-14f16, x, vl);
#endif

    return __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vor_vv_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(sign_y), __riscv_vreinterpret_v_f16m2_u16m2(x), vl));
}




vfloat16m4_t __riscv_vatan2_f16m4(vfloat16m4_t y, vfloat16m4_t x, size_t vl) {
    vfloat16m4_t w, xc, yc, wc, w_l, z, k, a, b, al, bl, A, Al, B, Bl, pol, minus, t, c, blend_x, blend_y, ch, cl, rh, rl, r, multiplicator;
    vuint16m4_t index;
    vfloat16m4_t sign_y;
    vbool4_t masky0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, sign_x;
#ifndef __FAST_MATH__
    vbool4_t mask_inf_x, mask_inf_y, mask_nan;
    vbool4_t lo_w_mask, hi_w_mask, mask_small, mask_big, maskx_uf, masky_uf;
    vint16m4_t p_diff;
#endif
    minus = __riscv_vfmv_v_f_f16m4(h_borders1[0], vl);

    sign_x = __riscv_vmsgeu_vv_u16m4_b4(__riscv_vreinterpret_v_f16m4_u16m4(x), __riscv_vreinterpret_v_f16m4_u16m4(minus), vl);
    masky0 = __riscv_vmfeq_vv_f16m4_b4(minus, y, vl);
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmor_mm_b4(__riscv_vmfne_vv_f16m4_b4(x, x, vl), __riscv_vmfne_vv_f16m4_b4(y, y, vl), vl);
#endif
    sign_y = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vv_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(minus), __riscv_vreinterpret_v_f16m4_u16m4(y), vl));
    x = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vv_u16m4(__riscv_vnot_v_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(minus), vl), __riscv_vreinterpret_v_f16m4_u16m4(x), vl));
    y = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vv_u16m4(__riscv_vnot_v_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(minus), vl), __riscv_vreinterpret_v_f16m4_u16m4(y), vl));
#ifndef __FAST_MATH__
    mask_inf_y = __riscv_vmfeq_vf_f16m4_b4(y, h_magic_consts1[0], vl);
    mask_inf_x = __riscv_vmfeq_vf_f16m4_b4(x, h_magic_consts1[0], vl);
    y = __riscv_vfmerge_vfm_f16m4(y, h_borders2[0], mask_inf_y, vl);
    x = mask_null_f16m4(mask_inf_y, x, vl);

    y = mask_null_f16m4(__riscv_vmor_mm_b4(__riscv_vmandn_mm_b4(mask_inf_x, mask_inf_y, vl), mask_nan, vl), y, vl);
    x = __riscv_vfmerge_vfm_f16m4(x, h_borders2[0],  __riscv_vmor_mm_b4(__riscv_vmor_mm_b4(mask_inf_x, masky0, vl), mask_nan, vl), vl);
#else
    x = __riscv_vfmerge_vfm_f16m4(x, h_borders2[0], masky0, vl);
#endif

    multiplicator = __riscv_vreinterpret_v_i16m4_f16m4(
			__riscv_vsll_vx_i16m4(
				__riscv_vsub_vv_i16m4(
					__riscv_vmv_v_x_i16m4(h_magic_consts2[2], vl), 
					__riscv_vreinterpret_v_u16m4_i16m4(__riscv_vsrl_vx_u16m4(
						__riscv_vadd_vv_u16m4(
								      __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(y), 0x7c00, vl), 
								      __riscv_vreinterpret_v_f16m4_u16m4(x), 
								      vl), 
						11, vl)), 
					vl),
				10, vl)
			);
#ifndef __FAST_MATH__
    hi_w_mask = __riscv_vmandn_mm_b4(__riscv_vmandn_mm_b4(sign_x, mask_nan, vl), mask_inf_x, vl);
#endif

    multiplicator = __riscv_vfmerge_vfm_f16m4(multiplicator, h_magic_consts1[1], __riscv_vmfle_vv_f16m4_b4(multiplicator, minus, vl), vl);

    x = __riscv_vfmul_vv_f16m4(x, multiplicator, vl);
    y = __riscv_vfmul_vv_f16m4(y, multiplicator, vl);

#ifndef __FAST_MATH__
    p_diff = __riscv_vsra_vx_i16m4(__riscv_vsub_vv_i16m4(__riscv_vreinterpret_v_f16m4_i16m4(y), __riscv_vreinterpret_v_f16m4_i16m4(x), vl), 10, vl);

    maskx_uf = __riscv_vmflt_vf_f16m4_b4(x, 0x1p-10f16, vl);
    masky_uf = __riscv_vmflt_vf_f16m4_b4(y, 0x1p-10f16, vl);
    mask_small = __riscv_vmsle_vx_i16m4_b4(p_diff, -0xb, vl);
    mask_big = __riscv_vmsge_vx_i16m4_b4(p_diff, 0x12, vl);
    xc = mask_null_f16m4(maskx_uf, x, vl);
    yc = mask_null_f16m4(masky_uf, y, vl);

    hi_w_mask = __riscv_vmand_mm_b4(hi_w_mask, __riscv_vmor_mm_b4(mask_small, mask_big, vl), vl);
#else
    xc = x;
    yc = y;
#endif

    mask1 = __riscv_vmfle_vv_f16m4_b4(__riscv_vfmul_vf_f16m4(xc, h_borders1[1], vl), y, vl);
    mask2 = __riscv_vmflt_vv_f16m4_b4(__riscv_vfmul_vf_f16m4(xc, h_borders1[2], vl), y, vl);
    mask3 = __riscv_vmflt_vv_f16m4_b4(__riscv_vfmul_vf_f16m4(xc, h_borders1[3], vl), y, vl);
    mask4 = __riscv_vmflt_vv_f16m4_b4(x, y, vl);
    mask5 = __riscv_vmfle_vv_f16m4_b4(x, __riscv_vfmul_vf_f16m4(yc, h_borders2[1], vl), vl);
    mask6 = __riscv_vmfle_vv_f16m4_b4(x, __riscv_vfmul_vf_f16m4(yc, h_borders2[2], vl), vl);
    mask7 = __riscv_vmfle_vv_f16m4_b4(x, __riscv_vfmul_vf_f16m4(yc, h_borders2[3], vl), vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmsge_vx_i16m4_b4(p_diff, 0xb, vl);
#endif

    index = __riscv_vmv_v_x_u16m4(0, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask1, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask2, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask3, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask4, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask5, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask6, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m4_mu(mask7, index, index, 2, vl);

#ifndef __FAST_MATH__
    lo_w_mask = __riscv_vmor_mm_b4(mask_small, mask_big, vl);
    hi_w_mask = __riscv_vmor_mm_b4(hi_w_mask, __riscv_vmandn_mm_b4(mask_big, sign_x, vl), vl);
#endif

    A = mask_null_f16m4(mask7, y, vl);
    k = __riscv_vloxei16_v_f16m4(h_k, index, vl);

    c = __riscv_vfmul_vv_f16m4(k, x, vl);
    al = __riscv_vfmsub_vv_f16m4(k, x, c, vl);
    a = __riscv_vfsub_vv_f16m4(A, c, vl);
    mask7 = __riscv_vmnot_m_b4(mask7, vl);
    Al = __riscv_vfsub_vv_f16m4(a, A, vl);
    al = __riscv_vfsub_vv_f16m4(__riscv_vfsub_vv_f16m4(__riscv_vfsub_vv_f16m4(A, __riscv_vfsub_vv_f16m4(a, Al, vl), vl), __riscv_vfadd_vv_f16m4(Al, c, vl), vl), al, vl);
    a = __riscv_vfmul_vf_f16m4(a, h_magic_consts1[2], vl);
#ifndef __FAST_MATH__
    a = mask_null_f16m4(hi_w_mask, a, vl);
#endif
    al = __riscv_vfmul_vf_f16m4(al, h_magic_consts1[2], vl);

    b = __riscv_vfmadd_vv_f16m4_mu(mask7, y, k, x, vl);
    Bl = __riscv_vfsub_vv_f16m4_mu(mask7, minus, b, x, vl);
    bl = __riscv_vfadd_vv_f16m4_mu(mask7, minus, __riscv_vfsub_vv_f16m4_mu(mask7, minus, x, __riscv_vfsub_vv_f16m4(b, Bl, vl), vl), __riscv_vfmsac_vv_f16m4(Bl, k, y, vl), vl);
    bl = __riscv_vfmul_vf_f16m4(bl, h_magic_consts1[2], vl);

    ch =  __riscv_vfdiv_vv_f16m4(a, b, vl);
    al = __riscv_vfmul_vf_f16m4(al, h_magic_consts1[2], vl);

    rh = __riscv_vfnmsub_vv_f16m4(ch, b, a, vl);
    rl = __riscv_vfnmsub_vv_f16m4(ch, bl, al, vl);
    r =  __riscv_vfmadd_vf_f16m4(rh, h_magic_consts1[2], rl, vl);
#ifndef __FAST_MATH__
    r = mask_null_f16m4(lo_w_mask, r, vl);
#endif
    cl = __riscv_vfdiv_vv_f16m4(r, b, vl);

    w = __riscv_vfmadd_vf_f16m4(cl, h_magic_consts1[3], ch, vl);
    w_l = __riscv_vfmadd_vf_f16m4(__riscv_vfsub_vv_f16m4(ch, w, vl), h_magic_consts1[2], cl, vl);

#ifndef __FAST_MATH__
    index = __riscv_vadd_vx_u16m4_mu(__riscv_vmandn_mm_b4(sign_x, mask_nan, vl), index, index, h_magic_consts2[1], vl);
#else
    index = __riscv_vadd_vx_u16m4_mu(sign_x, index, index, h_magic_consts2[1], vl);
#endif

    a = __riscv_vloxei16_v_f16m4(h_table_hi, index, vl);
    b = __riscv_vloxei16_v_f16m4(h_table_lo, index, vl);

#ifndef __FAST_MATH__
    wc = mask_null_f16m4(lo_w_mask, w, vl);
#else
    wc = w;
#endif

    pol = __riscv_vfmul_vv_f16m4(wc, wc, vl);
    pol = __riscv_vfmadd_vf_f16m4(pol, -0x1.52cp-13f16, __riscv_vfmv_v_f_f16m4(-0x1p-14f16, vl), vl);

    w  =  __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vxor_vv_u16m4_mu(sign_x, __riscv_vreinterpret_v_f16m4_u16m4(w),   __riscv_vreinterpret_v_f16m4_u16m4(minus), __riscv_vreinterpret_v_f16m4_u16m4(w),   vl));
    w_l = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vxor_vv_u16m4_mu(sign_x, __riscv_vreinterpret_v_f16m4_u16m4(w_l), __riscv_vreinterpret_v_f16m4_u16m4(minus), __riscv_vreinterpret_v_f16m4_u16m4(w_l), vl));

    x = __riscv_vfadd_vv_f16m4(a, w, vl);
    t = __riscv_vfsub_vv_f16m4(x, a, vl);
    w_l = __riscv_vfmadd_vf_f16m4(__riscv_vfsub_vv_f16m4(w, t, vl), h_magic_consts1[2], w_l, vl);
    x = __riscv_vfmadd_vf_f16m4(__riscv_vfsub_vf_f16m4(__riscv_vfadd_vv_f16m4(__riscv_vfmadd_vv_f16m4(pol, w, b, vl), w_l, vl), h_table_lo[0], vl), h_magic_consts1[3], x, vl);
    x = __riscv_vfmul_vf_f16m4(x, h_magic_consts1[3], vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmacc_vf_f16m4_mu(mask_inf_x, x, 0x1p-14f16, __riscv_vfmv_v_f_f16m4(0x1p-14f16, vl), vl);
    x = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vor_vx_u16m4_mu(mask_nan, __riscv_vreinterpret_v_f16m4_u16m4(x), __riscv_vreinterpret_v_f16m4_u16m4(x), h_magic_consts2[3], vl));
    x = __riscv_vfmadd_vf_f16m4(x, 0x1p-14f16, x, vl);
#endif

    return __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vor_vv_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(sign_y), __riscv_vreinterpret_v_f16m4_u16m4(x), vl));
}





vfloat16m8_t __riscv_vatan2_f16m8(vfloat16m8_t y, vfloat16m8_t x, size_t vl) {
    vfloat16m8_t w, xc, yc, wc, w_l, z, k, a, b, al, bl, A, Al, B, Bl, pol, minus, t, c, blend_x, blend_y, ch, cl, rh, rl, r, multiplicator;
    vuint16m8_t index;
    vfloat16m8_t sign_y;
    vbool2_t masky0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, sign_x;
#ifndef __FAST_MATH__
    vbool2_t mask_inf_x, mask_inf_y, mask_nan;
    vbool2_t lo_w_mask, hi_w_mask, mask_small, mask_big, maskx_uf, masky_uf;
    vint16m8_t p_diff;
#endif
    minus = __riscv_vfmv_v_f_f16m8(h_borders1[0], vl);

    sign_x = __riscv_vmsgeu_vv_u16m8_b2(__riscv_vreinterpret_v_f16m8_u16m8(x), __riscv_vreinterpret_v_f16m8_u16m8(minus), vl);
    masky0 = __riscv_vmfeq_vv_f16m8_b2(minus, y, vl);
#ifndef __FAST_MATH__
    mask_nan = __riscv_vmor_mm_b2(__riscv_vmfne_vv_f16m8_b2(x, x, vl), __riscv_vmfne_vv_f16m8_b2(y, y, vl), vl);
#endif
    sign_y = __riscv_vreinterpret_v_u16m8_f16m8(__riscv_vand_vv_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(minus), __riscv_vreinterpret_v_f16m8_u16m8(y), vl));
    x = __riscv_vreinterpret_v_u16m8_f16m8(__riscv_vand_vv_u16m8(__riscv_vnot_v_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(minus), vl), __riscv_vreinterpret_v_f16m8_u16m8(x), vl));
    y = __riscv_vreinterpret_v_u16m8_f16m8(__riscv_vand_vv_u16m8(__riscv_vnot_v_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(minus), vl), __riscv_vreinterpret_v_f16m8_u16m8(y), vl));
#ifndef __FAST_MATH__
    mask_inf_y = __riscv_vmfeq_vf_f16m8_b2(y, h_magic_consts1[0], vl);
    mask_inf_x = __riscv_vmfeq_vf_f16m8_b2(x, h_magic_consts1[0], vl);
    y = __riscv_vfmerge_vfm_f16m8(y, h_borders2[0], mask_inf_y, vl);
    x = mask_null_f16m8(mask_inf_y, x, vl);

    y = mask_null_f16m8(__riscv_vmor_mm_b2(__riscv_vmandn_mm_b2(mask_inf_x, mask_inf_y, vl), mask_nan, vl), y, vl);
    x = __riscv_vfmerge_vfm_f16m8(x, h_borders2[0],  __riscv_vmor_mm_b2(__riscv_vmor_mm_b2(mask_inf_x, masky0, vl), mask_nan, vl), vl);
#else
    x = __riscv_vfmerge_vfm_f16m8(x, h_borders2[0], masky0, vl);
#endif

    multiplicator = __riscv_vreinterpret_v_i16m8_f16m8(
			__riscv_vsll_vx_i16m8(
				__riscv_vsub_vv_i16m8(
					__riscv_vmv_v_x_i16m8(h_magic_consts2[2], vl), 
					__riscv_vreinterpret_v_u16m8_i16m8(__riscv_vsrl_vx_u16m8(
						__riscv_vadd_vv_u16m8(
								      __riscv_vand_vx_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(y), 0x7c00, vl), 
								      __riscv_vreinterpret_v_f16m8_u16m8(x), 
								      vl), 
						11, vl)), 
					vl),
				10, vl)
			);
#ifndef __FAST_MATH__
    hi_w_mask = __riscv_vmandn_mm_b2(__riscv_vmandn_mm_b2(sign_x, mask_nan, vl), mask_inf_x, vl);
#endif

    multiplicator = __riscv_vfmerge_vfm_f16m8(multiplicator, h_magic_consts1[1], __riscv_vmfle_vv_f16m8_b2(multiplicator, minus, vl), vl);

    x = __riscv_vfmul_vv_f16m8(x, multiplicator, vl);
    y = __riscv_vfmul_vv_f16m8(y, multiplicator, vl);

#ifndef __FAST_MATH__
    p_diff = __riscv_vsra_vx_i16m8(__riscv_vsub_vv_i16m8(__riscv_vreinterpret_v_f16m8_i16m8(y), __riscv_vreinterpret_v_f16m8_i16m8(x), vl), 10, vl);

    maskx_uf = __riscv_vmflt_vf_f16m8_b2(x, 0x1p-10f16, vl);
    masky_uf = __riscv_vmflt_vf_f16m8_b2(y, 0x1p-10f16, vl);
    mask_small = __riscv_vmsle_vx_i16m8_b2(p_diff, -0xb, vl);
    mask_big = __riscv_vmsge_vx_i16m8_b2(p_diff, 0x12, vl);
    xc = mask_null_f16m8(maskx_uf, x, vl);
    yc = mask_null_f16m8(masky_uf, y, vl);

    hi_w_mask = __riscv_vmand_mm_b2(hi_w_mask, __riscv_vmor_mm_b2(mask_small, mask_big, vl), vl);
#else
    xc = x;
    yc = y;
#endif

    mask1 = __riscv_vmfle_vv_f16m8_b2(__riscv_vfmul_vf_f16m8(xc, h_borders1[1], vl), y, vl);
    mask2 = __riscv_vmflt_vv_f16m8_b2(__riscv_vfmul_vf_f16m8(xc, h_borders1[2], vl), y, vl);
    mask3 = __riscv_vmflt_vv_f16m8_b2(__riscv_vfmul_vf_f16m8(xc, h_borders1[3], vl), y, vl);
    mask4 = __riscv_vmflt_vv_f16m8_b2(x, y, vl);
    mask5 = __riscv_vmfle_vv_f16m8_b2(x, __riscv_vfmul_vf_f16m8(yc, h_borders2[1], vl), vl);
    mask6 = __riscv_vmfle_vv_f16m8_b2(x, __riscv_vfmul_vf_f16m8(yc, h_borders2[2], vl), vl);
    mask7 = __riscv_vmfle_vv_f16m8_b2(x, __riscv_vfmul_vf_f16m8(yc, h_borders2[3], vl), vl);
#ifndef __FAST_MATH__
    mask_big = __riscv_vmsge_vx_i16m8_b2(p_diff, 0xb, vl);
#endif

    index = __riscv_vmv_v_x_u16m8(0, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask1, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask2, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask3, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask4, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask5, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask6, index, index, 2, vl);
    index = __riscv_vadd_vx_u16m8_mu(mask7, index, index, 2, vl);

#ifndef __FAST_MATH__
    lo_w_mask = __riscv_vmor_mm_b2(mask_small, mask_big, vl);
    hi_w_mask = __riscv_vmor_mm_b2(hi_w_mask, __riscv_vmandn_mm_b2(mask_big, sign_x, vl), vl);
#endif

    A = mask_null_f16m8(mask7, y, vl);
    k = __riscv_vloxei16_v_f16m8(h_k, index, vl);

    c = __riscv_vfmul_vv_f16m8(k, x, vl);
    al = __riscv_vfmsub_vv_f16m8(k, x, c, vl);
    a = __riscv_vfsub_vv_f16m8(A, c, vl);
    mask7 = __riscv_vmnot_m_b2(mask7, vl);
    Al = __riscv_vfsub_vv_f16m8(a, A, vl);
    al = __riscv_vfsub_vv_f16m8(__riscv_vfsub_vv_f16m8(__riscv_vfsub_vv_f16m8(A, __riscv_vfsub_vv_f16m8(a, Al, vl), vl), __riscv_vfadd_vv_f16m8(Al, c, vl), vl), al, vl);
    a = __riscv_vfmul_vf_f16m8(a, h_magic_consts1[2], vl);
#ifndef __FAST_MATH__
    a = mask_null_f16m8(hi_w_mask, a, vl);
#endif
    al = __riscv_vfmul_vf_f16m8(al, h_magic_consts1[2], vl);

    b = __riscv_vfmadd_vv_f16m8_mu(mask7, y, k, x, vl);
    Bl = __riscv_vfsub_vv_f16m8_mu(mask7, minus, b, x, vl);
    bl = __riscv_vfadd_vv_f16m8_mu(mask7, minus, __riscv_vfsub_vv_f16m8_mu(mask7, minus, x, __riscv_vfsub_vv_f16m8(b, Bl, vl), vl), __riscv_vfmsac_vv_f16m8(Bl, k, y, vl), vl);
    bl = __riscv_vfmul_vf_f16m8(bl, h_magic_consts1[2], vl);

    ch =  __riscv_vfdiv_vv_f16m8(a, b, vl);
    al = __riscv_vfmul_vf_f16m8(al, h_magic_consts1[2], vl);

    rh = __riscv_vfnmsub_vv_f16m8(ch, b, a, vl);
    rl = __riscv_vfnmsub_vv_f16m8(ch, bl, al, vl);
    r =  __riscv_vfmadd_vf_f16m8(rh, h_magic_consts1[2], rl, vl);
#ifndef __FAST_MATH__
    r = mask_null_f16m8(lo_w_mask, r, vl);
#endif
    cl = __riscv_vfdiv_vv_f16m8(r, b, vl);

    w = __riscv_vfmadd_vf_f16m8(cl, h_magic_consts1[3], ch, vl);
    w_l = __riscv_vfmadd_vf_f16m8(__riscv_vfsub_vv_f16m8(ch, w, vl), h_magic_consts1[2], cl, vl);

#ifndef __FAST_MATH__
    index = __riscv_vadd_vx_u16m8_mu(__riscv_vmandn_mm_b2(sign_x, mask_nan, vl), index, index, h_magic_consts2[1], vl);
#else
    index = __riscv_vadd_vx_u16m8_mu(sign_x, index, index, h_magic_consts2[1], vl);
#endif

    a = __riscv_vloxei16_v_f16m8(h_table_hi, index, vl);
    b = __riscv_vloxei16_v_f16m8(h_table_lo, index, vl);

#ifndef __FAST_MATH__
    wc = mask_null_f16m8(lo_w_mask, w, vl);
#else
    wc = w;
#endif

    pol = __riscv_vfmul_vv_f16m8(wc, wc, vl);
    pol = __riscv_vfmadd_vf_f16m8(pol, -0x1.52cp-13f16, __riscv_vfmv_v_f_f16m8(-0x1p-14f16, vl), vl);

    w  =  __riscv_vreinterpret_v_u16m8_f16m8(__riscv_vxor_vv_u16m8_mu(sign_x, __riscv_vreinterpret_v_f16m8_u16m8(w),   __riscv_vreinterpret_v_f16m8_u16m8(minus), __riscv_vreinterpret_v_f16m8_u16m8(w),   vl));
    w_l = __riscv_vreinterpret_v_u16m8_f16m8(__riscv_vxor_vv_u16m8_mu(sign_x, __riscv_vreinterpret_v_f16m8_u16m8(w_l), __riscv_vreinterpret_v_f16m8_u16m8(minus), __riscv_vreinterpret_v_f16m8_u16m8(w_l), vl));

    x = __riscv_vfadd_vv_f16m8(a, w, vl);
    t = __riscv_vfsub_vv_f16m8(x, a, vl);
    w_l = __riscv_vfmadd_vf_f16m8(__riscv_vfsub_vv_f16m8(w, t, vl), h_magic_consts1[2], w_l, vl);
    x = __riscv_vfmadd_vf_f16m8(__riscv_vfsub_vf_f16m8(__riscv_vfadd_vv_f16m8(__riscv_vfmadd_vv_f16m8(pol, w, b, vl), w_l, vl), h_table_lo[0], vl), h_magic_consts1[3], x, vl);
    x = __riscv_vfmul_vf_f16m8(x, h_magic_consts1[3], vl);

#ifndef __FAST_MATH__
    x = __riscv_vfmacc_vf_f16m8_mu(mask_inf_x, x, 0x1p-14f16, __riscv_vfmv_v_f_f16m8(0x1p-14f16, vl), vl);
    x = __riscv_vreinterpret_v_u16m8_f16m8(__riscv_vor_vx_u16m8_mu(mask_nan, __riscv_vreinterpret_v_f16m8_u16m8(x), __riscv_vreinterpret_v_f16m8_u16m8(x), h_magic_consts2[3], vl));
    x = __riscv_vfmadd_vf_f16m8(x, 0x1p-14f16, x, vl);
#endif

    return __riscv_vreinterpret_v_u16m8_f16m8(__riscv_vor_vv_u16m8(__riscv_vreinterpret_v_f16m8_u16m8(sign_y), __riscv_vreinterpret_v_f16m8_u16m8(x), vl));
}

#endif