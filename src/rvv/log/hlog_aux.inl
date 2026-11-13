/*
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025-2026 The RVVMF Authors (Elena Panova,
 * Vladislav Polozov)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  hlog_aux.inl                                 *
 *   Contains: helper built-in functions for log, log2   *
 *             log10, log1p functions (float16_t)        *
 *                                                       *
 *                                                       *
 *********************************************************
*/

constexpr _Float16 LOG_FACTOR_COEFF_F16 = 0x1p8f16;
constexpr _Float16 LOG_INV_FACTOR_COEFF_F16 = 0x1p-8f16;

// ==================================================================================================
// vfloat16m1_t
// ==================================================================================================

static inline void fast2sum_f16m1(vfloat16m1_t a, vfloat16m1_t b, vfloat16m1_t& s, vfloat16m1_t& t, size_t vl)
{
    s = __riscv_vfadd_vv_f16m1(a, b, vl);
    vfloat16m1_t b1 = __riscv_vfsub_vv_f16m1(s, a, vl);
    t = __riscv_vfsub_vv_f16m1(b, b1, vl);
}

static inline vfloat16m1_t mul21_f16m1(vfloat16m1_t ah, vfloat16m1_t al, _Float16 bh, _Float16 bl, size_t vl)
{
    constexpr _Float16 factorh = 0x1p2f16, invfactorh = 0x1p-2f16;
    constexpr _Float16 factorl = 0x1p15f16, invfactorl = 0x1p-15f16;
    constexpr _Float16 factorlmh = factorl*invfactorh;
    constexpr _Float16 factorlmhl = factorlmh*LOG_INV_FACTOR_COEFF_F16;

    vfloat16m1_t zh, zl;
    bh *= factorh; bl *= factorl;
    zh = __riscv_vfmul_vf_f16m1(ah, bh, vl);
    zl = __riscv_vfmul_vf_f16m1(__riscv_vfmsub_vf_f16m1(ah, bh, zh, vl), factorlmh, vl);
    zl = __riscv_vfadd_vv_f16m1(zl, __riscv_vfmadd_vf_f16m1(__riscv_vfmul_vf_f16m1(al, factorlmhl, vl),
        bh, __riscv_vfmul_vf_f16m1(ah, bl, vl), vl), vl);
    return __riscv_vfmadd_vf_f16m1(zl, invfactorl, __riscv_vfmul_vf_f16m1(zh, invfactorh, vl), vl);
}

static inline vfloat16m1_t abs_f16m1(vfloat16m1_t x, size_t vl)
{
    return __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vx_u16m1(
        __riscv_vreinterpret_v_f16m1_u16m1(x), 0x7fff, vl));
}

static inline void log_check_special_cases_f16m1(vfloat16m1_t& x, vfloat16m1_t& special,
    vbool16_t& specialMask, size_t vl, _Float16 leftDomainPoint = 0.0f16, _Float16 defaultArgValue = 1.0f16)
{
    uint16_t pinf = 0x7c00;
    uint16_t ninf = 0xfc00;
    uint16_t qnan = 0x7e00;
    vuint16m1_t absxui = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(x),
        (uint16_t)0x7fff, vl);

    vbool16_t nanMask = __riscv_vmsgtu_vx_u16m1_b16(absxui, pinf, vl);
    x = __riscv_vfmerge_vfm_f16m1(x, defaultArgValue, nanMask, vl);

    vbool16_t snanMask = __riscv_vmsltu_vx_u16m1_b16(absxui, qnan, vl);
    vbool16_t infMask = __riscv_vmseq_vx_u16m1_b16(__riscv_vreinterpret_v_f16m1_u16m1(x), pinf, vl);
    vbool16_t leftDomainPointMask = __riscv_vmfeq_vf_f16m1_b16(x, leftDomainPoint, vl);
    vbool16_t invalidDomainMask = __riscv_vmflt_vf_f16m1_b16(x, leftDomainPoint, vl);

    special = __riscv_vfmerge_vfm_f16m1(x, *(_Float16*)(&pinf), infMask, vl);
    special = __riscv_vfmerge_vfm_f16m1(special, *(_Float16*)(&ninf), leftDomainPointMask, vl);
    special = __riscv_vfmerge_vfm_f16m1(special, *(_Float16*)(&qnan),
        __riscv_vmor_mm_b16(invalidDomainMask, nanMask, vl), vl);

    specialMask = __riscv_vmor_mm_b16(__riscv_vmor_mm_b16(leftDomainPointMask, invalidDomainMask, vl),
        __riscv_vmor_mm_b16(infMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b16(__riscv_vmor_mm_b16(
        __riscv_vmand_mm_b16(nanMask, snanMask, vl), invalidDomainMask, vl), vl))
        volatile _Float16 x1 = 0.0f16/0.0f16;
    if (__riscv_vcpop_m_b16(leftDomainPointMask, vl))
        volatile _Float16 x1 = 1.0f16/0.0f16;

    x = __riscv_vfmerge_vfm_f16m1(x, defaultArgValue, specialMask, vl);
}

static inline void my_log_argument_reduction_f16m1(vfloat16m1_t x, vfloat16m1_t& y,
    vint16m1_t& e, vfloat16m1_t& ef, vuint16m1_t& index, size_t vl)
{
    constexpr uint16_t magic_const_1 = 0x3950;
    constexpr size_t tshift = 10 - 4 - 1;

    vuint16m1_t xi = __riscv_vreinterpret_v_f16m1_u16m1(x);
    vbool16_t msubn = __riscv_vmsltu_vx_u16m1_b16(xi, 0x0400, vl);
    x = __riscv_vfmul_vf_f16m1_mu(msubn, x, x, (_Float16)0x1p10f16, vl);
    xi = __riscv_vreinterpret_v_f16m1_u16m1(x);

    vuint16m1_t magic_number = __riscv_vsub_vx_u16m1(xi, magic_const_1, vl);
    e = __riscv_vsra_vx_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(magic_number), (size_t)10, vl);
    e = __riscv_vsub_vx_i16m1_mu(msubn, e, e, (int16_t)10, vl);
    xi = __riscv_vsub_vv_u16m1(xi, __riscv_vand_vx_u16m1(magic_number, (uint16_t)0xfc00, vl), vl);
    y = __riscv_vreinterpret_v_u16m1_f16m1(xi);

    index = __riscv_vsrl_vx_u16m1(__riscv_vand_vx_u16m1(magic_number, (uint16_t)0x03c0, vl), tshift, vl);
    ef = __riscv_vfcvt_f_x_v_f16m1(e, vl);
}

static inline void my_log_table_values_f16m1(vuint16m1_t index, vfloat16m1_t& r, vfloat16m1_t& logR,
    vfloat16m1_t& logRl, size_t vl)
{
    logR = __riscv_vloxei16_v_f16m1(table_half_log_R, index, vl);
    logRl = __riscv_vloxei16_v_f16m1(table_half_log_Rl, index, vl);
    r = __riscv_vloxei16_v_f16m1(table_half_R, index, vl);
}

static inline void my_log_second_reduction_f16m1(vfloat16m1_t y, vfloat16m1_t r,
    vfloat16m1_t& z, vfloat16m1_t& zh, vfloat16m1_t& zl, size_t vl)
{
    zh = __riscv_vfmul_vv_f16m1(y, r, vl);
    zl = __riscv_vfmsub_vv_f16m1(y, r, zh, vl);
    zh = __riscv_vfsub_vf_f16m1(zh, LOG_FACTOR_COEFF_F16, vl);
    z = __riscv_vfadd_vv_f16m1(zh, zl, vl);
}

static inline void my_log1p_second_reduction_f16m1(vfloat16m1_t xm1, vfloat16m1_t r,
    vint16m1_t e, vfloat16m1_t& z, vfloat16m1_t& zh, vfloat16m1_t& zl, size_t vl)
{
    vint16m1_t pow2mei = __riscv_vadd_vx_i16m1(__riscv_vneg_v_i16m1(e, vl), (int16_t)17, vl);
    vfloat16m1_t pow2me = __riscv_vreinterpret_v_i16m1_f16m1(__riscv_vsll_vx_i16m1(pow2mei, 10, vl));
    vfloat16m1_t c = __riscv_vfmul_vv_f16m1(pow2me, __riscv_vfmul_vf_f16m1(r, LOG_INV_FACTOR_COEFF_F16, vl), vl);
    constexpr _Float16 f = LOG_FACTOR_COEFF_F16 * 0.25f16;

    vfloat16m1_t tmph, tmpl, sh, sl;
    tmph = __riscv_vfmadd_vv_f16m1(xm1, c, c, vl);
    sh = __riscv_vfsub_vv_f16m1(tmph, c, vl);
    sl = __riscv_vfadd_vv_f16m1(sh, c, vl);
    sl = __riscv_vfsub_vv_f16m1(__riscv_vfsub_vv_f16m1(tmph, sl, vl),
        __riscv_vfadd_vv_f16m1(c, __riscv_vfsub_vv_f16m1(sh, sl, vl), vl), vl);
    tmpl = __riscv_vfmsub_vv_f16m1(xm1, c, sh, vl);
    zh = __riscv_vfsub_vf_f16m1(tmph, (_Float16)4.0f16, vl);
    zl = __riscv_vfsub_vv_f16m1(tmph, __riscv_vfadd_vf_f16m1(zh, (_Float16)4.0f16, vl), vl);
    zl = __riscv_vfadd_vv_f16m1(__riscv_vfsub_vv_f16m1(zl, sl, vl), tmpl, vl);

    zh = __riscv_vfmul_vf_f16m1(zh, f, vl);
    zl = __riscv_vfmul_vf_f16m1(zl, f, vl);
    z = __riscv_vfadd_vv_f16m1(zh, zl, vl);
}

static inline void my_log_approximation_f16m1(vfloat16m1_t z, vfloat16m1_t& res, size_t vl)
{
    constexpr _Float16 p0 = -0x1.004p-1f16;
    constexpr _Float16 p1invf = 0x1.564p-2f16 * LOG_INV_FACTOR_COEFF_F16;

    vfloat16m1_t tmp = __riscv_vfmadd_vf_f16m1(z, p1invf, __riscv_vfmv_v_f_f16m1(p0, vl), vl);
    res = __riscv_vfmul_vv_f16m1(z, tmp, vl);
}

static inline void my_log1p_approximation_f16m1(vfloat16m1_t z, vfloat16m1_t& resh, size_t vl)
{
    constexpr _Float16 p0 = -0x1p-1f16;
    constexpr _Float16 p1 = 0x1.56p-2f16;
    constexpr _Float16 p2invf = -0x1.008p-2f16 * LOG_INV_FACTOR_COEFF_F16;

    vfloat16m1_t tmp = __riscv_vfmadd_vf_f16m1(z, p2invf, __riscv_vfmv_v_f_f16m1(p1, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m1(z, __riscv_vfmul_vf_f16m1(tmp, LOG_INV_FACTOR_COEFF_F16, vl),
        __riscv_vfmv_v_f_f16m1(p0, vl), vl);
    resh = __riscv_vfmul_vv_f16m1(z, tmp, vl);
}

static inline void my_log_reconstruction_h_f16m1(vfloat16m1_t e, vfloat16m1_t zh, vfloat16m1_t zl,
    vfloat16m1_t p, vfloat16m1_t logR, vfloat16m1_t logRl, vfloat16m1_t& res, size_t vl)
{
    constexpr _Float16 log2h = 0x1.63p-1f16 * LOG_FACTOR_COEFF_F16;
    constexpr _Float16 log2l = (-0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16) * LOG_FACTOR_COEFF_F16;

    vfloat16m1_t rh, rl, tmp, tmpl;

    tmp = __riscv_vfmadd_vv_f16m1(p, zh, __riscv_vfmadd_vv_f16m1(
        p, zl, __riscv_vfmul_vf_f16m1(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmp = __riscv_vfmadd_vf_f16m1(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f16m1(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f16m1(zh, __riscv_vfadd_vv_f16m1(res, logR, vl), vl);
    tmpl = __riscv_vfmsub_vf_f16m1(tmpl, LOG_FACTOR_COEFF_F16, logRl, vl);
    rh = __riscv_vfmadd_vf_f16m1(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f16m1(res, __riscv_vfmsub_vf_f16m1(e, log2h, rh, vl), vl);
    rl = __riscv_vfmadd_vf_f16m1(rl, LOG_FACTOR_COEFF_F16, tmpl, vl);
    res = __riscv_vfmadd_vf_f16m1(__riscv_vfadd_vv_f16m1(rl, tmp, vl), LOG_INV_FACTOR_COEFF_F16, rh, vl);
}

static inline void my_log_reconstruction_hl_f16m1(vfloat16m1_t e, vfloat16m1_t zh, vfloat16m1_t zl,
    vfloat16m1_t p, vfloat16m1_t logR, vfloat16m1_t logRl, vfloat16m1_t& res, vfloat16m1_t& resl, size_t vl)
{
    constexpr _Float16 log2h = 0x1.63p-1f16 * LOG_FACTOR_COEFF_F16;
    constexpr _Float16 log2l = (-0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16) * LOG_FACTOR_COEFF_F16;

    vfloat16m1_t rh, rl, tmp;

    tmp = __riscv_vfmadd_vv_f16m1(p, zh, __riscv_vfmadd_vv_f16m1(
        p, zl, __riscv_vfmul_vf_f16m1(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmp = __riscv_vfmadd_vf_f16m1(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f16m1(zh, logR, vl);
    resl = __riscv_vfsub_vv_f16m1(zh, __riscv_vfadd_vv_f16m1(res, logR, vl), vl);
    resl = __riscv_vfmsub_vf_f16m1(resl, LOG_FACTOR_COEFF_F16, logRl, vl);
    rh = __riscv_vfmadd_vf_f16m1(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f16m1(res, __riscv_vfmsub_vf_f16m1(e, log2h, rh, vl), vl);
    rl = __riscv_vfadd_vv_f16m1(tmp, __riscv_vfmadd_vf_f16m1(rl, LOG_FACTOR_COEFF_F16, resl, vl), vl);
    res = __riscv_vfmadd_vf_f16m1(rl, LOG_INV_FACTOR_COEFF_F16, rh, vl);
    resl = __riscv_vfmadd_vf_f16m1(__riscv_vfsub_vv_f16m1(rh, res, vl), LOG_FACTOR_COEFF_F16, rl, vl);
}

static inline void my_log2_reconstruction_f16m1(vfloat16m1_t e, vfloat16m1_t zh, vfloat16m1_t zl,
    vfloat16m1_t p, vfloat16m1_t logR, vfloat16m1_t logRl, vfloat16m1_t& res, size_t vl)
{
    constexpr _Float16 invLog2h = 0x1.714p0f16, invLog2l = 0x1.478p-12f16*LOG_FACTOR_COEFF_F16;
    e = __riscv_vfmul_vf_f16m1(e, LOG_FACTOR_COEFF_F16, vl);

    vfloat16m1_t rh, rl, tmp, tmph, tmpl, sh, sl;

    tmp = __riscv_vfmadd_vv_f16m1(p, zh, __riscv_vfmadd_vv_f16m1(
        p, zl, __riscv_vfmul_vf_f16m1(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmph = __riscv_vfsub_vv_f16m1(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f16m1(zh, __riscv_vfadd_vv_f16m1(tmph, logR, vl), vl);
    tmpl = __riscv_vfmsub_vf_f16m1(tmpl, LOG_FACTOR_COEFF_F16, logRl, vl);
    tmp = __riscv_vfmadd_vf_f16m1(tmp, invLog2h, __riscv_vfmadd_vf_f16m1(tmpl, invLog2h,
        __riscv_vfmul_vf_f16m1(tmph, invLog2l, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f16m1(tmph, invLog2h, e, vl);
    sh = __riscv_vfsub_vv_f16m1(rh, e, vl);
    sl = __riscv_vfsub_vv_f16m1(rh, __riscv_vfadd_vv_f16m1(sh, e, vl), vl);
    rl = __riscv_vfmsub_vf_f16m1(tmph, invLog2h, sh, vl);
    res = __riscv_vfadd_vv_f16m1(rh, __riscv_vfmadd_vf_f16m1(tmp, LOG_INV_FACTOR_COEFF_F16,
        __riscv_vfsub_vv_f16m1(rl, sl, vl), vl), vl);
}

static inline void my_log1p_reconstruction_h_f16m1(vfloat16m1_t e, vfloat16m1_t zh,
    vfloat16m1_t zl, vfloat16m1_t p, vfloat16m1_t logR, vfloat16m1_t logRl, vfloat16m1_t& res, size_t vl)
{
    constexpr _Float16 log2h = 0x1.63p-1f16 * LOG_FACTOR_COEFF_F16;
    constexpr _Float16 log2l = (-0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16) * LOG_FACTOR_COEFF_F16;

    vfloat16m1_t rh, rl, tmp, tmph, tmpl, tmp1h, tmp1l, resl, zlf2;

#ifndef __FAST_MATH__
    vbool16_t mzlc = __riscv_vmflt_vf_f16m1_b16(abs_f16m1(zl, vl), (_Float16)0x1p-10f16, vl);
    vfloat16m1_t zlc = __riscv_vfadd_vf_f16m1_mu(mzlc, zl, zl, (_Float16)0x1p-10f16, vl);
#else
    vfloat16m1_t zlc = zl;
#endif
    p = __riscv_vfmul_vf_f16m1(p, LOG_FACTOR_COEFF_F16, vl);
    zlf2 = __riscv_vfmul_vf_f16m1(__riscv_vfmul_vf_f16m1(zl, LOG_FACTOR_COEFF_F16, vl), LOG_FACTOR_COEFF_F16, vl);

    tmp = __riscv_vfmadd_vv_f16m1(p, zl, zlf2, vl);
    tmpl = __riscv_vfmsac_vv_f16m1(__riscv_vfsub_vv_f16m1(tmp, zlf2, vl), p, zlc, vl);
    tmph = __riscv_vfmadd_vv_f16m1(p, zh, tmp, vl);
    tmpl = __riscv_vfadd_vv_f16m1(tmpl, __riscv_vfmsac_vv_f16m1(__riscv_vfsub_vv_f16m1(tmph, tmp, vl), p, zh, vl), vl);
    tmph = __riscv_vfmadd_vf_f16m1(tmph, LOG_INV_FACTOR_COEFF_F16, __riscv_vfmul_vf_f16m1(e, log2l, vl), vl);
    tmpl = __riscv_vfnmsac_vf_f16m1(tmpl, LOG_FACTOR_COEFF_F16, logRl, vl);
    tmp1h = __riscv_vfsub_vv_f16m1(zh, logR, vl);
    tmp1l = __riscv_vfsub_vv_f16m1(__riscv_vfsub_vv_f16m1(zh, tmp1h, vl), logR, vl);
    rh = __riscv_vfmadd_vf_f16m1(e, log2h, tmp1h, vl);
    rl = __riscv_vfadd_vv_f16m1(tmp1h, __riscv_vfmsub_vf_f16m1(e, log2h, rh, vl), vl);
    res = __riscv_vfmacc_vf_f16m1(rh, LOG_INV_FACTOR_COEFF_F16, tmph, vl);
    resl = __riscv_vfmadd_vf_f16m1(tmph, LOG_INV_FACTOR_COEFF_F16, __riscv_vfsub_vv_f16m1(rh, res, vl), vl);
    resl = __riscv_vfadd_vv_f16m1(resl, tmp1l, vl);
    resl = __riscv_vfadd_vv_f16m1(resl, rl, vl);
    resl = __riscv_vfmul_vf_f16m1(resl, LOG_FACTOR_COEFF_F16, vl);
    resl = __riscv_vfmacc_vf_f16m1(resl, LOG_INV_FACTOR_COEFF_F16, tmpl, vl);
    res = __riscv_vfmacc_vf_f16m1(res, LOG_INV_FACTOR_COEFF_F16, resl, vl);
}

static inline void my_log1p_process_linear_f16m1(const vfloat16m1_t& xm1, vfloat16m1_t& xm1c,
    vbool16_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f16m1_b16(abs_f16m1(xm1, vl), (_Float16)0x1.6ap-11f16, vl);
    xm1c = __riscv_vfmerge_vfm_f16m1(xm1, (_Float16)0.0f16, linearMask, vl);
}

static inline void my_log1p_set_fe_underflow_f16m1(vfloat16m1_t& res, size_t vl)
{
    res = __riscv_vfmadd_vf_f16m1(res, (_Float16)0x1p-14f16, res, vl);
}

// ==================================================================================================
// vfloat16m2_t
// ==================================================================================================

static inline void fast2sum_f16m2(vfloat16m2_t a, vfloat16m2_t b, vfloat16m2_t& s, vfloat16m2_t& t, size_t vl)
{
    s = __riscv_vfadd_vv_f16m2(a, b, vl);
    vfloat16m2_t b1 = __riscv_vfsub_vv_f16m2(s, a, vl);
    t = __riscv_vfsub_vv_f16m2(b, b1, vl);
}

static inline vfloat16m2_t mul21_f16m2(vfloat16m2_t ah, vfloat16m2_t al, _Float16 bh, _Float16 bl, size_t vl)
{
    constexpr _Float16 factorh = 0x1p2f16, invfactorh = 0x1p-2f16;
    constexpr _Float16 factorl = 0x1p15f16, invfactorl = 0x1p-15f16;
    constexpr _Float16 factorlmh = factorl*invfactorh;
    constexpr _Float16 factorlmhl = factorlmh*LOG_INV_FACTOR_COEFF_F16;

    vfloat16m2_t zh, zl;
    bh *= factorh; bl *= factorl;
    zh = __riscv_vfmul_vf_f16m2(ah, bh, vl);
    zl = __riscv_vfmul_vf_f16m2(__riscv_vfmsub_vf_f16m2(ah, bh, zh, vl), factorlmh, vl);
    zl = __riscv_vfadd_vv_f16m2(zl, __riscv_vfmadd_vf_f16m2(__riscv_vfmul_vf_f16m2(al, factorlmhl, vl),
        bh, __riscv_vfmul_vf_f16m2(ah, bl, vl), vl), vl);
    return __riscv_vfmadd_vf_f16m2(zl, invfactorl, __riscv_vfmul_vf_f16m2(zh, invfactorh, vl), vl);
}

static inline vfloat16m2_t abs_f16m2(vfloat16m2_t x, size_t vl)
{
    return __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vx_u16m2(
        __riscv_vreinterpret_v_f16m2_u16m2(x), 0x7fff, vl));
}

static inline void log_check_special_cases_f16m2(vfloat16m2_t& x, vfloat16m2_t& special,
    vbool8_t& specialMask, size_t vl, _Float16 leftDomainPoint = 0.0f16, _Float16 defaultArgValue = 1.0f16)
{
    uint16_t pinf = 0x7c00;
    uint16_t ninf = 0xfc00;
    uint16_t qnan = 0x7e00;
    vuint16m2_t absxui = __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(x),
        (uint16_t)0x7fff, vl);

    vbool8_t nanMask = __riscv_vmsgtu_vx_u16m2_b8(absxui, pinf, vl);
    x = __riscv_vfmerge_vfm_f16m2(x, defaultArgValue, nanMask, vl);

    vbool8_t snanMask = __riscv_vmsltu_vx_u16m2_b8(absxui, qnan, vl);
    vbool8_t infMask = __riscv_vmseq_vx_u16m2_b8(__riscv_vreinterpret_v_f16m2_u16m2(x), pinf, vl);
    vbool8_t leftDomainPointMask = __riscv_vmfeq_vf_f16m2_b8(x, leftDomainPoint, vl);
    vbool8_t invalidDomainMask = __riscv_vmflt_vf_f16m2_b8(x, leftDomainPoint, vl);

    special = __riscv_vfmerge_vfm_f16m2(x, *(_Float16*)(&pinf), infMask, vl);
    special = __riscv_vfmerge_vfm_f16m2(special, *(_Float16*)(&ninf), leftDomainPointMask, vl);
    special = __riscv_vfmerge_vfm_f16m2(special, *(_Float16*)(&qnan),
        __riscv_vmor_mm_b8(invalidDomainMask, nanMask, vl), vl);

    specialMask = __riscv_vmor_mm_b8(__riscv_vmor_mm_b8(leftDomainPointMask, invalidDomainMask, vl),
        __riscv_vmor_mm_b8(infMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b8(__riscv_vmor_mm_b8(
        __riscv_vmand_mm_b8(nanMask, snanMask, vl), invalidDomainMask, vl), vl))
        volatile _Float16 x1 = 0.0f16/0.0f16;
    if (__riscv_vcpop_m_b8(leftDomainPointMask, vl))
        volatile _Float16 x1 = 1.0f16/0.0f16;

    x = __riscv_vfmerge_vfm_f16m2(x, defaultArgValue, specialMask, vl);
}

static inline void my_log_argument_reduction_f16m2(vfloat16m2_t x, vfloat16m2_t& y,
    vint16m2_t& e, vfloat16m2_t& ef, vuint16m2_t& index, size_t vl)
{
    constexpr uint16_t magic_const_1 = 0x3950;
    constexpr size_t tshift = 10 - 4 - 1;

    vuint16m2_t xi = __riscv_vreinterpret_v_f16m2_u16m2(x);
    vbool8_t msubn = __riscv_vmsltu_vx_u16m2_b8(xi, 0x0400, vl);
    x = __riscv_vfmul_vf_f16m2_mu(msubn, x, x, (_Float16)0x1p10f16, vl);
    xi = __riscv_vreinterpret_v_f16m2_u16m2(x);

    vuint16m2_t magic_number = __riscv_vsub_vx_u16m2(xi, magic_const_1, vl);
    e = __riscv_vsra_vx_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(magic_number), (size_t)10, vl);
    e = __riscv_vsub_vx_i16m2_mu(msubn, e, e, (int16_t)10, vl);
    xi = __riscv_vsub_vv_u16m2(xi, __riscv_vand_vx_u16m2(magic_number, (uint16_t)0xfc00, vl), vl);
    y = __riscv_vreinterpret_v_u16m2_f16m2(xi);

    index = __riscv_vsrl_vx_u16m2(__riscv_vand_vx_u16m2(magic_number, (uint16_t)0x03c0, vl), tshift, vl);
    ef = __riscv_vfcvt_f_x_v_f16m2(e, vl);
}

static inline void my_log_table_values_f16m2(vuint16m2_t index, vfloat16m2_t& r, vfloat16m2_t& logR,
    vfloat16m2_t& logRl, size_t vl)
{
    logR = __riscv_vloxei16_v_f16m2(table_half_log_R, index, vl);
    logRl = __riscv_vloxei16_v_f16m2(table_half_log_Rl, index, vl);
    r = __riscv_vloxei16_v_f16m2(table_half_R, index, vl);
}

static inline void my_log_second_reduction_f16m2(vfloat16m2_t y, vfloat16m2_t r,
    vfloat16m2_t& z, vfloat16m2_t& zh, vfloat16m2_t& zl, size_t vl)
{
    zh = __riscv_vfmul_vv_f16m2(y, r, vl);
    zl = __riscv_vfmsub_vv_f16m2(y, r, zh, vl);
    zh = __riscv_vfsub_vf_f16m2(zh, LOG_FACTOR_COEFF_F16, vl);
    z = __riscv_vfadd_vv_f16m2(zh, zl, vl);
}

static inline void my_log1p_second_reduction_f16m2(vfloat16m2_t xm1, vfloat16m2_t r,
    vint16m2_t e, vfloat16m2_t& z, vfloat16m2_t& zh, vfloat16m2_t& zl, size_t vl)
{
    vint16m2_t pow2mei = __riscv_vadd_vx_i16m2(__riscv_vneg_v_i16m2(e, vl), (int16_t)17, vl);
    vfloat16m2_t pow2me = __riscv_vreinterpret_v_i16m2_f16m2(__riscv_vsll_vx_i16m2(pow2mei, 10, vl));
    vfloat16m2_t c = __riscv_vfmul_vv_f16m2(pow2me, __riscv_vfmul_vf_f16m2(r, LOG_INV_FACTOR_COEFF_F16, vl), vl);
    constexpr _Float16 f = LOG_FACTOR_COEFF_F16 * 0.25f16;

    vfloat16m2_t tmph, tmpl, sh, sl;
    tmph = __riscv_vfmadd_vv_f16m2(xm1, c, c, vl);
    sh = __riscv_vfsub_vv_f16m2(tmph, c, vl);
    sl = __riscv_vfadd_vv_f16m2(sh, c, vl);
    sl = __riscv_vfsub_vv_f16m2(__riscv_vfsub_vv_f16m2(tmph, sl, vl),
        __riscv_vfadd_vv_f16m2(c, __riscv_vfsub_vv_f16m2(sh, sl, vl), vl), vl);
    tmpl = __riscv_vfmsub_vv_f16m2(xm1, c, sh, vl);
    zh = __riscv_vfsub_vf_f16m2(tmph, (_Float16)4.0f16, vl);
    zl = __riscv_vfsub_vv_f16m2(tmph, __riscv_vfadd_vf_f16m2(zh, (_Float16)4.0f16, vl), vl);
    zl = __riscv_vfadd_vv_f16m2(__riscv_vfsub_vv_f16m2(zl, sl, vl), tmpl, vl);

    zh = __riscv_vfmul_vf_f16m2(zh, f, vl);
    zl = __riscv_vfmul_vf_f16m2(zl, f, vl);
    z = __riscv_vfadd_vv_f16m2(zh, zl, vl);
}

static inline void my_log_approximation_f16m2(vfloat16m2_t z, vfloat16m2_t& res, size_t vl)
{
    constexpr _Float16 p0 = -0x1.004p-1f16;
    constexpr _Float16 p1invf = 0x1.564p-2f16 * LOG_INV_FACTOR_COEFF_F16;

    vfloat16m2_t tmp = __riscv_vfmadd_vf_f16m2(z, p1invf, __riscv_vfmv_v_f_f16m2(p0, vl), vl);
    res = __riscv_vfmul_vv_f16m2(z, tmp, vl);
}

static inline void my_log1p_approximation_f16m2(vfloat16m2_t z, vfloat16m2_t& resh, size_t vl)
{
    constexpr _Float16 p0 = -0x1p-1f16;
    constexpr _Float16 p1 = 0x1.56p-2f16;
    constexpr _Float16 p2invf = -0x1.008p-2f16 * LOG_INV_FACTOR_COEFF_F16;

    vfloat16m2_t tmp = __riscv_vfmadd_vf_f16m2(z, p2invf, __riscv_vfmv_v_f_f16m2(p1, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m2(z, __riscv_vfmul_vf_f16m2(tmp, LOG_INV_FACTOR_COEFF_F16, vl),
        __riscv_vfmv_v_f_f16m2(p0, vl), vl);
    resh = __riscv_vfmul_vv_f16m2(z, tmp, vl);
}

static inline void my_log_reconstruction_h_f16m2(vfloat16m2_t e, vfloat16m2_t zh, vfloat16m2_t zl,
    vfloat16m2_t p, vfloat16m2_t logR, vfloat16m2_t logRl, vfloat16m2_t& res, size_t vl)
{
    constexpr _Float16 log2h = 0x1.63p-1f16 * LOG_FACTOR_COEFF_F16;
    constexpr _Float16 log2l = (-0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16) * LOG_FACTOR_COEFF_F16;

    vfloat16m2_t rh, rl, tmp, tmpl;

    tmp = __riscv_vfmadd_vv_f16m2(p, zh, __riscv_vfmadd_vv_f16m2(
        p, zl, __riscv_vfmul_vf_f16m2(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmp = __riscv_vfmadd_vf_f16m2(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f16m2(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f16m2(zh, __riscv_vfadd_vv_f16m2(res, logR, vl), vl);
    tmpl = __riscv_vfmsub_vf_f16m2(tmpl, LOG_FACTOR_COEFF_F16, logRl, vl);
    rh = __riscv_vfmadd_vf_f16m2(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f16m2(res, __riscv_vfmsub_vf_f16m2(e, log2h, rh, vl), vl);
    rl = __riscv_vfmadd_vf_f16m2(rl, LOG_FACTOR_COEFF_F16, tmpl, vl);
    res = __riscv_vfmadd_vf_f16m2(__riscv_vfadd_vv_f16m2(rl, tmp, vl), LOG_INV_FACTOR_COEFF_F16, rh, vl);
}

static inline void my_log_reconstruction_hl_f16m2(vfloat16m2_t e, vfloat16m2_t zh, vfloat16m2_t zl,
    vfloat16m2_t p, vfloat16m2_t logR, vfloat16m2_t logRl, vfloat16m2_t& res, vfloat16m2_t& resl, size_t vl)
{
    constexpr _Float16 log2h = 0x1.63p-1f16 * LOG_FACTOR_COEFF_F16;
    constexpr _Float16 log2l = (-0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16) * LOG_FACTOR_COEFF_F16;

    vfloat16m2_t rh, rl, tmp;

    tmp = __riscv_vfmadd_vv_f16m2(p, zh, __riscv_vfmadd_vv_f16m2(
        p, zl, __riscv_vfmul_vf_f16m2(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmp = __riscv_vfmadd_vf_f16m2(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f16m2(zh, logR, vl);
    resl = __riscv_vfsub_vv_f16m2(zh, __riscv_vfadd_vv_f16m2(res, logR, vl), vl);
    resl = __riscv_vfmsub_vf_f16m2(resl, LOG_FACTOR_COEFF_F16, logRl, vl);
    rh = __riscv_vfmadd_vf_f16m2(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f16m2(res, __riscv_vfmsub_vf_f16m2(e, log2h, rh, vl), vl);
    rl = __riscv_vfadd_vv_f16m2(tmp, __riscv_vfmadd_vf_f16m2(rl, LOG_FACTOR_COEFF_F16, resl, vl), vl);
    res = __riscv_vfmadd_vf_f16m2(rl, LOG_INV_FACTOR_COEFF_F16, rh, vl);
    resl = __riscv_vfmadd_vf_f16m2(__riscv_vfsub_vv_f16m2(rh, res, vl), LOG_FACTOR_COEFF_F16, rl, vl);
}

static inline void my_log2_reconstruction_f16m2(vfloat16m2_t e, vfloat16m2_t zh, vfloat16m2_t zl,
    vfloat16m2_t p, vfloat16m2_t logR, vfloat16m2_t logRl, vfloat16m2_t& res, size_t vl)
{
    constexpr _Float16 invLog2h = 0x1.714p0f16, invLog2l = 0x1.478p-12f16*LOG_FACTOR_COEFF_F16;
    e = __riscv_vfmul_vf_f16m2(e, LOG_FACTOR_COEFF_F16, vl);

    vfloat16m2_t rh, rl, tmp, tmph, tmpl, sh, sl;

    tmp = __riscv_vfmadd_vv_f16m2(p, zh, __riscv_vfmadd_vv_f16m2(
        p, zl, __riscv_vfmul_vf_f16m2(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmph = __riscv_vfsub_vv_f16m2(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f16m2(zh, __riscv_vfadd_vv_f16m2(tmph, logR, vl), vl);
    tmpl = __riscv_vfmsub_vf_f16m2(tmpl, LOG_FACTOR_COEFF_F16, logRl, vl);
    tmp = __riscv_vfmadd_vf_f16m2(tmp, invLog2h, __riscv_vfmadd_vf_f16m2(tmpl, invLog2h,
        __riscv_vfmul_vf_f16m2(tmph, invLog2l, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f16m2(tmph, invLog2h, e, vl);
    sh = __riscv_vfsub_vv_f16m2(rh, e, vl);
    sl = __riscv_vfsub_vv_f16m2(rh, __riscv_vfadd_vv_f16m2(sh, e, vl), vl);
    rl = __riscv_vfmsub_vf_f16m2(tmph, invLog2h, sh, vl);
    res = __riscv_vfadd_vv_f16m2(rh, __riscv_vfmadd_vf_f16m2(tmp, LOG_INV_FACTOR_COEFF_F16,
        __riscv_vfsub_vv_f16m2(rl, sl, vl), vl), vl);
}

static inline void my_log1p_reconstruction_h_f16m2(vfloat16m2_t e, vfloat16m2_t zh,
    vfloat16m2_t zl, vfloat16m2_t p, vfloat16m2_t logR, vfloat16m2_t logRl, vfloat16m2_t& res, size_t vl)
{
    constexpr _Float16 log2h = 0x1.63p-1f16 * LOG_FACTOR_COEFF_F16;
    constexpr _Float16 log2l = (-0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16) * LOG_FACTOR_COEFF_F16;

    vfloat16m2_t rh, rl, tmp, tmph, tmpl, tmp1h, tmp1l, resl, zlf2;

#ifndef __FAST_MATH__
    vbool8_t mzlc = __riscv_vmflt_vf_f16m2_b8(abs_f16m2(zl, vl), (_Float16)0x1p-10f16, vl);
    vfloat16m2_t zlc = __riscv_vfadd_vf_f16m2_mu(mzlc, zl, zl, (_Float16)0x1p-10f16, vl);
#else
    vfloat16m2_t zlc = zl;
#endif
    p = __riscv_vfmul_vf_f16m2(p, LOG_FACTOR_COEFF_F16, vl);
    zlf2 = __riscv_vfmul_vf_f16m2(__riscv_vfmul_vf_f16m2(zl, LOG_FACTOR_COEFF_F16, vl), LOG_FACTOR_COEFF_F16, vl);

    tmp = __riscv_vfmadd_vv_f16m2(p, zl, zlf2, vl);
    tmpl = __riscv_vfmsac_vv_f16m2(__riscv_vfsub_vv_f16m2(tmp, zlf2, vl), p, zlc, vl);
    tmph = __riscv_vfmadd_vv_f16m2(p, zh, tmp, vl);
    tmpl = __riscv_vfadd_vv_f16m2(tmpl, __riscv_vfmsac_vv_f16m2(__riscv_vfsub_vv_f16m2(tmph, tmp, vl), p, zh, vl), vl);
    tmph = __riscv_vfmadd_vf_f16m2(tmph, LOG_INV_FACTOR_COEFF_F16, __riscv_vfmul_vf_f16m2(e, log2l, vl), vl);
    tmpl = __riscv_vfnmsac_vf_f16m2(tmpl, LOG_FACTOR_COEFF_F16, logRl, vl);
    tmp1h = __riscv_vfsub_vv_f16m2(zh, logR, vl);
    tmp1l = __riscv_vfsub_vv_f16m2(__riscv_vfsub_vv_f16m2(zh, tmp1h, vl), logR, vl);
    rh = __riscv_vfmadd_vf_f16m2(e, log2h, tmp1h, vl);
    rl = __riscv_vfadd_vv_f16m2(tmp1h, __riscv_vfmsub_vf_f16m2(e, log2h, rh, vl), vl);
    res = __riscv_vfmacc_vf_f16m2(rh, LOG_INV_FACTOR_COEFF_F16, tmph, vl);
    resl = __riscv_vfmadd_vf_f16m2(tmph, LOG_INV_FACTOR_COEFF_F16, __riscv_vfsub_vv_f16m2(rh, res, vl), vl);
    resl = __riscv_vfadd_vv_f16m2(resl, tmp1l, vl);
    resl = __riscv_vfadd_vv_f16m2(resl, rl, vl);
    resl = __riscv_vfmul_vf_f16m2(resl, LOG_FACTOR_COEFF_F16, vl);
    resl = __riscv_vfmacc_vf_f16m2(resl, LOG_INV_FACTOR_COEFF_F16, tmpl, vl);
    res = __riscv_vfmacc_vf_f16m2(res, LOG_INV_FACTOR_COEFF_F16, resl, vl);
}

static inline void my_log1p_process_linear_f16m2(const vfloat16m2_t& xm1, vfloat16m2_t& xm1c,
    vbool8_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f16m2_b8(abs_f16m2(xm1, vl), (_Float16)0x1.6ap-11f16, vl);
    xm1c = __riscv_vfmerge_vfm_f16m2(xm1, (_Float16)0.0f16, linearMask, vl);
}

static inline void my_log1p_set_fe_underflow_f16m2(vfloat16m2_t& res, size_t vl)
{
    res = __riscv_vfmadd_vf_f16m2(res, (_Float16)0x1p-14f16, res, vl);
}

// ==================================================================================================
// vfloat16m4_t
// ==================================================================================================

static inline void fast2sum_f16m4(vfloat16m4_t a, vfloat16m4_t b, vfloat16m4_t& s, vfloat16m4_t& t, size_t vl)
{
    s = __riscv_vfadd_vv_f16m4(a, b, vl);
    vfloat16m4_t b1 = __riscv_vfsub_vv_f16m4(s, a, vl);
    t = __riscv_vfsub_vv_f16m4(b, b1, vl);
}

static inline vfloat16m4_t mul21_f16m4(vfloat16m4_t ah, vfloat16m4_t al, _Float16 bh, _Float16 bl, size_t vl)
{
    constexpr _Float16 factorh = 0x1p2f16, invfactorh = 0x1p-2f16;
    constexpr _Float16 factorl = 0x1p15f16, invfactorl = 0x1p-15f16;
    constexpr _Float16 factorlmh = factorl*invfactorh;
    constexpr _Float16 factorlmhl = factorlmh*LOG_INV_FACTOR_COEFF_F16;

    vfloat16m4_t zh, zl;
    bh *= factorh; bl *= factorl;
    zh = __riscv_vfmul_vf_f16m4(ah, bh, vl);
    zl = __riscv_vfmul_vf_f16m4(__riscv_vfmsub_vf_f16m4(ah, bh, zh, vl), factorlmh, vl);
    zl = __riscv_vfadd_vv_f16m4(zl, __riscv_vfmadd_vf_f16m4(__riscv_vfmul_vf_f16m4(al, factorlmhl, vl),
        bh, __riscv_vfmul_vf_f16m4(ah, bl, vl), vl), vl);
    return __riscv_vfmadd_vf_f16m4(zl, invfactorl, __riscv_vfmul_vf_f16m4(zh, invfactorh, vl), vl);
}

static inline vfloat16m4_t abs_f16m4(vfloat16m4_t x, size_t vl)
{
    return __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vx_u16m4(
        __riscv_vreinterpret_v_f16m4_u16m4(x), 0x7fff, vl));
}

static inline void log_check_special_cases_f16m4(vfloat16m4_t& x, vfloat16m4_t& special,
    vbool4_t& specialMask, size_t vl, _Float16 leftDomainPoint = 0.0f16, _Float16 defaultArgValue = 1.0f16)
{
    uint16_t pinf = 0x7c00;
    uint16_t ninf = 0xfc00;
    uint16_t qnan = 0x7e00;
    vuint16m4_t absxui = __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(x),
        (uint16_t)0x7fff, vl);

    vbool4_t nanMask = __riscv_vmsgtu_vx_u16m4_b4(absxui, pinf, vl);
    x = __riscv_vfmerge_vfm_f16m4(x, defaultArgValue, nanMask, vl);

    vbool4_t snanMask = __riscv_vmsltu_vx_u16m4_b4(absxui, qnan, vl);
    vbool4_t infMask = __riscv_vmseq_vx_u16m4_b4(__riscv_vreinterpret_v_f16m4_u16m4(x), pinf, vl);
    vbool4_t leftDomainPointMask = __riscv_vmfeq_vf_f16m4_b4(x, leftDomainPoint, vl);
    vbool4_t invalidDomainMask = __riscv_vmflt_vf_f16m4_b4(x, leftDomainPoint, vl);

    special = __riscv_vfmerge_vfm_f16m4(x, *(_Float16*)(&pinf), infMask, vl);
    special = __riscv_vfmerge_vfm_f16m4(special, *(_Float16*)(&ninf), leftDomainPointMask, vl);
    special = __riscv_vfmerge_vfm_f16m4(special, *(_Float16*)(&qnan),
        __riscv_vmor_mm_b4(invalidDomainMask, nanMask, vl), vl);

    specialMask = __riscv_vmor_mm_b4(__riscv_vmor_mm_b4(leftDomainPointMask, invalidDomainMask, vl),
        __riscv_vmor_mm_b4(infMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b4(__riscv_vmor_mm_b4(
        __riscv_vmand_mm_b4(nanMask, snanMask, vl), invalidDomainMask, vl), vl))
        volatile _Float16 x1 = 0.0f16/0.0f16;
    if (__riscv_vcpop_m_b4(leftDomainPointMask, vl))
        volatile _Float16 x1 = 1.0f16/0.0f16;

    x = __riscv_vfmerge_vfm_f16m4(x, defaultArgValue, specialMask, vl);
}

static inline void my_log_argument_reduction_f16m4(vfloat16m4_t x, vfloat16m4_t& y,
    vint16m4_t& e, vfloat16m4_t& ef, vuint16m4_t& index, size_t vl)
{
    constexpr uint16_t magic_const_1 = 0x3950;
    constexpr size_t tshift = 10 - 4 - 1;

    vuint16m4_t xi = __riscv_vreinterpret_v_f16m4_u16m4(x);
    vbool4_t msubn = __riscv_vmsltu_vx_u16m4_b4(xi, 0x0400, vl);
    x = __riscv_vfmul_vf_f16m4_mu(msubn, x, x, (_Float16)0x1p10f16, vl);
    xi = __riscv_vreinterpret_v_f16m4_u16m4(x);

    vuint16m4_t magic_number = __riscv_vsub_vx_u16m4(xi, magic_const_1, vl);
    e = __riscv_vsra_vx_i16m4(__riscv_vreinterpret_v_u16m4_i16m4(magic_number), (size_t)10, vl);
    e = __riscv_vsub_vx_i16m4_mu(msubn, e, e, (int16_t)10, vl);
    xi = __riscv_vsub_vv_u16m4(xi, __riscv_vand_vx_u16m4(magic_number, (uint16_t)0xfc00, vl), vl);
    y = __riscv_vreinterpret_v_u16m4_f16m4(xi);

    index = __riscv_vsrl_vx_u16m4(__riscv_vand_vx_u16m4(magic_number, (uint16_t)0x03c0, vl), tshift, vl);
    ef = __riscv_vfcvt_f_x_v_f16m4(e, vl);
}

static inline void my_log_table_values_f16m4(vuint16m4_t index, vfloat16m4_t& r, vfloat16m4_t& logR,
    vfloat16m4_t& logRl, size_t vl)
{
    logR = __riscv_vloxei16_v_f16m4(table_half_log_R, index, vl);
    logRl = __riscv_vloxei16_v_f16m4(table_half_log_Rl, index, vl);
    r = __riscv_vloxei16_v_f16m4(table_half_R, index, vl);
}

static inline void my_log_second_reduction_f16m4(vfloat16m4_t y, vfloat16m4_t r,
    vfloat16m4_t& z, vfloat16m4_t& zh, vfloat16m4_t& zl, size_t vl)
{
    zh = __riscv_vfmul_vv_f16m4(y, r, vl);
    zl = __riscv_vfmsub_vv_f16m4(y, r, zh, vl);
    zh = __riscv_vfsub_vf_f16m4(zh, LOG_FACTOR_COEFF_F16, vl);
    z = __riscv_vfadd_vv_f16m4(zh, zl, vl);
}

static inline void my_log1p_second_reduction_f16m4(vfloat16m4_t xm1, vfloat16m4_t r,
    vint16m4_t e, vfloat16m4_t& z, vfloat16m4_t& zh, vfloat16m4_t& zl, size_t vl)
{
    vint16m4_t pow2mei = __riscv_vadd_vx_i16m4(__riscv_vneg_v_i16m4(e, vl), (int16_t)17, vl);
    vfloat16m4_t pow2me = __riscv_vreinterpret_v_i16m4_f16m4(__riscv_vsll_vx_i16m4(pow2mei, 10, vl));
    vfloat16m4_t c = __riscv_vfmul_vv_f16m4(pow2me, __riscv_vfmul_vf_f16m4(r, LOG_INV_FACTOR_COEFF_F16, vl), vl);
    constexpr _Float16 f = LOG_FACTOR_COEFF_F16 * 0.25f16;

    vfloat16m4_t tmph, tmpl, sh, sl;
    tmph = __riscv_vfmadd_vv_f16m4(xm1, c, c, vl);
    sh = __riscv_vfsub_vv_f16m4(tmph, c, vl);
    sl = __riscv_vfadd_vv_f16m4(sh, c, vl);
    sl = __riscv_vfsub_vv_f16m4(__riscv_vfsub_vv_f16m4(tmph, sl, vl),
        __riscv_vfadd_vv_f16m4(c, __riscv_vfsub_vv_f16m4(sh, sl, vl), vl), vl);
    tmpl = __riscv_vfmsub_vv_f16m4(xm1, c, sh, vl);
    zh = __riscv_vfsub_vf_f16m4(tmph, (_Float16)4.0f16, vl);
    zl = __riscv_vfsub_vv_f16m4(tmph, __riscv_vfadd_vf_f16m4(zh, (_Float16)4.0f16, vl), vl);
    zl = __riscv_vfadd_vv_f16m4(__riscv_vfsub_vv_f16m4(zl, sl, vl), tmpl, vl);

    zh = __riscv_vfmul_vf_f16m4(zh, f, vl);
    zl = __riscv_vfmul_vf_f16m4(zl, f, vl);
    z = __riscv_vfadd_vv_f16m4(zh, zl, vl);
}

static inline void my_log_approximation_f16m4(vfloat16m4_t z, vfloat16m4_t& res, size_t vl)
{
    constexpr _Float16 p0 = -0x1.004p-1f16;
    constexpr _Float16 p1invf = 0x1.564p-2f16 * LOG_INV_FACTOR_COEFF_F16;

    vfloat16m4_t tmp = __riscv_vfmadd_vf_f16m4(z, p1invf, __riscv_vfmv_v_f_f16m4(p0, vl), vl);
    res = __riscv_vfmul_vv_f16m4(z, tmp, vl);
}

static inline void my_log1p_approximation_f16m4(vfloat16m4_t z, vfloat16m4_t& resh, size_t vl)
{
    constexpr _Float16 p0 = -0x1p-1f16;
    constexpr _Float16 p1 = 0x1.56p-2f16;
    constexpr _Float16 p2invf = -0x1.008p-2f16 * LOG_INV_FACTOR_COEFF_F16;

    vfloat16m4_t tmp = __riscv_vfmadd_vf_f16m4(z, p2invf, __riscv_vfmv_v_f_f16m4(p1, vl), vl);
    tmp = __riscv_vfmadd_vv_f16m4(z, __riscv_vfmul_vf_f16m4(tmp, LOG_INV_FACTOR_COEFF_F16, vl),
        __riscv_vfmv_v_f_f16m4(p0, vl), vl);
    resh = __riscv_vfmul_vv_f16m4(z, tmp, vl);
}

static inline void my_log_reconstruction_h_f16m4(vfloat16m4_t e, vfloat16m4_t zh, vfloat16m4_t zl,
    vfloat16m4_t p, vfloat16m4_t logR, vfloat16m4_t logRl, vfloat16m4_t& res, size_t vl)
{
    constexpr _Float16 log2h = 0x1.63p-1f16 * LOG_FACTOR_COEFF_F16;
    constexpr _Float16 log2l = (-0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16) * LOG_FACTOR_COEFF_F16;

    vfloat16m4_t rh, rl, tmp, tmpl;

    tmp = __riscv_vfmadd_vv_f16m4(p, zh, __riscv_vfmadd_vv_f16m4(
        p, zl, __riscv_vfmul_vf_f16m4(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmp = __riscv_vfmadd_vf_f16m4(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f16m4(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f16m4(zh, __riscv_vfadd_vv_f16m4(res, logR, vl), vl);
    tmpl = __riscv_vfmsub_vf_f16m4(tmpl, LOG_FACTOR_COEFF_F16, logRl, vl);
    rh = __riscv_vfmadd_vf_f16m4(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f16m4(res, __riscv_vfmsub_vf_f16m4(e, log2h, rh, vl), vl);
    rl = __riscv_vfmadd_vf_f16m4(rl, LOG_FACTOR_COEFF_F16, tmpl, vl);
    res = __riscv_vfmadd_vf_f16m4(__riscv_vfadd_vv_f16m4(rl, tmp, vl), LOG_INV_FACTOR_COEFF_F16, rh, vl);
}

static inline void my_log_reconstruction_hl_f16m4(vfloat16m4_t e, vfloat16m4_t zh, vfloat16m4_t zl,
    vfloat16m4_t p, vfloat16m4_t logR, vfloat16m4_t logRl, vfloat16m4_t& res, vfloat16m4_t& resl, size_t vl)
{
    constexpr _Float16 log2h = 0x1.63p-1f16 * LOG_FACTOR_COEFF_F16;
    constexpr _Float16 log2l = (-0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16) * LOG_FACTOR_COEFF_F16;

    vfloat16m4_t rh, rl, tmp;

    tmp = __riscv_vfmadd_vv_f16m4(p, zh, __riscv_vfmadd_vv_f16m4(
        p, zl, __riscv_vfmul_vf_f16m4(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmp = __riscv_vfmadd_vf_f16m4(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f16m4(zh, logR, vl);
    resl = __riscv_vfsub_vv_f16m4(zh, __riscv_vfadd_vv_f16m4(res, logR, vl), vl);
    resl = __riscv_vfmsub_vf_f16m4(resl, LOG_FACTOR_COEFF_F16, logRl, vl);
    rh = __riscv_vfmadd_vf_f16m4(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f16m4(res, __riscv_vfmsub_vf_f16m4(e, log2h, rh, vl), vl);
    rl = __riscv_vfadd_vv_f16m4(tmp, __riscv_vfmadd_vf_f16m4(rl, LOG_FACTOR_COEFF_F16, resl, vl), vl);
    res = __riscv_vfmadd_vf_f16m4(rl, LOG_INV_FACTOR_COEFF_F16, rh, vl);
    resl = __riscv_vfmadd_vf_f16m4(__riscv_vfsub_vv_f16m4(rh, res, vl), LOG_FACTOR_COEFF_F16, rl, vl);
}

static inline void my_log2_reconstruction_f16m4(vfloat16m4_t e, vfloat16m4_t zh, vfloat16m4_t zl,
    vfloat16m4_t p, vfloat16m4_t logR, vfloat16m4_t logRl, vfloat16m4_t& res, size_t vl)
{
    constexpr _Float16 invLog2h = 0x1.714p0f16, invLog2l = 0x1.478p-12f16*LOG_FACTOR_COEFF_F16;
    e = __riscv_vfmul_vf_f16m4(e, LOG_FACTOR_COEFF_F16, vl);

    vfloat16m4_t rh, rl, tmp, tmph, tmpl, sh, sl;

    tmp = __riscv_vfmadd_vv_f16m4(p, zh, __riscv_vfmadd_vv_f16m4(
        p, zl, __riscv_vfmul_vf_f16m4(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmph = __riscv_vfsub_vv_f16m4(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f16m4(zh, __riscv_vfadd_vv_f16m4(tmph, logR, vl), vl);
    tmpl = __riscv_vfmsub_vf_f16m4(tmpl, LOG_FACTOR_COEFF_F16, logRl, vl);
    tmp = __riscv_vfmadd_vf_f16m4(tmp, invLog2h, __riscv_vfmadd_vf_f16m4(tmpl, invLog2h,
        __riscv_vfmul_vf_f16m4(tmph, invLog2l, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f16m4(tmph, invLog2h, e, vl);
    sh = __riscv_vfsub_vv_f16m4(rh, e, vl);
    sl = __riscv_vfsub_vv_f16m4(rh, __riscv_vfadd_vv_f16m4(sh, e, vl), vl);
    rl = __riscv_vfmsub_vf_f16m4(tmph, invLog2h, sh, vl);
    res = __riscv_vfadd_vv_f16m4(rh, __riscv_vfmadd_vf_f16m4(tmp, LOG_INV_FACTOR_COEFF_F16,
        __riscv_vfsub_vv_f16m4(rl, sl, vl), vl), vl);
}

static inline void my_log1p_reconstruction_h_f16m4(vfloat16m4_t e, vfloat16m4_t zh,
    vfloat16m4_t zl, vfloat16m4_t p, vfloat16m4_t logR, vfloat16m4_t logRl, vfloat16m4_t& res, size_t vl)
{
    constexpr _Float16 log2h = 0x1.63p-1f16 * LOG_FACTOR_COEFF_F16;
    constexpr _Float16 log2l = (-0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16) * LOG_FACTOR_COEFF_F16;

    vfloat16m4_t rh, rl, tmp, tmph, tmpl, tmp1h, tmp1l, resl, zlf2;

#ifndef __FAST_MATH__
    vbool4_t mzlc = __riscv_vmflt_vf_f16m4_b4(abs_f16m4(zl, vl), (_Float16)0x1p-10f16, vl);
    vfloat16m4_t zlc = __riscv_vfadd_vf_f16m4_mu(mzlc, zl, zl, (_Float16)0x1p-10f16, vl);
#else
    vfloat16m4_t zlc = zl;
#endif
    p = __riscv_vfmul_vf_f16m4(p, LOG_FACTOR_COEFF_F16, vl);
    zlf2 = __riscv_vfmul_vf_f16m4(__riscv_vfmul_vf_f16m4(zl, LOG_FACTOR_COEFF_F16, vl), LOG_FACTOR_COEFF_F16, vl);

    tmp = __riscv_vfmadd_vv_f16m4(p, zl, zlf2, vl);
    tmpl = __riscv_vfmsac_vv_f16m4(__riscv_vfsub_vv_f16m4(tmp, zlf2, vl), p, zlc, vl);
    tmph = __riscv_vfmadd_vv_f16m4(p, zh, tmp, vl);
    tmpl = __riscv_vfadd_vv_f16m4(tmpl, __riscv_vfmsac_vv_f16m4(__riscv_vfsub_vv_f16m4(tmph, tmp, vl), p, zh, vl), vl);
    tmph = __riscv_vfmadd_vf_f16m4(tmph, LOG_INV_FACTOR_COEFF_F16, __riscv_vfmul_vf_f16m4(e, log2l, vl), vl);
    tmpl = __riscv_vfnmsac_vf_f16m4(tmpl, LOG_FACTOR_COEFF_F16, logRl, vl);
    tmp1h = __riscv_vfsub_vv_f16m4(zh, logR, vl);
    tmp1l = __riscv_vfsub_vv_f16m4(__riscv_vfsub_vv_f16m4(zh, tmp1h, vl), logR, vl);
    rh = __riscv_vfmadd_vf_f16m4(e, log2h, tmp1h, vl);
    rl = __riscv_vfadd_vv_f16m4(tmp1h, __riscv_vfmsub_vf_f16m4(e, log2h, rh, vl), vl);
    res = __riscv_vfmacc_vf_f16m4(rh, LOG_INV_FACTOR_COEFF_F16, tmph, vl);
    resl = __riscv_vfmadd_vf_f16m4(tmph, LOG_INV_FACTOR_COEFF_F16, __riscv_vfsub_vv_f16m4(rh, res, vl), vl);
    resl = __riscv_vfadd_vv_f16m4(resl, tmp1l, vl);
    resl = __riscv_vfadd_vv_f16m4(resl, rl, vl);
    resl = __riscv_vfmul_vf_f16m4(resl, LOG_FACTOR_COEFF_F16, vl);
    resl = __riscv_vfmacc_vf_f16m4(resl, LOG_INV_FACTOR_COEFF_F16, tmpl, vl);
    res = __riscv_vfmacc_vf_f16m4(res, LOG_INV_FACTOR_COEFF_F16, resl, vl);
}

static inline void my_log1p_process_linear_f16m4(const vfloat16m4_t& xm1, vfloat16m4_t& xm1c,
    vbool4_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f16m4_b4(abs_f16m4(xm1, vl), (_Float16)0x1.6ap-11f16, vl);
    xm1c = __riscv_vfmerge_vfm_f16m4(xm1, (_Float16)0.0f16, linearMask, vl);
}

static inline void my_log1p_set_fe_underflow_f16m4(vfloat16m4_t& res, size_t vl)
{
    res = __riscv_vfmadd_vf_f16m4(res, (_Float16)0x1p-14f16, res, vl);
}
