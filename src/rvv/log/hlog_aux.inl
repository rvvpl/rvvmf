/*
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of
 * Nizhny Novgorod and its affiliates. All rights reserved.
 *
 * Copyright 2026 The RVVMF Authors (Vladislav Polozov,
 * Elena Panova)
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

static inline void decompose_number_f16m1(vfloat16m1_t x, vfloat16m1_t& m, vint16m1_t& e, size_t vl)
{
    vuint16m1_t resu = __riscv_vreinterpret_v_f16m1_u16m1(x);
    resu = __riscv_vand_vx_u16m1(resu, (uint16_t)0x7c00, vl);
    resu = __riscv_vsrl_vx_u16m1(resu, (uint16_t)10, vl);
    e = __riscv_vreinterpret_v_u16m1_i16m1(resu);
    e = __riscv_vsub_vx_i16m1(e, (int16_t)15, vl);

    vuint16m1_t xi = __riscv_vreinterpret_v_f16m1_u16m1(x);
    xi = __riscv_vand_vx_u16m1(xi, (uint16_t)0x03ff, vl);
    xi = __riscv_vor_vx_u16m1(xi, (uint16_t)0x3c00, vl);
    m = __riscv_vreinterpret_v_u16m1_f16m1(xi);
}

static inline void my_log1p_argument_reduction_f16m1(vfloat16m1_t x, vfloat16m1_t& y,
    vint16m1_t& e, vfloat16m1_t& ef, size_t vl)
{
    constexpr _Float16 ap1 = 0x1.58p0f16;

    vint16m1_t et;
    vfloat16m1_t yt;
    decompose_number_f16m1(x, y, e, vl);

    vbool16_t subnormalMask = __riscv_vmflt_vf_f16m1_b16(x, (_Float16)0x1p-14f16, vl);
    yt = __riscv_vfsub_vf_f16m1(y, (_Float16)1.0f16, vl);
    decompose_number_f16m1(yt, yt, et, vl);
    et = __riscv_vadd_vv_i16m1(__riscv_vadd_vx_i16m1(e, (int16_t)1, vl), et, vl);
    y = __riscv_vmerge_vvm_f16m1(y, yt, subnormalMask, vl);
    e = __riscv_vmerge_vvm_i16m1(e, et, subnormalMask, vl);

    vbool16_t mask = __riscv_vmfge_vf_f16m1_b16(y, ap1, vl);
    yt = __riscv_vfmul_vf_f16m1(y, (_Float16)0.5f16, vl);
    et = __riscv_vadd_vx_i16m1(e, (int16_t)1, vl);
    y = __riscv_vmerge_vvm_f16m1(y, yt, mask, vl);
    e = __riscv_vmerge_vvm_i16m1(e, et, mask, vl);

    ef = __riscv_vfcvt_f_x_v_f16m1(e, vl);
}

static inline void my_log_argument_reduction_f16m1(vfloat16m1_t x, vfloat16m1_t& y, vfloat16m1_t& ef, size_t vl)
{
    vint16m1_t e;
    my_log1p_argument_reduction_f16m1(x, y, e, ef, vl);
}

static inline void my_log_table_values_f16m1(vfloat16m1_t y, vfloat16m1_t& logR, vfloat16m1_t& logRl,
    vfloat16m1_t& r, vuint16m1_t& index, size_t vl)
{
    constexpr size_t k = 4;
    constexpr _Float16 u = 0x1.5ccp0f16, v = 0x1.58cp-3f16;
    constexpr size_t tshift = 10 - k;

    index = __riscv_vreinterpret_v_f16m1_u16m1(
        __riscv_vfmadd_vf_f16m1(y, u, __riscv_vfmv_v_f_f16m1(v, vl), vl));
    index = __riscv_vand_vx_u16m1(index, (uint16_t)0x03ff, vl);
    index = __riscv_vsrl_vx_u16m1(index, tshift, vl);

    index = __riscv_vmul_vx_u16m1(index, sizeof(_Float16), vl);

    logR = __riscv_vloxei16_v_f16m1(table_half_log_R, index, vl);
    logRl = __riscv_vloxei16_v_f16m1(table_half_log_Rl, index, vl);
    r = __riscv_vloxei16_v_f16m1(table_half_R, index, vl);
}

static inline void my_log_second_reduction_f16m1(vfloat16m1_t y, vfloat16m1_t& z,
    vfloat16m1_t& zh, vfloat16m1_t& zl, vfloat16m1_t& logR, vfloat16m1_t& logRl, size_t vl)
{
    vfloat16m1_t r;
    vuint16m1_t index;
    my_log_table_values_f16m1(y, logR, logRl, r, index, vl);

    zh = __riscv_vfmul_vv_f16m1(y, r, vl);
    zl = __riscv_vfmsub_vv_f16m1(y, r, zh, vl);
    zh = __riscv_vfsub_vf_f16m1(zh, LOG_FACTOR_COEFF_F16, vl);
    z = __riscv_vfadd_vv_f16m1(zh, zl, vl);
}

static inline void my_log1p_second_reduction_f16m1(vfloat16m1_t xm1, vfloat16m1_t y, vint16m1_t e,
    vfloat16m1_t& z, vfloat16m1_t& zh, vfloat16m1_t& zl, vfloat16m1_t& logR, vfloat16m1_t& logRl, size_t vl)
{
    vfloat16m1_t r;
    vuint16m1_t index;
    my_log_table_values_f16m1(y, logR, logRl, r, index, vl);

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

static inline void my_log1p_approximation_f16m1(vfloat16m1_t z, vfloat16m1_t& resh, size_t vl) {
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
    constexpr _Float16 log2l = -0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16;

    vfloat16m1_t rh, rl, tmp, tmpl;

    tmp = __riscv_vfmadd_vv_f16m1(p, zh, __riscv_vfmadd_vv_f16m1(
        p, zl, __riscv_vfmul_vf_f16m1(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmp = __riscv_vfmadd_vf_f16m1(e, log2l*LOG_FACTOR_COEFF_F16, tmp, vl);
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
    constexpr _Float16 log2l = -0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16;

    vfloat16m1_t rh, rl, tmp;

    tmp = __riscv_vfmadd_vv_f16m1(p, zh, __riscv_vfmadd_vv_f16m1(
        p, zl, __riscv_vfmul_vf_f16m1(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmp = __riscv_vfmadd_vf_f16m1(e, log2l*LOG_FACTOR_COEFF_F16, tmp, vl);
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
    constexpr _Float16 invLog2h = 0x1.714p0f16, invLog2l = 0x1.478p-12f16;
    e = __riscv_vfmul_vf_f16m1(e, LOG_FACTOR_COEFF_F16, vl);

    vfloat16m1_t rh, rl, tmp, tmph, tmpl, sh, sl;

    tmp = __riscv_vfmadd_vv_f16m1(p, zh, __riscv_vfmadd_vv_f16m1(
        p, zl, __riscv_vfmul_vf_f16m1(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmph = __riscv_vfsub_vv_f16m1(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f16m1(zh, __riscv_vfadd_vv_f16m1(tmph, logR, vl), vl);
    tmpl = __riscv_vfmsub_vf_f16m1(tmpl, LOG_FACTOR_COEFF_F16, logRl, vl);
    tmp = __riscv_vfmadd_vf_f16m1(tmp, invLog2h, __riscv_vfmadd_vf_f16m1(tmpl, invLog2h,
        __riscv_vfmul_vf_f16m1(tmph, invLog2l*LOG_FACTOR_COEFF_F16, vl), vl), vl);
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
    constexpr _Float16 addfactor = 0x1p3f16, invaddfactor = 0x1p-3f16;
    constexpr _Float16 complexf1 = LOG_INV_FACTOR_COEFF_F16*addfactor;
    constexpr _Float16 complexf2 = LOG_FACTOR_COEFF_F16*addfactor;
    constexpr _Float16 complexf3 = LOG_INV_FACTOR_COEFF_F16*invaddfactor;

    vfloat16m1_t rh, rl, tmp, tmph, tmpl, resl, zhf, zlf;

    zl = __riscv_vfmul_vf_f16m1(zl, LOG_FACTOR_COEFF_F16, vl);
    zhf = __riscv_vfmul_vf_f16m1(zh, LOG_FACTOR_COEFF_F16, vl);
    zlf = __riscv_vfmul_vf_f16m1(zl, LOG_FACTOR_COEFF_F16, vl);
    tmp = __riscv_vfmadd_vv_f16m1(p, zl, zlf, vl);
    tmpl = __riscv_vfmsub_vv_f16m1(p, zl, __riscv_vfsub_vv_f16m1(tmp, zlf, vl), vl);
    tmph = __riscv_vfmadd_vv_f16m1(p, zhf, tmp, vl);
    tmpl = __riscv_vfadd_vv_f16m1(tmpl, __riscv_vfmsub_vv_f16m1(
        p, zhf, __riscv_vfsub_vv_f16m1(tmph, tmp, vl), vl), vl);
    tmph = __riscv_vfmadd_vf_f16m1(e, log2l, __riscv_vfmul_vf_f16m1(tmph, LOG_INV_FACTOR_COEFF_F16, vl), vl);
    res = __riscv_vfsub_vv_f16m1(zh, logR, vl);
    tmpl = __riscv_vfmadd_vf_f16m1(__riscv_vfmsub_vf_f16m1(
        __riscv_vfsub_vv_f16m1(zh, res, vl), LOG_FACTOR_COEFF_F16,
        __riscv_vfmul_vf_f16m1(logR, LOG_FACTOR_COEFF_F16, vl), vl), LOG_FACTOR_COEFF_F16, tmpl, vl);
    tmpl = __riscv_vfnmsub_vf_f16m1(logRl, LOG_FACTOR_COEFF_F16, tmpl, vl);
    rh = __riscv_vfmadd_vf_f16m1(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f16m1(res, __riscv_vfmsub_vf_f16m1(e, log2h, rh, vl), vl);
    rl = __riscv_vfmadd_vf_f16m1(tmpl, complexf1, __riscv_vfmul_vf_f16m1(rl, complexf2, vl), vl);
    vfloat16m1_t tmp1h = __riscv_vfmadd_vf_f16m1(tmph, LOG_INV_FACTOR_COEFF_F16, rh, vl);
    vfloat16m1_t tmp1l = __riscv_vfmadd_vf_f16m1(tmph, LOG_INV_FACTOR_COEFF_F16, __riscv_vfsub_vv_f16m1(rh, tmp1h, vl), vl);
    res = __riscv_vfmadd_vf_f16m1(__riscv_vfmadd_vf_f16m1(tmp1l, complexf2, rl, vl), complexf3, tmp1h, vl);
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

static inline void decompose_number_f16m2(vfloat16m2_t x, vfloat16m2_t& m, vint16m2_t& e, size_t vl)
{
    vuint16m2_t resu = __riscv_vreinterpret_v_f16m2_u16m2(x);
    resu = __riscv_vand_vx_u16m2(resu, (uint16_t)0x7c00, vl);
    resu = __riscv_vsrl_vx_u16m2(resu, (uint16_t)10, vl);
    e = __riscv_vreinterpret_v_u16m2_i16m2(resu);
    e = __riscv_vsub_vx_i16m2(e, (int16_t)15, vl);

    vuint16m2_t xi = __riscv_vreinterpret_v_f16m2_u16m2(x);
    xi = __riscv_vand_vx_u16m2(xi, (uint16_t)0x03ff, vl);
    xi = __riscv_vor_vx_u16m2(xi, (uint16_t)0x3c00, vl);
    m = __riscv_vreinterpret_v_u16m2_f16m2(xi);
}

static inline void my_log1p_argument_reduction_f16m2(vfloat16m2_t x, vfloat16m2_t& y,
    vint16m2_t& e, vfloat16m2_t& ef, size_t vl)
{
    constexpr _Float16 ap1 = 0x1.58p0f16;

    vint16m2_t et;
    vfloat16m2_t yt;
    decompose_number_f16m2(x, y, e, vl);

    vbool8_t subnormalMask = __riscv_vmflt_vf_f16m2_b8(x, (_Float16)0x1p-14f16, vl);
    yt = __riscv_vfsub_vf_f16m2(y, (_Float16)1.0f16, vl);
    decompose_number_f16m2(yt, yt, et, vl);
    et = __riscv_vadd_vv_i16m2(__riscv_vadd_vx_i16m2(e, (int16_t)1, vl), et, vl);
    y = __riscv_vmerge_vvm_f16m2(y, yt, subnormalMask, vl);
    e = __riscv_vmerge_vvm_i16m2(e, et, subnormalMask, vl);

    vbool8_t mask = __riscv_vmfge_vf_f16m2_b8(y, ap1, vl);
    yt = __riscv_vfmul_vf_f16m2(y, (_Float16)0.5f16, vl);
    et = __riscv_vadd_vx_i16m2(e, (int16_t)1, vl);
    y = __riscv_vmerge_vvm_f16m2(y, yt, mask, vl);
    e = __riscv_vmerge_vvm_i16m2(e, et, mask, vl);

    ef = __riscv_vfcvt_f_x_v_f16m2(e, vl);
}

static inline void my_log_argument_reduction_f16m2(vfloat16m2_t x, vfloat16m2_t& y, vfloat16m2_t& ef, size_t vl)
{
    vint16m2_t e;
    my_log1p_argument_reduction_f16m2(x, y, e, ef, vl);
}

static inline void my_log_table_values_f16m2(vfloat16m2_t y, vfloat16m2_t& logR, vfloat16m2_t& logRl,
    vfloat16m2_t& r, vuint16m2_t& index, size_t vl)
{
    constexpr size_t k = 4;
    constexpr _Float16 u = 0x1.5ccp0f16, v = 0x1.58cp-3f16;
    constexpr size_t tshift = 10 - k;

    index = __riscv_vreinterpret_v_f16m2_u16m2(
        __riscv_vfmadd_vf_f16m2(y, u, __riscv_vfmv_v_f_f16m2(v, vl), vl));
    index = __riscv_vand_vx_u16m2(index, (uint16_t)0x03ff, vl);
    index = __riscv_vsrl_vx_u16m2(index, tshift, vl);

    index = __riscv_vmul_vx_u16m2(index, sizeof(_Float16), vl);

    logR = __riscv_vloxei16_v_f16m2(table_half_log_R, index, vl);
    logRl = __riscv_vloxei16_v_f16m2(table_half_log_Rl, index, vl);
    r = __riscv_vloxei16_v_f16m2(table_half_R, index, vl);
}

static inline void my_log_second_reduction_f16m2(vfloat16m2_t y, vfloat16m2_t& z,
    vfloat16m2_t& zh, vfloat16m2_t& zl, vfloat16m2_t& logR, vfloat16m2_t& logRl, size_t vl)
{
    vfloat16m2_t r;
    vuint16m2_t index;
    my_log_table_values_f16m2(y, logR, logRl, r, index, vl);

    zh = __riscv_vfmul_vv_f16m2(y, r, vl);
    zl = __riscv_vfmsub_vv_f16m2(y, r, zh, vl);
    zh = __riscv_vfsub_vf_f16m2(zh, LOG_FACTOR_COEFF_F16, vl);
    z = __riscv_vfadd_vv_f16m2(zh, zl, vl);
}

static inline void my_log1p_second_reduction_f16m2(vfloat16m2_t xm2, vfloat16m2_t y, vint16m2_t e,
    vfloat16m2_t& z, vfloat16m2_t& zh, vfloat16m2_t& zl, vfloat16m2_t& logR, vfloat16m2_t& logRl, size_t vl)
{
    vfloat16m2_t r;
    vuint16m2_t index;
    my_log_table_values_f16m2(y, logR, logRl, r, index, vl);

    vint16m2_t pow2mei = __riscv_vadd_vx_i16m2(__riscv_vneg_v_i16m2(e, vl), (int16_t)17, vl);
    vfloat16m2_t pow2me = __riscv_vreinterpret_v_i16m2_f16m2(__riscv_vsll_vx_i16m2(pow2mei, 10, vl));
    vfloat16m2_t c = __riscv_vfmul_vv_f16m2(pow2me, __riscv_vfmul_vf_f16m2(r, LOG_INV_FACTOR_COEFF_F16, vl), vl);
    constexpr _Float16 f = LOG_FACTOR_COEFF_F16 * 0.25f16;

    vfloat16m2_t tmph, tmpl, sh, sl;
    tmph = __riscv_vfmadd_vv_f16m2(xm2, c, c, vl);
    sh = __riscv_vfsub_vv_f16m2(tmph, c, vl);
    sl = __riscv_vfadd_vv_f16m2(sh, c, vl);
    sl = __riscv_vfsub_vv_f16m2(__riscv_vfsub_vv_f16m2(tmph, sl, vl),
        __riscv_vfadd_vv_f16m2(c, __riscv_vfsub_vv_f16m2(sh, sl, vl), vl), vl);
    tmpl = __riscv_vfmsub_vv_f16m2(xm2, c, sh, vl);
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

static inline void my_log1p_approximation_f16m2(vfloat16m2_t z, vfloat16m2_t& resh, size_t vl) {
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
    constexpr _Float16 log2l = -0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16;

    vfloat16m2_t rh, rl, tmp, tmpl;

    tmp = __riscv_vfmadd_vv_f16m2(p, zh, __riscv_vfmadd_vv_f16m2(
        p, zl, __riscv_vfmul_vf_f16m2(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmp = __riscv_vfmadd_vf_f16m2(e, log2l*LOG_FACTOR_COEFF_F16, tmp, vl);
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
    constexpr _Float16 log2l = -0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16;

    vfloat16m2_t rh, rl, tmp;

    tmp = __riscv_vfmadd_vv_f16m2(p, zh, __riscv_vfmadd_vv_f16m2(
        p, zl, __riscv_vfmul_vf_f16m2(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmp = __riscv_vfmadd_vf_f16m2(e, log2l*LOG_FACTOR_COEFF_F16, tmp, vl);
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
    constexpr _Float16 invLog2h = 0x1.714p0f16, invLog2l = 0x1.478p-12f16;
    e = __riscv_vfmul_vf_f16m2(e, LOG_FACTOR_COEFF_F16, vl);

    vfloat16m2_t rh, rl, tmp, tmph, tmpl, sh, sl;

    tmp = __riscv_vfmadd_vv_f16m2(p, zh, __riscv_vfmadd_vv_f16m2(
        p, zl, __riscv_vfmul_vf_f16m2(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmph = __riscv_vfsub_vv_f16m2(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f16m2(zh, __riscv_vfadd_vv_f16m2(tmph, logR, vl), vl);
    tmpl = __riscv_vfmsub_vf_f16m2(tmpl, LOG_FACTOR_COEFF_F16, logRl, vl);
    tmp = __riscv_vfmadd_vf_f16m2(tmp, invLog2h, __riscv_vfmadd_vf_f16m2(tmpl, invLog2h,
        __riscv_vfmul_vf_f16m2(tmph, invLog2l*LOG_FACTOR_COEFF_F16, vl), vl), vl);
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
    constexpr _Float16 addfactor = 0x1p3f16, invaddfactor = 0x1p-3f16;
    constexpr _Float16 complexf1 = LOG_INV_FACTOR_COEFF_F16*addfactor;
    constexpr _Float16 complexf2 = LOG_FACTOR_COEFF_F16*addfactor;
    constexpr _Float16 complexf3 = LOG_INV_FACTOR_COEFF_F16*invaddfactor;

    vfloat16m2_t rh, rl, tmp, tmph, tmpl, resl, zhf, zlf;

    zl = __riscv_vfmul_vf_f16m2(zl, LOG_FACTOR_COEFF_F16, vl);
    zhf = __riscv_vfmul_vf_f16m2(zh, LOG_FACTOR_COEFF_F16, vl);
    zlf = __riscv_vfmul_vf_f16m2(zl, LOG_FACTOR_COEFF_F16, vl);
    tmp = __riscv_vfmadd_vv_f16m2(p, zl, zlf, vl);
    tmpl = __riscv_vfmsub_vv_f16m2(p, zl, __riscv_vfsub_vv_f16m2(tmp, zlf, vl), vl);
    tmph = __riscv_vfmadd_vv_f16m2(p, zhf, tmp, vl);
    tmpl = __riscv_vfadd_vv_f16m2(tmpl, __riscv_vfmsub_vv_f16m2(
        p, zhf, __riscv_vfsub_vv_f16m2(tmph, tmp, vl), vl), vl);
    tmph = __riscv_vfmadd_vf_f16m2(e, log2l, __riscv_vfmul_vf_f16m2(tmph, LOG_INV_FACTOR_COEFF_F16, vl), vl);
    res = __riscv_vfsub_vv_f16m2(zh, logR, vl);
    tmpl = __riscv_vfmadd_vf_f16m2(__riscv_vfmsub_vf_f16m2(
        __riscv_vfsub_vv_f16m2(zh, res, vl), LOG_FACTOR_COEFF_F16,
        __riscv_vfmul_vf_f16m2(logR, LOG_FACTOR_COEFF_F16, vl), vl), LOG_FACTOR_COEFF_F16, tmpl, vl);
    tmpl = __riscv_vfnmsub_vf_f16m2(logRl, LOG_FACTOR_COEFF_F16, tmpl, vl);
    rh = __riscv_vfmadd_vf_f16m2(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f16m2(res, __riscv_vfmsub_vf_f16m2(e, log2h, rh, vl), vl);
    rl = __riscv_vfmadd_vf_f16m2(tmpl, complexf1, __riscv_vfmul_vf_f16m2(rl, complexf2, vl), vl);
    vfloat16m2_t tmp1h = __riscv_vfmadd_vf_f16m2(tmph, LOG_INV_FACTOR_COEFF_F16, rh, vl);
    vfloat16m2_t tmp1l = __riscv_vfmadd_vf_f16m2(tmph, LOG_INV_FACTOR_COEFF_F16, __riscv_vfsub_vv_f16m2(rh, tmp1h, vl), vl);
    res = __riscv_vfmadd_vf_f16m2(__riscv_vfmadd_vf_f16m2(tmp1l, complexf2, rl, vl), complexf3, tmp1h, vl);
}

static inline void my_log1p_process_linear_f16m2(const vfloat16m2_t& xm2, vfloat16m2_t& xm2c,
    vbool8_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f16m2_b8(abs_f16m2(xm2, vl), (_Float16)0x1.6ap-11f16, vl);
    xm2c = __riscv_vfmerge_vfm_f16m2(xm2, (_Float16)0.0f16, linearMask, vl);
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

static inline void decompose_number_f16m4(vfloat16m4_t x, vfloat16m4_t& m, vint16m4_t& e, size_t vl)
{
    vuint16m4_t resu = __riscv_vreinterpret_v_f16m4_u16m4(x);
    resu = __riscv_vand_vx_u16m4(resu, (uint16_t)0x7c00, vl);
    resu = __riscv_vsrl_vx_u16m4(resu, (uint16_t)10, vl);
    e = __riscv_vreinterpret_v_u16m4_i16m4(resu);
    e = __riscv_vsub_vx_i16m4(e, (int16_t)15, vl);

    vuint16m4_t xi = __riscv_vreinterpret_v_f16m4_u16m4(x);
    xi = __riscv_vand_vx_u16m4(xi, (uint16_t)0x03ff, vl);
    xi = __riscv_vor_vx_u16m4(xi, (uint16_t)0x3c00, vl);
    m = __riscv_vreinterpret_v_u16m4_f16m4(xi);
}

static inline void my_log1p_argument_reduction_f16m4(vfloat16m4_t x, vfloat16m4_t& y,
    vint16m4_t& e, vfloat16m4_t& ef, size_t vl)
{
    constexpr _Float16 ap1 = 0x1.58p0f16;

    vint16m4_t et;
    vfloat16m4_t yt;
    decompose_number_f16m4(x, y, e, vl);

    vbool4_t subnormalMask = __riscv_vmflt_vf_f16m4_b4(x, (_Float16)0x1p-14f16, vl);
    yt = __riscv_vfsub_vf_f16m4(y, (_Float16)1.0f16, vl);
    decompose_number_f16m4(yt, yt, et, vl);
    et = __riscv_vadd_vv_i16m4(__riscv_vadd_vx_i16m4(e, (int16_t)1, vl), et, vl);
    y = __riscv_vmerge_vvm_f16m4(y, yt, subnormalMask, vl);
    e = __riscv_vmerge_vvm_i16m4(e, et, subnormalMask, vl);

    vbool4_t mask = __riscv_vmfge_vf_f16m4_b4(y, ap1, vl);
    yt = __riscv_vfmul_vf_f16m4(y, (_Float16)0.5f16, vl);
    et = __riscv_vadd_vx_i16m4(e, (int16_t)1, vl);
    y = __riscv_vmerge_vvm_f16m4(y, yt, mask, vl);
    e = __riscv_vmerge_vvm_i16m4(e, et, mask, vl);

    ef = __riscv_vfcvt_f_x_v_f16m4(e, vl);
}

static inline void my_log_argument_reduction_f16m4(vfloat16m4_t x, vfloat16m4_t& y, vfloat16m4_t& ef, size_t vl)
{
    vint16m4_t e;
    my_log1p_argument_reduction_f16m4(x, y, e, ef, vl);
}

static inline void my_log_table_values_f16m4(vfloat16m4_t y, vfloat16m4_t& logR, vfloat16m4_t& logRl,
    vfloat16m4_t& r, vuint16m4_t& index, size_t vl)
{
    constexpr size_t k = 4;
    constexpr _Float16 u = 0x1.5ccp0f16, v = 0x1.58cp-3f16;
    constexpr size_t tshift = 10 - k;

    index = __riscv_vreinterpret_v_f16m4_u16m4(
        __riscv_vfmadd_vf_f16m4(y, u, __riscv_vfmv_v_f_f16m4(v, vl), vl));
    index = __riscv_vand_vx_u16m4(index, (uint16_t)0x03ff, vl);
    index = __riscv_vsrl_vx_u16m4(index, tshift, vl);

    index = __riscv_vmul_vx_u16m4(index, sizeof(_Float16), vl);

    logR = __riscv_vloxei16_v_f16m4(table_half_log_R, index, vl);
    logRl = __riscv_vloxei16_v_f16m4(table_half_log_Rl, index, vl);
    r = __riscv_vloxei16_v_f16m4(table_half_R, index, vl);
}

static inline void my_log_second_reduction_f16m4(vfloat16m4_t y, vfloat16m4_t& z,
    vfloat16m4_t& zh, vfloat16m4_t& zl, vfloat16m4_t& logR, vfloat16m4_t& logRl, size_t vl)
{
    vfloat16m4_t r;
    vuint16m4_t index;
    my_log_table_values_f16m4(y, logR, logRl, r, index, vl);

    zh = __riscv_vfmul_vv_f16m4(y, r, vl);
    zl = __riscv_vfmsub_vv_f16m4(y, r, zh, vl);
    zh = __riscv_vfsub_vf_f16m4(zh, LOG_FACTOR_COEFF_F16, vl);
    z = __riscv_vfadd_vv_f16m4(zh, zl, vl);
}

static inline void my_log1p_second_reduction_f16m4(vfloat16m4_t xm4, vfloat16m4_t y, vint16m4_t e,
    vfloat16m4_t& z, vfloat16m4_t& zh, vfloat16m4_t& zl, vfloat16m4_t& logR, vfloat16m4_t& logRl, size_t vl)
{
    vfloat16m4_t r;
    vuint16m4_t index;
    my_log_table_values_f16m4(y, logR, logRl, r, index, vl);

    vint16m4_t pow2mei = __riscv_vadd_vx_i16m4(__riscv_vneg_v_i16m4(e, vl), (int16_t)17, vl);
    vfloat16m4_t pow2me = __riscv_vreinterpret_v_i16m4_f16m4(__riscv_vsll_vx_i16m4(pow2mei, 10, vl));
    vfloat16m4_t c = __riscv_vfmul_vv_f16m4(pow2me, __riscv_vfmul_vf_f16m4(r, LOG_INV_FACTOR_COEFF_F16, vl), vl);
    constexpr _Float16 f = LOG_FACTOR_COEFF_F16 * 0.25f16;

    vfloat16m4_t tmph, tmpl, sh, sl;
    tmph = __riscv_vfmadd_vv_f16m4(xm4, c, c, vl);
    sh = __riscv_vfsub_vv_f16m4(tmph, c, vl);
    sl = __riscv_vfadd_vv_f16m4(sh, c, vl);
    sl = __riscv_vfsub_vv_f16m4(__riscv_vfsub_vv_f16m4(tmph, sl, vl),
        __riscv_vfadd_vv_f16m4(c, __riscv_vfsub_vv_f16m4(sh, sl, vl), vl), vl);
    tmpl = __riscv_vfmsub_vv_f16m4(xm4, c, sh, vl);
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

static inline void my_log1p_approximation_f16m4(vfloat16m4_t z, vfloat16m4_t& resh, size_t vl) {
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
    constexpr _Float16 log2l = -0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16;

    vfloat16m4_t rh, rl, tmp, tmpl;

    tmp = __riscv_vfmadd_vv_f16m4(p, zh, __riscv_vfmadd_vv_f16m4(
        p, zl, __riscv_vfmul_vf_f16m4(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmp = __riscv_vfmadd_vf_f16m4(e, log2l*LOG_FACTOR_COEFF_F16, tmp, vl);
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
    constexpr _Float16 log2l = -0x1.bdp-13f16 * LOG_FACTOR_COEFF_F16;

    vfloat16m4_t rh, rl, tmp;

    tmp = __riscv_vfmadd_vv_f16m4(p, zh, __riscv_vfmadd_vv_f16m4(
        p, zl, __riscv_vfmul_vf_f16m4(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmp = __riscv_vfmadd_vf_f16m4(e, log2l*LOG_FACTOR_COEFF_F16, tmp, vl);
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
    constexpr _Float16 invLog2h = 0x1.714p0f16, invLog2l = 0x1.478p-12f16;
    e = __riscv_vfmul_vf_f16m4(e, LOG_FACTOR_COEFF_F16, vl);

    vfloat16m4_t rh, rl, tmp, tmph, tmpl, sh, sl;

    tmp = __riscv_vfmadd_vv_f16m4(p, zh, __riscv_vfmadd_vv_f16m4(
        p, zl, __riscv_vfmul_vf_f16m4(zl, LOG_FACTOR_COEFF_F16, vl), vl), vl);
    tmph = __riscv_vfsub_vv_f16m4(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f16m4(zh, __riscv_vfadd_vv_f16m4(tmph, logR, vl), vl);
    tmpl = __riscv_vfmsub_vf_f16m4(tmpl, LOG_FACTOR_COEFF_F16, logRl, vl);
    tmp = __riscv_vfmadd_vf_f16m4(tmp, invLog2h, __riscv_vfmadd_vf_f16m4(tmpl, invLog2h,
        __riscv_vfmul_vf_f16m4(tmph, invLog2l*LOG_FACTOR_COEFF_F16, vl), vl), vl);
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
    constexpr _Float16 addfactor = 0x1p3f16, invaddfactor = 0x1p-3f16;
    constexpr _Float16 complexf1 = LOG_INV_FACTOR_COEFF_F16*addfactor;
    constexpr _Float16 complexf2 = LOG_FACTOR_COEFF_F16*addfactor;
    constexpr _Float16 complexf3 = LOG_INV_FACTOR_COEFF_F16*invaddfactor;

    vfloat16m4_t rh, rl, tmp, tmph, tmpl, resl, zhf, zlf;

    zl = __riscv_vfmul_vf_f16m4(zl, LOG_FACTOR_COEFF_F16, vl);
    zhf = __riscv_vfmul_vf_f16m4(zh, LOG_FACTOR_COEFF_F16, vl);
    zlf = __riscv_vfmul_vf_f16m4(zl, LOG_FACTOR_COEFF_F16, vl);
    tmp = __riscv_vfmadd_vv_f16m4(p, zl, zlf, vl);
    tmpl = __riscv_vfmsub_vv_f16m4(p, zl, __riscv_vfsub_vv_f16m4(tmp, zlf, vl), vl);
    tmph = __riscv_vfmadd_vv_f16m4(p, zhf, tmp, vl);
    tmpl = __riscv_vfadd_vv_f16m4(tmpl, __riscv_vfmsub_vv_f16m4(
        p, zhf, __riscv_vfsub_vv_f16m4(tmph, tmp, vl), vl), vl);
    tmph = __riscv_vfmadd_vf_f16m4(e, log2l, __riscv_vfmul_vf_f16m4(tmph, LOG_INV_FACTOR_COEFF_F16, vl), vl);
    res = __riscv_vfsub_vv_f16m4(zh, logR, vl);
    tmpl = __riscv_vfmadd_vf_f16m4(__riscv_vfmsub_vf_f16m4(
        __riscv_vfsub_vv_f16m4(zh, res, vl), LOG_FACTOR_COEFF_F16,
        __riscv_vfmul_vf_f16m4(logR, LOG_FACTOR_COEFF_F16, vl), vl), LOG_FACTOR_COEFF_F16, tmpl, vl);
    tmpl = __riscv_vfnmsub_vf_f16m4(logRl, LOG_FACTOR_COEFF_F16, tmpl, vl);
    rh = __riscv_vfmadd_vf_f16m4(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f16m4(res, __riscv_vfmsub_vf_f16m4(e, log2h, rh, vl), vl);
    rl = __riscv_vfmadd_vf_f16m4(tmpl, complexf1, __riscv_vfmul_vf_f16m4(rl, complexf2, vl), vl);
    vfloat16m4_t tmp1h = __riscv_vfmadd_vf_f16m4(tmph, LOG_INV_FACTOR_COEFF_F16, rh, vl);
    vfloat16m4_t tmp1l = __riscv_vfmadd_vf_f16m4(tmph, LOG_INV_FACTOR_COEFF_F16, __riscv_vfsub_vv_f16m4(rh, tmp1h, vl), vl);
    res = __riscv_vfmadd_vf_f16m4(__riscv_vfmadd_vf_f16m4(tmp1l, complexf2, rl, vl), complexf3, tmp1h, vl);
}

static inline void my_log1p_process_linear_f16m4(const vfloat16m4_t& xm4, vfloat16m4_t& xm4c,
    vbool4_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f16m4_b4(abs_f16m4(xm4, vl), (_Float16)0x1.6ap-11f16, vl);
    xm4c = __riscv_vfmerge_vfm_f16m4(xm4, (_Float16)0.0f16, linearMask, vl);
}

static inline void my_log1p_set_fe_underflow_f16m4(vfloat16m4_t& res, size_t vl)
{
    res = __riscv_vfmadd_vf_f16m4(res, (_Float16)0x1p-14f16, res, vl);
}
