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
 *   File:  flog_aux.inl                                 *
 *   Contains: helper built-in functions for log, log2   *
 *             log10, log1p functions (float32_t)        *
 *                                                       *
 *                                                       *
 *********************************************************
*/

// ==================================================================================================
// vfloat32m1_t
// ==================================================================================================

static inline void fast2sum_f32m1(vfloat32m1_t a, vfloat32m1_t b, vfloat32m1_t& s, vfloat32m1_t& t, size_t vl)
{
    s = __riscv_vfadd_vv_f32m1(a, b, vl);
    vfloat32m1_t b1 = __riscv_vfsub_vv_f32m1(s, a, vl);
    t = __riscv_vfsub_vv_f32m1(b, b1, vl);
}

static inline vfloat32m1_t mul21_f32m1(vfloat32m1_t ah, vfloat32m1_t al, float bh, float bl, size_t vl)
{
    vfloat32m1_t zh, zl;
    zh = __riscv_vfmul_vf_f32m1(ah, bh, vl);
    zl = __riscv_vfmsub_vf_f32m1(ah, bh, zh, vl);
    zl = __riscv_vfadd_vv_f32m1(zl, __riscv_vfmadd_vf_f32m1(ah, bl,
        __riscv_vfmul_vf_f32m1(al, bh, vl), vl), vl);
    return __riscv_vfadd_vv_f32m1(zh, zl, vl);
}

static inline vfloat32m1_t abs_f32m1(vfloat32m1_t x, size_t vl)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vx_u32m1(
        __riscv_vreinterpret_v_f32m1_u32m1(x), 0x7fffffff, vl));
}

static inline void log_check_special_cases_f32m1(vfloat32m1_t& x, vfloat32m1_t& special,
    vbool32_t& specialMask, size_t vl, float leftDomainPoint = 0.0f, float defaultArgValue = 1.0f)
{
    uint32_t pinf = 0x7f800000;
    uint32_t ninf = 0xff800000;
    uint32_t qnan = 0x7fc00000;
    vuint32m1_t absxui = __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(x),
        (uint32_t)0x7fffffff, vl);

    vbool32_t nanMask = __riscv_vmsgtu_vx_u32m1_b32(absxui, pinf, vl);
    x = __riscv_vfmerge_vfm_f32m1(x, defaultArgValue, nanMask, vl);

    vbool32_t snanMask = __riscv_vmsltu_vx_u32m1_b32(absxui, qnan, vl);
    vbool32_t infMask = __riscv_vmseq_vx_u32m1_b32(__riscv_vreinterpret_v_f32m1_u32m1(x), pinf, vl);
    vbool32_t leftDomainPointMask = __riscv_vmfeq_vf_f32m1_b32(x, leftDomainPoint, vl);
    vbool32_t invalidDomainMask = __riscv_vmflt_vf_f32m1_b32(x, leftDomainPoint, vl);

    special = __riscv_vfmerge_vfm_f32m1(x, *(float*)(&pinf), infMask, vl);
    special = __riscv_vfmerge_vfm_f32m1(special, *(float*)(&ninf), leftDomainPointMask, vl);
    special = __riscv_vfmerge_vfm_f32m1(special, *(float*)(&qnan),
        __riscv_vmor_mm_b32(invalidDomainMask, nanMask, vl), vl);

    specialMask = __riscv_vmor_mm_b32(__riscv_vmor_mm_b32(leftDomainPointMask, invalidDomainMask, vl),
        __riscv_vmor_mm_b32(infMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b32(__riscv_vmor_mm_b32(
        __riscv_vmand_mm_b32(nanMask, snanMask, vl), invalidDomainMask, vl), vl))
        volatile float x1 = 0.0f/0.0f;
    if (__riscv_vcpop_m_b32(leftDomainPointMask, vl))
        volatile float x1 = 1.0f/0.0f;

    x = __riscv_vfmerge_vfm_f32m1(x, defaultArgValue, specialMask, vl);
}

static inline void decompose_number_f32m1(vfloat32m1_t x, vfloat32m1_t& m, vint32m1_t& e, size_t vl)
{
    // get exponent
    vuint32m1_t resu = __riscv_vreinterpret_v_f32m1_u32m1(x);
    resu = __riscv_vand_vx_u32m1(resu, (uint32_t)0x7f800000, vl);
    resu = __riscv_vsrl_vx_u32m1(resu, (uint32_t)23, vl);
    e = __riscv_vreinterpret_v_u32m1_i32m1(resu);
    e = __riscv_vsub_vx_i32m1(e, (int32_t)127, vl);

    // get significand
    vuint32m1_t xi = __riscv_vreinterpret_v_f32m1_u32m1(x);
    xi = __riscv_vand_vx_u32m1(xi, (uint32_t)0x007fffff, vl);
    xi = __riscv_vor_vx_u32m1(xi, (uint32_t)0x3f800000, vl);
    m = __riscv_vreinterpret_v_u32m1_f32m1(xi);
}

static inline void my_log1p_argument_reduction_f32m1(vfloat32m1_t x, vfloat32m1_t& y,
    vint32m1_t& e, vfloat32m1_t& ef, size_t vl)
{
    constexpr float ap1 = 0x1.56p0f;

    vint32m1_t et;
    vfloat32m1_t yt;
    decompose_number_f32m1(x, y, e, vl);

    vbool32_t subnormalMask = __riscv_vmflt_vf_f32m1_b32(x, (float)0x1p-126f, vl);
    yt = __riscv_vfsub_vf_f32m1(y, (float)1.0f, vl);
    decompose_number_f32m1(yt, yt, et, vl);
    et = __riscv_vadd_vv_i32m1(__riscv_vadd_vx_i32m1(e, (int32_t)1, vl), et, vl);
    y = __riscv_vmerge_vvm_f32m1(y, yt, subnormalMask, vl);
    e = __riscv_vmerge_vvm_i32m1(e, et, subnormalMask, vl);

    vbool32_t mask = __riscv_vmfge_vf_f32m1_b32(y, ap1, vl);
    yt = __riscv_vfmul_vf_f32m1(y, (float)0.5f, vl);
    et = __riscv_vadd_vx_i32m1(e, (int32_t)1, vl);
    y = __riscv_vmerge_vvm_f32m1(y, yt, mask, vl);
    e = __riscv_vmerge_vvm_i32m1(e, et, mask, vl);

    ef = __riscv_vfcvt_f_x_v_f32m1(e, vl);
}

static inline void my_log_argument_reduction_f32m1(vfloat32m1_t x, vfloat32m1_t& y, vfloat32m1_t& ef, size_t vl)
{
    vint32m1_t e;
    my_log1p_argument_reduction_f32m1(x, y, e, ef, vl);
}

static inline void my_log_table_values_f32m1(vfloat32m1_t y, vfloat32m1_t& logR, vfloat32m1_t& r, size_t vl)
{
    constexpr size_t k = 6;
    constexpr float u = 0x1.7711dcp0f, v = 0x1.5dc48cp-5f;
    constexpr size_t tshift = 23 - k;

    vuint32m1_t index = __riscv_vreinterpret_v_f32m1_u32m1(
        __riscv_vfmadd_vf_f32m1(y, u, __riscv_vfmv_v_f_f32m1(v, vl), vl));
    index = __riscv_vand_vx_u32m1(index, (uint32_t)0x007fffff, vl);
    index = __riscv_vsrl_vx_u32m1(index, tshift, vl);

    index = __riscv_vmul_vx_u32m1(index, sizeof(float), vl);

    logR = __riscv_vloxei32_v_f32m1(table_float_log_R, index, vl);
    r = __riscv_vloxei32_v_f32m1(table_float_R, index, vl);
}

static inline void my_log_second_reduction_f32m1(vfloat32m1_t y, vfloat32m1_t& z,
    vfloat32m1_t& zh, vfloat32m1_t& zl, vfloat32m1_t& logR, size_t vl)
{
    vfloat32m1_t r;
    my_log_table_values_f32m1(y, logR, r, vl);

    zh = __riscv_vfmul_vv_f32m1(y, r, vl);
    zl = __riscv_vfmsub_vv_f32m1(y, r, zh, vl);
    zh = __riscv_vfsub_vf_f32m1(zh, (float)1.0f, vl);
    z = __riscv_vfadd_vv_f32m1(zh, zl, vl);
}

static inline void my_log1p_second_reduction_f32m1(vfloat32m1_t xm1, vfloat32m1_t y, vint32m1_t e,
    vfloat32m1_t& z, vfloat32m1_t& zh, vfloat32m1_t& zl, vfloat32m1_t& logR, size_t vl)
{
    vfloat32m1_t r;
    my_log_table_values_f32m1(y, logR, r, vl);

    vint32m1_t pow2mei = __riscv_vadd_vx_i32m1(__riscv_vneg_v_i32m1(e, vl), (int32_t)129, vl);
    vfloat32m1_t pow2me = __riscv_vreinterpret_v_i32m1_f32m1(__riscv_vsll_vx_i32m1(pow2mei, 23, vl));
    vfloat32m1_t c = __riscv_vfmul_vv_f32m1(pow2me, r, vl);

    vfloat32m1_t tmph, tmpl, sh, sl;
    tmph = __riscv_vfmadd_vv_f32m1(xm1, c, c, vl);
    sh = __riscv_vfsub_vv_f32m1(tmph, c, vl);
    sl = __riscv_vfadd_vv_f32m1(sh, c, vl);
    sl = __riscv_vfsub_vv_f32m1(__riscv_vfsub_vv_f32m1(tmph, sl, vl),
        __riscv_vfadd_vv_f32m1(c, __riscv_vfsub_vv_f32m1(sh, sl, vl), vl), vl);
    tmpl = __riscv_vfmsub_vv_f32m1(xm1, c, sh, vl);
    zh = __riscv_vfsub_vf_f32m1(tmph, (float)4.0f, vl);
    zl = __riscv_vfsub_vv_f32m1(tmph, __riscv_vfadd_vf_f32m1(zh, (float)4.0f, vl), vl);
    zl = __riscv_vfadd_vv_f32m1(__riscv_vfsub_vv_f32m1(zl, sl, vl), tmpl, vl);

    zh = __riscv_vfmul_vf_f32m1(zh, (float)0.25f, vl);
    zl = __riscv_vfmul_vf_f32m1(zl, (float)0.25f, vl);

    z = __riscv_vfadd_vv_f32m1(zh, zl, vl);
}

static inline void my_log_approximation_f32m1(vfloat32m1_t z, vfloat32m1_t& res, size_t vl)
{
    constexpr float p0 = -0x1p-1f;
    constexpr float p1 = 0x1.555ffp-2f;
    constexpr float p2 = -0x1.0009cp-2f;
    constexpr float shift_avoid_underflow = 0x1p-41f;

    vfloat32m1_t zs = __riscv_vfadd_vf_f32m1(z, shift_avoid_underflow, vl);
    vfloat32m1_t sqrz = __riscv_vfmul_vv_f32m1(zs, zs, vl);
    vfloat32m1_t odd = __riscv_vfmadd_vf_f32m1(sqrz, p2, __riscv_vfmv_v_f_f32m1(p0, vl), vl);
    vfloat32m1_t even = __riscv_vfmul_vf_f32m1(sqrz, p1, vl);

    res = __riscv_vfmadd_vv_f32m1(odd, z, even, vl);
}

static inline void my_log_reconstruction_h_f32m1(vfloat32m1_t e, vfloat32m1_t zh,
    vfloat32m1_t zl, vfloat32m1_t p, vfloat32m1_t logR, vfloat32m1_t& res, size_t vl)
{
    constexpr float log2h = 0x1.62e43p-1f, log2l = -0x1.05c61p-29f;

    vfloat32m1_t rh, rl, tmp, resl;

    tmp = __riscv_vfmadd_vv_f32m1(p, zh, __riscv_vfmadd_vv_f32m1(p, zl, zl, vl), vl);
    tmp = __riscv_vfmadd_vf_f32m1(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f32m1(zh, logR, vl);
    resl = __riscv_vfsub_vv_f32m1(zh, __riscv_vfadd_vv_f32m1(res, logR, vl), vl);
    rh = __riscv_vfmadd_vf_f32m1(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f32m1(res, __riscv_vfmsub_vf_f32m1(e, log2h, rh, vl), vl);
    res = __riscv_vfadd_vv_f32m1(rh, __riscv_vfadd_vv_f32m1(
        __riscv_vfadd_vv_f32m1(rl, resl, vl), tmp, vl), vl);
}

static inline void my_log_reconstruction_hl_f32m1(vfloat32m1_t e, vfloat32m1_t zh,
    vfloat32m1_t zl, vfloat32m1_t p, vfloat32m1_t logR, vfloat32m1_t& res, vfloat32m1_t& resl, size_t vl)
{
    constexpr float log2h = 0x1.62e43p-1f, log2l = -0x1.05c61p-29f;

    vfloat32m1_t rh, rl, tmp;

    tmp = __riscv_vfmadd_vv_f32m1(p, zh, __riscv_vfmadd_vv_f32m1(p, zl, zl, vl), vl);
    tmp = __riscv_vfmadd_vf_f32m1(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f32m1(zh, logR, vl);
    resl = __riscv_vfsub_vv_f32m1(zh, __riscv_vfadd_vv_f32m1(res, logR, vl), vl);
    rh = __riscv_vfmadd_vf_f32m1(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f32m1(res, __riscv_vfmsub_vf_f32m1(e, log2h, rh, vl), vl);
    rl = __riscv_vfadd_vv_f32m1(tmp, __riscv_vfadd_vv_f32m1(rl, resl, vl), vl);
    fast2sum_f32m1(rh, rl, res, resl, vl);
}

static inline void my_log2_reconstruction_f32m1(vfloat32m1_t e, vfloat32m1_t zh,
    vfloat32m1_t zl, vfloat32m1_t p, vfloat32m1_t logR, vfloat32m1_t& res, size_t vl)
{
    constexpr float invLog2h = 0x1.715476p0f, invLog2l = 0x1.4ae0cp-26f;

    vfloat32m1_t rh, rl, tmp, tmph, tmpl, sh, sl;

    tmp = __riscv_vfmadd_vv_f32m1(p, zh, __riscv_vfmadd_vv_f32m1(p, zl, zl, vl), vl);
    tmph = __riscv_vfsub_vv_f32m1(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f32m1(zh, __riscv_vfadd_vv_f32m1(tmph, logR, vl), vl);
    tmp = __riscv_vfmadd_vf_f32m1(tmp, invLog2h, __riscv_vfmadd_vf_f32m1(tmpl, invLog2h,
        __riscv_vfmul_vf_f32m1(tmph, invLog2l, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f32m1(tmph, invLog2h, e, vl);
    sh = __riscv_vfsub_vv_f32m1(rh, e, vl);
    sl = __riscv_vfsub_vv_f32m1(rh, __riscv_vfadd_vv_f32m1(sh, e, vl), vl);
    rl = __riscv_vfmsub_vf_f32m1(tmph, invLog2h, sh, vl);
    res = __riscv_vfadd_vv_f32m1(rh, __riscv_vfadd_vv_f32m1(__riscv_vfsub_vv_f32m1(rl, sl, vl), tmp, vl), vl);
}

static inline void my_log1p_reconstruction_h_f32m1(vfloat32m1_t e, vfloat32m1_t zh,
    vfloat32m1_t zl, vfloat32m1_t p, vfloat32m1_t logR, vfloat32m1_t& res, size_t vl)
{
    constexpr float log2h = 0x1.62e43p-1f, log2l = -0x1.05c61p-29f;
    constexpr float factor = 0x1p100f, invfactor = 0x1p-100f;

    vfloat32m1_t rh, rl, tmp, tmph, tmpl, resl;

    zl = __riscv_vfmul_vf_f32m1(zl, factor, vl);
    tmp = __riscv_vfmadd_vv_f32m1(p, zl, zl, vl);
    tmpl = __riscv_vfmsub_vv_f32m1(p, zl, __riscv_vfsub_vv_f32m1(tmp, zl, vl), vl);
    p = __riscv_vfmul_vf_f32m1(p, factor, vl);
    tmph = __riscv_vfmadd_vv_f32m1(p, zh, tmp, vl);
    tmpl = __riscv_vfadd_vv_f32m1(tmpl, __riscv_vfmsub_vv_f32m1(
        p, zh, __riscv_vfsub_vv_f32m1(tmph, tmp, vl), vl), vl);
    tmph = __riscv_vfmadd_vf_f32m1(e, log2l*factor, tmph, vl);
    res = __riscv_vfsub_vv_f32m1(zh, logR, vl);
    tmpl = __riscv_vfadd_vv_f32m1(tmpl, __riscv_vfmsub_vf_f32m1(
        __riscv_vfsub_vv_f32m1(zh, res, vl), factor, __riscv_vfmul_vf_f32m1(logR, factor, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f32m1(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f32m1(res, __riscv_vfmsub_vf_f32m1(e, log2h, rh, vl), vl);
    rl = __riscv_vfmadd_vf_f32m1(rl, factor, tmpl, vl);
    vfloat32m1_t tmp1h = __riscv_vfmadd_vf_f32m1(tmph, invfactor, rh, vl);
    vfloat32m1_t tmp1l = __riscv_vfmadd_vf_f32m1(tmph, invfactor, __riscv_vfsub_vv_f32m1(rh, tmp1h, vl), vl);
    res = __riscv_vfmadd_vf_f32m1(__riscv_vfmadd_vf_f32m1(tmp1l, factor, rl, vl), invfactor, tmp1h, vl);
}

static inline void my_log1p_process_linear_f32m1(const vfloat32m1_t& xm1, vfloat32m1_t& xm1c,
    vbool32_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f32m1_b32(abs_f32m1(xm1, vl), (float)0x1.6a09e8p-24f, vl);
    xm1c = __riscv_vfmerge_vfm_f32m1(xm1, (float)0.0f, linearMask, vl);
}

static inline void my_log1p_set_fe_underflow_f32m1(vfloat32m1_t& res, size_t vl)
{
    res = __riscv_vfmadd_vf_f32m1(res, (float)0x1p-26f, res, vl);
}

// ==================================================================================================
// vfloat32m2_t
// ==================================================================================================

static inline void fast2sum_f32m2(vfloat32m2_t a, vfloat32m2_t b, vfloat32m2_t& s, vfloat32m2_t& t, size_t vl)
{
    s = __riscv_vfadd_vv_f32m2(a, b, vl);
    vfloat32m2_t b1 = __riscv_vfsub_vv_f32m2(s, a, vl);
    t = __riscv_vfsub_vv_f32m2(b, b1, vl);
}

static inline vfloat32m2_t mul21_f32m2(vfloat32m2_t ah, vfloat32m2_t al, float bh, float bl, size_t vl)
{
    vfloat32m2_t zh, zl;
    zh = __riscv_vfmul_vf_f32m2(ah, bh, vl);
    zl = __riscv_vfmsub_vf_f32m2(ah, bh, zh, vl);
    zl = __riscv_vfadd_vv_f32m2(zl, __riscv_vfmadd_vf_f32m2(ah, bl,
        __riscv_vfmul_vf_f32m2(al, bh, vl), vl), vl);
    return __riscv_vfadd_vv_f32m2(zh, zl, vl);
}

static inline vfloat32m2_t abs_f32m2(vfloat32m2_t x, size_t vl)
{
    return __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vand_vx_u32m2(
        __riscv_vreinterpret_v_f32m2_u32m2(x), 0x7fffffff, vl));
}

static inline void log_check_special_cases_f32m2(vfloat32m2_t& x, vfloat32m2_t& special,
    vbool16_t& specialMask, size_t vl, float leftDomainPoint = 0.0f, float defaultArgValue = 1.0f)
{
    uint32_t pinf = 0x7f800000;
    uint32_t ninf = 0xff800000;
    uint32_t qnan = 0x7fc00000;
    vuint32m2_t absxui = __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(x),
        (uint32_t)0x7fffffff, vl);

    vbool16_t nanMask = __riscv_vmsgtu_vx_u32m2_b16(absxui, pinf, vl);
    x = __riscv_vfmerge_vfm_f32m2(x, defaultArgValue, nanMask, vl);

    vbool16_t snanMask = __riscv_vmsltu_vx_u32m2_b16(absxui, qnan, vl);
    vbool16_t infMask = __riscv_vmseq_vx_u32m2_b16(__riscv_vreinterpret_v_f32m2_u32m2(x), pinf, vl);
    vbool16_t leftDomainPointMask = __riscv_vmfeq_vf_f32m2_b16(x, leftDomainPoint, vl);
    vbool16_t invalidDomainMask = __riscv_vmflt_vf_f32m2_b16(x, leftDomainPoint, vl);

    special = __riscv_vfmerge_vfm_f32m2(x, *(float*)(&pinf), infMask, vl);
    special = __riscv_vfmerge_vfm_f32m2(special, *(float*)(&ninf), leftDomainPointMask, vl);
    special = __riscv_vfmerge_vfm_f32m2(special, *(float*)(&qnan),
        __riscv_vmor_mm_b16(invalidDomainMask, nanMask, vl), vl);

    specialMask = __riscv_vmor_mm_b16(__riscv_vmor_mm_b16(leftDomainPointMask, invalidDomainMask, vl),
        __riscv_vmor_mm_b16(infMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b16(__riscv_vmor_mm_b16(
        __riscv_vmand_mm_b16(nanMask, snanMask, vl), invalidDomainMask, vl), vl))
        volatile float x1 = 0.0f/0.0f;
    if (__riscv_vcpop_m_b16(leftDomainPointMask, vl))
        volatile float x1 = 1.0f/0.0f;

    x = __riscv_vfmerge_vfm_f32m2(x, defaultArgValue, specialMask, vl);
}

static inline void decompose_number_f32m2(vfloat32m2_t x, vfloat32m2_t& m, vint32m2_t& e, size_t vl)
{
    vuint32m2_t resu = __riscv_vreinterpret_v_f32m2_u32m2(x);
    resu = __riscv_vand_vx_u32m2(resu, (uint32_t)0x7f800000, vl);
    resu = __riscv_vsrl_vx_u32m2(resu, (uint32_t)23, vl);
    e = __riscv_vreinterpret_v_u32m2_i32m2(resu);
    e = __riscv_vsub_vx_i32m2(e, (int32_t)127, vl);

    vuint32m2_t xi = __riscv_vreinterpret_v_f32m2_u32m2(x);
    xi = __riscv_vand_vx_u32m2(xi, (uint32_t)0x007fffff, vl);
    xi = __riscv_vor_vx_u32m2(xi, (uint32_t)0x3f800000, vl);
    m = __riscv_vreinterpret_v_u32m2_f32m2(xi);
}

static inline void my_log1p_argument_reduction_f32m2(vfloat32m2_t x, vfloat32m2_t& y,
    vint32m2_t& e, vfloat32m2_t& ef, size_t vl)
{
    constexpr float ap1 = 0x1.56p0f;

    vint32m2_t et;
    vfloat32m2_t yt;
    decompose_number_f32m2(x, y, e, vl);

    vbool16_t subnormalMask = __riscv_vmflt_vf_f32m2_b16(x, (float)0x1p-126f, vl);
    yt = __riscv_vfsub_vf_f32m2(y, (float)1.0f, vl);
    decompose_number_f32m2(yt, yt, et, vl);
    et = __riscv_vadd_vv_i32m2(__riscv_vadd_vx_i32m2(e, (int32_t)1, vl), et, vl);
    y = __riscv_vmerge_vvm_f32m2(y, yt, subnormalMask, vl);
    e = __riscv_vmerge_vvm_i32m2(e, et, subnormalMask, vl);

    vbool16_t mask = __riscv_vmfge_vf_f32m2_b16(y, ap1, vl);
    yt = __riscv_vfmul_vf_f32m2(y, (float)0.5f, vl);
    et = __riscv_vadd_vx_i32m2(e, (int32_t)1, vl);
    y = __riscv_vmerge_vvm_f32m2(y, yt, mask, vl);
    e = __riscv_vmerge_vvm_i32m2(e, et, mask, vl);

    ef = __riscv_vfcvt_f_x_v_f32m2(e, vl);
}

static inline void my_log_argument_reduction_f32m2(vfloat32m2_t x, vfloat32m2_t& y, vfloat32m2_t& ef, size_t vl)
{
    vint32m2_t e;
    my_log1p_argument_reduction_f32m2(x, y, e, ef, vl);
}

static inline void my_log_table_values_f32m2(vfloat32m2_t y, vfloat32m2_t& logR, vfloat32m2_t& r, size_t vl)
{
    constexpr size_t k = 6;
    constexpr float u = 0x1.7711dcp0f, v = 0x1.5dc48cp-5f;
    constexpr size_t tshift = 23 - k;

    vuint32m2_t index = __riscv_vreinterpret_v_f32m2_u32m2(
        __riscv_vfmadd_vf_f32m2(y, u, __riscv_vfmv_v_f_f32m2(v, vl), vl));
    index = __riscv_vand_vx_u32m2(index, (uint32_t)0x007fffff, vl);
    index = __riscv_vsrl_vx_u32m2(index, tshift, vl);

    index = __riscv_vmul_vx_u32m2(index, sizeof(float), vl);

    logR = __riscv_vloxei32_v_f32m2(table_float_log_R, index, vl);
    r = __riscv_vloxei32_v_f32m2(table_float_R, index, vl);
}

static inline void my_log_second_reduction_f32m2(vfloat32m2_t y, vfloat32m2_t& z,
    vfloat32m2_t& zh, vfloat32m2_t& zl, vfloat32m2_t& logR, size_t vl)
{
    vfloat32m2_t r;
    my_log_table_values_f32m2(y, logR, r, vl);

    zh = __riscv_vfmul_vv_f32m2(y, r, vl);
    zl = __riscv_vfmsub_vv_f32m2(y, r, zh, vl);
    zh = __riscv_vfsub_vf_f32m2(zh, (float)1.0f, vl);
    z = __riscv_vfadd_vv_f32m2(zh, zl, vl);
}

static inline void my_log1p_second_reduction_f32m2(vfloat32m2_t xm1, vfloat32m2_t y, vint32m2_t e,
    vfloat32m2_t& z, vfloat32m2_t& zh, vfloat32m2_t& zl, vfloat32m2_t& logR, size_t vl)
{
    vfloat32m2_t r;
    my_log_table_values_f32m2(y, logR, r, vl);

    vint32m2_t pow2mei = __riscv_vadd_vx_i32m2(__riscv_vneg_v_i32m2(e, vl), (int32_t)129, vl);
    vfloat32m2_t pow2me = __riscv_vreinterpret_v_i32m2_f32m2(__riscv_vsll_vx_i32m2(pow2mei, 23, vl));
    vfloat32m2_t c = __riscv_vfmul_vv_f32m2(pow2me, r, vl);

    vfloat32m2_t tmph, tmpl, sh, sl;
    tmph = __riscv_vfmadd_vv_f32m2(xm1, c, c, vl);
    sh = __riscv_vfsub_vv_f32m2(tmph, c, vl);
    sl = __riscv_vfadd_vv_f32m2(sh, c, vl);
    sl = __riscv_vfsub_vv_f32m2(__riscv_vfsub_vv_f32m2(tmph, sl, vl),
        __riscv_vfadd_vv_f32m2(c, __riscv_vfsub_vv_f32m2(sh, sl, vl), vl), vl);
    tmpl = __riscv_vfmsub_vv_f32m2(xm1, c, sh, vl);
    zh = __riscv_vfsub_vf_f32m2(tmph, (float)4.0f, vl);
    zl = __riscv_vfsub_vv_f32m2(tmph, __riscv_vfadd_vf_f32m2(zh, (float)4.0f, vl), vl);
    zl = __riscv_vfadd_vv_f32m2(__riscv_vfsub_vv_f32m2(zl, sl, vl), tmpl, vl);

    zh = __riscv_vfmul_vf_f32m2(zh, (float)0.25f, vl);
    zl = __riscv_vfmul_vf_f32m2(zl, (float)0.25f, vl);

    z = __riscv_vfadd_vv_f32m2(zh, zl, vl);
}

static inline void my_log_approximation_f32m2(vfloat32m2_t z, vfloat32m2_t& res, size_t vl)
{
    constexpr float p0 = -0x1p-1f;
    constexpr float p1 = 0x1.555ffp-2f;
    constexpr float p2 = -0x1.0009cp-2f;
    constexpr float shift_avoid_underflow = 0x1p-41f;

    vfloat32m2_t zs = __riscv_vfadd_vf_f32m2(z, shift_avoid_underflow, vl);
    vfloat32m2_t sqrz = __riscv_vfmul_vv_f32m2(zs, zs, vl);
    vfloat32m2_t odd = __riscv_vfmadd_vf_f32m2(sqrz, p2, __riscv_vfmv_v_f_f32m2(p0, vl), vl);
    vfloat32m2_t even = __riscv_vfmul_vf_f32m2(sqrz, p1, vl);

    res = __riscv_vfmadd_vv_f32m2(odd, z, even, vl);
}

static inline void my_log_reconstruction_h_f32m2(vfloat32m2_t e, vfloat32m2_t zh,
    vfloat32m2_t zl, vfloat32m2_t p, vfloat32m2_t logR, vfloat32m2_t& res, size_t vl)
{
    constexpr float log2h = 0x1.62e43p-1f, log2l = -0x1.05c61p-29f;

    vfloat32m2_t rh, rl, tmp, resl;

    tmp = __riscv_vfmadd_vv_f32m2(p, zh, __riscv_vfmadd_vv_f32m2(p, zl, zl, vl), vl);
    tmp = __riscv_vfmadd_vf_f32m2(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f32m2(zh, logR, vl);
    resl = __riscv_vfsub_vv_f32m2(zh, __riscv_vfadd_vv_f32m2(res, logR, vl), vl);
    rh = __riscv_vfmadd_vf_f32m2(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f32m2(res, __riscv_vfmsub_vf_f32m2(e, log2h, rh, vl), vl);
    res = __riscv_vfadd_vv_f32m2(rh, __riscv_vfadd_vv_f32m2(
        __riscv_vfadd_vv_f32m2(rl, resl, vl), tmp, vl), vl);
}

static inline void my_log_reconstruction_hl_f32m2(vfloat32m2_t e, vfloat32m2_t zh,
    vfloat32m2_t zl, vfloat32m2_t p, vfloat32m2_t logR, vfloat32m2_t& res, vfloat32m2_t& resl, size_t vl)
{
    constexpr float log2h = 0x1.62e43p-1f, log2l = -0x1.05c61p-29f;

    vfloat32m2_t rh, rl, tmp;

    tmp = __riscv_vfmadd_vv_f32m2(p, zh, __riscv_vfmadd_vv_f32m2(p, zl, zl, vl), vl);
    tmp = __riscv_vfmadd_vf_f32m2(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f32m2(zh, logR, vl);
    resl = __riscv_vfsub_vv_f32m2(zh, __riscv_vfadd_vv_f32m2(res, logR, vl), vl);
    rh = __riscv_vfmadd_vf_f32m2(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f32m2(res, __riscv_vfmsub_vf_f32m2(e, log2h, rh, vl), vl);
    rl = __riscv_vfadd_vv_f32m2(tmp, __riscv_vfadd_vv_f32m2(rl, resl, vl), vl);
    fast2sum_f32m2(rh, rl, res, resl, vl);
}

static inline void my_log2_reconstruction_f32m2(vfloat32m2_t e, vfloat32m2_t zh,
    vfloat32m2_t zl, vfloat32m2_t p, vfloat32m2_t logR, vfloat32m2_t& res, size_t vl)
{
    constexpr float invLog2h = 0x1.715476p0f, invLog2l = 0x1.4ae0cp-26f;

    vfloat32m2_t rh, rl, tmp, tmph, tmpl, sh, sl;

    tmp = __riscv_vfmadd_vv_f32m2(p, zh, __riscv_vfmadd_vv_f32m2(p, zl, zl, vl), vl);
    tmph = __riscv_vfsub_vv_f32m2(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f32m2(zh, __riscv_vfadd_vv_f32m2(tmph, logR, vl), vl);
    tmp = __riscv_vfmadd_vf_f32m2(tmp, invLog2h, __riscv_vfmadd_vf_f32m2(tmpl, invLog2h,
        __riscv_vfmul_vf_f32m2(tmph, invLog2l, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f32m2(tmph, invLog2h, e, vl);
    sh = __riscv_vfsub_vv_f32m2(rh, e, vl);
    sl = __riscv_vfsub_vv_f32m2(rh, __riscv_vfadd_vv_f32m2(sh, e, vl), vl);
    rl = __riscv_vfmsub_vf_f32m2(tmph, invLog2h, sh, vl);
    res = __riscv_vfadd_vv_f32m2(rh, __riscv_vfadd_vv_f32m2(__riscv_vfsub_vv_f32m2(rl, sl, vl), tmp, vl), vl);
}

static inline void my_log1p_reconstruction_h_f32m2(vfloat32m2_t e, vfloat32m2_t zh,
    vfloat32m2_t zl, vfloat32m2_t p, vfloat32m2_t logR, vfloat32m2_t& res, size_t vl)
{
    constexpr float log2h = 0x1.62e43p-1f, log2l = -0x1.05c61p-29f;
    constexpr float factor = 0x1p100f, invfactor = 0x1p-100f;

    vfloat32m2_t rh, rl, tmp, tmph, tmpl, resl;

    zl = __riscv_vfmul_vf_f32m2(zl, factor, vl);
    tmp = __riscv_vfmadd_vv_f32m2(p, zl, zl, vl);
    tmpl = __riscv_vfmsub_vv_f32m2(p, zl, __riscv_vfsub_vv_f32m2(tmp, zl, vl), vl);
    p = __riscv_vfmul_vf_f32m2(p, factor, vl);
    tmph = __riscv_vfmadd_vv_f32m2(p, zh, tmp, vl);
    tmpl = __riscv_vfadd_vv_f32m2(tmpl, __riscv_vfmsub_vv_f32m2(
        p, zh, __riscv_vfsub_vv_f32m2(tmph, tmp, vl), vl), vl);
    tmph = __riscv_vfmadd_vf_f32m2(e, log2l*factor, tmph, vl);
    res = __riscv_vfsub_vv_f32m2(zh, logR, vl);
    tmpl = __riscv_vfadd_vv_f32m2(tmpl, __riscv_vfmsub_vf_f32m2(
        __riscv_vfsub_vv_f32m2(zh, res, vl), factor, __riscv_vfmul_vf_f32m2(logR, factor, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f32m2(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f32m2(res, __riscv_vfmsub_vf_f32m2(e, log2h, rh, vl), vl);
    rl = __riscv_vfmadd_vf_f32m2(rl, factor, tmpl, vl);
    vfloat32m2_t tmp1h = __riscv_vfmadd_vf_f32m2(tmph, invfactor, rh, vl);
    vfloat32m2_t tmp1l = __riscv_vfmadd_vf_f32m2(tmph, invfactor, __riscv_vfsub_vv_f32m2(rh, tmp1h, vl), vl);
    res = __riscv_vfmadd_vf_f32m2(__riscv_vfmadd_vf_f32m2(tmp1l, factor, rl, vl), invfactor, tmp1h, vl);
}

static inline void my_log1p_process_linear_f32m2(const vfloat32m2_t& xm1, vfloat32m2_t& xm1c,
    vbool16_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f32m2_b16(abs_f32m2(xm1, vl), (float)0x1.6a09e8p-24f, vl);
    xm1c = __riscv_vfmerge_vfm_f32m2(xm1, (float)0.0f, linearMask, vl);
}

static inline void my_log1p_set_fe_underflow_f32m2(vfloat32m2_t& res, size_t vl)
{
    res = __riscv_vfmadd_vf_f32m2(res, (float)0x1p-26f, res, vl);
}

// ==================================================================================================
// vfloat32m4_t
// ==================================================================================================

static inline void fast2sum_f32m4(vfloat32m4_t a, vfloat32m4_t b, vfloat32m4_t& s, vfloat32m4_t& t, size_t vl)
{
    s = __riscv_vfadd_vv_f32m4(a, b, vl);
    vfloat32m4_t b1 = __riscv_vfsub_vv_f32m4(s, a, vl);
    t = __riscv_vfsub_vv_f32m4(b, b1, vl);
}

static inline vfloat32m4_t mul21_f32m4(vfloat32m4_t ah, vfloat32m4_t al, float bh, float bl, size_t vl)
{
    vfloat32m4_t zh, zl;
    zh = __riscv_vfmul_vf_f32m4(ah, bh, vl);
    zl = __riscv_vfmsub_vf_f32m4(ah, bh, zh, vl);
    zl = __riscv_vfadd_vv_f32m4(zl, __riscv_vfmadd_vf_f32m4(ah, bl,
        __riscv_vfmul_vf_f32m4(al, bh, vl), vl), vl);
    return __riscv_vfadd_vv_f32m4(zh, zl, vl);
}

static inline vfloat32m4_t abs_f32m4(vfloat32m4_t x, size_t vl)
{
    return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vand_vx_u32m4(
        __riscv_vreinterpret_v_f32m4_u32m4(x), 0x7fffffff, vl));
}

static inline void log_check_special_cases_f32m4(vfloat32m4_t& x, vfloat32m4_t& special,
    vbool8_t& specialMask, size_t vl, float leftDomainPoint = 0.0f, float defaultArgValue = 1.0f)
{
    uint32_t pinf = 0x7f800000;
    uint32_t ninf = 0xff800000;
    uint32_t qnan = 0x7fc00000;
    vuint32m4_t absxui = __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(x),
        (uint32_t)0x7fffffff, vl);

    vbool8_t nanMask = __riscv_vmsgtu_vx_u32m4_b8(absxui, pinf, vl);
    x = __riscv_vfmerge_vfm_f32m4(x, defaultArgValue, nanMask, vl);

    vbool8_t snanMask = __riscv_vmsltu_vx_u32m4_b8(absxui, qnan, vl);
    vbool8_t infMask = __riscv_vmseq_vx_u32m4_b8(__riscv_vreinterpret_v_f32m4_u32m4(x), pinf, vl);
    vbool8_t leftDomainPointMask = __riscv_vmfeq_vf_f32m4_b8(x, leftDomainPoint, vl);
    vbool8_t invalidDomainMask = __riscv_vmflt_vf_f32m4_b8(x, leftDomainPoint, vl);

    special = __riscv_vfmerge_vfm_f32m4(x, *(float*)(&pinf), infMask, vl);
    special = __riscv_vfmerge_vfm_f32m4(special, *(float*)(&ninf), leftDomainPointMask, vl);
    special = __riscv_vfmerge_vfm_f32m4(special, *(float*)(&qnan),
        __riscv_vmor_mm_b8(invalidDomainMask, nanMask, vl), vl);

    specialMask = __riscv_vmor_mm_b8(__riscv_vmor_mm_b8(leftDomainPointMask, invalidDomainMask, vl),
        __riscv_vmor_mm_b8(infMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b8(__riscv_vmor_mm_b8(
        __riscv_vmand_mm_b8(nanMask, snanMask, vl), invalidDomainMask, vl), vl))
        volatile float x1 = 0.0f/0.0f;
    if (__riscv_vcpop_m_b8(leftDomainPointMask, vl))
        volatile float x1 = 1.0f/0.0f;

    x = __riscv_vfmerge_vfm_f32m4(x, defaultArgValue, specialMask, vl);
}

static inline void decompose_number_f32m4(vfloat32m4_t x, vfloat32m4_t& m, vint32m4_t& e, size_t vl)
{
    vuint32m4_t resu = __riscv_vreinterpret_v_f32m4_u32m4(x);
    resu = __riscv_vand_vx_u32m4(resu, (uint32_t)0x7f800000, vl);
    resu = __riscv_vsrl_vx_u32m4(resu, (uint32_t)23, vl);
    e = __riscv_vreinterpret_v_u32m4_i32m4(resu);
    e = __riscv_vsub_vx_i32m4(e, (int32_t)127, vl);

    vuint32m4_t xi = __riscv_vreinterpret_v_f32m4_u32m4(x);
    xi = __riscv_vand_vx_u32m4(xi, (uint32_t)0x007fffff, vl);
    xi = __riscv_vor_vx_u32m4(xi, (uint32_t)0x3f800000, vl);
    m = __riscv_vreinterpret_v_u32m4_f32m4(xi);
}

static inline void my_log1p_argument_reduction_f32m4(vfloat32m4_t x, vfloat32m4_t& y,
    vint32m4_t& e, vfloat32m4_t& ef, size_t vl)
{
    constexpr float ap1 = 0x1.56p0f;

    vint32m4_t et;
    vfloat32m4_t yt;
    decompose_number_f32m4(x, y, e, vl);

    vbool8_t subnormalMask = __riscv_vmflt_vf_f32m4_b8(x, (float)0x1p-126f, vl);
    yt = __riscv_vfsub_vf_f32m4(y, (float)1.0f, vl);
    decompose_number_f32m4(yt, yt, et, vl);
    et = __riscv_vadd_vv_i32m4(__riscv_vadd_vx_i32m4(e, (int32_t)1, vl), et, vl);
    y = __riscv_vmerge_vvm_f32m4(y, yt, subnormalMask, vl);
    e = __riscv_vmerge_vvm_i32m4(e, et, subnormalMask, vl);

    vbool8_t mask = __riscv_vmfge_vf_f32m4_b8(y, ap1, vl);
    yt = __riscv_vfmul_vf_f32m4(y, (float)0.5f, vl);
    et = __riscv_vadd_vx_i32m4(e, (int32_t)1, vl);
    y = __riscv_vmerge_vvm_f32m4(y, yt, mask, vl);
    e = __riscv_vmerge_vvm_i32m4(e, et, mask, vl);

    ef = __riscv_vfcvt_f_x_v_f32m4(e, vl);
}

static inline void my_log_argument_reduction_f32m4(vfloat32m4_t x, vfloat32m4_t& y, vfloat32m4_t& ef, size_t vl)
{
    vint32m4_t e;
    my_log1p_argument_reduction_f32m4(x, y, e, ef, vl);
}

static inline void my_log_table_values_f32m4(vfloat32m4_t y, vfloat32m4_t& logR, vfloat32m4_t& r, size_t vl)
{
    constexpr size_t k = 6;
    constexpr float u = 0x1.7711dcp0f, v = 0x1.5dc48cp-5f;
    constexpr size_t tshift = 23 - k;

    vuint32m4_t index = __riscv_vreinterpret_v_f32m4_u32m4(
        __riscv_vfmadd_vf_f32m4(y, u, __riscv_vfmv_v_f_f32m4(v, vl), vl));
    index = __riscv_vand_vx_u32m4(index, (uint32_t)0x007fffff, vl);
    index = __riscv_vsrl_vx_u32m4(index, tshift, vl);

    index = __riscv_vmul_vx_u32m4(index, sizeof(float), vl);

    logR = __riscv_vloxei32_v_f32m4(table_float_log_R, index, vl);
    r = __riscv_vloxei32_v_f32m4(table_float_R, index, vl);
}

static inline void my_log_second_reduction_f32m4(vfloat32m4_t y, vfloat32m4_t& z,
    vfloat32m4_t& zh, vfloat32m4_t& zl, vfloat32m4_t& logR, size_t vl)
{
    vfloat32m4_t r;
    my_log_table_values_f32m4(y, logR, r, vl);

    zh = __riscv_vfmul_vv_f32m4(y, r, vl);
    zl = __riscv_vfmsub_vv_f32m4(y, r, zh, vl);
    zh = __riscv_vfsub_vf_f32m4(zh, (float)1.0f, vl);
    z = __riscv_vfadd_vv_f32m4(zh, zl, vl);
}

static inline void my_log1p_second_reduction_f32m4(vfloat32m4_t xm1, vfloat32m4_t y, vint32m4_t e,
    vfloat32m4_t& z, vfloat32m4_t& zh, vfloat32m4_t& zl, vfloat32m4_t& logR, size_t vl)
{
    vfloat32m4_t r;
    my_log_table_values_f32m4(y, logR, r, vl);

    vint32m4_t pow2mei = __riscv_vadd_vx_i32m4(__riscv_vneg_v_i32m4(e, vl), (int32_t)129, vl);
    vfloat32m4_t pow2me = __riscv_vreinterpret_v_i32m4_f32m4(__riscv_vsll_vx_i32m4(pow2mei, 23, vl));
    vfloat32m4_t c = __riscv_vfmul_vv_f32m4(pow2me, r, vl);

    vfloat32m4_t tmph, tmpl, sh, sl;
    tmph = __riscv_vfmadd_vv_f32m4(xm1, c, c, vl);
    sh = __riscv_vfsub_vv_f32m4(tmph, c, vl);
    sl = __riscv_vfadd_vv_f32m4(sh, c, vl);
    sl = __riscv_vfsub_vv_f32m4(__riscv_vfsub_vv_f32m4(tmph, sl, vl),
        __riscv_vfadd_vv_f32m4(c, __riscv_vfsub_vv_f32m4(sh, sl, vl), vl), vl);
    tmpl = __riscv_vfmsub_vv_f32m4(xm1, c, sh, vl);
    zh = __riscv_vfsub_vf_f32m4(tmph, (float)4.0f, vl);
    zl = __riscv_vfsub_vv_f32m4(tmph, __riscv_vfadd_vf_f32m4(zh, (float)4.0f, vl), vl);
    zl = __riscv_vfadd_vv_f32m4(__riscv_vfsub_vv_f32m4(zl, sl, vl), tmpl, vl);

    zh = __riscv_vfmul_vf_f32m4(zh, (float)0.25f, vl);
    zl = __riscv_vfmul_vf_f32m4(zl, (float)0.25f, vl);

    z = __riscv_vfadd_vv_f32m4(zh, zl, vl);
}

static inline void my_log_approximation_f32m4(vfloat32m4_t z, vfloat32m4_t& res, size_t vl)
{
    constexpr float p0 = -0x1p-1f;
    constexpr float p1 = 0x1.555ffp-2f;
    constexpr float p2 = -0x1.0009cp-2f;
    constexpr float shift_avoid_underflow = 0x1p-41f;

    vfloat32m4_t zs = __riscv_vfadd_vf_f32m4(z, shift_avoid_underflow, vl);
    vfloat32m4_t sqrz = __riscv_vfmul_vv_f32m4(zs, zs, vl);
    vfloat32m4_t odd = __riscv_vfmadd_vf_f32m4(sqrz, p2, __riscv_vfmv_v_f_f32m4(p0, vl), vl);
    vfloat32m4_t even = __riscv_vfmul_vf_f32m4(sqrz, p1, vl);

    res = __riscv_vfmadd_vv_f32m4(odd, z, even, vl);
}

static inline void my_log_reconstruction_h_f32m4(vfloat32m4_t e, vfloat32m4_t zh,
    vfloat32m4_t zl, vfloat32m4_t p, vfloat32m4_t logR, vfloat32m4_t& res, size_t vl)
{
    constexpr float log2h = 0x1.62e43p-1f, log2l = -0x1.05c61p-29f;

    vfloat32m4_t rh, rl, tmp, resl;

    tmp = __riscv_vfmadd_vv_f32m4(p, zh, __riscv_vfmadd_vv_f32m4(p, zl, zl, vl), vl);
    tmp = __riscv_vfmadd_vf_f32m4(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f32m4(zh, logR, vl);
    resl = __riscv_vfsub_vv_f32m4(zh, __riscv_vfadd_vv_f32m4(res, logR, vl), vl);
    rh = __riscv_vfmadd_vf_f32m4(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f32m4(res, __riscv_vfmsub_vf_f32m4(e, log2h, rh, vl), vl);
    res = __riscv_vfadd_vv_f32m4(rh, __riscv_vfadd_vv_f32m4(
        __riscv_vfadd_vv_f32m4(rl, resl, vl), tmp, vl), vl);
}

static inline void my_log_reconstruction_hl_f32m4(vfloat32m4_t e, vfloat32m4_t zh,
    vfloat32m4_t zl, vfloat32m4_t p, vfloat32m4_t logR, vfloat32m4_t& res, vfloat32m4_t& resl, size_t vl)
{
    constexpr float log2h = 0x1.62e43p-1f, log2l = -0x1.05c61p-29f;

    vfloat32m4_t rh, rl, tmp;

    tmp = __riscv_vfmadd_vv_f32m4(p, zh, __riscv_vfmadd_vv_f32m4(p, zl, zl, vl), vl);
    tmp = __riscv_vfmadd_vf_f32m4(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f32m4(zh, logR, vl);
    resl = __riscv_vfsub_vv_f32m4(zh, __riscv_vfadd_vv_f32m4(res, logR, vl), vl);
    rh = __riscv_vfmadd_vf_f32m4(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f32m4(res, __riscv_vfmsub_vf_f32m4(e, log2h, rh, vl), vl);
    rl = __riscv_vfadd_vv_f32m4(tmp, __riscv_vfadd_vv_f32m4(rl, resl, vl), vl);
    fast2sum_f32m4(rh, rl, res, resl, vl);
}

static inline void my_log2_reconstruction_f32m4(vfloat32m4_t e, vfloat32m4_t zh,
    vfloat32m4_t zl, vfloat32m4_t p, vfloat32m4_t logR, vfloat32m4_t& res, size_t vl)
{
    constexpr float invLog2h = 0x1.715476p0f, invLog2l = 0x1.4ae0cp-26f;

    vfloat32m4_t rh, rl, tmp, tmph, tmpl, sh, sl;

    tmp = __riscv_vfmadd_vv_f32m4(p, zh, __riscv_vfmadd_vv_f32m4(p, zl, zl, vl), vl);
    tmph = __riscv_vfsub_vv_f32m4(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f32m4(zh, __riscv_vfadd_vv_f32m4(tmph, logR, vl), vl);
    tmp = __riscv_vfmadd_vf_f32m4(tmp, invLog2h, __riscv_vfmadd_vf_f32m4(tmpl, invLog2h,
        __riscv_vfmul_vf_f32m4(tmph, invLog2l, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f32m4(tmph, invLog2h, e, vl);
    sh = __riscv_vfsub_vv_f32m4(rh, e, vl);
    sl = __riscv_vfsub_vv_f32m4(rh, __riscv_vfadd_vv_f32m4(sh, e, vl), vl);
    rl = __riscv_vfmsub_vf_f32m4(tmph, invLog2h, sh, vl);
    res = __riscv_vfadd_vv_f32m4(rh, __riscv_vfadd_vv_f32m4(__riscv_vfsub_vv_f32m4(rl, sl, vl), tmp, vl), vl);
}

static inline void my_log1p_reconstruction_h_f32m4(vfloat32m4_t e, vfloat32m4_t zh,
    vfloat32m4_t zl, vfloat32m4_t p, vfloat32m4_t logR, vfloat32m4_t& res, size_t vl)
{
    constexpr float log2h = 0x1.62e43p-1f, log2l = -0x1.05c61p-29f;
    constexpr float factor = 0x1p100f, invfactor = 0x1p-100f;

    vfloat32m4_t rh, rl, tmp, tmph, tmpl, resl;

    zl = __riscv_vfmul_vf_f32m4(zl, factor, vl);
    tmp = __riscv_vfmadd_vv_f32m4(p, zl, zl, vl);
    tmpl = __riscv_vfmsub_vv_f32m4(p, zl, __riscv_vfsub_vv_f32m4(tmp, zl, vl), vl);
    p = __riscv_vfmul_vf_f32m4(p, factor, vl);
    tmph = __riscv_vfmadd_vv_f32m4(p, zh, tmp, vl);
    tmpl = __riscv_vfadd_vv_f32m4(tmpl, __riscv_vfmsub_vv_f32m4(
        p, zh, __riscv_vfsub_vv_f32m4(tmph, tmp, vl), vl), vl);
    tmph = __riscv_vfmadd_vf_f32m4(e, log2l*factor, tmph, vl);
    res = __riscv_vfsub_vv_f32m4(zh, logR, vl);
    tmpl = __riscv_vfadd_vv_f32m4(tmpl, __riscv_vfmsub_vf_f32m4(
        __riscv_vfsub_vv_f32m4(zh, res, vl), factor, __riscv_vfmul_vf_f32m4(logR, factor, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f32m4(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f32m4(res, __riscv_vfmsub_vf_f32m4(e, log2h, rh, vl), vl);
    rl = __riscv_vfmadd_vf_f32m4(rl, factor, tmpl, vl);
    vfloat32m4_t tmp1h = __riscv_vfmadd_vf_f32m4(tmph, invfactor, rh, vl);
    vfloat32m4_t tmp1l = __riscv_vfmadd_vf_f32m4(tmph, invfactor, __riscv_vfsub_vv_f32m4(rh, tmp1h, vl), vl);
    res = __riscv_vfmadd_vf_f32m4(__riscv_vfmadd_vf_f32m4(tmp1l, factor, rl, vl), invfactor, tmp1h, vl);
}

static inline void my_log1p_process_linear_f32m4(const vfloat32m4_t& xm1, vfloat32m4_t& xm1c,
    vbool8_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f32m4_b8(abs_f32m4(xm1, vl), (float)0x1.6a09e8p-24f, vl);
    xm1c = __riscv_vfmerge_vfm_f32m4(xm1, (float)0.0f, linearMask, vl);
}

static inline void my_log1p_set_fe_underflow_f32m4(vfloat32m4_t& res, size_t vl)
{
    res = __riscv_vfmadd_vf_f32m4(res, (float)0x1p-26f, res, vl);
}
