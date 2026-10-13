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
 *   File:  dlog_aux.inl                                 *
 *   Contains: helper built-in functions for log, log2   *
 *             log10, log1p functions (float64_t)        *
 *                                                       *
 *                                                       *
 *********************************************************
*/

// ==================================================================================================
// vfloat64m1_t
// ==================================================================================================

static inline void fast2sum_f64m1(vfloat64m1_t a, vfloat64m1_t b, vfloat64m1_t& s, vfloat64m1_t& t, size_t vl)
{
    s = __riscv_vfadd_vv_f64m1(a, b, vl);
    vfloat64m1_t b1 = __riscv_vfsub_vv_f64m1(s, a, vl);
    t = __riscv_vfsub_vv_f64m1(b, b1, vl);
}

static inline vfloat64m1_t mul21_f64m1(vfloat64m1_t ah, vfloat64m1_t al, double bh, double bl, size_t vl)
{
    vfloat64m1_t zh, zl;
    zh = __riscv_vfmul_vf_f64m1(ah, bh, vl);
    zl = __riscv_vfmsub_vf_f64m1(ah, bh, zh, vl);
    zl = __riscv_vfadd_vv_f64m1(zl, __riscv_vfmadd_vf_f64m1(ah, bl,
        __riscv_vfmul_vf_f64m1(al, bh, vl), vl), vl);
    return __riscv_vfadd_vv_f64m1(zh, zl, vl);
}

static inline vfloat64m1_t abs_f64m1(vfloat64m1_t x, size_t vl)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vx_u64m1(
        __riscv_vreinterpret_v_f64m1_u64m1(x), 0x7fffffffffffffff, vl));
}

static inline void log_check_special_cases_f64m1(vfloat64m1_t& x, vfloat64m1_t& special,
    vbool64_t& specialMask, size_t vl, double leftDomainPoint = 0.0, double defaultArgValue = 1.0)
{
    uint64_t pinf = 0x7ff0000000000000;
    uint64_t ninf = 0xfff0000000000000;
    uint64_t qnan = 0x7ff8000000000000;
    vuint64m1_t absxui = __riscv_vand_vx_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(x),
        (uint64_t)0x7fffffffffffffff, vl);

    vbool64_t nanMask = __riscv_vmsgtu_vx_u64m1_b64(absxui, pinf, vl);
    x = __riscv_vfmerge_vfm_f64m1(x, defaultArgValue, nanMask, vl);

    vbool64_t snanMask = __riscv_vmsltu_vx_u64m1_b64(absxui, qnan, vl);
    vbool64_t infMask = __riscv_vmseq_vx_u64m1_b64(__riscv_vreinterpret_v_f64m1_u64m1(x), pinf, vl);
    vbool64_t leftDomainPointMask = __riscv_vmfeq_vf_f64m1_b64(x, leftDomainPoint, vl);
    vbool64_t invalidDomainMask = __riscv_vmflt_vf_f64m1_b64(x, leftDomainPoint, vl);

    special = __riscv_vfmerge_vfm_f64m1(x, *(double*)(&pinf), infMask, vl);
    special = __riscv_vfmerge_vfm_f64m1(special, *(double*)(&ninf), leftDomainPointMask, vl);
    special = __riscv_vfmerge_vfm_f64m1(special, *(double*)(&qnan),
        __riscv_vmor_mm_b64(invalidDomainMask, nanMask, vl), vl);

    specialMask = __riscv_vmor_mm_b64(__riscv_vmor_mm_b64(leftDomainPointMask, invalidDomainMask, vl),
        __riscv_vmor_mm_b64(infMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b64(__riscv_vmor_mm_b64(
        __riscv_vmand_mm_b64(nanMask, snanMask, vl), invalidDomainMask, vl), vl))
        volatile double x1 = 0.0/0.0;
    if (__riscv_vcpop_m_b64(leftDomainPointMask, vl))
        volatile double x1 = 1.0/0.0;

    x = __riscv_vfmerge_vfm_f64m1(x, defaultArgValue, specialMask, vl);
}

static inline void decompose_number_f64m1(vfloat64m1_t x, vfloat64m1_t& m, vint64m1_t& e, size_t vl)
{
    vuint64m1_t resu = __riscv_vreinterpret_v_f64m1_u64m1(x);
    resu = __riscv_vand_vx_u64m1(resu, (uint64_t)0x7ff0000000000000, vl);
    resu = __riscv_vsrl_vx_u64m1(resu, (uint64_t)52, vl);
    e = __riscv_vreinterpret_v_u64m1_i64m1(resu);
    e = __riscv_vsub_vx_i64m1(e, (int64_t)1023, vl);

    vuint64m1_t xi = __riscv_vreinterpret_v_f64m1_u64m1(x);
    xi = __riscv_vand_vx_u64m1(xi, (uint64_t)0x000fffffffffffff, vl);
    xi = __riscv_vor_vx_u64m1(xi, (uint64_t)0x3ff0000000000000, vl);
    m = __riscv_vreinterpret_v_u64m1_f64m1(xi);
}

static inline void my_log1p_argument_reduction_f64m1(vfloat64m1_t x, vfloat64m1_t& y,
    vint64m1_t& e, vfloat64m1_t& ef, size_t vl)
{
    constexpr double ap1 = 0x1.556p0;

    vint64m1_t et;
    vfloat64m1_t yt;
    decompose_number_f64m1(x, y, e, vl);

    vbool64_t subnormalMask = __riscv_vmflt_vf_f64m1_b64(x, (double)0x1p-1022, vl);
    yt = __riscv_vfsub_vf_f64m1(y, (double)1.0, vl);
    decompose_number_f64m1(yt, yt, et, vl);
    et = __riscv_vadd_vv_i64m1(__riscv_vadd_vx_i64m1(e, (int64_t)1, vl), et, vl);
    y = __riscv_vmerge_vvm_f64m1(y, yt, subnormalMask, vl);
    e = __riscv_vmerge_vvm_i64m1(e, et, subnormalMask, vl);

    vbool64_t mask = __riscv_vmfge_vf_f64m1_b64(y, ap1, vl);
    yt = __riscv_vfmul_vf_f64m1(y, (double)0.5, vl);
    et = __riscv_vadd_vx_i64m1(e, (int64_t)1, vl);
    y = __riscv_vmerge_vvm_f64m1(y, yt, mask, vl);
    e = __riscv_vmerge_vvm_i64m1(e, et, mask, vl);

    ef = __riscv_vfcvt_f_x_v_f64m1(e, vl);
}

static inline void my_log_argument_reduction_f64m1(vfloat64m1_t x, vfloat64m1_t& y, vfloat64m1_t& ef, size_t vl)
{
    vint64m1_t e;
    my_log1p_argument_reduction_f64m1(x, y, e, ef, vl);
}

static inline void my_log_table_values_f64m1(vfloat64m1_t y, vfloat64m1_t& logR, vfloat64m1_t& r, size_t vl)
{
    constexpr size_t k = 8;
    constexpr double u = 0x1.7e5035f940d7fp0, v = 0x1.17e5035f9409bp-7;
    constexpr size_t tshift = 52 - k;

    vuint64m1_t index = __riscv_vreinterpret_v_f64m1_u64m1(
        __riscv_vfmadd_vf_f64m1(y, u, __riscv_vfmv_v_f_f64m1(v, vl), vl));
    index = __riscv_vand_vx_u64m1(index, (uint64_t)0x000fffffffffffff, vl);
    index = __riscv_vsrl_vx_u64m1(index, tshift, vl);

    index = __riscv_vmul_vx_u64m1(index, sizeof(double), vl);

    logR = __riscv_vloxei64_v_f64m1(table_log_R, index, vl);
    r = __riscv_vloxei64_v_f64m1(table_R, index, vl);
}

static inline void my_log_second_reduction_f64m1(vfloat64m1_t y, vfloat64m1_t& z,
    vfloat64m1_t& zh, vfloat64m1_t& zl, vfloat64m1_t& logR, size_t vl)
{
    vfloat64m1_t r;
    my_log_table_values_f64m1(y, logR, r, vl);

    zh = __riscv_vfmul_vv_f64m1(y, r, vl);
    zl = __riscv_vfmsub_vv_f64m1(y, r, zh, vl);
    zh = __riscv_vfsub_vf_f64m1(zh, (double)1.0, vl);
    z = __riscv_vfadd_vv_f64m1(zh, zl, vl);
}

static inline void my_log1p_second_reduction_f64m1(vfloat64m1_t xm1, vfloat64m1_t& y,
    vint64m1_t e, vfloat64m1_t& z, vfloat64m1_t& zh, vfloat64m1_t& zl, vfloat64m1_t& logR, size_t vl)
{
    vfloat64m1_t r;
    my_log_table_values_f64m1(y, logR, r, vl);

    vint64m1_t pow2mei = __riscv_vadd_vx_i64m1(__riscv_vneg_v_i64m1(e, vl), (int64_t)1025, vl);
    vfloat64m1_t pow2me = __riscv_vreinterpret_v_i64m1_f64m1(__riscv_vsll_vx_i64m1(pow2mei, 52, vl));
    vfloat64m1_t c = __riscv_vfmul_vv_f64m1(pow2me, r, vl);

    vfloat64m1_t tmph, tmpl, sh, sl;
    tmph = __riscv_vfmadd_vv_f64m1(xm1, c, c, vl);
    sh = __riscv_vfsub_vv_f64m1(tmph, c, vl);
    sl = __riscv_vfsub_vv_f64m1(tmph, __riscv_vfadd_vv_f64m1(sh, c, vl), vl);
    tmpl = __riscv_vfmsub_vv_f64m1(xm1, c, sh, vl);
    zh = __riscv_vfsub_vf_f64m1(tmph, (double)4.0, vl);
    zl = __riscv_vfsub_vv_f64m1(tmph, __riscv_vfadd_vf_f64m1(zh, (double)4.0, vl), vl);
    zl = __riscv_vfadd_vv_f64m1(__riscv_vfsub_vv_f64m1(zl, sl, vl), tmpl, vl);
    zh = __riscv_vfmul_vf_f64m1(zh, (double)0.25, vl);
    zl = __riscv_vfmul_vf_f64m1(zl, (double)0.25, vl);

    z = __riscv_vfadd_vv_f64m1(zh, zl, vl);
}

static inline void my_log_approximation_f64m1(vfloat64m1_t z, vfloat64m1_t& res, size_t vl)
{
    constexpr double p0 = -0x1p-1;
    constexpr double p1 = 0x1.5555555555558p-2;
    constexpr double p2 = -0x1.ffffffff7995ep-3;
    constexpr double p3 = 0x1.999999981a42p-3;
    constexpr double p4 = -0x1.5556d55a23fddp-3;
    constexpr double p5 = 0x1.2494575b197f9p-3;
    constexpr double shift_avoid_underflow = 0x1p-84;

    vfloat64m1_t zs = __riscv_vfadd_vf_f64m1(z, shift_avoid_underflow, vl);
    vfloat64m1_t sqrz = __riscv_vfmul_vv_f64m1(zs, zs, vl);
    vfloat64m1_t odd = __riscv_vfmadd_vf_f64m1(sqrz, p4, __riscv_vfmv_v_f_f64m1(p2, vl), vl);
    odd = __riscv_vfmadd_vv_f64m1(sqrz, odd, __riscv_vfmv_v_f_f64m1(p0, vl), vl);
    vfloat64m1_t even = __riscv_vfmadd_vf_f64m1(sqrz, p5, __riscv_vfmv_v_f_f64m1(p3, vl), vl);
    even = __riscv_vfmadd_vv_f64m1(sqrz, even, __riscv_vfmv_v_f_f64m1(p1, vl), vl);
    even = __riscv_vfmul_vv_f64m1(sqrz, even, vl);

    res = __riscv_vfmadd_vv_f64m1(odd, z, even, vl);
}

static inline void my_log_reconstruction_h_f64m1(vfloat64m1_t e, vfloat64m1_t zh,
    vfloat64m1_t zl, vfloat64m1_t p, vfloat64m1_t logR, vfloat64m1_t& res, size_t vl)
{
    constexpr double log2h = 0x1.62e42fefa39efp-1, log2l = 0x1.abc9e3b39803fp-56;

    vfloat64m1_t rh, rl, tmp, resl;

    tmp = __riscv_vfmadd_vv_f64m1(p, zh, __riscv_vfmadd_vv_f64m1(p, zl, zl, vl), vl);
    tmp = __riscv_vfmadd_vf_f64m1(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f64m1(zh, logR, vl);
    resl = __riscv_vfsub_vv_f64m1(zh, __riscv_vfadd_vv_f64m1(res, logR, vl), vl);
    rh = __riscv_vfmadd_vf_f64m1(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f64m1(res, __riscv_vfmsub_vf_f64m1(e, log2h, rh, vl), vl);
    res = __riscv_vfadd_vv_f64m1(rh, __riscv_vfadd_vv_f64m1(
        __riscv_vfadd_vv_f64m1(rl, resl, vl), tmp, vl), vl);
}

static inline void my_log_reconstruction_hl_f64m1(vfloat64m1_t e, vfloat64m1_t zh,
    vfloat64m1_t zl, vfloat64m1_t p, vfloat64m1_t logR, vfloat64m1_t& res, vfloat64m1_t& resl, size_t vl)
{
    constexpr double log2h = 0x1.62e42fefa39efp-1, log2l = 0x1.abc9e3b39803fp-56;

    vfloat64m1_t rh, rl, tmp;

    tmp = __riscv_vfmadd_vv_f64m1(p, zh, __riscv_vfmadd_vv_f64m1(p, zl, zl, vl), vl);
    tmp = __riscv_vfmadd_vf_f64m1(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f64m1(zh, logR, vl);
    resl = __riscv_vfsub_vv_f64m1(zh, __riscv_vfadd_vv_f64m1(res, logR, vl), vl);
    rh = __riscv_vfmadd_vf_f64m1(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f64m1(res, __riscv_vfmsub_vf_f64m1(e, log2h, rh, vl), vl);
    rl = __riscv_vfadd_vv_f64m1(tmp, __riscv_vfadd_vv_f64m1(rl, resl, vl), vl);
    fast2sum_f64m1(rh, rl, res, resl, vl);
}

static inline void my_log2_reconstruction_f64m1(vfloat64m1_t e, vfloat64m1_t zh,
    vfloat64m1_t zl, vfloat64m1_t p, vfloat64m1_t logR, vfloat64m1_t& res, size_t vl)
{
    constexpr double invLog2h = 0x1.71547652b82fep0, invLog2l = 0x1.777d0ffda0d24p-56;

    vfloat64m1_t rh, rl, tmp, tmph, tmpl, sh, sl;

    tmp = __riscv_vfmadd_vv_f64m1(p, zh, __riscv_vfmadd_vv_f64m1(p, zl, zl, vl), vl);
    tmph = __riscv_vfsub_vv_f64m1(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f64m1(zh, __riscv_vfadd_vv_f64m1(tmph, logR, vl), vl);
    tmp = __riscv_vfmadd_vf_f64m1(tmp, invLog2h, __riscv_vfmadd_vf_f64m1(tmpl, invLog2h,
        __riscv_vfmul_vf_f64m1(tmph, invLog2l, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f64m1(tmph, invLog2h, e, vl);
    sh = __riscv_vfsub_vv_f64m1(rh, e, vl);
    sl = __riscv_vfsub_vv_f64m1(rh, __riscv_vfadd_vv_f64m1(sh, e, vl), vl);
    rl = __riscv_vfmsub_vf_f64m1(tmph, invLog2h, sh, vl);
    res = __riscv_vfadd_vv_f64m1(rh, __riscv_vfadd_vv_f64m1(__riscv_vfsub_vv_f64m1(rl, sl, vl), tmp, vl), vl);
}

static inline void my_log1p_process_linear_f64m1(const vfloat64m1_t& xm1, vfloat64m1_t& xm1c,
    vbool64_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f64m1_b64(abs_f64m1(xm1, vl), (double)0x1.6a09e667f3bcdp-53, vl);
    xm1c = __riscv_vfmerge_vfm_f64m1(xm1, (double)0.0, linearMask, vl);
}

static inline void my_log1p_set_fe_underflow_f64m1(vfloat64m1_t& res, size_t vl)
{
    res = __riscv_vfmadd_vf_f64m1(res, (double)0x1p-60, res, vl);
}

// ==================================================================================================
// vfloat64m2_t
// ==================================================================================================

static inline void fast2sum_f64m2(vfloat64m2_t a, vfloat64m2_t b, vfloat64m2_t& s, vfloat64m2_t& t, size_t vl)
{
    s = __riscv_vfadd_vv_f64m2(a, b, vl);
    vfloat64m2_t b1 = __riscv_vfsub_vv_f64m2(s, a, vl);
    t = __riscv_vfsub_vv_f64m2(b, b1, vl);
}

static inline vfloat64m2_t mul21_f64m2(vfloat64m2_t ah, vfloat64m2_t al, double bh, double bl, size_t vl)
{
    vfloat64m2_t zh, zl;
    zh = __riscv_vfmul_vf_f64m2(ah, bh, vl);
    zl = __riscv_vfmsub_vf_f64m2(ah, bh, zh, vl);
    zl = __riscv_vfadd_vv_f64m2(zl, __riscv_vfmadd_vf_f64m2(ah, bl,
        __riscv_vfmul_vf_f64m2(al, bh, vl), vl), vl);
    return __riscv_vfadd_vv_f64m2(zh, zl, vl);
}

static inline vfloat64m2_t abs_f64m2(vfloat64m2_t x, size_t vl)
{
    return __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vx_u64m2(
        __riscv_vreinterpret_v_f64m2_u64m2(x), 0x7fffffffffffffff, vl));
}

static inline void log_check_special_cases_f64m2(vfloat64m2_t& x, vfloat64m2_t& special,
    vbool32_t& specialMask, size_t vl, double leftDomainPoint = 0.0, double defaultArgValue = 1.0)
{
    uint64_t pinf = 0x7ff0000000000000;
    uint64_t ninf = 0xfff0000000000000;
    uint64_t qnan = 0x7ff8000000000000;
    vuint64m2_t absxui = __riscv_vand_vx_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(x),
        (uint64_t)0x7fffffffffffffff, vl);

    vbool32_t nanMask = __riscv_vmsgtu_vx_u64m2_b32(absxui, pinf, vl);
    x = __riscv_vfmerge_vfm_f64m2(x, defaultArgValue, nanMask, vl);

    vbool32_t snanMask = __riscv_vmsltu_vx_u64m2_b32(absxui, qnan, vl);
    vbool32_t infMask = __riscv_vmseq_vx_u64m2_b32(__riscv_vreinterpret_v_f64m2_u64m2(x), pinf, vl);
    vbool32_t leftDomainPointMask = __riscv_vmfeq_vf_f64m2_b32(x, leftDomainPoint, vl);
    vbool32_t invalidDomainMask = __riscv_vmflt_vf_f64m2_b32(x, leftDomainPoint, vl);

    special = __riscv_vfmerge_vfm_f64m2(x, *(double*)(&pinf), infMask, vl);
    special = __riscv_vfmerge_vfm_f64m2(special, *(double*)(&ninf), leftDomainPointMask, vl);
    special = __riscv_vfmerge_vfm_f64m2(special, *(double*)(&qnan),
        __riscv_vmor_mm_b32(invalidDomainMask, nanMask, vl), vl);

    specialMask = __riscv_vmor_mm_b32(__riscv_vmor_mm_b32(leftDomainPointMask, invalidDomainMask, vl),
        __riscv_vmor_mm_b32(infMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b32(__riscv_vmor_mm_b32(
        __riscv_vmand_mm_b32(nanMask, snanMask, vl), invalidDomainMask, vl), vl))
        volatile double x1 = 0.0/0.0;
    if (__riscv_vcpop_m_b32(leftDomainPointMask, vl))
        volatile double x1 = 1.0/0.0;

    x = __riscv_vfmerge_vfm_f64m2(x, defaultArgValue, specialMask, vl);
}

static inline void decompose_number_f64m2(vfloat64m2_t x, vfloat64m2_t& m, vint64m2_t& e, size_t vl)
{
    vuint64m2_t resu = __riscv_vreinterpret_v_f64m2_u64m2(x);
    resu = __riscv_vand_vx_u64m2(resu, (uint64_t)0x7ff0000000000000, vl);
    resu = __riscv_vsrl_vx_u64m2(resu, (uint64_t)52, vl);
    e = __riscv_vreinterpret_v_u64m2_i64m2(resu);
    e = __riscv_vsub_vx_i64m2(e, (int64_t)1023, vl);

    vuint64m2_t xi = __riscv_vreinterpret_v_f64m2_u64m2(x);
    xi = __riscv_vand_vx_u64m2(xi, (uint64_t)0x000fffffffffffff, vl);
    xi = __riscv_vor_vx_u64m2(xi, (uint64_t)0x3ff0000000000000, vl);
    m = __riscv_vreinterpret_v_u64m2_f64m2(xi);
}

static inline void my_log1p_argument_reduction_f64m2(vfloat64m2_t x, vfloat64m2_t& y,
    vint64m2_t& e, vfloat64m2_t& ef, size_t vl)
{
    constexpr double ap1 = 0x1.556p0;

    vint64m2_t et;
    vfloat64m2_t yt;
    decompose_number_f64m2(x, y, e, vl);

    vbool32_t subnormalMask = __riscv_vmflt_vf_f64m2_b32(x, (double)0x1p-1022, vl);
    yt = __riscv_vfsub_vf_f64m2(y, (double)1.0, vl);
    decompose_number_f64m2(yt, yt, et, vl);
    et = __riscv_vadd_vv_i64m2(__riscv_vadd_vx_i64m2(e, (int64_t)1, vl), et, vl);
    y = __riscv_vmerge_vvm_f64m2(y, yt, subnormalMask, vl);
    e = __riscv_vmerge_vvm_i64m2(e, et, subnormalMask, vl);

    vbool32_t mask = __riscv_vmfge_vf_f64m2_b32(y, ap1, vl);
    yt = __riscv_vfmul_vf_f64m2(y, (double)0.5, vl);
    et = __riscv_vadd_vx_i64m2(e, (int64_t)1, vl);
    y = __riscv_vmerge_vvm_f64m2(y, yt, mask, vl);
    e = __riscv_vmerge_vvm_i64m2(e, et, mask, vl);

    ef = __riscv_vfcvt_f_x_v_f64m2(e, vl);
}

static inline void my_log_argument_reduction_f64m2(vfloat64m2_t x, vfloat64m2_t& y, vfloat64m2_t& ef, size_t vl)
{
    vint64m2_t e;
    my_log1p_argument_reduction_f64m2(x, y, e, ef, vl);
}

static inline void my_log_table_values_f64m2(vfloat64m2_t y, vfloat64m2_t& logR, vfloat64m2_t& r, size_t vl)
{
    constexpr size_t k = 8;
    constexpr double u = 0x1.7e5035f940d7fp0, v = 0x1.17e5035f9409bp-7;
    constexpr size_t tshift = 52 - k;

    vuint64m2_t index = __riscv_vreinterpret_v_f64m2_u64m2(
        __riscv_vfmadd_vf_f64m2(y, u, __riscv_vfmv_v_f_f64m2(v, vl), vl));
    index = __riscv_vand_vx_u64m2(index, (uint64_t)0x000fffffffffffff, vl);
    index = __riscv_vsrl_vx_u64m2(index, tshift, vl);

    index = __riscv_vmul_vx_u64m2(index, sizeof(double), vl);

    logR = __riscv_vloxei64_v_f64m2(table_log_R, index, vl);
    r = __riscv_vloxei64_v_f64m2(table_R, index, vl);
}

static inline void my_log_second_reduction_f64m2(vfloat64m2_t y, vfloat64m2_t& z,
    vfloat64m2_t& zh, vfloat64m2_t& zl, vfloat64m2_t& logR, size_t vl)
{
    vfloat64m2_t r;
    my_log_table_values_f64m2(y, logR, r, vl);

    zh = __riscv_vfmul_vv_f64m2(y, r, vl);
    zl = __riscv_vfmsub_vv_f64m2(y, r, zh, vl);
    zh = __riscv_vfsub_vf_f64m2(zh, (double)1.0, vl);
    z = __riscv_vfadd_vv_f64m2(zh, zl, vl);
}

static inline void my_log1p_second_reduction_f64m2(vfloat64m2_t xm1, vfloat64m2_t& y,
    vint64m2_t e, vfloat64m2_t& z, vfloat64m2_t& zh, vfloat64m2_t& zl, vfloat64m2_t& logR, size_t vl)
{
    vfloat64m2_t r;
    my_log_table_values_f64m2(y, logR, r, vl);

    vint64m2_t pow2mei = __riscv_vadd_vx_i64m2(__riscv_vneg_v_i64m2(e, vl), (int64_t)1025, vl);
    vfloat64m2_t pow2me = __riscv_vreinterpret_v_i64m2_f64m2(__riscv_vsll_vx_i64m2(pow2mei, 52, vl));
    vfloat64m2_t c = __riscv_vfmul_vv_f64m2(pow2me, r, vl);

    vfloat64m2_t tmph, tmpl, sh, sl;
    tmph = __riscv_vfmadd_vv_f64m2(xm1, c, c, vl);
    sh = __riscv_vfsub_vv_f64m2(tmph, c, vl);
    sl = __riscv_vfsub_vv_f64m2(tmph, __riscv_vfadd_vv_f64m2(sh, c, vl), vl);
    tmpl = __riscv_vfmsub_vv_f64m2(xm1, c, sh, vl);
    zh = __riscv_vfsub_vf_f64m2(tmph, (double)4.0, vl);
    zl = __riscv_vfsub_vv_f64m2(tmph, __riscv_vfadd_vf_f64m2(zh, (double)4.0, vl), vl);
    zl = __riscv_vfadd_vv_f64m2(__riscv_vfsub_vv_f64m2(zl, sl, vl), tmpl, vl);
    zh = __riscv_vfmul_vf_f64m2(zh, (double)0.25, vl);
    zl = __riscv_vfmul_vf_f64m2(zl, (double)0.25, vl);

    z = __riscv_vfadd_vv_f64m2(zh, zl, vl);
}

static inline void my_log_approximation_f64m2(vfloat64m2_t z, vfloat64m2_t& res, size_t vl)
{
    constexpr double p0 = -0x1p-1;
    constexpr double p1 = 0x1.5555555555558p-2;
    constexpr double p2 = -0x1.ffffffff7995ep-3;
    constexpr double p3 = 0x1.999999981a42p-3;
    constexpr double p4 = -0x1.5556d55a23fddp-3;
    constexpr double p5 = 0x1.2494575b197f9p-3;
    constexpr double shift_avoid_underflow = 0x1p-84;

    vfloat64m2_t zs = __riscv_vfadd_vf_f64m2(z, shift_avoid_underflow, vl);
    vfloat64m2_t sqrz = __riscv_vfmul_vv_f64m2(zs, zs, vl);
    vfloat64m2_t odd = __riscv_vfmadd_vf_f64m2(sqrz, p4, __riscv_vfmv_v_f_f64m2(p2, vl), vl);
    odd = __riscv_vfmadd_vv_f64m2(sqrz, odd, __riscv_vfmv_v_f_f64m2(p0, vl), vl);
    vfloat64m2_t even = __riscv_vfmadd_vf_f64m2(sqrz, p5, __riscv_vfmv_v_f_f64m2(p3, vl), vl);
    even = __riscv_vfmadd_vv_f64m2(sqrz, even, __riscv_vfmv_v_f_f64m2(p1, vl), vl);
    even = __riscv_vfmul_vv_f64m2(sqrz, even, vl);

    res = __riscv_vfmadd_vv_f64m2(odd, z, even, vl);
}

static inline void my_log_reconstruction_h_f64m2(vfloat64m2_t e, vfloat64m2_t zh,
    vfloat64m2_t zl, vfloat64m2_t p, vfloat64m2_t logR, vfloat64m2_t& res, size_t vl)
{
    constexpr double log2h = 0x1.62e42fefa39efp-1, log2l = 0x1.abc9e3b39803fp-56;

    vfloat64m2_t rh, rl, tmp, resl;

    tmp = __riscv_vfmadd_vv_f64m2(p, zh, __riscv_vfmadd_vv_f64m2(p, zl, zl, vl), vl);
    tmp = __riscv_vfmadd_vf_f64m2(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f64m2(zh, logR, vl);
    resl = __riscv_vfsub_vv_f64m2(zh, __riscv_vfadd_vv_f64m2(res, logR, vl), vl);
    rh = __riscv_vfmadd_vf_f64m2(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f64m2(res, __riscv_vfmsub_vf_f64m2(e, log2h, rh, vl), vl);
    res = __riscv_vfadd_vv_f64m2(rh, __riscv_vfadd_vv_f64m2(
        __riscv_vfadd_vv_f64m2(rl, resl, vl), tmp, vl), vl);
}

static inline void my_log_reconstruction_hl_f64m2(vfloat64m2_t e, vfloat64m2_t zh,
    vfloat64m2_t zl, vfloat64m2_t p, vfloat64m2_t logR, vfloat64m2_t& res, vfloat64m2_t& resl, size_t vl)
{
    constexpr double log2h = 0x1.62e42fefa39efp-1, log2l = 0x1.abc9e3b39803fp-56;

    vfloat64m2_t rh, rl, tmp;

    tmp = __riscv_vfmadd_vv_f64m2(p, zh, __riscv_vfmadd_vv_f64m2(p, zl, zl, vl), vl);
    tmp = __riscv_vfmadd_vf_f64m2(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f64m2(zh, logR, vl);
    resl = __riscv_vfsub_vv_f64m2(zh, __riscv_vfadd_vv_f64m2(res, logR, vl), vl);
    rh = __riscv_vfmadd_vf_f64m2(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f64m2(res, __riscv_vfmsub_vf_f64m2(e, log2h, rh, vl), vl);
    rl = __riscv_vfadd_vv_f64m2(tmp, __riscv_vfadd_vv_f64m2(rl, resl, vl), vl);
    fast2sum_f64m2(rh, rl, res, resl, vl);
}

static inline void my_log2_reconstruction_f64m2(vfloat64m2_t e, vfloat64m2_t zh,
    vfloat64m2_t zl, vfloat64m2_t p, vfloat64m2_t logR, vfloat64m2_t& res, size_t vl)
{
    constexpr double invLog2h = 0x1.71547652b82fep0, invLog2l = 0x1.777d0ffda0d24p-56;

    vfloat64m2_t rh, rl, tmp, tmph, tmpl, sh, sl;

    tmp = __riscv_vfmadd_vv_f64m2(p, zh, __riscv_vfmadd_vv_f64m2(p, zl, zl, vl), vl);
    tmph = __riscv_vfsub_vv_f64m2(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f64m2(zh, __riscv_vfadd_vv_f64m2(tmph, logR, vl), vl);
    tmp = __riscv_vfmadd_vf_f64m2(tmp, invLog2h, __riscv_vfmadd_vf_f64m2(tmpl, invLog2h,
        __riscv_vfmul_vf_f64m2(tmph, invLog2l, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f64m2(tmph, invLog2h, e, vl);
    sh = __riscv_vfsub_vv_f64m2(rh, e, vl);
    sl = __riscv_vfsub_vv_f64m2(rh, __riscv_vfadd_vv_f64m2(sh, e, vl), vl);
    rl = __riscv_vfmsub_vf_f64m2(tmph, invLog2h, sh, vl);
    res = __riscv_vfadd_vv_f64m2(rh, __riscv_vfadd_vv_f64m2(__riscv_vfsub_vv_f64m2(rl, sl, vl), tmp, vl), vl);
}

static inline void my_log1p_process_linear_f64m2(const vfloat64m2_t& xm1, vfloat64m2_t& xm1c,
    vbool32_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f64m2_b32(abs_f64m2(xm1, vl), (double)0x1.6a09e667f3bcdp-53, vl);
    xm1c = __riscv_vfmerge_vfm_f64m2(xm1, (double)0.0, linearMask, vl);
}

static inline void my_log1p_set_fe_underflow_f64m2(vfloat64m2_t& res, size_t vl)
{
    res = __riscv_vfmadd_vf_f64m2(res, (double)0x1p-60, res, vl);
}

// ==================================================================================================
// vfloat64m4_t
// ==================================================================================================

static inline void fast2sum_f64m4(vfloat64m4_t a, vfloat64m4_t b, vfloat64m4_t& s, vfloat64m4_t& t, size_t vl)
{
    s = __riscv_vfadd_vv_f64m4(a, b, vl);
    vfloat64m4_t b1 = __riscv_vfsub_vv_f64m4(s, a, vl);
    t = __riscv_vfsub_vv_f64m4(b, b1, vl);
}

static inline vfloat64m4_t mul21_f64m4(vfloat64m4_t ah, vfloat64m4_t al, double bh, double bl, size_t vl)
{
    vfloat64m4_t zh, zl;
    zh = __riscv_vfmul_vf_f64m4(ah, bh, vl);
    zl = __riscv_vfmsub_vf_f64m4(ah, bh, zh, vl);
    zl = __riscv_vfadd_vv_f64m4(zl, __riscv_vfmadd_vf_f64m4(ah, bl,
        __riscv_vfmul_vf_f64m4(al, bh, vl), vl), vl);
    return __riscv_vfadd_vv_f64m4(zh, zl, vl);
}

static inline vfloat64m4_t abs_f64m4(vfloat64m4_t x, size_t vl)
{
    return __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vand_vx_u64m4(
        __riscv_vreinterpret_v_f64m4_u64m4(x), 0x7fffffffffffffff, vl));
}

static inline void log_check_special_cases_f64m4(vfloat64m4_t& x, vfloat64m4_t& special,
    vbool16_t& specialMask, size_t vl, double leftDomainPoint = 0.0, double defaultArgValue = 1.0)
{
    uint64_t pinf = 0x7ff0000000000000;
    uint64_t ninf = 0xfff0000000000000;
    uint64_t qnan = 0x7ff8000000000000;
    vuint64m4_t absxui = __riscv_vand_vx_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(x),
        (uint64_t)0x7fffffffffffffff, vl);

    vbool16_t nanMask = __riscv_vmsgtu_vx_u64m4_b16(absxui, pinf, vl);
    x = __riscv_vfmerge_vfm_f64m4(x, defaultArgValue, nanMask, vl);

    vbool16_t snanMask = __riscv_vmsltu_vx_u64m4_b16(absxui, qnan, vl);
    vbool16_t infMask = __riscv_vmseq_vx_u64m4_b16(__riscv_vreinterpret_v_f64m4_u64m4(x), pinf, vl);
    vbool16_t leftDomainPointMask = __riscv_vmfeq_vf_f64m4_b16(x, leftDomainPoint, vl);
    vbool16_t invalidDomainMask = __riscv_vmflt_vf_f64m4_b16(x, leftDomainPoint, vl);

    special = __riscv_vfmerge_vfm_f64m4(x, *(double*)(&pinf), infMask, vl);
    special = __riscv_vfmerge_vfm_f64m4(special, *(double*)(&ninf), leftDomainPointMask, vl);
    special = __riscv_vfmerge_vfm_f64m4(special, *(double*)(&qnan),
        __riscv_vmor_mm_b16(invalidDomainMask, nanMask, vl), vl);

    specialMask = __riscv_vmor_mm_b16(__riscv_vmor_mm_b16(leftDomainPointMask, invalidDomainMask, vl),
        __riscv_vmor_mm_b16(infMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b16(__riscv_vmor_mm_b16(
        __riscv_vmand_mm_b16(nanMask, snanMask, vl), invalidDomainMask, vl), vl))
        volatile double x1 = 0.0/0.0;
    if (__riscv_vcpop_m_b16(leftDomainPointMask, vl))
        volatile double x1 = 1.0/0.0;

    x = __riscv_vfmerge_vfm_f64m4(x, defaultArgValue, specialMask, vl);
}

static inline void decompose_number_f64m4(vfloat64m4_t x, vfloat64m4_t& m, vint64m4_t& e, size_t vl)
{
    vuint64m4_t resu = __riscv_vreinterpret_v_f64m4_u64m4(x);
    resu = __riscv_vand_vx_u64m4(resu, (uint64_t)0x7ff0000000000000, vl);
    resu = __riscv_vsrl_vx_u64m4(resu, (uint64_t)52, vl);
    e = __riscv_vreinterpret_v_u64m4_i64m4(resu);
    e = __riscv_vsub_vx_i64m4(e, (int64_t)1023, vl);

    vuint64m4_t xi = __riscv_vreinterpret_v_f64m4_u64m4(x);
    xi = __riscv_vand_vx_u64m4(xi, (uint64_t)0x000fffffffffffff, vl);
    xi = __riscv_vor_vx_u64m4(xi, (uint64_t)0x3ff0000000000000, vl);
    m = __riscv_vreinterpret_v_u64m4_f64m4(xi);
}

static inline void my_log1p_argument_reduction_f64m4(vfloat64m4_t x, vfloat64m4_t& y,
vint64m4_t& e, vfloat64m4_t& ef, size_t vl) {
    constexpr double ap1 = 0x1.556p0;

    vint64m4_t et;
    vfloat64m4_t yt;
    decompose_number_f64m4(x, y, e, vl);

    vbool16_t subnormalMask = __riscv_vmflt_vf_f64m4_b16(x, (double)0x1p-1022, vl);
    yt = __riscv_vfsub_vf_f64m4(y, (double)1.0, vl);
    decompose_number_f64m4(yt, yt, et, vl);
    et = __riscv_vadd_vv_i64m4(__riscv_vadd_vx_i64m4(e, (int64_t)1, vl), et, vl);
    y = __riscv_vmerge_vvm_f64m4(y, yt, subnormalMask, vl);
    e = __riscv_vmerge_vvm_i64m4(e, et, subnormalMask, vl);

    vbool16_t mask = __riscv_vmfge_vf_f64m4_b16(y, ap1, vl);
    yt = __riscv_vfmul_vf_f64m4(y, (double)0.5, vl);
    et = __riscv_vadd_vx_i64m4(e, (int64_t)1, vl);
    y = __riscv_vmerge_vvm_f64m4(y, yt, mask, vl);
    e = __riscv_vmerge_vvm_i64m4(e, et, mask, vl);

    ef = __riscv_vfcvt_f_x_v_f64m4(e, vl);
}

static inline void my_log_argument_reduction_f64m4(vfloat64m4_t x, vfloat64m4_t& y, vfloat64m4_t& ef, size_t vl)
{
    vint64m4_t e;
    my_log1p_argument_reduction_f64m4(x, y, e, ef, vl);
}

static inline void my_log_table_values_f64m4(vfloat64m4_t y, vfloat64m4_t& logR, vfloat64m4_t& r, size_t vl)
{
    constexpr size_t k = 8;
    constexpr double u = 0x1.7e5035f940d7fp0, v = 0x1.17e5035f9409bp-7;
    constexpr size_t tshift = 52 - k;

    vuint64m4_t index = __riscv_vreinterpret_v_f64m4_u64m4(
        __riscv_vfmadd_vf_f64m4(y, u, __riscv_vfmv_v_f_f64m4(v, vl), vl));
    index = __riscv_vand_vx_u64m4(index, (uint64_t)0x000fffffffffffff, vl);
    index = __riscv_vsrl_vx_u64m4(index, tshift, vl);

    index = __riscv_vmul_vx_u64m4(index, sizeof(double), vl);

    logR = __riscv_vloxei64_v_f64m4(table_log_R, index, vl);
    r = __riscv_vloxei64_v_f64m4(table_R, index, vl);
}

static inline void my_log_second_reduction_f64m4(vfloat64m4_t y, vfloat64m4_t& z,
    vfloat64m4_t& zh, vfloat64m4_t& zl, vfloat64m4_t& logR, size_t vl) {

    vfloat64m4_t r;
    my_log_table_values_f64m4(y, logR, r, vl);

    zh = __riscv_vfmul_vv_f64m4(y, r, vl);
    zl = __riscv_vfmsub_vv_f64m4(y, r, zh, vl);
    zh = __riscv_vfsub_vf_f64m4(zh, (double)1.0, vl);
    z = __riscv_vfadd_vv_f64m4(zh, zl, vl);
}

static inline void my_log1p_second_reduction_f64m4(vfloat64m4_t xm1, vfloat64m4_t& y,
    vint64m4_t e, vfloat64m4_t& z, vfloat64m4_t& zh, vfloat64m4_t& zl, vfloat64m4_t& logR, size_t vl)
{
    vfloat64m4_t r;
    my_log_table_values_f64m4(y, logR, r, vl);

    vint64m4_t pow2mei = __riscv_vadd_vx_i64m4(__riscv_vneg_v_i64m4(e, vl), (int64_t)1025, vl);
    vfloat64m4_t pow2me = __riscv_vreinterpret_v_i64m4_f64m4(__riscv_vsll_vx_i64m4(pow2mei, 52, vl));
    vfloat64m4_t c = __riscv_vfmul_vv_f64m4(pow2me, r, vl);

    vfloat64m4_t tmph, tmpl, sh, sl;
    tmph = __riscv_vfmadd_vv_f64m4(xm1, c, c, vl);
    sh = __riscv_vfsub_vv_f64m4(tmph, c, vl);
    sl = __riscv_vfsub_vv_f64m4(tmph, __riscv_vfadd_vv_f64m4(sh, c, vl), vl);
    tmpl = __riscv_vfmsub_vv_f64m4(xm1, c, sh, vl);
    zh = __riscv_vfsub_vf_f64m4(tmph, (double)4.0, vl);
    zl = __riscv_vfsub_vv_f64m4(tmph, __riscv_vfadd_vf_f64m4(zh, (double)4.0, vl), vl);
    zl = __riscv_vfadd_vv_f64m4(__riscv_vfsub_vv_f64m4(zl, sl, vl), tmpl, vl);
    zh = __riscv_vfmul_vf_f64m4(zh, (double)0.25, vl);
    zl = __riscv_vfmul_vf_f64m4(zl, (double)0.25, vl);

    z = __riscv_vfadd_vv_f64m4(zh, zl, vl);
}

static inline void my_log_approximation_f64m4(vfloat64m4_t z, vfloat64m4_t& res, size_t vl)
{
    constexpr double p0 = -0x1p-1;
    constexpr double p1 = 0x1.5555555555558p-2;
    constexpr double p2 = -0x1.ffffffff7995ep-3;
    constexpr double p3 = 0x1.999999981a42p-3;
    constexpr double p4 = -0x1.5556d55a23fddp-3;
    constexpr double p5 = 0x1.2494575b197f9p-3;
    constexpr double shift_avoid_underflow = 0x1p-84;

    vfloat64m4_t zs = __riscv_vfadd_vf_f64m4(z, shift_avoid_underflow, vl);
    vfloat64m4_t sqrz = __riscv_vfmul_vv_f64m4(zs, zs, vl);
    vfloat64m4_t odd = __riscv_vfmadd_vf_f64m4(sqrz, p4, __riscv_vfmv_v_f_f64m4(p2, vl), vl);
    odd = __riscv_vfmadd_vv_f64m4(sqrz, odd, __riscv_vfmv_v_f_f64m4(p0, vl), vl);
    vfloat64m4_t even = __riscv_vfmadd_vf_f64m4(sqrz, p5, __riscv_vfmv_v_f_f64m4(p3, vl), vl);
    even = __riscv_vfmadd_vv_f64m4(sqrz, even, __riscv_vfmv_v_f_f64m4(p1, vl), vl);
    even = __riscv_vfmul_vv_f64m4(sqrz, even, vl);

    res = __riscv_vfmadd_vv_f64m4(odd, z, even, vl);
}

static inline void my_log_reconstruction_h_f64m4(vfloat64m4_t e, vfloat64m4_t zh,
    vfloat64m4_t zl, vfloat64m4_t p, vfloat64m4_t logR, vfloat64m4_t& res, size_t vl)
{

    constexpr double log2h = 0x1.62e42fefa39efp-1, log2l = 0x1.abc9e3b39803fp-56;

    vfloat64m4_t rh, rl, tmp, resl;

    tmp = __riscv_vfmadd_vv_f64m4(p, zh, __riscv_vfmadd_vv_f64m4(p, zl, zl, vl), vl);
    tmp = __riscv_vfmadd_vf_f64m4(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f64m4(zh, logR, vl);
    resl = __riscv_vfsub_vv_f64m4(zh, __riscv_vfadd_vv_f64m4(res, logR, vl), vl);
    rh = __riscv_vfmadd_vf_f64m4(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f64m4(res, __riscv_vfmsub_vf_f64m4(e, log2h, rh, vl), vl);
    res = __riscv_vfadd_vv_f64m4(rh, __riscv_vfadd_vv_f64m4(
        __riscv_vfadd_vv_f64m4(rl, resl, vl), tmp, vl), vl);
}

static inline void my_log_reconstruction_hl_f64m4(vfloat64m4_t e, vfloat64m4_t zh,
    vfloat64m4_t zl, vfloat64m4_t p, vfloat64m4_t logR, vfloat64m4_t& res, vfloat64m4_t& resl, size_t vl)
{

    constexpr double log2h = 0x1.62e42fefa39efp-1, log2l = 0x1.abc9e3b39803fp-56;

    vfloat64m4_t rh, rl, tmp;

    tmp = __riscv_vfmadd_vv_f64m4(p, zh, __riscv_vfmadd_vv_f64m4(p, zl, zl, vl), vl);
    tmp = __riscv_vfmadd_vf_f64m4(e, log2l, tmp, vl);
    res = __riscv_vfsub_vv_f64m4(zh, logR, vl);
    resl = __riscv_vfsub_vv_f64m4(zh, __riscv_vfadd_vv_f64m4(res, logR, vl), vl);
    rh = __riscv_vfmadd_vf_f64m4(e, log2h, res, vl);
    rl = __riscv_vfadd_vv_f64m4(res, __riscv_vfmsub_vf_f64m4(e, log2h, rh, vl), vl);
    rl = __riscv_vfadd_vv_f64m4(tmp, __riscv_vfadd_vv_f64m4(rl, resl, vl), vl);
    fast2sum_f64m4(rh, rl, res, resl, vl);
}

static inline void my_log2_reconstruction_f64m4(vfloat64m4_t e, vfloat64m4_t zh,
    vfloat64m4_t zl, vfloat64m4_t p, vfloat64m4_t logR, vfloat64m4_t& res, size_t vl)
{
    constexpr double invLog2h = 0x1.71547652b82fep0, invLog2l = 0x1.777d0ffda0d24p-56;

    vfloat64m4_t rh, rl, tmp, tmph, tmpl, sh, sl;

    tmp = __riscv_vfmadd_vv_f64m4(p, zh, __riscv_vfmadd_vv_f64m4(p, zl, zl, vl), vl);
    tmph = __riscv_vfsub_vv_f64m4(zh, logR, vl);
    tmpl = __riscv_vfsub_vv_f64m4(zh, __riscv_vfadd_vv_f64m4(tmph, logR, vl), vl);
    tmp = __riscv_vfmadd_vf_f64m4(tmp, invLog2h, __riscv_vfmadd_vf_f64m4(tmpl, invLog2h,
        __riscv_vfmul_vf_f64m4(tmph, invLog2l, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f64m4(tmph, invLog2h, e, vl);
    sh = __riscv_vfsub_vv_f64m4(rh, e, vl);
    sl = __riscv_vfsub_vv_f64m4(rh, __riscv_vfadd_vv_f64m4(sh, e, vl), vl);
    rl = __riscv_vfmsub_vf_f64m4(tmph, invLog2h, sh, vl);
    res = __riscv_vfadd_vv_f64m4(rh, __riscv_vfadd_vv_f64m4(__riscv_vfsub_vv_f64m4(rl, sl, vl), tmp, vl), vl);
}

static inline void my_log1p_process_linear_f64m4(const vfloat64m4_t& xm1, vfloat64m4_t& xm1c,
    vbool16_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f64m4_b16(abs_f64m4(xm1, vl), (double)0x1.6a09e667f3bcdp-53, vl);
    xm1c = __riscv_vfmerge_vfm_f64m4(xm1, (double)0.0, linearMask, vl);
}

static inline void my_log1p_set_fe_underflow_f64m4(vfloat64m4_t& res, size_t vl)
{
    res = __riscv_vfmadd_vf_f64m4(res, (double)0x1p-60, res, vl);
}
