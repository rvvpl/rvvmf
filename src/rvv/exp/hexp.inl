/*
 *========================================================
 * Copyright (c) The Lobachevsky State University of
 * Nizhny Novgorod and its affiliates. All rights reserved.
 *
 * Copyright 2025-2026 The RVVPL Authors (Elena Panova)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  hexp.inl                                     *
 *   Contains: helper built-in functions for exp, exp2   *
 *             and expm1 functions (float16_t)           *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#include "exp_utilities.inl"

typedef _Float16 FLOAT16_T;

constexpr FLOAT16_T ZERO_F16 = 0.0f16;
constexpr FLOAT16_T ONE_F16 = 1.0f16;

constexpr FLOAT16_T EXP_EXPM1_OVERFLOW_THRESHOLD_F16 = 0x1.62cp3f16;
constexpr FLOAT16_T EXP2_EXP2M1_OVERFLOW_THRESHOLD_F16 = 0x1.ffcp3f16;
constexpr FLOAT16_T EXP_SUBNORMAL_THRESHOLD_F16 = -0x1.368p3f16;
constexpr FLOAT16_T EXP2_SUBNORMAL_THRESHOLD_F16 = -0x1.cp3f16;
constexpr FLOAT16_T EXP_ZERO_THRESHOLD_F16 = -0x1.154p4f16;
constexpr FLOAT16_T EXP2_ZERO_THRESHOLD_F16 = -0x1.8fcp4f16;
constexpr FLOAT16_T EXP_UNDERFLOW_VALUE_F16 = ZERO_F16;
constexpr FLOAT16_T EXPM1_UNDERFLOW_THRESHOLD_F16 = -0x1.0ap3f16;
constexpr FLOAT16_T EXPM1_LINEAR_THRESHOLD_F16 = 0x1.6ap-11f16;
constexpr FLOAT16_T EXPM1_UNDERFLOW_VALUE_F16 = -1.0f16;

constexpr size_t TABLE_SIZE_DEG_F16 = 3;
constexpr FLOAT16_T EXP2_TABLE_SIZE_DEG_F16 = 0x1p3f16;
constexpr FLOAT16_T M_EXP2_M_TABLE_SIZE_DEG_F16 = -0x1p-3f16;
constexpr uint16_t MASK_FI_BIT_F16 = 0x0007;
constexpr uint16_t MASK_HI_BIT_F16 = 0x01ff;
constexpr FLOAT16_T MAGIC_CONST_1_F16 = 1536.0f16;  // 1.5*2^10
constexpr FLOAT16_T INV_LOG2_2K_F16 = 0x1.714p3f16;
constexpr FLOAT16_T M_LOG2_2K_H_F16 = -0x1.6p-4f16;
constexpr FLOAT16_T M_LOG2_2K_L_F16 = -0x1.72p-11f16;
constexpr FLOAT16_T M_LOG2_2K_LL_F16 = -0x1.8p-23f16;

constexpr int16_t FACTOR_DEG_SIGNED_F16 = 11;
constexpr FLOAT16_T FACTOR_F16 = 0x1p11f16;
constexpr FLOAT16_T INV_FACTOR_F16 = 0x1p-11f16;
constexpr FLOAT16_T EXP_THRESHOLD_AVOID_UNDERFFLOW_IN_SQR_F16 = 0x1p-11f16;
constexpr FLOAT16_T EXP2_THRESHOLD_AVOID_UNDERFFLOW_IN_SQR_F16 = 0x1p-9f16;

static const FLOAT16_T LOOK_UP_TABLE_HIGH_F16[8] = {
    0x1p0f16, 0x1.174p0f16, 0x1.308p0f16, 0x1.4cp0f16,
    0x1.6ap0f16, 0x1.8acp0f16, 0x1.ae8p0f16, 0x1.d58p0f16
};
static const FLOAT16_T LOOK_UP_TABLE_LOW_F16[8] = {
    0.0f16*FACTOR_F16, -0x1.47cp-12f16*FACTOR_F16, -0x1.02p-12f16*FACTOR_F16, -0x1.298p-15f16*FACTOR_F16,
    0x1.3ccp-13f16*FACTOR_F16, 0x1.ca8p-13f16*FACTOR_F16, 0x1.3f4p-13f16*FACTOR_F16, 0x1.8ep-16f16*FACTOR_F16
};

constexpr FLOAT16_T EXP_POL_COEFF_2_F16 = 0x1p-1f16 * FACTOR_F16;
constexpr FLOAT16_T EXP_POL_COEFF_3_F16 = 0x1.55p-3f16 * FACTOR_F16;

constexpr FLOAT16_T EXP2_POL_COEFF_1_H_F16 = 0x1.63p-1f16 * FACTOR_F16;
constexpr FLOAT16_T EXP2_POL_COEFF_1_L_F16 = -0x1.bdp-13f16 * FACTOR_F16;
constexpr FLOAT16_T EXP2_POL_COEFF_2_F16 = 0x1.ecp-3f16 * FACTOR_F16;
constexpr FLOAT16_T EXP2_POL_COEFF_3_F16 = 0x1.c6cp-5f16 * FACTOR_F16;

// ---------------------------- m1 ----------------------------

forceinline void check_special_cases_f16m1(vfloat16m1_t& x, vfloat16m1_t& special, vbool16_t& specialMask,
    FLOAT16_T overflowThreshold, FLOAT16_T underflowThreshold, FLOAT16_T underflowValue, bool throwUnderflow, size_t vl)
{
    vuint16m1_t uxi = __riscv_vreinterpret_v_f16m1_u16m1(x);

    vbool16_t underflowMask = __riscv_vmsgtu_vx_u16m1_b16(uxi, RVVMF_EXP_AS_UINT16(underflowThreshold), vl);
    special = __riscv_vfmerge_vfm_f16m1(x, underflowValue, underflowMask, vl);
    specialMask = underflowMask;

#ifndef __FAST_MATH__
    uint16_t pinf = 0x7c00;
    uint16_t qnan = 0x7e00;
    vuint16m1_t absxui = __riscv_vand_vx_u16m1(uxi, (uint16_t)0x7fff, vl);
    vint16m1_t xi = __riscv_vreinterpret_v_f16m1_i16m1(x);

    vbool16_t nanMask = __riscv_vmsgtu_vx_u16m1_b16(absxui, pinf, vl);
    vbool16_t snanMask = __riscv_vmsltu_vx_u16m1_b16(absxui, qnan, vl);
    vbool16_t regularMask = __riscv_vmsltu_vx_u16m1_b16(absxui, pinf, vl);
    vbool16_t overflowMask = __riscv_vmsgt_vx_i16m1_b16(xi, RVVMF_EXP_AS_INT16(overflowThreshold), vl);

    special = __riscv_vfmerge_vfm_f16m1(special, RVVMF_EXP_AS_FP16(pinf), overflowMask, vl);
    special = __riscv_vfmerge_vfm_f16m1(special, RVVMF_EXP_AS_FP16(qnan), nanMask, vl);

    specialMask = __riscv_vmor_mm_b16(specialMask, __riscv_vmor_mm_b16(overflowMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b16(__riscv_vmand_mm_b16(nanMask, snanMask, vl), vl))
        RVVMF_EXP_CALL_FE_INVALID();
    if (__riscv_vcpop_m_b16(__riscv_vmand_mm_b16(overflowMask, regularMask, vl), vl))
        RVVMF_EXP_CALL_FE_OVERFLOW();
    if (throwUnderflow && __riscv_vcpop_m_b16(__riscv_vmand_mm_b16(underflowMask, regularMask, vl), vl))
        RVVMF_EXP_CALL_FE_UNDERFLOW();
#endif

    x = __riscv_vfmerge_vfm_f16m1(x, ZERO_F16, specialMask, vl);
}

forceinline void do_exp_argument_reduction_hl_f16m1(vfloat16m1_t x,
    vfloat16m1_t& yh, vfloat16m1_t& yl, vuint16m1_t& ei, vuint16m1_t& fi, size_t vl)
{
    vfloat16m1_t vmagicConst1 = __riscv_vfmv_v_f_f16m1(MAGIC_CONST_1_F16, vl);
    vfloat16m1_t h = __riscv_vfmadd_vf_f16m1(x, INV_LOG2_2K_F16, vmagicConst1, vl);
    vuint16m1_t hi = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(h), MASK_HI_BIT_F16, vl);
    fi = __riscv_vand_vx_u16m1(hi, MASK_FI_BIT_F16, vl);
    ei = __riscv_vsrl_vx_u16m1(hi, TABLE_SIZE_DEG_F16, vl);
    h = __riscv_vfsub_vv_f16m1(h, vmagicConst1, vl);
    fma12_ver2p2_vf_f16m1(h, M_LOG2_2K_L_F16, __riscv_vfmadd_vf_f16m1(h, M_LOG2_2K_H_F16, x, vl), yh, yl, vl);
    yl = __riscv_vfmacc_vf_f16m1(yl, M_LOG2_2K_LL_F16, h, vl);
    fast_2_sum_vv_f16m1(yh, yl, yh, yl, vl);
}

forceinline void do_exp2_argument_reduction_f16m1(vfloat16m1_t x, vfloat16m1_t& y,
    vuint16m1_t& ei, vuint16m1_t& fi, size_t vl)
{
    vfloat16m1_t vmagicConst1 = __riscv_vfmv_v_f_f16m1(MAGIC_CONST_1_F16, vl);
    vfloat16m1_t h = __riscv_vfmadd_vf_f16m1(x, EXP2_TABLE_SIZE_DEG_F16, vmagicConst1, vl);
    vuint16m1_t hi = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(h), MASK_HI_BIT_F16, vl);
    fi = __riscv_vand_vx_u16m1(hi, MASK_FI_BIT_F16, vl);
    ei = __riscv_vsrl_vx_u16m1(hi, TABLE_SIZE_DEG_F16, vl);
    h = __riscv_vfsub_vv_f16m1(h, vmagicConst1, vl);
    y = __riscv_vfmadd_vf_f16m1(h, M_EXP2_M_TABLE_SIZE_DEG_F16, x, vl);
}

forceinline void get_table_values_hl_f16m1(
    vuint16m1_t& index, vfloat16m1_t& th, vfloat16m1_t& tl, size_t vl)
{
    index = __riscv_vsll_vx_u16m1(index, (size_t)1, vl);
    th = __riscv_vloxei16_v_f16m1(LOOK_UP_TABLE_HIGH_F16, index, vl);
    tl = __riscv_vloxei16_v_f16m1(LOOK_UP_TABLE_LOW_F16, index, vl);
}

forceinline void calculate_exp_polynom_hl_f16m1(vfloat16m1_t yh, vfloat16m1_t yl,
    vfloat16m1_t& ph, vfloat16m1_t& pl, size_t vl)
{
#ifndef __FAST_MATH__
    vfloat16m1_t yhs = __riscv_vfmerge_vfm_f16m1(yh, ZERO_F16, __riscv_vmflt_vf_f16m1_b16(
        abs_f16m1(yh, vl), EXP_THRESHOLD_AVOID_UNDERFFLOW_IN_SQR_F16, vl), vl);
#else
    vfloat16m1_t yhs = yh;
#endif
    vfloat16m1_t r = calc_polynom_deg_1_f16m1(yh, EXP_POL_COEFF_2_F16, EXP_POL_COEFF_3_F16, vl);
    r = __riscv_vfmul_vv_f16m1(r, yh, vl);

    vfloat16m1_t tmp = __riscv_vfmul_vf_f16m1(yh, FACTOR_F16, vl);
    ph = __riscv_vfmadd_vv_f16m1(yh, r, tmp, vl);
    tmp = __riscv_vfsub_vv_f16m1(tmp, ph, vl);
    pl = __riscv_vfmacc_vv_f16m1(__riscv_vfmul_vf_f16m1(tmp, FACTOR_F16, vl), r,
        __riscv_vfmul_vf_f16m1(yhs, FACTOR_F16, vl), vl);

    pl = __riscv_vfmacc_vf_f16m1(pl, FACTOR_F16, __riscv_vfmul_vf_f16m1(yl, FACTOR_F16, vl), vl);
}

forceinline void calculate_expm1_polynom_hl_f16m1(vfloat16m1_t yh, vfloat16m1_t yl,
    vfloat16m1_t& ph, vfloat16m1_t& pl, size_t vl)
{
    calculate_exp_polynom_hl_f16m1(yh, yl, ph, pl, vl);
}

forceinline void calculate_exp2_polynom_hl12_f16m1(vfloat16m1_t yh, vfloat16m1_t& ph,
    vfloat16m1_t& pl, size_t vl)
{
#ifndef __FAST_MATH__
    vfloat16m1_t yhs = __riscv_vfmerge_vfm_f16m1(yh, ZERO_F16, __riscv_vmflt_vf_f16m1_b16(
        abs_f16m1(yh, vl), EXP2_THRESHOLD_AVOID_UNDERFFLOW_IN_SQR_F16, vl), vl);
#else
    vfloat16m1_t yhs = yh;
#endif
    vfloat16m1_t r = calc_polynom_deg_1_f16m1(yh, EXP2_POL_COEFF_2_F16, EXP2_POL_COEFF_3_F16, vl);
    r = __riscv_vfmul_vv_f16m1(r, yh, vl);
    r = __riscv_vfmadd_vv_f16m1(yhs, r, __riscv_vfmul_vf_f16m1(yhs, EXP2_POL_COEFF_1_L_F16, vl), vl);
    fma12_ver2p1_vf_f16m1(yh, EXP2_POL_COEFF_1_H_F16, r, ph, pl, vl);
    pl = __riscv_vfmul_vf_f16m1(pl, FACTOR_F16, vl);
}

forceinline void reconstruct_exp_hl_hl_f16m1(vfloat16m1_t th, vfloat16m1_t tl,
    vfloat16m1_t pm1h, vfloat16m1_t pm1l, vuint16m1_t ei, vfloat16m1_t& res, size_t vl)
{
    vfloat16m1_t zh, zl;
    fma12_vv_f16m1(th, pm1h, __riscv_vfmul_vf_f16m1(th, FACTOR_F16, vl), zh, zl, vl);
    zl = __riscv_vfmadd_vf_f16m1(zl, FACTOR_F16, __riscv_vfmadd_vv_f16m1(th, pm1l,
        __riscv_vfmadd_vv_f16m1(tl, pm1h, __riscv_vfmul_vf_f16m1(tl, FACTOR_F16, vl), vl), vl), vl);
    zl = __riscv_vfadd_vf_f16m1(zl, 0x1p-1f16, vl);

    ei = __riscv_vsll_vx_u16m1(__riscv_vadd_vx_u16m1(ei, (uint16_t)0x001e, vl), (size_t)9, vl);
    vuint16m1_t shifti = __riscv_vand_vx_u16m1(ei, (uint16_t)0x0200, vl);
    vuint16m1_t pow2e1i = __riscv_vadd_vv_u16m1(ei, shifti, vl);
    vuint16m1_t pow2e2i = __riscv_vsub_vv_u16m1(ei, shifti, vl);
    vfloat16m1_t pow2e1 = __riscv_vreinterpret_v_u16m1_f16m1(pow2e1i);
    vfloat16m1_t pow2e2 = __riscv_vreinterpret_v_u16m1_f16m1(pow2e2i);
    vfloat16m1_t invpow2e2 = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vrsub_vx_u16m1(pow2e2i, (uint16_t)0x7800, vl));
    vfloat16m1_t pow2e1f = __riscv_vfmul_vf_f16m1(pow2e1, INV_FACTOR_F16, vl);
    vfloat16m1_t pow2e2f = __riscv_vfmul_vf_f16m1(pow2e2, INV_FACTOR_F16, vl);

    zh = __riscv_vfmul_vv_f16m1(zh, pow2e1f, vl);
    res = __riscv_vfmul_vv_f16m1(zh, pow2e2, vl);
    vfloat16m1_t tmp = __riscv_vfmul_vf_f16m1(__riscv_vfnmsac_vv_f16m1(zh, res, invpow2e2, vl), FACTOR_F16, vl);
    res = __riscv_vfmacc_vv_f16m1(res, __riscv_vfmadd_vv_f16m1(zl, pow2e1f, tmp, vl), pow2e2f, vl);
}

forceinline void reconstruct_expm1_f16m1(vfloat16m1_t th, vfloat16m1_t tl,
    vfloat16m1_t pm1h, vfloat16m1_t pm1l, vuint16m1_t ei, vfloat16m1_t& res, size_t vl)
{
    vbool16_t mNearZero = __riscv_vmseq_vx_u16m1_b16(ei, (uint16_t)0, vl);

    vfloat16m1_t rh, rl, rll;
    fast_2_sum_fv_f16m1(FACTOR_F16, pm1h, rh, rl, vl);
    fast_2_sum_vv_f16m1(__riscv_vfmul_vf_f16m1(rl, FACTOR_F16, vl), pm1l, rl, rll, vl);

    vfloat16m1_t zh, zl;
    zh = __riscv_vfmul_vv_f16m1(th, rh, vl);
    zl = __riscv_vfmsub_vv_f16m1(th, rh, zh, vl);
    zl = __riscv_vfmadd_vf_f16m1(zl, FACTOR_F16, __riscv_vfmadd_vv_f16m1(
        th, rl, __riscv_vfmul_vv_f16m1(tl, rh, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f16m1(zl, INV_FACTOR_F16, zh, vl);
    rl = __riscv_vfmadd_vf_f16m1(__riscv_vfsub_vv_f16m1(zh, rh, vl), FACTOR_F16, zl, vl);

#ifndef __FAST_MATH__   
    rl  = __riscv_vfmerge_vfm_f16m1(rl, ZERO_F16, __riscv_vmflt_vf_f16m1_b16(abs_f16m1(rl, vl), 0x1p0f16, vl), vl);
#endif

    constexpr FLOAT16_T invfactore = INV_FACTOR_F16*0x1p1f16;
    vuint16m1_t pow2ei = __riscv_vsll_vx_u16m1(__riscv_vadd_vx_u16m1(ei, (uint16_t)14, vl), (size_t)10, vl);
    vfloat16m1_t pow2e = __riscv_vfmul_vf_f16m1(__riscv_vreinterpret_v_u16m1_f16m1(pow2ei), invfactore, vl);
    rh = __riscv_vfmul_vv_f16m1(rh, pow2e, vl);

    vfloat16m1_t sh1, sl1;
    sh1 = __riscv_vfadd_vf_f16m1(rh, EXPM1_UNDERFLOW_VALUE_F16, vl);
    sl1 = __riscv_vfsub_vv_f16m1(sh1, rh, vl);
    sl1 = __riscv_vfadd_vv_f16m1(__riscv_vfsub_vv_f16m1(rh, __riscv_vfsub_vv_f16m1(sh1, sl1, vl), vl),
        __riscv_vfrsub_vf_f16m1(sl1, EXPM1_UNDERFLOW_VALUE_F16, vl), vl);

    vfloat16m1_t tmp, resl;
    tmp = __riscv_vfmadd_vv_f16m1(rl, pow2e, __riscv_vfmul_vf_f16m1(sl1, FACTOR_F16, vl), vl);
    res = __riscv_vfmadd_vf_f16m1(tmp, INV_FACTOR_F16, sh1, vl);
    resl = __riscv_vfmadd_vf_f16m1(__riscv_vfsub_vv_f16m1(sh1, res, vl), FACTOR_F16, tmp, vl);
    resl = __riscv_vfmadd_vf_f16m1_mu(mNearZero, resl, FACTOR_F16, rll, vl);
    constexpr FLOAT16_T invfactor2 = INV_FACTOR_F16*INV_FACTOR_F16;
    res = __riscv_vfmacc_vf_f16m1_mu(mNearZero, res, invfactor2, resl, vl);
}

forceinline void process_linear_f16m1(vfloat16m1_t x, vfloat16m1_t& xc, vbool16_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f16m1_b16(abs_f16m1(x, vl), EXPM1_LINEAR_THRESHOLD_F16, vl);
    xc = __riscv_vfmerge_vfm_f16m1(x, ZERO_F16, linearMask, vl);
}

forceinline void set_fe_underflow_f16m1(vfloat16m1_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    volatile vfloat16m1_t tmp = __riscv_vfmadd_vf_f16m1(res, (FLOAT16_T)0x1p-14f16, res, vl);
#endif
}

// ---------------------------- m2 ----------------------------

forceinline void check_special_cases_f16m2(vfloat16m2_t& x, vfloat16m2_t& special, vbool8_t& specialMask,
    FLOAT16_T overflowThreshold, FLOAT16_T underflowThreshold, FLOAT16_T underflowValue, bool throwUnderflow, size_t vl)
{
    vuint16m2_t uxi = __riscv_vreinterpret_v_f16m2_u16m2(x);

    vbool8_t underflowMask = __riscv_vmsgtu_vx_u16m2_b8(uxi, RVVMF_EXP_AS_UINT16(underflowThreshold), vl);
    special = __riscv_vfmerge_vfm_f16m2(x, underflowValue, underflowMask, vl);
    specialMask = underflowMask;

#ifndef __FAST_MATH__
    uint16_t pinf = 0x7c00;
    uint16_t qnan = 0x7e00;
    vuint16m2_t absxui = __riscv_vand_vx_u16m2(uxi, (uint16_t)0x7fff, vl);
    vint16m2_t xi = __riscv_vreinterpret_v_f16m2_i16m2(x);

    vbool8_t nanMask = __riscv_vmsgtu_vx_u16m2_b8(absxui, pinf, vl);
    vbool8_t snanMask = __riscv_vmsltu_vx_u16m2_b8(absxui, qnan, vl);
    vbool8_t regularMask = __riscv_vmsltu_vx_u16m2_b8(absxui, pinf, vl);
    vbool8_t overflowMask = __riscv_vmsgt_vx_i16m2_b8(xi, RVVMF_EXP_AS_INT16(overflowThreshold), vl);

    special = __riscv_vfmerge_vfm_f16m2(special, RVVMF_EXP_AS_FP16(pinf), overflowMask, vl);
    special = __riscv_vfmerge_vfm_f16m2(special, RVVMF_EXP_AS_FP16(qnan), nanMask, vl);

    specialMask = __riscv_vmor_mm_b8(specialMask, __riscv_vmor_mm_b8(overflowMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b8(__riscv_vmand_mm_b8(nanMask, snanMask, vl), vl))
        RVVMF_EXP_CALL_FE_INVALID();
    if (__riscv_vcpop_m_b8(__riscv_vmand_mm_b8(overflowMask, regularMask, vl), vl))
        RVVMF_EXP_CALL_FE_OVERFLOW();
    if (throwUnderflow && __riscv_vcpop_m_b8(__riscv_vmand_mm_b8(underflowMask, regularMask, vl), vl))
        RVVMF_EXP_CALL_FE_UNDERFLOW();
#endif

    x = __riscv_vfmerge_vfm_f16m2(x, ZERO_F16, specialMask, vl);
}

forceinline void do_exp_argument_reduction_hl_f16m2(vfloat16m2_t x,
    vfloat16m2_t& yh, vfloat16m2_t& yl, vuint16m2_t& ei, vuint16m2_t& fi, size_t vl)
{
    vfloat16m2_t vmagicConst1 = __riscv_vfmv_v_f_f16m2(MAGIC_CONST_1_F16, vl);
    vfloat16m2_t h = __riscv_vfmadd_vf_f16m2(x, INV_LOG2_2K_F16, vmagicConst1, vl);
    vuint16m2_t hi = __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(h), MASK_HI_BIT_F16, vl);
    fi = __riscv_vand_vx_u16m2(hi, MASK_FI_BIT_F16, vl);
    ei = __riscv_vsrl_vx_u16m2(hi, TABLE_SIZE_DEG_F16, vl);
    h = __riscv_vfsub_vv_f16m2(h, vmagicConst1, vl);
    fma12_ver2p2_vf_f16m2(h, M_LOG2_2K_L_F16, __riscv_vfmadd_vf_f16m2(h, M_LOG2_2K_H_F16, x, vl), yh, yl, vl);
    yl = __riscv_vfmacc_vf_f16m2(yl, M_LOG2_2K_LL_F16, h, vl);
    fast_2_sum_vv_f16m2(yh, yl, yh, yl, vl);
}

forceinline void do_exp2_argument_reduction_f16m2(vfloat16m2_t x, vfloat16m2_t& y,
    vuint16m2_t& ei, vuint16m2_t& fi, size_t vl)
{
    vfloat16m2_t vmagicConst1 = __riscv_vfmv_v_f_f16m2(MAGIC_CONST_1_F16, vl);
    vfloat16m2_t h = __riscv_vfmadd_vf_f16m2(x, EXP2_TABLE_SIZE_DEG_F16, vmagicConst1, vl);
    vuint16m2_t hi = __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(h), MASK_HI_BIT_F16, vl);
    fi = __riscv_vand_vx_u16m2(hi, MASK_FI_BIT_F16, vl);
    ei = __riscv_vsrl_vx_u16m2(hi, TABLE_SIZE_DEG_F16, vl);
    h = __riscv_vfsub_vv_f16m2(h, vmagicConst1, vl);
    y = __riscv_vfmadd_vf_f16m2(h, M_EXP2_M_TABLE_SIZE_DEG_F16, x, vl);
}

forceinline void get_table_values_hl_f16m2(
    vuint16m2_t& index, vfloat16m2_t& th, vfloat16m2_t& tl, size_t vl)
{
    index = __riscv_vsll_vx_u16m2(index, (size_t)1, vl);
    th = __riscv_vloxei16_v_f16m2(LOOK_UP_TABLE_HIGH_F16, index, vl);
    tl = __riscv_vloxei16_v_f16m2(LOOK_UP_TABLE_LOW_F16, index, vl);
}

forceinline void calculate_exp_polynom_hl_f16m2(vfloat16m2_t yh, vfloat16m2_t yl,
    vfloat16m2_t& ph, vfloat16m2_t& pl, size_t vl)
{
#ifndef __FAST_MATH__
    vfloat16m2_t yhs = __riscv_vfmerge_vfm_f16m2(yh, ZERO_F16, __riscv_vmflt_vf_f16m2_b8(
        abs_f16m2(yh, vl), EXP_THRESHOLD_AVOID_UNDERFFLOW_IN_SQR_F16, vl), vl);
#else
    vfloat16m2_t yhs = yh;
#endif
    vfloat16m2_t r = calc_polynom_deg_1_f16m2(yh, EXP_POL_COEFF_2_F16, EXP_POL_COEFF_3_F16, vl);
    r = __riscv_vfmul_vv_f16m2(r, yh, vl);

    vfloat16m2_t tmp = __riscv_vfmul_vf_f16m2(yh, FACTOR_F16, vl);
    ph = __riscv_vfmadd_vv_f16m2(yh, r, tmp, vl);
    tmp = __riscv_vfsub_vv_f16m2(tmp, ph, vl);
    pl = __riscv_vfmacc_vv_f16m2(__riscv_vfmul_vf_f16m2(tmp, FACTOR_F16, vl), r,
        __riscv_vfmul_vf_f16m2(yhs, FACTOR_F16, vl), vl);

    pl = __riscv_vfmacc_vf_f16m2(pl, FACTOR_F16, __riscv_vfmul_vf_f16m2(yl, FACTOR_F16, vl), vl);
}

forceinline void calculate_expm1_polynom_hl_f16m2(vfloat16m2_t yh, vfloat16m2_t yl,
    vfloat16m2_t& ph, vfloat16m2_t& pl, size_t vl)
{
    calculate_exp_polynom_hl_f16m2(yh, yl, ph, pl, vl);
}

forceinline void calculate_exp2_polynom_hl12_f16m2(vfloat16m2_t yh, vfloat16m2_t& ph,
    vfloat16m2_t& pl, size_t vl)
{
#ifndef __FAST_MATH__
    vfloat16m2_t yhs = __riscv_vfmerge_vfm_f16m2(yh, ZERO_F16, __riscv_vmflt_vf_f16m2_b8(
        abs_f16m2(yh, vl), EXP2_THRESHOLD_AVOID_UNDERFFLOW_IN_SQR_F16, vl), vl);
#else
    vfloat16m2_t yhs = yh;
#endif
    vfloat16m2_t r = calc_polynom_deg_1_f16m2(yh, EXP2_POL_COEFF_2_F16, EXP2_POL_COEFF_3_F16, vl);
    r = __riscv_vfmul_vv_f16m2(r, yh, vl);
    r = __riscv_vfmadd_vv_f16m2(yhs, r, __riscv_vfmul_vf_f16m2(yhs, EXP2_POL_COEFF_1_L_F16, vl), vl);
    fma12_ver2p1_vf_f16m2(yh, EXP2_POL_COEFF_1_H_F16, r, ph, pl, vl);
    pl = __riscv_vfmul_vf_f16m2(pl, FACTOR_F16, vl);
}

forceinline void reconstruct_exp_hl_hl_f16m2(vfloat16m2_t th, vfloat16m2_t tl,
    vfloat16m2_t pm1h, vfloat16m2_t pm1l, vuint16m2_t ei, vfloat16m2_t& res, size_t vl)
{
    vfloat16m2_t zh, zl;
    fma12_vv_f16m2(th, pm1h, __riscv_vfmul_vf_f16m2(th, FACTOR_F16, vl), zh, zl, vl);
    zl = __riscv_vfmadd_vf_f16m2(zl, FACTOR_F16, __riscv_vfmadd_vv_f16m2(th, pm1l,
        __riscv_vfmadd_vv_f16m2(tl, pm1h, __riscv_vfmul_vf_f16m2(tl, FACTOR_F16, vl), vl), vl), vl);
    zl = __riscv_vfadd_vf_f16m2(zl, 0x1p-1f16, vl);

    ei = __riscv_vsll_vx_u16m2(__riscv_vadd_vx_u16m2(ei, (uint16_t)0x001e, vl), (size_t)9, vl);
    vuint16m2_t shifti = __riscv_vand_vx_u16m2(ei, (uint16_t)0x0200, vl);
    vuint16m2_t pow2e1i = __riscv_vadd_vv_u16m2(ei, shifti, vl);
    vuint16m2_t pow2e2i = __riscv_vsub_vv_u16m2(ei, shifti, vl);
    vfloat16m2_t pow2e1 = __riscv_vreinterpret_v_u16m2_f16m2(pow2e1i);
    vfloat16m2_t pow2e2 = __riscv_vreinterpret_v_u16m2_f16m2(pow2e2i);
    vfloat16m2_t invpow2e2 = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vrsub_vx_u16m2(pow2e2i, (uint16_t)0x7800, vl));
    vfloat16m2_t pow2e1f = __riscv_vfmul_vf_f16m2(pow2e1, INV_FACTOR_F16, vl);
    vfloat16m2_t pow2e2f = __riscv_vfmul_vf_f16m2(pow2e2, INV_FACTOR_F16, vl);

    zh = __riscv_vfmul_vv_f16m2(zh, pow2e1f, vl);
    res = __riscv_vfmul_vv_f16m2(zh, pow2e2, vl);
    vfloat16m2_t tmp = __riscv_vfmul_vf_f16m2(__riscv_vfnmsac_vv_f16m2(zh, res, invpow2e2, vl), FACTOR_F16, vl);
    res = __riscv_vfmacc_vv_f16m2(res, __riscv_vfmadd_vv_f16m2(zl, pow2e1f, tmp, vl), pow2e2f, vl);
}

forceinline void reconstruct_expm1_f16m2(vfloat16m2_t th, vfloat16m2_t tl,
    vfloat16m2_t pm1h, vfloat16m2_t pm1l, vuint16m2_t ei, vfloat16m2_t& res, size_t vl)
{
    vbool8_t mNearZero = __riscv_vmseq_vx_u16m2_b8(ei, (uint16_t)0, vl);

    vfloat16m2_t rh, rl, rll;
    fast_2_sum_fv_f16m2(FACTOR_F16, pm1h, rh, rl, vl);
    fast_2_sum_vv_f16m2(__riscv_vfmul_vf_f16m2(rl, FACTOR_F16, vl), pm1l, rl, rll, vl);

    vfloat16m2_t zh, zl;
    zh = __riscv_vfmul_vv_f16m2(th, rh, vl);
    zl = __riscv_vfmsub_vv_f16m2(th, rh, zh, vl);
    zl = __riscv_vfmadd_vf_f16m2(zl, FACTOR_F16, __riscv_vfmadd_vv_f16m2(
        th, rl, __riscv_vfmul_vv_f16m2(tl, rh, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f16m2(zl, INV_FACTOR_F16, zh, vl);
    rl = __riscv_vfmadd_vf_f16m2(__riscv_vfsub_vv_f16m2(zh, rh, vl), FACTOR_F16, zl, vl);

#ifndef __FAST_MATH__   
    rl  = __riscv_vfmerge_vfm_f16m2(rl, ZERO_F16, __riscv_vmflt_vf_f16m2_b8(abs_f16m2(rl, vl), 0x1p0f16, vl), vl);
#endif

    constexpr FLOAT16_T invfactore = INV_FACTOR_F16*0x1p1f16;
    vuint16m2_t pow2ei = __riscv_vsll_vx_u16m2(__riscv_vadd_vx_u16m2(ei, (uint16_t)14, vl), (size_t)10, vl);
    vfloat16m2_t pow2e = __riscv_vfmul_vf_f16m2(__riscv_vreinterpret_v_u16m2_f16m2(pow2ei), invfactore, vl);
    rh = __riscv_vfmul_vv_f16m2(rh, pow2e, vl);

    vfloat16m2_t sh1, sl1;
    sh1 = __riscv_vfadd_vf_f16m2(rh, EXPM1_UNDERFLOW_VALUE_F16, vl);
    sl1 = __riscv_vfsub_vv_f16m2(sh1, rh, vl);
    sl1 = __riscv_vfadd_vv_f16m2(__riscv_vfsub_vv_f16m2(rh, __riscv_vfsub_vv_f16m2(sh1, sl1, vl), vl),
        __riscv_vfrsub_vf_f16m2(sl1, EXPM1_UNDERFLOW_VALUE_F16, vl), vl);

    vfloat16m2_t tmp, resl;
    tmp = __riscv_vfmadd_vv_f16m2(rl, pow2e, __riscv_vfmul_vf_f16m2(sl1, FACTOR_F16, vl), vl);
    res = __riscv_vfmadd_vf_f16m2(tmp, INV_FACTOR_F16, sh1, vl);
    resl = __riscv_vfmadd_vf_f16m2(__riscv_vfsub_vv_f16m2(sh1, res, vl), FACTOR_F16, tmp, vl);
    resl = __riscv_vfmadd_vf_f16m2_mu(mNearZero, resl, FACTOR_F16, rll, vl);
    constexpr FLOAT16_T invfactor2 = INV_FACTOR_F16*INV_FACTOR_F16;
    res = __riscv_vfmacc_vf_f16m2_mu(mNearZero, res, invfactor2, resl, vl);
}

forceinline void process_linear_f16m2(vfloat16m2_t x, vfloat16m2_t& xc, vbool8_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f16m2_b8(abs_f16m2(x, vl), EXPM1_LINEAR_THRESHOLD_F16, vl);
    xc = __riscv_vfmerge_vfm_f16m2(x, ZERO_F16, linearMask, vl);
}

forceinline void set_fe_underflow_f16m2(vfloat16m2_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    volatile vfloat16m2_t tmp = __riscv_vfmadd_vf_f16m2(res, (FLOAT16_T)0x1p-14f16, res, vl);
#endif
}

// ---------------------------- m4 ----------------------------

forceinline void check_special_cases_f16m4(vfloat16m4_t& x, vfloat16m4_t& special, vbool4_t& specialMask,
    FLOAT16_T overflowThreshold, FLOAT16_T underflowThreshold, FLOAT16_T underflowValue, bool throwUnderflow, size_t vl)
{
    vuint16m4_t uxi = __riscv_vreinterpret_v_f16m4_u16m4(x);

    vbool4_t underflowMask = __riscv_vmsgtu_vx_u16m4_b4(uxi, RVVMF_EXP_AS_UINT16(underflowThreshold), vl);
    special = __riscv_vfmerge_vfm_f16m4(x, underflowValue, underflowMask, vl);
    specialMask = underflowMask;

#ifndef __FAST_MATH__
    uint16_t pinf = 0x7c00;
    uint16_t qnan = 0x7e00;
    vuint16m4_t absxui = __riscv_vand_vx_u16m4(uxi, (uint16_t)0x7fff, vl);
    vint16m4_t xi = __riscv_vreinterpret_v_f16m4_i16m4(x);

    vbool4_t nanMask = __riscv_vmsgtu_vx_u16m4_b4(absxui, pinf, vl);
    vbool4_t snanMask = __riscv_vmsltu_vx_u16m4_b4(absxui, qnan, vl);
    vbool4_t regularMask = __riscv_vmsltu_vx_u16m4_b4(absxui, pinf, vl);
    vbool4_t overflowMask = __riscv_vmsgt_vx_i16m4_b4(xi, RVVMF_EXP_AS_INT16(overflowThreshold), vl);

    special = __riscv_vfmerge_vfm_f16m4(special, RVVMF_EXP_AS_FP16(pinf), overflowMask, vl);
    special = __riscv_vfmerge_vfm_f16m4(special, RVVMF_EXP_AS_FP16(qnan), nanMask, vl);

    specialMask = __riscv_vmor_mm_b4(specialMask, __riscv_vmor_mm_b4(overflowMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b4(__riscv_vmand_mm_b4(nanMask, snanMask, vl), vl))
        RVVMF_EXP_CALL_FE_INVALID();
    if (__riscv_vcpop_m_b4(__riscv_vmand_mm_b4(overflowMask, regularMask, vl), vl))
        RVVMF_EXP_CALL_FE_OVERFLOW();
    if (throwUnderflow && __riscv_vcpop_m_b4(__riscv_vmand_mm_b4(underflowMask, regularMask, vl), vl))
        RVVMF_EXP_CALL_FE_UNDERFLOW();
#endif

    x = __riscv_vfmerge_vfm_f16m4(x, ZERO_F16, specialMask, vl);
}

forceinline void do_exp_argument_reduction_hl_f16m4(vfloat16m4_t x,
    vfloat16m4_t& yh, vfloat16m4_t& yl, vuint16m4_t& ei, vuint16m4_t& fi, size_t vl)
{
    vfloat16m4_t vmagicConst1 = __riscv_vfmv_v_f_f16m4(MAGIC_CONST_1_F16, vl);
    vfloat16m4_t h = __riscv_vfmadd_vf_f16m4(x, INV_LOG2_2K_F16, vmagicConst1, vl);
    vuint16m4_t hi = __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(h), MASK_HI_BIT_F16, vl);
    fi = __riscv_vand_vx_u16m4(hi, MASK_FI_BIT_F16, vl);
    ei = __riscv_vsrl_vx_u16m4(hi, TABLE_SIZE_DEG_F16, vl);
    h = __riscv_vfsub_vv_f16m4(h, vmagicConst1, vl);
    fma12_ver2p2_vf_f16m4(h, M_LOG2_2K_L_F16, __riscv_vfmadd_vf_f16m4(h, M_LOG2_2K_H_F16, x, vl), yh, yl, vl);
    yl = __riscv_vfmacc_vf_f16m4(yl, M_LOG2_2K_LL_F16, h, vl);
    fast_2_sum_vv_f16m4(yh, yl, yh, yl, vl);
}

forceinline void do_exp2_argument_reduction_f16m4(vfloat16m4_t x, vfloat16m4_t& y,
    vuint16m4_t& ei, vuint16m4_t& fi, size_t vl)
{
    vfloat16m4_t vmagicConst1 = __riscv_vfmv_v_f_f16m4(MAGIC_CONST_1_F16, vl);
    vfloat16m4_t h = __riscv_vfmadd_vf_f16m4(x, EXP2_TABLE_SIZE_DEG_F16, vmagicConst1, vl);
    vuint16m4_t hi = __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(h), MASK_HI_BIT_F16, vl);
    fi = __riscv_vand_vx_u16m4(hi, MASK_FI_BIT_F16, vl);
    ei = __riscv_vsrl_vx_u16m4(hi, TABLE_SIZE_DEG_F16, vl);
    h = __riscv_vfsub_vv_f16m4(h, vmagicConst1, vl);
    y = __riscv_vfmadd_vf_f16m4(h, M_EXP2_M_TABLE_SIZE_DEG_F16, x, vl);
}

forceinline void get_table_values_hl_f16m4(
    vuint16m4_t& index, vfloat16m4_t& th, vfloat16m4_t& tl, size_t vl)
{
    index = __riscv_vsll_vx_u16m4(index, (size_t)1, vl);
    th = __riscv_vloxei16_v_f16m4(LOOK_UP_TABLE_HIGH_F16, index, vl);
    tl = __riscv_vloxei16_v_f16m4(LOOK_UP_TABLE_LOW_F16, index, vl);
}

forceinline void calculate_exp_polynom_hl_f16m4(vfloat16m4_t yh, vfloat16m4_t yl,
    vfloat16m4_t& ph, vfloat16m4_t& pl, size_t vl)
{
#ifndef __FAST_MATH__
    vfloat16m4_t yhs = __riscv_vfmerge_vfm_f16m4(yh, ZERO_F16, __riscv_vmflt_vf_f16m4_b4(
        abs_f16m4(yh, vl), EXP_THRESHOLD_AVOID_UNDERFFLOW_IN_SQR_F16, vl), vl);
#else
    vfloat16m4_t yhs = yh;
#endif
    vfloat16m4_t r = calc_polynom_deg_1_f16m4(yh, EXP_POL_COEFF_2_F16, EXP_POL_COEFF_3_F16, vl);
    r = __riscv_vfmul_vv_f16m4(r, yh, vl);

    vfloat16m4_t tmp = __riscv_vfmul_vf_f16m4(yh, FACTOR_F16, vl);
    ph = __riscv_vfmadd_vv_f16m4(yh, r, tmp, vl);
    tmp = __riscv_vfsub_vv_f16m4(tmp, ph, vl);
    pl = __riscv_vfmacc_vv_f16m4(__riscv_vfmul_vf_f16m4(tmp, FACTOR_F16, vl), r,
        __riscv_vfmul_vf_f16m4(yhs, FACTOR_F16, vl), vl);

    pl = __riscv_vfmacc_vf_f16m4(pl, FACTOR_F16, __riscv_vfmul_vf_f16m4(yl, FACTOR_F16, vl), vl);
}

forceinline void calculate_expm1_polynom_hl_f16m4(vfloat16m4_t yh, vfloat16m4_t yl,
    vfloat16m4_t& ph, vfloat16m4_t& pl, size_t vl)
{
    calculate_exp_polynom_hl_f16m4(yh, yl, ph, pl, vl);
}

forceinline void calculate_exp2_polynom_hl12_f16m4(vfloat16m4_t yh, vfloat16m4_t& ph,
    vfloat16m4_t& pl, size_t vl)
{
#ifndef __FAST_MATH__
    vfloat16m4_t yhs = __riscv_vfmerge_vfm_f16m4(yh, ZERO_F16, __riscv_vmflt_vf_f16m4_b4(
        abs_f16m4(yh, vl), EXP2_THRESHOLD_AVOID_UNDERFFLOW_IN_SQR_F16, vl), vl);
#else
    vfloat16m4_t yhs = yh;
#endif
    vfloat16m4_t r = calc_polynom_deg_1_f16m4(yh, EXP2_POL_COEFF_2_F16, EXP2_POL_COEFF_3_F16, vl);
    r = __riscv_vfmul_vv_f16m4(r, yh, vl);
    r = __riscv_vfmadd_vv_f16m4(yhs, r, __riscv_vfmul_vf_f16m4(yhs, EXP2_POL_COEFF_1_L_F16, vl), vl);
    fma12_ver2p1_vf_f16m4(yh, EXP2_POL_COEFF_1_H_F16, r, ph, pl, vl);
    pl = __riscv_vfmul_vf_f16m4(pl, FACTOR_F16, vl);
}

forceinline void reconstruct_exp_hl_hl_f16m4(vfloat16m4_t th, vfloat16m4_t tl,
    vfloat16m4_t pm1h, vfloat16m4_t pm1l, vuint16m4_t ei, vfloat16m4_t& res, size_t vl)
{
    vfloat16m4_t zh, zl;
    fma12_vv_f16m4(th, pm1h, __riscv_vfmul_vf_f16m4(th, FACTOR_F16, vl), zh, zl, vl);
    zl = __riscv_vfmadd_vf_f16m4(zl, FACTOR_F16, __riscv_vfmadd_vv_f16m4(th, pm1l,
        __riscv_vfmadd_vv_f16m4(tl, pm1h, __riscv_vfmul_vf_f16m4(tl, FACTOR_F16, vl), vl), vl), vl);
    zl = __riscv_vfadd_vf_f16m4(zl, 0x1p-1f16, vl);

    ei = __riscv_vsll_vx_u16m4(__riscv_vadd_vx_u16m4(ei, (uint16_t)0x001e, vl), (size_t)9, vl);
    vuint16m4_t shifti = __riscv_vand_vx_u16m4(ei, (uint16_t)0x0200, vl);
    vuint16m4_t pow2e1i = __riscv_vadd_vv_u16m4(ei, shifti, vl);
    vuint16m4_t pow2e2i = __riscv_vsub_vv_u16m4(ei, shifti, vl);
    vfloat16m4_t pow2e1 = __riscv_vreinterpret_v_u16m4_f16m4(pow2e1i);
    vfloat16m4_t pow2e2 = __riscv_vreinterpret_v_u16m4_f16m4(pow2e2i);
    vfloat16m4_t invpow2e2 = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vrsub_vx_u16m4(pow2e2i, (uint16_t)0x7800, vl));
    vfloat16m4_t pow2e1f = __riscv_vfmul_vf_f16m4(pow2e1, INV_FACTOR_F16, vl);
    vfloat16m4_t pow2e2f = __riscv_vfmul_vf_f16m4(pow2e2, INV_FACTOR_F16, vl);

    zh = __riscv_vfmul_vv_f16m4(zh, pow2e1f, vl);
    res = __riscv_vfmul_vv_f16m4(zh, pow2e2, vl);
    vfloat16m4_t tmp = __riscv_vfmul_vf_f16m4(__riscv_vfnmsac_vv_f16m4(zh, res, invpow2e2, vl), FACTOR_F16, vl);
    res = __riscv_vfmacc_vv_f16m4(res, __riscv_vfmadd_vv_f16m4(zl, pow2e1f, tmp, vl), pow2e2f, vl);
}

forceinline void reconstruct_expm1_f16m4(vfloat16m4_t th, vfloat16m4_t tl,
    vfloat16m4_t pm1h, vfloat16m4_t pm1l, vuint16m4_t ei, vfloat16m4_t& res, size_t vl)
{
    vbool4_t mNearZero = __riscv_vmseq_vx_u16m4_b4(ei, (uint16_t)0, vl);

    vfloat16m4_t rh, rl, rll;
    fast_2_sum_fv_f16m4(FACTOR_F16, pm1h, rh, rl, vl);
    fast_2_sum_vv_f16m4(__riscv_vfmul_vf_f16m4(rl, FACTOR_F16, vl), pm1l, rl, rll, vl);

    vfloat16m4_t zh, zl;
    zh = __riscv_vfmul_vv_f16m4(th, rh, vl);
    zl = __riscv_vfmsub_vv_f16m4(th, rh, zh, vl);
    zl = __riscv_vfmadd_vf_f16m4(zl, FACTOR_F16, __riscv_vfmadd_vv_f16m4(
        th, rl, __riscv_vfmul_vv_f16m4(tl, rh, vl), vl), vl);
    rh = __riscv_vfmadd_vf_f16m4(zl, INV_FACTOR_F16, zh, vl);
    rl = __riscv_vfmadd_vf_f16m4(__riscv_vfsub_vv_f16m4(zh, rh, vl), FACTOR_F16, zl, vl);

#ifndef __FAST_MATH__   
    rl  = __riscv_vfmerge_vfm_f16m4(rl, ZERO_F16, __riscv_vmflt_vf_f16m4_b4(abs_f16m4(rl, vl), 0x1p0f16, vl), vl);
#endif

    constexpr FLOAT16_T invfactore = INV_FACTOR_F16*0x1p1f16;
    vuint16m4_t pow2ei = __riscv_vsll_vx_u16m4(__riscv_vadd_vx_u16m4(ei, (uint16_t)14, vl), (size_t)10, vl);
    vfloat16m4_t pow2e = __riscv_vfmul_vf_f16m4(__riscv_vreinterpret_v_u16m4_f16m4(pow2ei), invfactore, vl);
    rh = __riscv_vfmul_vv_f16m4(rh, pow2e, vl);

    vfloat16m4_t sh1, sl1;
    sh1 = __riscv_vfadd_vf_f16m4(rh, EXPM1_UNDERFLOW_VALUE_F16, vl);
    sl1 = __riscv_vfsub_vv_f16m4(sh1, rh, vl);
    sl1 = __riscv_vfadd_vv_f16m4(__riscv_vfsub_vv_f16m4(rh, __riscv_vfsub_vv_f16m4(sh1, sl1, vl), vl),
        __riscv_vfrsub_vf_f16m4(sl1, EXPM1_UNDERFLOW_VALUE_F16, vl), vl);

    vfloat16m4_t tmp, resl;
    tmp = __riscv_vfmadd_vv_f16m4(rl, pow2e, __riscv_vfmul_vf_f16m4(sl1, FACTOR_F16, vl), vl);
    res = __riscv_vfmadd_vf_f16m4(tmp, INV_FACTOR_F16, sh1, vl);
    resl = __riscv_vfmadd_vf_f16m4(__riscv_vfsub_vv_f16m4(sh1, res, vl), FACTOR_F16, tmp, vl);
    resl = __riscv_vfmadd_vf_f16m4_mu(mNearZero, resl, FACTOR_F16, rll, vl);
    constexpr FLOAT16_T invfactor2 = INV_FACTOR_F16*INV_FACTOR_F16;
    res = __riscv_vfmacc_vf_f16m4_mu(mNearZero, res, invfactor2, resl, vl);
}

forceinline void process_linear_f16m4(vfloat16m4_t x, vfloat16m4_t& xc, vbool4_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f16m4_b4(abs_f16m4(x, vl), EXPM1_LINEAR_THRESHOLD_F16, vl);
    xc = __riscv_vfmerge_vfm_f16m4(x, ZERO_F16, linearMask, vl);
}

forceinline void set_fe_underflow_f16m4(vfloat16m4_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    volatile vfloat16m4_t tmp = __riscv_vfmadd_vf_f16m4(res, (FLOAT16_T)0x1p-14f16, res, vl);
#endif
}
