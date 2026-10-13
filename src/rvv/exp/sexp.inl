/*
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of
 * Nizhny Novgorod and its affiliates. All rights reserved.
 *
 * Copyright 2025-2026 The RVVMF Authors (Elena Panova)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  sexp.inl                                     *
 *   Contains: helper built-in functions for exp, exp2   *
 *             and expm1 functions (float32_t)           *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#include "exp_utilities.inl"

constexpr float ZERO_F32 = 0.0f;
constexpr float ONE_F32 = 1.0f;

constexpr float EXP_EXPM1_OVERFLOW_THRESHOLD_F32 = 0x1.62e42ep6f;
constexpr float EXP2_EXP2M1_OVERFLOW_THRESHOLD_F32 = 0x1.fffffep6f;
constexpr float EXP_SUBNORMAL_THRESHOLD_F32 = -0x1.5d589ep6f;
constexpr float EXP2_SUBNORMAL_THRESHOLD_F32 = -0x1.f8p6f;
constexpr float EXP_ZERO_THRESHOLD_F32 = -0x1.9fe368p6f;
constexpr float EXP2_ZERO_THRESHOLD_F32 = -0x1.2bfffep7f;
constexpr float EXP_UNDERFLOW_VALUE_F32 = 0.0f;
constexpr float EXPM1_UNDERFLOW_THRESHOLD_F32 = -0x1.154244p4f;
constexpr float EXPM1_LINEAR_THRESHOLD_F32 = 0x1.6a09e8p-24f;
constexpr float EXPM1_UNDERFLOW_VALUE_F32 = -1.0f;

constexpr size_t TABLE_SIZE_DEG_F32 = 4;
constexpr float EXP2_TABLE_SIZE_DEG_F32 = 0x1p4f;
constexpr float M_EXP2_M_TABLE_SIZE_DEG_F32 = -0x1p-4f;
constexpr uint32_t MASK_FI_BIT_F32 = 0x0000000f;
constexpr uint32_t MASK_HI_BIT_F32 = 0x00001fff;
constexpr float MAGIC_CONST_1_F32 = 12582912.0f;    // 1.5*2^23
constexpr float INV_LOG2_2K_F32 = 0x1.715476p4f;
constexpr float M_LOG2_2K_H_F32 = -0x1.62ep-5f;
constexpr float M_LOG2_2K_L_F32 = -0x1.0bfbe8p-19f;
constexpr float M_LOG2_2K_LL_F32 = -0x1.cf79acp-44f;

constexpr uint32_t FACTOR_DEG_F32 = 127 - 57;
constexpr uint32_t MAGIC_CONST_2_F32 = 0x000001af;  // uint9_t(-24-57)
constexpr float FACTOR_F32 = 0x1p57f;
constexpr float SHIFT_AVOID_UNDERFFLOW_IN_SQR_F32 = 0x1p-41f;

static const float LOOK_UP_TABLE_HIGH_F32[16] = {
    0x1p0f, 0x1.0b5586p0f, 0x1.172b84p0f, 0x1.2387a6p0f,
    0x1.306fep0f, 0x1.3dea64p0f, 0x1.4bfdaep0f, 0x1.5ab07ep0f,
    0x1.6a09e6p0f, 0x1.7a1148p0f, 0x1.8ace54p0f, 0x1.9c4918p0f,
    0x1.ae89fap0f, 0x1.c199bep0f, 0x1.d5818ep0f, 0x1.ea4afap0f
};
static const float LOOK_UP_TABLE_LOW_F32[16] = {
    0.0f, 0x1.9f3122p-25f, -0x1.c15742p-27f, 0x1.ceac48p-25f,
    0x1.4636e2p-25f, 0x1.824684p-25f, -0x1.593abcp-25f, -0x1.5bd5ecp-27f,
    0x1.9fcef4p-26f, -0x1.829fdp-25f, 0x1.15506ep-27f, 0x1.51f848p-27f,
    -0x1.a94b14p-26f, -0x1.3d56b2p-27f, -0x1.822dbcp-27f, 0x1.52486cp-27f
};

constexpr float EXP_POL_COEFF_2_F32 = 0x1p-1f * FACTOR_F32;
constexpr float EXP_POL_COEFF_3_F32 = 0x1.5556dep-3f * FACTOR_F32;
constexpr float EXP_POL_COEFF_4_F32 = 0x1.555696p-5f * FACTOR_F32;

constexpr float EXP2_POL_COEFF_1_F32 = 0x1.62e43p-1f * FACTOR_F32;
constexpr float EXP2_POL_COEFF_2_F32 = 0x1.ebfbep-3f * FACTOR_F32;
constexpr float EXP2_POL_COEFF_3_F32 = 0x1.c6ae08p-5f * FACTOR_F32;
constexpr float EXP2_POL_COEFF_4_F32 = 0x1.3b27cep-7f * FACTOR_F32;

// ---------------------------- m1 ----------------------------

forceinline void check_special_cases_f32m1(vfloat32m1_t& x, vfloat32m1_t& special, vbool32_t& specialMask,
    float overflowThreshold, float underflowThreshold, float underflowValue, bool throwUnderflow, size_t vl)
{
    vuint32m1_t uxi = __riscv_vreinterpret_v_f32m1_u32m1(x);

    vbool32_t underflowMask = __riscv_vmsgtu_vx_u32m1_b32(uxi, RVVMF_EXP_AS_UINT32(underflowThreshold), vl);
    special = __riscv_vfmerge_vfm_f32m1(x, underflowValue, underflowMask, vl);
    specialMask = underflowMask;

#ifndef __FAST_MATH__
    uint32_t pinf = 0x7f800000;
    uint32_t qnan = 0x7fc00000;
    vuint32m1_t absxui = __riscv_vand_vx_u32m1(uxi, (uint32_t)0x7fffffff, vl);
    vint32m1_t xi = __riscv_vreinterpret_v_f32m1_i32m1(x);

    vbool32_t nanMask = __riscv_vmsgtu_vx_u32m1_b32(absxui, pinf, vl);
    vbool32_t snanMask = __riscv_vmsltu_vx_u32m1_b32(absxui, qnan, vl);
    vbool32_t regularMask = __riscv_vmsltu_vx_u32m1_b32(absxui, pinf, vl);
    vbool32_t overflowMask = __riscv_vmsgt_vx_i32m1_b32(xi, RVVMF_EXP_AS_INT32(overflowThreshold), vl);

    special = __riscv_vfmerge_vfm_f32m1(special, RVVMF_EXP_AS_FP32(pinf), overflowMask, vl);
    special = __riscv_vfmerge_vfm_f32m1(special, RVVMF_EXP_AS_FP32(qnan), nanMask, vl);

    specialMask = __riscv_vmor_mm_b32(specialMask, __riscv_vmor_mm_b32(overflowMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b32(__riscv_vmand_mm_b32(nanMask, snanMask, vl), vl))
        RVVMF_EXP_CALL_FE_INVALID();
    if (__riscv_vcpop_m_b32(__riscv_vmand_mm_b32(overflowMask, regularMask, vl), vl))
        RVVMF_EXP_CALL_FE_OVERFLOW();
    if (throwUnderflow && __riscv_vcpop_m_b32(__riscv_vmand_mm_b32(underflowMask, regularMask, vl), vl))
        RVVMF_EXP_CALL_FE_UNDERFLOW();
#endif

    x = __riscv_vfmerge_vfm_f32m1(x, ZERO_F32, specialMask, vl);
}

forceinline void do_exp_argument_reduction_hl_f32m1(vfloat32m1_t x,
    vfloat32m1_t& yh, vfloat32m1_t& yl, vuint32m1_t& ei, vuint32m1_t& fi, size_t vl)
{
    vfloat32m1_t vmagicConst1 = __riscv_vfmv_v_f_f32m1(MAGIC_CONST_1_F32, vl);
    vfloat32m1_t h = __riscv_vfmadd_vf_f32m1(x, INV_LOG2_2K_F32, vmagicConst1, vl);
    vuint32m1_t hi = __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(h), MASK_HI_BIT_F32, vl);
    fi = __riscv_vand_vx_u32m1(hi, MASK_FI_BIT_F32, vl);
    ei = __riscv_vsrl_vx_u32m1(hi, TABLE_SIZE_DEG_F32, vl);
    h = __riscv_vfsub_vv_f32m1(h, vmagicConst1, vl);
    fma12_vf_f32m1(h, M_LOG2_2K_L_F32, __riscv_vfmadd_vf_f32m1(h, M_LOG2_2K_H_F32, x, vl), yh, yl, vl);
    yl = __riscv_vfmacc_vf_f32m1(yl, M_LOG2_2K_LL_F32, h, vl);
    fast_2_sum_vv_f32m1(yh, yl, yh, yl, vl);
}

forceinline void do_exp2_argument_reduction_f32m1(vfloat32m1_t x, vfloat32m1_t& y,
    vuint32m1_t& ei, vuint32m1_t& fi, size_t vl)
{
    vfloat32m1_t vmagicConst1 = __riscv_vfmv_v_f_f32m1(MAGIC_CONST_1_F32, vl);
    vfloat32m1_t h = __riscv_vfmadd_vf_f32m1(x, EXP2_TABLE_SIZE_DEG_F32, vmagicConst1, vl);
    vuint32m1_t hi = __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(h), MASK_HI_BIT_F32, vl);
    fi = __riscv_vand_vx_u32m1(hi, MASK_FI_BIT_F32, vl);
    ei = __riscv_vsrl_vx_u32m1(hi, TABLE_SIZE_DEG_F32, vl);
    h = __riscv_vfsub_vv_f32m1(h, vmagicConst1, vl);
    y = __riscv_vfmadd_vf_f32m1(h, M_EXP2_M_TABLE_SIZE_DEG_F32, x, vl);
}

forceinline void get_table_values_hl_f32m1(
    vuint32m1_t& index, vfloat32m1_t& th, vfloat32m1_t& tl, size_t vl)
{
    index = __riscv_vmul_vx_u32m1(index, uint32_t(sizeof(float)), vl);
    th = __riscv_vloxei32_v_f32m1(LOOK_UP_TABLE_HIGH_F32, index, vl);
    tl = __riscv_vloxei32_v_f32m1(LOOK_UP_TABLE_LOW_F32, index, vl);
}

forceinline void calculate_exp_polynom_hl_f32m1(vfloat32m1_t yh, vfloat32m1_t yl,
    vfloat32m1_t& ph, vfloat32m1_t& pl, size_t vl)
{
    vfloat32m1_t yhs = __riscv_vfadd_vf_f32m1(yh, SHIFT_AVOID_UNDERFFLOW_IN_SQR_F32, vl);
    vfloat32m1_t sqryh = __riscv_vfmul_vv_f32m1(yhs, yhs, vl);
    vfloat32m1_t r = calc_polynom_deg_2_f32m1(yh, EXP_POL_COEFF_2_F32, EXP_POL_COEFF_3_F32, EXP_POL_COEFF_4_F32, vl);
    fma12_vv_f32m1(sqryh, r, __riscv_vfmul_vf_f32m1(yh, FACTOR_F32, vl), ph, pl, vl);
    pl = __riscv_vfadd_vv_f32m1(pl, __riscv_vfmul_vf_f32m1(yl, FACTOR_F32, vl), vl);
}

forceinline void calculate_expm1_polynom_hl_f32m1(vfloat32m1_t yh, vfloat32m1_t yl,
    vfloat32m1_t& ph, vfloat32m1_t& pl, size_t vl)
{
    vfloat32m1_t sqryh = __riscv_vfmul_vv_f32m1(yh, yh, vl);
    vfloat32m1_t r = calc_polynom_deg_2_f32m1(yh, EXP_POL_COEFF_2_F32, EXP_POL_COEFF_3_F32, EXP_POL_COEFF_4_F32, vl);
    fma12_vv_f32m1(sqryh, r, __riscv_vfmul_vf_f32m1(yh, FACTOR_F32, vl), ph, pl, vl);
    pl = __riscv_vfadd_vv_f32m1(pl, __riscv_vfmul_vf_f32m1(yl, FACTOR_F32, vl), vl);
}

forceinline void calculate_exp2_polynom_hl12_f32m1(vfloat32m1_t yh,
    vfloat32m1_t& ph, vfloat32m1_t& pl, size_t vl)
{
    vfloat32m1_t yhs = __riscv_vfadd_vf_f32m1(yh, SHIFT_AVOID_UNDERFFLOW_IN_SQR_F32, vl);
    vfloat32m1_t sqryh = __riscv_vfmul_vv_f32m1(yhs, yhs, vl);
    vfloat32m1_t r = calc_polynom_deg_2_f32m1(yh, EXP2_POL_COEFF_2_F32, EXP2_POL_COEFF_3_F32, EXP2_POL_COEFF_4_F32, vl);
    fma12_ver2p1_vf_f32m1(yh, EXP2_POL_COEFF_1_F32, __riscv_vfmul_vv_f32m1(sqryh, r, vl), ph, pl, vl);
}

forceinline void reconstruct_exp_hl_hl_f32m1(vfloat32m1_t th, vfloat32m1_t tl,
    vfloat32m1_t pm1h, vfloat32m1_t pm1l, vuint32m1_t ei, vfloat32m1_t& res, size_t vl)
{
    vfloat32m1_t sh, sl, sl1;
    fast_2_sum_fv_f32m1(FACTOR_F32, pm1h, sh, sl1, vl);
    sl = __riscv_vfadd_vv_f32m1(sl1, pm1l, vl);
    mul21_vv_f32m1(th, tl, sh, sl, res, vl);

    vuint32m1_t sign_ei = __riscv_vsrl_vx_u32m1(ei, (size_t)8, vl);
    vuint32m1_t shift = __riscv_vmul_vx_u32m1(sign_ei, MAGIC_CONST_2_F32, vl);

    vuint32m1_t pow2ei = __riscv_vadd_vv_u32m1(__riscv_vrsub_vx_u32m1(shift, FACTOR_DEG_F32, vl), ei, vl);
    pow2ei = __riscv_vsll_vx_u32m1(pow2ei, (size_t)23, vl);
    vuint32m1_t factori = __riscv_vadd_vx_u32m1(shift, (uint32_t)127, vl);
    factori = __riscv_vsll_vx_u32m1(factori, (size_t)23, vl);

    res = __riscv_vfmul_vv_f32m1(res, __riscv_vreinterpret_v_u32m1_f32m1(pow2ei), vl);
    res = __riscv_vfmul_vv_f32m1(res, __riscv_vreinterpret_v_u32m1_f32m1(factori), vl);

    res = abs_f32m1(res, vl);
}

forceinline void reconstruct_expm1_f32m1(vfloat32m1_t th, vfloat32m1_t tl,
    vfloat32m1_t pm1h, vfloat32m1_t pm1l, vuint32m1_t ei, vfloat32m1_t& res, size_t vl)
{
    vfloat32m1_t rh, rl, rll;
    fast_2_sum_fv_f32m1(FACTOR_F32, pm1h, rh, rl, vl);
    fast_2_sum_vv_f32m1(rl, pm1l, rl, rll, vl);
    mul22_vv_f32m1(th, tl, rh, rl, rh, rl, vl);

    vuint32m1_t pow2ei = __riscv_vsll_vx_u32m1(__riscv_vadd_vx_u32m1(ei, FACTOR_DEG_F32, vl), (size_t)23, vl);
    rh = __riscv_vfmul_vv_f32m1(rh, __riscv_vreinterpret_v_u32m1_f32m1(pow2ei), vl);
    rl = __riscv_vfmul_vv_f32m1(rl, __riscv_vreinterpret_v_u32m1_f32m1(pow2ei), vl);
    rll = __riscv_vfmul_vv_f32m1(rll, __riscv_vreinterpret_v_u32m1_f32m1(pow2ei), vl);

    vfloat32m1_t sh1, sl1, maxs, mins;
    vbool32_t sortMask = __riscv_vmsgtu_vx_u32m1_b32(__riscv_vand_vx_u32m1(
        __riscv_vreinterpret_v_f32m1_u32m1(rh), (uint32_t)0x7f800000, vl), (uint32_t)0x3f800000, vl);
    maxs = __riscv_vfmerge_vfm_f32m1(rh, EXPM1_UNDERFLOW_VALUE_F32, __riscv_vmnot_m_b32(sortMask, vl), vl);
    mins = __riscv_vfmerge_vfm_f32m1(rh, EXPM1_UNDERFLOW_VALUE_F32, sortMask, vl);
    fast_2_sum_vv_f32m1(maxs, mins, sh1, sl1, vl);

    vfloat32m1_t sh2, sl2, resl;
    sl2 = __riscv_vfadd_vv_f32m1(sl1, rll, vl);
    fast_2_sum_vv_f32m1(rl, sl2, sh2, sl2, vl);
    fast_2_sum_vv_f32m1(sh1, sh2, res, resl, vl);
    res = __riscv_vfadd_vv_f32m1(res, __riscv_vfadd_vv_f32m1(resl, sl2, vl), vl);
}

forceinline void process_linear_f32m1(vfloat32m1_t x, vfloat32m1_t& xc, vbool32_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f32m1_b32(abs_f32m1(x, vl), EXPM1_LINEAR_THRESHOLD_F32, vl);
    xc = __riscv_vfmerge_vfm_f32m1(x, ZERO_F32, linearMask, vl);
}

forceinline void set_fe_underflow_f32m1(vfloat32m1_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    res = __riscv_vfmadd_vf_f32m1(res, (float)0x1p-26f, res, vl);
#endif
}

forceinline void exp2_set_fe_underflow_f32m1(vfloat32m1_t x, vfloat32m1_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    vuint32m1_t mask = __riscv_vsrl_vx_u32m1(__riscv_vand_vx_u32m1(
        __riscv_vreinterpret_v_f32m1_u32m1(x), (uint32_t)0x7f800000, vl), (size_t)23, vl);
    mask = __riscv_vsub_vx_u32m1(__riscv_vsll_vv_u32m1(__riscv_vmv_v_x_u32m1((uint32_t)1, vl),
        __riscv_vrsub_vx_u32m1(mask, (uint32_t)150, vl), vl), (uint32_t)1, vl);
    vbool32_t intMask = __riscv_vmseq_vx_u32m1_b32(
        __riscv_vand_vv_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(x), mask, vl), (uint32_t)0, vl);

    vfloat32m1_t coeff = __riscv_vfmerge_vfm_f32m1(
        __riscv_vfmv_v_f_f32m1((float)0x1p-26f, vl), ZERO_F32, intMask, vl);
    res = __riscv_vfmadd_vv_f32m1(res, coeff, res, vl);
#endif
}

// ---------------------------- m2 ----------------------------

forceinline void check_special_cases_f32m2(vfloat32m2_t& x, vfloat32m2_t& special, vbool16_t& specialMask,
    float overflowThreshold, float underflowThreshold, float underflowValue, bool throwUnderflow, size_t vl)
{
    vuint32m2_t uxi = __riscv_vreinterpret_v_f32m2_u32m2(x);

    vbool16_t underflowMask = __riscv_vmsgtu_vx_u32m2_b16(uxi, RVVMF_EXP_AS_UINT32(underflowThreshold), vl);
    special = __riscv_vfmerge_vfm_f32m2(x, underflowValue, underflowMask, vl);
    specialMask = underflowMask;

#ifndef __FAST_MATH__
    uint32_t pinf = 0x7f800000;
    uint32_t qnan = 0x7fc00000;
    vuint32m2_t absxui = __riscv_vand_vx_u32m2(uxi, (uint32_t)0x7fffffff, vl);
    vint32m2_t xi = __riscv_vreinterpret_v_f32m2_i32m2(x);

    vbool16_t nanMask = __riscv_vmsgtu_vx_u32m2_b16(absxui, pinf, vl);
    vbool16_t snanMask = __riscv_vmsltu_vx_u32m2_b16(absxui, qnan, vl);
    vbool16_t regularMask = __riscv_vmsltu_vx_u32m2_b16(absxui, pinf, vl);
    vbool16_t overflowMask = __riscv_vmsgt_vx_i32m2_b16(xi, RVVMF_EXP_AS_INT32(overflowThreshold), vl);

    special = __riscv_vfmerge_vfm_f32m2(special, RVVMF_EXP_AS_FP32(pinf), overflowMask, vl);
    special = __riscv_vfmerge_vfm_f32m2(special, RVVMF_EXP_AS_FP32(qnan), nanMask, vl);

    specialMask = __riscv_vmor_mm_b16(specialMask, __riscv_vmor_mm_b16(overflowMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b16(__riscv_vmand_mm_b16(nanMask, snanMask, vl), vl))
        RVVMF_EXP_CALL_FE_INVALID();
    if (__riscv_vcpop_m_b16(__riscv_vmand_mm_b16(overflowMask, regularMask, vl), vl))
        RVVMF_EXP_CALL_FE_OVERFLOW();
    if (throwUnderflow && __riscv_vcpop_m_b16(__riscv_vmand_mm_b16(underflowMask, regularMask, vl), vl))
        RVVMF_EXP_CALL_FE_UNDERFLOW();
#endif

    x = __riscv_vfmerge_vfm_f32m2(x, ZERO_F32, specialMask, vl);
}

forceinline void do_exp_argument_reduction_hl_f32m2(vfloat32m2_t x,
    vfloat32m2_t& yh, vfloat32m2_t& yl, vuint32m2_t& ei, vuint32m2_t& fi, size_t vl)
{
    vfloat32m2_t vmagicConst1 = __riscv_vfmv_v_f_f32m2(MAGIC_CONST_1_F32, vl);
    vfloat32m2_t h = __riscv_vfmadd_vf_f32m2(x, INV_LOG2_2K_F32, vmagicConst1, vl);
    vuint32m2_t hi = __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(h), MASK_HI_BIT_F32, vl);
    fi = __riscv_vand_vx_u32m2(hi, MASK_FI_BIT_F32, vl);
    ei = __riscv_vsrl_vx_u32m2(hi, TABLE_SIZE_DEG_F32, vl);
    h = __riscv_vfsub_vv_f32m2(h, vmagicConst1, vl);
    fma12_vf_f32m2(h, M_LOG2_2K_L_F32, __riscv_vfmadd_vf_f32m2(h, M_LOG2_2K_H_F32, x, vl), yh, yl, vl);
    yl = __riscv_vfmacc_vf_f32m2(yl, M_LOG2_2K_LL_F32, h, vl);
    fast_2_sum_vv_f32m2(yh, yl, yh, yl, vl);
}

forceinline void do_exp2_argument_reduction_f32m2(vfloat32m2_t x, vfloat32m2_t& y,
    vuint32m2_t& ei, vuint32m2_t& fi, size_t vl)
{
    vfloat32m2_t vmagicConst1 = __riscv_vfmv_v_f_f32m2(MAGIC_CONST_1_F32, vl);
    vfloat32m2_t h = __riscv_vfmadd_vf_f32m2(x, EXP2_TABLE_SIZE_DEG_F32, vmagicConst1, vl);
    vuint32m2_t hi = __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(h), MASK_HI_BIT_F32, vl);
    fi = __riscv_vand_vx_u32m2(hi, MASK_FI_BIT_F32, vl);
    ei = __riscv_vsrl_vx_u32m2(hi, TABLE_SIZE_DEG_F32, vl);
    h = __riscv_vfsub_vv_f32m2(h, vmagicConst1, vl);
    y = __riscv_vfmadd_vf_f32m2(h, M_EXP2_M_TABLE_SIZE_DEG_F32, x, vl);
}

forceinline void get_table_values_hl_f32m2(
    vuint32m2_t& index, vfloat32m2_t& th, vfloat32m2_t& tl, size_t vl)
{
    index = __riscv_vmul_vx_u32m2(index, uint32_t(sizeof(float)), vl);
    th = __riscv_vloxei32_v_f32m2(LOOK_UP_TABLE_HIGH_F32, index, vl);
    tl = __riscv_vloxei32_v_f32m2(LOOK_UP_TABLE_LOW_F32, index, vl);
}

forceinline void calculate_exp_polynom_hl_f32m2(vfloat32m2_t yh, vfloat32m2_t yl,
    vfloat32m2_t& ph, vfloat32m2_t& pl, size_t vl)
{
    vfloat32m2_t yhs = __riscv_vfadd_vf_f32m2(yh, SHIFT_AVOID_UNDERFFLOW_IN_SQR_F32, vl);
    vfloat32m2_t sqryh = __riscv_vfmul_vv_f32m2(yhs, yhs, vl);
    vfloat32m2_t r = calc_polynom_deg_2_f32m2(yh, EXP_POL_COEFF_2_F32, EXP_POL_COEFF_3_F32, EXP_POL_COEFF_4_F32, vl);
    fma12_vv_f32m2(sqryh, r, __riscv_vfmul_vf_f32m2(yh, FACTOR_F32, vl), ph, pl, vl);
    pl = __riscv_vfadd_vv_f32m2(pl, __riscv_vfmul_vf_f32m2(yl, FACTOR_F32, vl), vl);
}

forceinline void calculate_expm1_polynom_hl_f32m2(vfloat32m2_t yh, vfloat32m2_t yl,
    vfloat32m2_t& ph, vfloat32m2_t& pl, size_t vl)
{
    vfloat32m2_t sqryh = __riscv_vfmul_vv_f32m2(yh, yh, vl);
    vfloat32m2_t r = calc_polynom_deg_2_f32m2(yh, EXP_POL_COEFF_2_F32, EXP_POL_COEFF_3_F32, EXP_POL_COEFF_4_F32, vl);
    fma12_vv_f32m2(sqryh, r, __riscv_vfmul_vf_f32m2(yh, FACTOR_F32, vl), ph, pl, vl);
    pl = __riscv_vfadd_vv_f32m2(pl, __riscv_vfmul_vf_f32m2(yl, FACTOR_F32, vl), vl);
}

forceinline void calculate_exp2_polynom_hl12_f32m2(vfloat32m2_t yh,
    vfloat32m2_t& ph, vfloat32m2_t& pl, size_t vl)
{
    vfloat32m2_t yhs = __riscv_vfadd_vf_f32m2(yh, SHIFT_AVOID_UNDERFFLOW_IN_SQR_F32, vl);
    vfloat32m2_t sqryh = __riscv_vfmul_vv_f32m2(yhs, yhs, vl);
    vfloat32m2_t r = calc_polynom_deg_2_f32m2(yh, EXP2_POL_COEFF_2_F32, EXP2_POL_COEFF_3_F32, EXP2_POL_COEFF_4_F32, vl);
    fma12_ver2p1_vf_f32m2(yh, EXP2_POL_COEFF_1_F32, __riscv_vfmul_vv_f32m2(sqryh, r, vl), ph, pl, vl);
}

forceinline void reconstruct_exp_hl_hl_f32m2(vfloat32m2_t th, vfloat32m2_t tl,
    vfloat32m2_t pm1h, vfloat32m2_t pm1l, vuint32m2_t ei, vfloat32m2_t& res, size_t vl)
{
    vfloat32m2_t sh, sl, sl1;
    fast_2_sum_fv_f32m2(FACTOR_F32, pm1h, sh, sl1, vl);
    sl = __riscv_vfadd_vv_f32m2(sl1, pm1l, vl);
    mul21_vv_f32m2(th, tl, sh, sl, res, vl);

    vuint32m2_t sign_ei = __riscv_vsrl_vx_u32m2(ei, (size_t)8, vl);
    vuint32m2_t shift = __riscv_vmul_vx_u32m2(sign_ei, MAGIC_CONST_2_F32, vl);

    vuint32m2_t pow2ei = __riscv_vadd_vv_u32m2(__riscv_vrsub_vx_u32m2(shift, FACTOR_DEG_F32, vl), ei, vl);
    pow2ei = __riscv_vsll_vx_u32m2(pow2ei, (size_t)23, vl);
    vuint32m2_t factori = __riscv_vadd_vx_u32m2(shift, (uint32_t)127, vl);
    factori = __riscv_vsll_vx_u32m2(factori, (size_t)23, vl);

    res = __riscv_vfmul_vv_f32m2(res, __riscv_vreinterpret_v_u32m2_f32m2(pow2ei), vl);
    res = __riscv_vfmul_vv_f32m2(res, __riscv_vreinterpret_v_u32m2_f32m2(factori), vl);

    res = abs_f32m2(res, vl);
}

forceinline void reconstruct_expm1_f32m2(vfloat32m2_t th, vfloat32m2_t tl,
    vfloat32m2_t pm1h, vfloat32m2_t pm1l, vuint32m2_t ei, vfloat32m2_t& res, size_t vl)
{
    vfloat32m2_t rh, rl, rll;
    fast_2_sum_fv_f32m2(FACTOR_F32, pm1h, rh, rl, vl);
    fast_2_sum_vv_f32m2(rl, pm1l, rl, rll, vl);
    mul22_vv_f32m2(th, tl, rh, rl, rh, rl, vl);

    vuint32m2_t pow2ei = __riscv_vsll_vx_u32m2(__riscv_vadd_vx_u32m2(ei, FACTOR_DEG_F32, vl), (size_t)23, vl);
    rh = __riscv_vfmul_vv_f32m2(rh, __riscv_vreinterpret_v_u32m2_f32m2(pow2ei), vl);
    rl = __riscv_vfmul_vv_f32m2(rl, __riscv_vreinterpret_v_u32m2_f32m2(pow2ei), vl);
    rll = __riscv_vfmul_vv_f32m2(rll, __riscv_vreinterpret_v_u32m2_f32m2(pow2ei), vl);

    vfloat32m2_t sh1, sl1, maxs, mins;
    vbool16_t sortMask = __riscv_vmsgtu_vx_u32m2_b16(__riscv_vand_vx_u32m2(
        __riscv_vreinterpret_v_f32m2_u32m2(rh), (uint32_t)0x7f800000, vl), (uint32_t)0x3f800000, vl);
    maxs = __riscv_vfmerge_vfm_f32m2(rh, EXPM1_UNDERFLOW_VALUE_F32, __riscv_vmnot_m_b16(sortMask, vl), vl);
    mins = __riscv_vfmerge_vfm_f32m2(rh, EXPM1_UNDERFLOW_VALUE_F32, sortMask, vl);
    fast_2_sum_vv_f32m2(maxs, mins, sh1, sl1, vl);

    vfloat32m2_t sh2, sl2, resl;
    sl2 = __riscv_vfadd_vv_f32m2(sl1, rll, vl);
    fast_2_sum_vv_f32m2(rl, sl2, sh2, sl2, vl);
    fast_2_sum_vv_f32m2(sh1, sh2, res, resl, vl);
    res = __riscv_vfadd_vv_f32m2(res, __riscv_vfadd_vv_f32m2(resl, sl2, vl), vl);
}

forceinline void process_linear_f32m2(vfloat32m2_t x, vfloat32m2_t& xc, vbool16_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f32m2_b16(abs_f32m2(x, vl), EXPM1_LINEAR_THRESHOLD_F32, vl);
    xc = __riscv_vfmerge_vfm_f32m2(x, ZERO_F32, linearMask, vl);
}

forceinline void set_fe_underflow_f32m2(vfloat32m2_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    res = __riscv_vfmadd_vf_f32m2(res, (float)0x1p-26f, res, vl);
#endif
}

forceinline void exp2_set_fe_underflow_f32m2(vfloat32m2_t x, vfloat32m2_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    vuint32m2_t mask = __riscv_vsrl_vx_u32m2(__riscv_vand_vx_u32m2(
        __riscv_vreinterpret_v_f32m2_u32m2(x), (uint32_t)0x7f800000, vl), (size_t)23, vl);
    mask = __riscv_vsub_vx_u32m2(__riscv_vsll_vv_u32m2(__riscv_vmv_v_x_u32m2((uint32_t)1, vl),
        __riscv_vrsub_vx_u32m2(mask, (uint32_t)150, vl), vl), (uint32_t)1, vl);
    vbool16_t intMask = __riscv_vmseq_vx_u32m2_b16(
        __riscv_vand_vv_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(x), mask, vl), (uint32_t)0, vl);

    vfloat32m2_t coeff = __riscv_vfmerge_vfm_f32m2(
        __riscv_vfmv_v_f_f32m2((float)0x1p-26f, vl), ZERO_F32, intMask, vl);
    res = __riscv_vfmadd_vv_f32m2(res, coeff, res, vl);
#endif
}

// ---------------------------- m4 ----------------------------

forceinline void check_special_cases_f32m4(vfloat32m4_t& x, vfloat32m4_t& special, vbool8_t& specialMask,
    float overflowThreshold, float underflowThreshold, float underflowValue, bool throwUnderflow, size_t vl)
{
    vuint32m4_t uxi = __riscv_vreinterpret_v_f32m4_u32m4(x);

    vbool8_t underflowMask = __riscv_vmsgtu_vx_u32m4_b8(uxi, RVVMF_EXP_AS_UINT32(underflowThreshold), vl);
    special = __riscv_vfmerge_vfm_f32m4(x, underflowValue, underflowMask, vl);
    specialMask = underflowMask;

#ifndef __FAST_MATH__
    uint32_t pinf = 0x7f800000;
    uint32_t qnan = 0x7fc00000;
    vuint32m4_t absxui = __riscv_vand_vx_u32m4(uxi, (uint32_t)0x7fffffff, vl);
    vint32m4_t xi = __riscv_vreinterpret_v_f32m4_i32m4(x);

    vbool8_t nanMask = __riscv_vmsgtu_vx_u32m4_b8(absxui, pinf, vl);
    vbool8_t snanMask = __riscv_vmsltu_vx_u32m4_b8(absxui, qnan, vl);
    vbool8_t regularMask = __riscv_vmsltu_vx_u32m4_b8(absxui, pinf, vl);
    vbool8_t overflowMask = __riscv_vmsgt_vx_i32m4_b8(xi, RVVMF_EXP_AS_INT32(overflowThreshold), vl);

    special = __riscv_vfmerge_vfm_f32m4(special, RVVMF_EXP_AS_FP32(pinf), overflowMask, vl);
    special = __riscv_vfmerge_vfm_f32m4(special, RVVMF_EXP_AS_FP32(qnan), nanMask, vl);

    specialMask = __riscv_vmor_mm_b8(specialMask, __riscv_vmor_mm_b8(overflowMask, nanMask, vl), vl);

    if (__riscv_vcpop_m_b8(__riscv_vmand_mm_b8(nanMask, snanMask, vl), vl))
        RVVMF_EXP_CALL_FE_INVALID();
    if (__riscv_vcpop_m_b8(__riscv_vmand_mm_b8(overflowMask, regularMask, vl), vl))
        RVVMF_EXP_CALL_FE_OVERFLOW();
    if (throwUnderflow && __riscv_vcpop_m_b8(__riscv_vmand_mm_b8(underflowMask, regularMask, vl), vl))
        RVVMF_EXP_CALL_FE_UNDERFLOW();
#endif

    x = __riscv_vfmerge_vfm_f32m4(x, ZERO_F32, specialMask, vl);
}

forceinline void do_exp_argument_reduction_hl_f32m4(vfloat32m4_t x,
    vfloat32m4_t& yh, vfloat32m4_t& yl, vuint32m4_t& ei, vuint32m4_t& fi, size_t vl)
{
    vfloat32m4_t vmagicConst1 = __riscv_vfmv_v_f_f32m4(MAGIC_CONST_1_F32, vl);
    vfloat32m4_t h = __riscv_vfmadd_vf_f32m4(x, INV_LOG2_2K_F32, vmagicConst1, vl);
    vuint32m4_t hi = __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(h), MASK_HI_BIT_F32, vl);
    fi = __riscv_vand_vx_u32m4(hi, MASK_FI_BIT_F32, vl);
    ei = __riscv_vsrl_vx_u32m4(hi, TABLE_SIZE_DEG_F32, vl);
    h = __riscv_vfsub_vv_f32m4(h, vmagicConst1, vl);
    fma12_vf_f32m4(h, M_LOG2_2K_L_F32, __riscv_vfmadd_vf_f32m4(h, M_LOG2_2K_H_F32, x, vl), yh, yl, vl);
    yl = __riscv_vfmacc_vf_f32m4(yl, M_LOG2_2K_LL_F32, h, vl);
    fast_2_sum_vv_f32m4(yh, yl, yh, yl, vl);
}

forceinline void do_exp2_argument_reduction_f32m4(vfloat32m4_t x, vfloat32m4_t& y,
    vuint32m4_t& ei, vuint32m4_t& fi, size_t vl)
{
    vfloat32m4_t vmagicConst1 = __riscv_vfmv_v_f_f32m4(MAGIC_CONST_1_F32, vl);
    vfloat32m4_t h = __riscv_vfmadd_vf_f32m4(x, EXP2_TABLE_SIZE_DEG_F32, vmagicConst1, vl);
    vuint32m4_t hi = __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(h), MASK_HI_BIT_F32, vl);
    fi = __riscv_vand_vx_u32m4(hi, MASK_FI_BIT_F32, vl);
    ei = __riscv_vsrl_vx_u32m4(hi, TABLE_SIZE_DEG_F32, vl);
    h = __riscv_vfsub_vv_f32m4(h, vmagicConst1, vl);
    y = __riscv_vfmadd_vf_f32m4(h, M_EXP2_M_TABLE_SIZE_DEG_F32, x, vl);
}

forceinline void get_table_values_hl_f32m4(
    vuint32m4_t& index, vfloat32m4_t& th, vfloat32m4_t& tl, size_t vl)
{
    index = __riscv_vmul_vx_u32m4(index, uint32_t(sizeof(float)), vl);
    th = __riscv_vloxei32_v_f32m4(LOOK_UP_TABLE_HIGH_F32, index, vl);
    tl = __riscv_vloxei32_v_f32m4(LOOK_UP_TABLE_LOW_F32, index, vl);
}

forceinline void calculate_exp_polynom_hl_f32m4(vfloat32m4_t yh, vfloat32m4_t yl,
    vfloat32m4_t& ph, vfloat32m4_t& pl, size_t vl)
{
    vfloat32m4_t yhs = __riscv_vfadd_vf_f32m4(yh, SHIFT_AVOID_UNDERFFLOW_IN_SQR_F32, vl);
    vfloat32m4_t sqryh = __riscv_vfmul_vv_f32m4(yhs, yhs, vl);
    vfloat32m4_t r = calc_polynom_deg_2_f32m4(yh, EXP_POL_COEFF_2_F32, EXP_POL_COEFF_3_F32, EXP_POL_COEFF_4_F32, vl);
    fma12_vv_f32m4(sqryh, r, __riscv_vfmul_vf_f32m4(yh, FACTOR_F32, vl), ph, pl, vl);
    pl = __riscv_vfadd_vv_f32m4(pl, __riscv_vfmul_vf_f32m4(yl, FACTOR_F32, vl), vl);
}

forceinline void calculate_expm1_polynom_hl_f32m4(vfloat32m4_t yh, vfloat32m4_t yl,
    vfloat32m4_t& ph, vfloat32m4_t& pl, size_t vl)
{
    vfloat32m4_t sqryh = __riscv_vfmul_vv_f32m4(yh, yh, vl);
    vfloat32m4_t r = calc_polynom_deg_2_f32m4(yh, EXP_POL_COEFF_2_F32, EXP_POL_COEFF_3_F32, EXP_POL_COEFF_4_F32, vl);
    fma12_vv_f32m4(sqryh, r, __riscv_vfmul_vf_f32m4(yh, FACTOR_F32, vl), ph, pl, vl);
    pl = __riscv_vfadd_vv_f32m4(pl, __riscv_vfmul_vf_f32m4(yl, FACTOR_F32, vl), vl);
}

forceinline void calculate_exp2_polynom_hl12_f32m4(vfloat32m4_t yh,
    vfloat32m4_t& ph, vfloat32m4_t& pl, size_t vl)
{
    vfloat32m4_t yhs = __riscv_vfadd_vf_f32m4(yh, SHIFT_AVOID_UNDERFFLOW_IN_SQR_F32, vl);
    vfloat32m4_t sqryh = __riscv_vfmul_vv_f32m4(yhs, yhs, vl);
    vfloat32m4_t r = calc_polynom_deg_2_f32m4(yh, EXP2_POL_COEFF_2_F32, EXP2_POL_COEFF_3_F32, EXP2_POL_COEFF_4_F32, vl);
    fma12_ver2p1_vf_f32m4(yh, EXP2_POL_COEFF_1_F32, __riscv_vfmul_vv_f32m4(sqryh, r, vl), ph, pl, vl);
}

forceinline void reconstruct_exp_hl_hl_f32m4(vfloat32m4_t th, vfloat32m4_t tl,
    vfloat32m4_t pm1h, vfloat32m4_t pm1l, vuint32m4_t ei, vfloat32m4_t& res, size_t vl)
{
    vfloat32m4_t sh, sl, sl1;
    fast_2_sum_fv_f32m4(FACTOR_F32, pm1h, sh, sl1, vl);
    sl = __riscv_vfadd_vv_f32m4(sl1, pm1l, vl);
    mul21_vv_f32m4(th, tl, sh, sl, res, vl);

    vuint32m4_t sign_ei = __riscv_vsrl_vx_u32m4(ei, (size_t)8, vl);
    vuint32m4_t shift = __riscv_vmul_vx_u32m4(sign_ei, MAGIC_CONST_2_F32, vl);

    vuint32m4_t pow2ei = __riscv_vadd_vv_u32m4(__riscv_vrsub_vx_u32m4(shift, FACTOR_DEG_F32, vl), ei, vl);
    pow2ei = __riscv_vsll_vx_u32m4(pow2ei, (size_t)23, vl);
    vuint32m4_t factori = __riscv_vadd_vx_u32m4(shift, (uint32_t)127, vl);
    factori = __riscv_vsll_vx_u32m4(factori, (size_t)23, vl);

    res = __riscv_vfmul_vv_f32m4(res, __riscv_vreinterpret_v_u32m4_f32m4(pow2ei), vl);
    res = __riscv_vfmul_vv_f32m4(res, __riscv_vreinterpret_v_u32m4_f32m4(factori), vl);

    res = abs_f32m4(res, vl);
}

forceinline void reconstruct_expm1_f32m4(vfloat32m4_t th, vfloat32m4_t tl,
    vfloat32m4_t pm1h, vfloat32m4_t pm1l, vuint32m4_t ei, vfloat32m4_t& res, size_t vl)
{
    vfloat32m4_t rh, rl, rll;
    fast_2_sum_fv_f32m4(FACTOR_F32, pm1h, rh, rl, vl);
    fast_2_sum_vv_f32m4(rl, pm1l, rl, rll, vl);
    mul22_vv_f32m4(th, tl, rh, rl, rh, rl, vl);

    vuint32m4_t pow2ei = __riscv_vsll_vx_u32m4(__riscv_vadd_vx_u32m4(ei, FACTOR_DEG_F32, vl), (size_t)23, vl);
    rh = __riscv_vfmul_vv_f32m4(rh, __riscv_vreinterpret_v_u32m4_f32m4(pow2ei), vl);
    rl = __riscv_vfmul_vv_f32m4(rl, __riscv_vreinterpret_v_u32m4_f32m4(pow2ei), vl);
    rll = __riscv_vfmul_vv_f32m4(rll, __riscv_vreinterpret_v_u32m4_f32m4(pow2ei), vl);

    vfloat32m4_t sh1, sl1, maxs, mins;
    vbool8_t sortMask = __riscv_vmsgtu_vx_u32m4_b8(__riscv_vand_vx_u32m4(
        __riscv_vreinterpret_v_f32m4_u32m4(rh), (uint32_t)0x7f800000, vl), (uint32_t)0x3f800000, vl);
    maxs = __riscv_vfmerge_vfm_f32m4(rh, EXPM1_UNDERFLOW_VALUE_F32, __riscv_vmnot_m_b8(sortMask, vl), vl);
    mins = __riscv_vfmerge_vfm_f32m4(rh, EXPM1_UNDERFLOW_VALUE_F32, sortMask, vl);
    fast_2_sum_vv_f32m4(maxs, mins, sh1, sl1, vl);

    vfloat32m4_t sh2, sl2, resl;
    sl2 = __riscv_vfadd_vv_f32m4(sl1, rll, vl);
    fast_2_sum_vv_f32m4(rl, sl2, sh2, sl2, vl);
    fast_2_sum_vv_f32m4(sh1, sh2, res, resl, vl);
    res = __riscv_vfadd_vv_f32m4(res, __riscv_vfadd_vv_f32m4(resl, sl2, vl), vl);
}

forceinline void process_linear_f32m4(vfloat32m4_t x, vfloat32m4_t& xc, vbool8_t& linearMask, size_t vl)
{
    linearMask = __riscv_vmflt_vf_f32m4_b8(abs_f32m4(x, vl), EXPM1_LINEAR_THRESHOLD_F32, vl);
    xc = __riscv_vfmerge_vfm_f32m4(x, ZERO_F32, linearMask, vl);
}

forceinline void set_fe_underflow_f32m4(vfloat32m4_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    res = __riscv_vfmadd_vf_f32m4(res, (float)0x1p-26f, res, vl);
#endif
}

forceinline void exp2_set_fe_underflow_f32m4(vfloat32m4_t x, vfloat32m4_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    vuint32m4_t mask = __riscv_vsrl_vx_u32m4(__riscv_vand_vx_u32m4(
        __riscv_vreinterpret_v_f32m4_u32m4(x), (uint32_t)0x7f800000, vl), (size_t)23, vl);
    mask = __riscv_vsub_vx_u32m4(__riscv_vsll_vv_u32m4(__riscv_vmv_v_x_u32m4((uint32_t)1, vl),
        __riscv_vrsub_vx_u32m4(mask, (uint32_t)150, vl), vl), (uint32_t)1, vl);
    vbool8_t intMask = __riscv_vmseq_vx_u32m4_b8(
        __riscv_vand_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(x), mask, vl), (uint32_t)0, vl);

    vfloat32m4_t coeff = __riscv_vfmerge_vfm_f32m4(
        __riscv_vfmv_v_f_f32m4((float)0x1p-26f, vl), ZERO_F32, intMask, vl);
    res = __riscv_vfmadd_vv_f32m4(res, coeff, res, vl);
#endif
}
