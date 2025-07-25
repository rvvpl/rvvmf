/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025 The RVVMF Authors (Elena Panova)
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

const float ZERO_F32 = 0.0f;
const float ONE_F32 = 1.0f;

const float EXP_EXPM1_OVERFLOW_THRESHOLD_F32 = 0x1.62e42ep6f;
const float EXP2_EXP2M1_OVERFLOW_THRESHOLD_F32 = 0x1.fffffep6f;
const float EXP_SUBNORMAL_THRESHOLD_F32 = -0x1.5d589ep6f;
const float EXP2_SUBNORMAL_THRESHOLD_F32 = -0x1.f8p6f;
const float EXP_ZERO_THRESHOLD_F32 = -0x1.9fe368p6f;
const float EXP2_ZERO_THRESHOLD_F32 = -0x1.2bfffep7f;
const float EXP_UNDERFLOW_VALUE_F32 = 0.0f;
const float EXPM1_UNDERFLOW_THRESHOLD_F32 = -0x1.154244p4f;
const float EXPM1_LINEAR_THRESHOLD_F32 = 0x1.6a09e8p-24f;
const float EXPM1_UNDERFLOW_VALUE_F32 = -1.0f;

const size_t TABLE_SIZE_DEG_F32 = 4;
const float EXP2_TABLE_SIZE_DEG_F32 = 0x1p4f;
const float M_EXP2_M_TABLE_SIZE_DEG_F32 = -0x1p-4f;
const uint32_t MASK_FI_BIT_F32 = 0x0000000f;
const uint32_t MASK_HI_BIT_F32 = 0x00001fff;
const float MAGIC_CONST_1_F32 = 12582912.0f; 
const float INV_LOG2_2K_F32 = 0x1.715476p4f;
const float M_LOG2_2K_H_F32 = -0x1.62ep-5f;
const float M_LOG2_2K_L_F32 = -0x1.0bfbe8p-19f;
const float M_LOG2_2K_LL_F32 = -0x1.cf79acp-44f;

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

const float EXP_POL_COEFF_2_F32 = 0x1p-1f;
const float EXP_POL_COEFF_3_F32 = 0x1.5556dep-3f;
const float EXP_POL_COEFF_4_F32 = 0x1.555696p-5f;

const float EXP2_POL_COEFF_1_F32 = 0x1.62e43p-1f;
const float EXP2_POL_COEFF_2_F32 = 0x1.ebfbep-3f;
const float EXP2_POL_COEFF_3_F32 = 0x1.c6ae08p-5f;
const float EXP2_POL_COEFF_4_F32 = 0x1.3b27cep-7f;

// ---------------------------- m1 ----------------------------

forceinline void check_special_cases_f32m1(vfloat32m1_t& x, vfloat32m1_t& special, vbool32_t& specialMask,
    const float& overflowThreshold, size_t vl)
{ 
    // check +inf
    uint32_t pinf = 0x7f800000;
    specialMask = __riscv_vmfeq_vf_f32m1_b32(x, RVVMF_EXP_AS_FP32(pinf), vl);
    special = __riscv_vfmerge_vfm_f32m1(x, RVVMF_EXP_AS_FP32(pinf), specialMask, vl);
    // check overflow
    vbool32_t mask = __riscv_vmand_mm_b32(__riscv_vmfgt_vf_f32m1_b32(x, overflowThreshold, vl),
        __riscv_vmflt_vf_f32m1_b32(x, RVVMF_EXP_AS_FP32(pinf), vl), vl);
    special = __riscv_vfmerge_vfm_f32m1(special, RVVMF_EXP_AS_FP32(pinf), mask, vl);
    specialMask = __riscv_vmor_mm_b32(specialMask, mask, vl);  
    if (__riscv_vcpop_m_b32(mask, vl)) RVVMF_EXP_CALL_FE_OVERFLOW();
    // NaNs, overflow, -inf -- automatically
    x = __riscv_vfmerge_vfm_f32m1(x, ZERO_F32, specialMask, vl);
}

forceinline void do_exp_argument_reduction_hl_f32m1(const vfloat32m1_t& x,
    vfloat32m1_t& yh, vfloat32m1_t& yl, vuint32m1_t& ei, vuint32m1_t& fi, size_t vl)
{
    vfloat32m1_t vmagicConst1 = __riscv_vfmv_v_f_f32m1(MAGIC_CONST_1_F32, vl);
    vfloat32m1_t h = __riscv_vfmadd_vf_f32m1(x, INV_LOG2_2K_F32, vmagicConst1, vl);
    vuint32m1_t hi = __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_f32m1_u32m1(h), MASK_HI_BIT_F32, vl);
    fi = __riscv_vand_vx_u32m1(hi, MASK_FI_BIT_F32, vl);
    ei = __riscv_vsrl_vx_u32m1(hi, TABLE_SIZE_DEG_F32, vl);
    h = __riscv_vfsub_vv_f32m1(h, vmagicConst1, vl);
    fma12_vf_f32m1(h, M_LOG2_2K_L_F32, __riscv_vfmadd_vf_f32m1(h, M_LOG2_2K_H_F32, x, vl), yh, yl, vl);
    yl = __riscv_vfmadd_vf_f32m1(h, M_LOG2_2K_LL_F32, yl, vl);
    fast_2_sum_vv_f32m1(yh, yl, yh, yl, vl);
}

forceinline void do_exp2_argument_reduction_f32m1(const vfloat32m1_t& x, vfloat32m1_t& y,
    vuint32m1_t& ei, vuint32m1_t& fi, size_t vl)  // exact
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

forceinline void calculate_exp_polynom_hl_f32m1(const vfloat32m1_t& yh, const vfloat32m1_t& yl, vfloat32m1_t& ph, vfloat32m1_t& pl, size_t vl)
{
    vfloat32m1_t sqryh = __riscv_vfmul_vv_f32m1(yh, yh, vl);
    vfloat32m1_t r = calc_polynom_deg_2_f32m1(yh, EXP_POL_COEFF_2_F32, EXP_POL_COEFF_3_F32, EXP_POL_COEFF_4_F32, vl); 
    fma12_vv_f32m1(sqryh, r, yh, ph, pl, vl);
    pl = __riscv_vfadd_vv_f32m1(pl, yl, vl);
}

forceinline void calculate_exp2_polynom_hl12_f32m1(const vfloat32m1_t& yh, vfloat32m1_t& ph, vfloat32m1_t& pl, size_t vl)
{
    vfloat32m1_t sqryh = __riscv_vfmul_vv_f32m1(yh, yh, vl);
    vfloat32m1_t r = calc_polynom_deg_2_f32m1(yh, EXP2_POL_COEFF_2_F32, EXP2_POL_COEFF_3_F32, EXP2_POL_COEFF_4_F32, vl); 
    fma12_ver2p1_vf_f32m1(yh, EXP2_POL_COEFF_1_F32, __riscv_vfmul_vv_f32m1(sqryh, r, vl), ph, pl, vl);
}

forceinline void update_exponent_f32m1(const vuint32m1_t& ei, vfloat32m1_t& res, size_t vl)
{
    res = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vadd_vv_u32m1(
        __riscv_vreinterpret_v_f32m1_u32m1(res), __riscv_vsll_vx_u32m1(ei, (size_t)23, vl), vl));
}

forceinline void update_exponent_with_subnormal_f32m1(const float& subnormalThreshold, const vfloat32m1_t& x,
    const vuint32m1_t& ei, vfloat32m1_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    uint32_t ninf = 0xff800000;
    vbool32_t subnormalMask = __riscv_vmand_mm_b32(__riscv_vmfgt_vf_f32m1_b32(x, RVVMF_EXP_AS_FP32(ninf), vl),
        __riscv_vmflt_vf_f32m1_b32(x, subnormalThreshold, vl), vl);
    if (__riscv_vcpop_m_b32(subnormalMask, vl)) RVVMF_EXP_CALL_FE_UNDERFLOW();  // FE_UNDERFLOW
    
    vuint32m1_t shiftNum = __riscv_vreinterpret_v_i32m1_u32m1(__riscv_vneg_v_i32m1(__riscv_vreinterpret_v_u32m1_i32m1(ei), vl));
    shiftNum = __riscv_vadd_vx_u32m1(__riscv_vand_vx_u32m1(shiftNum, (uint32_t)0x000001ff, vl), (uint32_t)1, vl);
    shiftNum = __riscv_vsll_vx_u32m1(shiftNum, (size_t)23, vl);
    vfloat32m1_t subnormalRes = __riscv_vfadd_vv_f32m1(res, __riscv_vreinterpret_v_u32m1_f32m1(shiftNum), vl);
    subnormalRes = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vx_u32m1(
        __riscv_vreinterpret_v_f32m1_u32m1(subnormalRes), (uint32_t)0x807fffff, vl));
#endif

    update_exponent_f32m1(ei, res, vl);
    
#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m1(res, subnormalRes, subnormalMask, vl);
#endif   
}

forceinline void reconstruct_exp_hl_hl_f32m1(const vfloat32m1_t& x, const vuint32m1_t& ei, const vfloat32m1_t& th, const vfloat32m1_t& tl,
    const vfloat32m1_t& pm1h, const vfloat32m1_t& pm1l, vfloat32m1_t& res, const float& subnormalThreshold, size_t vl)
{
    vfloat32m1_t sh, sl;
    fast_2_sum_fv_f32m1(ONE_F32, pm1h, sh, sl, vl);
    sl = __riscv_vfadd_vv_f32m1(sl, pm1l, vl);
    mul21_vv_f32m1(th, tl, sh, sl, res, vl);
    update_exponent_with_subnormal_f32m1(subnormalThreshold, x, ei, res, vl);
}

forceinline void reconstruct_expm1_f32m1(const vfloat32m1_t& th, const vfloat32m1_t& tl, 
    const vfloat32m1_t& pm1h, const vfloat32m1_t& pm1l, const vuint32m1_t& ei, vfloat32m1_t& res, size_t vl)
{        
    vfloat32m1_t rh, rl, sh, sl;
    fast_2_sum_fv_f32m1(ONE_F32, pm1h, rh, rl, vl);
    rl = __riscv_vfadd_vv_f32m1(rl, pm1l, vl);
    mul22_vv_f32m1(th, tl, rh, rl, sh, sl, vl);
    
    vuint32m1_t power = __riscv_vsll_vx_u32m1(ei, (size_t)23, vl);
    sh = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vadd_vv_u32m1(
        __riscv_vreinterpret_v_f32m1_u32m1(sh), power, vl));   
    vbool32_t slZeroMask = __riscv_vmfeq_vf_f32m1_b32(sl, ZERO_F32, vl);
    sl = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vadd_vv_u32m1(
        __riscv_vreinterpret_v_f32m1_u32m1(sl), power, vl));
    sl = __riscv_vfmerge_vfm_f32m1(sl, ZERO_F32, slZeroMask, vl);
    
    vbool32_t sortMask = __riscv_vmsgtu_vx_u32m1_b32(__riscv_vand_vx_u32m1(
        __riscv_vreinterpret_v_f32m1_u32m1(sh), (uint32_t)0x7f800000, vl), (uint32_t)0x3f800000, vl);
    vfloat32m1_t maxs = __riscv_vfmerge_vfm_f32m1(sh, EXPM1_UNDERFLOW_VALUE_F32, __riscv_vmnot_m_b32(sortMask, vl), vl);   
    vfloat32m1_t mins = __riscv_vfmerge_vfm_f32m1(sh, EXPM1_UNDERFLOW_VALUE_F32, sortMask, vl);
    fast_2_sum_vv_f32m1(maxs, mins, rh, rl, vl);
    
    res = __riscv_vfadd_vv_f32m1(rh, __riscv_vfadd_vv_f32m1(sl, rl, vl), vl);
}

forceinline void update_underflow_f32m1(const vfloat32m1_t& x, vfloat32m1_t& res,
    const float& underflowThreshold, const float& underflowValue, size_t vl)
{
    vbool32_t underflowMask = __riscv_vmflt_vf_f32m1_b32(x, underflowThreshold, vl);
    res = __riscv_vfmerge_vfm_f32m1(res, underflowValue, underflowMask, vl);
}

forceinline void set_sign_f32m1(const vfloat32m1_t& x, vfloat32m1_t& res, size_t vl)
{
    uint32_t signMask = 0x7fffffff;
    res = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vx_u32m1(
        __riscv_vreinterpret_v_f32m1_u32m1(res), signMask, vl));
    res = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vor_vv_u32m1(__riscv_vand_vx_u32m1(
        __riscv_vreinterpret_v_f32m1_u32m1(x), ~signMask, vl), __riscv_vreinterpret_v_f32m1_u32m1(res), vl));
}

forceinline void process_linear_f32m1(const vfloat32m1_t& x, vfloat32m1_t& res, size_t vl)
{
    uint32_t signMask = 0x7fffffff;
    vfloat32m1_t xabs = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vx_u32m1(
        __riscv_vreinterpret_v_f32m1_u32m1(x), signMask, vl));
    vbool32_t linearMask = __riscv_vmflt_vf_f32m1_b32(xabs, EXPM1_LINEAR_THRESHOLD_F32, vl);
    res = __riscv_vmerge_vvm_f32m1(res, x, linearMask, vl);
}

// ---------------------------- m2 ----------------------------

forceinline void check_special_cases_f32m2(vfloat32m2_t& x, vfloat32m2_t& special, vbool16_t& specialMask,
    const float& overflowThreshold, size_t vl)
{ 
    // check +inf
    uint32_t pinf = 0x7f800000;
    specialMask = __riscv_vmfeq_vf_f32m2_b16(x, RVVMF_EXP_AS_FP32(pinf), vl);
    special = __riscv_vfmerge_vfm_f32m2(x, RVVMF_EXP_AS_FP32(pinf), specialMask, vl);
    // check overflow
    vbool16_t mask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f32m2_b16(x, overflowThreshold, vl),
        __riscv_vmflt_vf_f32m2_b16(x, RVVMF_EXP_AS_FP32(pinf), vl), vl);
    special = __riscv_vfmerge_vfm_f32m2(special, RVVMF_EXP_AS_FP32(pinf), mask, vl);
    specialMask = __riscv_vmor_mm_b16(specialMask, mask, vl);  
    if (__riscv_vcpop_m_b16(mask, vl)) RVVMF_EXP_CALL_FE_OVERFLOW();
    // NaNs, overflow, -inf -- automatically
    x = __riscv_vfmerge_vfm_f32m2(x, ZERO_F32, specialMask, vl);
}

forceinline void do_exp_argument_reduction_hl_f32m2(const vfloat32m2_t& x,
    vfloat32m2_t& yh, vfloat32m2_t& yl, vuint32m2_t& ei, vuint32m2_t& fi, size_t vl)
{
    vfloat32m2_t vmagicConst1 = __riscv_vfmv_v_f_f32m2(MAGIC_CONST_1_F32, vl);
    vfloat32m2_t h = __riscv_vfmadd_vf_f32m2(x, INV_LOG2_2K_F32, vmagicConst1, vl);
    vuint32m2_t hi = __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(h), MASK_HI_BIT_F32, vl);
    fi = __riscv_vand_vx_u32m2(hi, MASK_FI_BIT_F32, vl);
    ei = __riscv_vsrl_vx_u32m2(hi, TABLE_SIZE_DEG_F32, vl);
    h = __riscv_vfsub_vv_f32m2(h, vmagicConst1, vl);
    fma12_vf_f32m2(h, M_LOG2_2K_L_F32, __riscv_vfmadd_vf_f32m2(h, M_LOG2_2K_H_F32, x, vl), yh, yl, vl);
    yl = __riscv_vfmadd_vf_f32m2(h, M_LOG2_2K_LL_F32, yl, vl);
    fast_2_sum_vv_f32m2(yh, yl, yh, yl, vl);
}

forceinline void do_exp2_argument_reduction_f32m2(const vfloat32m2_t& x, vfloat32m2_t& y,
    vuint32m2_t& ei, vuint32m2_t& fi, size_t vl)  // exact
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

forceinline void calculate_exp_polynom_hl_f32m2(const vfloat32m2_t& yh, const vfloat32m2_t& yl, vfloat32m2_t& ph, vfloat32m2_t& pl, size_t vl)
{
    vfloat32m2_t sqryh = __riscv_vfmul_vv_f32m2(yh, yh, vl);
    vfloat32m2_t r = calc_polynom_deg_2_f32m2(yh, EXP_POL_COEFF_2_F32, EXP_POL_COEFF_3_F32, EXP_POL_COEFF_4_F32, vl); 
    fma12_vv_f32m2(sqryh, r, yh, ph, pl, vl);
    pl = __riscv_vfadd_vv_f32m2(pl, yl, vl);
}

forceinline void calculate_exp2_polynom_hl12_f32m2(const vfloat32m2_t& yh, vfloat32m2_t& ph, vfloat32m2_t& pl, size_t vl)
{
    vfloat32m2_t sqryh = __riscv_vfmul_vv_f32m2(yh, yh, vl);
    vfloat32m2_t r = calc_polynom_deg_2_f32m2(yh, EXP2_POL_COEFF_2_F32, EXP2_POL_COEFF_3_F32, EXP2_POL_COEFF_4_F32, vl); 
    fma12_ver2p1_vf_f32m2(yh, EXP2_POL_COEFF_1_F32, __riscv_vfmul_vv_f32m2(sqryh, r, vl), ph, pl, vl);
}

forceinline void update_exponent_f32m2(const vuint32m2_t& ei, vfloat32m2_t& res, size_t vl)
{
    res = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vv_u32m2(
        __riscv_vreinterpret_v_f32m2_u32m2(res), __riscv_vsll_vx_u32m2(ei, (size_t)23, vl), vl));
}

forceinline void update_exponent_with_subnormal_f32m2(const float& subnormalThreshold, const vfloat32m2_t& x,
    const vuint32m2_t& ei, vfloat32m2_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    uint32_t ninf = 0xff800000;
    vbool16_t subnormalMask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f32m2_b16(x, RVVMF_EXP_AS_FP32(ninf), vl),
        __riscv_vmflt_vf_f32m2_b16(x, subnormalThreshold, vl), vl);
    if (__riscv_vcpop_m_b16(subnormalMask, vl)) RVVMF_EXP_CALL_FE_UNDERFLOW();  // FE_UNDERFLOW
    
    vuint32m2_t shiftNum = __riscv_vreinterpret_v_i32m2_u32m2(__riscv_vneg_v_i32m2(__riscv_vreinterpret_v_u32m2_i32m2(ei), vl));
    shiftNum = __riscv_vadd_vx_u32m2(__riscv_vand_vx_u32m2(shiftNum, (uint32_t)0x000001ff, vl), (uint32_t)1, vl);
    shiftNum = __riscv_vsll_vx_u32m2(shiftNum, (size_t)23, vl);
    vfloat32m2_t subnormalRes = __riscv_vfadd_vv_f32m2(res, __riscv_vreinterpret_v_u32m2_f32m2(shiftNum), vl);
    subnormalRes = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vand_vx_u32m2(
        __riscv_vreinterpret_v_f32m2_u32m2(subnormalRes), (uint32_t)0x807fffff, vl));
#endif

    update_exponent_f32m2(ei, res, vl);
    
#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m2(res, subnormalRes, subnormalMask, vl);
#endif   
}

forceinline void reconstruct_exp_hl_hl_f32m2(const vfloat32m2_t& x, const vuint32m2_t& ei, const vfloat32m2_t& th, const vfloat32m2_t& tl,
    const vfloat32m2_t& pm2h, const vfloat32m2_t& pm2l, vfloat32m2_t& res, const float& subnormalThreshold, size_t vl)
{
    vfloat32m2_t sh, sl;
    fast_2_sum_fv_f32m2(ONE_F32, pm2h, sh, sl, vl);
    sl = __riscv_vfadd_vv_f32m2(sl, pm2l, vl);
    mul21_vv_f32m2(th, tl, sh, sl, res, vl);
    update_exponent_with_subnormal_f32m2(subnormalThreshold, x, ei, res, vl);
}

forceinline void reconstruct_expm1_f32m2(const vfloat32m2_t& th, const vfloat32m2_t& tl, 
    const vfloat32m2_t& pm2h, const vfloat32m2_t& pm2l, const vuint32m2_t& ei, vfloat32m2_t& res, size_t vl)
{        
    vfloat32m2_t rh, rl, sh, sl;
    fast_2_sum_fv_f32m2(ONE_F32, pm2h, rh, rl, vl);
    rl = __riscv_vfadd_vv_f32m2(rl, pm2l, vl);
    mul22_vv_f32m2(th, tl, rh, rl, sh, sl, vl);
    
    vuint32m2_t power = __riscv_vsll_vx_u32m2(ei, (size_t)23, vl);
    sh = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vv_u32m2(
        __riscv_vreinterpret_v_f32m2_u32m2(sh), power, vl));   
    vbool16_t slZeroMask = __riscv_vmfeq_vf_f32m2_b16(sl, ZERO_F32, vl);
    sl = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vv_u32m2(
        __riscv_vreinterpret_v_f32m2_u32m2(sl), power, vl));
    sl = __riscv_vfmerge_vfm_f32m2(sl, ZERO_F32, slZeroMask, vl);
    
    vbool16_t sortMask = __riscv_vmsgtu_vx_u32m2_b16(__riscv_vand_vx_u32m2(
        __riscv_vreinterpret_v_f32m2_u32m2(sh), (uint32_t)0x7f800000, vl), (uint32_t)0x3f800000, vl);
    vfloat32m2_t maxs = __riscv_vfmerge_vfm_f32m2(sh, EXPM1_UNDERFLOW_VALUE_F32, __riscv_vmnot_m_b16(sortMask, vl), vl);   
    vfloat32m2_t mins = __riscv_vfmerge_vfm_f32m2(sh, EXPM1_UNDERFLOW_VALUE_F32, sortMask, vl);
    fast_2_sum_vv_f32m2(maxs, mins, rh, rl, vl);
    
    res = __riscv_vfadd_vv_f32m2(rh, __riscv_vfadd_vv_f32m2(sl, rl, vl), vl);
}

forceinline void update_underflow_f32m2(const vfloat32m2_t& x, vfloat32m2_t& res,
    const float& underflowThreshold, const float& underflowValue, size_t vl)
{
    vbool16_t underflowMask = __riscv_vmflt_vf_f32m2_b16(x, underflowThreshold, vl);
    res = __riscv_vfmerge_vfm_f32m2(res, underflowValue, underflowMask, vl);
}

forceinline void set_sign_f32m2(const vfloat32m2_t& x, vfloat32m2_t& res, size_t vl)
{
    uint32_t signMask = 0x7fffffff;
    res = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vand_vx_u32m2(
        __riscv_vreinterpret_v_f32m2_u32m2(res), signMask, vl));
    res = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vor_vv_u32m2(__riscv_vand_vx_u32m2(
        __riscv_vreinterpret_v_f32m2_u32m2(x), ~signMask, vl), __riscv_vreinterpret_v_f32m2_u32m2(res), vl));
}

forceinline void process_linear_f32m2(const vfloat32m2_t& x, vfloat32m2_t& res, size_t vl)
{
    uint32_t signMask = 0x7fffffff;
    vfloat32m2_t xabs = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vand_vx_u32m2(
        __riscv_vreinterpret_v_f32m2_u32m2(x), signMask, vl));
    vbool16_t linearMask = __riscv_vmflt_vf_f32m2_b16(xabs, EXPM1_LINEAR_THRESHOLD_F32, vl);
    res = __riscv_vmerge_vvm_f32m2(res, x, linearMask, vl);
}

// ---------------------------- m4 ----------------------------

forceinline void check_special_cases_f32m4(vfloat32m4_t& x, vfloat32m4_t& special, vbool8_t& specialMask,
    const float& overflowThreshold, size_t vl)
{ 
    // check +inf
    uint32_t pinf = 0x7f800000;
    specialMask = __riscv_vmfeq_vf_f32m4_b8(x, RVVMF_EXP_AS_FP32(pinf), vl);
    special = __riscv_vfmerge_vfm_f32m4(x, RVVMF_EXP_AS_FP32(pinf), specialMask, vl);
    // check overflow
    vbool8_t mask = __riscv_vmand_mm_b8(__riscv_vmfgt_vf_f32m4_b8(x, overflowThreshold, vl),
        __riscv_vmflt_vf_f32m4_b8(x, RVVMF_EXP_AS_FP32(pinf), vl), vl);
    special = __riscv_vfmerge_vfm_f32m4(special, RVVMF_EXP_AS_FP32(pinf), mask, vl);
    specialMask = __riscv_vmor_mm_b8(specialMask, mask, vl);  
    if (__riscv_vcpop_m_b8(mask, vl)) RVVMF_EXP_CALL_FE_OVERFLOW();
    // NaNs, overflow, -inf -- automatically
    x = __riscv_vfmerge_vfm_f32m4(x, ZERO_F32, specialMask, vl);
}

forceinline void do_exp_argument_reduction_hl_f32m4(const vfloat32m4_t& x,
    vfloat32m4_t& yh, vfloat32m4_t& yl, vuint32m4_t& ei, vuint32m4_t& fi, size_t vl)
{
    vfloat32m4_t vmagicConst1 = __riscv_vfmv_v_f_f32m4(MAGIC_CONST_1_F32, vl);
    vfloat32m4_t h = __riscv_vfmadd_vf_f32m4(x, INV_LOG2_2K_F32, vmagicConst1, vl);
    vuint32m4_t hi = __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(h), MASK_HI_BIT_F32, vl);
    fi = __riscv_vand_vx_u32m4(hi, MASK_FI_BIT_F32, vl);
    ei = __riscv_vsrl_vx_u32m4(hi, TABLE_SIZE_DEG_F32, vl);
    h = __riscv_vfsub_vv_f32m4(h, vmagicConst1, vl);
    fma12_vf_f32m4(h, M_LOG2_2K_L_F32, __riscv_vfmadd_vf_f32m4(h, M_LOG2_2K_H_F32, x, vl), yh, yl, vl);
    yl = __riscv_vfmadd_vf_f32m4(h, M_LOG2_2K_LL_F32, yl, vl);
    fast_2_sum_vv_f32m4(yh, yl, yh, yl, vl);
}

forceinline void do_exp2_argument_reduction_f32m4(const vfloat32m4_t& x, vfloat32m4_t& y,
    vuint32m4_t& ei, vuint32m4_t& fi, size_t vl)  // exact
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

forceinline void calculate_exp_polynom_hl_f32m4(const vfloat32m4_t& yh, const vfloat32m4_t& yl, vfloat32m4_t& ph, vfloat32m4_t& pl, size_t vl)
{
    vfloat32m4_t sqryh = __riscv_vfmul_vv_f32m4(yh, yh, vl);
    vfloat32m4_t r = calc_polynom_deg_2_f32m4(yh, EXP_POL_COEFF_2_F32, EXP_POL_COEFF_3_F32, EXP_POL_COEFF_4_F32, vl); 
    fma12_vv_f32m4(sqryh, r, yh, ph, pl, vl);
    pl = __riscv_vfadd_vv_f32m4(pl, yl, vl);
}

forceinline void calculate_exp2_polynom_hl12_f32m4(const vfloat32m4_t& yh, vfloat32m4_t& ph, vfloat32m4_t& pl, size_t vl)
{
    vfloat32m4_t sqryh = __riscv_vfmul_vv_f32m4(yh, yh, vl);
    vfloat32m4_t r = calc_polynom_deg_2_f32m4(yh, EXP2_POL_COEFF_2_F32, EXP2_POL_COEFF_3_F32, EXP2_POL_COEFF_4_F32, vl); 
    fma12_ver2p1_vf_f32m4(yh, EXP2_POL_COEFF_1_F32, __riscv_vfmul_vv_f32m4(sqryh, r, vl), ph, pl, vl);
}

forceinline void update_exponent_f32m4(const vuint32m4_t& ei, vfloat32m4_t& res, size_t vl)
{
    res = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vadd_vv_u32m4(
        __riscv_vreinterpret_v_f32m4_u32m4(res), __riscv_vsll_vx_u32m4(ei, (size_t)23, vl), vl));
}

forceinline void update_exponent_with_subnormal_f32m4(const float& subnormalThreshold, const vfloat32m4_t& x,
    const vuint32m4_t& ei, vfloat32m4_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    uint32_t ninf = 0xff800000;
    vbool8_t subnormalMask = __riscv_vmand_mm_b8(__riscv_vmfgt_vf_f32m4_b8(x, RVVMF_EXP_AS_FP32(ninf), vl),
        __riscv_vmflt_vf_f32m4_b8(x, subnormalThreshold, vl), vl);
    if (__riscv_vcpop_m_b8(subnormalMask, vl)) RVVMF_EXP_CALL_FE_UNDERFLOW();  // FE_UNDERFLOW
    
    vuint32m4_t shiftNum = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vneg_v_i32m4(__riscv_vreinterpret_v_u32m4_i32m4(ei), vl));
    shiftNum = __riscv_vadd_vx_u32m4(__riscv_vand_vx_u32m4(shiftNum, (uint32_t)0x000001ff, vl), (uint32_t)1, vl);
    shiftNum = __riscv_vsll_vx_u32m4(shiftNum, (size_t)23, vl);
    vfloat32m4_t subnormalRes = __riscv_vfadd_vv_f32m4(res, __riscv_vreinterpret_v_u32m4_f32m4(shiftNum), vl);
    subnormalRes = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vand_vx_u32m4(
        __riscv_vreinterpret_v_f32m4_u32m4(subnormalRes), (uint32_t)0x807fffff, vl));
#endif

    update_exponent_f32m4(ei, res, vl);
    
#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m4(res, subnormalRes, subnormalMask, vl);
#endif   
}

forceinline void reconstruct_exp_hl_hl_f32m4(const vfloat32m4_t& x, const vuint32m4_t& ei, const vfloat32m4_t& th, const vfloat32m4_t& tl,
    const vfloat32m4_t& pm4h, const vfloat32m4_t& pm4l, vfloat32m4_t& res, const float& subnormalThreshold, size_t vl)
{
    vfloat32m4_t sh, sl;
    fast_2_sum_fv_f32m4(ONE_F32, pm4h, sh, sl, vl);
    sl = __riscv_vfadd_vv_f32m4(sl, pm4l, vl);
    mul21_vv_f32m4(th, tl, sh, sl, res, vl);
    update_exponent_with_subnormal_f32m4(subnormalThreshold, x, ei, res, vl);
}

forceinline void reconstruct_expm1_f32m4(const vfloat32m4_t& th, const vfloat32m4_t& tl, 
    const vfloat32m4_t& pm4h, const vfloat32m4_t& pm4l, const vuint32m4_t& ei, vfloat32m4_t& res, size_t vl)
{        
    vfloat32m4_t rh, rl, sh, sl;
    fast_2_sum_fv_f32m4(ONE_F32, pm4h, rh, rl, vl);
    rl = __riscv_vfadd_vv_f32m4(rl, pm4l, vl);
    mul22_vv_f32m4(th, tl, rh, rl, sh, sl, vl);
    
    vuint32m4_t power = __riscv_vsll_vx_u32m4(ei, (size_t)23, vl);
    sh = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vadd_vv_u32m4(
        __riscv_vreinterpret_v_f32m4_u32m4(sh), power, vl));   
    vbool8_t slZeroMask = __riscv_vmfeq_vf_f32m4_b8(sl, ZERO_F32, vl);
    sl = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vadd_vv_u32m4(
        __riscv_vreinterpret_v_f32m4_u32m4(sl), power, vl));
    sl = __riscv_vfmerge_vfm_f32m4(sl, ZERO_F32, slZeroMask, vl);
    
    vbool8_t sortMask = __riscv_vmsgtu_vx_u32m4_b8(__riscv_vand_vx_u32m4(
        __riscv_vreinterpret_v_f32m4_u32m4(sh), (uint32_t)0x7f800000, vl), (uint32_t)0x3f800000, vl);
    vfloat32m4_t maxs = __riscv_vfmerge_vfm_f32m4(sh, EXPM1_UNDERFLOW_VALUE_F32, __riscv_vmnot_m_b8(sortMask, vl), vl);   
    vfloat32m4_t mins = __riscv_vfmerge_vfm_f32m4(sh, EXPM1_UNDERFLOW_VALUE_F32, sortMask, vl);
    fast_2_sum_vv_f32m4(maxs, mins, rh, rl, vl);
    
    res = __riscv_vfadd_vv_f32m4(rh, __riscv_vfadd_vv_f32m4(sl, rl, vl), vl);
}

forceinline void update_underflow_f32m4(const vfloat32m4_t& x, vfloat32m4_t& res,
    const float& underflowThreshold, const float& underflowValue, size_t vl)
{
    vbool8_t underflowMask = __riscv_vmflt_vf_f32m4_b8(x, underflowThreshold, vl);
    res = __riscv_vfmerge_vfm_f32m4(res, underflowValue, underflowMask, vl);
}

forceinline void set_sign_f32m4(const vfloat32m4_t& x, vfloat32m4_t& res, size_t vl)
{
    uint32_t signMask = 0x7fffffff;
    res = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vand_vx_u32m4(
        __riscv_vreinterpret_v_f32m4_u32m4(res), signMask, vl));
    res = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vv_u32m4(__riscv_vand_vx_u32m4(
        __riscv_vreinterpret_v_f32m4_u32m4(x), ~signMask, vl), __riscv_vreinterpret_v_f32m4_u32m4(res), vl));
}

forceinline void process_linear_f32m4(const vfloat32m4_t& x, vfloat32m4_t& res, size_t vl)
{
    uint32_t signMask = 0x7fffffff;
    vfloat32m4_t xabs = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vand_vx_u32m4(
        __riscv_vreinterpret_v_f32m4_u32m4(x), signMask, vl));
    vbool8_t linearMask = __riscv_vmflt_vf_f32m4_b8(xabs, EXPM1_LINEAR_THRESHOLD_F32, vl);
    res = __riscv_vmerge_vvm_f32m4(res, x, linearMask, vl);
}
