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
 *   File:  hexp.inl                                     *
 *   Contains: helper built-in functions for exp, exp2   *
 *             and expm1 functions (float16_t)           *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#include "exp_utilities.inl"

typedef _Float16 FLOAT16_T;

const FLOAT16_T ZERO_F16 = 0.0f16;
const FLOAT16_T ONE_F16 = 1.0f16;

const FLOAT16_T EXP_EXPM1_OVERFLOW_THRESHOLD_F16 = 0x1.62cp3f16;
const FLOAT16_T EXP2_EXP2M1_OVERFLOW_THRESHOLD_F16 = 0x1.ffcp3f16;
const FLOAT16_T EXP_SUBNORMAL_THRESHOLD_F16 = -0x1.368p3f16;
const FLOAT16_T EXP2_SUBNORMAL_THRESHOLD_F16 = -0x1.cp3f16;
const FLOAT16_T EXP_ZERO_THRESHOLD_F16 = -0x1.154p4f16;
const FLOAT16_T EXP2_ZERO_THRESHOLD_F16 = -0x1.8fcp4f16;
const FLOAT16_T EXP_UNDERFLOW_VALUE_F16 = 0.0f16;
const FLOAT16_T EXPM1_UNDERFLOW_THRESHOLD_F16 = -0x1.0ap3f16;
const FLOAT16_T EXPM1_LINEAR_THRESHOLD_F16 = 0x1.6ap-11f16;
const FLOAT16_T EXPM1_UNDERFLOW_VALUE_F16 = -1.0f16;

const size_t TABLE_SIZE_DEG_F16 = 3;
const FLOAT16_T EXP2_TABLE_SIZE_DEG_F16 = 0x1p3f16;
const FLOAT16_T M_EXP2_M_TABLE_SIZE_DEG_F16 = -0x1p-3f16;
const uint16_t MASK_FI_BIT_F16 = 0x0007;
const uint16_t MASK_HI_BIT_F16 = 0x01ff;
const FLOAT16_T MAGIC_CONST_1_F16 = 1536.0f16;
const FLOAT16_T INV_LOG2_2K_F16 = 0x1.714p3f16;
const FLOAT16_T M_LOG2_2K_H_F16 = -0x1.6p-4f16;
const FLOAT16_T M_LOG2_2K_L_F16 = -0x1.72p-11f16;
const FLOAT16_T M_LOG2_2K_LL_F16 = -0x1.8p-23f16;

static const FLOAT16_T LOOK_UP_TABLE_HIGH_F16[8] = {
    0x1p0f16, 0x1.174p0f16, 0x1.308p0f16, 0x1.4cp0f16,
    0x1.6ap0f16, 0x1.8acp0f16, 0x1.ae8p0f16, 0x1.d58p0f16
};
static const FLOAT16_T LOOK_UP_TABLE_LOW_F16[8] = {
    0.0f16, -0x1.47cp-12f16, -0x1.02p-12f16, -0x1.298p-15f16,
    0x1.3ccp-13f16, 0x1.ca8p-13f16, 0x1.3f4p-13f16, 0x1.8ep-16f16
};

const FLOAT16_T EXP_POL_COEFF_2_F16 = 0x1p-1f16;
const FLOAT16_T EXP_POL_COEFF_3_F16 = 0x1.55p-3f16;

const FLOAT16_T EXP2_POL_COEFF_1_F16 = 0x1.63p-1f16;
const FLOAT16_T EXP2_POL_COEFF_2_F16 = 0x1.ec4p-3f16;

// ---------------------------- m1 ----------------------------

forceinline void check_special_cases_f16m1(vfloat16m1_t& x, vfloat16m1_t& special, vbool16_t& specialMask,
    const FLOAT16_T& overflowThreshold, size_t vl)
{ 
    // check +inf
    uint16_t pinf = 0x7c00;
    specialMask = __riscv_vmfeq_vf_f16m1_b16(x, RVVMF_EXP_AS_FP16(pinf), vl);
    special = __riscv_vfmerge_vfm_f16m1(x, RVVMF_EXP_AS_FP16(pinf), specialMask, vl);
    // check overflow
    vbool16_t mask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f16m1_b16(x, overflowThreshold, vl),
        __riscv_vmflt_vf_f16m1_b16(x, RVVMF_EXP_AS_FP16(pinf), vl), vl);
    special = __riscv_vfmerge_vfm_f16m1(special, RVVMF_EXP_AS_FP16(pinf), mask, vl);
    specialMask = __riscv_vmor_mm_b16(specialMask, mask, vl);  
    if (__riscv_vcpop_m_b16(mask, vl)) RVVMF_EXP_CALL_FE_OVERFLOW();
    // NaNs, overflow, -inf -- automatically
    x = __riscv_vfmerge_vfm_f16m1(x, ZERO_F16, specialMask, vl);
}

forceinline void do_exp_argument_reduction_hl_f16m1(const vfloat16m1_t& x,
    vfloat16m1_t& yh, vfloat16m1_t& yl, vuint16m1_t& ei, vuint16m1_t& fi, size_t vl)
{
    vfloat16m1_t vmagicConst1 = __riscv_vfmv_v_f_f16m1(MAGIC_CONST_1_F16, vl);
    vfloat16m1_t h = __riscv_vfmadd_vf_f16m1(x, INV_LOG2_2K_F16, vmagicConst1, vl);
    vuint16m1_t hi = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(h), MASK_HI_BIT_F16, vl);
    fi = __riscv_vand_vx_u16m1(hi, MASK_FI_BIT_F16, vl);
    ei = __riscv_vsrl_vx_u16m1(hi, TABLE_SIZE_DEG_F16, vl);
    h = __riscv_vfsub_vv_f16m1(h, vmagicConst1, vl);
    fma12_ver2p2_vf_f16m1(h, M_LOG2_2K_L_F16, __riscv_vfmadd_vf_f16m1(h, M_LOG2_2K_H_F16, x, vl), yh, yl, vl);
    yl = __riscv_vfmadd_vf_f16m1(h, M_LOG2_2K_LL_F16, yl, vl);
    fast_2_sum_vv_f16m1(yh, yl, yh, yl, vl);
}

forceinline void do_exp2_argument_reduction_f16m1(const vfloat16m1_t& x, vfloat16m1_t& y,
    vuint16m1_t& ei, vuint16m1_t& fi, size_t vl)  // exact
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
    index = __riscv_vmul_vx_u16m1(index, uint16_t(sizeof(FLOAT16_T)), vl);
    th = __riscv_vloxei16_v_f16m1(LOOK_UP_TABLE_HIGH_F16, index, vl);
    tl = __riscv_vloxei16_v_f16m1(LOOK_UP_TABLE_LOW_F16, index, vl);
}

forceinline void calculate_exp_polynom_hl_f16m1(const vfloat16m1_t& yh, const vfloat16m1_t& yl, vfloat16m1_t& ph, vfloat16m1_t& pl, size_t vl)
{
    vfloat16m1_t sqryh = __riscv_vfmul_vv_f16m1(yh, yh, vl);
    vfloat16m1_t r = calc_polynom_deg_1_f16m1(yh, EXP_POL_COEFF_2_F16, EXP_POL_COEFF_3_F16, vl); 
    fma12_vv_f16m1(sqryh, r, yh, ph, pl, vl);
    pl = __riscv_vfadd_vv_f16m1(pl, yl, vl);
}

forceinline void calculate_exp2_polynom_hl12_f16m1(const vfloat16m1_t& yh, vfloat16m1_t& ph, vfloat16m1_t& pl, size_t vl)
{
    vfloat16m1_t sqryh = __riscv_vfmul_vv_f16m1(yh, yh, vl);
    vfloat16m1_t r = __riscv_vfmv_v_f_f16m1(EXP2_POL_COEFF_2_F16, vl); 
    fma12_ver2p1_vf_f16m1(yh, EXP2_POL_COEFF_1_F16, __riscv_vfmul_vv_f16m1(sqryh, r, vl), ph, pl, vl);
}

forceinline void update_exponent_f16m1(const vuint16m1_t& ei, vfloat16m1_t& res, size_t vl)
{
    res = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vadd_vv_u16m1(
        __riscv_vreinterpret_v_f16m1_u16m1(res), __riscv_vsll_vx_u16m1(ei, (size_t)10, vl), vl));
}

forceinline void update_exponent_with_subnormal_f16m1(const FLOAT16_T& subnormalThreshold, const vfloat16m1_t& x,
    const vuint16m1_t& ei, vfloat16m1_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    uint16_t ninf = 0xfc00;
    vbool16_t subnormalMask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f16m1_b16(x, RVVMF_EXP_AS_FP16(ninf), vl),
        __riscv_vmflt_vf_f16m1_b16(x, subnormalThreshold, vl), vl);
    if (__riscv_vcpop_m_b16(subnormalMask, vl)) RVVMF_EXP_CALL_FE_UNDERFLOW();  // FE_UNDERFLOW
    
    vuint16m1_t shiftNum = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vneg_v_i16m1(__riscv_vreinterpret_v_u16m1_i16m1(ei), vl));
    shiftNum = __riscv_vadd_vx_u16m1(__riscv_vand_vx_u16m1(shiftNum, (uint16_t)0x003f, vl), (uint16_t)1, vl);
    shiftNum = __riscv_vsll_vx_u16m1(shiftNum, (size_t)10, vl);
    vfloat16m1_t subnormalRes = __riscv_vfadd_vv_f16m1(res, __riscv_vreinterpret_v_u16m1_f16m1(shiftNum), vl);
    subnormalRes = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vx_u16m1(
        __riscv_vreinterpret_v_f16m1_u16m1(subnormalRes), (uint16_t)0x83ff, vl));
#endif

    update_exponent_f16m1(ei, res, vl);
    
#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m1(res, subnormalRes, subnormalMask, vl);
#endif   
}

forceinline void reconstruct_exp_hl_hl_f16m1(const vfloat16m1_t& x, const vuint16m1_t& ei, const vfloat16m1_t& th, const vfloat16m1_t& tl,
    const vfloat16m1_t& pm1h, const vfloat16m1_t& pm1l, vfloat16m1_t& res, const FLOAT16_T& subnormalThreshold, size_t vl)
{
    vfloat16m1_t sh, sl;
    fast_2_sum_fv_f16m1(ONE_F16, pm1h, sh, sl, vl);
    sl = __riscv_vfadd_vv_f16m1(sl, pm1l, vl);
    mul21_vv_f16m1(th, tl, sh, sl, res, vl);
    update_exponent_with_subnormal_f16m1(subnormalThreshold, x, ei, res, vl);
}

forceinline void reconstruct_expm1_f16m1(const vfloat16m1_t& th, const vfloat16m1_t& tl, 
    const vfloat16m1_t& pm1h, const vfloat16m1_t& pm1l, const vuint16m1_t& ei, vfloat16m1_t& res, size_t vl)
{        
    vfloat16m1_t rh, rl, sh, sl, sl1;
    fast_2_sum_fv_f16m1(ONE_F16, pm1h, rh, rl, vl);
    rl = __riscv_vfadd_vv_f16m1(rl, pm1l, vl);
    mul22_vv_f16m1(th, tl, rh, rl, sh, sl, vl);
    
    vuint16m1_t power = __riscv_vsll_vx_u16m1(ei, (size_t)10, vl);
    sh = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vadd_vv_u16m1(
        __riscv_vreinterpret_v_f16m1_u16m1(sh), power, vl));   
    vuint16m1_t power2 = __riscv_vsll_vx_u16m1(__riscv_vadd_vx_u16m1(ei, (uint16_t)15, vl), (size_t)10, vl);  
    sl1 = __riscv_vfmul_vv_f16m1(__riscv_vreinterpret_v_u16m1_f16m1(power2), sl, vl);
    vbool16_t slZeroMask = __riscv_vmfeq_vf_f16m1_b16(sl, ZERO_F16, vl);
    sl = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vadd_vv_u16m1(
        __riscv_vreinterpret_v_f16m1_u16m1(sl), power, vl));
    sl = __riscv_vfmerge_vfm_f16m1(sl, ZERO_F16, slZeroMask, vl);
    vbool16_t infPowerMask = __riscv_vmseq_vx_u16m1_b16(power2, (uint16_t)0x7c00, vl);
    sl = __riscv_vmerge_vvm_f16m1(sl1, sl, infPowerMask, vl);
    
    vbool16_t sortMask = __riscv_vmsgtu_vx_u16m1_b16(__riscv_vand_vx_u16m1(__riscv_vreinterpret_v_f16m1_u16m1(sh),
        (uint16_t)0x7c00, vl), (uint16_t)0x3c00, vl);
    vfloat16m1_t maxs = __riscv_vfmerge_vfm_f16m1(sh, EXPM1_UNDERFLOW_VALUE_F16, __riscv_vmnot_m_b16(sortMask, vl), vl);   
    vfloat16m1_t mins = __riscv_vfmerge_vfm_f16m1(sh, EXPM1_UNDERFLOW_VALUE_F16, sortMask, vl);
    fast_2_sum_vv_f16m1(maxs, mins, rh, rl, vl);
    
    res = __riscv_vfadd_vv_f16m1(rh, __riscv_vfadd_vv_f16m1(sl, rl, vl), vl);
}

forceinline void update_underflow_f16m1(const vfloat16m1_t& x, vfloat16m1_t& res,
    const FLOAT16_T& underflowThreshold, const FLOAT16_T& underflowValue, size_t vl)
{
    vbool16_t underflowMask = __riscv_vmflt_vf_f16m1_b16(x, underflowThreshold, vl);
    res = __riscv_vfmerge_vfm_f16m1(res, underflowValue, underflowMask, vl);
}

forceinline void set_pos_sign_f16m1(vfloat16m1_t& res, size_t vl)
{
    uint16_t signMask = 0x7fff;
    res = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vx_u16m1(
        __riscv_vreinterpret_v_f16m1_u16m1(res), signMask, vl));
}

forceinline void set_sign_f16m1(const vfloat16m1_t& x, vfloat16m1_t& res, size_t vl)
{
    uint16_t signMask = 0x7fff;
    res = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vx_u16m1(
        __riscv_vreinterpret_v_f16m1_u16m1(res), signMask, vl));
    res = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vor_vv_u16m1(__riscv_vand_vx_u16m1(
        __riscv_vreinterpret_v_f16m1_u16m1(x), ~signMask, vl), __riscv_vreinterpret_v_f16m1_u16m1(res), vl));
}

forceinline void process_linear_f16m1(const vfloat16m1_t& x, vfloat16m1_t& res, size_t vl)
{
    uint16_t signMask = 0x7fff;
    vfloat16m1_t xabs = __riscv_vreinterpret_v_u16m1_f16m1(__riscv_vand_vx_u16m1(
        __riscv_vreinterpret_v_f16m1_u16m1(x), signMask, vl));
    vbool16_t linearMask = __riscv_vmflt_vf_f16m1_b16(xabs, EXPM1_LINEAR_THRESHOLD_F16, vl);
    res = __riscv_vmerge_vvm_f16m1(res, x, linearMask, vl);
}

// ---------------------------- m2 ----------------------------

forceinline void check_special_cases_f16m2(vfloat16m2_t& x, vfloat16m2_t& special, vbool8_t& specialMask,
    const FLOAT16_T& overflowThreshold, size_t vl)
{ 
    // check +inf
    uint16_t pinf = 0x7c00;
    specialMask = __riscv_vmfeq_vf_f16m2_b8(x, RVVMF_EXP_AS_FP16(pinf), vl);
    special = __riscv_vfmerge_vfm_f16m2(x, RVVMF_EXP_AS_FP16(pinf), specialMask, vl);
    // check overflow
    vbool8_t mask = __riscv_vmand_mm_b8(__riscv_vmfgt_vf_f16m2_b8(x, overflowThreshold, vl),
        __riscv_vmflt_vf_f16m2_b8(x, RVVMF_EXP_AS_FP16(pinf), vl), vl);
    special = __riscv_vfmerge_vfm_f16m2(special, RVVMF_EXP_AS_FP16(pinf), mask, vl);
    specialMask = __riscv_vmor_mm_b8(specialMask, mask, vl);  
    if (__riscv_vcpop_m_b8(mask, vl)) RVVMF_EXP_CALL_FE_OVERFLOW();
    // NaNs, overflow, -inf -- automatically
    x = __riscv_vfmerge_vfm_f16m2(x, ZERO_F16, specialMask, vl);
}

forceinline void do_exp_argument_reduction_hl_f16m2(const vfloat16m2_t& x,
    vfloat16m2_t& yh, vfloat16m2_t& yl, vuint16m2_t& ei, vuint16m2_t& fi, size_t vl)
{
    vfloat16m2_t vmagicConst1 = __riscv_vfmv_v_f_f16m2(MAGIC_CONST_1_F16, vl);
    vfloat16m2_t h = __riscv_vfmadd_vf_f16m2(x, INV_LOG2_2K_F16, vmagicConst1, vl);
    vuint16m2_t hi = __riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(h), MASK_HI_BIT_F16, vl);
    fi = __riscv_vand_vx_u16m2(hi, MASK_FI_BIT_F16, vl);
    ei = __riscv_vsrl_vx_u16m2(hi, TABLE_SIZE_DEG_F16, vl);
    h = __riscv_vfsub_vv_f16m2(h, vmagicConst1, vl);
    fma12_ver2p2_vf_f16m2(h, M_LOG2_2K_L_F16, __riscv_vfmadd_vf_f16m2(h, M_LOG2_2K_H_F16, x, vl), yh, yl, vl);
    yl = __riscv_vfmadd_vf_f16m2(h, M_LOG2_2K_LL_F16, yl, vl);
    fast_2_sum_vv_f16m2(yh, yl, yh, yl, vl);
}

forceinline void do_exp2_argument_reduction_f16m2(const vfloat16m2_t& x, vfloat16m2_t& y,
    vuint16m2_t& ei, vuint16m2_t& fi, size_t vl)  // exact
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
    index = __riscv_vmul_vx_u16m2(index, uint16_t(sizeof(FLOAT16_T)), vl);
    th = __riscv_vloxei16_v_f16m2(LOOK_UP_TABLE_HIGH_F16, index, vl);
    tl = __riscv_vloxei16_v_f16m2(LOOK_UP_TABLE_LOW_F16, index, vl);
}

forceinline void calculate_exp_polynom_hl_f16m2(const vfloat16m2_t& yh, const vfloat16m2_t& yl, vfloat16m2_t& ph, vfloat16m2_t& pl, size_t vl)
{
    vfloat16m2_t sqryh = __riscv_vfmul_vv_f16m2(yh, yh, vl);
    vfloat16m2_t r = calc_polynom_deg_1_f16m2(yh, EXP_POL_COEFF_2_F16, EXP_POL_COEFF_3_F16, vl); 
    fma12_vv_f16m2(sqryh, r, yh, ph, pl, vl);
    pl = __riscv_vfadd_vv_f16m2(pl, yl, vl);
}

forceinline void calculate_exp2_polynom_hl12_f16m2(const vfloat16m2_t& yh, vfloat16m2_t& ph, vfloat16m2_t& pl, size_t vl)
{
    vfloat16m2_t sqryh = __riscv_vfmul_vv_f16m2(yh, yh, vl);
    vfloat16m2_t r = __riscv_vfmv_v_f_f16m2(EXP2_POL_COEFF_2_F16, vl); 
    fma12_ver2p1_vf_f16m2(yh, EXP2_POL_COEFF_1_F16, __riscv_vfmul_vv_f16m2(sqryh, r, vl), ph, pl, vl);
}

forceinline void update_exponent_f16m2(const vuint16m2_t& ei, vfloat16m2_t& res, size_t vl)
{
    res = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vadd_vv_u16m2(
        __riscv_vreinterpret_v_f16m2_u16m2(res), __riscv_vsll_vx_u16m2(ei, (size_t)10, vl), vl));
}

forceinline void update_exponent_with_subnormal_f16m2(const FLOAT16_T& subnormalThreshold, const vfloat16m2_t& x,
    const vuint16m2_t& ei, vfloat16m2_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    uint16_t ninf = 0xfc00;
    vbool8_t subnormalMask = __riscv_vmand_mm_b8(__riscv_vmfgt_vf_f16m2_b8(x, RVVMF_EXP_AS_FP16(ninf), vl),
        __riscv_vmflt_vf_f16m2_b8(x, subnormalThreshold, vl), vl);
    if (__riscv_vcpop_m_b8(subnormalMask, vl)) RVVMF_EXP_CALL_FE_UNDERFLOW();  // FE_UNDERFLOW
    
    vuint16m2_t shiftNum = __riscv_vreinterpret_v_i16m2_u16m2(__riscv_vneg_v_i16m2(__riscv_vreinterpret_v_u16m2_i16m2(ei), vl));
    shiftNum = __riscv_vadd_vx_u16m2(__riscv_vand_vx_u16m2(shiftNum, (uint16_t)0x003f, vl), (uint16_t)1, vl);
    shiftNum = __riscv_vsll_vx_u16m2(shiftNum, (size_t)10, vl);
    vfloat16m2_t subnormalRes = __riscv_vfadd_vv_f16m2(res, __riscv_vreinterpret_v_u16m2_f16m2(shiftNum), vl);
    subnormalRes = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vx_u16m2(
        __riscv_vreinterpret_v_f16m2_u16m2(subnormalRes), (uint16_t)0x83ff, vl));
#endif

    update_exponent_f16m2(ei, res, vl);
    
#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m2(res, subnormalRes, subnormalMask, vl);
#endif   
}

forceinline void reconstruct_exp_hl_hl_f16m2(const vfloat16m2_t& x, const vuint16m2_t& ei, const vfloat16m2_t& th, const vfloat16m2_t& tl,
    const vfloat16m2_t& pm2h, const vfloat16m2_t& pm2l, vfloat16m2_t& res, const FLOAT16_T& subnormalThreshold, size_t vl)
{
    vfloat16m2_t sh, sl;
    fast_2_sum_fv_f16m2(ONE_F16, pm2h, sh, sl, vl);
    sl = __riscv_vfadd_vv_f16m2(sl, pm2l, vl);
    mul21_vv_f16m2(th, tl, sh, sl, res, vl);
    update_exponent_with_subnormal_f16m2(subnormalThreshold, x, ei, res, vl);
}

forceinline void reconstruct_expm1_f16m2(const vfloat16m2_t& th, const vfloat16m2_t& tl, 
    const vfloat16m2_t& pm2h, const vfloat16m2_t& pm2l, const vuint16m2_t& ei, vfloat16m2_t& res, size_t vl)
{        
    vfloat16m2_t rh, rl, sh, sl, sl1;
    fast_2_sum_fv_f16m2(ONE_F16, pm2h, rh, rl, vl);
    rl = __riscv_vfadd_vv_f16m2(rl, pm2l, vl);
    mul22_vv_f16m2(th, tl, rh, rl, sh, sl, vl);
    
    vuint16m2_t power = __riscv_vsll_vx_u16m2(ei, (size_t)10, vl);
    sh = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vadd_vv_u16m2(
        __riscv_vreinterpret_v_f16m2_u16m2(sh), power, vl));   
    vuint16m2_t power2 = __riscv_vsll_vx_u16m2(__riscv_vadd_vx_u16m2(ei, (uint16_t)15, vl), (size_t)10, vl);  
    sl1 = __riscv_vfmul_vv_f16m2(__riscv_vreinterpret_v_u16m2_f16m2(power2), sl, vl);
    vbool8_t slZeroMask = __riscv_vmfeq_vf_f16m2_b8(sl, ZERO_F16, vl);
    sl = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vadd_vv_u16m2(
        __riscv_vreinterpret_v_f16m2_u16m2(sl), power, vl));
    sl = __riscv_vfmerge_vfm_f16m2(sl, ZERO_F16, slZeroMask, vl);
    vbool8_t infPowerMask = __riscv_vmseq_vx_u16m2_b8(power2, (uint16_t)0x7c00, vl);
    sl = __riscv_vmerge_vvm_f16m2(sl1, sl, infPowerMask, vl);
    
    vbool8_t sortMask = __riscv_vmsgtu_vx_u16m2_b8(__riscv_vand_vx_u16m2(__riscv_vreinterpret_v_f16m2_u16m2(sh),
        (uint16_t)0x7c00, vl), (uint16_t)0x3c00, vl);
    vfloat16m2_t maxs = __riscv_vfmerge_vfm_f16m2(sh, EXPM1_UNDERFLOW_VALUE_F16, __riscv_vmnot_m_b8(sortMask, vl), vl);   
    vfloat16m2_t mins = __riscv_vfmerge_vfm_f16m2(sh, EXPM1_UNDERFLOW_VALUE_F16, sortMask, vl);
    fast_2_sum_vv_f16m2(maxs, mins, rh, rl, vl);
    
    res = __riscv_vfadd_vv_f16m2(rh, __riscv_vfadd_vv_f16m2(sl, rl, vl), vl);
}

forceinline void update_underflow_f16m2(const vfloat16m2_t& x, vfloat16m2_t& res,
    const FLOAT16_T& underflowThreshold, const FLOAT16_T& underflowValue, size_t vl)
{
    vbool8_t underflowMask = __riscv_vmflt_vf_f16m2_b8(x, underflowThreshold, vl);
    res = __riscv_vfmerge_vfm_f16m2(res, underflowValue, underflowMask, vl);
}

forceinline void set_pos_sign_f16m2(vfloat16m2_t& res, size_t vl)
{
    uint16_t signMask = 0x7fff;
    res = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vx_u16m2(
        __riscv_vreinterpret_v_f16m2_u16m2(res), signMask, vl));
}

forceinline void set_sign_f16m2(const vfloat16m2_t& x, vfloat16m2_t& res, size_t vl)
{
    uint16_t signMask = 0x7fff;
    res = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vx_u16m2(
        __riscv_vreinterpret_v_f16m2_u16m2(res), signMask, vl));
    res = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vor_vv_u16m2(__riscv_vand_vx_u16m2(
        __riscv_vreinterpret_v_f16m2_u16m2(x), ~signMask, vl), __riscv_vreinterpret_v_f16m2_u16m2(res), vl));
}

forceinline void process_linear_f16m2(const vfloat16m2_t& x, vfloat16m2_t& res, size_t vl)
{
    uint16_t signMask = 0x7fff;
    vfloat16m2_t xabs = __riscv_vreinterpret_v_u16m2_f16m2(__riscv_vand_vx_u16m2(
        __riscv_vreinterpret_v_f16m2_u16m2(x), signMask, vl));
    vbool8_t linearMask = __riscv_vmflt_vf_f16m2_b8(xabs, EXPM1_LINEAR_THRESHOLD_F16, vl);
    res = __riscv_vmerge_vvm_f16m2(res, x, linearMask, vl);
}

// ---------------------------- m4 ----------------------------

forceinline void check_special_cases_f16m4(vfloat16m4_t& x, vfloat16m4_t& special, vbool4_t& specialMask,
    const FLOAT16_T& overflowThreshold, size_t vl)
{ 
    // check +inf
    uint16_t pinf = 0x7c00;
    specialMask = __riscv_vmfeq_vf_f16m4_b4(x, RVVMF_EXP_AS_FP16(pinf), vl);
    special = __riscv_vfmerge_vfm_f16m4(x, RVVMF_EXP_AS_FP16(pinf), specialMask, vl);
    // check overflow
    vbool4_t mask = __riscv_vmand_mm_b4(__riscv_vmfgt_vf_f16m4_b4(x, overflowThreshold, vl),
        __riscv_vmflt_vf_f16m4_b4(x, RVVMF_EXP_AS_FP16(pinf), vl), vl);
    special = __riscv_vfmerge_vfm_f16m4(special, RVVMF_EXP_AS_FP16(pinf), mask, vl);
    specialMask = __riscv_vmor_mm_b4(specialMask, mask, vl);  
    if (__riscv_vcpop_m_b4(mask, vl)) RVVMF_EXP_CALL_FE_OVERFLOW();
    // NaNs, overflow, -inf -- automatically
    x = __riscv_vfmerge_vfm_f16m4(x, ZERO_F16, specialMask, vl);
}

forceinline void do_exp_argument_reduction_hl_f16m4(const vfloat16m4_t& x,
    vfloat16m4_t& yh, vfloat16m4_t& yl, vuint16m4_t& ei, vuint16m4_t& fi, size_t vl)
{
    vfloat16m4_t vmagicConst1 = __riscv_vfmv_v_f_f16m4(MAGIC_CONST_1_F16, vl);
    vfloat16m4_t h = __riscv_vfmadd_vf_f16m4(x, INV_LOG2_2K_F16, vmagicConst1, vl);
    vuint16m4_t hi = __riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(h), MASK_HI_BIT_F16, vl);
    fi = __riscv_vand_vx_u16m4(hi, MASK_FI_BIT_F16, vl);
    ei = __riscv_vsrl_vx_u16m4(hi, TABLE_SIZE_DEG_F16, vl);
    h = __riscv_vfsub_vv_f16m4(h, vmagicConst1, vl);
    fma12_ver2p2_vf_f16m4(h, M_LOG2_2K_L_F16, __riscv_vfmadd_vf_f16m4(h, M_LOG2_2K_H_F16, x, vl), yh, yl, vl);
    yl = __riscv_vfmadd_vf_f16m4(h, M_LOG2_2K_LL_F16, yl, vl);
    fast_2_sum_vv_f16m4(yh, yl, yh, yl, vl);
}

forceinline void do_exp2_argument_reduction_f16m4(const vfloat16m4_t& x, vfloat16m4_t& y,
    vuint16m4_t& ei, vuint16m4_t& fi, size_t vl)  // exact
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
    index = __riscv_vmul_vx_u16m4(index, uint16_t(sizeof(FLOAT16_T)), vl);
    th = __riscv_vloxei16_v_f16m4(LOOK_UP_TABLE_HIGH_F16, index, vl);
    tl = __riscv_vloxei16_v_f16m4(LOOK_UP_TABLE_LOW_F16, index, vl);
}

forceinline void calculate_exp_polynom_hl_f16m4(const vfloat16m4_t& yh, const vfloat16m4_t& yl, vfloat16m4_t& ph, vfloat16m4_t& pl, size_t vl)
{
    vfloat16m4_t sqryh = __riscv_vfmul_vv_f16m4(yh, yh, vl);
    vfloat16m4_t r = calc_polynom_deg_1_f16m4(yh, EXP_POL_COEFF_2_F16, EXP_POL_COEFF_3_F16, vl); 
    fma12_vv_f16m4(sqryh, r, yh, ph, pl, vl);
    pl = __riscv_vfadd_vv_f16m4(pl, yl, vl);
}

forceinline void calculate_exp2_polynom_hl12_f16m4(const vfloat16m4_t& yh, vfloat16m4_t& ph, vfloat16m4_t& pl, size_t vl)
{
    vfloat16m4_t sqryh = __riscv_vfmul_vv_f16m4(yh, yh, vl);
    vfloat16m4_t r = __riscv_vfmv_v_f_f16m4(EXP2_POL_COEFF_2_F16, vl); 
    fma12_ver2p1_vf_f16m4(yh, EXP2_POL_COEFF_1_F16, __riscv_vfmul_vv_f16m4(sqryh, r, vl), ph, pl, vl);
}

forceinline void update_exponent_f16m4(const vuint16m4_t& ei, vfloat16m4_t& res, size_t vl)
{
    res = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vadd_vv_u16m4(
        __riscv_vreinterpret_v_f16m4_u16m4(res), __riscv_vsll_vx_u16m4(ei, (size_t)10, vl), vl));
}

forceinline void update_exponent_with_subnormal_f16m4(const FLOAT16_T& subnormalThreshold, const vfloat16m4_t& x,
    const vuint16m4_t& ei, vfloat16m4_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    uint16_t ninf = 0xfc00;
    vbool4_t subnormalMask = __riscv_vmand_mm_b4(__riscv_vmfgt_vf_f16m4_b4(x, RVVMF_EXP_AS_FP16(ninf), vl),
        __riscv_vmflt_vf_f16m4_b4(x, subnormalThreshold, vl), vl);
    if (__riscv_vcpop_m_b4(subnormalMask, vl)) RVVMF_EXP_CALL_FE_UNDERFLOW();  // FE_UNDERFLOW
    
    vuint16m4_t shiftNum = __riscv_vreinterpret_v_i16m4_u16m4(__riscv_vneg_v_i16m4(__riscv_vreinterpret_v_u16m4_i16m4(ei), vl));
    shiftNum = __riscv_vadd_vx_u16m4(__riscv_vand_vx_u16m4(shiftNum, (uint16_t)0x003f, vl), (uint16_t)1, vl);
    shiftNum = __riscv_vsll_vx_u16m4(shiftNum, (size_t)10, vl);
    vfloat16m4_t subnormalRes = __riscv_vfadd_vv_f16m4(res, __riscv_vreinterpret_v_u16m4_f16m4(shiftNum), vl);
    subnormalRes = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vx_u16m4(
        __riscv_vreinterpret_v_f16m4_u16m4(subnormalRes), (uint16_t)0x83ff, vl));
#endif

    update_exponent_f16m4(ei, res, vl);
    
#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m4(res, subnormalRes, subnormalMask, vl);
#endif   
}

forceinline void reconstruct_exp_hl_hl_f16m4(const vfloat16m4_t& x, const vuint16m4_t& ei, const vfloat16m4_t& th, const vfloat16m4_t& tl,
    const vfloat16m4_t& pm4h, const vfloat16m4_t& pm4l, vfloat16m4_t& res, const FLOAT16_T& subnormalThreshold, size_t vl)
{
    vfloat16m4_t sh, sl;
    fast_2_sum_fv_f16m4(ONE_F16, pm4h, sh, sl, vl);
    sl = __riscv_vfadd_vv_f16m4(sl, pm4l, vl);
    mul21_vv_f16m4(th, tl, sh, sl, res, vl);
    update_exponent_with_subnormal_f16m4(subnormalThreshold, x, ei, res, vl);
}

forceinline void reconstruct_expm1_f16m4(const vfloat16m4_t& th, const vfloat16m4_t& tl, 
    const vfloat16m4_t& pm4h, const vfloat16m4_t& pm4l, const vuint16m4_t& ei, vfloat16m4_t& res, size_t vl)
{        
    vfloat16m4_t rh, rl, sh, sl, sl1;
    fast_2_sum_fv_f16m4(ONE_F16, pm4h, rh, rl, vl);
    rl = __riscv_vfadd_vv_f16m4(rl, pm4l, vl);
    mul22_vv_f16m4(th, tl, rh, rl, sh, sl, vl);
    
    vuint16m4_t power = __riscv_vsll_vx_u16m4(ei, (size_t)10, vl);
    sh = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vadd_vv_u16m4(
        __riscv_vreinterpret_v_f16m4_u16m4(sh), power, vl));   
    vuint16m4_t power2 = __riscv_vsll_vx_u16m4(__riscv_vadd_vx_u16m4(ei, (uint16_t)15, vl), (size_t)10, vl);  
    sl1 = __riscv_vfmul_vv_f16m4(__riscv_vreinterpret_v_u16m4_f16m4(power2), sl, vl);
    vbool4_t slZeroMask = __riscv_vmfeq_vf_f16m4_b4(sl, ZERO_F16, vl);
    sl = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vadd_vv_u16m4(
        __riscv_vreinterpret_v_f16m4_u16m4(sl), power, vl));
    sl = __riscv_vfmerge_vfm_f16m4(sl, ZERO_F16, slZeroMask, vl);
    vbool4_t infPowerMask = __riscv_vmseq_vx_u16m4_b4(power2, (uint16_t)0x7c00, vl);
    sl = __riscv_vmerge_vvm_f16m4(sl1, sl, infPowerMask, vl);
    
    vbool4_t sortMask = __riscv_vmsgtu_vx_u16m4_b4(__riscv_vand_vx_u16m4(__riscv_vreinterpret_v_f16m4_u16m4(sh),
        (uint16_t)0x7c00, vl), (uint16_t)0x3c00, vl);
    vfloat16m4_t maxs = __riscv_vfmerge_vfm_f16m4(sh, EXPM1_UNDERFLOW_VALUE_F16, __riscv_vmnot_m_b4(sortMask, vl), vl);   
    vfloat16m4_t mins = __riscv_vfmerge_vfm_f16m4(sh, EXPM1_UNDERFLOW_VALUE_F16, sortMask, vl);
    fast_2_sum_vv_f16m4(maxs, mins, rh, rl, vl);
    
    res = __riscv_vfadd_vv_f16m4(rh, __riscv_vfadd_vv_f16m4(sl, rl, vl), vl);
}

forceinline void update_underflow_f16m4(const vfloat16m4_t& x, vfloat16m4_t& res,
    const FLOAT16_T& underflowThreshold, const FLOAT16_T& underflowValue, size_t vl)
{
    vbool4_t underflowMask = __riscv_vmflt_vf_f16m4_b4(x, underflowThreshold, vl);
    res = __riscv_vfmerge_vfm_f16m4(res, underflowValue, underflowMask, vl);
}

forceinline void set_pos_sign_f16m4(vfloat16m4_t& res, size_t vl)
{
    uint16_t signMask = 0x7fff;
    res = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vx_u16m4(
        __riscv_vreinterpret_v_f16m4_u16m4(res), signMask, vl));
}

forceinline void set_sign_f16m4(const vfloat16m4_t& x, vfloat16m4_t& res, size_t vl)
{
    uint16_t signMask = 0x7fff;
    res = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vx_u16m4(
        __riscv_vreinterpret_v_f16m4_u16m4(res), signMask, vl));
    res = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vor_vv_u16m4(__riscv_vand_vx_u16m4(
        __riscv_vreinterpret_v_f16m4_u16m4(x), ~signMask, vl), __riscv_vreinterpret_v_f16m4_u16m4(res), vl));
}

forceinline void process_linear_f16m4(const vfloat16m4_t& x, vfloat16m4_t& res, size_t vl)
{
    uint16_t signMask = 0x7fff;
    vfloat16m4_t xabs = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vand_vx_u16m4(
        __riscv_vreinterpret_v_f16m4_u16m4(x), signMask, vl));
    vbool4_t linearMask = __riscv_vmflt_vf_f16m4_b4(xabs, EXPM1_LINEAR_THRESHOLD_F16, vl);
    res = __riscv_vmerge_vvm_f16m4(res, x, linearMask, vl);
}
