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
 *   File:  dexp.inl                                     *
 *   Contains: helper built-in functions for exp, exp2   *
 *             and expm1 functions (float64_t)           *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#include "exp_utilities.inl"

const double ZERO_F64 = 0.0;
const double ONE_F64 = 1.0;

const double EXP_EXPM1_OVERFLOW_THRESHOLD_F64 = 0x1.62e42fefa39efp9;
const double EXP2_EXP2M1_OVERFLOW_THRESHOLD_F64 = 0x1.fffffffffffffp9;
const double EXP_SUBNORMAL_THRESHOLD_F64 = -0x1.6232bdd7abcd2p9;
const double EXP2_SUBNORMAL_THRESHOLD_F64 = -0x1.ffp9;
const double EXP_ZERO_THRESHOLD_F64 = -0x1.74910d52d3051p9;
const double EXP2_ZERO_THRESHOLD_F64 = -0x1.0cbffffffffffp10;
const double EXP_UNDERFLOW_VALUE_F64 = 0.0;
const double EXPM1_UNDERFLOW_THRESHOLD_F64 = -0x1.2b708872320e1p5;
const double EXPM1_LINEAR_THRESHOLD_F64 = 0x1.6a09e667f3bcdp-53;
const double EXPM1_UNDERFLOW_VALUE_F64 = -1.0;

const size_t TABLE_SIZE_DEG_F64 = 6;
const double EXP2_TABLE_SIZE_DEG_F64 = 0x1p6;
const double M_EXP2_M_TABLE_SIZE_DEG_F64 = -0x1p-6;
const uint64_t MASK_FI_BIT_F64 = 0x000000000000003f;
const uint64_t MASK_HI_BIT_F64 = 0x000000000003ffff;
const double MAGIC_CONST_1_F64 = 6755399441055744.0;
const double INV_LOG2_2K_F64 = 0x1.71547652b82fep6;
const double M_LOG2_2K_H_F64 = -0x1.62e42fefap-7;
const double M_LOG2_2K_L_F64 = -0x1.cf79abc9e3b3ap-46;
const double M_LOG2_2K_LL_F64 = -0x1.ff0342542fc33p-100;

static const double LOOK_UP_TABLE_HIGH_F64[64] = {
    0x1p0, 0x1.02c9a3e778061p0, 0x1.059b0d3158574p0, 0x1.0874518759bc8p0,
    0x1.0b5586cf9890fp0, 0x1.0e3ec32d3d1a2p0, 0x1.11301d0125b51p0, 0x1.1429aaea92dep0,
    0x1.172b83c7d517bp0, 0x1.1a35beb6fcb75p0, 0x1.1d4873168b9aap0, 0x1.2063b88628cd6p0,
    0x1.2387a6e756238p0, 0x1.26b4565e27cddp0, 0x1.29e9df51fdee1p0, 0x1.2d285a6e4030bp0,
    0x1.306fe0a31b715p0, 0x1.33c08b26416ffp0, 0x1.371a7373aa9cbp0, 0x1.3a7db34e59ff7p0,
    0x1.3dea64c123422p0, 0x1.4160a21f72e2ap0, 0x1.44e086061892dp0, 0x1.486a2b5c13cdp0,
    0x1.4bfdad5362a27p0, 0x1.4f9b2769d2ca7p0, 0x1.5342b569d4f82p0, 0x1.56f4736b527dap0,
    0x1.5ab07dd485429p0, 0x1.5e76f15ad2148p0, 0x1.6247eb03a5585p0, 0x1.6623882552225p0,
    0x1.6a09e667f3bcdp0, 0x1.6dfb23c651a2fp0, 0x1.71f75e8ec5f74p0, 0x1.75feb564267c9p0,
    0x1.7a11473eb0187p0, 0x1.7e2f336cf4e62p0, 0x1.82589994cce13p0, 0x1.868d99b4492edp0,
    0x1.8ace5422aa0dbp0, 0x1.8f1ae99157736p0, 0x1.93737b0cdc5e5p0, 0x1.97d829fde4e5p0,
    0x1.9c49182a3f09p0, 0x1.a0c667b5de565p0, 0x1.a5503b23e255dp0, 0x1.a9e6b5579fdbfp0,
    0x1.ae89f995ad3adp0, 0x1.b33a2b84f15fbp0, 0x1.b7f76f2fb5e47p0, 0x1.bcc1e904bc1d2p0,
    0x1.c199bdd85529cp0, 0x1.c67f12e57d14bp0, 0x1.cb720dcef9069p0, 0x1.d072d4a07897cp0,
    0x1.d5818dcfba487p0, 0x1.da9e603db3285p0, 0x1.dfc97337b9b5fp0, 0x1.e502ee78b3ff6p0,
    0x1.ea4afa2a490dap0, 0x1.efa1bee615a27p0, 0x1.f50765b6e454p0, 0x1.fa7c1819e90d8p0 
};
static const double LOOK_UP_TABLE_LOW_F64[64] = {
    0, -0x1.19083535b085dp-56, 0x1.d73e2a475b465p-55, 0x1.186be4bb284ffp-57,
    0x1.8a62e4adc610bp-54, 0x1.03a1727c57b53p-59, -0x1.6c51039449b3ap-54, -0x1.32fbf9af1369ep-54,
    -0x1.19041b9d78a76p-55, 0x1.e5b4c7b4968e4p-55, 0x1.e016e00a2643cp-54, 0x1.dc775814a8495p-55,
    0x1.9b07eb6c70573p-54, 0x1.2bd339940e9d9p-55, 0x1.612e8afad1255p-55, 0x1.0024754db41d5p-54,
    0x1.6f46ad23182e4p-55, 0x1.32721843659a6p-54, -0x1.63aeabf42eae2p-54, -0x1.5e436d661f5e3p-56,
    0x1.ada0911f09ebcp-55, -0x1.ef3691c309278p-58, 0x1.89b7a04ef80dp-59, 0x1.3c1a3b69062fp-56,
    0x1.d4397afec42e2p-56, -0x1.4b309d25957e3p-54, -0x1.07abe1db13cadp-55, 0x1.9bb2c011d93adp-54,
    0x1.6324c054647adp-54, 0x1.ba6f93080e65ep-54, -0x1.383c17e40b497p-54, -0x1.bb60987591c34p-54,
    -0x1.bdd3413b26456p-54, -0x1.bbe3a683c88abp-57, -0x1.16e4786887a99p-55, -0x1.0245957316dd3p-54,
    -0x1.41577ee04992fp-55, 0x1.05d02ba15797ep-56, -0x1.d4c1dd41532d8p-54, -0x1.fc6f89bd4f6bap-54,
    0x1.6e9f156864b27p-54, 0x1.5cc13a2e3976cp-55, -0x1.75fc781b57ebcp-57, -0x1.d185b7c1b85d1p-54,
    0x1.c7c46b071f2bep-56, -0x1.359495d1cd533p-54, -0x1.d2f6edb8d41e1p-54, 0x1.0fac90ef7fd31p-54,
    0x1.7a1cd345dcc81p-54, -0x1.2805e3084d708p-57, -0x1.5584f7e54ac3bp-56, 0x1.23dd07a2d9e84p-55,
    0x1.11065895048ddp-55, 0x1.2884dff483cadp-54, 0x1.503cbd1e949dbp-56, -0x1.cbc3743797a9cp-54,
    0x1.2ed02d75b3707p-55, 0x1.c2300696db532p-54, -0x1.1a5cd4f184b5cp-54, 0x1.39e8980a9cc8fp-55,
    -0x1.e9c23179c2893p-54, 0x1.dc7f486a4b6bp-54, 0x1.9d3e12dd8a18bp-54, 0x1.74853f3a5931ep-55
}; 

const double EXP_POL_COEFF_2_F64 = 0x1p-1;
const double EXP_POL_COEFF_3_F64 = 0x1.55555555548bap-3;
const double EXP_POL_COEFF_4_F64 = 0x1.5555555555abcp-5;
const double EXP_POL_COEFF_5_F64 = 0x1.111123cf189c3p-7;
const double EXP_POL_COEFF_6_F64 = 0x1.6c16c6a1679b3p-10;

const double EXP2_POL_COEFF_1_F64 = 0x1.62e42fefa39efp-1;
const double EXP2_POL_COEFF_2_F64 = 0x1.ebfbdff82c58dp-3;
const double EXP2_POL_COEFF_3_F64 = 0x1.c6b08d7073c6bp-5;
const double EXP2_POL_COEFF_4_F64 = 0x1.3b2ab6fcef62fp-7;
const double EXP2_POL_COEFF_5_F64 = 0x1.5d872202a7a6ep-10;
const double EXP2_POL_COEFF_6_F64 = 0x1.42fa95beb52fbp-13;

// ---------------------------- m1 ----------------------------

forceinline void check_special_cases_f64m1(vfloat64m1_t& x, vfloat64m1_t& special, vbool64_t& specialMask,
    const double& overflowThreshold, size_t vl)
{ 
    // check +inf
    uint64_t pinf = 0x7ff0000000000000;
    specialMask = __riscv_vmfeq_vf_f64m1_b64(x, RVVMF_EXP_AS_FP64(pinf), vl);
    special = __riscv_vfmerge_vfm_f64m1(x, RVVMF_EXP_AS_FP64(pinf), specialMask, vl);
    // check overflow
    vbool64_t mask = __riscv_vmand_mm_b64(__riscv_vmfgt_vf_f64m1_b64(x, overflowThreshold, vl),
        __riscv_vmflt_vf_f64m1_b64(x, RVVMF_EXP_AS_FP64(pinf), vl), vl);
    special = __riscv_vfmerge_vfm_f64m1(special, RVVMF_EXP_AS_FP64(pinf), mask, vl);
    specialMask = __riscv_vmor_mm_b64(specialMask, mask, vl);  
    if (__riscv_vcpop_m_b64(mask, vl)) RVVMF_EXP_CALL_FE_OVERFLOW(); 
    // NaNs, -inf -- automatically
    x = __riscv_vfmerge_vfm_f64m1(x, ZERO_F64, specialMask, vl);
}

forceinline void do_exp_argument_reduction_h_f64m1(const vfloat64m1_t& x,
    vfloat64m1_t& yh, vuint64m1_t& ei, vuint64m1_t& fi, size_t vl)
{
    vfloat64m1_t vmagicConst1 = __riscv_vfmv_v_f_f64m1(MAGIC_CONST_1_F64, vl);
    vfloat64m1_t h = __riscv_vfmadd_vf_f64m1(x, INV_LOG2_2K_F64, vmagicConst1, vl);
    vuint64m1_t hi = __riscv_vand_vx_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(h), MASK_HI_BIT_F64, vl);
    fi = __riscv_vand_vx_u64m1(hi, MASK_FI_BIT_F64, vl);
    ei = __riscv_vsrl_vx_u64m1(hi, TABLE_SIZE_DEG_F64, vl);
    h = __riscv_vfsub_vv_f64m1(h, vmagicConst1, vl);
    yh = __riscv_vfmadd_vf_f64m1(h, M_LOG2_2K_L_F64, __riscv_vfmadd_vf_f64m1(h, M_LOG2_2K_H_F64, x, vl), vl);
}

forceinline void do_exp2_argument_reduction_f64m1(const vfloat64m1_t& x, vfloat64m1_t& y,
    vuint64m1_t& ei, vuint64m1_t& fi, size_t vl)  // exact
{
    vfloat64m1_t vmagicConst1 = __riscv_vfmv_v_f_f64m1(MAGIC_CONST_1_F64, vl);
    vfloat64m1_t h = __riscv_vfmadd_vf_f64m1(x, EXP2_TABLE_SIZE_DEG_F64, vmagicConst1, vl);
    vuint64m1_t hi = __riscv_vand_vx_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(h), MASK_HI_BIT_F64, vl);
    fi = __riscv_vand_vx_u64m1(hi, MASK_FI_BIT_F64, vl);
    ei = __riscv_vsrl_vx_u64m1(hi, TABLE_SIZE_DEG_F64, vl);
    h = __riscv_vfsub_vv_f64m1(h, vmagicConst1, vl);
    y = __riscv_vfmadd_vf_f64m1(h, M_EXP2_M_TABLE_SIZE_DEG_F64, x, vl);
}

forceinline void get_table_values_hl_f64m1(
    vuint64m1_t& index, vfloat64m1_t& th, vfloat64m1_t& tl, size_t vl)
{
    index = __riscv_vmul_vx_u64m1(index, uint64_t(sizeof(double)), vl);
    th = __riscv_vloxei64_v_f64m1(LOOK_UP_TABLE_HIGH_F64, index, vl);
    tl = __riscv_vloxei64_v_f64m1(LOOK_UP_TABLE_LOW_F64, index, vl);
}

forceinline void calculate_exp_polynom_hl12_f64m1(const vfloat64m1_t& yh, vfloat64m1_t& ph, vfloat64m1_t& pl, size_t vl)
{
    vfloat64m1_t sqryh = __riscv_vfmul_vv_f64m1(yh, yh, vl);
    vfloat64m1_t r = calc_polynom_deg_4_parallel_f64m1(yh, sqryh, EXP_POL_COEFF_2_F64, EXP_POL_COEFF_3_F64,
        EXP_POL_COEFF_4_F64, EXP_POL_COEFF_5_F64, EXP_POL_COEFF_6_F64, vl);        
    fma12_vv_f64m1(sqryh, r, yh, ph, pl, vl);
}

forceinline void calculate_exp2_polynom_hl12_f64m1(const vfloat64m1_t& yh, vfloat64m1_t& ph, vfloat64m1_t& pl, size_t vl)
{
    vfloat64m1_t sqryh = __riscv_vfmul_vv_f64m1(yh, yh, vl);
    vfloat64m1_t r = calc_polynom_deg_4_parallel_f64m1(yh, sqryh, EXP2_POL_COEFF_2_F64, EXP2_POL_COEFF_3_F64,
        EXP2_POL_COEFF_4_F64, EXP2_POL_COEFF_5_F64, EXP2_POL_COEFF_6_F64, vl); 
    fma12_vf_f64m1(yh, EXP2_POL_COEFF_1_F64, __riscv_vfmul_vv_f64m1(sqryh, r, vl), ph, pl, vl);
}

forceinline void update_exponent_f64m1(const vuint64m1_t& ei, vfloat64m1_t& res, size_t vl)
{
    res = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vadd_vv_u64m1(
        __riscv_vreinterpret_v_f64m1_u64m1(res), __riscv_vsll_vx_u64m1(ei, (size_t)52, vl), vl));
}

forceinline void update_exponent_with_subnormal_f64m1(const double& subnormalThreshold, const vfloat64m1_t& x,
    const vuint64m1_t& ei, vfloat64m1_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    uint64_t ninf = 0xfff0000000000000;
    vbool64_t subnormalMask = __riscv_vmand_mm_b64(__riscv_vmfgt_vf_f64m1_b64(x, RVVMF_EXP_AS_FP64(ninf), vl),
        __riscv_vmflt_vf_f64m1_b64(x, subnormalThreshold, vl), vl);
    if (__riscv_vcpop_m_b64(subnormalMask, vl)) RVVMF_EXP_CALL_FE_UNDERFLOW();  // FE_UNDERFLOW
    
    vuint64m1_t shiftNum = __riscv_vreinterpret_v_i64m1_u64m1(__riscv_vneg_v_i64m1(__riscv_vreinterpret_v_u64m1_i64m1(ei), vl));
    shiftNum = __riscv_vand_vx_u64m1(__riscv_vadd_vx_u64m1(shiftNum, (uint64_t)1, vl), (uint64_t)0x0000000000000fff, vl);
    shiftNum = __riscv_vsll_vx_u64m1(shiftNum, (size_t)52, vl);
    vfloat64m1_t subnormalRes = __riscv_vfadd_vv_f64m1(res, __riscv_vreinterpret_v_u64m1_f64m1(shiftNum), vl);
    subnormalRes = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vx_u64m1(
        __riscv_vreinterpret_v_f64m1_u64m1(subnormalRes), (uint64_t)0x800fffffffffffff, vl));
#endif

    update_exponent_f64m1(ei, res, vl);
    
#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m1(res, subnormalRes, subnormalMask, vl);  
#endif
}

forceinline void reconstruct_exp_hl_hl_f64m1(const vfloat64m1_t& x, const vuint64m1_t& ei, const vfloat64m1_t& th, const vfloat64m1_t& tl,
    const vfloat64m1_t& pm1h, const vfloat64m1_t& pm1l, vfloat64m1_t& res, const double& subnormalThreshold, size_t vl)
{
    vfloat64m1_t sh, sl;
    fast_2_sum_fv_f64m1(ONE_F64, pm1h, sh, sl, vl);
    sl = __riscv_vfadd_vv_f64m1(sl, pm1l, vl);
    mul21_vv_f64m1(th, tl, sh, sl, res, vl);
    update_exponent_with_subnormal_f64m1(subnormalThreshold, x, ei, res, vl);
}

forceinline void reconstruct_expm1_f64m1(const vfloat64m1_t& th, const vfloat64m1_t& tl, 
    const vfloat64m1_t& pm1h, const vfloat64m1_t& pm1l, const vuint64m1_t& ei, vfloat64m1_t& res, size_t vl)
{        
    vfloat64m1_t rh, rl, sh, sl;
    fast_2_sum_fv_f64m1(ONE_F64, pm1h, rh, rl, vl);
    rl = __riscv_vfadd_vv_f64m1(rl, pm1l, vl);
    mul22_vv_f64m1(th, tl, rh, rl, sh, sl, vl);
    
    vuint64m1_t power = __riscv_vsll_vx_u64m1(ei, (size_t)52, vl);
    sh = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vadd_vv_u64m1(
        __riscv_vreinterpret_v_f64m1_u64m1(sh), power, vl));   
    vbool64_t slZeroMask = __riscv_vmfeq_vf_f64m1_b64(sl, ZERO_F64, vl);
    sl = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vadd_vv_u64m1(
        __riscv_vreinterpret_v_f64m1_u64m1(sl), power, vl));
    sl = __riscv_vfmerge_vfm_f64m1(sl, ZERO_F64, slZeroMask, vl);
    
    vbool64_t sortMask = __riscv_vmsgtu_vx_u64m1_b64(__riscv_vand_vx_u64m1(
        __riscv_vreinterpret_v_f64m1_u64m1(sh), (uint64_t)0x7ff0000000000000, vl), (uint64_t)0x3ff0000000000000, vl);
    vfloat64m1_t maxs = __riscv_vfmerge_vfm_f64m1(sh, EXPM1_UNDERFLOW_VALUE_F64, __riscv_vmnot_m_b64(sortMask, vl), vl);   
    vfloat64m1_t mins = __riscv_vfmerge_vfm_f64m1(sh, EXPM1_UNDERFLOW_VALUE_F64, sortMask, vl);
    fast_2_sum_vv_f64m1(maxs, mins, rh, rl, vl);
    
    res = __riscv_vfadd_vv_f64m1(rh, __riscv_vfadd_vv_f64m1(sl, rl, vl), vl);
}

forceinline void update_underflow_f64m1(const vfloat64m1_t& x, vfloat64m1_t& res,
    const double& underflowThreshold, const double& underflowValue, size_t vl)
{
    vbool64_t underflowMask = __riscv_vmflt_vf_f64m1_b64(x, underflowThreshold, vl);
    res = __riscv_vfmerge_vfm_f64m1(res, underflowValue, underflowMask, vl);
}

forceinline void set_sign_f64m1(const vfloat64m1_t& x, vfloat64m1_t& res, size_t vl)
{
    uint64_t signMask = 0x7fffffffffffffff;
    res = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vx_u64m1(
        __riscv_vreinterpret_v_f64m1_u64m1(res), signMask, vl));
    res = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vor_vv_u64m1(__riscv_vand_vx_u64m1(
        __riscv_vreinterpret_v_f64m1_u64m1(x), ~signMask, vl), __riscv_vreinterpret_v_f64m1_u64m1(res), vl));
}

forceinline void process_linear_f64m1(const vfloat64m1_t& x, vfloat64m1_t& res, size_t vl)
{
    uint64_t signMask = 0x7fffffffffffffff;
    vfloat64m1_t xabs = __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vand_vx_u64m1(
        __riscv_vreinterpret_v_f64m1_u64m1(x), signMask, vl));
    vbool64_t linearMask = __riscv_vmflt_vf_f64m1_b64(xabs, EXPM1_LINEAR_THRESHOLD_F64, vl);
    res = __riscv_vmerge_vvm_f64m1(res, x, linearMask, vl);
}

// ---------------------------- m2 ----------------------------

forceinline void check_special_cases_f64m2(vfloat64m2_t& x, vfloat64m2_t& special, vbool32_t& specialMask,
    const double& overflowThreshold, size_t vl)
{ 
    // check +inf
    uint64_t pinf = 0x7ff0000000000000;
    specialMask = __riscv_vmfeq_vf_f64m2_b32(x, RVVMF_EXP_AS_FP64(pinf), vl);
    special = __riscv_vfmerge_vfm_f64m2(x, RVVMF_EXP_AS_FP64(pinf), specialMask, vl);
    // check overflow
    vbool32_t mask = __riscv_vmand_mm_b32(__riscv_vmfgt_vf_f64m2_b32(x, overflowThreshold, vl),
        __riscv_vmflt_vf_f64m2_b32(x, RVVMF_EXP_AS_FP64(pinf), vl), vl);
    special = __riscv_vfmerge_vfm_f64m2(special, RVVMF_EXP_AS_FP64(pinf), mask, vl);
    specialMask = __riscv_vmor_mm_b32(specialMask, mask, vl);  
    if (__riscv_vcpop_m_b32(mask, vl)) RVVMF_EXP_CALL_FE_OVERFLOW(); 
    // NaNs, -inf -- automatically
    x = __riscv_vfmerge_vfm_f64m2(x, ZERO_F64, specialMask, vl);
}

forceinline void do_exp_argument_reduction_h_f64m2(const vfloat64m2_t& x,
    vfloat64m2_t& yh, vuint64m2_t& ei, vuint64m2_t& fi, size_t vl)
{
    vfloat64m2_t vmagicConst1 = __riscv_vfmv_v_f_f64m2(MAGIC_CONST_1_F64, vl);
    vfloat64m2_t h = __riscv_vfmadd_vf_f64m2(x, INV_LOG2_2K_F64, vmagicConst1, vl);
    vuint64m2_t hi = __riscv_vand_vx_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(h), MASK_HI_BIT_F64, vl);
    fi = __riscv_vand_vx_u64m2(hi, MASK_FI_BIT_F64, vl);
    ei = __riscv_vsrl_vx_u64m2(hi, TABLE_SIZE_DEG_F64, vl);
    h = __riscv_vfsub_vv_f64m2(h, vmagicConst1, vl);
    yh = __riscv_vfmadd_vf_f64m2(h, M_LOG2_2K_L_F64, __riscv_vfmadd_vf_f64m2(h, M_LOG2_2K_H_F64, x, vl), vl);
}

forceinline void do_exp2_argument_reduction_f64m2(const vfloat64m2_t& x, vfloat64m2_t& y,
    vuint64m2_t& ei, vuint64m2_t& fi, size_t vl)  // exact
{
    vfloat64m2_t vmagicConst1 = __riscv_vfmv_v_f_f64m2(MAGIC_CONST_1_F64, vl);   
    vfloat64m2_t h = __riscv_vfmadd_vf_f64m2(x, EXP2_TABLE_SIZE_DEG_F64, vmagicConst1, vl);
    vuint64m2_t hi = __riscv_vand_vx_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(h), MASK_HI_BIT_F64, vl);
    fi = __riscv_vand_vx_u64m2(hi, MASK_FI_BIT_F64, vl);
    ei = __riscv_vsrl_vx_u64m2(hi, TABLE_SIZE_DEG_F64, vl);
    h = __riscv_vfsub_vv_f64m2(h, vmagicConst1, vl);
    y = __riscv_vfmadd_vf_f64m2(h, M_EXP2_M_TABLE_SIZE_DEG_F64, x, vl);
}

forceinline void get_table_values_hl_f64m2(
    vuint64m2_t& index, vfloat64m2_t& th, vfloat64m2_t& tl, size_t vl)
{
    index = __riscv_vmul_vx_u64m2(index, uint64_t(sizeof(double)), vl);
    th = __riscv_vloxei64_v_f64m2(LOOK_UP_TABLE_HIGH_F64, index, vl);
    tl = __riscv_vloxei64_v_f64m2(LOOK_UP_TABLE_LOW_F64, index, vl);
}

forceinline void calculate_exp_polynom_hl12_f64m2(const vfloat64m2_t& yh, vfloat64m2_t& ph, vfloat64m2_t& pl, size_t vl)
{
    vfloat64m2_t sqryh = __riscv_vfmul_vv_f64m2(yh, yh, vl);
    vfloat64m2_t r = calc_polynom_deg_4_parallel_f64m2(yh, sqryh, EXP_POL_COEFF_2_F64, EXP_POL_COEFF_3_F64,
        EXP_POL_COEFF_4_F64, EXP_POL_COEFF_5_F64, EXP_POL_COEFF_6_F64, vl);        
    fma12_vv_f64m2(sqryh, r, yh, ph, pl, vl);
}

forceinline void calculate_exp2_polynom_hl12_f64m2(const vfloat64m2_t& yh, vfloat64m2_t& ph, vfloat64m2_t& pl, size_t vl)
{
    vfloat64m2_t sqryh = __riscv_vfmul_vv_f64m2(yh, yh, vl);
    vfloat64m2_t r = calc_polynom_deg_4_parallel_f64m2(yh, sqryh, EXP2_POL_COEFF_2_F64, EXP2_POL_COEFF_3_F64,
        EXP2_POL_COEFF_4_F64, EXP2_POL_COEFF_5_F64, EXP2_POL_COEFF_6_F64, vl); 
    fma12_vf_f64m2(yh, EXP2_POL_COEFF_1_F64, __riscv_vfmul_vv_f64m2(sqryh, r, vl), ph, pl, vl);
}

forceinline void update_exponent_f64m2(const vuint64m2_t& ei, vfloat64m2_t& res, size_t vl)
{
    res = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vadd_vv_u64m2(
        __riscv_vreinterpret_v_f64m2_u64m2(res), __riscv_vsll_vx_u64m2(ei, (size_t)52, vl), vl));
}

forceinline void update_exponent_with_subnormal_f64m2(const double& subnormalThreshold, const vfloat64m2_t& x,
    const vuint64m2_t& ei, vfloat64m2_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    uint64_t ninf = 0xfff0000000000000;
    vbool32_t subnormalMask = __riscv_vmand_mm_b32(__riscv_vmfgt_vf_f64m2_b32(x, RVVMF_EXP_AS_FP64(ninf), vl),
        __riscv_vmflt_vf_f64m2_b32(x, subnormalThreshold, vl), vl);
    if (__riscv_vcpop_m_b32(subnormalMask, vl)) RVVMF_EXP_CALL_FE_UNDERFLOW();  // FE_UNDERFLOW
    
    vuint64m2_t shiftNum = __riscv_vreinterpret_v_i64m2_u64m2(__riscv_vneg_v_i64m2(__riscv_vreinterpret_v_u64m2_i64m2(ei), vl));
    shiftNum = __riscv_vand_vx_u64m2(__riscv_vadd_vx_u64m2(shiftNum, (uint64_t)1, vl), (uint64_t)0x0000000000000fff, vl);
    shiftNum = __riscv_vsll_vx_u64m2(shiftNum, (size_t)52, vl);
    vfloat64m2_t subnormalRes = __riscv_vfadd_vv_f64m2(res, __riscv_vreinterpret_v_u64m2_f64m2(shiftNum), vl);
    subnormalRes = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vx_u64m2(
        __riscv_vreinterpret_v_f64m2_u64m2(subnormalRes), (uint64_t)0x800fffffffffffff, vl));
#endif

    update_exponent_f64m2(ei, res, vl);
    
#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m2(res, subnormalRes, subnormalMask, vl);  
#endif
}

forceinline void reconstruct_exp_hl_hl_f64m2(const vfloat64m2_t& x, const vuint64m2_t& ei, const vfloat64m2_t& th, const vfloat64m2_t& tl,
    const vfloat64m2_t& pm2h, const vfloat64m2_t& pm2l, vfloat64m2_t& res, const double& subnormalThreshold, size_t vl)
{
    vfloat64m2_t sh, sl;
    fast_2_sum_fv_f64m2(ONE_F64, pm2h, sh, sl, vl);
    sl = __riscv_vfadd_vv_f64m2(sl, pm2l, vl);
    mul21_vv_f64m2(th, tl, sh, sl, res, vl);
    update_exponent_with_subnormal_f64m2(subnormalThreshold, x, ei, res, vl);
}

forceinline void reconstruct_expm1_f64m2(const vfloat64m2_t& th, const vfloat64m2_t& tl, 
    const vfloat64m2_t& pm2h, const vfloat64m2_t& pm2l, const vuint64m2_t& ei, vfloat64m2_t& res, size_t vl)
{        
    vfloat64m2_t rh, rl, sh, sl;
    fast_2_sum_fv_f64m2(ONE_F64, pm2h, rh, rl, vl);
    rl = __riscv_vfadd_vv_f64m2(rl, pm2l, vl);
    mul22_vv_f64m2(th, tl, rh, rl, sh, sl, vl);
    
    vuint64m2_t power = __riscv_vsll_vx_u64m2(ei, (size_t)52, vl);
    sh = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vadd_vv_u64m2(
        __riscv_vreinterpret_v_f64m2_u64m2(sh), power, vl));   
    vbool32_t slZeroMask = __riscv_vmfeq_vf_f64m2_b32(sl, ZERO_F64, vl);
    sl = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vadd_vv_u64m2(
        __riscv_vreinterpret_v_f64m2_u64m2(sl), power, vl));
    sl = __riscv_vfmerge_vfm_f64m2(sl, ZERO_F64, slZeroMask, vl);
    
    vbool32_t sortMask = __riscv_vmsgtu_vx_u64m2_b32(__riscv_vand_vx_u64m2(
        __riscv_vreinterpret_v_f64m2_u64m2(sh), (uint64_t)0x7ff0000000000000, vl), (uint64_t)0x3ff0000000000000, vl);
    vfloat64m2_t maxs = __riscv_vfmerge_vfm_f64m2(sh, EXPM1_UNDERFLOW_VALUE_F64, __riscv_vmnot_m_b32(sortMask, vl), vl);   
    vfloat64m2_t mins = __riscv_vfmerge_vfm_f64m2(sh, EXPM1_UNDERFLOW_VALUE_F64, sortMask, vl);
    fast_2_sum_vv_f64m2(maxs, mins, rh, rl, vl);
    
    res = __riscv_vfadd_vv_f64m2(rh, __riscv_vfadd_vv_f64m2(sl, rl, vl), vl);
}

forceinline void update_underflow_f64m2(const vfloat64m2_t& x, vfloat64m2_t& res,
    const double& underflowThreshold, const double& underflowValue, size_t vl)
{
    vbool32_t underflowMask = __riscv_vmflt_vf_f64m2_b32(x, underflowThreshold, vl);
    res = __riscv_vfmerge_vfm_f64m2(res, underflowValue, underflowMask, vl);
}

forceinline void set_sign_f64m2(const vfloat64m2_t& x, vfloat64m2_t& res, size_t vl)
{
    uint64_t signMask = 0x7fffffffffffffff;
    res = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vx_u64m2(
        __riscv_vreinterpret_v_f64m2_u64m2(res), signMask, vl));
    res = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vor_vv_u64m2(__riscv_vand_vx_u64m2(
        __riscv_vreinterpret_v_f64m2_u64m2(x), ~signMask, vl), __riscv_vreinterpret_v_f64m2_u64m2(res), vl));
}

forceinline void process_linear_f64m2(const vfloat64m2_t& x, vfloat64m2_t& res, size_t vl)
{
    uint64_t signMask = 0x7fffffffffffffff;
    vfloat64m2_t xabs = __riscv_vreinterpret_v_u64m2_f64m2(__riscv_vand_vx_u64m2(
        __riscv_vreinterpret_v_f64m2_u64m2(x), signMask, vl));
    vbool32_t linearMask = __riscv_vmflt_vf_f64m2_b32(xabs, EXPM1_LINEAR_THRESHOLD_F64, vl);
    res = __riscv_vmerge_vvm_f64m2(res, x, linearMask, vl);
}

// ---------------------------- m4 ----------------------------

forceinline void check_special_cases_f64m4(vfloat64m4_t& x, vfloat64m4_t& special, vbool16_t& specialMask,
    const double& overflowThreshold, size_t vl)
{ 
    // check +inf
    uint64_t pinf = 0x7ff0000000000000;
    specialMask = __riscv_vmfeq_vf_f64m4_b16(x, RVVMF_EXP_AS_FP64(pinf), vl);
    special = __riscv_vfmerge_vfm_f64m4(x, RVVMF_EXP_AS_FP64(pinf), specialMask, vl);
    // check overflow
    vbool16_t mask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f64m4_b16(x, overflowThreshold, vl),
        __riscv_vmflt_vf_f64m4_b16(x, RVVMF_EXP_AS_FP64(pinf), vl), vl);
    special = __riscv_vfmerge_vfm_f64m4(special, RVVMF_EXP_AS_FP64(pinf), mask, vl);
    specialMask = __riscv_vmor_mm_b16(specialMask, mask, vl);  
    if (__riscv_vcpop_m_b16(mask, vl)) RVVMF_EXP_CALL_FE_OVERFLOW(); 
    // NaNs, -inf -- automatically
    x = __riscv_vfmerge_vfm_f64m4(x, ZERO_F64, specialMask, vl);
}

forceinline void do_exp_argument_reduction_h_f64m4(const vfloat64m4_t& x,
    vfloat64m4_t& yh, vuint64m4_t& ei, vuint64m4_t& fi, size_t vl)
{
    vfloat64m4_t vmagicConst1 = __riscv_vfmv_v_f_f64m4(MAGIC_CONST_1_F64, vl);
    vfloat64m4_t h = __riscv_vfmadd_vf_f64m4(x, INV_LOG2_2K_F64, vmagicConst1, vl);
    vuint64m4_t hi = __riscv_vand_vx_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(h), MASK_HI_BIT_F64, vl);
    fi = __riscv_vand_vx_u64m4(hi, MASK_FI_BIT_F64, vl);
    ei = __riscv_vsrl_vx_u64m4(hi, TABLE_SIZE_DEG_F64, vl);
    h = __riscv_vfsub_vv_f64m4(h, vmagicConst1, vl);
    yh = __riscv_vfmadd_vf_f64m4(h, M_LOG2_2K_L_F64, __riscv_vfmadd_vf_f64m4(h, M_LOG2_2K_H_F64, x, vl), vl);
}

forceinline void do_exp2_argument_reduction_f64m4(const vfloat64m4_t& x, vfloat64m4_t& y,
    vuint64m4_t& ei, vuint64m4_t& fi, size_t vl)  // exact
{
    vfloat64m4_t vmagicConst1 = __riscv_vfmv_v_f_f64m4(MAGIC_CONST_1_F64, vl);
    vfloat64m4_t h = __riscv_vfmadd_vf_f64m4(x, EXP2_TABLE_SIZE_DEG_F64, vmagicConst1, vl);
    vuint64m4_t hi = __riscv_vand_vx_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(h), MASK_HI_BIT_F64, vl);
    fi = __riscv_vand_vx_u64m4(hi, MASK_FI_BIT_F64, vl);
    ei = __riscv_vsrl_vx_u64m4(hi, TABLE_SIZE_DEG_F64, vl);
    h = __riscv_vfsub_vv_f64m4(h, vmagicConst1, vl);
    y = __riscv_vfmadd_vf_f64m4(h, M_EXP2_M_TABLE_SIZE_DEG_F64, x, vl);
}

forceinline void get_table_values_hl_f64m4(
    vuint64m4_t& index, vfloat64m4_t& th, vfloat64m4_t& tl, size_t vl)
{
    index = __riscv_vmul_vx_u64m4(index, uint64_t(sizeof(double)), vl);
    th = __riscv_vloxei64_v_f64m4(LOOK_UP_TABLE_HIGH_F64, index, vl);
    tl = __riscv_vloxei64_v_f64m4(LOOK_UP_TABLE_LOW_F64, index, vl);
}

forceinline void calculate_exp_polynom_hl12_f64m4(const vfloat64m4_t& yh, vfloat64m4_t& ph, vfloat64m4_t& pl, size_t vl)
{
    vfloat64m4_t sqryh = __riscv_vfmul_vv_f64m4(yh, yh, vl);
    vfloat64m4_t r = calc_polynom_deg_4_parallel_f64m4(yh, sqryh, EXP_POL_COEFF_2_F64, EXP_POL_COEFF_3_F64,
        EXP_POL_COEFF_4_F64, EXP_POL_COEFF_5_F64, EXP_POL_COEFF_6_F64, vl);        
    fma12_vv_f64m4(sqryh, r, yh, ph, pl, vl);
}

forceinline void calculate_exp2_polynom_hl12_f64m4(const vfloat64m4_t& yh, vfloat64m4_t& ph, vfloat64m4_t& pl, size_t vl)
{
    vfloat64m4_t sqryh = __riscv_vfmul_vv_f64m4(yh, yh, vl);
    vfloat64m4_t r = calc_polynom_deg_4_parallel_f64m4(yh, sqryh, EXP2_POL_COEFF_2_F64, EXP2_POL_COEFF_3_F64,
        EXP2_POL_COEFF_4_F64, EXP2_POL_COEFF_5_F64, EXP2_POL_COEFF_6_F64, vl); 
    fma12_vf_f64m4(yh, EXP2_POL_COEFF_1_F64, __riscv_vfmul_vv_f64m4(sqryh, r, vl), ph, pl, vl);
}

forceinline void update_exponent_f64m4(const vuint64m4_t& ei, vfloat64m4_t& res, size_t vl)
{
    res = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vadd_vv_u64m4(
        __riscv_vreinterpret_v_f64m4_u64m4(res), __riscv_vsll_vx_u64m4(ei, (size_t)52, vl), vl));
}

forceinline void update_exponent_with_subnormal_f64m4(const double& subnormalThreshold, const vfloat64m4_t& x,
    const vuint64m4_t& ei, vfloat64m4_t& res, size_t vl)
{
#ifndef __FAST_MATH__
    uint64_t ninf = 0xfff0000000000000;
    vbool16_t subnormalMask = __riscv_vmand_mm_b16(__riscv_vmfgt_vf_f64m4_b16(x, RVVMF_EXP_AS_FP64(ninf), vl),
        __riscv_vmflt_vf_f64m4_b16(x, subnormalThreshold, vl), vl);
    if (__riscv_vcpop_m_b16(subnormalMask, vl)) RVVMF_EXP_CALL_FE_UNDERFLOW();  // FE_UNDERFLOW
    
    vuint64m4_t shiftNum = __riscv_vreinterpret_v_i64m4_u64m4(__riscv_vneg_v_i64m4(__riscv_vreinterpret_v_u64m4_i64m4(ei), vl));
    shiftNum = __riscv_vand_vx_u64m4(__riscv_vadd_vx_u64m4(shiftNum, (uint64_t)1, vl), (uint64_t)0x0000000000000fff, vl);
    shiftNum = __riscv_vsll_vx_u64m4(shiftNum, (size_t)52, vl);
    vfloat64m4_t subnormalRes = __riscv_vfadd_vv_f64m4(res, __riscv_vreinterpret_v_u64m4_f64m4(shiftNum), vl);
    subnormalRes = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vand_vx_u64m4(
        __riscv_vreinterpret_v_f64m4_u64m4(subnormalRes), (uint64_t)0x800fffffffffffff, vl));
#endif

    update_exponent_f64m4(ei, res, vl);
    
#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m4(res, subnormalRes, subnormalMask, vl);  
#endif
}

forceinline void reconstruct_exp_hl_hl_f64m4(const vfloat64m4_t& x, const vuint64m4_t& ei, const vfloat64m4_t& th, const vfloat64m4_t& tl,
    const vfloat64m4_t& pm4h, const vfloat64m4_t& pm4l, vfloat64m4_t& res, const double& subnormalThreshold, size_t vl)
{
    vfloat64m4_t sh, sl;
    fast_2_sum_fv_f64m4(ONE_F64, pm4h, sh, sl, vl);
    sl = __riscv_vfadd_vv_f64m4(sl, pm4l, vl);
    mul21_vv_f64m4(th, tl, sh, sl, res, vl);
    update_exponent_with_subnormal_f64m4(subnormalThreshold, x, ei, res, vl);
}

forceinline void reconstruct_expm1_f64m4(const vfloat64m4_t& th, const vfloat64m4_t& tl, 
    const vfloat64m4_t& pm4h, const vfloat64m4_t& pm4l, const vuint64m4_t& ei, vfloat64m4_t& res, size_t vl)
{        
    vfloat64m4_t rh, rl, sh, sl;
    fast_2_sum_fv_f64m4(ONE_F64, pm4h, rh, rl, vl);
    rl = __riscv_vfadd_vv_f64m4(rl, pm4l, vl);
    mul22_vv_f64m4(th, tl, rh, rl, sh, sl, vl);
    
    vuint64m4_t power = __riscv_vsll_vx_u64m4(ei, (size_t)52, vl);
    sh = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vadd_vv_u64m4(
        __riscv_vreinterpret_v_f64m4_u64m4(sh), power, vl));   
    vbool16_t slZeroMask = __riscv_vmfeq_vf_f64m4_b16(sl, ZERO_F64, vl);
    sl = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vadd_vv_u64m4(
        __riscv_vreinterpret_v_f64m4_u64m4(sl), power, vl));
    sl = __riscv_vfmerge_vfm_f64m4(sl, ZERO_F64, slZeroMask, vl);
    
    vbool16_t sortMask = __riscv_vmsgtu_vx_u64m4_b16(__riscv_vand_vx_u64m4(
        __riscv_vreinterpret_v_f64m4_u64m4(sh), (uint64_t)0x7ff0000000000000, vl), (uint64_t)0x3ff0000000000000, vl);
    vfloat64m4_t maxs = __riscv_vfmerge_vfm_f64m4(sh, EXPM1_UNDERFLOW_VALUE_F64, __riscv_vmnot_m_b16(sortMask, vl), vl);   
    vfloat64m4_t mins = __riscv_vfmerge_vfm_f64m4(sh, EXPM1_UNDERFLOW_VALUE_F64, sortMask, vl);
    fast_2_sum_vv_f64m4(maxs, mins, rh, rl, vl);
    
    res = __riscv_vfadd_vv_f64m4(rh, __riscv_vfadd_vv_f64m4(sl, rl, vl), vl);
}

forceinline void update_underflow_f64m4(const vfloat64m4_t& x, vfloat64m4_t& res,
    const double& underflowThreshold, const double& underflowValue, size_t vl)
{
    vbool16_t underflowMask = __riscv_vmflt_vf_f64m4_b16(x, underflowThreshold, vl);
    res = __riscv_vfmerge_vfm_f64m4(res, underflowValue, underflowMask, vl);
}

forceinline void set_sign_f64m4(const vfloat64m4_t& x, vfloat64m4_t& res, size_t vl)
{
    uint64_t signMask = 0x7fffffffffffffff;
    res = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vand_vx_u64m4(
        __riscv_vreinterpret_v_f64m4_u64m4(res), signMask, vl));
    res = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vor_vv_u64m4(__riscv_vand_vx_u64m4(
        __riscv_vreinterpret_v_f64m4_u64m4(x), ~signMask, vl), __riscv_vreinterpret_v_f64m4_u64m4(res), vl));
}

forceinline void process_linear_f64m4(const vfloat64m4_t& x, vfloat64m4_t& res, size_t vl)
{
    uint64_t signMask = 0x7fffffffffffffff;
    vfloat64m4_t xabs = __riscv_vreinterpret_v_u64m4_f64m4(__riscv_vand_vx_u64m4(
        __riscv_vreinterpret_v_f64m4_u64m4(x), signMask, vl));
    vbool16_t linearMask = __riscv_vmflt_vf_f64m4_b16(xabs, EXPM1_LINEAR_THRESHOLD_F64, vl);
    res = __riscv_vmerge_vvm_f64m4(res, x, linearMask, vl);
}
