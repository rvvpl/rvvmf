/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2026 The RVVMF Authors (Egor Korotin, 
 * Vladislav Polozov)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  exp2_components.inl                          *
 *   Contains: internal components of exp2 function      *
 *   modified to implement pow                           *
 *                                                       *
 *   Based on exp2 implementation from src/rvv/exp       *
 *                                                       *
 *                                                       *
 *********************************************************
*/


#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
    #define forceinline __attribute__((always_inline)) inline
#else
    #define forceinline inline
#endif

const double EXP2_POL_COEFF_1_F64 = 0x1.62e42fefa39efp-1;
const double EXP2_POL_COEFF_2_F64 = 0x1.ebfbdff82c58dp-3;
const double EXP2_POL_COEFF_3_F64 = 0x1.c6b08d7073c6bp-5;
const double EXP2_POL_COEFF_4_F64 = 0x1.3b2ab6fcef62fp-7;
const double EXP2_POL_COEFF_5_F64 = 0x1.5d872202a7a6ep-10;
const double EXP2_POL_COEFF_6_F64 = 0x1.42fa95beb52fbp-13;

const float EXP2_POL_COEFF_1_F32 = 0x1.62e43p-1f;
const float EXP2_POL_COEFF_2_F32 = 0x1.ebfbep-3f;
const float EXP2_POL_COEFF_3_F32 = 0x1.c6ae08p-5f;
const float EXP2_POL_COEFF_4_F32 = 0x1.3b27cep-7f;

#define RISCV_POW_EXP2_F64(LMUL, BOOL)   \
forceinline void do_exact_spec_exp_argument_reduction_hl_f64##LMUL(const vfloat64##LMUL##_t& x, vfloat64##LMUL##_t& yh, vuint64##LMUL##_t& ei, vuint64##LMUL##_t& fi, size_t vl)\
{    \
    vfloat64##LMUL##_t vmagicConst1 = __riscv_vfmv_v_f_f64##LMUL(6755399441055744.0, vl);\
    \
    vfloat64##LMUL##_t h = __riscv_vfmadd_vf_f64##LMUL(x, 0x1p6, vmagicConst1, vl);\
    vuint64##LMUL##_t hi = __riscv_vand_vx_u64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(h), 0x000000000003ffff, vl);\
    fi = __riscv_vand_vx_u64##LMUL(hi, 0x000000000000003f, vl);\
    ei = __riscv_vsrl_vx_u64##LMUL(hi, 6, vl);\
    h = __riscv_vfsub_vv_f64##LMUL(h, vmagicConst1, vl);\
\
    yh = __riscv_vfmadd_vf_f64##LMUL(h, -0x1p-6, x, vl);\
}\
\
forceinline void get_table_values_hl_f64##LMUL(vuint64##LMUL##_t& index, vfloat64##LMUL##_t& th, vfloat64##LMUL##_t& tl, size_t vl)\
{\
    index = __riscv_vmul_vx_u64##LMUL(index, uint64_t(sizeof(double)), vl);\
    th = __riscv_vloxei64_v_f64##LMUL(LOOK_UP_TABLE_HIGH_F64, index, vl);\
    tl = __riscv_vloxei64_v_f64##LMUL(LOOK_UP_TABLE_LOW_F64, index, vl);\
}\
\
forceinline void calculate_super_exact_exp2_polynom_hl_f64##LMUL(const vfloat64##LMUL##_t& yh, const vfloat64##LMUL##_t& yl, vfloat64##LMUL##_t& ph, vfloat64##LMUL##_t& pl, size_t vl)\
{\
vfloat64##LMUL##_t r1, r2, r;\
\
  vfloat64##LMUL##_t sqr = __riscv_vfmul_vv_f64##LMUL(yh, yh, vl);\
\
  r1 = __riscv_vfmadd_vf_f64##LMUL(sqr, EXP2_POL_COEFF_6_F64, __riscv_vfmv_v_f_f64##LMUL(EXP2_POL_COEFF_4_F64, vl), vl);\
  r2 = __riscv_vfmadd_vf_f64##LMUL(sqr, EXP2_POL_COEFF_5_F64, __riscv_vfmv_v_f_f64##LMUL(EXP2_POL_COEFF_3_F64, vl), vl);\
  r1 = __riscv_vfmadd_vv_f64##LMUL(sqr, r1, __riscv_vfmv_v_f_f64##LMUL(EXP2_POL_COEFF_2_F64, vl), vl);\
\
  r = __riscv_vfmadd_vv_f64##LMUL(yh, r2, r1, vl);\
    \
  fma12_f64##LMUL(yh, __riscv_vfmv_v_f_f64##LMUL(EXP2_POL_COEFF_1_F64, vl), __riscv_vfmul_vv_f64##LMUL(r, sqr, vl), ph, pl, vl);\
}\
\
forceinline void reconstruct_exp_hl_hl_f64##LMUL(const vfloat64##LMUL##_t& x, const vuint64##LMUL##_t& ei, const vfloat64##LMUL##_t& th, const vfloat64##LMUL##_t& tl,\
    const vfloat64##LMUL##_t& pm1h, const vfloat64##LMUL##_t& pm1l, vfloat64##LMUL##_t& res, size_t vl)\
{\
    vfloat64##LMUL##_t sh, sl;\
    fast_2_sum_f64##LMUL(__riscv_vfmv_v_f_f64##LMUL(1.0, vl), pm1h, sh, sl, vl);\
    sl = __riscv_vfadd_vv_f64##LMUL(sl, pm1l, vl);\
    mul21_f64##LMUL(th, tl, sh, sl, res, vl);\
    update_exponent_f64##LMUL(x, ei, res, vl);\
}\
\
vfloat64##LMUL##_t special_exp_f64##LMUL(vfloat64##LMUL##_t xh, vfloat64##LMUL##_t xl, size_t vl)\
{\
    vfloat64##LMUL##_t res;\
\
    const double subnormalThreshold = -0x1.ffp9;\
    const double zeroThreshold = -0x1.0cbffffffffffp10;\
\
    vfloat64##LMUL##_t yh, yl, zh, zl, th, tl, pm1h, pm1l;\
    vuint64##LMUL##_t ei, fi;\
    \
    do_exact_spec_exp_argument_reduction_hl_f64##LMUL(xh, yh, ei, fi, vl);\
    fast_2_sum_f64##LMUL(yh, xl, yh, yl, vl);\
\
    get_table_values_hl_f64##LMUL(fi, th, tl, vl);\
\
    calculate_super_exact_exp2_polynom_hl_f64##LMUL(yh, yl, pm1h, pm1l, vl);\
    reconstruct_exp_hl_hl_f64##LMUL(xh, ei, th, tl, pm1h, pm1l, res, vl);\
    \
    return res;\
}

#ifndef __FAST_MATH__
#define RISCV_POW_UPDEXP_F64(LMUL, BOOL)   \
forceinline void update_exponent_f64##LMUL(const vfloat64##LMUL##_t& x, const vuint64##LMUL##_t& ei, vfloat64##LMUL##_t& res, size_t vl)\
{\
    vint64##LMUL##_t sign_x = __riscv_vand_vx_i64##LMUL(__riscv_vreinterpret_v_f64##LMUL##_i64##LMUL(x), SIGN_MASK_F64, vl);\
    vuint64##LMUL##_t subDiff = __riscv_vand_vx_u64##LMUL(__riscv_vreinterpret_v_i64##LMUL##_u64##LMUL(__riscv_vsra_vx_i64##LMUL(sign_x, BITS_F64, vl)), EXPADD_F64, vl);\
    \
    vuint64##LMUL##_t pow2ei = __riscv_vand_vx_u64##LMUL(__riscv_vadd_vv_u64##LMUL(__riscv_vadd_vx_u64##LMUL(__riscv_vsll_vx_u64##LMUL(ei, (size_t)52, vl), EXPADD_F64, vl), subDiff, vl), EXP_MASK_F64, vl);\
\
    res = __riscv_vfmul_vv_f64##LMUL(res, __riscv_vreinterpret_v_u64##LMUL##_f64##LMUL(pow2ei), vl);\
    res = __riscv_vfmul_vv_f64##LMUL(res, __riscv_vreinterpret_v_u64##LMUL##_f64##LMUL(__riscv_vrsub_vx_u64##LMUL(subDiff, (uint64_t)0x4000000000000000, vl)), vl);\
}

#else

#define RISCV_POW_UPDEXP_F64(LMUL, BOOL)   \
forceinline void update_exponent_f64##LMUL(const vfloat64##LMUL##_t& x, const vuint64##LMUL##_t& ei, vfloat64##LMUL##_t& res, size_t vl)\
{\
    vbool##BOOL##_t subnormalMask = __riscv_vmflt_vf_f64##LMUL##_b##BOOL(x, 0.0, vl);\
    vuint64##LMUL##_t subDiff = __riscv_vmerge_vxm_u64##LMUL(__riscv_vmv_v_x_u64##LMUL(uint64_t(0), vl), EXPADD_F64, subnormalMask, vl);\
    \
    vuint64##LMUL##_t pow2ei = __riscv_vand_vx_u64##LMUL(__riscv_vadd_vv_u64##LMUL(__riscv_vadd_vx_u64##LMUL(__riscv_vsll_vx_u64##LMUL(ei, (size_t)52, vl), EXPADD_F64, vl), subDiff, vl), EXP_MASK_F64, vl);\
\
    res = __riscv_vfmul_vv_f64##LMUL(res, __riscv_vreinterpret_v_u64##LMUL##_f64##LMUL(pow2ei), vl);\
    res = __riscv_vfmul_vv_f64##LMUL(res, __riscv_vreinterpret_v_u64##LMUL##_f64##LMUL(__riscv_vrsub_vx_u64##LMUL(subDiff, (uint64_t)0x4000000000000000, vl)), vl);\
}

#endif



#define RISCV_POW_EXP2_F32(LMUL, BOOL)\
forceinline void do_exact_spec_exp_argument_reduction_hl_f32##LMUL(const vfloat32##LMUL##_t& x, vfloat32##LMUL##_t& yh, vuint32##LMUL##_t& ei, vuint32##LMUL##_t& fi, size_t vl)\
{\
    vfloat32##LMUL##_t vmagicConst1 = __riscv_vfmv_v_f_f32##LMUL(12582912.0f, vl);\
    vfloat32##LMUL##_t h = __riscv_vfmadd_vf_f32##LMUL(x, 0x1p4, vmagicConst1, vl);\
    vuint32##LMUL##_t hi = __riscv_vand_vx_u32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(h), 0x00001fff, vl);\
    fi = __riscv_vand_vx_u32##LMUL(hi, 0x0000000f, vl);\
    ei = __riscv_vsrl_vx_u32##LMUL(hi, 4, vl);\
    h = __riscv_vfsub_vv_f32##LMUL(h, vmagicConst1, vl);\
\
    yh = __riscv_vfmadd_vf_f32##LMUL(h, -0x1p-4, x, vl);\
}\
\
forceinline void get_table_values_hl_f32##LMUL(vuint32##LMUL##_t& index, vfloat32##LMUL##_t& th, vfloat32##LMUL##_t& tl, size_t vl)\
{\
    index = __riscv_vmul_vx_u32##LMUL(index, uint32_t(sizeof(float)), vl);\
    th = __riscv_vloxei32_v_f32##LMUL(LOOK_UP_TABLE_HIGH_F32, index, vl);\
    tl = __riscv_vloxei32_v_f32##LMUL(LOOK_UP_TABLE_LOW_F32, index, vl);\
}\
\
forceinline void calculate_super_exact_exp2_polynom_hl_f32##LMUL(const vfloat32##LMUL##_t& yh, const vfloat32##LMUL##_t& yl, vfloat32##LMUL##_t& ph, vfloat32##LMUL##_t& pl, size_t vl)\
{\
    vfloat32##LMUL##_t sqryh = __riscv_vfmul_vv_f32##LMUL(yh, yh, vl);\
    vfloat32##LMUL##_t r = __riscv_vfmadd_vv_f32##LMUL(\
					yh, \
					__riscv_vfmadd_vf_f32##LMUL(yh, EXP2_POL_COEFF_4_F32, __riscv_vfmv_v_f_f32##LMUL(EXP2_POL_COEFF_3_F32, vl), vl), \
					__riscv_vfmv_v_f_f32##LMUL(EXP2_POL_COEFF_2_F32, vl), \
					vl);\
    fma12_ver2p1_f32##LMUL(yh, __riscv_vfmv_v_f_f32##LMUL(EXP2_POL_COEFF_1_F32, vl), __riscv_vfmul_vv_f32##LMUL(sqryh, r, vl), ph, pl, vl);\
}\
\
forceinline void reconstruct_exp_hl_hl_f32##LMUL(const vfloat32##LMUL##_t& x, const vuint32##LMUL##_t& ei, const vfloat32##LMUL##_t& th, const vfloat32##LMUL##_t& tl,\
    const vfloat32##LMUL##_t& pm1h, const vfloat32##LMUL##_t& pm1l, vfloat32##LMUL##_t& res, size_t vl)\
{\
    vfloat32##LMUL##_t sh, sl;\
    fast_2_sum_f32##LMUL(__riscv_vfmv_v_f_f32##LMUL(1.0f, vl), pm1h, sh, sl, vl);\
    sl = __riscv_vfadd_vv_f32##LMUL(sl, pm1l, vl);\
    mul21_f32##LMUL(th, tl, sh, sl, res, vl);\
    update_exponent_f32##LMUL(x, ei, res, vl);\
}\
\
vfloat32##LMUL##_t special_exp_f32##LMUL(vfloat32##LMUL##_t xh, vfloat32##LMUL##_t xl, size_t vl)\
{\
    vfloat32##LMUL##_t res, yh, yl, zh, zl, th, tl, pm1h, pm1l;\
    vuint32##LMUL##_t ei, fi;\
        \
    do_exact_spec_exp_argument_reduction_hl_f32##LMUL(xh, yh, ei, fi, vl);\
    fast_2_sum_f32##LMUL(yh, xl, yh, yl, vl);\
\
    get_table_values_hl_f32##LMUL(fi, th, tl, vl);\
    calculate_super_exact_exp2_polynom_hl_f32##LMUL(yh, yl, pm1h, pm1l, vl);\
    reconstruct_exp_hl_hl_f32##LMUL(xh, ei, th, tl, pm1h, pm1l, res, vl);\
\
    return res;\
}

#ifndef __FAST_MATH__

#define RISCV_POW_UPDEXP_F32(LMUL, BOOL)\
forceinline void update_exponent_f32##LMUL(const vfloat32##LMUL##_t& x, const vuint32##LMUL##_t& ei, vfloat32##LMUL##_t& res, size_t vl)\
{\
    vint32##LMUL##_t sign_x = __riscv_vand_vx_i32##LMUL(__riscv_vreinterpret_v_f32##LMUL##_i32##LMUL(x), SIGN_MASK_F32, vl);\
    vuint32##LMUL##_t subDiff = __riscv_vand_vx_u32##LMUL(__riscv_vreinterpret_v_i32##LMUL##_u32##LMUL(__riscv_vsra_vx_i32##LMUL(sign_x, BITS_F32, vl)), EXPADD_F32, vl);\
    \
    vuint32##LMUL##_t pow2ei = __riscv_vand_vx_u32##LMUL(__riscv_vadd_vv_u32##LMUL(__riscv_vadd_vx_u32##LMUL(__riscv_vsll_vx_u32##LMUL(ei, (size_t)23, vl), EXPADD_F32, vl), subDiff, vl), EXP_MASK_F32, vl);\
\
    res = __riscv_vfmul_vv_f32##LMUL(res, __riscv_vreinterpret_v_u32##LMUL##_f32##LMUL(pow2ei), vl);\
    res = __riscv_vfmul_vv_f32##LMUL(res, __riscv_vreinterpret_v_u32##LMUL##_f32##LMUL(__riscv_vrsub_vx_u32##LMUL(subDiff, (uint32_t)0x40000000, vl)), vl);\
}

#else

#define RISCV_POW_UPDEXP_F32(LMUL, BOOL)\
forceinline void update_exponent_f32##LMUL(const vfloat32##LMUL##_t& x, const vuint32##LMUL##_t& ei, vfloat32##LMUL##_t& res, size_t vl)\
{\
    vbool##BOOL##_t subnormalMask = __riscv_vmflt_vf_f32##LMUL##_b##BOOL(x, 0.0f, vl);\
    vuint32##LMUL##_t subDiff = __riscv_vmerge_vxm_u32##LMUL(__riscv_vmv_v_x_u32##LMUL(uint32_t(0), vl), EXPADD_F32, subnormalMask, vl);\
    \
    vuint32##LMUL##_t pow2ei = __riscv_vand_vx_u32##LMUL(__riscv_vadd_vv_u32##LMUL(__riscv_vadd_vx_u32##LMUL(__riscv_vsll_vx_u32##LMUL(ei, (size_t)23, vl), EXPADD_F32, vl), subDiff, vl), EXP_MASK_F32, vl);\
\
    res = __riscv_vfmul_vv_f32##LMUL(res, __riscv_vreinterpret_v_u32##LMUL##_f32##LMUL(pow2ei), vl);\
    res = __riscv_vfmul_vv_f32##LMUL(res, __riscv_vreinterpret_v_u32##LMUL##_f32##LMUL(__riscv_vrsub_vx_u32##LMUL(subDiff, (uint32_t)0x40000000, vl)), vl);\
}

#endif

RISCV_POW_UPDEXP_F32(m1, 32)
RISCV_POW_UPDEXP_F32(m2, 16)
RISCV_POW_UPDEXP_F32(m4, 8)
RISCV_POW_UPDEXP_F32(m8, 4)

RISCV_POW_EXP2_F32(m1, 32)
RISCV_POW_EXP2_F32(m2, 16)
RISCV_POW_EXP2_F32(m4, 8)
RISCV_POW_EXP2_F32(m8, 4)

RISCV_POW_UPDEXP_F64(m1, 64)
RISCV_POW_UPDEXP_F64(m2, 32)
RISCV_POW_UPDEXP_F64(m4, 16)
RISCV_POW_UPDEXP_F64(m8, 8)

RISCV_POW_EXP2_F64(m1, 64)
RISCV_POW_EXP2_F64(m2, 32)
RISCV_POW_EXP2_F64(m4, 16)
RISCV_POW_EXP2_F64(m8, 8)

#ifdef __riscv_zvfh

#ifndef __FAST_MATH__

const _Float16 EXP2_POL_COEFF_1_H_F16 = 0x1.63p-1f16;
const _Float16 EXP2_POL_COEFF_1_P5_H_F16 = 0x1.63p4f16;
const _Float16 EXP2_POL_COEFF_1_P14_H_F16 = 0x1.63p13f16;
const _Float16 EXP2_POL_COEFF_1_P14_L_F16 = -0x1.bdp1f16;
const _Float16 EXP2_POL_COEFF_2_P8_F16 = 0x1.ecp5f16;
const _Float16 EXP2_POL_COEFF_3_P18_F16 = 0x1.c6cp13f16;
const _Float16 EXP2_POL_COEFF_3_P2_F16 = 0x1.c6cp-3f16;


#define RISCV_POW_EXP2_F16(LMUL, BOOL)\
forceinline void do_exact_spec_exp_argument_reduction_f16##LMUL(const vfloat16##LMUL##_t& x, vfloat16##LMUL##_t& y, vuint16##LMUL##_t& ei, vuint16##LMUL##_t& fi, size_t vl)\
{\
    vfloat16##LMUL##_t vmagicConst1 = __riscv_vfmv_v_f_f16##LMUL(1536.0f16, vl);\
    vfloat16##LMUL##_t h = __riscv_vfmadd_vf_f16##LMUL(x, 0x1p3f16, vmagicConst1, vl);\
    vuint16##LMUL##_t hi = __riscv_vand_vx_u16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(h), 0x01ff, vl);\
    fi = __riscv_vand_vx_u16##LMUL(hi, 0x0007, vl);\
    ei = __riscv_vsrl_vx_u16##LMUL(hi, 3, vl);\
    h = __riscv_vfsub_vv_f16##LMUL(h, vmagicConst1, vl);\
\
    y = __riscv_vfmadd_vf_f16##LMUL(h, -0x1p-3f16, x, vl);\
}\
\
forceinline void get_table_values_hl_f16##LMUL(\
    vuint16##LMUL##_t& index, vfloat16##LMUL##_t& th, vfloat16##LMUL##_t& tl, size_t vl)\
{\
    index = __riscv_vmul_vx_u16##LMUL(index, uint16_t(sizeof(_Float16)), vl);\
    th = __riscv_vloxei16_v_f16##LMUL(LOOK_UP_TABLE_HIGH_F16, index, vl);\
    tl = __riscv_vloxei16_v_f16##LMUL(LOOK_UP_TABLE_LOW_P15_F16, index, vl);\
}\
\
forceinline void calculate_super_exact_exp2_polynom_hl_f16##LMUL(const vfloat16##LMUL##_t& yh, const vfloat16##LMUL##_t& yl, vfloat16##LMUL##_t& ph, vfloat16##LMUL##_t& pl, size_t vl)\
{\
    vfloat16##LMUL##_t r = __riscv_vfmadd_vf_f16##LMUL(yh, EXP2_POL_COEFF_3_P2_F16, __riscv_vfmv_v_f_f16##LMUL(EXP2_POL_COEFF_2_P8_F16, vl), vl);\
    ph = __riscv_vfmadd_vv_f16##LMUL(yh, r, __riscv_vfmv_v_f_f16##LMUL(EXP2_POL_COEFF_1_P14_L_F16, vl), vl);\
\
    r = __riscv_vfmul_vf_f16##LMUL(yh, EXP2_POL_COEFF_1_P14_H_F16, vl);\
    pl = __riscv_vfmsub_vf_f16##LMUL(yh, EXP2_POL_COEFF_1_P14_H_F16, r, vl);\
    pl = __riscv_vfmadd_vv_f16##LMUL(__riscv_vfmul_vf_f16##LMUL(yh, 0x1p5f16, vl), ph, __riscv_vfmul_vf_f16##LMUL(pl, 0x1p5f16, vl), vl);\
    ph = __riscv_vfmadd_vf_f16##LMUL(pl, 0x1p-5f16, r, vl);\
    pl = __riscv_vfnmsac_vf_f16##LMUL(pl, 0x1p5f16, __riscv_vfsub_vv_f16##LMUL(ph, r, vl), vl);\
    pl = __riscv_vfmadd_vf_f16##LMUL(yl, EXP2_POL_COEFF_1_P5_H_F16, pl, vl);\
}\
\
forceinline void update_exponent_f16##LMUL(const vfloat16##LMUL##_t& x, const vfloat16##LMUL##_t& xl, const vuint16##LMUL##_t& ei, vfloat16##LMUL##_t& res, size_t vl)\
{\
\
    vint16##LMUL##_t sign_x = __riscv_vand_vx_i16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(x), SIGN_MASK_F16, vl);\
    vuint16##LMUL##_t subDiff = __riscv_vand_vx_u16##LMUL(__riscv_vreinterpret_v_i16##LMUL##_u16##LMUL(__riscv_vsra_vx_i16##LMUL(sign_x, BITS_F16, vl)), EXPADD_F16, vl);\
    \
    vuint16##LMUL##_t pow2ei = __riscv_vand_vx_u16##LMUL(__riscv_vadd_vv_u16##LMUL(__riscv_vadd_vx_u16##LMUL(__riscv_vsll_vx_u16##LMUL(ei, (size_t)10, vl), EXPADD_F16, vl), subDiff, vl), EXP_MASK_F16, vl);\
\
    res = __riscv_vfmul_vv_f16##LMUL(res, __riscv_vreinterpret_v_u16##LMUL##_f16##LMUL(pow2ei), vl);\
    res = __riscv_vfmul_vv_f16##LMUL(res, __riscv_vreinterpret_v_u16##LMUL##_f16##LMUL(__riscv_vrsub_vx_u16##LMUL(subDiff, (uint16_t)0x4000, vl)), vl);\
}\
\
forceinline void reconstruct_exp_hl_hl_f16##LMUL(const vfloat16##LMUL##_t& x, const vfloat16##LMUL##_t& xl, const vuint16##LMUL##_t& ei, const vfloat16##LMUL##_t& th, const vfloat16##LMUL##_t& tl,\
    const vfloat16##LMUL##_t& pm1h, const vfloat16##LMUL##_t& pm1l, vfloat16##LMUL##_t& res, size_t vl)\
{\
    vfloat16##LMUL##_t sh, sl;\
    fma12_f16##LMUL(pm1h, __riscv_vfmv_v_f_f16##LMUL(0x1p-6f16, vl), __riscv_vfmv_v_f_f16##LMUL(0x1p14f16, vl), sh, sl, vl);\
    sl = __riscv_vfmadd_vf_f16##LMUL(sl, 0x1p11f16, pm1l, vl);\
\
    res = __riscv_vfmul_vv_f16##LMUL(th, sh, vl);\
    sl = __riscv_vfmadd_vv_f16##LMUL(th, sl, __riscv_vfmul_vv_f16##LMUL(tl, sh, vl), vl);\
    sh = __riscv_vfmsub_vv_f16##LMUL(th, sh, res, vl);\
    sl = __riscv_vfmadd_vf_f16##LMUL(sh, 0x1p11f16, sl, vl);\
    res = __riscv_vfmadd_vf_f16##LMUL(\
				sl, 0x1p-11f16,\
				res,\
				vl);\
    res = __riscv_vfmul_vf_f16##LMUL(res, 0x1p-14f16, vl);\
\
    update_exponent_f16##LMUL(x, xl, ei, res, vl);\
}\
\
vfloat16##LMUL##_t special_exp_f16##LMUL(vfloat16##LMUL##_t xh, vfloat16##LMUL##_t xl, size_t vl)\
{\
    vfloat16##LMUL##_t res, yh, yl, zh, zl, th, tl, pm1h, pm1l;\
    vuint16##LMUL##_t ei, fi;\
    vbool##BOOL##_t nzero_z;\
\
    do_exact_spec_exp_argument_reduction_f16##LMUL(xh, zh, ei, fi, vl);\
\
    nzero_z = __riscv_vmor_mm_b##BOOL(__riscv_vmfne_vf_f16##LMUL##_b##BOOL(zh, 0.0f16, vl),\
				       __riscv_vmfge_vf_f16##LMUL##_b##BOOL(__riscv_vreinterpret_v_i16##LMUL##_f16##LMUL(__riscv_vand_vx_i16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_i16##LMUL(xl), ABS_MASK_F16, vl)), 0x1p0f16, vl),\
				       vl);\
    zh = __riscv_vfmul_vf_f16##LMUL(zh, 0x1p6f16, vl);\
    yh = __riscv_vfmacc_vf_f16##LMUL##_mu(nzero_z, zh, 0x1p-14f16, xl, vl);\
    yl = __riscv_vfnmsub_vf_f16##LMUL##_mu(nzero_z, __riscv_vfsub_vv_f16##LMUL(yh, zh, vl), 0x1p14f16, xl, vl);\
\
    get_table_values_hl_f16##LMUL(fi, th, tl, vl);\
\
    calculate_super_exact_exp2_polynom_hl_f16##LMUL(yh, yl, pm1h, pm1l, vl);\
\
    reconstruct_exp_hl_hl_f16##LMUL(xh, xl, ei, th, tl, pm1h, pm1l, res, vl);\
\
    return res;\
}

#else

const _Float16 EXP2_POL_COEFF_1_H_F16 = 0x1.63p-1f16;
const _Float16 EXP2_POL_COEFF_1_L_F16 = -0x1.bdp-13f16;
const _Float16 EXP2_POL_COEFF_2_F16 = 0x1.ecp-3f16;
const _Float16 EXP2_POL_COEFF_3_F16 = 0x1.c6cp-5f16;

#define RISCV_POW_EXP2_F16(LMUL, BOOL)\
forceinline void do_exact_spec_exp_argument_reduction_f16##LMUL(const vfloat16##LMUL##_t& x, vfloat16##LMUL##_t& y, vuint16##LMUL##_t& ei, vuint16##LMUL##_t& fi, size_t vl)\
{\
    vfloat16##LMUL##_t vmagicConst1 = __riscv_vfmv_v_f_f16##LMUL(1536.0f16, vl);\
    vfloat16##LMUL##_t h = __riscv_vfmadd_vf_f16##LMUL(x, 0x1p3f16, vmagicConst1, vl);\
    vuint16##LMUL##_t hi = __riscv_vand_vx_u16##LMUL(__riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(h), 0x01ff, vl);\
    fi = __riscv_vand_vx_u16##LMUL(hi, 0x0007, vl);\
    ei = __riscv_vsrl_vx_u16##LMUL(hi, 3, vl);\
    h = __riscv_vfsub_vv_f16##LMUL(h, vmagicConst1, vl);\
\
    y = __riscv_vfmadd_vf_f16##LMUL(h, -0x1p-3f16, x, vl);\
}\
\
forceinline void get_table_values_hl_f16##LMUL(\
    vuint16##LMUL##_t& index, vfloat16##LMUL##_t& th, vfloat16##LMUL##_t& tl, size_t vl)\
{\
    index = __riscv_vmul_vx_u16##LMUL(index, uint16_t(sizeof(_Float16)), vl);\
    th = __riscv_vloxei16_v_f16##LMUL(LOOK_UP_TABLE_HIGH_F16, index, vl);\
    tl = __riscv_vloxei16_v_f16##LMUL(LOOK_UP_TABLE_LOW_F16, index, vl);\
}\
\
forceinline void calculate_super_exact_exp2_polynom_hl_f16##LMUL(const vfloat16##LMUL##_t& yh, const vfloat16##LMUL##_t& yl, vfloat16##LMUL##_t& ph, vfloat16##LMUL##_t& pl, size_t vl)\
{\
    vfloat16##LMUL##_t sqryh = __riscv_vfmul_vv_f16##LMUL(yh, yh, vl);\
    vfloat16##LMUL##_t r = __riscv_vfmadd_vf_f16##LMUL(yh, EXP2_POL_COEFF_3_F16, __riscv_vfmv_v_f_f16##LMUL(EXP2_POL_COEFF_2_F16, vl), vl);\
    r = __riscv_vfmadd_vf_f16##LMUL(yh, EXP2_POL_COEFF_1_L_F16, __riscv_vfmul_vv_f16##LMUL(sqryh, r, vl), vl);\
    fma12_ver2p1_f16##LMUL(yh, __riscv_vfmv_v_f_f16##LMUL(EXP2_POL_COEFF_1_H_F16, vl), r, ph, pl, vl);\
    pl = __riscv_vfmadd_vf_f16##LMUL(yl, EXP2_POL_COEFF_1_H_F16, __riscv_vfmul_vf_f16##LMUL(pl, 0x1p10f16, vl), vl);\
}\
\
forceinline void update_exponent_f16##LMUL(const vfloat16##LMUL##_t& x, const vfloat16##LMUL##_t& xl, const vuint16##LMUL##_t& ei, vfloat16##LMUL##_t& res, size_t vl)\
{\
\
    vbool##BOOL##_t subnormalMask = __riscv_vmflt_vf_f16##LMUL##_b##BOOL(x, 0.0f16, vl);\
    vuint16##LMUL##_t subDiff = __riscv_vmerge_vxm_u16##LMUL(__riscv_vmv_v_x_u16##LMUL(uint16_t(0), vl), EXPADD_F16, subnormalMask, vl);\
    \
    vuint16##LMUL##_t pow2ei = __riscv_vand_vx_u16##LMUL(__riscv_vadd_vv_u16##LMUL(__riscv_vadd_vx_u16##LMUL(__riscv_vsll_vx_u16##LMUL(ei, (size_t)10, vl), EXPADD_F16, vl), subDiff, vl), EXP_MASK_F16, vl);\
\
    res = __riscv_vfmul_vv_f16##LMUL(res, __riscv_vreinterpret_v_u16##LMUL##_f16##LMUL(pow2ei), vl);\
    res = __riscv_vfmul_vv_f16##LMUL(res, __riscv_vreinterpret_v_u16##LMUL##_f16##LMUL(__riscv_vrsub_vx_u16##LMUL(subDiff, (uint16_t)0x4000, vl)), vl);\
}\
\
forceinline void reconstruct_exp_hl_hl_f16##LMUL(const vfloat16##LMUL##_t& x, const vfloat16##LMUL##_t& xl, const vuint16##LMUL##_t& ei, const vfloat16##LMUL##_t& th, const vfloat16##LMUL##_t& tl,\
    const vfloat16##LMUL##_t& pm1h, const vfloat16##LMUL##_t& pm1l, vfloat16##LMUL##_t& res, size_t vl)\
{\
    vfloat16##LMUL##_t sh, sl;\
    fast_2_sum_f16##LMUL(__riscv_vfmv_v_f_f16##LMUL(1.0f16, vl), pm1h, sh, sl, vl);\
    sl = __riscv_vfmadd_vf_f16##LMUL(pm1l, 0x1p-10f16, sl, vl);\
\
    mul21_f16##LMUL(th, tl, sh, sl, res, vl);\
\
    update_exponent_f16##LMUL(x, xl, ei, res, vl);\
}\
\
vfloat16##LMUL##_t special_exp_f16##LMUL(vfloat16##LMUL##_t xh, vfloat16##LMUL##_t xl, size_t vl)\
{\
    vfloat16##LMUL##_t res, yh, yl, zh, zl, th, tl, pm1h, pm1l;\
    vuint16##LMUL##_t ei, fi;\
        \
    do_exact_spec_exp_argument_reduction_f16##LMUL(xh, zh, ei, fi, vl);\
\
    yh = __riscv_vfmadd_vf_f16##LMUL(xl, 0x1p-10f16, zh, vl);\
    yl = __riscv_vfnmsub_vf_f16##LMUL(__riscv_vfsub_vv_f16##LMUL(yh, zh, vl), 0x1p10f16, xl, vl);\
\
    get_table_values_hl_f16##LMUL(fi, th, tl, vl);\
\
    calculate_super_exact_exp2_polynom_hl_f16##LMUL(yh, yl, pm1h, pm1l, vl);\
\
    reconstruct_exp_hl_hl_f16##LMUL(xh, xl, ei, th, tl, pm1h, pm1l, res, vl);\
\
    return res;\
}

#endif


RISCV_POW_EXP2_F16(m1, 16)
RISCV_POW_EXP2_F16(m2, 8)
RISCV_POW_EXP2_F16(m4, 4)
RISCV_POW_EXP2_F16(m8, 2)

#endif