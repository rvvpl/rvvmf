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
 *   File:  log2_components.inl                          *
 *   Contains: internal components of extended-precision *
 *   log2 function for pow implementation                *
 *                                                       *
 *   Based on log2 implementation from src/rvv/log       *
 *                                                       *
 *                                                       *
 *********************************************************
*/


const double LOG2_POL_COEFF_0_F64 = -0x1.71547652b82ffp-1;
const double LOG2_POL_COEFF_1_F64 = 0x1.ec709dc3a04p-2;
const double LOG2_POL_COEFF_2_F64 = -0x1.71547651fa392p-2;
const double LOG2_POL_COEFF_3_F64 = 0x1.2776c50e27398p-2;
const double LOG2_POL_COEFF_4_F64 = -0x1.ec734ca48adc8p-3;
const double LOG2_POL_COEFF_5_F64 = 0x1.a619fe301a757p-3;

const float LOG2_POL_COEFF_0_F32 = -0x1.715478p-1f;
const float LOG2_POL_COEFF_1_F32 = 0x1.ec70ap-2f;
const float LOG2_POL_COEFF_2_F32 = -0x1.712188p-2f;
const float LOG2_POL_COEFF_3_F32 = 0x1.2753ccp-2f;

#define RISCV_POW_LOG2_F64(LMUL, BOOL)   \
static inline void decompose_number_f64##LMUL(vfloat64##LMUL##_t x, vfloat64##LMUL##_t& m, vint64##LMUL##_t& e, size_t vl) {\
    vuint64##LMUL##_t resu = __riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(x);\
    resu = __riscv_vand_vx_u64##LMUL(resu, (uint64_t)0x7ff0000000000000, vl);\
    resu = __riscv_vsrl_vx_u64##LMUL(resu, (uint64_t)52, vl);\
    e = __riscv_vreinterpret_v_u64##LMUL##_i64##LMUL(resu);\
    e = __riscv_vsub_vx_i64##LMUL(e, (int64_t)1023, vl);\
    \
    vuint64##LMUL##_t xi = __riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(x);\
    xi = __riscv_vand_vx_u64##LMUL(xi, (uint64_t)0x000fffffffffffff, vl);\
    xi = __riscv_vor_vx_u64##LMUL(xi, (uint64_t)0x3ff0000000000000, vl);\
    m = __riscv_vreinterpret_v_u64##LMUL##_f64##LMUL(xi);\
}\
\
static inline void my_log_argument_reduction_f64##LMUL(vfloat64##LMUL##_t x, vfloat64##LMUL##_t& y, vfloat64##LMUL##_t& ef, size_t vl) {\
    constexpr double a = 0x1.558p-2;\
    constexpr double ap1 = 0x1.556p0;\
    \
    vint64##LMUL##_t e, et;\
    vfloat64##LMUL##_t yt;\
    decompose_number_f64##LMUL(x, y, e, vl);\
    \
    vbool##BOOL##_t subnormalMask = __riscv_vmflt_vf_f64##LMUL##_b##BOOL(x, (double)0x1p-1022, vl);\
    yt = __riscv_vfsub_vf_f64##LMUL(y, (double)1.0, vl);\
    decompose_number_f64##LMUL(yt, yt, et, vl);\
    et = __riscv_vadd_vv_i64##LMUL(__riscv_vadd_vx_i64##LMUL(e, (int64_t)1, vl), et, vl);\
    y = __riscv_vmerge_vvm_f64##LMUL(y, yt, subnormalMask, vl);\
    e = __riscv_vmerge_vvm_i64##LMUL(e, et, subnormalMask, vl);\
    \
    vbool##BOOL##_t mask = __riscv_vmfge_vf_f64##LMUL##_b##BOOL(y, ap1, vl);\
    yt = __riscv_vfmul_vf_f64##LMUL(y, (double)0.5, vl);\
    et = __riscv_vadd_vx_i64##LMUL(e, (int64_t)1, vl);\
    y = __riscv_vmerge_vvm_f64##LMUL(y, yt, mask, vl);\
    e = __riscv_vmerge_vvm_i64##LMUL(e, et, mask, vl);\
    \
    ef = __riscv_vfcvt_f_x_v_f64##LMUL(e, vl);\
}\
\
static inline void my_log2_table_values_f64##LMUL(vfloat64##LMUL##_t y, vfloat64##LMUL##_t& logR, vfloat64##LMUL##_t& r, size_t vl) {\
    constexpr size_t k = 8;\
    constexpr double u = 0x1.7e5035f940d7fp0, v = 0x1.17e5035f9409bp-7;\
    constexpr size_t tshift = 52 - k;\
    \
    vuint64##LMUL##_t index = __riscv_vreinterpret_v_f64##LMUL##_u64##LMUL(\
        __riscv_vfmadd_vf_f64##LMUL(y, u, __riscv_vfmv_v_f_f64##LMUL(v, vl), vl));\
    index = __riscv_vand_vx_u64##LMUL(index, (uint64_t)0x000fffffffffffff, vl);\
    index = __riscv_vsrl_vx_u64##LMUL(index, tshift, vl);\
    \
    index = __riscv_vmul_vx_u64##LMUL(index, sizeof(double), vl);\
    \
    logR = __riscv_vloxei64_v_f64##LMUL(my_table_log2_R, index, vl);\
    r = __riscv_vloxei64_v_f64##LMUL(my_table2_R, index, vl);\
}\
\
static inline void my_log2_second_reduction_f64##LMUL(vfloat64##LMUL##_t y, vfloat64##LMUL##_t& zh, vfloat64##LMUL##_t& zl, vfloat64##LMUL##_t& logR, size_t vl) {\
    vfloat64##LMUL##_t r, ph, pl;\
    my_log2_table_values_f64##LMUL(y, logR, r, vl);\
    \
    ph = __riscv_vfmul_vv_f64##LMUL(y, r, vl);\
    pl = __riscv_vfmsub_vv_f64##LMUL(y, r, ph, vl);\
    ph = __riscv_vfsub_vf_f64##LMUL(ph, (double)1.0, vl);\
\
    fast_2_sum_f64##LMUL(ph, pl, zh, zl, vl);\
}\
\
static inline vfloat64##LMUL##_t my_log2_approximation_f64##LMUL(vfloat64##LMUL##_t z, size_t vl) {\
    vfloat64##LMUL##_t sqrz = __riscv_vfmul_vv_f64##LMUL(z, z, vl);\
    vfloat64##LMUL##_t odd = __riscv_vfmadd_vf_f64##LMUL(sqrz, LOG2_POL_COEFF_4_F64, __riscv_vfmv_v_f_f64##LMUL(LOG2_POL_COEFF_2_F64, vl), vl);\
    vfloat64##LMUL##_t even = __riscv_vfmadd_vf_f64##LMUL(sqrz, LOG2_POL_COEFF_5_F64, __riscv_vfmv_v_f_f64##LMUL(LOG2_POL_COEFF_3_F64, vl), vl);\
    odd = __riscv_vfmadd_vv_f64##LMUL(sqrz, odd, __riscv_vfmv_v_f_f64##LMUL(LOG2_POL_COEFF_0_F64, vl), vl);\
    even = __riscv_vfmadd_vv_f64##LMUL(sqrz, even, __riscv_vfmv_v_f_f64##LMUL(LOG2_POL_COEFF_1_F64, vl), vl);\
    even = __riscv_vfmul_vv_f64##LMUL(sqrz, even, vl);\
    \
    return __riscv_vfmadd_vv_f64##LMUL(odd, z, even, vl);\
}\
\
static inline void my_log2_reconstruction_h_f64##LMUL(vfloat64##LMUL##_t e, vfloat64##LMUL##_t zh,\
    vfloat64##LMUL##_t zl, vfloat64##LMUL##_t px, vfloat64##LMUL##_t logR, vfloat64##LMUL##_t& resh, vfloat64##LMUL##_t& resl, size_t vl) {\
    \
    constexpr double invlog2h = 0x1.71547652b82fep0;\
    constexpr double invlog2l = 0x1.777d0ffda0d24p-56;\
    \
    vfloat64##LMUL##_t rh, rl1, rl2, t;\
\
    fast_2_sum_f64##LMUL(__riscv_vfmv_v_f_f64##LMUL(invlog2h, vl), px, rh, rl1, vl);\
    rl1 = __riscv_vfadd_vf_f64##LMUL(rl1, invlog2l, vl);\
\
    mul22_f64##LMUL(rh, rl1, zh, zl, rh, rl1, vl);\
\
    fast_2_sum_f64##LMUL(e, __riscv_vfmul_vf_f64##LMUL(logR, -1.0, vl), t, resl, vl);\
    fast_2_sum_f64##LMUL(t, rh, resh, rl2, vl);\
\
    resl = __riscv_vfadd_vv_f64##LMUL(resl, __riscv_vfadd_vv_f64##LMUL(rl1, rl2, vl), vl);\
}\
\
static inline void log22_f64##LMUL(vfloat64##LMUL##_t x, vfloat64##LMUL##_t& resh, vfloat64##LMUL##_t& resl, size_t vl) {\
    vfloat64##LMUL##_t y, e, zh, zl, logR;\
    \
    my_log_argument_reduction_f64##LMUL(x, y, e, vl);\
\
    my_log2_second_reduction_f64##LMUL(y, zh, zl, logR, vl);\
\
    vfloat64##LMUL##_t px;\
    px = my_log2_approximation_f64##LMUL(zh, vl);\
\
    my_log2_reconstruction_h_f64##LMUL(e, zh, zl, px, logR, resh, resl, vl);\
}

#define RISCV_POW_LOG2_F32(LMUL, BOOL)\
static inline void decompose_number_f32##LMUL(vfloat32##LMUL##_t x, vfloat32##LMUL##_t& m, vint32##LMUL##_t& e, size_t vl) {\
    vuint32##LMUL##_t resu = __riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(x);\
    resu = __riscv_vand_vx_u32##LMUL(resu, (uint32_t)0x7f800000, vl);\
    resu = __riscv_vsrl_vx_u32##LMUL(resu, (uint32_t)23, vl);\
    e = __riscv_vreinterpret_v_u32##LMUL##_i32##LMUL(resu);\
    e = __riscv_vsub_vx_i32##LMUL(e, (int32_t)127, vl);\
    \
    vuint32##LMUL##_t xi = __riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(x);\
    xi = __riscv_vand_vx_u32##LMUL(xi, (uint32_t)0x007fffff, vl);\
    xi = __riscv_vor_vx_u32##LMUL(xi, (uint32_t)0x3f800000, vl);\
    m = __riscv_vreinterpret_v_u32##LMUL##_f32##LMUL(xi);\
}\
\
static inline void my_log_argument_reduction_f32##LMUL(vfloat32##LMUL##_t x, vfloat32##LMUL##_t& y, vfloat32##LMUL##_t& ef, size_t vl) {\
    constexpr float a = 0x1.58p-2f;\
    constexpr float ap1 = 0x1.56p0f;\
    \
    vint32##LMUL##_t e, et;\
    vfloat32##LMUL##_t yt;\
    decompose_number_f32##LMUL(x, y, e, vl);\
    \
    vbool##BOOL##_t subnormalMask = __riscv_vmflt_vf_f32##LMUL##_b##BOOL(x, (float)0x1p-126f, vl);\
    yt = __riscv_vfsub_vf_f32##LMUL(y, (float)1.0f, vl);\
    decompose_number_f32##LMUL(yt, yt, et, vl);\
    et = __riscv_vadd_vv_i32##LMUL(__riscv_vadd_vx_i32##LMUL(e, (int32_t)1, vl), et, vl);\
    y = __riscv_vmerge_vvm_f32##LMUL(y, yt, subnormalMask, vl);\
    e = __riscv_vmerge_vvm_i32##LMUL(e, et, subnormalMask, vl);\
    \
    vbool##BOOL##_t mask = __riscv_vmfge_vf_f32##LMUL##_b##BOOL(y, ap1, vl);\
    yt = __riscv_vfmul_vf_f32##LMUL(y, (float)0.5f, vl);\
    et = __riscv_vadd_vx_i32##LMUL(e, (int32_t)1, vl);\
    y = __riscv_vmerge_vvm_f32##LMUL(y, yt, mask, vl);\
    e = __riscv_vmerge_vvm_i32##LMUL(e, et, mask, vl);\
    \
    ef = __riscv_vfcvt_f_x_v_f32##LMUL(e, vl);\
}\
\
static inline void my_log2_table_values_f32##LMUL(vfloat32##LMUL##_t y, vfloat32##LMUL##_t& logR, vfloat32##LMUL##_t& r, size_t vl) {\
    constexpr size_t k = 6;\
    constexpr float u = 0x1.7711dcp0f, v = 0x1.5dc48cp-5f;\
    constexpr size_t tshift = 23 - k;\
    \
    vuint32##LMUL##_t index = __riscv_vreinterpret_v_f32##LMUL##_u32##LMUL(\
        __riscv_vfmadd_vf_f32##LMUL(y, u, __riscv_vfmv_v_f_f32##LMUL(v, vl), vl));\
    index = __riscv_vand_vx_u32##LMUL(index, (uint32_t)0x007fffff, vl);\
    index = __riscv_vsrl_vx_u32##LMUL(index, tshift, vl);\
    \
    index = __riscv_vmul_vx_u32##LMUL(index, sizeof(float), vl);\
    \
    logR = __riscv_vloxei32_v_f32##LMUL(table_float_log2_R, index, vl);\
    r = __riscv_vloxei32_v_f32##LMUL(table2_float_R, index, vl);\
}\
\
static inline void my_log2_second_reduction_f32##LMUL(vfloat32##LMUL##_t y, vfloat32##LMUL##_t& zh, vfloat32##LMUL##_t& zl, vfloat32##LMUL##_t& logR, size_t vl) {\
    vfloat32##LMUL##_t r, ph, pl;\
    my_log2_table_values_f32##LMUL(y, logR, r, vl);\
\
    ph = __riscv_vfmul_vv_f32##LMUL(y, r, vl);\
    pl = __riscv_vfmsub_vv_f32##LMUL(y, r, ph, vl);\
    ph = __riscv_vfsub_vf_f32##LMUL(ph, (float)1.0f, vl);\
\
    fast_2_sum_f32##LMUL(ph, pl, zh, zl, vl);\
}\
\
static inline vfloat32##LMUL##_t my_log2_approximation_f32##LMUL(vfloat32##LMUL##_t z, size_t vl) {        \
    vfloat32##LMUL##_t sqrz = __riscv_vfmul_vv_f32##LMUL(z, z, vl);\
    vfloat32##LMUL##_t pol1 = __riscv_vfmadd_vf_f32##LMUL(sqrz, LOG2_POL_COEFF_3_F32, __riscv_vfmv_v_f_f32##LMUL(LOG2_POL_COEFF_1_F32, vl), vl);\
    vfloat32##LMUL##_t pol2 = __riscv_vfmadd_vf_f32##LMUL(sqrz, LOG2_POL_COEFF_2_F32, __riscv_vfmv_v_f_f32##LMUL(LOG2_POL_COEFF_0_F32, vl), vl);\
    \
    return __riscv_vfmadd_vv_f32##LMUL(pol1, z, pol2, vl);\
}\
\
static inline void my_log2_reconstruction_h_f32##LMUL(vfloat32##LMUL##_t e, vfloat32##LMUL##_t zh,\
    vfloat32##LMUL##_t zl, vfloat32##LMUL##_t px, vfloat32##LMUL##_t logR, vfloat32##LMUL##_t& resh, vfloat32##LMUL##_t& resl, size_t vl) {\
    \
    constexpr float ch = 0x1.715476p0f;\
    constexpr float cl = 0x1.4ae2dp-26f;\
    \
    vfloat32##LMUL##_t rh, rl1, rl2, t;\
\
    fma12_f32##LMUL(px, zh, __riscv_vfmv_v_f_f32##LMUL(ch, vl), rh, rl1, vl);\
    rl1 = __riscv_vfadd_vf_f32##LMUL(rl1, cl, vl);\
\
    mul22_f32##LMUL(rh, rl1, zh, zl, rh, rl1, vl);\
\
    fast_2_sum_f32##LMUL(e, __riscv_vfneg_v_f32##LMUL(logR, vl), t, resl, vl);\
    fast_2_sum_f32##LMUL(t, rh, resh, rl2, vl);\
\
    resl = __riscv_vfadd_vv_f32##LMUL(resl, __riscv_vfadd_vv_f32##LMUL(rl1, rl2, vl), vl);\
}\
\
static inline void log22_f32##LMUL(vfloat32##LMUL##_t x, vfloat32##LMUL##_t& resh, vfloat32##LMUL##_t& resl, size_t vl) {\
    vfloat32##LMUL##_t y, e, zh, zl, logR;\
    \
    my_log_argument_reduction_f32##LMUL(x, y, e, vl);\
\
    my_log2_second_reduction_f32##LMUL(y, zh, zl, logR, vl);\
\
    vfloat32##LMUL##_t px;\
    px = my_log2_approximation_f32##LMUL(zh, vl);\
\
    my_log2_reconstruction_h_f32##LMUL(e, zh, zl, px, logR, resh, resl, vl);\
}

RISCV_POW_LOG2_F64(m1, 64)
RISCV_POW_LOG2_F64(m2, 32)
RISCV_POW_LOG2_F64(m4, 16)
RISCV_POW_LOG2_F64(m8, 8)

RISCV_POW_LOG2_F32(m1, 32)
RISCV_POW_LOG2_F32(m2, 16)
RISCV_POW_LOG2_F32(m4, 8)
RISCV_POW_LOG2_F32(m8, 4)

#ifdef __riscv_zvfh

#ifndef __FAST_MATH__

const _Float16 LOG2_POL_COEFF_0_F16 = -0x1.718p9f16;
const _Float16 LOG2_POL_COEFF_1_F16 = 0x1.ed8p8f16;
const _Float16 LOG2_POL_COEFF_2_F16 = -0x1.098p8f16;

#define RISCV_POW_LOG2_F16(LMUL, BOOL)\
static inline void decompose_number_f16##LMUL(vfloat16##LMUL##_t x, vfloat16##LMUL##_t& m, vint16##LMUL##_t& e, size_t vl) {\
    vuint16##LMUL##_t resu = __riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(x);\
    resu = __riscv_vand_vx_u16##LMUL(resu, (uint16_t)0x7c00, vl);\
    resu = __riscv_vsrl_vx_u16##LMUL(resu, (uint16_t)10, vl);\
    e = __riscv_vreinterpret_v_u16##LMUL##_i16##LMUL(resu);\
    e = __riscv_vsub_vx_i16##LMUL(e, (int16_t)15, vl);\
    \
    vuint16##LMUL##_t xi = __riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(x);\
    xi = __riscv_vand_vx_u16##LMUL(xi, (uint16_t)0x03ff, vl);\
    xi = __riscv_vor_vx_u16##LMUL(xi, (uint16_t)0x3c00, vl);\
    m = __riscv_vreinterpret_v_u16##LMUL##_f16##LMUL(xi);\
}\
\
static inline void my_log_argument_reduction_f16##LMUL(vfloat16##LMUL##_t x, vfloat16##LMUL##_t& y, vfloat16##LMUL##_t& ef, size_t vl) {\
    constexpr _Float16 a = 0x1.6p-2f16;\
    constexpr _Float16 ap1 = 0x1.58p0f16;\
    \
    vint16##LMUL##_t e, et;\
    vfloat16##LMUL##_t yt;\
    decompose_number_f16##LMUL(x, y, e, vl);\
    \
    vbool##BOOL##_t subnormalMask = __riscv_vmflt_vf_f16##LMUL##_b##BOOL(x, (_Float16)0x1p-14f16, vl);\
    yt = __riscv_vfsub_vf_f16##LMUL(y, (_Float16)1.0f16, vl);\
    decompose_number_f16##LMUL(yt, yt, et, vl);\
    et = __riscv_vadd_vv_i16##LMUL(__riscv_vadd_vx_i16##LMUL(e, (int16_t)1, vl), et, vl);\
    y = __riscv_vmerge_vvm_f16##LMUL(y, yt, subnormalMask, vl);\
    e = __riscv_vmerge_vvm_i16##LMUL(e, et, subnormalMask, vl);\
    \
    vbool##BOOL##_t mask = __riscv_vmfge_vf_f16##LMUL##_b##BOOL(y, ap1, vl);\
    yt = __riscv_vfmul_vf_f16##LMUL(y, (_Float16)0.5f16, vl);\
    et = __riscv_vadd_vx_i16##LMUL(e, (int16_t)1, vl);\
    y = __riscv_vmerge_vvm_f16##LMUL(y, yt, mask, vl);\
    e = __riscv_vmerge_vvm_i16##LMUL(e, et, mask, vl);\
    \
    ef = __riscv_vfcvt_f_x_v_f16##LMUL(e, vl);\
}\
\
static inline void my_log2_table_values_f16##LMUL(vfloat16##LMUL##_t y, vfloat16##LMUL##_t& logR, vfloat16##LMUL##_t& logRl, vfloat16##LMUL##_t& r, size_t vl) {\
    constexpr size_t k = 4;\
    constexpr _Float16 u = 0x1.5ccp0f16, v =  0x1.58cp-3f16;\
    constexpr size_t tshift = 10 - k;\
    \
    vuint16##LMUL##_t index = __riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(\
        __riscv_vfmadd_vf_f16##LMUL(y, u, __riscv_vfmv_v_f_f16##LMUL(v, vl), vl));\
    index = __riscv_vand_vx_u16##LMUL(index, (uint16_t)0x03ff, vl);\
    index = __riscv_vsrl_vx_u16##LMUL(index, tshift, vl);\
    \
    index = __riscv_vmul_vx_u16##LMUL(index, sizeof(_Float16), vl);\
    \
    logR = __riscv_vloxei16_v_f16##LMUL(table_half_log2_R, index, vl);\
    logRl = __riscv_vloxei16_v_f16##LMUL(table_half_log2_Rl, index, vl);\
    r = __riscv_vloxei16_v_f16##LMUL(table2_half_R, index, vl);\
}\
\
static inline void my_log2_second_reduction_f16##LMUL(vfloat16##LMUL##_t y, vfloat16##LMUL##_t& zh, vfloat16##LMUL##_t& zl, vfloat16##LMUL##_t& logR, vfloat16##LMUL##_t& logRl, size_t vl) {\
    vfloat16##LMUL##_t r, ph, pl;\
\
    my_log2_table_values_f16##LMUL(y, logR, logRl, r, vl);\
    ph = __riscv_vfmul_vv_f16##LMUL(y, r, vl);\
    pl = __riscv_vfmsub_vv_f16##LMUL(y, r, ph, vl);\
    ph = __riscv_vfsub_vf_f16##LMUL(ph, 1.0f16, vl);\
\
    fast_2_sum_f16##LMUL(ph, pl, zh, zl, vl);\
}\
\
static inline vfloat16##LMUL##_t my_log2_approximation_f16##LMUL(vfloat16##LMUL##_t z, size_t vl) {        \
    vfloat16##LMUL##_t pol = __riscv_vfmadd_vf_f16##LMUL(z, LOG2_POL_COEFF_2_F16, __riscv_vfmv_v_f_f16##LMUL(LOG2_POL_COEFF_1_F16, vl), vl);\
    \
    return __riscv_vfmadd_vv_f16##LMUL(pol, z, __riscv_vfmv_v_f_f16##LMUL(LOG2_POL_COEFF_0_F16, vl), vl);\
}\
\
static inline void my_log2_reconstruction_h_f16##LMUL(vfloat16##LMUL##_t e, vfloat16##LMUL##_t zh,\
    vfloat16##LMUL##_t zl, vfloat16##LMUL##_t px, vfloat16##LMUL##_t logR, vfloat16##LMUL##_t logRl, vfloat16##LMUL##_t& resh, vfloat16##LMUL##_t& resl, size_t vl) {\
    \
    constexpr _Float16 ch = 0x1.714p10f16;\
    constexpr _Float16 cl = 0x1.47p8f16;\
    \
    vfloat16##LMUL##_t rh, rl1, rl2, t;\
\
    rh = __riscv_vfmadd_vv_f16##LMUL(px, zh, __riscv_vfmv_v_f_f16##LMUL(ch, vl), vl);\
    rl1 = __riscv_vfmadd_vv_f16##LMUL(px, zh, __riscv_vfrsub_vf_f16##LMUL(rh, ch, vl), vl);\
    rl1 = __riscv_vfmadd_vf_f16##LMUL(rl1, 0x1p10f16, __riscv_vfmv_v_f_f16##LMUL(cl, vl), vl);\
\
    t = __riscv_vfmul_vv_f16##LMUL(rh, zh, vl);\
    zl = __riscv_vfmul_vf_f16##LMUL(zl, 0x1p10f16, vl);\
    rl1 = __riscv_vfmadd_vf_f16##LMUL(__riscv_vfmsub_vv_f16##LMUL(rh, zh, t, vl), 0x1p10f16,\
				     __riscv_vfmadd_vv_f16##LMUL(\
								rh,\
								zl,\
								__riscv_vfmul_vv_f16##LMUL(rl1, zh, vl),\
								vl\
								),\
				     vl);\
    rh = t;\
\
    fast_2_sum_f16##LMUL(e, __riscv_vfneg_v_f16##LMUL(logR, vl), t, resl, vl);\
    resh = __riscv_vfmadd_vf_f16##LMUL(t, 0x1p10f16, rh, vl);\
    rl2 = __riscv_vfsub_vv_f16##LMUL(rh, __riscv_vfnmsub_vf_f16##LMUL(t, 0x1p10f16, resh, vl), vl);\
\
    resl = __riscv_vfmsub_vf_f16##LMUL(__riscv_vfmadd_vf_f16##LMUL(resl, 0x1p10f16, __riscv_vfmadd_vf_f16##LMUL(rl1, 0x1p-10f16, rl2, vl), vl), 0x1p4f16, logRl, vl);\
}\
\
static inline void log22_f16##LMUL(vfloat16##LMUL##_t x, vfloat16##LMUL##_t& resh, vfloat16##LMUL##_t& resl, size_t vl) {\
    vfloat16##LMUL##_t y, e, zh, zl, logR, logRl;\
    \
    my_log_argument_reduction_f16##LMUL(x, y, e, vl);\
    my_log2_second_reduction_f16##LMUL(y, zh, zl, logR, logRl, vl);\
\
    vfloat16##LMUL##_t px;\
    px = my_log2_approximation_f16##LMUL(zh, vl);\
    my_log2_reconstruction_h_f16##LMUL(e, zh, zl, px, logR, logRl, resh, resl, vl);\
}

#else

const _Float16 LOG2_POL_COEFF_0_F16 = -0x1.718p-1f16;
const _Float16 LOG2_POL_COEFF_1_F16 = 0x1.ed8p-2f16;
const _Float16 LOG2_POL_COEFF_2_F16 = -0x1.098p-2f16;

#define RISCV_POW_LOG2_F16(LMUL, BOOL)\
static inline void decompose_number_f16##LMUL(vfloat16##LMUL##_t x, vfloat16##LMUL##_t& m, vint16##LMUL##_t& e, size_t vl) {\
    vuint16##LMUL##_t resu = __riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(x);\
    resu = __riscv_vand_vx_u16##LMUL(resu, (uint16_t)0x7c00, vl);\
    resu = __riscv_vsrl_vx_u16##LMUL(resu, (uint16_t)10, vl);\
    e = __riscv_vreinterpret_v_u16##LMUL##_i16##LMUL(resu);\
    e = __riscv_vsub_vx_i16##LMUL(e, (int16_t)15, vl);\
    \
    vuint16##LMUL##_t xi = __riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(x);\
    xi = __riscv_vand_vx_u16##LMUL(xi, (uint16_t)0x03ff, vl);\
    xi = __riscv_vor_vx_u16##LMUL(xi, (uint16_t)0x3c00, vl);\
    m = __riscv_vreinterpret_v_u16##LMUL##_f16##LMUL(xi);\
}\
\
static inline void my_log_argument_reduction_f16##LMUL(vfloat16##LMUL##_t x, vfloat16##LMUL##_t& y, vfloat16##LMUL##_t& ef, size_t vl) {\
    constexpr _Float16 a = 0x1.6p-2f16;\
    constexpr _Float16 ap1 = 0x1.58p0f16;\
    \
    vint16##LMUL##_t e, et;\
    vfloat16##LMUL##_t yt;\
    decompose_number_f16##LMUL(x, y, e, vl);\
    \
    vbool##BOOL##_t subnormalMask = __riscv_vmflt_vf_f16##LMUL##_b##BOOL(x, (_Float16)0x1p-14f16, vl);\
    yt = __riscv_vfsub_vf_f16##LMUL(y, (_Float16)1.0f16, vl);\
    decompose_number_f16##LMUL(yt, yt, et, vl);\
    et = __riscv_vadd_vv_i16##LMUL(__riscv_vadd_vx_i16##LMUL(e, (int16_t)1, vl), et, vl);\
    y = __riscv_vmerge_vvm_f16##LMUL(y, yt, subnormalMask, vl);\
    e = __riscv_vmerge_vvm_i16##LMUL(e, et, subnormalMask, vl);\
    \
    vbool##BOOL##_t mask = __riscv_vmfge_vf_f16##LMUL##_b##BOOL(y, ap1, vl);\
    yt = __riscv_vfmul_vf_f16##LMUL(y, (_Float16)0.5f16, vl);\
    et = __riscv_vadd_vx_i16##LMUL(e, (int16_t)1, vl);\
    y = __riscv_vmerge_vvm_f16##LMUL(y, yt, mask, vl);\
    e = __riscv_vmerge_vvm_i16##LMUL(e, et, mask, vl);\
    \
    ef = __riscv_vfcvt_f_x_v_f16##LMUL(e, vl);\
}\
\
static inline void my_log2_table_values_f16##LMUL(vfloat16##LMUL##_t y, vfloat16##LMUL##_t& logR, vfloat16##LMUL##_t& logRl, vfloat16##LMUL##_t& r, size_t vl) {\
    constexpr size_t k = 4;\
    constexpr _Float16 u = 0x1.5ccp0f16, v =  0x1.58cp-3f16;\
    constexpr size_t tshift = 10 - k;\
    \
    vuint16##LMUL##_t index = __riscv_vreinterpret_v_f16##LMUL##_u16##LMUL(\
        __riscv_vfmadd_vf_f16##LMUL(y, u, __riscv_vfmv_v_f_f16##LMUL(v, vl), vl));\
    index = __riscv_vand_vx_u16##LMUL(index, (uint16_t)0x03ff, vl);\
    index = __riscv_vsrl_vx_u16##LMUL(index, tshift, vl);\
    \
    index = __riscv_vmul_vx_u16##LMUL(index, sizeof(_Float16), vl);\
    \
    logR = __riscv_vloxei16_v_f16##LMUL(table_half_log2_R, index, vl);\
    logRl = __riscv_vloxei16_v_f16##LMUL(table_half_log2_Rl, index, vl);\
    r = __riscv_vloxei16_v_f16##LMUL(table2_half_R, index, vl);\
}\
\
static inline void my_log2_second_reduction_f16##LMUL(vfloat16##LMUL##_t y, vfloat16##LMUL##_t& zh, vfloat16##LMUL##_t& zl, vfloat16##LMUL##_t& logR, vfloat16##LMUL##_t& logRl, size_t vl) {\
    vfloat16##LMUL##_t r, ph, pl;\
\
    my_log2_table_values_f16##LMUL(y, logR, logRl, r, vl);\
    ph = __riscv_vfmul_vv_f16##LMUL(y, r, vl);\
    pl = __riscv_vfmsub_vv_f16##LMUL(y, r, ph, vl);\
    ph = __riscv_vfsub_vf_f16##LMUL(ph, 1.0f16, vl);\
\
    fast_2_sum_f16##LMUL(ph, pl, zh, zl, vl);\
}\
\
static inline vfloat16##LMUL##_t my_log2_approximation_f16##LMUL(vfloat16##LMUL##_t z, size_t vl) {        \
    vfloat16##LMUL##_t pol = __riscv_vfmadd_vf_f16##LMUL(z, LOG2_POL_COEFF_2_F16, __riscv_vfmv_v_f_f16##LMUL(LOG2_POL_COEFF_1_F16, vl), vl);\
    \
    return __riscv_vfmadd_vv_f16##LMUL(pol, z, __riscv_vfmv_v_f_f16##LMUL(LOG2_POL_COEFF_0_F16, vl), vl);\
}\
\
static inline void my_log2_reconstruction_h_f16##LMUL(vfloat16##LMUL##_t e, vfloat16##LMUL##_t zh,\
    vfloat16##LMUL##_t zl, vfloat16##LMUL##_t px, vfloat16##LMUL##_t logR, vfloat16##LMUL##_t logRl, vfloat16##LMUL##_t& resh, vfloat16##LMUL##_t& resl, size_t vl) {\
    \
    constexpr _Float16 ch = 0x1.714p0f16;\
    constexpr _Float16 cl = 0x1.47p-12f16;\
    \
    vfloat16##LMUL##_t rh, rl1, rl2, t;\
\
    fma12_f16##LMUL(px, zh, __riscv_vfmv_v_f_f16##LMUL(ch, vl), rh, rl1, vl);\
    rl1 = __riscv_vfadd_vf_f16##LMUL(rl1, cl, vl);\
\
    mul22_f16##LMUL(rh, rl1, zh, zl, rh, rl1, vl);\
\
    fast_2_sum_f16##LMUL(e, __riscv_vfneg_v_f16##LMUL(logR, vl), t, resl, vl);\
    fast_2_sum_f16##LMUL(t, rh, resh, rl2, vl);\
\
    resl = __riscv_vfnmsub_vf_f16##LMUL(logRl, 0x1p-4f16, __riscv_vfmul_vf_f16##LMUL(\
									     __riscv_vfadd_vv_f16##LMUL(resl, __riscv_vfadd_vv_f16##LMUL(rl1, rl2, vl), vl),\
									     0x1p10f16, \
									     vl),\
						      vl);\
}\
\
static inline void log22_f16##LMUL(vfloat16##LMUL##_t x, vfloat16##LMUL##_t& resh, vfloat16##LMUL##_t& resl, size_t vl) {\
    vfloat16##LMUL##_t y, e, zh, zl, logR, logRl;\
    \
    my_log_argument_reduction_f16##LMUL(x, y, e, vl);\
    my_log2_second_reduction_f16##LMUL(y, zh, zl, logR, logRl, vl);\
\
    vfloat16##LMUL##_t px;\
    px = my_log2_approximation_f16##LMUL(zh, vl);\
    my_log2_reconstruction_h_f16##LMUL(e, zh, zl, px, logR, logRl, resh, resl, vl);\
}


#endif

RISCV_POW_LOG2_F16(m1, 16)
RISCV_POW_LOG2_F16(m2, 8)
RISCV_POW_LOG2_F16(m4, 4)
RISCV_POW_LOG2_F16(m8, 2)

#endif