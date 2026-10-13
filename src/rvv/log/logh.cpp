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
 *   File:  logh.cpp                                     *
 *   Contains: intrinsic functions log, log2, log10,     *
 *             log1p for float16                         *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the logarithm of input vector V              *
 *                                                       *
 * Algorithm:                                            *
 *    1) Argument reduction to [1-a, 1+a], a = 11/32     *
 *    2) Additional reduction using two look-up tables   *
 *       of size 2^k (k=4)                               *
 *    3) Polynomial approximation (degree = 2)           *
 *    4) Reconstruction of the result                    *
 *                                                       *
 * The log2 and log10 functions are implemented using    *
 * the log function                                      *
 *                                                       *
 * The log1p function is implemented using an accurate   *
 * (a + 1) * b - 1 operation, a polynom of degree 3 and  *
 * three look-up tables                                  *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic

#ifdef __riscv_zvfh

#include "riscv_vector.h"
#include <cfloat>

#include "hlog.data"
#include "hlog_aux.inl"

vfloat16m1_t __riscv_vlog_f16m1(vfloat16m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m1(avl);
    x = __riscv_vmv_v_v_f16m1_tu(__riscv_vfmv_v_f_f16m1(
        (_Float16)1.0f16, __riscv_vsetvlmax_e16m1()), x, vl);

#ifndef __FAST_MATH__
    vfloat16m1_t special;
    vbool16_t specialMask;
    log_check_special_cases_f16m1(x, special, specialMask, vl);
#endif

    vfloat16m1_t y, e, z, zh, zl, p, logR, logRl, res;

    my_log_argument_reduction_f16m1(x, y, e, vl);
    my_log_second_reduction_f16m1(y, z, zh, zl, logR, logRl, vl);
    my_log_approximation_f16m1(z, p, vl);
    my_log_reconstruction_h_f16m1(e, zh, zl, p, logR, logRl, res, vl);

    res = __riscv_vfmul_vf_f16m1(res, LOG_INV_FACTOR_COEFF_F16, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m1(res, special, specialMask, vl);
#endif

    return res;
}

vfloat16m1_t __riscv_vlog2_f16m1(vfloat16m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m1(avl);
    x = __riscv_vmv_v_v_f16m1_tu(__riscv_vfmv_v_f_f16m1(
        (_Float16)1.0f16, __riscv_vsetvlmax_e16m1()), x, vl);

#ifndef __FAST_MATH__
    vfloat16m1_t special;
    vbool16_t specialMask;
    log_check_special_cases_f16m1(x, special, specialMask, vl);
#endif

    vfloat16m1_t y, e, z, zh, zl, p, logR, logRl, res;

    my_log_argument_reduction_f16m1(x, y, e, vl);
    my_log_second_reduction_f16m1(y, z, zh, zl, logR, logRl, vl);
    my_log_approximation_f16m1(z, p, vl);
    my_log2_reconstruction_f16m1(e, zh, zl, p, logR, logRl, res, vl);

    res = __riscv_vfmul_vf_f16m1(res, LOG_INV_FACTOR_COEFF_F16, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m1(res, special, specialMask, vl);
#endif

    return res;
}

vfloat16m1_t __riscv_vlog10_f16m1(vfloat16m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m1(avl);
    x = __riscv_vmv_v_v_f16m1_tu(__riscv_vfmv_v_f_f16m1(
        (_Float16)1.0f16, __riscv_vsetvlmax_e16m1()), x, vl);

#ifndef __FAST_MATH__
    vfloat16m1_t special;
    vbool16_t specialMask;
    log_check_special_cases_f16m1(x, special, specialMask, vl);
#endif

    constexpr _Float16 invLog10h = 0x1.bccp-2f16, invLog10l = -0x1.0ap-15f16;
    vfloat16m1_t y, e, z, zh, zl, p, logR, logRl, res, resl;

    my_log_argument_reduction_f16m1(x, y, e, vl);
    my_log_second_reduction_f16m1(y, z, zh, zl, logR, logRl, vl);
    my_log_approximation_f16m1(z, p, vl);
    my_log_reconstruction_hl_f16m1(e, zh, zl, p, logR, logRl, res, resl, vl);

    res = mul21_f16m1(res, resl, invLog10h, invLog10l, vl);
    res = __riscv_vfmul_vf_f16m1(res, LOG_INV_FACTOR_COEFF_F16, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m1(res, special, specialMask, vl);
#endif

    return res;
}

vfloat16m1_t __riscv_vlog1p_f16m1(vfloat16m1_t xm1, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m1(avl);
    xm1 = __riscv_vmv_v_v_f16m1_tu(__riscv_vfmv_v_f_f16m1(
        (_Float16)0.0f16, __riscv_vsetvlmax_e16m1()), xm1, vl);

#ifndef __FAST_MATH__
    vfloat16m1_t special;
    vbool16_t specialMask;
    log_check_special_cases_f16m1(xm1, special, specialMask, vl, (_Float16)-1.0f, (_Float16)0.0f);
#endif

    vfloat16m1_t xm1c, ef, y, p, logR, logRl, z, zh, zl, res;
    vint16m1_t e;
    vbool16_t linearMask;

    my_log1p_process_linear_f16m1(xm1, xm1c, linearMask, vl);
    my_log1p_argument_reduction_f16m1(__riscv_vfadd_vf_f16m1(xm1c, (_Float16)1.0f, vl), y, e, ef, vl);
    my_log1p_second_reduction_f16m1(xm1, y, e, z, zh, zl, logR, logRl, vl);
    my_log1p_approximation_f16m1(z, p, vl);
    my_log1p_reconstruction_h_f16m1(ef, zh, zl, p, logR, logRl, res, vl);
    res = __riscv_vfmul_vf_f16m1(res, LOG_INV_FACTOR_COEFF_F16, vl);
    res = __riscv_vmerge_vvm_f16m1(res, xm1, linearMask, vl);

#ifndef __FAST_MATH__
    my_log1p_set_fe_underflow_f16m1(res, vl);
    res = __riscv_vmerge_vvm_f16m1(res, special, specialMask, vl);
#endif

    return res;
}

// ==================================================================================================
// vfloat16m2_t
// ==================================================================================================

vfloat16m2_t __riscv_vlog_f16m2(vfloat16m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m2(avl);
    x = __riscv_vmv_v_v_f16m2_tu(__riscv_vfmv_v_f_f16m2(
        (_Float16)1.0f16, __riscv_vsetvlmax_e16m2()), x, vl);

#ifndef __FAST_MATH__
    vfloat16m2_t special;
    vbool8_t specialMask;
    log_check_special_cases_f16m2(x, special, specialMask, vl);
#endif

    vfloat16m2_t y, e, z, zh, zl, p, logR, logRl, res;

    my_log_argument_reduction_f16m2(x, y, e, vl);
    my_log_second_reduction_f16m2(y, z, zh, zl, logR, logRl, vl);
    my_log_approximation_f16m2(z, p, vl);
    my_log_reconstruction_h_f16m2(e, zh, zl, p, logR, logRl, res, vl);

    res = __riscv_vfmul_vf_f16m2(res, LOG_INV_FACTOR_COEFF_F16, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m2(res, special, specialMask, vl);
#endif

    return res;
}

vfloat16m2_t __riscv_vlog2_f16m2(vfloat16m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m2(avl);
    x = __riscv_vmv_v_v_f16m2_tu(__riscv_vfmv_v_f_f16m2(
        (_Float16)1.0f16, __riscv_vsetvlmax_e16m2()), x, vl);

#ifndef __FAST_MATH__
    vfloat16m2_t special;
    vbool8_t specialMask;
    log_check_special_cases_f16m2(x, special, specialMask, vl);
#endif

    vfloat16m2_t y, e, z, zh, zl, p, logR, logRl, res;

    my_log_argument_reduction_f16m2(x, y, e, vl);
    my_log_second_reduction_f16m2(y, z, zh, zl, logR, logRl, vl);
    my_log_approximation_f16m2(z, p, vl);
    my_log2_reconstruction_f16m2(e, zh, zl, p, logR, logRl, res, vl);

    res = __riscv_vfmul_vf_f16m2(res, LOG_INV_FACTOR_COEFF_F16, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m2(res, special, specialMask, vl);
#endif

    return res;
}

vfloat16m2_t __riscv_vlog10_f16m2(vfloat16m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m2(avl);
    x = __riscv_vmv_v_v_f16m2_tu(__riscv_vfmv_v_f_f16m2(
        (_Float16)1.0f16, __riscv_vsetvlmax_e16m2()), x, vl);

#ifndef __FAST_MATH__
    vfloat16m2_t special;
    vbool8_t specialMask;
    log_check_special_cases_f16m2(x, special, specialMask, vl);
#endif

    constexpr _Float16 invLog10h = 0x1.bccp-2f16, invLog10l = -0x1.0ap-15f16;
    vfloat16m2_t y, e, z, zh, zl, p, logR, logRl, res, resl;

    my_log_argument_reduction_f16m2(x, y, e, vl);
    my_log_second_reduction_f16m2(y, z, zh, zl, logR, logRl, vl);
    my_log_approximation_f16m2(z, p, vl);
    my_log_reconstruction_hl_f16m2(e, zh, zl, p, logR, logRl, res, resl, vl);

    res = mul21_f16m2(res, resl, invLog10h, invLog10l, vl);
    res = __riscv_vfmul_vf_f16m2(res, LOG_INV_FACTOR_COEFF_F16, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m2(res, special, specialMask, vl);
#endif

    return res;
}

vfloat16m2_t __riscv_vlog1p_f16m2(vfloat16m2_t xm1, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m2(avl);
    xm1 = __riscv_vmv_v_v_f16m2_tu(__riscv_vfmv_v_f_f16m2(
        (_Float16)0.0f16, __riscv_vsetvlmax_e16m2()), xm1, vl);

#ifndef __FAST_MATH__
    vfloat16m2_t special;
    vbool8_t specialMask;
    log_check_special_cases_f16m2(xm1, special, specialMask, vl, (_Float16)-1.0f, (_Float16)0.0f);
#endif

    vfloat16m2_t xm1c, ef, y, p, logR, logRl, z, zh, zl, res;
    vint16m2_t e;
    vbool8_t linearMask;

    my_log1p_process_linear_f16m2(xm1, xm1c, linearMask, vl);
    my_log1p_argument_reduction_f16m2(__riscv_vfadd_vf_f16m2(xm1c, (_Float16)1.0f, vl), y, e, ef, vl);
    my_log1p_second_reduction_f16m2(xm1, y, e, z, zh, zl, logR, logRl, vl);
    my_log1p_approximation_f16m2(z, p, vl);
    my_log1p_reconstruction_h_f16m2(ef, zh, zl, p, logR, logRl, res, vl);
    res = __riscv_vfmul_vf_f16m2(res, LOG_INV_FACTOR_COEFF_F16, vl);
    res = __riscv_vmerge_vvm_f16m2(res, xm1, linearMask, vl);

#ifndef __FAST_MATH__
    my_log1p_set_fe_underflow_f16m2(res, vl);
    res = __riscv_vmerge_vvm_f16m2(res, special, specialMask, vl);
#endif

    return res;
}

// ==================================================================================================
// vfloat16m4_t
// ==================================================================================================

vfloat16m4_t __riscv_vlog_f16m4(vfloat16m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m4(avl);
    x = __riscv_vmv_v_v_f16m4_tu(__riscv_vfmv_v_f_f16m4(
        (_Float16)1.0f16, __riscv_vsetvlmax_e16m4()), x, vl);

#ifndef __FAST_MATH__
    vfloat16m4_t special;
    vbool4_t specialMask;
    log_check_special_cases_f16m4(x, special, specialMask, vl);
#endif

    vfloat16m4_t y, e, z, zh, zl, p, logR, logRl, res;

    my_log_argument_reduction_f16m4(x, y, e, vl);
    my_log_second_reduction_f16m4(y, z, zh, zl, logR, logRl, vl);
    my_log_approximation_f16m4(z, p, vl);
    my_log_reconstruction_h_f16m4(e, zh, zl, p, logR, logRl, res, vl);

    res = __riscv_vfmul_vf_f16m4(res, LOG_INV_FACTOR_COEFF_F16, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m4(res, special, specialMask, vl);
#endif

    return res;
}

vfloat16m4_t __riscv_vlog2_f16m4(vfloat16m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m4(avl);
    x = __riscv_vmv_v_v_f16m4_tu(__riscv_vfmv_v_f_f16m4(
        (_Float16)1.0f16, __riscv_vsetvlmax_e16m4()), x, vl);

#ifndef __FAST_MATH__
    vfloat16m4_t special;
    vbool4_t specialMask;
    log_check_special_cases_f16m4(x, special, specialMask, vl);
#endif

    vfloat16m4_t y, e, z, zh, zl, p, logR, logRl, res;

    my_log_argument_reduction_f16m4(x, y, e, vl);
    my_log_second_reduction_f16m4(y, z, zh, zl, logR, logRl, vl);
    my_log_approximation_f16m4(z, p, vl);
    my_log2_reconstruction_f16m4(e, zh, zl, p, logR, logRl, res, vl);

    res = __riscv_vfmul_vf_f16m4(res, LOG_INV_FACTOR_COEFF_F16, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m4(res, special, specialMask, vl);
#endif

    return res;
}

vfloat16m4_t __riscv_vlog10_f16m4(vfloat16m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m4(avl);
    x = __riscv_vmv_v_v_f16m4_tu(__riscv_vfmv_v_f_f16m4(
        (_Float16)1.0f16, __riscv_vsetvlmax_e16m4()), x, vl);

#ifndef __FAST_MATH__
    vfloat16m4_t special;
    vbool4_t specialMask;
    log_check_special_cases_f16m4(x, special, specialMask, vl);
#endif

    constexpr _Float16 invLog10h = 0x1.bccp-2f16, invLog10l = -0x1.0ap-15f16;
    vfloat16m4_t y, e, z, zh, zl, p, logR, logRl, res, resl;

    my_log_argument_reduction_f16m4(x, y, e, vl);
    my_log_second_reduction_f16m4(y, z, zh, zl, logR, logRl, vl);
    my_log_approximation_f16m4(z, p, vl);
    my_log_reconstruction_hl_f16m4(e, zh, zl, p, logR, logRl, res, resl, vl);

    res = mul21_f16m4(res, resl, invLog10h, invLog10l, vl);
    res = __riscv_vfmul_vf_f16m4(res, LOG_INV_FACTOR_COEFF_F16, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m4(res, special, specialMask, vl);
#endif

    return res;
}

vfloat16m4_t __riscv_vlog1p_f16m4(vfloat16m4_t xm1, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m4(avl);
    xm1 = __riscv_vmv_v_v_f16m4_tu(__riscv_vfmv_v_f_f16m4(
        (_Float16)0.0f16, __riscv_vsetvlmax_e16m4()), xm1, vl);

#ifndef __FAST_MATH__
    vfloat16m4_t special;
    vbool4_t specialMask;
    log_check_special_cases_f16m4(xm1, special, specialMask, vl, (_Float16)-1.0f, (_Float16)0.0f);
#endif

    vfloat16m4_t xm1c, ef, y, p, logR, logRl, z, zh, zl, res;
    vint16m4_t e;
    vbool4_t linearMask;

    my_log1p_process_linear_f16m4(xm1, xm1c, linearMask, vl);
    my_log1p_argument_reduction_f16m4(__riscv_vfadd_vf_f16m4(xm1c, (_Float16)1.0f, vl), y, e, ef, vl);
    my_log1p_second_reduction_f16m4(xm1, y, e, z, zh, zl, logR, logRl, vl);
    my_log1p_approximation_f16m4(z, p, vl);
    my_log1p_reconstruction_h_f16m4(ef, zh, zl, p, logR, logRl, res, vl);
    res = __riscv_vfmul_vf_f16m4(res, LOG_INV_FACTOR_COEFF_F16, vl);
    res = __riscv_vmerge_vvm_f16m4(res, xm1, linearMask, vl);

#ifndef __FAST_MATH__
    my_log1p_set_fe_underflow_f16m4(res, vl);
    res = __riscv_vmerge_vvm_f16m4(res, special, specialMask, vl);
#endif

    return res;
}

// ==================================================================================================
// vfloat16m8_t
// ==================================================================================================

vfloat16m8_t __riscv_vlog_f16m8(vfloat16m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m8(avl);
    x = __riscv_vmv_v_v_f16m8_tu(__riscv_vfmv_v_f_f16m8(
        (_Float16)1.0f16, __riscv_vsetvlmax_e16m8()), x, vl);

    vfloat16m8_t res;
    vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1 = __riscv_vlog_f16m4(x1, vl);
    res = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e16m4(avl - vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1 = __riscv_vlog_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}

vfloat16m8_t __riscv_vlog2_f16m8(vfloat16m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m8(avl);
    x = __riscv_vmv_v_v_f16m8_tu(__riscv_vfmv_v_f_f16m8(
        (_Float16)1.0f16, __riscv_vsetvlmax_e16m8()), x, vl);

    vfloat16m8_t res;
    vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1 = __riscv_vlog2_f16m4(x1, vl);
    res = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e16m4(avl - vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1 = __riscv_vlog2_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}

vfloat16m8_t __riscv_vlog10_f16m8(vfloat16m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m8(avl);
    x = __riscv_vmv_v_v_f16m8_tu(__riscv_vfmv_v_f_f16m8(
        (_Float16)1.0f16, __riscv_vsetvlmax_e16m8()), x, vl);

    vfloat16m8_t res;
    vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1 = __riscv_vlog10_f16m4(x1, vl);
    res = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e16m4(avl - vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1 = __riscv_vlog10_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}

vfloat16m8_t __riscv_vlog1p_f16m8(vfloat16m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m8(avl);
    x = __riscv_vmv_v_v_f16m8_tu(__riscv_vfmv_v_f_f16m8(
        (_Float16)0.0f16, __riscv_vsetvlmax_e16m8()), x, vl);

    vfloat16m8_t res;
    vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1 = __riscv_vlog1p_f16m4(x1, vl);
    res = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e16m4(avl - vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1 = __riscv_vlog1p_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}

#endif /* __riscv_zvfh */

#endif /* __riscv_v_intrinsic */
