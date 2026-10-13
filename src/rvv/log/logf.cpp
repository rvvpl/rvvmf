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
 *   File:  logf.cpp                                     *
 *   Contains: intrinsic functions log, log2, log10,     *
 *             log1p for float32                         *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the logarithm of input vector V              *
 *                                                       *
 * Algorithm:                                            *
 *    1) Argument reduction to [1-a, 1+a], a = 43/128    *
 *    2) Additional reduction using two look-up tables   *
 *       of size 2^k (k=6)                               *
 *    3) Polynomial approximation (degree = 3)           *
 *    4) Reconstruction of the result                    *
 *                                                       *
 * The log2 and log10 functions are implemented using    *
 * the log function                                      *
 *                                                       *
 * The log1p function is implemented using an accurate   *
 * (a + 1) * b - 1 operation                             *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic

#include "riscv_vector.h"
#include <cfloat>

#include "flog.data"
#include "flog_aux.inl"

vfloat32m1_t __riscv_vlog_f32m1(vfloat32m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m1(avl);
    x = __riscv_vmv_v_v_f32m1_tu(__riscv_vfmv_v_f_f32m1(
        (float)1.0f, __riscv_vsetvlmax_e32m1()), x, vl);

#ifndef __FAST_MATH__
    vfloat32m1_t special;
    vbool32_t specialMask;
    log_check_special_cases_f32m1(x, special, specialMask, vl);
#endif

    vfloat32m1_t y, e, z, zh, zl, p, logR, res;

    my_log_argument_reduction_f32m1(x, y, e, vl);
    my_log_second_reduction_f32m1(y, z, zh, zl, logR, vl);
    my_log_approximation_f32m1(z, p, vl);
    my_log_reconstruction_h_f32m1(e, zh, zl, p, logR, res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m1(res, special, specialMask, vl);
#endif

    return res;
}

vfloat32m1_t __riscv_vlog2_f32m1(vfloat32m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m1(avl);
    x = __riscv_vmv_v_v_f32m1_tu(__riscv_vfmv_v_f_f32m1(
        (float)1.0f, __riscv_vsetvlmax_e32m1()), x, vl);

#ifndef __FAST_MATH__
    vfloat32m1_t special;
    vbool32_t specialMask;
    log_check_special_cases_f32m1(x, special, specialMask, vl);
#endif

    vfloat32m1_t y, e, z, zh, zl, p, logR, res;

    my_log_argument_reduction_f32m1(x, y, e, vl);
    my_log_second_reduction_f32m1(y, z, zh, zl, logR, vl);
    my_log_approximation_f32m1(z, p, vl);
    my_log2_reconstruction_f32m1(e, zh, zl, p, logR, res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m1(res, special, specialMask, vl);
#endif

    return res;
}

vfloat32m1_t __riscv_vlog10_f32m1(vfloat32m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m1(avl);
    x = __riscv_vmv_v_v_f32m1_tu(__riscv_vfmv_v_f_f32m1(
        (float)1.0f, __riscv_vsetvlmax_e32m1()), x, vl);

#ifndef __FAST_MATH__
    vfloat32m1_t special;
    vbool32_t specialMask;
    log_check_special_cases_f32m1(x, special, specialMask, vl);
#endif

    constexpr float invLog10h = 0x1.bcb7b2p-2f, invLog10l = -0x1.5b235ep-27f;
    vfloat32m1_t y, e, z, zh, zl, p, logR, res, resl;

    my_log_argument_reduction_f32m1(x, y, e, vl);
    my_log_second_reduction_f32m1(y, z, zh, zl, logR, vl);
    my_log_approximation_f32m1(z, p, vl);
    my_log_reconstruction_hl_f32m1(e, zh, zl, p, logR, res, resl, vl);

    res = mul21_f32m1(res, resl, invLog10h, invLog10l, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m1(res, special, specialMask, vl);
#endif

    return res;
}

vfloat32m1_t __riscv_vlog1p_f32m1(vfloat32m1_t xm1, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m1(avl);
    xm1 = __riscv_vmv_v_v_f32m1_tu(__riscv_vfmv_v_f_f32m1(
        (float)0.0f, __riscv_vsetvlmax_e32m1()), xm1, vl);

#ifndef __FAST_MATH__
    vfloat32m1_t special;
    vbool32_t specialMask;
    log_check_special_cases_f32m1(xm1, special, specialMask, vl, (float)-1.0f, (float)0.0f);
#endif

    vfloat32m1_t xm1c, ef, y, p, logR, z, zh, zl, res;
    vint32m1_t e;
    vbool32_t linearMask;

    my_log1p_process_linear_f32m1(xm1, xm1c, linearMask, vl);
    my_log1p_argument_reduction_f32m1(__riscv_vfadd_vf_f32m1(xm1c, (float)1.0f, vl), y, e, ef, vl);
    my_log1p_second_reduction_f32m1(xm1, y, e, z, zh, zl, logR, vl);
    my_log_approximation_f32m1(z, p, vl);
    my_log1p_reconstruction_h_f32m1(ef, zh, zl, p, logR, res, vl);
    res = __riscv_vmerge_vvm_f32m1(res, xm1, linearMask, vl);

#ifndef __FAST_MATH__
    my_log1p_set_fe_underflow_f32m1(res, vl);
    res = __riscv_vmerge_vvm_f32m1(res, special, specialMask, vl);
#endif

    return res;
}

// ==================================================================================================
// vfloat32m2_t
// ==================================================================================================

vfloat32m2_t __riscv_vlog_f32m2(vfloat32m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m2(avl);
    x = __riscv_vmv_v_v_f32m2_tu(__riscv_vfmv_v_f_f32m2(
        (float)1.0f, __riscv_vsetvlmax_e32m2()), x, vl);

#ifndef __FAST_MATH__
    vfloat32m2_t special;
    vbool16_t specialMask;
    log_check_special_cases_f32m2(x, special, specialMask, vl);
#endif

    vfloat32m2_t y, e, z, zh, zl, p, logR, res;

    my_log_argument_reduction_f32m2(x, y, e, vl);
    my_log_second_reduction_f32m2(y, z, zh, zl, logR, vl);
    my_log_approximation_f32m2(z, p, vl);
    my_log_reconstruction_h_f32m2(e, zh, zl, p, logR, res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m2(res, special, specialMask, vl);
#endif

    return res;
}

vfloat32m2_t __riscv_vlog2_f32m2(vfloat32m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m2(avl);
    x = __riscv_vmv_v_v_f32m2_tu(__riscv_vfmv_v_f_f32m2(
        (float)1.0f, __riscv_vsetvlmax_e32m2()), x, vl);

#ifndef __FAST_MATH__
    vfloat32m2_t special;
    vbool16_t specialMask;
    log_check_special_cases_f32m2(x, special, specialMask, vl);
#endif

    vfloat32m2_t y, e, z, zh, zl, p, logR, res;

    my_log_argument_reduction_f32m2(x, y, e, vl);
    my_log_second_reduction_f32m2(y, z, zh, zl, logR, vl);
    my_log_approximation_f32m2(z, p, vl);
    my_log2_reconstruction_f32m2(e, zh, zl, p, logR, res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m2(res, special, specialMask, vl);
#endif

    return res;
}

vfloat32m2_t __riscv_vlog10_f32m2(vfloat32m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m2(avl);
    x = __riscv_vmv_v_v_f32m2_tu(__riscv_vfmv_v_f_f32m2(
        (float)1.0f, __riscv_vsetvlmax_e32m2()), x, vl);

#ifndef __FAST_MATH__
    vfloat32m2_t special;
    vbool16_t specialMask;
    log_check_special_cases_f32m2(x, special, specialMask, vl);
#endif

    constexpr float invLog10h = 0x1.bcb7b2p-2f, invLog10l = -0x1.5b235ep-27f;
    vfloat32m2_t y, e, z, zh, zl, p, logR, res, resl;

    my_log_argument_reduction_f32m2(x, y, e, vl);
    my_log_second_reduction_f32m2(y, z, zh, zl, logR, vl);
    my_log_approximation_f32m2(z, p, vl);
    my_log_reconstruction_hl_f32m2(e, zh, zl, p, logR, res, resl, vl);

    res = mul21_f32m2(res, resl, invLog10h, invLog10l, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m2(res, special, specialMask, vl);
#endif

    return res;
}

vfloat32m2_t __riscv_vlog1p_f32m2(vfloat32m2_t xm1, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m2(avl);
    xm1 = __riscv_vmv_v_v_f32m2_tu(__riscv_vfmv_v_f_f32m2(
        (float)0.0f, __riscv_vsetvlmax_e32m2()), xm1, vl);

#ifndef __FAST_MATH__
    vfloat32m2_t special;
    vbool16_t specialMask;
    log_check_special_cases_f32m2(xm1, special, specialMask, vl, (float)-1.0f, (float)0.0f);
#endif

    vfloat32m2_t xm1c, ef, y, p, logR, z, zh, zl, res;
    vint32m2_t e;
    vbool16_t linearMask;

    my_log1p_process_linear_f32m2(xm1, xm1c, linearMask, vl);
    my_log1p_argument_reduction_f32m2(__riscv_vfadd_vf_f32m2(xm1c, (float)1.0f, vl), y, e, ef, vl);
    my_log1p_second_reduction_f32m2(xm1, y, e, z, zh, zl, logR, vl);
    my_log_approximation_f32m2(z, p, vl);
    my_log1p_reconstruction_h_f32m2(ef, zh, zl, p, logR, res, vl);
    res = __riscv_vmerge_vvm_f32m2(res, xm1, linearMask, vl);

#ifndef __FAST_MATH__
    my_log1p_set_fe_underflow_f32m2(res, vl);
    res = __riscv_vmerge_vvm_f32m2(res, special, specialMask, vl);
#endif

    return res;
}

// ==================================================================================================
// vfloat32m4_t
// ==================================================================================================

vfloat32m4_t __riscv_vlog_f32m4(vfloat32m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m4(avl);
    x = __riscv_vmv_v_v_f32m4_tu(__riscv_vfmv_v_f_f32m4(
        (float)1.0f, __riscv_vsetvlmax_e32m4()), x, vl);

#ifndef __FAST_MATH__
    vfloat32m4_t special;
    vbool8_t specialMask;
    log_check_special_cases_f32m4(x, special, specialMask, vl);
#endif

    vfloat32m4_t y, e, z, zh, zl, p, logR, res;

    my_log_argument_reduction_f32m4(x, y, e, vl);
    my_log_second_reduction_f32m4(y, z, zh, zl, logR, vl);
    my_log_approximation_f32m4(z, p, vl);
    my_log_reconstruction_h_f32m4(e, zh, zl, p, logR, res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m4(res, special, specialMask, vl);
#endif

    return res;
}

vfloat32m4_t __riscv_vlog2_f32m4(vfloat32m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m4(avl);
    x = __riscv_vmv_v_v_f32m4_tu(__riscv_vfmv_v_f_f32m4(
        (float)1.0f, __riscv_vsetvlmax_e32m4()), x, vl);

#ifndef __FAST_MATH__
    vfloat32m4_t special;
    vbool8_t specialMask;
    log_check_special_cases_f32m4(x, special, specialMask, vl);
#endif

    vfloat32m4_t y, e, z, zh, zl, p, logR, res;

    my_log_argument_reduction_f32m4(x, y, e, vl);
    my_log_second_reduction_f32m4(y, z, zh, zl, logR, vl);
    my_log_approximation_f32m4(z, p, vl);
    my_log2_reconstruction_f32m4(e, zh, zl, p, logR, res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m4(res, special, specialMask, vl);
#endif

    return res;
}

vfloat32m4_t __riscv_vlog10_f32m4(vfloat32m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m4(avl);
    x = __riscv_vmv_v_v_f32m4_tu(__riscv_vfmv_v_f_f32m4(
        (float)1.0f, __riscv_vsetvlmax_e32m4()), x, vl);

#ifndef __FAST_MATH__
    vfloat32m4_t special;
    vbool8_t specialMask;
    log_check_special_cases_f32m4(x, special, specialMask, vl);
#endif

    constexpr float invLog10h = 0x1.bcb7b2p-2f, invLog10l = -0x1.5b235ep-27f;
    vfloat32m4_t y, e, z, zh, zl, p, logR, res, resl;

    my_log_argument_reduction_f32m4(x, y, e, vl);
    my_log_second_reduction_f32m4(y, z, zh, zl, logR, vl);
    my_log_approximation_f32m4(z, p, vl);
    my_log_reconstruction_hl_f32m4(e, zh, zl, p, logR, res, resl, vl);

    res = mul21_f32m4(res, resl, invLog10h, invLog10l, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m4(res, special, specialMask, vl);
#endif

    return res;
}

vfloat32m4_t __riscv_vlog1p_f32m4(vfloat32m4_t xm1, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m4(avl);
    xm1 = __riscv_vmv_v_v_f32m4_tu(__riscv_vfmv_v_f_f32m4(
        (float)0.0f, __riscv_vsetvlmax_e32m4()), xm1, vl);

#ifndef __FAST_MATH__
    vfloat32m4_t special;
    vbool8_t specialMask;
    log_check_special_cases_f32m4(xm1, special, specialMask, vl, (float)-1.0f, (float)0.0f);
#endif

    vfloat32m4_t xm1c, ef, y, p, logR, z, zh, zl, res;
    vint32m4_t e;
    vbool8_t linearMask;

    my_log1p_process_linear_f32m4(xm1, xm1c, linearMask, vl);
    my_log1p_argument_reduction_f32m4(__riscv_vfadd_vf_f32m4(xm1c, (float)1.0f, vl), y, e, ef, vl);
    my_log1p_second_reduction_f32m4(xm1, y, e, z, zh, zl, logR, vl);
    my_log_approximation_f32m4(z, p, vl);
    my_log1p_reconstruction_h_f32m4(ef, zh, zl, p, logR, res, vl);
    res = __riscv_vmerge_vvm_f32m4(res, xm1, linearMask, vl);

#ifndef __FAST_MATH__
    my_log1p_set_fe_underflow_f32m4(res, vl);
    res = __riscv_vmerge_vvm_f32m4(res, special, specialMask, vl);
#endif

    return res;
}

// ==================================================================================================
// vfloat32m8_t
// ==================================================================================================

vfloat32m8_t __riscv_vlog_f32m8(vfloat32m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m8(avl);
    x = __riscv_vmv_v_v_f32m8_tu(__riscv_vfmv_v_f_f32m8(
        (float)1.0f, __riscv_vsetvlmax_e32m8()), x, vl);

    vfloat32m8_t res;
    vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1 = __riscv_vlog_f32m4(x1, vl);
    res = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e32m4(avl - vl);
        x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1 = __riscv_vlog_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}

vfloat32m8_t __riscv_vlog2_f32m8(vfloat32m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m8(avl);
    x = __riscv_vmv_v_v_f32m8_tu(__riscv_vfmv_v_f_f32m8(
        (float)1.0f, __riscv_vsetvlmax_e32m8()), x, vl);

    vfloat32m8_t res;
    vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1 = __riscv_vlog2_f32m4(x1, vl);
    res = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e32m4(avl - vl);
        x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1 = __riscv_vlog2_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}

vfloat32m8_t __riscv_vlog10_f32m8(vfloat32m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m8(avl);
    x = __riscv_vmv_v_v_f32m8_tu(__riscv_vfmv_v_f_f32m8(
        (float)1.0f, __riscv_vsetvlmax_e32m8()), x, vl);

    vfloat32m8_t res;
    vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1 = __riscv_vlog10_f32m4(x1, vl);
    res = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e32m4(avl - vl);
        x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1 = __riscv_vlog10_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}

vfloat32m8_t __riscv_vlog1p_f32m8(vfloat32m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m8(avl);
    x = __riscv_vmv_v_v_f32m8_tu(__riscv_vfmv_v_f_f32m8(
        (float)0.0f, __riscv_vsetvlmax_e32m8()), x, vl);

    vfloat32m8_t res;
    vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1 = __riscv_vlog1p_f32m4(x1, vl);
    res = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e32m4(avl - vl);
        x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1 = __riscv_vlog1p_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}

#endif /* __riscv_v_intrinsic */
