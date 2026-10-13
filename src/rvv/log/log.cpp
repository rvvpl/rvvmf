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
 *   File:  log.cpp                                      *
 *   Contains: intrinsic functions log, log2, log10,     *
 *             log1p for float64                         *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the logarithm of input vector V              *
 *                                                       *
 * Algorithm:                                            *
 *    1) Argument reduction to [1-a, 1+a], a = 683/2048  *
 *    2) Additional reduction using two look-up tables   *
 *       of size 2^k (k=8)                               *
 *    3) Polynomial approximation (degree = 6)           *
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

#include "dlog.data"
#include "dlog_aux.inl"

vfloat64m1_t __riscv_vlog_f64m1(vfloat64m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m1(avl);
    x = __riscv_vmv_v_v_f64m1_tu(__riscv_vfmv_v_f_f64m1(
        (double)1.0, __riscv_vsetvlmax_e64m1()), x, vl);

#ifndef __FAST_MATH__
    vfloat64m1_t special;
    vbool64_t specialMask;
    log_check_special_cases_f64m1(x, special, specialMask, vl);
#endif

    vfloat64m1_t y, e, z, zh, zl, p, logR, res;

    my_log_argument_reduction_f64m1(x, y, e, vl);
    my_log_second_reduction_f64m1(y, z, zh, zl, logR, vl);
    my_log_approximation_f64m1(z, p, vl);
    my_log_reconstruction_h_f64m1(e, zh, zl, p, logR, res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m1(res, special, specialMask, vl);
#endif

    return res;
}

vfloat64m1_t __riscv_vlog2_f64m1(vfloat64m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m1(avl);
    x = __riscv_vmv_v_v_f64m1_tu(__riscv_vfmv_v_f_f64m1(
        (double)1.0, __riscv_vsetvlmax_e64m1()), x, vl);

#ifndef __FAST_MATH__
    vfloat64m1_t special;
    vbool64_t specialMask;
    log_check_special_cases_f64m1(x, special, specialMask, vl);
#endif

    vfloat64m1_t y, e, z, zh, zl, p, logR, res;

    my_log_argument_reduction_f64m1(x, y, e, vl);
    my_log_second_reduction_f64m1(y, z, zh, zl, logR, vl);
    my_log_approximation_f64m1(z, p, vl);
    my_log2_reconstruction_f64m1(e, zh, zl, p, logR, res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m1(res, special, specialMask, vl);
#endif

    return res;
}

vfloat64m1_t __riscv_vlog10_f64m1(vfloat64m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m1(avl);
    x = __riscv_vmv_v_v_f64m1_tu(__riscv_vfmv_v_f_f64m1(
        (double)1.0, __riscv_vsetvlmax_e64m1()), x, vl);

#ifndef __FAST_MATH__
    vfloat64m1_t special;
    vbool64_t specialMask;
    log_check_special_cases_f64m1(x, special, specialMask, vl);
#endif

    constexpr double invLog10h = 0x1.bcb7b1526e50ep-2, invLog10l = 0x1.95355baaafad3p-57;
    vfloat64m1_t y, e, z, zh, zl, p, logR, res, resl;

    my_log_argument_reduction_f64m1(x, y, e, vl);
    my_log_second_reduction_f64m1(y, z, zh, zl, logR, vl);
    my_log_approximation_f64m1(z, p, vl);
    my_log_reconstruction_hl_f64m1(e, zh, zl, p, logR, res, resl, vl);

    res = mul21_f64m1(res, resl, invLog10h, invLog10l, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m1(res, special, specialMask, vl);
#endif

    return res;
}

vfloat64m1_t __riscv_vlog1p_f64m1(vfloat64m1_t xm1, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m1(avl);
    xm1 = __riscv_vmv_v_v_f64m1_tu(__riscv_vfmv_v_f_f64m1(
        (double)0.0, __riscv_vsetvlmax_e64m1()), xm1, vl);

#ifndef __FAST_MATH__
    vfloat64m1_t special;
    vbool64_t specialMask;
    log_check_special_cases_f64m1(xm1, special, specialMask, vl, (double)-1.0, (double)0.0);
#endif

    vfloat64m1_t xm1c, ef, y, p, logR, z, zh, zl, res;
    vint64m1_t e;
    vbool64_t linearMask;

    my_log1p_process_linear_f64m1(xm1, xm1c, linearMask, vl);
    my_log1p_argument_reduction_f64m1(__riscv_vfadd_vf_f64m1(xm1c, (double)1.0, vl), y, e, ef, vl);
    my_log1p_second_reduction_f64m1(xm1, y, e, z, zh, zl, logR, vl);
    my_log_approximation_f64m1(z, p, vl);

    my_log_reconstruction_h_f64m1(ef, zh, zl, p, logR, res, vl);
    res = __riscv_vmerge_vvm_f64m1(res, xm1, linearMask, vl);

#ifndef __FAST_MATH__
    my_log1p_set_fe_underflow_f64m1(res, vl);
    res = __riscv_vmerge_vvm_f64m1(res, special, specialMask, vl);
#endif

    return res;
}

// ==================================================================================================
// vfloat64m2_t
// ==================================================================================================

vfloat64m2_t __riscv_vlog_f64m2(vfloat64m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m2(avl);
    x = __riscv_vmv_v_v_f64m2_tu(__riscv_vfmv_v_f_f64m2(
        (double)1.0, __riscv_vsetvlmax_e64m2()), x, vl);

#ifndef __FAST_MATH__
    vfloat64m2_t special;
    vbool32_t specialMask;
    log_check_special_cases_f64m2(x, special, specialMask, vl);
#endif

    vfloat64m2_t y, e, z, zh, zl, p, logR, res;

    my_log_argument_reduction_f64m2(x, y, e, vl);
    my_log_second_reduction_f64m2(y, z, zh, zl, logR, vl);
    my_log_approximation_f64m2(z, p, vl);
    my_log_reconstruction_h_f64m2(e, zh, zl, p, logR, res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m2(res, special, specialMask, vl);
#endif

    return res;
}

vfloat64m2_t __riscv_vlog2_f64m2(vfloat64m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m2(avl);
    x = __riscv_vmv_v_v_f64m2_tu(__riscv_vfmv_v_f_f64m2(
        (double)1.0, __riscv_vsetvlmax_e64m2()), x, vl);

#ifndef __FAST_MATH__
    vfloat64m2_t special;
    vbool32_t specialMask;
    log_check_special_cases_f64m2(x, special, specialMask, vl);
#endif

    vfloat64m2_t y, e, z, zh, zl, p, logR, res;

    my_log_argument_reduction_f64m2(x, y, e, vl);
    my_log_second_reduction_f64m2(y, z, zh, zl, logR, vl);
    my_log_approximation_f64m2(z, p, vl);
    my_log2_reconstruction_f64m2(e, zh, zl, p, logR, res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m2(res, special, specialMask, vl);
#endif

    return res;
}

vfloat64m2_t __riscv_vlog10_f64m2(vfloat64m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m2(avl);
    x = __riscv_vmv_v_v_f64m2_tu(__riscv_vfmv_v_f_f64m2(
        (double)1.0, __riscv_vsetvlmax_e64m2()), x, vl);

#ifndef __FAST_MATH__
    vfloat64m2_t special;
    vbool32_t specialMask;
    log_check_special_cases_f64m2(x, special, specialMask, vl);
#endif

    constexpr double invLog10h = 0x1.bcb7b1526e50ep-2, invLog10l = 0x1.95355baaafad3p-57;
    vfloat64m2_t y, e, z, zh, zl, p, logR, res, resl;

    my_log_argument_reduction_f64m2(x, y, e, vl);
    my_log_second_reduction_f64m2(y, z, zh, zl, logR, vl);
    my_log_approximation_f64m2(z, p, vl);
    my_log_reconstruction_hl_f64m2(e, zh, zl, p, logR, res, resl, vl);

    res = mul21_f64m2(res, resl, invLog10h, invLog10l, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m2(res, special, specialMask, vl);
#endif

    return res;
}

vfloat64m2_t __riscv_vlog1p_f64m2(vfloat64m2_t xm1, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m2(avl);
    xm1 = __riscv_vmv_v_v_f64m2_tu(__riscv_vfmv_v_f_f64m2(
        (double)0.0, __riscv_vsetvlmax_e64m2()), xm1, vl);

#ifndef __FAST_MATH__
    vfloat64m2_t special;
    vbool32_t specialMask;
    log_check_special_cases_f64m2(xm1, special, specialMask, vl, (double)-1.0, (double)0.0);
#endif

    vfloat64m2_t xm1c, ef, y, p, logR, z, zh, zl, res;
    vint64m2_t e;
    vbool32_t linearMask;

    my_log1p_process_linear_f64m2(xm1, xm1c, linearMask, vl);
    my_log1p_argument_reduction_f64m2(__riscv_vfadd_vf_f64m2(xm1c, (double)1.0, vl), y, e, ef, vl);
    my_log1p_second_reduction_f64m2(xm1, y, e, z, zh, zl, logR, vl);
    my_log_approximation_f64m2(z, p, vl);

    my_log_reconstruction_h_f64m2(ef, zh, zl, p, logR, res, vl);
    res = __riscv_vmerge_vvm_f64m2(res, xm1, linearMask, vl);

#ifndef __FAST_MATH__
    my_log1p_set_fe_underflow_f64m2(res, vl);
    res = __riscv_vmerge_vvm_f64m2(res, special, specialMask, vl);
#endif

    return res;
}

// ==================================================================================================
// vfloat64m4_t
// ==================================================================================================

vfloat64m4_t __riscv_vlog_f64m4(vfloat64m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m4(avl);
    x = __riscv_vmv_v_v_f64m4_tu(__riscv_vfmv_v_f_f64m4(
        (double)1.0, __riscv_vsetvlmax_e64m4()), x, vl);

#ifndef __FAST_MATH__
    vfloat64m4_t special;
    vbool16_t specialMask;
    log_check_special_cases_f64m4(x, special, specialMask, vl);
#endif

    vfloat64m4_t y, e, z, zh, zl, p, logR, res;

    my_log_argument_reduction_f64m4(x, y, e, vl);
    my_log_second_reduction_f64m4(y, z, zh, zl, logR, vl);
    my_log_approximation_f64m4(z, p, vl);
    my_log_reconstruction_h_f64m4(e, zh, zl, p, logR, res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m4(res, special, specialMask, vl);
#endif

    return res;
}

vfloat64m4_t __riscv_vlog2_f64m4(vfloat64m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m4(avl);
    x = __riscv_vmv_v_v_f64m4_tu(__riscv_vfmv_v_f_f64m4(
        (double)1.0, __riscv_vsetvlmax_e64m4()), x, vl);

#ifndef __FAST_MATH__
    vfloat64m4_t special;
    vbool16_t specialMask;
    log_check_special_cases_f64m4(x, special, specialMask, vl);
#endif

    vfloat64m4_t y, e, z, zh, zl, p, logR, res;

    my_log_argument_reduction_f64m4(x, y, e, vl);
    my_log_second_reduction_f64m4(y, z, zh, zl, logR, vl);
    my_log_approximation_f64m4(z, p, vl);
    my_log2_reconstruction_f64m4(e, zh, zl, p, logR, res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m4(res, special, specialMask, vl);
#endif

    return res;
}

vfloat64m4_t __riscv_vlog10_f64m4(vfloat64m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m4(avl);
    x = __riscv_vmv_v_v_f64m4_tu(__riscv_vfmv_v_f_f64m4(
        (double)1.0, __riscv_vsetvlmax_e64m4()), x, vl);

#ifndef __FAST_MATH__
    vfloat64m4_t special;
    vbool16_t specialMask;
    log_check_special_cases_f64m4(x, special, specialMask, vl);
#endif

    constexpr double invLog10h = 0x1.bcb7b1526e50ep-2, invLog10l = 0x1.95355baaafad3p-57;
    vfloat64m4_t y, e, z, zh, zl, p, logR, res, resl;

    my_log_argument_reduction_f64m4(x, y, e, vl);
    my_log_second_reduction_f64m4(y, z, zh, zl, logR, vl);
    my_log_approximation_f64m4(z, p, vl);
    my_log_reconstruction_hl_f64m4(e, zh, zl, p, logR, res, resl, vl);

    res = mul21_f64m4(res, resl, invLog10h, invLog10l, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m4(res, special, specialMask, vl);
#endif

    return res;
}

vfloat64m4_t __riscv_vlog1p_f64m4(vfloat64m4_t xm1, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m4(avl);
    xm1 = __riscv_vmv_v_v_f64m4_tu(__riscv_vfmv_v_f_f64m4(
        (double)0.0, __riscv_vsetvlmax_e64m4()), xm1, vl);

#ifndef __FAST_MATH__
    vfloat64m4_t special;
    vbool16_t specialMask;
    log_check_special_cases_f64m4(xm1, special, specialMask, vl, (double)-1.0, (double)0.0);
#endif

    vfloat64m4_t xm1c, ef, y, p, logR, z, zh, zl, res;
    vint64m4_t e;
    vbool16_t linearMask;

    my_log1p_process_linear_f64m4(xm1, xm1c, linearMask, vl);
    my_log1p_argument_reduction_f64m4(__riscv_vfadd_vf_f64m4(xm1c, (double)1.0, vl), y, e, ef, vl);
    my_log1p_second_reduction_f64m4(xm1, y, e, z, zh, zl, logR, vl);
    my_log_approximation_f64m4(z, p, vl);

    my_log_reconstruction_h_f64m4(ef, zh, zl, p, logR, res, vl);
    res = __riscv_vmerge_vvm_f64m4(res, xm1, linearMask, vl);

#ifndef __FAST_MATH__
    my_log1p_set_fe_underflow_f64m4(res, vl);
    res = __riscv_vmerge_vvm_f64m4(res, special, specialMask, vl);
#endif

    return res;
}

// ==================================================================================================
// vfloat64m8_t
// ==================================================================================================

vfloat64m8_t __riscv_vlog_f64m8(vfloat64m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m8(avl);
    x = __riscv_vmv_v_v_f64m8_tu(__riscv_vfmv_v_f_f64m8(
        (double)1.0, __riscv_vsetvlmax_e64m8()), x, vl);

    vfloat64m8_t res;
    vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1 = __riscv_vlog_f64m4(x1, vl);
    res = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e64m4(avl - vl);
        x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1 = __riscv_vlog_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}

vfloat64m8_t __riscv_vlog2_f64m8(vfloat64m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m8(avl);
    x = __riscv_vmv_v_v_f64m8_tu(__riscv_vfmv_v_f_f64m8(
        (double)1.0, __riscv_vsetvlmax_e64m8()), x, vl);

    vfloat64m8_t res;
    vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1 = __riscv_vlog2_f64m4(x1, vl);
    res = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e64m4(avl - vl);
        x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1 = __riscv_vlog2_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}

vfloat64m8_t __riscv_vlog10_f64m8(vfloat64m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m8(avl);
    x = __riscv_vmv_v_v_f64m8_tu(__riscv_vfmv_v_f_f64m8(
        (double)1.0, __riscv_vsetvlmax_e64m8()), x, vl);

    vfloat64m8_t res;
    vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1 = __riscv_vlog10_f64m4(x1, vl);
    res = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e64m4(avl - vl);
        x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1 = __riscv_vlog10_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}

vfloat64m8_t __riscv_vlog1p_f64m8(vfloat64m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m8(avl);
    x = __riscv_vmv_v_v_f64m8_tu(__riscv_vfmv_v_f_f64m8(
        (double)0.0, __riscv_vsetvlmax_e64m8()), x, vl);

    vfloat64m8_t res;
    vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1 = __riscv_vlog1p_f64m4(x1, vl);
    res = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e64m4(avl - vl);
        x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1 = __riscv_vlog1p_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}

#endif /* __riscv_v_intrinsic */
