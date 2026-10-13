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
 *   File:  expm1.cpp                                    *
 *   Contains: intrinsic function expm1 for f64,f32,f16  *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the e-base exponent minus 1 of input vector  *
 *                                                       *
 * Algorithm:                                            *
 *    1) Argument reduction to a small interval near 0   *
 *    2) Additional reduction using the look-up table    *
 *       of size 2^k (k: f64 - 6, f32 - 4, f16 - 3)      *
 *    3) Polynomial degrees: f64 - 6, f32 - 4, f16 - 3   *
 *    4) Reconstruction of the result, accurate          *
 *       subtraction of 1                                *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

#include <cstdint>
#include <cfloat>
#include <cmath>

#include "dexp.inl"
#include "sexp.inl"


vfloat64m1_t __riscv_vexpm1_f64m1(vfloat64m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m1(avl);
    x = __riscv_vmv_v_v_f64m1_tu(__riscv_vfmv_v_f_f64m1(
        0.0, __riscv_vsetvlmax_e64m1()), x, vl);
        
    vfloat64m1_t special = x;
    vbool64_t specialMask = __riscv_vmclr_m_b64(vl);
    check_special_cases_f64m1(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F64,
        EXPM1_UNDERFLOW_THRESHOLD_F64, EXPM1_UNDERFLOW_VALUE_F64, false, vl);

    vfloat64m1_t xc, res, yh, th, tl, pm1h, pm1l;
    vuint64m1_t ei, fi;
    vbool64_t linearMask;

    process_linear_f64m1(x, xc, linearMask, vl);
    do_exp_argument_reduction_h_f64m1(xc, yh, ei, fi, vl);
    get_table_values_hl_f64m1(fi, th, tl, vl);
    calculate_expm1_polynom_hl12_f64m1(yh, pm1h, pm1l, vl);
    reconstruct_expm1_f64m1(th, tl, pm1h, pm1l, ei, res, vl);
    res = __riscv_vmerge_vvm_f64m1(res, x, linearMask, vl);

    set_fe_underflow_f64m1(res, vl);
    res = __riscv_vmerge_vvm_f64m1(res, special, specialMask, vl);

    return res;
}

vfloat64m2_t __riscv_vexpm1_f64m2(vfloat64m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m2(avl);
    x = __riscv_vmv_v_v_f64m2_tu(__riscv_vfmv_v_f_f64m2(
        0.0, __riscv_vsetvlmax_e64m2()), x, vl);

    vfloat64m2_t special = x;
    vbool32_t specialMask = __riscv_vmclr_m_b32(vl);
    check_special_cases_f64m2(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F64,
        EXPM1_UNDERFLOW_THRESHOLD_F64, EXPM1_UNDERFLOW_VALUE_F64, false, vl);

    vfloat64m2_t xc, res, yh, th, tl, pm1h, pm1l;
    vuint64m2_t ei, fi;
    vbool32_t linearMask;

    process_linear_f64m2(x, xc, linearMask, vl);
    do_exp_argument_reduction_h_f64m2(xc, yh, ei, fi, vl);
    get_table_values_hl_f64m2(fi, th, tl, vl);
    calculate_expm1_polynom_hl12_f64m2(yh, pm1h, pm1l, vl);
    reconstruct_expm1_f64m2(th, tl, pm1h, pm1l, ei, res, vl);
    res = __riscv_vmerge_vvm_f64m2(res, x, linearMask, vl);

    set_fe_underflow_f64m2(res, vl);
    res = __riscv_vmerge_vvm_f64m2(res, special, specialMask, vl);

    return res;
}

vfloat64m4_t __riscv_vexpm1_f64m4(vfloat64m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m4(avl);
    x = __riscv_vmv_v_v_f64m4_tu(__riscv_vfmv_v_f_f64m4(
        0.0, __riscv_vsetvlmax_e64m4()), x, vl);

    vfloat64m4_t special = x;
    vbool16_t specialMask = __riscv_vmclr_m_b16(vl);
    check_special_cases_f64m4(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F64,
        EXPM1_UNDERFLOW_THRESHOLD_F64, EXPM1_UNDERFLOW_VALUE_F64, false, vl);

    vfloat64m4_t xc, res, yh, th, tl, pm1h, pm1l;
    vuint64m4_t ei, fi;
    vbool16_t linearMask;

    process_linear_f64m4(x, xc, linearMask, vl);
    do_exp_argument_reduction_h_f64m4(xc, yh, ei, fi, vl);
    get_table_values_hl_f64m4(fi, th, tl, vl);
    calculate_expm1_polynom_hl12_f64m4(yh, pm1h, pm1l, vl);
    reconstruct_expm1_f64m4(th, tl, pm1h, pm1l, ei, res, vl);
    res = __riscv_vmerge_vvm_f64m4(res, x, linearMask, vl);

    set_fe_underflow_f64m4(res, vl);
    res = __riscv_vmerge_vvm_f64m4(res, special, specialMask, vl);

    return res;
}

vfloat64m8_t __riscv_vexpm1_f64m8(vfloat64m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m8(avl);
    x = __riscv_vmv_v_v_f64m8_tu(__riscv_vfmv_v_f_f64m8(
        0.0, __riscv_vsetvlmax_e64m8()), x, vl);

    vfloat64m8_t res;
    vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1 = __riscv_vexpm1_f64m4(x1, vl);
    res = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e64m4(avl - vl);
        x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1 = __riscv_vexpm1_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}


vfloat32m1_t __riscv_vexpm1_f32m1(vfloat32m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m1(avl);
    x = __riscv_vmv_v_v_f32m1_tu(__riscv_vfmv_v_f_f32m1(
        0.0f, __riscv_vsetvlmax_e32m1()), x, vl);

    vfloat32m1_t special = x;
    vbool32_t specialMask = __riscv_vmclr_m_b32(vl);
    check_special_cases_f32m1(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F32,
        EXPM1_UNDERFLOW_THRESHOLD_F32, EXPM1_UNDERFLOW_VALUE_F32, false, vl);

    vfloat32m1_t xc, res, yh, yl, th, tl, pm1h, pm1l;
    vuint32m1_t ei, fi;
    vbool32_t linearMask;

    process_linear_f32m1(x, xc, linearMask, vl);
    do_exp_argument_reduction_hl_f32m1(xc, yh, yl, ei, fi, vl);
    get_table_values_hl_f32m1(fi, th, tl, vl);
    calculate_expm1_polynom_hl_f32m1(yh, yl, pm1h, pm1l, vl);
    reconstruct_expm1_f32m1(th, tl, pm1h, pm1l, ei, res, vl);
    res = __riscv_vmerge_vvm_f32m1(res, x, linearMask, vl);

    set_fe_underflow_f32m1(res, vl);
    res = __riscv_vmerge_vvm_f32m1(res, special, specialMask, vl);

    return res;
}

vfloat32m2_t __riscv_vexpm1_f32m2(vfloat32m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m2(avl);
    x = __riscv_vmv_v_v_f32m2_tu(__riscv_vfmv_v_f_f32m2(
        0.0f, __riscv_vsetvlmax_e32m2()), x, vl);

    vfloat32m2_t special = x;
    vbool16_t specialMask = __riscv_vmclr_m_b16(vl);
    check_special_cases_f32m2(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F32,
        EXPM1_UNDERFLOW_THRESHOLD_F32, EXPM1_UNDERFLOW_VALUE_F32, false, vl);

    vfloat32m2_t xc, res, yh, yl, th, tl, pm1h, pm1l;
    vuint32m2_t ei, fi;
    vbool16_t linearMask;

    process_linear_f32m2(x, xc, linearMask, vl);
    do_exp_argument_reduction_hl_f32m2(xc, yh, yl, ei, fi, vl);
    get_table_values_hl_f32m2(fi, th, tl, vl);
    calculate_expm1_polynom_hl_f32m2(yh, yl, pm1h, pm1l, vl);
    reconstruct_expm1_f32m2(th, tl, pm1h, pm1l, ei, res, vl);
    res = __riscv_vmerge_vvm_f32m2(res, x, linearMask, vl);

    set_fe_underflow_f32m2(res, vl);
    res = __riscv_vmerge_vvm_f32m2(res, special, specialMask, vl);

    return res;
}

vfloat32m4_t __riscv_vexpm1_f32m4(vfloat32m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m4(avl);
    x = __riscv_vmv_v_v_f32m4_tu(__riscv_vfmv_v_f_f32m4(
        0.0f, __riscv_vsetvlmax_e32m4()), x, vl);

    vfloat32m4_t special = x;
    vbool8_t specialMask = __riscv_vmclr_m_b8(vl);
    check_special_cases_f32m4(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F32,
        EXPM1_UNDERFLOW_THRESHOLD_F32, EXPM1_UNDERFLOW_VALUE_F32, false, vl);

    vfloat32m4_t xc, res, yh, yl, th, tl, pm1h, pm1l;
    vuint32m4_t ei, fi;
    vbool8_t linearMask;

    process_linear_f32m4(x, xc, linearMask, vl);
    do_exp_argument_reduction_hl_f32m4(xc, yh, yl, ei, fi, vl);
    get_table_values_hl_f32m4(fi, th, tl, vl);
    calculate_expm1_polynom_hl_f32m4(yh, yl, pm1h, pm1l, vl);
    reconstruct_expm1_f32m4(th, tl, pm1h, pm1l, ei, res, vl);
    res = __riscv_vmerge_vvm_f32m4(res, x, linearMask, vl);

    set_fe_underflow_f32m4(res, vl);
    res = __riscv_vmerge_vvm_f32m4(res, special, specialMask, vl);

    return res;
}

vfloat32m8_t __riscv_vexpm1_f32m8(vfloat32m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m8(avl);
    x = __riscv_vmv_v_v_f32m8_tu(__riscv_vfmv_v_f_f32m8(
        0.0f, __riscv_vsetvlmax_e32m8()), x, vl);

    vfloat32m8_t res;
    vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1 = __riscv_vexpm1_f32m4(x1, vl);
    res = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e32m4(avl - vl);
        x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1 = __riscv_vexpm1_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}


#ifdef __riscv_zvfh

#include "hexp.inl"

vfloat16m1_t __riscv_vexpm1_f16m1(vfloat16m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m1(avl);
    x = __riscv_vmv_v_v_f16m1_tu(__riscv_vfmv_v_f_f16m1(
        0.0f16, __riscv_vsetvlmax_e16m1()), x, vl);

    vfloat16m1_t special = x;
    vbool16_t specialMask = __riscv_vmclr_m_b16(vl);
    check_special_cases_f16m1(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F16,
        EXPM1_UNDERFLOW_THRESHOLD_F16, EXPM1_UNDERFLOW_VALUE_F16, false, vl);

    vfloat16m1_t xc, res, yh, yl, th, tl, pm1h, pm1l;
    vuint16m1_t ei, fi;
    vbool16_t linearMask;

    process_linear_f16m1(x, xc, linearMask, vl);
    do_exp_argument_reduction_hl_f16m1(xc, yh, yl, ei, fi, vl);
    get_table_values_hl_f16m1(fi, th, tl, vl);
    calculate_expm1_polynom_hl_f16m1(yh, yl, pm1h, pm1l, vl);
    reconstruct_expm1_f16m1(th, tl, pm1h, pm1l, ei, res, vl);
    res = __riscv_vmerge_vvm_f16m1(res, x, linearMask, vl);

    set_fe_underflow_f16m1(res, vl);
    res = __riscv_vmerge_vvm_f16m1(res, special, specialMask, vl);

    return res;
}

vfloat16m2_t __riscv_vexpm1_f16m2(vfloat16m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m2(avl);
    x = __riscv_vmv_v_v_f16m2_tu(__riscv_vfmv_v_f_f16m2(
        0.0f16, __riscv_vsetvlmax_e16m2()), x, vl);

    vfloat16m2_t special = x;
    vbool8_t specialMask = __riscv_vmclr_m_b8(vl);
    check_special_cases_f16m2(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F16,
        EXPM1_UNDERFLOW_THRESHOLD_F16, EXPM1_UNDERFLOW_VALUE_F16, false, vl);

    vfloat16m2_t xc, res, yh, yl, th, tl, pm1h, pm1l;
    vuint16m2_t ei, fi;
    vbool8_t linearMask;

    process_linear_f16m2(x, xc, linearMask, vl);
    do_exp_argument_reduction_hl_f16m2(xc, yh, yl, ei, fi, vl);
    get_table_values_hl_f16m2(fi, th, tl, vl);
    calculate_expm1_polynom_hl_f16m2(yh, yl, pm1h, pm1l, vl);
    reconstruct_expm1_f16m2(th, tl, pm1h, pm1l, ei, res, vl);
    res = __riscv_vmerge_vvm_f16m2(res, x, linearMask, vl);

    set_fe_underflow_f16m2(res, vl);
    res = __riscv_vmerge_vvm_f16m2(res, special, specialMask, vl);

    return res;
}

vfloat16m4_t __riscv_vexpm1_f16m4(vfloat16m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m4(avl);
    x = __riscv_vmv_v_v_f16m4_tu(__riscv_vfmv_v_f_f16m4(
        0.0f16, __riscv_vsetvlmax_e16m4()), x, vl);

    vfloat16m4_t special = x;
    vbool4_t specialMask = __riscv_vmclr_m_b4(vl);
    check_special_cases_f16m4(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F16,
        EXPM1_UNDERFLOW_THRESHOLD_F16, EXPM1_UNDERFLOW_VALUE_F16, false, vl);

    vfloat16m4_t xc, res, yh, yl, th, tl, pm1h, pm1l;
    vuint16m4_t ei, fi;
    vbool4_t linearMask;

    process_linear_f16m4(x, xc, linearMask, vl);
    do_exp_argument_reduction_hl_f16m4(xc, yh, yl, ei, fi, vl);
    get_table_values_hl_f16m4(fi, th, tl, vl);
    calculate_expm1_polynom_hl_f16m4(yh, yl, pm1h, pm1l, vl);
    reconstruct_expm1_f16m4(th, tl, pm1h, pm1l, ei, res, vl);
    res = __riscv_vmerge_vvm_f16m4(res, x, linearMask, vl);

    set_fe_underflow_f16m4(res, vl);
    res = __riscv_vmerge_vvm_f16m4(res, special, specialMask, vl);

    return res;
}

vfloat16m8_t __riscv_vexpm1_f16m8(vfloat16m8_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m8(avl);
    x = __riscv_vmv_v_v_f16m8_tu(__riscv_vfmv_v_f_f16m8(
        0.0f16, __riscv_vsetvlmax_e16m8()), x, vl);

    vfloat16m8_t res;
    vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1 = __riscv_vexpm1_f16m4(x1, vl);
    res = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e16m4(avl - vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1 = __riscv_vexpm1_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}

#endif /* __riscv_zvfh */

#endif /* __riscv_v_intrinsic */
