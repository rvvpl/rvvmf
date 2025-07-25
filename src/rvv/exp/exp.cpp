/* 
 *========================================================
 * Copyright (c) The Lobachevsky State University of 
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
 *   File:  exp.cpp                                      *
 *   Contains: intrinsic function exp for f64, f32, f16  *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Computes the e-base exponent of input vector V        *
 *                                                       *
 * Algorithm:                                            *
 *    1) Argument reduction to a small interval near 0   *
 *    2) Additional reduction using the look-up table    *
 *       of size 2^k (k: f64 - 6, f32 - 4, f16 - 3)      *
 *    3) Polynomial degrees: f64 - 6, f32 - 4, f16 - 3   *
 *    4) Reconstruction of the result                    *
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


vfloat64m1_t __riscv_vexp_f64m1(vfloat64m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m1(avl);
    
#ifndef __FAST_MATH__
    const double zeroThreshold = EXP_ZERO_THRESHOLD_F64;
    vfloat64m1_t special;
    vbool64_t specialMask;
    check_special_cases_f64m1(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F64, vl);
#else
    const double zeroThreshold = EXP_SUBNORMAL_THRESHOLD_F64;    
#endif

    vfloat64m1_t res, yh, th, tl, pm1h, pm1l;
    vuint64m1_t ei, fi;
    
    do_exp_argument_reduction_h_f64m1(x, yh, ei, fi, vl);
    get_table_values_hl_f64m1(fi, th, tl, vl);
    calculate_exp_polynom_hl12_f64m1(yh, pm1h, pm1l, vl);
    reconstruct_exp_hl_hl_f64m1(x, ei, th, tl, pm1h, pm1l, res, EXP_SUBNORMAL_THRESHOLD_F64, vl);
    update_underflow_f64m1(x, res, zeroThreshold, EXP_UNDERFLOW_VALUE_F64, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m1(res, special, specialMask, vl);
#endif

    return res;
}

vfloat64m2_t __riscv_vexp_f64m2(vfloat64m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m2(avl);
    
#ifndef __FAST_MATH__
    const double zeroThreshold = EXP_ZERO_THRESHOLD_F64;
    vfloat64m2_t special;
    vbool32_t specialMask;
    check_special_cases_f64m2(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F64, vl);
#else
    const double zeroThreshold = EXP_SUBNORMAL_THRESHOLD_F64;    
#endif

    vfloat64m2_t res, yh, th, tl, pm1h, pm1l;
    vuint64m2_t ei, fi;
    
    do_exp_argument_reduction_h_f64m2(x, yh, ei, fi, vl);
    get_table_values_hl_f64m2(fi, th, tl, vl);
    calculate_exp_polynom_hl12_f64m2(yh, pm1h, pm1l, vl);
    reconstruct_exp_hl_hl_f64m2(x, ei, th, tl, pm1h, pm1l, res, EXP_SUBNORMAL_THRESHOLD_F64, vl);
    update_underflow_f64m2(x, res, zeroThreshold, EXP_UNDERFLOW_VALUE_F64, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m2(res, special, specialMask, vl);
#endif

    return res;
}

vfloat64m4_t __riscv_vexp_f64m4(vfloat64m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e64m4(avl);
    
#ifndef __FAST_MATH__
    const double zeroThreshold = EXP_ZERO_THRESHOLD_F64;
    vfloat64m4_t special;
    vbool16_t specialMask;
    check_special_cases_f64m4(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F64, vl);
#else
    const double zeroThreshold = EXP_SUBNORMAL_THRESHOLD_F64;    
#endif

    vfloat64m4_t res, yh, th, tl, pm1h, pm1l;
    vuint64m4_t ei, fi;
    
    do_exp_argument_reduction_h_f64m4(x, yh, ei, fi, vl);
    get_table_values_hl_f64m4(fi, th, tl, vl);
    calculate_exp_polynom_hl12_f64m4(yh, pm1h, pm1l, vl);
    reconstruct_exp_hl_hl_f64m4(x, ei, th, tl, pm1h, pm1l, res, EXP_SUBNORMAL_THRESHOLD_F64, vl);
    update_underflow_f64m4(x, res, zeroThreshold, EXP_UNDERFLOW_VALUE_F64, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f64m4(res, special, specialMask, vl);
#endif

    return res;
}

vfloat64m8_t __riscv_vexp_f64m8(vfloat64m8_t x, size_t avl)
{
    vfloat64m8_t res;
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1 = __riscv_vexp_f64m4(x1, vl);
    res = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e64m4(avl - vl);
        x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1 = __riscv_vexp_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}


vfloat32m1_t __riscv_vexp_f32m1(vfloat32m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m1(avl);
    
#ifndef __FAST_MATH__
    const float zeroThreshold = EXP_ZERO_THRESHOLD_F32;
    vfloat32m1_t special;
    vbool32_t specialMask;
    check_special_cases_f32m1(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F32, vl);
#else
    const float zeroThreshold = EXP_SUBNORMAL_THRESHOLD_F32;    
#endif

    vfloat32m1_t res, yh, yl, th, tl, pm1h, pm1l;
    vuint32m1_t ei, fi;
    
    do_exp_argument_reduction_hl_f32m1(x, yh, yl, ei, fi, vl);
    get_table_values_hl_f32m1(fi, th, tl, vl);
    calculate_exp_polynom_hl_f32m1(yh, yl, pm1h, pm1l, vl);
    reconstruct_exp_hl_hl_f32m1(x, ei, th, tl, pm1h, pm1l, res, EXP_SUBNORMAL_THRESHOLD_F32, vl);
    update_underflow_f32m1(x, res, zeroThreshold, EXP_UNDERFLOW_VALUE_F32, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m1(res, special, specialMask, vl);
#endif

    return res;
}

vfloat32m2_t __riscv_vexp_f32m2(vfloat32m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m2(avl);
    
#ifndef __FAST_MATH__
    const float zeroThreshold = EXP_ZERO_THRESHOLD_F32;
    vfloat32m2_t special;
    vbool16_t specialMask;
    check_special_cases_f32m2(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F32, vl);
#else
    const float zeroThreshold = EXP_SUBNORMAL_THRESHOLD_F32;    
#endif

    vfloat32m2_t res, yh, yl, th, tl, pm1h, pm1l;
    vuint32m2_t ei, fi;
    
    do_exp_argument_reduction_hl_f32m2(x, yh, yl, ei, fi, vl);
    get_table_values_hl_f32m2(fi, th, tl, vl);
    calculate_exp_polynom_hl_f32m2(yh, yl, pm1h, pm1l, vl);
    reconstruct_exp_hl_hl_f32m2(x, ei, th, tl, pm1h, pm1l, res, EXP_SUBNORMAL_THRESHOLD_F32, vl);
    update_underflow_f32m2(x, res, zeroThreshold, EXP_UNDERFLOW_VALUE_F32, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m2(res, special, specialMask, vl);
#endif

    return res;
}

vfloat32m4_t __riscv_vexp_f32m4(vfloat32m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e32m4(avl);
    
#ifndef __FAST_MATH__
    const float zeroThreshold = EXP_ZERO_THRESHOLD_F32;
    vfloat32m4_t special;
    vbool8_t specialMask;
    check_special_cases_f32m4(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F32, vl);
#else
    const float zeroThreshold = EXP_SUBNORMAL_THRESHOLD_F32;    
#endif

    vfloat32m4_t res, yh, yl, th, tl, pm1h, pm1l;
    vuint32m4_t ei, fi;
    
    do_exp_argument_reduction_hl_f32m4(x, yh, yl, ei, fi, vl);
    get_table_values_hl_f32m4(fi, th, tl, vl);
    calculate_exp_polynom_hl_f32m4(yh, yl, pm1h, pm1l, vl);
    reconstruct_exp_hl_hl_f32m4(x, ei, th, tl, pm1h, pm1l, res, EXP_SUBNORMAL_THRESHOLD_F32, vl);
    update_underflow_f32m4(x, res, zeroThreshold, EXP_UNDERFLOW_VALUE_F32, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f32m4(res, special, specialMask, vl);
#endif

    return res;
}

vfloat32m8_t __riscv_vexp_f32m8(vfloat32m8_t x, size_t avl)
{
    vfloat32m8_t res;
    size_t vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t x1 = __riscv_vget_v_f32m8_f32m4(x, 0);
    x1 = __riscv_vexp_f32m4(x1, vl);
    res = __riscv_vset_v_f32m4_f32m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e32m4(avl - vl);
        x1 = __riscv_vget_v_f32m8_f32m4(x, 1);
        x1 = __riscv_vexp_f32m4(x1, vl);
        res = __riscv_vset_v_f32m4_f32m8(res, 1, x1);
    }
    return res;
}


#ifdef __riscv_zvfh

#include "hexp.inl"

vfloat16m1_t __riscv_vexp_f16m1(vfloat16m1_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m1(avl);
    
#ifndef __FAST_MATH__
    const FLOAT16_T zeroThreshold = EXP_ZERO_THRESHOLD_F16;
    vfloat16m1_t special;
    vbool16_t specialMask;
    check_special_cases_f16m1(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F16, vl);
#else
    const FLOAT16_T zeroThreshold = EXP_SUBNORMAL_THRESHOLD_F16;    
#endif

    vfloat16m1_t res, yh, yl, th, tl, pm1h, pm1l;
    vuint16m1_t ei, fi;
    
    do_exp_argument_reduction_hl_f16m1(x, yh, yl, ei, fi, vl);
    get_table_values_hl_f16m1(fi, th, tl, vl);
    calculate_exp_polynom_hl_f16m1(yh, yl, pm1h, pm1l, vl);
    reconstruct_exp_hl_hl_f16m1(x, ei, th, tl, pm1h, pm1l, res, EXP_SUBNORMAL_THRESHOLD_F16, vl);
    update_underflow_f16m1(x, res, zeroThreshold, EXP_UNDERFLOW_VALUE_F16, vl);
    set_pos_sign_f16m1(res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m1(res, special, specialMask, vl);
#endif

    return res;
}

vfloat16m2_t __riscv_vexp_f16m2(vfloat16m2_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m2(avl);
    
#ifndef __FAST_MATH__
    const FLOAT16_T zeroThreshold = EXP_ZERO_THRESHOLD_F16;
    vfloat16m2_t special;
    vbool8_t specialMask;
    check_special_cases_f16m2(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F16, vl);
#else
    const FLOAT16_T zeroThreshold = EXP_SUBNORMAL_THRESHOLD_F16;    
#endif

    vfloat16m2_t res, yh, yl, th, tl, pm1h, pm1l;
    vuint16m2_t ei, fi;
    
    do_exp_argument_reduction_hl_f16m2(x, yh, yl, ei, fi, vl);
    get_table_values_hl_f16m2(fi, th, tl, vl);
    calculate_exp_polynom_hl_f16m2(yh, yl, pm1h, pm1l, vl);
    reconstruct_exp_hl_hl_f16m2(x, ei, th, tl, pm1h, pm1l, res, EXP_SUBNORMAL_THRESHOLD_F16, vl);
    update_underflow_f16m2(x, res, zeroThreshold, EXP_UNDERFLOW_VALUE_F16, vl);
    set_pos_sign_f16m2(res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m2(res, special, specialMask, vl);
#endif

    return res;
}

vfloat16m4_t __riscv_vexp_f16m4(vfloat16m4_t x, size_t avl)
{
    size_t vl = __riscv_vsetvl_e16m4(avl);
    
#ifndef __FAST_MATH__
    const FLOAT16_T zeroThreshold = EXP_ZERO_THRESHOLD_F16;
    vfloat16m4_t special;
    vbool4_t specialMask;
    check_special_cases_f16m4(x, special, specialMask, EXP_EXPM1_OVERFLOW_THRESHOLD_F16, vl);
#else
    const FLOAT16_T zeroThreshold = EXP_SUBNORMAL_THRESHOLD_F16;    
#endif

    vfloat16m4_t res, yh, yl, th, tl, pm1h, pm1l;
    vuint16m4_t ei, fi;
    
    do_exp_argument_reduction_hl_f16m4(x, yh, yl, ei, fi, vl);
    get_table_values_hl_f16m4(fi, th, tl, vl);
    calculate_exp_polynom_hl_f16m4(yh, yl, pm1h, pm1l, vl);
    reconstruct_exp_hl_hl_f16m4(x, ei, th, tl, pm1h, pm1l, res, EXP_SUBNORMAL_THRESHOLD_F16, vl);
    update_underflow_f16m4(x, res, zeroThreshold, EXP_UNDERFLOW_VALUE_F16, vl);
    set_pos_sign_f16m4(res, vl);

#ifndef __FAST_MATH__
    res = __riscv_vmerge_vvm_f16m4(res, special, specialMask, vl);
#endif

    return res;
}

vfloat16m8_t __riscv_vexp_f16m8(vfloat16m8_t x, size_t avl)
{
    vfloat16m8_t res;
    size_t vl = __riscv_vsetvl_e16m4(avl);
    vfloat16m4_t x1 = __riscv_vget_v_f16m8_f16m4(x, 0);
    x1 = __riscv_vexp_f16m4(x1, vl);
    res = __riscv_vset_v_f16m4_f16m8(res, 0, x1);
    if (avl > vl) {
        vl = __riscv_vsetvl_e16m4(avl - vl);
        x1 = __riscv_vget_v_f16m8_f16m4(x, 1);
        x1 = __riscv_vexp_f16m4(x1, vl);
        res = __riscv_vset_v_f16m4_f16m8(res, 1, x1);
    }
    return res;
}


#endif /* __riscv_zvfh */

#endif /* __riscv_v_intrinsic */
