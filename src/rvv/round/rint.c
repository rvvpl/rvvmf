/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025 The RVVMF Authors (Valentin Volokitin)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *  File:  rint.c                                        *
 *  Contains: intrinsic function rint, lrint, llrint     *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * These are interface functions for other functions or  *
 * standard intrinsics                                   *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

#include <fenv.h>

#include "rounding.h"

vfloat64m1_t __riscv_vrint_f64m1(vfloat64m1_t x, size_t avl)
{
    switch (fegetround())
    {
        case FE_TONEAREST:  return __riscv_vtrunc_f64m1(x, avl);
        case FE_DOWNWARD:   return __riscv_vfloor_f64m1(x, avl);
        case FE_UPWARD:     return __riscv_vceil_f64m1(x, avl);
        case FE_TOWARDZERO: return __riscv_vround_f64m1(x, avl);
    };
}
vfloat64m2_t __riscv_vrint_f64m2(vfloat64m2_t x, size_t avl)
{
    switch (fegetround())
    {
        case FE_TONEAREST:  return __riscv_vtrunc_f64m2(x, avl);
        case FE_DOWNWARD:   return __riscv_vfloor_f64m2(x, avl);
        case FE_UPWARD:     return __riscv_vceil_f64m2(x, avl);
        case FE_TOWARDZERO: return __riscv_vround_f64m2(x, avl);
    };
}
vfloat64m4_t __riscv_vrint_f64m4(vfloat64m4_t x, size_t avl)
{
    switch (fegetround())
    {
        case FE_TONEAREST:  return __riscv_vtrunc_f64m4(x, avl);
        case FE_DOWNWARD:   return __riscv_vfloor_f64m4(x, avl);
        case FE_UPWARD:     return __riscv_vceil_f64m4(x, avl);
        case FE_TOWARDZERO: return __riscv_vround_f64m4(x, avl);
    };
}
vfloat64m8_t __riscv_vrint_f64m8(vfloat64m8_t x, size_t avl)
{
    switch (fegetround())
    {
        case FE_TONEAREST:  return __riscv_vtrunc_f64m8(x, avl);
        case FE_DOWNWARD:   return __riscv_vfloor_f64m8(x, avl);
        case FE_UPWARD:     return __riscv_vceil_f64m8(x, avl);
        case FE_TOWARDZERO: return __riscv_vround_f64m8(x, avl);
    };
}

vint32m1_t __riscv_vlrint_i32m1(vfloat64m2_t x, size_t avl)
{
    return __riscv_vfncvt_x_f_w_i32m1(x, avl);
}
vint32m2_t __riscv_vlrint_i32m2(vfloat64m4_t x, size_t avl)
{
    return __riscv_vfncvt_x_f_w_i32m2(x, avl);
}
vint32m4_t __riscv_vlrint_i32m4(vfloat64m8_t x, size_t avl)
{
    return __riscv_vfncvt_x_f_w_i32m4(x, avl);
}

vint64m1_t __riscv_vllrint_i64m1(vfloat64m1_t x, size_t avl)
{
    return __riscv_vfcvt_x_f_v_i64m1(x, avl);
}
vint64m2_t __riscv_vllrint_i64m2(vfloat64m2_t x, size_t avl)
{
    return __riscv_vfcvt_x_f_v_i64m2(x, avl);
}
vint64m4_t __riscv_vllrint_i64m4(vfloat64m4_t x, size_t avl)
{
    return __riscv_vfcvt_x_f_v_i64m4(x, avl);
}
vint64m8_t __riscv_vllrint_i64m8(vfloat64m8_t x, size_t avl)
{
    return __riscv_vfcvt_x_f_v_i64m8(x, avl);
}

vfloat32m1_t __riscv_vrint_f32m1(vfloat32m1_t x, size_t avl)
{
    switch (fegetround())
    {
        case FE_TONEAREST:  return __riscv_vtrunc_f32m1(x, avl);
        case FE_DOWNWARD:   return __riscv_vfloor_f32m1(x, avl);
        case FE_UPWARD:     return __riscv_vceil_f32m1(x, avl);
        case FE_TOWARDZERO: return __riscv_vround_f32m1(x, avl);
    };
}
vfloat32m2_t __riscv_vrint_f32m2(vfloat32m2_t x, size_t avl)
{
    switch (fegetround())
    {
        case FE_TONEAREST:  return __riscv_vtrunc_f32m2(x, avl);
        case FE_DOWNWARD:   return __riscv_vfloor_f32m2(x, avl);
        case FE_UPWARD:     return __riscv_vceil_f32m2(x, avl);
        case FE_TOWARDZERO: return __riscv_vround_f32m2(x, avl);
    };
}
vfloat32m4_t __riscv_vrint_f32m4(vfloat32m4_t x, size_t avl)
{
    switch (fegetround())
    {
        case FE_TONEAREST:  return __riscv_vtrunc_f32m4(x, avl);
        case FE_DOWNWARD:   return __riscv_vfloor_f32m4(x, avl);
        case FE_UPWARD:     return __riscv_vceil_f32m4(x, avl);
        case FE_TOWARDZERO: return __riscv_vround_f32m4(x, avl);
    };
}
vfloat32m8_t __riscv_vrint_f32m8(vfloat32m8_t x, size_t avl)
{
    switch (fegetround())
    {
        case FE_TONEAREST:  return __riscv_vtrunc_f32m8(x, avl);
        case FE_DOWNWARD:   return __riscv_vfloor_f32m8(x, avl);
        case FE_UPWARD:     return __riscv_vceil_f32m8(x, avl);
        case FE_TOWARDZERO: return __riscv_vround_f32m8(x, avl);
    };
}

vint32m1_t __riscv_vlrint_i32m1(vfloat32m1_t x, size_t avl)
{
    return __riscv_vfcvt_x_f_v_i32m1(x, avl);
}
vint32m2_t __riscv_vlrint_i32m2(vfloat32m2_t x, size_t avl)
{
    return __riscv_vfcvt_x_f_v_i32m2(x, avl);
}
vint32m4_t __riscv_vlrint_i32m4(vfloat32m4_t x, size_t avl)
{
    return __riscv_vfcvt_x_f_v_i32m4(x, avl);
}
vint32m8_t __riscv_vlrint_i32m8(vfloat32m8_t x, size_t avl)
{
    return __riscv_vfcvt_x_f_v_i32m8(x, avl);
}

vint64m2_t __riscv_vllrint_i64m2(vfloat32m1_t x, size_t avl)
{
    return __riscv_vfwcvt_x_f_v_i64m2(x, avl);
}
vint64m4_t __riscv_vllrint_i64m4(vfloat32m2_t x, size_t avl)
{
    return __riscv_vfwcvt_x_f_v_i64m4(x, avl);
}
vint64m8_t __riscv_vllrint_i64m8(vfloat32m4_t x, size_t avl)
{
    return __riscv_vfwcvt_x_f_v_i64m8(x, avl);
}


#if (defined(__riscv_zvfh) || defined(__riscv_zvfhmin))

vfloat16m1_t __riscv_vrint_f16m1(vfloat16m1_t x, size_t avl)
{
    switch (fegetround())
    {
        case FE_TONEAREST:  return __riscv_vtrunc_f16m1(x, avl);
        case FE_DOWNWARD:   return __riscv_vfloor_f16m1(x, avl);
        case FE_UPWARD:     return __riscv_vceil_f16m1(x, avl);
        case FE_TOWARDZERO: return __riscv_vround_f16m1(x, avl);
    };
}
vfloat16m2_t __riscv_vrint_f16m2(vfloat16m2_t x, size_t avl)
{
    switch (fegetround())
    {
        case FE_TONEAREST:  return __riscv_vtrunc_f16m2(x, avl);
        case FE_DOWNWARD:   return __riscv_vfloor_f16m2(x, avl);
        case FE_UPWARD:     return __riscv_vceil_f16m2(x, avl);
        case FE_TOWARDZERO: return __riscv_vround_f16m2(x, avl);
    };
}
vfloat16m4_t __riscv_vrint_f16m4(vfloat16m4_t x, size_t avl)
{
    switch (fegetround())
    {
        case FE_TONEAREST:  return __riscv_vtrunc_f16m4(x, avl);
        case FE_DOWNWARD:   return __riscv_vfloor_f16m4(x, avl);
        case FE_UPWARD:     return __riscv_vceil_f16m4(x, avl);
        case FE_TOWARDZERO: return __riscv_vround_f16m4(x, avl);
    };
}
vfloat16m8_t __riscv_vrint_f16m8(vfloat16m8_t x, size_t avl)
{
    switch (fegetround())
    {
        case FE_TONEAREST:  return __riscv_vtrunc_f16m8(x, avl);
        case FE_DOWNWARD:   return __riscv_vfloor_f16m8(x, avl);
        case FE_UPWARD:     return __riscv_vceil_f16m8(x, avl);
        case FE_TOWARDZERO: return __riscv_vround_f16m8(x, avl);
    };
}

vint32m2_t __riscv_vlrint_i32m2(vfloat16m1_t x, size_t avl)
{
    return __riscv_vfwcvt_x_f_v_i32m2(x, avl);
}
vint32m4_t __riscv_vlrint_i32m4(vfloat16m2_t x, size_t avl)
{
    return __riscv_vfwcvt_x_f_v_i32m4(x, avl);
}
vint32m8_t __riscv_vlrint_i32m8(vfloat16m4_t x, size_t avl)
{
    return __riscv_vfwcvt_x_f_v_i32m8(x, avl);
}

vint64m4_t __riscv_vllrint_i64m4(vfloat16m1_t x, size_t avl)
{
    return __riscv_vwcvt_x_x_v_i64m4(__riscv_vfwcvt_x_f_v_i32m2(x, avl), avl);
}
vint64m8_t __riscv_vllrint_i64m8(vfloat16m2_t x, size_t avl)
{
    return __riscv_vwcvt_x_x_v_i64m8(__riscv_vfwcvt_x_f_v_i32m4(x, avl), avl);
}

#endif /* __riscv_zvfh || __riscv_zvfhmin */

#endif /* __riscv_v_intrinsic */