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
 *   File:  exp_utilities.inl                            *
 *   Contains: helper built-in functions for exp impl    *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#ifndef __RVVMF_EXP_HELPER_BUILT_IN_FUNCTIONS__
#define __RVVMF_EXP_HELPER_BUILT_IN_FUNCTIONS__

#include "exp_macro.inl"


/* double-FP arithmetic functions */

#define RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(postfix, stype, vtype) \
    forceinline void fast_2_sum_vv_##postfix(vtype a, vtype b, vtype& sh, vtype& sl, size_t vl) \
        { RVVMF_EXP_FAST2SUM_VV(postfix, vtype, a, b, sh, sl, vl); }
    
#define RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(postfix, stype, vtype) \
    forceinline void fast_2_sum_fv_##postfix(stype a, vtype b, vtype& sh, vtype& sl, size_t vl) \
        { RVVMF_EXP_FAST2SUM_FV(postfix, vtype, a, b, sh, sl, vl); }

#define RVVMF_EXP_DEF_MUL22_VV_FUNC(postfix, stype, vtype) \
    forceinline void mul22_vv_##postfix(vtype ah, vtype al, vtype bh, vtype bl, vtype& zh, vtype& zl, size_t vl) \
        { RVVMF_EXP_MUL22_VV(postfix, vtype, ah, al, bh, bl, zh, zl, vl); }
        
#define RVVMF_EXP_DEF_MUL21_VV_FUNC(postfix, stype, vtype) \
    forceinline void mul21_vv_##postfix(vtype ah, vtype al, vtype bh, vtype bl, vtype& rh, size_t vl) \
        { RVVMF_EXP_MUL21_VV(postfix, vtype, ah, al, bh, bl, rh, vl); }

#define RVVMF_EXP_DEF_FMA12_VV_FUNC(postfix, stype, vtype) \
    forceinline void fma12_vv_##postfix(vtype ah, vtype bh, vtype ch, vtype& zh, vtype& zl, size_t vl) \
        { RVVMF_EXP_FMA12_VER1_VV(postfix, vtype, ah, bh, ch, zh, zl, vl); }

#define RVVMF_EXP_DEF_FMA12_VF_FUNC(postfix, stype, vtype) \
    forceinline void fma12_vf_##postfix(vtype ah, stype bh, vtype ch, vtype& zh, vtype& zl, size_t vl) \
        { RVVMF_EXP_FMA12_VER1_VF(postfix, vtype, ah, bh, ch, zh, zl, vl); }

#define RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(postfix, stype, vtype) \
    forceinline void fma12_ver2p1_vf_##postfix(vtype ah, stype bh, vtype ch, vtype& zh, vtype& zl, size_t vl) \
        { RVVMF_EXP_FMA12_VER2P1_VF(postfix, vtype, ah, bh, ch, zh, zl, vl); }

#define RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(postfix, stype, vtype) \
    forceinline void fma12_ver2p2_vf_##postfix(vtype ah, stype bh, vtype ch, vtype& zh, vtype& zl, size_t vl) \
        { RVVMF_EXP_FMA12_VER2P2_VF(postfix, vtype, ah, bh, ch, zh, zl, vl); }


/* fast2sum operations, a+b=sh+sl, exponent a >= exponent b */
RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(f32m8, float, vfloat32m8_t)

RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(f32m8, float, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_FAST2SUM_VV_FUNC(f16m8, _Float16, vfloat16m8_t)
    
    RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_FAST2SUM_FV_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif


/* mul22 operations, (ah+al)*(bh+bl)=sh+sl */
RVVMF_EXP_DEF_MUL22_VV_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_MUL22_VV_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_MUL22_VV_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_MUL22_VV_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_EXP_DEF_MUL22_VV_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_MUL22_VV_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_MUL22_VV_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_MUL22_VV_FUNC(f32m8, float, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    RVVMF_EXP_DEF_MUL22_VV_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_MUL22_VV_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_MUL22_VV_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_MUL22_VV_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

/* mul21 operations, (ah+al)*(bh+bl)=s */
RVVMF_EXP_DEF_MUL21_VV_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_MUL21_VV_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_MUL21_VV_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_MUL21_VV_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_EXP_DEF_MUL21_VV_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_MUL21_VV_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_MUL21_VV_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_MUL21_VV_FUNC(f32m8, float, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    RVVMF_EXP_DEF_MUL21_VV_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_MUL21_VV_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_MUL21_VV_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_MUL21_VV_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif


/* simple fma12 operations, a*b+c=sh+sl */
RVVMF_EXP_DEF_FMA12_VV_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_FMA12_VV_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_FMA12_VV_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_FMA12_VV_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_EXP_DEF_FMA12_VF_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_FMA12_VF_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_FMA12_VF_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_FMA12_VF_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_EXP_DEF_FMA12_VV_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_FMA12_VV_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_FMA12_VV_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_FMA12_VV_FUNC(f32m8, float, vfloat32m8_t)

RVVMF_EXP_DEF_FMA12_VF_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_FMA12_VF_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_FMA12_VF_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_FMA12_VF_FUNC(f32m8, float, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    RVVMF_EXP_DEF_FMA12_VV_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_FMA12_VV_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_FMA12_VV_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_FMA12_VV_FUNC(f16m8, _Float16, vfloat16m8_t)
    
    RVVMF_EXP_DEF_FMA12_VF_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_FMA12_VF_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_FMA12_VF_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_FMA12_VF_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

/* more exact fma12 operations, a*b+c=sh+sl, exponent zh >= exponent ch */
RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(f32m8, float, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_FMA12_VER2P1_VF_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

/* more exact fma12 operations, a*b+c=sh+sl, exponent ch >= exponent zh */
RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(f32m8, float, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_FMA12_VER2P2_VF_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif


/* polynom calculation functions */

#define RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(postfix, stype, vtype) \
    forceinline vtype calc_polynom_deg_1_##postfix(const vtype& x, stype a0, stype a1, size_t vl) \
        { return RVVMF_EXP_CALC_POLYNOM_DEG_1(postfix, x, a0, a1, vl); }
        
#define RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(postfix, stype, vtype) \
    forceinline vtype calc_polynom_deg_2_##postfix(const vtype& x, stype a0, stype a1, stype a2, size_t vl) \
        { return RVVMF_EXP_CALC_POLYNOM_DEG_2(postfix, x, a0, a1, a2, vl); }

#define RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(postfix, stype, vtype) \
    forceinline vtype calc_polynom_deg_3_parallel_##postfix(const vtype& x, const vtype& sqrx, stype a0, stype a1, \
        stype a2, stype a3, size_t vl) \
        { return RVVMF_EXP_CALC_POLYNOM_DEG_3_PARALLEL(postfix, x, sqrx, a0, a1, a2, a3, vl); }

#define RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(postfix, stype, vtype) \
    forceinline vtype calc_polynom_deg_4_parallel_##postfix(const vtype& x, const vtype& sqrx, stype a0, stype a1, \
        stype a2, stype a3, stype a4, size_t vl) \
        { return RVVMF_EXP_CALC_POLYNOM_DEG_4_PARALLEL(postfix, x, sqrx, a0, a1, a2, a3, a4, vl); }

#define RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(postfix, stype, vtype) \
    forceinline vtype calc_polynom_deg_5_parallel_##postfix(const vtype& x, const vtype& sqrx, stype a0, stype a1, \
        stype a2, stype a3, stype a4, stype a5, size_t vl) \
        { return RVVMF_EXP_CALC_POLYNOM_DEG_5_PARALLEL(postfix, x, sqrx, a0, a1, a2, a3, a4, a5, vl); }

#define RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(postfix, stype, vtype) \
    forceinline vtype calc_polynom_deg_6_parallel_##postfix(const vtype& x, const vtype& sqrx, stype a0, stype a1, \
        stype a2, stype a3, stype a4, stype a5, stype a6, size_t vl) \
        { return RVVMF_EXP_CALC_POLYNOM_DEG_6_PARALLEL(postfix, x, sqrx, a0, a1, a2, a3, a4, a5, a6, vl); }


RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(f64m8, double, vfloat64m8_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(f32m8, float, vfloat32m8_t)
#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_1_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(f64m8, double, vfloat64m8_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(f32m8, float, vfloat32m8_t)
#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_2_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(f64m8, double, vfloat64m8_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(f32m8, float, vfloat32m8_t)
#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_3_PARALLEL_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(f64m8, double, vfloat64m8_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(f32m8, float, vfloat32m8_t)
#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_4_PARALLEL_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(f64m8, double, vfloat64m8_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(f32m8, float, vfloat32m8_t)
#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_5_PARALLEL_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(f64m8, double, vfloat64m8_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(f32m8, float, vfloat32m8_t)
#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(f16m1, _Float16, vfloat16m1_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(f16m2, _Float16, vfloat16m2_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(f16m4, _Float16, vfloat16m4_t)
    RVVMF_EXP_DEF_CALC_POLYNOM_DEG_6_PARALLEL_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

#endif
