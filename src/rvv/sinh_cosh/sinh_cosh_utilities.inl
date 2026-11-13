/*
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of
 * Nizhny Novgorod and its affiliates. All rights reserved.
 *
 * Copyright 2026 The RVVMF Authors (Kabalova Valeriia)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  sinh_cosh_utilities.inl                      *
 *   Contains: helper built-in functions for             *
 *             sinh and cosh implementations             *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#ifndef __RVVMF_SC_HELPER_BUILT_IN_FUNCTIONS__
#define __RVVMF_SC_HELPER_BUILT_IN_FUNCTIONS__

#include "sinh_cosh_macro.inl"

/* double-FP arithmetic functions */

#define RVVMF_SC_DEF_FAST2SUM_VV_FUNC(postfix, stype, vtype) \
  forceinline void fast2sum_vv_##postfix(vtype a, vtype b, vtype& sh, vtype& sl, size_t vl) \
    { RVVMF_SC_FAST2SUM_VV(postfix, vtype, a, b, sh, sl, vl); }

#define RVVMF_SC_DEF_MUL12_VV_FUNC(postfix, stype, vtype) \
  forceinline void mul12_vv_##postfix(vtype ah, vtype bh, vtype& zh, vtype& zl, size_t vl) \
    { RVVMF_SC_MUL12_VV(postfix, vtype, ah, bh, zh, zl, vl);}

#define RVVMF_SC_DEF_MUL22_VV_FUNC(postfix, stype, vtype) \
  forceinline void mul22_vv_##postfix(vtype ah, vtype al, vtype bh, vtype bl, vtype& zh, vtype& zl, size_t vl) \
    { RVVMF_SC_MUL22_VV(postfix, vtype, ah, al, bh, bl, zh, zl, vl);}

#define RVVMF_SC_DEF_FMA12_VV_FUNC(postfix, stype, vtype) \
  forceinline void fma12_vv_##postfix(vtype ah, vtype bh, vtype ch, vtype& zh, vtype& zl, size_t vl) \
    { RVVMF_SC_FMA12_VV(postfix, vtype, ah, bh, ch, zh, zl, vl);}

#define RVVMF_SC_DEF_FMA12_VER2_VF_FUNC(postfix, stype, vtype, literal) \
  forceinline void fma12_ver2_vf_##postfix(vtype ah, stype bh, vtype ch, vtype& zh, vtype& zl, size_t vl) \
    { RVVMF_SC_FMA12_VER2_VF(postfix, vtype, ah, bh, ch, zh, zl, vl, literal);}

#define RVVMF_SC_DEF_ADD22_VV_FUNC(postfix, stype, vtype) \
  forceinline void add22_vv_##postfix(vtype ah, vtype al, vtype bh, vtype bl, vtype& zh, vtype& zl, size_t vl) \
    { RVVMF_SC_ADD22_VV(postfix, vtype, ah, al, bh, bl, zh, zl, vl);}

#define RVVMF_SC_DEF_SUB22_VV_FUNC(postfix, stype, vtype) \
  forceinline void sub22_vv_##postfix(vtype ah, vtype al, vtype bh, vtype bl, vtype& zh, vtype& zl, size_t vl) \
    { RVVMF_SC_SUB22_VV(postfix, vtype, ah, al, bh, bl, zh, zl, vl);}

#define RVVMF_SC_DEF_ADD21_VV_FUNC(postfix, stype, vtype) \
  forceinline void add21_vv_##postfix(vtype ah, vtype al, vtype bh, vtype bl, vtype& zh, size_t vl) \
    { RVVMF_SC_ADD21_VV(postfix, vtype, ah, al, bh, bl, zh, vl);}

#define RVVMF_SC_DEF_ADD21_2_VV_FUNC(postfix, stype, vtype, literal) \
  forceinline void add21_2_vv_##postfix(vtype ah, vtype al, vtype bh, vtype bl, vtype& zh, size_t vl) \
    { RVVMF_SC_ADD21_2_VV(postfix, vtype, ah, al, bh, bl, zh, vl, literal);}

/*fast2sum operations a + b = sh + sl*/
RVVMF_SC_DEF_FAST2SUM_VV_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_SC_DEF_FAST2SUM_VV_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_SC_DEF_FAST2SUM_VV_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_SC_DEF_FAST2SUM_VV_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_SC_DEF_FAST2SUM_VV_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_SC_DEF_FAST2SUM_VV_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_SC_DEF_FAST2SUM_VV_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_SC_DEF_FAST2SUM_VV_FUNC(f32m8, float, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
RVVMF_SC_DEF_FAST2SUM_VV_FUNC(f16m1, _Float16, vfloat16m1_t)
RVVMF_SC_DEF_FAST2SUM_VV_FUNC(f16m2, _Float16, vfloat16m2_t)
RVVMF_SC_DEF_FAST2SUM_VV_FUNC(f16m4, _Float16, vfloat16m4_t)
RVVMF_SC_DEF_FAST2SUM_VV_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

/*mul12 operation ah * bh = zh + zl*/
RVVMF_SC_DEF_MUL12_VV_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_SC_DEF_MUL12_VV_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_SC_DEF_MUL12_VV_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_SC_DEF_MUL12_VV_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_SC_DEF_MUL12_VV_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_SC_DEF_MUL12_VV_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_SC_DEF_MUL12_VV_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_SC_DEF_MUL12_VV_FUNC(f32m8, float, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
RVVMF_SC_DEF_MUL12_VV_FUNC(f16m1, _Float16, vfloat16m1_t)
RVVMF_SC_DEF_MUL12_VV_FUNC(f16m2, _Float16, vfloat16m2_t)
RVVMF_SC_DEF_MUL12_VV_FUNC(f16m4, _Float16, vfloat16m4_t)
RVVMF_SC_DEF_MUL12_VV_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

/*mul22 operation (ah + al) * (bh + bl) = zh + zl */
RVVMF_SC_DEF_MUL22_VV_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_SC_DEF_MUL22_VV_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_SC_DEF_MUL22_VV_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_SC_DEF_MUL22_VV_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_SC_DEF_MUL22_VV_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_SC_DEF_MUL22_VV_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_SC_DEF_MUL22_VV_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_SC_DEF_MUL22_VV_FUNC(f32m8, float, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
RVVMF_SC_DEF_MUL22_VV_FUNC(f16m1, _Float16, vfloat16m1_t)
RVVMF_SC_DEF_MUL22_VV_FUNC(f16m2, _Float16, vfloat16m2_t)
RVVMF_SC_DEF_MUL22_VV_FUNC(f16m4, _Float16, vfloat16m4_t)
RVVMF_SC_DEF_MUL22_VV_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

/*fma12 operation ah * bh + ch = zh + zl    |zh| > |ch|*/
RVVMF_SC_DEF_FMA12_VV_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_SC_DEF_FMA12_VV_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_SC_DEF_FMA12_VV_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_SC_DEF_FMA12_VV_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_SC_DEF_FMA12_VV_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_SC_DEF_FMA12_VV_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_SC_DEF_FMA12_VV_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_SC_DEF_FMA12_VV_FUNC(f32m8, float, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
RVVMF_SC_DEF_FMA12_VV_FUNC(f16m1, _Float16, vfloat16m1_t)
RVVMF_SC_DEF_FMA12_VV_FUNC(f16m2, _Float16, vfloat16m2_t)
RVVMF_SC_DEF_FMA12_VV_FUNC(f16m4, _Float16, vfloat16m4_t)
RVVMF_SC_DEF_FMA12_VV_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

/*fma12 operation ah * bh + ch = zh + zl    |ch| > |zh|*/
RVVMF_SC_DEF_FMA12_VER2_VF_FUNC(f32m1, float, vfloat32m1_t, f)
RVVMF_SC_DEF_FMA12_VER2_VF_FUNC(f32m2, float, vfloat32m2_t, f)
RVVMF_SC_DEF_FMA12_VER2_VF_FUNC(f32m4, float, vfloat32m4_t, f)
RVVMF_SC_DEF_FMA12_VER2_VF_FUNC(f32m8, float, vfloat32m8_t, f)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
RVVMF_SC_DEF_FMA12_VER2_VF_FUNC(f16m1, _Float16, vfloat16m1_t, f16)
RVVMF_SC_DEF_FMA12_VER2_VF_FUNC(f16m2, _Float16, vfloat16m2_t, f16)
RVVMF_SC_DEF_FMA12_VER2_VF_FUNC(f16m4, _Float16, vfloat16m4_t, f16)
RVVMF_SC_DEF_FMA12_VER2_VF_FUNC(f16m8, _Float16, vfloat16m8_t, f16)
#endif

/*add22 operation (ah + al) + (bh + bl) = zh + zl*/
RVVMF_SC_DEF_ADD22_VV_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_SC_DEF_ADD22_VV_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_SC_DEF_ADD22_VV_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_SC_DEF_ADD22_VV_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_SC_DEF_ADD22_VV_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_SC_DEF_ADD22_VV_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_SC_DEF_ADD22_VV_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_SC_DEF_ADD22_VV_FUNC(f32m8, float, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
RVVMF_SC_DEF_ADD22_VV_FUNC(f16m1, _Float16, vfloat16m1_t)
RVVMF_SC_DEF_ADD22_VV_FUNC(f16m2, _Float16, vfloat16m2_t)
RVVMF_SC_DEF_ADD22_VV_FUNC(f16m4, _Float16, vfloat16m4_t)
RVVMF_SC_DEF_ADD22_VV_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

/*sub22 operation (ah + al) - (bh + bl) = zh + zl*/
RVVMF_SC_DEF_SUB22_VV_FUNC(f64m1, double, vfloat64m1_t)
RVVMF_SC_DEF_SUB22_VV_FUNC(f64m2, double, vfloat64m2_t)
RVVMF_SC_DEF_SUB22_VV_FUNC(f64m4, double, vfloat64m4_t)
RVVMF_SC_DEF_SUB22_VV_FUNC(f64m8, double, vfloat64m8_t)

RVVMF_SC_DEF_SUB22_VV_FUNC(f32m1, float, vfloat32m1_t)
RVVMF_SC_DEF_SUB22_VV_FUNC(f32m2, float, vfloat32m2_t)
RVVMF_SC_DEF_SUB22_VV_FUNC(f32m4, float, vfloat32m4_t)
RVVMF_SC_DEF_SUB22_VV_FUNC(f32m8, float, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
RVVMF_SC_DEF_SUB22_VV_FUNC(f16m1, _Float16, vfloat16m1_t)
RVVMF_SC_DEF_SUB22_VV_FUNC(f16m2, _Float16, vfloat16m2_t)
RVVMF_SC_DEF_SUB22_VV_FUNC(f16m4, _Float16, vfloat16m4_t)
RVVMF_SC_DEF_SUB22_VV_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

RVVMF_SC_DEF_ADD21_VV_FUNC(f32m1, _Float32, vfloat32m1_t)
RVVMF_SC_DEF_ADD21_VV_FUNC(f32m2, _Float32, vfloat32m2_t)
RVVMF_SC_DEF_ADD21_VV_FUNC(f32m4, _Float32, vfloat32m4_t)
RVVMF_SC_DEF_ADD21_VV_FUNC(f32m8, _Float32, vfloat32m8_t)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
RVVMF_SC_DEF_ADD21_VV_FUNC(f16m1, _Float16, vfloat16m1_t)
RVVMF_SC_DEF_ADD21_VV_FUNC(f16m2, _Float16, vfloat16m2_t)
RVVMF_SC_DEF_ADD21_VV_FUNC(f16m4, _Float16, vfloat16m4_t)
RVVMF_SC_DEF_ADD21_VV_FUNC(f16m8, _Float16, vfloat16m8_t)
#endif

/*add21 operation with 2*a */
RVVMF_SC_DEF_ADD21_2_VV_FUNC(f32m1, float, vfloat32m1_t, f)
RVVMF_SC_DEF_ADD21_2_VV_FUNC(f32m2, float, vfloat32m2_t, f)
RVVMF_SC_DEF_ADD21_2_VV_FUNC(f32m4, float, vfloat32m4_t, f)
RVVMF_SC_DEF_ADD21_2_VV_FUNC(f32m8, float, vfloat32m8_t, f)

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
RVVMF_SC_DEF_ADD21_2_VV_FUNC(f16m1, _Float16, vfloat16m1_t, f16)
RVVMF_SC_DEF_ADD21_2_VV_FUNC(f16m2, _Float16, vfloat16m2_t, f16)
RVVMF_SC_DEF_ADD21_2_VV_FUNC(f16m4, _Float16, vfloat16m4_t, f16)
RVVMF_SC_DEF_ADD21_2_VV_FUNC(f16m8, _Float16, vfloat16m8_t, f16)
#endif

#endif
