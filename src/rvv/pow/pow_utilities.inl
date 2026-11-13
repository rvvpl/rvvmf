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
 *   File:  pow_utilities.inl                            *
 *   Contains: helper built-in functions for pow impl    *
 *                                                       *
 *                                                       *
 *********************************************************
*/


#include <riscv_vector.h>

#define RISCV_POW_UTILS(TYPE)   \
static inline void mul22_f##TYPE(vfloat##TYPE##_t ah, vfloat##TYPE##_t al, vfloat##TYPE##_t bh, vfloat##TYPE##_t bl, vfloat##TYPE##_t& zh, vfloat##TYPE##_t& zl, size_t vl)\
{\
    zh = __riscv_vfmul_vv_f##TYPE(ah, bh, vl);\
    zl = __riscv_vfadd_vv_f##TYPE(__riscv_vfmsub_vv_f##TYPE(ah, bh, zh, vl), \
				__riscv_vfmadd_vv_f##TYPE(\
							ah, \
							bl, \
							__riscv_vfmul_vv_f##TYPE(al, bh, vl), \
							vl\
							),\
				vl);\
}\
\
static inline void fast_2_sum_f##TYPE(vfloat##TYPE##_t a, vfloat##TYPE##_t b, vfloat##TYPE##_t& sh, vfloat##TYPE##_t& sl, size_t vl)\
{\
    sh = __riscv_vfadd_vv_f##TYPE(a, b, vl);\
    sl = __riscv_vfsub_vv_f##TYPE(b, __riscv_vfsub_vv_f##TYPE(sh, a, vl), vl);\
}\
\
static inline void fma12_f##TYPE(vfloat##TYPE##_t ah, vfloat##TYPE##_t bh, vfloat##TYPE##_t ch, vfloat##TYPE##_t& zh, vfloat##TYPE##_t& zl, size_t vl)\
{\
    zh = __riscv_vfmadd_vv_f##TYPE(ah, bh, ch, vl);\
    zl = __riscv_vfmadd_vv_f##TYPE(ah, bh, __riscv_vfsub_vv_f##TYPE(ch, zh, vl), vl);\
}\
\
static inline void fma12_ver2p1_f##TYPE(vfloat##TYPE##_t ah, vfloat##TYPE##_t bh, vfloat##TYPE##_t ch, vfloat##TYPE##_t& zh, vfloat##TYPE##_t& zl, size_t vl)\
{\
    vfloat##TYPE##_t sh, sl;\
    zh = __riscv_vfmadd_vv_f##TYPE(ah, bh, ch, vl);\
    fast_2_sum_f##TYPE(__riscv_vfneg_v_f##TYPE(zh, vl), ch, sh, sl, vl);\
    zl = __riscv_vfmadd_vv_f##TYPE(ah, bh, sh, vl);\
    zl = __riscv_vfadd_vv_f##TYPE(zl, sl, vl);\
}\
\
static inline void mul21_f##TYPE(vfloat##TYPE##_t ah, vfloat##TYPE##_t al, vfloat##TYPE##_t bh, vfloat##TYPE##_t bl, vfloat##TYPE##_t& rh, size_t vl)\
{\
    vfloat##TYPE##_t zh, zl;\
    mul22_f##TYPE(ah, al, bh, bl, zh, zl, vl);\
    rh = __riscv_vfadd_vv_f##TYPE(zh, zl, vl);\
}

RISCV_POW_UTILS(64m1)
RISCV_POW_UTILS(64m2)
RISCV_POW_UTILS(64m4)
RISCV_POW_UTILS(64m8)
RISCV_POW_UTILS(32m1)
RISCV_POW_UTILS(32m2)
RISCV_POW_UTILS(32m4)
RISCV_POW_UTILS(32m8)

#ifdef __riscv_zvfh
RISCV_POW_UTILS(16m1)
RISCV_POW_UTILS(16m2)
RISCV_POW_UTILS(16m4)
RISCV_POW_UTILS(16m8)

static inline void mul22_acc_f16m1(vfloat16m1_t ah, vfloat16m1_t al, vfloat16m1_t bh, vfloat16m1_t bl, vfloat16m1_t& zh, vfloat16m1_t& zl, size_t vl)
{
    vfloat16m1_t Bh = __riscv_vfmul_vf_f16m1(bh, 0x1p10f16, vl);
    zh = __riscv_vfmul_vv_f16m1(ah, Bh, vl);
    zl = __riscv_vfadd_vv_f16m1(__riscv_vfmsub_vv_f16m1(ah, Bh, zh, vl),
				__riscv_vfmadd_vv_f16m1(
							ah,
							bl,
							__riscv_vfmul_vv_f16m1(al, bh, vl),
							vl
							),
				vl);
    zh = __riscv_vfmul_vf_f16m1(zh, 0x1p-10f16, vl);
}
static inline void mul22_acc_f16m2(vfloat16m2_t ah, vfloat16m2_t al, vfloat16m2_t bh, vfloat16m2_t bl, vfloat16m2_t& zh, vfloat16m2_t& zl, size_t vl)
{
    vfloat16m2_t Bh = __riscv_vfmul_vf_f16m2(bh, 0x1p10f16, vl);
    zh = __riscv_vfmul_vv_f16m2(ah, Bh, vl);
    zl = __riscv_vfadd_vv_f16m2(__riscv_vfmsub_vv_f16m2(ah, Bh, zh, vl),
				__riscv_vfmadd_vv_f16m2(
							ah,
							bl,
							__riscv_vfmul_vv_f16m2(al, bh, vl),
							vl
							),
				vl);
    zh = __riscv_vfmul_vf_f16m2(zh, 0x1p-10f16, vl);
}
static inline void mul22_acc_f16m4(vfloat16m4_t ah, vfloat16m4_t al, vfloat16m4_t bh, vfloat16m4_t bl, vfloat16m4_t& zh, vfloat16m4_t& zl, size_t vl)
{
    vfloat16m4_t Bh = __riscv_vfmul_vf_f16m4(bh, 0x1p10f16, vl);
    zh = __riscv_vfmul_vv_f16m4(ah, Bh, vl);
    zl = __riscv_vfadd_vv_f16m4(__riscv_vfmsub_vv_f16m4(ah, Bh, zh, vl),
				__riscv_vfmadd_vv_f16m4(
							ah,
							bl,
							__riscv_vfmul_vv_f16m4(al, bh, vl),
							vl
							),
				vl);
    zh = __riscv_vfmul_vf_f16m4(zh, 0x1p-10f16, vl);
}
static inline void mul22_acc_f16m8(vfloat16m8_t ah, vfloat16m8_t al, vfloat16m8_t bh, vfloat16m8_t bl, vfloat16m8_t& zh, vfloat16m8_t& zl, size_t vl)
{
    vfloat16m8_t Bh = __riscv_vfmul_vf_f16m8(bh, 0x1p10f16, vl);
    zh = __riscv_vfmul_vv_f16m8(ah, Bh, vl);
    zl = __riscv_vfadd_vv_f16m8(__riscv_vfmsub_vv_f16m8(ah, Bh, zh, vl),
				__riscv_vfmadd_vv_f16m8(
							ah,
							bl,
							__riscv_vfmul_vv_f16m8(al, bh, vl),
							vl
							),
				vl);
    zh = __riscv_vfmul_vf_f16m8(zh, 0x1p-10f16, vl);
}
#endif