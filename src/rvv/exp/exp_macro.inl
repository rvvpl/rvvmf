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
 *   File:  exp_macro.inl                                *
 *   Contains: helper macros for exp functions           *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#ifndef __RVVMF_EXP_HELPER_MACRO__
#define __RVVMF_EXP_HELPER_MACRO__

#ifndef forceinline 
    #if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
        #define forceinline __attribute__((always_inline)) inline
    #else
        #define forceinline inline
    #endif
#endif


/* fe exception macros */
#define RVVMF_EXP_CALL_FE_INVALID() volatile double exception = 0.0/0.0

#define RVVMF_EXP_CALL_FE_OVERFLOW() volatile double exception = DBL_MAX*2.0

#define RVVMF_EXP_CALL_FE_UNDERFLOW() volatile double exception = nextafter(DBL_MIN/(double((uint64_t)1 << 52)), 0.0)


/* c reinterpret macros */
#define RVVMF_EXP_AS_FP64(x) (*(double*)(&x))

#define RVVMF_EXP_AS_FP32(x) (*(float*)(&x))

#if defined(__riscv_zfh) || defined(__riscv_zvfh)
    #define RVVMF_EXP_AS_FP16(x) (*(_Float16*)(&x))
#endif


/* polynom calculation macros */
#define RVVMF_EXP_CALC_POLYNOM_DEG_1(postfix, x, a0, a1, vl) \
    __riscv_vfmadd_vf_##postfix(x, a1, __riscv_vfmv_v_f_##postfix(a0, vl), vl)

#define RVVMF_EXP_CALC_POLYNOM_DEG_2(postfix, x, a0, a1, a2, vl) \
    __riscv_vfmadd_vv_##postfix(x, __riscv_vfmadd_vf_##postfix(x, a2, __riscv_vfmv_v_f_##postfix(a1, vl), vl), \
    __riscv_vfmv_v_f_##postfix(a0, vl), vl)

#define RVVMF_EXP_CALC_POLYNOM_DEG_3(postfix, x, a0, a1, a2, a3, vl) \
    __riscv_vfmadd_vv_##postfix(x, __riscv_vfmadd_vv_##postfix(x, \
    __riscv_vfmadd_vf_##postfix(x, a3, __riscv_vfmv_v_f_##postfix(a2, vl), vl), \
    __riscv_vfmv_v_f_##postfix(a1, vl), vl), __riscv_vfmv_v_f_##postfix(a0, vl), vl)

#define RVVMF_EXP_CALC_POLYNOM_DEG_3_PARALLEL(postfix, x, sqrx, a0, a1, a2, a3, vl) \
    __riscv_vfmadd_vv_##postfix(x, RVVMF_EXP_CALC_POLYNOM_DEG_1(postfix, sqrx, a1, a3, vl), \
    RVVMF_EXP_CALC_POLYNOM_DEG_1(postfix, sqrx, a0, a2, vl), vl)

#define RVVMF_EXP_CALC_POLYNOM_DEG_4_PARALLEL(postfix, x, sqrx, a0, a1, a2, a3, a4, vl) \
    __riscv_vfmadd_vv_##postfix(x, RVVMF_EXP_CALC_POLYNOM_DEG_1(postfix, sqrx, a1, a3, vl), \
    RVVMF_EXP_CALC_POLYNOM_DEG_2(postfix, sqrx, a0, a2, a4, vl), vl)

#define RVVMF_EXP_CALC_POLYNOM_DEG_5_PARALLEL(postfix, x, sqrx, a0, a1, a2, a3, a4, a5, vl) \
    __riscv_vfmadd_vv_##postfix(x, RVVMF_EXP_CALC_POLYNOM_DEG_2(postfix, sqrx, a1, a3, a5, vl), \
    RVVMF_EXP_CALC_POLYNOM_DEG_2(postfix, sqrx, a0, a2, a4, vl), vl)

#define RVVMF_EXP_CALC_POLYNOM_DEG_6_PARALLEL(postfix, x, sqrx, a0, a1, a2, a3, a4, a5, a6, vl) \
    __riscv_vfmadd_vv_##postfix(x, RVVMF_EXP_CALC_POLYNOM_DEG_2(postfix, sqrx, a1, a3, a5, vl), \
    RVVMF_EXP_CALC_POLYNOM_DEG_3_PARALLEL(postfix, sqrx, __riscv_vfmul_vv_##postfix(sqrx, sqrx, vl), a0, a2, a4, a6, vl), vl)


/* double-FP arithmetic macros */
#define RVVMF_EXP_FAST2SUM_VV(postfix, vtype, a, b, sh, sl, vl) /* |a| > |b| */ \
        sh = __riscv_vfadd_vv_##postfix(a, b, vl); \
        sl = __riscv_vfsub_vv_##postfix(b, __riscv_vfsub_vv_##postfix(sh, a, vl), vl)

#define RVVMF_EXP_FAST2SUM_FV(postfix, vtype, a, b, sh, sl, vl) /* |a| > |b| */ \
    sh = __riscv_vfadd_vf_##postfix(b, a, vl); \
    sl = __riscv_vfsub_vv_##postfix(b, __riscv_vfsub_vf_##postfix(sh, a, vl), vl)

#define RVVMF_EXP_MUL22_VV(postfix, vtype, ah, al, bh, bl, zh, zl, vl) \
    zh = __riscv_vfmul_vv_##postfix(ah, bh, vl); \
    zl = __riscv_vfmsub_vv_##postfix(ah, bh, zh, vl); \
    zl = __riscv_vfadd_vv_##postfix(zl, __riscv_vfmadd_vv_##postfix(ah, bl, __riscv_vfmul_vv_##postfix(al, bh, vl), vl), vl);\

#define RVVMF_EXP_MUL21_VV(postfix, vtype, ah, al, bh, bl, rh, vl) \
    vtype __var_zh_rvvmf_exp_mul21_vv__, __var_zl_rvvmf_exp_mul21_vv__; \
    RVVMF_EXP_MUL22_VV(postfix, vtype, ah, al, bh, bl, __var_zh_rvvmf_exp_mul21_vv__, __var_zl_rvvmf_exp_mul21_vv__, vl); \
    rh = __riscv_vfadd_vv_##postfix(__var_zh_rvvmf_exp_mul21_vv__, __var_zl_rvvmf_exp_mul21_vv__, vl)

#define RVVMF_EXP_FMA12_VER1_VV(postfix, vtype, ah, bh, ch, zh, zl, vl) \
    zh = __riscv_vfmadd_vv_##postfix(ah, bh, ch, vl); \
    zl = __riscv_vfmadd_vv_##postfix(ah, bh, __riscv_vfsub_vv_##postfix(ch, zh, vl), vl)

#define RVVMF_EXP_FMA12_VER1_VF(postfix, vtype, ah, bh, ch, zh, zl, vl) \
    zh = __riscv_vfmadd_vf_##postfix(ah, bh, ch, vl); \
    zl = __riscv_vfmadd_vf_##postfix(ah, bh, __riscv_vfsub_vv_##postfix(ch, zh, vl), vl)

#define RVVMF_EXP_FMA12_VER2P1_VF(postfix, vtype, ah, bh, ch, zh, zl, vl) /* |zh| > |ch| */ \
    vtype __var_sh_rvvmf_exp_fma12_ver2p1_vf__, __var_sl_rvvmf_exp_fma12_ver2p1_vf__; \
    zh = __riscv_vfmadd_vf_##postfix(ah, bh, ch, vl); \
    RVVMF_EXP_FAST2SUM_VV(postfix, vtype, __riscv_vfneg_v_##postfix(zh, vl), ch, \
        __var_sh_rvvmf_exp_fma12_ver2p1_vf__, __var_sl_rvvmf_exp_fma12_ver2p1_vf__, vl); \
    zl = __riscv_vfmadd_vf_##postfix(ah, bh, __var_sh_rvvmf_exp_fma12_ver2p1_vf__, vl); \
    zl = __riscv_vfadd_vv_##postfix(zl, __var_sl_rvvmf_exp_fma12_ver2p1_vf__, vl)

#define RVVMF_EXP_FMA12_VER2P2_VF(postfix, vtype, ah, bh, ch, zh, zl, vl) /* |ch| > |zh| */ \
    vtype __var_sh_rvvmf_exp_fma12_ver2p2_vf__, __var_sl_rvvmf_exp_fma12_ver2p2_vf__; \
    zh = __riscv_vfmadd_vf_##postfix(ah, bh, ch, vl); \
    RVVMF_EXP_FAST2SUM_VV(postfix, vtype, ch, __riscv_vfneg_v_##postfix(zh, vl), \
        __var_sh_rvvmf_exp_fma12_ver2p2_vf__, __var_sl_rvvmf_exp_fma12_ver2p2_vf__, vl); \
    zl = __riscv_vfmadd_vf_##postfix(ah, bh, __var_sh_rvvmf_exp_fma12_ver2p2_vf__, vl); \
    zl = __riscv_vfadd_vv_##postfix(zl, __var_sl_rvvmf_exp_fma12_ver2p2_vf__, vl)

#endif
