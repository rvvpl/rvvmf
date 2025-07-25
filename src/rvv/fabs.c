/* 
*========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2024 The RVVMF Authors (Valentin Volokitin)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  fabs.c                                       *
 *   Contains: intrinsic function fabs for f64, f32, f16 *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Return value absolute value of vector V               *
 *                                                       *
 * Algorithm:                                            *
 *    1) Assigning 0-bit to sign                         *
 *                                                       *
 *                                                       *
 * Note that this intrinsic is less efficient than       *
 * __riscv_vfabs_v                                       *
 *********************************************************
*/
 
#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

vfloat64m1_t __riscv_vfabs_f64m1(vfloat64m1_t v, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m1(avl);
    return __riscv_vreinterpret_v_u64m1_f64m1(
        __riscv_vand_vv_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(v),
            __riscv_vmv_v_x_u64m1(0x7fffffffffffffff, vl), vl
         )
    );
}

vfloat64m2_t __riscv_vfabs_f64m2(vfloat64m2_t v, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m2(avl);
    return __riscv_vreinterpret_v_u64m2_f64m2(
        __riscv_vand_vv_u64m2(
            __riscv_vreinterpret_v_f64m2_u64m2(v),
            __riscv_vmv_v_x_u64m2(0x7fffffffffffffff, vl), vl
         )
    );
}

vfloat64m4_t __riscv_vfabs_f64m4(vfloat64m4_t v, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m4(avl);
    return __riscv_vreinterpret_v_u64m4_f64m4(
        __riscv_vand_vv_u64m4(
            __riscv_vreinterpret_v_f64m4_u64m4(v),
            __riscv_vmv_v_x_u64m4(0x7fffffffffffffff, vl), vl
         )
    );
}

vfloat64m8_t __riscv_vfabs_f64m8(vfloat64m8_t v, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m8(avl);
    return __riscv_vreinterpret_v_u64m8_f64m8(
        __riscv_vand_vv_u64m8(
            __riscv_vreinterpret_v_f64m8_u64m8(v),
            __riscv_vmv_v_x_u64m8(0x7fffffffffffffff, vl), vl
         )
    );
}


vfloat32m1_t __riscv_vfabs_f32m1(vfloat32m1_t v, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m1(avl);
    return __riscv_vreinterpret_v_u32m1_f32m1(
        __riscv_vand_vv_u32m1(
            __riscv_vreinterpret_v_f32m1_u32m1(v),
            __riscv_vmv_v_x_u32m1(0x7fffffff, vl), vl
         )
    );
}

vfloat32m2_t __riscv_vfabs_f32m2(vfloat32m2_t v, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m2(avl);
    return __riscv_vreinterpret_v_u32m2_f32m2(
        __riscv_vand_vv_u32m2(
            __riscv_vreinterpret_v_f32m2_u32m2(v),
            __riscv_vmv_v_x_u32m2(0x7fffffff, vl), vl
         )
    );
}

vfloat32m4_t __riscv_vfabs_f32m4(vfloat32m4_t v, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m4(avl);
    return __riscv_vreinterpret_v_u32m4_f32m4(
        __riscv_vand_vv_u32m4(
            __riscv_vreinterpret_v_f32m4_u32m4(v),
            __riscv_vmv_v_x_u32m4(0x7fffffff, vl), vl
         )
    );
}

vfloat32m8_t __riscv_vfabs_f32m8(vfloat32m8_t v, size_t avl) {
    size_t vl = __riscv_vsetvl_e32m8(avl);
    return __riscv_vreinterpret_v_u32m8_f32m8(
        __riscv_vand_vv_u32m8(
            __riscv_vreinterpret_v_f32m8_u32m8(v),
            __riscv_vmv_v_x_u32m8(0x7fffffff, vl), vl
         )
    );
}

#if (defined(__riscv_zvfh) || defined(__riscv_zvfhmin))

vfloat16m1_t __riscv_vfabs_f16m1(vfloat16m1_t v, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m1(avl);
    return __riscv_vreinterpret_v_u16m1_f16m1(
        __riscv_vand_vv_u16m1(
            __riscv_vreinterpret_v_f16m1_u16m1(v),
            __riscv_vmv_v_x_u16m1(0x7fff, vl), vl
         )
    );
}

vfloat16m2_t __riscv_vfabs_f16m2(vfloat16m2_t v, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m2(avl);
    return __riscv_vreinterpret_v_u16m2_f16m2(
        __riscv_vand_vv_u16m2(
            __riscv_vreinterpret_v_f16m2_u16m2(v),
            __riscv_vmv_v_x_u16m2(0x7fff, vl), vl
         )
    );
}

vfloat16m4_t __riscv_vfabs_f16m4(vfloat16m4_t v, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m4(avl);
    return __riscv_vreinterpret_v_u16m4_f16m4(
        __riscv_vand_vv_u16m4(
            __riscv_vreinterpret_v_f16m4_u16m4(v),
            __riscv_vmv_v_x_u16m4(0x7fff, vl), vl
         )
    );
}

vfloat16m8_t __riscv_vfabs_f16m8(vfloat16m8_t v, size_t avl) {
    size_t vl = __riscv_vsetvl_e16m8(avl);
    return __riscv_vreinterpret_v_u16m8_f16m8(
        __riscv_vand_vv_u16m8(
            __riscv_vreinterpret_v_f16m8_u16m8(v),
            __riscv_vmv_v_x_u16m8(0x7fff, vl), vl
         )
    );
}

#endif /* __riscv_zvfh || __riscv_zvfhmin */

#endif /* __riscv_v_intrinsic */