/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025-2026 The RVVMF Authors 
 *(Anastasia Vodeneeva, Valentin Volokitin)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  trig_reduction.h                             *
 *   Contains: prototypes intrinsic functions for        *
 *             argument reduction                        *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

#include <cstdint>
#include <cfloat>

void __riscv_vreduction_pi4_f64m1(vfloat64m1_t vx, vfloat64m1_t& redx, vfloat64m1_t& redx_lo, 
                                  vuint64m1_t& quadrants, vbool64_t& mask_pi_4, size_t avl);
void __riscv_vreduction_pi4_f64m2(vfloat64m2_t vx, vfloat64m2_t& redx, vfloat64m2_t& redx_lo, 
                                  vuint64m2_t& quadrants, vbool32_t& mask_pi_4, size_t avl);
void __riscv_vreduction_pi4_f64m4(vfloat64m4_t vx, vfloat64m4_t& redx, vfloat64m4_t& redx_lo, 
                                  vuint64m4_t& quadrants, vbool16_t& mask_pi_4, size_t avl);

void __riscv_vreduction_pi4_f32m1(vfloat32m1_t vx, vfloat32m1_t& redx, vfloat32m1_t& redx_lo, 
                                  vuint32m1_t& quadrants, vbool32_t& mask_pi_4, size_t avl);
void __riscv_vreduction_pi4_f32m2(vfloat32m2_t vx, vfloat32m2_t& redx, vfloat32m2_t& redx_lo, 
                                  vuint32m2_t& quadrants, vbool16_t& mask_pi_4, size_t avl);
void __riscv_vreduction_pi4_f32m4(vfloat32m4_t vx, vfloat32m4_t& redx, vfloat32m4_t& redx_lo, 
                                  vuint32m4_t& quadrants, vbool8_t& mask_pi_4, size_t avl);

#ifdef __riscv_zvfh

void __riscv_vreduction_pi4_f16m1(vfloat16m1_t vx, vfloat16m1_t& redx, vfloat16m1_t& redx_lo, 
                                  vuint16m1_t& quadrants, vbool16_t& mask_pi_4, size_t avl);
void __riscv_vreduction_pi4_f16m2(vfloat16m2_t vx, vfloat16m2_t& redx, vfloat16m2_t& redx_lo, 
                                  vuint16m2_t& quadrants, vbool8_t& mask_pi_4, size_t avl);
void __riscv_vreduction_pi4_f16m4(vfloat16m4_t vx, vfloat16m4_t& redx, vfloat16m4_t& redx_lo, 
                                  vuint16m4_t& quadrants, vbool4_t& mask_pi_4, size_t avl);

#endif /* __riscv_zvfh */

#endif /* __riscv_v_intrinsic */