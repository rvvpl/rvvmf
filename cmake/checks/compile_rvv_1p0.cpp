/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2026 The RVVMF Authors (Evgeny Kozinov)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  compile_rvv_1p0.cpp                          *
 *   Contains: checking for compiler support for RVV 1.0 *
 *                                                       *
 *                                                       *
 *********************************************************
*/

#include <riscv_vector.h>

#define ARRAY_LEN 100

void test()
{
  size_t bs_m1 = __riscv_vsetvl_e32m1(ARRAY_LEN);
  size_t bs_m2 = __riscv_vsetvl_e32m2(ARRAY_LEN);
  size_t bs_m4 = __riscv_vsetvl_e32m4(ARRAY_LEN);
  size_t bs_m8 = __riscv_vsetvl_e32m8(ARRAY_LEN);
}

int main() 
{
  test();
  return 0; 
}