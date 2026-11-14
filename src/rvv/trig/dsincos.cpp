/* 
 *========================================================
 * Copyright (c) RVVPL and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2025-2026 The RVVMF Authors  
 * (Valentin Volokitin, Dmitrii Suvorov)
 *
 * Distributed under the BSD 4-Clause License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 *
 *********************************************************
 *                                                       *
 *   File:  dsincos.cpp                                  *
 *   Contains: intrinsic functions sin, cos, sincos      *
 *             for float64                               *
 *                                                       *
 * Input vector register V with any floating point value *
 * Input AVL number of elements in vector register       *
 *                                                       *
 * Returns the value of a trigonometric function         *
 * of vector V                                           *
 *                                                       *
 * Algorithm:                                            *
 *    1) Argument reduction by modified                  *
 *       Payne-Hanek algorithm to [0, Pi/4]              *
 *    2) Reduction to [-0x1p-6, 0x1p-6]                  *
 *    3) Calculating sincos                              *
 *    4) Restoring the value of a function using         *
 *       formulas and table values:                      *
 *       sin(x+a)=sin(x)*cos(a)+cos(x)*sin(a)            *
 *       cos(x+a)=cos(x)*cos(a)-sin(x)*sin(a)            *
 *                                                       *
 * Note that this implementation provides 0.501 ulp      *
 *                                                       *
 *********************************************************
*/

#ifdef __riscv_v_intrinsic
#include "riscv_vector.h"

#include <cstdint>
#include <cfloat>
#include <cmath>

#include "trig_reduction.h"

double table_cos_hi [] = {
    0x1p+0,
    0x1.ffc00155527d3p-1, 0x1.ffb7c1b2fd3dbp-1, 0x1.ffaf0222ba3bdp-1, 0x1.ffa5c2a6b95b7p-1,
    0x1.ff9c03414a7bap-1, 0x1.ff91c3f4dd75ap-1, 0x1.ff8704c4021c8p-1, 0x1.ff7bc5b1683c6p-1,
    0x1.ff7006bfdf99fp-1, 0x1.ff63c7f257f16p-1, 0x1.ff57094be0f62p-1, 0x1.ff49cacfaa51bp-1,
    0x1.ff3c0c8103a31p-1, 0x1.ff2dce635c7ddp-1, 0x1.ff1f107a44695p-1, 0x1.ff0fd2c96adfcp-1,
    0x1.ff0015549f4d3p-1, 0x1.fedf1b2f0f71dp-1, 0x1.febc222a8ef9fp-1, 0x1.fe972a6a16e73p-1,
    0x1.fe7034129ef6fp-1, 0x1.fe473f4b1d7d6p-1, 0x1.fe1c4c3c873ebp-1, 0x1.fdef5b11cf45dp-1,
    0x1.fdc06bf7e6b9bp-1, 0x1.fd8f7f1dbcb07p-1, 0x1.fd5c94b43ep-1, 0x1.fd27acee550dbp-1,
    0x1.fcf0c800e99b1p-1, 0x1.fcb7e622e090ep-1, 0x1.fc7d078d1bc88p-1, 0x1.fc402c7a79d2cp-1,
    0x1.fc015527d5bd3p-1, 0x1.fb7db2bfe0695p-1, 0x1.faf22263c4bd3p-1, 0x1.fa5ea641c36f2p-1,
    0x1.f9c340a7cc428p-1, 0x1.f91ff40374d01p-1, 0x1.f874c2e1eecf6p-1, 0x1.f7c1afeffde24p-1,
    0x1.f706bdf9ece1cp-1, 0x1.f643efeb82acdp-1, 0x1.f57948cff6797p-1, 0x1.f4a6cbd1e3a79p-1,
    0x1.f3cc7c3b3d16ep-1, 0x1.f2ea5d753ffedp-1, 0x1.f20073086649fp-1, 0x1.f10ec09c5873bp-1,
    0x1.f01549f7deea1p-1, 0x1.ee0b1fbc0f11cp-1, 0x1.ebe214f76efa8p-1, 0x1.e99a4c3a7cd83p-1,
    0x1.e733ea0193d4p-1, 0x1.e4af14b2a449cp-1, 0x1.e20bf49acd6c1p-1, 0x1.df4ab3ebd875ep-1,
    0x1.dc6b7eb995912p-1, 0x1.d96e82f71a9dcp-1, 0x1.d653f073e404p-1, 0x1.d31bf8d8d7c06p-1,
    0x1.cfc6cfa52ad9fp-1, 0x1.cc54aa2b2972ep-1, 0x1.c8c5bf8ce1a84p-1, 0x1.c51a48b8b175ep-1,
    0x1.c1528065b7d5p-1, 0x1.b96eeef58840ep-1, 0x1.b11d04162a4c6p-1, 0x1.a85ed4373e02dp-1,
    0x1.9f368ed912f85p-1, 0x1.95a67e00cb1fdp-1, 0x1.8bb105a5dc9p-1, 0x1.8158a31916d5dp-1,
    0x1.769fec655211fp-1, 0x1.6b898fa9efb5dp-1
};
    
double table_cos_lo [] = {
    0x0p0,
    -0x1.3b54492d89b5bp-55, 0x1.d9ba5b9bd6747p-55, -0x1.426013ff0feb7p-57, 0x1.8b4dba2185a75p-55,
    0x1.991f4be6c59bfp-57, -0x1.178e491e22647p-59, -0x1.d80c59264b631p-57, 0x1.592b157bdcdbbp-55,
    -0x1.8b3b560648d5fp-56, 0x1.cd79b5650bf14p-55, 0x1.780a6d6b9aac9p-55, 0x1.6a3bc928b4a38p-55,
    0x1.4856dbddc0e66p-56, 0x1.f7fd699bcd953p-56, 0x1.2552d1d5613d6p-58, -0x1.4a81bed63934bp-55,
    0x1.328387b99426fp-55, -0x1.358030423989ep-58, 0x1.7934934f54c77p-58, -0x1.9b4127ab3d548p-58,
    -0x1.cbf4337c96f97p-57, 0x1.a001dd2227ed4p-55, -0x1.5a9c9057c4a02p-60, -0x1.e833a0863483p-56,
    0x1.31902b535f8dbp-55, -0x1.c752ecfaa985bp-55, -0x1.2e768cb4f92f9p-57, 0x1.c3f64be936b41p-55,
    0x1.ea3d786d186acp-57, 0x1.efee620ded043p-55, 0x1.075d2447db685p-55, 0x1.f515d61e3b7d1p-56,
    0x1.b68f35094efb8p-55, 0x1.21dadf4f65ab1p-55, -0x1.52ace133a2769p-58, 0x1.04da6ed17cc7cp-59,
    0x1.c5b6b063b7462p-55, -0x1.7d03f4d3a9e4cp-57, -0x1.c6514e1332b16p-55, -0x1.8f55bc47540b1p-56,
    -0x1.698c80c36dcb4p-55, 0x1.6b00ac1fe28acp-56, 0x1.e3a0d3e03b1d4p-57, 0x1.13df0edaebb57p-55,
    -0x1.21a3ad28a3494p-57, 0x1.cc4215f56d583p-55, 0x1.b940416c1984bp-56, 0x1.d9072762c1283p-55,
    0x1.d3c1e99e5cafdp-55, -0x1.bfd2380bbc3b1p-59, -0x1.02f9f12ba543ep-55, -0x1.2264b1bc53ce8p-55,
    -0x1.6428b3546ce13p-55, -0x1.68ca02e8a6833p-55, -0x1.660aec7ef636bp-58, -0x1.e2d8a7e6736c4p-55,
    0x1.4b364776dcd35p-58, 0x1.ff61bd5d2039dp-55, -0x1.76236434bec37p-55, 0x1.e60dd3089cbddp-56,
    0x1.8b5b5508f2a0dp-55, 0x1.4ee162ba83a98p-57, 0x1.ab3d1a1590123p-56, -0x1.1bbb43b9aa88p-57,
    -0x1.892111312e828p-55, 0x1.45a3cc78fadep-58, 0x1.1dd561efbc0c2p-56, 0x1.9be06385ec792p-57,
    -0x1.1d200c5791606p-55, -0x1.0befda21f862dp-55, 0x1.863e03e9474c1p-55, -0x1.de8b90b8228dep-57,
    -0x1.827d5cf8c68c5p-57, 0x1.15ac786ccf4b2p-56
};
    
double table_sin_hi [] = {
    0x0p+0,
    0x1.ffeaaaeeee86fp-6, 0x1.0ff334d8e27f7p-5, 0x1.1ff0d03d81efcp-5, 0x1.2fee23a5eed0bp-5,
    0x1.3feb2b12d45d5p-5, 0x1.4fe7e284f0cfdp-5, 0x1.5fe445fd1660fp-5, 0x1.6fe0517c2c48p-5,
    0x1.7fdc01032fba9p-5, 0x1.8fd7509334ec1p-5, 0x1.9fd23c2d680dfp-5, 0x1.afccbfd30e4eep-5,
    0x1.bfc6d78586dacp-5, 0x1.cfc07f464bda5p-5, 0x1.dfb9b316f372ap-5, 0x1.efb26ef930c4dp-5,
    0x1.ffaaaeeed4edbp-5, 0x1.0fccd58e190ecp-4, 0x1.1fc343d808befp-4, 0x1.2fb8925ecc92p-4,
    0x1.3facb12d1755bp-4, 0x1.4f9f904ecb8ffp-4, 0x1.5f911fd10b737p-4, 0x1.6f814fc248d1fp-4,
    0x1.7f701032550e4p-4, 0x1.8f5d5132710c4p-4, 0x1.9f4902d55d1f9p-4, 0x1.af33152f68f8fp-4,
    0x1.bf1b78568391dp-4, 0x1.cf021c624b168p-4, 0x1.dee6f16c1cce6p-4, 0x1.eec9e78f2502bp-4,
    0x1.feaaeee86ee36p-4, 0x1.0f3378ddd71d1p-3, 0x1.1f0d3d7afceafp-3, 0x1.2ee285e4ab88fp-3,
    0x1.3eb312c5d66cbp-3, 0x1.4e7ea4dc5f27bp-3, 0x1.5e44fcfa126f3p-3, 0x1.6e05dc05a4d4cp-3,
    0x1.7dc102fbaf2b5p-3, 0x1.8d7632efaa944p-3, 0x1.9d252d0cec312p-3, 0x1.accdb297a0765p-3,
    0x1.bc6f84edc6199p-3, 0x1.cc0a6588289a3p-3, 0x1.db9e15fb5a5dp-3, 0x1.eb2a57f8ae5a3p-3,
    0x1.faaeed4f31577p-3, 0x1.0cd00cef36436p-2, 0x1.1c37d64c6b876p-2, 0x1.2b8ddc43eb49fp-2,
    0x1.3ad129769d3d8p-2, 0x1.4a00c9b0f3d2p-2, 0x1.591bc9fa2f597p-2, 0x1.682138a38d7f7p-2,
    0x1.7710255764214p-2, 0x1.85e7a12826949p-2, 0x1.94a6be9f546c5p-2, 0x1.a34c91cc50ccap-2,
    0x1.b1d8305321617p-2, 0x1.c048b17b140a3p-2, 0x1.ce9d2e3d4a51fp-2, 0x1.dcd4c15329c9ap-2,
    0x1.eaee8744b05fp-2, 0x1.0362939c69955p-1, 0x1.110d0c4b69c3bp-1, 0x1.1e7343236574cp-1,
    0x1.2b91dea88421ep-1, 0x1.386597456282bp-1, 0x1.44eb381cf386bp-1, 0x1.511f9fd7b351cp-1,
    0x1.5cffc16bf8f0dp-1, 0x1.6888a4e134b2fp-1
};
    
double table_sin_lo [] = {
    0x0p+0,
    -0x1.cd406fb224ae2p-60, -0x1.6d8cae1647ff2p-60, 0x1.b74e4aa1f4c7fp-59, -0x1.0a51c2de848a2p-59,
    0x1.4ec54203d1c11p-60, 0x1.96c944d7c4a46p-60, 0x1.d7a6e37a23793p-59, 0x1.9433cf7d4616ap-59,
    -0x1.599bdf46e997ap-59, 0x1.78626c4629dcdp-60, -0x1.70b1a2ba5f59p-60, -0x1.4584e4b331072p-59,
    0x1.8e4fd03dbf236p-62, -0x1.f7e9c7fb257aap-61, -0x1.b3adb2cd8fe77p-61, 0x1.f9586df0c98ddp-60,
    -0x1.2d16d32684b69p-59, -0x1.ef54e5b79b73cp-58, -0x1.f3d32e6f3be4fp-58, -0x1.1aceeb6944bb7p-60,
    -0x1.921915299468bp-58, 0x1.6fdd8bf8cda76p-58, -0x1.0184f02be9102p-58, -0x1.1e7a14e279b6ap-59,
    0x1.afc2d1800501ap-60, 0x1.ae29046de7c88p-61, 0x1.2696d7eac1dc1p-58, 0x1.3f7698bdd23f1p-58,
    0x1.e91841dea4cc8p-58, -0x1.f6aedf73fb13fp-62, -0x1.50f8e2fb71673p-59, 0x1.3535175887f66p-60,
    -0x1.afcb2bcc6f03bp-59, 0x1.d8468724f0f9ep-57, -0x1.6ef95099769a5p-57, -0x1.e4d0f05dee058p-57,
    0x1.47d666b66cb91p-57, 0x1.949db2ac072fcp-58, -0x1.6f443063f89b6p-57, -0x1.32c5c8b81c919p-66,
    0x1.5ab50e23c97c3p-59, -0x1.20fa262cbb953p-57, 0x1.9c43d80b1137dp-58, -0x1.9883b57d6cdeap-58,
    0x1.9c1a56a7b0cabp-57, -0x1.868d09bc87c6bp-57, -0x1.32e20d6cc6fc2p-57, -0x1.0be06af572cebp-57,
    -0x1.15d88508e32b8p-57, -0x1.9fb0a0c93e2b4p-56, 0x1.46076fe0dcff4p-56, 0x1.1553899f2d807p-57,
    0x1.03d550487839ap-63, 0x1.823ba6bb08eadp-56, 0x1.7c74bac3fe0cbp-57, -0x1.d889202444aadp-56,
    -0x1.6ead7314bb6cep-57, 0x1.8a40e9b5facep-56, -0x1.69ce13e683f58p-56, -0x1.a310e3b50cecdp-58,
    -0x1.ae242cb99f519p-56, 0x1.19fe6757e9fa7p-57, -0x1.2fc8a12dae298p-57, 0x1.0d4c6e171fd9ap-56,
    -0x1.789b43c9b027dp-58, -0x1.2d8cd78397b01p-55, 0x1.d918998809981p-55, 0x1.22a3fa4f41d5ap-56,
    -0x1.fa371db216abp-55, -0x1.10fada93b07a8p-56, -0x1.3ed6c1e6a5505p-55, -0x1.5c0e861c48831p-55,
    0x1.96cb370eb578ap-55, -0x1.6b7d37644d5e6p-55
};

void __riscv_vsincos_kernel_f64m1(vfloat64m1_t x, vfloat64m1_t x_l, vfloat64m1_t& sinrx, vfloat64m1_t& cosrx, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m1(avl);
    vfloat64m1_t x_lo, x_hi, tmp;
    x_hi = __riscv_vmv_v_v_f64m1(x, vl);
    x_lo = __riscv_vmv_v_v_f64m1(x_l, vl);

    vuint64m1_t ix = __riscv_vreinterpret_v_f64m1_u64m1(x_hi);
    vuint64m1_t c_u64 = __riscv_vadd_vx_u64m1(ix, 0x0000800000000000, vl);
    c_u64 = __riscv_vand_vx_u64m1(c_u64, 0x7fff000000000000, vl);
    vuint64m1_t ind = __riscv_vsrl_vx_u64m1(c_u64, 48, vl);
    ind = __riscv_vsub_vx_u64m1(ind, 16287, vl);
    
    vbool64_t mask = __riscv_vmsltu_vx_u64m1_b64(c_u64, 0x3fa0000000000000, vl);
    vfloat64m1_t c = __riscv_vfmerge_vfm_f64m1(
               __riscv_vreinterpret_v_u64m1_f64m1(c_u64), 0.0, mask, vl);
    ind = __riscv_vmerge_vxm_u64m1(ind, 0, mask, vl);
    ind = __riscv_vsll_vx_u64m1(ind, 3, vl);

    tmp = __riscv_vfsub_vv_f64m1(x_hi, c, vl);
    x_hi = __riscv_vfadd_vv_f64m1(tmp, x_lo, vl);
    tmp = __riscv_vfsub_vv_f64m1(x_hi, tmp, vl);
    x_lo = __riscv_vfsub_vv_f64m1(x_lo, tmp, vl);
    
    tmp = __riscv_vfmul_vv_f64m1(x_lo, x_hi, vl);
    tmp = __riscv_vfmul_vf_f64m1(tmp, 2.0, vl);
    vfloat64m1_t x2 = __riscv_vfmul_vv_f64m1(x_hi, x_hi, vl);
    vfloat64m1_t x2_lo = __riscv_vfmsub_vv_f64m1(x_hi, x_hi, x2, vl);
    x2_lo = __riscv_vfadd_vv_f64m1(tmp, x2_lo, vl);

    vfloat64m1_t sin_r_h = __riscv_vfmadd_vf_f64m1(x2, -0x1.a01947af8f235p-13, 
                            __riscv_vfmv_v_f_f64m1(0x1.11111110f434dp-7, vl), vl);
    sin_r_h = __riscv_vfmadd_vv_f64m1(sin_r_h, x2, 
               __riscv_vfmv_v_f_f64m1(-0x1.5555555555555p-3, vl), vl);
    tmp = __riscv_vfmul_vv_f64m1(sin_r_h, x_lo, vl);
    tmp = __riscv_vfmadd_vv_f64m1(sin_r_h, x_hi, tmp, vl);
    sin_r_h = __riscv_vfmadd_vv_f64m1(tmp, x2, x_hi, vl);
    vfloat64m1_t sin_r_l = __riscv_vfsub_vv_f64m1(x_hi, sin_r_h, vl);
    sin_r_l = __riscv_vfmadd_vv_f64m1(tmp, x2, sin_r_l, vl);
    sin_r_l = __riscv_vfadd_vv_f64m1(sin_r_l, x_lo, vl);
    
    vfloat64m1_t cos_r_l = __riscv_vfmadd_vf_f64m1(x2, -0x1.6c15ff7a5b7a5p-10,
                            __riscv_vfmv_v_f_f64m1(0x1.555555551fd37p-5, vl), vl);
    cos_r_l = __riscv_vfmadd_vv_f64m1(cos_r_l, x2, 
               __riscv_vfmv_v_f_f64m1(-0x1.fffffffffffffp-2, vl), vl);
    vfloat64m1_t cos_r_h = __riscv_vfmadd_vv_f64m1(cos_r_l, x2, 
               __riscv_vfmv_v_f_f64m1(0x1p0, vl), vl);
    tmp = __riscv_vfrsub_vf_f64m1(cos_r_h, 1.0, vl);
    tmp = __riscv_vfmadd_vv_f64m1(cos_r_l, x2, tmp, vl);
    cos_r_l = __riscv_vfmadd_vv_f64m1(cos_r_l, x2_lo, tmp, vl);
    
    vfloat64m1_t cos_hi = __riscv_vloxei64_v_f64m1(table_cos_hi, ind, vl);
    vfloat64m1_t cos_lo = __riscv_vloxei64_v_f64m1(table_cos_lo, ind, vl);
    vfloat64m1_t sin_hi = __riscv_vloxei64_v_f64m1(table_sin_hi, ind, vl);
    vfloat64m1_t sin_lo = __riscv_vloxei64_v_f64m1(table_sin_lo, ind, vl);
    
    sinrx = __riscv_vfmul_vv_f64m1(cos_hi, sin_r_l, vl);
    sinrx = __riscv_vfmacc_vv_f64m1(sinrx, cos_lo, sin_r_h, vl);
    sinrx = __riscv_vfmacc_vv_f64m1(sinrx, sin_lo, cos_r_h, vl);
    sinrx = __riscv_vfmacc_vv_f64m1(sinrx, sin_hi, cos_r_l, vl);
    sinrx = __riscv_vfmacc_vv_f64m1(sinrx, cos_hi, sin_r_h, vl);
    sinrx = __riscv_vfmacc_vv_f64m1(sinrx, sin_hi, cos_r_h, vl);
    
    cosrx = __riscv_vfmul_vv_f64m1(sin_hi, sin_r_l, vl);
    cosrx = __riscv_vfmacc_vv_f64m1(cosrx, sin_lo, sin_r_h, vl);
    cosrx = __riscv_vfmsac_vv_f64m1(cosrx, cos_lo, cos_r_h, vl);
    cosrx = __riscv_vfmacc_vv_f64m1(cosrx, cos_hi, cos_r_l, vl);
    cosrx = __riscv_vfnmsac_vv_f64m1(cosrx, sin_hi, sin_r_h, vl);
    cosrx = __riscv_vfmacc_vv_f64m1(cosrx, cos_hi, cos_r_h, vl);    
}

void __riscv_vsincos_kernel_f64m2(vfloat64m2_t x, vfloat64m2_t x_l, vfloat64m2_t& sinrx, vfloat64m2_t& cosrx, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m2(avl);
    vfloat64m2_t x_lo, x_hi, tmp;
    x_hi = __riscv_vmv_v_v_f64m2(x, vl);
    x_lo = __riscv_vmv_v_v_f64m2(x_l, vl);

    vuint64m2_t ix = __riscv_vreinterpret_v_f64m2_u64m2(x_hi);
    vuint64m2_t c_u64 = __riscv_vadd_vx_u64m2(ix, 0x0000800000000000, vl);
    c_u64 = __riscv_vand_vx_u64m2(c_u64, 0x7fff000000000000, vl);
    vuint64m2_t ind = __riscv_vsrl_vx_u64m2(c_u64, 48, vl);
    ind = __riscv_vsub_vx_u64m2(ind, 16287, vl);
    
    vbool32_t mask = __riscv_vmsltu_vx_u64m2_b32(c_u64, 0x3fa0000000000000, vl);
    vfloat64m2_t c = __riscv_vfmerge_vfm_f64m2(
               __riscv_vreinterpret_v_u64m2_f64m2(c_u64), 0.0, mask, vl);
    ind = __riscv_vmerge_vxm_u64m2(ind, 0, mask, vl);
    ind = __riscv_vsll_vx_u64m2(ind, 3, vl);

    tmp = __riscv_vfsub_vv_f64m2(x_hi, c, vl);
    x_hi = __riscv_vfadd_vv_f64m2(tmp, x_lo, vl);
    tmp = __riscv_vfsub_vv_f64m2(x_hi, tmp, vl);
    x_lo = __riscv_vfsub_vv_f64m2(x_lo, tmp, vl);
    
    tmp = __riscv_vfmul_vv_f64m2(x_lo, x_hi, vl);
    tmp = __riscv_vfmul_vf_f64m2(tmp, 2.0, vl);
    vfloat64m2_t x2 = __riscv_vfmul_vv_f64m2(x_hi, x_hi, vl);
    vfloat64m2_t x2_lo = __riscv_vfmsub_vv_f64m2(x_hi, x_hi, x2, vl);
    x2_lo = __riscv_vfadd_vv_f64m2(tmp, x2_lo, vl);

    vfloat64m2_t sin_r_h = __riscv_vfmadd_vf_f64m2(x2, -0x1.a01947af8f235p-13, 
                            __riscv_vfmv_v_f_f64m2(0x1.11111110f434dp-7, vl), vl);
    sin_r_h = __riscv_vfmadd_vv_f64m2(sin_r_h, x2, 
               __riscv_vfmv_v_f_f64m2(-0x1.5555555555555p-3, vl), vl);
    tmp = __riscv_vfmul_vv_f64m2(sin_r_h, x_lo, vl);
    tmp = __riscv_vfmadd_vv_f64m2(sin_r_h, x_hi, tmp, vl);
    sin_r_h = __riscv_vfmadd_vv_f64m2(tmp, x2, x_hi, vl);
    vfloat64m2_t sin_r_l = __riscv_vfsub_vv_f64m2(x_hi, sin_r_h, vl);
    sin_r_l = __riscv_vfmadd_vv_f64m2(tmp, x2, sin_r_l, vl);
    sin_r_l = __riscv_vfadd_vv_f64m2(sin_r_l, x_lo, vl);
    
    vfloat64m2_t cos_r_l = __riscv_vfmadd_vf_f64m2(x2, -0x1.6c15ff7a5b7a5p-10,
                            __riscv_vfmv_v_f_f64m2(0x1.555555551fd37p-5, vl), vl);
    cos_r_l = __riscv_vfmadd_vv_f64m2(cos_r_l, x2, 
               __riscv_vfmv_v_f_f64m2(-0x1.fffffffffffffp-2, vl), vl);
    vfloat64m2_t cos_r_h = __riscv_vfmadd_vv_f64m2(cos_r_l, x2, 
               __riscv_vfmv_v_f_f64m2(0x1p0, vl), vl);
    tmp = __riscv_vfrsub_vf_f64m2(cos_r_h, 1.0, vl);
    tmp = __riscv_vfmadd_vv_f64m2(cos_r_l, x2, tmp, vl);
    cos_r_l = __riscv_vfmadd_vv_f64m2(cos_r_l, x2_lo, tmp, vl);
    
    vfloat64m2_t cos_hi = __riscv_vloxei64_v_f64m2(table_cos_hi, ind, vl);
    vfloat64m2_t cos_lo = __riscv_vloxei64_v_f64m2(table_cos_lo, ind, vl);
    vfloat64m2_t sin_hi = __riscv_vloxei64_v_f64m2(table_sin_hi, ind, vl);
    vfloat64m2_t sin_lo = __riscv_vloxei64_v_f64m2(table_sin_lo, ind, vl);
    
    sinrx = __riscv_vfmul_vv_f64m2(cos_hi, sin_r_l, vl);
    sinrx = __riscv_vfmacc_vv_f64m2(sinrx, cos_lo, sin_r_h, vl);
    sinrx = __riscv_vfmacc_vv_f64m2(sinrx, sin_lo, cos_r_h, vl);
    sinrx = __riscv_vfmacc_vv_f64m2(sinrx, sin_hi, cos_r_l, vl);
    sinrx = __riscv_vfmacc_vv_f64m2(sinrx, cos_hi, sin_r_h, vl);
    sinrx = __riscv_vfmacc_vv_f64m2(sinrx, sin_hi, cos_r_h, vl);
    
    cosrx = __riscv_vfmul_vv_f64m2(sin_hi, sin_r_l, vl);
    cosrx = __riscv_vfmacc_vv_f64m2(cosrx, sin_lo, sin_r_h, vl);
    cosrx = __riscv_vfmsac_vv_f64m2(cosrx, cos_lo, cos_r_h, vl);
    cosrx = __riscv_vfmacc_vv_f64m2(cosrx, cos_hi, cos_r_l, vl);
    cosrx = __riscv_vfnmsac_vv_f64m2(cosrx, sin_hi, sin_r_h, vl);
    cosrx = __riscv_vfmacc_vv_f64m2(cosrx, cos_hi, cos_r_h, vl);    
}

void __riscv_vsincos_kernel_f64m4(vfloat64m4_t x, vfloat64m4_t x_l, vfloat64m4_t& sinrx, vfloat64m4_t& cosrx, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x_lo, x_hi, tmp;
    x_hi = __riscv_vmv_v_v_f64m4(x, vl);
    x_lo = __riscv_vmv_v_v_f64m4(x_l, vl);

    vuint64m4_t ix = __riscv_vreinterpret_v_f64m4_u64m4(x_hi);
    vuint64m4_t c_u64 = __riscv_vadd_vx_u64m4(ix, 0x0000800000000000, vl);
    c_u64 = __riscv_vand_vx_u64m4(c_u64, 0x7fff000000000000, vl);
    vuint64m4_t ind = __riscv_vsrl_vx_u64m4(c_u64, 48, vl);
    ind = __riscv_vsub_vx_u64m4(ind, 16287, vl);
    
    vbool16_t mask = __riscv_vmsltu_vx_u64m4_b16(c_u64, 0x3fa0000000000000, vl);
    vfloat64m4_t c = __riscv_vfmerge_vfm_f64m4(
               __riscv_vreinterpret_v_u64m4_f64m4(c_u64), 0.0, mask, vl);
    ind = __riscv_vmerge_vxm_u64m4(ind, 0, mask, vl);
    ind = __riscv_vsll_vx_u64m4(ind, 3, vl);

    tmp = __riscv_vfsub_vv_f64m4(x_hi, c, vl);
    x_hi = __riscv_vfadd_vv_f64m4(tmp, x_lo, vl);
    tmp = __riscv_vfsub_vv_f64m4(x_hi, tmp, vl);
    x_lo = __riscv_vfsub_vv_f64m4(x_lo, tmp, vl);
    
    tmp = __riscv_vfmul_vv_f64m4(x_lo, x_hi, vl);
    tmp = __riscv_vfmul_vf_f64m4(tmp, 2.0, vl);
    vfloat64m4_t x2 = __riscv_vfmul_vv_f64m4(x_hi, x_hi, vl);
    vfloat64m4_t x2_lo = __riscv_vfmsub_vv_f64m4(x_hi, x_hi, x2, vl);
    x2_lo = __riscv_vfadd_vv_f64m4(tmp, x2_lo, vl);

    vfloat64m4_t sin_r_h = __riscv_vfmadd_vf_f64m4(x2, -0x1.a01947af8f235p-13, 
                            __riscv_vfmv_v_f_f64m4(0x1.11111110f434dp-7, vl), vl);
    sin_r_h = __riscv_vfmadd_vv_f64m4(sin_r_h, x2, 
               __riscv_vfmv_v_f_f64m4(-0x1.5555555555555p-3, vl), vl);
    tmp = __riscv_vfmul_vv_f64m4(sin_r_h, x_lo, vl);
    tmp = __riscv_vfmadd_vv_f64m4(sin_r_h, x_hi, tmp, vl);
    sin_r_h = __riscv_vfmadd_vv_f64m4(tmp, x2, x_hi, vl);
    vfloat64m4_t sin_r_l = __riscv_vfsub_vv_f64m4(x_hi, sin_r_h, vl);
    sin_r_l = __riscv_vfmadd_vv_f64m4(tmp, x2, sin_r_l, vl);
    sin_r_l = __riscv_vfadd_vv_f64m4(sin_r_l, x_lo, vl);
    
    vfloat64m4_t cos_r_l = __riscv_vfmadd_vf_f64m4(x2, -0x1.6c15ff7a5b7a5p-10,
                            __riscv_vfmv_v_f_f64m4(0x1.555555551fd37p-5, vl), vl);
    cos_r_l = __riscv_vfmadd_vv_f64m4(cos_r_l, x2, 
               __riscv_vfmv_v_f_f64m4(-0x1.fffffffffffffp-2, vl), vl);
    vfloat64m4_t cos_r_h = __riscv_vfmadd_vv_f64m4(cos_r_l, x2, 
               __riscv_vfmv_v_f_f64m4(0x1p0, vl), vl);
    tmp = __riscv_vfrsub_vf_f64m4(cos_r_h, 1.0, vl);
    tmp = __riscv_vfmadd_vv_f64m4(cos_r_l, x2, tmp, vl);
    cos_r_l = __riscv_vfmadd_vv_f64m4(cos_r_l, x2_lo, tmp, vl);
    
    vfloat64m4_t cos_hi = __riscv_vloxei64_v_f64m4(table_cos_hi, ind, vl);
    vfloat64m4_t cos_lo = __riscv_vloxei64_v_f64m4(table_cos_lo, ind, vl);
    vfloat64m4_t sin_hi = __riscv_vloxei64_v_f64m4(table_sin_hi, ind, vl);
    vfloat64m4_t sin_lo = __riscv_vloxei64_v_f64m4(table_sin_lo, ind, vl);
    
    sinrx = __riscv_vfmul_vv_f64m4(cos_hi, sin_r_l, vl);
    sinrx = __riscv_vfmacc_vv_f64m4(sinrx, cos_lo, sin_r_h, vl);
    sinrx = __riscv_vfmacc_vv_f64m4(sinrx, sin_lo, cos_r_h, vl);
    sinrx = __riscv_vfmacc_vv_f64m4(sinrx, sin_hi, cos_r_l, vl);
    sinrx = __riscv_vfmacc_vv_f64m4(sinrx, cos_hi, sin_r_h, vl);
    sinrx = __riscv_vfmacc_vv_f64m4(sinrx, sin_hi, cos_r_h, vl);
    
    cosrx = __riscv_vfmul_vv_f64m4(sin_hi, sin_r_l, vl);
    cosrx = __riscv_vfmacc_vv_f64m4(cosrx, sin_lo, sin_r_h, vl);
    cosrx = __riscv_vfmsac_vv_f64m4(cosrx, cos_lo, cos_r_h, vl);
    cosrx = __riscv_vfmacc_vv_f64m4(cosrx, cos_hi, cos_r_l, vl);
    cosrx = __riscv_vfnmsac_vv_f64m4(cosrx, sin_hi, sin_r_h, vl);
    cosrx = __riscv_vfmacc_vv_f64m4(cosrx, cos_hi, cos_r_h, vl);  
}

vfloat64m1_t __riscv_vsin_f64m1(vfloat64m1_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m1(avl);
    vfloat64m1_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool64_t mask_pi_4;
    vuint64m1_t quadrants;
    
    __riscv_vreduction_pi4_f64m1(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl); 
    __riscv_vsincos_kernel_f64m1(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f64m1(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f64m1(cosr, sinr, mask_pi_4, vl);
    
    vfloat64m1_t res;
    vfloat64m1_t nsinrx = __riscv_vfmul_vf_f64m1(sinrx, -1.0, vl);
    vfloat64m1_t ncosrx = __riscv_vfmul_vf_f64m1(cosrx, -1.0, vl);
    
    vuint64m1_t signx = __riscv_vand_vx_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(x), 
                                            0x8000000000000000, vl);  
    vbool64_t mask = __riscv_vmseq_vx_u64m1_b64(signx, 0, vl);
    res = __riscv_vmv_v_v_f64m1(sinrx, vl);
    sinrx = __riscv_vmerge_vvm_f64m1(nsinrx, sinrx, mask, vl);
    nsinrx = __riscv_vmerge_vvm_f64m1(res, nsinrx, mask, vl);
    res = __riscv_vmv_v_v_f64m1(cosrx, vl);
    cosrx = __riscv_vmerge_vvm_f64m1(ncosrx, cosrx, mask, vl);
    ncosrx = __riscv_vmerge_vvm_f64m1(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f64m1(res, sinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f64m1(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f64m1(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f64m1(res, ncosrx, mask, vl);

#ifndef __FAST_MATH__
    vuint64m1_t ix = __riscv_vand_vx_u64m1(
                 __riscv_vreinterpret_v_f64m1_u64m1(x), 0x7fffffffffffffff, vl);
    vbool64_t mask_NaN = __riscv_vmsgeu_vx_u64m1_b64 (ix, 0x7ff0000000000000, vl);
    res = __riscv_vmerge_vvm_f64m1(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b64(mask_NaN,
                  __riscv_vmsltu_vx_u64m1_b64(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b64(mask_NaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        res = __riscv_vfmerge_vfm_f64m1(res, x1, mask_NaN, vl);
    }
#endif

    return res;            
}

vfloat64m2_t __riscv_vsin_f64m2(vfloat64m2_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m2(avl);
    vfloat64m2_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool32_t mask_pi_4;
    vuint64m2_t quadrants;
    
    __riscv_vreduction_pi4_f64m2(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl); 
    __riscv_vsincos_kernel_f64m2(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f64m2(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f64m2(cosr, sinr, mask_pi_4, vl);
    
    vfloat64m2_t res;
    vfloat64m2_t nsinrx = __riscv_vfmul_vf_f64m2(sinrx, -1.0, vl);
    vfloat64m2_t ncosrx = __riscv_vfmul_vf_f64m2(cosrx, -1.0, vl);
    
    vuint64m2_t signx = __riscv_vand_vx_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(x), 
                                            0x8000000000000000, vl);  
    vbool32_t mask = __riscv_vmseq_vx_u64m2_b32(signx, 0, vl);
    res = __riscv_vmv_v_v_f64m2(sinrx, vl);
    sinrx = __riscv_vmerge_vvm_f64m2(nsinrx, sinrx, mask, vl);
    nsinrx = __riscv_vmerge_vvm_f64m2(res, nsinrx, mask, vl);
    res = __riscv_vmv_v_v_f64m2(cosrx, vl);
    cosrx = __riscv_vmerge_vvm_f64m2(ncosrx, cosrx, mask, vl);
    ncosrx = __riscv_vmerge_vvm_f64m2(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f64m2(res, sinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f64m2(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f64m2(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f64m2(res, ncosrx, mask, vl);

#ifndef __FAST_MATH__
    vuint64m2_t ix = __riscv_vand_vx_u64m2(
                 __riscv_vreinterpret_v_f64m2_u64m2(x), 0x7fffffffffffffff, vl);
    vbool32_t mask_NaN = __riscv_vmsgeu_vx_u64m2_b32 (ix, 0x7ff0000000000000, vl);
    res = __riscv_vmerge_vvm_f64m2(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b32(mask_NaN,
                  __riscv_vmsltu_vx_u64m2_b32(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b32(mask_NaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        res = __riscv_vfmerge_vfm_f64m2(res, x1, mask_NaN, vl);
    }
#endif

    return res;            
}

vfloat64m4_t __riscv_vsin_f64m4(vfloat64m4_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool16_t mask_pi_4;
    vuint64m4_t quadrants;
    
    __riscv_vreduction_pi4_f64m4(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl); 
    __riscv_vsincos_kernel_f64m4(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f64m4(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f64m4(cosr, sinr, mask_pi_4, vl);
    
    vfloat64m4_t res;
    vfloat64m4_t nsinrx = __riscv_vfmul_vf_f64m4(sinrx, -1.0, vl);
    vfloat64m4_t ncosrx = __riscv_vfmul_vf_f64m4(cosrx, -1.0, vl);
    
    vuint64m4_t signx = __riscv_vand_vx_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(x), 
                                            0x8000000000000000, vl);  
    vbool16_t mask = __riscv_vmseq_vx_u64m4_b16(signx, 0, vl);
    res = __riscv_vmv_v_v_f64m4(sinrx, vl);
    sinrx = __riscv_vmerge_vvm_f64m4(nsinrx, sinrx, mask, vl);
    nsinrx = __riscv_vmerge_vvm_f64m4(res, nsinrx, mask, vl);
    res = __riscv_vmv_v_v_f64m4(cosrx, vl);
    cosrx = __riscv_vmerge_vvm_f64m4(ncosrx, cosrx, mask, vl);
    ncosrx = __riscv_vmerge_vvm_f64m4(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f64m4(res, sinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f64m4(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f64m4(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f64m4(res, ncosrx, mask, vl);

#ifndef __FAST_MATH__
    vuint64m4_t ix = __riscv_vand_vx_u64m4(
                 __riscv_vreinterpret_v_f64m4_u64m4(x), 0x7fffffffffffffff, vl);
    vbool16_t mask_NaN = __riscv_vmsgeu_vx_u64m4_b16 (ix, 0x7ff0000000000000, vl);
    res = __riscv_vmerge_vvm_f64m4(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b16(mask_NaN,
                  __riscv_vmsltu_vx_u64m4_b16(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_NaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        res = __riscv_vfmerge_vfm_f64m4(res, x1, mask_NaN, vl);
    }
#endif

    return res;            
}

vfloat64m8_t __riscv_vsin_f64m8(vfloat64m8_t x, size_t avl)
{ 
    vfloat64m8_t res;
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1 = __riscv_vsin_f64m4(x1, vl);
    res = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e64m4(avl-vl);
        x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1 = __riscv_vsin_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}

vfloat64m1_t __riscv_vcos_f64m1(vfloat64m1_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m1(avl);
    vfloat64m1_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool64_t mask_pi_4;
    vuint64m1_t quadrants;
    
    __riscv_vreduction_pi4_f64m1(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f64m1(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f64m1(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f64m1(cosr, sinr, mask_pi_4, vl);
    
    vfloat64m1_t res = __riscv_vfmv_v_f_f64m1(0.0, vl);
    vfloat64m1_t nsinrx = __riscv_vfmul_vf_f64m1(sinrx, -1.0, vl);
    vfloat64m1_t ncosrx = __riscv_vfmul_vf_f64m1(cosrx, -1.0, vl);
    
    vbool64_t mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f64m1(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f64m1(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f64m1(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f64m1(res, sinrx, mask, vl);

#ifndef __FAST_MATH__
    vuint64m1_t ix = __riscv_vand_vx_u64m1(
                 __riscv_vreinterpret_v_f64m1_u64m1(x), 0x7fffffffffffffff, vl);
    vbool64_t mask_NaN = __riscv_vmsgeu_vx_u64m1_b64 (ix, 0x7ff0000000000000, vl);
    res = __riscv_vmerge_vvm_f64m1(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b64(mask_NaN,
                  __riscv_vmsltu_vx_u64m1_b64(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b64(mask_NaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        res = __riscv_vfmerge_vfm_f64m1(res, x1, mask_NaN, vl);
    }
#endif

    return res;
}

vfloat64m2_t __riscv_vcos_f64m2(vfloat64m2_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m2(avl);
    vfloat64m2_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool32_t mask_pi_4;
    vuint64m2_t quadrants;
    
    __riscv_vreduction_pi4_f64m2(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f64m2(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f64m2(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f64m2(cosr, sinr, mask_pi_4, vl);
    
    vfloat64m2_t res = __riscv_vfmv_v_f_f64m2(0.0, vl);
    vfloat64m2_t nsinrx = __riscv_vfmul_vf_f64m2(sinrx, -1.0, vl);
    vfloat64m2_t ncosrx = __riscv_vfmul_vf_f64m2(cosrx, -1.0, vl);
    
    vbool32_t mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f64m2(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f64m2(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f64m2(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f64m2(res, sinrx, mask, vl);

#ifndef __FAST_MATH__
    vuint64m2_t ix = __riscv_vand_vx_u64m2(
                 __riscv_vreinterpret_v_f64m2_u64m2(x), 0x7fffffffffffffff, vl);
    vbool32_t mask_NaN = __riscv_vmsgeu_vx_u64m2_b32 (ix, 0x7ff0000000000000, vl);
    res = __riscv_vmerge_vvm_f64m2(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b32(mask_NaN,
                  __riscv_vmsltu_vx_u64m2_b32(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b32(mask_NaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        res = __riscv_vfmerge_vfm_f64m2(res, x1, mask_NaN, vl);
    }
#endif

    return res;
}

vfloat64m4_t __riscv_vcos_f64m4(vfloat64m4_t x, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool16_t mask_pi_4;
    vuint64m4_t quadrants;
    
    __riscv_vreduction_pi4_f64m4(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f64m4(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f64m4(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f64m4(cosr, sinr, mask_pi_4, vl);
    
    vfloat64m4_t res = __riscv_vfmv_v_f_f64m4(0.0, vl);
    vfloat64m4_t nsinrx = __riscv_vfmul_vf_f64m4(sinrx, -1.0, vl);
    vfloat64m4_t ncosrx = __riscv_vfmul_vf_f64m4(cosrx, -1.0, vl);
    
    vbool16_t mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 0, vl);
    res = __riscv_vmerge_vvm_f64m4(res, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 1, vl);
    res = __riscv_vmerge_vvm_f64m4(res, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 2, vl);
    res = __riscv_vmerge_vvm_f64m4(res, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 3, vl);
    res = __riscv_vmerge_vvm_f64m4(res, sinrx, mask, vl);

#ifndef __FAST_MATH__
    vuint64m4_t ix = __riscv_vand_vx_u64m4(
                 __riscv_vreinterpret_v_f64m4_u64m4(x), 0x7fffffffffffffff, vl);
    vbool16_t mask_NaN = __riscv_vmsgeu_vx_u64m4_b16 (ix, 0x7ff0000000000000, vl);
    res = __riscv_vmerge_vvm_f64m4(res, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b16(mask_NaN,
                  __riscv_vmsltu_vx_u64m4_b16(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_NaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        res = __riscv_vfmerge_vfm_f64m4(res, x1, mask_NaN, vl);
    }
#endif

    return res;
}

vfloat64m8_t __riscv_vcos_f64m8(vfloat64m8_t x, size_t avl)
{ 
    vfloat64m8_t res;
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    x1 = __riscv_vcos_f64m4(x1, vl);
    res = __riscv_vset_v_f64m4_f64m8(res, 0, x1);
    if(avl > vl){
        vl = __riscv_vsetvl_e64m4(avl-vl);
        x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
        x1 = __riscv_vcos_f64m4(x1, vl);
        res = __riscv_vset_v_f64m4_f64m8(res, 1, x1);
    }
    return res;
}

void __riscv_vsincos_f64m1(vfloat64m1_t x, vfloat64m1_t& sinx, vfloat64m1_t& cosx, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m1(avl);
    vfloat64m1_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool64_t mask_pi_4;
    vuint64m1_t quadrants;
    
    __riscv_vreduction_pi4_f64m1(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f64m1(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f64m1(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f64m1(cosr, sinr, mask_pi_4, vl);
    
    vfloat64m1_t nsinrx = __riscv_vfmul_vf_f64m1(sinrx, -1.0, vl);
    vfloat64m1_t ncosrx = __riscv_vfmul_vf_f64m1(cosrx, -1.0, vl);
    
    vuint64m1_t signx = __riscv_vand_vx_u64m1(__riscv_vreinterpret_v_f64m1_u64m1(x), 
                                            0x8000000000000000, vl);  
    vbool64_t mask = __riscv_vmseq_vx_u64m1_b64(signx, 0, vl);
    vfloat64m1_t ssinrx = __riscv_vmerge_vvm_f64m1(nsinrx, sinrx, mask, vl);
    vfloat64m1_t snsinrx = __riscv_vmerge_vvm_f64m1(sinrx, nsinrx, mask, vl);
    vfloat64m1_t scosrx = __riscv_vmerge_vvm_f64m1(ncosrx, cosrx, mask, vl);
    vfloat64m1_t sncosrx = __riscv_vmerge_vvm_f64m1(cosrx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 0, vl);
    sinx = __riscv_vmerge_vvm_f64m1(sinx, ssinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f64m1(cosx, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 1, vl);
    sinx = __riscv_vmerge_vvm_f64m1(sinx, scosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f64m1(cosx, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 2, vl);
    sinx = __riscv_vmerge_vvm_f64m1(sinx, snsinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f64m1(cosx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m1_b64(quadrants, 3, vl);
    sinx = __riscv_vmerge_vvm_f64m1(sinx, sncosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f64m1(cosx, sinrx, mask, vl);
    
#ifndef __FAST_MATH__
    vuint64m1_t ix = __riscv_vand_vx_u64m1(
                 __riscv_vreinterpret_v_f64m1_u64m1(x), 0x7fffffffffffffff, vl);
    vbool64_t mask_NaN = __riscv_vmsgeu_vx_u64m1_b64 (ix, 0x7ff0000000000000, vl);
    sinx = __riscv_vmerge_vvm_f64m1(sinx, x, mask_NaN, vl);
    cosx = __riscv_vmerge_vvm_f64m1(cosx, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b64(mask_NaN,
                  __riscv_vmsltu_vx_u64m1_b64(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b64(mask_NaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        sinx = __riscv_vfmerge_vfm_f64m1(sinx, x1, mask_NaN, vl);
        cosx = __riscv_vfmerge_vfm_f64m1(cosx, x1, mask_NaN, vl);
    }
#endif
}

void __riscv_vsincos_f64m2(vfloat64m2_t x, vfloat64m2_t& sinx, vfloat64m2_t& cosx, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m2(avl);
    vfloat64m2_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool32_t mask_pi_4;
    vuint64m2_t quadrants;
    
    __riscv_vreduction_pi4_f64m2(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f64m2(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f64m2(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f64m2(cosr, sinr, mask_pi_4, vl);
    
    vfloat64m2_t nsinrx = __riscv_vfmul_vf_f64m2(sinrx, -1.0, vl);
    vfloat64m2_t ncosrx = __riscv_vfmul_vf_f64m2(cosrx, -1.0, vl);
    
    vuint64m2_t signx = __riscv_vand_vx_u64m2(__riscv_vreinterpret_v_f64m2_u64m2(x), 
                                            0x8000000000000000, vl);  
    vbool32_t mask = __riscv_vmseq_vx_u64m2_b32(signx, 0, vl);
    vfloat64m2_t ssinrx = __riscv_vmerge_vvm_f64m2(nsinrx, sinrx, mask, vl);
    vfloat64m2_t snsinrx = __riscv_vmerge_vvm_f64m2(sinrx, nsinrx, mask, vl);
    vfloat64m2_t scosrx = __riscv_vmerge_vvm_f64m2(ncosrx, cosrx, mask, vl);
    vfloat64m2_t sncosrx = __riscv_vmerge_vvm_f64m2(cosrx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 0, vl);
    sinx = __riscv_vmerge_vvm_f64m2(sinx, ssinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f64m2(cosx, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 1, vl);
    sinx = __riscv_vmerge_vvm_f64m2(sinx, scosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f64m2(cosx, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 2, vl);
    sinx = __riscv_vmerge_vvm_f64m2(sinx, snsinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f64m2(cosx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m2_b32(quadrants, 3, vl);
    sinx = __riscv_vmerge_vvm_f64m2(sinx, sncosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f64m2(cosx, sinrx, mask, vl);
    
#ifndef __FAST_MATH__
    vuint64m2_t ix = __riscv_vand_vx_u64m2(
                 __riscv_vreinterpret_v_f64m2_u64m2(x), 0x7fffffffffffffff, vl);
    vbool32_t mask_NaN = __riscv_vmsgeu_vx_u64m2_b32 (ix, 0x7ff0000000000000, vl);
    sinx = __riscv_vmerge_vvm_f64m2(sinx, x, mask_NaN, vl);
    cosx = __riscv_vmerge_vvm_f64m2(cosx, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b32(mask_NaN,
                  __riscv_vmsltu_vx_u64m2_b32(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b32(mask_NaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        sinx = __riscv_vfmerge_vfm_f64m2(sinx, x1, mask_NaN, vl);
        cosx = __riscv_vfmerge_vfm_f64m2(cosx, x1, mask_NaN, vl);
    }
#endif
}

void __riscv_vsincos_f64m4(vfloat64m4_t x, vfloat64m4_t& sinx, vfloat64m4_t& cosx, size_t avl) {
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t redx_hi, redx_lo, sinr, cosr, sinrx, cosrx;
    vbool16_t mask_pi_4;
    vuint64m4_t quadrants;
    
    __riscv_vreduction_pi4_f64m4(x, redx_hi, redx_lo, quadrants, mask_pi_4, vl);
    __riscv_vsincos_kernel_f64m4(redx_hi, redx_lo, sinr, cosr, vl);
    
    sinrx = __riscv_vmerge_vvm_f64m4(sinr, cosr, mask_pi_4, vl);
    cosrx = __riscv_vmerge_vvm_f64m4(cosr, sinr, mask_pi_4, vl);
    
    vfloat64m4_t nsinrx = __riscv_vfmul_vf_f64m4(sinrx, -1.0, vl);
    vfloat64m4_t ncosrx = __riscv_vfmul_vf_f64m4(cosrx, -1.0, vl);
    
    vuint64m4_t signx = __riscv_vand_vx_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(x), 
                                            0x8000000000000000, vl);  
    vbool16_t mask = __riscv_vmseq_vx_u64m4_b16(signx, 0, vl);
    vfloat64m4_t ssinrx = __riscv_vmerge_vvm_f64m4(nsinrx, sinrx, mask, vl);
    vfloat64m4_t snsinrx = __riscv_vmerge_vvm_f64m4(sinrx, nsinrx, mask, vl);
    vfloat64m4_t scosrx = __riscv_vmerge_vvm_f64m4(ncosrx, cosrx, mask, vl);
    vfloat64m4_t sncosrx = __riscv_vmerge_vvm_f64m4(cosrx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 0, vl);
    sinx = __riscv_vmerge_vvm_f64m4(sinx, ssinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f64m4(cosx, cosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 1, vl);
    sinx = __riscv_vmerge_vvm_f64m4(sinx, scosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f64m4(cosx, nsinrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 2, vl);
    sinx = __riscv_vmerge_vvm_f64m4(sinx, snsinrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f64m4(cosx, ncosrx, mask, vl);
    
    mask = __riscv_vmseq_vx_u64m4_b16(quadrants, 3, vl);
    sinx = __riscv_vmerge_vvm_f64m4(sinx, sncosrx, mask, vl);
    cosx = __riscv_vmerge_vvm_f64m4(cosx, sinrx, mask, vl);
    
#ifndef __FAST_MATH__
    vuint64m4_t ix = __riscv_vand_vx_u64m4(
                 __riscv_vreinterpret_v_f64m4_u64m4(x), 0x7fffffffffffffff, vl);
    vbool16_t mask_NaN = __riscv_vmsgeu_vx_u64m4_b16 (ix, 0x7ff0000000000000, vl);
    sinx = __riscv_vmerge_vvm_f64m4(sinx, x, mask_NaN, vl);
    cosx = __riscv_vmerge_vvm_f64m4(cosx, x, mask_NaN, vl);
    mask_NaN = __riscv_vmand_mm_b16(mask_NaN,
                  __riscv_vmsltu_vx_u64m4_b16(ix, 0x7ff8000000000000, vl), vl);
    unsigned int issNaN = __riscv_vcpop_m_b16(mask_NaN, vl);
    if (issNaN) {
        volatile double x1 = 0.0/0.0;
        sinx = __riscv_vfmerge_vfm_f64m4(sinx, x1, mask_NaN, vl);
        cosx = __riscv_vfmerge_vfm_f64m4(cosx, x1, mask_NaN, vl);
    }
#endif
}

void __riscv_vsincos_f64m8(vfloat64m8_t x, vfloat64m8_t& sinx, vfloat64m8_t& cosx, size_t avl)
{ 
    vfloat64m8_t res;
    size_t vl = __riscv_vsetvl_e64m4(avl);
    vfloat64m4_t x1 = __riscv_vget_v_f64m8_f64m4(x, 0);
    vfloat64m4_t sinx1 = __riscv_vget_v_f64m8_f64m4(sinx, 0);
    vfloat64m4_t cosx1 = __riscv_vget_v_f64m8_f64m4(cosx, 0);
    __riscv_vsincos_f64m4(x1, sinx1, cosx1, vl);
    sinx = __riscv_vset_v_f64m4_f64m8(sinx, 0, sinx1);
    cosx = __riscv_vset_v_f64m4_f64m8(cosx, 0, cosx1);
    if(avl > vl){
        vl = __riscv_vsetvl_e64m4(avl-vl);
        x1 = __riscv_vget_v_f64m8_f64m4(x, 1);
        sinx1 = __riscv_vget_v_f64m8_f64m4(sinx, 1);
        cosx1 = __riscv_vget_v_f64m8_f64m4(cosx, 1);
        __riscv_vsincos_f64m4(x1, sinx1, cosx1, vl);
        sinx = __riscv_vset_v_f64m4_f64m8(sinx, 1, sinx1);
        cosx = __riscv_vset_v_f64m4_f64m8(cosx, 1, cosx1);
    }
}

#endif /* __riscv_v_intrinsic */
