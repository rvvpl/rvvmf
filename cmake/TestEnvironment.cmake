#========================================================
# Copyright (c) The Lobachevsky State University of 
# Nizhny Novgorod and its affiliates. All rights reserved.
# 
# Copyright 2025 The RVVPL Authors (Evgeny Kozinov)
#
# Distributed under the BSD 4-Clause License
# (See file LICENSE in the root directory of this 
# source tree)
#========================================================
#
#********************************************************
#                                                       *
#   File:  TestEnvironment.cmake                        *
#   Contains: CMake build system                        *
#                                                       *
#********************************************************

macro(assign_bool var)
     if(${ARGN})
         set(${var} ON)
     else()
         set(${var} OFF)
     endif()
endmacro()

message ("    check HAVE_RVV_1P0")
try_compile(HAVE_RVV_1P0
            ${CMAKE_BINARY_DIR}
            ${CMAKE_SOURCE_DIR}/cmake/checks/compile_rvv_1p0.cpp
            )

assign_bool(BUILD_RVV ((HAVE_RVV_0P7 OR HAVE_RVV_1P0) AND WITH_RVV_LIB))
