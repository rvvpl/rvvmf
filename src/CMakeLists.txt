#========================================================
# Copyright (c) RVVPL and Lobachevsky State University of 
# Nizhny Novgorod and its affiliates. All rights reserved.
# 
# Copyright 2025-2026 The RVVMF Authors (Evgeny Kozinov)
#
# Distributed under the BSD 4-Clause License
# (See file LICENSE in the root directory of this 
# source tree)
#========================================================
#
#********************************************************
#                                                       *
#   File: CMakeLists.txt                                *
#   Contains: CMake build system                        *
#                                                       *
#********************************************************

if(BUILD_RVV)
  add_subdirectory(rvv)
endif()