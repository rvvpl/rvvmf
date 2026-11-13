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
#   File: CMakeLists.txt                                *
#   Contains: CMake build system                        *
#                                                       *
#********************************************************

if(BUILD_RVV)
  add_subdirectory(rvv)
endif()