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

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(rvvmf)

set (CMAKE_CONFIGURATION_TYPES "Debug;Release") 
set (CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(WITH_RVV_LIB        "Build the RVV library" ON)
option(WITH_ERRNO          "Use   the errno macro" OFF)

include(cmake/TargetCreationFunctions.cmake)
include(cmake/TestEnvironment.cmake)
include(cmake/OutputFlagsOption.cmake)

if(WITH_ERRNO)
  add_compile_options(-DRVVMF_WITH_ERRNO)
endif()

if(BUILD_RVV)
  if(HAVE_RVV_1P0)
    add_compile_options(-DRVVMF_RVV_1P0)
  endif()
endif()

add_subdirectory(src)

install(DIRECTORY include/ 
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
        PATTERN "*.hpp")