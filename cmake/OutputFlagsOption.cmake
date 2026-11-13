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
#   File:  OutputFlagsOption.cmake                      *
#   Contains: CMake build system                        *
#                                                       *
#********************************************************

function(status MESSEGE FLAG)
    if (FLAG)
      message("${MESSEGE} YES")
    else()
      message("${MESSEGE} NO")
    endif()
endfunction()


message ("======================= RVVMV build option ========================= ")

message ("-----------------------  compile options    ------------------------ ")

if(DEFINED CMAKE_CXX_STANDARD AND CMAKE_CXX_STANDARD)
  message("C++ standard        :" ${CMAKE_CXX_STANDARD})
endif()
message("C++ Compiler        :" ${CMAKE_CXX_COMPILER})
message("C Compiler          :" ${CMAKE_C_COMPILER})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "----- Using GCC compiler")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "----- Using Clang compiler")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("C++ flags (Debug)   :" ${CMAKE_CXX_FLAGS} " " ${CMAKE_CXX_FLAGS_DEBUG})
  message("C   flags (Debug)   :" ${CMAKE_C_FLAGS} " " ${CMAKE_C_FLAGS_DEBUG})
else()
  message("C++ flags (Release) :" ${CMAKE_CXX_FLAGS} " " ${CMAKE_CXX_FLAGS_RELEASE})
  message("C   flags (Release) :" ${CMAKE_C_FLAGS} " " ${CMAKE_C_FLAGS_RELEASE})
endif()

message ("------------------------ having components -------------------------- ")

status("HAVE_RVV_1P0      : " ${HAVE_RVV_1P0})
status("Use errno macro   : " ${WITH_ERRNO})

message ("---------------------- compiling components ------------------------- ")

status("Build the RVV    library                : " ${BUILD_RVV})

message ("======================= RVVMV build option ========================= ")
