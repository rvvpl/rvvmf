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
#   File:  TargetCreationFunctions.cmake                *
#   Contains: CMake build system                        *
#                                                       *
#********************************************************

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${curdir}/${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

function(create_project_lib TARGET )
    file(GLOB_RECURSE TARGET_SRC "*.c*")
    file(GLOB_RECURSE TARGET_HD "*.h*")
    add_library(${TARGET} STATIC ${TARGET_SRC} ${TARGET_HD})
    target_include_directories(${TARGET} PUBLIC "${CMAKE_SOURCE_DIR}/include")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      if(CMAKE_CXX_COMPILER MATCHES "icpx?$|icpc$")
          target_compile_options(${TARGET} PUBLIC "-fp-model=strict")
      else()
          target_compile_options(${TARGET} PUBLIC "-ffp-model=strict")
      endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(${TARGET} PUBLIC "-fp-model=strict")
    endif()

    get_property ( LIB_LIST GLOBAL PROPERTY LIBS_P)
    list(APPEND LIB_LIST ${TARGET})
    set_property ( GLOBAL PROPERTY LIBS_P ${LIB_LIST})

    add_library("${TARGET}_ffast_math" STATIC ${TARGET_SRC} ${TARGET_HD})
    target_compile_options("${TARGET}_ffast_math" PUBLIC "-ffast-math")
    target_include_directories("${TARGET}_ffast_math" PUBLIC "${CMAKE_SOURCE_DIR}/include")

    get_property ( LIB_LIST_FAST GLOBAL PROPERTY LIBS_P_FAST)
    list(APPEND LIB_LIST_FAST "${TARGET}_ffast_math")
    set_property ( GLOBAL PROPERTY LIBS_P_FAST ${LIB_LIST_FAST})

    install(TARGETS ${TARGET} "${TARGET}_ffast_math"
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
endfunction()
