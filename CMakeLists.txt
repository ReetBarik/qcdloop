cmake_minimum_required(VERSION 3.12...3.31)

# Disable the use of RPATHS - we probably are not
# that interested in relocatable binaries and it
# causes a lot of trouble on macOS
SET(CMAKE_SKIP_RPATH TRUE)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

project(qcdloop-kokkos)
include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(VERSION 1.0.0)


# Set default CXXFLAGS but allow for environment override
# c.f. https://cmake.org/cmake/help/v3.31/envvar/CXXFLAGS.html
if (NOT CMAKE_CXX_FLAGS)
  # c.f. https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
  if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -fvisibility-inlines-hidden -fmessage-length=0 -ftree-vectorize -fstack-protector-strong -O2 -pipe -fext-numeric-literals")
  else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
  endif()
else()
  if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
    # Ensure -fext-numeric-literals is in CMAKE_CXX_FLAGS
    string(FIND "${CMAKE_CXX_FLAGS}" "-fext-numeric-literals" _found_ext_numeric_literals)
    if (_found_ext_numeric_literals EQUAL -1)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals" CACHE STRING "Update environment CXXFLAGS" FORCE)
    endif()
  endif()
endif()

# TODO: QCDLoop is currently only well tested on GNU, so other compiler
# and linker defaults should be added as they become known
if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address ${CMAKE_CXX_FLAGS}" CACHE STRING "debug CXXFLAGS" FORCE)
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address" CACHE STRING "debug linker flags" FORCE)
endif()

include_directories(src/qcdloop src)

find_package(Kokkos REQUIRED)


add_executable(tadpoleGPU_test examples/tadpoleGPU_test.cc)
target_link_libraries(tadpoleGPU_test PUBLIC Kokkos::kokkos)
target_compile_options(tadpoleGPU_test PRIVATE -w)

add_executable(bubbleGPU_test examples/bubbleGPU_test.cc)
target_link_libraries(bubbleGPU_test PUBLIC Kokkos::kokkos)
target_compile_options(bubbleGPU_test PRIVATE -w)

add_executable(triangleGPU_test examples/triangleGPU_test.cc)
target_link_libraries(triangleGPU_test PUBLIC Kokkos::kokkos)
target_compile_options(triangleGPU_test PRIVATE -w)

add_executable(boxGPU_test examples/boxGPU_test.cc)
target_link_libraries(boxGPU_test PUBLIC Kokkos::kokkos)
target_compile_options(boxGPU_test PRIVATE -w)


IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  set(QUADMATH_NAMES ${QUADMATH_NAMES} libquadmath.so quadmath)
  find_library(QUADMATH_LIBRARY
    NAMES ${QUADMATH_NAMES}
    PATHS /usr/lib64/atlas /usr/lib/atlas
          /usr/lib64 /usr/lib /usr/local/lib64
          /usr/local/lib /usr/x86_64-linux-gnu/*
          /usr/lib/gcc/x86_64-linux-gnu/*
          /usr/lib/gcc/x86_64-redhat-linux/*
          /usr/lib/gcc/x86_64-pc-linux-gnu/*
          /soft/compilers/gcc/x86_64-suse-linux/14.1.0/lib64
          "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
  )

  if(QUADMATH_LIBRARY)
    set(QUADMATH_LIBRARIES ${QUADMATH_LIBRARY})
    set(QUADMATH_FOUND "YES")
    message(STATUS "Found QUADMATH: ${QUADMATH_LIBRARIES}")
  else()
    set(QUADMATH_FOUND "NO")
    message(WARNING "Could not find QuadMath library - binary may not compile with quadmath.h")
  endif()
ENDIF()

# Individual test executables split from boxGPU_test_quad
set(BOX_QUAD_TESTS
  boxGPU_test_quad_BIN0
  boxGPU_test_quad_BIN1
  boxGPU_test_quad_BIN2
  boxGPU_test_quad_BIN3
  boxGPU_test_quad_BIN4
  boxGPU_test_quad_B1
  boxGPU_test_quad_B2
  boxGPU_test_quad_B3
  boxGPU_test_quad_B4
  boxGPU_test_quad_B5
  boxGPU_test_quad_B6
  boxGPU_test_quad_B7
  boxGPU_test_quad_B8
  boxGPU_test_quad_B9
  boxGPU_test_quad_B10
  boxGPU_test_quad_B11
  boxGPU_test_quad_B12
  boxGPU_test_quad_B13
  boxGPU_test_quad_B14
  boxGPU_test_quad_B15
  boxGPU_test_quad_B16
)

foreach(test_target ${BOX_QUAD_TESTS})
  add_executable(${test_target} examples/box/${test_target}.cc)
  target_link_libraries(${test_target} PUBLIC Kokkos::kokkos)
  if(QUADMATH_FOUND)
    target_link_libraries(${test_target} PUBLIC ${QUADMATH_LIBRARY})
  endif()
  target_compile_options(${test_target} PRIVATE -w)
endforeach()