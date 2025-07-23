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
set(VERSION 2.1.0)

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

set(prefix ${CMAKE_INSTALL_PREFIX})
set(exec_prefix ${CMAKE_INSTALL_PREFIX})
set(includedir ${CMAKE_INSTALL_INCLUDEDIR})
set(libdir ${CMAKE_INSTALL_LIBDIR})

configure_file(
  "${PROJECT_SOURCE_DIR}/src/qcdloop/config.h.in"
  "${PROJECT_SOURCE_DIR}/src/qcdloop/config.h"
  )

configure_file(
  "${PROJECT_SOURCE_DIR}/src/qcdloop-config.in"
  "${PROJECT_BINARY_DIR}/src/qcdloop-config"
  )

configure_file(
  "${PROJECT_SOURCE_DIR}/src/qcdloop.pc.in"
  "${PROJECT_SOURCE_DIR}/src/qcdloop.pc"
  )

# libqcdloop configuration
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

