# Install script for directory: /tmp/tmp.ZO2yVo1pOb/src/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/aes/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/bench_operations/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/euclidean_distance/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/float/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/innerproduct/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/lowmc/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/millionaire_prob/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/min-euclidean-dist/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/psi_2D_CH/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/psi_phasing/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/psi_scs/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/sha1/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/threshold_euclidean_dist_2d_simd/cmake_install.cmake")
  include("/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/uc_circuit/cmake_install.cmake")

endif()

