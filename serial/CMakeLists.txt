# -------------------------------------------------------------------
# CMake configuration for coanda_fem
# -------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13)
project(CoandaFEM LANGUAGES CXX)

# -------------------------------------------------------------------
# Find deal.II (requires deal.II >= 9.3)
# -------------------------------------------------------------------
find_package(deal.II 9.3 REQUIRED HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

# -------------------------------------------------------------------
# Set up build type and compiler flags
# -------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release or Debug)" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Using deal.II from: ${deal.II_DIR}")

# -------------------------------------------------------------------
# Declare the executable and link deal.II
# -------------------------------------------------------------------
deal_ii_initialize_cached_variables()

add_executable(coanda_ssfem coanda_ssfem_serial.cpp)
deal_ii_setup_target(coanda_ssfem)

# -------------------------------------------------------------------
# Installation rule (optional)
# -------------------------------------------------------------------
install(TARGETS coanda_ssfem RUNTIME DESTINATION bin)
