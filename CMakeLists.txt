cmake_minimum_required(VERSION 3.13.4)
find_package(pybind11 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
project(coanda_gtest)
find_package(deal.II 9.5.0 REQUIRED
  HINTS $ENV{mkDealiiPrefix} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
deal_ii_initialize_cached_variables()

# add_library(mylibrary source/coanda_NL.cpp source/coanda_L.cpp source/coanda.cpp)
# target_include_directories(mylibrary PUBLIC ${CMAKE_SOURCE_DIR}/include/)
# deal_ii_setup_target(mylibrary)

# add_executable(coanda source/main.cpp)
# target_include_directories(coanda PUBLIC ${CMAKE_SOURCE_DIR}/include/)
# target_link_libraries(coanda mylibrary)
# deal_ii_setup_target(coanda)

file(GLOB source_files source/main.cpp)
# add_executable(coandaex source/main.cpp)
add_library(coanda MODULE source/coanda.cpp)
target_compile_options(coanda PUBLIC -fPIC)
target_include_directories(coanda PUBLIC ${CMAKE_SOURCE_DIR}/include/)
# target_link_libraries(coanda pybind11::module pybind11::lto)
target_link_libraries(coanda pybind11::module pybind11::lto Eigen3::Eigen)
deal_ii_setup_target(coanda)

# add_library(coanda-lib SHARED $<TARGET_OBJECTS:coanda>)
# deal_ii_setup_target(coanda-lib)

# find_package(GTest CONFIG REQUIRED)
# file(GLOB test_files tests/*.cpp)
# add_executable(coanda_gtest ${test_files})
# target_link_libraries(coanda_gtest GTest::gtest_main coanda-lib)
# deal_ii_setup_target(coanda_gtest)

# enable_testing()
# include(GoogleTest)
# gtest_discover_tests(coanda_gtest)

