cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(test_combustion test_combustion.cpp)
target_link_libraries(test_combustion "${TORCH_LIBRARIES}")
set_property(TARGET test_combustion PROPERTY CXX_STANDARD 11)
