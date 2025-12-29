cmake_minimum_required(VERSION 3.14)
project(memory_pool VERSION 0.1.0 LANGUAGES CXX CUDA)

# Enable compile commands export
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test programs" OFF)
option(BUILD_PERFORMANCE_TESTS "Build performance test programs" OFF)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Library source files
set(SOURCES
    src/common.cpp
    src/memory_pool_manager.cpp
    src/cpu/cpu_memory_pool.cpp
    src/cpu/fixed_size_allocator.cpp
    src/cpu/variable_size_allocator.cpp
    src/gpu/gpu_memory_pool.cpp
    src/gpu/cuda_allocator.cpp
    src/gpu/cuda_utils.cpp
    src/stats/memory_stats.cpp
    src/utils/thread_safety.cpp
    src/utils/debug_tools.cpp
    src/utils/error_handling.cpp
)

# Create the library
add_library(memory_pool ${SOURCES})
target_link_libraries(memory_pool CUDA::cudart)

# Examples
if(BUILD_EXAMPLES)
    add_executable(basic_example examples/basic_usage.cpp)
    target_link_libraries(basic_example memory_pool)

    add_executable(gpu_example examples/gpu_examples.cpp)
    target_link_libraries(gpu_example memory_pool)
endif()

# Enable testing
if(BUILD_TESTS)
    enable_testing()

    # Unit tests
    add_executable(cpu_tests tests/unit/cpu_tests.cpp)
    target_link_libraries(cpu_tests memory_pool)
    add_test(NAME cpu_tests COMMAND cpu_tests)

    add_executable(gpu_tests tests/unit/gpu_tests.cpp)
    target_link_libraries(gpu_tests memory_pool)
    add_test(NAME gpu_tests COMMAND gpu_tests)

    add_executable(manager_tests tests/unit/manager_tests.cpp)
    target_link_libraries(manager_tests memory_pool)
    add_test(NAME manager_tests COMMAND manager_tests)

    # Integration tests
    add_executable(cpu_gpu_integration_tests tests/integration/cpu_gpu_integration_tests.cpp)
    target_link_libraries(cpu_gpu_integration_tests memory_pool)
    add_test(NAME cpu_gpu_integration_tests COMMAND cpu_gpu_integration_tests)
endif()

# Performance tests
if(BUILD_PERFORMANCE_TESTS)
    add_executable(cpu_benchmarks tests/performance/cpu_benchmarks.cpp)
    target_link_libraries(cpu_benchmarks memory_pool)

    add_executable(gpu_benchmarks tests/performance/gpu_benchmarks.cpp)
    target_link_libraries(gpu_benchmarks memory_pool)

    add_executable(comparison_benchmarks tests/performance/comparison_benchmarks.cpp)
    target_link_libraries(comparison_benchmarks memory_pool)
endif()

# Installation
install(TARGETS memory_pool
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include)