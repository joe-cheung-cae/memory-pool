cmake_minimum_required(VERSION 3.14)

# Set the CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Choose the CUDA architectures")
set_property(CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY STRINGS native all all-major)

project(memory_pool VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set CMAKE build type
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug)
message(STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

# Enable compile commands export
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Find Valgrind (required)
find_package(PkgConfig REQUIRED)
pkg_check_modules(VALGRIND REQUIRED valgrind)
add_definitions(-DHAVE_VALGRIND)
include_directories(${VALGRIND_INCLUDE_DIRS})
link_directories(${VALGRIND_LIBRARY_DIRS})

# Find NUMA (optional)
pkg_check_modules(NUMA numa)
if(NUMA_FOUND)
    add_definitions(-DHAVE_NUMA)
    include_directories(${NUMA_INCLUDE_DIRS})
    link_directories(${NUMA_LIBRARY_DIRS})
endif()

# Find PMEM (optional)
pkg_check_modules(PMEM libpmem)
if(PMEM_FOUND)
    add_definitions(-DHAVE_PMEM)
    include_directories(${PMEM_INCLUDE_DIRS})
    link_directories(${PMEM_LIBRARY_DIRS})
endif()

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
    src/utils/numa_utils.cpp
)

if(PMEM_FOUND)
    list(APPEND SOURCES
        src/pmem/pmem_memory_pool.cpp
        src/pmem/pmem_fixed_size_allocator.cpp
        src/pmem/pmem_variable_size_allocator.cpp
    )
endif()

# Create the library
add_library(memory_pool ${SOURCES})
target_link_libraries(memory_pool CUDA::cudart)
if(NUMA_FOUND)
    target_link_libraries(memory_pool ${NUMA_LIBRARIES})
endif()
if(PMEM_FOUND)
    target_link_libraries(memory_pool ${PMEM_LIBRARIES})
endif()

# Examples
if(BUILD_EXAMPLES)
    add_executable(basic_example examples/basic_usage.cpp)
    target_link_libraries(basic_example memory_pool)

    add_executable(gpu_example examples/gpu_examples.cpp)
    target_link_libraries(gpu_example memory_pool)

    if(PMEM_FOUND)
        add_executable(pmem_example examples/pmem_examples.cpp)
        target_link_libraries(pmem_example memory_pool)
    endif()
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

    add_executable(numa_tests tests/unit/numa_tests.cpp)
    target_link_libraries(numa_tests memory_pool)
    add_test(NAME numa_tests COMMAND numa_tests)

    add_executable(multi_gpu_tests tests/unit/multi_gpu_tests.cpp)
    target_link_libraries(multi_gpu_tests memory_pool)
    add_test(NAME multi_gpu_tests COMMAND multi_gpu_tests)

    add_executable(fuzz_tests tests/unit/fuzz_tests.cpp)
    target_link_libraries(fuzz_tests memory_pool)
    add_test(NAME fuzz_tests COMMAND fuzz_tests 2)  # Run for 2 seconds

    if(PMEM_FOUND)
        add_executable(pmem_tests tests/unit/pmem_tests.cpp)
        target_link_libraries(pmem_tests memory_pool)
        add_test(NAME pmem_tests COMMAND pmem_tests)
    endif()

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

# Doxygen documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    # Set Doxygen input and output directories
    set(DOXYGEN_INPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs)

    # Create Doxygen configuration file
    doxygen_add_docs(doxygen
        ${DOXYGEN_INPUT_DIR}
        README.md
        COMMENT "Generate HTML documentation"
    )

    # Configure Doxygen
    set(DOXYGEN_PROJECT_NAME "Memory Pool Management System")
    set(DOXYGEN_PROJECT_VERSION ${PROJECT_VERSION})
    set(DOXYGEN_PROJECT_BRIEF "High-performance memory pool management for CPU and GPU")
    set(DOXYGEN_OUTPUT_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
    set(DOXYGEN_HTML_OUTPUT html)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_EXTRACT_PRIVATE NO)
    set(DOXYGEN_EXTRACT_STATIC NO)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    set(DOXYGEN_WARN_IF_DOC_ERROR YES)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)

    # Custom target to generate docs
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Installation
install(TARGETS memory_pool
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include)