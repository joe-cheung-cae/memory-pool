#include "memory_pool/memory_pool.hpp"
#include "memory_pool/cpu/cpu_memory_pool.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstring>

using namespace memory_pool;

// Benchmark configuration
const int NUM_ITERATIONS = 1000;
const int WARMUP_ITERATIONS = 100;
const size_t MIN_SIZE = 16;
const size_t MAX_SIZE = 4096;

// Helper function to measure time
template<typename Func>
double measureTime(Func&& func, int iterations = NUM_ITERATIONS) {
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations + WARMUP_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::micro> duration = end - start;
        if (i >= WARMUP_ITERATIONS) {
            times.push_back(duration.count());
        }
    }

    // Calculate average and standard deviation
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / times.size() - mean * mean);

    std::cout << "  Average time: " << mean << " μs, StdDev: " << stdev << " μs" << std::endl;
    return mean;
}

// Benchmark CPU memory pool allocation/deallocation
void benchmarkCPUPoolAllocation() {
    std::cout << "Benchmarking CPU Memory Pool Allocation/Deallocation..." << std::endl;

    PoolConfig config = PoolConfig::DefaultCPU();
    CPUMemoryPool pool("benchmark_pool", config);

    // Fixed size allocation
    std::cout << "Fixed size (256 bytes) allocation/deallocation:" << std::endl;
    measureTime([&]() {
        void* ptr = pool.allocate(256);
        pool.deallocate(ptr);
    });

    // Variable size allocation
    std::cout << "Variable size (16-4096 bytes) allocation/deallocation:" << std::endl;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> sizeDist(MIN_SIZE, MAX_SIZE);

    measureTime([&]() {
        size_t size = sizeDist(rng);
        void* ptr = pool.allocate(size);
        pool.deallocate(ptr);
    });

    std::cout << "CPU Memory Pool benchmark completed." << std::endl;
}

// Benchmark standard malloc/free
void benchmarkStandardAllocator() {
    std::cout << "Benchmarking Standard malloc/free..." << std::endl;

    // Fixed size allocation
    std::cout << "Fixed size (256 bytes) malloc/free:" << std::endl;
    measureTime([]() {
        void* ptr = malloc(256);
        free(ptr);
    });

    // Variable size allocation
    std::cout << "Variable size (16-4096 bytes) malloc/free:" << std::endl;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> sizeDist(MIN_SIZE, MAX_SIZE);

    measureTime([&]() {
        size_t size = sizeDist(rng);
        void* ptr = malloc(size);
        free(ptr);
    });

    std::cout << "Standard allocator benchmark completed." << std::endl;
}

// Benchmark new/delete
void benchmarkNewDelete() {
    std::cout << "Benchmarking new/delete..." << std::endl;

    // Fixed size allocation
    std::cout << "Fixed size (256 bytes) new/delete:" << std::endl;
    measureTime([]() {
        char* ptr = new char[256];
        delete[] ptr;
    });

    // Variable size allocation
    std::cout << "Variable size (16-4096 bytes) new/delete:" << std::endl;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> sizeDist(MIN_SIZE, MAX_SIZE);

    measureTime([&]() {
        size_t size = sizeDist(rng);
        char* ptr = new char[size];
        delete[] ptr;
    });

    std::cout << "new/delete benchmark completed." << std::endl;
}

// Benchmark fragmentation
void benchmarkFragmentation() {
    std::cout << "Benchmarking Memory Fragmentation..." << std::endl;

    PoolConfig config = PoolConfig::DefaultCPU();
    CPUMemoryPool pool("fragmentation_pool", config);

    const int NUM_ALLOCS = 10000;
    std::vector<void*> pointers;
    pointers.reserve(NUM_ALLOCS);

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> sizeDist(MIN_SIZE, MAX_SIZE);

    // Allocate many blocks
    std::cout << "Allocating " << NUM_ALLOCS << " blocks..." << std::endl;
    auto allocTime = measureTime([&]() {
        for (int i = 0; i < NUM_ALLOCS; ++i) {
            size_t size = sizeDist(rng);
            void* ptr = pool.allocate(size);
            pointers.push_back(ptr);
        }
    }, 1);  // Only one iteration for this test

    // Deallocate every other block to create fragmentation
    std::cout << "Creating fragmentation by deallocating every other block..." << std::endl;
    for (size_t i = 0; i < pointers.size(); i += 2) {
        pool.deallocate(pointers[i]);
        pointers[i] = nullptr;
    }

    // Try to allocate large blocks
    std::cout << "Attempting to allocate large blocks in fragmented memory..." << std::endl;
    std::vector<void*> largePointers;

    auto fragTime = measureTime([&]() {
        for (int i = 0; i < 100; ++i) {
            void* ptr = pool.allocate(MAX_SIZE * 2);  // Try to allocate large blocks
            if (ptr) {
                largePointers.push_back(ptr);
            }
        }
    }, 1);

    std::cout << "Successfully allocated " << largePointers.size() << " large blocks in fragmented memory" << std::endl;

    // Clean up
    for (void* ptr : pointers) {
        if (ptr) pool.deallocate(ptr);
    }
    for (void* ptr : largePointers) {
        pool.deallocate(ptr);
    }

    std::cout << "Fragmentation benchmark completed." << std::endl;
}

// Benchmark thread safety (simplified - doesn't use actual threads)
void benchmarkThreadSafety() {
    std::cout << "Benchmarking Thread Safety (simulated)..." << std::endl;

    PoolConfig config = PoolConfig::DefaultCPU();
    config.threadSafe = true;
    CPUMemoryPool pool("thread_safe_pool", config);

    // Simulate concurrent operations
    std::cout << "Simulating concurrent allocations/deallocations..." << std::endl;
    measureTime([&]() {
        // Simulate multiple operations
        for (int i = 0; i < 100; ++i) {
            void* ptr1 = pool.allocate(128);
            void* ptr2 = pool.allocate(256);
            void* ptr3 = pool.allocate(64);

            pool.deallocate(ptr1);
            pool.deallocate(ptr2);
            pool.deallocate(ptr3);
        }
    });

    std::cout << "Thread safety benchmark completed." << std::endl;
}

// Compare performance
void comparePerformance() {
    std::cout << "\n=== Performance Comparison ===" << std::endl;

    std::cout << "CPU Memory Pool vs Standard Allocators:" << std::endl;
    std::cout << "Fixed size (256 bytes):" << std::endl;

    // CPU Pool
    PoolConfig config = PoolConfig::DefaultCPU();
    CPUMemoryPool pool("compare_pool", config);

    double poolTime = measureTime([&]() {
        void* ptr = pool.allocate(256);
        pool.deallocate(ptr);
    });

    // malloc/free
    double mallocTime = measureTime([]() {
        void* ptr = malloc(256);
        free(ptr);
    });

    // new/delete
    double newTime = measureTime([]() {
        char* ptr = new char[256];
        delete[] ptr;
    });

    std::cout << "Performance ratios (lower is better):" << std::endl;
    std::cout << "  Pool vs malloc: " << (poolTime / mallocTime) << std::endl;
    std::cout << "  Pool vs new: " << (poolTime / newTime) << std::endl;
    std::cout << "  malloc vs new: " << (mallocTime / newTime) << std::endl;
}

int main() {
    std::cout << "CPU Memory Pool Performance Benchmarks" << std::endl;
    std::cout << "=====================================" << std::endl;

    try {
        benchmarkCPUPoolAllocation();
        std::cout << std::endl;

        benchmarkStandardAllocator();
        std::cout << std::endl;

        benchmarkNewDelete();
        std::cout << std::endl;

        benchmarkFragmentation();
        std::cout << std::endl;

        benchmarkThreadSafety();
        std::cout << std::endl;

        comparePerformance();

        std::cout << "\nAll CPU benchmarks completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed with exception: " << e.what() << std::endl;
        return 1;
    }
}