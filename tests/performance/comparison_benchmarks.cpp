#include "memory_pool/memory_pool.hpp"
#include "memory_pool/cpu/cpu_memory_pool.hpp"
#include "memory_pool/gpu/gpu_memory_pool.hpp"
#include "memory_pool/gpu/cuda_utils.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <map>
#include <string>
#include <iomanip>

using namespace memory_pool;

// Benchmark configuration
const int NUM_ITERATIONS = 1000;
const int WARMUP_ITERATIONS = 100;
const size_t TEST_SIZES[] = {16, 64, 256, 1024, 4096};
const int NUM_TEST_SIZES = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);

// Helper function to check CUDA availability
bool isCudaAvailable() {
    try {
        int deviceCount = getDeviceCount();
        return deviceCount > 0 && isDeviceAvailable(0);
    } catch (const std::exception&) {
        return false;
    }
}

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

    // Calculate average
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return sum / times.size();
}

// Structure to hold benchmark results
struct BenchmarkResult {
    std::string name;
    std::map<size_t, double> times;  // size -> average time in microseconds

    void print() const {
        std::cout << name << ":" << std::endl;
        for (size_t size : TEST_SIZES) {
            auto it = times.find(size);
            if (it != times.end()) {
                std::cout << "  " << size << " bytes: " << it->second << " μs" << std::endl;
            }
        }
    }
};

// Benchmark CPU allocators
std::vector<BenchmarkResult> benchmarkCPUAllocators() {
    std::cout << "Benchmarking CPU Allocators..." << std::endl;

    std::vector<BenchmarkResult> results;

    // Memory Pool
    {
        BenchmarkResult result;
        result.name = "CPU Memory Pool";

        PoolConfig config = PoolConfig::DefaultCPU();
        CPUMemoryPool pool("cpu_compare_pool", config);

        for (size_t size : TEST_SIZES) {
            double time = measureTime([&]() {
                void* ptr = pool.allocate(size);
                pool.deallocate(ptr);
            });
            result.times[size] = time;
        }

        results.push_back(result);
    }

    // Standard malloc/free
    {
        BenchmarkResult result;
        result.name = "malloc/free";

        for (size_t size : TEST_SIZES) {
            double time = measureTime([size]() {
                void* ptr = malloc(size);
                free(ptr);
            });
            result.times[size] = time;
        }

        results.push_back(result);
    }

    // new/delete
    {
        BenchmarkResult result;
        result.name = "new/delete";

        for (size_t size : TEST_SIZES) {
            double time = measureTime([size]() {
                char* ptr = new char[size];
                delete[] ptr;
            });
            result.times[size] = time;
        }

        results.push_back(result);
    }

    return results;
}

// Benchmark GPU allocators
std::vector<BenchmarkResult> benchmarkGPUAllocators() {
    std::cout << "Benchmarking GPU Allocators..." << std::endl;

    std::vector<BenchmarkResult> results;

    if (!isCudaAvailable()) {
        std::cout << "CUDA not available - skipping GPU benchmarks" << std::endl;
        return results;
    }

    // GPU Memory Pool
    {
        BenchmarkResult result;
        result.name = "GPU Memory Pool";

        PoolConfig config;
        config.allocatorType = AllocatorType::FixedSize;
        config.blockSize = 4096;  // Large enough for all test sizes
        config.deviceId = 0;
        GPUMemoryPool pool("gpu_compare_pool", config);

        for (size_t size : TEST_SIZES) {
            double time = measureTime([&]() {
                void* ptr = pool.allocate(size);
                pool.deallocate(ptr);
            });
            result.times[size] = time;
        }

        results.push_back(result);
    }

    // CUDA malloc/free
    {
        BenchmarkResult result;
        result.name = "cudaMalloc/cudaFree";

        for (size_t size : TEST_SIZES) {
            double time = measureTime([size]() {
                void* ptr;
                cudaMalloc(&ptr, size);
                cudaFree(ptr);
            });
            result.times[size] = time;
        }

        results.push_back(result);
    }

    return results;
}

// Benchmark data transfer
std::vector<BenchmarkResult> benchmarkDataTransfer() {
    std::cout << "Benchmarking Data Transfer..." << std::endl;

    std::vector<BenchmarkResult> results;

    if (!isCudaAvailable()) {
        std::cout << "CUDA not available - skipping data transfer benchmarks" << std::endl;
        return results;
    }

    // Set up pools
    PoolConfig cpuConfig = PoolConfig::DefaultCPU();
    CPUMemoryPool cpuPool("cpu_transfer_pool", cpuConfig);

    PoolConfig gpuConfig;
    gpuConfig.allocatorType = AllocatorType::FixedSize;
    gpuConfig.blockSize = 4096;
    gpuConfig.deviceId = 0;
    GPUMemoryPool gpuPool("gpu_transfer_pool", gpuConfig);

    // Host to device transfer
    {
        BenchmarkResult result;
        result.name = "Host to Device Transfer";

        for (size_t size : TEST_SIZES) {
            void* hostData = cpuPool.allocate(size);
            void* deviceData = gpuPool.allocate(size);

            memset(hostData, 0xAA, size);

            double time = measureTime([&]() {
                gpuPool.copyHostToDevice(deviceData, hostData, size);
            });

            result.times[size] = time;

            cpuPool.deallocate(hostData);
            gpuPool.deallocate(deviceData);
        }

        results.push_back(result);
    }

    // Device to host transfer
    {
        BenchmarkResult result;
        result.name = "Device to Host Transfer";

        for (size_t size : TEST_SIZES) {
            void* hostData = cpuPool.allocate(size);
            void* deviceData = gpuPool.allocate(size);

            double time = measureTime([&]() {
                gpuPool.copyDeviceToHost(hostData, deviceData, size);
            });

            result.times[size] = time;

            cpuPool.deallocate(hostData);
            gpuPool.deallocate(deviceData);
        }

        results.push_back(result);
    }

    return results;
}

// Print comparison table
void printComparisonTable(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n=== Performance Comparison Table ===" << std::endl;
    std::cout << std::setw(20) << "Allocator" << std::setw(10) << "16B" << std::setw(10) << "64B"
              << std::setw(10) << "256B" << std::setw(10) << "1KB" << std::setw(10) << "4KB" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (const auto& result : results) {
        std::cout << std::setw(20) << result.name;
        for (size_t size : TEST_SIZES) {
            auto it = result.times.find(size);
            if (it != result.times.end()) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(2) << it->second;
            } else {
                std::cout << std::setw(10) << "N/A";
            }
        }
        std::cout << std::endl;
    }
}

// Calculate performance ratios
void printPerformanceRatios(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n=== Performance Ratios (vs Standard Allocators) ===" << std::endl;

    if (results.size() < 2) return;

    const BenchmarkResult& baseline = results[1];  // Assume second result is baseline (malloc/free or cudaMalloc)

    for (size_t i = 0; i < results.size(); ++i) {
        if (i == 1) continue;  // Skip baseline

        const BenchmarkResult& result = results[i];
        std::cout << result.name << " vs " << baseline.name << ":" << std::endl;

        for (size_t size : TEST_SIZES) {
            auto resultIt = result.times.find(size);
            auto baselineIt = baseline.times.find(size);

            if (resultIt != result.times.end() && baselineIt != baseline.times.end()) {
                double ratio = resultIt->second / baselineIt->second;
                std::cout << "  " << size << " bytes: " << std::fixed << std::setprecision(3) << ratio << "x";
                if (ratio < 1.0) {
                    std::cout << " (faster)";
                } else {
                    std::cout << " (slower)";
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "Memory Pool Comparison Benchmarks" << std::endl;
    std::cout << "=================================" << std::endl;

    try {
        // CPU benchmarks
        auto cpuResults = benchmarkCPUAllocators();
        std::cout << std::endl;

        // GPU benchmarks
        auto gpuResults = benchmarkGPUAllocators();
        std::cout << std::endl;

        // Data transfer benchmarks
        auto transferResults = benchmarkDataTransfer();
        std::cout << std::endl;

        // Print detailed results
        std::cout << "\nDetailed Results:" << std::endl;
        for (const auto& result : cpuResults) {
            result.print();
        }
        for (const auto& result : gpuResults) {
            result.print();
        }
        for (const auto& result : transferResults) {
            result.print();
        }

        // Print comparison tables
        if (!cpuResults.empty()) {
            printComparisonTable(cpuResults);
            printPerformanceRatios(cpuResults);
        }

        if (!gpuResults.empty()) {
            printComparisonTable(gpuResults);
            printPerformanceRatios(gpuResults);
        }

        if (!transferResults.empty()) {
            printComparisonTable(transferResults);
        }

        std::cout << "\nAll comparison benchmarks completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Comparison benchmark failed with exception: " << e.what() << std::endl;
        return 1;
    }
}