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

using namespace memory_pool;

// Benchmark configuration
const int NUM_ITERATIONS = 1000;
const int WARMUP_ITERATIONS = 100;
const size_t MIN_SIZE = 16;
const size_t MAX_SIZE = 4096;

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

    // Calculate average and standard deviation
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / times.size() - mean * mean);

    std::cout << "  Average time: " << mean << " μs, StdDev: " << stdev << " μs" << std::endl;
    return mean;
}

// Benchmark GPU memory pool allocation/deallocation
void benchmarkGPUPoolAllocation() {
    std::cout << "Benchmarking GPU Memory Pool Allocation/Deallocation..." << std::endl;

    const size_t BLOCK_SIZE = 512;

    PoolConfig config;
    config.allocatorType = AllocatorType::FixedSize;
    config.blockSize = BLOCK_SIZE;
    config.deviceId = 0;
    GPUMemoryPool pool("gpu_benchmark_pool", config);

    // Fixed size allocation
    std::cout << "Fixed size (512 bytes) allocation/deallocation:" << std::endl;
    measureTime([&]() {
        void* ptr = pool.allocate(BLOCK_SIZE);
        pool.deallocate(ptr);
    });

    // Variable size allocation (using smaller sizes that fit in block)
    std::cout << "Variable size (16-512 bytes) allocation/deallocation:" << std::endl;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> sizeDist(MIN_SIZE, BLOCK_SIZE);

    measureTime([&]() {
        size_t size = sizeDist(rng);
        void* ptr = pool.allocate(size);
        pool.deallocate(ptr);
    });

    std::cout << "GPU Memory Pool benchmark completed." << std::endl;
}

// Benchmark standard CUDA allocation
void benchmarkCudaAllocator() {
    std::cout << "Benchmarking Standard CUDA cudaMalloc/cudaFree..." << std::endl;

    // Fixed size allocation
    std::cout << "Fixed size (512 bytes) cudaMalloc/cudaFree:" << std::endl;
    measureTime([]() {
        void* ptr;
        cudaMalloc(&ptr, 512);
        cudaFree(ptr);
    });

    // Variable size allocation
    std::cout << "Variable size (16-4096 bytes) cudaMalloc/cudaFree:" << std::endl;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> sizeDist(MIN_SIZE, MAX_SIZE);

    measureTime([&]() {
        size_t size = sizeDist(rng);
        void* ptr;
        cudaMalloc(&ptr, size);
        cudaFree(ptr);
    });

    std::cout << "CUDA allocator benchmark completed." << std::endl;
}

// Benchmark CPU-GPU data transfer
void benchmarkDataTransfer() {
    std::cout << "Benchmarking CPU-GPU Data Transfer..." << std::endl;

    // Create pools
    PoolConfig cpuConfig = PoolConfig::DefaultCPU();
    cpuConfig.usePinnedMemory = true;  // Use pinned memory for efficient transfers
    CPUMemoryPool cpuPool("cpu_transfer_pool", cpuConfig);

    PoolConfig gpuConfig;
    gpuConfig.allocatorType = AllocatorType::VariableSize;
    gpuConfig.initialSize = 1024 * 1024;  // 1MB
    gpuConfig.deviceId = 0;
    GPUMemoryPool gpuPool("gpu_transfer_pool", gpuConfig);

    // Allocate memory
    void* hostData = cpuPool.allocate(1024);
    void* deviceData = gpuPool.allocate(1024);

    // Initialize host data
    memset(hostData, 0xAA, 1024);

    // Host to device transfer
    std::cout << "Host to device transfer (1KB):" << std::endl;
    measureTime([&]() {
        gpuPool.copyHostToDevice(deviceData, hostData, 1024);
    });

    // Device to host transfer
    std::cout << "Device to host transfer (1KB):" << std::endl;
    measureTime([&]() {
        gpuPool.copyDeviceToHost(hostData, deviceData, 1024);
    });

    // Larger transfers
    void* largeHostData = cpuPool.allocate(64 * 1024);  // 64KB
    void* largeDeviceData = gpuPool.allocate(64 * 1024);

    memset(largeHostData, 0xBB, 64 * 1024);

    std::cout << "Host to device transfer (64KB):" << std::endl;
    measureTime([&]() {
        gpuPool.copyHostToDevice(largeDeviceData, largeHostData, 64 * 1024);
    });

    std::cout << "Device to host transfer (64KB):" << std::endl;
    measureTime([&]() {
        gpuPool.copyDeviceToHost(largeHostData, largeDeviceData, 64 * 1024);
    });

    // Clean up
    cpuPool.deallocate(hostData);
    cpuPool.deallocate(largeHostData);
    gpuPool.deallocate(deviceData);
    gpuPool.deallocate(largeDeviceData);

    std::cout << "Data transfer benchmark completed." << std::endl;
}

// Benchmark device-to-device transfer
void benchmarkDeviceToDeviceTransfer() {
    std::cout << "Benchmarking Device-to-Device Transfer..." << std::endl;

    // Create two GPU pools (could be on different devices if available)
    PoolConfig gpuConfig1;
    gpuConfig1.allocatorType = AllocatorType::VariableSize;
    gpuConfig1.initialSize = 1024 * 1024;  // 1MB
    gpuConfig1.deviceId = 0;
    GPUMemoryPool gpuPool1("gpu_pool_1", gpuConfig1);

    PoolConfig gpuConfig2;
    gpuConfig2.allocatorType = AllocatorType::VariableSize;
    gpuConfig2.initialSize = 1024 * 1024;  // 1MB
    gpuConfig2.deviceId = 0;  // Same device for now
    GPUMemoryPool gpuPool2("gpu_pool_2", gpuConfig2);

    // Allocate memory
    void* deviceData1 = gpuPool1.allocate(1024);
    void* deviceData2 = gpuPool2.allocate(1024);

    // Initialize first device buffer
    std::vector<char> hostData(1024, 0xCC);
    gpuPool1.copyHostToDevice(deviceData1, hostData.data(), 1024);

    // Device to device transfer
    std::cout << "Device to device transfer (1KB):" << std::endl;
    measureTime([&]() {
        gpuPool1.copyDeviceToDevice(deviceData2, deviceData1, 1024);
    });

    // Larger transfers
    void* largeDeviceData1 = gpuPool1.allocate(64 * 1024);
    void* largeDeviceData2 = gpuPool2.allocate(64 * 1024);

    std::vector<char> largeHostData(64 * 1024, 0xDD);
    gpuPool1.copyHostToDevice(largeDeviceData1, largeHostData.data(), 64 * 1024);

    std::cout << "Device to device transfer (64KB):" << std::endl;
    measureTime([&]() {
        gpuPool1.copyDeviceToDevice(largeDeviceData2, largeDeviceData1, 64 * 1024);
    });

    // Clean up
    gpuPool1.deallocate(deviceData1);
    gpuPool1.deallocate(largeDeviceData1);
    gpuPool2.deallocate(deviceData2);
    gpuPool2.deallocate(largeDeviceData2);

    std::cout << "Device-to-device transfer benchmark completed." << std::endl;
}

// Benchmark memory pool manager GPU operations
void benchmarkMemoryPoolManagerGPU() {
    std::cout << "Benchmarking MemoryPoolManager GPU Operations..." << std::endl;

    MemoryPoolManager& manager = MemoryPoolManager::getInstance();

    // Create GPU pool
    PoolConfig config;
    config.allocatorType = AllocatorType::FixedSize;
    config.blockSize = 512;
    config.deviceId = 0;

    IMemoryPool* gpuPool = manager.createGPUPool("manager_gpu_benchmark", config);

    // Benchmark allocation through manager
    std::cout << "Allocation/deallocation through manager:" << std::endl;
    measureTime([&]() {
        void* ptr = gpuPool->allocate(256);
        gpuPool->deallocate(ptr);
    });

    // Benchmark statistics retrieval
    std::cout << "Statistics retrieval:" << std::endl;
    measureTime([&]() {
        auto stats = manager.getAllStats();
        (void)stats;  // Suppress unused variable warning
    });

    // Clean up
    manager.destroyPool("manager_gpu_benchmark");

    std::cout << "MemoryPoolManager GPU benchmark completed." << std::endl;
}

// Compare GPU performance
void compareGPUPerformance() {
    std::cout << "\n=== GPU Performance Comparison ===" << std::endl;

    std::cout << "GPU Memory Pool vs Standard CUDA:" << std::endl;
    std::cout << "Fixed size (512 bytes):" << std::endl;

    // GPU Pool
    PoolConfig config;
    config.allocatorType = AllocatorType::FixedSize;
    config.blockSize = 512;
    config.deviceId = 0;
    GPUMemoryPool pool("gpu_compare_pool", config);

    double poolTime = measureTime([&]() {
        void* ptr = pool.allocate(512);
        pool.deallocate(ptr);
    });

    // CUDA malloc/free
    double cudaTime = measureTime([]() {
        void* ptr;
        cudaMalloc(&ptr, 512);
        cudaFree(ptr);
    });

    std::cout << "Performance ratios (lower is better):" << std::endl;
    std::cout << "  Pool vs cudaMalloc: " << (poolTime / cudaTime) << std::endl;
}

int main() {
    std::cout << "GPU Memory Pool Performance Benchmarks" << std::endl;
    std::cout << "=====================================" << std::endl;

    if (!isCudaAvailable()) {
        std::cout << "CUDA not available - skipping GPU benchmarks" << std::endl;
        return 0;
    }

    try {
        benchmarkGPUPoolAllocation();
        std::cout << std::endl;

        benchmarkCudaAllocator();
        std::cout << std::endl;

        benchmarkDataTransfer();
        std::cout << std::endl;

        benchmarkDeviceToDeviceTransfer();
        std::cout << std::endl;

        benchmarkMemoryPoolManagerGPU();
        std::cout << std::endl;

        compareGPUPerformance();

        std::cout << "\nAll GPU benchmarks completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "GPU benchmark failed with exception: " << e.what() << std::endl;
        return 1;
    }
}