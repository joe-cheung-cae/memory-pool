#include "../include/memory_pool/memory_pool.hpp"
#include "../include/memory_pool/gpu/gpu_memory_pool.hpp"
#include <iostream>
#include <vector>
#include <chrono>

using namespace memory_pool;

// Simple vector addition kernel (this would be in a .cu file in a real project)
// Here we're just showing the memory management part
void vectorAdd(const float* a, const float* b, float* c, int n) {
    // In a real CUDA project, this would be a kernel call like:
    // vectorAddKernel<<<blocks, threads>>>(a, b, c, n);
    
    // For demonstration purposes, we'll just simulate the operation on CPU
    std::cout << "Simulating GPU vector addition (would be a CUDA kernel in real code)...\n";
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Example of using GPU memory pool for a simple vector addition
void gpuVectorAddExample() {
    std::cout << "\n=== GPU Vector Addition Example ===\n";
    
    // Create a GPU memory pool
    auto& manager = MemoryPoolManager::getInstance();
    PoolConfig config = PoolConfig::DefaultGPU();
    GPUMemoryPool* pool = static_cast<GPUMemoryPool*>(manager.createGPUPool("vector_add_pool", config));
    
    // Vector size
    const int N = 1000000;
    const size_t size = N * sizeof(float);
    
    std::cout << "Allocating memory for vectors of size " << N << "...\n";
    
    // Allocate device memory
    float* d_a = static_cast<float*>(pool->allocate(size));
    float* d_b = static_cast<float*>(pool->allocate(size));
    float* d_c = static_cast<float*>(pool->allocate(size));
    
    // Allocate host memory
    std::vector<float> h_a(N, 1.0f);  // Initialize with 1.0
    std::vector<float> h_b(N, 2.0f);  // Initialize with 2.0
    std::vector<float> h_c(N);
    
    // Copy data from host to device
    std::cout << "Copying data from host to device...\n";
    pool->copyHostToDevice(d_a, h_a.data(), size);
    pool->copyHostToDevice(d_b, h_b.data(), size);
    
    // Launch kernel
    std::cout << "Launching vector addition kernel...\n";
    auto startTime = std::chrono::high_resolution_clock::now();
    
    vectorAdd(d_a, d_b, d_c, N);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    std::cout << "Kernel execution time: " << duration.count() << " microseconds\n";
    
    // Copy result back to host
    std::cout << "Copying result back to host...\n";
    pool->copyDeviceToHost(h_c.data(), d_c, size);
    
    // Verify result
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != 3.0f) {  // 1.0 + 2.0 = 3.0
            correct = false;
            std::cout << "Error at index " << i << ": " << h_c[i] << " != 3.0\n";
            break;
        }
    }
    
    if (correct) {
        std::cout << "Vector addition completed successfully!\n";
    } else {
        std::cout << "Vector addition failed!\n";
    }
    
    // Print memory statistics
    std::cout << "\nMemory pool statistics:\n";
    std::cout << pool->getStats().getStatsString() << std::endl;
    
    // Deallocate device memory
    std::cout << "Deallocating device memory...\n";
    pool->deallocate(d_a);
    pool->deallocate(d_b);
    pool->deallocate(d_c);
}

// Example of using pinned memory for faster host-device transfers
void pinnedMemoryExample() {
    std::cout << "\n=== Pinned Memory Example ===\n";
    
    // Create a GPU memory pool with pinned memory
    auto& manager = MemoryPoolManager::getInstance();
    PoolConfig config = PoolConfig::DefaultGPU();
    config.usePinnedMemory = true;
    GPUMemoryPool* pool = static_cast<GPUMemoryPool*>(manager.createGPUPool("pinned_memory_pool", config));
    
    // Vector size
    const int N = 10000000;
    const size_t size = N * sizeof(float);
    
    std::cout << "Allocating pinned host memory and device memory...\n";
    
    // Allocate device memory
    float* d_data = static_cast<float*>(pool->allocate(size));
    
    // Allocate pinned host memory
    float* h_data_pinned = static_cast<float*>(pool->allocate(size, AllocFlags::Pinned));
    
    // Allocate regular (pageable) host memory for comparison
    float* h_data_pageable = new float[N];
    
    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_data_pinned[i] = static_cast<float>(i);
        h_data_pageable[i] = static_cast<float>(i);
    }
    
    // Transfer with pinned memory
    std::cout << "Transferring data using pinned memory...\n";
    auto startTimePinned = std::chrono::high_resolution_clock::now();
    
    pool->copyHostToDevice(d_data, h_data_pinned, size);
    
    auto endTimePinned = std::chrono::high_resolution_clock::now();
    auto durationPinned = std::chrono::duration_cast<std::chrono::microseconds>(endTimePinned - startTimePinned);
    
    // Transfer with pageable memory
    std::cout << "Transferring data using pageable memory...\n";
    auto startTimePageable = std::chrono::high_resolution_clock::now();
    
    pool->copyHostToDevice(d_data, h_data_pageable, size);
    
    auto endTimePageable = std::chrono::high_resolution_clock::now();
    auto durationPageable = std::chrono::duration_cast<std::chrono::microseconds>(endTimePageable - startTimePageable);
    
    // Print results
    std::cout << "Transfer time with pinned memory: " << durationPinned.count() << " microseconds\n";
    std::cout << "Transfer time with pageable memory: " << durationPageable.count() << " microseconds\n";
    std::cout << "Speedup: " << static_cast<float>(durationPageable.count()) / durationPinned.count() << "x\n";
    
    // Deallocate memory
    pool->deallocate(d_data);
    pool->deallocate(h_data_pinned);
    delete[] h_data_pageable;
}

int main() {
    std::cout << "GPU Memory Pool Examples\n";
    std::cout << "=======================\n";
    
    try {
        // Check if CUDA is available
        int deviceCount = getDeviceCount();
        if (deviceCount == 0) {
            std::cout << "No CUDA devices found. Running in simulation mode.\n";
        } else {
            std::cout << "Found " << deviceCount << " CUDA device(s).\n";
            
            // Print device information
            for (int i = 0; i < deviceCount; ++i) {
                setCurrentDevice(i);
                size_t memory = getDeviceMemory(i);
                std::cout << "Device " << i << ": " << memory / (1024 * 1024) << " MB of memory\n";
            }
        }
        
        // Run examples
        gpuVectorAddExample();
        pinnedMemoryExample();
        
        std::cout << "\nAll examples completed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}