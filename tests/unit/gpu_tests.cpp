#include "memory_pool/memory_pool.hpp"
#include "memory_pool/gpu/gpu_memory_pool.hpp"
#include "memory_pool/gpu/cuda_allocator.hpp"
#include "memory_pool/gpu/cuda_utils.hpp"
#include "memory_pool/common.hpp"
#include <iostream>
#include <vector>
#include <cstring>

using namespace memory_pool;

// Test helper functions
bool testCudaDeviceAvailable() {
    try {
        int deviceCount = getDeviceCount();
        if (deviceCount == 0) {
            std::cout << "No CUDA devices available, skipping GPU tests" << std::endl;
            return false;
        }

        if (!isDeviceAvailable(0)) {
            std::cout << "CUDA device 0 not available, skipping GPU tests" << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cout << "CUDA not available: " << e.what() << ", skipping GPU tests" << std::endl;
        return false;
    }
}

void testCudaFixedSizeAllocator() {
    std::cout << "Testing CudaFixedSizeAllocator..." << std::endl;

    const size_t BLOCK_SIZE     = 256;
    const size_t INITIAL_BLOCKS = 10;

    CudaFixedSizeAllocator allocator(BLOCK_SIZE, INITIAL_BLOCKS, 0);

    // Test basic allocation
    void* ptr1 = allocator.allocate(BLOCK_SIZE, AllocFlags::None);
    if (ptr1 == nullptr) {
        throw std::runtime_error("Failed to allocate block");
    }

    void* ptr2 = allocator.allocate(BLOCK_SIZE, AllocFlags::None);
    if (ptr2 == nullptr) {
        throw std::runtime_error("Failed to allocate second block");
    }

    // Test ownership
    if (!allocator.owns(ptr1) || !allocator.owns(ptr2)) {
        throw std::runtime_error("Allocator should own allocated pointers");
    }

    // Test block size
    if (allocator.getBlockSize(ptr1) != BLOCK_SIZE) {
        throw std::runtime_error("Block size mismatch");
    }

    // Test statistics
    if (allocator.getUsedBlocks() != 2) {
        throw std::runtime_error("Used blocks count incorrect");
    }

    if (allocator.getFreeBlocks() != INITIAL_BLOCKS - 2) {
        throw std::runtime_error("Free blocks count incorrect");
    }

    // Test deallocation
    allocator.deallocate(ptr1);
    if (allocator.getUsedBlocks() != 1) {
        throw std::runtime_error("Deallocation failed");
    }

    allocator.deallocate(ptr2);
    if (allocator.getUsedBlocks() != 0) {
        throw std::runtime_error("Second deallocation failed");
    }

    // Test reset
    void* ptr3 = allocator.allocate(BLOCK_SIZE, AllocFlags::None);
    void* ptr4 = allocator.allocate(BLOCK_SIZE, AllocFlags::None);
    allocator.reset();

    if (allocator.getUsedBlocks() != 0) {
        throw std::runtime_error("Reset failed");
    }

    std::cout << "CudaFixedSizeAllocator tests passed!" << std::endl;
}

void testCudaVariableSizeAllocator() {
    std::cout << "Testing CudaVariableSizeAllocator..." << std::endl;

    const size_t INITIAL_SIZE = 1024;

    CudaVariableSizeAllocator allocator(INITIAL_SIZE, 0);

    // Test basic allocation
    void* ptr1 = allocator.allocate(128, AllocFlags::None);
    if (ptr1 == nullptr) {
        throw std::runtime_error("Failed to allocate memory");
    }

    void* ptr2 = allocator.allocate(256, AllocFlags::None);
    if (ptr2 == nullptr) {
        throw std::runtime_error("Failed to allocate second memory block");
    }

    // Test ownership
    if (!allocator.owns(ptr1) || !allocator.owns(ptr2)) {
        throw std::runtime_error("Allocator should own allocated pointers");
    }

    // Test block sizes
    if (allocator.getBlockSize(ptr1) != align_size(128)) {
        throw std::runtime_error("Block size mismatch for ptr1");
    }

    if (allocator.getBlockSize(ptr2) != align_size(256)) {
        throw std::runtime_error("Block size mismatch for ptr2");
    }

    // Test statistics
    size_t expectedUsed = align_size(128) + align_size(256);
    if (allocator.getUsedSize() != expectedUsed) {
        throw std::runtime_error("Used size incorrect");
    }

    // Test deallocation
    allocator.deallocate(ptr1);
    if (allocator.getUsedSize() != align_size(256)) {
        throw std::runtime_error("Deallocation failed");
    }

    allocator.deallocate(ptr2);
    if (allocator.getUsedSize() != 0) {
        throw std::runtime_error("Second deallocation failed");
    }

    // Test reset
    void* ptr3 = allocator.allocate(64, AllocFlags::None);
    void* ptr4 = allocator.allocate(32, AllocFlags::None);
    allocator.reset();

    if (allocator.getUsedSize() != 0) {
        throw std::runtime_error("Reset failed");
    }

    std::cout << "CudaVariableSizeAllocator tests passed!" << std::endl;
}

void testGPUMemoryPool() {
    std::cout << "Testing GPUMemoryPool..." << std::endl;

    // Create GPU memory pool
    PoolConfig config;
    config.allocatorType = AllocatorType::FixedSize;
    config.blockSize     = 512;
    config.initialSize   = 1024 * 10;  // 10KB
    config.deviceId      = 0;

    GPUMemoryPool pool("test_gpu_pool", config);

    // Test basic allocation
    void* ptr1 = pool.allocate(256);
    if (ptr1 == nullptr) {
        throw std::runtime_error("Failed to allocate from GPU pool");
    }

    void* ptr2 = pool.allocate(512);
    if (ptr2 == nullptr) {
        throw std::runtime_error("Failed to allocate second block from GPU pool");
    }

    // Test memory type
    if (pool.getMemoryType() != MemoryType::GPU) {
        throw std::runtime_error("Memory type should be GPU");
    }

    // Test pool name
    if (pool.getName() != "test_gpu_pool") {
        throw std::runtime_error("Pool name mismatch");
    }

    // Test statistics
    const MemoryStats& stats = pool.getStats();
    if (stats.getCurrentUsed() == 0) {
        throw std::runtime_error("Statistics not tracking allocations");
    }

    // Test deallocation
    pool.deallocate(ptr1);
    pool.deallocate(ptr2);

    // Test reset
    pool.reset();
    const MemoryStats& resetStats = pool.getStats();
    if (resetStats.getCurrentUsed() != 0) {
        throw std::runtime_error("Reset failed");
    }

    std::cout << "GPUMemoryPool tests passed!" << std::endl;
}

void testGPUMemoryPoolVariableSize() {
    std::cout << "Testing GPUMemoryPool with variable size allocator..." << std::endl;

    // Create GPU memory pool with variable size allocator
    PoolConfig config;
    config.allocatorType = AllocatorType::VariableSize;
    config.initialSize   = 1024 * 10;  // 10KB
    config.deviceId      = 0;

    GPUMemoryPool pool("test_gpu_var_pool", config);

    // Test allocation of different sizes
    void* ptr1 = pool.allocate(100);
    void* ptr2 = pool.allocate(200);
    void* ptr3 = pool.allocate(300);

    if (ptr1 == nullptr || ptr2 == nullptr || ptr3 == nullptr) {
        throw std::runtime_error("Failed to allocate variable sizes");
    }

    // Test statistics
    const MemoryStats& stats = pool.getStats();
    if (stats.getCurrentUsed() < 600) {  // At least 600 bytes used
        throw std::runtime_error("Statistics not tracking variable allocations correctly");
    }

    // Test deallocation
    pool.deallocate(ptr1);
    pool.deallocate(ptr2);
    pool.deallocate(ptr3);

    std::cout << "GPUMemoryPool variable size tests passed!" << std::endl;
}

void testMemoryPoolManagerGPU() {
    std::cout << "Testing MemoryPoolManager GPU functionality..." << std::endl;

    MemoryPoolManager& manager = MemoryPoolManager::getInstance();

    // Create GPU pool
    PoolConfig config;
    config.allocatorType = AllocatorType::FixedSize;
    config.blockSize     = 256;
    config.deviceId      = 0;

    IMemoryPool* gpuPool = manager.createGPUPool("test_manager_gpu", config);
    if (gpuPool == nullptr) {
        throw std::runtime_error("Failed to create GPU pool through manager");
    }

    // Test retrieval
    IMemoryPool* retrievedPool = manager.getGPUPool("test_manager_gpu");
    if (retrievedPool != gpuPool) {
        throw std::runtime_error("Failed to retrieve GPU pool from manager");
    }

    // Test allocation through manager
    void* ptr = gpuPool->allocate(128);
    if (ptr == nullptr) {
        throw std::runtime_error("Failed to allocate through manager-created pool");
    }

    gpuPool->deallocate(ptr);

    // Test statistics
    std::map<std::string, std::string> allStats = manager.getAllStats();
    if (allStats.find("test_manager_gpu") == allStats.end()) {
        throw std::runtime_error("GPU pool not found in manager statistics");
    }

    // Clean up
    if (!manager.destroyPool("test_manager_gpu")) {
        throw std::runtime_error("Failed to destroy GPU pool");
    }

    std::cout << "MemoryPoolManager GPU tests passed!" << std::endl;
}

int main() {
    std::cout << "Running GPU unit tests..." << std::endl;

    try {
        // Check if CUDA is available
        if (!testCudaDeviceAvailable()) {
            std::cout << "CUDA not available - all GPU tests skipped" << std::endl;
            return 0;
        }

        // Run tests
        testCudaFixedSizeAllocator();
        testCudaVariableSizeAllocator();
        testGPUMemoryPool();
        testGPUMemoryPoolVariableSize();
        testMemoryPoolManagerGPU();

        std::cout << "All GPU unit tests passed!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "GPU test failed: " << e.what() << std::endl;
        return 1;
    }
}