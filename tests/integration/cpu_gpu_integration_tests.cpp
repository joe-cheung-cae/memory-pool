#include "../../include/memory_pool/memory_pool.hpp"
#include "../../include/memory_pool/gpu/gpu_memory_pool.hpp"
#include <iostream>
#include <vector>
#include <cstring>

using namespace memory_pool;

// Test helper functions
bool testCudaDeviceAvailable() {
    try {
        int deviceCount = getDeviceCount();
        if (deviceCount == 0) {
            std::cout << "No CUDA devices available, skipping integration tests" << std::endl;
            return false;
        }

        if (!isDeviceAvailable(0)) {
            std::cout << "CUDA device 0 not available, skipping integration tests" << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cout << "CUDA not available: " << e.what() << ", skipping integration tests" << std::endl;
        return false;
    }
}

void testBasicCpuGpuDataTransfer() {
    std::cout << "Testing basic CPU-GPU data transfer..." << std::endl;

    MemoryPoolManager& manager = MemoryPoolManager::getInstance();

    // Create CPU pool
    PoolConfig cpuConfig;
    cpuConfig.allocatorType = AllocatorType::FixedSize;
    cpuConfig.blockSize = 1024;
    IMemoryPool* cpuPool = manager.createCPUPool("cpu_transfer_pool", cpuConfig);

    // Create GPU pool
    PoolConfig gpuConfig;
    gpuConfig.allocatorType = AllocatorType::FixedSize;
    gpuConfig.blockSize = 1024;
    gpuConfig.deviceId = 0;
    GPUMemoryPool* gpuPool = static_cast<GPUMemoryPool*>(manager.createGPUPool("gpu_transfer_pool", gpuConfig));

    // Allocate CPU memory
    void* hostData = cpuPool->allocate(512);
    if (hostData == nullptr) {
        throw std::runtime_error("Failed to allocate host memory");
    }

    // Fill host data
    const char* testString = "Hello GPU World!";
    std::strncpy(static_cast<char*>(hostData), testString, 512);

    // Allocate GPU memory
    void* deviceData = gpuPool->allocate(512);
    if (deviceData == nullptr) {
        cpuPool->deallocate(hostData);
        throw std::runtime_error("Failed to allocate device memory");
    }

    // Copy host to device
    gpuPool->copyHostToDevice(deviceData, hostData, 512);

    // Allocate another host buffer for result
    void* resultData = cpuPool->allocate(512);
    if (resultData == nullptr) {
        cpuPool->deallocate(hostData);
        gpuPool->deallocate(deviceData);
        throw std::runtime_error("Failed to allocate result buffer");
    }

    // Copy device to host
    gpuPool->copyDeviceToHost(resultData, deviceData, 512);

    // Verify data
    if (std::strcmp(static_cast<char*>(hostData), static_cast<char*>(resultData)) != 0) {
        cpuPool->deallocate(hostData);
        cpuPool->deallocate(resultData);
        gpuPool->deallocate(deviceData);
        throw std::runtime_error("Data transfer verification failed");
    }

    // Clean up
    cpuPool->deallocate(hostData);
    cpuPool->deallocate(resultData);
    gpuPool->deallocate(deviceData);

    std::cout << "Basic CPU-GPU data transfer tests passed!" << std::endl;
}

void testDeviceToDeviceTransfer() {
    std::cout << "Testing device-to-device data transfer..." << std::endl;

    MemoryPoolManager& manager = MemoryPoolManager::getInstance();

    // Create two GPU pools (could be on different devices if available)
    PoolConfig gpuConfig1;
    gpuConfig1.allocatorType = AllocatorType::FixedSize;
    gpuConfig1.blockSize = 512;
    gpuConfig1.deviceId = 0;
    GPUMemoryPool* gpuPool1 = static_cast<GPUMemoryPool*>(manager.createGPUPool("gpu_pool_1", gpuConfig1));

    PoolConfig gpuConfig2;
    gpuConfig2.allocatorType = AllocatorType::FixedSize;
    gpuConfig2.blockSize = 512;
    gpuConfig2.deviceId = 0; // Same device for now
    GPUMemoryPool* gpuPool2 = static_cast<GPUMemoryPool*>(manager.createGPUPool("gpu_pool_2", gpuConfig2));

    // Allocate memory on both devices
    void* deviceData1 = gpuPool1->allocate(256);
    void* deviceData2 = gpuPool2->allocate(256);

    if (deviceData1 == nullptr || deviceData2 == nullptr) {
        if (deviceData1) gpuPool1->deallocate(deviceData1);
        if (deviceData2) gpuPool2->deallocate(deviceData2);
        throw std::runtime_error("Failed to allocate device memory");
    }

    // Copy device to device
    gpuPool1->copyDeviceToDevice(deviceData2, deviceData1, 256);

    // Clean up
    gpuPool1->deallocate(deviceData1);
    gpuPool2->deallocate(deviceData2);

    std::cout << "Device-to-device data transfer tests passed!" << std::endl;
}

void testMemoryPoolManagerIntegration() {
    std::cout << "Testing MemoryPoolManager CPU-GPU integration..." << std::endl;

    MemoryPoolManager& manager = MemoryPoolManager::getInstance();

    // Create CPU and GPU pools
    PoolConfig cpuConfig;
    cpuConfig.allocatorType = AllocatorType::VariableSize;
    IMemoryPool* cpuPool = manager.createCPUPool("integration_cpu", cpuConfig);

    PoolConfig gpuConfig;
    gpuConfig.allocatorType = AllocatorType::VariableSize;
    gpuConfig.deviceId = 0;
    IMemoryPool* gpuPool = manager.createGPUPool("integration_gpu", gpuConfig);

    // Test allocation from both pools
    void* cpuPtr = cpuPool->allocate(1024);
    void* gpuPtr = gpuPool->allocate(1024);

    if (cpuPtr == nullptr || gpuPtr == nullptr) {
        if (cpuPtr) cpuPool->deallocate(cpuPtr);
        if (gpuPtr) gpuPool->deallocate(gpuPtr);
        throw std::runtime_error("Failed to allocate from integrated pools");
    }

    // Test statistics collection
    std::map<std::string, std::string> stats = manager.getAllStats();
    if (stats.find("integration_cpu") == stats.end() ||
        stats.find("integration_gpu") == stats.end()) {
        cpuPool->deallocate(cpuPtr);
        gpuPool->deallocate(gpuPtr);
        throw std::runtime_error("Pool statistics not found");
    }

    // Clean up
    cpuPool->deallocate(cpuPtr);
    gpuPool->deallocate(gpuPtr);

    // Destroy pools
    if (!manager.destroyPool("integration_cpu") ||
        !manager.destroyPool("integration_gpu")) {
        throw std::runtime_error("Failed to destroy integrated pools");
    }

    std::cout << "MemoryPoolManager CPU-GPU integration tests passed!" << std::endl;
}

void testCrossPoolDataTransfer() {
    std::cout << "Testing cross-pool data transfer..." << std::endl;

    MemoryPoolManager& manager = MemoryPoolManager::getInstance();

    // Create CPU pool with pinned memory for efficient transfer
    PoolConfig cpuConfig;
    cpuConfig.allocatorType = AllocatorType::FixedSize;
    cpuConfig.blockSize = 1024;
    cpuConfig.usePinnedMemory = true;
    IMemoryPool* cpuPool = manager.createCPUPool("pinned_cpu", cpuConfig);

    // Create GPU pool
    PoolConfig gpuConfig;
    gpuConfig.allocatorType = AllocatorType::FixedSize;
    gpuConfig.blockSize = 1024;
    gpuConfig.deviceId = 0;
    GPUMemoryPool* gpuPool = static_cast<GPUMemoryPool*>(manager.createGPUPool("transfer_gpu", gpuConfig));

    // Allocate and initialize data
    void* hostData = cpuPool->allocate(512);
    if (hostData == nullptr) {
        throw std::runtime_error("Failed to allocate pinned host memory");
    }

    // Initialize with pattern
    for (int i = 0; i < 512; ++i) {
        static_cast<char*>(hostData)[i] = static_cast<char>(i % 256);
    }

    void* deviceData = gpuPool->allocate(512);
    if (deviceData == nullptr) {
        cpuPool->deallocate(hostData);
        throw std::runtime_error("Failed to allocate device memory");
    }

    // Transfer data
    gpuPool->copyHostToDevice(deviceData, hostData, 512);

    // Allocate result buffer
    void* resultData = cpuPool->allocate(512);
    if (resultData == nullptr) {
        cpuPool->deallocate(hostData);
        gpuPool->deallocate(deviceData);
        throw std::runtime_error("Failed to allocate result buffer");
    }

    // Transfer back
    gpuPool->copyDeviceToHost(resultData, deviceData, 512);

    // Verify
    if (std::memcmp(hostData, resultData, 512) != 0) {
        cpuPool->deallocate(hostData);
        cpuPool->deallocate(resultData);
        gpuPool->deallocate(deviceData);
        throw std::runtime_error("Cross-pool transfer verification failed");
    }

    // Clean up
    cpuPool->deallocate(hostData);
    cpuPool->deallocate(resultData);
    gpuPool->deallocate(deviceData);

    std::cout << "Cross-pool data transfer tests passed!" << std::endl;
}

int main() {
    std::cout << "Running CPU-GPU integration tests..." << std::endl;

    try {
        // Check if CUDA is available
        if (!testCudaDeviceAvailable()) {
            std::cout << "CUDA not available - all integration tests skipped" << std::endl;
            return 0;
        }

        // Run integration tests
        testBasicCpuGpuDataTransfer();
        testDeviceToDeviceTransfer();
        testMemoryPoolManagerIntegration();
        testCrossPoolDataTransfer();

        std::cout << "All CPU-GPU integration tests passed!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Integration test failed: " << e.what() << std::endl;
        return 1;
    }
}