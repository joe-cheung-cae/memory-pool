#include "memory_pool/memory_pool.hpp"
#include "memory_pool/custom/custom_allocator.hpp"
#include "memory_pool/custom/custom_memory_pool.hpp"
#include "memory_pool/custom/rdma_allocator.hpp"
#include <iostream>
#include <vector>

using namespace memory_pool;

/**
 * @brief Example demonstrating custom allocator usage for RDMA hardware.
 *
 * This example shows how to:
 * 1. Register a custom RDMA allocator
 * 2. Create a custom memory pool
 * 3. Allocate and register memory for RDMA operations
 * 4. Perform RDMA-specific operations
 */
int main() {
    try {
        std::cout << "Custom Allocator Example - RDMA Memory Pool" << std::endl;
        std::cout << "==========================================" << std::endl;

        // Step 1: Register the RDMA allocator factory
        std::cout << "\n1. Registering RDMA allocator factory..." << std::endl;
        CustomAllocatorRegistry::registerAllocator(
            HardwareType::RDMA,
            createRDMAAllocator,
            "RDMA_Example"
        );

        // Step 2: Create RDMA configuration
        std::cout << "\n2. Creating RDMA configuration..." << std::endl;
        RDMAConfig rdmaConfig;
        rdmaConfig.deviceName = "mlx5_0";
        rdmaConfig.transportType = "RoCE";
        rdmaConfig.maxMemorySize = 1024 * 1024 * 1024;  // 1GB
        rdmaConfig.enableZeroCopy = true;

        // Step 3: Create pool configuration
        std::cout << "\n3. Creating custom memory pool configuration..." << std::endl;
        PoolConfig poolConfig;
        poolConfig.allocatorType = AllocatorType::Custom;
        poolConfig.hardwareType = HardwareType::RDMA;
        poolConfig.hardwareConfig = rdmaConfig;
        poolConfig.trackStats = true;

        // Step 4: Create custom memory pool
        std::cout << "\n4. Creating custom RDMA memory pool..." << std::endl;
        auto& manager = MemoryPoolManager::getInstance();
        IMemoryPool* pool = manager.createCustomPool("rdma_pool", poolConfig);
        if (!pool) {
            throw std::runtime_error("Failed to create custom pool");
        }

        // Step 5: Allocate memory
        std::cout << "\n5. Allocating RDMA memory..." << std::endl;
        const size_t bufferSize = 64 * 1024;  // 64KB
        void* buffer = pool->allocate(bufferSize);
        std::cout << "Allocated " << bufferSize << " bytes at " << buffer << std::endl;

        // Step 6: Register memory for RDMA
        std::cout << "\n6. Registering memory for RDMA operations..." << std::endl;
        uint64_t rdmaHandle = static_cast<CustomMemoryPool*>(pool)->registerMemory(buffer, bufferSize);
        std::cout << "Memory registered with RDMA handle: " << rdmaHandle << std::endl;

        // Step 7: Get hardware information
        std::cout << "\n7. Getting hardware information..." << std::endl;
        auto hardwareInfo = static_cast<CustomMemoryPool*>(pool)->getHardwareInfo(buffer);
        for (const auto& pair : hardwareInfo) {
            std::cout << "  " << pair.first << ": " << pair.second << std::endl;
        }

        // Step 8: Simulate RDMA operations
        std::cout << "\n8. Simulating RDMA operations..." << std::endl;
        // Fill buffer with test data
        uint32_t* data = static_cast<uint32_t*>(buffer);
        for (size_t i = 0; i < bufferSize / sizeof(uint32_t); ++i) {
            data[i] = static_cast<uint32_t>(i);
        }

        // Synchronize memory (ensure RDMA operations are complete)
        static_cast<CustomMemoryPool*>(pool)->synchronize(buffer, bufferSize);

        // Step 9: Get memory statistics
        std::cout << "\n9. Memory pool statistics:" << std::endl;
        const auto& stats = pool->getStats();
        std::cout << "  Total allocated: " << stats.getTotalAllocated() << " bytes" << std::endl;
        std::cout << "  Current used: " << stats.getCurrentUsed() << " bytes" << std::endl;
        std::cout << "  Peak usage: " << stats.getPeakUsage() << " bytes" << std::endl;

        // Step 10: Cleanup
        std::cout << "\n10. Cleaning up..." << std::endl;
        static_cast<CustomMemoryPool*>(pool)->unregisterMemory(rdmaHandle);
        pool->deallocate(buffer);

        std::cout << "\nExample completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}