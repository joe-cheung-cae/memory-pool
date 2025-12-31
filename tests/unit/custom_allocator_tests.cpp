#include "memory_pool/custom/custom_allocator.hpp"
#include "memory_pool/custom/rdma_allocator.hpp"
#include "memory_pool/custom/custom_memory_pool.hpp"
#include "memory_pool/memory_pool.hpp"
#include <functional>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <string>

using namespace memory_pool;

// Test fixture for custom allocator tests
class CustomAllocatorTest {
public:
    CustomAllocatorTest() {
        // Set up test environment
        std::cout << "Setting up custom allocator test environment...\n";

        // Register RDMA allocator for testing
        CustomAllocatorRegistry::registerAllocator(
            HardwareType::RDMA,
            createRDMAAllocator,
            "Test_RDMA"
        );
    }

    ~CustomAllocatorTest() {
        // Clean up test environment
        std::cout << "Cleaning up custom allocator test environment...\n";

        // Clean up any created pools
        auto& manager = MemoryPoolManager::getInstance();
        manager.destroyPool("test_custom_pool");
        manager.destroyPool("manager_test_pool");
        manager.destroyPool("invalid_pool");
    }

    // Test HardwareType enum
    void testHardwareTypeValues() {
        std::cout << "Testing HardwareType enum values...\n";

        assert(static_cast<int>(HardwareType::RDMA) == 0);
        assert(static_cast<int>(HardwareType::FPGA) == 1);
        assert(static_cast<int>(HardwareType::ASIC) == 2);
        assert(static_cast<int>(HardwareType::Custom) == 3);

        std::cout << "HardwareType enum test passed!\n";
    }

    // Test RDMAConfig structure
    void testRDMAConfigDefaults() {
        std::cout << "Testing RDMAConfig defaults...\n";

        RDMAConfig config;
        assert(config.deviceName.empty());
        assert(config.transportType.empty());
        assert(config.maxMemorySize == 0u);
        assert(!config.enableZeroCopy);

        std::cout << "RDMAConfig defaults test passed!\n";
    }

    // Test CustomAllocatorRegistry
    void testRegistryRegistration() {
        std::cout << "Testing allocator registry...\n";

        // Check if RDMA allocator is registered
        assert(CustomAllocatorRegistry::hasAllocator(HardwareType::RDMA));

        // Get registered allocators
        auto names = CustomAllocatorRegistry::getRegisteredAllocators(HardwareType::RDMA);
        assert(!names.empty());
        assert(names[0] == "Test_RDMA");

        std::cout << "Registry registration test passed!\n";
    }

    // Test RDMA allocator creation
    void testRDMAAllocatorCreation() {
        std::cout << "Testing RDMA allocator creation...\n";

        RDMAConfig config;
        config.deviceName = "test_device";
        config.transportType = "RoCE";
        config.maxMemorySize = 1024 * 1024;
        config.enableZeroCopy = true;

        HardwareConfig hwConfig = config;
        auto allocator = CustomAllocatorRegistry::createAllocator(HardwareType::RDMA, hwConfig);
        assert(allocator != nullptr);

        assert(allocator->getHardwareType() == HardwareType::RDMA);
        assert(allocator->isHardwareAvailable());

        std::cout << "RDMA allocator creation test passed!\n";
    }

    // Test RDMA allocator memory operations
    void testRDMAAllocatorMemoryOperations() {
        std::cout << "Testing RDMA allocator memory operations...\n";

        RDMAConfig config;
        config.deviceName = "test_device";
        config.transportType = "RoCE";

        HardwareConfig hwConfig = config;
        auto allocator = CustomAllocatorRegistry::createAllocator(HardwareType::RDMA, hwConfig);
        assert(allocator != nullptr);

        // Test allocation
        const size_t testSize = 4096;
        void* ptr = allocator->allocate(testSize);
        assert(ptr != nullptr);

        // Test ownership
        assert(allocator->owns(ptr));

        // Test block size (should be aligned)
        size_t blockSize = allocator->getBlockSize(ptr);
        assert(blockSize >= testSize);
        assert(blockSize % 4096 == 0u);  // Should be page-aligned

        // Test memory registration
        uint64_t handle = allocator->registerMemory(ptr, testSize);
        assert(handle != 0u);

        // Test hardware info
        auto info = allocator->getHardwareInfo(ptr);
        assert(!info.empty());
        assert(info["device"] == "test_device");
        assert(info["transport"] == "RoCE");
        assert(info["registered"] == "true");

        // Test synchronization
        allocator->synchronize(ptr, testSize);

        // Test unregistration
        allocator->unregisterMemory(handle);

        // Test deallocation
        allocator->deallocate(ptr);

        // Test reset
        allocator->reset();

        std::cout << "RDMA allocator memory operations test passed!\n";
    }

    // Test CustomMemoryPool creation
    void testCustomMemoryPoolCreation() {
        std::cout << "Testing custom memory pool creation...\n";

        RDMAConfig rdmaConfig;
        rdmaConfig.deviceName = "test_device";

        PoolConfig poolConfig;
        poolConfig.allocatorType = AllocatorType::Custom;
        poolConfig.hardwareType = HardwareType::RDMA;
        poolConfig.hardwareConfig = rdmaConfig;
        poolConfig.trackStats = true;

        auto& manager = MemoryPoolManager::getInstance();
        IMemoryPool* pool = manager.createCustomPool("test_custom_pool", poolConfig);
        assert(pool != nullptr);

        assert(pool->getMemoryType() == MemoryType::Custom);
        assert(pool->getName() == "test_custom_pool");

        std::cout << "Custom memory pool creation test passed!\n";
    }

    // Test CustomMemoryPool operations
    void testCustomMemoryPoolOperations() {
        std::cout << "Testing custom memory pool operations...\n";

        RDMAConfig rdmaConfig;
        rdmaConfig.deviceName = "test_device";

        PoolConfig poolConfig;
        poolConfig.allocatorType = AllocatorType::Custom;
        poolConfig.hardwareType = HardwareType::RDMA;
        poolConfig.hardwareConfig = rdmaConfig;
        poolConfig.trackStats = true;

        auto& manager = MemoryPoolManager::getInstance();
        IMemoryPool* pool = manager.createCustomPool("test_custom_pool", poolConfig);
        assert(pool != nullptr);

        // Test allocation
        const size_t testSize = 8192;
        void* buffer = pool->allocate(testSize);
        assert(buffer != nullptr);

        // Test statistics
        const auto& stats = pool->getStats();
        assert(stats.getCurrentUsed() == testSize);

        // Test hardware-specific operations
        auto customPool = static_cast<CustomMemoryPool*>(pool);
        uint64_t handle = customPool->registerMemory(buffer, testSize);
        assert(handle != 0u);

        auto info = customPool->getHardwareInfo(buffer);
        assert(!info.empty());

        customPool->synchronize(buffer, testSize);
        customPool->unregisterMemory(handle);

        // Test deallocation
        pool->deallocate(buffer);
        assert(stats.getCurrentUsed() == 0u);

        // Test reset
        pool->reset();

        std::cout << "Custom memory pool operations test passed!\n";
    }

    // Test error handling
    void testErrorHandling() {
        std::cout << "Testing error handling...\n";

        // Test invalid hardware type
        HardwareConfig invalidConfig;
        try {
            CustomAllocatorRegistry::createAllocator(HardwareType::FPGA, invalidConfig);
            assert(false);  // Should throw
        } catch (const InvalidOperationException&) {
            // Expected
        }

        // Test invalid config type for RDMA
        HardwareConfig wrongConfig = std::unordered_map<std::string, std::string>{};
        try {
            CustomAllocatorRegistry::createAllocator(HardwareType::RDMA, wrongConfig);
            assert(false);  // Should throw
        } catch (const InvalidOperationException&) {
            // Expected
        }

        // Test invalid pool config
        PoolConfig invalidPoolConfig;
        invalidPoolConfig.allocatorType = AllocatorType::Custom;
        invalidPoolConfig.hardwareType = HardwareType::FPGA;  // No allocator registered

        auto& manager = MemoryPoolManager::getInstance();
        assert(manager.createCustomPool("invalid_pool", invalidPoolConfig) == nullptr);

        std::cout << "Error handling test passed!\n";
    }

    // Test MemoryPoolManager custom pool methods
    void testMemoryPoolManagerCustomPools() {
        std::cout << "Testing MemoryPoolManager custom pool methods...\n";

        RDMAConfig rdmaConfig;
        rdmaConfig.deviceName = "test_device";

        PoolConfig poolConfig;
        poolConfig.allocatorType = AllocatorType::Custom;
        poolConfig.hardwareType = HardwareType::RDMA;
        poolConfig.hardwareConfig = rdmaConfig;

        auto& manager = MemoryPoolManager::getInstance();

        // Create pool
        IMemoryPool* pool = manager.createCustomPool("manager_test_pool", poolConfig);
        assert(pool != nullptr);

        // Get pool
        IMemoryPool* retrieved = manager.getCustomPool("manager_test_pool");
        assert(retrieved == pool);

        // Test non-existent pool
        assert(manager.getCustomPool("non_existent") == nullptr);

        // Clean up
        manager.destroyPool("manager_test_pool");

        std::cout << "MemoryPoolManager custom pools test passed!\n";
    }

    // Test PoolConfig custom fields
    void testPoolConfigCustomFields() {
        std::cout << "Testing PoolConfig custom fields...\n";

        PoolConfig config;
        assert(config.hardwareType == HardwareType::Custom);

        // Test RDMA config assignment
        RDMAConfig rdmaConfig;
        rdmaConfig.deviceName = "mlx5_0";
        config.hardwareConfig = rdmaConfig;

        // Verify the config is stored
        assert(std::holds_alternative<RDMAConfig>(config.hardwareConfig));
        auto storedConfig = std::get<RDMAConfig>(config.hardwareConfig);
        assert(storedConfig.deviceName == "mlx5_0");

        std::cout << "PoolConfig custom fields test passed!\n";
    }

    // Test multiple allocator registrations
    void testMultipleAllocatorRegistration() {
        std::cout << "Testing multiple allocator registration...\n";

        // Register another RDMA allocator
        CustomAllocatorRegistry::registerAllocator(
            HardwareType::RDMA,
            createRDMAAllocator,
            "Test_RDMA_v2"
        );

        auto names = CustomAllocatorRegistry::getRegisteredAllocators(HardwareType::RDMA);
        assert(names.size() >= 2u);

        // Should still create allocator (uses first registered)
        RDMAConfig config;
        HardwareConfig hwConfig = config;
        auto allocator = CustomAllocatorRegistry::createAllocator(HardwareType::RDMA, hwConfig);
        assert(allocator != nullptr);

        std::cout << "Multiple allocator registration test passed!\n";
    }

    // Run all tests
    void runAllTests() {
        testHardwareTypeValues();
        testRDMAConfigDefaults();
        testRegistryRegistration();
        testRDMAAllocatorCreation();
        testRDMAAllocatorMemoryOperations();
        testCustomMemoryPoolCreation();
        testCustomMemoryPoolOperations();
        testErrorHandling();
        testMemoryPoolManagerCustomPools();
        testPoolConfigCustomFields();
        testMultipleAllocatorRegistration();

        std::cout << "All custom allocator tests passed!\n";
    }
};

int main() {
    try {
        CustomAllocatorTest test;
        test.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}