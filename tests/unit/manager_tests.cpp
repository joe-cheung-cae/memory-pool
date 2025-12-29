#include "memory_pool/memory_pool.hpp"
#include "memory_pool/cpu/cpu_memory_pool.hpp"
#include "memory_pool/gpu/gpu_memory_pool.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>

using namespace memory_pool;

// Test fixture
class MemoryPoolManagerTest {
public:
    MemoryPoolManagerTest() {
        // Clean up any existing test pools
        MemoryPoolManager& manager = MemoryPoolManager::getInstance();
        manager.destroyPool("test_cpu_pool");
        manager.destroyPool("test_gpu_pool");
        manager.destroyPool("another_test_pool");

        std::cout << "Setting up MemoryPoolManager test environment...\n";
    }

    ~MemoryPoolManagerTest() {
        // Clean up test pools
        MemoryPoolManager& manager = MemoryPoolManager::getInstance();
        manager.destroyPool("test_cpu_pool");
        manager.destroyPool("test_gpu_pool");
        manager.destroyPool("another_test_pool");

        std::cout << "Cleaning up MemoryPoolManager test environment...\n";
    }

    // Test creating and getting CPU pools
    static void testCreateAndGetCPUPool() {
        std::cout << "Testing CPU pool creation and retrieval...\n";

        MemoryPoolManager& manager = MemoryPoolManager::getInstance();

        // Create a CPU pool
        PoolConfig config = PoolConfig::DefaultCPU();
        IMemoryPool* pool = manager.createCPUPool("test_cpu_pool", config);
        assert(pool != nullptr);
        assert(pool->getMemoryType() == MemoryType::CPU);

        // Get the pool back
        IMemoryPool* retrievedPool = manager.getCPUPool("test_cpu_pool");
        assert(retrievedPool == pool);
        (void)retrievedPool;  // Suppress unused variable warning

        // Test allocation
        void* ptr = pool->allocate(1024);
        assert(ptr != nullptr);
        pool->deallocate(ptr);

        std::cout << "CPU pool creation and retrieval test passed!\n";
    }

    // Test creating and getting GPU pools
    static void testCreateAndGetGPUPool() {
        std::cout << "Testing GPU pool creation and retrieval...\n";

        MemoryPoolManager& manager = MemoryPoolManager::getInstance();

        // Create a GPU pool
        PoolConfig config = PoolConfig::DefaultGPU();
        config.deviceId = 0;
        IMemoryPool* pool = manager.createGPUPool("test_gpu_pool", config);
        assert(pool != nullptr);
        assert(pool->getMemoryType() == MemoryType::GPU);

        // Get the pool back
        IMemoryPool* retrievedPool = manager.getGPUPool("test_gpu_pool");
        assert(retrievedPool == pool);
        (void)retrievedPool;  // Suppress unused variable warning

        // Test allocation (only if CUDA is available)
        try {
            void* ptr = pool->allocate(1024);
            if (ptr != nullptr) {
                pool->deallocate(ptr);
            }
        } catch (const std::exception&) {
            // CUDA not available, skip allocation test
        }

        std::cout << "GPU pool creation and retrieval test passed!\n";
    }

    // Test pool destruction
    static void testDestroyPool() {
        std::cout << "Testing pool destruction...\n";

        MemoryPoolManager& manager = MemoryPoolManager::getInstance();

        // Create a pool
        PoolConfig config = PoolConfig::DefaultCPU();
        IMemoryPool* pool = manager.createCPUPool("another_test_pool", config);
        assert(pool != nullptr);
        (void)pool;  // Suppress unused variable warning

        // Verify it exists
        IMemoryPool* retrievedPool = manager.getCPUPool("another_test_pool");
        assert(retrievedPool != nullptr);
        (void)retrievedPool;  // Suppress unused variable warning

        // Destroy the pool
        bool destroyed = manager.destroyPool("another_test_pool");
        assert(destroyed);
        (void)destroyed;  // Suppress unused variable warning

        // Verify it's gone
        retrievedPool = manager.getCPUPool("another_test_pool");
        assert(retrievedPool == nullptr);

        std::cout << "Pool destruction test passed!\n";
    }

    // Test default pools
    static void testDefaultPools() {
        std::cout << "Testing default pools...\n";

        MemoryPoolManager& manager = MemoryPoolManager::getInstance();

        // Default CPU pool should exist
        IMemoryPool* defaultCPUPool = manager.getCPUPool("default");
        assert(defaultCPUPool != nullptr);
        assert(defaultCPUPool->getMemoryType() == MemoryType::CPU);
        (void)defaultCPUPool;  // Suppress unused variable warning

        // Default GPU pool should exist
        IMemoryPool* defaultGPUPool = manager.getGPUPool("default_gpu");
        assert(defaultGPUPool != nullptr);
        assert(defaultGPUPool->getMemoryType() == MemoryType::GPU);
        (void)defaultGPUPool;  // Suppress unused variable warning

        // Should not be able to destroy default pools
        bool destroyedCPU = manager.destroyPool("default");
        assert(!destroyedCPU);
        (void)destroyedCPU;  // Suppress unused variable warning

        bool destroyedGPU = manager.destroyPool("default_gpu");
        assert(!destroyedGPU);
        (void)destroyedGPU;  // Suppress unused variable warning

        std::cout << "Default pools test passed!\n";
    }

    // Test reset all pools
    static void testResetAllPools() {
        std::cout << "Testing reset all pools...\n";

        MemoryPoolManager& manager = MemoryPoolManager::getInstance();

        // Create a test pool and allocate some memory
        PoolConfig config = PoolConfig::DefaultCPU();
        IMemoryPool* pool = manager.createCPUPool("reset_test_pool", config);

        void* ptr = pool->allocate(1024);
        assert(ptr != nullptr);

        // Check stats before reset
        const MemoryStats& statsBefore = pool->getStats();
        assert(statsBefore.getCurrentUsed() >= 1024);
        (void)statsBefore;  // Suppress unused variable warning

        // Reset all pools
        manager.resetAllPools();

        // Check stats after reset
        const MemoryStats& statsAfter = pool->getStats();
        assert(statsAfter.getCurrentUsed() == 0);
        (void)statsAfter;  // Suppress unused variable warning

        // Clean up
        pool->deallocate(ptr);
        manager.destroyPool("reset_test_pool");

        std::cout << "Reset all pools test passed!\n";
    }

    // Test get all stats
    static void testGetAllStats() {
        std::cout << "Testing get all stats...\n";

        MemoryPoolManager& manager = MemoryPoolManager::getInstance();

        // Get stats
        std::map<std::string, std::string> stats = manager.getAllStats();

        // Should have at least default pools
        assert(stats.find("default") != stats.end());
        assert(stats.find("default_gpu") != stats.end());

        // Stats strings should not be empty
        assert(!stats["default"].empty());
        assert(!stats["default_gpu"].empty());

        std::cout << "Get all stats test passed!\n";
    }

    // Test helper functions
    static void testHelperFunctions() {
        std::cout << "Testing helper functions...\n";

        // Test CPU allocation/deallocation
        void* ptr = allocate(512, "default");
        assert(ptr != nullptr);
        deallocate(ptr, "default");

        // Test GPU allocation/deallocation (if available)
        try {
            void* gpuPtr = allocateGPU(512, "default_gpu");
            if (gpuPtr != nullptr) {
                deallocateGPU(gpuPtr, "default_gpu");
            }
        } catch (const std::exception&) {
            // CUDA not available, skip
        }

        std::cout << "Helper functions test passed!\n";
    }

    // Test error cases
    static void testErrorCases() {
        std::cout << "Testing error cases...\n";

        MemoryPoolManager& manager = MemoryPoolManager::getInstance();

        // Try to get non-existent pool
        IMemoryPool* nonExistent = manager.getCPUPool("non_existent");
        assert(nonExistent == nullptr);
        (void)nonExistent;  // Suppress unused variable warning

        nonExistent = manager.getGPUPool("non_existent");
        assert(nonExistent == nullptr);

        // Try to destroy non-existent pool
        bool destroyed = manager.destroyPool("non_existent");
        assert(!destroyed);
        (void)destroyed;  // Suppress unused variable warning

        // Try to create pool with existing name
        PoolConfig config = PoolConfig::DefaultCPU();
        IMemoryPool* pool1 = manager.createCPUPool("duplicate_test", config);
        IMemoryPool* pool2 = manager.createCPUPool("duplicate_test", config);
        assert(pool1 == pool2);  // Should return existing pool
        (void)pool1;  // Suppress unused variable warning
        (void)pool2;  // Suppress unused variable warning

        manager.destroyPool("duplicate_test");

        std::cout << "Error cases test passed!\n";
    }

    // Run all tests
    static void runAllTests() {
        testCreateAndGetCPUPool();
        testCreateAndGetGPUPool();
        testDestroyPool();
        testDefaultPools();
        testResetAllPools();
        testGetAllStats();
        testHelperFunctions();
        testErrorCases();

        std::cout << "All MemoryPoolManager tests passed!\n";
    }
};

int main() {
    try {
        MemoryPoolManagerTest test;
        test.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "MemoryPoolManager test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}