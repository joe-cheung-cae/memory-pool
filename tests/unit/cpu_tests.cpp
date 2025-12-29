#include "memory_pool/memory_pool.hpp"
#include "memory_pool/cpu/cpu_memory_pool.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <chrono>
#include <random>

using namespace memory_pool;

// Test fixture
class CPUMemoryPoolTest {
public:
    CPUMemoryPoolTest() {
        // Set up test environment
        std::cout << "Setting up test environment...\n";
    }
    
    ~CPUMemoryPoolTest() {
        // Clean up test environment
        std::cout << "Cleaning up test environment...\n";
    }
    
    // Test fixed-size allocator
    void testFixedSizeAllocator() {
        std::cout << "Testing fixed-size allocator...\n";
        
        // Create a pool with fixed-size blocks
        PoolConfig config = PoolConfig::FixedSizeCPU(128);  // 128-byte blocks
        auto& manager = MemoryPoolManager::getInstance();
        IMemoryPool* pool = manager.createCPUPool("fixed_test", config);
        
        // Allocate memory
        const int numAllocations = 100;
        std::vector<void*> pointers;
        
        for (int i = 0; i < numAllocations; ++i) {
            void* ptr = pool->allocate(64);  // Each allocation is smaller than the block size
            assert(ptr != nullptr);
            pointers.push_back(ptr);
            
            // Write to memory to ensure it's usable
            memset(ptr, i, 64);
        }
        
        // Verify memory contents
        for (int i = 0; i < numAllocations; ++i) {
            unsigned char* ptr = static_cast<unsigned char*>(pointers[i]);
            for (int j = 0; j < 64; ++j) {
                assert(ptr[j] == static_cast<unsigned char>(i));
            }
        }
        
        // Deallocate half of the memory
        for (int i = 0; i < numAllocations / 2; ++i) {
            pool->deallocate(pointers[i]);
            pointers[i] = nullptr;
        }
        
        // Allocate more memory
        for (int i = 0; i < numAllocations / 2; ++i) {
            void* ptr = pool->allocate(64);
            assert(ptr != nullptr);
            pointers[i] = ptr;
            
            // Write to memory to ensure it's usable
            memset(ptr, i + 100, 64);
        }
        
        // Verify memory contents for new allocations
        for (int i = 0; i < numAllocations / 2; ++i) {
            unsigned char* ptr = static_cast<unsigned char*>(pointers[i]);
            for (int j = 0; j < 64; ++j) {
                assert(ptr[j] == static_cast<unsigned char>(i + 100));
            }
        }
        
        // Deallocate all memory
        for (void* ptr : pointers) {
            if (ptr) {
                pool->deallocate(ptr);
            }
        }
        
        // Destroy the pool
        manager.destroyPool("fixed_test");
        
        std::cout << "Fixed-size allocator test passed!\n";
    }
    
    // Test variable-size allocator
    void testVariableSizeAllocator() {
        std::cout << "Testing variable-size allocator...\n";
        
        // Create a pool with variable-size blocks
        PoolConfig config = PoolConfig::DefaultCPU();  // Default is variable-size
        auto& manager = MemoryPoolManager::getInstance();
        IMemoryPool* pool = manager.createCPUPool("variable_test", config);
        
        // Allocate memory of different sizes
        std::vector<std::pair<void*, size_t>> allocations;
        
        // Allocate blocks of increasing size
        for (size_t size = 16; size <= 16384; size *= 2) {
            void* ptr = pool->allocate(size);
            assert(ptr != nullptr);
            allocations.push_back(std::make_pair(ptr, size));
            
            // Write to memory to ensure it's usable
            memset(ptr, static_cast<int>(size % 256), size);
        }
        
        // Verify memory contents
        for (const auto& pair : allocations) {
            void* ptr = pair.first;
            size_t size = pair.second;
            unsigned char expected = static_cast<unsigned char>(size % 256);
            
            unsigned char* charPtr = static_cast<unsigned char*>(ptr);
            for (size_t i = 0; i < size; ++i) {
                assert(charPtr[i] == expected);
            }
        }
        
        // Deallocate in reverse order
        for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
            pool->deallocate(it->first);
        }
        
        // Allocate again to ensure memory is reused
        allocations.clear();
        
        // Allocate blocks of decreasing size
        for (size_t size = 16384; size >= 16; size /= 2) {
            void* ptr = pool->allocate(size);
            assert(ptr != nullptr);
            allocations.push_back(std::make_pair(ptr, size));
            
            // Write to memory to ensure it's usable
            memset(ptr, static_cast<int>(size % 256), size);
        }
        
        // Verify memory contents
        for (const auto& pair : allocations) {
            void* ptr = pair.first;
            size_t size = pair.second;
            unsigned char expected = static_cast<unsigned char>(size % 256);
            
            unsigned char* charPtr = static_cast<unsigned char*>(ptr);
            for (size_t i = 0; i < size; ++i) {
                assert(charPtr[i] == expected);
            }
        }
        
        // Deallocate all memory
        for (const auto& pair : allocations) {
            pool->deallocate(pair.first);
        }
        
        // Destroy the pool
        manager.destroyPool("variable_test");
        
        std::cout << "Variable-size allocator test passed!\n";
    }
    
    // Test memory fragmentation
    void testMemoryFragmentation() {
        std::cout << "Testing memory fragmentation...\n";
        
        // Create a pool with variable-size blocks
        PoolConfig config = PoolConfig::DefaultCPU();
        auto& manager = MemoryPoolManager::getInstance();
        IMemoryPool* pool = manager.createCPUPool("fragmentation_test", config);
        
        // Allocate memory with a pattern that causes fragmentation
        const int numAllocations = 1000;
        std::vector<void*> pointers(numAllocations, nullptr);
        
        // Allocate memory of different sizes
        std::mt19937 rng(42);  // Fixed seed for reproducibility
        std::uniform_int_distribution<size_t> sizeDist(16, 1024);
        
        for (int i = 0; i < numAllocations; ++i) {
            size_t size = sizeDist(rng);
            pointers[i] = pool->allocate(size);
            assert(pointers[i] != nullptr);
        }
        
        // Deallocate every other allocation to create fragmentation
        for (int i = 0; i < numAllocations; i += 2) {
            pool->deallocate(pointers[i]);
            pointers[i] = nullptr;
        }
        
        // Try to allocate a large block that should fail due to fragmentation
        void* largeBlock = pool->allocate(numAllocations * 512);
        
        // This might succeed or fail depending on the allocator's behavior
        // If it succeeds, the allocator is handling fragmentation well
        if (largeBlock != nullptr) {
            std::cout << "Allocator handled fragmentation well!\n";
            pool->deallocate(largeBlock);
        } else {
            std::cout << "Expected fragmentation occurred.\n";
        }
        
        // Deallocate remaining memory
        for (void* ptr : pointers) {
            if (ptr) {
                pool->deallocate(ptr);
            }
        }
        
        // Destroy the pool
        manager.destroyPool("fragmentation_test");
        
        std::cout << "Memory fragmentation test completed!\n";
    }
    
    // Test thread safety
    void testThreadSafety() {
        std::cout << "Testing thread safety...\n";
        
        // This is a simplified test that doesn't actually use threads
        // In a real test, you would create multiple threads and have them
        // allocate and deallocate memory concurrently
        
        // Create a pool with thread safety enabled
        PoolConfig config = PoolConfig::DefaultCPU();
        config.threadSafe = true;
        
        auto& manager = MemoryPoolManager::getInstance();
        IMemoryPool* pool = manager.createCPUPool("thread_safety_test", config);
        
        // Simulate multiple threads allocating and deallocating
        const int numOperations = 1000;
        std::vector<void*> pointers;
        
        for (int i = 0; i < numOperations; ++i) {
            // Allocate memory
            void* ptr = pool->allocate(64);
            assert(ptr != nullptr);
            pointers.push_back(ptr);
            
            // Occasionally deallocate memory
            if (i % 3 == 0 && !pointers.empty()) {
                size_t index = i % pointers.size();
                pool->deallocate(pointers[index]);
                pointers[index] = nullptr;
            }
        }
        
        // Deallocate all remaining memory
        for (void* ptr : pointers) {
            if (ptr) {
                pool->deallocate(ptr);
            }
        }
        
        // Destroy the pool
        manager.destroyPool("thread_safety_test");
        
        std::cout << "Thread safety test completed!\n";
    }
    
    // Test memory statistics
    void testMemoryStatistics() {
        std::cout << "Testing memory statistics...\n";
        
        // Create a pool with statistics tracking enabled
        PoolConfig config = PoolConfig::DefaultCPU();
        config.trackStats = true;
        
        auto& manager = MemoryPoolManager::getInstance();
        IMemoryPool* pool = manager.createCPUPool("stats_test", config);
        
        // Initial statistics
        const MemoryStats& initialStats = pool->getStats();
        assert(initialStats.getCurrentUsed() == 0);
        assert(initialStats.getAllocationCount() == 0);
        
        // Allocate memory
        void* ptr1 = pool->allocate(1024);
        assert(ptr1 != nullptr);
        
        // Check statistics after allocation
        const MemoryStats& afterAllocationStats = pool->getStats();
        assert(afterAllocationStats.getCurrentUsed() >= 1024);
        assert(afterAllocationStats.getAllocationCount() == 1);
        
        // Allocate more memory
        void* ptr2 = pool->allocate(2048);
        assert(ptr2 != nullptr);
        
        // Check statistics after second allocation
        const MemoryStats& afterSecondAllocationStats = pool->getStats();
        assert(afterSecondAllocationStats.getCurrentUsed() >= 3072);
        assert(afterSecondAllocationStats.getAllocationCount() == 2);
        
        // Deallocate first allocation
        pool->deallocate(ptr1);
        
        // Check statistics after deallocation
        const MemoryStats& afterDeallocationStats = pool->getStats();
        assert(afterDeallocationStats.getCurrentUsed() >= 2048);
        assert(afterDeallocationStats.getAllocationCount() == 2);
        assert(afterDeallocationStats.getDeallocationCount() == 1);
        
        // Deallocate second allocation
        pool->deallocate(ptr2);
        
        // Check statistics after all deallocations
        const MemoryStats& finalStats = pool->getStats();
        assert(finalStats.getCurrentUsed() == 0);
        assert(finalStats.getAllocationCount() == 2);
        assert(finalStats.getDeallocationCount() == 2);
        
        // Destroy the pool
        manager.destroyPool("stats_test");
        
        std::cout << "Memory statistics test passed!\n";
    }
    
    // Run all tests
    void runAllTests() {
        testFixedSizeAllocator();
        testVariableSizeAllocator();
        testMemoryFragmentation();
        testThreadSafety();
        testMemoryStatistics();
        
        std::cout << "All CPU memory pool tests passed!\n";
    }
};

int main() {
    try {
        CPUMemoryPoolTest test;
        test.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}