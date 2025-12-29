#include "memory_pool/memory_pool.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using namespace memory_pool;

// Helper function to print memory stats
void printStats(const std::string& poolName) {
    auto& manager = MemoryPoolManager::getInstance();
    IMemoryPool* pool = manager.getCPUPool(poolName);
    
    if (pool) {
        std::cout << "Stats for pool '" << poolName << "':\n";
        std::cout << pool->getStats().getStatsString() << std::endl;
    }
}

// Example of using the fixed-size allocator
void fixedSizeExample() {
    std::cout << "\n=== Fixed-Size Allocator Example ===\n";
    
    // Create a pool with fixed-size blocks
    auto& manager = MemoryPoolManager::getInstance();
    PoolConfig config = PoolConfig::FixedSizeCPU(128);  // 128-byte blocks
    IMemoryPool* pool = manager.createCPUPool("fixed_example", config);
    
    // Allocate some memory
    const int numAllocations = 100;
    std::vector<void*> pointers;
    
    std::cout << "Allocating " << numAllocations << " blocks of 64 bytes...\n";
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numAllocations; ++i) {
        void* ptr = pool->allocate(64);  // Each allocation is smaller than the block size
        pointers.push_back(ptr);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    std::cout << "Allocation time: " << duration.count() << " microseconds\n";
    printStats("fixed_example");
    
    // Deallocate half of the memory
    std::cout << "Deallocating half of the blocks...\n";
    
    startTime = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numAllocations / 2; ++i) {
        pool->deallocate(pointers[i]);
    }
    
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    std::cout << "Deallocation time: " << duration.count() << " microseconds\n";
    printStats("fixed_example");
    
    // Allocate more memory
    std::cout << "Allocating " << numAllocations / 2 << " more blocks...\n";
    
    startTime = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numAllocations / 2; ++i) {
        void* ptr = pool->allocate(64);
        pointers.push_back(ptr);
    }
    
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    std::cout << "Allocation time: " << duration.count() << " microseconds\n";
    printStats("fixed_example");
    
    // Deallocate all memory
    std::cout << "Deallocating all blocks...\n";
    
    startTime = std::chrono::high_resolution_clock::now();
    
    for (void* ptr : pointers) {
        if (ptr) {
            pool->deallocate(ptr);
        }
    }
    
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    std::cout << "Deallocation time: " << duration.count() << " microseconds\n";
    printStats("fixed_example");
}

// Example of using the variable-size allocator
void variableSizeExample() {
    std::cout << "\n=== Variable-Size Allocator Example ===\n";
    
    // Create a pool with variable-size blocks
    auto& manager = MemoryPoolManager::getInstance();
    PoolConfig config = PoolConfig::DefaultCPU();  // Default is variable-size
    IMemoryPool* pool = manager.createCPUPool("variable_example", config);
    
    // Allocate memory of different sizes
    std::vector<std::pair<void*, size_t>> allocations;
    
    std::cout << "Allocating blocks of various sizes...\n";
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Allocate blocks of increasing size
    for (size_t size = 16; size <= 16384; size *= 2) {
        void* ptr = pool->allocate(size);
        allocations.push_back(std::make_pair(ptr, size));
        std::cout << "Allocated " << size << " bytes at " << ptr << std::endl;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    std::cout << "Allocation time: " << duration.count() << " microseconds\n";
    printStats("variable_example");
    
    // Deallocate in reverse order
    std::cout << "Deallocating in reverse order...\n";
    
    startTime = std::chrono::high_resolution_clock::now();
    
    for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
        std::cout << "Deallocating " << it->second << " bytes from " << it->first << std::endl;
        pool->deallocate(it->first);
    }
    
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    std::cout << "Deallocation time: " << duration.count() << " microseconds\n";
    printStats("variable_example");
}

// Example of using the helper functions
void helperFunctionsExample() {
    std::cout << "\n=== Helper Functions Example ===\n";
    
    // Allocate memory using the helper functions
    std::cout << "Allocating memory using helper functions...\n";
    
    void* ptr1 = allocate(1024);  // Uses the default CPU pool
    std::cout << "Allocated 1024 bytes at " << ptr1 << std::endl;
    
    int* intArray = allocate<int>(100);  // Allocate an array of 100 integers
    std::cout << "Allocated 100 integers at " << intArray << std::endl;
    
    // Use the allocated memory
    for (int i = 0; i < 100; ++i) {
        intArray[i] = i;
    }
    
    std::cout << "First few values: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << intArray[i] << " ";
    }
    std::cout << std::endl;
    
    // Deallocate the memory
    deallocate(ptr1);
    deallocate(intArray);
    
    printStats("default");
}

int main() {
    std::cout << "Memory Pool Management System Example\n";
    std::cout << "====================================\n";
    
    try {
        fixedSizeExample();
        variableSizeExample();
        helperFunctionsExample();
        
        std::cout << "\nAll examples completed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}