#include "memory_pool/memory_pool.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>

using namespace memory_pool;

/**
 * @brief Advanced usage examples demonstrating multi-threading,
 * performance optimization, and complex allocation patterns.
 */

// Example of multi-threaded memory pool usage
void multiThreadedExample() {
    std::cout << "\n=== Multi-Threaded Memory Pool Example ===\n";

    // Create a high-performance CPU pool
    auto& manager = MemoryPoolManager::getInstance();
    PoolConfig config = PoolConfig::HighPerformanceCPU();
    config.initialSize = 16 * 1024 * 1024;  // 16MB
    IMemoryPool* pool = manager.createCPUPool("mt_pool", config);

    const int numThreads = 4;
    const int allocationsPerThread = 1000;
    const size_t allocationSize = 1024;  // 1KB per allocation

    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> threadAllocations(numThreads);

    // Allocation phase
    auto allocateFunc = [&](int threadId) {
        auto& allocations = threadAllocations[threadId];
        for (int i = 0; i < allocationsPerThread; ++i) {
            void* ptr = pool->allocate(allocationSize);
            allocations.push_back(ptr);

            // Simulate some work
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    };

    std::cout << "Starting allocation phase with " << numThreads << " threads...\n";
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(allocateFunc, i);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "Allocation completed in " << duration.count() << " ms\n";
    std::cout << "Total allocations: " << numThreads * allocationsPerThread << "\n";

    // Print statistics
    const auto& stats = pool->getStats();
    std::cout << "Memory pool statistics:\n" << stats.getStatsString() << "\n";

    // Deallocation phase
    threads.clear();
    auto deallocateFunc = [&](int threadId) {
        auto& allocations = threadAllocations[threadId];
        for (void* ptr : allocations) {
            pool->deallocate(ptr);
        }
    };

    std::cout << "Starting deallocation phase...\n";
    startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(deallocateFunc, i);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "Deallocation completed in " << duration.count() << " ms\n";
    std::cout << "Final statistics:\n" << pool->getStats().getStatsString() << "\n";
}

// Example of performance optimization techniques
void performanceOptimizationExample() {
    std::cout << "\n=== Performance Optimization Example ===\n";

    auto& manager = MemoryPoolManager::getInstance();

    // Compare different pool configurations
    std::vector<std::pair<std::string, PoolConfig>> configs = {
        {"Fixed Size (256B)", PoolConfig::FixedSizeCPU(256)},
        {"Fixed Size (1KB)", PoolConfig::FixedSizeCPU(1024)},
        {"Variable Size", PoolConfig::DefaultCPU()},
        {"High Performance", PoolConfig::HighPerformanceCPU()}
    };

    const int numAllocations = 10000;
    const size_t baseSize = 512;

    for (const auto& [name, config] : configs) {
        IMemoryPool* pool = manager.createCPUPool("perf_test_" + name, config);

        std::vector<void*> allocations;
        allocations.reserve(numAllocations);

        // Warm up
        for (int i = 0; i < 100; ++i) {
            void* ptr = pool->allocate(baseSize);
            pool->deallocate(ptr);
        }

        // Benchmark allocation
        auto startTime = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < numAllocations; ++i) {
            size_t size = baseSize + (i % 100);  // Vary size slightly
            void* ptr = pool->allocate(size);
            allocations.push_back(ptr);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto allocDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        // Benchmark deallocation
        startTime = std::chrono::high_resolution_clock::now();

        for (void* ptr : allocations) {
            pool->deallocate(ptr);
        }

        endTime = std::chrono::high_resolution_clock::now();
        auto deallocDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        std::cout << name << ":\n";
        std::cout << "  Allocation time: " << allocDuration.count() << " μs\n";
        std::cout << "  Deallocation time: " << deallocDuration.count() << " μs\n";
        std::cout << "  Average alloc time: " << static_cast<double>(allocDuration.count()) / numAllocations << " μs\n";
        std::cout << "  Average dealloc time: " << static_cast<double>(deallocDuration.count()) / numAllocations << " μs\n";

        manager.destroyPool("perf_test_" + name);
    }
}

// Example of memory fragmentation analysis
void fragmentationAnalysisExample() {
    std::cout << "\n=== Memory Fragmentation Analysis Example ===\n";

    auto& manager = MemoryPoolManager::getInstance();
    PoolConfig config = PoolConfig::DefaultCPU();
    config.initialSize = 4 * 1024 * 1024;  // 4MB
    config.trackStats = true;

    IMemoryPool* pool = manager.createCPUPool("frag_analysis", config);

    // Simulate a pattern that causes fragmentation
    std::vector<void*> smallBlocks;
    std::vector<void*> largeBlocks;

    std::cout << "Phase 1: Allocate small blocks\n";
    for (int i = 0; i < 100; ++i) {
        void* ptr = pool->allocate(64);  // 64B blocks
        smallBlocks.push_back(ptr);
    }

    std::cout << "Phase 2: Allocate large blocks\n";
    for (int i = 0; i < 10; ++i) {
        void* ptr = pool->allocate(64 * 1024);  // 64KB blocks
        largeBlocks.push_back(ptr);
    }

    std::cout << "Phase 3: Deallocate every other small block (creating holes)\n";
    for (size_t i = 0; i < smallBlocks.size(); i += 2) {
        pool->deallocate(smallBlocks[i]);
    }

    // Print fragmentation statistics
    const auto& stats = pool->getStats();
    std::cout << "Fragmentation analysis:\n";
    std::cout << stats.getStatsString() << "\n";
    std::cout << "Fragmentation ratio: " << stats.getFragmentationRatio() << "\n";

    // Try to allocate a medium-sized block
    std::cout << "Phase 4: Try to allocate a 32KB block\n";
    void* mediumBlock = pool->allocate(32 * 1024);
    if (mediumBlock) {
        std::cout << "Successfully allocated 32KB block\n";
        pool->deallocate(mediumBlock);
    } else {
        std::cout << "Failed to allocate 32KB block due to fragmentation\n";
    }

    // Cleanup
    for (void* ptr : smallBlocks) {
        if (ptr) pool->deallocate(ptr);
    }
    for (void* ptr : largeBlocks) {
        pool->deallocate(ptr);
    }

    std::cout << "Final statistics:\n" << pool->getStats().getStatsString() << "\n";
}

// Example of custom allocation patterns
void customAllocationPatternExample() {
    std::cout << "\n=== Custom Allocation Pattern Example ===\n";

    auto& manager = MemoryPoolManager::getInstance();

    // Create multiple pools for different allocation patterns
    PoolConfig smallPoolConfig = PoolConfig::FixedSizeCPU(64);   // For small objects
    PoolConfig mediumPoolConfig = PoolConfig::FixedSizeCPU(1024); // For medium objects
    PoolConfig largePoolConfig = PoolConfig::DefaultCPU();        // For large objects

    IMemoryPool* smallPool = manager.createCPUPool("small_objects", smallPoolConfig);
    IMemoryPool* mediumPool = manager.createCPUPool("medium_objects", mediumPoolConfig);
    IMemoryPool* largePool = manager.createCPUPool("large_objects", largePoolConfig);

    // Simulate object allocation based on size
    struct TestObject {
        size_t size;
        void* data;
        std::string type;
    };

    std::vector<TestObject> objects;

    // Generate objects of different sizes
    for (int i = 0; i < 1000; ++i) {
        size_t size;
        IMemoryPool* targetPool;
        std::string type;

        if (i % 10 == 0) {
            // 10% large objects
            size = 10000 + (rand() % 90000);
            targetPool = largePool;
            type = "large";
        } else if (i % 3 == 0) {
            // 30% medium objects
            size = 512 + (rand() % 512);
            targetPool = mediumPool;
            type = "medium";
        } else {
            // 60% small objects
            size = 32 + (rand() % 32);
            targetPool = smallPool;
            type = "small";
        }

        void* data = targetPool->allocate(size);
        objects.push_back({size, data, type});
    }

    std::cout << "Allocated " << objects.size() << " objects across multiple pools\n";

    // Print statistics for each pool
    std::cout << "\nSmall objects pool:\n" << smallPool->getStats().getStatsString() << "\n";
    std::cout << "Medium objects pool:\n" << mediumPool->getStats().getStatsString() << "\n";
    std::cout << "Large objects pool:\n" << largePool->getStats().getStatsString() << "\n";

    // Cleanup
    for (const auto& obj : objects) {
        IMemoryPool* targetPool;
        if (obj.type == "small") targetPool = smallPool;
        else if (obj.type == "medium") targetPool = mediumPool;
        else targetPool = largePool;

        targetPool->deallocate(obj.data);
    }
}

int main() {
    std::cout << "Advanced Memory Pool Usage Examples\n";
    std::cout << "===================================\n";

    try {
        multiThreadedExample();
        performanceOptimizationExample();
        fragmentationAnalysisExample();
        customAllocationPatternExample();

        std::cout << "\nAll advanced examples completed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}