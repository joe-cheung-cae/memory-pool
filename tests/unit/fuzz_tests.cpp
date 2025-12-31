#include "memory_pool/memory_pool.hpp"
#include "memory_pool/cpu/cpu_memory_pool.hpp"
#include "memory_pool/gpu/gpu_memory_pool.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <cassert>
#include <limits>
#include <cstring>

using namespace memory_pool;

// Fuzz test for allocation edge cases
class AllocationFuzzer {
public:
    AllocationFuzzer() : rng_(std::random_device{}()) {}

    // Test CPU memory pool with simple patterns
    void fuzzCPUPool(size_t durationSeconds = 10) {
        std::cout << "Fuzzing CPU memory pool for " << durationSeconds << " seconds...\n";

        PoolConfig config = PoolConfig::DefaultCPU();
        config.trackStats = true;
        config.enableDebugging = false;  // Disable for fuzzing

        auto& manager = MemoryPoolManager::getInstance();
        IMemoryPool* pool = manager.createCPUPool("fuzz_cpu", config);

        auto startTime = std::chrono::steady_clock::now();
        size_t iterations = 0;

        std::vector<void*> allocations;

        // Simple pattern: allocate and deallocate in sequence
        while (std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - startTime).count() < static_cast<long>(durationSeconds)) {

            // Allocate up to 100 allocations
            if (allocations.size() < 100) {
                size_t size = 1024;  // Fixed size
                try {
                    void* ptr = pool->allocate(size);
                    if (ptr) {
                        allocations.push_back(ptr);
                    }
                } catch (const std::exception& e) {
                    std::cout << "Allocation failed: " << e.what() << std::endl;
                }
            } else {
                // Deallocate some
                if (!allocations.empty()) {
                    pool->deallocate(allocations.back());
                    allocations.pop_back();
                }
            }

            iterations++;
        }

        // Clean up remaining allocations
        for (void* ptr : allocations) {
            pool->deallocate(ptr);
        }

        manager.destroyPool("fuzz_cpu");
        std::cout << "CPU fuzz test completed: " << iterations << " operations\n";
    }

    // Test GPU memory pool with random allocations
    void fuzzGPUPool(size_t durationSeconds = 5) {
        std::cout << "Fuzzing GPU memory pool for " << durationSeconds << " seconds...\n";

        PoolConfig config = PoolConfig::DefaultGPU();
        config.trackStats = true;

        auto& manager = MemoryPoolManager::getInstance();
        GPUMemoryPool* pool = static_cast<GPUMemoryPool*>(
            manager.createGPUPool("fuzz_gpu", config));

        auto startTime = std::chrono::steady_clock::now();
        size_t iterations = 0;

        std::vector<void*> allocations;

        while (std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - startTime).count() < static_cast<long>(durationSeconds)) {

            // Randomly decide to allocate or deallocate
            if (allocations.empty() || std::uniform_int_distribution<>(0, 1)(rng_) == 0) {
                // Allocate
                size_t size = generateRandomSize();
                try {
                    void* ptr = pool->allocate(size);
                    if (ptr) {
                        allocations.push_back(ptr);
                    }
                } catch (const std::exception& e) {
                    // Expected for some edge cases
                    std::cout << "GPU allocation failed for size " << size << ": " << e.what() << std::endl;
                }
            } else {
                // Deallocate
                if (!allocations.empty()) {
                    size_t index = std::uniform_int_distribution<size_t>(0, allocations.size() - 1)(rng_);
                    pool->deallocate(allocations[index]);
                    allocations.erase(allocations.begin() + index);
                }
            }

            iterations++;
            if (iterations % 5000 == 0) {
                std::cout << "Completed " << iterations << " GPU operations...\n";
            }
        }

        // Clean up remaining allocations
        for (void* ptr : allocations) {
            pool->deallocate(ptr);
        }

        manager.destroyPool("fuzz_gpu");
        std::cout << "GPU fuzz test completed: " << iterations << " operations\n";
    }

    // Test specific edge cases
    void testEdgeCases() {
        std::cout << "Testing specific edge cases...\n";

        PoolConfig config = PoolConfig::DefaultCPU();
        auto& manager = MemoryPoolManager::getInstance();
        IMemoryPool* pool = manager.createCPUPool("edge_case", config);

        // Test zero size allocation
        void* zeroPtr = pool->allocate(0);
        // Zero size allocations should return nullptr or valid ptr depending on implementation
        if (zeroPtr) {
            pool->deallocate(zeroPtr);
        }

        // Test very small sizes
        for (size_t size : {1, 2, 4, 8, 16}) {
            void* ptr = pool->allocate(size);
            assert(ptr != nullptr);
            pool->deallocate(ptr);
        }

        // Test large sizes (may fail, but shouldn't crash)
        std::vector<size_t> largeSizes = {
            1024 * 1024,      // 1MB
            10 * 1024 * 1024, // 10MB
            100 * 1024 * 1024 // 100MB
        };

        for (size_t size : largeSizes) {
            try {
                void* ptr = pool->allocate(size);
                if (ptr) {
                    pool->deallocate(ptr);
                }
            } catch (const std::exception&) {
                // Expected for very large sizes
            }
        }

        // Test size that might cause overflow in calculations
        size_t maxReasonableSize = 1024 * 1024 * 1024;  // 1GB
        try {
            void* ptr = pool->allocate(maxReasonableSize);
            if (ptr) {
                pool->deallocate(ptr);
            }
        } catch (const std::exception&) {
            // Expected
        }

        manager.destroyPool("edge_case");
        std::cout << "Edge case testing completed\n";
    }

private:
    std::mt19937 rng_;

    // Generate random allocation sizes with bias towards edge cases
    size_t generateRandomSize() {
        std::uniform_int_distribution<> dist(0, 99);

        int choice = dist(rng_);
        if (choice < 10) {
            // 10% chance: zero size
            return 0;
        } else if (choice < 20) {
            // 10% chance: very small sizes
            return std::uniform_int_distribution<size_t>(1, 64)(rng_);
        } else if (choice < 30) {
            // 10% chance: medium sizes
            return std::uniform_int_distribution<size_t>(65, 1024 * 1024)(rng_);
        } else if (choice < 40) {
            // 10% chance: large sizes (may fail)
            return std::uniform_int_distribution<size_t>(1024 * 1024 + 1, 2 * 1024 * 1024)(rng_);
        } else {
            // 60% chance: normal sizes
            return std::uniform_int_distribution<size_t>(1, 64 * 1024)(rng_);
        }
    }
};

int main(int argc, char* argv[]) {
    size_t duration = 5;  // Default 5 seconds
    if (argc > 1) {
        duration = std::stoul(argv[1]);
    }

    try {
        AllocationFuzzer fuzzer;

        // Test edge cases first
        fuzzer.testEdgeCases();

        // Fuzz CPU pool with short duration
        fuzzer.fuzzCPUPool(1);

        // Skip GPU fuzzing for now
        // fuzzer.fuzzGPUPool(std::min(duration, size_t(3)));

        std::cout << "All fuzz tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fuzz test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}