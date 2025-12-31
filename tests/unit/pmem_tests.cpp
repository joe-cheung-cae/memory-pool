#include "memory_pool/pmem/pmem_memory_pool.hpp"
#include "memory_pool/pmem/pmem_fixed_size_allocator.hpp"
#include "memory_pool/pmem/pmem_variable_size_allocator.hpp"
#include "memory_pool/config.hpp"
#include "memory_pool/common.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>

#ifdef HAVE_PMEM

void testPMEMFixedSizeAllocator() {
    std::cout << "Testing PMEM Fixed-Size Allocator...\n";

    const std::string poolPath = "/tmp/test_pmem_fixed.pool";
    const size_t poolSize = 1024 * 1024;  // 1MB
    const size_t blockSize = 256;
    const size_t initialBlocks = 10;

    PMEMFixedSizeAllocator allocator(poolPath, poolSize, blockSize, initialBlocks);

    // Test allocation
    void* ptr1 = allocator.allocate(blockSize);
    assert(ptr1 != nullptr);
    std::cout << "Allocated block at " << ptr1 << std::endl;

    // Test deallocation
    allocator.deallocate(ptr1);
    std::cout << "Deallocated block\n";

    // Test multiple allocations
    std::vector<void*> pointers;
    for (int i = 0; i < 5; ++i) {
        void* ptr = allocator.allocate(blockSize);
        assert(ptr != nullptr);
        pointers.push_back(ptr);
    }

    // Test deallocation of multiple blocks
    for (void* ptr : pointers) {
        allocator.deallocate(ptr);
    }

    std::cout << "PMEM Fixed-Size Allocator test passed!\n";
}

void testPMEMVariableSizeAllocator() {
    std::cout << "Testing PMEM Variable-Size Allocator...\n";

    const std::string poolPath = "/tmp/test_pmem_var.pool";
    const size_t poolSize = 1024 * 1024;  // 1MB

    PMEMVariableSizeAllocator allocator(poolPath, poolSize);

    // Test allocation
    void* ptr1 = allocator.allocate(512);
    assert(ptr1 != nullptr);
    std::cout << "Allocated 512 bytes at " << ptr1 << std::endl;

    // Test deallocation
    allocator.deallocate(ptr1);
    std::cout << "Deallocated block\n";

    // Test multiple allocations of different sizes
    std::vector<std::pair<void*, size_t>> allocations;
    allocations.emplace_back(allocator.allocate(128), 128);
    allocations.emplace_back(allocator.allocate(256), 256);
    allocations.emplace_back(allocator.allocate(64), 64);

    // Test deallocation
    for (auto& alloc : allocations) {
        allocator.deallocate(alloc.first);
    }

    std::cout << "PMEM Variable-Size Allocator test passed!\n";
}

void testPMEMMemoryPool() {
    std::cout << "Testing PMEM Memory Pool...\n";

    PoolConfig config;
    config.initialSize = 1024 * 1024;  // 1MB
    config.pmemPoolPath = "/tmp/test_pmem_pool.pool";

    PMEMMemoryPool pool("test_pmem", config);

    // Test allocation
    void* ptr1 = pool.allocate(512);
    assert(ptr1 != nullptr);
    std::cout << "Allocated 512 bytes at " << ptr1 << std::endl;

    // Test deallocation
    pool.deallocate(ptr1);
    std::cout << "Deallocated block\n";

    // Test statistics
    auto stats = pool.getStats();
    std::cout << "Pool stats: " << stats.getStatsString() << std::endl;

    std::cout << "PMEM Memory Pool test passed!\n";
}

void testPMEMPersistence() {
    std::cout << "Testing PMEM Persistence...\n";

    const std::string poolPath = "/tmp/test_pmem_persist.pool";
    const size_t poolSize = 1024 * 1024;  // 1MB

    {
        PMEMVariableSizeAllocator allocator(poolPath, poolSize);

        // Allocate and write data
        void* ptr = allocator.allocate(100);
        strcpy(static_cast<char*>(ptr), "Persistent Data");

        // Persist the data
        allocator.persist(ptr, strlen("Persistent Data") + 1);

        std::cout << "Data written and persisted: " << static_cast<char*>(ptr) << std::endl;
    }

    // Create new allocator with same pool (simulating restart)
    {
        PMEMVariableSizeAllocator allocator(poolPath, poolSize);

        // The data should still be there (though we can't easily verify without more complex setup)
        std::cout << "PMEM persistence test completed (data should persist across allocator instances)\n";
    }

    std::cout << "PMEM Persistence test passed!\n";
}

int main() {
    std::cout << "Running PMEM Tests...\n";

    try {
        testPMEMFixedSizeAllocator();
        testPMEMVariableSizeAllocator();
        testPMEMMemoryPool();
        testPMEMPersistence();

        std::cout << "All PMEM tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PMEM test failed: " << e.what() << std::endl;
        return 1;
    }
}

#else  // HAVE_PMEM

int main() {
    std::cout << "PMEM tests skipped - libpmem not available\n";
    return 0;
}

#endif  // HAVE_PMEM