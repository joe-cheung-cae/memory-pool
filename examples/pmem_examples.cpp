#include "memory_pool/memory_pool.hpp"
#include <iostream>
#include <cstring>
#include <vector>

#ifdef HAVE_PMEM

void pmemBasicExample() {
    std::cout << "\n=== PMEM Basic Usage Example ===\n";

    // Create a PMEM memory pool
    auto& manager = MemoryPoolManager::getInstance();

    PoolConfig pmemConfig;
    pmemConfig.initialSize = 2 * 1024 * 1024;  // 2MB
    pmemConfig.pmemPoolPath = "/tmp/pmem_example.pool";

    auto* pool = manager.createPMEMPool("pmem_example", pmemConfig);

    // Allocate persistent memory
    void* data = pool->allocate(1024);
    std::cout << "Allocated 1024 bytes of persistent memory at " << data << std::endl;

    // Write data that will persist
    strcpy(static_cast<char*>(data), "This data persists across program restarts!");
    std::cout << "Data written: " << static_cast<char*>(data) << std::endl;

    // Persist the data explicitly
    if (auto pmemPool = dynamic_cast<PMEMMemoryPool*>(pool)) {
        pmemPool->persist(data, strlen(static_cast<char*>(data)) + 1);
        std::cout << "Data persisted to PMEM\n";
    }

    // Use the memory...
    std::cout << "Data in memory: " << static_cast<char*>(data) << std::endl;

    // Deallocate
    pool->deallocate(data);
    std::cout << "Memory deallocated\n";

    // Get statistics
    auto stats = pool->getStats();
    std::cout << "PMEM Pool Statistics:\n" << stats.getStatsString() << std::endl;
}

void pmemPersistenceExample() {
    std::cout << "\n=== PMEM Persistence Example ===\n";

    const std::string poolName = "persistent_data";
    const std::string poolPath = "/tmp/persistent_data.pool";

    auto& manager = MemoryPoolManager::getInstance();

    // First run - create and populate data
    {
        PoolConfig config;
        config.initialSize = 1024 * 1024;  // 1MB
        config.pmemPoolPath = poolPath;

        auto* pool = manager.createPMEMPool(poolName, config);

        // Allocate and initialize persistent data structures
        struct PersistentData {
            int id;
            char name[50];
            double value;
        };

        std::vector<PersistentData*> records;
        for (int i = 0; i < 5; ++i) {
            auto* record = static_cast<PersistentData*>(pool->allocate(sizeof(PersistentData)));
            record->id = i + 1;
            sprintf(record->name, "Record_%d", i + 1);
            record->value = 3.14159 * (i + 1);

            records.push_back(record);
        }

        std::cout << "Created persistent records:\n";
        for (const auto* record : records) {
            std::cout << "  ID: " << record->id
                     << ", Name: " << record->name
                     << ", Value: " << record->value << std::endl;
        }

        // Persist all data
        for (auto* record : records) {
            if (auto pmemPool = dynamic_cast<PMEMMemoryPool*>(pool)) {
                pmemPool->persist(record, sizeof(PersistentData));
            }
        }

        std::cout << "All data persisted. Simulating program restart...\n";
    }

    // Second run - access persisted data (in a real scenario, this would be a separate program run)
    {
        PoolConfig config;
        config.initialSize = 1024 * 1024;  // 1MB
        config.pmemPoolPath = poolPath;

        auto* pool = manager.createPMEMPool(poolName + "_reloaded", config);

        std::cout << "Reloaded persistent pool. Data should still be available.\n";
        std::cout << "(Note: In this example, the data layout would need to be managed more carefully)\n";

        // Get pool statistics
        auto stats = pool->getStats();
        std::cout << "Reloaded Pool Statistics:\n" << stats.getStatsString() << std::endl;
    }
}

int main() {
    std::cout << "PMEM Examples\n";
    std::cout << "=============\n";

    try {
        pmemBasicExample();
        pmemPersistenceExample();

        std::cout << "\nAll PMEM examples completed successfully!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "PMEM example failed: " << e.what() << std::endl;
        return 1;
    }
}

#else  // HAVE_PMEM

int main() {
    std::cout << "PMEM examples skipped - libpmem not available\n";
    std::cout << "To run PMEM examples, install PMDK (Persistent Memory Development Kit)\n";
    return 0;
}

#endif  // HAVE_PMEM