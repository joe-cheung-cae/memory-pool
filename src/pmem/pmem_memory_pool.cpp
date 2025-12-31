#include "memory_pool/pmem/pmem_memory_pool.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

namespace memory_pool {

#ifdef HAVE_PMEM

PMEMMemoryPool::PMEMMemoryPool(const std::string& name, const PoolConfig& config)
    : name(name), config(config), poolPath(config.pmemPoolPath) {
    if (poolPath.empty()) {
        // Generate default pool path
        poolPath = "/tmp/memory_pool_" + name + ".pmem";
    }

    // Ensure the directory exists (simple version without filesystem)
    size_t lastSlash = poolPath.find_last_of('/');
    if (lastSlash != std::string::npos) {
        std::string dir = poolPath.substr(0, lastSlash);
        mkdir(dir.c_str(), 0755);  // Create directory if it doesn't exist
    }

    initialize();
}

PMEMMemoryPool::~PMEMMemoryPool() = default;

#endif  // HAVE_PMEM

void* PMEMMemoryPool::allocate(size_t size) {
#ifdef HAVE_PMEM
    return allocateInternal(size, AllocFlags::None);
#else
    throw PMEMException("PMEM support not available");
#endif
}

void PMEMMemoryPool::deallocate(void* ptr) {
#ifdef HAVE_PMEM
    if (!ptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);

    try {
        size_t blockSize = allocator->getBlockSize(ptr);
        allocator->deallocate(ptr);
        stats.recordDeallocation(blockSize);  // Record deallocation
    } catch (const std::exception& e) {
        throw MemoryPoolException(std::string("PMEM deallocation failed: ") + e.what());
    }
#else
    throw PMEMException("PMEM support not available");
#endif
}

void* PMEMMemoryPool::allocate(size_t size, AllocFlags flags) {
#ifdef HAVE_PMEM
    return allocateInternal(size, flags);
#else
    throw PMEMException("PMEM support not available");
#endif
}

void PMEMMemoryPool::reset() {
#ifdef HAVE_PMEM
    std::lock_guard<std::mutex> lock(mutex);

    try {
        allocator->reset();
        stats.reset();
    } catch (const std::exception& e) {
        throw MemoryPoolException(std::string("PMEM pool reset failed: ") + e.what());
    }
#else
    throw PMEMException("PMEM support not available");
#endif
}

const MemoryStats& PMEMMemoryPool::getStats() const {
#ifdef HAVE_PMEM
    return stats;
#else
    throw PMEMException("PMEM support not available");
#endif
}

MemoryType PMEMMemoryPool::getMemoryType() const {
    return MemoryType::PMEM;
}

std::string PMEMMemoryPool::getName() const {
    return name;
}

const PoolConfig& PMEMMemoryPool::getConfig() const {
    return config;
}

void PMEMMemoryPool::persist(void* ptr, size_t size) {
#ifdef HAVE_PMEM
    if (auto pmemAllocator = dynamic_cast<PMEMFixedSizeAllocator*>(allocator.get())) {
        pmemAllocator->persist(ptr, size);
    } else if (auto pmemVarAllocator = dynamic_cast<PMEMVariableSizeAllocator*>(allocator.get())) {
        pmemVarAllocator->persist(ptr, size);
    }
#else
    throw PMEMException("PMEM support not available");
#endif
}

std::string PMEMMemoryPool::getPoolPath() const {
    return poolPath;
}

#ifdef HAVE_PMEM

void PMEMMemoryPool::initialize() {
    try {
        if (config.allocatorType == AllocatorType::FixedSize) {
            allocator = std::make_unique<PMEMFixedSizeAllocator>(
                poolPath, config.initialSize, config.blockSize,
                config.initialSize / config.blockSize, config.alignment,
                config.syncType == SyncType::LockFree);
        } else {
            allocator = std::make_unique<PMEMVariableSizeAllocator>(
                poolPath, config.initialSize, config.alignment,
                config.syncType == SyncType::LockFree);
        }

        std::cout << "Initialized PMEM memory pool '" << name << "' with "
                  << config.initialSize << " bytes at " << poolPath << std::endl;

    } catch (const std::exception& e) {
        throw MemoryPoolException(std::string("Failed to initialize PMEM pool: ") + e.what());
    }
}

void* PMEMMemoryPool::allocateInternal(size_t size, AllocFlags flags) {
    std::lock_guard<std::mutex> lock(mutex);

    try {
        void* ptr = allocator->allocate(size);
        stats.recordAllocation(size);  // Record allocation

        // If persistence is requested, persist immediately
        if (has_flag(flags, AllocFlags::Persist)) {
            persist(ptr, size);
        }

        return ptr;
    } catch (const std::exception& e) {
        throw MemoryPoolException(std::string("PMEM allocation failed: ") + e.what());
    }
}

#endif  // HAVE_PMEM

}  // namespace memory_pool