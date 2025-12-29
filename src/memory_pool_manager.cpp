#include "memory_pool/memory_pool.hpp"
#include "memory_pool/cpu/cpu_memory_pool.hpp"
#include "memory_pool/gpu/gpu_memory_pool.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <stdexcept>

namespace memory_pool {

// Singleton instance
MemoryPoolManager& MemoryPoolManager::getInstance() {
    static MemoryPoolManager instance;
    return instance;
}

MemoryPoolManager::MemoryPoolManager() {
    // Create default pools
    createCPUPool("default", PoolConfig::DefaultCPU());
    createGPUPool("default_gpu", PoolConfig::DefaultGPU());
}

MemoryPoolManager::~MemoryPoolManager() {
    // Pools will be automatically destroyed by unique_ptr
}

IMemoryPool* MemoryPoolManager::getCPUPool(const std::string& name) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    auto it = pools.find(name);
    if (it != pools.end() && it->second->getMemoryType() == MemoryType::CPU) {
        return it->second.get();
    }

    reportError(ErrorSeverity::Warning, "MemoryPoolManager: CPU pool '" + name + "' not found");
    return nullptr;
}

IMemoryPool* MemoryPoolManager::createCPUPool(const std::string& name, const PoolConfig& config) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    // Check if a pool with this name already exists
    if (pools.find(name) != pools.end()) {
        reportError(ErrorSeverity::Warning, "MemoryPoolManager: Pool '" + name + "' already exists");
        return pools[name].get();
    }

    // Create a new CPU pool
    auto         pool    = std::make_unique<CPUMemoryPool>(name, config);
    IMemoryPool* poolPtr = pool.get();

    // Add the pool to the map
    pools[name] = std::move(pool);

    return poolPtr;
}

IMemoryPool* MemoryPoolManager::getGPUPool(const std::string& name) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    auto it = pools.find(name);
    if (it != pools.end() && it->second->getMemoryType() == MemoryType::GPU) {
        return it->second.get();
    }

    reportError(ErrorSeverity::Warning, "MemoryPoolManager: GPU pool '" + name + "' not found");
    return nullptr;
}

IMemoryPool* MemoryPoolManager::createGPUPool(const std::string& name, const PoolConfig& config) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    // Check if a pool with this name already exists
    if (pools.find(name) != pools.end()) {
        reportError(ErrorSeverity::Warning, "MemoryPoolManager: Pool '" + name + "' already exists");
        return pools[name].get();
    }

    // Create a new GPU pool
    auto         pool    = std::make_unique<GPUMemoryPool>(name, config);
    IMemoryPool* poolPtr = pool.get();

    // Add the pool to the map
    pools[name] = std::move(pool);

    return poolPtr;
}

bool MemoryPoolManager::destroyPool(const std::string& name) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    auto it = pools.find(name);
    if (it == pools.end()) {
        reportError(ErrorSeverity::Warning, "MemoryPoolManager: Cannot destroy pool '" + name + "', not found");
        return false;
    }

    // Don't allow destroying default pools
    if (name == "default" || name == "default_gpu") {
        reportError(ErrorSeverity::Warning, "MemoryPoolManager: Cannot destroy default pool '" + name + "'");
        return false;
    }

    // Remove the pool from the map
    pools.erase(it);
    return true;
}

void MemoryPoolManager::resetAllPools() {
    std::lock_guard<std::mutex> lock(poolsMutex);

    for (auto& pair : pools) {
        pair.second->reset();
    }
}

std::map<std::string, std::string> MemoryPoolManager::getAllStats() const {
    std::lock_guard<std::mutex> lock(poolsMutex);

    // Instead of returning a map of MemoryStats, return a map of strings
    std::map<std::string, std::string> stats;
    for (const auto& pair : pools) {
        // Get a reference to the stats
        const MemoryStats& poolStats = pair.second->getStats();
        // Convert to string representation
        stats[pair.first] = poolStats.getStatsString();
    }

    return stats;
}

// Helper functions for common operations
void* allocate(size_t size, const std::string& poolName) {
    IMemoryPool* pool = MemoryPoolManager::getInstance().getCPUPool(poolName);
    if (pool == nullptr) {
        throw InvalidOperationException("Pool '" + poolName + "' not found");
    }

    return pool->allocate(size);
}

void deallocate(void* ptr, const std::string& poolName) {
    if (ptr == nullptr) {
        return;
    }

    IMemoryPool* pool = MemoryPoolManager::getInstance().getCPUPool(poolName);
    if (pool == nullptr) {
        throw InvalidOperationException("Pool '" + poolName + "' not found");
    }

    pool->deallocate(ptr);
}

void* allocateGPU(size_t size, const std::string& poolName) {
    IMemoryPool* pool = MemoryPoolManager::getInstance().getGPUPool(poolName);
    if (pool == nullptr) {
        throw InvalidOperationException("GPU pool '" + poolName + "' not found");
    }

    return pool->allocate(size);
}

void deallocateGPU(void* ptr, const std::string& poolName) {
    if (ptr == nullptr) {
        return;
    }

    IMemoryPool* pool = MemoryPoolManager::getInstance().getGPUPool(poolName);
    if (pool == nullptr) {
        throw InvalidOperationException("GPU pool '" + poolName + "' not found");
    }

    pool->deallocate(ptr);
}

}  // namespace memory_pool