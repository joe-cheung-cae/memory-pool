#ifndef MEMORY_POOL_MEMORY_POOL_HPP
#define MEMORY_POOL_MEMORY_POOL_HPP

#include "common.hpp"
#include "config.hpp"
#include "stats/memory_stats.hpp"
#include <string>
#include <map>
#include <memory>
#include <mutex>

namespace memory_pool {

// Memory Pool Interface
class IMemoryPool {
  public:
    virtual ~IMemoryPool() = default;

    // Core allocation functions
    virtual void* allocate(size_t size) = 0;
    virtual void  deallocate(void* ptr) = 0;

    // Allocation with flags
    virtual void* allocate(size_t size, AllocFlags flags) = 0;

    // Pool management
    virtual void               reset()          = 0;
    virtual const MemoryStats& getStats() const = 0;

    // Pool information
    virtual MemoryType        getMemoryType() const = 0;
    virtual std::string       getName() const       = 0;
    virtual const PoolConfig& getConfig() const     = 0;
};

// Memory Pool Manager
class MemoryPoolManager {
  public:
    // Singleton access
    static MemoryPoolManager& getInstance();

    // CPU pool management
    IMemoryPool* getCPUPool(const std::string& name);
    IMemoryPool* createCPUPool(const std::string& name, const PoolConfig& config);

    // GPU pool management
    IMemoryPool* getGPUPool(const std::string& name);
    IMemoryPool* createGPUPool(const std::string& name, const PoolConfig& config);

    // Pool operations
    bool destroyPool(const std::string& name);
    void resetAllPools();

    // Statistics
    std::map<std::string, std::string> getAllStats() const;

  private:
    MemoryPoolManager();
    ~MemoryPoolManager();

    // Prevent copying and assignment
    MemoryPoolManager(const MemoryPoolManager&)            = delete;
    MemoryPoolManager& operator=(const MemoryPoolManager&) = delete;

    // Pool storage
    std::map<std::string, std::unique_ptr<IMemoryPool>> pools;
    mutable std::mutex                                  poolsMutex;
};

// Helper functions for common operations
void* allocate(size_t size, const std::string& poolName = "default");
void  deallocate(void* ptr, const std::string& poolName = "default");
void* allocateGPU(size_t size, const std::string& poolName = "default_gpu");
void  deallocateGPU(void* ptr, const std::string& poolName = "default_gpu");

// Template functions for typed allocations
template <typename T>
T* allocate(size_t count = 1, const std::string& poolName = "default") {
    return static_cast<T*>(allocate(sizeof(T) * count, poolName));
}

template <typename T>
T* allocateGPU(size_t count = 1, const std::string& poolName = "default_gpu") {
    return static_cast<T*>(allocateGPU(sizeof(T) * count, poolName));
}

template <typename T>
void deallocate(T* ptr, const std::string& poolName = "default") {
    deallocate(static_cast<void*>(ptr), poolName);
}

template <typename T>
void deallocateGPU(T* ptr, const std::string& poolName = "default_gpu") {
    deallocateGPU(static_cast<void*>(ptr), poolName);
}

}  // namespace memory_pool

#endif  // MEMORY_POOL_MEMORY_POOL_HPP