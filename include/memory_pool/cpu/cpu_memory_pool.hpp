#ifndef MEMORY_POOL_CPU_MEMORY_POOL_HPP
#define MEMORY_POOL_CPU_MEMORY_POOL_HPP

#include "memory_pool/memory_pool.hpp"
#include "memory_pool/config.hpp"
#include "memory_pool/stats/memory_stats.hpp"
#include <mutex>
#include <string>
#include <memory>

namespace memory_pool {

// Forward declarations
class IAllocator;

// CPU Memory Pool implementation
class CPUMemoryPool : public IMemoryPool {
  public:
    // Constructor and destructor
    CPUMemoryPool(const std::string& name, const PoolConfig& config);
    ~CPUMemoryPool() override;

    // IMemoryPool interface implementation
    void*              allocate(size_t size) override;
    void               deallocate(void* ptr) override;
    void*              allocate(size_t size, AllocFlags flags) override;
    void               reset() override;
    const MemoryStats& getStats() const override;
    MemoryType         getMemoryType() const override;
    std::string        getName() const override;
    const PoolConfig&  getConfig() const override;

  private:
    // Pool identification
    std::string name;
    PoolConfig  config;

    // Allocator
    std::unique_ptr<IAllocator> allocator;

    // Thread safety
    std::mutex mutex;

    // Statistics
    MemoryStats stats;

    // Helper methods
    void  initialize();
    void* allocateInternal(size_t size, AllocFlags flags);
};

// Allocator interface
class IAllocator {
  public:
    virtual ~IAllocator() = default;

    virtual void*  allocate(size_t size)         = 0;
    virtual void   deallocate(void* ptr)         = 0;
    virtual void   reset()                       = 0;
    virtual size_t getBlockSize(void* ptr) const = 0;
    virtual bool   owns(void* ptr) const         = 0;
};

}  // namespace memory_pool

#endif  // MEMORY_POOL_CPU_MEMORY_POOL_HPP