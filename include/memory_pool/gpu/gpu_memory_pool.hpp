#ifndef MEMORY_POOL_GPU_MEMORY_POOL_HPP
#define MEMORY_POOL_GPU_MEMORY_POOL_HPP

#include "memory_pool/memory_pool.hpp"
#include "memory_pool/config.hpp"
#include "memory_pool/stats/memory_stats.hpp"
#include "cuda_utils.hpp"
#include <mutex>
#include <string>
#include <memory>

namespace memory_pool {

// Forward declarations
class ICudaAllocator;

// GPU Memory Pool implementation
class GPUMemoryPool : public IMemoryPool {
public:
    // Constructor and destructor
    GPUMemoryPool(const std::string& name, const PoolConfig& config);
    ~GPUMemoryPool() override;
    
    // IMemoryPool interface implementation
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void* allocate(size_t size, AllocFlags flags) override;
    void reset() override;
    const MemoryStats& getStats() const override;
    MemoryType getMemoryType() const override;
    std::string getName() const override;
    const PoolConfig& getConfig() const override;
    
    // GPU-specific methods
    void setDevice(int deviceId);
    int getDevice() const;
    void setStream(cudaStream_t stream);
    cudaStream_t getStream() const;
    
    // Memory transfer helpers
    void copyHostToDevice(void* dst, const void* src, size_t size);
    void copyDeviceToHost(void* dst, const void* src, size_t size);
    void copyDeviceToDevice(void* dst, const void* src, size_t size);
    
private:
    // Pool identification
    std::string name;
    PoolConfig config;
    
    // CUDA properties
    int deviceId;
    cudaStream_t stream;
    
    // Allocator
    std::unique_ptr<ICudaAllocator> allocator;
    
    // Thread safety
    std::mutex mutex;
    
    // Statistics
    MemoryStats stats;
    
    // Helper methods
    void initialize();
    void* allocateInternal(size_t size, AllocFlags flags);
    void ensureCorrectDevice() const;
};

// CUDA Allocator interface
class ICudaAllocator {
public:
    virtual ~ICudaAllocator() = default;
    
    virtual void* allocate(size_t size, AllocFlags flags) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void reset() = 0;
    virtual size_t getBlockSize(void* ptr) const = 0;
    virtual bool owns(void* ptr) const = 0;
    
    // CUDA-specific methods
    virtual void setDevice(int deviceId) = 0;
    virtual int getDevice() const = 0;
    virtual void setStream(cudaStream_t stream) = 0;
    virtual cudaStream_t getStream() const = 0;
};

} // namespace memory_pool

#endif // MEMORY_POOL_GPU_MEMORY_POOL_HPP