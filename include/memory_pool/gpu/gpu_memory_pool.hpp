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

/**
 * @brief GPU memory pool implementation using CUDA.
 *
 * This class provides memory management for GPU devices, implementing the IMemoryPool
 * interface with CUDA-specific optimizations and features.
 */
class GPUMemoryPool : public IMemoryPool {
  public:
    /**
     * @brief Constructor for GPU memory pool.
     * @param name The name of the memory pool.
     * @param config Configuration parameters for the pool.
     */
    GPUMemoryPool(const std::string& name, const PoolConfig& config);

    /** @brief Destructor */
    ~GPUMemoryPool() override;

    // IMemoryPool interface implementation
    void*              allocate(size_t size) override;
    void               deallocate(void* ptr) override;
    void*              allocate(size_t size, AllocFlags flags) override;
    void               reset() override;
    const MemoryStats& getStats() const override;
    MemoryType         getMemoryType() const override;
    std::string        getName() const override;
    const PoolConfig&  getConfig() const override;

    /**
     * @brief Sets the CUDA device for this pool.
     * @param deviceId The CUDA device ID.
     */
    void setDevice(int deviceId);

    /** @brief Gets the current CUDA device ID. */
    int getDevice() const;

    /**
     * @brief Sets the CUDA stream for asynchronous operations.
     * @param stream The CUDA stream to use.
     */
    void setStream(cudaStream_t stream);

    /** @brief Gets the current CUDA stream. */
    cudaStream_t getStream() const;

    /**
     * @brief Copies data from host memory to device memory.
     * @param dst Destination device pointer.
     * @param src Source host pointer.
     * @param size Number of bytes to copy.
     */
    void copyHostToDevice(void* dst, const void* src, size_t size);

    /**
     * @brief Copies data from device memory to host memory.
     * @param dst Destination host pointer.
     * @param src Source device pointer.
     * @param size Number of bytes to copy.
     */
    void copyDeviceToHost(void* dst, const void* src, size_t size);

    /**
     * @brief Copies data from device memory to device memory.
     * @param dst Destination device pointer.
     * @param src Source device pointer.
     * @param size Number of bytes to copy.
     */
    void copyDeviceToDevice(void* dst, const void* src, size_t size);

  private:
    // Pool identification
    std::string name;
    PoolConfig  config;

    // CUDA properties
    int          deviceId;
    cudaStream_t stream;

    // Allocator
    std::unique_ptr<ICudaAllocator> allocator;

    // Thread safety
    std::mutex mutex;

    // Statistics
    MemoryStats stats;

    // Helper methods
    void  initialize();
    void* allocateInternal(size_t size, AllocFlags flags);
    void  ensureCorrectDevice() const;
};

/**
 * @brief Interface for CUDA memory allocators.
 *
 * This abstract interface defines the contract for CUDA memory allocators,
 * providing both standard allocation methods and CUDA-specific functionality.
 */
class ICudaAllocator {
  public:
    virtual ~ICudaAllocator() = default;

    /**
     * @brief Allocates CUDA memory with specified flags.
     * @param size Size of memory to allocate in bytes.
     * @param flags Allocation flags (e.g., pinned memory).
     * @return Pointer to allocated memory, or nullptr on failure.
     */
    virtual void* allocate(size_t size, AllocFlags flags) = 0;

    /**
     * @brief Deallocates previously allocated CUDA memory.
     * @param ptr Pointer to memory to deallocate.
     */
    virtual void deallocate(void* ptr) = 0;

    /** @brief Resets the allocator, deallocating all memory. */
    virtual void reset() = 0;

    /**
     * @brief Gets the block size for a given allocation.
     * @param ptr Pointer to the allocated memory.
     * @return Size of the memory block.
     */
    virtual size_t getBlockSize(void* ptr) const = 0;

    /**
     * @brief Checks if the allocator owns a given pointer.
     * @param ptr Pointer to check.
     * @return True if the allocator owns this pointer.
     */
    virtual bool owns(void* ptr) const = 0;

    // CUDA-specific methods
    /**
     * @brief Sets the CUDA device for this allocator.
     * @param deviceId The CUDA device ID.
     */
    virtual void setDevice(int deviceId) = 0;

    /** @brief Gets the current CUDA device ID. */
    virtual int getDevice() const = 0;

    /**
     * @brief Sets the CUDA stream for asynchronous operations.
     * @param stream The CUDA stream to use.
     */
    virtual void setStream(cudaStream_t stream) = 0;

    /** @brief Gets the current CUDA stream. */
    virtual cudaStream_t getStream() const = 0;
};

}  // namespace memory_pool

#endif  // MEMORY_POOL_GPU_MEMORY_POOL_HPP