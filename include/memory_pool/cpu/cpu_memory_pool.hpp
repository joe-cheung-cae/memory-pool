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

/**
 * @brief CPU memory pool implementation.
 *
 * This class provides a CPU-based memory pool that manages memory allocation
 * and deallocation using configurable allocators.
 */
class CPUMemoryPool : public IMemoryPool {
  public:
    /**
     * @brief Constructs a CPU memory pool with the given configuration.
     * @param name The name of the pool.
     * @param config The pool configuration.
     */
    CPUMemoryPool(const std::string& name, const PoolConfig& config);

    /**
     * @brief Destroys the CPU memory pool.
     */
    ~CPUMemoryPool() override;

    // IMemoryPool interface implementation
    /**
     * @brief Allocates memory of the specified size.
     * @param size The size in bytes to allocate.
     * @return Pointer to the allocated memory.
     */
    void* allocate(size_t size) override;

    /**
     * @brief Deallocates previously allocated memory.
     * @param ptr Pointer to the memory to deallocate.
     */
    void deallocate(void* ptr) override;

    /**
     * @brief Allocates memory with additional flags.
     * @param size The size in bytes to allocate.
     * @param flags Allocation flags.
     * @return Pointer to the allocated memory.
     */
    void* allocate(size_t size, AllocFlags flags) override;

    /**
     * @brief Resets the memory pool, deallocating all memory.
     */
    void reset() override;

    /**
     * @brief Gets memory usage statistics.
     * @return Reference to the memory statistics.
     */
    const MemoryStats& getStats() const override;

    /**
     * @brief Gets the type of memory managed by this pool.
     * @return MemoryType::CPU.
     */
    MemoryType getMemoryType() const override;

    /**
     * @brief Gets the name of this memory pool.
     * @return The pool name.
     */
    std::string getName() const override;

    /**
     * @brief Gets the configuration used to create this pool.
     * @return Reference to the pool configuration.
     */
    const PoolConfig& getConfig() const override;

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
    /**
     * @brief Initializes the memory pool.
     */
    void initialize();

    /**
     * @brief Internal allocation method with flags.
     * @param size The size to allocate.
     * @param flags Allocation flags.
     * @return Pointer to allocated memory.
     */
    void* allocateInternal(size_t size, AllocFlags flags);
};

/**
 * @brief Interface for memory allocators.
 *
 * This abstract base class defines the interface that all allocator
 * implementations must provide.
 */
class IAllocator {
  public:
    virtual ~IAllocator() = default;

    /**
     * @brief Allocates memory of the specified size.
     * @param size The size in bytes to allocate.
     * @return Pointer to the allocated memory.
     */
    virtual void* allocate(size_t size) = 0;

    /**
     * @brief Deallocates previously allocated memory.
     * @param ptr Pointer to the memory to deallocate.
     */
    virtual void deallocate(void* ptr) = 0;

    /**
     * @brief Resets the allocator, deallocating all memory.
     */
    virtual void reset() = 0;

    /**
     * @brief Gets the block size of the allocated memory.
     * @param ptr Pointer to the allocated memory.
     * @return The block size in bytes.
     */
    virtual size_t getBlockSize(void* ptr) const = 0;

    /**
     * @brief Checks if the allocator owns the given pointer.
     * @param ptr Pointer to check.
     * @return True if the allocator owns the pointer.
     */
    virtual bool owns(void* ptr) const = 0;
};

}  // namespace memory_pool

#endif  // MEMORY_POOL_CPU_MEMORY_POOL_HPP