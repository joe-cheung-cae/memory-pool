#ifndef MEMORY_POOL_PMEM_MEMORY_POOL_HPP
#define MEMORY_POOL_PMEM_MEMORY_POOL_HPP

#include "memory_pool/memory_pool.hpp"
#include "memory_pool/config.hpp"
#include "memory_pool/stats/memory_stats.hpp"
#include "memory_pool/cpu/cpu_memory_pool.hpp"
#include <mutex>
#include <string>
#include <memory>

#ifdef HAVE_PMEM
#include "memory_pool/pmem/pmem_fixed_size_allocator.hpp"
#include "memory_pool/pmem/pmem_variable_size_allocator.hpp"
#endif

namespace memory_pool {

#ifdef HAVE_PMEM
/**
 * @brief PMEM memory pool implementation.
 *
 * This class provides a persistent memory pool that manages memory allocation
 * and deallocation using configurable allocators with persistence guarantees.
 */
class PMEMMemoryPool : public IMemoryPool {
#else
/**
 * @brief Stub PMEM memory pool implementation (PMEM not available).
 *
 * This is a stub implementation that throws exceptions when PMEM is not available.
 */
class PMEMMemoryPool : public IMemoryPool {
#endif
  public:
    /**
     * @brief Constructs a PMEM memory pool with the given configuration.
     * @param name The name of the pool.
     * @param config The pool configuration.
     */
    PMEMMemoryPool(const std::string& name, const PoolConfig& config);

    /**
     * @brief Destroys the PMEM memory pool.
     */
    ~PMEMMemoryPool() override;

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
     * @return MemoryType::PMEM.
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

    // PMEM-specific methods
    /**
     * @brief Persists data to persistent memory.
     * @param ptr Pointer to the data to persist.
     * @param size Size of the data to persist.
     */
    void persist(void* ptr, size_t size);

    /**
     * @brief Gets the path to the persistent memory pool file.
     * @return The pool file path.
     */
    std::string getPoolPath() const;

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

    // PMEM-specific data
    std::string poolPath;

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

}  // namespace memory_pool

#endif  // MEMORY_POOL_PMEM_MEMORY_POOL_HPP