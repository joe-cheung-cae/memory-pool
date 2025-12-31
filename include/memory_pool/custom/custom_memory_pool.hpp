#ifndef MEMORY_POOL_CUSTOM_CUSTOM_MEMORY_POOL_HPP
#define MEMORY_POOL_CUSTOM_CUSTOM_MEMORY_POOL_HPP

#include "memory_pool/memory_pool.hpp"
#include "memory_pool/config.hpp"
#include "memory_pool/stats/memory_stats.hpp"
#include "memory_pool/custom/custom_allocator.hpp"
#include <mutex>
#include <string>
#include <memory>

namespace memory_pool {

/**
 * @brief Custom memory pool implementation for specialized hardware.
 *
 * This class provides a memory pool that uses custom allocators for
 * specialized hardware like RDMA, FPGA, or ASIC devices.
 */
class CustomMemoryPool : public IMemoryPool {
public:
    /**
     * @brief Constructs a custom memory pool with the given configuration.
     * @param name The name of the pool.
     * @param config The pool configuration.
     * @param allocator The custom allocator to use.
     */
    CustomMemoryPool(const std::string& name, const PoolConfig& config,
                     std::unique_ptr<ICustomAllocator> allocator);

    /**
     * @brief Destroys the custom memory pool.
     */
    ~CustomMemoryPool() override;

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
     * @return MemoryType::Custom.
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

    /**
     * @brief Registers memory for hardware access.
     * @param ptr Pointer to the memory to register.
     * @param size Size of the memory region.
     * @return Hardware-specific handle.
     */
    uint64_t registerMemory(void* ptr, size_t size);

    /**
     * @brief Unregisters previously registered memory.
     * @param handle The registration handle.
     */
    void unregisterMemory(uint64_t handle);

    /**
     * @brief Synchronizes memory operations with hardware.
     * @param ptr Pointer to the memory region.
     * @param size Size of the region to synchronize.
     */
    void synchronize(void* ptr, size_t size);

    /**
     * @brief Gets hardware-specific memory information.
     * @param ptr Pointer to allocated memory.
     * @return Hardware-specific information as key-value pairs.
     */
    std::unordered_map<std::string, std::string> getHardwareInfo(void* ptr) const;

private:
    // Pool identification
    std::string name;
    PoolConfig  config;

    // Custom allocator
    std::unique_ptr<ICustomAllocator> allocator;

    // Thread safety
    std::mutex mutex;

    // Statistics
    MemoryStats stats;

    // Track allocated sizes for deallocation
    std::unordered_map<void*, size_t> allocatedSizes;

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

#endif  // MEMORY_POOL_CUSTOM_CUSTOM_MEMORY_POOL_HPP