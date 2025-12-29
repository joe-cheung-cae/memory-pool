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

/**
 * @brief Abstract interface for memory pool implementations.
 *
 * This interface defines the core functionality that all memory pool
 * implementations must provide, including allocation, deallocation,
 * and statistics gathering.
 */
class IMemoryPool {
  public:
    virtual ~IMemoryPool() = default;

    /**
     * @brief Allocates memory of the specified size.
     * @param size The size in bytes to allocate.
     * @return Pointer to the allocated memory, or nullptr on failure.
     */
    virtual void* allocate(size_t size) = 0;

    /**
     * @brief Deallocates previously allocated memory.
     * @param ptr Pointer to the memory to deallocate.
     */
    virtual void deallocate(void* ptr) = 0;

    /**
     * @brief Allocates memory with additional flags.
     * @param size The size in bytes to allocate.
     * @param flags Allocation flags (e.g., pinned memory for GPU).
     * @return Pointer to the allocated memory, or nullptr on failure.
     */
    virtual void* allocate(size_t size, AllocFlags flags) = 0;

    /**
     * @brief Resets the memory pool, deallocating all memory.
     */
    virtual void reset() = 0;

    /**
     * @brief Gets memory usage statistics.
     * @return Reference to the memory statistics object.
     */
    virtual const MemoryStats& getStats() const = 0;

    /**
     * @brief Gets the type of memory managed by this pool.
     * @return The memory type (CPU or GPU).
     */
    virtual MemoryType getMemoryType() const = 0;

    /**
     * @brief Gets the name of this memory pool.
     * @return The pool name as a string.
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Gets the configuration used to create this pool.
     * @return Reference to the pool configuration.
     */
    virtual const PoolConfig& getConfig() const = 0;
};

/**
 * @brief Singleton manager for memory pools.
 *
 * This class provides centralized management of memory pools, allowing
 * creation, retrieval, and destruction of both CPU and GPU memory pools.
 * It follows the singleton pattern to ensure only one instance exists.
 */
class MemoryPoolManager {
  public:
    /**
     * @brief Gets the singleton instance of the memory pool manager.
     * @return Reference to the memory pool manager instance.
     */
    static MemoryPoolManager& getInstance();

    /**
     * @brief Gets an existing CPU memory pool by name.
     * @param name The name of the pool to retrieve.
     * @return Pointer to the memory pool, or nullptr if not found.
     */
    IMemoryPool* getCPUPool(const std::string& name);

    /**
     * @brief Creates a new CPU memory pool with the specified configuration.
     * @param name The name for the new pool.
     * @param config The configuration for the pool.
     * @return Pointer to the created memory pool.
     */
    IMemoryPool* createCPUPool(const std::string& name, const PoolConfig& config);

    /**
     * @brief Gets an existing GPU memory pool by name.
     * @param name The name of the pool to retrieve.
     * @return Pointer to the memory pool, or nullptr if not found.
     */
    IMemoryPool* getGPUPool(const std::string& name);

    /**
     * @brief Creates a new GPU memory pool with the specified configuration.
     * @param name The name for the new pool.
     * @param config The configuration for the pool.
     * @return Pointer to the created memory pool.
     */
    IMemoryPool* createGPUPool(const std::string& name, const PoolConfig& config);

    /**
     * @brief Destroys a memory pool by name.
     * @param name The name of the pool to destroy.
     * @return True if the pool was successfully destroyed, false otherwise.
     */
    bool destroyPool(const std::string& name);

    /**
     * @brief Resets all managed memory pools.
     */
    void resetAllPools();

    /**
     * @brief Gets statistics for all managed pools.
     * @return Map of pool names to their statistics strings.
     */
    std::map<std::string, std::string> getAllStats() const;

    // Prevent copying and assignment
    MemoryPoolManager(const MemoryPoolManager&)            = delete;
    MemoryPoolManager& operator=(const MemoryPoolManager&) = delete;

  private:
    MemoryPoolManager();
    ~MemoryPoolManager();

    // Pool storage
    std::map<std::string, std::unique_ptr<IMemoryPool>> pools;
    mutable std::mutex                                  poolsMutex;
};

/**
 * @brief Allocates memory from the default CPU pool.
 * @param size The size in bytes to allocate.
 * @param poolName The name of the pool to use (default: "default").
 * @return Pointer to the allocated memory.
 */
void* allocate(size_t size, const std::string& poolName = "default");

/**
 * @brief Deallocates memory from the default CPU pool.
 * @param ptr Pointer to the memory to deallocate.
 * @param poolName The name of the pool to use (default: "default").
 */
void deallocate(void* ptr, const std::string& poolName = "default");

/**
 * @brief Allocates GPU memory from the default GPU pool.
 * @param size The size in bytes to allocate.
 * @param poolName The name of the pool to use (default: "default_gpu").
 * @return Pointer to the allocated GPU memory.
 */
void* allocateGPU(size_t size, const std::string& poolName = "default_gpu");

/**
 * @brief Deallocates GPU memory from the default GPU pool.
 * @param ptr Pointer to the GPU memory to deallocate.
 * @param poolName The name of the pool to use (default: "default_gpu").
 */
void deallocateGPU(void* ptr, const std::string& poolName = "default_gpu");

/**
 * @brief Template function for typed CPU memory allocation.
 * @tparam T The type of objects to allocate.
 * @param count The number of objects to allocate (default: 1).
 * @param poolName The name of the pool to use (default: "default").
 * @return Pointer to the allocated typed memory.
 */
template <typename T>
T* allocate(size_t count = 1, const std::string& poolName = "default") {
    return static_cast<T*>(allocate(sizeof(T) * count, poolName));
}

/**
 * @brief Template function for typed GPU memory allocation.
 * @tparam T The type of objects to allocate.
 * @param count The number of objects to allocate (default: 1).
 * @param poolName The name of the pool to use (default: "default_gpu").
 * @return Pointer to the allocated typed GPU memory.
 */
template <typename T>
T* allocateGPU(size_t count = 1, const std::string& poolName = "default_gpu") {
    return static_cast<T*>(allocateGPU(sizeof(T) * count, poolName));
}

/**
 * @brief Template function for typed CPU memory deallocation.
 * @tparam T The type of objects being deallocated.
 * @param ptr Pointer to the memory to deallocate.
 * @param poolName The name of the pool to use (default: "default").
 */
template <typename T>
void deallocate(T* ptr, const std::string& poolName = "default") {
    deallocate(static_cast<void*>(ptr), poolName);
}

/**
 * @brief Template function for typed GPU memory deallocation.
 * @tparam T The type of objects being deallocated.
 * @param ptr Pointer to the GPU memory to deallocate.
 * @param poolName The name of the pool to use (default: "default_gpu").
 */
template <typename T>
void deallocateGPU(T* ptr, const std::string& poolName = "default_gpu") {
    deallocateGPU(static_cast<void*>(ptr), poolName);
}

}  // namespace memory_pool

#endif  // MEMORY_POOL_MEMORY_POOL_HPP