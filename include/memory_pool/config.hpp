#ifndef MEMORY_POOL_CONFIG_HPP
#define MEMORY_POOL_CONFIG_HPP

#include <cstddef>
#include <string>
#include "common.hpp"

namespace memory_pool {

/**
 * @brief Types of allocators supported by the memory pool system.
 */
enum class AllocatorType {
    FixedSize,    /**< Fixed-size block allocator for uniform allocations */
    VariableSize  /**< Variable-size allocator for arbitrary allocation sizes */
};

/**
 * @brief Types of synchronization mechanisms.
 */
enum class SyncType {
    Mutex,    /**< Standard mutex-based synchronization */
    LockFree  /**< Lock-free synchronization for high-performance scenarios */
};

/**
 * @brief Configuration structure for memory pool creation.
 *
 * This structure contains all the parameters needed to configure a memory pool,
 * including allocator type, size settings, threading options, and GPU-specific
 * parameters.
 */
struct PoolConfig {
    /** @brief Type of allocator to use */
    AllocatorType allocatorType   = AllocatorType::VariableSize;
    /** @brief Initial size of the memory pool in bytes */
    size_t        initialSize     = 1024 * 1024;  // 1MB default
    /** @brief Block size for fixed-size allocator */
    size_t        blockSize       = 256;          // For fixed-size allocator
    /** @brief Whether the pool should be thread-safe */
    bool          threadSafe      = true;
    /** @brief Type of synchronization to use */
    SyncType      syncType        = SyncType::Mutex;  // Synchronization method
    /** @brief Whether to track memory statistics */
    bool          trackStats      = true;
    /** @brief Whether to enable debugging features */
    bool          enableDebugging = false;

    /** @brief CUDA device ID for GPU pools */
    int  deviceId         = 0;      // CUDA device ID
    /** @brief Whether to use pinned memory for CPU-GPU transfers */
    bool usePinnedMemory  = false;  // Use pinned memory for CPU-GPU transfers
    /** @brief Whether to use CUDA managed memory */
    bool useManagedMemory = false;  // Use CUDA managed memory

    /** @brief Memory alignment requirement */
    size_t alignment    = DEFAULT_ALIGNMENT;
    /** @brief Growth factor when expanding the pool */
    size_t growthFactor = 2;  // How much to grow when out of memory
    /** @brief Maximum pool size (0 means no limit) */
    size_t maxSize      = 0;  // 0 means no limit

    /** @brief Default constructor */
    PoolConfig() {
#ifndef NDEBUG
        enableDebugging = true;  // Enable debugging in debug builds by default
#endif
    }

    /**
     * @brief Creates a default CPU pool configuration.
     * @return A PoolConfig with default CPU settings.
     */
    static PoolConfig DefaultCPU() {
        PoolConfig config;
        return config;
    }

    /**
     * @brief Creates a default GPU pool configuration.
     * @return A PoolConfig with default GPU settings.
     */
    static PoolConfig DefaultGPU() {
        PoolConfig config;
        config.deviceId = 0;
        return config;
    }

    /**
     * @brief Creates a fixed-size CPU pool configuration.
     * @param blockSize The size of each block in bytes.
     * @return A PoolConfig for fixed-size CPU allocation.
     */
    static PoolConfig FixedSizeCPU(size_t blockSize) {
        PoolConfig config;
        config.allocatorType = AllocatorType::FixedSize;
        config.blockSize     = blockSize;
        return config;
    }

    /**
     * @brief Creates a fixed-size GPU pool configuration.
     * @param blockSize The size of each block in bytes.
     * @param deviceId The CUDA device ID to use.
     * @return A PoolConfig for fixed-size GPU allocation.
     */
    static PoolConfig FixedSizeGPU(size_t blockSize, int deviceId = 0) {
        PoolConfig config;
        config.allocatorType = AllocatorType::FixedSize;
        config.blockSize     = blockSize;
        config.deviceId      = deviceId;
        return config;
    }

    /**
     * @brief Creates a high-performance CPU pool configuration.
     * @return A PoolConfig optimized for CPU performance.
     */
    static PoolConfig HighPerformanceCPU() {
        PoolConfig config;
        config.initialSize     = 16 * 1024 * 1024;  // 16MB
        config.enableDebugging = false;
        return config;
    }

    /**
     * @brief Creates a high-performance GPU pool configuration.
     * @param deviceId The CUDA device ID to use.
     * @return A PoolConfig optimized for GPU performance.
     */
    static PoolConfig HighPerformanceGPU(int deviceId = 0) {
        PoolConfig config;
        config.initialSize     = 64 * 1024 * 1024;  // 64MB
        config.enableDebugging = false;
        config.deviceId        = deviceId;
        return config;
    }

    /**
     * @brief Creates a debug-enabled CPU pool configuration.
     * @return A PoolConfig with debugging features enabled.
     */
    static PoolConfig DebugCPU() {
        PoolConfig config;
        config.enableDebugging = true;
        config.trackStats      = true;
        return config;
    }

    /**
     * @brief Creates a debug-enabled GPU pool configuration.
     * @param deviceId The CUDA device ID to use.
     * @return A PoolConfig with debugging features enabled.
     */
    static PoolConfig DebugGPU(int deviceId = 0) {
        PoolConfig config;
        config.enableDebugging = true;
        config.trackStats      = true;
        config.deviceId        = deviceId;
        return config;
    }
};

}  // namespace memory_pool

#endif  // MEMORY_POOL_CONFIG_HPP