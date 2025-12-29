#ifndef MEMORY_POOL_CONFIG_HPP
#define MEMORY_POOL_CONFIG_HPP

#include <cstddef>
#include <string>
#include "common.hpp"

namespace memory_pool {

// Allocator types
enum class AllocatorType { FixedSize, VariableSize };

// Synchronization types
enum class SyncType {
    Mutex,    // Standard mutex-based synchronization
    LockFree  // Lock-free synchronization for high-performance scenarios
};

// Configuration for memory pools
struct PoolConfig {
    // General configuration
    AllocatorType allocatorType   = AllocatorType::VariableSize;
    size_t        initialSize     = 1024 * 1024;  // 1MB default
    size_t        blockSize       = 256;          // For fixed-size allocator
    bool          threadSafe      = true;
    SyncType      syncType        = SyncType::Mutex;  // Synchronization method
    bool          trackStats      = true;
    bool          enableDebugging = false;

    // GPU-specific options
    int  deviceId         = 0;      // CUDA device ID
    bool usePinnedMemory  = false;  // Use pinned memory for CPU-GPU transfers
    bool useManagedMemory = false;  // Use CUDA managed memory

    // Advanced options
    size_t alignment    = DEFAULT_ALIGNMENT;
    size_t growthFactor = 2;  // How much to grow when out of memory
    size_t maxSize      = 0;  // 0 means no limit

    // Constructor with basic options
    PoolConfig() = default;

    // Named constructors for common configurations
    static PoolConfig DefaultCPU() {
        PoolConfig config;
        return config;
    }

    static PoolConfig DefaultGPU() {
        PoolConfig config;
        config.deviceId = 0;
        return config;
    }

    static PoolConfig FixedSizeCPU(size_t blockSize) {
        PoolConfig config;
        config.allocatorType = AllocatorType::FixedSize;
        config.blockSize     = blockSize;
        return config;
    }

    static PoolConfig FixedSizeGPU(size_t blockSize, int deviceId = 0) {
        PoolConfig config;
        config.allocatorType = AllocatorType::FixedSize;
        config.blockSize     = blockSize;
        config.deviceId      = deviceId;
        return config;
    }

    static PoolConfig HighPerformanceCPU() {
        PoolConfig config;
        config.initialSize     = 16 * 1024 * 1024;  // 16MB
        config.enableDebugging = false;
        return config;
    }

    static PoolConfig HighPerformanceGPU(int deviceId = 0) {
        PoolConfig config;
        config.initialSize     = 64 * 1024 * 1024;  // 64MB
        config.enableDebugging = false;
        config.deviceId        = deviceId;
        return config;
    }

    static PoolConfig DebugCPU() {
        PoolConfig config;
        config.enableDebugging = true;
        config.trackStats      = true;
        return config;
    }

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