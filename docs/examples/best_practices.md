# Memory Pool Best Practices

This guide provides recommendations for effectively using the Memory Pool Management System in your applications.

## General Guidelines

### 1. Choose the Right Allocator Type

- **Fixed-Size Allocators**: Use for objects of uniform size
  - Provides O(1) allocation/deallocation
  - Minimal overhead
  - Best for: Game objects, particles, small data structures

- **Variable-Size Allocators**: Use for objects of varying sizes
  - Better memory utilization
  - Handles fragmentation
  - Best for: Dynamic data, strings, complex objects

### 2. Pool Sizing

- **Initial Size**: Set based on expected memory usage
  - Too small: Frequent reallocations
  - Too large: Wasted memory
  - Rule of thumb: 2-4x expected peak usage

- **Growth Factor**: Default is 2x
  - Balance between memory waste and reallocation frequency
  - Consider application memory constraints

### 3. Thread Safety

- **Default**: All pools are thread-safe
- **Performance**: Use `LockFree` sync for high-contention scenarios
- **Single-threaded**: Disable thread safety for maximum performance

```cpp
// High-performance single-threaded pool
PoolConfig config = PoolConfig::HighPerformanceCPU();
config.threadSafe = false;

// Lock-free multi-threaded pool
PoolConfig mtConfig = PoolConfig::DefaultCPU();
mtConfig.syncType = SyncType::LockFree;
```

## CPU Memory Management

### Memory Alignment

- Use default alignment (typically 16 or 32 bytes) for general purpose
- Specify custom alignment for SIMD operations or cache optimization

```cpp
PoolConfig config;
config.alignment = 64;  // For AVX-512 operations
```

### Memory Access Patterns

- **Sequential Access**: Fixed-size allocators excel
- **Random Access**: Ensure proper alignment
- **Bulk Operations**: Use variable-size for large contiguous blocks

## GPU Memory Management

### Device Selection

- Specify device ID explicitly for multi-GPU systems
- Use `cudaGetDevice()` to get current device

```cpp
PoolConfig gpuConfig = PoolConfig::DefaultGPU();
gpuConfig.deviceId = 0;  // Use first GPU
```

### Memory Transfer Optimization

- **Pinned Memory**: Use for frequent CPU-GPU transfers
- **Managed Memory**: Use for automatic migration (Pascal+ GPUs)
- **Streams**: Use CUDA streams for asynchronous operations

```cpp
// Pinned memory for fast transfers
PoolConfig pinnedConfig = PoolConfig::DefaultGPU();
pinnedConfig.usePinnedMemory = true;

// Managed memory for automatic migration
PoolConfig managedConfig = PoolConfig::DefaultGPU();
managedConfig.useManagedMemory = true;
```

### GPU Pool Usage

```cpp
auto& manager = MemoryPoolManager::getInstance();
GPUMemoryPool* gpuPool = static_cast<GPUMemoryPool*>(
    manager.createGPUPool("gpu_pool", gpuConfig));

// Set CUDA stream for asynchronous operations
cudaStream_t stream;
cudaStreamCreate(&stream);
gpuPool->setStream(stream);
```

## Performance Optimization

### 1. Pool Reuse

- Reuse pools across multiple operations
- Avoid creating/destroying pools frequently
- Use named pools for different allocation patterns

```cpp
// Good: Reuse pools
auto& manager = MemoryPoolManager::getInstance();
auto* pool = manager.getCPUPool("reusable_pool");
if (!pool) {
    pool = manager.createCPUPool("reusable_pool", config);
}
```

### 2. Bulk Operations

- Allocate multiple objects together when possible
- Use appropriate block sizes for your access patterns

### 3. Memory Lifetime Management

- Allocate at setup, deallocate at teardown
- Minimize allocations during performance-critical code
- Use object pools for frequently created/destroyed objects

### 4. Statistics and Monitoring

- Enable statistics in development builds
- Monitor fragmentation and memory usage
- Use debug builds to detect leaks

```cpp
PoolConfig debugConfig = PoolConfig::DebugCPU();
// Check stats periodically
const auto& stats = pool->getStats();
if (stats.getFragmentationRatio() > 0.5) {
    // Consider defragmentation or pool reset
}
```

## Error Handling

### Exception Safety

- All operations are exception-safe
- Check return values for allocation failures
- Use RAII patterns for automatic cleanup

```cpp
void* data = nullptr;
try {
    data = pool->allocate(size);
    // Use data...
} catch (const OutOfMemoryException& e) {
    std::cerr << "Allocation failed: " << e.what() << std::endl;
}

if (data) {
    pool->deallocate(data);
}
```

### CUDA Error Handling

- GPU operations may throw `CudaException`
- Check CUDA error codes in release builds
- Handle device memory exhaustion gracefully

## Memory Leak Detection

### Debug Features

- Enable debugging in development builds
- Use memory leak detection tools
- Check allocation statistics
- Boundary checking prevents buffer overflows

```cpp
PoolConfig debugConfig = PoolConfig::DebugCPU();
auto* pool = manager.createCPUPool("debug_pool", debugConfig);

// Boundary checking is automatically enabled in debug builds
// It adds canary values around allocations to detect buffer overflows

// At application exit, check for boundary violations
std::string report = getMemoryReport();
if (!report.empty()) {
    std::cout << "Memory report:\n" << report << std::endl;
}
```

## Platform-Specific Considerations

### Linux
- Use huge pages for large allocations
- Consider NUMA effects in multi-socket systems

### Windows
- Be aware of memory fragmentation in long-running processes
- Use appropriate page sizes

### CUDA
- Consider GPU architecture (compute capability)
- Optimize for specific GPU memory types
- Use appropriate CUDA versions

## Testing and Validation

### Unit Testing
- Test allocation/deallocation cycles
- Test edge cases (zero size, large sizes)
- Test thread safety

### Integration Testing
- Test with real workloads
- Monitor memory usage patterns
- Validate performance requirements

### Performance Benchmarking
- Compare against standard allocators
- Test under various load conditions
- Profile memory access patterns

## Common Pitfalls

1. **Forgetting to deallocate**: Always match allocations with deallocations
2. **Wrong pool for deallocation**: Use the same pool for alloc/dealloc
3. **Threading issues**: Ensure proper synchronization
4. **GPU/CPU confusion**: Don't mix GPU and CPU pointers
5. **Ignoring statistics**: Monitor memory usage regularly
6. **Over-allocation**: Plan memory usage carefully

## Advanced Patterns

### Object Pools

```cpp
template <typename T>
class ObjectPool {
private:
    IMemoryPool* pool_;
    std::vector<T*> freeObjects_;

public:
    ObjectPool(IMemoryPool* pool) : pool_(pool) {}

    T* allocate() {
        if (!freeObjects_.empty()) {
            T* obj = freeObjects_.back();
            freeObjects_.pop_back();
            return new(obj) T();  // Placement new
        }
        return new(pool_->allocate(sizeof(T))) T();
    }

    void deallocate(T* obj) {
        obj->~T();  // Call destructor
        freeObjects_.push_back(obj);
    }
};
```

### Memory Arena

```cpp
class MemoryArena {
private:
    IMemoryPool* pool_;
    std::vector<void*> allocations_;

public:
    MemoryArena(IMemoryPool* pool) : pool_(pool) {}

    ~MemoryArena() {
        reset();
    }

    void* allocate(size_t size) {
        void* ptr = pool_->allocate(size);
        allocations_.push_back(ptr);
        return ptr;
    }

    void reset() {
        for (void* ptr : allocations_) {
            pool_->deallocate(ptr);
        }
        allocations_.clear();
    }
};
```

Following these best practices will help you get the most out of the Memory Pool Management System while avoiding common pitfalls.