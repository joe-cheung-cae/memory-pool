# Memory Pool API Reference

This document provides a comprehensive reference for the Memory Pool Management System API.

## Core Classes

### IMemoryPool

The abstract base class for all memory pool implementations.

```cpp
class IMemoryPool {
public:
    virtual ~IMemoryPool() = default;

    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void* allocate(size_t size, AllocFlags flags) = 0;
    virtual void reset() = 0;
    virtual const MemoryStats& getStats() const = 0;
    virtual MemoryType getMemoryType() const = 0;
    virtual std::string getName() const = 0;
    virtual const PoolConfig& getConfig() const = 0;
};
```

### MemoryPoolManager

Singleton manager for creating and managing memory pools.

```cpp
class MemoryPoolManager {
public:
    static MemoryPoolManager& getInstance();

    IMemoryPool* getCPUPool(const std::string& name);
    IMemoryPool* createCPUPool(const std::string& name, const PoolConfig& config);
    IMemoryPool* getGPUPool(const std::string& name);
    IMemoryPool* createGPUPool(const std::string& name, const PoolConfig& config);

    bool destroyPool(const std::string& name);
    void resetAllPools();
    std::map<std::string, std::string> getAllStats() const;
};
```

## Configuration

### PoolConfig

Configuration structure for memory pool creation.

```cpp
struct PoolConfig {
    AllocatorType allocatorType = AllocatorType::VariableSize;
    size_t initialSize = 1024 * 1024;
    size_t blockSize = 256;
    bool threadSafe = true;
    SyncType syncType = SyncType::Mutex;
    bool trackStats = true;
    bool enableDebugging = false;

    // GPU-specific
    int deviceId = 0;
    bool usePinnedMemory = false;
    bool useManagedMemory = false;

    // Advanced
    size_t alignment = DEFAULT_ALIGNMENT;
    size_t growthFactor = 2;
    size_t maxSize = 0;

    // Named constructors
    static PoolConfig DefaultCPU();
    static PoolConfig DefaultGPU();
    static PoolConfig FixedSizeCPU(size_t blockSize);
    static PoolConfig FixedSizeGPU(size_t blockSize, int deviceId = 0);
    static PoolConfig HighPerformanceCPU();
    static PoolConfig HighPerformanceGPU(int deviceId = 0);
    static PoolConfig DebugCPU();
    static PoolConfig DebugGPU(int deviceId = 0);
};
```

## Helper Functions

### Global Allocation Functions

```cpp
// CPU memory allocation
void* allocate(size_t size, const std::string& poolName = "default");
void deallocate(void* ptr, const std::string& poolName = "default");

// GPU memory allocation
void* allocateGPU(size_t size, const std::string& poolName = "default_gpu");
void deallocateGPU(void* ptr, const std::string& poolName = "default_gpu");

// Template functions for typed allocation
template <typename T>
T* allocate(size_t count = 1, const std::string& poolName = "default");

template <typename T>
T* allocateGPU(size_t count = 1, const std::string& poolName = "default_gpu");

template <typename T>
void deallocate(T* ptr, const std::string& poolName = "default");

template <typename T>
void deallocateGPU(T* ptr, const std::string& poolName = "default_gpu");
```

## Memory Statistics

### MemoryStats

Class for tracking memory allocation statistics.

```cpp
class MemoryStats {
public:
    void recordAllocation(size_t size);
    void recordDeallocation(size_t size);
    void trackAllocation(void* ptr, size_t size);
    void trackDeallocation(void* ptr);

    size_t getTotalAllocated() const;
    size_t getCurrentUsed() const;
    size_t getPeakUsage() const;
    size_t getAllocationCount() const;
    size_t getDeallocationCount() const;
    double getFragmentationRatio() const;

    std::vector<AllocationInfo> getActiveAllocations() const;
    bool hasMemoryLeaks() const;
    std::string getStatsString() const;

    void setTrackingEnabled(bool enable);
    bool isTrackingEnabled() const;
    void reset();
};
```

## GPU-Specific Classes

### GPUMemoryPool

GPU memory pool implementation.

```cpp
class GPUMemoryPool : public IMemoryPool {
public:
    GPUMemoryPool(const std::string& name, const PoolConfig& config);
    ~GPUMemoryPool() override;

    // IMemoryPool interface...
    // GPU-specific methods
    void setDevice(int deviceId);
    int getDevice() const;
    void setStream(cudaStream_t stream);
    cudaStream_t getStream() const;

    // Memory transfer helpers
    void copyHostToDevice(void* dst, const void* src, size_t size);
    void copyDeviceToHost(void* dst, const void* src, size_t size);
    void copyDeviceToDevice(void* dst, const void* src, size_t size);
};
```

## Enums and Types

### AllocatorType

```cpp
enum class AllocatorType {
    FixedSize,    // Fixed-size block allocator
    VariableSize  // Variable-size allocator
};
```

### SyncType

```cpp
enum class SyncType {
    Mutex,    // Standard mutex-based synchronization
    LockFree  // Lock-free synchronization for high-performance scenarios
};
```

### MemoryType

```cpp
enum class MemoryType {
    CPU,  // Host memory
    GPU   // Device memory
};
```

### AllocFlags

```cpp
enum class AllocFlags {
    None = 0,      // Default allocation
    Pinned = 1,    // Pinned memory for GPU transfers
    Managed = 2    // CUDA managed memory
};
```

## Error Handling

The library uses custom exceptions derived from `MemoryPoolException`:

- `OutOfMemoryException`: Thrown when memory allocation fails
- `InvalidOperationException`: Thrown for invalid operations
- `InvalidPointerException`: Thrown for invalid pointer operations
- `CudaException`: Thrown for CUDA-related errors

## Thread Safety

All memory pools are thread-safe by default. The synchronization mechanism can be configured using the `syncType` parameter in `PoolConfig`.

## Performance Considerations

- Fixed-size allocators provide O(1) allocation/deallocation
- Variable-size allocators may have higher overhead but provide better memory utilization
- GPU memory pools include optimizations for CUDA operations
- Pinned memory can improve transfer performance between CPU and GPU

For detailed API documentation with examples, see the generated Doxygen HTML documentation.