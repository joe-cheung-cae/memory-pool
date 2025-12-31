# Memory Pool Management System

A high-performance memory pool management system implemented in C++ that supports both CPU and GPU (CUDA) memory management.

## Overview

This library provides efficient memory allocation and deallocation for high-performance computing applications. It includes support for both CPU and GPU memory, with various allocation strategies and features for debugging and performance monitoring.

## Features

- **Cross-Hardware Support**:
  - CPU memory pool management
  - GPU memory pool management via CUDA
  - Persistent memory (PMEM) support via PMDK/libpmem
  - Seamless integration between CPU, GPU, and PMEM operations

- **Allocation Strategies**:
  - Fixed-size block allocation
  - Variable-size allocation
  - Configurable allocation policies

- **Thread Safety**:
  - Support for multi-threaded applications
  - Lock-based and lock-free synchronization options
  - Thread-local storage options for high-performance scenarios

- **Memory Tracking and Statistics**:
  - Memory usage tracking
  - Allocation/deallocation statistics
  - Performance metrics

- **Debugging Capabilities**:
  - Memory leak detection
  - Boundary checking
  - Allocation history tracking

## Requirements

- C++17 compatible compiler
- CUDA Toolkit 11.0 or higher (for GPU support)
- PMDK (Persistent Memory Development Kit) (for PMEM support, optional)
- CMake 3.14 or higher

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Usage Examples

### Basic CPU Memory Pool

```cpp
#include "memory_pool/memory_pool.hpp"
#include <iostream>

using namespace memory_pool;

int main() {
    // Get the memory pool manager
    auto& manager = MemoryPoolManager::getInstance();
    
    // Create a CPU memory pool with default configuration
    PoolConfig config;
    auto* pool = manager.createCPUPool("main_cpu_pool", config);
    
    // Allocate memory
    void* data = pool->allocate(1024);
    
    // Use the memory...
    
    // Deallocate memory
    pool->deallocate(data);
    
    // Get memory statistics
    const MemoryStats& stats = pool->getStats();
    std::cout << stats.getStatsString() << std::endl;
    
    return 0;
}
```

### Basic GPU Memory Pool

```cpp
#include "memory_pool/memory_pool.hpp"
#include <iostream>

using namespace memory_pool;

int main() {
    // Get the memory pool manager
    auto& manager = MemoryPoolManager::getInstance();
    
    // Create a GPU memory pool with custom configuration
    PoolConfig config;
    config.allocatorType = PoolConfig::AllocatorType::FixedSize;
    config.blockSize = 1024;
    config.deviceId = 0;  // Use first CUDA device
    
    auto* pool = manager.createGPUPool("main_gpu_pool", config);
    
    // Allocate GPU memory
    void* deviceData = pool->allocate(1024 * 10);
    
    // Allocate CPU memory for data transfer
    void* hostData = allocate(1024 * 10);
    
    // Fill host data
    // ...
    
    // Copy data from host to device
    GPUMemoryPool* gpuPool = static_cast<GPUMemoryPool*>(pool);
    gpuPool->copyHostToDevice(deviceData, hostData, 1024 * 10);
    
    // Use the memory with CUDA kernels...
    
    // Copy results back to host
    gpuPool->copyDeviceToHost(hostData, deviceData, 1024 * 10);
    
    // Deallocate memory
    pool->deallocate(deviceData);
    deallocate(hostData);
    
    return 0;
}
```

### Basic PMEM Memory Pool

```cpp
#include "memory_pool/memory_pool.hpp"
#include <iostream>
#include <cstring>

using namespace memory_pool;

int main() {
    // Get the memory pool manager
    auto& manager = MemoryPoolManager::getInstance();

    // Create a PMEM memory pool with custom configuration
    PoolConfig config;
    config.initialSize = 2 * 1024 * 1024;  // 2MB
    config.pmemPoolPath = "/tmp/my_persistent_pool.pool";

    auto* pool = manager.createPMEMPool("persistent_pool", config);

    // Allocate persistent memory
    void* data = pool->allocate(1024);
    strcpy(static_cast<char*>(data), "This data persists!");

    // Persist the data explicitly
    PMEMMemoryPool* pmemPool = static_cast<PMEMMemoryPool*>(pool);
    pmemPool->persist(data, strlen("This data persists!") + 1);

    std::cout << "Data written and persisted: " << static_cast<char*>(data) << std::endl;

    // Deallocate memory
    pool->deallocate(data);

    return 0;
}
```

### Using Helper Functions

```cpp
#include "memory_pool/memory_pool.hpp"
#include <iostream>

using namespace memory_pool;

int main() {
    // Allocate memory using helper functions
    int* array = allocate<int>(100);  // Allocate array of 100 integers
    
    // Use the memory
    for (int i = 0; i < 100; ++i) {
        array[i] = i;
    }
    
    // Deallocate memory
    deallocate(array);
    
    // GPU memory allocation
    float* deviceArray = allocateGPU<float>(1000);
    
    // Deallocate GPU memory
    deallocateGPU(deviceArray);
    
    return 0;
}
```

## Advanced Configuration

The library provides various configuration options for fine-tuning memory pools:

```cpp
// Create a high-performance CPU pool
PoolConfig config = PoolConfig::HighPerformanceCPU();
auto* pool = manager.createCPUPool("high_perf_pool", config);

// Create a debug-friendly GPU pool
PoolConfig debugConfig = PoolConfig::DebugGPU();
auto* debugPool = manager.createGPUPool("debug_pool", debugConfig);

// Create a fixed-size pool with specific block size
PoolConfig fixedConfig = PoolConfig::FixedSizeCPU(512);  // 512-byte blocks
auto* fixedPool = manager.createCPUPool("fixed_pool", fixedConfig);
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.