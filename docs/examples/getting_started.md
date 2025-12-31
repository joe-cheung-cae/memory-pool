# Getting Started with Memory Pool Management System

This tutorial will guide you through the basic usage of the Memory Pool Management System.

## Prerequisites

- C++17 compatible compiler
- CUDA Toolkit 11.0+ (for GPU support)
- CMake 3.14+

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd memory-pool
```

2. Build the library:
```bash
mkdir build
cd build
cmake ..
make
```

## Basic CPU Memory Pool Usage

### Step 1: Include the header

```cpp
#include "memory_pool/memory_pool.hpp"
#include <iostream>

using namespace memory_pool;
```

### Step 2: Create a memory pool

```cpp
int main() {
    // Get the memory pool manager (singleton)
    auto& manager = MemoryPoolManager::getInstance();

    // Create a CPU memory pool with default configuration
    PoolConfig config;
    IMemoryPool* pool = manager.createCPUPool("my_first_pool", config);
```

### Step 3: Allocate and use memory

```cpp
    // Allocate memory
    void* data = pool->allocate(1024);  // Allocate 1KB

    // Use the memory (cast to appropriate type)
    int* intArray = static_cast<int*>(data);
    for (int i = 0; i < 256; ++i) {  // 256 * 4 = 1024 bytes
        intArray[i] = i;
    }

    std::cout << "First element: " << intArray[0] << std::endl;
    std::cout << "Last element: " << intArray[255] << std::endl;
```

### Step 4: Deallocate memory

```cpp
    // Deallocate the memory
    pool->deallocate(data);
```

### Step 5: Check statistics

```cpp
    // Get memory statistics
    const MemoryStats& stats = pool->getStats();
    std::cout << "Memory usage statistics:" << std::endl;
    std::cout << stats.getStatsString() << std::endl;

    return 0;
}
```

## Complete Example

```cpp
#include "memory_pool/memory_pool.hpp"
#include <iostream>

using namespace memory_pool;

int main() {
    // Get the memory pool manager
    auto& manager = MemoryPoolManager::getInstance();

    // Create a CPU memory pool
    PoolConfig config;
    IMemoryPool* pool = manager.createCPUPool("tutorial_pool", config);

    // Allocate memory
    void* data = pool->allocate(1024);

    // Use the memory
    int* intArray = static_cast<int*>(data);
    for (int i = 0; i < 256; ++i) {
        intArray[i] = i * 2;
    }

    std::cout << "Array values: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << intArray[i] << " ";
    }
    std::cout << std::endl;

    // Deallocate memory
    pool->deallocate(data);

    // Print statistics
    const MemoryStats& stats = pool->getStats();
    std::cout << stats.getStatsString() << std::endl;

    return 0;
}
```

## Using Helper Functions

The library provides convenient helper functions for common operations:

```cpp
#include "memory_pool/memory_pool.hpp"
#include <iostream>

using namespace memory_pool;

int main() {
    // Allocate using helper functions
    int* array = allocate<int>(100);  // Allocate array of 100 integers

    // Use the array
    for (int i = 0; i < 100; ++i) {
        array[i] = i * i;
    }

    std::cout << "First 10 squares: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;

    // Deallocate
    deallocate(array);

    return 0;
}
```

## Next Steps

- Learn about [GPU memory pools](gpu_tutorial.md)
- Explore [advanced configuration options](configuration_tutorial.md)
- Understand [performance optimization](performance_tutorial.md)
- Check [cross-platform compatibility](cross_platform.md) for building on different operating systems
- Read the [API reference](../api/README.md)