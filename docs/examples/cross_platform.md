# Cross-Platform Compatibility Guide

This guide covers building and using the Memory Pool Management System on different operating systems: Linux, Windows, and macOS.

## Supported Platforms

The memory pool system is designed to be cross-platform and supports:

- **Linux**: Full support for CPU, GPU (CUDA), and PMEM operations
- **Windows**: Full support for CPU and GPU (CUDA) operations
- **macOS**: CPU operations only (GPU support limited due to CUDA discontinuation on macOS)

## Prerequisites

### Linux

- C++17 compatible compiler (GCC 7+, Clang 5+)
- CUDA Toolkit 11.0+ (for GPU support)
- CMake 3.14+
- Valgrind (for memory leak detection)
- NUMA libraries (optional, for NUMA-aware allocation)
- PMDK (optional, for persistent memory support)

**Installation on Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install build-essential cmake cuda-toolkit valgrind libnuma-dev libpmem-dev
```

**Installation on CentOS/RHEL:**
```bash
sudo yum groupinstall "Development Tools"
sudo yum install cmake cuda-toolkit valgrind-devel numactl-devel libpmem-devel
```

### Windows

- Visual Studio 2019+ with C++17 support
- CUDA Toolkit 11.0+ (for GPU support)
- CMake 3.14+
- Windows SDK

**Installation:**
1. Download and install [Visual Studio](https://visualstudio.microsoft.com/)
2. Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
3. Download and install [CMake](https://cmake.org/download/)

**Note:** Valgrind and PMDK are not available on Windows, so memory leak detection and persistent memory features are not supported on this platform.

### macOS

- Xcode 10+ with Command Line Tools
- CMake 3.14+

**Installation:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake via Homebrew
brew install cmake
```

**Note:** CUDA is not supported on macOS (discontinued after CUDA 10.2), so GPU features are not available. PMEM support is also not available on macOS.

## Building

### Linux

```bash
git clone <repository-url>
cd memory-pool
mkdir build
cd build
cmake ..
make
```

### Windows

Using Visual Studio Developer Command Prompt:

```cmd
git clone <repository-url>
cd memory-pool
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

Or using Ninja:

```cmd
cmake .. -G Ninja
ninja
```

### macOS

```bash
git clone <repository-url>
cd memory-pool
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Platform-Specific Configuration

### Linux-Specific Features

- NUMA-aware memory allocation
- Persistent memory (PMEM) support via PMDK
- Valgrind integration for memory leak detection

**Example: NUMA-aware allocation**
```cpp
#include "memory_pool/memory_pool.hpp"

using namespace memory_pool;

int main() {
    PoolConfig config = PoolConfig::HighPerformanceCPU();
    // NUMA support is automatically enabled if libnuma is available
    auto* pool = manager.createCPUPool("numa_pool", config);
    // Memory will be allocated on the appropriate NUMA node
}
```

### Windows-Specific Considerations

- Use forward slashes (/) or escaped backslashes (\\\\) in paths
- GPU support requires proper CUDA installation and PATH configuration
- Memory leak detection is not available (Valgrind not supported)

**Example: Windows path handling**
```cpp
#include "memory_pool/memory_pool.hpp"

using namespace memory_pool;

int main() {
    PoolConfig config;
    // Use forward slashes for consistency across platforms
    config.pmemPoolPath = "C:/temp/persistent_pool.pool";  // Note: PMEM not supported on Windows

    auto* pool = manager.createCPUPool("windows_pool", config);
}
```

### macOS-Specific Considerations

- Only CPU memory pools are supported
- GPU and PMEM features are not available
- Use standard macOS development tools

**Example: macOS CPU-only usage**
```cpp
#include "memory_pool/memory_pool.hpp"

using namespace memory_pool;

int main() {
    // Only CPU pools are supported on macOS
    PoolConfig config = PoolConfig::HighPerformanceCPU();
    auto* pool = manager.createCPUPool("macos_pool", config);

    // GPU operations will throw exceptions
    // auto* gpuPool = manager.createGPUPool("gpu_pool", config); // Not supported
}
```

## CMake Configuration Options

### Platform-Independent Options

- `BUILD_EXAMPLES`: Build example programs (default: ON)
- `BUILD_TESTS`: Build unit tests (default: OFF)
- `BUILD_PERFORMANCE_TESTS`: Build performance benchmarks (default: OFF)

### Platform-Specific Options

On Linux, additional libraries are automatically detected:
- `HAVE_NUMA`: NUMA support enabled
- `HAVE_PMEM`: Persistent memory support enabled
- `HAVE_VALGRIND`: Memory leak detection enabled

## Troubleshooting

### Common Issues

**CUDA not found on Linux/Windows:**
- Ensure CUDA Toolkit is installed and `CUDA_PATH` environment variable is set
- Add CUDA bin directory to PATH

**Build failures on Windows:**
- Use Visual Studio Developer Command Prompt
- Ensure Windows SDK is installed
- Try building with Ninja instead of Visual Studio generator

**Missing dependencies on Linux:**
- Install development packages for required libraries
- Use `pkg-config` to verify library installation

**macOS build issues:**
- Ensure Command Line Tools are installed: `xcode-select --install`
- Update CMake if using old version

### Platform Limitations

| Feature | Linux | Windows | macOS |
|---------|-------|---------|-------|
| CPU Memory Pools | ✅ | ✅ | ✅ |
| GPU Memory Pools | ✅ | ✅ | ❌ |
| Persistent Memory | ✅ | ❌ | ❌ |
| NUMA Support | ✅ | ❌ | ❌ |
| Memory Leak Detection | ✅ | ❌ | ❌ |
| Multi-threading | ✅ | ✅ | ✅ |
| Custom Allocators | ✅ | ✅ | ✅ |

## Contributing

When contributing platform-specific code:

1. Use conditional compilation with standard platform macros:
   ```cpp
   #ifdef _WIN32
       // Windows-specific code
   #elif __APPLE__
       // macOS-specific code
   #else
       // Linux and other Unix-like systems
   #endif
   ```

2. Test builds on all supported platforms
3. Update this documentation for any new platform requirements
4. Ensure CMake configurations work across platforms

## Getting Help

- Check the [API documentation](../api/README.md) for detailed function references
- Review [best practices](best_practices.md) for optimization tips
- File issues on the project repository for platform-specific problems