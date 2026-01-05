#include "memory_pool/memory_pool.hpp"
#include "memory_pool/cpu/cpu_memory_pool.hpp"
#include "memory_pool/custom/custom_allocator.hpp"
#include "memory_pool/custom/custom_memory_pool.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include "memory_pool/utils/platform_utils.hpp"
#include <stdexcept>
#include <algorithm>

#if HAVE_CUDA
#include "memory_pool/gpu/gpu_memory_pool.hpp"
#include "memory_pool/gpu/cuda_utils.hpp"
#endif

#ifdef HAVE_PMEM
#include "memory_pool/pmem/pmem_memory_pool.hpp"
#endif

namespace memory_pool {

// Singleton instance
MemoryPoolManager& MemoryPoolManager::getInstance() {
    static MemoryPoolManager instance;
    return instance;
}

MemoryPoolManager::MemoryPoolManager() {
    // Create default pools
    createCPUPool("default", PoolConfig::DefaultCPU());

#if HAVE_CUDA
    // Create default GPU pool on the best available device
    int bestDevice = selectBestGPUDevice();
    if (bestDevice >= 0) {
        PoolConfig gpuConfig = PoolConfig::DefaultGPU();
        gpuConfig.deviceId = bestDevice;
        createGPUPool("default_gpu", gpuConfig);
    } else {
        // Fallback to device 0 if no devices available
        createGPUPool("default_gpu", PoolConfig::DefaultGPU());
    }
#endif
}

MemoryPoolManager::~MemoryPoolManager() {
    // Pools will be automatically destroyed by unique_ptr
}

IMemoryPool* MemoryPoolManager::getCPUPool(const std::string& name) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    auto it = pools.find(name);
    if (it != pools.end() && it->second->getMemoryType() == MemoryType::CPU) {
        return it->second.get();
    }

    reportError(ErrorSeverity::Warning, "MemoryPoolManager: CPU pool '" + name + "' not found");
    return nullptr;
}

IMemoryPool* MemoryPoolManager::createCPUPool(const std::string& name, const PoolConfig& config) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    // Check if a pool with this name already exists
    if (pools.find(name) != pools.end()) {
        reportError(ErrorSeverity::Warning, "MemoryPoolManager: Pool '" + name + "' already exists");
        return pools[name].get();
    }

    // Create a new CPU pool
    auto         pool    = std::make_unique<CPUMemoryPool>(name, config);
    IMemoryPool* poolPtr = pool.get();

    // Add the pool to the map
    pools[name] = std::move(pool);

    return poolPtr;
}

#if HAVE_CUDA

IMemoryPool* MemoryPoolManager::getGPUPool(const std::string& name) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    auto it = pools.find(name);
    if (it != pools.end() && it->second->getMemoryType() == MemoryType::GPU) {
        return it->second.get();
    }

    reportError(ErrorSeverity::Warning, "MemoryPoolManager: GPU pool '" + name + "' not found");
    return nullptr;
}

IMemoryPool* MemoryPoolManager::createGPUPool(const std::string& name, const PoolConfig& config) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    // Check if a pool with this name already exists
    if (pools.find(name) != pools.end()) {
        reportError(ErrorSeverity::Warning, "MemoryPoolManager: Pool '" + name + "' already exists");
        return pools[name].get();
    }

    // Create a new GPU pool
    auto         pool    = std::make_unique<GPUMemoryPool>(name, config);
    IMemoryPool* poolPtr = pool.get();

    // Add the pool to the map
    pools[name] = std::move(pool);

    return poolPtr;
}

#else  // HAVE_CUDA

IMemoryPool* MemoryPoolManager::getGPUPool(const std::string& name) {
    (void)name;
    reportError(ErrorSeverity::Error, "MemoryPoolManager: GPU support not available - CUDA not found");
    return nullptr;
}

IMemoryPool* MemoryPoolManager::createGPUPool(const std::string& name, const PoolConfig& config) {
    (void)name;
    (void)config;
    reportError(ErrorSeverity::Error, "MemoryPoolManager: GPU support not available - CUDA not found");
    return nullptr;
}

#endif  // HAVE_CUDA

bool MemoryPoolManager::destroyPool(const std::string& name) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    auto it = pools.find(name);
    if (it == pools.end()) {
        reportError(ErrorSeverity::Warning, "MemoryPoolManager: Cannot destroy pool '" + name + "', not found");
        return false;
    }

    // Don't allow destroying default pools
    if (name == "default" || name == "default_gpu") {
        reportError(ErrorSeverity::Warning, "MemoryPoolManager: Cannot destroy default pool '" + name + "'");
        return false;
    }

    // Remove the pool from the map
    pools.erase(it);
    return true;
}

void MemoryPoolManager::resetAllPools() {
    std::lock_guard<std::mutex> lock(poolsMutex);

    for (auto& pair : pools) {
        pair.second->reset();
    }
}

std::map<std::string, std::string> MemoryPoolManager::getAllStats() const {
    std::lock_guard<std::mutex> lock(poolsMutex);

    // Instead of returning a map of MemoryStats, return a map of strings
    std::map<std::string, std::string> stats;
    for (const auto& pair : pools) {
        // Get a reference to the stats
        const MemoryStats& poolStats = pair.second->getStats();
        // Convert to string representation
        stats[pair.first] = poolStats.getStatsString();
    }

    return stats;
}

#if HAVE_CUDA

int MemoryPoolManager::getGPUDeviceCount() {
    return getDeviceCount();
}

bool MemoryPoolManager::isGPUDeviceAvailable(int deviceId) {
    return isDeviceAvailable(deviceId);
}

size_t MemoryPoolManager::getGPUDeviceMemory(int deviceId) {
    return getDeviceMemory(deviceId);
}

int MemoryPoolManager::selectBestGPUDevice() {
    int deviceCount = getGPUDeviceCount();
    if (deviceCount == 0) {
        return -1;
    }

    // Select device with most available memory
    int    bestDevice = -1;
    size_t maxMemory  = 0;

    for (int i = 0; i < deviceCount; ++i) {
        if (isGPUDeviceAvailable(i)) {
            size_t memory = getGPUDeviceMemory(i);
            if (memory > maxMemory) {
                maxMemory  = memory;
                bestDevice = i;
            }
        }
    }

    return bestDevice;
}

IMemoryPool* MemoryPoolManager::createGPUPoolForDevice(int deviceId, const PoolConfig& config) {
    if (!isGPUDeviceAvailable(deviceId)) {
        reportError(ErrorSeverity::Error, "MemoryPoolManager: GPU device " + std::to_string(deviceId) + " not available");
        return nullptr;
    }

    std::string poolName = "gpu_" + std::to_string(deviceId);
    PoolConfig  deviceConfig = config;
    deviceConfig.deviceId = deviceId;

    return createGPUPool(poolName, deviceConfig);
}

#else  // HAVE_CUDA

int MemoryPoolManager::getGPUDeviceCount() {
    return 0;
}

bool MemoryPoolManager::isGPUDeviceAvailable(int deviceId) {
    (void)deviceId;
    return false;
}

size_t MemoryPoolManager::getGPUDeviceMemory(int deviceId) {
    (void)deviceId;
    return 0;
}

int MemoryPoolManager::selectBestGPUDevice() {
    return -1;
}

IMemoryPool* MemoryPoolManager::createGPUPoolForDevice(int deviceId, const PoolConfig& config) {
    (void)deviceId;
    (void)config;
    reportError(ErrorSeverity::Error, "MemoryPoolManager: GPU support not available - CUDA not found");
    return nullptr;
}

#endif  // HAVE_CUDA

#ifdef HAVE_PMEM

IMemoryPool* MemoryPoolManager::getPMEMPool(const std::string& name) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    auto it = pools.find(name);
    if (it != pools.end() && it->second->getMemoryType() == MemoryType::PMEM) {
        return it->second.get();
    }

    reportError(ErrorSeverity::Warning, "MemoryPoolManager: PMEM pool '" + name + "' not found");
    return nullptr;
}

IMemoryPool* MemoryPoolManager::createPMEMPool(const std::string& name, const PoolConfig& config) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    // Check if a pool with this name already exists
    if (pools.find(name) != pools.end()) {
        reportError(ErrorSeverity::Warning, "MemoryPoolManager: Pool '" + name + "' already exists");
        return pools[name].get();
    }

    // Create a new PMEM pool
    auto         pool    = std::make_unique<PMEMMemoryPool>(name, config);
    IMemoryPool* poolPtr = pool.get();

    // Add the pool to the map
    pools[name] = std::move(pool);

    return poolPtr;
}

#else  // HAVE_PMEM

IMemoryPool* MemoryPoolManager::getPMEMPool(const std::string& name) {
    reportError(ErrorSeverity::Error, "MemoryPoolManager: PMEM support not available - libpmem not found");
    return nullptr;
}

IMemoryPool* MemoryPoolManager::createPMEMPool(const std::string& name, const PoolConfig& config) {
    reportError(ErrorSeverity::Error, "MemoryPoolManager: PMEM support not available - libpmem not found");
    return nullptr;
}

#endif  // HAVE_PMEM

IMemoryPool* MemoryPoolManager::getCustomPool(const std::string& name) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    auto it = pools.find(name);
    if (it != pools.end() && it->second->getMemoryType() == MemoryType::Custom) {
        return it->second.get();
    }

    reportError(ErrorSeverity::Warning, "MemoryPoolManager: Custom pool '" + name + "' not found");
    return nullptr;
}

IMemoryPool* MemoryPoolManager::createCustomPool(const std::string& name, const PoolConfig& config) {
    std::lock_guard<std::mutex> lock(poolsMutex);

    // Check if a pool with this name already exists
    if (pools.find(name) != pools.end()) {
        reportError(ErrorSeverity::Warning, "MemoryPoolManager: Pool '" + name + "' already exists");
        return pools[name].get();
    }

    // Check if allocator type is Custom
    if (config.allocatorType != AllocatorType::Custom) {
        reportError(ErrorSeverity::Error, "MemoryPoolManager: Pool config must have allocatorType = Custom for custom pools");
        return nullptr;
    }

    // Create custom allocator
    std::unique_ptr<ICustomAllocator> customAllocator;
    try {
        customAllocator = CustomAllocatorRegistry::createAllocator(config.hardwareType, config.hardwareConfig);
    } catch (const std::exception& e) {
        reportError(ErrorSeverity::Error, std::string("MemoryPoolManager: Failed to create custom allocator: ") + e.what());
        return nullptr;
    }

    if (!customAllocator) {
        reportError(ErrorSeverity::Error, "MemoryPoolManager: Failed to create custom allocator for hardware type " +
                   std::to_string(static_cast<int>(config.hardwareType)));
        return nullptr;
    }

    // Create custom memory pool
    auto pool = std::make_unique<CustomMemoryPool>(name, config, std::move(customAllocator));
    IMemoryPool* poolPtr = pool.get();

    // Add the pool to the map
    pools[name] = std::move(pool);

    return poolPtr;
}

// Helper functions for common operations
void* allocate(size_t size, const std::string& poolName) {
    IMemoryPool* pool = MemoryPoolManager::getInstance().getCPUPool(poolName);
    if (pool == nullptr) {
        throw InvalidOperationException("Pool '" + poolName + "' not found");
    }

    return pool->allocate(size);
}

void deallocate(void* ptr, const std::string& poolName) {
    if (ptr == nullptr) {
        return;
    }

    IMemoryPool* pool = MemoryPoolManager::getInstance().getCPUPool(poolName);
    if (pool == nullptr) {
        throw InvalidOperationException("Pool '" + poolName + "' not found");
    }

    pool->deallocate(ptr);
}

#if HAVE_CUDA

void* allocateGPU(size_t size, const std::string& poolName) {
    IMemoryPool* pool = MemoryPoolManager::getInstance().getGPUPool(poolName);
    if (pool == nullptr) {
        throw InvalidOperationException("GPU pool '" + poolName + "' not found");
    }

    return pool->allocate(size);
}

void* allocateGPU(size_t size, int deviceId) {
    MemoryPoolManager& manager = MemoryPoolManager::getInstance();

    if (!manager.isGPUDeviceAvailable(deviceId)) {
        throw InvalidOperationException("GPU device " + std::to_string(deviceId) + " not available");
    }

    // Create pool for the device if not exists
    std::string poolName = "gpu_" + std::to_string(deviceId);
    IMemoryPool* pool = manager.getGPUPool(poolName);
    if (pool == nullptr) {
        pool = manager.createGPUPoolForDevice(deviceId);
        if (pool == nullptr) {
            throw InvalidOperationException("Failed to create GPU pool for device " + std::to_string(deviceId));
        }
    }

    return pool->allocate(size);
}

void deallocateGPU(void* ptr, const std::string& poolName) {
    if (ptr == nullptr) {
        return;
    }

    IMemoryPool* pool = MemoryPoolManager::getInstance().getGPUPool(poolName);
    if (pool == nullptr) {
        throw InvalidOperationException("GPU pool '" + poolName + "' not found");
    }

    pool->deallocate(ptr);
}

#else  // HAVE_CUDA

void* allocateGPU(size_t size, const std::string& poolName) {
    (void)size;
    (void)poolName;
    throw InvalidOperationException("GPU support not available - CUDA not found");
}

void* allocateGPU(size_t size, int deviceId) {
    (void)size;
    (void)deviceId;
    throw InvalidOperationException("GPU support not available - CUDA not found");
}

void deallocateGPU(void* ptr, const std::string& poolName) {
    (void)ptr;
    (void)poolName;
    throw InvalidOperationException("GPU support not available - CUDA not found");
}

#endif  // HAVE_CUDA

}  // namespace memory_pool