#include "memory_pool/gpu/gpu_memory_pool.hpp"
#include "memory_pool/gpu/cuda_allocator.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <cstring>

namespace memory_pool {

GPUMemoryPool::GPUMemoryPool(const std::string& name, const PoolConfig& config)
    : name(name), config(config), deviceId(config.deviceId), stream(nullptr), allocator(nullptr) {
    initialize();
}

GPUMemoryPool::~GPUMemoryPool() {
    // The allocator will be automatically destroyed by the unique_ptr
    
    // Destroy the CUDA stream if it was created
    if (stream != nullptr) {
        destroyStream(stream);
        stream = nullptr;
    }
}

void GPUMemoryPool::initialize() {
    // Set the device
    setDevice(deviceId);
    
    // Create a CUDA stream
    stream = createStream();
    
    // Create the appropriate allocator based on the configuration
    if (config.allocatorType == AllocatorType::FixedSize) {
        allocator = std::make_unique<CudaFixedSizeAllocator>(
            config.blockSize,
            config.initialSize / config.blockSize,
            deviceId,
            config.usePinnedMemory ? AllocFlags::Pinned : 
                (config.useManagedMemory ? AllocFlags::Managed : AllocFlags::None)
        );
    } else {
        allocator = std::make_unique<CudaVariableSizeAllocator>(
            config.initialSize,
            deviceId,
            config.usePinnedMemory ? AllocFlags::Pinned : 
                (config.useManagedMemory ? AllocFlags::Managed : AllocFlags::None)
        );
    }
    
    // Set the stream for the allocator
    allocator->setStream(stream);
}

void* GPUMemoryPool::allocate(size_t size) {
    return allocateInternal(size, AllocFlags::None);
}

void* GPUMemoryPool::allocate(size_t size, AllocFlags flags) {
    return allocateInternal(size, flags);
}

void* GPUMemoryPool::allocateInternal(size_t size, AllocFlags flags) {
    if (size == 0) {
        return nullptr;
    }
    
    // Ensure we're on the correct device
    ensureCorrectDevice();
    
    void* ptr = nullptr;
    
    // Use thread safety if configured
    if (config.threadSafe) {
        std::lock_guard<std::mutex> lock(mutex);
        ptr = allocator->allocate(size, flags);
    } else {
        ptr = allocator->allocate(size, flags);
    }
    
    if (ptr == nullptr) {
        throw OutOfMemoryException("Failed to allocate memory in GPU pool: " + name);
    }
    
    // Track statistics if enabled
    if (config.trackStats) {
        stats.recordAllocation(size);
        
        if (config.enableDebugging) {
            stats.trackAllocation(ptr, size);
        }
    }
    
    return ptr;
}

void GPUMemoryPool::deallocate(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    
    // Ensure we're on the correct device
    ensureCorrectDevice();
    
    size_t size = 0;
    
    // Use thread safety if configured
    if (config.threadSafe) {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Get the size before deallocating for statistics
        if (config.trackStats) {
            size = allocator->getBlockSize(ptr);
        }
        
        allocator->deallocate(ptr);
    } else {
        // Get the size before deallocating for statistics
        if (config.trackStats) {
            size = allocator->getBlockSize(ptr);
        }
        
        allocator->deallocate(ptr);
    }
    
    // Track statistics if enabled
    if (config.trackStats) {
        stats.recordDeallocation(size);
        
        if (config.enableDebugging) {
            stats.trackDeallocation(ptr);
        }
    }
}

void GPUMemoryPool::reset() {
    // Ensure we're on the correct device
    ensureCorrectDevice();
    
    if (config.threadSafe) {
        std::lock_guard<std::mutex> lock(mutex);
        allocator->reset();
        
        if (config.trackStats) {
            stats.reset();
        }
    } else {
        allocator->reset();
        
        if (config.trackStats) {
            stats.reset();
        }
    }
}

const MemoryStats& GPUMemoryPool::getStats() const {
    return stats;
}

MemoryType GPUMemoryPool::getMemoryType() const {
    return MemoryType::GPU;
}

std::string GPUMemoryPool::getName() const {
    return name;
}

const PoolConfig& GPUMemoryPool::getConfig() const {
    return config;
}

void GPUMemoryPool::setDevice(int deviceId) {
    this->deviceId = deviceId;
    setCurrentDevice(deviceId);
    
    if (allocator) {
        allocator->setDevice(deviceId);
    }
}

int GPUMemoryPool::getDevice() const {
    return deviceId;
}

void GPUMemoryPool::setStream(cudaStream_t stream) {
    this->stream = stream;
    
    if (allocator) {
        allocator->setStream(stream);
    }
}

cudaStream_t GPUMemoryPool::getStream() const {
    return stream;
}

void GPUMemoryPool::copyHostToDevice(void* dst, const void* src, size_t size) {
    if (dst == nullptr || src == nullptr || size == 0) {
        return;
    }
    
    // Ensure we're on the correct device
    ensureCorrectDevice();
    
    // Copy the data
    cudaMemcpyAsync(dst, src, size, true, stream);
    
    // Synchronize to ensure the copy is complete
    synchronizeStream(stream);
}

void GPUMemoryPool::copyDeviceToHost(void* dst, const void* src, size_t size) {
    if (dst == nullptr || src == nullptr || size == 0) {
        return;
    }
    
    // Ensure we're on the correct device
    ensureCorrectDevice();
    
    // Copy the data
    cudaMemcpyAsync(dst, src, size, false, stream);
    
    // Synchronize to ensure the copy is complete
    synchronizeStream(stream);
}

void GPUMemoryPool::copyDeviceToDevice(void* dst, const void* src, size_t size) {
    if (dst == nullptr || src == nullptr || size == 0) {
        return;
    }
    
    // Ensure we're on the correct device
    ensureCorrectDevice();
    
    // Copy the data (using our utility function)
    cudaMemcpyAsync(dst, src, size, false, stream);
    
    // Synchronize to ensure the copy is complete
    synchronizeStream(stream);
}

void GPUMemoryPool::ensureCorrectDevice() const {
    int currentDevice = getCurrentDevice();
    if (currentDevice != deviceId) {
        setCurrentDevice(deviceId);
    }
}

} // namespace memory_pool