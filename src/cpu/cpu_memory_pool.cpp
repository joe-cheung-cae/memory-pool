#include "memory_pool/cpu/cpu_memory_pool.hpp"
#include "memory_pool/cpu/fixed_size_allocator.hpp"
#include "memory_pool/cpu/variable_size_allocator.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <cstring>

namespace memory_pool {

CPUMemoryPool::CPUMemoryPool(const std::string& name, const PoolConfig& config)
    : name(name), config(config), allocator(nullptr) {
    initialize();
}

CPUMemoryPool::~CPUMemoryPool() {
    // The allocator will be automatically destroyed by the unique_ptr
}

void CPUMemoryPool::initialize() {
    // Create the appropriate allocator based on the configuration
    if (config.allocatorType == AllocatorType::FixedSize) {
        allocator = std::make_unique<FixedSizeAllocator>(
            config.blockSize,
            config.initialSize / config.blockSize,
            config.alignment
        );
    } else {
        allocator = std::make_unique<VariableSizeAllocator>(
            config.initialSize,
            config.alignment
        );
    }
}

void* CPUMemoryPool::allocate(size_t size) {
    return allocateInternal(size, AllocFlags::None);
}

void* CPUMemoryPool::allocate(size_t size, AllocFlags flags) {
    return allocateInternal(size, flags);
}

void* CPUMemoryPool::allocateInternal(size_t size, AllocFlags flags) {
    if (size == 0) {
        return nullptr;
    }
    
    void* ptr = nullptr;
    
    // Use thread safety if configured
    if (config.threadSafe) {
        std::lock_guard<std::mutex> lock(mutex);
        ptr = allocator->allocate(size);
    } else {
        ptr = allocator->allocate(size);
    }
    
    if (ptr == nullptr) {
        throw OutOfMemoryException("Failed to allocate memory in CPU pool: " + name);
    }
    
    // Zero memory if requested
    if (has_flag(flags, AllocFlags::ZeroMemory)) {
        std::memset(ptr, 0, size);
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

void CPUMemoryPool::deallocate(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    
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

void CPUMemoryPool::reset() {
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

const MemoryStats& CPUMemoryPool::getStats() const {
    return stats;
}

MemoryType CPUMemoryPool::getMemoryType() const {
    return MemoryType::CPU;
}

std::string CPUMemoryPool::getName() const {
    return name;
}

const PoolConfig& CPUMemoryPool::getConfig() const {
    return config;
}

} // namespace memory_pool