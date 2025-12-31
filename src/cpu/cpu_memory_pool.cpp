#include "memory_pool/cpu/cpu_memory_pool.hpp"
#include "memory_pool/cpu/fixed_size_allocator.hpp"
#include "memory_pool/cpu/variable_size_allocator.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include "memory_pool/utils/debug_tools.hpp"
#include <cstring>

namespace memory_pool {

// Constructor
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
        allocator = std::make_unique<FixedSizeAllocator>(config.blockSize, config.initialSize / config.blockSize,
                                                          config.alignment, config.syncType == SyncType::LockFree);
    } else {
        allocator = std::make_unique<VariableSizeAllocator>(config.initialSize, config.alignment);
    }

    // Enable debugging tools if configured
    if (config.enableDebugging) {
        enableDebugging(true);
    }
}

void* CPUMemoryPool::allocate(size_t size) { return allocateInternal(size, AllocFlags::None); }

void* CPUMemoryPool::allocate(size_t size, AllocFlags flags) { return allocateInternal(size, flags); }

void* CPUMemoryPool::allocateInternal(size_t size, AllocFlags flags) {
    if (size == 0) {
        return nullptr;
    }

    void* ptr = nullptr;
    size_t actualSize = size;

    // If boundary checking is enabled, allocate extra space for canaries
    if (BoundaryChecker::getInstance().isEnabled()) {
        actualSize += 2 * BoundaryChecker::BOUNDARY_SIZE;
    }

    // Use thread safety if configured
    if (config.threadSafe) {
        std::lock_guard<std::mutex> lock(mutex);
        ptr = allocator->allocate(actualSize);
    } else {
        ptr = allocator->allocate(actualSize);
    }

    if (ptr == nullptr) {
        throw OutOfMemoryException("Failed to allocate memory in CPU pool: " + name);
    }

    // If boundary checking is enabled, add canary markers
    if (BoundaryChecker::getInstance().isEnabled()) {
        ptr = BoundaryChecker::getInstance().trackAllocation(ptr, size);
    }

    // Zero memory if requested (only the user data, not canaries)
    if (has_flag(flags, AllocFlags::ZeroMemory)) {
        std::memset(ptr, 0, size);
    }

    // Track statistics if enabled
    if (config.trackStats) {
        stats.recordAllocation(size);

        if (config.enableDebugging) {
            stats.trackAllocation(ptr, size);
            MemoryLeakDetector::getInstance().trackAllocation(ptr, size, name);
        }
    }

    return ptr;
}

// Deallocate memory
void CPUMemoryPool::deallocate(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    // Handle boundary checking
    bool boundaryEnabled = BoundaryChecker::getInstance().isEnabled();
    void* trackPtr = ptr;  // User ptr for tracking

    if (boundaryEnabled) {
        if (!BoundaryChecker::getInstance().checkAndDeallocate(ptr)) {
            // Boundary violation detected, but continue with deallocation
        }
        // Adjust ptr back to the original allocated ptr for allocator
        ptr = static_cast<char*>(ptr) - BoundaryChecker::BOUNDARY_SIZE;
    }

    size_t allocatedSize = 0;
    size_t userSize = 0;

    // Use thread safety if configured
    if (config.threadSafe) {
        std::lock_guard<std::mutex> lock(mutex);

        // Get the allocated size before deallocating for statistics
        if (config.trackStats) {
            allocatedSize = allocator->getBlockSize(ptr);
            userSize = boundaryEnabled ? allocatedSize - 2 * BoundaryChecker::BOUNDARY_SIZE : allocatedSize;
        }

        allocator->deallocate(ptr);
    } else {
        // Get the allocated size before deallocating for statistics
        if (config.trackStats) {
            allocatedSize = allocator->getBlockSize(ptr);
            userSize = boundaryEnabled ? allocatedSize - 2 * BoundaryChecker::BOUNDARY_SIZE : allocatedSize;
        }

        allocator->deallocate(ptr);
    }

    // Track statistics if enabled
    if (config.trackStats) {
        stats.recordDeallocation(userSize);

        if (config.enableDebugging) {
            stats.trackDeallocation(trackPtr);
            MemoryLeakDetector::getInstance().trackDeallocation(trackPtr, name);
        }
    }
}

// Reset the memory pool
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

const MemoryStats& CPUMemoryPool::getStats() const { return stats; }

MemoryType CPUMemoryPool::getMemoryType() const { return MemoryType::CPU; }

std::string CPUMemoryPool::getName() const { return name; }

const PoolConfig& CPUMemoryPool::getConfig() const { return config; }

}  // namespace memory_pool