#include "memory_pool/custom/custom_memory_pool.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <stdexcept>

namespace memory_pool {

CustomMemoryPool::CustomMemoryPool(const std::string& name, const PoolConfig& config,
                                   std::unique_ptr<ICustomAllocator> allocator)
    : name(name), config(config), allocator(std::move(allocator)) {
    if (!this->allocator) {
        throw InvalidOperationException("CustomMemoryPool: allocator cannot be null");
    }
    initialize();
}

CustomMemoryPool::~CustomMemoryPool() {
    // Custom allocators will be automatically cleaned up
}

void* CustomMemoryPool::allocate(size_t size) {
    return allocateInternal(size, AllocFlags::None);
}

void CustomMemoryPool::deallocate(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);
    allocator->deallocate(ptr);

    // Record deallocation with actual size
    if (config.trackStats) {
        auto it = allocatedSizes.find(ptr);
        if (it != allocatedSizes.end()) {
            stats.recordDeallocation(it->second);
            allocatedSizes.erase(it);
        }
    }
}

void* CustomMemoryPool::allocate(size_t size, AllocFlags flags) {
    return allocateInternal(size, flags);
}

void CustomMemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mutex);
    allocator->reset();
    stats.reset();
    allocatedSizes.clear();
}

const MemoryStats& CustomMemoryPool::getStats() const {
    return stats;
}

MemoryType CustomMemoryPool::getMemoryType() const {
    return MemoryType::Custom;
}

std::string CustomMemoryPool::getName() const {
    return name;
}

const PoolConfig& CustomMemoryPool::getConfig() const {
    return config;
}

uint64_t CustomMemoryPool::registerMemory(void* ptr, size_t size) {
    return allocator->registerMemory(ptr, size);
}

void CustomMemoryPool::unregisterMemory(uint64_t handle) {
    allocator->unregisterMemory(handle);
}

void CustomMemoryPool::synchronize(void* ptr, size_t size) {
    allocator->synchronize(ptr, size);
}

std::unordered_map<std::string, std::string> CustomMemoryPool::getHardwareInfo(void* ptr) const {
    return allocator->getHardwareInfo(ptr);
}

void CustomMemoryPool::initialize() {
    // Custom pools don't need initialization like CPU/GPU pools
    // The allocator is already provided
}

void* CustomMemoryPool::allocateInternal(size_t size, AllocFlags /*flags*/) {
    std::lock_guard<std::mutex> lock(mutex);
    void* ptr = allocator->allocate(size);
    if (ptr) {
        if (config.trackStats) {
            stats.recordAllocation(size);
            allocatedSizes[ptr] = size;
        }
    }
    return ptr;
}

}  // namespace memory_pool