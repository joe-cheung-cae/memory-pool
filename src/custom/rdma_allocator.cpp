#include "memory_pool/custom/rdma_allocator.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace memory_pool {

RDMAAllocator::RDMAAllocator(const RDMAConfig& config)
    : config(config), nextHandle(1), deviceAvailable(true), deviceName(config.deviceName) {
    // In a real implementation, this would initialize RDMA device context
    // For now, we simulate device availability
    if (config.deviceName.empty()) {
        deviceName = "mlx5_0";  // Default device
    }

    std::cout << "RDMA Allocator initialized for device: " << deviceName
              << " with transport: " << config.transportType << std::endl;
}

RDMAAllocator::~RDMAAllocator() {
    // Clean up all registered memory
    for (auto& pair : registeredHandles) {
        mockMemoryUnregistration(pair.first);
    }
    registeredHandles.clear();

    // Deallocate all memory
    for (auto& pair : allocatedBlocks) {
        std::free(pair.second.ptr);
    }
    allocatedBlocks.clear();
}

void* RDMAAllocator::allocate(size_t size) {
    if (!deviceAvailable) {
        throw InvalidOperationException("RDMA device not available");
    }

    // Align size to page boundary for RDMA
    size_t alignedSize = (size + 4095) & ~4095;  // 4KB alignment

    void* ptr = std::malloc(alignedSize);
    if (!ptr) {
        throw OutOfMemoryException("Failed to allocate RDMA memory");
    }

    // Initialize memory to zero if requested
    if (config.enableZeroCopy) {
        std::memset(ptr, 0, alignedSize);
    }

    // Track the allocation
    MemoryBlock block{ptr, alignedSize, 0, false};
    allocatedBlocks[ptr] = block;

    return ptr;
}

void RDMAAllocator::deallocate(void* ptr) {
    if (!ptr) return;

    auto it = allocatedBlocks.find(ptr);
    if (it == allocatedBlocks.end()) {
        throw InvalidPointerException("Pointer not allocated by this RDMA allocator");
    }

    // Unregister if registered
    if (it->second.registered) {
        unregisterMemory(it->second.handle);
    }

    std::free(ptr);
    allocatedBlocks.erase(it);
}

void RDMAAllocator::reset() {
    // Unregister all memory
    for (auto& pair : registeredHandles) {
        mockMemoryUnregistration(pair.first);
    }
    registeredHandles.clear();

    // Deallocate all memory
    for (auto& pair : allocatedBlocks) {
        std::free(pair.second.ptr);
    }
    allocatedBlocks.clear();
}

size_t RDMAAllocator::getBlockSize(void* ptr) const {
    auto it = allocatedBlocks.find(ptr);
    if (it == allocatedBlocks.end()) {
        return 0;
    }
    return it->second.size;
}

bool RDMAAllocator::owns(void* ptr) const {
    return allocatedBlocks.find(ptr) != allocatedBlocks.end();
}

HardwareType RDMAAllocator::getHardwareType() const {
    return HardwareType::RDMA;
}

uint64_t RDMAAllocator::registerMemory(void* ptr, size_t size) {
    if (!owns(ptr)) {
        throw InvalidPointerException("Pointer not owned by this allocator");
    }

    auto it = allocatedBlocks.find(ptr);
    if (it->second.registered) {
        return it->second.handle;  // Already registered
    }

    uint64_t handle = generateHandle();
    if (mockMemoryRegistration(ptr, size)) {
        it->second.handle = handle;
        it->second.registered = true;
        registeredHandles[handle] = ptr;
        return handle;
    }

    throw InvalidOperationException("Failed to register memory with RDMA device");
}

void RDMAAllocator::unregisterMemory(uint64_t handle) {
    auto it = registeredHandles.find(handle);
    if (it == registeredHandles.end()) {
        throw InvalidOperationException("Invalid registration handle");
    }

    void* ptr = it->second;  // it->second is the pointer, it->first is the handle
    auto blockIt = allocatedBlocks.find(ptr);
    if (blockIt != allocatedBlocks.end()) {
        mockMemoryUnregistration(handle);
        blockIt->second.registered = false;
        blockIt->second.handle = 0;
    }

    registeredHandles.erase(it);
}

void RDMAAllocator::synchronize(void* ptr, size_t size) {
    // In a real implementation, this would ensure RDMA operations are complete
    // For simulation, we just do a memory barrier
    if (owns(ptr)) {
        // Simulate synchronization delay
        std::cout << "Synchronizing RDMA memory region of " << size << " bytes" << std::endl;
    }
}

std::unordered_map<std::string, std::string> RDMAAllocator::getHardwareInfo(void* ptr) const {
    std::unordered_map<std::string, std::string> info;

    if (owns(ptr)) {
        auto it = allocatedBlocks.find(ptr);
        info["device"] = deviceName;
        info["transport"] = config.transportType;
        info["registered"] = it->second.registered ? "true" : "false";
        info["size"] = std::to_string(it->second.size);
        info["handle"] = std::to_string(it->second.handle);
    }

    return info;
}

bool RDMAAllocator::isHardwareAvailable() const {
    return deviceAvailable;
}

uint64_t RDMAAllocator::generateHandle() {
    return nextHandle++;
}

bool RDMAAllocator::mockMemoryRegistration(void* ptr, size_t size) {
    // Simulate RDMA memory registration
    // In real implementation, this would call ibv_reg_mr or similar
    std::cout << "Mock registering " << size << " bytes at " << ptr
              << " with RDMA device " << deviceName << std::endl;
    return true;
}

void RDMAAllocator::mockMemoryUnregistration(uint64_t handle) {
    // Simulate RDMA memory unregistration
    std::cout << "Mock unregistering memory with handle " << handle << std::endl;
}

// Factory function for RDMA allocator
std::unique_ptr<ICustomAllocator> createRDMAAllocator(const HardwareConfig& config) {
    if (std::holds_alternative<RDMAConfig>(config)) {
        return std::make_unique<RDMAAllocator>(std::get<RDMAConfig>(config));
    }
    throw InvalidOperationException("Invalid config type for RDMA allocator");
}

}  // namespace memory_pool