#ifndef MEMORY_POOL_CUSTOM_RDMA_ALLOCATOR_HPP
#define MEMORY_POOL_CUSTOM_RDMA_ALLOCATOR_HPP

#include "memory_pool/custom/custom_allocator.hpp"
#include <unordered_map>
#include <memory>

namespace memory_pool {

/**
 * @brief RDMA memory allocator implementation.
 *
 * This allocator provides memory management for RDMA (Remote Direct Memory Access)
 * operations, supporting both RoCE and iWARP transports. It handles memory
 * registration with RDMA devices for efficient zero-copy operations.
 */
class RDMAAllocator : public ICustomAllocator {
public:
    /**
     * @brief Constructs an RDMA allocator with the given configuration.
     * @param config The RDMA configuration.
     */
    explicit RDMAAllocator(const RDMAConfig& config);

    /**
     * @brief Destroys the RDMA allocator.
     */
    ~RDMAAllocator() override;

    // IAllocator interface
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void reset() override;
    size_t getBlockSize(void* ptr) const override;
    bool owns(void* ptr) const override;

    // ICustomAllocator interface
    HardwareType getHardwareType() const override;
    uint64_t registerMemory(void* ptr, size_t size) override;
    void unregisterMemory(uint64_t handle) override;
    void synchronize(void* ptr, size_t size) override;
    std::unordered_map<std::string, std::string> getHardwareInfo(void* ptr) const override;
    bool isHardwareAvailable() const override;

private:
    RDMAConfig config;

    // Memory tracking
    struct MemoryBlock {
        void* ptr;
        size_t size;
        uint64_t handle;  // Mock handle for registration
        bool registered;
    };

    std::unordered_map<void*, MemoryBlock> allocatedBlocks;
    std::unordered_map<uint64_t, void*> registeredHandles;

    uint64_t nextHandle;

    // Mock RDMA device state
    bool deviceAvailable;
    std::string deviceName;

    // Helper methods
    uint64_t generateHandle();
    bool mockMemoryRegistration(void* ptr, size_t size);
    void mockMemoryUnregistration(uint64_t handle);
};

    // Factory function declaration
    std::unique_ptr<ICustomAllocator> createRDMAAllocator(const HardwareConfig& config);

}  // namespace memory_pool

#endif  // MEMORY_POOL_CUSTOM_RDMA_ALLOCATOR_HPP