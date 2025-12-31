#ifndef MEMORY_POOL_CUSTOM_CUSTOM_ALLOCATOR_HPP
#define MEMORY_POOL_CUSTOM_CUSTOM_ALLOCATOR_HPP

#include "memory_pool/cpu/cpu_memory_pool.hpp"
#include "memory_pool/custom/custom_types.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace memory_pool {

/**
 * @brief Extended allocator interface for specialized hardware.
 *
 * This interface extends the basic IAllocator with hardware-specific
 * operations like memory registration, hardware synchronization, and
 * device-specific memory management.
 */
class ICustomAllocator : public IAllocator {
public:
    /**
     * @brief Gets the hardware type supported by this allocator.
     * @return The hardware type.
     */
    virtual HardwareType getHardwareType() const = 0;

    /**
     * @brief Registers memory for hardware access.
     *
     * For RDMA, this registers the memory region with the RDMA device.
     * For accelerators, this may pin memory or set up DMA mappings.
     *
     * @param ptr Pointer to the memory to register.
     * @param size Size of the memory region.
     * @return Hardware-specific handle or key for the registered memory.
     */
    virtual uint64_t registerMemory(void* ptr, size_t size) = 0;

    /**
     * @brief Unregisters previously registered memory.
     *
     * @param handle The registration handle returned by registerMemory.
     */
    virtual void unregisterMemory(uint64_t handle) = 0;

    /**
     * @brief Synchronizes memory operations with hardware.
     *
     * This ensures that all pending hardware operations on the memory
     * are completed before returning.
     *
     * @param ptr Pointer to the memory region.
     * @param size Size of the region to synchronize.
     */
    virtual void synchronize(void* ptr, size_t size) = 0;

    /**
     * @brief Gets hardware-specific memory information.
     *
     * @param ptr Pointer to allocated memory.
     * @return Hardware-specific information as key-value pairs.
     */
    virtual std::unordered_map<std::string, std::string> getHardwareInfo(void* ptr) const = 0;

    /**
     * @brief Checks if the hardware is available and operational.
     * @return True if hardware is available.
     */
    virtual bool isHardwareAvailable() const = 0;
};

/**
 * @brief Factory function type for creating custom allocators.
 */
using CustomAllocatorFactory = std::function<std::unique_ptr<ICustomAllocator>(const HardwareConfig& config)>;

/**
 * @brief Registry for custom allocator factories.
 *
 * This class manages the registration and creation of custom allocators
 * for different hardware types.
 */
class CustomAllocatorRegistry {
public:
    /**
     * @brief Registers a custom allocator factory.
     *
     * @param hardwareType The hardware type.
     * @param factory The factory function.
     * @param name Optional name for the allocator implementation.
     */
    static void registerAllocator(HardwareType hardwareType,
                                  CustomAllocatorFactory factory,
                                  const std::string& name = "");

    /**
     * @brief Creates a custom allocator instance.
     *
     * @param hardwareType The hardware type.
     * @param config The hardware configuration.
     * @return Unique pointer to the created allocator.
     */
    static std::unique_ptr<ICustomAllocator> createAllocator(HardwareType hardwareType,
                                                             const HardwareConfig& config);

    /**
     * @brief Gets the list of registered allocator names for a hardware type.
     *
     * @param hardwareType The hardware type.
     * @return Vector of registered allocator names.
     */
    static std::vector<std::string> getRegisteredAllocators(HardwareType hardwareType);

    /**
     * @brief Checks if a hardware type has registered allocators.
     *
     * @param hardwareType The hardware type.
     * @return True if allocators are registered.
     */
    static bool hasAllocator(HardwareType hardwareType);

private:
    static std::unordered_map<HardwareType,
                              std::unordered_map<std::string, CustomAllocatorFactory>> registry;
};

}  // namespace memory_pool

#endif  // MEMORY_POOL_CUSTOM_CUSTOM_ALLOCATOR_HPP