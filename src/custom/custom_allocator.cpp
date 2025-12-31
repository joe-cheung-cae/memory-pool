#include "memory_pool/custom/custom_allocator.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <stdexcept>

namespace memory_pool {

// Initialize the static registry
std::unordered_map<HardwareType,
                   std::unordered_map<std::string, CustomAllocatorFactory>>
    CustomAllocatorRegistry::registry;

void CustomAllocatorRegistry::registerAllocator(HardwareType hardwareType,
                                                CustomAllocatorFactory factory,
                                                const std::string& name) {
    registry[hardwareType][name] = std::move(factory);
}

std::unique_ptr<ICustomAllocator> CustomAllocatorRegistry::createAllocator(HardwareType hardwareType,
                                                                           const HardwareConfig& config) {
    auto hardwareIt = registry.find(hardwareType);
    if (hardwareIt == registry.end()) {
        throw InvalidOperationException("No allocators registered for hardware type: " +
                                       std::to_string(static_cast<int>(hardwareType)));
    }

    // Use the first registered allocator if no specific name is given
    // For now, we take the first one in the map
    if (!hardwareIt->second.empty()) {
        auto factoryIt = hardwareIt->second.begin();
        return factoryIt->second(config);
    }

    throw InvalidOperationException("No allocator implementations available for hardware type: " +
                                   std::to_string(static_cast<int>(hardwareType)));
}

std::vector<std::string> CustomAllocatorRegistry::getRegisteredAllocators(HardwareType hardwareType) {
    std::vector<std::string> names;
    auto hardwareIt = registry.find(hardwareType);
    if (hardwareIt != registry.end()) {
        for (const auto& pair : hardwareIt->second) {
            names.push_back(pair.first);
        }
    }
    return names;
}

bool CustomAllocatorRegistry::hasAllocator(HardwareType hardwareType) {
    auto hardwareIt = registry.find(hardwareType);
    return hardwareIt != registry.end() && !hardwareIt->second.empty();
}

}  // namespace memory_pool