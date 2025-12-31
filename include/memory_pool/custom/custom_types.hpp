#ifndef MEMORY_POOL_CUSTOM_CUSTOM_TYPES_HPP
#define MEMORY_POOL_CUSTOM_CUSTOM_TYPES_HPP

#include <string>
#include <unordered_map>
#include <variant>

namespace memory_pool {

/**
 * @brief Hardware type enumeration for custom allocators.
 */
enum class HardwareType {
    RDMA,      /**< Remote Direct Memory Access */
    FPGA,      /**< Field Programmable Gate Array */
    ASIC,      /**< Application Specific Integrated Circuit */
    Custom     /**< Generic custom hardware */
};

/**
 * @brief Configuration for RDMA hardware.
 */
struct RDMAConfig {
    std::string deviceName;     /**< RDMA device name (e.g., "mlx5_0") */
    std::string transportType;  /**< Transport type: "RoCE", "iWARP", "IB" */
    size_t      maxMemorySize;  /**< Maximum memory size for registration */
    bool        enableZeroCopy; /**< Enable zero-copy operations */
};

/**
 * @brief Configuration for FPGA hardware.
 */
struct FPGAConfig {
    std::string devicePath;     /**< Device file path (e.g., "/dev/fpga0") */
    size_t      memoryBankSize; /**< Size of FPGA memory bank */
    bool        enableDMA;      /**< Enable DMA transfers */
};

/**
 * @brief Configuration for ASIC hardware.
 */
struct ASICConfig {
    std::string devicePath;     /**< Device file path */
    size_t      memoryAlignment;/**< Required memory alignment */
    bool        enableCache;    /**< Enable hardware caching */
};

/**
 * @brief Configuration variant for different hardware types.
 */
using HardwareConfig = std::variant<
    std::monostate,  // For no specific config
    RDMAConfig,
    FPGAConfig,
    ASICConfig,
    std::unordered_map<std::string, std::string>  // Generic key-value config
>;

}  // namespace memory_pool

#endif  // MEMORY_POOL_CUSTOM_CUSTOM_TYPES_HPP