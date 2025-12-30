#ifndef MEMORY_POOL_NUMA_UTILS_HPP
#define MEMORY_POOL_NUMA_UTILS_HPP

#include <cstddef>
#include <cstdint>

namespace memory_pool {

/**
 * @brief NUMA (Non-Uniform Memory Access) utility functions.
 *
 * This namespace provides utilities for NUMA-aware memory allocation
 * and topology detection.
 */
namespace numa_utils {

/**
 * @brief Check if NUMA support is available on this system.
 * @return True if NUMA is available, false otherwise.
 */
bool is_numa_available();

/**
 * @brief Get the number of NUMA nodes on the system.
 * @return Number of NUMA nodes, or 0 if NUMA is not available.
 */
int get_num_numa_nodes();

/**
 * @brief Get the NUMA node of the current CPU.
 * @return NUMA node ID, or -1 if NUMA is not available or cannot be determined.
 */
int get_current_numa_node();

/**
 * @brief Allocate memory on a specific NUMA node.
 * @param size The size in bytes to allocate.
 * @param alignment Memory alignment requirement (must be power of 2).
 * @param node The NUMA node to allocate on, or -1 for local allocation.
 * @return Pointer to allocated memory, or nullptr on failure.
 */
void* allocate_on_node(size_t size, size_t alignment, int node);

/**
 * @brief Deallocate memory allocated with allocate_on_node.
 * @param ptr Pointer to the memory to deallocate.
 */
void deallocate(void* ptr);

}  // namespace numa_utils

}  // namespace memory_pool

#endif  // MEMORY_POOL_NUMA_UTILS_HPP