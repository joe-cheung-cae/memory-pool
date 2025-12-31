#ifndef MEMORY_POOL_PLATFORM_UTILS_HPP
#define MEMORY_POOL_PLATFORM_UTILS_HPP

#include <string>

namespace memory_pool {
namespace utils {

/**
 * @brief Normalizes a file path to use the correct path separator for the current platform.
 * @param path The path to normalize.
 * @return The normalized path.
 */
std::string normalize_path(const std::string& path);

/**
 * @brief Gets the preferred temporary directory path for the current platform.
 * @return The temporary directory path.
 */
std::string get_temp_directory();

/**
 * @brief Checks if the current platform supports CUDA.
 * @return True if CUDA is supported, false otherwise.
 */
bool cuda_supported();

/**
 * @brief Checks if the current platform supports NUMA.
 * @return True if NUMA is supported, false otherwise.
 */
bool numa_supported();

/**
 * @brief Checks if the current platform supports persistent memory.
 * @return True if PMEM is supported, false otherwise.
 */
bool pmem_supported();

/**
 * @brief Checks if the current platform supports Valgrind.
 * @return True if Valgrind is supported, false otherwise.
 */
bool valgrind_supported();

}  // namespace utils
}  // namespace memory_pool

#endif  // MEMORY_POOL_PLATFORM_UTILS_HPP