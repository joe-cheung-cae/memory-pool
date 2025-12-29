#ifndef MEMORY_POOL_COMMON_HPP
#define MEMORY_POOL_COMMON_HPP

#include <cstddef>
#include <string>
#include <stdexcept>

namespace memory_pool {

// Forward declarations
class MemoryStats;

/**
 * @brief Base exception class for memory pool errors.
 */
class MemoryPoolException : public std::runtime_error {
  public:
    /**
     * @brief Constructs a memory pool exception with a message.
     * @param message The error message.
     */
    explicit MemoryPoolException(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Exception thrown when memory allocation fails due to insufficient memory.
 */
class OutOfMemoryException : public MemoryPoolException {
  public:
    /**
     * @brief Constructs an out-of-memory exception with a message.
     * @param message The error message.
     */
    explicit OutOfMemoryException(const std::string& message) : MemoryPoolException(message) {}
};

/**
 * @brief Exception thrown for invalid operations on memory pools.
 */
class InvalidOperationException : public MemoryPoolException {
  public:
    /**
     * @brief Constructs an invalid operation exception with a message.
     * @param message The error message.
     */
    explicit InvalidOperationException(const std::string& message) : MemoryPoolException(message) {}
};

/**
 * @brief Exception thrown when an invalid pointer is used.
 */
class InvalidPointerException : public MemoryPoolException {
  public:
    /**
     * @brief Constructs an invalid pointer exception with a message.
     * @param message The error message.
     */
    explicit InvalidPointerException(const std::string& message) : MemoryPoolException(message) {}
};

/**
 * @brief Default memory alignment in bytes.
 */
constexpr size_t DEFAULT_ALIGNMENT = 16;

/**
 * @brief Aligns a size to the specified alignment.
 * @param size The size to align.
 * @param alignment The alignment boundary (default: DEFAULT_ALIGNMENT).
 * @return The aligned size.
 */
inline size_t align_size(size_t size, size_t alignment = DEFAULT_ALIGNMENT) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Enumeration for memory types.
 */
enum class MemoryType { CPU, GPU };

/**
 * @brief Flags for memory allocation options.
 */
enum class AllocFlags { None = 0, ZeroMemory = 1 << 0, Pinned = 1 << 1, Managed = 1 << 2 };

/**
 * @brief Bitwise OR operator for AllocFlags.
 * @param a First flag.
 * @param b Second flag.
 * @return Combined flags.
 */
inline AllocFlags operator|(AllocFlags a, AllocFlags b) {
    return static_cast<AllocFlags>(static_cast<int>(a) | static_cast<int>(b));
}

/**
 * @brief Bitwise AND operator for AllocFlags.
 * @param a First flag.
 * @param b Second flag.
 * @return Intersection of flags.
 */
inline AllocFlags operator&(AllocFlags a, AllocFlags b) {
    return static_cast<AllocFlags>(static_cast<int>(a) & static_cast<int>(b));
}

/**
 * @brief Checks if a flag is set in the flags.
 * @param flags The flags to check.
 * @param flag The flag to test.
 * @return True if the flag is set.
 */
inline bool has_flag(AllocFlags flags, AllocFlags flag) { return (flags & flag) == flag; }

}  // namespace memory_pool

#endif  // MEMORY_POOL_COMMON_HPP