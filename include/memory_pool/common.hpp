#ifndef MEMORY_POOL_COMMON_HPP
#define MEMORY_POOL_COMMON_HPP

#include <cstddef>
#include <string>
#include <stdexcept>

namespace memory_pool {

// Forward declarations
class MemoryStats;

// Memory pool exceptions
class MemoryPoolException : public std::runtime_error {
public:
    explicit MemoryPoolException(const std::string& message)
        : std::runtime_error(message) {}
};

class OutOfMemoryException : public MemoryPoolException {
public:
    explicit OutOfMemoryException(const std::string& message)
        : MemoryPoolException(message) {}
};

class InvalidOperationException : public MemoryPoolException {
public:
    explicit InvalidOperationException(const std::string& message)
        : MemoryPoolException(message) {}
};

class InvalidPointerException : public MemoryPoolException {
public:
    explicit InvalidPointerException(const std::string& message)
        : MemoryPoolException(message) {}
};

// Memory alignment utilities
constexpr size_t DEFAULT_ALIGNMENT = 16;

inline size_t align_size(size_t size, size_t alignment = DEFAULT_ALIGNMENT) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Memory pool types
enum class MemoryType {
    CPU,
    GPU
};

// Memory allocation flags
enum class AllocFlags {
    None = 0,
    ZeroMemory = 1 << 0,
    Pinned = 1 << 1,
    Managed = 1 << 2
};

inline AllocFlags operator|(AllocFlags a, AllocFlags b) {
    return static_cast<AllocFlags>(static_cast<int>(a) | static_cast<int>(b));
}

inline AllocFlags operator&(AllocFlags a, AllocFlags b) {
    return static_cast<AllocFlags>(static_cast<int>(a) & static_cast<int>(b));
}

inline bool has_flag(AllocFlags flags, AllocFlags flag) {
    return (flags & flag) == flag;
}

} // namespace memory_pool

#endif // MEMORY_POOL_COMMON_HPP