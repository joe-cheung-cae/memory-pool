#ifndef MEMORY_POOL_MEMORY_STATS_HPP
#define MEMORY_POOL_MEMORY_STATS_HPP

#include <cstddef>
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>
#include <string>

namespace memory_pool {

/**
 * @brief Structure to hold information about individual allocations for debugging.
 */
struct AllocationInfo {
    void*                                 ptr;       /**< Pointer to the allocated memory */
    size_t                                size;      /**< Size of the allocation in bytes */
    std::chrono::steady_clock::time_point timestamp; /**< Time when the allocation was made */

    /**
     * @brief Constructor for AllocationInfo.
     * @param p Pointer to the allocated memory.
     * @param s Size of the allocation.
     */
    AllocationInfo(void* p, size_t s) : ptr(p), size(s), timestamp(std::chrono::steady_clock::now()) {}
};

/**
 * @brief Class for tracking memory allocation statistics and debugging information.
 *
 * This class provides comprehensive memory usage tracking, including allocation counts,
 * memory usage statistics, and debugging features like leak detection.
 */
class MemoryStats {
  public:
    /** @brief Default constructor */
    MemoryStats();
    /** @brief Destructor */
    ~MemoryStats() = default;

    /**
     * @brief Records a memory allocation.
     * @param size The size of the allocation in bytes.
     */
    void recordAllocation(size_t size);

    /**
     * @brief Records a memory deallocation.
     * @param size The size of the deallocation in bytes.
     */
    void recordDeallocation(size_t size);

    /**
     * @brief Tracks an individual allocation for debugging.
     * @param ptr Pointer to the allocated memory.
     * @param size Size of the allocation.
     */
    void trackAllocation(void* ptr, size_t size);

    /**
     * @brief Tracks a deallocation for debugging.
     * @param ptr Pointer to the memory being deallocated.
     */
    void trackDeallocation(void* ptr);

    /** @brief Gets the total amount of memory allocated. */
    size_t getTotalAllocated() const;
    /** @brief Gets the current amount of memory in use. */
    size_t getCurrentUsed() const;
    /** @brief Gets the peak memory usage. */
    size_t getPeakUsage() const;
    /** @brief Gets the total number of allocations. */
    size_t getAllocationCount() const;
    /** @brief Gets the total number of deallocations. */
    size_t getDeallocationCount() const;
    /** @brief Gets the memory fragmentation ratio. */
    double getFragmentationRatio() const;

    /** @brief Gets a list of currently active allocations. */
    std::vector<AllocationInfo> getActiveAllocations() const;
    /** @brief Checks if there are any memory leaks. */
    bool                        hasMemoryLeaks() const;
    /** @brief Gets a formatted string with statistics. */
    std::string                 getStatsString() const;

    /**
     * @brief Enables or disables detailed allocation tracking.
     * @param enable True to enable tracking, false to disable.
     */
    void setTrackingEnabled(bool enable);

    /** @brief Checks if tracking is currently enabled. */
    bool isTrackingEnabled() const;

    /** @brief Resets all statistics to zero. */
    void reset();

  private:
    // Basic statistics
    std::atomic<size_t> totalAllocated;
    std::atomic<size_t> currentUsed;
    std::atomic<size_t> peakUsage;
    std::atomic<size_t> allocationCount;
    std::atomic<size_t> deallocationCount;

    // Debug tracking
    bool                        trackingEnabled;
    mutable std::mutex          trackingMutex;
    std::vector<AllocationInfo> activeAllocations;
};

}  // namespace memory_pool

#endif  // MEMORY_POOL_MEMORY_STATS_HPP