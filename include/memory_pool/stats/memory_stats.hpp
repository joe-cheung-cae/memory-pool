#ifndef MEMORY_POOL_MEMORY_STATS_HPP
#define MEMORY_POOL_MEMORY_STATS_HPP

#include <cstddef>
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>
#include <string>

namespace memory_pool {

// Structure to hold allocation information for debugging
struct AllocationInfo {
    void*                                 ptr;
    size_t                                size;
    std::chrono::steady_clock::time_point timestamp;

    AllocationInfo(void* p, size_t s) : ptr(p), size(s), timestamp(std::chrono::steady_clock::now()) {}
};

// Memory statistics class
class MemoryStats {
  public:
    MemoryStats();
    ~MemoryStats() = default;

    // Record allocation/deallocation
    void recordAllocation(size_t size);
    void recordDeallocation(size_t size);

    // Debug tracking
    void trackAllocation(void* ptr, size_t size);
    void trackDeallocation(void* ptr);

    // Statistics getters
    size_t getTotalAllocated() const;
    size_t getCurrentUsed() const;
    size_t getPeakUsage() const;
    size_t getAllocationCount() const;
    size_t getDeallocationCount() const;
    double getFragmentationRatio() const;

    // Debug information
    std::vector<AllocationInfo> getActiveAllocations() const;
    bool                        hasMemoryLeaks() const;
    std::string                 getStatsString() const;

    // Enable/disable tracking
    void setTrackingEnabled(bool enable);
    bool isTrackingEnabled() const;

    // Reset statistics
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