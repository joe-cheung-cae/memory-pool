#include "memory_pool/stats/memory_stats.hpp"
#include <sstream>
#include <algorithm>

namespace memory_pool {

MemoryStats::MemoryStats()
    : totalAllocated(0),
      currentUsed(0),
      peakUsage(0),
      allocationCount(0),
      deallocationCount(0),
      trackingEnabled(false) {}

void MemoryStats::recordAllocation(size_t size) {
    totalAllocated += size;
    currentUsed += size;

    // Update peak usage if current usage is higher
    size_t current = currentUsed.load();
    size_t peak    = peakUsage.load();
    while (current > peak && !peakUsage.compare_exchange_weak(peak, current)) {
        // If compare_exchange_weak fails, peak is updated with the current value
        // and we need to check again
        current = currentUsed.load();
    }

    allocationCount++;
}

void MemoryStats::recordDeallocation(size_t size) {
    // Ensure we don't go below zero
    size_t current = currentUsed.load();
    while (current >= size && !currentUsed.compare_exchange_weak(current, current - size)) {
        current = currentUsed.load();
    }

    deallocationCount++;
}

void MemoryStats::trackAllocation(void* ptr, size_t size) {
    if (!trackingEnabled) {
        return;
    }

    std::lock_guard<std::mutex> lock(trackingMutex);
    activeAllocations.emplace_back(ptr, size);
}

void MemoryStats::trackDeallocation(void* ptr) {
    if (!trackingEnabled) {
        return;
    }

    std::lock_guard<std::mutex> lock(trackingMutex);
    auto                        it = std::find_if(activeAllocations.begin(), activeAllocations.end(),
                                                  [ptr](const AllocationInfo& info) { return info.ptr == ptr; });

    if (it != activeAllocations.end()) {
        activeAllocations.erase(it);
    }
}

size_t MemoryStats::getTotalAllocated() const { return totalAllocated.load(); }

size_t MemoryStats::getCurrentUsed() const { return currentUsed.load(); }

size_t MemoryStats::getPeakUsage() const { return peakUsage.load(); }

size_t MemoryStats::getAllocationCount() const { return allocationCount.load(); }

size_t MemoryStats::getDeallocationCount() const { return deallocationCount.load(); }

double MemoryStats::getFragmentationRatio() const {
    // A simple fragmentation metric:
    // If we've allocated and deallocated a lot, but still have high memory usage,
    // fragmentation might be high
    size_t allocCount   = allocationCount.load();
    size_t deallocCount = deallocationCount.load();

    if (allocCount == 0 || deallocCount == 0) {
        return 0.0;
    }

    // Calculate the ratio of deallocations to allocations
    double deallocRatio = static_cast<double>(deallocCount) / allocCount;

    // Calculate the ratio of current memory to peak memory
    size_t current     = currentUsed.load();
    size_t peak        = peakUsage.load();
    double memoryRatio = peak > 0 ? static_cast<double>(current) / peak : 0.0;

    // Fragmentation is high when we've deallocated a lot but still use a lot of memory
    return deallocRatio * memoryRatio;
}

std::vector<AllocationInfo> MemoryStats::getActiveAllocations() const {
    if (!trackingEnabled) {
        return {};
    }

    std::lock_guard<std::mutex> lock(trackingMutex);
    return activeAllocations;
}

bool MemoryStats::hasMemoryLeaks() const {
    if (!trackingEnabled) {
        return false;
    }

    std::lock_guard<std::mutex> lock(trackingMutex);
    return !activeAllocations.empty();
}

std::string MemoryStats::getStatsString() const {
    std::ostringstream oss;
    oss << "Memory Statistics:\n"
        << "  Total Allocated: " << getTotalAllocated() << " bytes\n"
        << "  Current Used: " << getCurrentUsed() << " bytes\n"
        << "  Peak Usage: " << getPeakUsage() << " bytes\n"
        << "  Allocation Count: " << getAllocationCount() << "\n"
        << "  Deallocation Count: " << getDeallocationCount() << "\n"
        << "  Fragmentation Ratio: " << getFragmentationRatio() << "\n";

    if (trackingEnabled) {
        std::lock_guard<std::mutex> lock(trackingMutex);
        oss << "  Active Allocations: " << activeAllocations.size() << "\n";
    }

    return oss.str();
}

void MemoryStats::setTrackingEnabled(bool enable) {
    std::lock_guard<std::mutex> lock(trackingMutex);

    if (enable && !trackingEnabled) {
        // Clear existing allocations when enabling
        activeAllocations.clear();
    }

    trackingEnabled = enable;
}

bool MemoryStats::isTrackingEnabled() const { return trackingEnabled; }

void MemoryStats::reset() {
    totalAllocated    = 0;
    currentUsed       = 0;
    peakUsage         = 0;
    allocationCount   = 0;
    deallocationCount = 0;

    if (trackingEnabled) {
        std::lock_guard<std::mutex> lock(trackingMutex);
        activeAllocations.clear();
    }
}

}  // namespace memory_pool