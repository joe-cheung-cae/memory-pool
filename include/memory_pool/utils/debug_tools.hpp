#ifndef MEMORY_POOL_DEBUG_TOOLS_HPP
#define MEMORY_POOL_DEBUG_TOOLS_HPP

#include "../common.hpp"
#include "../stats/memory_stats.hpp"
#include <string>
#include <vector>
#include <map>
#include <functional>

namespace memory_pool {

// Memory leak detector
class MemoryLeakDetector {
public:
    // Get singleton instance
    static MemoryLeakDetector& getInstance();
    
    // Track allocation
    void trackAllocation(void* ptr, size_t size, const std::string& poolName);
    
    // Track deallocation
    void trackDeallocation(void* ptr, const std::string& poolName);
    
    // Check for leaks
    bool hasLeaks() const;
    
    // Get leak report
    std::string getLeakReport() const;
    
    // Reset tracking
    void reset();
    
    // Enable/disable tracking
    void setEnabled(bool enable);
    bool isEnabled() const;
    
private:
    MemoryLeakDetector();
    ~MemoryLeakDetector();
    
    // Prevent copying
    MemoryLeakDetector(const MemoryLeakDetector&) = delete;
    MemoryLeakDetector& operator=(const MemoryLeakDetector&) = delete;
    
    // Allocation tracking
    struct AllocationRecord {
        size_t size;
        std::string poolName;
        std::string stackTrace;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::map<void*, AllocationRecord> allocations;
    bool enabled;
    mutable std::mutex mutex;
    
    // Stack trace capture
    std::string captureStackTrace();
};

// Boundary checker for detecting buffer overflows
class BoundaryChecker {
public:
    // Get singleton instance
    static BoundaryChecker& getInstance();
    
    // Track allocation with boundary markers
    void* trackAllocation(void* ptr, size_t size);
    
    // Check boundaries and deallocate
    bool checkAndDeallocate(void* ptr);
    
    // Check all tracked allocations
    bool checkAll();
    
    // Get violation report
    std::string getViolationReport() const;
    
    // Enable/disable checking
    void setEnabled(bool enable);
    bool isEnabled() const;
    
private:
    BoundaryChecker();
    ~BoundaryChecker();
    
    // Prevent copying
    BoundaryChecker(const BoundaryChecker&) = delete;
    BoundaryChecker& operator=(const BoundaryChecker&) = delete;
    
    // Boundary marker pattern
    static constexpr uint32_t BOUNDARY_PATTERN = 0xFEEDFACE;
    static constexpr size_t BOUNDARY_SIZE = sizeof(uint32_t);
    
    // Allocation tracking
    struct BoundaryRecord {
        void* originalPtr;
        size_t size;
        std::string stackTrace;
    };
    
    std::map<void*, BoundaryRecord> trackedAllocations;
    bool enabled;
    mutable std::mutex mutex;
    
    // Helper methods
    bool checkBoundaries(void* ptr, const BoundaryRecord& record);
    void* addBoundaryMarkers(void* ptr, size_t size);
};

// Performance tracker for memory operations
class PerformanceTracker {
public:
    // Get singleton instance
    static PerformanceTracker& getInstance();
    
    // Track allocation time
    void trackAllocation(size_t size, double timeMs, const std::string& poolName);
    
    // Track deallocation time
    void trackDeallocation(size_t size, double timeMs, const std::string& poolName);
    
    // Get performance report
    std::string getPerformanceReport() const;
    
    // Reset tracking
    void reset();
    
    // Enable/disable tracking
    void setEnabled(bool enable);
    bool isEnabled() const;
    
private:
    PerformanceTracker();
    ~PerformanceTracker() = default;
    
    // Prevent copying
    PerformanceTracker(const PerformanceTracker&) = delete;
    PerformanceTracker& operator=(const PerformanceTracker&) = delete;
    
    // Performance metrics
    struct PoolMetrics {
        // Allocation metrics
        size_t totalAllocations = 0;
        double totalAllocationTimeMs = 0.0;
        double minAllocationTimeMs = std::numeric_limits<double>::max();
        double maxAllocationTimeMs = 0.0;
        
        // Deallocation metrics
        size_t totalDeallocations = 0;
        double totalDeallocationTimeMs = 0.0;
        double minDeallocationTimeMs = std::numeric_limits<double>::max();
        double maxDeallocationTimeMs = 0.0;
        
        // Size-based metrics
        std::map<size_t, double> sizeToTimeMap;
    };
    
    std::map<std::string, PoolMetrics> poolMetrics;
    bool enabled;
    mutable std::mutex mutex;
};

// Timer class for measuring operation duration
class Timer {
public:
    Timer();
    
    // Reset the timer
    void reset();
    
    // Get elapsed time in milliseconds
    double elapsedMs() const;
    
private:
    std::chrono::steady_clock::time_point startTime;
};

// Convenience functions
void enableDebugging(bool enable);
bool isDebuggingEnabled();
std::string getMemoryReport();

} // namespace memory_pool

#endif // MEMORY_POOL_DEBUG_TOOLS_HPP