#include "memory_pool/utils/debug_tools.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <sstream>
#include <iomanip>
#include <cstring>
#include <execinfo.h>
#include <cxxabi.h>

namespace memory_pool {

// Memory Leak Detector implementation
MemoryLeakDetector& MemoryLeakDetector::getInstance() {
    static MemoryLeakDetector instance;
    return instance;
}

MemoryLeakDetector::MemoryLeakDetector() : enabled(false) {}

MemoryLeakDetector::~MemoryLeakDetector() {
    if (!allocations.empty()) {
        reportWarning("Memory leaks detected at program exit: " + getLeakReport());
    }
}

void MemoryLeakDetector::trackAllocation(void* ptr, size_t size, const std::string& poolName) {
    if (!enabled || ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);

    AllocationRecord record;
    record.size       = size;
    record.poolName   = poolName;
    record.stackTrace = captureStackTrace();
    record.timestamp  = std::chrono::steady_clock::now();

    allocations[ptr] = record;
}

void MemoryLeakDetector::trackDeallocation(void* ptr, const std::string& poolName) {
    if (!enabled || ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);

    auto it = allocations.find(ptr);
    if (it != allocations.end()) {
        // Check if the pool name matches
        if (it->second.poolName != poolName) {
            reportWarning("Memory allocated in pool '" + it->second.poolName + "' but deallocated in pool '" +
                          poolName + "'");
        }

        allocations.erase(it);
    } else {
        reportWarning("Attempted to deallocate untracked memory at " +
                      std::to_string(reinterpret_cast<uintptr_t>(ptr)));
    }
}

bool MemoryLeakDetector::hasLeaks() const {
    if (!enabled) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex);
    return !allocations.empty();
}

std::string MemoryLeakDetector::getLeakReport() const {
    if (!enabled) {
        return "Memory leak detection is disabled";
    }

    std::lock_guard<std::mutex> lock(mutex);

    if (allocations.empty()) {
        return "No memory leaks detected";
    }

    std::ostringstream oss;
    oss << "Memory Leak Report:\n";
    oss << "-------------------\n";
    oss << "Total leaks: " << allocations.size() << "\n\n";

    size_t totalLeakedBytes = 0;

    for (const auto& pair : allocations) {
        const void*             ptr    = pair.first;
        const AllocationRecord& record = pair.second;

        totalLeakedBytes += record.size;

        oss << "Leak at address " << ptr << ":\n";
        oss << "  Size: " << record.size << " bytes\n";
        oss << "  Pool: " << record.poolName << "\n";

        // Calculate time since allocation
        auto now      = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - record.timestamp);
        oss << "  Allocated " << duration.count() << " seconds ago\n";

        if (!record.stackTrace.empty()) {
            oss << "  Stack trace:\n" << record.stackTrace << "\n";
        }

        oss << "\n";
    }

    oss << "Total leaked memory: " << totalLeakedBytes << " bytes\n";

    return oss.str();
}

void MemoryLeakDetector::reset() {
    std::lock_guard<std::mutex> lock(mutex);
    allocations.clear();
}

void MemoryLeakDetector::setEnabled(bool enable) {
    std::lock_guard<std::mutex> lock(mutex);

    if (enable && !enabled) {
        // Clear existing allocations when enabling
        allocations.clear();
    }

    enabled = enable;
}

bool MemoryLeakDetector::isEnabled() const { return enabled; }

std::string MemoryLeakDetector::captureStackTrace() {
    const int maxFrames = 32;
    void* frames[maxFrames];
    int numFrames = backtrace(frames, maxFrames);
    char** symbols = backtrace_symbols(frames, numFrames);

    std::ostringstream oss;
    oss << "Stack trace:\n";

    for (int i = 0; i < numFrames; ++i) {
        // Demangle C++ symbols
        char* symbol = symbols[i];
        char* mangled = nullptr;

        // Find the mangled name between '(' and '+'
        char* openParen = strchr(symbol, '(');
        if (openParen) {
            char* plus = strchr(openParen, '+');
            if (plus) {
                *plus = '\0';
                mangled = openParen + 1;

                int status;
                char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
                if (status == 0 && demangled) {
                    oss << "  " << i << ": " << demangled << "\n";
                    free(demangled);
                } else {
                    oss << "  " << i << ": " << mangled << "\n";
                }
                *plus = '+';  // Restore
            } else {
                oss << "  " << i << ": " << symbol << "\n";
            }
        } else {
            oss << "  " << i << ": " << symbol << "\n";
        }
    }

    free(symbols);
    return oss.str();
}

// Boundary Checker implementation
BoundaryChecker& BoundaryChecker::getInstance() {
    static BoundaryChecker instance;
    return instance;
}

BoundaryChecker::BoundaryChecker() : enabled(false) {}

BoundaryChecker::~BoundaryChecker() {
    if (enabled && !trackedAllocations.empty()) {
        reportWarning("Boundary checker detected unfreed memory at program exit");
    }
}

void* BoundaryChecker::trackAllocation(void* ptr, size_t size) {
    if (!enabled || ptr == nullptr) {
        return ptr;
    }

    void* markedPtr = addBoundaryMarkers(ptr, size);

    std::lock_guard<std::mutex> lock(mutex);

    BoundaryRecord record;
    record.originalPtr = ptr;
    record.size        = size;
    record.stackTrace  = captureStackTrace();

    trackedAllocations[markedPtr] = record;

    return markedPtr;
}

bool BoundaryChecker::checkAndDeallocate(void* ptr) {
    if (!enabled || ptr == nullptr) {
        return true;
    }

    std::lock_guard<std::mutex> lock(mutex);

    auto it = trackedAllocations.find(ptr);
    if (it == trackedAllocations.end()) {
        reportWarning("Attempted to check boundaries of untracked memory at " +
                      std::to_string(reinterpret_cast<uintptr_t>(ptr)));
        return false;
    }

    bool valid = checkBoundaries(ptr, it->second);

    if (!valid) {
        reportError(ErrorSeverity::Error, "Memory boundary violation detected");
    }

    trackedAllocations.erase(it);

    return valid;
}

bool BoundaryChecker::checkAll() {
    if (!enabled) {
        return true;
    }

    std::lock_guard<std::mutex> lock(mutex);

    bool allValid = true;

    for (const auto& pair : trackedAllocations) {
        if (!checkBoundaries(pair.first, pair.second)) {
            allValid = false;

            reportError(ErrorSeverity::Error, "Memory boundary violation detected at " +
                                                  std::to_string(reinterpret_cast<uintptr_t>(pair.first)));
        }
    }

    return allValid;
}

std::string BoundaryChecker::getViolationReport() const {
    if (!enabled) {
        return "Boundary checking is disabled";
    }

    std::lock_guard<std::mutex> lock(mutex);

    std::ostringstream oss;
    oss << "Boundary Violation Report:\n";
    oss << "-------------------------\n";

    bool foundViolations = false;

    for (const auto& pair : trackedAllocations) {
        if (!const_cast<BoundaryChecker*>(this)->checkBoundaries(pair.first, pair.second)) {
            foundViolations = true;

            const void*           ptr    = pair.first;
            const BoundaryRecord& record = pair.second;

            oss << "Violation at address " << ptr << ":\n";
            oss << "  Size: " << record.size << " bytes\n";

            if (!record.stackTrace.empty()) {
                oss << "  Allocation stack trace:\n" << record.stackTrace << "\n";
            }

            oss << "\n";
        }
    }

    if (!foundViolations) {
        oss << "No boundary violations detected\n";
    }

    return oss.str();
}

void BoundaryChecker::setEnabled(bool enable) {
    std::lock_guard<std::mutex> lock(mutex);

    if (enable && !enabled) {
        // Clear existing allocations when enabling
        trackedAllocations.clear();
    }

    enabled = enable;
}

bool BoundaryChecker::isEnabled() const { return enabled; }

bool BoundaryChecker::checkBoundaries(void* ptr, const BoundaryRecord& record) {
    // Check the boundary markers
    char* charPtr = static_cast<char*>(ptr);

    // Check the prefix markers (before the user data)
    uint32_t prefixMarkers[4];
    std::memcpy(prefixMarkers, charPtr - BOUNDARY_SIZE, BOUNDARY_SIZE);

    if (prefixMarkers[0] != BOUNDARY_PATTERN1 || prefixMarkers[1] != BOUNDARY_PATTERN2 ||
        prefixMarkers[2] != BOUNDARY_PATTERN3 || prefixMarkers[3] != BOUNDARY_PATTERN4) {
        return false;
    }

    // Check the suffix markers (after the user data)
    uint32_t suffixMarkers[4];
    std::memcpy(suffixMarkers, charPtr + record.size, BOUNDARY_SIZE);

    if (suffixMarkers[0] != BOUNDARY_PATTERN1 || suffixMarkers[1] != BOUNDARY_PATTERN2 ||
        suffixMarkers[2] != BOUNDARY_PATTERN3 || suffixMarkers[3] != BOUNDARY_PATTERN4) {
        return false;
    }

    return true;
}

void* BoundaryChecker::addBoundaryMarkers(void* ptr, size_t size) {
    // Add boundary markers before and after the user data
    char* charPtr = static_cast<char*>(ptr);

    // Add the prefix markers
    uint32_t prefixMarkers[4] = {BOUNDARY_PATTERN1, BOUNDARY_PATTERN2, BOUNDARY_PATTERN3, BOUNDARY_PATTERN4};
    std::memcpy(charPtr, prefixMarkers, BOUNDARY_SIZE);

    // The user data starts after the prefix marker
    void* userPtr = charPtr + BOUNDARY_SIZE;

    // Add the suffix markers
    uint32_t suffixMarkers[4] = {BOUNDARY_PATTERN1, BOUNDARY_PATTERN2, BOUNDARY_PATTERN3, BOUNDARY_PATTERN4};
    std::memcpy(charPtr + BOUNDARY_SIZE + size, suffixMarkers, BOUNDARY_SIZE);

    return userPtr;
}

std::string BoundaryChecker::captureStackTrace() {
    const int maxFrames = 32;
    void* frames[maxFrames];
    int numFrames = backtrace(frames, maxFrames);
    char** symbols = backtrace_symbols(frames, numFrames);

    std::ostringstream oss;
    oss << "Stack trace:\n";

    for (int i = 0; i < numFrames; ++i) {
        // Demangle C++ symbols
        char* symbol = symbols[i];
        char* mangled = nullptr;

        // Find the mangled name between '(' and '+'
        char* openParen = strchr(symbol, '(');
        if (openParen) {
            char* plus = strchr(openParen, '+');
            if (plus) {
                *plus = '\0';
                mangled = openParen + 1;

                int status;
                char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
                if (status == 0 && demangled) {
                    oss << "  " << i << ": " << demangled << "\n";
                    free(demangled);
                } else {
                    oss << "  " << i << ": " << mangled << "\n";
                }
                *plus = '+';  // Restore
            } else {
                oss << "  " << i << ": " << symbol << "\n";
            }
        } else {
            oss << "  " << i << ": " << symbol << "\n";
        }
    }

    free(symbols);
    return oss.str();
}

// Performance Tracker implementation
PerformanceTracker& PerformanceTracker::getInstance() {
    static PerformanceTracker instance;
    return instance;
}

PerformanceTracker::PerformanceTracker() : enabled(false) {}

void PerformanceTracker::trackAllocation(size_t size, double timeMs, const std::string& poolName) {
    if (!enabled) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);

    PoolMetrics& metrics = poolMetrics[poolName];

    // Update allocation metrics
    metrics.totalAllocations++;
    metrics.totalAllocationTimeMs += timeMs;
    metrics.minAllocationTimeMs = std::min(metrics.minAllocationTimeMs, timeMs);
    metrics.maxAllocationTimeMs = std::max(metrics.maxAllocationTimeMs, timeMs);

    // Update size-based metrics
    metrics.sizeToTimeMap[size] += timeMs;
}

void PerformanceTracker::trackDeallocation(size_t size, double timeMs, const std::string& poolName) {
    if (!enabled) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);

    PoolMetrics& metrics = poolMetrics[poolName];

    // Update deallocation metrics
    metrics.totalDeallocations++;
    metrics.totalDeallocationTimeMs += timeMs;
    metrics.minDeallocationTimeMs = std::min(metrics.minDeallocationTimeMs, timeMs);
    metrics.maxDeallocationTimeMs = std::max(metrics.maxDeallocationTimeMs, timeMs);
}

std::string PerformanceTracker::getPerformanceReport() const {
    if (!enabled) {
        return "Performance tracking is disabled";
    }

    std::lock_guard<std::mutex> lock(mutex);

    std::ostringstream oss;
    oss << "Performance Report:\n";
    oss << "------------------\n";

    for (const auto& pair : poolMetrics) {
        const std::string& poolName = pair.first;
        const PoolMetrics& metrics  = pair.second;

        oss << "Pool: " << poolName << "\n";

        // Allocation metrics
        oss << "  Allocations:\n";
        oss << "    Count: " << metrics.totalAllocations << "\n";

        if (metrics.totalAllocations > 0) {
            double avgAllocationTime = metrics.totalAllocationTimeMs / metrics.totalAllocations;

            oss << "    Total time: " << metrics.totalAllocationTimeMs << " ms\n";
            oss << "    Average time: " << avgAllocationTime << " ms\n";
            oss << "    Min time: " << metrics.minAllocationTimeMs << " ms\n";
            oss << "    Max time: " << metrics.maxAllocationTimeMs << " ms\n";
        }

        // Deallocation metrics
        oss << "  Deallocations:\n";
        oss << "    Count: " << metrics.totalDeallocations << "\n";

        if (metrics.totalDeallocations > 0) {
            double avgDeallocationTime = metrics.totalDeallocationTimeMs / metrics.totalDeallocations;

            oss << "    Total time: " << metrics.totalDeallocationTimeMs << " ms\n";
            oss << "    Average time: " << avgDeallocationTime << " ms\n";
            oss << "    Min time: " << metrics.minDeallocationTimeMs << " ms\n";
            oss << "    Max time: " << metrics.maxDeallocationTimeMs << " ms\n";
        }

        // Size-based metrics
        oss << "  Size-based metrics:\n";

        for (const auto& sizeTimePair : metrics.sizeToTimeMap) {
            size_t size      = sizeTimePair.first;
            double totalTime = sizeTimePair.second;

            oss << "    Size " << size << " bytes: " << totalTime << " ms total\n";
        }

        oss << "\n";
    }

    return oss.str();
}

void PerformanceTracker::reset() {
    std::lock_guard<std::mutex> lock(mutex);
    poolMetrics.clear();
}

void PerformanceTracker::setEnabled(bool enable) {
    std::lock_guard<std::mutex> lock(mutex);

    if (enable && !enabled) {
        // Clear existing metrics when enabling
        poolMetrics.clear();
    }

    enabled = enable;
}

bool PerformanceTracker::isEnabled() const { return enabled; }

// Timer implementation
Timer::Timer() : startTime(std::chrono::steady_clock::now()) {}

void Timer::reset() { startTime = std::chrono::steady_clock::now(); }

double Timer::elapsedMs() const {
    auto now      = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - startTime);
    return duration.count() / 1000.0;
}

// Convenience functions
void enableDebugging(bool enable) {
    MemoryLeakDetector::getInstance().setEnabled(enable);
    BoundaryChecker::getInstance().setEnabled(enable);
    PerformanceTracker::getInstance().setEnabled(enable);
}

bool isDebuggingEnabled() {
    return MemoryLeakDetector::getInstance().isEnabled() || BoundaryChecker::getInstance().isEnabled() ||
           PerformanceTracker::getInstance().isEnabled();
}

std::string getMemoryReport() {
    std::ostringstream oss;

    if (MemoryLeakDetector::getInstance().isEnabled()) {
        oss << MemoryLeakDetector::getInstance().getLeakReport() << "\n\n";
    }

    if (BoundaryChecker::getInstance().isEnabled()) {
        oss << BoundaryChecker::getInstance().getViolationReport() << "\n\n";
    }

    if (PerformanceTracker::getInstance().isEnabled()) {
        oss << PerformanceTracker::getInstance().getPerformanceReport() << "\n\n";
    }

    return oss.str();
}

}  // namespace memory_pool