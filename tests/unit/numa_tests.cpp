#include "memory_pool/utils/numa_utils.hpp"
#include <iostream>
#include <cassert>
#include <cstddef>

using namespace memory_pool::numa_utils;

// Test fixture
class NUMATests {
public:
    NUMATests() {
        std::cout << "Setting up NUMA tests...\n";
    }

    ~NUMATests() {
        std::cout << "Cleaning up NUMA tests...\n";
    }

    void testIsNUMAAvailable() {
        std::cout << "Testing NUMA availability...\n";
        bool available = is_numa_available();
        std::cout << "NUMA available: " << (available ? "yes" : "no") << "\n";
    }

    void testGetNumNUMANodes() {
        std::cout << "Testing NUMA node count...\n";
        int nodes = get_num_numa_nodes();
        assert(nodes >= 0);
        std::cout << "Number of NUMA nodes: " << nodes << "\n";
    }

    void testGetCurrentNUMANode() {
        std::cout << "Testing current NUMA node...\n";
        int node = get_current_numa_node();
        if (is_numa_available()) {
            assert(node >= 0);
            std::cout << "Current NUMA node: " << node << "\n";
        } else {
            assert(node == -1);
            std::cout << "NUMA not available, node: " << node << "\n";
        }
    }

    void testAllocateAndDeallocate() {
        std::cout << "Testing allocate and deallocate...\n";
        const size_t size = 1024;
        const size_t alignment = 64;

        void* ptr = allocate_on_node(size, alignment, get_current_numa_node());
        assert(ptr != nullptr);

        // Check alignment
        assert(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);

        deallocate(ptr);
        std::cout << "Allocate/deallocate test passed!\n";
    }

    void testAllocateZeroSize() {
        std::cout << "Testing zero size allocation...\n";
        void* ptr = allocate_on_node(0, 64, get_current_numa_node());
        assert(ptr == nullptr);
        std::cout << "Zero size allocation test passed!\n";
    }

    void testAllocateLargeSize() {
        std::cout << "Testing large size allocation...\n";
        const size_t size = 1024 * 1024;  // 1MB
        const size_t alignment = 4096;

        void* ptr = allocate_on_node(size, alignment, get_current_numa_node());
        if (ptr != nullptr) {
            assert(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
            deallocate(ptr);
            std::cout << "Large allocation test passed!\n";
        } else {
            std::cout << "Large allocation failed (possibly due to limited memory), but that's OK.\n";
        }
    }

    void runAllTests() {
        testIsNUMAAvailable();
        testGetNumNUMANodes();
        testGetCurrentNUMANode();
        testAllocateAndDeallocate();
        testAllocateZeroSize();
        testAllocateLargeSize();

        std::cout << "All NUMA tests passed!\n";
    }
};

int main() {
    try {
        NUMATests tests;
        tests.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "NUMA test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}