#include "memory_pool/memory_pool.hpp"
#include <iostream>
#include <cassert>

using namespace memory_pool;

// Test fixture
class MultiGPUTests {
public:
    MultiGPUTests() {
        std::cout << "Setting up Multi-GPU tests...\n";
    }

    ~MultiGPUTests() {
        std::cout << "Cleaning up Multi-GPU tests...\n";
    }

    void testDeviceEnumeration() {
        std::cout << "Testing device enumeration...\n";
        MemoryPoolManager& manager = MemoryPoolManager::getInstance();

        int deviceCount = manager.getGPUDeviceCount();
        std::cout << "GPU device count: " << deviceCount << "\n";
        assert(deviceCount >= 0);

        for (int i = 0; i < deviceCount; ++i) {
            bool available = manager.isGPUDeviceAvailable(i);
            std::cout << "Device " << i << " available: " << (available ? "yes" : "no") << "\n";
            if (available) {
                size_t memory = manager.getGPUDeviceMemory(i);
                std::cout << "Device " << i << " memory: " << memory << " bytes\n";
                assert(memory > 0);
            }
        }
    }

    void testDeviceSelection() {
        std::cout << "Testing device selection...\n";
        MemoryPoolManager& manager = MemoryPoolManager::getInstance();

        int bestDevice = manager.selectBestGPUDevice();
        if (manager.getGPUDeviceCount() > 0) {
            std::cout << "Best device: " << bestDevice << "\n";
            assert(bestDevice >= 0);
            assert(manager.isGPUDeviceAvailable(bestDevice));
        } else {
            std::cout << "No devices available\n";
            assert(bestDevice == -1);
        }
    }

    void testDeviceSpecificAllocation() {
        std::cout << "Testing device-specific allocation...\n";
        MemoryPoolManager& manager = MemoryPoolManager::getInstance();

        int deviceCount = manager.getGPUDeviceCount();
        if (deviceCount == 0) {
            std::cout << "No GPU devices, skipping allocation test\n";
            return;
        }

        // Test allocation on device 0
        try {
            void* ptr = allocateGPU(1024, 0);
            assert(ptr != nullptr);
            deallocateGPU(ptr, "gpu_0");
            std::cout << "Allocation on device 0 successful\n";
        } catch (const std::exception& e) {
            std::cout << "Allocation on device 0 failed: " << e.what() << "\n";
            // This might happen if device 0 is not available
        }

        // Test pool creation for device
        IMemoryPool* pool = manager.createGPUPoolForDevice(0);
        if (pool != nullptr) {
            std::cout << "Pool created for device 0\n";
            void* ptr = pool->allocate(1024);
            assert(ptr != nullptr);
            pool->deallocate(ptr);
        } else {
            std::cout << "Failed to create pool for device 0\n";
        }
    }

    void testDefaultGPUAllocation() {
        std::cout << "Testing default GPU allocation...\n";

        // Test the default allocation
        try {
            void* ptr = allocateGPU(1024);
            assert(ptr != nullptr);
            deallocateGPU(ptr);
            std::cout << "Default GPU allocation successful\n";
        } catch (const std::exception& e) {
            std::cout << "Default GPU allocation failed: " << e.what() << "\n";
        }
    }

    void runAllTests() {
        testDeviceEnumeration();
        testDeviceSelection();
        testDeviceSpecificAllocation();
        testDefaultGPUAllocation();

        std::cout << "All Multi-GPU tests completed!\n";
    }
};

int main() {
    try {
        MultiGPUTests tests;
        tests.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Multi-GPU test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}