# 性能分析工具使用场景和示例详解

## 目录

1. [真实场景用例](#真实场景用例)
2. [详细代码示例](#详细代码示例)
3. [性能优化工作流](#性能优化工作流)
4. [常见问题诊断](#常见问题诊断)
5. [最佳实践模式](#最佳实践模式)

## 真实场景用例

### 场景1：游戏引擎中的粒子系统

**背景**：
- 每帧需要分配/释放数千个粒子对象
- 粒子大小固定（64字节）
- 高频率分配导致性能问题

**分析目标**：
- 测量分配延迟是否影响帧率
- 比较固定大小分配器 vs 可变大小分配器
- 确定最佳池配置

**代码示例**：

```cpp
#include "memory_pool/memory_pool.hpp"
#include "memory_pool/utils/profiling_tools.hpp"

struct Particle {
    float position[3];
    float velocity[3];
    float color[4];
    float lifetime;
    // Total: 64 bytes
};

void profile_particle_system() {
    using namespace memory_pool;
    using namespace memory_pool::profiling;
    
    const int PARTICLES_PER_FRAME = 5000;
    const int FRAMES = 1000;
    
    // Test 1: Fixed-size allocator
    {
        std::cout << "\n=== Testing Fixed-Size Allocator ===" << std::endl;
        
        PoolConfig config = PoolConfig::FixedSizeCPU(sizeof(Particle));
        config.initialSize = PARTICLES_PER_FRAME * sizeof(Particle) * 2;
        
        auto& manager = MemoryPoolManager::getInstance();
        auto* pool = manager.createCPUPool("particle_fixed", config);
        
        MemoryProfiler profiler;
        std::vector<Particle*> particles;
        particles.reserve(PARTICLES_PER_FRAME);
        
        // Simulate game loop
        for (int frame = 0; frame < FRAMES; ++frame) {
            // Allocate particles for this frame
            for (int i = 0; i < PARTICLES_PER_FRAME; ++i) {
                Timer timer;
                timer.start();
                Particle* p = static_cast<Particle*>(pool->allocate(sizeof(Particle)));
                timer.stop();
                
                profiler.record_allocation(sizeof(Particle), timer.elapsed_microseconds());
                particles.push_back(p);
            }
            
            // Update particles (simulated work)
            for (Particle* p : particles) {
                p->lifetime -= 0.016f; // 60 FPS
            }
            
            // Remove dead particles
            auto it = std::remove_if(particles.begin(), particles.end(),
                [&](Particle* p) {
                    if (p->lifetime <= 0) {
                        Timer timer;
                        timer.start();
                        pool->deallocate(p);
                        timer.stop();
                        profiler.record_deallocation(sizeof(Particle), timer.elapsed_microseconds());
                        return true;
                    }
                    return false;
                });
            particles.erase(it, particles.end());
        }
        
        // Print results
        std::cout << profiler.generate_report() << std::endl;
        std::cout << "Average allocation time: " 
                  << profiler.alloc_stats().average() << " μs" << std::endl;
        std::cout << "Max allocation time: " 
                  << profiler.alloc_stats().max() << " μs" << std::endl;
        
        // Check if suitable for 60 FPS (16.67ms per frame)
        double total_alloc_time_per_frame = 
            profiler.alloc_stats().average() * PARTICLES_PER_FRAME;
        std::cout << "Total allocation time per frame: " 
                  << total_alloc_time_per_frame / 1000.0 << " ms" << std::endl;
        
        if (total_alloc_time_per_frame < 1000.0) { // < 1ms
            std::cout << "✓ Allocation overhead acceptable for 60 FPS!" << std::endl;
        } else {
            std::cout << "✗ Allocation overhead too high!" << std::endl;
        }
    }
}
```

**预期结果**：
- 固定大小分配器应提供 < 0.1 μs 的分配延迟
- 总分配开销应 < 1% 的帧时间预算

---

### 场景2：深度学习训练中的GPU内存管理

**背景**：
- 频繁分配/释放张量（tensor）内存
- 大小从几KB到几GB不等
- GPU内存是瓶颈

**分析目标**：
- 测量GPU内存分配延迟
- 分析CPU-GPU传输带宽
- 识别内存碎片问题

**代码示例**：

```cpp
void profile_deep_learning_workload() {
    using namespace memory_pool;
    using namespace memory_pool::profiling;
    
    if (!isDeviceAvailable(0)) {
        std::cerr << "CUDA device not available" << std::endl;
        return;
    }
    
    std::cout << "\n=== Profiling Deep Learning Workload ===" << std::endl;
    
    // Create GPU pool
    PoolConfig config = PoolConfig::HighPerformanceGPU();
    config.initialSize = 512 * 1024 * 1024; // 512MB
    config.deviceId = 0;
    
    auto& manager = MemoryPoolManager::getInstance();
    auto* gpu_pool = static_cast<GPUMemoryPool*>(
        manager.createGPUPool("dl_gpu_pool", config));
    
    GPUProfiler gpu_profiler;
    MemoryProfiler mem_profiler;
    
    // Simulate tensor allocations (typical in training)
    struct TensorAllocation {
        void* ptr;
        size_t size;
    };
    std::vector<TensorAllocation> tensors;
    
    // Layer sizes for a typical neural network
    std::vector<size_t> layer_sizes = {
        1024 * 1024 * 4,      // 4MB - Input layer
        2048 * 2048 * 4,      // 16MB - Hidden layer 1
        2048 * 2048 * 4,      // 16MB - Hidden layer 2
        1024 * 1000 * 4,      // 4MB - Output layer
        2048 * 2048 * 4,      // 16MB - Gradients
        2048 * 2048 * 4,      // 16MB - Optimizer state
    };
    
    // Training loop simulation
    const int ITERATIONS = 100;
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        // Forward pass - allocate tensors
        {
            ScopedProfiler profiler("Forward Pass");
            
            for (size_t size : layer_sizes) {
                gpu_profiler.start_kernel_timing();
                
                Timer timer;
                timer.start();
                void* ptr = gpu_pool->allocate(size);
                timer.stop();
                
                gpu_profiler.stop_kernel_timing();
                
                mem_profiler.record_allocation(size, timer.elapsed_microseconds());
                tensors.push_back({ptr, size});
            }
        }
        
        // Simulate computation (kernel execution)
        {
            ScopedProfiler profiler("Kernel Execution");
            cudaDeviceSynchronize();
        }
        
        // Backward pass - more allocations for gradients
        {
            ScopedProfiler profiler("Backward Pass");
            
            for (size_t size : layer_sizes) {
                Timer timer;
                timer.start();
                void* ptr = gpu_pool->allocate(size);
                timer.stop();
                
                mem_profiler.record_allocation(size, timer.elapsed_microseconds());
                tensors.push_back({ptr, size});
            }
        }
        
        // Weight update - deallocate temporary tensors
        {
            ScopedProfiler profiler("Weight Update");
            
            for (const auto& tensor : tensors) {
                Timer timer;
                timer.start();
                gpu_pool->deallocate(tensor.ptr);
                timer.stop();
                
                mem_profiler.record_deallocation(tensor.size, timer.elapsed_microseconds());
            }
            tensors.clear();
        }
        
        // Check memory stats every 10 iterations
        if ((iter + 1) % 10 == 0) {
            const auto& stats = gpu_pool->getStats();
            double utilization = static_cast<double>(stats.getCurrentUsed()) / 
                               stats.getTotalAllocated();
            double fragmentation = stats.getFragmentationRatio();
            
            std::cout << "Iteration " << (iter + 1) << ":" << std::endl;
            std::cout << "  Memory utilization: " << (utilization * 100) << "%" << std::endl;
            std::cout << "  Fragmentation ratio: " << (fragmentation * 100) << "%" << std::endl;
            
            if (fragmentation > 0.3) {
                std::cout << "  ⚠ Warning: High fragmentation detected!" << std::endl;
            }
        }
    }
    
    // Final report
    std::cout << "\n=== Final Performance Report ===" << std::endl;
    std::cout << mem_profiler.generate_report() << std::endl;
    
    // Bandwidth analysis
    double total_bytes = mem_profiler.alloc_stats().count() * 
                        (layer_sizes[0] + layer_sizes[1]); // Simplified
    double total_time_sec = mem_profiler.alloc_stats().total() / 1e6;
    double bandwidth_gbps = (total_bytes / 1e9) / total_time_sec;
    
    std::cout << "Effective memory bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
    
    // Compare with cudaMalloc
    std::cout << "\n=== Comparison with cudaMalloc ===" << std::endl;
    {
        Timer timer;
        timer.start();
        for (int i = 0; i < 100; ++i) {
            void* ptr;
            cudaMalloc(&ptr, 16 * 1024 * 1024);
            cudaFree(ptr);
        }
        timer.stop();
        
        double cuda_malloc_avg = timer.elapsed_microseconds() / 100.0;
        double pool_avg = mem_profiler.alloc_stats().average();
        double speedup = cuda_malloc_avg / pool_avg;
        
        std::cout << "cudaMalloc average: " << cuda_malloc_avg << " μs" << std::endl;
        std::cout << "Memory pool average: " << pool_avg << " μs" << std::endl;
        std::cout << "Speedup: " << speedup << "x" << std::endl;
    }
}
```

**预期结果**：
- Memory pool 应该比 cudaMalloc 快 50-200倍
- 碎片率应该保持在 < 30%
- 内存利用率应该 > 80%

---

### 场景3：Web服务器的请求处理

**背景**：
- 高并发HTTP请求处理
- 每个请求需要临时缓冲区
- 内存分配是性能热点

**分析目标**：
- 测量多线程竞争
- 对比不同同步策略
- 优化池配置

**代码示例**：

```cpp
void profile_web_server_workload() {
    using namespace memory_pool;
    using namespace memory_pool::profiling;
    
    const int NUM_THREADS = 8;
    const int REQUESTS_PER_THREAD = 10000;
    const size_t BUFFER_SIZE = 4096; // 4KB per request
    
    std::cout << "\n=== Profiling Web Server Workload ===" << std::endl;
    std::cout << "Threads: " << NUM_THREADS << std::endl;
    std::cout << "Requests per thread: " << REQUESTS_PER_THREAD << std::endl;
    
    // Test different synchronization strategies
    struct TestConfig {
        std::string name;
        SyncType sync_type;
    };
    
    std::vector<TestConfig> configs = {
        {"Mutex-based", SyncType::Mutex},
        {"Lock-free", SyncType::LockFree},
    };
    
    for (const auto& config : configs) {
        std::cout << "\n--- Testing: " << config.name << " ---" << std::endl;
        
        PoolConfig pool_config = PoolConfig::DefaultCPU();
        pool_config.syncType = config.sync_type;
        pool_config.initialSize = BUFFER_SIZE * REQUESTS_PER_THREAD * NUM_THREADS;
        
        auto& manager = MemoryPoolManager::getInstance();
        std::string pool_name = "webserver_" + config.name;
        auto* pool = manager.createCPUPool(pool_name, pool_config);
        
        // Thread-local profilers
        std::vector<MemoryProfiler> profilers(NUM_THREADS);
        std::vector<std::thread> threads;
        
        Timer total_timer;
        total_timer.start();
        
        // Launch worker threads
        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back([&, t]() {
                auto& profiler = profilers[t];
                
                for (int req = 0; req < REQUESTS_PER_THREAD; ++req) {
                    // Simulate request processing
                    Timer timer;
                    timer.start();
                    void* buffer = pool->allocate(BUFFER_SIZE);
                    timer.stop();
                    
                    profiler.record_allocation(BUFFER_SIZE, timer.elapsed_microseconds());
                    
                    // Simulate work (parsing, processing, etc.)
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                    
                    // Response sent, deallocate buffer
                    timer.start();
                    pool->deallocate(buffer);
                    timer.stop();
                    
                    profiler.record_deallocation(BUFFER_SIZE, timer.elapsed_microseconds());
                }
            });
        }
        
        // Wait for completion
        for (auto& thread : threads) {
            thread.join();
        }
        
        total_timer.stop();
        
        // Aggregate results
        PerformanceCounter total_alloc;
        PerformanceCounter total_dealloc;
        
        for (const auto& profiler : profilers) {
            // Aggregate stats (simplified - in real code, merge properly)
            std::cout << "Thread stats: " << std::endl;
            std::cout << "  Alloc avg: " << profiler.alloc_stats().average() << " μs" << std::endl;
            std::cout << "  Alloc max: " << profiler.alloc_stats().max() << " μs" << std::endl;
        }
        
        // Throughput analysis
        size_t total_requests = NUM_THREADS * REQUESTS_PER_THREAD;
        double total_time_sec = total_timer.elapsed_seconds();
        double throughput = total_requests / total_time_sec;
        
        std::cout << "\nOverall Performance:" << std::endl;
        std::cout << "  Total time: " << total_time_sec << " seconds" << std::endl;
        std::cout << "  Throughput: " << throughput << " requests/sec" << std::endl;
        std::cout << "  Avg latency: " << (total_time_sec * 1e6 / total_requests) << " μs/request" << std::endl;
        
        manager.destroyPool(pool_name);
    }
}
```

**预期结果**：
- Lock-free 在高并发下应该优于 Mutex
- 吞吐量应该 > 100K requests/sec
- 平均延迟应该 < 50 μs

---

## 详细代码示例

### 示例1：基础性能分析流程

```cpp
#include "memory_pool/memory_pool.hpp"
#include "memory_pool/utils/profiling_tools.hpp"
#include <iostream>
#include <vector>

void basic_profiling_workflow() {
    using namespace memory_pool;
    using namespace memory_pool::profiling;
    
    std::cout << "=== Basic Profiling Workflow ===" << std::endl;
    
    // Step 1: Create a memory pool
    auto& manager = MemoryPoolManager::getInstance();
    PoolConfig config = PoolConfig::DefaultCPU();
    auto* pool = manager.createCPUPool("profile_test", config);
    
    // Step 2: Create profiling tools
    Timer timer;
    MemoryProfiler profiler;
    
    // Step 3: Profile a series of allocations
    std::cout << "\nPhase 1: Sequential Allocations" << std::endl;
    {
        ScopedProfiler scope("Sequential Allocation Phase");
        
        std::vector<void*> pointers;
        for (size_t size = 64; size <= 4096; size *= 2) {
            timer.start();
            void* ptr = pool->allocate(size);
            timer.stop();
            
            profiler.record_allocation(size, timer.elapsed_microseconds());
            pointers.push_back(ptr);
            
            std::cout << "  Allocated " << size << " bytes in " 
                      << timer.elapsed_microseconds() << " μs" << std::endl;
        }
        
        // Step 4: Deallocate
        for (void* ptr : pointers) {
            timer.start();
            pool->deallocate(ptr);
            timer.stop();
            
            profiler.record_deallocation(0, timer.elapsed_microseconds());
        }
    }
    
    // Step 5: Generate report
    std::cout << "\n=== Profiling Report ===" << std::endl;
    std::cout << profiler.generate_report() << std::endl;
    
    // Step 6: Analyze results
    const auto& alloc_stats = profiler.alloc_stats();
    std::cout << "\nStatistical Analysis:" << std::endl;
    std::cout << "  Count: " << alloc_stats.count() << std::endl;
    std::cout << "  Min: " << alloc_stats.min() << " μs" << std::endl;
    std::cout << "  Max: " << alloc_stats.max() << " μs" << std::endl;
    std::cout << "  Average: " << alloc_stats.average() << " μs" << std::endl;
    std::cout << "  Std Dev: " << alloc_stats.std_deviation() << " μs" << std::endl;
    
    // Step 7: Performance recommendations
    if (alloc_stats.average() > 1.0) {
        std::cout << "\n⚠ Warning: Average allocation time > 1 μs" << std::endl;
        std::cout << "Consider using fixed-size allocator or increasing pool size" << std::endl;
    } else {
        std::cout << "\n✓ Performance looks good!" << std::endl;
    }
}
```

### 示例2：GPU内存传输带宽分析

```cpp
void profile_gpu_memory_bandwidth() {
    using namespace memory_pool;
    using namespace memory_pool::profiling;
    
    if (!isDeviceAvailable(0)) {
        std::cout << "CUDA device not available" << std::endl;
        return;
    }
    
    std::cout << "=== GPU Memory Bandwidth Profiling ===" << std::endl;
    
    // Create pools
    auto& manager = MemoryPoolManager::getInstance();
    auto* cpu_pool = manager.createCPUPool("cpu", PoolConfig::DefaultCPU());
    auto* gpu_pool = static_cast<GPUMemoryPool*>(
        manager.createGPUPool("gpu", PoolConfig::DefaultGPU()));
    
    GPUProfiler gpu_profiler;
    
    // Test different transfer sizes
    std::vector<size_t> sizes = {
        1024,           // 1 KB
        1024 * 1024,    // 1 MB
        16 * 1024 * 1024,   // 16 MB
        64 * 1024 * 1024,   // 64 MB
        256 * 1024 * 1024,  // 256 MB
    };
    
    std::cout << "\nHost-to-Device Transfer:" << std::endl;
    std::cout << std::setw(15) << "Size" << std::setw(15) << "Time (ms)" 
              << std::setw(20) << "Bandwidth (GB/s)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    for (size_t size : sizes) {
        // Allocate memory
        void* host_ptr = cpu_pool->allocate(size);
        void* device_ptr = gpu_pool->allocate(size);
        
        // Initialize host memory
        memset(host_ptr, 0xAA, size);
        
        // Warmup
        for (int i = 0; i < 3; ++i) {
            gpu_pool->copyHostToDevice(device_ptr, host_ptr, size);
        }
        
        // Measure transfer
        gpu_profiler.start_transfer_timing();
        for (int i = 0; i < 10; ++i) {
            gpu_pool->copyHostToDevice(device_ptr, host_ptr, size);
        }
        gpu_profiler.stop_transfer_timing();
        
        float time_ms = gpu_profiler.get_transfer_time_ms() / 10.0f;
        double bandwidth = gpu_profiler.calculate_bandwidth_gbps(size, time_ms);
        
        std::string size_str;
        if (size < 1024 * 1024) {
            size_str = std::to_string(size / 1024) + " KB";
        } else {
            size_str = std::to_string(size / (1024 * 1024)) + " MB";
        }
        
        std::cout << std::setw(15) << size_str 
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(20) << std::fixed << std::setprecision(2) << bandwidth
                  << std::endl;
        
        // Cleanup
        cpu_pool->deallocate(host_ptr);
        gpu_pool->deallocate(device_ptr);
    }
    
    std::cout << "\n📊 Bandwidth Analysis:" << std::endl;
    std::cout << "  Typical PCIe 3.0 x16: ~12 GB/s" << std::endl;
    std::cout << "  Typical PCIe 4.0 x16: ~25 GB/s" << std::endl;
    std::cout << "  If bandwidth is significantly lower, check for:" << std::endl;
    std::cout << "    - CPU-GPU PCIe connection" << std::endl;
    std::cout << "    - System load" << std::endl;
    std::cout << "    - Use of pinned memory (should improve bandwidth)" << std::endl;
}
```

### 示例3：内存碎片分析

```cpp
void profile_memory_fragmentation() {
    using namespace memory_pool;
    using namespace memory_pool::profiling;
    
    std::cout << "=== Memory Fragmentation Analysis ===" << std::endl;
    
    // Create pool
    PoolConfig config = PoolConfig::DefaultCPU();
    config.initialSize = 10 * 1024 * 1024; // 10 MB
    
    auto& manager = MemoryPoolManager::getInstance();
    auto* pool = manager.createCPUPool("frag_test", config);
    
    // Simulate fragmentation-inducing pattern
    std::vector<void*> allocations;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(64, 4096);
    
    std::cout << "\nSimulating workload..." << std::endl;
    
    for (int phase = 1; phase <= 5; ++phase) {
        std::cout << "\nPhase " << phase << ":" << std::endl;
        
        // Allocate many blocks
        for (int i = 0; i < 1000; ++i) {
            size_t size = size_dist(gen);
            void* ptr = pool->allocate(size);
            allocations.push_back(ptr);
        }
        
        // Deallocate every other block (creates fragmentation)
        for (size_t i = 0; i < allocations.size(); i += 2) {
            pool->deallocate(allocations[i]);
        }
        allocations.erase(
            std::remove_if(allocations.begin(), allocations.end(),
                [&, idx = 0](void* ptr) mutable { return (idx++ % 2) == 0; }),
            allocations.end()
        );
        
        // Check fragmentation
        const auto& stats = pool->getStats();
        double utilization = static_cast<double>(stats.getCurrentUsed()) / 
                           stats.getTotalAllocated();
        double fragmentation = stats.getFragmentationRatio();
        
        std::cout << "  Current used: " << (stats.getCurrentUsed() / 1024) << " KB" << std::endl;
        std::cout << "  Total allocated: " << (stats.getTotalAllocated() / 1024) << " KB" << std::endl;
        std::cout << "  Utilization: " << (utilization * 100) << "%" << std::endl;
        std::cout << "  Fragmentation: " << (fragmentation * 100) << "%" << std::endl;
        
        // Provide recommendations
        if (fragmentation > 0.5) {
            std::cout << "  ⚠ High fragmentation detected!" << std::endl;
            std::cout << "  💡 Recommendations:" << std::endl;
            std::cout << "     - Consider using fixed-size allocator for uniform sizes" << std::endl;
            std::cout << "     - Reset pool periodically if workload allows" << std::endl;
            std::cout << "     - Increase pool size to reduce reallocation" << std::endl;
        } else if (fragmentation > 0.3) {
            std::cout << "  ⚠ Moderate fragmentation" << std::endl;
            std::cout << "  💡 Monitor in production environment" << std::endl;
        } else {
            std::cout << "  ✓ Fragmentation is acceptable" << std::endl;
        }
    }
    
    // Cleanup
    for (void* ptr : allocations) {
        pool->deallocate(ptr);
    }
}
```

---

## 性能优化工作流

### 完整的优化流程示例

```cpp
class PerformanceOptimizationWorkflow {
public:
    void run() {
        step1_baseline_measurement();
        step2_identify_bottlenecks();
        step3_apply_optimizations();
        step4_verify_improvements();
        step5_document_results();
    }
    
private:
    void step1_baseline_measurement() {
        std::cout << "\n📊 Step 1: Baseline Measurement" << std::endl;
        std::cout << "================================" << std::endl;
        
        // Measure current performance
        // ... profiling code ...
        
        std::cout << "Baseline established:" << std::endl;
        std::cout << "  Allocation latency: 2.5 μs" << std::endl;
        std::cout << "  Memory utilization: 65%" << std::endl;
        std::cout << "  Fragmentation: 45%" << std::endl;
    }
    
    void step2_identify_bottlenecks() {
        std::cout << "\n🔍 Step 2: Identify Bottlenecks" << std::endl;
        std::cout << "================================" << std::endl;
        
        std::cout << "Analysis reveals:" << std::endl;
        std::cout << "  ❌ High fragmentation (45%) due to variable sizes" << std::endl;
        std::cout << "  ❌ Lock contention in multi-threaded scenarios" << std::endl;
        std::cout << "  ✓ Allocation latency is acceptable" << std::endl;
    }
    
    void step3_apply_optimizations() {
        std::cout << "\n🔧 Step 3: Apply Optimizations" << std::endl;
        std::cout << "===============================" << std::endl;
        
        std::cout << "Optimization 1: Use size-class buckets" << std::endl;
        std::cout << "Optimization 2: Switch to lock-free synchronization" << std::endl;
        std::cout << "Optimization 3: Increase initial pool size" << std::endl;
        
        // ... apply optimizations ...
    }
    
    void step4_verify_improvements() {
        std::cout << "\n✅ Step 4: Verify Improvements" << std::endl;
        std::cout << "==============================" << std::endl;
        
        std::cout << "After optimizations:" << std::endl;
        std::cout << "  Allocation latency: 1.8 μs (-28%)" << std::endl;
        std::cout << "  Memory utilization: 85% (+20%)" << std::endl;
        std::cout << "  Fragmentation: 15% (-30%)" << std::endl;
        std::cout << "\n🎉 All metrics improved!" << std::endl;
    }
    
    void step5_document_results() {
        std::cout << "\n📝 Step 5: Document Results" << std::endl;
        std::cout << "===========================" << std::endl;
        
        std::cout << "Optimization Report:" << std::endl;
        std::cout << "  Date: 2025-12-31" << std::endl;
        std::cout << "  Goal: Reduce fragmentation and improve utilization" << std::endl;
        std::cout << "  Changes Applied:" << std::endl;
        std::cout << "    - Implemented size-class buckets (64, 256, 1024, 4096 bytes)" << std::endl;
        std::cout << "    - Switched from Mutex to LockFree synchronization" << std::endl;
        std::cout << "    - Increased initial pool size from 1MB to 10MB" << std::endl;
        std::cout << "  Results: All metrics improved by 20-30%" << std::endl;
        std::cout << "  Recommendation: Deploy to production" << std::endl;
    }
};
```

---

## 常见问题诊断

### 问题1：分配延迟过高

**症状**：
```
Average allocation time: 15.3 μs
Max allocation time: 234.7 μs
```

**诊断流程**：

```cpp
void diagnose_high_allocation_latency() {
    std::cout << "🔍 Diagnosing high allocation latency..." << std::endl;
    
    // Check 1: Pool size
    const auto& stats = pool->getStats();
    if (stats.getCurrentUsed() > stats.getTotalAllocated() * 0.9) {
        std::cout << "❌ Pool is nearly full (>90%)" << std::endl;
        std::cout << "💡 Solution: Increase pool size or implement auto-growth" << std::endl;
    }
    
    // Check 2: Fragmentation
    if (stats.getFragmentationRatio() > 0.5) {
        std::cout << "❌ High fragmentation detected" << std::endl;
        std::cout << "💡 Solution: Use fixed-size allocator or defragment" << std::endl;
    }
    
    // Check 3: Lock contention (if multi-threaded)
    // Profile with thread timing
    
    // Check 4: Memory alignment issues
    // Check pool configuration
}
```

### 问题2：GPU内存传输慢

**症状**：
```
Host-to-Device bandwidth: 2.3 GB/s (expected: 12 GB/s)
```

**诊断和解决**：

```cpp
void diagnose_gpu_transfer_slowness() {
    std::cout << "🔍 Diagnosing slow GPU transfers..." << std::endl;
    
    // Test 1: Pinned vs pageable memory
    std::cout << "\nTest 1: Memory type comparison" << std::endl;
    
    // Pageable memory
    PoolConfig pageable_config = PoolConfig::DefaultGPU();
    pageable_config.usePinnedMemory = false;
    test_transfer_speed("Pageable", pageable_config);
    
    // Pinned memory
    PoolConfig pinned_config = PoolConfig::DefaultGPU();
    pinned_config.usePinnedMemory = true;
    test_transfer_speed("Pinned", pinned_config);
    
    std::cout << "\n💡 If pinned memory is much faster, use it for frequent transfers" << std::endl;
    
    // Test 2: Transfer size
    std::cout << "\nTest 2: Optimal transfer size" << std::endl;
    std::cout << "Small transfers (<1MB): Use batching" << std::endl;
    std::cout << "Large transfers (>64MB): Should achieve near-peak bandwidth" << std::endl;
    
    // Test 3: CUDA streams
    std::cout << "\nTest 3: Asynchronous transfers" << std::endl;
    std::cout << "💡 Use CUDA streams for overlapping computation and transfer" << std::endl;
}
```

---

## 最佳实践模式

### 模式1：RAII性能分析器

```cpp
class ProfiledSection {
public:
    ProfiledSection(const std::string& name, MemoryProfiler& profiler)
        : name_(name), profiler_(profiler) {
        std::cout << "\n=== " << name_ << " ===" << std::endl;
        timer_.start();
    }
    
    ~ProfiledSection() {
        timer_.stop();
        std::cout << "Section '" << name_ << "' completed in " 
                  << timer_.elapsed_milliseconds() << " ms" << std::endl;
    }
    
    void checkpoint(const std::string& msg) {
        std::cout << "  ✓ " << msg << " (at " 
                  << timer_.elapsed_milliseconds() << " ms)" << std::endl;
    }
    
private:
    std::string name_;
    Timer timer_;
    MemoryProfiler& profiler_;
};

// Usage
void example_with_profiled_sections() {
    MemoryProfiler profiler;
    
    {
        ProfiledSection section("Data Loading", profiler);
        // ... load data ...
        section.checkpoint("Loaded 1000 records");
        // ... process ...
        section.checkpoint("Processing complete");
    }
    
    {
        ProfiledSection section("Computation", profiler);
        // ... compute ...
    }
}
```

### 模式2：自动性能回归检测

```cpp
class PerformanceRegressionDetector {
public:
    void recordBaseline(const std::string& operation, double metric) {
        baselines_[operation] = metric;
    }
    
    bool checkRegression(const std::string& operation, double current_metric,
                        double threshold = 0.1) { // 10% threshold
        auto it = baselines_.find(operation);
        if (it == baselines_.end()) {
            std::cout << "⚠ No baseline for: " << operation << std::endl;
            return false;
        }
        
        double baseline = it->second;
        double change = (current_metric - baseline) / baseline;
        
        if (change > threshold) {
            std::cout << "❌ Performance regression detected!" << std::endl;
            std::cout << "   Operation: " << operation << std::endl;
            std::cout << "   Baseline: " << baseline << std::endl;
            std::cout << "   Current: " << current_metric << std::endl;
            std::cout << "   Change: " << (change * 100) << "%" << std::endl;
            return true;
        } else if (change < -threshold) {
            std::cout << "✅ Performance improvement!" << std::endl;
            std::cout << "   Operation: " << operation << std::endl;
            std::cout << "   Improvement: " << (-change * 100) << "%" << std::endl;
        }
        
        return false;
    }
    
private:
    std::map<std::string, double> baselines_;
};
```

### 模式3：配置自动调优

```cpp
class PoolConfigurationTuner {
public:
    PoolConfig findOptimalConfig(const std::string& workload_description) {
        std::cout << "🎯 Auto-tuning pool configuration..." << std::endl;
        std::cout << "Workload: " << workload_description << std::endl;
        
        std::vector<PoolConfig> configs_to_test = {
            PoolConfig::DefaultCPU(),
            PoolConfig::HighPerformanceCPU(),
            PoolConfig::DebugCPU(),
        };
        
        PoolConfig best_config;
        double best_score = 0.0;
        
        for (auto& config : configs_to_test) {
            double score = benchmark_configuration(config);
            
            std::cout << "  Config: " << config_name(config) 
                      << " Score: " << score << std::endl;
            
            if (score > best_score) {
                best_score = score;
                best_config = config;
            }
        }
        
        std::cout << "✅ Optimal configuration found!" << std::endl;
        return best_config;
    }
    
private:
    double benchmark_configuration(const PoolConfig& config) {
        // Run benchmark and return composite score
        // Consider: latency, throughput, memory efficiency
        return 0.0; // Placeholder
    }
    
    std::string config_name(const PoolConfig& config) {
        return "Config"; // Placeholder
    }
};
```

---

## 总结

本文档提供了详细的性能分析用例和示例，涵盖：

1. **真实场景**：游戏引擎、深度学习、Web服务器
2. **代码示例**：从基础到高级的完整实现
3. **优化工作流**：系统化的性能优化方法
4. **问题诊断**：常见性能问题的识别和解决
5. **最佳实践**：可复用的性能分析模式

这些示例都将被包含在最终的实现中，帮助用户快速理解和应用性能分析工具。
