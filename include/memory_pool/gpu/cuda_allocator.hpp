#ifndef MEMORY_POOL_CUDA_ALLOCATOR_HPP
#define MEMORY_POOL_CUDA_ALLOCATOR_HPP

#include "gpu_memory_pool.hpp"
#include "cuda_utils.hpp"
#include <vector>
#include <deque>
#include <map>
#include <cstddef>

namespace memory_pool {

// Base CUDA allocator
class CudaAllocatorBase : public ICudaAllocator {
  public:
    // Constructor and destructor
    CudaAllocatorBase(int deviceId);
    ~CudaAllocatorBase() override;

    // ICudaAllocator interface implementation for device management
    void         setDevice(int deviceId) override;
    int          getDevice() const override;
    void         setStream(cudaStream_t stream) override;
    cudaStream_t getStream() const override;

  protected:
    // CUDA properties
    int          deviceId;
    cudaStream_t stream;

    // Helper methods
    void ensureCorrectDevice() const;
};

// CUDA fixed-size block allocator
class CudaFixedSizeAllocator : public CudaAllocatorBase {
  public:
    // Constructor and destructor
    CudaFixedSizeAllocator(size_t blockSize, size_t initialBlocks, int deviceId,
                           AllocFlags defaultFlags = AllocFlags::None);
    ~CudaFixedSizeAllocator() override;

    // ICudaAllocator interface implementation
    void*  allocate(size_t size, AllocFlags flags) override;
    void   deallocate(void* ptr) override;
    void   reset() override;
    size_t getBlockSize(void* ptr) const override;
    bool   owns(void* ptr) const override;

    // Fixed-size allocator specific methods
    size_t getBlockSize() const;
    size_t getCapacity() const;
    size_t getFreeBlocks() const;
    size_t getUsedBlocks() const;

  private:
    // Block structure (stored in host memory)
    struct Block {
        void* devicePtr;
        bool  isFree;

        Block(void* ptr) : devicePtr(ptr), isFree(true) {}
    };

    // Chunk structure (contains multiple blocks)
    struct Chunk {
        void*              deviceMemory;
        size_t             blockCount;
        std::vector<Block> blocks;

        Chunk(void* mem, size_t count) : deviceMemory(mem), blockCount(count) { blocks.reserve(count); }
    };

    // Allocator properties
    size_t     blockSize;
    AllocFlags defaultFlags;

    // Memory management
    std::vector<Chunk>      chunks;
    std::vector<Block*>     freeBlocks;
    std::map<void*, Block*> allocatedBlocks;

    // Statistics
    size_t totalBlocks;
    size_t usedBlocks;

    // Helper methods
    void allocateChunk(size_t blockCount);
    bool isPointerInChunk(const void* ptr, const Chunk& chunk) const;
};

// CUDA variable-size allocator
class CudaVariableSizeAllocator : public CudaAllocatorBase {
  public:
    // Constructor and destructor
    CudaVariableSizeAllocator(size_t initialSize, int deviceId, AllocFlags defaultFlags = AllocFlags::None);
    ~CudaVariableSizeAllocator() override;

    // ICudaAllocator interface implementation
    void*  allocate(size_t size, AllocFlags flags) override;
    void   deallocate(void* ptr) override;
    void   reset() override;
    size_t getBlockSize(void* ptr) const override;
    bool   owns(void* ptr) const override;

    // Variable-size allocator specific methods
    size_t getTotalSize() const;
    size_t getUsedSize() const;
    size_t getFreeSize() const;

  private:
    // Block structure (stored in host memory)
    struct Block {
        void*  devicePtr;
        size_t size;
        bool   isFree;

        Block(void* ptr, size_t s) : devicePtr(ptr), size(s), isFree(true) {}
    };

    // Memory region structure
    struct MemoryRegion {
        void*             deviceMemory;
        size_t            size;
        std::deque<Block> blocks;

        MemoryRegion(void* mem, size_t s) : deviceMemory(mem), size(s) {}
    };

    // Allocator properties
    AllocFlags defaultFlags;

    // Memory management
    std::vector<MemoryRegion>     regions;
    std::multimap<size_t, Block*> freeBlocks;  // Map of size -> free block
    std::map<void*, Block*>       allocatedBlocks;

    // Statistics
    size_t totalSize;
    size_t usedSize;

    // Helper methods
    void   addRegion(size_t size);
    void   addToFreeList(Block* block);
    void   removeFromFreeList(Block* block);
    Block* findBestFit(size_t size);
    void   splitBlock(Block* block, size_t size);
    void   mergeAdjacentBlocks();
    bool   isPointerInRegion(const void* ptr, const MemoryRegion& region) const;
};

}  // namespace memory_pool

#endif  // MEMORY_POOL_CUDA_ALLOCATOR_HPP