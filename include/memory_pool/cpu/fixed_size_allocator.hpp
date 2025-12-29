#ifndef MEMORY_POOL_FIXED_SIZE_ALLOCATOR_HPP
#define MEMORY_POOL_FIXED_SIZE_ALLOCATOR_HPP

#include "cpu_memory_pool.hpp"
#include <vector>
#include <cstddef>
#include <cstdint>

namespace memory_pool {

// Fixed-size block allocator
class FixedSizeAllocator : public IAllocator {
public:
    // Constructor and destructor
    FixedSizeAllocator(size_t blockSize, size_t initialBlocks, size_t alignment = DEFAULT_ALIGNMENT);
    ~FixedSizeAllocator() override;
    
    // IAllocator interface implementation
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void reset() override;
    size_t getBlockSize(void* ptr) const override;
    bool owns(void* ptr) const override;
    
    // Fixed-size allocator specific methods
    size_t getBlockSize() const;
    size_t getCapacity() const;
    size_t getFreeBlocks() const;
    size_t getUsedBlocks() const;
    
private:
    // Block structure
    struct Block {
        Block* next;
    };
    
    // Chunk structure (contains multiple blocks)
    struct Chunk {
        void* memory;
        size_t blockCount;
        
        Chunk(void* mem, size_t count) : memory(mem), blockCount(count) {}
    };
    
    // Allocator properties
    size_t blockSize;
    size_t alignedBlockSize;
    size_t alignment;
    
    // Memory management
    Block* freeList;
    std::vector<Chunk> chunks;
    
    // Statistics
    size_t totalBlocks;
    size_t usedBlocks;
    
    // Helper methods
    void allocateChunk(size_t blockCount);
    bool isPointerInChunk(const void* ptr, const Chunk& chunk) const;
    Block* getBlockFromIndex(void* chunkStart, size_t index) const;
};

} // namespace memory_pool

#endif // MEMORY_POOL_FIXED_SIZE_ALLOCATOR_HPP