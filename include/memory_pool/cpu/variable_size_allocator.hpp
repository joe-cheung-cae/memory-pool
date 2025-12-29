#ifndef MEMORY_POOL_VARIABLE_SIZE_ALLOCATOR_HPP
#define MEMORY_POOL_VARIABLE_SIZE_ALLOCATOR_HPP

#include "cpu_memory_pool.hpp"
#include <map>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace memory_pool {

// Variable-size allocator
class VariableSizeAllocator : public IAllocator {
public:
    // Constructor and destructor
    VariableSizeAllocator(size_t initialSize, size_t alignment = DEFAULT_ALIGNMENT);
    ~VariableSizeAllocator() override;
    
    // IAllocator interface implementation
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void reset() override;
    size_t getBlockSize(void* ptr) const override;
    bool owns(void* ptr) const override;
    
    // Variable-size allocator specific methods
    size_t getTotalSize() const;
    size_t getUsedSize() const;
    size_t getFreeSize() const;
    size_t getLargestFreeBlock() const;
    double getFragmentationRatio() const;
    
private:
    // Block header structure
    struct BlockHeader {
        size_t size;       // Size of the block (including header)
        bool isFree;       // Whether the block is free
        BlockHeader* prev; // Previous block in memory
        BlockHeader* next; // Next block in memory
        
        // Get pointer to the data area
        void* getData() {
            return reinterpret_cast<void*>(reinterpret_cast<char*>(this) + sizeof(BlockHeader));
        }
        
        // Get block header from data pointer
        static BlockHeader* fromData(void* data) {
            return reinterpret_cast<BlockHeader*>(reinterpret_cast<char*>(data) - sizeof(BlockHeader));
        }
        
        // Get next block in memory
        BlockHeader* getNextPhysical() {
            if (next && next > this) {
                return next;
            }
            return nullptr;
        }
        
        // Get previous block in memory
        BlockHeader* getPrevPhysical() {
            if (prev && prev < this) {
                return prev;
            }
            return nullptr;
        }
    };
    
    // Memory region structure
    struct MemoryRegion {
        void* memory;
        size_t size;
        BlockHeader* firstBlock;
        
        MemoryRegion(void* mem, size_t s) : memory(mem), size(s), firstBlock(nullptr) {}
    };
    
    // Allocator properties
    size_t alignment;
    size_t minBlockSize;
    
    // Memory management
    std::vector<MemoryRegion> regions;
    std::multimap<size_t, BlockHeader*> freeBlocks; // Map of size -> free block
    
    // Statistics
    size_t totalSize;
    size_t usedSize;
    
    // Helper methods
    void addRegion(size_t size);
    void addToFreeList(BlockHeader* block);
    void removeFromFreeList(BlockHeader* block);
    BlockHeader* findBestFit(size_t size);
    void splitBlock(BlockHeader* block, size_t size);
    void mergeWithNeighbors(BlockHeader* block);
    bool isPointerInRegion(const void* ptr, const MemoryRegion& region) const;
};

} // namespace memory_pool

#endif // MEMORY_POOL_VARIABLE_SIZE_ALLOCATOR_HPP