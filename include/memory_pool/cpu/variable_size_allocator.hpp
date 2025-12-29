#ifndef MEMORY_POOL_VARIABLE_SIZE_ALLOCATOR_HPP
#define MEMORY_POOL_VARIABLE_SIZE_ALLOCATOR_HPP

#include "cpu_memory_pool.hpp"
#include <map>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace memory_pool {

/**
 * @brief Variable-size allocator.
 *
 * This allocator manages memory in variable-size blocks, providing
 * efficient allocation and deallocation for objects of different sizes.
 */
class VariableSizeAllocator : public IAllocator {
  public:
    /**
     * @brief Constructs a variable-size allocator.
     * @param initialSize The initial size of the memory pool.
     * @param alignment Memory alignment requirement.
     */
    VariableSizeAllocator(size_t initialSize, size_t alignment = DEFAULT_ALIGNMENT);

    /**
     * @brief Destroys the variable-size allocator.
     */
    ~VariableSizeAllocator() override;

    // IAllocator interface implementation
    void*  allocate(size_t size) override;
    void   deallocate(void* ptr) override;
    void   reset() override;
    size_t getBlockSize(void* ptr) const override;
    bool   owns(void* ptr) const override;

    // Variable-size allocator specific methods
    /**
     * @brief Gets the total size of the allocator.
     * @return The total size in bytes.
     */
    size_t getTotalSize() const;

    /**
     * @brief Gets the used size of the allocator.
     * @return The used size in bytes.
     */
    size_t getUsedSize() const;

    /**
     * @brief Gets the free size of the allocator.
     * @return The free size in bytes.
     */
    size_t getFreeSize() const;

    /**
     * @brief Gets the size of the largest free block.
     * @return The largest free block size in bytes.
     */
    size_t getLargestFreeBlock() const;

    /**
     * @brief Gets the fragmentation ratio.
     * @return The fragmentation ratio (0.0 to 1.0).
     */
    double getFragmentationRatio() const;

  private:
    // Block header structure
    struct BlockHeader {
        size_t       size;    // Size of the block (including header)
        bool         isFree;  // Whether the block is free
        BlockHeader* prev;    // Previous block in memory
        BlockHeader* next;    // Next block in memory

        // Get pointer to the data area
        void* getData() { return reinterpret_cast<void*>(reinterpret_cast<char*>(this) + sizeof(BlockHeader)); }

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
        void*        memory;
        size_t       size;
        BlockHeader* firstBlock;

        MemoryRegion(void* mem, size_t s) : memory(mem), size(s), firstBlock(nullptr) {}
    };

    // Allocator properties
    size_t alignment;
    size_t minBlockSize;

    // Memory management
    std::vector<MemoryRegion>           regions;
    std::multimap<size_t, BlockHeader*> freeBlocks;  // Map of size -> free block

    // Statistics
    size_t totalSize;
    size_t usedSize;

    // Helper methods
    void         addRegion(size_t size);
    void         addToFreeList(BlockHeader* block);
    void         removeFromFreeList(BlockHeader* block);
    BlockHeader* findBestFit(size_t size);
    void         splitBlock(BlockHeader* block, size_t size);
    void         mergeWithNeighbors(BlockHeader* block);
    bool         isPointerInRegion(const void* ptr, const MemoryRegion& region) const;
};

}  // namespace memory_pool

#endif  // MEMORY_POOL_VARIABLE_SIZE_ALLOCATOR_HPP