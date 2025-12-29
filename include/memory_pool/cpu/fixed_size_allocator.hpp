#ifndef MEMORY_POOL_FIXED_SIZE_ALLOCATOR_HPP
#define MEMORY_POOL_FIXED_SIZE_ALLOCATOR_HPP

#include "cpu_memory_pool.hpp"
#include "memory_pool/utils/thread_safety.hpp"
#include <vector>
#include <cstddef>
#include <cstdint>
#include <atomic>

namespace memory_pool {

/**
 * @brief Fixed-size block allocator.
 *
 * This allocator manages memory in fixed-size blocks, providing efficient
 * allocation and deallocation for objects of uniform size.
 */
class FixedSizeAllocator : public IAllocator {
  public:
    /**
     * @brief Constructs a fixed-size allocator.
     * @param blockSize The size of each block in bytes.
     * @param initialBlocks The initial number of blocks to allocate.
     * @param alignment Memory alignment requirement.
     * @param lockFree Whether to use lock-free operations.
     */
    FixedSizeAllocator(size_t blockSize, size_t initialBlocks, size_t alignment = DEFAULT_ALIGNMENT,
                       bool lockFree = false);

    /**
     * @brief Destroys the fixed-size allocator.
     */
    ~FixedSizeAllocator() override;

    // IAllocator interface implementation
    /**
     * @brief Allocates a fixed-size block.
     * @param size The size to allocate (must match block size).
     * @return Pointer to the allocated block.
     */
    void* allocate(size_t size) override;

    /**
     * @brief Deallocates a fixed-size block.
     * @param ptr Pointer to the block to deallocate.
     */
    void deallocate(void* ptr) override;

    /**
     * @brief Resets the allocator, deallocating all blocks.
     */
    void reset() override;

    /**
     * @brief Gets the block size of the allocated memory.
     * @param ptr Pointer to the allocated memory.
     * @return The block size.
     */
    size_t getBlockSize(void* ptr) const override;

    /**
     * @brief Checks if the allocator owns the given pointer.
     * @param ptr Pointer to check.
     * @return True if the allocator owns the pointer.
     */
    bool owns(void* ptr) const override;

    // Fixed-size allocator specific methods
    /**
     * @brief Gets the fixed block size.
     * @return The block size in bytes.
     */
    size_t getBlockSize() const;

    /**
     * @brief Gets the total capacity in blocks.
     * @return The total number of blocks.
     */
    size_t getCapacity() const;

    /**
     * @brief Gets the number of free blocks.
     * @return The number of free blocks.
     */
    size_t getFreeBlocks() const;

    /**
     * @brief Gets the number of used blocks.
     * @return The number of used blocks.
     */
    size_t getUsedBlocks() const;

  private:
    // Block structure
    struct Block {
        Block* next;
    };

    // Chunk structure (contains multiple blocks)
    struct Chunk {
        void*  memory;
        size_t blockCount;

        Chunk(void* mem, size_t count) : memory(mem), blockCount(count) {}
    };

    // Allocator properties
    size_t blockSize;
    size_t alignedBlockSize;
    size_t alignment;
    bool   lockFree;

    // Memory management
    std::atomic<Block*> freeList;
    std::vector<Chunk>  chunks;

    // Statistics
    std::atomic<size_t> totalBlocks;
    std::atomic<size_t> usedBlocks;

    // Helper methods
    /**
     * @brief Allocates a new chunk of memory.
     * @param blockCount The number of blocks in the chunk.
     */
    void allocateChunk(size_t blockCount);

    /**
     * @brief Checks if a pointer is within a chunk.
     * @param ptr The pointer to check.
     * @param chunk The chunk to check against.
     * @return True if the pointer is in the chunk.
     */
    bool isPointerInChunk(const void* ptr, const Chunk& chunk) const;

    /**
     * @brief Gets a block from a chunk by index.
     * @param chunkStart The start of the chunk.
     * @param index The block index.
     * @return Pointer to the block.
     */
    Block* getBlockFromIndex(void* chunkStart, size_t index) const;
};

}  // namespace memory_pool

#endif  // MEMORY_POOL_FIXED_SIZE_ALLOCATOR_HPP