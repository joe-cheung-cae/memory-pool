#ifndef MEMORY_POOL_PMEM_VARIABLE_SIZE_ALLOCATOR_HPP
#define MEMORY_POOL_PMEM_VARIABLE_SIZE_ALLOCATOR_HPP

#include "memory_pool/cpu/cpu_memory_pool.hpp"
#include "memory_pool/utils/thread_safety.hpp"
#include <map>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <atomic>

#ifdef HAVE_PMEM
#include <libpmem.h>
#endif

namespace memory_pool {

/**
 * @brief Variable-size allocator for persistent memory.
 *
 * This allocator manages persistent memory with variable-size allocations,
 * using segregated free lists for different size classes with persistence guarantees.
 */
class PMEMVariableSizeAllocator : public IAllocator {
  public:
    /**
     * @brief Constructs a PMEM variable-size allocator.
     * @param poolPath Path to the persistent memory pool file.
     * @param poolSize Total size of the persistent memory pool.
     * @param alignment Memory alignment requirement.
     * @param lockFree Whether to use lock-free operations.
     */
    PMEMVariableSizeAllocator(const std::string& poolPath, size_t poolSize,
                             size_t alignment = DEFAULT_ALIGNMENT, bool lockFree = false);

    /**
     * @brief Destroys the PMEM variable-size allocator.
     */
    ~PMEMVariableSizeAllocator() override;

    // IAllocator interface implementation
    /**
     * @brief Allocates memory of the specified size.
     * @param size The size in bytes to allocate.
     * @return Pointer to the allocated memory.
     */
    void* allocate(size_t size) override;

    /**
     * @brief Deallocates previously allocated memory.
     * @param ptr Pointer to the memory to deallocate.
     */
    void deallocate(void* ptr) override;

    /**
     * @brief Resets the allocator, deallocating all memory.
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

    // PMEM-specific methods
    /**
     * @brief Persists data to persistent memory.
     * @param ptr Pointer to the data to persist.
     * @param size Size of the data to persist.
     */
    void persist(void* ptr, size_t size);

    /**
     * @brief Gets the total allocated memory.
     * @return Total allocated bytes.
     */
    size_t getTotalAllocated() const;

    /**
     * @brief Gets the current used memory.
     * @return Current used bytes.
     */
    size_t getCurrentUsed() const;

  private:
    // Block header for allocated blocks
    struct BlockHeader {
        size_t size;
        bool   isFree;
    };

    // Free block structure
    struct FreeBlock {
        size_t     size;
        FreeBlock* next;
    };

    // PMEM pool information
    std::string poolPath;
    size_t      poolSize;
    void*       pmemAddr;
    size_t      mappedLen;
    int         isPmem;

    // Allocator properties
    size_t alignment;
    bool   lockFree;

    // Free lists for different size classes
    std::map<size_t, std::atomic<FreeBlock*>> freeLists;

    // Statistics
    std::atomic<size_t> totalAllocated;
    std::atomic<size_t> currentUsed;

    // Helper methods
    /**
     * @brief Initializes the persistent memory pool.
     */
    void initializePMEM();

    /**
     * @brief Gets the size class for a given size.
     * @param size The allocation size.
     * @return The size class.
     */
    size_t getSizeClass(size_t size) const;

    /**
     * @brief Finds a suitable free block for the given size.
     * @param size The requested size.
     * @return Pointer to a suitable free block, or nullptr.
     */
    FreeBlock* findFreeBlock(size_t size);

    /**
     * @brief Splits a free block if it's larger than needed.
     * @param block The block to split.
     * @param size The requested size.
     * @return Pointer to the allocated block.
     */
    void* splitBlock(FreeBlock* block, size_t size);

    /**
     * @brief Coalesces adjacent free blocks.
     * @param block The block to coalesce with neighbors.
     */
    void coalesceBlock(FreeBlock* block);

    /**
     * @brief Gets the block header for a pointer.
     * @param ptr The pointer.
     * @return Pointer to the block header.
     */
    BlockHeader* getBlockHeader(void* ptr) const;

    /**
     * @brief Checks if a pointer is within the PMEM pool.
     * @param ptr The pointer to check.
     * @return True if the pointer is in the pool.
     */
    bool isPointerInPool(const void* ptr) const;
};

}  // namespace memory_pool

#endif  // MEMORY_POOL_PMEM_VARIABLE_SIZE_ALLOCATOR_HPP