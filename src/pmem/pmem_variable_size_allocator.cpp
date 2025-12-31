#include "memory_pool/pmem/pmem_variable_size_allocator.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <cstring>
#include <algorithm>
#include <iostream>

namespace memory_pool {

PMEMVariableSizeAllocator::PMEMVariableSizeAllocator(const std::string& poolPath, size_t poolSize,
                                                   size_t alignment, bool lockFree)
    : poolPath(poolPath), poolSize(poolSize), pmemAddr(nullptr), mappedLen(0), isPmem(0),
      alignment(alignment), lockFree(lockFree), totalAllocated(0), currentUsed(0) {
    initializePMEM();

    // Initialize with one large free block
    FreeBlock* initialBlock = static_cast<FreeBlock*>(pmemAddr);
    initialBlock->size = poolSize - sizeof(BlockHeader) - sizeof(FreeBlock);
    initialBlock->next = nullptr;

    freeLists[getSizeClass(initialBlock->size)] = initialBlock;

    // Persist the initial block
    persist(initialBlock, sizeof(FreeBlock));
}

PMEMVariableSizeAllocator::~PMEMVariableSizeAllocator() {
    if (pmemAddr) {
        pmem_unmap(pmemAddr, mappedLen);
    }
}

void* PMEMVariableSizeAllocator::allocate(size_t size) {
    size_t alignedSize = align_size(size + sizeof(BlockHeader), alignment);

    FreeBlock* block = findFreeBlock(alignedSize);
    if (!block) {
        throw OutOfMemoryException("No suitable free block found for allocation");
    }

    void* allocatedPtr = splitBlock(block, alignedSize);
    if (allocatedPtr) {
        currentUsed.fetch_add(alignedSize, std::memory_order_relaxed);
        totalAllocated.fetch_add(alignedSize, std::memory_order_relaxed);
    }

    return allocatedPtr;
}

void PMEMVariableSizeAllocator::deallocate(void* ptr) {
    if (!ptr || !owns(ptr)) {
        throw InvalidPointerException("Invalid pointer for deallocation");
    }

    BlockHeader* header = getBlockHeader(ptr);
    size_t blockSize = header->size;

    // Mark block as free
    header->isFree = true;

    // Create free block structure
    FreeBlock* freeBlock = static_cast<FreeBlock*>(ptr);
    freeBlock->size = blockSize;
    freeBlock->next = nullptr;

    // Add to appropriate free list
    size_t sizeClass = getSizeClass(blockSize);
    freeBlock->next = freeLists[sizeClass].load(std::memory_order_acquire);
    freeLists[sizeClass].store(freeBlock, std::memory_order_release);

    // Try to coalesce with adjacent blocks
    coalesceBlock(freeBlock);

    currentUsed.fetch_sub(blockSize, std::memory_order_relaxed);

    // Persist the deallocation
    persist(header, sizeof(BlockHeader));
}

void PMEMVariableSizeAllocator::reset() {
    // Reset the entire pool to one large free block
    freeLists.clear();

    FreeBlock* initialBlock = static_cast<FreeBlock*>(pmemAddr);
    initialBlock->size = poolSize - sizeof(BlockHeader) - sizeof(FreeBlock);
    initialBlock->next = nullptr;

    freeLists[getSizeClass(initialBlock->size)] = initialBlock;

    totalAllocated.store(0, std::memory_order_release);
    currentUsed.store(0, std::memory_order_release);

    // Persist the reset
    persist(pmemAddr, poolSize);
}

size_t PMEMVariableSizeAllocator::getBlockSize(void* ptr) const {
    if (!owns(ptr)) {
        return 0;
    }

    BlockHeader* header = getBlockHeader(ptr);
    return header->size - sizeof(BlockHeader);
}

bool PMEMVariableSizeAllocator::owns(void* ptr) const {
    return isPointerInPool(ptr);
}

void PMEMVariableSizeAllocator::persist(void* ptr, size_t size) {
    if (isPmem) {
        pmem_persist(ptr, size);
    } else {
        pmem_msync(ptr, size);
    }
}

size_t PMEMVariableSizeAllocator::getTotalAllocated() const {
    return totalAllocated.load(std::memory_order_relaxed);
}

size_t PMEMVariableSizeAllocator::getCurrentUsed() const {
    return currentUsed.load(std::memory_order_relaxed);
}

void PMEMVariableSizeAllocator::initializePMEM() {
    // Create and map the persistent memory file
    pmemAddr = pmem_map_file(poolPath.c_str(), poolSize, PMEM_FILE_CREATE, 0666,
                            &mappedLen, &isPmem);
    if (!pmemAddr) {
        throw MemoryPoolException("Failed to map persistent memory file: " + poolPath);
    }

    std::cout << "Mapped " << mappedLen << " bytes of persistent memory at " << pmemAddr
              << " (is_pmem: " << isPmem << ")" << std::endl;
}

size_t PMEMVariableSizeAllocator::getSizeClass(size_t size) const {
    // Simple size class calculation - round up to next power of 2
    size_t sizeClass = 1;
    while (sizeClass < size) {
        sizeClass <<= 1;
    }
    return sizeClass;
}

PMEMVariableSizeAllocator::FreeBlock* PMEMVariableSizeAllocator::findFreeBlock(size_t size) {
    // Find the smallest size class that can accommodate the request
    for (auto& pair : freeLists) {
        if (pair.first >= size) {
            FreeBlock* block = pair.second.load(std::memory_order_acquire);
            if (block) {
                // Remove from free list
                pair.second.store(block->next, std::memory_order_release);
                return block;
            }
        }
    }

    // Try first-fit in larger size classes
    for (auto& pair : freeLists) {
        FreeBlock* block = pair.second.load(std::memory_order_acquire);
        while (block) {
            if (block->size >= size) {
                // Remove from free list
                pair.second.store(block->next, std::memory_order_release);
                return block;
            }
            block = block->next;
        }
    }

    return nullptr;
}

void* PMEMVariableSizeAllocator::splitBlock(FreeBlock* block, size_t size) {
    if (block->size < size) {
        return nullptr;
    }

    // Calculate remaining size after allocation
    size_t remainingSize = block->size - size;

    // Set up block header at the start of the block
    BlockHeader* header = reinterpret_cast<BlockHeader*>(block);
    header->size = size;
    header->isFree = false;

    void* userPtr = reinterpret_cast<char*>(block) + sizeof(BlockHeader);

    // If there's remaining space, create a new free block
    if (remainingSize > sizeof(FreeBlock)) {
        FreeBlock* newFreeBlock = reinterpret_cast<FreeBlock*>(
            reinterpret_cast<char*>(block) + size);
        newFreeBlock->size = remainingSize;
        newFreeBlock->next = nullptr;

        // Add to appropriate free list
        size_t sizeClass = getSizeClass(newFreeBlock->size);
        newFreeBlock->next = freeLists[sizeClass].load(std::memory_order_acquire);
        freeLists[sizeClass].store(newFreeBlock, std::memory_order_release);
    }

    // Persist the allocation (header + user data)
    persist(header, size);

    return userPtr;
}

void PMEMVariableSizeAllocator::coalesceBlock(FreeBlock* block) {
    // For simplicity, we'll skip complex coalescing in this implementation
    // A full implementation would check adjacent blocks and merge them
}

PMEMVariableSizeAllocator::BlockHeader* PMEMVariableSizeAllocator::getBlockHeader(void* ptr) const {
    return reinterpret_cast<BlockHeader*>(
        reinterpret_cast<char*>(ptr) - sizeof(BlockHeader));
}

bool PMEMVariableSizeAllocator::isPointerInPool(const void* ptr) const {
    if (!ptr || !pmemAddr) {
        return false;
    }

    const char* ptrChar = static_cast<const char*>(ptr);
    const char* poolStart = static_cast<const char*>(pmemAddr);
    const char* poolEnd = poolStart + mappedLen;

    return (ptrChar >= poolStart && ptrChar < poolEnd);
}

}  // namespace memory_pool