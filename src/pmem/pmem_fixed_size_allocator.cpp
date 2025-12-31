#include "memory_pool/pmem/pmem_fixed_size_allocator.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace memory_pool {

PMEMFixedSizeAllocator::PMEMFixedSizeAllocator(const std::string& poolPath, size_t poolSize,
                                             size_t blockSize, size_t initialBlocks,
                                             size_t alignment, bool lockFree)
    : poolPath(poolPath), poolSize(poolSize), pmemAddr(nullptr), mappedLen(0), isPmem(0),
      blockSize(blockSize), alignedBlockSize(align_size(blockSize, alignment)),
      alignment(alignment), lockFree(lockFree), freeList(nullptr),
      totalBlocks(0), usedBlocks(0) {
    initializePMEM();
    allocateChunk(initialBlocks);
}

PMEMFixedSizeAllocator::~PMEMFixedSizeAllocator() {
    if (pmemAddr) {
        pmem_unmap(pmemAddr, mappedLen);
    }
}

void* PMEMFixedSizeAllocator::allocate(size_t size) {
    if (size > blockSize) {
        throw InvalidOperationException("Requested size exceeds block size");
    }

    Block* block = nullptr;

    if (lockFree) {
        // Lock-free allocation using atomic operations
        Block* oldHead = freeList.load(std::memory_order_acquire);
        do {
            if (!oldHead) {
                // No free blocks, allocate new chunk
                allocateChunk(1);
                oldHead = freeList.load(std::memory_order_acquire);
                if (!oldHead) {
                    throw OutOfMemoryException("Failed to allocate new chunk");
                }
            }
            block = oldHead;
        } while (!freeList.compare_exchange_weak(oldHead, oldHead->next,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire));
    } else {
        // Mutex-based allocation
        static std::mutex allocMutex;
        std::lock_guard<std::mutex> lock(allocMutex);

        if (!freeList) {
            allocateChunk(1);
        }

        if (!freeList) {
            throw OutOfMemoryException("Failed to allocate memory block");
        }

        block = freeList;
        freeList = freeList->next;
    }

    usedBlocks.fetch_add(1, std::memory_order_relaxed);
    return block;
}

void PMEMFixedSizeAllocator::deallocate(void* ptr) {
    if (!ptr || !owns(ptr)) {
        throw InvalidPointerException("Invalid pointer for deallocation");
    }

    Block* block = static_cast<Block*>(ptr);

    if (lockFree) {
        // Lock-free deallocation
        Block* oldHead = freeList.load(std::memory_order_acquire);
        do {
            block->next = oldHead;
        } while (!freeList.compare_exchange_weak(oldHead, block,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire));
    } else {
        // Mutex-based deallocation
        static std::mutex deallocMutex;
        std::lock_guard<std::mutex> lock(deallocMutex);

        block->next = freeList;
        freeList = block;
    }

    usedBlocks.fetch_sub(1, std::memory_order_relaxed);
}

void PMEMFixedSizeAllocator::reset() {
    // Reset all chunks and free lists
    freeList.store(nullptr, std::memory_order_release);
    chunks.clear();
    totalBlocks.store(0, std::memory_order_release);
    usedBlocks.store(0, std::memory_order_release);

    // Reinitialize with initial blocks
    allocateChunk(1);
}

size_t PMEMFixedSizeAllocator::getBlockSize(void* ptr) const {
    if (!owns(ptr)) {
        return 0;
    }
    return blockSize;
}

bool PMEMFixedSizeAllocator::owns(void* ptr) const {
    if (!ptr || !pmemAddr) {
        return false;
    }

    // Check if pointer is within any chunk
    for (const auto& chunk : chunks) {
        if (isPointerInChunk(ptr, chunk)) {
            return true;
        }
    }

    return false;
}

void PMEMFixedSizeAllocator::persist(void* ptr, size_t size) {
    if (isPmem) {
        pmem_persist(ptr, size);
    } else {
        pmem_msync(ptr, size);
    }
}

size_t PMEMFixedSizeAllocator::getBlockSize() const {
    return blockSize;
}

size_t PMEMFixedSizeAllocator::getCapacity() const {
    return totalBlocks.load(std::memory_order_relaxed);
}

size_t PMEMFixedSizeAllocator::getFreeBlocks() const {
    return totalBlocks.load(std::memory_order_relaxed) - usedBlocks.load(std::memory_order_relaxed);
}

size_t PMEMFixedSizeAllocator::getUsedBlocks() const {
    return usedBlocks.load(std::memory_order_relaxed);
}

void PMEMFixedSizeAllocator::initializePMEM() {
    // Create and map the persistent memory file
    pmemAddr = pmem_map_file(poolPath.c_str(), poolSize, PMEM_FILE_CREATE, 0666,
                            &mappedLen, &isPmem);
    if (!pmemAddr) {
        throw MemoryPoolException("Failed to map persistent memory file: " + poolPath);
    }

    std::cout << "Mapped " << mappedLen << " bytes of persistent memory at " << pmemAddr
              << " (is_pmem: " << isPmem << ")" << std::endl;
}

void PMEMFixedSizeAllocator::allocateChunk(size_t blockCount) {
    size_t chunkSize = blockCount * alignedBlockSize;
    if (chunkSize > poolSize) {
        throw OutOfMemoryException("Requested chunk size exceeds pool capacity");
    }

    // Allocate from PMEM pool
    void* chunkMemory = pmemAddr;

    // Initialize blocks in the chunk
    for (size_t i = 0; i < blockCount; ++i) {
        Block* block = getBlockFromIndex(chunkMemory, i);
        block->next = freeList.load(std::memory_order_acquire);
        freeList.store(block, std::memory_order_release);
    }

    chunks.emplace_back(chunkMemory, blockCount);
    totalBlocks.fetch_add(blockCount, std::memory_order_relaxed);

    // Persist the chunk initialization
    persist(chunkMemory, chunkSize);
}

bool PMEMFixedSizeAllocator::isPointerInChunk(const void* ptr, const Chunk& chunk) const {
    const char* ptrChar = static_cast<const char*>(ptr);
    const char* chunkStart = static_cast<const char*>(chunk.memory);
    const char* chunkEnd = chunkStart + (chunk.blockCount * alignedBlockSize);

    return (ptrChar >= chunkStart && ptrChar < chunkEnd);
}

PMEMFixedSizeAllocator::Block* PMEMFixedSizeAllocator::getBlockFromIndex(void* chunkStart, size_t index) const {
    char* blockPtr = static_cast<char*>(chunkStart) + (index * alignedBlockSize);
    return reinterpret_cast<Block*>(blockPtr);
}

}  // namespace memory_pool