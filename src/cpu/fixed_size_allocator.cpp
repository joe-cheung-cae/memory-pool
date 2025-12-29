#include "memory_pool/cpu/fixed_size_allocator.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace memory_pool {

FixedSizeAllocator::FixedSizeAllocator(size_t blockSize, size_t initialBlocks, size_t alignment, bool lockFree)
    : blockSize(blockSize),
      alignedBlockSize(align_size(std::max(blockSize, sizeof(Block)), alignment)),
      alignment(alignment),
      lockFree(lockFree),
      freeList(nullptr),
      totalBlocks(0),
      usedBlocks(0) {
    // Allocate initial chunk if initialBlocks > 0
    if (initialBlocks > 0) {
        allocateChunk(initialBlocks);
    }
}

FixedSizeAllocator::~FixedSizeAllocator() {
    // Free all allocated chunks
    for (const auto& chunk : chunks) {
        free(chunk.memory);
    }

    // Clear data structures
    chunks.clear();
    freeList = nullptr;
}

void* FixedSizeAllocator::allocate(size_t size) {
    // Check if the requested size fits in our block size
    if (size > blockSize) {
        reportError(ErrorSeverity::Warning, "FixedSizeAllocator: Requested size " + std::to_string(size) +
                                                " exceeds block size " + std::to_string(blockSize));
        return nullptr;
    }

    // If no free blocks, allocate a new chunk
    if (freeList == nullptr) {
        // Calculate how many blocks to allocate
        // Start with at least 1, or double the current total
        size_t newBlockCount = std::max<size_t>(1, totalBlocks);
        allocateChunk(newBlockCount);

        // If still no free blocks after allocation, we're out of memory
        if (freeList == nullptr) {
            reportError(ErrorSeverity::Error, "FixedSizeAllocator: Out of memory");
            return nullptr;
        }
    }

    // Get a block from the free list
    Block* block;
    if (lockFree) {
        // Lock-free allocation using atomic operations
        do {
            block = freeList.load();
            if (block == nullptr) break;
        } while (!freeList.compare_exchange_weak(block, block->next));
    } else {
        block    = freeList;
        freeList = block->next;
    }

    // Update statistics
    usedBlocks++;

    // Return the block's memory
    return block;
}

void FixedSizeAllocator::deallocate(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    // Verify that the pointer belongs to this allocator
    if (!owns(ptr)) {
        reportError(ErrorSeverity::Error,
                    "FixedSizeAllocator: Attempted to deallocate memory not owned by this allocator");
        return;
    }

    // Cast the pointer to a Block
    Block* block = static_cast<Block*>(ptr);

    // Add the block to the free list
    if (lockFree) {
        // Lock-free deallocation using atomic operations
        Block* oldHead;
        do {
            oldHead     = freeList.load();
            block->next = oldHead;
        } while (!freeList.compare_exchange_weak(oldHead, block));
    } else {
        block->next = freeList;
        freeList    = block;
    }

    // Update statistics
    usedBlocks--;
}

void FixedSizeAllocator::reset() {
    // Reset the free list
    freeList = nullptr;

    // Rebuild the free list from all chunks
    for (auto& chunk : chunks) {
        for (size_t i = 0; i < chunk.blockCount; ++i) {
            Block* block = getBlockFromIndex(chunk.memory, i);
            block->next  = freeList;
            freeList     = block;
        }
    }

    // Reset statistics
    usedBlocks = 0;
}

size_t FixedSizeAllocator::getBlockSize(void* ptr) const {
    // All blocks have the same size
    return blockSize;
}

bool FixedSizeAllocator::owns(void* ptr) const {
    if (ptr == nullptr) {
        return false;
    }

    // Check if the pointer is within any of our chunks
    for (const auto& chunk : chunks) {
        if (isPointerInChunk(ptr, chunk)) {
            return true;
        }
    }

    return false;
}

size_t FixedSizeAllocator::getBlockSize() const { return blockSize; }

size_t FixedSizeAllocator::getCapacity() const { return totalBlocks; }

size_t FixedSizeAllocator::getFreeBlocks() const { return totalBlocks - usedBlocks; }

size_t FixedSizeAllocator::getUsedBlocks() const { return usedBlocks; }

void FixedSizeAllocator::allocateChunk(size_t blockCount) {
    // Allocate memory for the chunk
    size_t chunkSize = alignedBlockSize * blockCount;
    void*  memory    = aligned_alloc(alignment, chunkSize);

    if (memory == nullptr) {
        reportError(ErrorSeverity::Error, "FixedSizeAllocator: Failed to allocate memory for chunk");
        return;
    }

    // Initialize the chunk
    Chunk chunk(memory, blockCount);

    // Initialize the blocks in the chunk
    for (size_t i = 0; i < blockCount; ++i) {
        Block* block = getBlockFromIndex(memory, i);
        block->next  = freeList;
        freeList     = block;
    }

    // Add the chunk to our list
    chunks.push_back(chunk);

    // Update statistics
    totalBlocks += blockCount;
}

bool FixedSizeAllocator::isPointerInChunk(const void* ptr, const Chunk& chunk) const {
    const char* charPtr    = static_cast<const char*>(ptr);
    const char* chunkStart = static_cast<const char*>(chunk.memory);
    const char* chunkEnd   = chunkStart + (alignedBlockSize * chunk.blockCount);

    return charPtr >= chunkStart && charPtr < chunkEnd;
}

FixedSizeAllocator::Block* FixedSizeAllocator::getBlockFromIndex(void* chunkStart, size_t index) const {
    char* blockPtr = static_cast<char*>(chunkStart) + (index * alignedBlockSize);
    return reinterpret_cast<Block*>(blockPtr);
}

}  // namespace memory_pool