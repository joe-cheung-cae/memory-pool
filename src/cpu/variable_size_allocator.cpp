#include "../../include/memory_pool/cpu/variable_size_allocator.hpp"
#include "../../include/memory_pool/utils/error_handling.hpp"
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace memory_pool {

VariableSizeAllocator::VariableSizeAllocator(size_t initialSize, size_t alignment)
    : alignment(alignment),
      minBlockSize(align_size(sizeof(BlockHeader) + 1, alignment)),
      totalSize(0),
      usedSize(0) {
    
    // Allocate initial region if initialSize > 0
    if (initialSize > 0) {
        addRegion(initialSize);
    }
}

VariableSizeAllocator::~VariableSizeAllocator() {
    // Free all allocated regions
    for (const auto& region : regions) {
        free(region.memory);
    }
    
    // Clear data structures
    regions.clear();
    freeBlocks.clear();
}

void* VariableSizeAllocator::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    
    // Calculate the total size needed including header
    size_t totalSize = align_size(sizeof(BlockHeader) + size, alignment);
    
    // Find a suitable block
    BlockHeader* block = findBestFit(totalSize);
    
    // If no suitable block found, allocate a new region
    if (block == nullptr) {
        // Calculate the size of the new region
        // Either double the current total size or use the requested size, whichever is larger
        size_t newRegionSize = std::max(totalSize, this->totalSize * 2);
        
        // Ensure minimum size
        newRegionSize = std::max(newRegionSize, minBlockSize * 16);
        
        // Add the new region
        addRegion(newRegionSize);
        
        // Try to find a suitable block again
        block = findBestFit(totalSize);
        
        // If still no suitable block, we're out of memory
        if (block == nullptr) {
            reportError(ErrorSeverity::Error, 
                "VariableSizeAllocator: Out of memory, failed to allocate " + 
                std::to_string(size) + " bytes");
            return nullptr;
        }
    }
    
    // Remove the block from the free list
    removeFromFreeList(block);
    
    // Split the block if it's much larger than needed
    if (block->size >= totalSize + minBlockSize) {
        splitBlock(block, totalSize);
    }
    
    // Mark the block as used
    block->isFree = false;
    
    // Update statistics
    usedSize += block->size;
    
    // Return the data portion of the block
    return block->getData();
}

void VariableSizeAllocator::deallocate(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    
    // Verify that the pointer belongs to this allocator
    if (!owns(ptr)) {
        reportError(ErrorSeverity::Error, 
            "VariableSizeAllocator: Attempted to deallocate memory not owned by this allocator");
        return;
    }
    
    // Get the block header from the pointer
    BlockHeader* block = BlockHeader::fromData(ptr);
    
    // Update statistics
    usedSize -= block->size;
    
    // Mark the block as free
    block->isFree = true;
    
    // Merge with adjacent free blocks
    mergeWithNeighbors(block);
    
    // Add the block to the free list
    addToFreeList(block);
}

void VariableSizeAllocator::reset() {
    // Clear the free list
    freeBlocks.clear();
    
    // Reset all regions
    for (auto& region : regions) {
        // Initialize the first block in the region
        BlockHeader* block = static_cast<BlockHeader*>(region.memory);
        block->size = region.size;
        block->isFree = true;
        block->prev = nullptr;
        block->next = nullptr;
        
        // Set the first block in the region
        region.firstBlock = block;
        
        // Add the block to the free list
        addToFreeList(block);
    }
    
    // Reset statistics
    usedSize = 0;
}

size_t VariableSizeAllocator::getBlockSize(void* ptr) const {
    if (ptr == nullptr) {
        return 0;
    }
    
    // Verify that the pointer belongs to this allocator
    if (!owns(ptr)) {
        reportError(ErrorSeverity::Error, 
            "VariableSizeAllocator: Attempted to get size of memory not owned by this allocator");
        return 0;
    }
    
    // Get the block header from the pointer
    BlockHeader* block = BlockHeader::fromData(ptr);
    
    // Return the size of the data portion
    return block->size - sizeof(BlockHeader);
}

bool VariableSizeAllocator::owns(void* ptr) const {
    if (ptr == nullptr) {
        return false;
    }
    
    // Check if the pointer is within any of our regions
    for (const auto& region : regions) {
        if (isPointerInRegion(ptr, region)) {
            return true;
        }
    }
    
    return false;
}

size_t VariableSizeAllocator::getTotalSize() const {
    return totalSize;
}

size_t VariableSizeAllocator::getUsedSize() const {
    return usedSize;
}

size_t VariableSizeAllocator::getFreeSize() const {
    return totalSize - usedSize;
}

size_t VariableSizeAllocator::getLargestFreeBlock() const {
    if (freeBlocks.empty()) {
        return 0;
    }
    
    // The largest block is at the end of the multimap
    return freeBlocks.rbegin()->first;
}

double VariableSizeAllocator::getFragmentationRatio() const {
    if (totalSize == 0) {
        return 0.0;
    }
    
    // Calculate the number of free blocks
    size_t freeBlockCount = freeBlocks.size();
    
    // If there are no free blocks, there's no fragmentation
    if (freeBlockCount == 0) {
        return 0.0;
    }
    
    // Calculate the average free block size
    size_t freeSize = totalSize - usedSize;
    double avgFreeBlockSize = static_cast<double>(freeSize) / freeBlockCount;
    
    // Calculate the largest possible free block size
    size_t largestFreeBlock = getLargestFreeBlock();
    
    // Calculate the fragmentation ratio
    // 0.0 means no fragmentation (one large free block)
    // 1.0 means maximum fragmentation (many small free blocks)
    return 1.0 - (avgFreeBlockSize / largestFreeBlock);
}

void VariableSizeAllocator::addRegion(size_t size) {
    // Ensure the size is aligned
    size_t alignedSize = align_size(size, alignment);
    
    // Allocate memory for the region
    void* memory = aligned_alloc(alignment, alignedSize);
    
    if (memory == nullptr) {
        reportError(ErrorSeverity::Error, 
            "VariableSizeAllocator: Failed to allocate memory for region");
        return;
    }
    
    // Initialize the region
    MemoryRegion region(memory, alignedSize);
    
    // Initialize the first block in the region
    BlockHeader* block = static_cast<BlockHeader*>(memory);
    block->size = alignedSize;
    block->isFree = true;
    block->prev = nullptr;
    block->next = nullptr;
    
    // Set the first block in the region
    region.firstBlock = block;
    
    // Add the region to our list
    regions.push_back(region);
    
    // Add the block to the free list
    addToFreeList(block);
    
    // Update statistics
    totalSize += alignedSize;
}

void VariableSizeAllocator::addToFreeList(BlockHeader* block) {
    // Add the block to the free list
    freeBlocks.insert(std::make_pair(block->size, block));
}

void VariableSizeAllocator::removeFromFreeList(BlockHeader* block) {
    // Find the block in the free list
    auto range = freeBlocks.equal_range(block->size);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second == block) {
            freeBlocks.erase(it);
            break;
        }
    }
}

VariableSizeAllocator::BlockHeader* VariableSizeAllocator::findBestFit(size_t size) {
    // Find the first block that is large enough
    auto it = freeBlocks.lower_bound(size);
    
    // If no block is large enough, return nullptr
    if (it == freeBlocks.end()) {
        return nullptr;
    }
    
    // Return the block
    return it->second;
}

void VariableSizeAllocator::splitBlock(BlockHeader* block, size_t size) {
    // Calculate the size of the remaining block
    size_t remainingSize = block->size - size;
    
    // Only split if the remaining size is large enough
    if (remainingSize < minBlockSize) {
        return;
    }
    
    // Calculate the address of the new block
    char* blockAddr = reinterpret_cast<char*>(block);
    BlockHeader* newBlock = reinterpret_cast<BlockHeader*>(blockAddr + size);
    
    // Initialize the new block
    newBlock->size = remainingSize;
    newBlock->isFree = true;
    newBlock->prev = block;
    newBlock->next = block->next;
    
    // Update the next block's prev pointer
    if (block->next != nullptr) {
        block->next->prev = newBlock;
    }
    
    // Update the current block
    block->size = size;
    block->next = newBlock;
    
    // Add the new block to the free list
    addToFreeList(newBlock);
}

void VariableSizeAllocator::mergeWithNeighbors(BlockHeader* block) {
    // Try to merge with the next block
    if (block->next != nullptr && block->next->isFree) {
        // Remove the next block from the free list
        removeFromFreeList(block->next);
        
        // Merge the blocks
        block->size += block->next->size;
        block->next = block->next->next;
        
        // Update the next block's prev pointer
        if (block->next != nullptr) {
            block->next->prev = block;
        }
    }
    
    // Try to merge with the previous block
    if (block->prev != nullptr && block->prev->isFree) {
        // Remove the previous block from the free list
        removeFromFreeList(block->prev);
        
        // Merge the blocks
        block->prev->size += block->size;
        block->prev->next = block->next;
        
        // Update the next block's prev pointer
        if (block->next != nullptr) {
            block->next->prev = block->prev;
        }
        
        // The previous block is now the merged block
        block = block->prev;
    }
}

bool VariableSizeAllocator::isPointerInRegion(const void* ptr, const MemoryRegion& region) const {
    const char* charPtr = static_cast<const char*>(ptr);
    const char* regionStart = static_cast<const char*>(region.memory);
    const char* regionEnd = regionStart + region.size;
    
    return charPtr >= regionStart && charPtr < regionEnd;
}

} // namespace memory_pool