#include "memory_pool/gpu/cuda_allocator.hpp"
#include "memory_pool/gpu/cuda_utils.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <algorithm>
#include <limits>

namespace memory_pool {

// CudaAllocatorBase implementation
CudaAllocatorBase::CudaAllocatorBase(int deviceId) : deviceId(deviceId), stream(nullptr) {}

CudaAllocatorBase::~CudaAllocatorBase() {
    // Stream is managed by the memory pool
}

void CudaAllocatorBase::setDevice(int deviceId) { this->deviceId = deviceId; }

int CudaAllocatorBase::getDevice() const { return deviceId; }

void CudaAllocatorBase::setStream(cudaStream_t stream) { this->stream = stream; }

cudaStream_t CudaAllocatorBase::getStream() const { return stream; }

void CudaAllocatorBase::ensureCorrectDevice() const {
    int currentDevice = getCurrentDevice();
    if (currentDevice != deviceId) {
        setCurrentDevice(deviceId);
    }
}

// CudaFixedSizeAllocator implementation
CudaFixedSizeAllocator::CudaFixedSizeAllocator(size_t blockSize, size_t initialBlocks, int deviceId,
                                               AllocFlags defaultFlags)
    : CudaAllocatorBase(deviceId),
      blockSize(blockSize),
      defaultFlags(defaultFlags),
      totalBlocks(initialBlocks),
      usedBlocks(0) {
    if (blockSize == 0) {
        throw InvalidOperationException("Block size cannot be zero");
    }

    // Allocate initial chunk
    allocateChunk(initialBlocks);
}

CudaFixedSizeAllocator::~CudaFixedSizeAllocator() {
    // Free all allocated chunks
    for (auto& chunk : chunks) {
        cudaDeallocate(chunk.deviceMemory);
    }
    chunks.clear();
    freeBlocks.clear();
    allocatedBlocks.clear();
}

void* CudaFixedSizeAllocator::allocate(size_t size, AllocFlags flags) {
    if (size == 0) {
        return nullptr;
    }

    // Check if requested size fits in our block size
    if (size > blockSize) {
        throw InvalidOperationException("Requested size exceeds block size");
    }

    ensureCorrectDevice();

    // Combine flags
    AllocFlags combinedFlags = defaultFlags | flags;

    // Find a free block
    if (freeBlocks.empty()) {
        // Allocate a new chunk
        allocateChunk(std::max(size_t(16), totalBlocks / 4));  // Grow by at least 16 blocks or 25%
    }

    if (freeBlocks.empty()) {
        throw OutOfMemoryException("Failed to allocate GPU memory block");
    }

    // Get the first free block
    Block* block = freeBlocks.back();
    freeBlocks.pop_back();
    block->isFree = false;

    // Track allocation
    allocatedBlocks[block->devicePtr] = block;
    usedBlocks++;

    // Zero memory if requested
    if (has_flag(combinedFlags, AllocFlags::ZeroMemory)) {
        cudaMemsetValue(block->devicePtr, 0, blockSize);
    }

    return block->devicePtr;
}

void CudaFixedSizeAllocator::deallocate(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    ensureCorrectDevice();

    auto it = allocatedBlocks.find(ptr);
    if (it == allocatedBlocks.end()) {
        throw InvalidPointerException("Pointer not allocated by this allocator");
    }

    Block* block  = it->second;
    block->isFree = true;
    allocatedBlocks.erase(it);
    freeBlocks.push_back(block);
    usedBlocks--;
}

void CudaFixedSizeAllocator::reset() {
    // Mark all blocks as free
    for (auto& chunk : chunks) {
        for (auto& block : chunk.blocks) {
            if (!block.isFree) {
                block.isFree = true;
                freeBlocks.push_back(&block);
            }
        }
    }
    allocatedBlocks.clear();
    usedBlocks = 0;
}

size_t CudaFixedSizeAllocator::getBlockSize(void* ptr) const {
    if (ptr == nullptr) {
        return 0;
    }

    auto it = allocatedBlocks.find(ptr);
    if (it == allocatedBlocks.end()) {
        return 0;
    }

    return blockSize;
}

bool CudaFixedSizeAllocator::owns(void* ptr) const {
    if (ptr == nullptr) {
        return false;
    }

    // Check if pointer is in any of our chunks
    for (const auto& chunk : chunks) {
        if (isPointerInChunk(ptr, chunk)) {
            return true;
        }
    }

    return false;
}

size_t CudaFixedSizeAllocator::getBlockSize() const { return blockSize; }

size_t CudaFixedSizeAllocator::getCapacity() const { return totalBlocks; }

size_t CudaFixedSizeAllocator::getFreeBlocks() const { return freeBlocks.size(); }

size_t CudaFixedSizeAllocator::getUsedBlocks() const { return usedBlocks; }

void CudaFixedSizeAllocator::allocateChunk(size_t blockCount) {
    size_t chunkSize = blockCount * blockSize;

    // Allocate device memory
    void* deviceMemory = cudaAllocate(chunkSize, defaultFlags);
    if (deviceMemory == nullptr) {
        throw OutOfMemoryException("Failed to allocate GPU memory chunk");
    }

    // Create chunk structure
    chunks.emplace_back(deviceMemory, blockCount);

    Chunk& chunk = chunks.back();

    // Initialize blocks
    for (size_t i = 0; i < blockCount; ++i) {
        void* blockPtr = static_cast<char*>(deviceMemory) + (i * blockSize);
        chunk.blocks.emplace_back(blockPtr);
        freeBlocks.push_back(&chunk.blocks.back());
    }

    totalBlocks += blockCount;
}

bool CudaFixedSizeAllocator::isPointerInChunk(const void* ptr, const Chunk& chunk) const {
    const char* ptrChar    = static_cast<const char*>(ptr);
    const char* chunkStart = static_cast<const char*>(chunk.deviceMemory);
    const char* chunkEnd   = chunkStart + (chunk.blockCount * blockSize);

    return (ptrChar >= chunkStart && ptrChar < chunkEnd);
}

// CudaVariableSizeAllocator implementation
CudaVariableSizeAllocator::CudaVariableSizeAllocator(size_t initialSize, int deviceId, AllocFlags defaultFlags)
    : CudaAllocatorBase(deviceId), defaultFlags(defaultFlags), totalSize(0), usedSize(0) {
    if (initialSize > 0) {
        addRegion(initialSize);
    }
}

CudaVariableSizeAllocator::~CudaVariableSizeAllocator() {
    // Free all allocated regions
    for (auto& region : regions) {
        cudaDeallocate(region.deviceMemory);
    }
    regions.clear();
    freeBlocks.clear();
    allocatedBlocks.clear();
}

void* CudaVariableSizeAllocator::allocate(size_t size, AllocFlags flags) {
    if (size == 0) {
        return nullptr;
    }

    ensureCorrectDevice();

    // Combine flags
    AllocFlags combinedFlags = defaultFlags | flags;

    // Align size
    size = align_size(size);

    // Find best fit block
    Block* block = findBestFit(size);
    if (block == nullptr) {
        // Allocate new region
        size_t regionSize = std::max(size * 2, totalSize / 2);  // Double the requested size or half of total size
        addRegion(regionSize);

        block = findBestFit(size);
        if (block == nullptr) {
            throw OutOfMemoryException("Failed to allocate GPU memory");
        }
    }

    // Split block if necessary
    splitBlock(block, size);

    // Mark as allocated
    block->isFree                     = false;
    allocatedBlocks[block->devicePtr] = block;
    usedSize += block->size;

    // Zero memory if requested
    if (has_flag(combinedFlags, AllocFlags::ZeroMemory)) {
        cudaMemsetValue(block->devicePtr, 0, block->size);
    }

    return block->devicePtr;
}

void CudaVariableSizeAllocator::deallocate(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    ensureCorrectDevice();

    auto it = allocatedBlocks.find(ptr);
    if (it == allocatedBlocks.end()) {
        throw InvalidPointerException("Pointer not allocated by this allocator");
    }

    Block* block = it->second;
    usedSize -= block->size;
    block->isFree = true;
    allocatedBlocks.erase(it);

    // Add to free list
    addToFreeList(block);

    // Try to merge adjacent free blocks
    mergeAdjacentBlocks();
}

void CudaVariableSizeAllocator::reset() {
    // Mark all blocks as free and rebuild free list
    freeBlocks.clear();
    for (auto& region : regions) {
        for (auto& block : region.blocks) {
            if (!block.isFree) {
                block.isFree = true;
                usedSize -= block.size;
            }
            if (block.isFree) {
                addToFreeList(&block);
            }
        }
    }
    allocatedBlocks.clear();
    usedSize = 0;
}

size_t CudaVariableSizeAllocator::getBlockSize(void* ptr) const {
    if (ptr == nullptr) {
        return 0;
    }

    auto it = allocatedBlocks.find(ptr);
    if (it == allocatedBlocks.end()) {
        return 0;
    }

    return it->second->size;
}

bool CudaVariableSizeAllocator::owns(void* ptr) const {
    if (ptr == nullptr) {
        return false;
    }

    // Check if pointer is in any of our regions
    for (const auto& region : regions) {
        if (isPointerInRegion(ptr, region)) {
            return true;
        }
    }

    return false;
}

size_t CudaVariableSizeAllocator::getTotalSize() const { return totalSize; }

size_t CudaVariableSizeAllocator::getUsedSize() const { return usedSize; }

size_t CudaVariableSizeAllocator::getFreeSize() const { return totalSize - usedSize; }

void CudaVariableSizeAllocator::addRegion(size_t size) {
    // Allocate device memory
    void* deviceMemory = cudaAllocate(size, defaultFlags);
    if (deviceMemory == nullptr) {
        throw OutOfMemoryException("Failed to allocate GPU memory region");
    }

    // Create region structure
    regions.emplace_back(deviceMemory, size);

    MemoryRegion& region = regions.back();

    // Create initial free block
    region.blocks.emplace_back(deviceMemory, size);
    addToFreeList(&region.blocks.back());

    totalSize += size;
}

void CudaVariableSizeAllocator::addToFreeList(Block* block) { freeBlocks.emplace(block->size, block); }

void CudaVariableSizeAllocator::removeFromFreeList(Block* block) {
    auto it = freeBlocks.find(block->size);
    if (it != freeBlocks.end()) {
        // Find the specific block (there might be multiple with same size)
        auto range = freeBlocks.equal_range(block->size);
        for (auto i = range.first; i != range.second; ++i) {
            if (i->second == block) {
                freeBlocks.erase(i);
                break;
            }
        }
    }
}

CudaVariableSizeAllocator::Block* CudaVariableSizeAllocator::findBestFit(size_t size) {
    // Find the smallest block that can fit the requested size
    Block* bestFit     = nullptr;
    size_t bestFitSize = std::numeric_limits<size_t>::max();

    auto it = freeBlocks.lower_bound(size);
    if (it != freeBlocks.end()) {
        bestFit     = it->second;
        bestFitSize = it->first;
    }

    return bestFit;
}

void CudaVariableSizeAllocator::splitBlock(Block* block, size_t size) {
    if (block->size <= size + align_size(sizeof(Block))) {
        // Not enough space for splitting
        return;
    }

    // Remove from free list
    removeFromFreeList(block);

    // Calculate remaining size
    size_t remainingSize = block->size - size;

    // Create new block for remaining space
    void* remainingPtr = static_cast<char*>(block->devicePtr) + size;

    // Find which region this block belongs to
    for (auto& region : regions) {
        if (isPointerInRegion(block->devicePtr, region)) {
            region.blocks.emplace_back(remainingPtr, remainingSize);
            Block* newBlock = &region.blocks.back();
            addToFreeList(newBlock);
            break;
        }
    }

    // Update original block size
    block->size = size;
}

void CudaVariableSizeAllocator::mergeAdjacentBlocks() {
    // This is a simplified implementation - in a real-world scenario,
    // you'd want more sophisticated merging logic
    // For now, we'll skip complex merging to keep the implementation simpler
}

bool CudaVariableSizeAllocator::isPointerInRegion(const void* ptr, const MemoryRegion& region) const {
    const char* ptrChar     = static_cast<const char*>(ptr);
    const char* regionStart = static_cast<const char*>(region.deviceMemory);
    const char* regionEnd   = regionStart + region.size;

    return (ptrChar >= regionStart && ptrChar < regionEnd);
}

}  // namespace memory_pool