#include "memory_pool/gpu/cuda_utils.hpp"
#include "memory_pool/utils/error_handling.hpp"
#include <sstream>
#include <iostream>

// Include CUDA runtime for implementation
#include <cuda_runtime.h>

namespace memory_pool {

// CUDA error handling
std::string CudaException::getCudaErrorString(cudaError_t error) {
    return cudaGetErrorString(error);
}

void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line << ": " 
            << cudaGetErrorString(error) << " (" << error << ")";
        
        reportError(ErrorSeverity::Error, oss.str());
        throw CudaException(error);
    }
}

// CUDA device management
int getCurrentDevice() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

void setCurrentDevice(int deviceId) {
    CUDA_CHECK(cudaSetDevice(deviceId));
}

int getDeviceCount() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

size_t getDeviceMemory(int deviceId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    return prop.totalGlobalMem;
}

bool isDeviceAvailable(int deviceId) {
    int count = getDeviceCount();
    if (deviceId < 0 || deviceId >= count) {
        return false;
    }
    
    // Try to set the device
    cudaError_t error = cudaSetDevice(deviceId);
    return (error == cudaSuccess);
}

// CUDA stream management
cudaStream_t createStream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return stream;
}

void destroyStream(cudaStream_t stream) {
    if (stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

void synchronizeStream(cudaStream_t stream) {
    if (stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// CUDA memory operations
void* cudaAllocate(size_t size, AllocFlags flags) {
    if (size == 0) {
        return nullptr;
    }
    
    void* ptr = nullptr;
    
    if (has_flag(flags, AllocFlags::Pinned)) {
        // Allocate pinned memory
        CUDA_CHECK(cudaMallocHost(&ptr, size));
    } else if (has_flag(flags, AllocFlags::Managed)) {
        // Allocate managed memory
        CUDA_CHECK(cudaMallocManaged(&ptr, size));
    } else {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&ptr, size));
    }
    
    // Zero memory if requested
    if (has_flag(flags, AllocFlags::ZeroMemory)) {
        CUDA_CHECK(cudaMemset(ptr, 0, size));
    }
    
    return ptr;
}

void cudaDeallocate(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    
    // Determine if this is host or device memory
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
    
    if (error != cudaSuccess) {
        // Reset the error state
        cudaGetLastError();
        
        // Try to free as device memory
        error = cudaFree(ptr);
        if (error != cudaSuccess) {
            // Reset the error state again
            cudaGetLastError();
            
            // Try to free as host memory
            error = cudaFreeHost(ptr);
            if (error != cudaSuccess) {
                // Reset the error state one more time
                cudaGetLastError();
                
                reportError(ErrorSeverity::Warning, 
                    "Failed to deallocate CUDA memory: " + std::string(cudaGetErrorString(error)));
            }
        }
    } else {
        // Free based on memory type
#if CUDART_VERSION >= 10000
        if (attributes.type == cudaMemoryTypeHost || 
            attributes.type == cudaMemoryTypeManaged) {
            CUDA_CHECK(cudaFreeHost(ptr));
        } else {
            CUDA_CHECK(cudaFree(ptr));
        }
#else
        if (attributes.memoryType == cudaMemoryTypeHost || 
            attributes.isManaged) {
            CUDA_CHECK(cudaFreeHost(ptr));
        } else {
            CUDA_CHECK(cudaFree(ptr));
        }
#endif
    }
}

void cudaMemset(void* ptr, int value, size_t size) {
    if (ptr == nullptr || size == 0) {
        return;
    }
    
    CUDA_CHECK(cudaMemset(ptr, value, size));
}

void cudaMemcpy(void* dst, const void* src, size_t size, bool hostToDevice) {
    if (dst == nullptr || src == nullptr || size == 0) {
        return;
    }
    
    cudaMemcpyKind kind = hostToDevice ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    CUDA_CHECK(cudaMemcpy(dst, src, size, kind));
}

void cudaMemcpyAsync(void* dst, const void* src, size_t size, bool hostToDevice, cudaStream_t stream) {
    if (dst == nullptr || src == nullptr || size == 0) {
        return;
    }
    
    cudaMemcpyKind kind = hostToDevice ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, kind, stream));
}

} // namespace memory_pool