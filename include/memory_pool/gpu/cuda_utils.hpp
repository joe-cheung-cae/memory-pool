#ifndef MEMORY_POOL_CUDA_UTILS_HPP
#define MEMORY_POOL_CUDA_UTILS_HPP

#include "../common.hpp"
#include <string>

// Forward declarations for CUDA types to avoid requiring CUDA headers
// in client code that doesn't need them
#ifndef __CUDACC__
typedef struct CUstream_st* cudaStream_t;
typedef int cudaError_t;
#else
#include <cuda_runtime.h>
#endif

namespace memory_pool {

// CUDA error handling
class CudaException : public MemoryPoolException {
public:
    explicit CudaException(const std::string& message)
        : MemoryPoolException(message) {}
    
    explicit CudaException(cudaError_t error)
        : MemoryPoolException(getCudaErrorString(error)),
          errorCode(error) {}
    
    cudaError_t getErrorCode() const { return errorCode; }
    
private:
    cudaError_t errorCode;
    
    static std::string getCudaErrorString(cudaError_t error);
};

// Check CUDA error and throw exception if needed
void checkCudaError(cudaError_t error, const char* file, int line);

// Macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    memory_pool::checkCudaError(error, __FILE__, __LINE__); \
} while(0)

// CUDA device management
int getCurrentDevice();
void setCurrentDevice(int deviceId);
int getDeviceCount();
size_t getDeviceMemory(int deviceId);
bool isDeviceAvailable(int deviceId);

// CUDA stream management
cudaStream_t createStream();
void destroyStream(cudaStream_t stream);
void synchronizeStream(cudaStream_t stream);

// CUDA memory operations
void* cudaAllocate(size_t size, AllocFlags flags = AllocFlags::None);
void cudaDeallocate(void* ptr);
void cudaMemset(void* ptr, int value, size_t size);
void cudaMemcpy(void* dst, const void* src, size_t size, bool hostToDevice);
void cudaMemcpyAsync(void* dst, const void* src, size_t size, bool hostToDevice, cudaStream_t stream);

} // namespace memory_pool

#endif // MEMORY_POOL_CUDA_UTILS_HPP