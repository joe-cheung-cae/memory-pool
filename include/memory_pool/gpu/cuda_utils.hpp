#ifndef MEMORY_POOL_CUDA_UTILS_HPP
#define MEMORY_POOL_CUDA_UTILS_HPP

#include "memory_pool/common.hpp"
#include <string>

// Forward declarations for CUDA types to avoid requiring CUDA headers
// in client code that doesn't need them
#include <cuda_runtime.h>

// Valgrind integration for memory leak detection
#ifdef HAVE_VALGRIND
#include <valgrind/valgrind.h>
#include <valgrind/memcheck.h>
#endif

namespace memory_pool {

/**
 * @brief Exception class for CUDA-related errors.
 *
 * This class extends MemoryPoolException to provide specific handling
 * for CUDA API errors, including error code information.
 */
class CudaException : public MemoryPoolException {
  public:
    /**
     * @brief Constructs a CUDA exception with a custom message.
     * @param message The error message.
     */
    explicit CudaException(const std::string& message) : MemoryPoolException(message) {}

    /**
     * @brief Constructs a CUDA exception from a CUDA error code.
     * @param error The CUDA error code.
     */
    explicit CudaException(cudaError_t error) : MemoryPoolException(getCudaErrorString(error)), errorCode(error) {}

    /**
     * @brief Gets the CUDA error code associated with this exception.
     * @return The CUDA error code.
     */
    cudaError_t getErrorCode() const { return errorCode; }

  private:
    cudaError_t errorCode;

    /**
     * @brief Converts a CUDA error code to a string.
     * @param error The CUDA error code.
     * @return The error string.
     */
    static std::string getCudaErrorString(cudaError_t error);
};

/**
 * @brief Checks a CUDA error code and throws an exception if an error occurred.
 * @param error The CUDA error code to check.
 * @param file The source file name where the error occurred.
 * @param line The line number where the error occurred.
 */
void checkCudaError(cudaError_t error, const char* file, int line);

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                        \
    do {                                                        \
        cudaError_t error = call;                               \
        memory_pool::checkCudaError(error, __FILE__, __LINE__); \
    } while (0)

/**
 * @brief Gets the current CUDA device.
 * @return The current device ID.
 */
int getCurrentDevice();

/**
 * @brief Sets the current CUDA device.
 * @param deviceId The device ID to set as current.
 */
void setCurrentDevice(int deviceId);

/**
 * @brief Gets the number of available CUDA devices.
 * @return The number of CUDA devices.
 */
int getDeviceCount();

/**
 * @brief Gets the total memory of a CUDA device.
 * @param deviceId The device ID.
 * @return The total memory in bytes.
 */
size_t getDeviceMemory(int deviceId);

/**
 * @brief Checks if a CUDA device is available.
 * @param deviceId The device ID to check.
 * @return True if the device is available, false otherwise.
 */
bool isDeviceAvailable(int deviceId);

/**
 * @brief Creates a new CUDA stream.
 * @return The created CUDA stream.
 */
cudaStream_t createStream();

/**
 * @brief Destroys a CUDA stream.
 * @param stream The stream to destroy.
 */
void destroyStream(cudaStream_t stream);

/**
 * @brief Synchronizes a CUDA stream.
 * @param stream The stream to synchronize. If nullptr, synchronizes the device.
 */
void synchronizeStream(cudaStream_t stream);

/**
 * @brief Allocates CUDA memory with specified flags.
 * @param size The size in bytes to allocate.
 * @param flags Allocation flags (pinned, managed, zero memory).
 * @return Pointer to the allocated memory, or nullptr on failure.
 */
void* cudaAllocate(size_t size, AllocFlags flags = AllocFlags::None);

/**
 * @brief Deallocates CUDA memory.
 * @param ptr Pointer to the memory to deallocate.
 */
void cudaDeallocate(void* ptr);

/**
 * @brief Sets a value to a range of CUDA memory.
 * @param ptr Pointer to the memory.
 * @param value The value to set.
 * @param size The size in bytes to set.
 */
void cudaMemsetValue(void* ptr, int value, size_t size);

/**
 * @brief Copies memory between host and device synchronously.
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param size Size in bytes to copy.
 * @param hostToDevice True if copying from host to device, false otherwise.
 */
void cudaMemcpy(void* dst, const void* src, size_t size, bool hostToDevice);

/**
 * @brief Copies memory between host and device asynchronously.
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param size Size in bytes to copy.
 * @param hostToDevice True if copying from host to device, false otherwise.
 * @param stream The CUDA stream for asynchronous operation.
 */
void cudaMemcpyAsync(void* dst, const void* src, size_t size, bool hostToDevice, cudaStream_t stream);

}  // namespace memory_pool

#endif  // MEMORY_POOL_CUDA_UTILS_HPP