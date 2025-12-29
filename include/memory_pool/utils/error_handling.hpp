#ifndef MEMORY_POOL_ERROR_HANDLING_HPP
#define MEMORY_POOL_ERROR_HANDLING_HPP

#include "memory_pool/common.hpp"
#include <string>
#include <functional>
#include <iostream>

namespace memory_pool {

// Error severity levels
enum class ErrorSeverity { Info, Warning, Error, Fatal };

// Error callback function type
using ErrorCallback = std::function<void(ErrorSeverity severity, const std::string& message)>;

// Error handler class
class ErrorHandler {
  public:
    // Get singleton instance
    static ErrorHandler& getInstance();

    // Set error callback
    void setErrorCallback(ErrorCallback callback);

    // Report an error
    void reportError(ErrorSeverity severity, const std::string& message);

    // Check a condition and report an error if it fails
    bool check(bool condition, ErrorSeverity severity, const std::string& message);

    // Throw an exception if enabled
    void throwIfEnabled(const std::string& message);

    // Enable/disable exception throwing
    void setThrowExceptions(bool enable);
    bool isThrowingExceptions() const;

  private:
    ErrorHandler();
    ~ErrorHandler() = default;

    // Prevent copying
    ErrorHandler(const ErrorHandler&)            = delete;
    ErrorHandler& operator=(const ErrorHandler&) = delete;

    // Error callback
    ErrorCallback errorCallback;

    // Exception throwing flag
    bool throwExceptions;

    // Default error callback
    static void defaultErrorCallback(ErrorSeverity severity, const std::string& message);
};

// Convenience functions
void reportError(ErrorSeverity severity, const std::string& message);
void reportWarning(const std::string& message);
void reportInfo(const std::string& message);

// Assertion macros
#ifdef NDEBUG
    #define MP_ASSERT(condition, message) ((void)0)
    #define MP_ASSERT_RETURN(condition, message, returnValue) \
        if (!(condition)) return returnValue
#else
    #define MP_ASSERT(condition, message)                                                                     \
        do {                                                                                                  \
            if (!(condition)) {                                                                               \
                memory_pool::ErrorHandler::getInstance().reportError(                                         \
                    memory_pool::ErrorSeverity::Error, std::string("Assertion failed: ") + (message) + " [" + \
                                                           __FILE__ + ":" + std::to_string(__LINE__) + "]");  \
                memory_pool::ErrorHandler::getInstance().throwIfEnabled(std::string("Assertion failed: ") +   \
                                                                        (message));                           \
            }                                                                                                 \
        } while (0)

    #define MP_ASSERT_RETURN(condition, message, returnValue)                                                 \
        do {                                                                                                  \
            if (!(condition)) {                                                                               \
                memory_pool::ErrorHandler::getInstance().reportError(                                         \
                    memory_pool::ErrorSeverity::Error, std::string("Assertion failed: ") + (message) + " [" + \
                                                           __FILE__ + ":" + std::to_string(__LINE__) + "]");  \
                return (returnValue);                                                                         \
            }                                                                                                 \
        } while (0)
#endif

}  // namespace memory_pool

#endif  // MEMORY_POOL_ERROR_HANDLING_HPP