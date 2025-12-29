#include "memory_pool/utils/error_handling.hpp"
#include <iostream>

namespace memory_pool {

// Singleton instance
ErrorHandler& ErrorHandler::getInstance() {
    static ErrorHandler instance;
    return instance;
}

ErrorHandler::ErrorHandler() : errorCallback(defaultErrorCallback), throwExceptions(true) {}

void ErrorHandler::setErrorCallback(ErrorCallback callback) {
    if (callback) {
        errorCallback = callback;
    } else {
        errorCallback = defaultErrorCallback;
    }
}

void ErrorHandler::reportError(ErrorSeverity severity, const std::string& message) {
    if (errorCallback) {
        errorCallback(severity, message);
    }

    if (severity == ErrorSeverity::Fatal) {
        throwIfEnabled("Fatal error: " + message);
    }
}

bool ErrorHandler::check(bool condition, ErrorSeverity severity, const std::string& message) {
    if (!condition) {
        reportError(severity, message);
    }

    return condition;
}

void ErrorHandler::throwIfEnabled(const std::string& message) {
    if (throwExceptions) {
        throw MemoryPoolException(message);
    }
}

void ErrorHandler::setThrowExceptions(bool enable) { throwExceptions = enable; }

bool ErrorHandler::isThrowingExceptions() const { return throwExceptions; }

void ErrorHandler::defaultErrorCallback(ErrorSeverity severity, const std::string& message) {
    switch (severity) {
        case ErrorSeverity::Info:
            std::cout << "Info: " << message << std::endl;
            break;
        case ErrorSeverity::Warning:
            std::cerr << "Warning: " << message << std::endl;
            break;
        case ErrorSeverity::Error:
            std::cerr << "Error: " << message << std::endl;
            break;
        case ErrorSeverity::Fatal:
            std::cerr << "Fatal Error: " << message << std::endl;
            break;
    }
}

// Convenience functions
void reportError(ErrorSeverity severity, const std::string& message) {
    ErrorHandler::getInstance().reportError(severity, message);
}

void reportWarning(const std::string& message) { reportError(ErrorSeverity::Warning, message); }

void reportInfo(const std::string& message) { reportError(ErrorSeverity::Info, message); }

}  // namespace memory_pool