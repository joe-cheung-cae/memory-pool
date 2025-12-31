#include "memory_pool/utils/platform_utils.hpp"
#include "memory_pool/common.hpp"
#include <algorithm>

#ifdef MEMORY_POOL_PLATFORM_WINDOWS
    #include <windows.h>
    #include <shlobj.h>
#else
    #include <cstdlib>
#endif

namespace memory_pool {
namespace utils {

std::string normalize_path(const std::string& path) {
#ifdef MEMORY_POOL_PLATFORM_WINDOWS
    std::string result = path;
    std::replace(result.begin(), result.end(), '/', '\\');
    return result;
#else
    std::string result = path;
    std::replace(result.begin(), result.end(), '\\', '/');
    return result;
#endif
}

std::string get_temp_directory() {
#ifdef MEMORY_POOL_PLATFORM_WINDOWS
    char temp_path[MAX_PATH];
    if (GetTempPathA(MAX_PATH, temp_path) != 0) {
        return normalize_path(std::string(temp_path));
    }
    return "C:\\Temp\\";
#else
    const char* temp_dir = std::getenv("TMPDIR");
    if (!temp_dir) {
        temp_dir = std::getenv("TEMP");
    }
    if (!temp_dir) {
        temp_dir = std::getenv("TMP");
    }
    if (!temp_dir) {
        temp_dir = "/tmp";
    }
    return std::string(temp_dir) + "/";
#endif
}

bool cuda_supported() {
#if HAVE_CUDA
    return true;
#else
    return false;
#endif
}

bool numa_supported() {
#if HAVE_NUMA
    return true;
#else
    return false;
#endif
}

bool pmem_supported() {
#if HAVE_PMEM
    return true;
#else
    return false;
#endif
}

bool valgrind_supported() {
#if HAVE_VALGRIND
    return true;
#else
    return false;
#endif
}

}  // namespace utils
}  // namespace memory_pool