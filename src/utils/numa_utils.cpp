#include "memory_pool/utils/numa_utils.hpp"

#ifdef HAVE_NUMA
#include <numa.h>
#include <sched.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace memory_pool {
namespace numa_utils {

bool is_numa_available() {
#ifdef HAVE_NUMA
    return numa_available() >= 0;
#else
    return false;
#endif
}

int get_num_numa_nodes() {
#ifdef HAVE_NUMA
    if (!is_numa_available()) {
        return 0;
    }
    return numa_max_node() + 1;
#else
    return 0;
#endif
}

int get_current_numa_node() {
#ifdef HAVE_NUMA
    if (!is_numa_available()) {
        return -1;
    }
    return numa_node_of_cpu(sched_getcpu());
#else
    return -1;
#endif
}

void* allocate_on_node(size_t size, size_t alignment, int node) {
    if (size == 0) {
        return nullptr;
    }

#ifdef HAVE_NUMA
    if (false) {  // Temporarily disable NUMA
        // numa_alloc_onnode doesn't guarantee alignment beyond page size
        // So we allocate extra space and align manually
        size_t page_size = getpagesize();
        size_t header_size = sizeof(void*) + sizeof(size_t);
        size_t alloc_size = size + alignment - 1 + page_size + header_size;

        void* raw_ptr;
        if (node >= 0 && node < get_num_numa_nodes()) {
            raw_ptr = numa_alloc_onnode(alloc_size, node);
        } else {
            raw_ptr = numa_alloc_local(alloc_size);
        }

        if (raw_ptr == nullptr) {
            return nullptr;
        }

        // Align the pointer
        uintptr_t addr = reinterpret_cast<uintptr_t>(raw_ptr);
        uintptr_t data_start = addr + header_size;
        uintptr_t aligned_addr = (data_start + alignment - 1) & ~(alignment - 1);

        // Store the original pointer and size just before the aligned address
        void** raw_ptr_ptr = reinterpret_cast<void**>(aligned_addr - header_size);
        *raw_ptr_ptr = raw_ptr;
        size_t* size_ptr = reinterpret_cast<size_t*>(aligned_addr - header_size + sizeof(void*));
        *size_ptr = alloc_size;

        return reinterpret_cast<void*>(aligned_addr);
    }
#endif

    // Fallback to standard aligned allocation
    return aligned_alloc(alignment, size);
}

void deallocate(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

#ifdef HAVE_NUMA
    if (is_numa_available()) {
        // Retrieve the original pointer and size from the header
        size_t header_size = sizeof(void*) + sizeof(size_t);
        void** raw_ptr_ptr = reinterpret_cast<void**>(reinterpret_cast<uintptr_t>(ptr) - header_size);
        void* raw_ptr = *raw_ptr_ptr;
        size_t* size_ptr = reinterpret_cast<size_t*>(reinterpret_cast<uintptr_t>(ptr) - header_size + sizeof(void*));
        size_t alloc_size = *size_ptr;

        numa_free(raw_ptr, alloc_size);
        return;
    }
#endif

    // Fallback to standard free
    free(ptr);
}

}  // namespace numa_utils
}  // namespace memory_pool