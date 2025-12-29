#ifndef MEMORY_POOL_THREAD_SAFETY_HPP
#define MEMORY_POOL_THREAD_SAFETY_HPP

#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>

namespace memory_pool {

// Thread-safe counter
class AtomicCounter {
public:
    AtomicCounter(size_t initial = 0) : value(initial) {}
    
    size_t increment() {
        return ++value;
    }
    
    size_t decrement() {
        return --value;
    }
    
    size_t add(size_t amount) {
        return value.fetch_add(amount) + amount;
    }
    
    size_t subtract(size_t amount) {
        return value.fetch_sub(amount) - amount;
    }
    
    size_t get() const {
        return value.load();
    }
    
    void set(size_t newValue) {
        value.store(newValue);
    }
    
private:
    std::atomic<size_t> value;
};

// Read-write lock for operations that need different levels of access
class ReadWriteLock {
public:
    ReadWriteLock() = default;
    
    // Acquire read lock (shared access)
    void lockRead() {
        mutex.lock_shared();
    }
    
    // Try to acquire read lock
    bool tryLockRead() {
        return mutex.try_lock_shared();
    }
    
    // Release read lock
    void unlockRead() {
        mutex.unlock_shared();
    }
    
    // Acquire write lock (exclusive access)
    void lockWrite() {
        mutex.lock();
    }
    
    // Try to acquire write lock
    bool tryLockWrite() {
        return mutex.try_lock();
    }
    
    // Release write lock
    void unlockWrite() {
        mutex.unlock();
    }
    
private:
    std::shared_mutex mutex;
};

// RAII wrapper for read lock
class ReadLockGuard {
public:
    explicit ReadLockGuard(ReadWriteLock& lock) : lock(lock) {
        lock.lockRead();
    }
    
    ~ReadLockGuard() {
        lock.unlockRead();
    }
    
private:
    ReadWriteLock& lock;
    
    // Prevent copying
    ReadLockGuard(const ReadLockGuard&) = delete;
    ReadLockGuard& operator=(const ReadLockGuard&) = delete;
};

// RAII wrapper for write lock
class WriteLockGuard {
public:
    explicit WriteLockGuard(ReadWriteLock& lock) : lock(lock) {
        lock.lockWrite();
    }
    
    ~WriteLockGuard() {
        lock.unlockWrite();
    }
    
private:
    ReadWriteLock& lock;
    
    // Prevent copying
    WriteLockGuard(const WriteLockGuard&) = delete;
    WriteLockGuard& operator=(const WriteLockGuard&) = delete;
};

// Thread-local storage for per-thread caching
template<typename T>
class ThreadLocalStorage {
public:
    ThreadLocalStorage() = default;
    
    T& get() {
        return storage;
    }
    
    const T& get() const {
        return storage;
    }
    
private:
    static thread_local T storage;
};

template<typename T>
thread_local T ThreadLocalStorage<T>::storage;

} // namespace memory_pool

#endif // MEMORY_POOL_THREAD_SAFETY_HPP