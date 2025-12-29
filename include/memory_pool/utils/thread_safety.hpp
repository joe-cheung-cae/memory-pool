#ifndef MEMORY_POOL_THREAD_SAFETY_HPP
#define MEMORY_POOL_THREAD_SAFETY_HPP

#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <memory>

namespace memory_pool {

// Thread-safe counter
class AtomicCounter {
  public:
    AtomicCounter(size_t initial = 0) : value(initial) {}

    size_t increment() { return ++value; }

    size_t decrement() { return --value; }

    size_t add(size_t amount) { return value.fetch_add(amount) + amount; }

    size_t subtract(size_t amount) { return value.fetch_sub(amount) - amount; }

    size_t get() const { return value.load(); }

    void set(size_t newValue) { value.store(newValue); }

  private:
    std::atomic<size_t> value;
};

// Read-write lock for operations that need different levels of access
class ReadWriteLock {
  public:
    ReadWriteLock() = default;

    // Acquire read lock (shared access)
    void lockRead() { mutex.lock_shared(); }

    // Try to acquire read lock
    bool tryLockRead() { return mutex.try_lock_shared(); }

    // Release read lock
    void unlockRead() { mutex.unlock_shared(); }

    // Acquire write lock (exclusive access)
    void lockWrite() { mutex.lock(); }

    // Try to acquire write lock
    bool tryLockWrite() { return mutex.try_lock(); }

    // Release write lock
    void unlockWrite() { mutex.unlock(); }

  private:
    std::shared_mutex mutex;
};

// RAII wrapper for read lock
class ReadLockGuard {
  public:
    explicit ReadLockGuard(ReadWriteLock& lock) : lock(lock) { lock.lockRead(); }

    ~ReadLockGuard() { lock.unlockRead(); }

  private:
    ReadWriteLock& lock;

    // Prevent copying
    ReadLockGuard(const ReadLockGuard&)            = delete;
    ReadLockGuard& operator=(const ReadLockGuard&) = delete;
};

// RAII wrapper for write lock
class WriteLockGuard {
  public:
    explicit WriteLockGuard(ReadWriteLock& lock) : lock(lock) { lock.lockWrite(); }

    ~WriteLockGuard() { lock.unlockWrite(); }

  private:
    ReadWriteLock& lock;

    // Prevent copying
    WriteLockGuard(const WriteLockGuard&)            = delete;
    WriteLockGuard& operator=(const WriteLockGuard&) = delete;
};

// Thread-local storage for per-thread caching
template <typename T>
class ThreadLocalStorage {
  public:
    ThreadLocalStorage() = default;

    T& get() { return storage; }

    const T& get() const { return storage; }

  private:
    static thread_local T storage;
};

template <typename T>
thread_local T ThreadLocalStorage<T>::storage;

// Lock-free queue for single-producer, single-consumer scenarios
template <typename T>
class LockFreeQueue {
  public:
    LockFreeQueue() : head(new Node()), tail(head.load()) {}

    ~LockFreeQueue() {
        while (T* item = dequeue()) {
            delete item;
        }
        Node* node = head.load();
        while (node) {
            Node* next = node->next.load();
            delete node;
            node = next;
        }
    }

    void enqueue(T* item) {
        Node* newNode = new Node(item);
        Node* oldTail = tail.exchange(newNode);
        oldTail->next.store(newNode);
    }

    T* dequeue() {
        Node* oldHead = head.load();
        Node* next    = oldHead->next.load();
        if (next == nullptr) {
            return nullptr;  // Queue is empty
        }
        T* item    = next->data;
        next->data = nullptr;  // Mark as dequeued
        head.store(next);
        delete oldHead;
        return item;
    }

    bool isEmpty() const {
        Node* currentHead = head.load();
        return currentHead->next.load() == nullptr;
    }

  private:
    struct Node {
        T*                 data;
        std::atomic<Node*> next;

        Node(T* d = nullptr) : data(d), next(nullptr) {}
    };

    std::atomic<Node*> head;
    std::atomic<Node*> tail;
};

// Lock-free stack using elimination backoff for better performance
template <typename T>
class LockFreeStack {
  public:
    LockFreeStack() : top(nullptr), size(0) {}

    ~LockFreeStack() {
        while (T* item = pop()) {
            delete item;
        }
    }

    void push(T* item) {
        Node* newNode = new Node(item);
        newNode->next = top.load();
        while (!top.compare_exchange_weak(newNode->next, newNode)) {
            // Retry
        }
        size.fetch_add(1);
    }

    T* pop() {
        Node* oldTop = top.load();
        while (oldTop && !top.compare_exchange_weak(oldTop, oldTop->next.load())) {
            // Retry
        }
        if (oldTop) {
            size.fetch_sub(1);
            T* item      = oldTop->data;
            oldTop->data = nullptr;
            delete oldTop;
            return item;
        }
        return nullptr;
    }

    bool isEmpty() const { return top.load() == nullptr; }

    size_t getSize() const { return size.load(); }

  private:
    struct Node {
        T*                 data;
        std::atomic<Node*> next;

        Node(T* d) : data(d), next(nullptr) {}
    };

    std::atomic<Node*>  top;
    std::atomic<size_t> size;
};

}  // namespace memory_pool

#endif  // MEMORY_POOL_THREAD_SAFETY_HPP