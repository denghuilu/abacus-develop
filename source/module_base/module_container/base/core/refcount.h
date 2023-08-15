#ifndef BASE_CORE_REFCOUNT_H_
#define BASE_CORE_REFCOUNT_H_

#include <atomic>
#include <memory>
#include <iostream>

namespace container {
namespace base {

/**
 * @brief The base class for reference-counted objects.
 */
class counted_base {
 public:
    /**
     * @brief Default constructor. Initializes the reference count to one.
     */
    counted_base() : ref_(1) {}

    /**
     * @brief Increases the reference count by one.
     */
    void ref() const {
      ref_.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Decreases the reference count by one.
     * @return True if the object is deleted, otherwise false.
     */
    bool unref() const {
      int_fast32_t oldCount = ref_.fetch_sub(1, std::memory_order_release) - 1;
      if (oldCount == 1) {
        std::atomic_thread_fence(std::memory_order_acquire);
        delete this;
        return true;
      }
      return false;
    }

    /**
     * @brief Gets the current reference count.
     * @return The current reference count.
     */
    int_fast32_t ref_count() const {
      return ref_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Checks if the reference count is one.
     * @return True if the reference count is one, otherwise false.
     */
    bool ref_count_is_one() const {
      int_fast32_t count = ref_.load(std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_acquire);
      return count == 1;
    }

 protected:
    /**
     * @brief Virtual destructor.
     * @details The destructor is protected to prevent the explicit initialization of the base class.
     */
    virtual ~counted_base() {}

    /**
     * @brief Increases the reference count by one if the object is not being destructed.
     * @return True if the reference is successfully acquired, otherwise false.
     */
    bool try_ref() const {
      int_fast32_t count = ref_.load(std::memory_order_acquire);
      while (count > 0) {
        if (ref_.compare_exchange_weak(count, count + 1, std::memory_order_acquire)) {
          return true;
        }
      }
      return false;
    }

    /**
     * @brief Notifies that the instance is deleted.
     */
    virtual void notify_deleted() const {}

 private:
    mutable std::atomic_int_fast32_t ref_;
};

/**
 * @brief A deleter functor for creating std::unique_ptr that unrefs objects.
 */
struct ref_count_deleter {
    /**
     * @brief Calls unref on the object.
     * @param o Pointer to the object.
     */
    void operator()(const counted_base* o) const {
      o->unref();
    }
};

/**
 * @brief A smart pointer that holds a reference-counted object and releases it on destruction.
 * @tparam T Type of the object.
 */
template <typename T>
class ref_count_ptr;

/**
 * @brief Adds a new reference to a counted_base pointer.
 * @tparam T Type of the object.
 * @param ptr Pointer to the object.
 * @return A smart pointer holding the reference to the object.
 */
template <typename T>
std::unique_ptr<T, ref_count_deleter> get_new_ref(T* ptr) {
    static_assert(std::is_base_of<counted_base, T>::value,
                  "T must be derived from counted_base");

    if (ptr == nullptr) {
      return std::unique_ptr<T, ref_count_deleter>();
    }
    ptr->ref();
    return std::unique_ptr<T, ref_count_deleter>(ptr);
}

/**
 * @brief A smart pointer that unrefs the owned object on destruction.
 * @tparam T Type of the object.
 */
template <typename T>
class ref_count_ptr : public std::unique_ptr<T, ref_count_deleter> {
 public:
    using std::unique_ptr<T, ref_count_deleter>::unique_ptr;

    /**
     * @brief Adds a new reference to the owned object.
     * @return A smart pointer holding the reference to the object.
     */
    std::unique_ptr<T, ref_count_deleter> get_new_ref() const {
        if (this->get() == nullptr) {
            return std::unique_ptr<T, ref_count_deleter>();
        }
        this->get()->ref();
        return std::unique_ptr<T, ref_count_deleter>(this->get());
    }
};

} // namespace base
} // namespace container

#endif // BASE_CORE_REFCOUNT_H_