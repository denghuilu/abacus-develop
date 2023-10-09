#ifndef ATEN_CORE_TENSOR_ACCESSOR_H_
#define ATEN_CORE_TENSOR_ACCESSOR_H_

#include <cstddef> // Include the <cstddef> header file to define size_t
#include <stdint.h>
#include <base/macros/macros.h>
#include <base/utils/array_ref.h>

namespace container {

template <typename T, size_t N>
class TensorAccessorBase {
  public:
    TensorAccessorBase(T* data, const int64_t* sizes, const int64_t* strides)
        : data_(data), sizes_(sizes), strides_(strides) {}
    
    AT_HOST int_array_ref sizes() const {
        return int_array_ref(sizes_, N);
    }

    AT_HOST int_array_ref strides() const {
        return int_array_ref(strides_, N);
    }

    AT_HOST_DEVICE int64_t stride(int64_t idx) const {
        return strides_[idx];
    }

    AT_HOST_DEVICE int64_t size(int64_t idx) const {
        return sizes_[idx];
    }

    AT_HOST_DEVICE T* data() {
        return data_;
    }

    AT_HOST_DEVICE const T* data() const {
        return data_;
    }

  protected:
    T* data_;
    const int64_t* sizes_;
    const int64_t* strides_;
};

template <typename T, size_t N>
class TensorAccessor : public TensorAccessorBase<T, N> {
  public:
    AT_HOST_DEVICE TensorAccessor(T* data, const int64_t* sizes, const int64_t* strides)
        : TensorAccessorBase<T, N>(data, sizes, strides) {}

    AT_HOST_DEVICE TensorAccessor<T, N - 1> operator[](int64_t idx) {
        return TensorAccessor<T, N - 1>(this->data_ + idx * this->strides_[0], this->sizes_ + 1, this->strides_ + 1);
    }

    AT_HOST_DEVICE const TensorAccessor<T, N - 1> operator[](int64_t idx) const {
        return TensorAccessor<T, N - 1>(this->data_ + idx * this->strides_[0], this->sizes_ + 1, this->strides_ + 1);
    }
};

template <typename T>
class TensorAccessor<T, 1> : public TensorAccessorBase<T, 1> {
  public:
    AT_HOST_DEVICE TensorAccessor(T* data, const int64_t* sizes, const int64_t* strides)
        : TensorAccessorBase<T, 1>(data, sizes, strides) {}

    AT_HOST_DEVICE T& operator[](int64_t idx) {
        return this->data_[idx * this->strides_[0]];
    }

    AT_HOST_DEVICE const T& operator[](int64_t idx) const {
        return this->data_[idx * this->strides_[0]];
    }
};


} // namespace container

#endif // ATEN_CORE_TENSOR_ACCESSOR_H_