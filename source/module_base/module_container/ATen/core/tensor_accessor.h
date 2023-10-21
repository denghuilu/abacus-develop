#ifndef ATEN_CORE_TENSOR_ACCESSOR_H_
#define ATEN_CORE_TENSOR_ACCESSOR_H_

#include <cstddef> // Include the <cstddef> header file to define size_t
#include <cstdint>
#include <base/macros/macros.h>
#include <base/utils/array_ref.h>

namespace container {

template <typename T>
struct DefaultPtrTraits {
    using PtrType = T*;
};

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
struct RestrictPtrTraits<T*> {
    using PtrType = T* __restrict__;
};
#endif

template <typename T, size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessorBase {
  public:

    using PtrType = typename PtrTraits<T>::PtrType;

    AT_HOST_DEVICE TensorAccessorBase(
            PtrType data,
            index_t* sizes,
            index_t* strides)
            : data_(data), sizes_(sizes), strides_(strides) {}
    
    AT_HOST int_array_ref sizes() const {
        return {sizes_, N};
    }

    AT_HOST int_array_ref strides() const {
        return {strides_, N};
    }

    AT_HOST_DEVICE index_t stride(index_t idx) const {
        return strides_[idx];
    }

    AT_HOST_DEVICE index_t size(index_t idx) const {
        return sizes_[idx];
    }

    AT_HOST_DEVICE PtrType data() {
        return data_;
    }

    AT_HOST_DEVICE const PtrType data() const {
        return data_;
    }

  protected:
    T* data_;
    const index_t* sizes_;
    const index_t* strides_;
};

template <typename T, size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessor : public TensorAccessorBase<T, N, PtrTraits, index_t> {
  public:
    using PtrType = typename PtrTraits<T>::PtrType;

    AT_HOST_DEVICE TensorAccessor(PtrType data, const index_t* sizes, const index_t* strides)
        : TensorAccessorBase<T, N, PtrTraits, index_t>(data, sizes, strides) {}

    AT_HOST_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t idx) {
        return TensorAccessor<T, N - 1, PtrTraits, index_t>(this->data_ + idx * this->strides_[0], this->sizes_ + 1, this->strides_ + 1);
    }

    AT_HOST_DEVICE const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t idx) const {
        return TensorAccessor<T, N - 1, PtrTraits, index_t>(this->data_ + idx * this->strides_[0], this->sizes_ + 1, this->strides_ + 1);
    }
};

template <typename T,
          template <typename U> class PtrTraits, typename index_t>
class TensorAccessor<T, 1, PtrTraits, index_t> : public TensorAccessorBase<T, 1, PtrTraits, index_t> {
  public:
    using PtrType = typename PtrTraits<T>::PtrType;
    AT_HOST_DEVICE TensorAccessor(T* data, const index_t* sizes, const index_t* strides)
        : TensorAccessorBase<T, 1, PtrTraits, index_t>(data, sizes, strides) {}

    AT_HOST_DEVICE T& operator[](index_t idx) {
        return this->data_[idx * this->strides_[0]];
    }

    AT_HOST_DEVICE const T& operator[](index_t idx) const {
        return this->data_[idx * this->strides_[0]];
    }
};


} // namespace container

#endif // ATEN_CORE_TENSOR_ACCESSOR_H_