#ifndef ATEN_CORE_TENSOR_ITERATOR_H_
#define ATEN_CORE_TENSOR_ITERATOR_H_

#include <base/macros/macros.h>

namespace container {

class TensorIteratorConfig final {
    friend class TensorIterator;
    friend class TensorIteratorBase;

    TensorIteratorConfig() = default;
    ~TensorIteratorConfig() = default;

    DISALLOW_COPY_MOVE_AND_ASSIGN(TensorIteratorConfig);

    TensorIteratorConfig& check_all_same_dtype(bool check_all_same_dtype) {
        check_all_same_dtype_ = check_all_same_dtype;
        return *this;
    }

    TensorIteratorConfig& add_input(const Tensor& input);
    TensorIteratorConfig& add_output(const Tensor& output);

    TensorIteratorConfig& add_borrowed_input(const Tensor& input);
    TensorIteratorConfig& add_borrowed_output(const Tensor& output);

  private:
    bool check_all_same_dtype_ = false;

};

class TensorIterator {
public:
    TensorIterator() {}
    ~TensorIterator() {}

    template<typename Func>
    void for_each(Func& func) {
        for (int i = 0; i < N; i++) {
            func(i);
        }
    }
};

} // namespace container

#endif // ATEN_CORE_TENSOR_ITERATOR_H_