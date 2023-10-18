#ifndef ATEN_CORE_DISPATCH_H_
#define ATEN_CORE_DISPATCH_H_

#include <base/macros/macros.h>

template <int N>
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

template <typename Derived>
struct DispatchBase {
    void operator()() { (static_cast<Derived*>(this))->impl() };
};

#define DECLARE_DISPATCH(name)                          \
template <typename T, typename Device>                  \
struct name : DispatchBase<name<T, Device>>;            \
                                                        \
template <typename T, typename Device,                  \
          typename rT, typename... Args>                \
struct name<T, Device, rT (*)(Args...)>                 \
{                                                       \
    void impl();                                        \
};

#define DECLARE_DISPATCH(name)                          \
struct name : DispatchBase<name> {                      \
    void impl();                                        \
}                       

#define DEFINE_DISPATCHER(name)                         \
template <typename T, typename Device>                  \
void name<T, Device>::impl() {                          \
    /* do something */                                  \
}                                                       \
                                                        \

                                                      

#endif // ATEN_CORE_DISPATCH_H_