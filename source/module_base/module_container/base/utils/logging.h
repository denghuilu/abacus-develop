#ifndef BASE_CORE_LOGGING_H_
#define BASE_CORE_LOGGING_H_

#include <base/macros/macros.h>
#ifdef __ENABLE_BACKTRACE
#include <base/third_party/backward-cpp/backward.hpp>
#endif // __ENABLE_BACKTRACE
namespace base {
namespace utils {

// Note while in the calling situation of check_msg_impl and check_exit_impl, 
// the check has been failed, so we don't need to release the char* msg
inline static const char* check_msg_impl(const char* msg) {
  return msg;
}

inline static void check_exit_impl(const char* func, const char* file, uint32_t line, const char* msg) {
    fprintf(stderr, "Fatal error in function %s, file %s, line %u, \nwith message: \n\t\t%s\n", func, file, line, msg);
#ifdef __ENABLE_BACKTRACE
    backward::StackTrace st; st.load_here(32); st.skip_n_firsts(1);
    backward::Printer p; p.print(st);
#endif // __ENABLE_BACKTRACE
    std::abort();
}

} // namespace logging
} // namespace base
#endif // BASE_CORE_LOGGING_H_