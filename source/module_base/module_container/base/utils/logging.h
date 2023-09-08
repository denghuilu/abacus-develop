#ifndef BASE_CORE_LOGGING_H_
#define BASE_CORE_LOGGING_H_

#include <base/macros/macros.h>

namespace base {
namespace utils {

inline static const char* check_msg_impl(const char* msg) {
  return msg;
}

inline static void check_exit_impl(const char* func, const char* file, uint32_t line, const char* msg) {
    fprintf(stderr, "Fatal error in function %s, file %s, line %u:\nwith message: %s\n", func, file, line, msg);
    std::abort();
}

} // namespace logging
} // namespace base
#endif // BASE_CORE_LOGGING_H_