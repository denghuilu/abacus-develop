# obtain third_party dependencies
# spdlog is a header only C++ logging library. It is very fast and has a low memory footprint.
include(FetchContent)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY    https://github.com/gabime/spdlog.git
  GIT_TAG           v1.12.0
)

FetchContent_MakeAvailable(spdlog)

set(ATen_CPU_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS} spdlog PARENT_SCOPE)
