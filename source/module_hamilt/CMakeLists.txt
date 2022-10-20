add_subdirectory(ks_pw)
add_subdirectory(ks_lcao)

list(APPEND hamilt_srcs
    operator.cpp
    hamilt_pw.cpp
    hamilt_lcao.cpp
    ks_pw/operator_pw.cpp
    ks_pw/ekinetic_pw.cpp
    ks_pw/veff_pw.cpp
    ks_pw/nonlocal_pw.cpp
    ks_pw/meta_pw.cpp
    ks_pw/velocity_pw.cpp    
    ks_lcao/operator_lcao.cpp
    ks_lcao/ekinetic_lcao.cpp
    ks_lcao/nonlocal_lcao.cpp
    ks_lcao/overlap_lcao.cpp
    ks_lcao/veff_lcao.cpp
    ks_lcao/meta_lcao.cpp
    ks_lcao/op_dftu_lcao.cpp
    ks_lcao/deepks_lcao.cpp
    ks_lcao/op_exx_lcao.cpp
    src/ekinetic.cpp
    src/nonlocal.cpp
)

if (USE_CUDA) 
    list(APPEND hamilt_srcs src/cuda/ekinetic.cu src/cuda/nonlocal.cu)
elseif(USE_ROCM)
    list(APPEND hamilt_srcs src/rocm/ekinetic.cu src/cuda/nonlocal.cu)
endif()

add_library(
    hamilt
    OBJECT
    ${hamilt_srcs}
)

IF (BUILD_TESTING)
  add_subdirectory(test)
endif()