if(USE_CUDA)
    list(APPEND objects
        psi.cu
    )
endif()
add_library(
    psi
    OBJECT
    psi.cpp
    ${objects}
)