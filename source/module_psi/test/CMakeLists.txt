AddTest(
    TARGET psi_UT
    LIBS ${math_libs} base device container
    SOURCES 
        psi_test.cpp
        ../psi.cpp 
)