#include <complex>
#include <vector>
#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include "module_psi/include/memory.h"
#include "module_base/complexmatrix.h"
#include "module_base/lapack_connector.h"
#include "module_hsolver/include/dngvd_op.h"


class TestModuleHsolverMathDngvd : public ::testing::Test
{
    protected:
    using resize_memory_op_Z = psi::memory::resize_memory_op<std::complex<double>, psi::DEVICE_GPU>;
    using delete_memory_op_Z = psi::memory::delete_memory_op<std::complex<double>, psi::DEVICE_GPU>;
    using resize_memory_op_D = psi::memory::resize_memory_op<double, psi::DEVICE_GPU>;
    using delete_memory_op_D = psi::memory::delete_memory_op<double, psi::DEVICE_GPU>;
    // from CPU to GPU
    using synchronize_memory_op_C2G_Z = psi::memory::synchronize_memory_op<std::complex<double>, psi::DEVICE_GPU, psi::DEVICE_CPU>;
    using synchronize_memory_op_C2G_D = psi::memory::synchronize_memory_op<double, psi::DEVICE_GPU, psi::DEVICE_CPU>;
    using synchronize_memory_op_G2C_Z = psi::memory::synchronize_memory_op<std::complex<double>, psi::DEVICE_CPU, psi::DEVICE_GPU>;
    using synchronize_memory_op_G2C_D = psi::memory::synchronize_memory_op<double, psi::DEVICE_CPU, psi::DEVICE_GPU>;

    const psi::DEVICE_CPU * cpu_ctx = {};
    const psi::DEVICE_GPU * gpu_ctx = {};

    // prepare A & B in CPU
    std::vector<complex<double> > matrix_A = {
        {-0.65412617, -0.74208893}, {-2.21731157,  0.42540039}, {3.41302551,  -2.3175205 },
        {3.36373004,  -2.51647562}, {-2.985111  , -0.53251562}, {-0.27628221, -1.35701656},
        {0.37908265,   0.81605825}, { 1.66281318,  2.71761869}, {-2.36701143,  1.23009056}
    };
    std::vector<complex<double> > matrix_B = {
        {-2.21731157,  0.42540039}, {3.41302551,  -2.3175205 }, {3.36373004,  -2.51647562},
        {-0.65412617, -0.74208893}, {-2.985111  , -0.53251562}, {-0.27628221, -1.35701656},
        {0.37908265,   0.81605825}, { 1.66281318,  2.71761869}, {-2.36701143,  1.23009056}
    };
    const int matrix_size = 9;

    // // prepare A & B in GPU
    // std::complex<double>* device_matrix_A = nullptr;
    // std::complex<double>* device_matrix_B = nullptr;
    // resize_memory_op_Z()(gpu_ctx, device_matrix_A, matrix_size);
    // resize_memory_op_Z()(gpu_ctx, device_matrix_B, matrix_size);
    // synchronize_memory_op_C2G_Z()(gpu_ctx, cpu_ctx, device_matrix_A, matrix_A.data(), matrix_size);
    // synchronize_memory_op_C2G_Z()(gpu_ctx, cpu_ctx, device_matrix_B, matrix_B.data(), matrix_size);

    // prepare W & V in CPU in dngv_op 
    std::vector<double> W_dngv_op = {0.0, 0.0, 0.0};
    std::vector<complex<double> > matrix_V_dngv_op = {
        {0.0,  0.0}, {0.0,  0.0}, {0.0,  0.0},
        {0.0,  0.0}, {0.0,  0.0}, {0.0,  0.0},
        {0.0,  0.0}, {0.0,  0.0}, {0.0,  0.0}
    };

    // // prepare W & V in GPU in dngv_op 
    // double* device_W_dngv_op = nullptr;
    // resize_memory_op_D()(gpu_ctx, device_W_dngv_op, W_dngv_op.size());
    // synchronize_memory_op_C2G_D()(gpu_ctx, cpu_ctx, device_W_dngv_op, W_dngv_op.data(), W_dngv_op.size());

    // std::complex<double>* device_matrix_V_dngv_op = nullptr;
    // resize_memory_op_Z()(gpu_ctx, device_matrix_V_dngv_op, matrix_V_dngv_op.size());
    // synchronize_memory_op_C2G_Z()(gpu_ctx, cpu_ctx, device_matrix_V_dngv_op, matrix_V_dngv_op.data(), matrix_V_dngv_op.size());
    

    // prepare W & V in CPU in dngvx_op 
    std::vector<double> W_DNGVX = {0.0, 0.0};
    std::vector<complex<double> > matrix_V_DNGVX = {
        {0.0,  0.0}, {0.0,  0.0}, {0.0,  0.0},
        {0.0,  0.0}, {0.0,  0.0}, {0.0,  0.0}
    };
    // // prepare W & V in GPU in dngvx_op 
    // double* device_W_DNGVX = nullptr;
    // resize_memory_op_D()(gpu_ctx, device_W_DNGVX, W_DNGVX.size());
    // synchronize_memory_op_C2G_D()(gpu_ctx, cpu_ctx, device_W_DNGVX, W_DNGVX.data(), W_DNGVX.size());
    // std::complex<double>* device_matrix_V_DNGVX = nullptr;
    // resize_memory_op_Z()(gpu_ctx, device_matrix_V_DNGVX, matrix_V_DNGVX.size());
    // synchronize_memory_op_C2G_Z()(gpu_ctx, cpu_ctx, device_matrix_V_DNGVX, matrix_V_DNGVX.data(), matrix_V_DNGVX.size());
    

    // ModuleBase::ComplexMatrix A(3, 3);
    // ModuleBase::ComplexMatrix B(3, 3);
    // ModuleBase::ComplexMatrix V(3, 3);
    // A.c = matrix_A.data();
    // B.c = matrix_B.data();
    // V.c = matrix_V.data();

};

#if __UT_USE_CUDA || __UT_USE_ROCM

// computes all the eigenvalues and eigenvectors
TEST_F(TestModuleHsolverMathDngvd, dngv_gpu)
{
    // prepare A & B in GPU
    std::complex<double>* device_matrix_A = nullptr;
    std::complex<double>* device_matrix_B = nullptr;
    resize_memory_op_Z()(gpu_ctx, device_matrix_A, matrix_size);
    resize_memory_op_Z()(gpu_ctx, device_matrix_B, matrix_size);
    synchronize_memory_op_C2G_Z()(gpu_ctx, cpu_ctx, device_matrix_A, matrix_A.data(), matrix_size);
    synchronize_memory_op_C2G_Z()(gpu_ctx, cpu_ctx, device_matrix_B, matrix_B.data(), matrix_size);
    // prepare W & V in GPU in dngv_op 
    double* device_W_dngv_op = nullptr;
    resize_memory_op_D()(gpu_ctx, device_W_dngv_op, W_dngv_op.size());
    synchronize_memory_op_C2G_D()(gpu_ctx, cpu_ctx, device_W_dngv_op, W_dngv_op.data(), W_dngv_op.size());
    std::complex<double>* device_matrix_V_dngv_op = nullptr;
    resize_memory_op_Z()(gpu_ctx, device_matrix_V_dngv_op, matrix_V_dngv_op.size());
    synchronize_memory_op_C2G_Z()(gpu_ctx, cpu_ctx, device_matrix_V_dngv_op, matrix_V_dngv_op.data(), matrix_V_dngv_op.size());


    // run in GPU
    hsolver::dngv_op<double, psi::DEVICE_GPU>()(
        gpu_ctx,
        3,
        3,
        device_matrix_A,
        device_matrix_B,
        device_W_dngv_op,
        device_matrix_V_dngv_op
    );

    // copy W data from GPU to CPU
    std::vector<double> W_result = {0.0, 0.0, 0.0};
    synchronize_memory_op_G2C_D()(cpu_ctx, gpu_ctx, W_result.data(), device_W_dngv_op, W_result.size());
    // copy V data from GPU to CPU 
    std::vector<complex<double> > V_result = {
        {0.0,  0.0}, {0.0,  0.0}, {0.0,  0.0},
        {0.0,  0.0}, {0.0,  0.0}, {0.0,  0.0},
        {0.0,  0.0}, {0.0,  0.0}, {0.0,  0.0}
    };
    synchronize_memory_op_G2C_Z()(cpu_ctx, gpu_ctx, V_result.data(), device_matrix_V_dngv_op, V_result.size());


    // run in CPU
    hsolver::dngv_op<double, psi::DEVICE_CPU>()(
        cpu_ctx,
        3,
        3,
        matrix_A.data(),
        matrix_B.data(),
        W_dngv_op.data(),
        matrix_V_dngv_op.data()
    );
    

    // we need to compare 
    //          1. W with W_result 
    //          2. matrix_V with V_result
    for (int i = 0; i < W_dngv_op.size(); i++)
    {
        EXPECT_LT(fabs(W_dngv_op[i] - W_result[i]), 1e-8);
    }
    for (int i = 0; i < matrix_V_dngv_op.size(); i++)
    {
        EXPECT_LT(fabs(matrix_V_dngv_op[i].imag() - V_result[i].imag()), 1e-8);
        EXPECT_LT(fabs(matrix_V_dngv_op[i].real() - V_result[i].real()), 1e-8);
    }
    
    // delete values in GPU
    delete_memory_op_Z()(gpu_ctx, device_matrix_A);
    delete_memory_op_Z()(gpu_ctx, device_matrix_B);
    delete_memory_op_Z()(gpu_ctx, device_matrix_V_dngv_op);
    delete_memory_op_D()(gpu_ctx, device_W_dngv_op);
}


// computes the first m eigenvalues ​​and their corresponding eigenvectors
TEST_F(TestModuleHsolverMathDngvd, dngvx_gpu)
{
    // prepare A & B in GPU
    std::complex<double>* device_matrix_A = nullptr;
    std::complex<double>* device_matrix_B = nullptr;
    resize_memory_op_Z()(gpu_ctx, device_matrix_A, matrix_size);
    resize_memory_op_Z()(gpu_ctx, device_matrix_B, matrix_size);
    synchronize_memory_op_C2G_Z()(gpu_ctx, cpu_ctx, device_matrix_A, matrix_A.data(), matrix_size);
    synchronize_memory_op_C2G_Z()(gpu_ctx, cpu_ctx, device_matrix_B, matrix_B.data(), matrix_size);
    // prepare W & V in GPU in dngvx_op 
    double* device_W_DNGVX = nullptr;
    resize_memory_op_D()(gpu_ctx, device_W_DNGVX, W_DNGVX.size());
    synchronize_memory_op_C2G_D()(gpu_ctx, cpu_ctx, device_W_DNGVX, W_DNGVX.data(), W_DNGVX.size());
    std::complex<double>* device_matrix_V_DNGVX = nullptr;
    resize_memory_op_Z()(gpu_ctx, device_matrix_V_DNGVX, matrix_V_DNGVX.size());
    synchronize_memory_op_C2G_Z()(gpu_ctx, cpu_ctx, device_matrix_V_DNGVX, matrix_V_DNGVX.data(), matrix_V_DNGVX.size());
    
    // run in GPU
    hsolver::dngvx_op<double, psi::DEVICE_GPU>()(
        gpu_ctx,
        3,
        3,
        device_matrix_A,
        device_matrix_B,
        2,
        device_W_DNGVX,
        device_matrix_V_DNGVX
    );
    // copy W data from GPU to CPU
    std::vector<double> W_result = {0.0, 0.0};
    synchronize_memory_op_G2C_D()(cpu_ctx, gpu_ctx, W_result.data(), device_W_DNGVX, W_result.size());
    // copy V data from GPU to CPU 
    std::vector<complex<double> > V_result = {
        {0.0,  0.0}, {0.0,  0.0}, {0.0,  0.0},
        {0.0,  0.0}, {0.0,  0.0}, {0.0,  0.0}
    };
    synchronize_memory_op_G2C_Z()(cpu_ctx, gpu_ctx, V_result.data(), device_matrix_V_DNGVX, V_result.size());

    // std::cout << W_result[0] << "\t" <<  W_result[1] << std::endl;
    

    // run in CPU
    hsolver::dngvx_op<double, psi::DEVICE_CPU>()(
        cpu_ctx,
        3,
        3,
        matrix_A.data(),
        matrix_B.data(),
        2,
        W_DNGVX.data(),
        matrix_V_DNGVX.data()
    );
    
    // std::cout << W_DNGVX[0] << "\t" <<  W_DNGVX[1] << std::endl;

    // std::fstream output;
    // std::string filename = "/home/haozhihan/code/1031/abacus-develop/build/hhz.txt";
    // output.open(filename, fstream::out);
    // output << "GPU  " << std::endl;
    // output << "GPU  " << W_result[0] << "\t" <<  W_result[1] << std::endl;
    // output << "CPU  " << W_DNGVX[0] << "\t" <<  W_DNGVX[1] << std::endl;
    // output.close();


    // we need to compare 
    //          1. W with W_result 
    //          2. matrix_V with V_result
    for (int i = 0; i < W_DNGVX.size(); i++)
    {
        EXPECT_LT(fabs(W_DNGVX[i] - W_result[i]), 1e-8);
    }
    for (int i = 0; i < matrix_V_DNGVX.size(); i++)
    {
        EXPECT_LT(fabs(matrix_V_DNGVX[i].imag() - V_result[i].imag()), 1e-8);
        EXPECT_LT(fabs(matrix_V_DNGVX[i].real() - V_result[i].real()), 1e-8);
    }
    
    // delete values in GPU
    delete_memory_op_Z()(gpu_ctx, device_matrix_A);
    delete_memory_op_Z()(gpu_ctx, device_matrix_B);
    delete_memory_op_Z()(gpu_ctx, device_matrix_V_DNGVX);
    delete_memory_op_D()(gpu_ctx, device_W_DNGVX);
}



#endif // __UT_USE_CUDA || __UT_USE_ROCM
