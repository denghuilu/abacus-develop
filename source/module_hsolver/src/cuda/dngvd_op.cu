#include "module_hsolver/include/dngvd_op.h"
#include "module_hsolver/include/math_kernel.h"

#include <cusolverDn.h>


namespace hsolver {

template <>
void dngvd_op<double, psi::DEVICE_GPU>::operator()(
        const psi::DEVICE_GPU* d,
        const int row,
        const int col,
        const std::complex<double>* A,
        const std::complex<double>* B,
        const int m,
        double* W, 
        std::complex<double>* V)
{
    matrixTranspose_op<double, psi::DEVICE_GPU>()(d, row, col, A, A);
    matrixTranspose_op<double, psi::DEVICE_GPU>()(d, row, col, B, B);
    // init A_eigenvectors and all_W 
    double2* A_eigenvectors;
    checkCudaErrors( cudaMalloc ((void**)&A_eigenvectors, sizeof(double2) * row * col) );
    checkCudaErrors( cudaMemcpy(A_eigenvectors, A, sizeof(double2) * row * col, cudaMemcpyDeviceToDevice) );
    double* all_W ;
    checkCudaErrors( cudaMalloc ((void**)&all_W, sizeof(double) * row) );

    // init
    cusolverDnHandle_t cusolverH;
    checkCudaErrors( cusolverDnCreate(&cusolverH) );
    int * devInfo ;
    checkCudaErrors( cudaMalloc ((void**)&devInfo, sizeof(int)) );
    
    // calculate the sizes needed for pre-allocated buffer.
    int lwork = 0;
    checkCudaErrors( 
        cusolverDnZhegvd_bufferSize(
            cusolverH,
            CUSOLVER_EIG_TYPE_1,        // itype = CUSOLVER_EIG_TYPE_1: A*x = (lambda)*B*x.
            CUSOLVER_EIG_MODE_VECTOR,   // jobz = CUSOLVER_EIG_MODE_VECTOR : Compute eigenvalues and eigenvectors.
            CUBLAS_FILL_MODE_LOWER,
            row,
            A_eigenvectors,
            col,
            (double2*)B,
            col,
            all_W,
            &lwork)
    );

    // allocate memery
    cuDoubleComplex *d_work;
    checkCudaErrors( cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork) );

    // compute eigenvalues and eigenvectors. 
    checkCudaErrors(
        cusolverDnZhegvd(
            cusolverH,
            CUSOLVER_EIG_TYPE_1,        // itype = CUSOLVER_EIG_TYPE_1: A*x = (lambda)*B*x.
            CUSOLVER_EIG_MODE_VECTOR,   // jobz = CUSOLVER_EIG_MODE_VECTOR : Compute eigenvalues and eigenvectors.
            CUBLAS_FILL_MODE_LOWER,
            row,
            A_eigenvectors,
            col,
            (double2*)B,
            col,
            all_W,
            d_work,
            lwork,
            devInfo)
    );

    checkCudaErrors( cudaDeviceSynchronize() );

    // get eigenvalues and eigenvectors.  only m !
    checkCudaErrors( cudaMemcpy(W, all_W, sizeof(double)*m, cudaMemcpyDeviceToDevice) );
    matrixTranspose_op<double, psi::DEVICE_GPU>()(d, row, col, A_eigenvectors, A_eigenvectors);
    checkCudaErrors( cudaMemcpy(V, A_eigenvectors, sizeof(std::complex<double>)*col*m, cudaMemcpyDeviceToDevice) );
    // free the buffer
    checkCudaErrors( cudaFree(d_work) );
    // free resources and destroy
    checkCudaErrors( cudaFree(A_eigenvectors) );
    checkCudaErrors( cudaFree(all_W) );
    checkCudaErrors( cudaFree(devInfo) );
    checkCudaErrors( cusolverDnDestroy(cusolverH) ); 
}


}