#include "module_hsolver/include/dngvd_op.h"


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
    createBLAShandle();
    matrixTranspose_op<double, psi::DEVICE_GPU>()(d, row, col, A, A);
    matrixTranspose_op<double, psi::DEVICE_GPU>()(d, row, col, B, B);
    // init A_eigenvectors and all_W 
    double2* A_eigenvectors;
    cudaMalloc ((void**)&A_eigenvectors, sizeof(double2) * row * col);
    cudaMemcpy(A_eigenvectors, A, sizeof(double2) * row * col, cudaMemcpyDeviceToDevice);
    double* all_W ;
    cudaMalloc ((void**)&all_W, sizeof(double) * row);

    // init
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);
    int * devInfo ;
    cudaMalloc ((void**)&devInfo, sizeof(int));
    
    // calculate the sizes needed for pre-allocated buffer.
    int lwork = 0;
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
        &lwork);

    // allocate memery
    cuDoubleComplex *d_work;
    cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);

    // compute eigenvalues and eigenvectors. 
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
        devInfo);

    cudaDeviceSynchronize();

    // get eigenvalues and eigenvectors.  only m !
    cudaMemcpy(W, all_W, sizeof(double)*m, cudaMemcpyDeviceToDevice);
    matrixTranspose_op<double, psi::DEVICE_GPU>()(d, row, col, A_eigenvectors, A_eigenvectors);
    cudaMemcpy(V, A_eigenvectors, sizeof(std::complex<double>)*col*m, cudaMemcpyDeviceToDevice);
    // free the buffer
    cudaFree(d_work);
    // free resources and destroy
    cudaFree(A_eigenvectors);
    cudaFree(all_W);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolverH); 
    destoryBLAShandle();
}


}