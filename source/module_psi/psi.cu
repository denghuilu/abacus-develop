#include "psi.h"
#include <cuComplex.h>
#include <thrust/extrema.h>

#include <fstream>
#include <iostream>
#include <unistd.h>

namespace psi
{

    
// __global__ void gpu_norm(cuDoubleComplex *psi_band, double *result, int size){
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if(tid < size){
//         result[tid] =  cuCreal(psi_band[tid]) * cuCreal(psi_band[tid]) + cuCimag(psi_band[tid]) * cuCimag(psi_band[tid]);
//     }
// }


// template<>
// void psi_gpu_test_in( Psi<std::complex<double>> &psi){
//     for (int ik = 0; ik < psi.get_nk(); ++ik)
//     {
//         psi.fix_k(ik);
//         for (int ibnd = 0; ibnd < psi.get_nbands(); ibnd++)
//         {
//             double sum = 0.0;
//             double result[psi.get_nbasis()+1];
//             double *dev_result;       
//             cuDoubleComplex psi_band[psi.get_nbasis()+1];
//             cuDoubleComplex *dev_psi_band;
//             for (int ibasis = 0; ibasis < psi.get_nbasis(); ibasis++){
//                 psi_band[ibasis] = make_cuDoubleComplex(psi(ibnd, ibasis).real(), psi(ibnd, ibasis).imag());
//                 result[ibasis] = 0.0;
//             }
//             cudaMalloc((void**)&dev_psi_band, (psi.get_nbasis()+1) * sizeof(cuDoubleComplex)); 
//             cudaMalloc((void**)&dev_result, (psi.get_nbasis()+1) * sizeof(double));
//             cudaMemcpy(dev_psi_band, psi_band, (psi.get_nbasis()+1) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice); 
//             cudaMemcpy(dev_result, result, (psi.get_nbasis()+1) * sizeof(double), cudaMemcpyHostToDevice);  
//             int thread = 512;
//             int block = (psi.get_nbasis() + thread - 1) / thread;
//             gpu_norm<<<block,thread>>>(dev_psi_band, dev_result, psi.get_nbasis()); 
//             cudaMemcpy(result, dev_result, (psi.get_nbasis()+1) * sizeof(double), cudaMemcpyDeviceToHost);
//             for (int ibasis = 0; ibasis < psi.get_nbasis(); ibasis++){
//                 sum = sum + result[ibasis];
//             }
//             std::cout << "GPU: SUM = " << sum << std::endl;  
//         }   
//     }
// }



__global__ void gpu_norm_test_double2(double2 *psi_band, double *result, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size){
        // result[tid] =  cuCreal(psi_band[tid]) * cuCreal(psi_band[tid]) + cuCimag(psi_band[tid]) * cuCimag(psi_band[tid]);

        result[tid] = psi_band[tid].x * psi_band[tid].x + psi_band[tid].y * psi_band[tid].y;

    }
}


template<>
void psi_gpu_test_in( Psi<std::complex<double>> &psi){

    // int nk = psi.get_nk();
    // int nband = psi.get_nbands();
    // int nbasis = psi.get_nbasis();
    
    psi.fix_k(0);
    
    double2 *dev_psi;
    cudaMalloc((void**)&dev_psi, (psi.size()+1) * sizeof(double2));
    cudaMemcpy(dev_psi, psi.get_pointer(), (psi.size()+1) * sizeof(double2), cudaMemcpyHostToDevice);

    double *result;
    result = (double*) malloc((psi.size()+1) * sizeof(double));
    double *dev_result;
    cudaMalloc((void**)&dev_result, (psi.size()+1) * sizeof(double));

    // 调用核函数
    int thread = 512;
    int block = (psi.size() + thread - 1) / thread;
    gpu_norm_test_double2<<<block,thread>>>(dev_psi, dev_result, psi.size());
    
    cudaMemcpy(result, dev_result, (psi.size()+1) * sizeof(double), cudaMemcpyDeviceToHost);

    for (int ik = 0; ik < psi.get_nk(); ++ik)
    {
        for (int ibnd = 0; ibnd < psi.get_nbands(); ibnd++)
        {
            double sum = 0.0;
            for (int ibasis = 0; ibasis < psi.get_nbasis(); ibasis++){
                sum = sum + result[(ik * psi.get_nbands() + ibnd) * psi.get_nbasis() + ibasis];
            }
            std::cout << "Test2: Double2 : SUM = " << sum << std::endl;  
        }   
    }
}


template<>
void psi_gpu_test_in(Psi<double> &psi){
    return;
}


template void psi_gpu_test_in <double> (Psi<double> &psi);
template void psi_gpu_test_in <std::complex<double> > (Psi<complex<double> > &psi);

}