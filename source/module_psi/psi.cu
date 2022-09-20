#include "psi.h"
#include <cuComplex.h>
#include <thrust/extrema.h>

#include <fstream>
#include <iostream>
#include <unistd.h>

namespace psi
{

__global__ void kernel_norm(double2 *psi_band, double *result, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size){
        result[tid] = psi_band[tid].x * psi_band[tid].x + psi_band[tid].y * psi_band[tid].y;
    }
}


template<>
void psi_gpu_test_in( Psi<std::complex<double>> &psi){

    // result in cpu
    double *result;
    result = (double*) malloc((psi.size()) * sizeof(double));
    // result in gpu
    double *dev_result;
    cudaMalloc((void**)&dev_result, (psi.size()) * sizeof(double));
    // complex in GPU
    psi.fix_k(0); // important!!!
    Psi<std::complex<double>, DEVICE_GPU>* psi_complex_gpu = new Psi<std::complex<double>, DEVICE_GPU>(psi);

    // 调用核函数
    int thread = 512;
    int block = (psi.size() + thread - 1) / thread;
    // 这里有complex 到 double2 的强转
    kernel_norm<<<block,thread>>>((double2*)psi_complex_gpu->get_pointer(), dev_result, psi.size());
    // result from gpu to cpu
    cudaMemcpy(result, dev_result, (psi.size()) * sizeof(double), cudaMemcpyDeviceToHost);

    // check result
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




// 检查 将Complex in CPU 通过cudaMemcpy 到 double2 in GPU 数据的正确性
template<>
void checkComplexToDouble2(Psi<std::complex<double>> &psi){
    psi.fix_k(0);
    // psi object in GPU
    double2 *gpu_psi;
    cudaMalloc((void**)&gpu_psi, (psi.size()+1) * sizeof(double2));
    // Complex Psi in CPU --> double2 psi in GPU
    cudaMemcpy(gpu_psi, psi.get_pointer(), (psi.size()+1) * sizeof(double2), cudaMemcpyHostToDevice);

    // psi object in CPU
    double2* cpu_psi = (double2*) malloc((psi.size()+1) * sizeof(double2));
    // double2 psi in GPU --> double2 psi in CPU
    cudaMemcpy(cpu_psi, gpu_psi, (psi.size()+1) * sizeof(double2), cudaMemcpyDeviceToHost);

    std::cout << "double2 :Real:" << cpu_psi[0].x << ", Imag:" <<  cpu_psi[0].y << std::endl;

    psi.fix_k(0);
    std::cout << "complex :Real:" << psi.get_pointer()[0].real() << ", Imag:" <<  psi.get_pointer()[0].imag() << std::endl;
}


template<>
void psiTo1(Psi<std::complex<double>> &psi){
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
    kernel_norm<<<block,thread>>>(dev_psi, dev_result, psi.size());
    
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
void checkComplexToDouble2(Psi<double> &psi){
}

template<>
void psiTo1(Psi<double> &psi){
    return;
}

template<>
void psi_gpu_test_in(Psi<double> &psi){
    return;
}


template void psi_gpu_test_in <double> (Psi<double> &psi);
template void psi_gpu_test_in <std::complex<double> > (Psi<complex<double> > &psi);

template void psiTo1 <double> (Psi<double> &psi);
template void psiTo1 <std::complex<double> > (Psi<complex<double> > &psi);

template void checkComplexToDouble2 <double> (Psi<double> &psi);
template void checkComplexToDouble2 <std::complex<double> > (Psi<complex<double> > &psi);
}