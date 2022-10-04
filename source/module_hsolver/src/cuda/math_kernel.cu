#include "module_hsolver/include/math_kernel.h"
#include "module_psi/psi.h"
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>

using namespace hsolver;

// for this implementation, please check https://thrust.github.io/doc/group__transformed__reductions_ga321192d85c5f510e52300ae762c7e995.html
// denghui modify 2022-10-03
// Note that ddot_(2*dim,a,1,b,1) = REAL( zdotc_(dim,a,1,b,1) )
template<typename FPTYPE>
FPTYPE hsolver::zdot_real_gpu_cuda(
    const int &dim, 
    const std::complex<FPTYPE>* psi_L, 
    const std::complex<FPTYPE>* psi_R, 
    const psi::AbacusDevice_t device,
    const bool reduce)
{
    const FPTYPE* pL = reinterpret_cast<const FPTYPE*>(psi_L);
    const FPTYPE* pR = reinterpret_cast<const FPTYPE*>(psi_R);
    FPTYPE result = thrust::inner_product(thrust::device, pL, pL + dim * 2, pR, FPTYPE(0.0));
    if (reduce) {
        Parallel_Reduce::reduce_double_pool(result);
    }
    return result;
}

namespace hsolver {
    template double zdot_real_gpu_cuda<double>(const int &dim, const std::complex<double>* psi_L, const std::complex<double>* psi_R, const psi::AbacusDevice_t device, const bool reduce);
}

