#include "elecstate_pw.h"

#include "module_base/constants.h"
#include "src_parallel/parallel_reduce.h"
#include "src_pw/global.h"
#include "module_base/timer.h"

namespace elecstate
{


    // 没有考虑GlobalV::NSPIN ！= 1 /2 的时候
void ElecStatePW::psiToRho_CUDA(const psi::Psi<std::complex<double>, psi::DEVICE_GPU>& psi){
    ModuleBase::TITLE("ElecStatePW", "psiToRho_CUDA");
    ModuleBase::timer::tick("ElecStatePW", "psiToRho_CUDA");

    this->calculate_weights();
    this->calEBand();

    for(int is=0; is<GlobalV::NSPIN; is++)
	{
		ModuleBase::GlobalFunc::ZEROS(this->charge->rho[is], this->charge->nrxx);
		if (XC_Functional::get_func_type() == 3)
		{
            ModuleBase::GlobalFunc::ZEROS(this->charge->kin_r[is], this->charge->nrxx);
        }
	}

    for (int ik =0; ik < psi.get_nk(); ++ik){
        psi.fix_k(ik);

        // used for plane wavefunction FFT3D to real space in GPU
        double2* wfcr;
        cudaMalloc((void**)&wfcr, this->charge->nrxx * sizeof(float2));


        int npw = psi.get_current_nbas();
        int current_spin = 0;
        if (GlobalV::NSPIN == 2){
            current_spin = this->klist->isk[ik];
        }

        int nbands = psi.get_nbands();

        //  here we compute the band energy: the sum of the eigenvalues

        for (int ibnd = 0; ibnd < nbands; ibnd++){

            ///
            /// only occupied band should be calculated.
            if (this->wg(ik, ibnd) < ModuleBase::threshold_wg)
                continue;
            ///

            // this->basis->recip2real(&psi(ibnd,0), wfcr.data(), ik);
        }



    }





    ModuleBase::timer::tick("ElecStatePW", "psiToRho");
}



} // namespace elecstate