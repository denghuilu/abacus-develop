#ifndef MODULE_PW_FFT_H
#define MODULE_PW_FFT_H

#include <string>
#include <complex>
#include "module_psi/include/memory.h"
#include "module_pw/include/fft_multi_device.h"

namespace ModulePW {

template<typename FPTYPE, typename Device>
class FFT
{
  public:

	FFT();
	~FFT();
	void clear(); //reset fft

	// init parameters of fft
	void initfft(int nx_in, int ny_in, int nz_in, int lixy_in, int rixy_in, int ns_in, int nplane_in, 
				 int nproc_in, bool gamma_only_in, bool xprime_in = true, bool mpifft_in = false);

	//init fftw_plans
	void setupFFT(); 

	//destroy fftw_plans
	void cleanFFT(); 

	void fftzfor(std::complex<FPTYPE>* & in, std::complex<FPTYPE>* & out);
	void fftzbac(std::complex<FPTYPE>* & in, std::complex<FPTYPE>* & out);
	void fftxyfor(std::complex<FPTYPE>* & in, std::complex<FPTYPE>* & out);
	void fftxybac(std::complex<FPTYPE>* & in, std::complex<FPTYPE>* & out);
	void fftxyr2c(FPTYPE * &in, std::complex<FPTYPE>* & out);
	void fftxyc2r(std::complex<FPTYPE>* & in, FPTYPE* & out);

	//init fftw_plans
	void initplan(); 
	void initplan_mpi();

  public:
	int fftnx=0, fftny=0;
	int fftnxy=0;
	int ny=0, nx=0, nz=0;
	int nxy=0;
	bool xprime = true; // true: when do recip2real, x-fft will be done last and when doing real2recip, x-fft will be done first; false: y-fft
                        // For gamma_only, true: we use half x; false: we use half y
	int lixy=0,rixy=0;  // lixy: the left edge of the pw ball in the y direction; rixy: the right edge of the pw ball in the x or y direction
	int ns=0; //number of sticks
	int nplane=0; //number of x-y planes
	int nproc=1; // number of proc.
	std::complex<FPTYPE> *auxg=nullptr, *auxr=nullptr; //fft space
	FPTYPE *r_rspace=nullptr; //real number space for r, [nplane * nx *ny]


  private:
	bool gamma_only=false;
	bool mpifft=false; // if use mpi fft, only used when define __FFTW3_MPI
	bool destroyp=true;
    using plan = typename fft::fft_plan<FPTYPE, Device>::plan;

	plan planzfor;
	plan planzbac;
	plan planxfor1;
	plan planxbac1;
	plan planxfor2;
	plan planxbac2;
	plan planyfor;
	plan planybac;
	plan planxr2c;
	plan planxc2r;
	plan planyr2c;
	plan planyc2r;

    Device* ctx = {};
    using delete_memory_op = psi::memory::delete_memory_op<std::complex<FPTYPE>, Device>;
    using resize_memory_op = psi::memory::resize_memory_op<std::complex<FPTYPE>, Device>;
    using fft_destroy_plan_op = fft::fft_destroy_plan_op<plan>;

};

} // namespace ModulePW

#endif // MODULE_PW_FFT_H

