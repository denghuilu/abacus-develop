#include "stress_func.h"
#include "../module_base/math_polyint.h"
#include "../module_base/math_ylmreal.h"
#include "../module_base/timer.h"
#include "global.h"

//calculate the nonlocal pseudopotential stress in PW
template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::stress_nl(ModuleBase::matrix& sigma, const ModuleBase::matrix& wg, const psi::Psi<complex<FPTYPE>, Device>* psi_in)
{
	ModuleBase::TITLE("Stress_Func","stres_nl");
	ModuleBase::timer::tick("Stress_Func","stres_nl");
	
	const int nkb = GlobalC::ppcell.nkb;
	if(nkb == 0) 
	{
		ModuleBase::timer::tick("Stress_Func","stres_nl");
		return;
	}
	FPTYPE sigmanlc[3][3];
	for(int l=0;l<3;l++)
	{
		for(int m=0;m<3;m++)
		{
			sigmanlc[l][m]=0.0;
		}
	}
	
	// dbecp: conj( -iG * <Beta(nkb,npw)|psi(nbnd,npw)> )
	ModuleBase::ComplexMatrix dbecp( GlobalV::NBANDS, nkb );
	ModuleBase::ComplexMatrix becp( GlobalV::NBANDS, nkb );

	// vkb1: |Beta(nkb,npw)><Beta(nkb,npw)|psi(nbnd,npw)>
	ModuleBase::ComplexMatrix vkb1( nkb, GlobalC::wf.npwx );
	ModuleBase::ComplexMatrix vkb0[3];
	for(int i=0;i<3;i++){
		vkb0[i].create(nkb, GlobalC::wf.npwx);
	}
	ModuleBase::ComplexMatrix vkb2( nkb, GlobalC::wf.npwx );
    for (int ik = 0;ik < GlobalC::kv.nks;ik++)
    {   	  
		if (GlobalV::NSPIN==2) GlobalV::CURRENT_SPIN = GlobalC::kv.isk[ik];
		const int npw = GlobalC::kv.ngk[ik];
		// generate vkb
		if (GlobalC::ppcell.nkb > 0)
		{
			GlobalC::ppcell.getvnl(ik, GlobalC::ppcell.vkb);
		}

		// get becp according to wave functions and vkb
		// important here ! becp must set zero!!
		// vkb: Beta(nkb,npw)
		// becp(nkb,nbnd): <Beta(nkb,npw)|psi(nbnd,npw)>
        becp.zero_out();
		const std::complex<FPTYPE>* ppsi=nullptr;
		if(psi_in!=nullptr)
		{
			ppsi = &(psi_in[0](ik, 0, 0));
		}
		else
		{
			ppsi = &(GlobalC::wf.evc[ik](0, 0));
		}
		char transa = 'C';
        char transb = 'N';
        ///
        ///only occupied band should be calculated.
        ///
        int nbands_occ = GlobalV::NBANDS;
        while(wg(ik, nbands_occ-1) < ModuleBase::threshold_wg)
        {
            nbands_occ--;
        }
        int npm = GlobalV::NPOL * nbands_occ;
        zgemm_(&transa,
            &transb,
            &nkb,
            &npm,
            &npw,
            &ModuleBase::ONE,
            GlobalC::ppcell.vkb.c,
            &GlobalC::wf.npwx,
            ppsi,
            &GlobalC::wf.npwx,
            &ModuleBase::ZERO,
            becp.c,
            &nkb);
		//becp calculate is over , now we should broadcast this data.
		Parallel_Reduce::reduce_complex_double_pool( becp.c, becp.size);

		for (int i = 0; i < 3; i++) 
		{
			get_dvnl1(vkb0[i], ik, i);
		}
        get_dvnl2(vkb2, ik);

        ModuleBase::Vector3<FPTYPE> qvec;
        FPTYPE* qvec0[3];
		qvec0[0] = &(qvec.x);
		qvec0[1] = &(qvec.y);
		qvec0[2] = &(qvec.z);

        for (int ipol = 0; ipol < 3; ipol++) 
		{
            for (int jpol = 0; jpol < ipol + 1; jpol++) 
			{
				dbecp.zero_out();
				vkb1.zero_out();
				for (int i = 0; i < nkb; i++) 
				{
					std::complex<FPTYPE>* pvkb0i = &vkb0[ipol](i, 0);
					std::complex<FPTYPE>* pvkb0j = &vkb0[jpol](i, 0);
					std::complex<FPTYPE>* pvkb1 = &vkb1(i, 0);
					// third term of dbecp_noevc
					//std::complex<FPTYPE>* pvkb = &vkb2(i,0);
					//std::complex<FPTYPE>* pdbecp_noevc = &dbecp_noevc(i, 0);
					for (int ig = 0; ig < npw; ig++) 
					{
						qvec = GlobalC::wfcpw->getgpluskcar(ik, ig);

						pvkb1[ig] += 0.5 * qvec0[ipol][0] * pvkb0j[ig] +
									0.5 * qvec0[jpol][0] * pvkb0i[ig];
						
					} // end ig
					  
				}//end nkb
				ModuleBase::ComplexMatrix dbecp_noevc(nkb, GlobalC::wf.npwx, true);
				for (int i = 0; i < nkb; i++) 
				{
					std::complex<FPTYPE>* pdbecp_noevc = &dbecp_noevc(i, 0);
					std::complex<FPTYPE>* pvkb = &vkb1(i, 0);
					// first term
					for (int ig = 0; ig < npw;ig++) 
					{
						pdbecp_noevc[ig] -= 2.0 * pvkb[ig];
					}
					// second termi
					if (ipol == jpol)
					{
						pvkb = &GlobalC::ppcell.vkb(i, 0);
						for (int ig = 0; ig < npw;ig++) 
						{
							pdbecp_noevc[ig] -= pvkb[ig];
						}
					}
					// third term
					pvkb = &vkb2(i,0);
					for (int ig = 0; ig < npw;ig++) 
					{
						qvec =	GlobalC::wfcpw->getgpluskcar(ik, ig);
						FPTYPE qm1;
						if(qvec.norm2() > 1e-16) qm1 = 1.0 / qvec.norm(); 
						else qm1 = 0; 
						pdbecp_noevc[ig] -= 2.0 * pvkb[ig] * qvec0[ipol][0] * 
							qvec0[jpol][0] * qm1 *	GlobalC::ucell.tpiba;
					} // end ig
				}     // end i
				zgemm_(&transa,
					&transb,
					&nkb,
					&npm,
					&npw,
					&ModuleBase::ONE,
					dbecp_noevc.c,
					&GlobalC::wf.npwx,
					ppsi,
					&GlobalC::wf.npwx,
					&ModuleBase::ZERO,
					dbecp.c,
					&nkb);

				//              don't need to reduce here, keep
				//              dbecp different in each
				//              processor, and at last sum up
				//              all the forces.
				//              Parallel_Reduce::reduce_complex_double_pool(
				//              dbecp.ptr, dbecp.ndata);

				//              FPTYPE *cf = new
				//              FPTYPE[GlobalC::ucell.nat*3];
				//              ModuleBase::GlobalFunc::ZEROS(cf,
				//              GlobalC::ucell.nat);
				for (int ib=0; ib<nbands_occ; ib++)
				{
					FPTYPE fac = wg(ik, ib) * 1.0;
					int iat = 0;
					int sum = 0;
					for (int it=0; it<GlobalC::ucell.ntype; it++)
					{
						const int Nprojs = GlobalC::ucell.atoms[it].ncpp.nh;
						for (int ia=0; ia<GlobalC::ucell.atoms[it].na; ia++)
						{
							for (int ip1=0; ip1<Nprojs; ip1++)
							{
								for(int ip2=0; ip2<Nprojs; ip2++)
								{
									if(!GlobalC::ppcell.multi_proj && ip1 != ip2) 
									{
										continue;
									}
									FPTYPE ps = GlobalC::ppcell.deeq(GlobalV::CURRENT_SPIN, iat, ip1, ip2) ;
									const int inkb1 = sum + ip1;
									const int inkb2 = sum + ip2;
									//out<<"\n ps = "<<ps;

								
									const FPTYPE dbb = ( conj( dbecp( ib, inkb1) ) * becp( ib, inkb2) ).real();
									sigmanlc[ipol][ jpol] -= ps * fac * dbb;
								}
							 
							}//end ip
							++iat;        
							sum+=Nprojs;
						}//ia
					} //end it
				} //end band
            }//end jpol
		}//end ipol
	}// end ik

	// sum up forcenl from all processors
	for(int l=0;l<3;l++)
	{
		for(int m=0;m<3;m++)
		{
			if(m>l) 
			{
				sigmanlc[l][m] = sigmanlc[m][l];
			}
			Parallel_Reduce::reduce_double_all( sigmanlc[l][m] ); //qianrui fix a bug for kpar > 1
		}
	}

//        Parallel_Reduce::reduce_double_all(sigmanl.c, sigmanl.nr * sigmanl.nc);
        
	for (int ipol = 0; ipol<3; ipol++)
	{
		for(int jpol = 0; jpol < 3; jpol++)
		{
			sigmanlc[ipol][jpol] *= 1.0 / GlobalC::ucell.omega;
		}
	}
	
	for (int ipol = 0; ipol<3; ipol++)
	{
		for(int jpol = 0; jpol < 3; jpol++)
		{
			sigma(ipol,jpol) = sigmanlc[ipol][jpol] ;
		}
	}
	//do symmetry
	if(ModuleSymmetry::Symmetry::symm_flag)
	{
		GlobalC::symm.stress_symmetry(sigma, GlobalC::ucell);
	}//end symmetry
	
	//  this->print(GlobalV::ofs_running, "nonlocal stress", stresnl);
	ModuleBase::timer::tick("Stress_Func","stres_nl");
	return;
}

template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::get_dvnl1
(
	ModuleBase::ComplexMatrix &vkb,
	const int ik,
	const int ipol
)
{
	if(GlobalV::test_pp) ModuleBase::TITLE("Stress_Func","get_dvnl1");

	const int lmaxkb = GlobalC::ppcell.lmaxkb;
	if(lmaxkb < 0)
	{
		return;
	}

	const int npw = GlobalC::kv.ngk[ik];
	const int nhm = GlobalC::ppcell.nhm;
	int ig, ia, nb, ih;
	ModuleBase::matrix vkb1(nhm, npw);
	vkb1.zero_out();
	FPTYPE *vq = new FPTYPE[npw];
	const int x1= (lmaxkb + 1)*(lmaxkb + 1);

	ModuleBase::matrix dylm(x1, npw);
	ModuleBase::Vector3<FPTYPE> *gk = new ModuleBase::Vector3<FPTYPE>[npw];
	for (ig = 0;ig < npw;ig++)
	{
		gk[ig] = GlobalC::wf.get_1qvec_cartesian(ik, ig);
	}
			   
	dylmr2(x1, npw, gk, dylm, ipol);

	int jkb = 0;
	for(int it = 0;it < GlobalC::ucell.ntype;it++)
	{
		if(GlobalV::test_pp>1) ModuleBase::GlobalFunc::OUT("it",it);
		// calculate beta in G-space using an interpolation table
		const int nbeta = GlobalC::ucell.atoms[it].ncpp.nbeta;
		const int nh = GlobalC::ucell.atoms[it].ncpp.nh;

		if(GlobalV::test_pp>1) ModuleBase::GlobalFunc::OUT("nbeta",nbeta);

		for (nb = 0;nb < nbeta;nb++)
		{
			if(GlobalV::test_pp>1) ModuleBase::GlobalFunc::OUT("ib",nb);
			for (ig = 0;ig < npw;ig++)
			{
				const FPTYPE gnorm = gk[ig].norm() * GlobalC::ucell.tpiba;

				//cout << "\n gk[ig] = " << gk[ig].x << " " << gk[ig].y << " " << gk[ig].z;
				//cout << "\n gk.norm = " << gnorm;

				vq [ig] = ModuleBase::PolyInt::Polynomial_Interpolation(
						GlobalC::ppcell.tab, it, nb, GlobalV::NQX, GlobalV::DQ, gnorm );

			} // enddo

			// add spherical harmonic part
			for (ih = 0;ih < nh;ih++)
			{
				if (nb == GlobalC::ppcell.indv(it, ih))
				{
					const int lm = static_cast<int>( GlobalC::ppcell.nhtolm(it, ih) );
					for (ig = 0;ig < npw;ig++)
					{
						vkb1(ih, ig) = dylm(lm, ig) * vq [ig];
					}

				}

			} // end ih

		} // end nbeta

		// vkb1 contains all betas including angular part for type nt
		// now add the structure factor and factor (-i)^l
		for (ia=0; ia<GlobalC::ucell.atoms[it].na; ia++)
		{
			std::complex<FPTYPE> *sk = GlobalC::wf.get_sk(ik, it, ia,GlobalC::wfcpw);
			for (ih = 0;ih < nh;ih++)
			{
				std::complex<FPTYPE> pref = pow( ModuleBase::NEG_IMAG_UNIT, GlobalC::ppcell.nhtol(it, ih));      //?
				for (ig = 0;ig < npw;ig++)
				{
					vkb(jkb, ig) = vkb1(ih, ig) * sk [ig] * pref;
				}
				++jkb;
			} // end ih
		delete [] sk;
		} // end ia
	} // enddo
	delete [] gk;
	delete [] vq;
	return;
}//end get_dvnl1

template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::get_dvnl2(ModuleBase::ComplexMatrix &vkb,
		const int ik)
{
	if(GlobalV::test_pp) ModuleBase::TITLE("Stress","get_dvnl2");
//	ModuleBase::timer::tick("Stress","get_dvnl2");

	const int lmaxkb = GlobalC::ppcell.lmaxkb;
	if(lmaxkb < 0)
	{
		return;
	}

	const int npw = GlobalC::kv.ngk[ik];
	const int nhm = GlobalC::ppcell.nhm;
	int ig, ia, nb, ih;
	ModuleBase::matrix vkb1(nhm, npw);
	FPTYPE *vq = new FPTYPE[npw];
	const int x1= (lmaxkb + 1)*(lmaxkb + 1);

	ModuleBase::matrix ylm(x1, npw);
	ModuleBase::Vector3<FPTYPE> *gk = new ModuleBase::Vector3<FPTYPE>[npw];
	for (ig = 0;ig < npw;ig++)
	{
		gk[ig] = GlobalC::wf.get_1qvec_cartesian(ik, ig);
	}
	ModuleBase::YlmReal::Ylm_Real(x1, npw, gk, ylm);

	int jkb = 0;
	for(int it = 0;it < GlobalC::ucell.ntype;it++)
	{
		if(GlobalV::test_pp>1) ModuleBase::GlobalFunc::OUT("it",it);
		// calculate beta in G-space using an interpolation table
		const int nbeta = GlobalC::ucell.atoms[it].ncpp.nbeta;
		const int nh = GlobalC::ucell.atoms[it].ncpp.nh;

		if(GlobalV::test_pp>1) ModuleBase::GlobalFunc::OUT("nbeta",nbeta);

		for (nb = 0;nb < nbeta;nb++)
		{
			if(GlobalV::test_pp>1) ModuleBase::GlobalFunc::OUT("ib",nb);
			for (ig = 0;ig < npw;ig++)
			{
				const FPTYPE gnorm = gk[ig].norm() * GlobalC::ucell.tpiba;
	//cout << "\n gk[ig] = " << gk[ig].x << " " << gk[ig].y << " " << gk[ig].z;
	//cout << "\n gk.norm = " << gnorm;
				vq [ig] = Polynomial_Interpolation_nl(
						GlobalC::ppcell.tab, it, nb, GlobalV::DQ, gnorm );

			} // enddo

							// add spherical harmonic part
			for (ih = 0;ih < nh;ih++)
			{
				if (nb == GlobalC::ppcell.indv(it, ih))
				{
					const int lm = static_cast<int>( GlobalC::ppcell.nhtolm(it, ih) );
					for (ig = 0;ig < npw;ig++)
					{
						vkb1(ih, ig) = ylm(lm, ig) * vq [ig];
					}
				}
			} // end ih
		} // end nbeta

		// vkb1 contains all betas including angular part for type nt
		// now add the structure factor and factor (-i)^l
		for (ia=0; ia<GlobalC::ucell.atoms[it].na; ia++)
		{
			std::complex<FPTYPE> *sk = GlobalC::wf.get_sk(ik, it, ia,GlobalC::wfcpw);
			for (ih = 0;ih < nh;ih++)
			{
				std::complex<FPTYPE> pref = pow( ModuleBase::NEG_IMAG_UNIT, GlobalC::ppcell.nhtol(it, ih));      //?
				for (ig = 0;ig < npw;ig++)
				{
					vkb(jkb, ig) = vkb1(ih, ig) * sk [ig] * pref;
				}
			++jkb;
			} // end ih
			delete [] sk;
		} // end ia
	}	 // enddo

	delete [] gk;
	delete [] vq;
//	ModuleBase::timer::tick("Stress","get_dvnl2");

	return;
}


template <typename FPTYPE, typename Device>
FPTYPE Stress_Func<FPTYPE, Device>::Polynomial_Interpolation_nl
(
    const ModuleBase::realArray &table,
    const int &dim1,
    const int &dim2,
    const FPTYPE &table_interval,
    const FPTYPE &x                             // input value
)
{

	assert(table_interval>0.0);
	const FPTYPE position = x  / table_interval;
	const int iq = static_cast<int>(position);

	const FPTYPE x0 = position - static_cast<FPTYPE>(iq);
	const FPTYPE x1 = 1.0 - x0;
	const FPTYPE x2 = 2.0 - x0;
	const FPTYPE x3 = 3.0 - x0;
	const FPTYPE y=
			( table(dim1, dim2, iq)   * (-x2*x3-x1*x3-x1*x2) / 6.0 +
			table(dim1, dim2, iq+1) * (+x2*x3-x0*x3-x0*x2) / 2.0 -
			table(dim1, dim2, iq+2) * (+x1*x3-x0*x3-x0*x1) / 2.0 +
			table(dim1, dim2, iq+3) * (+x1*x2-x0*x2-x0*x1) / 6.0 )/table_interval ;


	return y;
}

template <typename FPTYPE, typename Device>
void Stress_Func<FPTYPE, Device>::dylmr2 (
	const int nylm,
	const int ngy,
	ModuleBase::Vector3<FPTYPE> *gk,
	ModuleBase::matrix &dylm,
	const int ipol)
{
  //-----------------------------------------------------------------------
  //
  //     compute \partial Y_lm(G) \over \partial (G)_ipol
  //     using simple numerical derivation (SdG)
  //     The spherical harmonics are calculated in ylmr2
  //
  //int nylm, ngy, ipol;
  // number of spherical harmonics
  // the number of g vectors to compute
  // desired polarization
  //FPTYPE g (3, ngy), gg (ngy), dylm (ngy, nylm)
  // the coordinates of g vectors
  // the moduli of g vectors
  // the spherical harmonics derivatives
  //
	int ig, lm;
	// counter on g vectors
	// counter on l,m component

	const FPTYPE delta = 1e-6;
	FPTYPE *dg, *dgi;

	ModuleBase::matrix ylmaux;
	// dg is the finite increment for numerical derivation:
	// dg = delta |G| = delta * sqrt(gg)
	// dgi= 1 /(delta * sqrt(gg))
	// gx = g +/- dg


	ModuleBase::Vector3<FPTYPE> *gx = new ModuleBase::Vector3<FPTYPE> [ngy];
	 

	dg = new FPTYPE [ngy];
	dgi = new FPTYPE [ngy];

	ylmaux.create (nylm, ngy);

	dylm.zero_out();
	ylmaux.zero_out();

	for( ig = 0;ig< ngy;ig++){
		gx[ig] = gk[ig];
	}
	//$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ig)
	for( ig = 0;ig< ngy;ig++){
		dg [ig] = delta * gx[ig].norm() ;
		if (gx[ig].norm2() > 1e-9) {
			dgi [ig] = 1.0 / dg [ig];
		}
		else{
			dgi [ig] = 0.0;
		}
	}
	//$OMP END PARALLEL DO

	//$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ig)
	for( ig = 0;ig< ngy;ig++)
	{
		if(ipol==0)
			gx [ig].x = gk[ ig].x + dg [ig];
		else if(ipol==1)
			gx [ig].y = gk [ ig].y + dg [ig];
		else if(ipol==2)
			gx [ig].z = gk [ ig].z + dg [ig];
	}
	//$OMP END PARALLEL DO

	ModuleBase::YlmReal::Ylm_Real(nylm, ngy, gx, dylm);
	//$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ig)
	for(ig = 0;ig< ngy;ig++)
	{
		if(ipol==0)
			gx [ig].x = gk [ ig].x - dg [ig];
		else if(ipol==1)
			gx [ig].y = gk [ ig].y - dg [ig];
		else if(ipol==2)
			gx [ig].z = gk [ ig].z - dg [ig];
	}
	//$OMP END PARALLEL DO

	ModuleBase::YlmReal::Ylm_Real(nylm, ngy, gx, ylmaux);


	//  zaxpy ( - 1.0, ylmaux, 1, dylm, 1);
	for( lm = 0;lm< nylm;lm++)
	{
		for(ig = 0;ig< ngy;ig++)
		{
			dylm (lm,ig) = dylm(lm,ig) - ylmaux(lm,ig);
		}
	}


	for( lm = 0;lm< nylm;lm++)
	{
		//$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ig)
		for(ig = 0;ig< ngy;ig++)
		{
			dylm (lm,ig) = dylm(lm,ig) * 0.5 * dgi [ig];
		}
		//$OMP END PARALLEL DO
	}
	delete[] gx;
	delete[] dg;
	delete[] dgi;

	return;
}

template class Stress_Func<double, psi::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class Stress_Func<double, psi::DEVICE_GPU>;
#endif