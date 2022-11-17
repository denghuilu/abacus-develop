#include "diago_iter_assist.h"

#include "module_base/blas_connector.h"
#include "module_base/complexmatrix.h"
#include "module_base/constants.h"
#include "module_base/global_variable.h"
#include "module_base/lapack_connector.h"
#include "module_base/timer.h"
#include "src_parallel/parallel_reduce.h"
#include "module_hsolver/include/math_kernel.h"
#include "module_hsolver/include/dngvd_op.h"
#include "module_psi/include/device.h"

using namespace hsolver;

template<typename FPTYPE, typename Device>
FPTYPE DiagoIterAssist<FPTYPE, Device>::avg_iter = 0.0;

template<typename FPTYPE, typename Device>
int DiagoIterAssist<FPTYPE, Device>::PW_DIAG_NMAX = 30;

template<typename FPTYPE, typename Device>
FPTYPE DiagoIterAssist<FPTYPE, Device>::PW_DIAG_THR = 1.0e-2;

template<typename FPTYPE, typename Device>
bool DiagoIterAssist<FPTYPE, Device>::need_subspace = false;

//----------------------------------------------------------------------
// Hamiltonian diagonalization in the subspace spanned
// by nstart states psi (atomic or random wavefunctions).
// Produces on output n_band eigenvectors (n_band <= nstart) in evc.
//----------------------------------------------------------------------
template<typename FPTYPE, typename Device>
void DiagoIterAssist<FPTYPE, Device>::diagH_subspace(
    hamilt::Hamilt<FPTYPE, Device>* pHamilt,
    const psi::Psi<std::complex<FPTYPE>, Device> &psi,
    psi::Psi<std::complex<FPTYPE>, Device> &evc,
    FPTYPE *en,
    int n_band)
{
    ModuleBase::TITLE("DiagoIterAssist", "diagH_subspace");
    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace");

    // two case:
    // 1. pw base: nstart = n_band, psi(nbands * npwx)
    // 2. lcao_in_pw base: nstart >= n_band, psi(NLOCAL * npwx)
    const int nstart = psi.get_nbands();
    if (n_band == 0)
        n_band = nstart;
    assert(n_band <= nstart);

    Device* ctx = {};

    std::complex<FPTYPE>* hcc = nullptr;
    std::complex<FPTYPE>* scc = nullptr;
    std::complex<FPTYPE>* vcc = nullptr;
    psi::memory::resize_memory_op<std::complex<FPTYPE>, Device>()(ctx, hcc, nstart * nstart);
    psi::memory::resize_memory_op<std::complex<FPTYPE>, Device>()(ctx, scc, nstart * nstart);
    psi::memory::resize_memory_op<std::complex<FPTYPE>, Device>()(ctx, vcc, nstart * nstart);
    psi::memory::set_memory_op<std::complex<FPTYPE>, Device>()(ctx, hcc, 0, nstart * nstart);
    psi::memory::set_memory_op<std::complex<FPTYPE>, Device>()(ctx, scc, 0, nstart * nstart);
    psi::memory::set_memory_op<std::complex<FPTYPE>, Device>()(ctx, vcc, 0, nstart * nstart);

    const int dmin = psi.get_current_nbas();
    const int dmax = psi.get_nbasis();

    // qianrui improve this part 2021-3-14
    const std::complex<FPTYPE>* ppsi = psi.get_pointer();

    // allocated hpsi 
    // std::vector<std::complex<FPTYPE>> hpsi(psi.get_nbands() * psi.get_nbasis());
    std::complex<FPTYPE>* hphi = nullptr;
    psi::memory::resize_memory_op<std::complex<FPTYPE>, Device>()(ctx, hphi, psi.get_nbands() * psi.get_nbasis());
    psi::memory::set_memory_op<std::complex<FPTYPE>, Device>()(ctx, hphi, 0, psi.get_nbands() * psi.get_nbasis());
    // do hPsi for all bands
    psi::Range all_bands_range(1, psi.get_current_k(), 0, psi.get_nbands()-1);
    hpsi_info hpsi_in(&psi, all_bands_range, hphi);
    pHamilt->ops->hPsi(hpsi_in);

    gemm_op<FPTYPE, Device>()(
        ctx,
        'C',
        'N',
        nstart,
        nstart,
        dmin,
        &ModuleBase::ONE,
        ppsi,
        dmax,
        hphi,
        dmax,
        &ModuleBase::ZERO,
        hcc,
        nstart
    );
    matrixTranspose_op<FPTYPE, Device>()(
        ctx,
        nstart,
        nstart,
        hcc,
        hcc
    );

    gemm_op<FPTYPE, Device>()(
        ctx,
        'C',
        'N',
        nstart,
        nstart,
        dmin,
        &ModuleBase::ONE,
        ppsi,
        dmax,
        ppsi,
        dmax,
        &ModuleBase::ZERO,
        scc,
        nstart
    );
    matrixTranspose_op<FPTYPE, Device>()(
        ctx,
        nstart,
        nstart,
        scc,
        scc
    );

    if (GlobalV::NPROC_IN_POOL > 1)
    {
        Parallel_Reduce::reduce_complex_double_pool(hcc, nstart * nstart);
        Parallel_Reduce::reduce_complex_double_pool(scc, nstart * nstart);
    }

    // after generation of H and S matrix, diag them
    DiagoIterAssist::diagH_LAPACK(nstart, n_band, hcc, scc, nstart, en, vcc);

    //=======================
    // diagonize the H-matrix
    //=======================
    if ((GlobalV::BASIS_TYPE == "lcao" || GlobalV::BASIS_TYPE == "lcao_in_pw") && GlobalV::CALCULATION == "nscf")
    {
        GlobalV::ofs_running << " Not do zgemm to get evc." << std::endl;
    }
    else if ((GlobalV::BASIS_TYPE == "lcao" || GlobalV::BASIS_TYPE == "lcao_in_pw")
             && (GlobalV::CALCULATION == "scf" || GlobalV::CALCULATION == "md"
                 || GlobalV::CALCULATION == "relax")) // pengfei 2014-10-13
    {
        // because psi and evc are different here,
        // I think if psi and evc are the same,
        // there may be problems, mohan 2011-01-01
        gemm_op<FPTYPE, Device>()(
            ctx,
            'N',
            'T',
            dmax,
            n_band,
            nstart,
            &ModuleBase::ONE,
            ppsi,
            dmax,
            vcc,
            n_band,
            &ModuleBase::ZERO,
            evc.get_pointer(),
            dmax
        );
    }
    else
    {
        // As the evc and psi may refer to the same matrix, we first
        // create a temporary matrix to store the result. (by wangjp)
        // qianrui improve this part 2021-3-13
        std::complex<FPTYPE>* evctemp = nullptr;
        psi::memory::resize_memory_op<std::complex<FPTYPE>, Device>()(ctx, evctemp, n_band * dmin);
        psi::memory::set_memory_op<std::complex<FPTYPE>, Device>()(ctx, evctemp, 0, n_band * dmin);
        
        gemm_op<FPTYPE, Device>()(
            ctx,
            'N',
            'T',
            dmin,
            n_band,
            nstart,
            &ModuleBase::ONE,
            ppsi,
            dmax,
            vcc,
            n_band,
            &ModuleBase::ZERO,
            evctemp,
            dmin
        );

        matrixSetToAnother<FPTYPE, Device>()(ctx, n_band, evctemp, dmin, evc.get_pointer(), dmax);
        // for (int ib = 0; ib < n_band; ib++)
        // {
        //     for (int ig = 0; ig < dmin; ig++)
        //     {
                // evc(ib, ig) = evctmp(ib, ig);
        //     }
        // }
        psi::memory::delete_memory_op<std::complex<double>, Device>()(ctx, evctemp);
    }

    psi::memory::delete_memory_op<std::complex<FPTYPE>, Device>()(ctx, hcc);
    psi::memory::delete_memory_op<std::complex<FPTYPE>, Device>()(ctx, scc);
    psi::memory::delete_memory_op<std::complex<FPTYPE>, Device>()(ctx, vcc);
    psi::memory::delete_memory_op<std::complex<FPTYPE>, Device>()(ctx, hphi);

    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace");
    return;
}

template<typename FPTYPE, typename Device>
void DiagoIterAssist<FPTYPE, Device>::diagH_subspace_init(
    hamilt::Hamilt<FPTYPE, Device>* pHamilt,
    const ModuleBase::ComplexMatrix &psi,
    psi::Psi<std::complex<FPTYPE>, Device> &evc,
    FPTYPE *en)
{
    ModuleBase::TITLE("DiagoIterAssist", "diagH_subspace_init");
    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace");

    // two case:
    // 1. pw base: nstart = n_band, psi(nbands * npwx)
    // 2. lcao_in_pw base: nstart >= n_band, psi(NLOCAL * npwx)
    const int nstart = psi.nr;
    const int n_band = evc.get_nbands();

    ModuleBase::ComplexMatrix hc(nstart, nstart);
    ModuleBase::ComplexMatrix sc(nstart, nstart);
    ModuleBase::ComplexMatrix hvec(nstart, n_band);

    const int dmin = evc.get_current_nbas();
    const int dmax = evc.get_nbasis();

    // qianrui improve this part 2021-3-14
    //std::complex<double> *aux = new std::complex<double>[dmax * nstart];
    //const std::complex<double> *paux = aux;
    psi::Psi<std::complex<FPTYPE>, Device> psi_temp(1, nstart, psi.nc, &evc.get_ngk(0));
    ModuleBase::GlobalFunc::COPYARRAY(psi.c, psi_temp.get_pointer(), psi_temp.size());
    const std::complex<FPTYPE> *ppsi = psi_temp.get_pointer();

    //allocated hpsi 
    std::vector<std::complex<FPTYPE>> hpsi(psi_temp.get_nbands() * psi_temp.get_nbasis());
    //do hPsi for all bands
    psi::Range all_bands_range(1, psi_temp.get_current_k(), 0, psi_temp.get_nbands()-1);
    hpsi_info hpsi_in(&psi_temp, all_bands_range, hpsi.data());
    pHamilt->ops->hPsi(hpsi_in);
    //use aux as a data pointer for hpsi
    const std::complex<FPTYPE> *aux = hpsi.data();

    char trans1 = 'C';
    char trans2 = 'N';
    zgemm_(&trans1,
           &trans2,
           &nstart,
           &nstart,
           &dmin,
           &ModuleBase::ONE,
           ppsi,
           &dmax,
           aux,
           &dmax,
           &ModuleBase::ZERO,
           hc.c,
           &nstart);
    hc = transpose(hc, false);

    zgemm_(&trans1,
           &trans2,
           &nstart,
           &nstart,
           &dmin,
           &ModuleBase::ONE,
           ppsi,
           &dmax,
           ppsi,
           &dmax,
           &ModuleBase::ZERO,
           sc.c,
           &nstart);
    sc = transpose(sc, false);

    if (GlobalV::NPROC_IN_POOL > 1)
    {
        Parallel_Reduce::reduce_complex_double_pool(hc.c, nstart * nstart);
        Parallel_Reduce::reduce_complex_double_pool(sc.c, nstart * nstart);
    }

    // after generation of H and S matrix, diag them
    ///this part only for test, eigenvector would have different phase caused by micro numerical perturbation
    ///set 8 bit effective accuracy would help for debugging
    /*for(int i=0;i<nstart;i++)
    {
        for(int j=0;j<nstart;j++)
        {
            if(std::norm(hc(i,j))<1e-10) hc(i,j) = ModuleBase::ZERO;
            else hc(i,j) = std::complex<double>(double(int(hc(i,j).real()*100000000))/100000000, 0);
            if(std::norm(sc(i,j))<1e-10) sc(i,j) = ModuleBase::ZERO;
            else sc(i,j) = std::complex<double>(double(int(sc(i,j).real()*100000000))/100000000, 0);
        }
    }*/
    DiagoIterAssist::diagH_LAPACK(nstart, n_band, hc, sc, nstart, en, hvec);

    //=======================
    // diagonize the H-matrix
    //=======================
    if ((GlobalV::BASIS_TYPE == "lcao" || GlobalV::BASIS_TYPE == "lcao_in_pw") && GlobalV::CALCULATION == "nscf")
    {
        GlobalV::ofs_running << " Not do zgemm to get evc." << std::endl;
    }
    else if ((GlobalV::BASIS_TYPE == "lcao" || GlobalV::BASIS_TYPE == "lcao_in_pw")
             && (GlobalV::CALCULATION == "scf" || GlobalV::CALCULATION == "md"
                 || GlobalV::CALCULATION == "relax")) // pengfei 2014-10-13
    {
        // because psi and evc are different here,
        // I think if psi and evc are the same,
        // there may be problems, mohan 2011-01-01
        char transa = 'N';
        char transb = 'T';
        zgemm_(&transa,
               &transb,
               &dmax, // m: row of A,C
               &n_band, // n: col of B,C
               &nstart, // k: col of A, row of B
               &ModuleBase::ONE, // alpha
               ppsi, // A
               &dmax, // LDA: if(N) max(1,m) if(T) max(1,k)
               hvec.c, // B
               &n_band, // LDB: if(N) max(1,k) if(T) max(1,n)
               &ModuleBase::ZERO, // belta
               evc.get_pointer(), // C
               &dmax); // LDC: if(N) max(1, m)
    }
    else
    {
        // As the evc and psi may refer to the same matrix, we first
        // create a temporary matrix to store the result. (by wangjp)
        // qianrui improve this part 2021-3-13
        char transa = 'N';
        char transb = 'T';
        ModuleBase::ComplexMatrix evctmp(n_band, dmin, false);
        zgemm_(&transa,
               &transb,
               &dmin,
               &n_band,
               &nstart,
               &ModuleBase::ONE,
               ppsi,
               &dmax,
               hvec.c,
               &n_band,
               &ModuleBase::ZERO,
               evctmp.c,
               &dmin);
        for (int ib = 0; ib < n_band; ib++)
        {
            for (int ig = 0; ig < dmin; ig++)
            {
                evc(ib, ig) = evctmp(ib, ig);
            }
        }
    }

    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace");
    return;
}

template<typename FPTYPE, typename Device>
void DiagoIterAssist<FPTYPE, Device>::diagH_subspace_init(
    hamilt::Hamilt<FPTYPE, Device>* pHamilt,
    const std::complex<FPTYPE>* psi,
    int psi_nr,
    int psi_nc,
    psi::Psi<std::complex<FPTYPE>, Device> &evc,
    FPTYPE *en)
{
    ModuleBase::TITLE("DiagoIterAssist", "diagH_subspace_init");
    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace");

    // two case:
    // 1. pw base: nstart = n_band, psi(nbands * npwx)
    // 2. lcao_in_pw base: nstart >= n_band, psi(NLOCAL * npwx)
    
    const int nstart = psi_nr;
    const int n_band = evc.get_nbands();

    Device* ctx = {};

    // ModuleBase::ComplexMatrix hc(nstart, nstart);
    // ModuleBase::ComplexMatrix sc(nstart, nstart);
    // ModuleBase::ComplexMatrix hvec(nstart, n_band);
    std::complex<FPTYPE>* hcc = nullptr;
    std::complex<FPTYPE>* scc = nullptr;
    std::complex<FPTYPE>* vcc = nullptr;
    psi::memory::resize_memory_op<std::complex<FPTYPE>, Device>()(ctx, hcc, nstart * nstart);
    psi::memory::resize_memory_op<std::complex<FPTYPE>, Device>()(ctx, scc, nstart * nstart);
    psi::memory::resize_memory_op<std::complex<FPTYPE>, Device>()(ctx, vcc, nstart * nstart);
    psi::memory::set_memory_op<std::complex<FPTYPE>, Device>()(ctx, hcc, 0, nstart * nstart);
    psi::memory::set_memory_op<std::complex<FPTYPE>, Device>()(ctx, scc, 0, nstart * nstart);
    psi::memory::set_memory_op<std::complex<FPTYPE>, Device>()(ctx, vcc, 0, nstart * nstart);

    const int dmin = evc.get_current_nbas();
    const int dmax = evc.get_nbasis();

    // qianrui improve this part 2021-3-14
    // std::complex<double> *aux = new std::complex<double>[dmax * nstart];
    // const std::complex<double> *paux = aux;
    psi::Psi<std::complex<FPTYPE>, Device> psi_temp(1, nstart, psi_nc, &evc.get_ngk(0));
    psi::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, Device>()(
        ctx,
        ctx, 
        psi_temp.get_pointer(), 
        psi, 
        psi_temp.size()
    );
    // ModuleBase::GlobalFunc::COPYARRAY(psi, psi_temp.get_pointer(), psi_temp.size());

    const std::complex<FPTYPE> *ppsi = psi_temp.get_pointer();

    // allocated hpsi 
    std::complex<FPTYPE>* hpsi = nullptr;
    psi::memory::resize_memory_op<std::complex<FPTYPE>, Device>()(ctx, hpsi, psi_temp.get_nbands() * psi_temp.get_nbasis());
    psi::memory::set_memory_op<std::complex<FPTYPE>, Device>()(ctx, hpsi, 0, psi_temp.get_nbands() * psi_temp.get_nbasis());
    // ================================================
    // std::vector<std::complex<FPTYPE>> hpsi(psi_temp.get_nbands() * psi_temp.get_nbasis());


    // do hPsi for all bands
    psi::Range all_bands_range(1, psi_temp.get_current_k(), 0, psi_temp.get_nbands()-1);
    hpsi_info hpsi_in(&psi_temp, all_bands_range, hpsi);
    pHamilt->ops->hPsi(hpsi_in);

    gemm_op<FPTYPE, Device>()(
        ctx,
        'C',
        'N',
        nstart,
        nstart,
        dmin,
        &ModuleBase::ONE,
        ppsi,
        dmax,
        hpsi,
        dmax,
        &ModuleBase::ZERO,
        hcc,
        nstart
    );
    matrixTranspose_op<FPTYPE, Device>()(
        ctx,
        nstart,
        nstart,
        hcc,
        hcc
    );

    gemm_op<FPTYPE, Device>()(
        ctx,
        'C',
        'N',
        nstart,
        nstart,
        dmin,
        &ModuleBase::ONE,
        ppsi,
        dmax,
        ppsi,
        dmax,
        &ModuleBase::ZERO,
        scc,
        nstart
    );
    matrixTranspose_op<FPTYPE, Device>()(
        ctx,
        nstart,
        nstart,
        scc,
        scc
    );

    if (GlobalV::NPROC_IN_POOL > 1)
    {
        Parallel_Reduce::reduce_complex_double_pool(hcc, nstart * nstart);
        Parallel_Reduce::reduce_complex_double_pool(scc, nstart * nstart);
    }

    // after generation of H and S matrix, diag them
    ///this part only for test, eigenvector would have different phase caused by micro numerical perturbation
    ///set 8 bit effective accuracy would help for debugging
    /*for(int i=0;i<nstart;i++)
    {
        for(int j=0;j<nstart;j++)
        {
            if(std::norm(hc(i,j))<1e-10) hc(i,j) = ModuleBase::ZERO;
            else hc(i,j) = std::complex<double>(double(int(hc(i,j).real()*100000000))/100000000, 0);
            if(std::norm(sc(i,j))<1e-10) sc(i,j) = ModuleBase::ZERO;
            else sc(i,j) = std::complex<double>(double(int(sc(i,j).real()*100000000))/100000000, 0);
        }
    }*/

    DiagoIterAssist::diagH_LAPACK(nstart, n_band, hcc, scc, nstart, en, vcc);

    //=======================
    // diagonize the H-matrix
    //=======================
    if ((GlobalV::BASIS_TYPE == "lcao" || GlobalV::BASIS_TYPE == "lcao_in_pw") && GlobalV::CALCULATION == "nscf")
    {
        GlobalV::ofs_running << " Not do zgemm to get evc." << std::endl;
    }
    else if ((GlobalV::BASIS_TYPE == "lcao" || GlobalV::BASIS_TYPE == "lcao_in_pw")
             && (GlobalV::CALCULATION == "scf" || GlobalV::CALCULATION == "md"
                 || GlobalV::CALCULATION == "relax")) // pengfei 2014-10-13
    {
        // because psi and evc are different here,
        // I think if psi and evc are the same,
        // there may be problems, mohan 2011-01-01
        gemm_op<FPTYPE, Device>()(
            ctx,
            'N',
            'T',
            dmax,
            n_band,
            nstart,
            &ModuleBase::ONE,
            ppsi,
            dmax,
            vcc,
            n_band,
            &ModuleBase::ZERO,
            evc.get_pointer(),
            dmax
        );
    }
    else
    {
        // As the evc and psi may refer to the same matrix, we first
        // create a temporary matrix to store the result. (by wangjp)
        // qianrui improve this part 2021-3-13
        std::complex<FPTYPE>* evctemp = nullptr;
        psi::memory::resize_memory_op<std::complex<FPTYPE>, Device>()(ctx, evctemp, n_band * dmin);
        psi::memory::set_memory_op<std::complex<FPTYPE>, Device>()(ctx, evctemp, 0, n_band * dmin);
        
        gemm_op<FPTYPE, Device>()(
            ctx,
            'N',
            'T',
            dmin,
            n_band,
            nstart,
            &ModuleBase::ONE,
            ppsi,
            dmax,
            vcc,
            n_band,
            &ModuleBase::ZERO,
            evctemp,
            dmin
        );

        matrixSetToAnother<FPTYPE, Device>()(ctx, n_band, evctemp, dmin, evc.get_pointer(), dmax);

        psi::memory::delete_memory_op<std::complex<double>, Device>()(ctx, evctemp);
    }

    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace");
}

template<typename FPTYPE, typename Device>
void DiagoIterAssist<FPTYPE, Device>::diagH_LAPACK(
    const int nstart,
    const int nbands,
    const ModuleBase::ComplexMatrix &hc,
    const ModuleBase::ComplexMatrix &sc,
    const int ldh, // nstart
    FPTYPE *e,
    ModuleBase::ComplexMatrix &hvec)
{
    ModuleBase::TITLE("DiagoIterAssist", "diagH_LAPACK");
    ModuleBase::timer::tick("DiagoIterAssist", "diagH_LAPACK");

    int lwork = 0;

    ModuleBase::ComplexMatrix sdum(nstart, ldh);
    ModuleBase::ComplexMatrix hdum;

    sdum = sc;

    const bool all_eigenvalues = (nstart == nbands);

    // workspace query
    int nb = LapackConnector::ilaenv(1, "ZHETRD", "U", nstart, -1, -1, -1);

    if (nb < 1)
    {
        nb = std::max(1, nstart);
    }

    if (nb == 1 || nb >= nstart)
    {
        lwork = 2 * nstart; // mohan modify 2009-08-02
    }
    else
    {
        lwork = (nb + 1) * nstart;
    }

    std::complex<FPTYPE> *work = new std::complex<FPTYPE>[lwork];
    ModuleBase::GlobalFunc::ZEROS(work, lwork);

    //=====================================================================
    // input s and (see below) h are copied so that they are not destroyed
    //=====================================================================

    int info = 0;
    int rwork_dim;
    if (all_eigenvalues)
    {
        rwork_dim = 3 * nstart - 2;
    }
    else
    {
        rwork_dim = 7 * nstart;
    }

    FPTYPE *rwork = new FPTYPE[rwork_dim];
    ModuleBase::GlobalFunc::ZEROS(rwork, rwork_dim);

    if (all_eigenvalues)
    {
        //===========================
        // calculate all eigenvalues
        //===========================
        hvec = hc;
        LapackConnector::zhegv(1, 'V', 'U', nstart, hvec, ldh, sdum, ldh, e, work, lwork, rwork, info);
    }
    else
    {
        //=====================================
        // calculate only m lowest eigenvalues
        //=====================================
        int *iwork = new int[5 * nstart];
        int *ifail = new int[nstart];

        ModuleBase::GlobalFunc::ZEROS(rwork, 7 * nstart);
        ModuleBase::GlobalFunc::ZEROS(iwork, 5 * nstart);
        ModuleBase::GlobalFunc::ZEROS(ifail, nstart);

        hdum.create(nstart, ldh);
        hdum = hc;

        //=============================
        // Number of calculated bands
        //=============================
        int mm = nbands;

        LapackConnector::zhegvx(1, // INTEGER
                                'V', // CHARACTER*1
                                'I', // CHARACTER*1
                                'U', // CHARACTER*1
                                nstart, // INTEGER
                                hdum, // COMPLEX*16 array
                                ldh, // INTEGER
                                sdum, // COMPLEX*16 array
                                ldh, // INTEGER
                                0.0, // DOUBLE PRECISION
                                0.0, // DOUBLE PRECISION
                                1, // INTEGER
                                nbands, // INTEGER
                                0.0, // DOUBLE PRECISION
                                mm, // INTEGER
                                e, // DOUBLE PRECISION array
                                hvec, // COMPLEX*16 array
                                ldh, // INTEGER
                                work, // DOUBLE array, dimension (MAX(1,LWORK))
                                lwork, // INTEGER
                                rwork, // DOUBLE PRECISION array, dimension (7*N)
                                iwork, // INTEGER array, dimension (5*N)
                                ifail, // INTEGER array, dimension (N)
                                info // INTEGER
        );

        delete[] iwork;
        delete[] ifail;
    }
    delete[] rwork;
    delete[] work;

    ModuleBase::timer::tick("DiagoIterAssist", "diagH_LAPACK");
    return;
}


template<typename FPTYPE, typename Device>
void DiagoIterAssist<FPTYPE, Device>::diagH_LAPACK(
    const int nstart,
    const int nbands,
    const std::complex<FPTYPE>* hcc,
    const std::complex<FPTYPE>* scc,
    const int ldh, // nstart
    FPTYPE *e, // always in CPU
    std::complex<FPTYPE>* vcc)
{
    ModuleBase::TITLE("DiagoIterAssist", "LAPACK_subspace");
    ModuleBase::timer::tick("DiagoIterAssist", "LAPACK_subspace");

    const bool all_eigenvalues = (nstart == nbands);

    FPTYPE * res = e, *e_gpu = nullptr; 
#if ((defined __CUDA) || (defined __ROCM))
    if (psi::device::get_device_type<Device>(ctx) == psi::GpuDevice) {
        psi::memory::resize_memory_op<FPTYPE, psi::DEVICE_GPU>()(gpu_ctx, e_gpu, nbands);
        // set e in CPU value to e_gpu
        syncmem_var_h2d_op()(gpu_ctx, cpu_ctx, e_gpu, e, nbands);
        res = e_gpu;
    }
#endif

    if (all_eigenvalues) {
        //===========================
        // calculate all eigenvalues
        //===========================
        dngv_op<FPTYPE, Device>()(ctx, nstart, ldh, hcc, scc, res, vcc);
    }
    else {
        //=====================================
        // calculate only m lowest eigenvalues
        //=====================================
        dngvx_op<FPTYPE, Device>()(ctx, nstart, ldh, hcc, scc, nbands, res, vcc);
    }

#if ((defined __CUDA) || (defined __ROCM))
    if (psi::device::get_device_type<Device>(ctx) == psi::GpuDevice) {
        // set e_gpu value to e in CPU
        syncmem_var_d2h_op()(cpu_ctx, gpu_ctx, e, res, nbands);
        delmem_var_op()(gpu_ctx, e_gpu);
    }
#endif

    ModuleBase::timer::tick("DiagoIterAssist", "LAPACK_subspace");
}

template<typename FPTYPE, typename Device>
bool DiagoIterAssist<FPTYPE, Device>::test_exit_cond(const int &ntry, const int &notconv)
{
    //================================================================
    // If this logical function is true, need to do diagH_subspace
    // and cg again.
    //================================================================

    bool scf = true;
    if (GlobalV::CALCULATION == "nscf")
        scf = false;

    // If ntry <=5, try to do it better, if ntry > 5, exit.
    const bool f1 = (ntry <= 5);

    // In non-self consistent calculation, do until totally converged.
    const bool f2 = ((!scf && (notconv > 0)));

    // if self consistent calculation, if not converged > 5,
    // using diagH_subspace and cg method again. ntry++
    const bool f3 = ((scf && (notconv > 5)));
    return (f1 && (f2 || f3));
}

namespace hsolver {
template class DiagoIterAssist<double, psi::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoIterAssist<double, psi::DEVICE_GPU>;
#endif 
} // namespace hsolver