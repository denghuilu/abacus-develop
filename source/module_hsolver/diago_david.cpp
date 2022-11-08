#include "diago_david.h"

#include "diago_iter_assist.h"
#include "module_base/constants.h"
#include "module_base/blas_connector.h"
#include "module_base/lapack_connector.h"
#include "module_base/timer.h"
#include "src_parallel/parallel_reduce.h"
#include "src_parallel/parallel_common.h"
#include "module_hsolver/include/math_kernel.h"
#include "module_hsolver/include/dngvd_op.h"

using namespace hsolver;

template<typename FPTYPE, typename Device>
int DiagoDavid<FPTYPE, Device>::PW_DIAG_NDIM = 4;

template<typename FPTYPE, typename Device>
DiagoDavid<FPTYPE, Device>::DiagoDavid(const FPTYPE* precondition_in)
{
    this->precondition = precondition_in;

    test_david = 2;
    // 1: check which function is called and which step is executed
    // 2: check the eigenvalues of the result of each iteration
    // 3: check the eigenvalues and errors of the last result
    // default: no check
}

template<typename FPTYPE, typename Device>
DiagoDavid<FPTYPE, Device>::~DiagoDavid() {
    delete_memory_op()(this->ctx, this->hphi);
    delete_memory_op()(this->ctx, this->sphi);
    delete_memory_op()(this->ctx, this->hcc);
    delete_memory_op()(this->ctx, this->scc);
    delete_memory_op()(this->ctx, this->vcc);
    psi::memory::delete_memory_op<FPTYPE, Device>()(this->ctx, this->eigenvalue);
    // psi::memory::delete_memory_op<FPTYPE, Device>()(this->ctx, this->precondition);
}

template<typename FPTYPE, typename Device>
void DiagoDavid<FPTYPE, Device>::diag_mock(hamilt::Hamilt<double>* phm_in, psi::Psi<std::complex<FPTYPE>, Device> &psi, FPTYPE *eigenvalue_in)
{
    if (test_david == 1)
        ModuleBase::TITLE("DiagoDavid", "diag_mock");
    ModuleBase::timer::tick("DiagoDavid", "diag_mock");

    assert(DiagoDavid::PW_DIAG_NDIM > 1);
    assert(DiagoDavid::PW_DIAG_NDIM * psi.get_nbands() < psi.get_current_nbas() * GlobalV::NPROC_IN_POOL);
    // qianrui change it 2021-7-25.
    // In strictly speaking, it shoule be PW_DIAG_NDIM*nband < npw sum of all pools. We roughly estimate it here.
    // However, in most cases, total number of plane waves should be much larger than nband*PW_DIAG_NDIM


    /// initialize variables
    this->dim = psi.get_current_nbas();
    this->dmx = psi.get_nbasis();
    this->n_band = psi.get_nbands();
    this->nbase_x = DiagoDavid::PW_DIAG_NDIM * this->n_band; // maximum dimension of the reduced basis set


    psi::Psi<std::complex<FPTYPE>, Device> basis(1, this->nbase_x, this->dim, &(psi.get_ngk(0))); // the reduced basis set

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // ModuleBase::ComplexMatrix hp(nbase_x, this->dim); // the product of H and psi in the reduced basis set
    resize_memory_op()(this->ctx, this->hphi, this->nbase_x * this->dim);
    set_memory_op()(this->ctx, this->hphi, 0, this->nbase_x * this->dim);


    // ModuleBase::ComplexMatrix sp(nbase_x, this->dim); // the Product of S and psi in the reduced basis set
    resize_memory_op()(this->ctx, this->sphi, this->nbase_x * this->dim);
    set_memory_op()(this->ctx, this->sphi, 0, this->nbase_x * this->dim);


    // ModuleBase::ComplexMatrix hc(this->nbase_x, this->nbase_x); // Hamiltonian on the reduced basis
    resize_memory_op()(this->ctx, this->hcc, this->nbase_x * this->nbase_x);
    set_memory_op()(this->ctx, this->hcc, 0, this->nbase_x * this->nbase_x);

    // ModuleBase::ComplexMatrix sc(this->nbase_x, this->nbase_x); // Overlap on the reduced basis
    resize_memory_op()(this->ctx, this->scc, this->nbase_x * this->nbase_x);
    set_memory_op()(this->ctx, this->scc, 0, this->nbase_x * this->nbase_x);

    // ModuleBase::ComplexMatrix vc(this->nbase_x, this->nbase_x); // Eigenvectors of hc
    resize_memory_op()(this->ctx, this->vcc, this->nbase_x * this->nbase_x);
    set_memory_op()(this->ctx, this->vcc, 0, this->nbase_x * this->nbase_x);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    
    // the lowest N eigenvalues of hc
    psi::memory::resize_memory_op<FPTYPE, Device>()(this->ctx, this->eigenvalue, this->nbase_x);
    psi::memory::set_memory_op<FPTYPE, Device>()(this->ctx, this->eigenvalue, 0, this->nbase_x);

    // convflag[m] = true if the m th band is convergent
    std::vector<bool> convflag(this->n_band, false);
    // unconv[m] store the number of the m th unconvergent band
    std::vector<int> unconv(this->n_band);

    int nbase = 0; // the dimension of the reduced basis set
    
    this->notconv = this->n_band; // the number of the unconvergent bands
    
    for (int m = 0; m < this->n_band; m++)
        unconv[m] = m;

    ModuleBase::timer::tick("DiagoDavid", "first");
    
    // orthogonalise the initial trial psi(0~nband-1)
    
    // ModuleBase::ComplexMatrix lagrange_matrix(this->n_band, this->n_band);
    resize_memory_op()(this->ctx, this->lagrange_matrix, this->n_band * this->n_band);
    set_memory_op()(this->ctx, this->lagrange_matrix, 0, this->n_band * this->n_band);
    
    // plan for SchmitOrth
    std::vector<int> pre_matrix_mm_m(this->n_band, 0);
    std::vector<int> pre_matrix_mv_m(this->n_band, 1);
    this->planSchmitOrth(this->n_band, pre_matrix_mm_m.data(), pre_matrix_mv_m.data());
    
    
    for( int m = 0; m < this->n_band; m++)
    {
        phm_in->sPsi(&psi(m, 0), &this->sphi[m * this->dim],  (size_t)this->dim);
    }
    //begin SchmitOrth
    for (int m = 0; m < this->n_band; m++)
    {
        // psi_m = psi(m)
        // haozhihan replace 2022-10-23
        psi::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, Device>()(this->ctx, this->ctx, &basis(m, 0), &psi(m, 0), this->dim);
        // ModuleBase::GlobalFunc::COPYARRAY(&psi(m, 0), &basis(m, 0), this->dim);
        /*for (int ig = 0; ig < dim; ig++)
        {
            psi_m[ig] = psi(m, ig);
        }*/

        //phm_in->sPsi(psi_m.data(), spsi.data(),  (size_t)dim);
        this->SchmitOrth(
            this->dim,
            this->n_band,
            m,
            basis,
            this->sphi,
            &this->lagrange_matrix[m * this->n_band],
            pre_matrix_mm_m[m],
            pre_matrix_mv_m[m]
        );
        phm_in->sPsi(&basis(m, 0), &this->sphi[m * this->dim],  (size_t)this->dim);

        // basis(m) = psi_m, hp(m) = H |psi_m>, sp(m) = S |psi_m>
        //ModuleBase::GlobalFunc::COPYARRAY(psi_m.data(), &basis(m, 0), dim);
        //ModuleBase::GlobalFunc::COPYARRAY(spsi.data(), &sp(m, 0), dim);
        /*std::complex<double>* sp_p = &sp(m, 0);
        std::complex<double>* basis_p = &basis(m, 0);
        for (int ig = 0; ig < dim; ig++)
        {
            basis_p[ig] = psi_m[ig];
            //hp(m, ig) = hpsi[ig];
            sp_p[ig] = spsi[ig];
        }*/
    }
    //end of SchmitOrth and calculate H|psi>
    hpsi_info dav_hpsi_in(&basis, psi::Range(1, 0, 0, this->n_band-1), this->hphi);
    phm_in->ops->hPsi(dav_hpsi_in);

    this->cal_elem(this->dim, nbase, this->notconv, basis, this->hphi, this->sphi, this->hcc, this->scc);

    this->diag_zhegvx(nbase, this->n_band, this->hcc, this->scc, this->nbase_x, this->eigenvalue, this->vcc);

    for (int m = 0; m < this->n_band; m++)
    {
        eigenvalue_in[m] = this->eigenvalue[m];
    }

    ModuleBase::timer::tick("DiagoDavid", "first");

    int dav_iter = 0;
    do
    {
        dav_iter++;

        this->cal_grad(phm_in,
                       this->dim,
                       nbase,
                       this->notconv,
                       basis,
                       this->hphi,
                       this->sphi,
                       this->vcc,
                       unconv.data(),
                       this->eigenvalue);

        this->cal_elem(this->dim, nbase, this->notconv, basis, this->hphi, this->sphi, this->hcc, this->scc);

        this->diag_zhegvx(nbase, this->n_band, this->hcc, this->scc, this->nbase_x, this->eigenvalue, this->vcc);

        // check convergence and update eigenvalues
        ModuleBase::timer::tick("DiagoDavid", "check_update");

        this->notconv = 0;
        for (int m = 0; m < this->n_band; m++)
        {
            convflag[m] = (abs(this->eigenvalue[m] - eigenvalue_in[m]) < DiagoIterAssist<FPTYPE, Device>::PW_DIAG_THR);

            if (!convflag[m])
            {
                unconv[this->notconv] = m;
                this->notconv++;
            }

            eigenvalue_in[m] = this->eigenvalue[m];
        }

        ModuleBase::timer::tick("DiagoDavid", "check_update");
        if (!this->notconv || (nbase + this->notconv > this->nbase_x) || (dav_iter == DiagoIterAssist<FPTYPE, Device>::PW_DIAG_NMAX))
        {
            ModuleBase::timer::tick("DiagoDavid", "last");

            // updata eigenvectors of Hamiltonian
            // ModuleBase::GlobalFunc::ZEROS(psi.get_pointer(), psi.get_nbands() * psi.get_nbasis());
            set_memory_op()(this->ctx, psi.get_pointer(), 0, psi.get_nbands() * psi.get_nbasis());
            //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            // haozhihan repalce 2022-10-18
            gemm_op<FPTYPE, Device>()(
                this->ctx,
                'N',
                'T',
                this->dim, // m: row of A,C
                this->n_band, // n: col of B,C
                nbase, // k: col of A, row of B
                &ModuleBase::ONE, // alpha
                basis.get_pointer(), // A
                basis.get_nbasis(), // LDA: if(N) max(1,m) if(T) max(1,k)
                this->vcc, // B
                this->nbase_x, // LDB: if(N) max(1,k) if(T) max(1,n)
                &ModuleBase::ZERO, // belta
                psi.get_pointer(), // C
                psi.get_nbasis() // LDC: if(N) max(1, m)
            );
            //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            // zgemm_(&transa,
            //         &transb,
            //         &dim, // m: row of A,C
            //         &nband, // n: col of B,C
            //         &nbase, // k: col of A, row of B
            //         &ModuleBase::ONE, // alpha
            //         basis.get_pointer(), // A
            //         &basis.get_nbasis(), // LDA: if(N) max(1,m) if(T) max(1,k)
            //         vc.c, // B
            //         &nbase_x, // LDB: if(N) max(1,k) if(T) max(1,n)
            //         &ModuleBase::ZERO, // belta
            //         psi.get_pointer(), // C
            //         &psi.get_nbasis()); // LDC: if(N) max(1, m)
            //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            /*for (int m = 0; m < nband; m++)
            {
                for (int j = 0; j < nbase; j++)
                {
                    for (int ig = 0; ig < dim; ig++)
                    {
                        psi(m, ig) += vc(j, m) * basis(j, ig);
                    }
                }
            }*/

            if (!this->notconv || (dav_iter == DiagoIterAssist<FPTYPE, Device>::PW_DIAG_NMAX))
            {
                // overall convergence or last iteration: exit the iteration

                ModuleBase::timer::tick("DiagoDavid", "last");
                break;
            }
            else
            {
                // if the dimension of the reduced basis set is becoming too large,
                // then replace the first N (=nband) basis vectors with the current
                // estimate of the eigenvectors and set the basis dimension to N;

                this->refresh(this->dim, this->n_band, nbase, eigenvalue_in, psi, basis, this->hphi, this->sphi, this->hcc, this->scc, this->vcc);
                ModuleBase::timer::tick("DiagoDavid", "last");
            }

        } // end of if

    } while (1);

    DiagoIterAssist<FPTYPE, Device>::avg_iter += static_cast<double>(dav_iter);

    ModuleBase::timer::tick("DiagoDavid", "diag_mock");

    return;
}




template<typename FPTYPE, typename Device>
void DiagoDavid<FPTYPE, Device>::cal_grad(hamilt::Hamilt<double>* phm_in,
                          const int &npw,
                          const int &nbase, // current dimension of the reduced basis
                          const int &notconv,
                          psi::Psi<std::complex<FPTYPE>, Device>& basis,
                          std::complex<FPTYPE>* hphi,
                          std::complex<FPTYPE>* sphi,
                          const std::complex<FPTYPE>* vcc,
                          const int *unconv,
                          const FPTYPE *eigenvalue)
{
    if (test_david == 1)
        ModuleBase::TITLE("DiagoDavid", "cal_grad");
    if(notconv == 0) return;
    ModuleBase::timer::tick("DiagoDavid", "cal_grad");

    //use template pointer for accelerate
    std::complex<double> *spsi;
    std::complex<double> *ppsi;

    // expand the reduced basis set with the new basis vectors P|R(psi)>...
    // in which psi are the last eigenvectors
    // we define |R(psi)> as (H-ES)*|Psi>, E = <psi|H|psi>/<psi|S|psi>

    // ModuleBase::ComplexMatrix vc_ev_vector(notconv, nbase);
    std::complex<FPTYPE>* vc_ev_vector = nullptr;
    resize_memory_op()(this->ctx, vc_ev_vector, notconv * nbase);
    set_memory_op()(this->ctx, vc_ev_vector, 0, notconv * nbase);

    for (int m = 0; m < notconv; m++)
    {
        for(int i = 0; i < nbase; i++)
        {
            // vc_ev_vector(m, i) = vc(i, unconv[m]);
            vc_ev_vector[m * nbase + i] = vcc[i * this->nbase_x + unconv[m]];
        }
    }
    ppsi = &basis(nbase, 0);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan repalce 2022-10-18
    gemm_op<FPTYPE, Device>()(
            this->ctx,
            'N',
            'N',
            npw, // m: row of A,C
            notconv, // n: col of B,C
            nbase, // k: col of A, row of B
            &ModuleBase::ONE, // alpha
            hphi, // A npw * nbase
            this->dim, // LDA: if(N) max(1,m) if(T) max(1,k)
            vc_ev_vector, // B nbase * notconv
            nbase, // LDB: if(N) max(1,k) if(T) max(1,n)
            &ModuleBase::ZERO, // belta
            ppsi, // C npw * notconv
            basis.get_nbasis() // LDC: if(N) max(1, m)
    );
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // zgemm_(&trans,
    //         &transb,
    //         &npw, // m: row of A,C
    //         &notconv, // n: col of B,C
    //         &nbase, // k: col of A, row of B
    //         &ModuleBase::ONE, // alpha
    //         hp.c, // A
    //         &hp.nc, // LDA: if(N) max(1,m) if(T) max(1,k)
    //         vc_ev_vector.c, // B
    //         &vc_ev_vector.nc, // LDB: if(N) max(1,k) if(T) max(1,n)
    //         &ModuleBase::ZERO, // belta
    //         ppsi, // C
    //         &basis.get_nbasis()); // LDC: if(N) max(1, m)
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    /*zgemv_(&trans,
        &npw,
        &nbase,
        &ModuleBase::ONE,
        hp.c,
        &hp.nc,
        vc_ev_vector.data(),
        &inc,
        &ModuleBase::ZERO,
        respsi,
        &inc);*/
    for (int m = 0; m < notconv; m++)
    {
        for(int i = 0; i < nbase; i++)
        {
            vc_ev_vector[m * nbase + i] *= -1 * this->eigenvalue[unconv[m]];
        }
    }

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan repalce 2022-10-18
    gemm_op<FPTYPE, Device>()(
        this->ctx,
        'N',
        'N',
        npw, // m: row of A,C
        notconv, // n: col of B,C
        nbase, // k: col of A, row of B
        &ModuleBase::ONE, // alpha
        sphi, // A
        this->dim, // LDA: if(N) max(1,m) if(T) max(1,k)
        vc_ev_vector, // B
        nbase, // LDB: if(N) max(1,k) if(T) max(1,n)
        &ModuleBase::ONE, // belta
        ppsi, // C npw * notconv
        basis.get_nbasis() // LDC: if(N) max(1, m)
    );
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // zgemm_(&trans,
    //         &transb,
    //         &npw, // m: row of A,C
    //         &notconv, // n: col of B,C
    //         &nbase, // k: col of A, row of B
    //         &ModuleBase::ONE, // alpha
    //         sp.c, // A
    //         &sp.nc, // LDA: if(N) max(1,m) if(T) max(1,k)
    //         vc_ev_vector.c, // B
    //         &vc_ev_vector.nc, // LDB: if(N) max(1,k) if(T) max(1,n)
    //         &ModuleBase::ONE, // belta
    //         ppsi, // C
    //         &basis.get_nbasis()); // LDC: if(N) max(1, m)
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        /*zgemv_(&trans,
            &npw,
            &nbase,
            &ModuleBase::ONE,
            sp.c,
            &sp.nc,
            vc_ev_vector.data(),
            &inc,
            &ModuleBase::ONE,
            respsi,
            &inc);*/

        /*ModuleBase::GlobalFunc::ZEROS(respsi, npw);
        for (int i = 0; i < nbase; i++)
        {
            hpsi = &(hp(i, 0));
            spsi = &(sp(i, 0));
            auto vc_value = vc(i, unconv[m]);
            auto ev_value = eigenvalue[unconv[m]];
            for (int ig = 0; ig < npw; ig++)
            {
                respsi[ig] += vc_value * (hpsi[ig] - ev_value * spsi[ig]);
            }
        }*/
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    for (int m = 0; m < notconv; m++)
    {
        ppsi = &basis(nbase + m, 0);
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // haozhihan replace 2022-10-18
        vector_div_vector_op<FPTYPE, Device>()(this->ctx, npw, ppsi, ppsi, this->precondition);
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // for (int ig = 0; ig < npw; ig++)
        // {
        //     ppsi[ig] /= this->precondition[ig];
        // }
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    }

    //there is a nbase to nbase + notconv band orthogonalise
    //plan for SchmitOrth
    // ModuleBase::ComplexMatrix lagrange_matrix(notconv, nbase + notconv);
    std::complex<FPTYPE>* lagrange = nullptr;
    resize_memory_op()(this->ctx, lagrange, notconv * (nbase + notconv));
    set_memory_op()(this->ctx, lagrange, 0, notconv * (nbase + notconv));



    std::vector<int> pre_matrix_mm_m(notconv, 0);
    std::vector<int> pre_matrix_mv_m(notconv, 1);
    this->planSchmitOrth(notconv, pre_matrix_mm_m.data(), pre_matrix_mv_m.data());
    for( int m = 0; m < notconv; m++)
    {
        phm_in->sPsi(&basis(nbase + m, 0), &sphi[ (nbase + m) * this->dim ],  (size_t)npw);
    }
    //first nbase bands psi* dot notconv bands spsi to prepare lagrange_matrix

    //calculate the square matrix for future lagranges
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan replace 2022-10-18
    gemm_op<FPTYPE, Device>()(
        this->ctx,
        'C',
        'N',
        nbase, // m: row of A,C
        notconv, // n: col of B,C
        npw, // k: col of A, row of B
        &ModuleBase::ONE, // alpha
        &basis(0, 0), // A
        basis.get_nbasis(), // LDA: if(N) max(1,m) if(T) max(1,k)
        &sphi[nbase * this->dim], // B
        this->dim, // LDB: if(N) max(1,k) if(T) max(1,n)
        &ModuleBase::ZERO, // belta
        lagrange, // C
        nbase + notconv // LDC: if(N) max(1, m)
    );
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // zgemm_(&trans,
    //         &transb,
    //         &nbase, // m: row of A,C
    //         &notconv, // n: col of B,C
    //         &npw, // k: col of A, row of B
    //         &ModuleBase::ONE, // alpha
    //         &basis(0, 0), // A
    //         &basis.get_nbasis(), // LDA: if(N) max(1,m) if(T) max(1,k)
    //         &sp(nbase, 0), // B
    //         &sp.nc, // LDB: if(N) max(1,k) if(T) max(1,n)
    //         &ModuleBase::ZERO, // belta
    //         &lagrange_matrix(0, 0), // C
    //         &lagrange_matrix.nc); // LDC: if(N) max(1, m)
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    for (int m = 0; m < notconv; m++)
    {
        ppsi = &basis(nbase + m, 0);
        spsi = &sphi[ (nbase + m) * this->dim ];

        this->SchmitOrth(
            npw,
            nbase + notconv,
            nbase + m,
            basis,
            sphi,
            &lagrange[m * (nbase + notconv)],
            pre_matrix_mm_m[m],
            pre_matrix_mv_m[m]
        );
        phm_in->sPsi(ppsi, spsi, (size_t)npw);

    }
    //calculate H|psi> for not convergence bands
    hpsi_info dav_hpsi_in(&basis, psi::Range(1, 0, nbase, nbase + notconv-1), &hphi[nbase * this->dim]); // &hp(nbase, 0)
    phm_in->ops->hPsi(dav_hpsi_in);
    
    ModuleBase::timer::tick("DiagoDavid", "cal_grad");
    return;
}

template<typename FPTYPE, typename Device>
void DiagoDavid<FPTYPE, Device>::cal_elem(const int &npw,
                          int &nbase, // current dimension of the reduced basis
                          const int &notconv, // number of newly added basis vectors
                          const psi::Psi<std::complex<FPTYPE>, Device> &basis,
                          const std::complex<FPTYPE>* hphi,
                          const std::complex<FPTYPE>* sphi,
                          std::complex<FPTYPE>* hcc,
                          std::complex<FPTYPE>* scc)
{
    if (test_david == 1)
        ModuleBase::TITLE("DiagoDavid", "cal_elem");

    if(notconv == 0) return;
    ModuleBase::timer::tick("DiagoDavid", "cal_elem");

    // update the reduced Hamiltonian
    int offset_h = nbase * this->nbase_x;
    int offset_s = nbase * this->nbase_x;
    //	ModuleBase::GlobalFunc::ZEROS( hc.c+offset_h, notconv*hc.nr );
    //	ModuleBase::GlobalFunc::ZEROS( sc.c+offset_s, notconv*sc.nr );

    const int nb_notc = (nbase + notconv);


    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan replace 2022-10-18
    matrixTranspose_op<FPTYPE, Device>()(this->ctx, this->nbase_x, this->nbase_x, hcc, hcc);
    // hc = transpose(hc, false);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan repalce 2022-10-16
    gemm_op<FPTYPE, Device>()(
        this->ctx,
        'C',
        'N',
        notconv,
        nb_notc,
        npw,
        &ModuleBase::ONE,
        &basis(nbase, 0), // this->dim * notconv
        basis.get_nbasis(), // this->dim
        hphi, // this->dim * (nbase + notconv)
        this->dim,
        &ModuleBase::ZERO,
        hcc + nbase, // notconv * (nbase + notconv)
        this->nbase_x
    );
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // zgemm_(&trans1,
    //        &trans2,
    //        &notconv,
    //        &nb_notc,
    //        &npw,
    //        &ModuleBase::ONE,
    //        &basis(nbase, 0),
    //        &basis.get_nbasis(),
    //        hp.c,
    //        &hp.nc,
    //        &ModuleBase::ONE,
    //        hc.c + nbase,
    //        &hc.nr);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // replace transpose
    matrixTranspose_op<FPTYPE, Device>()(this->ctx, this->nbase_x, this->nbase_x, hcc, hcc);
    // hc = transpose(hc, false);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // replace transpose
    matrixTranspose_op<FPTYPE, Device>()(this->ctx, this->nbase_x, this->nbase_x, scc, scc);
    // sc = transpose(sc, false);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan repalce 2022-10-16
    gemm_op<FPTYPE, Device>()(
        this->ctx,
        'C',
        'N',
        notconv,
        nb_notc,
        npw,
        &ModuleBase::ONE,
        &basis(nbase, 0),
        basis.get_nbasis(),
        sphi,
        this->dim,
        &ModuleBase::ZERO,
        scc + nbase,
        this->nbase_x
    );
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // zgemm_(&trans1,
    //        &trans2,
    //        &notconv,
    //        &nb_notc,
    //        &npw,
    //        &ModuleBase::ONE,
    //        &basis(nbase, 0),
    //        &basis.get_nbasis(),
    //        sp.c,
    //        &sp.nc,
    //        &ModuleBase::ONE,
    //        sc.c + nbase,
    //        &sc.nr);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // replace transpose
    matrixTranspose_op<FPTYPE, Device>()(this->ctx, this->nbase_x, this->nbase_x, scc, scc);
    // sc = transpose(sc, false);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 



//    Parallel_Reduce::reduce_complex_double_pool(hc.c + offset_h, notconv * hc.nr);
//    Parallel_Reduce::reduce_complex_double_pool(sc.c + offset_s, notconv * sc.nr);
#ifdef __MPI
	std::complex<double> *swap = new std::complex<double>[notconv * this->nbase_x];

    
    // ModuleBase::GlobalFunc::COPYARRAY(hcc+offset_h, swap, notconv * this->nbase_x);
    psi::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, Device>()(this->ctx, this->ctx, swap, hcc+offset_h, notconv * this->nbase_x);

    MPI_Reduce(swap,hcc+offset_h,notconv * this->nbase_x,MPI_DOUBLE_COMPLEX,MPI_SUM,0,MPI_COMM_WORLD);
    
    // ModuleBase::GlobalFunc::COPYARRAY(scc+offset_h, swap, notconv * this->nbase_x);
    psi::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, Device>()(this->ctx, this->ctx, swap, scc+offset_h, notconv * this->nbase_x);
    
    MPI_Reduce(swap,scc+offset_h,notconv * this->nbase_x,MPI_DOUBLE_COMPLEX,MPI_SUM,0,MPI_COMM_WORLD);
    delete[] swap;
#endif
    /*
        for( int i = nbase; i < nbase+notconv; i++ )
        {
            for( int j = 0; j <i; j++ )
            {
                hc(j,i) = conj( hc(i,j) );
                sc(j,i) = conj( sc(i,j) );
            }
        }
    */
    nbase += notconv;
    ModuleBase::timer::tick("DiagoDavid", "cal_elem");
    return;
}

//==============================================================================
// optimize diag_zhegvx().

// 09-05-09 wangjp
// fixed a bug in diag_zhegvx().
// modify the dimension of h and s as (n,n) and copy the leading N*N
// part of hc & sc into h & s

// 09-05-10 wangjp
// As the complexmatrixs will be copied again in the subroutine ZHEGVX(...  ),
// i.e ZHEGVX(...) will not destroy the input complexmatrixs,
// we needn't creat another two complexmatrixs in diag_zhegvx().
//==============================================================================
template<typename FPTYPE, typename Device>
void DiagoDavid<FPTYPE, Device>::diag_zhegvx(const int &n,      // nbase
                             const int &m,                      // this->n_band
                             const std::complex<FPTYPE>* hc,    // hc
                             const std::complex<FPTYPE>* sc,    // sc
                             const int &ldh,                    // this->nbase_x
                             FPTYPE *eigenvalue,
                             std::complex<FPTYPE>* vc)
{
    //	ModuleBase::TITLE("DiagoDavid","diag_zhegvx");
    ModuleBase::timer::tick("DiagoDavid", "diag_zhegvx");
if(GlobalV::MY_RANK == 0)
{
    assert(ldh >= max(1, n));


    dngvx_op<FPTYPE, Device>()(
        this->ctx,
        this->nbase_x,
        n,
        this->hcc,
        this->scc,
        m,
        this->eigenvalue,
        this->vcc
    );


    // int lwork;
    // int info = 0;
    // int mm = m;
    // std::string name1 = "ZHETRD";
    // std::string name2 = "L";

    // int nb = LapackConnector::ilaenv(1, name1.c_str(), name2.c_str(), n, -1, -1, -1);
    // if (nb < 1)
    // {
    //     nb = max(1, n);
    // }
    // if (nb == 1 || nb >= n)
    // {
    //     lwork = 2 * n; // qianrui fix a bug 2021-7-25 : lwork should be at least max(1,2*n)
    // }
    // else
    // {
    //     lwork = (nb + 1) * n;
    // }
    // std::complex<double> *work = new std::complex<double>[2 * lwork];
    // assert(work != 0);
    // double *rwork = new double[7 * n];
    // assert(rwork != 0);
    // int *iwork = new int[5 * n];
    // assert(iwork != 0);
    // int *ifail = new int[n];
    // assert(ifail != 0);
    // ModuleBase::GlobalFunc::ZEROS(work, lwork); // qianrui change it, only first lwork numbers are used in zhegvx
    // ModuleBase::GlobalFunc::ZEROS(rwork, 7 * n);
    // ModuleBase::GlobalFunc::ZEROS(iwork, 5 * n);
    // ModuleBase::GlobalFunc::ZEROS(ifail, n);

    // // ModuleBase::WARNING_QUIT("divid","open zhegvx!");


    // std::cout << "n is " << n << "; And m is " << m << "; And mm is " << mm << std::endl;
    

    // LapackConnector::zhegvx(1,       // ITYPE = 1:  A*x = (lambda)*B*x
    //                         'V',     // JOBZ = 'V':  Compute eigenvalues and eigenvectors.
    //                         'I',     // RANGE = 'I': the IL-th through IU-th eigenvalues will be found.
    //                         'L',     // UPLO = 'L':  Lower triangles of A and B are stored.
    //                         n,       // N = base 
    //                         hcc,     // A is COMPLEX*16 array  dimension (LDA, N)
    //                         n,       // LDA = base
    //                         scc,     // B is COMPLEX*16 array, dimension (LDB, N)
    //                         n,       // LDB = base
    //                         0.0,     // Not referenced if RANGE = 'A' or 'I'.
    //                         0.0,     // Not referenced if RANGE = 'A' or 'I'.
    //                         1,       // IL: If RANGE='I', the index of the smallest eigenvalue to be returned. 1 <= IL <= IU <= N,
    //                         m,       // IU: If RANGE='I', the index of the largest eigenvalue to be returned. 1 <= IL <= IU <= N,
    //                         0.0,     // ABSTOL
    //                         mm,      // M: The total number of eigenvalues found.  0 <= M <= N. if RANGE = 'I', M = IU-IL+1.
    //                         this->eigenvalue, // W store eigenvalues
    //                         vcc,     // store eigenvector
    //                         n,       // LDZ: The leading dimension of the array Z.
    //                         work,
    //                         lwork,
    //                         rwork,
    //                         iwork,
    //                         ifail,
    //                         info,
    //                         this->nbase_x );


    // delete[] work;
    // delete[] rwork;
    // delete[] iwork;
    // delete[] ifail;
}

#ifdef __MPI
    for (int i=0;i<n;i++)
    {
        Parallel_Common::bcast_complex_double(&vcc[i * this->nbase_x], m);
    }
    Parallel_Common::bcast_double(this->eigenvalue,m);
#endif

    ModuleBase::timer::tick("DiagoDavid", "diag_zhegvx");
    return;
}

template<typename FPTYPE, typename Device>
void DiagoDavid<FPTYPE, Device>::refresh(const int &npw,
                         const int &nband,
                         int &nbase,
                         const FPTYPE *eigenvalue_in,
                         const psi::Psi<std::complex<FPTYPE>, Device> &psi,
                         psi::Psi<std::complex<FPTYPE>, Device> &basis,
                         std::complex<FPTYPE>* hp,
                         std::complex<FPTYPE>* sp,
                         std::complex<FPTYPE>* hc,
                         std::complex<FPTYPE>* sc,
                         std::complex<FPTYPE>* vc)
{
    if (test_david == 1)
        ModuleBase::TITLE("DiagoDavid", "refresh");
    ModuleBase::timer::tick("DiagoDavid", "refresh");

    // update hp,sp
    basis.zero_out();
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan repalce 2022-10-18
    gemm_op<FPTYPE, Device>()(
        this->ctx,
        'N',
        'T',
        npw, // m: row of A,C
        nband, // n: col of B,C
        nbase, // k: col of A, row of B
        &ModuleBase::ONE, // alpha
        hphi, // A
        this->dim, // LDA: if(N) max(1,m) if(T) max(1,k)
        vcc, // B
        this->nbase_x, // LDB: if(N) max(1,k) if(T) max(1,n)
        &ModuleBase::ZERO, // belta
        basis.get_pointer(), // C
        basis.get_nbasis() // LDC: if(N) max(1, m)
    );
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // zgemm_(&transa,
    //         &transb,
    //         &npw, // m: row of A,C
    //         &nband, // n: col of B,C
    //         &nbase, // k: col of A, row of B
    //         &ModuleBase::ONE, // alpha
    //         hp.c, // A
    //         &hp.nc, // LDA: if(N) max(1,m) if(T) max(1,k)
    //         vc.c, // B
    //         &vc.nc, // LDB: if(N) max(1,k) if(T) max(1,n)
    //         &ModuleBase::ZERO, // belta
    //         basis.get_pointer(), // C
    //         &basis.get_nbasis()); // LDC: if(N) max(1, m)
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan repalce 2022-10-18
    gemm_op<FPTYPE, Device>()(
        this->ctx,
        'N',
        'T',
        npw, // m: row of A,C
        nband, // n: col of B,C
        nbase, // k: col of A, row of B
        &ModuleBase::ONE, // alpha
        sphi, // A
        this->dim, // LDA: if(N) max(1,m) if(T) max(1,k)
        vcc, // B
        this->nbase_x, // LDB: if(N) max(1,k) if(T) max(1,n)
        &ModuleBase::ZERO, // belta
        &basis(nband, 0), // C
        basis.get_nbasis() // LDC: if(N) max(1, m)
    );
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // zgemm_(&transa,
    //         &transb,
    //         &npw, // m: row of A,C
    //         &nband, // n: col of B,C
    //         &nbase, // k: col of A, row of B
    //         &ModuleBase::ONE, // alpha
    //         sp.c, // A
    //         &sp.nc, // LDA: if(N) max(1,m) if(T) max(1,k)
    //         vc.c, // B
    //         &vc.nc, // LDB: if(N) max(1,k) if(T) max(1,n)
    //         &ModuleBase::ZERO, // belta
    //         &basis(nband, 0), // C
    //         &basis.get_nbasis()); // LDC: if(N) max(1, m)
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    /*for (int m = 0; m < nband; m++)
    {
        for (int j = 0; j < nbase; j++)
        {
            for (int ig = 0; ig < npw; ig++)
            {
                basis(m, ig) += vc(j, m) * hp(j, ig);
                basis(m + nband, ig) += vc(j, m) * sp(j, ig);
            }
        }
    }*/
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    // ModuleBase::GlobalFunc::COPYARRAY(&basis(0, 0), hphi, npw * nband);
    psi::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, Device>()(this->ctx, this->ctx, hphi, &basis(0, 0), npw * nband);
    
    // ModuleBase::GlobalFunc::COPYARRAY(&basis(nband, 0), sphi, npw * nband);
    psi::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, Device>()(this->ctx, this->ctx, sphi, &basis(nband, 0), npw * nband);
    /*for (int m = 0; m < nband; m++)
    {
        for (int ig = 0; ig < npw; ig++)
        {
            hp(m, ig) = basis(m, ig);
            sp(m, ig) = basis(m + nband, ig);
        }
    }*/

    // update basis
    basis.zero_out();
    for (int m = 0; m < nband; m++)
    {
        // ModuleBase::GlobalFunc::COPYARRAY(&psi(m, 0), &basis(m, 0), npw);
        psi::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, Device>()(this->ctx, this->ctx, &basis(m, 0), &psi(m, 0), npw);
        /*for (int ig = 0; ig < npw; ig++)
            basis(m, ig) = psi(m, ig);*/
    }

    // updata the reduced Hamiltonian
    nbase = nband;
    
    
    // hc.zero_out();
    set_memory_op()(this->ctx, hcc, 0, this->nbase_x * this->nbase_x);
    
    // sc.zero_out();
    set_memory_op()(this->ctx, scc, 0, this->nbase_x * this->nbase_x);

    for (int i = 0; i < nbase; i++)
    {
        hcc[i * this->nbase_x + i] = eigenvalue_in[i];
        // sc(i, i) = ModuleBase::ONE;
        scc[i * this->nbase_x + i] = ModuleBase::ONE;
        // vc(i, i) = ModuleBase::ONE;
        vcc[i * this->nbase_x + i] = ModuleBase::ONE;
    }

    ModuleBase::timer::tick("DiagoDavid", "refresh");
    return;
}



template<typename FPTYPE, typename Device>
void DiagoDavid<FPTYPE, Device>::SchmitOrth(const int &npw,
                            const int n_band,
                            const int m,
                            psi::Psi<std::complex<FPTYPE>, Device>& psi,
                            const std::complex<FPTYPE>* spsi,
                            std::complex<FPTYPE>* lagrange_m,
                            const int mm_size,
                            const int mv_size)
{
    //	if(test_david == 1) ModuleBase::TITLE("DiagoDavid","SchmitOrth");
    ModuleBase::timer::tick("DiagoDavid", "SchmitOrth");

    // orthogonalize starting eigenfunction to those already calculated
    // psi_m orthogonalize to psi(0) ~ psi(m-1)
    // Attention, the orthogonalize here read as
    // psi(m) -> psi(m) - \sum_{i < m} \langle psi(i)|S|psi(m) \rangle psi(i)
    // so the orthogonalize is performed about S.

    assert(psi.get_nbands() >= n_band);
    assert(m >= 0);
    assert(m < n_band);

    std::complex<double>* psi_m = &psi(m, 0);

    //std::complex<double> *lagrange = new std::complex<double>[m + 1];
    //ModuleBase::GlobalFunc::ZEROS(lagrange, m + 1);

    //calculate the square matrix for future lagranges
    if(mm_size != 0)
    {
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // haozhihan repalce 2022-10-16
        gemm_op<FPTYPE, Device>()(
            this->ctx,
            'C',
            'N',
            mm_size, // m: row of A,C
            mm_size, // n: col of B,C
            npw, // k: col of A, row of B
            &ModuleBase::ONE, // alpha
            &psi(m-mv_size+1-mm_size, 0), // A
            psi.get_nbasis(), // LDA: if(N) max(1,m) if(T) max(1,k)
            &sphi[m * this->dim], // B
            this->dim, // LDB: if(N) max(1,k) if(T) max(1,n)
            &ModuleBase::ZERO, // belta
            &lagrange_m[m-mv_size+1-mm_size], // C
            n_band // LDC: if(N) max(1, m)
        );
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // zgemm_(&trans,
        //     &transb,
        //     &mm_size, // m: row of A,C
        //     &mm_size, // n: col of B,C
        //     &npw, // k: col of A, row of B
        //     &ModuleBase::ONE, // alpha
        //     &psi(m-mv_size+1-mm_size, 0), // A
        //     &psi.get_nbasis(), // LDA: if(N) max(1,m) if(T) max(1,k)
        //     &spsi(m, 0), // B
        //     &spsi.nc, // LDB: if(N) max(1,k) if(T) max(1,n)
        //     &ModuleBase::ZERO, // belta
        //     &lagrange_m[m-mv_size+1-mm_size], // C
        //     &n_band); // LDC: if(N) max(1, m)
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    }
    //calculate other lagranges for this band
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan repalce 2022-10-16
    gemv_op<FPTYPE, Device>()(this->ctx,
        'C',
        npw,
        mv_size,
        &ModuleBase::ONE,
        &psi(m-mv_size+1, 0),
        psi.get_nbasis(),
        &sphi[m * this->dim],
        1,
        &ModuleBase::ZERO,
        &lagrange_m[m-mv_size+1],
        1);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // zgemv_(&trans,
    //        &npw,
    //        &mv_size,
    //        &ModuleBase::ONE,
    //        &psi(m-mv_size+1, 0),
    //        &psi.get_nbasis(),
    //        &spsi(m, 0),
    //        &inc,
    //        &ModuleBase::ZERO,
    //        &lagrange_m[m-mv_size+1],
    //        &inc);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    /*for (int j = 0; j < m; j++)
    {
        const std::complex<double>* psi_p = &(psi(j, 0));
        zdotc_(&lagrange[j], &npw, psi_p, &one, spsi, &one);
        for (int ig = 0; ig < npw; ig++)
        {
            lagrange[j] += conj(psi(j, ig)) * spsi[ig];
        }
        //	lagrange[j] = Diago_CG::ddot( npw, psi, j, spsi );
    }*/
    //zdotc_(&lagrange[m], &npw, psi_m, &inc, spsi, &inc);
    /*for (int ig = 0; ig < npw; ig++)
    {
        lagrange[m] += conj(psi_m[ig]) * spsi[ig];
    }*/
    //	lagrange[m] = Diago_CG::ddot( npw, psi_m, spsi );

    Parallel_Reduce::reduce_complex_double_pool(lagrange_m, m + 1);

    //	out.printr1_d("lagrange", lagrange, m+1 );

    double psi_norm = lagrange_m[m].real();
    assert(psi_norm > 0.0);
    //	std::cout << "m = " << m << std::endl;


    // haozhihan replace 2022-10-24
    gemv_op<FPTYPE, Device>()(this->ctx,
                            'N',
                            npw,
                            m,
                            &ModuleBase::NEG_ONE,
                            &psi(0,0),
                            npw,
                            lagrange_m,
                            1,
                            &ModuleBase::ONE,
                            psi_m,
                            1);

    psi_norm -= zdot_real_op<FPTYPE, Device>()(this->ctx, m, lagrange_m, lagrange_m, false);

    // for (int j = 0; j < m; j++)
    // {
    //     const std::complex<double> alpha = std::complex<double>(-1, 0) * lagrange_m[j];
    //     zaxpy_(&npw, &alpha, &psi(j,0), &inc, psi_m, &inc);
    //     /*for (int ig = 0; ig < npw; ig++)
    //     {
    //         psi_m[ig] -= lagrange[j] * psi(j, ig);
    //     }*/
    //     psi_norm -= (conj(lagrange_m[j]) * lagrange_m[j]).real();
    // }

    assert(psi_norm > 0.0);

    psi_norm = sqrt(psi_norm);

    if (psi_norm < 1.0e-12)
    {
        std::cout << "DiagoDavid::SchmitOrth:aborted for psi_norm <1.0e-12" << std::endl;
        std::cout << "n_band = " << n_band << std::endl;
        std::cout << "m = " << m << std::endl;
        exit(0);
    }
    else
    {
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // haozhihan repalce 2022-10-16      
        vector_div_constant_op<FPTYPE, Device>()(this->ctx, npw, psi_m, psi_m, psi_norm);
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // for (int i = 0; i < npw; i++)
        // {
        //     psi_m[i] /= psi_norm;
        // }

    }

    //delete[] lagrange;
    ModuleBase::timer::tick("DiagoDavid", "SchmitOrth");
    return;
}

template<typename FPTYPE, typename Device>
void DiagoDavid<FPTYPE, Device>::planSchmitOrth(
    const int nband,
    int* pre_matrix_mm_m,
    int* pre_matrix_mv_m
)
{
    if(nband<=0)return;
    ModuleBase::GlobalFunc::ZEROS(pre_matrix_mm_m, nband);
    ModuleBase::GlobalFunc::ZEROS(pre_matrix_mv_m, nband);
    int last_matrix_size = nband;
    int matrix_size = int(nband / 2);
    int divide_times = 0;
    std::vector<int> divide_points(nband);
    int res_nband = nband - matrix_size;
    while(matrix_size>1)
    {
        int index = nband - matrix_size;
        if(divide_times == 0)
        {
            divide_points[0] = index;
            pre_matrix_mm_m[index] = matrix_size;
            if(res_nband == matrix_size) pre_matrix_mv_m[index] = 1;
            else pre_matrix_mv_m[index] = 2;
            divide_times = 1;
        }
        else
        {
            for(int i=divide_times-1; i>=0; i--)
            {
                divide_points[i*2] = divide_points[i] - matrix_size;
                divide_points[i*2+1] = divide_points[i*2] + last_matrix_size;
                pre_matrix_mm_m[ divide_points[i*2] ] = matrix_size;
                pre_matrix_mm_m[ divide_points[i*2+1]] = matrix_size;
                if(res_nband == matrix_size)
                {
                    pre_matrix_mv_m[divide_points[i*2]] = 1;
                    pre_matrix_mv_m[divide_points[i*2+1]] = 1;
                }
                else
                {
                    pre_matrix_mv_m[divide_points[i*2]] = 2;
                    pre_matrix_mv_m[divide_points[i*2+1]] = 2;
                }
            }
            divide_times *= 2;
        }
        last_matrix_size = matrix_size;
        matrix_size = int(res_nband / 2);
        res_nband -= matrix_size;
    }
    //fill the pre_matrix_mv_m array
    pre_matrix_mv_m[0] = 1;
    for(int m = 1; m < nband; m++)
    {
        if(pre_matrix_mv_m[m] == 0)
        {
            pre_matrix_mv_m[m] = pre_matrix_mv_m[m-1]+1;
        }
    }
}


#if ((defined __CUDA) || (defined __ROCM))
template<>
void DiagoDavid<double, psi::DEVICE_GPU>::cal_grad(hamilt::Hamilt* phm_in,
                          const int &npw,
                          const int &nbase, // current dimension of the reduced basis
                          const int &notconv,
                          psi::Psi<std::complex<double>, psi::DEVICE_GPU> &basis,
                          std::complex<double>* hp,
                          std::complex<double>* sp,
                          const std::complex<double>* vc,
                          const int *unconv,
                          const double *eigenvalue)
{
    return;
}

template<>
void DiagoDavid<double, psi::DEVICE_GPU>::diag_mock(hamilt::Hamilt* phm_in, psi::Psi<std::complex<double>, psi::DEVICE_GPU> &psi, double* eigenvalue_in)
{
    return;
}
#endif // ((defined __CUDA) || (defined __ROCM))






template<typename FPTYPE, typename Device>
void DiagoDavid<FPTYPE, Device>::diag(hamilt::Hamilt<double> *phm_in, psi::Psi<std::complex<FPTYPE>, Device> &psi, FPTYPE* eigenvalue_in)
{
    /// record the times of trying iterative diagonalization
    int ntry = 0;
    this->notconv = 0;
    do
    {
        this->diag_mock(phm_in, psi, eigenvalue_in);
        ++ntry;
    } while (DiagoIterAssist<FPTYPE, Device>::test_exit_cond(ntry, this->notconv));

    if (notconv > max(5, psi.get_nbands() / 4))
    {
        std::cout << "\n notconv = " << this->notconv;
        std::cout << "\n DiagoDavid::diag', too many bands are not converged! \n";
    }
    return;
}

namespace hsolver {
template class DiagoDavid<double, psi::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoDavid<double, psi::DEVICE_GPU>;
#endif 
} // namespace hsolver
