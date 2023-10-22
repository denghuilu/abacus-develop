#include "diago_david.h"

#include "diago_iter_assist.h"
#include "module_base/blas_connector.h"
#include "module_base/constants.h"
#include "module_base/lapack_connector.h"
#include "module_base/memory.h"
#include "module_base/parallel_common.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_hsolver/kernels/dngvd_op.h"
#include "module_hsolver/kernels/math_kernel_op.h"

#include <ATen/core/tensor_map.h>

#ifdef USE_PAW
#include "module_cell/module_paw/paw_cell.h"
#endif

using namespace hsolver;

template <typename T, typename Device> 
DiagoDavid<T, Device>::DiagoDavid(const ct::Tensor& prec_in, const ct::Tensor& n_basis_in)
{
    // reference to prec_in and n_basis_in
    this->h_prec_.CopyFrom(prec_in);
    this->n_basis_.CopyFrom(n_basis_in);

    this->test_david_ = 2;
    this->one_ = ct::Tensor({static_cast<T>(1.0)});
    this->zero_ = ct::Tensor({static_cast<T>(0.0)});
    this->neg_one_ = ct::Tensor({static_cast<T>(-1.0)});

    this->device = psi::device::get_device_type<Device>(this->ctx_);
    // 1: check which function is called and which step is executed
    // 2: check the eigenvalues of the result of each iteration
    // 3: check the eigenvalues and errors of the last result
    // default: no check
}

template <typename T, typename Device>
void DiagoDavid<T, Device>::diag_mock(
    hamilt::Hamilt* phm_in,
    ct::Tensor& psi,
    ct::Tensor& eigen_in)
{
    if (this->test_david_ == 1) {
        ModuleBase::TITLE("DiagoDavid", "diag_mock");
    }
    ModuleBase::timer::tick("DiagoDavid", "diag_mock");

    assert(DiagoDavid::PW_DIAG_NDIM > 1);
    assert(DiagoDavid::PW_DIAG_NDIM * psi.get_nbands() < psi.get_current_nbas() * GlobalV::NPROC_IN_POOL);
    // qianrui change it 2021-7-25.
    // In strictly speaking, it shoule be PW_DIAG_NDIM*nband < npw sum of all pools. We roughly estimate it here.
    // However, in most cases, total number of plane waves should be much larger than nband*PW_DIAG_NDIM

    /// initialize variables
    this->n_band_ = psi.shape().dim_size(1);
    this->n_basis_max_ = psi.shape().dim_size(2);
    // Note: this->ik_ is a member value of basis class DiagH, 
    // and this value was initialized by a function call within the HsolverPW's solve function
    this->current_n_basis_ = this->n_basis_.data<int>()[this->ik_];

    this->n_base_x_ = DiagoDavid::PW_DIAG_NDIM * this->n_band_; // maximum dimension of the reduced basis set

    // the lowest N eigenvalues
    this->eigen_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {DiagoDavid::PW_DIAG_NDIM, this->n_band_}));
    this->eigen_.zero();

    ct::Tensor basis = ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {1, this->n_base_x_, this->n_basis_max_});
    ModuleBase::Memory::record("DAV::basis", this->n_base_x_ * this->n_basis_max_ * sizeof(T));

    // ModuleBase::ComplexMatrix hp(nbase_x, this->current_n_basis_); // the product of H and psi in the reduced basis set
    this->hphi_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_base_x_, this->current_n_basis_}));
    this->hphi_.zero();
    // ModuleBase::ComplexMatrix sp(nbase_x, this->current_n_basis_); // the Product of S and psi in the reduced basis set
    this->sphi_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_base_x_, this->current_n_basis_}));
    this->sphi_.zero();
    // ModuleBase::ComplexMatrix hc(this->n_base_x_, this->n_base_x_); // Hamiltonian on the reduced basis
    this->hcc_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_base_x_, this->n_base_x_}));
    this->hcc_.zero();
    // ModuleBase::ComplexMatrix sc(this->n_base_x_, this->n_base_x_); // Overlap on the reduced basis
    this->scc_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_base_x_, this->n_base_x_}));
    this->scc_.zero();
    // ModuleBase::ComplexMatrix vc(this->n_base_x_, this->n_base_x_); // Eigenvectors of hc
    this->vcc_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_base_x_, this->n_base_x_}));
    this->vcc_.zero();

    // convflag[m] = true if the m th band is convergent
    std::vector<bool> convflag(this->n_band_, false);
    // unconv[m] store the number of the m th unconvergent band
    std::vector<int> unconv(this->n_band_);

    int nbase = 0; // the dimension of the reduced basis set

    this->notconv_ = this->n_band_; // the number of the unconvergent bands

    for (int m = 0; m < this->n_band_; m++)
        unconv[m] = m;

    ModuleBase::timer::tick("DiagoDavid", "first");

    // orthogonalise the initial trial psi(0~nband-1)
    // ModuleBase::ComplexMatrix lagrange_matrix(this->n_band_, this->n_band_);
    this->lagrange_matrix_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_band_, this->n_band_}));
    this->lagrange_matrix_.zero();

    // plan for SchmitOrth
    std::vector<int> pre_matrix_mm_m(this->n_band_, 0);
    std::vector<int> pre_matrix_mv_m(this->n_band_, 1);
    this->planSchmitOrth(this->n_band_, pre_matrix_mm_m, pre_matrix_mv_m);

    auto psi_pack = psi.accessor<T, 3>();
    auto basis_pack = basis.accessor<T, 3>();
    auto sphi_pack = this->sphi_.accessor<T, 2>();
    auto eigen_pack = this->eigen_.accessor<T, 2>();
    auto eigen_in_pack  = eigen_in.accessor<T, 1>();
    for (int m = 0; m < this->n_band_; m++)
    {
        if(GlobalV::use_paw)
        {
#ifdef USE_PAW
#ifdef __DEBUG
            assert(psi.get_k_first());
#endif 
            GlobalC::paw_cell.paw_nl_psi(1, reinterpret_cast<const std::complex<double>*> (&psi(m, 0)),
                reinterpret_cast<std::complex<double>*>(&this->sphi[m * this->dim]));
#endif
        }
        else
        {
            auto sphi_m = this->sphi_[m];
            phm_in->sPsi(psi[this->ik_][m], sphi_m);
        }
    }
    // begin SchmitOrth
    for (int m = 0; m < this->n_band_; m++)
    {
        // denghuilu replace 2023-10-09
        // syncmem_complex_op()(this->ctx_, this->ctx_, &basis(m, 0), &psi(m, 0), this->current_n_basis_);
        basis[0][m].sync(psi[this->ik_][m]);
        auto lagrange_matrix_m = ct::TensorMap(
            this->lagrange_matrix_[m].data<T>(), this->lagrange_matrix_.data_type(), this->lagrange_matrix_.device_type(), {this->n_band_ - m, this->n_band_});
        this->SchmitOrth(this->current_n_basis_,
                         this->n_band_,
                         m,
                         basis,
                         this->sphi_,
                         lagrange_matrix_m,
                         pre_matrix_mm_m[m],
                         pre_matrix_mv_m[m]);
        if(GlobalV::use_paw)
        {
#ifdef USE_PAW
            GlobalC::paw_cell.paw_nl_psi(1,reinterpret_cast<const std::complex<double>*> (&basis(m, 0)),
                reinterpret_cast<std::complex<double>*>(&this->sphi[m * this->current_n_basis_]));
#endif
        }
        else
        {
            auto sphi_m = this->sphi_[m];
            phm_in->sPsi(basis[0][m], sphi_m);
        }
    }

    // end of SchmitOrth and calculate H|psi>
    phm_in->ops->hPsi(basis, this->hphi_);

    this->cal_elem(this->current_n_basis_, nbase, this->notconv_, basis, this->hphi_, this->sphi_, this->hcc_, this->scc_);

    this->diag_zhegvx(nbase, this->n_band_, this->hcc_, this->scc_, this->n_base_x_, this->eigen_, this->vcc_);
    eigen_in.sync(this->eigen_[0]);

    ModuleBase::timer::tick("DiagoDavid", "first");

    int dav_iter = 0;
    do
    {
        dav_iter++;

        this->cal_grad(phm_in,
                       this->current_n_basis_,
                       nbase,
                       this->notconv_,
                       basis,
                       this->hphi_,
                       this->sphi_,
                       this->vcc_,
                       unconv,
                       this->eigen_);

        this->cal_elem(this->current_n_basis_, nbase, this->notconv_, basis, this->hphi_, this->sphi_, this->hcc_, this->scc_);

        this->diag_zhegvx(nbase, this->n_band_, this->hcc_, this->scc_, this->n_base_x_, this->eigen_, this->vcc_);

        // check convergence and update eigenvalues
        ModuleBase::timer::tick("DiagoDavid", "check_update");

        this->notconv_ = 0;
        for (int m = 0; m < this->n_band_; m++)
        {
            convflag[m] = (std::abs(eigen_pack[0][m] - eigen_in_pack[m]) < DiagoIterAssist<T, Device>::PW_DIAG_THR);

            if (!convflag[m])
            {
                unconv[this->notconv_] = m;
                this->notconv_++;
            }

            eigen_in_pack[m] = eigen_pack[0][m];
        }

        ModuleBase::timer::tick("DiagoDavid", "check_update");
        if (!this->notconv_ || (nbase + this->notconv_ > this->n_base_x_)
            || (dav_iter == DiagoIterAssist<T, Device>::PW_DIAG_NMAX))
        {
            ModuleBase::timer::tick("DiagoDavid", "last");

            // updata eigenvectors of Hamiltonian

            // ModuleBase::GlobalFunc::ZEROS(psi.get_pointer(), n_band * this->n_basis_max_);
            psi[this->ik_].zero();
            // TODO: Use Tensor operators
            // haozhihan repalce 2022-10-18
            gemm_op<Real, Device>()(this->ctx_,
                                      'N',
                                      'N',
                                      this->current_n_basis_,     // m: row of A,C
                                      this->n_band_,        // n: col of B,C
                                      nbase,               // k: col of A, row of B
                                      this->one_.data<T>(),
                                      &basis_pack[0][0][0], // A dim * nbase
                                      this->current_n_basis_,
                                      this->vcc_.data<T>(),           // B nbase * n_band
                                      this->n_base_x_,
                                      this->zero_.data<T>(),
                                      &psi_pack[this->ik_][0][0],   // C dim * n_band
                                      this->n_basis_max_
            );

            if (!this->notconv_ || (dav_iter == DiagoIterAssist<T, Device>::PW_DIAG_NMAX))
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

                this->refresh(this->current_n_basis_,
                              this->n_band_,
                              nbase,
                              eigen_in,
                              psi,
                              basis,
                              this->hphi_,
                              this->sphi_,
                              this->hcc_,
                              this->scc_,
                              this->vcc_);
                ModuleBase::timer::tick("DiagoDavid", "last");
            }

        } // end of if

    } while (1);

    DiagoIterAssist<T, Device>::avg_iter += static_cast<double>(dav_iter);

    ModuleBase::timer::tick("DiagoDavid", "diag_mock");

    return;
}

template <typename T, typename Device>
void DiagoDavid<T, Device>::cal_grad(hamilt::Hamilt* phm_in,
                                          const int& dim,
                                          const int& nbase, // current dimension of the reduced basis
                                          const int& notconv,
                                          ct::Tensor& basis,
                                          ct::Tensor& hphi,
                                          ct::Tensor& sphi,
                                          const ct::Tensor& vcc,
                                          const std::vector<int>& unconv,
                                          const ct::Tensor& eigen)
{
    if (test_david_ == 1)
        ModuleBase::TITLE("DiagoDavid", "cal_grad");
    if (notconv == 0)
        return;
    ModuleBase::timer::tick("DiagoDavid", "cal_grad");

    auto basis_pack = basis.accessor<T, 3>();

    // expand the reduced basis set with the new basis vectors P|Real(psi)>...
    // in which psi are the last eigenvectors
    // we define |Real(psi)> as (H-ES)*|Psi>, E = <psi|H|psi>/<psi|S|psi>

    // ModuleBase::ComplexMatrix vc_ev_vector(notconv, nbase);
    auto vc_ev_vector = ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {notconv, nbase});
    vc_ev_vector.zero();

    auto vc_ev_vector_pack = vc_ev_vector.accessor<T, 2>();

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // for (int m = 0; m < notconv; m++)
    // {
    //     for (int i = 0; i < nbase; i++)
    //     {
    //         // vc_ev_vector(m, i) = vc(i, unconv[m]);
    //         vc_ev_vector[m * nbase + i] = vcc[i * this->n_base_x_ + unconv[m]];
    //     }
    // }
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // replace by haozhihan
    for (int m = 0; m < notconv; m++)
    {
        vc_ev_vector[m].sync(
            ct::TensorMap(vcc[unconv[m]].data<T>(), vc_ev_vector.data_type(), vc_ev_vector.device_type(), {nbase}));
    }


    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan repalce 2022-10-18
    gemm_op<Real, Device>()(this->ctx_,
                              'N',
                              'N',
                              this->current_n_basis_, // m: row of A,C
                              notconv, // n: col of B,C
                              nbase, // k: col of A, row of B
                              this->one_.data<T>(), // alpha
                              hphi.data<T>(), // A dim * nbase
                              this->current_n_basis_, // LDA: if(N) max(1,m) if(T) max(1,k)
                              vc_ev_vector.data<T>(), // B nbase * notconv
                              nbase, // LDB: if(N) max(1,k) if(T) max(1,n)
                              this->zero_.data<T>(), // belta
                              &basis_pack[0][nbase][0], // C dim * notconv
                              this->current_n_basis_ // LDC: if(N) max(1, m)
    );
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // for (int m = 0; m < notconv; m++)
    // {
    //     for (int i = 0; i < nbase; i++)
    //     {
    //         vc_ev_vector[m * nbase + i] *= -1 * this->eigenvalue[unconv[m]];
    //     }
    // }
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan replace 2022.11.18
    for (int m = 0; m < notconv; m++)
    {
        std::vector<Real> e_temp_cpu(nbase, (-1.0 * this->eigen_.data<Real>()[unconv[m]]));

        if (this->device == psi::GpuDevice)
        {
#if defined(__CUDA) || defined(__ROCM)
            Real* e_temp_gpu = nullptr;
            resmem_var_op()(this->ctx_, e_temp_gpu, nbase);
            syncmem_var_h2d_op()(this->ctx_, this->cpu_ctx_, e_temp_gpu, e_temp_cpu.data(), nbase);
            vector_mul_vector_op<Real, Device>()(this->ctx_,
                                                   nbase,
                                                   &vc_ev_vector_pack[m][0],
                                                   &vc_ev_vector_pack[m][0],
                                                   e_temp_gpu);
            delmem_var_op()(this->ctx_, e_temp_gpu);
#endif
        }
        else
        {
            vector_mul_vector_op<Real, Device>()(this->ctx_,
                                                   nbase,
                                                   &vc_ev_vector_pack[m][0],
                                                   &vc_ev_vector_pack[m][0],
                                                   e_temp_cpu.data());
        }
    }
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan repalce 2022-10-18
    gemm_op<Real, Device>()(this->ctx_,
                            'N',
                            'N',
                            this->current_n_basis_, // m: row of A,C
                            notconv, // n: col of B,C
                            nbase, // k: col of A, row of B
                            this->one_.data<T>(), // alpha
                            sphi.data<T>(), // A
                            this->current_n_basis_, // LDA: if(N) max(1,m) if(T) max(1,k)
                            vc_ev_vector.data<T>(), // B
                            nbase, // LDB: if(N) max(1,k) if(T) max(1,n)
                            this->one_.data<T>(), // belta
                            &basis_pack[0][nbase][0], // C dim * notconv
                            this->current_n_basis_ // LDC: if(N) max(1, m)
    );
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    for (int m = 0; m < notconv; m++)
    {
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // haozhihan replace 2022-10-18
        if (this->device == psi::GpuDevice)
        {
#if defined(__CUDA) || defined(__ROCM)
            vector_div_vector_op<Real, Device>()(this->ctx_,
                                                   this->current_n_basis_,
                                                   &basis_pack[0][nbase + m][0],
                                                   &basis_pack[0][nbase + m][0],
                                                   this->d_precondition);
#endif
        }
        else
        {
            vector_div_vector_op<Real, Device>()(this->ctx_,
                                                   this->current_n_basis_,
                                                   &basis_pack[0][nbase + m][0],
                                                   &basis_pack[0][nbase + m][0],
                                                   this->prec_.data<Real>());
        }
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // for (int ig = 0; ig < this->current_n_basis_; ig++)
        // {
        //     ppsi[ig] /= this->precondition[ig];
        // }
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    }

    // there is a nbase to nbase + notconv band orthogonalise
    // plan for SchmitOrth
    auto lagrange = ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {notconv, nbase + notconv});
    lagrange.zero();

    std::vector<int> pre_matrix_mm_m(notconv, 0);
    std::vector<int> pre_matrix_mv_m(notconv, 1);
    this->planSchmitOrth(notconv, pre_matrix_mm_m, pre_matrix_mv_m);
    for (int m = 0; m < notconv; m++)
    {
        if(GlobalV::use_paw)
        {
#ifdef USE_PAW
            GlobalC::paw_cell.paw_nl_psi(1,reinterpret_cast<const std::complex<double>*> (&basis_pack[0][nbase + m][0]),
                reinterpret_cast<std::complex<double>*>(&sphi[(nbase + m) * this->current_n_basis_]));
#endif
        }
        else
        {
            auto sphi_m = sphi[nbase + m];
            phm_in->sPsi(basis[0][nbase + m], sphi_m);
        }
    }
    // first nbase bands psi* dot notconv bands spsi to prepare lagrange_matrix

    // calculate the square matrix for future lagranges
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    //  haozhihan replace 2022-10-18
    gemm_op<Real, Device>()(this->ctx_,
                              'C',
                              'N',
                              nbase, // m: row of A,C
                              notconv, // n: col of B,C
                              this->current_n_basis_, // k: col of A, row of B
                              this->one_.data<T>(), // alpha
                              &basis_pack[0][0][0], // A
                              this->current_n_basis_, // LDA: if(N) max(1,m) if(T) max(1,k)
                              &sphi.data<T>()[nbase * this->current_n_basis_], // B
                              this->current_n_basis_, // LDB: if(N) max(1,k) if(T) max(1,n)
                              this->zero_.data<T>(), // belta
                              lagrange.data<T>(), // C
                              nbase + notconv // LDC: if(N) max(1, m)
    );

    for (int m = 0; m < notconv; m++)
    {
        auto lagrange_m = ct::TensorMap(
            lagrange[m].data<T>(), lagrange.data_type(), lagrange.device_type(), {notconv - m, nbase + notconv});
        this->SchmitOrth(this->current_n_basis_,
                         nbase + notconv,
                         nbase + m,
                         basis,
                         sphi,
                         lagrange_m,
                         pre_matrix_mm_m[m],
                         pre_matrix_mv_m[m]);
        if(GlobalV::use_paw)
        {
#ifdef USE_PAW
            GlobalC::paw_cell.paw_nl_psi(1,reinterpret_cast<const std::complex<double>*> (&basis_pack[0][nbase + m][0]),
                reinterpret_cast<std::complex<double>*>(&sphi[(nbase + m) * this->current_n_basis_]));
#endif
        }
        else
        {
            auto sphi_m = sphi[nbase + m];
            phm_in->sPsi(basis[0][nbase + m], sphi_m);
        }
    }
    // calculate H|psi> for not convergence bands
    auto hphi_m = ct::TensorMap(
        hphi[nbase].data<T>(), hphi.data_type(), hphi.device_type(), {this->n_base_x_ - nbase, this->current_n_basis_});
    phm_in->ops->hPsi(basis, hphi_m);

    ModuleBase::timer::tick("DiagoDavid", "cal_grad");
    return;
}

template <typename T, typename Device>
void DiagoDavid<T, Device>::cal_elem(
    const int& current_n_basis,
    int& nbase,
    const int& notconv,
    const ct::Tensor& basis,
    const ct::Tensor& hphi,
    const ct::Tensor& sphi,
    ct::Tensor& hcc,
    ct::Tensor& scc)
{
    if (test_david_ == 1)
        ModuleBase::TITLE("DiagoDavid", "cal_elem");

    if (notconv == 0)
        return;
    ModuleBase::timer::tick("DiagoDavid", "cal_elem");

    auto basis_pack = basis.accessor<T, 3>();

    gemm_op<Real, Device>()(this->ctx_,
                              'C',
                              'N',
                              notconv,
                              nbase + notconv,
                              this->current_n_basis_,
                              this->one_.data<T>(),
                              &basis_pack[0][nbase][0], // this->current_n_basis_ * notconv
                              this->current_n_basis_,
                              hphi.data<T>(),               // this->current_n_basis_ * (nbase + notconv)
                              this->current_n_basis_,
                              this->zero_.data<T>(),
                              hcc.data<T>() + nbase,        // notconv * (nbase + notconv)
                              this->n_base_x_);

    gemm_op<Real, Device>()(this->ctx_,
                              'C',
                              'N',
                              notconv,
                              nbase + notconv,
                              this->current_n_basis_,
                              this->one_.data<T>(),
                              &basis_pack[0][nbase][0],   // this->current_n_basis_ * notconv
                              this->current_n_basis_,
                              sphi.data<T>(),               // this->current_n_basis_ * (nbase + notconv)
                              this->current_n_basis_,
                              this->zero_.data<T>(),
                              scc.data<T>() + nbase,        // notconv * (nbase + notconv)
                              this->n_base_x_);


#ifdef __MPI
    if (GlobalV::NPROC_IN_POOL > 1)
    {
        matrixTranspose_op<Real, Device>()(this->ctx_, this->n_base_x_, this->n_base_x_, hcc.data<T>(), hcc.data<T>());
        matrixTranspose_op<Real, Device>()(this->ctx_, this->n_base_x_, this->n_base_x_, scc.data<T>(), scc.data<T>());

        auto* swap = new T[notconv * this->n_base_x_];
        syncmem_complex_op()(this->ctx_, this->ctx_, swap, hcc.data<T>() + nbase * this->n_base_x_, notconv * this->n_base_x_);
        if (psi::device::get_current_precision(swap) == "single") {
            MPI_Reduce(swap, hcc.data<T>() + nbase * this->n_base_x_, notconv * this->n_base_x_, MPI_COMPLEX, MPI_SUM, 0,POOL_WORLD);
        }
        else {
            MPI_Reduce(swap, hcc.data<T>() + nbase * this->n_base_x_, notconv * this->n_base_x_, MPI_DOUBLE_COMPLEX, MPI_SUM, 0,POOL_WORLD);
        }
        syncmem_complex_op()(this->ctx_, this->ctx_, swap, scc.data<T>() + nbase * this->n_base_x_, notconv * this->n_base_x_);
        if (psi::device::get_current_precision(swap) == "single") {
            MPI_Reduce(swap, scc.data<T>() + nbase * this->n_base_x_, notconv * this->n_base_x_, MPI_COMPLEX, MPI_SUM, 0, POOL_WORLD);
        }
        else {
            MPI_Reduce(swap, scc.data<T>() + nbase * this->n_base_x_, notconv * this->n_base_x_, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, POOL_WORLD);
        }
        delete[] swap;

        // Parallel_Reduce::reduce_complex_double_pool( hcc.data<T>() + nbase * this->n_base_x_, notconv * this->n_base_x_ );
        // Parallel_Reduce::reduce_complex_double_pool( scc.data<T>() + nbase * this->n_base_x_, notconv * this->n_base_x_ );

        matrixTranspose_op<Real, Device>()(this->ctx_, this->n_base_x_, this->n_base_x_, hcc.data<T>(), hcc.data<T>());
        matrixTranspose_op<Real, Device>()(this->ctx_, this->n_base_x_, this->n_base_x_, scc.data<T>(), scc.data<T>());
    }
#endif

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
template <typename T, typename Device>
void DiagoDavid<T, Device>::diag_zhegvx(
    const int& nbase,
    const int& nband,
    const ct::Tensor& hcc,
    const ct::Tensor& scc,
    const int& nbase_x,
    ct::Tensor& eigenvalue,
    ct::Tensor& vcc)
{
    //	ModuleBase::TITLE("DiagoDavid","diag_zhegvx");
    ModuleBase::timer::tick("DiagoDavid", "diag_zhegvx");
    if (GlobalV::RANK_IN_POOL == 0)
    {
        assert(nbase_x >= std::max(1, nbase));

        if (this->device == psi::GpuDevice)
        {
#if defined(__CUDA) || defined(__ROCM)
            Real* eigenvalue_gpu = nullptr;
            resmem_var_op()(this->ctx_, eigenvalue_gpu, this->n_base_x_);
            syncmem_var_h2d_op()(this->ctx_, this->cpu_ctx_, eigenvalue_gpu, this->eigenvalue, this->n_base_x_);

            dnevx_op<Real, Device>()(this->ctx_, nbase, this->n_base_x_, this->hcc_.data<T>(), nband, eigenvalue_gpu, this->vcc_.data<T>());

            syncmem_var_d2h_op()(this->cpu_ctx_, this->ctx_, this->eigen_.data<T>(), eigenvalue_gpu, this->n_base_x_);
            delmem_var_op()(this->ctx_, eigenvalue_gpu);
#endif
        }
        else
        {
            dnevx_op<Real, Device>()(this->ctx_, nbase, this->n_base_x_, this->hcc_.data<T>(), nband, this->eigen_.data<Real>(), this->vcc_.data<T>());
        }
    }

#ifdef __MPI
    if (GlobalV::NPROC_IN_POOL > 1)
    {
        // vcc: nbase * nband
        for (int i = 0; i < nband; i++)
        {
            MPI_Bcast(vcc.data<T>() + i * this->n_base_x_, nbase, MPI_DOUBLE_COMPLEX, 0, POOL_WORLD);
        }
        MPI_Bcast(this->eigen_.data<T>(), nband, MPI_DOUBLE, 0, POOL_WORLD);
    }
#endif

    ModuleBase::timer::tick("DiagoDavid", "diag_zhegvx");
    return;
}

template <typename T, typename Device>
void DiagoDavid<T, Device>::refresh(
    const int& current_n_basis,
    const int& nband,
    int& nbase,
    const ct::Tensor& eigen_in,
    const ct::Tensor& psi,
    ct::Tensor& basis,
    ct::Tensor& hphi,
    ct::Tensor& sphi,
    ct::Tensor& hcc,
    ct::Tensor& scc,
    ct::Tensor& vcc)
{
    if (test_david_ == 1)
        ModuleBase::TITLE("DiagoDavid", "refresh");
    ModuleBase::timer::tick("DiagoDavid", "refresh");

    // update hp,sp
    basis.zero();

    auto basis_pack = basis.accessor<T, 3>();
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan repalce 2022-10-18
    gemm_op<Real, Device>()(this->ctx_,
                              'N',
                              'N',
                              this->current_n_basis_,            // m: row of A,C
                              nband,                // n: col of B,C
                              nbase,                // k: col of A, row of B
                              this->one_.data<T>(),
                              this->hphi_.data<T>(),           // A dim * nbase
                              this->current_n_basis_,
                              this->vcc_.data<T>(),            // B nbase * nband
                              this->n_base_x_,
                              this->zero_.data<T>(),
                              basis.data<T>(),  // C dim * nband
                              this->current_n_basis_
    );

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan repalce 2022-10-18
    gemm_op<Real, Device>()(this->ctx_,
                              'N',
                              'N',
                              this->current_n_basis_,                // m: row of A,C
                              nband,                    // n: col of B,C
                              nbase,                    // k: col of A, row of B
                              this->one_.data<T>(),
                              this->sphi_.data<T>(),               // A dim * nbase
                              this->current_n_basis_,
                              this->vcc_.data<T>(),                // B nbase * nband
                              this->n_base_x_,
                              this->zero_.data<T>(),
                              &basis_pack[0][nband][0],         // C dim * nband
                              this->current_n_basis_
    );

    hphi.sync(basis[0][0]);
    sphi.sync(basis[0][nband]);

    // update basis
    basis.zero();
    basis[0].sync(psi[0]);

    // updata the reduced Hamiltonian
    nbase = nband;
    // hc.zero_out();
    hcc.zero();
    // sc.zero_out();
    scc.zero();

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    if (this->device == psi::GpuDevice)
    {
#if defined(__CUDA) || defined(__ROCM)
        T* hcc_cpu = nullptr;
        T* scc_cpu = nullptr;
        T* vcc_cpu = nullptr;
        psi::memory::resize_memory_op<T, psi::DEVICE_CPU>()(this->cpu_ctx_,
                                                                               hcc_cpu,
                                                                               this->n_base_x_ * this->n_base_x_,
                                                                               "DAV::hcc");
        psi::memory::resize_memory_op<T, psi::DEVICE_CPU>()(this->cpu_ctx_,
                                                                               scc_cpu,
                                                                               this->n_base_x_ * this->n_base_x_,
                                                                               "DAV::scc");
        psi::memory::resize_memory_op<T, psi::DEVICE_CPU>()(this->cpu_ctx_,
                                                                               vcc_cpu,
                                                                               this->n_base_x_ * this->n_base_x_,
                                                                               "DAV::vcc");

        syncmem_complex_d2h_op()(this->cpu_ctx_, this->ctx_, hcc_cpu, hcc, this->n_base_x_ * this->n_base_x_);
        syncmem_complex_d2h_op()(this->cpu_ctx_, this->ctx_, scc_cpu, scc, this->n_base_x_ * this->n_base_x_);
        syncmem_complex_d2h_op()(this->cpu_ctx_, this->ctx_, vcc_cpu, vcc, this->n_base_x_ * this->n_base_x_);

        for (int i = 0; i < nbase; i++)
        {
            hcc_cpu[i * this->n_base_x_ + i] = eigenvalue_in[i];
            scc_cpu[i * this->n_base_x_ + i] = this->one_[0];
            vcc_cpu[i * this->n_base_x_ + i] = this->one_[0];
        }

        syncmem_complex_h2d_op()(this->ctx_, this->cpu_ctx_, hcc, hcc_cpu, this->n_base_x_ * this->n_base_x_);
        syncmem_complex_h2d_op()(this->ctx_, this->cpu_ctx_, scc, scc_cpu, this->n_base_x_ * this->n_base_x_);
        syncmem_complex_h2d_op()(this->ctx_, this->cpu_ctx_, vcc, vcc_cpu, this->n_base_x_ * this->n_base_x_);

        psi::memory::delete_memory_op<T, psi::DEVICE_CPU>()(this->cpu_ctx_, hcc_cpu);
        psi::memory::delete_memory_op<T, psi::DEVICE_CPU>()(this->cpu_ctx_, scc_cpu);
        psi::memory::delete_memory_op<T, psi::DEVICE_CPU>()(this->cpu_ctx_, vcc_cpu);
#endif
    }
    else
    {
        for (int i = 0; i < nbase; i++)
        {
            hcc.data<T>()[i * this->n_base_x_ + i] = eigen_in.data<T>()[i];
            // sc(i, i) = this->one_;
            scc.data<T>()[i * this->n_base_x_ + i] = this->one_.data<T>()[0];
            // vc(i, i) = this->one_;
            vcc.data<T>()[i * this->n_base_x_ + i] = this->one_.data<T>()[0];
        }
    }
    ModuleBase::timer::tick("DiagoDavid", "refresh");
    return;
}

template <typename T, typename Device>
void DiagoDavid<T, Device>::SchmitOrth(
    const int& current_n_basis,
    const int& nband,
    const int& m,
    ct::Tensor& basis,
    const ct::Tensor& sphi,
    ct::Tensor& lagrange_m,
    const int& mm_size,
    const int& mv_size)
{
    //	if(test_david == 1) ModuleBase::TITLE("DiagoDavid","SchmitOrth");
    ModuleBase::timer::tick("DiagoDavid", "SchmitOrth");

    // orthogonalize starting eigenfunction to those already calculated
    // psi_m orthogonalize to psi(0) ~ psi(m-1)
    // Attention, the orthogonalize here read as
    // psi(m) -> psi(m) - \sum_{i < m} \langle psi(i)|S|psi(m) \rangle psi(i)
    // so the orthogonalize is performed about S.

    assert(this->n_band_ >= nband);
    assert(m >= 0);
    assert(m < nband);

    auto sphi_pack = sphi.accessor<T, 2>();
    auto basis_pack = basis.accessor<T, 3>();
    auto lagrange_m_pack = lagrange_m.accessor<T, 2>();

    // std::complex<double> *lagrange = new std::complex<double>[m + 1];
    // ModuleBase::GlobalFunc::ZEROS(lagrange, m + 1);

    // calculate the square matrix for future lagranges
    if (mm_size != 0)
    {
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // haozhihan repalce 2022-10-16
        gemm_op<Real, Device>()(this->ctx_,
                                'C',
                                'N',
                                mm_size, // m: row of A,C
                                mm_size, // n: col of B,C
                                this->current_n_basis_, // k: col of A, row of B
                                this->one_.data<T>(), // alpha
                                &basis_pack[0][m - mm_size + 1][0], // A
                                this->current_n_basis_, // LDA: if(N) max(1,m) if(T) max(1,k)
                                &sphi_pack[m][0], // B
                                this->current_n_basis_, // LDB: if(N) max(1,k) if(T) max(1,n)
                                this->zero_.data<T>(), // belta
                                &lagrange_m_pack[m - mv_size + 1 - mm_size][0], // C
                                nband // LDC: if(N) max(1, m)
        );
    }
    // calculate other lagranges for this band
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    //  haozhihan repalce 2022-10-16
    gemv_op<Real, Device>()(this->ctx_,
                              'C',
                              this->current_n_basis_,
                              mv_size,
                              this->one_.data<T>(),
                              &basis_pack[0][m - mv_size + 1][0],
                              this->current_n_basis_,
                              &sphi_pack[m][0],
                              1,
                              this->zero_.data<T>(),
                              &lagrange_m_pack[m - mv_size + 1][0],
                              1);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    Parallel_Reduce::reduce_complex_double_pool(lagrange_m.data<T>(), m + 1);

    T var = {0, 0};
    syncmem_complex_d2h_op()(this->cpu_ctx_, this->ctx_, &var, lagrange_m.data<T>() + m, 1);
    double psi_norm = var.real();

    assert(psi_norm > 0.0);

    // haozhihan replace 2022-10-24
    gemv_op<Real, Device>()(this->ctx_,
                            'N',
                            this->current_n_basis_,
                            m,
                            this->neg_one_.data<T>(),
                            &basis_pack[0][0][0],
                            this->current_n_basis_,
                            &lagrange_m_pack[0][0],
                            1,
                            this->one_.data<T>(),
                            &basis_pack[0][m][0],
                            1);

    psi_norm -= zdot_real_op<Real, Device>()(this->ctx_, m, lagrange_m.data<T>(), lagrange_m.data<T>(), false);

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
        std::cout << "nband = " << nband << std::endl;
        std::cout << "m = " << m << std::endl;
        exit(0);
    }
    else
    {
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // haozhihan repalce 2022-10-16
        vector_div_constant_op<Real, Device>()(this->ctx_, this->current_n_basis_, &basis_pack[0][m][0], &basis_pack[0][m][0], psi_norm);
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // for (int i = 0; i < npw; i++)
        // {
        //     psi_m[i] /= psi_norm;
        // }
    }

    // delete[] lagrange;
    ModuleBase::timer::tick("DiagoDavid", "SchmitOrth");
    return;
}

template <typename T, typename Device>
void DiagoDavid<T, Device>::planSchmitOrth(const int& nband, std::vector<int>& pre_matrix_mm_m, std::vector<int>& pre_matrix_mv_m)
{
    if (nband <= 0)
        return;
    ModuleBase::GlobalFunc::ZEROS(pre_matrix_mm_m.data(), nband);
    ModuleBase::GlobalFunc::ZEROS(pre_matrix_mv_m.data(), nband);
    int last_matrix_size = nband;
    int matrix_size = int(nband / 2);
    int divide_times = 0;
    std::vector<int> divide_points(nband);
    int res_nband = nband - matrix_size;
    while (matrix_size > 1)
    {
        int index = nband - matrix_size;
        if (divide_times == 0)
        {
            divide_points[0] = index;
            pre_matrix_mm_m[index] = matrix_size;
            if (res_nband == matrix_size)
                pre_matrix_mv_m[index] = 1;
            else
                pre_matrix_mv_m[index] = 2;
            divide_times = 1;
        }
        else
        {
            for (int i = divide_times - 1; i >= 0; i--)
            {
                divide_points[i * 2] = divide_points[i] - matrix_size;
                divide_points[i * 2 + 1] = divide_points[i * 2] + last_matrix_size;
                pre_matrix_mm_m[divide_points[i * 2]] = matrix_size;
                pre_matrix_mm_m[divide_points[i * 2 + 1]] = matrix_size;
                if (res_nband == matrix_size)
                {
                    pre_matrix_mv_m[divide_points[i * 2]] = 1;
                    pre_matrix_mv_m[divide_points[i * 2 + 1]] = 1;
                }
                else
                {
                    pre_matrix_mv_m[divide_points[i * 2]] = 2;
                    pre_matrix_mv_m[divide_points[i * 2 + 1]] = 2;
                }
            }
            divide_times *= 2;
        }
        last_matrix_size = matrix_size;
        matrix_size = int(res_nband / 2);
        res_nband -= matrix_size;
    }
    // fill the pre_matrix_mv_m array
    pre_matrix_mv_m[0] = 1;
    for (int m = 1; m < nband; m++)
    {
        if (pre_matrix_mv_m[m] == 0)
        {
            pre_matrix_mv_m[m] = pre_matrix_mv_m[m - 1] + 1;
        }
    }
}

template <typename T, typename Device>
void DiagoDavid<T, Device>::diag(
    hamilt::Hamilt* phm_in,
    ct::Tensor& psi,
    ct::Tensor& eigen_in)
{
    /// record the times of trying iterative diagonalization
    int ntry = 0;
    this->notconv_ = 0;

#if defined(__CUDA) || defined(__ROCM)
    if (this->device == psi::GpuDevice)
    {
        resmem_var_op()(this->ctx_, this->d_precondition, psi.get_nbasis());
        syncmem_var_h2d_op()(this->ctx_, this->cpu_ctx_, this->d_precondition, this->precondition, psi.get_nbasis());
    }
#endif

    do
    {
        this->diag_mock(phm_in, psi, eigen_in);
        ++ntry;
    } while (DiagoIterAssist<T, Device>::test_exit_cond(ntry, this->notconv_));

    if (notconv_ > std::max(5, this->n_band_ / 4))
    {
        std::cout << "\n notconv = " << this->notconv_;
        std::cout << "\n DiagoDavid::diag', too many bands are not converged! \n";
    }
    return;
}

namespace hsolver {
template class DiagoDavid<std::complex<float>, psi::DEVICE_CPU>;
template class DiagoDavid<std::complex<double>, psi::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoDavid<std::complex<float>, psi::DEVICE_GPU>;
template class DiagoDavid<std::complex<double>, psi::DEVICE_GPU>;
#endif
#ifdef __LCAO
template class DiagoDavid<double, psi::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoDavid<double, psi::DEVICE_GPU>;
#endif
#endif
} // namespace hsolver
