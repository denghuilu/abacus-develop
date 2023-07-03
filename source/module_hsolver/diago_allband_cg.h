#ifndef DIAGO_ALL_BAND_CG_H
#define DIAGO_ALL_BAND_CG_H

#include "diagh.h"
#include "module_base/complexmatrix.h"
#include "module_hamilt_pw/hamilt_pwdft/structure_factor.h"

#include "module_psi/kernels/types.h"
#include "module_psi/kernels/device.h"
#include "module_psi/kernels/memory_op.h"

#include "module_hsolver/kernels/math_kernel_op.h"
#include "module_hsolver/kernels/dngvd_op.h"

namespace hsolver {

/**
 * @class DiagoAllBandCG
 * @brief A class for diagonalization  using the All-Band CG method.
 * @tparam FPTYPE The floating-point type used for calculations.
 * @tparam Device The device used for calculations (e.g., cpu or gpu).
 */
template<typename FPTYPE = double, typename Device = psi::DEVICE_CPU>
class DiagoAllBandCG : public DiagH<FPTYPE, Device>
{
  // Column major psi in this class
  public:
    /**
     * @brief Constructor for DiagoAllBandCG class.
     *
     * @param precondition precondition data passed by the "Hamilt_PW" class.
     */
    explicit DiagoAllBandCG(const FPTYPE * precondition);

    /**
     * @brief Destructor for DiagoAllBandCG class.
     */
    ~DiagoAllBandCG();

    /**
     * @brief Initialize the class before diagonalization.
     *
     * This function allocates all the related variables, such as hpsi, hsub, before the diag call.
     * It is called by the HsolverPW::initDiagh() function.
     *
     * @param psi_in The input wavefunction psi.
     */
    void init_iter(const psi::Psi<std::complex<FPTYPE>, Device> &psi_in);

    /**
     * @brief Diagonalize the Hamiltonian using the CG method.
     *
     * This function is an override function for the CG method. It is called by the HsolverPW::solve() function.
     *
     * @param phm_in A pointer to the hamilt::Hamilt object representing the Hamiltonian operator.
     * @param psi The input wavefunction psi matrix with [dim: n_basis x n_band, column major].
     * @param eigenvalue_in Pointer to the eigen array with [dim: n_band, column major].
     */
    void diag(hamilt::Hamilt<FPTYPE, Device> *phm_in, psi::Psi<std::complex<FPTYPE>, Device> &psi, FPTYPE *eigenvalue_in) override;


  private:
    /// col size for input psi matrix, column major.
    int n_band = 0;
    /// row size for input psi matrix, column major.
    int n_basis = 0;
    /// lda for input psi matrix, column major.
    int n_basis_max = 0;
    /// max iter steps for all-band cg loop
    int nline = 4;
    /// cg convergence thr
    FPTYPE all_band_cg_thr = 1E-5;


    /// Precondition data, reference to h_prec pointer or d_prec pointer.
    /// Note: this pointer does not own memory but instead references either h_prec (for CPU runtime)
    /// or d_prec (for GPU runtime), depending on the device type used in this class.
    /// Dim: n_basis_max, column major.
    const FPTYPE *prec = nullptr;
    /// Host precondition data, reference to the `precondition` parameter of the constructor.
    /// Note: this pointer does not own memory.
    /// Dim: n_basis_max, column major.
    const FPTYPE *h_prec = nullptr;
    /// Device precondition data.
    /// Note: Copy precondition data from h_prec to d_prec.
    /// Dim: n_basis_max, column major.
    FPTYPE *d_prec = nullptr;
    /// The coefficient for mixing the current and previous step gradients, used in iterative methods.
    /// Dim: n_band, column major.
    FPTYPE *beta = nullptr;
    /// Error state value, if it is smaller than the given threshold, then exit the iteration.
    /// Dim: n_band, column major.
    FPTYPE *err_st = nullptr;
    /// Calculated eigen
    /// Dim: n_band, column major.
    FPTYPE *eigen = nullptr;

    /// Pointer to the input wavefunction.
    /// Note: this pointer does not own memory, instead it ref the psi_in object.
    /// Dim: n_basis * n_band, column major, lda = n_basis_max.
    std::complex<FPTYPE>* psi = nullptr;
    /// H|psi> matrix.
    /// wDim: n_basis * n_band, column major, lda = n_basis_max.
    std::complex<FPTYPE>* hpsi = nullptr;
    /// <psi_i|H|psi_j> matrix.
    /// Dim: n_basis * n_band, column major, lda = n_basis_max.
    std::complex<FPTYPE>* hsub = nullptr;
    /// H|psi> - epsilo * psi, grad of the given problem.
    /// Dim: n_basis * n_band, column major, lda = n_basis_max.
    std::complex<FPTYPE>* grad = nullptr;
    /// H|grad> matrix.
    /// Dim: n_basis * n_band, column major, lda = n_basis_max.
    std::complex<FPTYPE>* hgrad = nullptr;
    /// Store the last step grad, used in iterative methods.
    /// Dim: n_basis * n_band, column major, lda = n_basis_max.
    std::complex<FPTYPE>* grad_old = nullptr;
    /// work for some calculations within this class, including rotate_wf call
    /// Dim: n_basis x n_band, column major, lda = n_basis_max.
    std::complex<FPTYPE>* work = nullptr;

    /**
     * @brief Specify which device(currently cpu or gpu) used in the calculations of this class.
     *
     * It controls the ops used in this class to use the corresponding device to calculate results
     */
    Device * ctx = {};
    /// CPU device type, template type
    psi::DEVICE_CPU *cpu_ctx = {};
    /// Current device used in this class
    psi::AbacusDevice_t device = {};

    psi::Psi<std::complex<FPTYPE>, Device> * grad_wrapper;
    /**
     * @brief Update the precondition array.
     *
     * This function updates the precondition array by copying the host precondition
     * to the device in a 'gpu' runtime environment. The address of the precondition
     * array is passed by the constructor function called by hsolver::HSolverPW::initDiagh.
     * The precondition will be updated before the DiagoAllBandCG<FPTYPE, Device>::diag call.
     *
     * @note prec[dim: n_band]
     *
     * @param dev Reference to the AbacusDevice_t object, speciy which device used in the calc_prec function.
     * @param prec Pointer to the host precondition array with [dim: n_band, column major]
     * @param h_prec Pointer to the host precondition array with [dim: n_band, column major].
     * @param d_prec Pointer to the device precondition array with [dim: n_band, column major].
     */
    void calc_prec();

    /**
     *
     * @brief Apply the H operator to psi and obtain the hpsi matrix.
     *
     * This function calculates the matrix product of the Hamiltonian operator (H) and the input wavefunction (psi).
     * The resulting matrix is stored in the output array hpsi_out.
     *
     * @note hpsi = H|psi>;
     *
     * psi_in[dim: n_basis x n_band, column major, lda = n_basis_max],
     * hpsi_out[dim: n_basis x n_band, column major, lda = n_basis_max].
     *
     * @param hamilt_in A pointer to the hamilt::Hamilt object representing the Hamiltonian operator.
     * @param psi_in The input wavefunction psi.
     * @param hpsi_out Pointer to the array where the resulting hpsi matrix will be stored.
     */
    void calc_hpsi_all_band (hamilt::Hamilt<FPTYPE, Device> *hamilt_in, const psi::Psi<std::complex<FPTYPE>, Device> &psi_in,  std::complex<FPTYPE> * hpsi_out );

    /**
     * @brief Diagonalization of the subspace matrix.
     *
     * All the matrix used in this function are stored and used as the column major.
     * psi_in[dim: n_basis x n_band, column major, lda = n_basis_max],
     * hpsi_in[dim: n_basis x n_band, column major, lda = n_basis_max],
     * hpsi_out[dim: n_basis x n_band, column major, lda = n_basis_max],
     * eigenvalue_out[dim: n_basis_max, column major].
     *
     * @param psi_in Input wavefunction matrix with [dim: n_basis x n_band, column major].
     * @param hpsi_in H|psi> matrix with [dim: n_basis x n_band, column major].
     * @param hsub_out Output Hamiltonian subtracted matrix with [dim: n_band x n_band, column major]
     * @param eigenvalue_out Computed eigen array with [dim: n_band]
     */
    void diag_hsub(const std::complex<FPTYPE> * psi_in, const std::complex<FPTYPE> * hpsi_in,
                   std::complex<FPTYPE> * hsub_out, FPTYPE * eigenvalue_out);

    /**
     * @brief Inplace matrix multiplication to obtain the initial guessed wavefunction.
     *
     * hsub_in[dim: n_band x n_band, column major, lda = n_band],
     * workspace_in[dim: n_basis x n_band, column major, lda = n_basis_max],
     * psi_out[dim: n_basis x n_band, column major, lda = n_basis_max],
     *
     * @param hsub_in Subspace matrix input, dim [n_basis, n_band] with column major.
     * @param workspace_in Workspace matrix, dim [n_basis, n_band] with column major..
     * @param psi_out output wavefunction matrix with dim [n_basis, n_band], column major.
     */
    void rotate_wf(const std::complex<FPTYPE> * hsub_in,
                   std::complex<FPTYPE> * psi_out, std::complex<FPTYPE> * workspace_in);

    /**
     * @brief Calculate the gradient for all bands used in CG method.
     *
     * prec_in[dim: n_basis_max, column major],
     * err_out[dim: n_band, column major],
     * beta_out[dim: n_band, column major],
     * psi_in[dim: n_basis x n_band, column major, lda = n_basis_max],
     * hpsi_in[dim: n_basis x n_band, column major, lda = n_basis_max],
     * grad_out[dim: n_basis x n_band, column major, lda = n_basis_max],
     * grad_old_out[dim: n_basis x n_band, column major, lda = n_basis_max],
     *
     * @param prec_in Input preconditioner.
     * @param err_out Output error state value. If it is smaller than a given threshold, exit the iteration.
     * @param beta_out Output beta coefficient.
     * @param psi_in Input wavefunction matrix.
     * @param hpsi_in Product of psi_in and Hamiltonian.
     * @param grad_out Output gradient matrix.
     * @param grad_old_out Previous gradient matrix.
     * @note The steps involved in optimization are:
     *   1. normalize psi
     *   2. calculate the epsilo
     *   3. calculate the gradient by hpsi - epsilo * psi
     *   4. gradient mix with the previous gradient
     *   5. Do precondition
     */
    void calc_grad_all_band(const FPTYPE *prec_in, FPTYPE *err_out, FPTYPE *beta_out,
                            complex<FPTYPE> *psi_in, complex<FPTYPE> *hpsi_in,
                            complex<FPTYPE> *grad_out, complex<FPTYPE> *grad_old_out);

    /**
     *
     * @brief Apply the Hamiltonian operator to psi and obtain the hpsi matrix.
     *
     * psi_out[dim: n_basis x n_band, column major, lda = n_basis_max],
     * hpsi_out[dim: n_basis x n_band, column major, lda = n_basis_max],
     * hsub_out[dim: n_band x n_band, column major, lda = n_band],
     * eigenvalue_out[dim: n_basis_max, column major].
     *
     * @param hamilt_in Pointer to the Hamiltonian object.
     * @param psi_in Input wavefunction.
     * @param psi_out Output wavefunction.
     * @param hpsi_out Product of psi_out and Hamiltonian.
     * @param hsub_out Subspace matrix output.
     * @param eigenvalue_out Computed eigen.
     */
    void calc_hsub_all_band(hamilt::Hamilt<FPTYPE, Device> *hamilt_in,
                        const psi::Psi<std::complex<FPTYPE>, Device> &psi_in,
                        std::complex<FPTYPE> * psi_out, std::complex<FPTYPE> * hpsi_out,
                        std::complex<FPTYPE> * hsub_out, std::complex<FPTYPE> * workspace_in,
                        FPTYPE * eigenvalue_out);

    /**
     * @brief Orthogonalize column vectors in grad to column vectors in psi.
     *
     * hsub_in and workspace_in are only used to store intermediate variables of the gemm operator.
     *
     * @param psi_in Input wavefunction array, [dim: n_basis x n_band, column major, lda = n_basis_max].
     * @param hsub_in Subspace matrix input, [dim: n_band x n_band, column major, lda = n_band].
     * @param grad_out Input and output gradient array, [dim: n_basis x n_band, column major, lda = n_basis_max]..
     * @note This function is a member of the DiagoAllBandCG class.
     */
    void orth_projection(const std::complex<FPTYPE> * psi_in,
            std::complex<FPTYPE> * hsub_in,
            std::complex<FPTYPE> * grad_out);

    /**
     *
     *@brief Optimize psi as well as the hpsi.
     *
     *@param grad_in Input gradient array, [dim: n_basis x n_band, column major, lda = n_basis_max].
     *@param hgrad_in Product of grad_in and Hamiltonian, [dim: n_basis x n_band, column major, lda = n_basis_max].
     *@param psi_out Input and output wavefunction array, [dim: n_basis x n_band, column major, lda = n_basis_max].
     *@param hpsi_out Product of psi_out and Hamiltonian, [dim: n_basis x n_band, column major, lda = n_basis_max].
     *@note The steps involved in optimization are:
     *  1. Normalize the gradient.
     *  2. Calculate theta.
     *  3. Update psi as well as hpsi.
     */
    void line_minimize(std::complex<FPTYPE> * grad_in,
            std::complex<FPTYPE> * hgrad_in,
            std::complex<FPTYPE> * psi_out,
            std::complex<FPTYPE> * hpsi_out);

    /**
     * @brief Orthogonalize and normalize the column vectors in psi_out using Cholesky decomposition.
     *
     * @param workspace_in Workspace memory, [dim: n_basis x n_band, column major, lda = n_basis_max]..
     * @param psi_out Input and output wavefunction array. [dim: n_basis x n_band, column major, lda = n_basis_max].
     * @param hpsi_out Input and output hpsi array. [dim: n_basis x n_band, column major, lda = n_basis_max].
     * @param hsub_out Input Hamiltonian product array. [dim: n_band x n_band, column major, lda = n_band].
     */
    void orth_cholesky(std::complex<FPTYPE> * workspace_in, std::complex<FPTYPE> * psi_out, std::complex<FPTYPE> * hpsi_out, std::complex<FPTYPE> * hsub_out);

    /**
     * @brief Checks if the error satisfies the given threshold.
     *
     * @param err_in Pointer to the error array.[dim: n_band, column major]
     * @param thr_in The threshold.
     * @return Returns true if all error values are less than or equal to the threshold, false otherwise.
     */
    bool test_error(const FPTYPE * err_in, FPTYPE thr_in);

    using hpsi_info = typename hamilt::Operator<std::complex<FPTYPE>, Device>::hpsi_info;

    using setmem_var_op = psi::memory::set_memory_op<FPTYPE, Device>;
    using resmem_var_op = psi::memory::resize_memory_op<FPTYPE, Device>;
    using delmem_var_op = psi::memory::delete_memory_op<FPTYPE, Device>;
    using syncmem_var_h2d_op = psi::memory::synchronize_memory_op<FPTYPE, Device, psi::DEVICE_CPU>;
    using syncmem_var_d2h_op = psi::memory::synchronize_memory_op<FPTYPE, psi::DEVICE_CPU, Device>;

    using setmem_complex_op = psi::memory::set_memory_op<std::complex<FPTYPE>, Device>;
    using delmem_complex_op = psi::memory::delete_memory_op<std::complex<FPTYPE>, Device>;
    using resmem_complex_op = psi::memory::resize_memory_op<std::complex<FPTYPE>, Device>;
    using syncmem_complex_op = psi::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, Device>;

    using dnevd_op = hsolver::dnevd_op<FPTYPE, Device>;
    using zpotrf_op = hsolver::zpotrf_op<FPTYPE, Device>;
    using ztrtri_op = hsolver::ztrtri_op<FPTYPE, Device>;
    using set_matrix_op = hsolver::set_matrix_op<FPTYPE, Device>;
    using mat_add_inplace_op = hsolver::mat_add_inplace_op<FPTYPE, Device>;
    using calc_grad_all_band_op = hsolver::calc_grad_all_band_op<FPTYPE, Device>;
    using line_minimize_all_band_op = hsolver::line_minimize_all_band_op<FPTYPE, Device>;

    const std::complex<FPTYPE> * one = nullptr, * zero = nullptr, * neg_one = nullptr;
};

} // namespace hsolver
#endif // DIAGO_ALL_BAND_CG_H