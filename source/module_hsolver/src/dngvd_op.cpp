#include "module_hsolver/include/dngvd_op.h"

#include <algorithm>

namespace hsolver
{

template <>
void dngvd_op<double, psi::DEVICE_CPU>::operator()(
        const psi::DEVICE_CPU* d,
        const int row,
        const int col,
        const std::complex<double>* A,
        const std::complex<double>* B,
        const int m,
        double* W, 
        std::complex<double>* V)
{
    int lwork;
    int info = 0;

    std::string name1 = "ZHETRD";
    std::string name2 = "L";

    // TODO check col?
    int nb = LapackConnector::ilaenv(1, name1.c_str(), name2.c_str(), col, -1, -1, -1);
    if (nb < 1)
    {
        nb = std::max(1, col);
    }

    if (nb == 1 || nb >= col)
    {
        lwork = 2 * col; // qianrui fix a bug 2021-7-25 : lwork should be at least max(1,2*n)
    } else
    {
        lwork = (nb + 1) * col;
    }

    std::complex<double> *work = new std::complex<double>[2 * lwork];
    assert(work != 0);
    double *rwork = new double[7 * col];
    assert(rwork != 0);
    int *iwork = new int[5 * col];
    assert(iwork != 0);
    int *ifail = new int[col];
    assert(ifail != 0);
    ModuleBase::GlobalFunc::ZEROS(work, lwork); // qianrui change it, only first lwork numbers are used in zhegvx
    ModuleBase::GlobalFunc::ZEROS(rwork, 7 * col);
    ModuleBase::GlobalFunc::ZEROS(iwork, 5 * col);
    ModuleBase::GlobalFunc::ZEROS(ifail, col);

    LapackConnector::zhegvx(1,       // ITYPE = 1:  A*x = (lambda)*B*x
                            'V',     // JOBZ = 'V':  Compute eigenvalues and eigenvectors.
                            'I',     // RANGE = 'I': the IL-th through IU-th eigenvalues will be found.
                            'L',     // UPLO = 'L':  Lower triangles of A and B are stored.
                            col,     // N = base 
                            A,       // A is COMPLEX*16 array  dimension (LDA, N)
                            col,     // LDA = base
                            B,       // B is COMPLEX*16 array, dimension (LDB, N)
                            col,     // LDB = base
                            0.0,     // Not referenced if RANGE = 'A' or 'I'.
                            0.0,     // Not referenced if RANGE = 'A' or 'I'.
                            1,       // IL: If RANGE='I', the index of the smallest eigenvalue to be returned. 1 <= IL <= IU <= N,
                            m,       // IU: If RANGE='I', the index of the largest eigenvalue to be returned. 1 <= IL <= IU <= N,
                            0.0,     // ABSTOL
                            m,       // M: The total number of eigenvalues found.  0 <= M <= N. if RANGE = 'I', M = IU-IL+1.
                            W,       // W store eigenvalues
                            V,       // store eigenvector
                            col,     // LDZ: The leading dimension of the array Z.
                            work,
                            lwork,
                            rwork,
                            iwork,
                            ifail,
                            info,
                            row);

    delete[] work;
    delete[] rwork;
    delete[] iwork;
    delete[] ifail;

};


} // namespace hsolver