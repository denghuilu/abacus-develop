#ifndef OPERATOR_H
#define OPERATOR_H

#include<complex>
#include "module_psi/psi.h"
#include "module_base/global_function.h"
#include "module_base/tool_quit.h"

#include <ATen/core/tensor.h>

namespace hamilt
{

enum calculation_type
{
    no,
    pw_ekinetic, 
    pw_nonlocal,
    pw_veff,
    pw_meta,
    lcao_overlap,
    lcao_fixed,
    lcao_gint,
    lcao_deepks,
    lcao_exx,
    lcao_dftu,
};

// Basic class for operator module, 
// it is designed for "O|psi>" and "<psi|O|psi>"
// Operator "O" might have several different types, which should be calculated one by one.
// In basic class , function add() is designed for combine all operators together with a chain. 
class Operator
{
    public:
    Operator();
    virtual ~Operator();

    //this is the core function for Operator
    // do H|psi> from input |psi> , 

    /// as default, different operators donate hPsi independently
    /// run this->act function for the first operator and run all act() for other nodes in chain table 
    /// if this procedure is not suitable for your operator, just override this function.
    /// output of hpsi would be first member of the returned tuple 
    virtual void hPsi(const ct::Tensor* psi, ct::Tensor* h_psi) const;

    virtual void init(int ik_in);

    virtual void add(Operator* next);

    virtual int get_ik() const { return this->ik; }

    ///do operation : |hpsi_choosed> = V|psi_choosed>
    ///V is the target operator act on choosed psi, the consequence should be added to choosed hpsi
    virtual void act(const int64_t nbands,
        const int64_t nbasis,
        const int npol,
        const ct::Tensor* tmpsi_in,
        ct::Tensor* tmhpsi,
        const int ngk_ik) const {};

    /// an developer-friendly interface for act() function
    virtual ct::Tensor act(const ct::Tensor& psi_in) const { return psi_in; };

    Operator* next_op = nullptr;

    protected:
    int ik = 0;

    mutable bool in_place = false;

    //calculation type, only different type can be in main chain table 
    enum calculation_type cal_type = calculation_type::no;
    Operator* next_sub_op = nullptr;
    bool is_first_node = true;

    //if this Operator is first node in chain table, hpsi would not be empty
    mutable ct::Tensor* hpsi = nullptr;

    int npol_ = 0;
    int* ngk_ = nullptr;
};

}//end namespace hamilt

#endif