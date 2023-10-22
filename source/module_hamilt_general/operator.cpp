#include "operator.h"

#include "module_base/timer.h"

using namespace hamilt;


Operator::Operator() = default;

Operator::~Operator()
{
    delete this->hpsi;
    Operator* last = this->next_op;
    Operator* last_sub = this->next_sub_op;
    while(last != nullptr || last_sub != nullptr)
    {
        if(last_sub != nullptr)
        {//delete sub_chain first
            Operator* node_delete = last_sub;
            last_sub = last_sub->next_sub_op;
            node_delete->next_sub_op = nullptr;
            delete node_delete;
        }
        else
        {//delete main chain if sub_chain is deleted
            Operator* node_delete = last;
            last_sub = last->next_sub_op;
            node_delete->next_sub_op = nullptr;
            last = last->next_op;
            node_delete->next_op = nullptr;
            delete node_delete;
        }
    }
}

void Operator::hPsi(const ct::Tensor& psi, ct::Tensor& h_psi) const
{
    ModuleBase::timer::tick("Operator", "hPsi");
    // psi are stored as a dim-3 tensor with shape [k, band, g]
    auto nbands = psi.shape().dim_size(1);
    auto nbasis = psi.shape().dim_size(2);

    // how to get npol and ngk info
    this->act(nbands, nbasis, this->npol_, psi, h_psi, this->ngk_[this->ik]);
    auto* node = this->next_op;
    while (node != nullptr)
    {
        node->act(nbands, nbasis, this->npol_, psi, h_psi, this->ngk_[node->ik]);
        node = node->next_op;
    }

    ModuleBase::timer::tick("Operator", "hPsi");
}

void Operator::init(const int ik_in)
{
    this->ik = ik_in;
    if (this->next_op != nullptr) {
        this->next_op->init(ik_in);
    }
}

void Operator::add(Operator* next)
{
    if(next==nullptr) return;
    next->is_first_node = false;
    if(next->next_op != nullptr) this->add(next->next_op);
    Operator* last = this;
    //loop to end of the chain
    while(last->next_op != nullptr)
    {
        if(next->cal_type==last->cal_type)
        {
            break;
        }
        last = last->next_op;
    }
    if(next->cal_type == last->cal_type)
    {
        //insert next to sub chain of current node
        Operator* sub_last = last;
        while(sub_last->next_sub_op != nullptr)
        {
            sub_last = sub_last->next_sub_op;
        }
        sub_last->next_sub_op = next;
        return;
    }
    else
    {
        last->next_op = next;
    }
}
