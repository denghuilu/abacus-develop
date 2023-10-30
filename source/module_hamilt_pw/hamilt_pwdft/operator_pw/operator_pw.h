#ifndef OPERATORPW_H
#define OPERATORPW_H
#include"module_hamilt_general/operator.h"

namespace hamilt {
template<typename T, typename Device = psi::DEVICE_CPU>
class OperatorPW : public Operator
{
    public:
        ~OperatorPW() override;
        calculation_type type_ = calculation_type::no;
};

}//end namespace hamilt

#endif