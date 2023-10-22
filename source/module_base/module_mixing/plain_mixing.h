#ifndef PLAIN_MIXING_H_
#define PLAIN_MIXING_H_
#include "mixing.h"
#include "module_base/memory.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"

namespace Base_Mixing
{
/**
 * @brief Plain mixing : rho_new = rho_in + mixing_beta * (rho_out - rho_in)
 *
 */
class Plain_Mixing : public Mixing
{
  public:
    Plain_Mixing(const double& mixing_beta)
    {
        this->mixing_beta = mixing_beta;
        this->data_ndim = 1;
        this->coef = std::vector<double>(1, 1.0);
    }
    virtual ~Plain_Mixing() override{};

    /**
     * @brief reset mixing
     *
     */
    virtual void reset() override
    {
        return;
    }

    /**
     * @brief
     *
     * @param mdata store information of this iterative step
     * @param data_in x_in
     * @param data_out x_out = f(x_in)
     * @param screen pointer to the screen function for Ker-Ker mixing
     * @param need_calcoef whether need to calculate the coef
     *
     */
    virtual void push_data(Mixing_Data& mdata,
                           const double* data_in,
                           const double* data_out,
                           std::function<void(double*)> screen,
                           const bool& need_calcoef) override
    {
        this->tem_push_data(mdata, data_in, data_out, screen, need_calcoef);
    };
    virtual void push_data(Mixing_Data& mdata,
                           const std::complex<double>* data_in,
                           const std::complex<double>* data_out,
                           std::function<void(std::complex<double>*)> screen,
                           const bool& need_calcoef) override
    {
        this->tem_push_data(mdata, data_in, data_out, screen, need_calcoef);
    };

    /**
     * @brief calculate coeficients for mixing
     *
     * @param mdata Mixing_Data
     * @param inner_product pointer to the inner dot function
     */
    virtual void cal_coef(const Mixing_Data& mdata, std::function<double(double*, double*)> inner_product) override
    {
        return;
    }
    virtual void cal_coef(const Mixing_Data& mdata,
                          std::function<double(std::complex<double>*, std::complex<double>*)> inner_product) override
    {
        return;
    }

  private:
    /**
     * @brief
     *
     * @param mdata store information of this iterative step
     * @param data_in x_in
     * @param data_out x_out = f(x_in)
     * @param screen pointer to the screen function for Ker-Ker mixing
     * @param need_calcoef whether need to calculate the coef
     *
     */
    template <class FPTYPE>
    void tem_push_data(Mixing_Data& mdata,
                       const FPTYPE* data_in,
                       const FPTYPE* data_out,
                       std::function<void(FPTYPE*)> screen,
                       const bool& need_calcoef)
    {
        const size_t length = mdata.length;
        std::vector<FPTYPE> F_tmp(length);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int i = 0; i < length; ++i)
        {
            F_tmp[i] = data_out[i] - data_in[i];
        }

        // get screened F
        if (screen != nullptr)
            screen(F_tmp.data());

        // container::Tensor data = data_in + mixing_beta * F;
        std::vector<FPTYPE> data(length);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(FPTYPE))
#endif
        for (int i = 0; i < length; ++i)
        {
            data[i] = data_in[i] + this->mixing_beta * F_tmp[i];
        }

        mdata.push(data.data());
    };
};
} // namespace Base_Mixing
#endif