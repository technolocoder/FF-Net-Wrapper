#ifndef __FF__FC__WRAPPER__
#define __FF__FC__WRAPPER__

#include "common.hpp"
#include "activations.hpp"

TEMPLATE struct _FC{
    T *input_ptr, *output_ptr, *weights_ptr, *bias_ptr, *weights_deriv, *bias_deriv, *input_deriv_ptr, *output_deriv_ptr, *activations_deriv_ptr,lr;
    int input_size, layer_size, total_output_size, total_weight_size ,total_size, sample_size;
    ACTIVATION_FUNCTIONS activation;

    void init(int _input_size, int _layer_size, int _sample_size, ACTIVATION_FUNCTIONS _activation);
    void init_mem(T *&base_ptr);

    void process();
    void process_deriv();

    void backprop_process();
    void backprop_process(T *label);

    void backprop_process_noprev();
    void backprop_process_noprev(T *label);

    void apply_deriv();

    T get_loss(T *label);
};

#endif 