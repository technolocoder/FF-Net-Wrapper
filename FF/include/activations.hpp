#ifndef __FF_ACTIVATIONS_WRAPPER__
#define __FF_ACTIVATIONS_WRAPPER__

enum ACTIVATION_FUNCTIONS {TANH,SIGMOID,RELU,LEAKY_RELU,LINEAR};
#include "common.hpp"
#include <cmath>
TEMPLATE T sigmoid(T sum){
    return 1.0/(1.0+exp(-sum));
}

TEMPLATE T relu(T sum){
    return sum>0?sum:0;
}

TEMPLATE T leaky_relu(T sum){
    return sum>0?sum:sum*0.1;
}

TEMPLATE T sigmoid_deriv(T out){
    return out*(1-out);
}

TEMPLATE T tanh_deriv(T out){
    return 1-out*out;
}

TEMPLATE T relu_deriv(T sum){
    return sum>0;
}

TEMPLATE T leaky_relu_deriv(T sum){
    return sum>0?1:0.1;
}

#endif 