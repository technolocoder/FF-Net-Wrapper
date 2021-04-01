#include "FC.hpp"
#include <cstring>
#include <random>

using namespace std;
TEMPLATE void _FC<T>::init(int _input_size, int _layer_size, int _sample_size, ACTIVATION_FUNCTIONS _activation){
    input_size = _input_size;
    layer_size = _layer_size;
    sample_size = _sample_size;
    activation = _activation;
    total_output_size = layer_size*sample_size;
    total_weight_size = input_size*layer_size;
    total_size = total_output_size*3+total_weight_size*2+layer_size*2;
    lr = 0.05;
}

TEMPLATE void _FC<T>::init_mem(T *&base_ptr){
    random_device rd;
    mt19937_64 engine(rd());
    weights_ptr = base_ptr;
    weights_deriv = weights_ptr+total_weight_size;
    memset(weights_deriv,0,sizeof(T)*total_weight_size);
    if(activation == TANH || activation == SIGMOID){
        T range = sqrt(6.0/(input_size+layer_size));
        uniform_real_distribution<T> dist(-range,range);
        for(int i = 0; i < total_weight_size; ++i) weights_ptr[i] = dist(engine);
    }else if(activation == RELU || activation == LEAKY_RELU){
        normal_distribution<T> dist((T)0.0,sqrt(2.0/input_size));
        for(int i = 0; i < total_weight_size; ++i) weights_ptr[i] = dist(engine);
    }else{
        normal_distribution<T> dist((T)0.0,sqrt(1.0/input_size));
        for(int i = 0; i < total_weight_size; ++i) weights_ptr[i] = dist(engine);
    }
    bias_ptr = weights_deriv+total_weight_size;
    bias_deriv = bias_ptr+layer_size;
    memset(bias_ptr,0,sizeof(T)*layer_size);
    memset(bias_deriv,0,sizeof(T)*layer_size);
    output_ptr = bias_deriv+layer_size;
    activations_deriv_ptr = output_ptr+total_output_size;
    output_deriv_ptr = activations_deriv_ptr+total_output_size; 
    memset(output_deriv_ptr,0,sizeof(T)*total_output_size);
    base_ptr += total_size;
} 

#define LOOP for(int i = 0; i < total_output_size; ++i)
TEMPLATE void _FC<T>::process(){
    int index = 0;
    for(int i = 0; i < sample_size; ++i){
        for(int j = 0; j < layer_size; ++j){
            T sum = bias_ptr[j];
            for(int k = 0; k < input_size; ++k) sum += input_ptr[i*input_size+k]*weights_ptr[j*input_size+k];
            output_ptr[index++] = sum;
        }
    }
    if(activation==TANH) LOOP output_ptr[i] = tanh(output_ptr[i]);
    else if(activation==SIGMOID) LOOP output_ptr[i] = sigmoid(output_ptr[i]);
    else if(activation==RELU) LOOP output_ptr[i] = relu(output_ptr[i]);
    else if(activation==LEAKY_RELU) LOOP output_ptr[i] = leaky_relu(output_ptr[i]); 
}

#include <iostream>
using namespace std;


TEMPLATE void _FC<T>::process_deriv(){
    int index = 0;
    for(int i = 0; i < sample_size; ++i){
        for(int j = 0; j < layer_size; ++j){
            T sum = bias_ptr[j];
            for(int k = 0; k < input_size; ++k) sum += input_ptr[i*input_size+k]*weights_ptr[j*input_size+k];
            output_ptr[index++] = sum;
        }
    }
    if(activation==TANH) LOOP{
        output_ptr[i] = tanh(output_ptr[i]);
        activations_deriv_ptr[i] = tanh_deriv(output_ptr[i]);
    }
    else if(activation==SIGMOID) LOOP {
        output_ptr[i] = sigmoid(output_ptr[i]);
        activations_deriv_ptr[i] = sigmoid_deriv(output_ptr[i]);
    }
    else if(activation==RELU) LOOP{
        activations_deriv_ptr[i] = relu_deriv(output_ptr[i]);
        output_ptr[i] = relu(output_ptr[i]);
    }
    else if(activation==LEAKY_RELU) LOOP{
        activations_deriv_ptr[i] = leaky_relu_deriv(output_ptr[i]);
        output_ptr[i] = leaky_relu(output_ptr[i]); 
    }
}

TEMPLATE T _FC<T>::get_loss(T *label){
    T loss = 0;
    for(int i = 0; i < total_output_size; ++i){
        T diff = output_ptr[i]-label[i];
        loss += diff*diff;
    }
    return loss/sample_size;
}

TEMPLATE void _FC<T>::backprop_process(){
    for(int i = 0; i < sample_size; ++i){
        for(int j = 0; j < layer_size; ++j){
            T deriv = output_deriv_ptr[i*layer_size+j]*activations_deriv_ptr [i*layer_size+j];
            for(int k = 0; k < input_size; ++k){
                weights_deriv[j*input_size+k] += deriv*input_ptr[i*input_size+k];
                input_deriv_ptr[i*input_size+k] += deriv*weights_ptr[j*input_size+k];
            }
            bias_deriv[j] += deriv;
        }
    }
}

TEMPLATE void _FC<T>::backprop_process_noprev(){
    for(int i = 0; i < sample_size; ++i){
        for(int j = 0; j < layer_size; ++j){
            T deriv = output_deriv_ptr[i*layer_size+j]*activations_deriv_ptr [i*layer_size+j];
            for(int k = 0; k < input_size; ++k){
                weights_deriv[j*input_size+k] += deriv*input_ptr[i*input_size+k];
            }
            bias_deriv[j] += deriv;
        }
    }
}

TEMPLATE void _FC<T>::backprop_process(T *label){
    for(int i = 0; i < sample_size; ++i){
        for(int j = 0; j < layer_size; ++j){
            output_deriv_ptr[i*layer_size+j] = 2*(label[i*layer_size+j]-output_ptr[i*layer_size+j]);
            T deriv = output_deriv_ptr[i*layer_size+j]*activations_deriv_ptr [i*layer_size+j];
            for(int k = 0; k < input_size; ++k){
                weights_deriv[j*input_size+k] += deriv*input_ptr[i*input_size+k];
                input_deriv_ptr[i*input_size+k] += deriv*weights_ptr[j*input_size+k];
            }
            bias_deriv[j] += deriv;
        }
    }
}

TEMPLATE void _FC<T>::backprop_process_noprev(T *label){
    for(int i = 0; i < sample_size; ++i){
        for(int j = 0; j < layer_size; ++j){
            output_deriv_ptr[i*layer_size+j] = 2*(label[i*layer_size+j]-output_ptr[i*layer_size+j]);
            T deriv = output_deriv_ptr[i*layer_size+j]*activations_deriv_ptr [i*layer_size+j];
            for(int k = 0; k < input_size; ++k){
                weights_deriv[j*input_size+k] += deriv*input_ptr[i*input_size+k];
            }
            bias_deriv[j] += deriv;
        }
    }
}

TEMPLATE void _FC<T>::apply_deriv(){
    for(int i = 0; i < total_weight_size; ++i) weights_ptr[i] += weights_deriv[i]/sample_size*lr;
    for(int i = 0; i < layer_size; ++i) bias_ptr[i] += bias_deriv[i]/sample_size*lr;
    memset(output_deriv_ptr,0,sizeof(T)*total_output_size);
    memset(weights_deriv,0,sizeof(T)*total_weight_size);
    memset(bias_deriv,0,sizeof(T)*layer_size);
}

template struct _FC<float>;
template struct _FC<double>;