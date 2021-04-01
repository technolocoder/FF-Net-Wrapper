#include "SL.hpp"
#include <cmath>
#define MAX(a,b) (a)>(b)?(a):(b)

TEMPLATE void _SL<T>::init(int _sample_size ,int _layer_size){
    sample_size = _sample_size;
    layer_size = _layer_size;
    total_size = sample_size*layer_size;
}

TEMPLATE void _SL<T>::init_mem(T *& base_ptr){
    output_ptr = base_ptr;
    base_ptr += total_size;
}

TEMPLATE void _SL<T>::process(){
    for(int i = 0; i < sample_size; ++i){
        T max = 0,sum = 0;
        for(int j = 0; j < layer_size; ++j) max = MAX(max,input_ptr[i*layer_size+j]);
        for(int j = 0; j < layer_size; ++j) sum += exp(input_ptr[i*sample_size+j]-max);
        for(int j = 0; j < layer_size; ++j) output_ptr[i*layer_size+j] = exp(input_ptr[i*sample_size+j]-max)/sum;
    }
}

template struct _SL<float>;
template struct _SL<double>;