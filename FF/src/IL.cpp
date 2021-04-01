#include "IL.hpp"

TEMPLATE void _IL<T>::init(int _sample_size, int _input_size, bool _batch_normalized){
    sample_size = _sample_size;
    input_size = _input_size;
    batch_normalized = _batch_normalized;
    total_size = batch_normalized*sample_size*input_size;
}

TEMPLATE void _IL<T>::process(){
    if(batch_normalized){
        //TODO Implement Batch Normalization    
        return;
    }
    output_ptr = input_ptr;
}

TEMPLATE void _IL<T>::init_mem(T *& base_ptr){
    if(batch_normalized){
        output_ptr = base_ptr;
        base_ptr += total_size;
    }
}

template struct _IL<float>;
template struct _IL<double>;