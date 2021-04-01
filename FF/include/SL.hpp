#ifndef __FF__SL__WRAPPER__
#define __FF__SL__WRAPPER__

#include "common.hpp"

TEMPLATE struct _SL{
    T *input_ptr, *output_ptr;
    int sample_size, layer_size, total_size;

    void init(int _sample_size ,int _layer_size);
    void init_mem(T *& base_ptr);
    void process();
};

#endif 