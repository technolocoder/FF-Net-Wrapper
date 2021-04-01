#ifndef __FF_IL_WRAPPER__
#define __FF_IL_WRAPPER__

#include "common.hpp"

TEMPLATE struct _IL{
    T *input_ptr,*output_ptr;
    bool batch_normalized=false;
    int sample_size, input_size,total_size;

    void init(int _sample_size, int _input_size,bool _batch_normalized);
    void init_mem(T *& base_ptr);
    void process();
};

#endif 