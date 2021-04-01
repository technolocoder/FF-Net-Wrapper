#ifndef __LAYER_FF_WRAPPER__
#define __LAYER_FF_WRAPPER__

#include "FC.hpp"
#include "SL.hpp"
#include "IL.hpp"

enum LAYER_TYPE {INPUT_LAYER,FULLY_CONNECTED_LAYER,SOFTMAX_LAYER};

TEMPLATE struct layer{
    LAYER_TYPE type;
    union{
        _FC<T> FC;
        _SL<T> SL;
        _IL<T> IL;
    };
};

#endif 