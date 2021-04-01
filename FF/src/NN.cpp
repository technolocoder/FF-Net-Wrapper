#include "NN.hpp"
#include <random>
#define TEMPLATE template<typename T> 

using namespace std;

TEMPLATE neural_network<T>::neural_network() {}
TEMPLATE neural_network<T>::neural_network(int _sample_size ,int _layer_count){
    init(_sample_size,_layer_count);
}

TEMPLATE void neural_network<T>::init(int _sample_size, int _layer_count){
    sample_size = _sample_size;
    layer_count = _layer_count;
    layers = (layer<T>*)malloc(sizeof(layer<T>)*layer_count);
    init_layer = true;
}

TEMPLATE void neural_network<T>::add_input_layer(int input_size,bool batch_normalized){
    layers[layer_index].IL.init(sample_size,input_size,batch_normalized);
    layers[layer_index].type = INPUT_LAYER;
    total_size += layers[layer_index].IL.total_size;
    ++layer_index;
}

TEMPLATE void neural_network<T>::add_fully_connected_layer(int layer_size, ACTIVATION_FUNCTIONS activation){
    int input_size;
    layers[layer_index].type = FULLY_CONNECTED_LAYER;
    if(layers[layer_index-1].type == INPUT_LAYER) input_size = layers[layer_index-1].IL.input_size;
    else input_size = layers[layer_index-1].FC.layer_size;
    layers[layer_index].FC.init(input_size,layer_size,sample_size,activation);
    total_size += layers[layer_index].FC.total_size;
    ++layer_index;
}

TEMPLATE void neural_network<T>::add_softmax_layer(){
    int input_size;
    layers[layer_index].type = SOFTMAX_LAYER;
    if(layers[layer_index-1].type == INPUT_LAYER) input_size = layers[layer_index-1].IL.input_size;
    else input_size = layers[layer_index-1].FC.layer_size;
    layers[layer_index].SL.init(sample_size,input_size);
    total_size += layers[layer_index].SL.total_size;
    ++layer_index;
}

TEMPLATE void neural_network<T>::construct_net(){
    data = (T*)malloc(sizeof(T)*total_size);
    T *data_ptr = data, *prev_deriv=NULL;
    layers[0].IL.init_mem(data_ptr);
    T *prev_ptr = layers[0].IL.output_ptr;
    for(int i = 1; i < layer_count; ++i){
        if(layers[i].type == FULLY_CONNECTED_LAYER){
            layers[i].FC.init_mem(data_ptr);
            layers[i].FC.input_ptr = prev_ptr;
            prev_ptr = layers[i].FC.output_ptr;
            layers[i].FC.input_deriv_ptr = prev_deriv;
            prev_deriv = layers[i].FC.output_deriv_ptr;
        }else{
            layers[i].SL.init_mem(data_ptr);
            layers[i].SL.input_ptr = prev_ptr;
            prev_ptr = layers[i].SL.output_ptr;
        }
    }
    init_mem = true;
}

TEMPLATE T* neural_network<T>::feedforward(T *input){
    layers[0].IL.input_ptr = input;
    layers[0].IL.process();
    if(layer_index > 1){
        if(layers[1].type == FULLY_CONNECTED_LAYER){
            layers[1].FC.input_ptr = layers[0].IL.output_ptr;
            layers[1].FC.process();
        }else{
            layers[1].SL.input_ptr = layers[0].IL.output_ptr;
            layers[1].SL.process();
        }
        
        for(int i = 2; i < layer_index; ++i){
            if(layers[i].type == FULLY_CONNECTED_LAYER){
                layers[i].FC.process();
            }else{
                layers[i].SL.process();
            }
        }
        return layers[layer_index-1].type==FULLY_CONNECTED_LAYER?layers[layer_index-1].FC.output_ptr:layers[layer_index-1].SL.output_ptr;
    }
    return layers[0].IL.output_ptr;
}

TEMPLATE void neural_network<T>::feedforward_deriv(T *input){
    layers[0].IL.input_ptr = input;
    layers[0].IL.process();
    if(layer_index > 1){
        if(layers[1].type == FULLY_CONNECTED_LAYER){
            layers[1].FC.input_ptr = layers[0].IL.output_ptr;
            layers[1].FC.process_deriv();
        }else{
            layers[1].SL.input_ptr = layers[0].IL.output_ptr;
            layers[1].SL.process();
        }
        
        for(int i = 2; i < layer_index; ++i){
            if(layers[i].type == FULLY_CONNECTED_LAYER){
                layers[i].FC.process_deriv();
            }else{
                layers[i].SL.process();
            }
        }
    }
}

TEMPLATE T neural_network<T>::get_loss(T *input, T *label){
    feedforward(input);
    if(layers[layer_index-1].type == FULLY_CONNECTED_LAYER){
        return layers[layer_index-1].FC.get_loss(label);
    }else{
        // TODO Softmax Loss
    }
}

TEMPLATE void neural_network<T>::backpropagate(T *input, T *label){
    feedforward_deriv(input);
    if(layer_index > 2){
    layers[layer_index-1].FC.backprop_process(label);
        layers[layer_index-1].FC.apply_deriv();
        for(int i = layer_index-2; i > 1; --i){
            layers[layer_index-1].FC.backprop_process();
            layers[layer_index-1].FC.apply_deriv();
        }
        layers[1].FC.backprop_process_noprev();
        layers[1].FC.apply_deriv();
    }else{
        layers[1].FC.backprop_process_noprev(label);
        layers[1].FC.apply_deriv();
    }
}

TEMPLATE neural_network<T>::~neural_network(){
    if(init_layer) free(layers);
    if(init_mem) free(data);
}

template class neural_network<float>;
template class neural_network<double>;