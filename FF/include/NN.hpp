#ifndef __NN_FF__WRAPPER__
#define __NN_FF__WRAPPER__

#include "common.hpp"
#include "layer.hpp"

TEMPLATE class neural_network{
public:
    neural_network();
    neural_network(int _sample_size,int _layer_count);
    void init(int _sample_size, int _layer_count);

    void add_input_layer(int input_size, bool batch_normalized);
    void add_fully_connected_layer(int layer_size,ACTIVATION_FUNCTIONS activation);
    void add_softmax_layer();

    void construct_net();
    T *feedforward(T *input);
    T get_loss(T *input, T *label);

    void feedforward_deriv(T *input);
    void backpropagate(T *input, T *label);

    ~neural_network();
private:
    int sample_size,layer_count,layer_index=0,total_size=0;
    bool init_layer=false, init_mem=false;
    T *data;
    layer<T> *layers;
};

#endif