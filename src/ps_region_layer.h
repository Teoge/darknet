#ifndef ps_region_LAYER_H
#define ps_region_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer ps_region_layer;

ps_region_layer make_ps_region_layer(int batch, int h, int w, int n, int classes, int coords);
void forward_ps_region_layer(const ps_region_layer l, network_state state);
void backward_ps_region_layer(const ps_region_layer l, network_state state);
void get_ps_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);
void resize_ps_region_layer(layer *l, int w, int h);

#ifdef GPU
void forward_ps_region_layer_gpu(const ps_region_layer l, network_state state);
void backward_ps_region_layer_gpu(ps_region_layer l, network_state state);
#endif

#endif
