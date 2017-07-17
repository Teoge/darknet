#ifndef POINT_REGION_LAYER_H
#define POINT_REGION_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer point_region_layer;

point_region_layer make_point_region_layer(int batch, int h, int w);
void forward_point_region_layer(const point_region_layer l, network_state state);
void backward_point_region_layer(const point_region_layer l, network_state state);
void get_point_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);
void resize_point_region_layer(layer *l, int w, int h);

#ifdef GPU
void forward_point_region_layer_gpu(const point_region_layer l, network_state state);
void backward_point_region_layer_gpu(point_region_layer l, network_state state);
#endif

#endif