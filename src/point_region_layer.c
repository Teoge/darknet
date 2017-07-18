#include "point_region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#define DOABS 1

point_region_layer make_point_region_layer(int batch, int w, int h)
{
	point_region_layer l = { 0 };
	l.type = POINT_REGION;

	l.batch = batch;
	l.h = h;
	l.w = w;
	l.cost = calloc(1, sizeof(float));
	l.outputs = h*w * 3;
	l.inputs = l.outputs;
	l.truths = 30 * 2;
	l.delta = calloc(batch*l.outputs, sizeof(float));
	l.output = calloc(batch*l.outputs, sizeof(float));
	l.forward = forward_point_region_layer;
	l.backward = backward_point_region_layer;
#ifdef GPU
	l.forward_gpu = forward_point_region_layer_gpu;
	l.backward_gpu = backward_point_region_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

	fprintf(stderr, "detection\n");
	srand(0);

	return l;
}

void resize_point_region_layer(layer *l, int w, int h)
{
	l->w = w;
	l->h = h;

	l->outputs = h*w*l->n*(l->classes + l->coords + 1);
	l->inputs = l->outputs;

	l->output = realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = realloc(l->delta, l->batch*l->outputs * sizeof(float));

#ifdef GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
	l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_point_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
	box b;
	b.x = (i + logistic_activate(x[index + 0])) / w;
	b.y = (j + logistic_activate(x[index + 1])) / h;
	b.w = exp(x[index + 2]) * biases[2 * n];
	b.h = exp(x[index + 3]) * biases[2 * n + 1];
	if (DOABS) {
		b.w = exp(x[index + 2]) * biases[2 * n] / w;
		b.h = exp(x[index + 3]) * biases[2 * n + 1] / h;
	}
	return b;
}

void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output);
void forward_point_region_layer(const point_region_layer l, network_state state)
{
	int i, j, b, t, n;
	int size = 3;
	memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));
	for (b = 0; b < l.batch; ++b) {
		for (i = 0; i < l.h*l.w; ++i) {
			int index = size*i + b*l.outputs;
			l.output[index + 2] = logistic_activate(l.output[index + 2]);
		}
	}

	if (!state.train) return;

	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
	float avg_confid = 0;
	float recall = 0;
	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	*(l.cost) = 0;

	for (b = 0; b < l.batch; ++b) {
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				float shortest_dist = 1.0f;
				int index = size*(j*l.w + i) + b*l.outputs;
				float x = (logistic_activate(l.output[index + 0]) + i) / l.w;
				float y = (logistic_activate(l.output[index + 1]) + j) / l.h;
				for (t = 0; t < 30; ++t) {
					int truth_index = t * 2 + b*l.truths;
					if (!state.truth[truth_index]) break;
					float dx = x - state.truth[truth_index + 0];
					float dy = y - state.truth[truth_index + 1];
					float dist = (dx*dx + dy*dy) * 2704;//2704 = 52^2
					if (dist < shortest_dist)shortest_dist = dist;
				}
				avg_anyobj += l.output[index + 2];
				if (shortest_dist > 1)
					l.delta[index + 2] = l.noobject_scale * ((0 - l.output[index + 2]) * logistic_gradient(l.output[index + 2]));
			}
		}
		for (t = 0; t < 30; ++t) {
			int truth_index = t * 2 + b*l.truths;
			if (!state.truth[truth_index]) break;
			i = (state.truth[truth_index] * l.w);
			j = (state.truth[truth_index + 1] * l.h);
			int index = size*(j*l.w + i) + b*l.outputs;
			float tx = (state.truth[truth_index + 0] * l.w - i);
			float ty = (state.truth[truth_index + 1] * l.h - j);
			float dx = logistic_activate(l.output[index + 0]);
			float dy = logistic_activate(l.output[index + 1]);
			l.delta[index + 0] = l.coord_scale * (tx - dx) * logistic_gradient(dx);
			l.delta[index + 1] = l.coord_scale * (ty - dy) * logistic_gradient(dy);
			float dist = ((tx - dx)*(tx - dx) + (ty - dy)*(ty - dy)) * 16;
			if (dist < .2f) recall += 1;
			avg_obj += l.output[index + 2];
			if (dist < 1) {
				avg_confid += 1 - dist;
				l.delta[index + 2] = l.object_scale * (1 - dist - l.output[index + 2]) * logistic_gradient(l.output[index + 2]);
			}
			else {
				l.delta[index + 2] = l.noobject_scale * (0 - l.output[index + 2]) * logistic_gradient(l.output[index + 2]);
			}
			++count;
		}
	}
	*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
	printf("Region Avg CONFIDENCE: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_confid / count, avg_obj / count, avg_anyobj / (l.w*l.h*l.batch), recall / count, count);
}

void backward_point_region_layer(const point_region_layer l, network_state state)
{
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void get_point_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map)
{
	int i, j, n;
	float *predictions = l.output;
	for (i = 0; i < l.w*l.h; ++i) {
		int row = i / l.w;
		int col = i % l.w;
		for (n = 0; n < l.n; ++n) {
			int index = i*l.n + n;
			int p_index = index * (l.classes + 5) + 4;
			float scale = predictions[p_index];
			if (l.classfix == -1 && scale < .5) scale = 0;
			int box_index = index * (l.classes + 5);
			boxes[index] = get_point_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);
			boxes[index].x *= w;
			boxes[index].y *= h;
			boxes[index].w *= w;
			boxes[index].h *= h;

			int class_index = index * (l.classes + 5) + 5;
			if (l.softmax_tree) {

				hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);
				int found = 0;
				if (map) {
					for (j = 0; j < 200; ++j) {
						float prob = scale*predictions[class_index + map[j]];
						probs[index][j] = (prob > thresh) ? prob : 0;
					}
				}
				else {
					for (j = l.classes - 1; j >= 0; --j) {
						if (!found && predictions[class_index + j] > .5) {
							found = 1;
						}
						else {
							predictions[class_index + j] = 0;
						}
						float prob = predictions[class_index + j];
						probs[index][j] = (scale > thresh) ? prob : 0;
					}
				}
			}
			else {
				for (j = 0; j < l.classes; ++j) {
					float prob = scale*predictions[class_index + j];
					probs[index][j] = (prob > thresh) ? prob : 0;
				}
			}
			if (only_objectness) {
				probs[index][0] = scale;
			}
		}
	}
}

#ifdef GPU

void forward_point_region_layer_gpu(const point_region_layer l, network_state state)
{
	flatten_ongpu(state.input, l.h*l.w, 3, l.batch, 1, l.output_gpu);

	float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
	float *truth_cpu = 0;
	if (state.truth) {
		int num_truth = l.batch*l.truths;
		truth_cpu = calloc(num_truth, sizeof(float));
		cuda_pull_array(state.truth, truth_cpu, num_truth);
	}
	cuda_pull_array(l.output_gpu, in_cpu, l.batch*l.inputs);
	network_state cpu_state = state;
	cpu_state.train = state.train;
	cpu_state.truth = truth_cpu;
	cpu_state.input = in_cpu;
	forward_point_region_layer(l, cpu_state);
	//cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
	free(cpu_state.input);
	if (!state.train) return;
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
	if (cpu_state.truth) free(cpu_state.truth);
}

void backward_point_region_layer_gpu(point_region_layer l, network_state state)
{
	flatten_ongpu(l.delta_gpu, l.h*l.w, 3, l.batch, 0, state.delta);
}
#endif

