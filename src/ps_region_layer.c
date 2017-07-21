#include "ps_region_layer.h"
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

ps_region_layer make_ps_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
	ps_region_layer l = { 0 };
	l.type = PS_REGION;

	l.n = n;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.classes = classes;
	l.coords = coords;
	l.cost = calloc(1, sizeof(float));
	l.outputs = h*w*(classes + coords + 1);
	l.inputs = l.outputs;
	l.truths = 30 * (5);
	l.delta = calloc(batch*l.outputs, sizeof(float));
	l.output = calloc(batch*l.outputs, sizeof(float));

	l.forward = forward_ps_region_layer;
	l.backward = backward_ps_region_layer;
#ifdef GPU
	l.forward_gpu = forward_ps_region_layer_gpu;
	l.backward_gpu = backward_ps_region_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

	fprintf(stderr, "detection\n");
	srand(0);

	return l;
}

void resize_ps_region_layer(layer *l, int w, int h)
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

box get_ps_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h)
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

float delta_ps_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale)
{
	box pred = get_ps_region_box(x, biases, n, index, i, j, w, h);
	float iou = box_iou(pred, truth);

	float tx = (truth.x*w - i);
	float ty = (truth.y*h - j);
	float tw = log(truth.w / biases[2 * n]);
	float th = log(truth.h / biases[2 * n + 1]);
	if (DOABS) {
		tw = log(truth.w*w / biases[2 * n]);
		th = log(truth.h*h / biases[2 * n + 1]);
	}

	delta[index + 0] = scale * (tx - logistic_activate(x[index + 0])) * logistic_gradient(logistic_activate(x[index + 0]));
	delta[index + 1] = scale * (ty - logistic_activate(x[index + 1])) * logistic_gradient(logistic_activate(x[index + 1]));
	delta[index + 2] = scale * (tw - x[index + 2]);
	delta[index + 3] = scale * (th - x[index + 3]);
	return iou;
}

void delta_ps_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, float *avg_cat)
{
	int i, n;
	for (n = 0; n < classes; ++n) {
		delta[index + n] = scale * (((n == class) ? 1 : 0) - output[index + n]);
		if (n == class) *avg_cat += output[index + n];
	}
}

typedef struct {
	float cx, cy, cos_angle, sin_angle, half_dist;
	int ps_type;
	int min_x, max_x, min_y, max_y;
} RotatedBox;

RotatedBox get_rotated_box(float x, float y, float angle, float dist, int label)
{
	//printf("%f %f %f %f %d\n", x, y, angle, dist, label);
	RotatedBox b;
	angle = angle * 2 * 3.141592653f;
	b.cos_angle = cos(angle);
	b.sin_angle = sin(angle);
	b.ps_type = label;
	float separator_length;
	if (label != 3) {
		separator_length = 195;
	}
	else {
		separator_length = 83;
	}
	dist = min(dist, 0.9f);
	dist = max(dist, 0.1f);
	b.half_dist = dist * 208;
	x = x * 416;
	y = y * 416;
	float x1 = x + b.half_dist*b.cos_angle;
	float x2 = x - b.half_dist*b.cos_angle;
	float x3 = x1 + separator_length*b.sin_angle;
	float x4 = x2 + separator_length*b.sin_angle;
	float y1 = y + b.half_dist*b.sin_angle;
	float y2 = y - b.half_dist*b.sin_angle;
	float y3 = y1 - separator_length*b.cos_angle;
	float y4 = y2 - separator_length*b.cos_angle;
	//printf("%f %f\n%f %f\n%f %f\n%f %f\n", x1, y1, x2, y2, x3, y3, x4, y4);
	if (b.cos_angle > 0) {
		if (b.sin_angle > 0) {
			b.min_x = (int)floorf(x2);
			b.max_x = (int)ceilf(x3);
			b.min_y = (int)floorf(y4);
			b.max_y = (int)ceilf(y1);
		}
		else {
			b.min_x = (int)floorf(x4);
			b.max_x = (int)ceilf(x1);
			b.min_y = (int)floorf(y3);
			b.max_y = (int)ceilf(y2);
		}
	}
	else {
		if (b.sin_angle > 0) {
			b.min_x = (int)floorf(x1);
			b.max_x = (int)ceilf(x4);
			b.min_y = (int)floorf(y2);
			b.max_y = (int)ceilf(y3);
		}
		else {
			b.min_x = (int)floorf(x3);
			b.max_x = (int)ceilf(x2);
			b.min_y = (int)floorf(y1);
			b.max_y = (int)ceilf(y4);
		}
	}
	//printf("%d %d %d %d\n", b.min_x, b.max_x, b.min_y, b.max_y);
	b.cx = x + separator_length*b.sin_angle / 2.0f;
	b.cy = y - separator_length*b.cos_angle / 2.0f;
	return b;
}

int within_rotated_box(float cx, float cy, float cos_angle, float sin_angle, float half_dist, int ps_type, int px, int py)
{
	float vx = px - cx;
	float vy = py - cy;
	float rvx = cos_angle*vx + sin_angle*vy;
	float rvy = cos_angle*vy - sin_angle*vx;
	if (ps_type != 3) {
		if (abs(rvx) < half_dist && abs(rvy) < 97.5f)return 1;
		else return 0;
	}
	else {
		if (abs(rvx) < half_dist && abs(rvy) < 41.5f)return 1;
		else return 0;
	}
}

float calculate_iou(RotatedBox pred, RotatedBox truth)
{
	int min_x = min(pred.min_x, truth.min_x);
	int min_y = min(pred.min_y, truth.min_y);
	int max_x = max(pred.max_x, truth.max_x);
	int max_y = max(pred.max_y, truth.max_y);
	int intercetion_count = 0;
	int pred_count = 0;
	int truth_count = 0;
	for (int i = min_y; i <= max_y; i++) {
		for (int j = min_x; j <= max_x; j++) {
			int within_pred = within_rotated_box(pred.cx, pred.cy, pred.cos_angle, pred.sin_angle, pred.half_dist, pred.ps_type, j, i);
			int within_truth = within_rotated_box(truth.cx, truth.cy, truth.cos_angle, truth.sin_angle, truth.half_dist, truth.ps_type, j, i);
			if (within_pred == 1 && within_truth == 1)intercetion_count++;
			pred_count += within_pred;
			truth_count += within_truth;
		}
	}
	//printf("%d %d %d\n", intercetion_count, pred_count, truth_count);
	if (intercetion_count == 0)return 0;
	return (double)intercetion_count / (pred_count + truth_count - intercetion_count);
}

int get_best_class(float *x, int classes, int index)
{
	int max_confid = 0;
	int best_class = -1;
	for (int i = 0; i < classes; i++)
	{
		if (x[index + i] > max_confid) {
			max_confid = x[index + i];
			best_class = i;
		}
	}
	return best_class;
}

void forward_ps_region_layer(const ps_region_layer l, network_state state)
{
	int i, j, b, t;
	int size = l.coords + l.classes + 1;
	memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));
	for (b = 0; b < l.batch; ++b) {
		for (i = 0; i < l.h*l.w; ++i) {
			int index = size*i + b*l.outputs;
			l.output[index + 0] = logistic_activate(l.output[index + 0]);
			l.output[index + 1] = logistic_activate(l.output[index + 1]);
			l.output[index + 2] = logistic_activate(l.output[index + 2]);
			l.output[index + 4] = logistic_activate(l.output[index + 4]);
		}
	}
	if (!state.train) return;
	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
	float avg_iou = 0;
	float recall = 0;
	float avg_cat = 0;
	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;
	for (b = 0; b < l.batch; ++b) {
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				int index = size*(j*l.w + i) + b*l.outputs;
				float x = (l.output[index + 0] + i) / l.w;
				float y = (l.output[index + 1] + j) / l.h;
				int label = get_best_class(l.output, l.classes, index + 5);
				float dist;
				if (label != 3) dist = (float)exp(dist) * 106 / 416.0f;
				else dist = (float)exp(dist) * 265 / 416.0f;
				RotatedBox pred = get_rotated_box(x, y, l.output[index + 2], dist, label);
				float best_iou = 0;
				for (t = 0; t < 30; ++t) {
					int truth_index = t * 5 + b*l.truths;
					if (!state.truth[truth_index]) break;
					RotatedBox truth = get_rotated_box(state.truth[truth_index], state.truth[truth_index + 1], state.truth[truth_index + 2], state.truth[truth_index + 3], (int)state.truth[truth_index + 4]);
					float iou = calculate_iou(pred, truth);
					if (iou > best_iou) {
						best_iou = iou;
					}
				}
				avg_anyobj += l.output[index + 4];
				if (best_iou <= l.thresh)l.delta[index + 4] = l.noobject_scale * ((0 - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
			}
		}
		for (t = 0; t < 30; ++t) {
			int truth_index = t * 5 + b*l.truths;
			if (!state.truth[truth_index]) break;
			RotatedBox truth = get_rotated_box(state.truth[truth_index], state.truth[truth_index + 1], state.truth[truth_index + 2], state.truth[truth_index + 3], (int)state.truth[truth_index + 4]);

			i = (int)(state.truth[truth_index + 0] * l.w);
			j = (int)(state.truth[truth_index + 1] * l.h);
			int index = size*(j*l.w + i) + b*l.outputs;
			float tx = state.truth[truth_index + 0] * l.w - i;
			float ty = state.truth[truth_index + 1] * l.h - j;
			int tclass = (int)state.truth[truth_index + 4];
			float tdist;
			if (tclass != 3)  tdist = log(state.truth[truth_index + 3] * 416 / 106.0f);
			else  tdist = log(state.truth[truth_index + 3] * 416 / 265.0f);
			float x = (l.output[index + 0] + i) / l.w;
			if (isnan(x)) {
				printf("%d %d %d %d %f %f", b, index, i, j, state.truth[truth_index + 0], state.truth[truth_index + 1]);
			}
			float y = (l.output[index + 1] + j) / l.h;
			int label = get_best_class(l.output, l.classes, index + 5);
			float dist;
			if (label != 3) dist = (float)exp(dist) * 106 / 416.0f;
			else dist = (float)exp(dist) * 265 / 416.0f;
			RotatedBox pred = get_rotated_box(x, y, l.output[index + 2], dist, label);
			float iou = calculate_iou(pred, truth);
			l.delta[index + 0] = l.coord_scale * (tx - l.output[index + 0]) * logistic_gradient(l.output[index + 0]);
			l.delta[index + 1] = l.coord_scale * (ty - l.output[index + 1]) * logistic_gradient(l.output[index + 1]);
			l.delta[index + 2] = l.coord_scale * (state.truth[truth_index + 2] - l.output[index + 2]) * logistic_gradient(l.output[index + 2]);
			l.delta[index + 3] = l.coord_scale * (ty - l.output[index + 3]) * logistic_gradient(l.output[index + 3]);
			l.delta[index + 4] = l.object_scale * (iou - l.output[index + 4]) * logistic_gradient(l.output[index + 4]);
			delta_ps_region_class(l.output, l.delta, index + 5, tclass, l.classes, l.softmax_tree, l.class_scale, &avg_cat);

			if (iou > .5) recall += 1;
			avg_iou += iou;
			avg_obj += l.output[index + 4];
			++count;
			++class_count;
		}
	}
	//printf("\n");
#ifndef GPU
	flatten(l.delta, l.w*l.h, size*l.n, l.batch, 0);
#endif
	*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
	printf("ps_region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, count);
}

void backward_ps_region_layer(const ps_region_layer l, network_state state)
{
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_ps_region_layer_gpu(const ps_region_layer l, network_state state)
{
	flatten_ongpu(state.input, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 1, l.output_gpu);
	softmax_gpu(l.output_gpu + 5, l.classes, l.classes + 5, l.w*l.h*l.batch, 1, l.output_gpu + 5);

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
	forward_ps_region_layer(l, cpu_state);
	//cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
	free(cpu_state.input);
	if (!state.train) return;
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
	if (cpu_state.truth) free(cpu_state.truth);
}

void backward_ps_region_layer_gpu(ps_region_layer l, network_state state)
{
	flatten_ongpu(l.delta_gpu, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 0, state.delta);
}
#endif

