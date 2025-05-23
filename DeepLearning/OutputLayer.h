#pragma once

#include "Layer.h"
#include "../data.h"

class OutputLayer : public Layer
{
public:
	OutputLayer(int prev, int current) : Layer(prev, current) {}
	void feedForward(Layer);
	void backProp(data* data);
	void updateWeights(double, Layer*);
};