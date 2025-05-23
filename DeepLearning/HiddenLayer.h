#pragma once

#include "Layer.h"

class HiddenLayer : public Layer
{
public:
	HiddenLayer(int prev, int current) : Layer(prev, current){}
	void feedForward(Layer perv);
	void backProp(Layer next);
	void updateWeights(double, Layer*);
};