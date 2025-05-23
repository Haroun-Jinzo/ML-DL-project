#pragma once

#include "Layer.h"
#include "../data.h"

class InputLayer : public Layer
{
public:
	InputLayer(int prev, int current) : Layer(prev, current){}
	void setLayerOutputs(data* d);
};