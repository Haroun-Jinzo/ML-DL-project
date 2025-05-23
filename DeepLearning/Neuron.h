#pragma once

#include <cmath>
#include <vector>

class Neuron
{
public:
    double output;
    double delta;
    std::vector<double> weights;
    Neuron(int, int);
    void initializeWeights(int);
};