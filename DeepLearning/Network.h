#pragma once

#include "../data.h"
#include "Neuron.h"
#include "Layer.h"
#include "HiddenLayer.h"
#include "InputLayer.h"
#include "OutputLayer.h"
#include"../common.h"

class Network : public CommonData
{
public:
    std::vector<Layer*> layers;
    double learningRate;
    double testPerformance = 0;
    double lambda;  // L2 regularization parameter
    double dropoutRate;

    std::vector<std::vector<double>> bestWeights;
    double bestValidation = 0.0;
    int patienceCounter = 0;

    std::vector<std::vector<double>> m; // First moment vector
    std::vector<std::vector<double>> v; // Second moment vector
    double beta1 = 0.9;
    double beta2 = 0.999;
    int timestep = 0;

    void initializeAdam();

    Network(std::vector<int> spec, int inputSize, int numClasses, double lr, double l2 = 0.01, double dropout = 0.5f);
    ~Network();


    std::vector<double> fprop(data* data);
    double activate(std::vector<double>, std::vector<double>); // dot product
    double transfer(double);
    double transferDerivative(double); // used for backprop
    void bprop(data* data);
    void updateWeights(data* data);
    int predict(data* data); // return the index of the maximum value in the output array.
    void train(int); // num iterations
    double test();
    void validate();

    void saveBestWeights();
    void restoreBestWeights();
    double crossEntropyLoss(const std::vector<double>& outputs, const std::vector<int>& expected);
    double outputTransfer(double activation, const std::vector<double>& layerOutputs);

    std::vector<double> softmax(const std::vector<double>& activations);


};