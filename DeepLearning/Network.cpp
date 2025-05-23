#include "Network.h"
#include "Layer.h"
#include "../DataHandler.h"
#include <numeric>
#include <cassert>

Network::Network(std::vector<int> spec, int inputSize, int numClasses, double learningRate, double l2, double dropout)
    : CommonData(), learningRate(learningRate), lambda(l2), dropoutRate(dropout)
{
    for (int i = 0; i < spec.size(); i++)
    {
        if (i == 0)
            layers.push_back(new Layer(inputSize, spec.at(i)));
        else
            layers.push_back(new Layer(layers.at(i - 1)->neurons.size(), spec.at(i)));

    }
    layers.push_back(new Layer(layers.at(layers.size() - 1)->neurons.size(), numClasses));
    this->learningRate = learningRate;
}

Network::~Network() {}

double Network::activate(std::vector<double> weights, std::vector<double> input)
{
    double activation = weights.back(); // bias term
    for (int i = 0; i < weights.size() - 1; i++)
    {
        activation += weights[i] * input[i];
    }
    return activation;
}

double Network::transfer(double activation)
{
    return std::max(0.0, activation);
}

double Network::transferDerivative(double output)
{
    return output > 0 ? 1.0 : 0.0;
}

std::vector<double> Network::fprop(data* data)
{
    if (!data || data->getNormalizedFeatureVector().empty()) {
        throw std::invalid_argument("Invalid data sample");
    }

    std::vector<double> inputs = data->getNormalizedFeatureVector();

    // Process hidden layers with ReLU
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        Layer* layer = layers[i];
        std::vector<double> newInputs;
        newInputs.reserve(layer->neurons.size());

        for (Neuron* neuron : layer->neurons) {
            if (neuron->weights.size() != inputs.size() + 1) {
                throw std::runtime_error("Weight/input dimension mismatch");
            }

            double activation = activate(neuron->weights, inputs);
            neuron->output = transfer(activation);  // ReLU
            newInputs.push_back(neuron->output);
        }

        inputs = std::move(newInputs);
    }

    // Process output layer with softmax
    Layer* outputLayer = layers.back();
    std::vector<double> activations;
    activations.reserve(outputLayer->neurons.size());

    // Get raw activations (pre-softmax)
    for (Neuron* neuron : outputLayer->neurons) {
        if (neuron->weights.size() != inputs.size() + 1) {
            throw std::runtime_error("Output layer weight/input mismatch");
        }
        double activation = activate(neuron->weights, inputs);
        activations.push_back(activation);
    }

    // Apply softmax
    std::vector<double> outputs = softmax(activations);

    // Store softmax results in output neurons
    for (size_t i = 0; i < outputLayer->neurons.size(); ++i) {
        outputLayer->neurons[i]->output = outputs[i];
    }

    return outputs;
}

std::vector<double> Network::softmax(const std::vector<double>& activations) {
    std::vector<double> exponents;
    double max_activation = *std::max_element(activations.begin(), activations.end());
    double sum = 0.0;

    // Numerical stability: subtract max before exp
    for (double a : activations) {
        double exp_a = exp(a - max_activation);
        exponents.push_back(exp_a);
        sum += exp_a;
    }

    std::vector<double> outputs;
    for (double exp_a : exponents) {
        outputs.push_back(exp_a / (sum + 1e-8));  // Avoid division by zero
    }

    return outputs;
}

void Network::bprop(data* data)
{
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        Layer* layer = layers.at(i);
        std::vector<double> errors;
        if (i != layers.size() - 1)
        {
            for (int j = 0; j < layer->neurons.size(); j++)
            {
                double error = 0.0;
                for (Neuron* n : layers.at(i + 1)->neurons)
                {
                    error += (n->weights.at(j) * n->delta);
                }
                errors.push_back(error);
            }
        }
        else {
            for (int j = 0; j < layer->neurons.size(); j++)
            {
                Neuron* n = layer->neurons.at(j);
                errors.push_back(n->output - (double)data->getClassVector().at(j)); // expected - actual
            }
        }
        for (int j = 0; j < layer->neurons.size(); j++)
        {
            Neuron* n = layer->neurons.at(j);
            n->delta = errors.at(j) * this->transferDerivative(n->output); //gradient / derivative part of back prop.
        }
    }
}

void Network::updateWeights(data* data)
{
    if (m.empty()) initializeAdam();
    timestep++;

    size_t weightIdx = 0;  // Track position in m/v vectors
    std::vector<double> inputs = data->getNormalizedFeatureVector();

    for (size_t i = 0; i < layers.size(); ++i) {
        Layer* layer = layers[i];

        // Build inputs for next layer
        if (i != 0) {
            for (Neuron* n : layers[i - 1]->neurons) {
                inputs.push_back(n->output);
            }
        }

        // Update each neuron's weights
        for (Neuron* n : layer->neurons) {
            // Adam optimization for each weight
            for (size_t j = 0; j < n->weights.size(); ++j) {
                double g = n->delta * (j < inputs.size() ? inputs[j] : 1.0);  // Handle bias

                // Update moments
                m[weightIdx][j] = beta1 * m[weightIdx][j] + (1 - beta1) * g;
                v[weightIdx][j] = beta2 * v[weightIdx][j] + (1 - beta2) * g * g;

                // Bias-corrected estimates
                double m_hat = m[weightIdx][j] / (1 - pow(beta1, timestep));
                double v_hat = v[weightIdx][j] / (1 - pow(beta2, timestep));

                // Apply update
                n->weights[j] -= learningRate * m_hat / (sqrt(v_hat) + 1e-8);
            }
            weightIdx++;
        }
        inputs.clear();
    }
}

int Network::predict(data* data)
{
    std::vector<double> outputs = fprop(data);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void Network::train(int numEpochs)  
{  
    if (TrainingSet.empty()) {
        throw std::runtime_error("No training data available");
    }

    const size_t inputSize = TrainingSet[0]->getNormalizedFeatureVector().size();
    if (layers.empty() || layers[0]->neurons.empty()) {
        throw std::logic_error("Network not properly initialized");
    }

    // Validate first layer input size
    if (layers[0]->neurons[0]->weights.size() != inputSize + 1) { // +1 for bias
        std::string err = "Input size mismatch. Expected " +
            std::to_string(inputSize) +
            " features, first layer expects " +
            std::to_string(layers[0]->neurons[0]->weights.size() - 1);
        throw std::runtime_error(err);
    }

    const size_t numOutputs = layers.back()->neurons.size();

    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        double sumError = 0.0;

        learningRate *= 0.95;  // Decay every epoch
        learningRate = std::max(learningRate, 1e-5);  // Minimum rate

        for (const auto& sample : TrainingSet) {
            // Forward pass with validation
            std::vector<double> outputs = fprop(sample.get());
            const std::vector<int>& expected = sample->getClassVector();

            // Validate dimensions
            if (outputs.size() != numOutputs) {
                throw std::runtime_error("Output size mismatch");
            }
            if (expected.size() != numOutputs) {
                throw std::runtime_error("Label size mismatch");
            }

            // Calculate error
            double error = crossEntropyLoss(outputs, expected);
            sumError += error;

            // Backprop and update
            bprop(sample.get());
            updateWeights(sample.get());
        }

        printf("Epoch: %d \t Error=%.4f\n", epoch, sumError);

        // Early stopping check
        validate();
        if (sumError < 0.001) break;
    }
}

double Network::test()
{
    double numCorrect = 0.0;
    double count = 0.0;
    for (const auto& data : TestSet)
    {
        count++;
        int index = predict(data.get());
        if (data->getClassVector().at(index) == 1)
        {
            numCorrect++;
        }
    }

    testPerformance = (numCorrect / count);
    return testPerformance;
}

void Network::validate()
{
    double numCorrect = 0.0;
    double count = 0.0;
    for (const auto& data : ValidationSet)
    {
        count++;
        int index = predict(data.get());
        if (data->getClassVector().at(index) == 1)
        {
            numCorrect++;
        }

    }
    printf("Validation Performance: %.4f\n", numCorrect / count);
}

void Network::saveBestWeights()
{
    bestWeights.clear();
    for (const auto& layer : layers) {
        for (const auto& neuron : layer->neurons) {
            bestWeights.push_back(neuron->weights);
        }
    }
}

void Network::restoreBestWeights()
{
    if (!bestWeights.empty()) {
        size_t idx = 0;
        for (auto& layer : layers) {
            for (auto& neuron : layer->neurons) {
                if (idx < bestWeights.size()) {
                    neuron->weights = bestWeights[idx++];
                }
            }
        }
    }
}

double Network::crossEntropyLoss(const std::vector<double>& outputs, const std::vector<int>& expected)
{
    double loss = 0.0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        loss += -expected[i] * log(outputs[i] + 1e-8);
    }
    return loss;
}
double Network::outputTransfer(double activation, const std::vector<double>& layerOutputs)
{
    double sum = 0.0;
    for (const auto& output : layerOutputs) {
        sum += exp(output);
    }
    return exp(activation) / (sum + 1e-8);
}

void Network::initializeAdam()
{
    m.clear();
    v.clear();
    for (const auto& layer : layers) {
        for (const auto& neuron : layer->neurons) {
            m.push_back(std::vector<double>(neuron->weights.size(), 0.0));
            v.push_back(std::vector<double>(neuron->weights.size(), 0.0));
        }
    }
}