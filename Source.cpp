#include<vector>
#include "DataHandler.h"
#include "Algorithms/Knn.h"
#include "Algorithms/Kmeans.h"
#include "DeepLearning/Network.h"
#include <iostream>
#include<memory>

int main()
{
    // Data handling
    /*DataHandler* dh = new DataHandler();
    dh->readFeatureVector("train-images-idx3-ubyte");
    dh->readFeatureLabels("train-labels-idx1-ubyte");
    dh->splitData();
    dh->countClasses();*/
	

    // kmeans unspervised training and evaluation
   /* double performance = 0.0f;
    double best_performance = 0.0f;
    int best_k = 1;


    // putthing the set inside dont do shit
    for (int k = dh->getClassCount(); k < dh->getTrainingData()->size() * 0.1; k++)
    {
        Kmeans* km = new Kmeans(k);
        km->set_trainingSet(dh->getTrainingData());
        km->set_testSet(dh->getTestData());
        km->set_validationSet(dh->getValidationData());
        km->initClusters();
        km->train();
        performance = km->validate();
        printf("current training Performance @ K = %d: %0.2f", k, performance);
        if (performance > best_performance)
        {
            best_performance = performance;
            best_k = k;
        }
    }


    Kmeans* km = new Kmeans(best_k);
    km->set_trainingSet(dh->getTrainingData());
    km->set_testSet(dh->getTestData());
    km->set_validationSet(dh->getValidationData());
    km->initClusters();
    performance = km->test();
    printf("current testing Performance @ K = %d: %0.2f", best_k, performance);
    */


    // deep learning training
    auto dataHandler = std::make_unique<DataHandler>();

    /*dataHandler->readInputData("../train-images-idx3-ubyte");
    dataHandler->readLabelData("../train-labels-idx1-ubyte");
    dataHandler->countClasses();*/

    //encoding the specific dataset features
    DataConfig config;
    config.columnRules = {
        {0, FeatureType::Numerical},    // Age
        {1, FeatureType::Categorical}, // Gender
        {2, FeatureType::Categorical}, // Education
        {3, FeatureType::Ignore},      // Job Title
        {4, FeatureType::Numerical}    // Experience
    };
    config.labelColumn = 5;
    config.categoricalEncodings = {  
           {1, {{"Male", 0.0}, {"Female", 1.0}}},  
           {2, {{"Bachelor's", 0.0}, {"Master's", 1.0}, {"PhD", 2.0}}}  };

    config.autoGenerateEncodings = true;

    try {
        dataHandler->read_csv("Salary-Data.csv", config, ",");
        //dataHandler->read_csv("iris.data", ",");
        dataHandler->countClasses();
        dataHandler->splitData();
        dataHandler->normalize();

        // Get data as references to unique_ptr vectors
        auto& trainingData = dataHandler->getTrainingData();
        auto& testData = dataHandler->getTestData();
        auto& validationData = dataHandler->getValidationData();

        // Verify we have data before proceeding
        if (trainingData.empty() || testData.empty()) {
            throw std::runtime_error("No training/test data available");
        }

        // Deep learning setup
        std::vector<int> hiddenLayers = { 128, 64 };
        const int inputSize = trainingData.front()->getNormalizedFeatureVector().size();
        const int numClasses = dataHandler->getClassCount();

        auto net = std::make_unique<Network>(
            hiddenLayers,
            inputSize,
            numClasses,
            0.001,
            0.0001,
            0.3
        );

        // Pass data references to network
        net->set_trainingSet(std::move(dataHandler->getTrainingData()));
        net->set_testSet(std::move(dataHandler->getTestData()));
        net->set_validationSet(std::move(dataHandler->getValidationData()));

        net->train(50);
        net->validate();
        printf("Test Performance: %.3f\n", net->test());

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}