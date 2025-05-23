#pragma once
#define _CRT_SECURE_NO_DEPRECATE

#include "fstream"
#include "stdint.h"
#include "data.h"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <math.h>
#include <random>
#include <algorithm>
#include <memory>
#include <cmath>
#include <stdexcept>
#include "DataConfig.hpp"
#include <unordered_map>
#include <sstream>
#include <cctype>

class DataHandler
{
    std::vector<std::unique_ptr<data>> dataArray; // all of the data pre-Split
    std::vector<std::unique_ptr<data>> trainingData;
    std::vector<std::unique_ptr<data>> testData;
    std::vector<std::unique_ptr<data>> validationData;

    int numClasses;
    int featureVectorSize;
    std::map<uint8_t, int> classFromInt;
    std::map<std::string, int> classFromString; //string key

    const double TrainSetPercent = 0.75;
    const double TestSetPercent = 0.20;
    const double ValidationPercent = 0.05;

public:
    DataHandler();
    ~DataHandler();

    void read_csv(const std::string& path, const DataConfig& config, const std::string& delimiter);
    void read_csv(std::string, std::string);
    void readFeatureVector(const std::string& path);
    void readFeatureLabels(std::string path);
    void splitData();
    void countClasses();
    void normalize();
    void print();

    uint32_t convertToLittleEndian(const unsigned char* bytes);

    int getClassCount();
    int getDataArraySize();
    int getTrainingDataSize();
    int getTestDataSize();
    int getValidationSize();
    void Trim(std::string& s);

    uint32_t format(const unsigned char* bytes);

    std::vector<std::unique_ptr<data>>& getTrainingData();
    std::vector<std::unique_ptr<data>>& getTestData();
    std::vector<std::unique_ptr<data>>& getValidationData();

private:
    std::unordered_map<int, std::unordered_map<std::string, double>> categoricalMaps;
    void processLine(const std::string& line, const DataConfig& config, const std::string& delimiter, size_t lineNumber);
    double handleNumerical(const std::string& token);
    double handleCategorical(int colIndex, const std::string& token, const DataConfig& config);
    void handleLabel(const std::string& token, std::unique_ptr<data>& sample, const DataConfig& config);
};