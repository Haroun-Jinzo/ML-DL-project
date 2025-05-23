#pragma once

#include<vector>
#include"stdint.h"
#include"stdio.h"
#include<memory>

class data
{
    std::vector<uint8_t>* feature_vector; // no class at end
    std::unique_ptr<std::vector<double>> normalizedFeatureVector;
    std::vector<int>* class_vector;
    uint8_t label;
    int enum_label; // A-> 1, B-> 2
    double distance;

public:
    data();
    ~data();
    void setDistance(double val);
    void set_feature_vector(std::vector<uint8_t>*);
    void append_to_feature_vector(uint8_t);
    void setNormalizedFeatureVector(std::unique_ptr<std::vector<double>>);
    void append_to_feature_vector(double);
    void set_class_vector(int count);
    void set_Label(uint8_t);
    void setEnumeratedLabel(int);

    void printVector();
    void printNormalizedVector();

    double getDistance();
    int getFeatureVectorSize();
    uint8_t get_enumeratedLable();
    uint8_t get_Label();

    std::vector<uint8_t>* getFeatureVector();
    std::vector<double>& getNormalizedFeatureVector();
    std::vector<int> getClassVector();


};