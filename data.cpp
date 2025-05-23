#include"data.h"

data::data()
{
    feature_vector = new std::vector<uint8_t>;
}

data::~data()
{
    //free allocated memeory
}

void data::setDistance(double dist)
{
    distance = dist;
}

void data::set_feature_vector(std::vector<uint8_t>* vect)
{
    feature_vector = vect;
}

void data::setNormalizedFeatureVector(std::unique_ptr<std::vector<double>> vect)  
{  
   normalizedFeatureVector = std::move(vect);  
}

void data::append_to_feature_vector(uint8_t val)
{
    feature_vector->push_back(val);
}


void data::append_to_feature_vector(double val)
{
    normalizedFeatureVector->push_back(val);
}


void data::set_Label(uint8_t val)
{
    label = val;
}

void data::setEnumeratedLabel(int val)
{
    enum_label = val;
}

void data::set_class_vector(int count)
{
    class_vector = new std::vector<int>();
    for (int i = 0; i < count; i++)
    {
        if (i == label)
            class_vector->push_back(1);
        else
            class_vector->push_back(0);
    }
}

void data::printVector()
{
    printf("[ ");
    for (uint8_t val : *feature_vector)
    {
        printf("%u ", val);
    }
    printf("]\n");
}

void data::printNormalizedVector()
{
    printf("[ ");
    for (auto val : *normalizedFeatureVector)
    {
        printf("%.2f ", val);
    }
    printf("]\n");

}

int data::getFeatureVectorSize()
{
    return feature_vector->size();
}

uint8_t data::get_Label()
{
    return label;
}

uint8_t data::get_enumeratedLable()
{
    return enum_label;
}

std::vector<uint8_t>* data::getFeatureVector()
{
    return feature_vector;
}

std::vector<double>& data::getNormalizedFeatureVector()
{  
    return *normalizedFeatureVector;
}

double data::getDistance()
{
    return distance;
}

std::vector<int> data::getClassVector()
{
    return *class_vector;
}
