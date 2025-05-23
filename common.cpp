#include "common.h"

void CommonData::set_trainingSet(std::vector<std::unique_ptr<data>>&& vect)
{
	TrainingSet = std::move(vect);
}
void CommonData::set_testSet(std::vector<std::unique_ptr<data>>&& vect)
{
	TestSet = std::move(vect);
}
void CommonData::set_validationSet(std::vector<std::unique_ptr<data>>&& vect)
{
	ValidationSet = std::move(vect);
}
