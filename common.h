#pragma once

#include "data.h"
#include <vector>

class CommonData
{
protected:
	std::vector<std::unique_ptr<data>> TrainingSet;
	std::vector<std::unique_ptr<data>> TestSet;
	std::vector<std::unique_ptr<data>> ValidationSet;
public:
	CommonData() = default;
	void set_trainingSet(std::vector<std::unique_ptr<data>>&& trainingSet);
	void set_testSet(std::vector<std::unique_ptr<data>>&& testSet);
	void set_validationSet(std::vector<std::unique_ptr<data>>&& validationSet);
};
