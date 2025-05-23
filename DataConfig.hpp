#pragma once
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <string>

enum class FeatureType
{
	Numerical,
	Categorical,
	Ignore,
};

struct DataConfig
{
	std::unordered_map<int, FeatureType> columnRules;

	std::unordered_map<int, std::unordered_map<std::string, double>> categoricalEncodings;

	int labelColumn;

	bool autoGenerateEncodings = false;
};