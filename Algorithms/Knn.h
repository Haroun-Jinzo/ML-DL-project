#pragma once

#include"../common.h"

class KNN : public CommonData
{
	int k; // Number of neighbors
	std::vector<data*>* neightbors;

public:
	KNN(int);
	KNN();
	~KNN();

	void find_knearest(data* query_point);
	void set_k(int k);

	int predict();
	double calculate_distance(data* query_point, data* input);
	double validate_performance();
	double test_performance();
};