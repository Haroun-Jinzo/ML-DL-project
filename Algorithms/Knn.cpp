#include "Knn.h"
#include<cmath>
#include<limits>
#include<map>
#include"stdint.h"
#include"../DataHandler.h"
#define EUCLID

KNN::KNN(int val)
{
	k = val;
}

KNN::KNN()
{

}
KNN::~KNN()
{

}

void KNN::find_knearest(data* query_point)
{
	neightbors = new std::vector<data*>;
	double current_min = std::numeric_limits<double>::max();
	double previous_min = current_min;
	int index = 0;
	for (int i = 0; i < k; i++)
	{
		if (i == 0)
		{
			for (int j = 0; j < TrainingSet->size(); j++)
			{
				double distance = calculate_distance(query_point, TrainingSet->at(j));
				TrainingSet->at(j)->setDistance(distance);
				if (distance < current_min)
				{
					current_min = distance;
					index = j;
				}
			}
			neightbors->push_back(TrainingSet->at(index));
			previous_min = current_min;
			current_min = std::numeric_limits<double>::max();
		}
		else
		{
			for (int j = 0; j < TrainingSet->size(); j++)
			{
				double distance = calculate_distance(query_point, TrainingSet->at(j));
				TrainingSet->at(j)->setDistance(distance);
				if (distance < current_min && distance > previous_min)
				{
					current_min = distance;
					index = j;
				}
			}
			neightbors->push_back(TrainingSet->at(index));
			previous_min = current_min;
			current_min = std::numeric_limits<double>::max();
		}
	}
}

void KNN::set_k(int val)
{
	this->k = val;
}

int KNN::predict()
{
	std::map<uint8_t, int> class_map;
	for (int i = 0; i < neightbors->size(); i++)
	{
		uint8_t label = neightbors->at(i)->get_Label();
		if (class_map.find(label) == class_map.end())
		{
			class_map[label] = 1;
		}
		else
		{
			class_map[label]++;
		}
	}
	int max_count = 0;
	uint8_t predicted_label = 0;
	for (const auto& pair : class_map)
	{
		if (pair.second > max_count)
		{
			max_count = pair.second;
			predicted_label = pair.first;
		}
	}
	delete neightbors; // Free the memory allocated for neighbors
	return predicted_label;
}
double KNN::calculate_distance(data* query_point, data* input)
{
	double distance = 0.0f;
	if (query_point->getFeatureVectorSize() != input->getFeatureVectorSize())
	{
		throw std::invalid_argument("Feature vector sizes do not match");
	}
#ifdef EUCLID
	for (unsigned i = 0; i < query_point->getFeatureVectorSize(); i++)
	{
		distance += pow(query_point->getFeatureVector()->at(i) - input->getFeatureVector()->at(i), 2);
	}
	distance = sqrt(distance);
#elif defined MANHATTAN
#endif
	return distance;
}
double KNN::validate_performance()
{
	double current_performance = 0;
	int count = 0;
	int data_index = 0;
	for (data* query_point : *ValidationSet)
	{
		find_knearest(query_point);
		int predicted_label = predict();
		if (predicted_label == query_point->get_Label())
		{
			count++;
		}
		data_index++;
		printf("current performance = %.3f %%\n", ((double)count*100.0f)/((double)data_index));
	}
	current_performance = ((double)count * 100.0f) / ((double)ValidationSet->size());
	printf("Validation performance for k = %d: %.3f %%\n",k, current_performance);
	return current_performance;
}
double KNN::test_performance()
{
	double current_performance = 0;
	int count = 0;
	for (data* query_point : *TestSet)
	{
		find_knearest(query_point);
		int predicted_label = predict();
		if (predicted_label == query_point->get_Label())
		{
			count++;
		}
	}
	current_performance = ((double)count * 100.0f) / ((double)TestSet->size());
	printf("Test performance = %.3f %%\n", current_performance);
	return current_performance;
}