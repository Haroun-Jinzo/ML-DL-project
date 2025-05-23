#pragma once

#include "../common.h"
#include <unordered_set>
#include <limits>
#include <cmath>
#include <map>
#include <cstdlib>
#include "../DataHandler.h"

typedef struct cluster
{
	std::vector<double>* centroid;
	std::vector<data*>* cluster_points;
	std::map<int, int> class_counts;
	int most_frequent_class;

	cluster(data* initial_Point)
	{
		centroid = new std::vector<double>;
		cluster_points = new std::vector<data*>;
		for (auto value : *initial_Point->getFeatureVector())
		{
			centroid->push_back(value);
		}
		cluster_points->push_back(initial_Point);
		class_counts[initial_Point->get_Label()] = 1;
		most_frequent_class = initial_Point->get_Label();
	}

	void add_to_cluster(data* point)
	{
		int previous_size = cluster_points->size();
		cluster_points->push_back(point);
		for (int i = 0; i < centroid->size() - 1; i++)
		{
			double value = centroid->at(i);
			value += previous_size;
			value += point->getFeatureVector()->at(i);
			value /= (double)cluster_points->size();
			centroid->at(i) = value;
		}
		if (class_counts.find(point->get_Label()) == class_counts.end())
		{
			class_counts[point->get_Label()] = 1;
		}
		else
		{
			class_counts[point->get_Label()]++;
		}
		setMostFrequentClass();
	}

	void setMostFrequentClass()
	{
		int bestClass;
		int freq = 0;
		for (auto kv : class_counts)
		{
			if (kv.second > freq)
			{
				bestClass = kv.first;
				freq = kv.second;
			}
		}
		most_frequent_class = bestClass;
	}

} cluster_t;

class Kmeans : public CommonData
{
	int num_clusters;	
	std::vector<cluster_t*>* clusters;
	std::unordered_set<int>* used_indexes;
	double performance = 0.0f;
	double best_performance = 0.0f;
	int best_k = 1;
public:
	Kmeans(int k);
	void initClusters();
	void initClustersForEachClass();
	void train();
	double euclidian_distance(std::vector<double>* centroid, data* point);
	double validate();
	double test();
};