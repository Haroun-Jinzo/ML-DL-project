#include"Kmeans.h"

Kmeans::Kmeans(int k)
{
	num_clusters = k;
	clusters = new std::vector<cluster_t *>;
	used_indexes = new std::unordered_set<int>;
}
void Kmeans::initClusters()
{
	for (int i = 0; i < num_clusters; i++)
	{
		int index = rand() % TrainingSet->size();
		while (used_indexes->find(index) != used_indexes->end())
		{
			index = rand() % TrainingSet->size();
		}
		data* initial_point = TrainingSet->at(index);
		cluster_t* new_cluster = new cluster_t(initial_point);
		clusters->push_back(new_cluster);
		used_indexes->insert(index);
	}
}
void Kmeans::initClustersForEachClass()
{
	std::unordered_set<int> classes_used;
	for (int i = 0; i < TrainingSet->size(); i++)
	{
		if (classes_used.find(TrainingSet->at(i)->get_Label()) == classes_used.end())
		{
			data* initial_point = TrainingSet->at(i);
			cluster_t* new_cluster = new cluster_t(initial_point);
			clusters->push_back(new_cluster);
			used_indexes->insert(i);
			classes_used.insert(TrainingSet->at(i)->get_Label());
		}
		else
		{
			continue;
		}
	}
}
void Kmeans::train()
{
	int index = 0;
	while (used_indexes->size() < TrainingSet->size())
	{
		while (used_indexes->find(index) != used_indexes->end())
		{
			index++;
		}
		double min_distance = std::numeric_limits<double>::max();
		int best_cluster_index = 0;
		for (int j = 0; j < clusters->size(); j++)
		{
			double currentDistance = euclidian_distance(clusters->at(j)->centroid, TrainingSet->at(index));
			if (currentDistance < min_distance)
			{
				min_distance = currentDistance;
				best_cluster_index = j;
			}
		}
		clusters->at(best_cluster_index)->add_to_cluster(TrainingSet->at(index));
		used_indexes->insert(index);
	}
}
double Kmeans::euclidian_distance(std::vector<double>* centroid, data* point)
{
	double distance = 0.0f;
	for (int i = 0; i < centroid->size(); i++)
	{
		distance += pow(centroid->at(i) - point->getFeatureVector()->at(i), 2);
	}

	return sqrt(distance);
}
double Kmeans::validate()
{
	double num_Correct = 0.0;
	for (auto queryPoint : *ValidationSet)
	{
		double min_distance = std::numeric_limits<double>::max();
		int best_cluster_index = 0;
		for (int j = 0; j < clusters->size(); j++)
		{
			double currentDistance = euclidian_distance(clusters->at(j)->centroid, queryPoint);
			if (currentDistance < min_distance)
			{
				min_distance = currentDistance;
				best_cluster_index = j;
			}
		}
		if (clusters->at(best_cluster_index)->most_frequent_class == queryPoint->get_Label())
		{
			num_Correct++;
		}
	}
	return 100.0 * (num_Correct / (double)ValidationSet->size());
}
double Kmeans::test()
{
	double num_Correct = 0.0;
	for (auto queryPoint : *TestSet)
	{
		double min_distance = std::numeric_limits<double>::max();
		int best_cluster_index = 0;
		for (int j = 0; j < clusters->size(); j++)
		{
			double currentDistance = euclidian_distance(clusters->at(j)->centroid, queryPoint);
			if (currentDistance < min_distance)
			{
				min_distance = currentDistance;
				best_cluster_index = j;
			}
		}
		if (clusters->at(best_cluster_index)->most_frequent_class == queryPoint->get_Label())
		{
			num_Correct++;
		}
	}
	return 100.0 * (num_Correct / (double)TestSet->size());
}