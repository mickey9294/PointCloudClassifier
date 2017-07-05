#ifndef FEATURESLOADER_H
#define FEATURESLOADER_H

#include <fstream>
#include <Eigen/Core>
#include <vector>
#include <list>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/convenience.hpp>
#include <map>

class FeaturesLoader
{
public:
	FeaturesLoader();
	~FeaturesLoader();

	static void load_features_from_files(const std::string features_list_file,
		std::list<Eigen::VectorXd> &features, std::list<int> &labels, std::vector<int> &cluster_count);
};

#endif // !FEATURESLOADER_H