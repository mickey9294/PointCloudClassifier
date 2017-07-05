#ifndef PARTSSPLITTER_H
#define PARTSSPLITTER_H

#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <Eigen/Core>
#include <set>
#include <map>
#include "..\ComputeFeatures\PointCloudLoader.h"
#include "..\ComputeFeatures\PointCloudGraph.h"

#define PI 3.14159265359

class PartsSplitter
{
public:
	PartsSplitter();
	PartsSplitter(std::string shape_category);
	~PartsSplitter();

	void split_parts(const std::string shapes_list_file, const std::string labels_dir,
		const std::string new_labels_dir, std::vector<Eigen::Vector3f> &order_directions, int force_match = 0);

	void count_parts(const std::string shapes_list_file, const int num_labels, const std::string labels_dir);

	void count_parts_once(const std::string shape_path, const std::string labels_dir, 
		const int num_labels, std::vector<int> &parts_counter);

	void split_once(const std::string shape_path, const std::string labels_dir,
		const std::string new_labels_dir, std::vector<Eigen::Vector3f> &order_directions, int force_match = 0);

private:
	std::vector<int> m_labels;
	std::vector<std::string> m_label_names;
	std::list<std::vector<int>> m_symmetry_groups;
	int m_null_label;
	std::string m_shape_category;
	std::list<std::string> m_wrong_list;

	bool load_label_info(std::string regions_file, bool _verbose = false);
	bool load_regions_symmetry(std::string regions_symmetry_file);
	int get_label_from_label_name(std::string label_name);
};

#endif // !PARTSSPLITTER_H