#ifndef FEATURESEXTRACTOR_H
#define FEATURESEXTRACTOR_H

#include <string>
#include <list>
#include <vector>
#include <fstream>
#include <pcl/io/ply_io.h>

#include "FeaturesEstimator.h"
#include "PointCloudLoader.h"

class FeaturesExtractor
{
public:
	FeaturesExtractor();
	FeaturesExtractor(const std::string shape_category);
	~FeaturesExtractor();

	int num_threads = 4;

	void set_shape_category(const std::string shape_category);
	void load_shapes_list(const std::string shapes_list_path);
	void extract_features(const std::string shapes_list_path, const std::string output_dir, const std::string labels_dir);

private:
	std::string m_shape_category;
	std::string m_output_dir;
	std::list<std::string> m_shapes_list;
	std::string m_off_dir;

	boost::shared_mutex dir_mutex;
	boost::shared_mutex name_mutex;
	//boost::mutex cout_mutex;

	void extract_features_once(int id, const std::list<std::string> &sub_list, const std::string labels_dir);
	void save_labels(const std::string save_path, const std::vector<int> &labels);
	void save_pointcloud(const std::string save_path, pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud);
	void save_pointcloud(const std::string save_path, pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud,
		const std::vector<int> &face_index_map, const std::vector<Eigen::Vector3f> &bary_coords);
};

#endif // !FEATURESEXTRACTOR_H