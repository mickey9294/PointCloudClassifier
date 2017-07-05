#ifndef POINTSSAMPLER_H
#define POINTSSAMPLER_H

#include <list>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <algorithm>
#include <boost/filesystem.hpp>
#include "../ComputeFeatures/MyMesh.h"

struct FacetProb
{
	int index;		//the no. of triangle in the tri_coordIndex;
	double area;	//area of the triangle;
	FacetProb(void) {};
	FacetProb(const FacetProb &info)
	{
		this->index = info.index;
		this->area = info.area;
	}
	void SetInfo(const int &index, const double &area)
	{
		this->index = index;
		this->area = area;
	}

};
class PointsSampler
{
public:
	PointsSampler();
	~PointsSampler();

	void batch_sample(const std::string shapes_list_file, const std::string samples_dir,
		const std::string dense_samples_dir, const int num_samples, const int num_dense_samples,
		const std::string modified_shapes_dir, const std::string labels_dir, const std::string output_labels_dir);

	void sample_on_mesh(const MyMesh &mesh, const int num_samples, std::vector<Eigen::Vector3f> &samples,
		std::vector<int> &face_index_map, std::vector<Eigen::Vector3f> &bary_coords);

	void sample_on_one_shape(const std::string shape_file_path, const std::string samples_dir,
		const std::string dense_samples_dir, const int num_samples, const int num_dense_samples,
		const std::string modified_shapes_dir, const std::string labels_dir, const std::string output_labels_dir,
		std::list<std::string> &sample_files_list);

private:
	int load_shapes_paths(const std::string shapes_list_file, std::list<std::string> &shapes_list);
	void setFaceProb(const MyMesh &mesh, std::vector<FacetProb> &faceProbs);
	bool getRandomPoint(const MyMesh &mesh, const std::vector<FacetProb> &faceProbs,
		Eigen::Vector3f &v, Eigen::Vector3f & bary_coord, int & face_idx);
	double TriangleArea(const Eigen::Vector3f &pA, const Eigen::Vector3f &pB, const Eigen::Vector3f &pC);
	void save_sample_points(const std::string save_path, const std::vector<Eigen::Vector3f> &samples,
		const std::vector<int> &face_index_map, const std::vector<Eigen::Vector3f> &bary_coords);

	void load_samples_labels(const std::string mesh_labels_path, const int nfaces, const std::vector<int> &face_index_map, std::vector<int> &labels);
	void save_labels(const std::string save_path, const std::vector<int> &labels);
};

#endif