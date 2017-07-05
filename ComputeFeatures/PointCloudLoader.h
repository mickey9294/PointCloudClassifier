#ifndef POINTCLOUDLOADER_H
#define POINTCLOUDLOADER_H

#include "MyMesh.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <time.h>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <vector>
#include <list>
#include "Seb.h"

typedef Seb::Point<float> MiniPoint;
typedef std::vector<MiniPoint> PointVector;
typedef Seb::Smallest_enclosing_ball<float> Miniball;

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

class PointCloudLoader
{
public:
	PointCloudLoader();
	~PointCloudLoader();

	static bool load_pointcloud(const std::string &file_path, pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud);
	static bool load_pointcloud(const std::string &file_path, const std::string labels_dir, pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, 
		std::vector<int> &points_labels, std::vector<int> &face_index_map, std::vector<Eigen::Vector3f> &bary_coords);
	static void sample_on_mesh(const MyMesh &mesh, const int num_samples, pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, 
		std::vector<int> &face_index_map, std::vector<Eigen::Vector3f> &bary_coords);

private:
	static void setFaceProb(const MyMesh &mesh, std::vector<FacetProb> &faceProbs);
	static bool getRandomPoint(const MyMesh &mesh, const std::vector<FacetProb> &faceProbs,
		Eigen::Vector3f &v, Eigen::Vector3f & bary_coord, int & face_idx);
	static double TriangleArea(const Eigen::Vector3f &pA, const Eigen::Vector3f &pB, const Eigen::Vector3f &pC);
};

#endif