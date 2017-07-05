#ifndef FEATURESESTIMATOR_H
#define FEATURESESTIMATOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <Eigen/Core>
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include "Seb.h"
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/impl/centroid.hpp>
#include "PointCloudGraph.h"
#include <assert.h>
#include <algorithm>

#define DIMEN 36
#define MAX_DISTANCE 1.3

class FeaturesEstimator
{
public:
	FeaturesEstimator();
	FeaturesEstimator(int _pid);
	~FeaturesEstimator();

	boost::shared_mutex cloud_mutex;
	boost::shared_mutex graph_mutex;

	static const int num_bins = 8;

	void set_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud);
	void compute_features(Eigen::MatrixXd &features);
	void compute_part_of_features(int id, boost::promise<Eigen::MatrixXd> &promise);

private:
	int pid;
	pcl::PointCloud<pcl::PointXYZ>::Ptr m_pointcloud;
	PointCloudGraph m_graph;

};

#endif