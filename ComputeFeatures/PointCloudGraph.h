#ifndef POINTCLOUDGRAPH_H
#define POINTCLOUDGRAPH_H

#include <iostream>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/shared_ptr.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <cmath>
#include <queue>
#include <set>
#include <algorithm>

// type for weight/distance on each edge
typedef double t_weight;

// define the graph type
typedef boost::property<boost::edge_weight_t, t_weight> EdgeWeightProperty;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
	boost::no_property, EdgeWeightProperty> Graph;

typedef boost::property_map<Graph, boost::edge_weight_t>::type WeightMap;
typedef boost::graph_traits < Graph >::vertex_descriptor vertex_descriptor;

// Declare a matrix type and its corresponding property map that
// will contain the distances between each pair of vertices.
typedef boost::exterior_vertex_property<Graph, t_weight> DistanceProperty;
typedef DistanceProperty::matrix_type DistanceMatrix;
typedef DistanceProperty::matrix_map_type DistanceMatrixMap;

class Node {
public:
	Node(int _id);
	Node(int _id, boost::shared_ptr<Node> _parent);

	int id;
	boost::shared_ptr<Node> parent;
	//std::list<boost::shared_ptr<Node>> children;
};

class PointCloudGraph
{
public:
	PointCloudGraph();
	~PointCloudGraph();

	void set_input_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud);
	double distance(int index_1, int index_2);
	void shortest_paths(int vert_index, std::vector<double> & distances);
	void radius_search(int index, double radius, 
		std::vector<int> &neighbors, std::vector<double> &neighborhood_distances);
	void nearest_k_search(int index, int k, std::vector<int> &neighbors);
	void compute_all_pairs_distances();
	double get_max_distance();
	void check_adjacency(int vert_index);
	void get_adjacency(int vert_index, std::vector<int> &adjacent_verts, std::vector<double> &adjacent_distances);
	int connected_components(std::vector<int> &component_indices);

	static const int num_adjacent_vertices = 10;

private:
	Graph m_graph;
	boost::shared_ptr<DistanceMatrix> m_distances_mat;

	bool is_duplicate(int neighbor_id, boost::shared_ptr<Node> current_vert);
};

#endif