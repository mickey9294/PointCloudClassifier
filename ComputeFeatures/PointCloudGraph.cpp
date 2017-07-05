#include "PointCloudGraph.h"

Node::Node(int _id)
	: id(_id)
{}

Node::Node(int _id, boost::shared_ptr<Node> _parent)
	: id(_id)
{
	parent = _parent;
}

PointCloudGraph::PointCloudGraph()
	: m_distances_mat(NULL)
{
}


PointCloudGraph::~PointCloudGraph()
{
}

void PointCloudGraph::set_input_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud)
{
	m_graph.clear();
	m_distances_mat.reset();

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(pointcloud);
	bool first = true;

	for (int i = 0; i < pointcloud->size(); i++)
	{
		pcl::PointXYZ search_point;
		search_point.x = pointcloud->points[i].x;
		search_point.y = pointcloud->points[i].y;
		search_point.z = pointcloud->points[i].z;

		std::vector<int> pointIdxNKNSearch(num_adjacent_vertices);
		std::vector<float> pointNKNSquaredDistance(num_adjacent_vertices);

		kdtree.nearestKSearch(search_point, num_adjacent_vertices, pointIdxNKNSearch, pointNKNSquaredDistance);
	
		for (int j = 0; j < pointIdxNKNSearch.size(); j++)
		{
			int neighbor_idx = pointIdxNKNSearch[j];
			if (neighbor_idx > i)
			{
				double dist = std::sqrt(pointNKNSquaredDistance[j]);
				boost::add_edge(i, neighbor_idx, dist, m_graph);

				/*if (first)
				{
					std::cout << i << " - " << neighbor_idx << ": " << dist << std::endl;
					first = false;
				}*/
			}
		}
	}
}

void PointCloudGraph::compute_all_pairs_distances()
{
	WeightMap weight_pmap = boost::get(boost::edge_weight, m_graph);
	m_distances_mat.reset(new DistanceMatrix(num_vertices(m_graph)));
	DistanceMatrixMap dm(*m_distances_mat, m_graph);

	std::cout << "Run Floyd-Warshall algorithm." << std::endl;
	bool valid = boost::floyd_warshall_all_pairs_shortest_paths(m_graph, dm, boost::weight_map(weight_pmap));
	std::cout << "done.";

	if (!valid)
	{
		std::cerr << "Error - Negative cycle in matrix" << std::endl;
	}
}

double PointCloudGraph::distance(int index_1, int index_2)
{
	int i, j;
	if (index_1 < index_2)
	{
		i = index_1;
		j = index_2;
	}
	else
	{
		i = index_2;
		j = index_1;
	}
	
	if (!m_distances_mat)
	{
		std::vector<double> d;
		shortest_paths(i, d);

		return d[j];
	}
	else
		return m_distances_mat->operator[](i)[j];
}

void PointCloudGraph::shortest_paths(int vert_index, std::vector<double>& distances)
{
	std::vector<vertex_descriptor> p(num_vertices(m_graph));
	distances.resize(num_vertices(m_graph));
	vertex_descriptor root = boost::vertex(vert_index, m_graph);

	boost::dijkstra_shortest_paths(m_graph, root,
		predecessor_map(boost::make_iterator_property_map(p.begin(), boost::get(boost::vertex_index, m_graph))).
		distance_map(boost::make_iterator_property_map(distances.begin(), boost::get(boost::vertex_index, m_graph))));
}

void PointCloudGraph::check_adjacency(int vert_index)
{
	Graph::out_edge_iterator eit, eend;
	std::tie(eit, eend) = boost::out_edges(vert_index, m_graph);
	for (; eit != eend; ++eit)
	{
		Graph::vertex_descriptor v = boost::target(*eit, m_graph);
		double dist = boost::get(boost::edge_weight_t(), m_graph, *eit);

		std::cout << "neighborhood dist(" << vert_index << ", " << v << ") = " << dist << std::endl;
	}
}

void PointCloudGraph::get_adjacency(int vert_index, std::vector<int> &adjacent_verts, std::vector<double> &adjacent_distances)
{
	adjacent_verts.clear();
	adjacent_distances.clear();
	adjacent_verts.reserve(num_adjacent_vertices);
	adjacent_distances.reserve(num_adjacent_vertices);

	Graph::out_edge_iterator eit, eend;
	std::tie(eit, eend) = boost::out_edges(vert_index, m_graph);
	for (; eit != eend; ++eit)
	{
		Graph::vertex_descriptor v = boost::target(*eit, m_graph);
		double dist = boost::get(boost::edge_weight_t(), m_graph, *eit);

		adjacent_verts.push_back(v);
		adjacent_distances.push_back(dist);
	}
}

void PointCloudGraph::radius_search(int index, double radius, 
	std::vector<int>& neighbors, std::vector<double> &neighborhood_distances)
{
	std::list<boost::shared_ptr<Node>> queue;
	std::set<int> neighbors_set;

	boost::shared_ptr<Node> root(new Node(index, boost::shared_ptr<Node>(NULL)));

	queue.push_back(root);

	std::vector<double> distances(num_vertices(m_graph), std::numeric_limits<double>::max());
	//bool first = true;
	distances[index] = 0;

	while (!queue.empty())
	{
		boost::shared_ptr<Node> current_node = queue.front();
		int current_id = current_node->id;
		queue.pop_front();

		/* get all adjacent vertices of the current node */
		Graph::out_edge_iterator eit, eend;
		std::tie(eit, eend) = boost::out_edges(current_id, m_graph);
		for(; eit != eend; ++eit)
		{
			Graph::vertex_descriptor v = boost::target(*eit, m_graph);
			double dist = boost::get(boost::edge_weight_t(), m_graph, *eit);
			
			if (!is_duplicate(v, current_node))
			{
				double new_dist = distances[current_id] + dist;
				if (new_dist < distances[v])
				{
					distances[v] = new_dist;
					if (new_dist < radius)
					{
						boost::shared_ptr<Node> neighbor_node(new Node(v, current_node));
						//current_node->children.push_back(neighbor_node);
						queue.push_back(neighbor_node);
						neighbors_set.insert(v);
					}
				}
			}
		}
	}

	neighbors.resize(neighbors_set.size());
	neighborhood_distances.resize(neighbors_set.size());
	int idx = 0;
	for (std::set<int>::iterator it = neighbors_set.begin(); it != neighbors_set.end(); ++it, ++idx)
	{
		neighbors[idx] = *it;
		neighborhood_distances[idx] = distances[*it];
	}
}

void PointCloudGraph::nearest_k_search(int index, int k, std::vector<int>& neighbors)
{
	assert(m_distances_mat);
	std::priority_queue<std::pair<double, int>> q;
	for (int i = 0; i < num_vertices(m_graph); i++)
		q.push(std::pair<double, int>(m_distances_mat->operator[](index)[i], i));

	neighbors.resize(k);
	for (int i = 0; i < k; i++)
	{
		int neighbor = q.top().second;
		neighbors[i] = neighbor;
		q.pop();
	}
}

double PointCloudGraph::get_max_distance()
{
	if (!m_distances_mat)
		compute_all_pairs_distances();

	double max = 0;
	for (int i = 0; i < num_vertices(m_graph); i++)
	{
		for (int j = 0; j < num_vertices(m_graph); j++)
		{
			double dist = m_distances_mat->operator[](i)[j];
			max = std::max(max, dist);
		}
	}

	return max;
}

bool PointCloudGraph::is_duplicate(int neighbor_id, boost::shared_ptr<Node> current_vert)
{
	boost::shared_ptr<Node> predecessor = current_vert->parent;
	
	while (predecessor)
	{
		if (neighbor_id == predecessor->id)
			return true;
		predecessor = predecessor->parent;
	}

	return false;
}

int PointCloudGraph::connected_components(std::vector<int> &component_indices)
{
	component_indices.resize(boost::num_vertices(m_graph));
	int num_connected_components = boost::connected_components(m_graph, &component_indices[0]);

	return num_connected_components;
}