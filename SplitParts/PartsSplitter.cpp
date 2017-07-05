#include "PartsSplitter.h"



PartsSplitter::PartsSplitter()
{
}

PartsSplitter::PartsSplitter(std::string shape_category)
	: m_shape_category(shape_category)
{
	std::string data_dir = "D:\\Projects\\shape2pose\\data\\";
	std::string body_dir = data_dir + "0_body\\";
	std::string info_dir = body_dir + m_shape_category + "\\";
	std::string regions_file = info_dir + "regions.txt";
	std::string regions_symmetry_file = info_dir + "regions_symmetry.txt";
	load_label_info(regions_file);
	load_regions_symmetry(regions_symmetry_file);
}


PartsSplitter::~PartsSplitter()
{
}

void PartsSplitter::split_parts(const std::string shapes_list_file, const std::string labels_dir, 
	const std::string new_labels_dir, std::vector<Eigen::Vector3f>& order_directions, int force_match)
{
	std::list<std::string> shapes_list;
	std::ifstream list_in(shapes_list_file.c_str());
	if (list_in.is_open())
	{
		while (!list_in.eof())
		{
			std::string line;
			std::getline(list_in, line);
			if (line.length() > 0)
			{
				shapes_list.push_back(line);
			}
		}

		list_in.close();
	}
	else
	{
		std::cerr << "Error: cannot open list file " << shapes_list_file << std::endl;
	}

	for (std::list<std::string>::iterator shape_it = shapes_list.begin();
		shape_it != shapes_list.end(); ++shape_it)
	{
		split_once(*shape_it, labels_dir, new_labels_dir, order_directions, force_match);
	}

	/* Output wrong list */
	if (!m_wrong_list.empty())
	{
		std::string wrong_out_path = new_labels_dir + "\\..\\wrong_shapes_list.txt";
		std::ofstream wrong_out(wrong_out_path.c_str());
		if (wrong_out.is_open())
		{
			for (std::list<std::string>::iterator wrong_it = m_wrong_list.begin();
				wrong_it != m_wrong_list.end(); ++wrong_it)
			{
				wrong_out << *wrong_it << std::endl;
			}
		}
		else
			std::cerr << "Error: cannot save wrong shapes list to " << wrong_out_path << std::endl;
	}
}

void PartsSplitter::count_parts(const std::string shapes_list_file, const int num_labels, const std::string labels_dir)
{
	std::list<std::string> shapes_list;
	std::ifstream list_in(shapes_list_file.c_str());
	if (list_in.is_open())
	{
		while (!list_in.eof())
		{
			std::string line;
			std::getline(list_in, line);
			if (line.length() > 0)
			{
				shapes_list.push_back(line);
			}
		}

		list_in.close();
	}
	else
	{
		std::cerr << "Error: cannot open list file " << shapes_list_file << std::endl;
	}

	std::vector<int> parts_counter(num_labels, 0);
	for (std::list<std::string>::iterator shape_it = shapes_list.begin();
		shape_it != shapes_list.end(); ++shape_it)
	{
		count_parts_once(*shape_it, labels_dir, num_labels, parts_counter);
	}

	for (int i = 0; i < parts_counter.size(); i++)
	{
		std::cout << "Part_" << i << " has " << parts_counter[i] << " point clusters" << std::endl;
	}
}

void PartsSplitter::count_parts_once(const std::string shape_path, const std::string labels_dir, 
	const int num_labels, std::vector<int>& parts_counter)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<int> original_labels, face_index_map;
	std::vector<Eigen::Vector3f> bary_coords;

	PointCloudLoader::load_pointcloud(shape_path, labels_dir, pointcloud, original_labels, face_index_map, bary_coords);

	std::vector<std::list<int>> original_parts(num_labels);

	for (int i = 0; i < original_labels.size(); i++)
	{
		int original_label = original_labels[i];
		original_parts[original_label].push_back(i);
	}

	int idx = 0;
	for (std::vector<std::list<int>>::iterator part_it = original_parts.begin();
		part_it != original_parts.end(); ++part_it, ++idx)
	{
		if (part_it->size() < 3)
			continue;

		pcl::PointCloud<pcl::PointXYZ>::Ptr part_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		part_cloud->width = part_it->size();
		part_cloud->height = 1;
		part_cloud->is_dense = false;
		part_cloud->points.resize(part_cloud->width * part_cloud->height);

		std::vector<int> part_to_origin_map(part_it->size());

		int jdx = 0;
		for (std::list<int>::iterator point_it = part_it->begin(); point_it != part_it->end(); ++point_it, ++jdx)
		{
			int point_index = *point_it;
			part_cloud->points[jdx].x = pointcloud->points[point_index].x;
			part_cloud->points[jdx].y = pointcloud->points[point_index].y;
			part_cloud->points[jdx].z = pointcloud->points[point_index].z;
			part_to_origin_map[jdx] = point_index;
		}

		boost::shared_ptr<PointCloudGraph> graph(new PointCloudGraph());
		graph->set_input_pointcloud(part_cloud);

		std::vector<int> components_indices;
		int num_components = graph->connected_components(components_indices);

		std::vector<int> cluster_counter(num_components, 0);
		int real_num_components = 0;
		for (std::vector<int>::iterator it = components_indices.begin(); it != components_indices.end(); ++it)
		{
			cluster_counter[*it]++;
		}
		for (int i = 0; i < cluster_counter.size(); i++)
		{
			if (cluster_counter[i] > 20)
				real_num_components++;
		}

		if (parts_counter[idx] < real_num_components)
			parts_counter[idx] = real_num_components;
	}
}

void PartsSplitter::split_once(const std::string shape_path, const std::string labels_dir,
	const std::string new_labels_dir, std::vector<Eigen::Vector3f>& order_directions, int force_match)
{
	std::cout << "Split parts for " << shape_path << std::endl;

	/* Normalize the order directions */
	for (std::vector<Eigen::Vector3f>::iterator dir_it = order_directions.begin();
		dir_it != order_directions.end(); ++dir_it)
		dir_it->normalize();

	pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<int> original_labels, face_index_map;
	std::vector<Eigen::Vector3f> bary_coords;

	PointCloudLoader::load_pointcloud(shape_path, labels_dir, pointcloud, original_labels, face_index_map, bary_coords);

	std::vector<std::list<int>> original_parts(m_symmetry_groups.size());
	std::vector<int> original_label_indices(m_symmetry_groups.size());

	for (int i = 0; i < original_labels.size(); i++)
	{
		int original_label = original_labels[i];
		original_parts[original_label].push_back(i);
	}

	std::list<int> single_label, group_label;
	int l_idx = 0;
	for (std::list<std::vector<int>>::iterator group_it = m_symmetry_groups.begin();
		group_it != m_symmetry_groups.end(); ++group_it, ++l_idx)
	{
		if (group_it->size() == 1)
			single_label.push_back(l_idx);
		else
			group_label.push_back(l_idx);
	}

	std::vector<std::vector<std::list<int>>> point_clusters(m_symmetry_groups.size());
	
	/* For each original part, split the points into several point clusters if it has */
	int idx = 0;
	std::vector<std::list<int>>::iterator original_part_it;
	std::list<std::vector<int>>::iterator sym_it;
	for (original_part_it = original_parts.begin(), sym_it = m_symmetry_groups.begin();
		original_part_it != original_parts.end() && sym_it != m_symmetry_groups.end(); ++original_part_it, ++sym_it, ++idx)
	{
		if (sym_it->size() > 1)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr part_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			part_cloud->width = original_part_it->size();
			part_cloud->height = 1;
			part_cloud->is_dense = false;
			part_cloud->points.resize(part_cloud->width * part_cloud->height);

			std::vector<int> part_to_origin_map(original_part_it->size());

			int jdx = 0;
			for (std::list<int>::iterator point_it = original_part_it->begin(); point_it != original_part_it->end(); ++point_it, ++jdx)
			{
				int point_index = *point_it;
				part_cloud->points[jdx].x = pointcloud->points[point_index].x;
				part_cloud->points[jdx].y = pointcloud->points[point_index].y;
				part_cloud->points[jdx].z = pointcloud->points[point_index].z;
				part_to_origin_map[jdx] = point_index;
			}

			if (part_cloud->size() < 10)
			{
				if (force_match == 1 || force_match == 2)
				{
					if (m_wrong_list.empty() || shape_path.compare(m_wrong_list.back()) != 0)
						m_wrong_list.push_back(shape_path);
				}
				continue;
			}

			boost::shared_ptr<PointCloudGraph> graph(new PointCloudGraph());
			graph->set_input_pointcloud(part_cloud);

			std::vector<int> components_indices;
			int num_components = graph->connected_components(components_indices);

			if (force_match == 1)
			{
				if (num_components < sym_it->size())
				{
					if (m_wrong_list.empty() || shape_path.compare(m_wrong_list.back()) != 0)
						m_wrong_list.push_back(shape_path);
				}
			}
			else if (force_match == 2)
			{
				if (num_components != sym_it->size())
				{
					if (m_wrong_list.empty() || shape_path.compare(m_wrong_list.back()) != 0)
						m_wrong_list.push_back(shape_path);
				}
			}
			else if (force_match == 3)
			{
				if (num_components > sym_it->size())
				{
					if(m_wrong_list.empty() || shape_path.compare(m_wrong_list.back()) != 0)
						m_wrong_list.push_back(shape_path);
				}
			}

			if (num_components > 1)
				group_label.push_back(idx);
			else
				single_label.push_back(idx);

			point_clusters[idx].resize(num_components);

			int kdx = 0;
			for (std::vector<int>::iterator idx_it = components_indices.begin(); idx_it != components_indices.end(); ++idx_it, ++kdx)
			{
				int component_id = *idx_it;
				int original_idx = part_to_origin_map[kdx];
				point_clusters[idx][component_id].push_back(original_idx);
			}
		}
		else
		{
			point_clusters[idx].resize(1);

			for (std::list<int>::iterator p_it = original_part_it->begin();
				p_it != original_part_it->end(); ++p_it)
			{
				point_clusters[idx][0].push_back(*p_it);
			}
		}
	}

	std::vector<int> new_points_labels(pointcloud->size());

	int new_label_idx = 0;
	int part_idx = 0;
	std::vector<std::vector<std::list<int>>>::iterator part_it;
	for (part_it = point_clusters.begin(), sym_it = m_symmetry_groups.begin();
		part_it != point_clusters.end() && sym_it != m_symmetry_groups.end(); ++part_it, ++sym_it, ++part_idx)
	{
		/* If the current part only contains one point cluster */
		if (sym_it->size() == 1)
		{
			if (part_it->size() > 0)
			{
				std::list<int> &points_indices = part_it->operator[](0);
				for (std::list<int>::iterator i_it = points_indices.begin(); i_it != points_indices.end(); ++i_it)
				{
					int original_index = *i_it;
					new_points_labels[original_index] = new_label_idx;
				}
			}
			new_label_idx++;
		}
		/* If the current part conatins more than 1 point clusters */
		else
		{
			std::vector<std::list<int>> &clusters = *part_it;

			/* If the number of clusters is larger than expected, abandon the smallest ones */
			std::set<int> skip_cluster_indices;
			if (clusters.size() > sym_it->size())
			{
				int num_abandoned = sym_it->size() - clusters.size();
				std::vector<int> cluster_sizes(clusters.size());
				std::vector<int> cluster_indices(clusters.size());
				for (int i = 0; i < cluster_indices.size(); i++)
				{
					cluster_sizes[i] = clusters[i].size();
					cluster_indices[i] = i;
				}

				std::sort(cluster_indices.begin(), cluster_indices.end(),
					[&](const int a, const int &b) {
					return (cluster_sizes[a] < cluster_sizes[b]);
				});

				for (int i = 0; i < num_abandoned; i++)
					skip_cluster_indices.insert(cluster_indices[i]);
			}

			int real_num_clusters = clusters.size() - skip_cluster_indices.size();

			/* get the center of each point cluster */
			std::vector<int> center_cluster_map(real_num_clusters);
			std::vector<Eigen::Vector3f> centers(real_num_clusters, Eigen::Vector3f::Zero());
			int jdx = 0, cluster_index = 0;
			for (std::vector<std::list<int>>::iterator cluster_it = clusters.begin();
				cluster_it != clusters.end(); ++cluster_it, cluster_index++)
			{
				if (skip_cluster_indices.find(cluster_index) != skip_cluster_indices.end())
					continue;

				for (std::list<int>::iterator point_it = cluster_it->begin();
					point_it != cluster_it->end(); ++point_it)
				{
					int point_index = *point_it;
					Eigen::Vector3f point(pointcloud->points[point_index].x,
						pointcloud->points[point_index].y,
						pointcloud->points[point_index].z);
					centers[jdx] += point;
				}
				centers[jdx] /= (float)cluster_it->size();
				center_cluster_map[jdx] = cluster_index;
				jdx++;
			}

			/* If the point clusters are arranged approximately in a line */
			if (order_directions.size() == 1)
			{
				Eigen::Vector3f &direction = order_directions[0];

				/* Project centers to order direction */
				std::vector<float> projections(centers.size());
				std::vector<int> centers_indices(centers.size());
				for (int i = 0; i < centers.size(); i++)
				{
					float projection = direction.dot(centers[i]);
					projections[i] = projection;
					centers_indices[i] = i;
				}

				/* Sort the centers along the order direction */
				std::sort(centers_indices.begin(), centers_indices.end(),
					[&](const int a, const int &b) {
					return (projections[a] < projections[b]);
				});

				/* Update the labels of each clusters */
				int current_label = new_label_idx;
				int center_idx = 0;
				for (std::vector<int>::iterator center_it = centers_indices.begin();
					center_it != centers_indices.end(); ++center_it, ++center_idx)
				{
					int cluster_index = center_cluster_map[center_idx];

					std::list<int> &points_indices = clusters[cluster_index];
					for (std::list<int>::iterator p_it = points_indices.begin();
						p_it != points_indices.end(); ++p_it)
					{
						int original_label = *p_it;
						new_points_labels[original_label] = current_label;
					}
					current_label++;
				}

				new_label_idx += sym_it->size();
			}
			else
			{
				/* Compute the center of the centers of point clusters */
				Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
				for (std::vector<Eigen::Vector3f>::iterator c_it = centers.begin();
					c_it != centers.end(); ++c_it)
					centroid += *c_it;
				centroid /= (float)centers.size();

				/* Compute the angle between cluster center and centroid */
				std::vector<int> centers_indices(centers.size());
				std::vector<float> angles(centers.size());
				for (int i = 0; i < centers.size(); i++)
				{
					Eigen::Vector3f center_dir = centers[i] - centroid;
					Eigen::Vector3f &order_dir_1 = order_directions[0];
					Eigen::Vector3f &order_dir_2 = order_directions[1];

					float theta1 = std::acos(center_dir.dot(order_dir_1) / (center_dir.norm() * order_dir_1.norm()));
					if (center_dir.dot(order_dir_2) < 0)
						theta1 = 2 * PI - theta1;
					angles[i] = theta1;

					centers_indices[i] = i;
				}

				/* Sort the centers according to the angles */
				std::sort(centers_indices.begin(), centers_indices.end(),
					[&](const int a, const int &b) {
					return (angles[a] < angles[b]);
				});

				/* Update the labels of each clusters */
				int current_label = new_label_idx;
				//int center_idx = 0;
				for (std::vector<int>::iterator center_it = centers_indices.begin();
					center_it != centers_indices.end(); ++center_it)
				{
					int cluster_index = center_cluster_map[*center_it];

					std::list<int> &points_indices = clusters[cluster_index];
					for (std::list<int>::iterator p_it = points_indices.begin();
						p_it != points_indices.end(); ++p_it)
					{
						int original_index = *p_it;
						new_points_labels[original_index] = current_label;
					}
					current_label++;
				}

				new_label_idx += sym_it->size();
			}
		}
	}

	/* Output new labels */
	boost::filesystem::path boost_shape_path(shape_path);
	std::string file_name = boost::filesystem::basename(boost_shape_path);
	std::string new_labels_path = new_labels_dir + "\\" + file_name + ".seg";
	std::ofstream new_labels_out(new_labels_path.c_str());
	if (new_labels_out.is_open())
	{
		for (std::vector<int>::iterator label_it = new_points_labels.begin();
			label_it != new_points_labels.end(); ++label_it)
			new_labels_out << *label_it << std::endl;

		new_labels_out.close();
	}
	else
	{
		std::cerr << "Error: cannot write new labels file to " << new_labels_path << std::endl;
	}

	std::cout << "done." << std::endl;
}

bool PartsSplitter::load_label_info(std::string regions_file, bool _verbose)
{
	const char *_filename = regions_file.c_str();

	std::ifstream file(_filename);
	if (!file)
	{
		std::cerr << "Can't open file: \"" << _filename << "\"" << std::endl;
		return false;
	}

	if (_verbose)
		std::cout << "Loading " << _filename << "..." << std::endl;


	// NOTE:
	// All cuboids are also deleted.
	m_labels.clear();
	m_label_names.clear();


	std::string buffer;
	int new_label = 0;

	while (!file.eof())
	{
		std::getline(file, buffer);
		if (buffer == "") break;

		std::stringstream sstr(buffer);

		const unsigned int num_tokens = 3;
		std::string tokens[num_tokens];

		for (unsigned int i = 0; i < num_tokens; ++i)
		{
			if (sstr.eof())
			{
				std::cerr << "Error: Wrong file format: \"" << _filename << "\"" << std::endl;
				return false;
			}

			std::getline(sstr, tokens[i], ' ');
		}

		if (tokens[1] != "pnts" || tokens[2] != "1")
		{
			std::cerr << "Error: Wrong file format: \"" << _filename << "\"" << std::endl;
			return false;
		}

		// NOTE:
		// In this file format, labels are defined by the recorded order.
		m_labels.push_back(new_label);
		m_label_names.push_back(tokens[0]);
		++new_label;
	}

	m_null_label = new_label;

	file.close();
}

bool PartsSplitter::load_regions_symmetry(std::string regions_symmetry_file)
{
	std::ifstream in(regions_symmetry_file.c_str());
	if (in.is_open())
	{
		while (!in.eof())
		{
			std::string line;
			std::getline(in, line);

			if (line.length() > 0)
			{
				std::vector<std::string> line_list;
				boost::split(line_list, line, boost::is_any_of(" "), boost::token_compress_on);

				std::vector<int> sym_labels(line_list.size());
				for (int i = 0; i < line_list.size(); i++)
				{
					int label = get_label_from_label_name(line_list[i]);

					sym_labels[i] = label;
				}

				m_symmetry_groups.push_back(sym_labels);
			}
		}
	}
	else
		return false;

	return true;
}

int PartsSplitter::get_label_from_label_name(std::string label_name)
{
	boost::trim(label_name);
	for (int i = 0; i < m_label_names.size(); i++)
	{
		if (label_name.compare(m_label_names[i]) == 0)
			return i;
	}

	return m_labels.size();
}
