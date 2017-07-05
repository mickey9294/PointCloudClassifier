#include "FeaturesExtractor.h"

extern boost::mutex cout_mutex;

FeaturesExtractor::FeaturesExtractor()
{
}

FeaturesExtractor::FeaturesExtractor(const std::string shape_category)
{
	m_shape_category = shape_category;
}

FeaturesExtractor::~FeaturesExtractor()
{
}

void FeaturesExtractor::set_shape_category(const std::string shape_category)
{
	m_shape_category = shape_category;
}

void FeaturesExtractor::load_shapes_list(const std::string shapes_list_path)
{
	std::ifstream in(shapes_list_path.c_str());
	if (in.is_open())
	{
		while (!in.eof())
		{
			std::string line;
			std::getline(in, line);
			if(line.length() > 0)
				m_shapes_list.push_back(line);
		}

		in.close();
	}
}

void FeaturesExtractor::extract_features(const std::string shapes_list_path, const std::string output_dir, const std::string labels_dir)
{
	cout_mutex.lock();
	std::cout << "Extract features of shapes set." << std::endl;
	cout_mutex.unlock();

	/* Load shapes paths from list file */
	boost::shared_ptr<FeaturesEstimator> estimator(new FeaturesEstimator());
	load_shapes_list(shapes_list_path);

	/* Check whether the output directory exists */
	boost::filesystem::path boost_output_dir(output_dir);
	if (!boost::filesystem::exists(boost_output_dir))
	{
		std::cerr << "Error: The output directory does not exist!" << std::endl;
		return;
	}
	else
		m_output_dir = output_dir;

	m_off_dir = "D:\\Projects\\shape2pose\\data\\1_input\\" + m_shape_category + "\\off";
	boost::filesystem::path boost_off_dir(m_off_dir);
	if (!boost::filesystem::exists(boost_off_dir))
	{
		boost::filesystem::create_directories(boost_off_dir);
	}

	/* Compute the number of shapes that per extractor thread should process */
	int num_shapes_per_thread = m_shapes_list.size() / num_threads;
	if (num_shapes_per_thread == 0)
	{
		num_shapes_per_thread = 1;
		num_threads = m_shapes_list.size();
	}

	/* Extractor threads definition */
	std::vector<boost::function0<void>> functions(num_threads);
	std::vector<boost::thread> threads(num_threads);
	std::list<std::string>::iterator shape_it = m_shapes_list.begin();
	for (int i = 0; i < num_threads; i++)
	{
		std::list<std::string> sub_list;
		for (int j = 0; j < num_shapes_per_thread && shape_it != m_shapes_list.end(); ++j, ++shape_it)
			sub_list.push_back(*shape_it);
		if (i == num_threads - 1)
		{
			for (; shape_it != m_shapes_list.end(); ++shape_it)
				sub_list.push_back(*shape_it);
		}
		
		functions[i] = boost::bind(&FeaturesExtractor::extract_features_once, this, i + 1, sub_list, labels_dir);
		threads[i] = boost::thread(functions[i]);
	}
	/* Start the extractor threads and wait */
	for (int i = 0; i < num_threads; i++)
		threads[i].join();

	cout_mutex.lock();
	std::cout << "Features extraction all done." << std::endl;
	cout_mutex.unlock();

	/* Generate features list file */
	std::string features_list_file = output_dir + "\\" + m_shape_category + "_feats.txt";
	std::ofstream feats_list_out(features_list_file.c_str());
	if (feats_list_out.is_open())
	{
		for (std::list<std::string>::iterator file_it = m_shapes_list.begin(); file_it != m_shapes_list.end(); ++file_it)
		{
			boost::filesystem::path shape_path(*file_it);
			std::string shape_name = shape_path.stem().string();

			std::string feat_path = output_dir + "\\" + shape_name + ".csv";
			feats_list_out << feat_path << std::endl;
		}

		feats_list_out.close();
	}
	else
	{
		std::cerr << "Error: cannot save features list file to " << features_list_file << std::endl;
	}
}

void FeaturesExtractor::extract_features_once(int id, const std::list<std::string> &sub_list, const std::string labels_dir)
{
	cout_mutex.lock();
	std::cout << "extractor_thread_" << id << " starts." << std::endl;
	cout_mutex.unlock();

	boost::shared_lock<boost::shared_mutex> name_lock{ name_mutex };
	std::string features_dir = "../data/features/" + m_shape_category;
	name_lock.unlock();

	/* Initialize FeaturesEstimator */
	boost::shared_ptr<FeaturesEstimator> estimator(new FeaturesEstimator(id));
	
	/* Define output format of .csv file */
	const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

	/* Process shapes in sub_list one by one */
	for (std::list<std::string>::const_iterator it = sub_list.begin(); it != sub_list.end(); ++it)
	{
		/* Get the shape name */
		boost::filesystem::path shape_path(*it);
		std::string shape_name = shape_path.stem().string();

		cout_mutex.lock();
		std::cout << "extractor_thread_" << id << ": extract features of " << m_shape_category << "_" << shape_name << std::endl;
		cout_mutex.unlock();

		/* Load shape from file to point cloud */
		pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
		std::vector<int> points_labels;
		std::vector<int> face_index_map;
		std::vector<Eigen::Vector3f> bary_coords;
		bool is_mesh = PointCloudLoader::load_pointcloud(*it, labels_dir, pointcloud, points_labels, face_index_map, bary_coords);

		/* Save the sample points in ply file */
		//pcl::PLYWriter ply_writer;
		//std::string ply_file = m_output_dir + "\\" + shape_name + ".ply";
		//ply_writer.write(ply_file, *pointcloud);
		/* Save the sample Points in off file */
		
		std::string off_file = m_off_dir + "\\" + shape_name + ".off";
		save_pointcloud(off_file, pointcloud);
		if (is_mesh)
		{
			std::string pts_file = m_output_dir + "\\" + shape_name + ".pts";
			save_pointcloud(pts_file, pointcloud, face_index_map, bary_coords);
		}

		/* Set the pointcloud loaded into features estimator */
		estimator->set_pointcloud(pointcloud);

		/* Compute features */
		Eigen::MatrixXd features_mat;
		estimator->compute_features(features_mat);

		/* Generate output path of features file and labels file */
		boost::shared_lock<boost::shared_mutex> dir_lock{ dir_mutex };
		std::string features_out_path = m_output_dir + "\\" + shape_name + ".csv";
		std::string labels_out_path = m_output_dir + "\\" + shape_name + ".seg";
		dir_lock.unlock();

		/* Output features into file */
		std::ofstream features_out(features_out_path.c_str());
		if (features_out.is_open())
		{
			features_out << features_mat.format(CSVFormat);
			features_out.close();
		}
		/* Output labels */
		if(is_mesh)
			save_labels(labels_out_path, points_labels);

		cout_mutex.lock();
		std::cout << "extractor_thread_" << id << ": extraction of " << m_shape_category << "_" << shape_name << " done." << std::endl;
		cout_mutex.unlock();
	}

	cout_mutex.lock();
	std::cout << "extractor_thread_" << id << " finished." << std::endl;
	cout_mutex.unlock();
}

void FeaturesExtractor::save_labels(const std::string save_path, const std::vector<int> &labels)
{
	std::ofstream out(save_path.c_str());

	if (out.is_open())
	{
		for (std::vector<int>::const_iterator it = labels.begin(); it != labels.end(); ++it)
			out << *it << std::endl;

		out.close();
	}
}

void FeaturesExtractor::save_pointcloud(const std::string save_path, pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud)
{
	std::ofstream out(save_path.c_str());

	boost::filesystem::path boost_file_path(save_path.c_str());
	std::string format_extension = boost_file_path.extension().string();

	if (out.is_open())
	{
		if (format_extension.compare(".off") == 0)
		{
			out << "OFF" << std::endl;

			out << pointcloud->size() << " 0 0" << std::endl;
			for (int i = 0; i < pointcloud->size(); i++)
			{
				out << pointcloud->points[i].x << " "
					<< pointcloud->points[i].y << " "
					<< pointcloud->points[i].z << std::endl;
			}
		}
		else if (format_extension.compare(".pts") == 0)
		{
			for (int i = 0; i < pointcloud->size(); i++)
			{
				out << "0 0.333 0.333 0.333 "
					<< pointcloud->points[i].x << " "
					<< pointcloud->points[i].y << " "
					<< pointcloud->points[i].z << std::endl;
			}
		}

		out.close();
	}
	else
		std::cerr << "Error: Cannot save point cloud into " << save_path << "." << std::endl;
}

void FeaturesExtractor::save_pointcloud(const std::string save_path, pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, 
	const std::vector<int>& face_index_map, const std::vector<Eigen::Vector3f>& bary_coords)
{
	std::ofstream out(save_path.c_str());

	boost::filesystem::path boost_file_path(save_path.c_str());
	std::string format_extension = boost_file_path.extension().string();

	if (out.is_open())
	{
		if (format_extension.compare(".pts") == 0)
		{
			for (int i = 0; i < pointcloud->size(); i++)
			{
				int face_index = face_index_map[i];
				Eigen::Vector3f bary_coord = bary_coords[i];

				out << face_index << " " 
					<< bary_coord[0] << " " << bary_coord[1] << " " << bary_coord[2] << " "
					<< pointcloud->points[i].x << " "
					<< pointcloud->points[i].y << " "
					<< pointcloud->points[i].z << std::endl;
			}
		}

		out.close();
	}
	else
		std::cerr << "Error: Cannot save point cloud into " << save_path << "." << std::endl;
}
