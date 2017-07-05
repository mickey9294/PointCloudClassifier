#include "FeaturesEstimator.h"

boost::mutex cout_mutex;

FeaturesEstimator::FeaturesEstimator()
	: pid(0)
{
}

FeaturesEstimator::FeaturesEstimator(int _pid)
	: pid(_pid)
{
}

FeaturesEstimator::~FeaturesEstimator()
{
}

void FeaturesEstimator::set_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud)
{
	m_pointcloud = pointcloud;
	m_graph.set_input_pointcloud(m_pointcloud);
}

void FeaturesEstimator::compute_features(Eigen::MatrixXd & features)
{
	cout_mutex.lock();
	std::cout << "Estimator_" << pid << ": Compute features." << std::endl;
	cout_mutex.unlock();

	std::vector<boost::promise<Eigen::MatrixXd>> promises(6);
	std::vector<boost::function0<void>> functions(6);
	std::vector<boost::thread> threads(6);
	for (int i =  0; i < 6; i++)
	{
		functions[i] = boost::bind(&FeaturesEstimator::compute_part_of_features, this, i + 1, std::ref(promises[i]));
		threads[i] = boost::thread(functions[i]);
	}
	for (int i = 0; i < 6; i++)
		threads[i].join();

	//Eigen::MatrixXd features_mat = Eigen::MatrixXd::Zero(m_pointcloud->size(), DIMEN);
	features.setZero(m_pointcloud->size(), DIMEN);
	for (int i = 0; i < 6; i++)
	{
		boost::unique_future<Eigen::MatrixXd> f = promises[i].get_future();
		features += f.get();
	}

	cout_mutex.lock();
	std::cout << "Estimator_" << pid << ": all done." << std::endl;
	cout_mutex.unlock();
}

void FeaturesEstimator::compute_part_of_features(int id, boost::promise<Eigen::MatrixXd> & promise)
{
	cout_mutex.lock();
	std::cout << "subthread_" << pid << "_" << id << " start." << std::endl;
	cout_mutex.unlock();

	Eigen::MatrixXd features;

	/* Copy the piont cloud to thread storage space */
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	{
		boost::shared_lock<boost::shared_mutex> lock{ cloud_mutex };
		cloud->width = m_pointcloud->width;
		cloud->height = 1;
		cloud->is_dense = false;
		cloud->points.resize(cloud->height * cloud->width);
		for (int i = 0; i < cloud->size(); i++)
		{
			cloud->points[i].x = m_pointcloud->points[i].x;
			cloud->points[i].y = m_pointcloud->points[i].y;
			cloud->points[i].z = m_pointcloud->points[i].z;
		}
		lock.unlock();
	}

	features = Eigen::MatrixXd::Zero(cloud->size(), DIMEN);

	/* if the thread is used to compute neighborhood features */
	if (id < 6)  
	{
		std::vector<double> abs_curvatures;
		if (id == 3)
		{
			/* Compute the normals of the piont cloud */
			pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
			ne.setInputCloud(cloud);
			pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>());
			ne.setSearchMethod(kdtree);

			pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

			ne.setRadiusSearch(0.05);
			ne.compute(*cloud_normals);

			/* Compute curvatures */
			abs_curvatures.resize(cloud->size());
			for (int i = 0; i < cloud->size(); i++)
			{
				std::vector<int> neighbors_indices;
				std::vector<double> neighbors_distances;
				boost::shared_lock<boost::shared_mutex> graph_lock{ graph_mutex };
				m_graph.radius_search(i, 0.05, neighbors_indices, neighbors_distances);
				graph_lock.unlock();

				Eigen::Vector3d sum_offsets;
				sum_offsets.setZero();
				int num_offsets = 0;

				Eigen::Vector3d normal;
				normal[0] = cloud_normals->points[i].normal_x;
				normal[1] = cloud_normals->points[i].normal_y;
				normal[2] = cloud_normals->points[i].normal_z;
				Eigen::Vector3d position;
				position[0] = cloud->points[i].x;
				position[1] = cloud->points[i].y;
				position[2] = cloud->points[i].z;

				for (int j = 0; j < neighbors_indices.size(); j++)
				{
					int neighbor_idx = neighbors_indices[j];

					Eigen::Vector3d pntNormal;
					pntNormal[0] = cloud_normals->points[neighbor_idx].normal_x;
					pntNormal[1] = cloud_normals->points[neighbor_idx].normal_y;
					pntNormal[2] = cloud_normals->points[neighbor_idx].normal_z;

					Eigen::Vector3d pntPos;
					pntPos[0] = cloud->points[neighbor_idx].x;
					pntPos[1] = cloud->points[neighbor_idx].y;
					pntPos[2] = cloud->points[neighbor_idx].z;

					Eigen::Vector3d offset = pntPos - position;
					offset.normalize();
					sum_offsets += offset;
					num_offsets++;
				}
				double curv = num_offsets > 0 ? (std::abs(sum_offsets.dot(normal)) / num_offsets) : 0;
				if (curv != curv)
					curv = 0;
				abs_curvatures[i] = curv;
			}
		}

		/* Find the neighbors of each point */
		for (int i = 0; i < cloud->size(); i++)
		{
			Eigen::VectorXd feature(DIMEN);
			feature.setZero();

			std::vector<int> neighborhood;
			std::vector<double> neighborhood_distances;
			boost::shared_lock<boost::shared_mutex> graph_lock{ graph_mutex };
			m_graph.radius_search(i, id * 0.1, neighborhood, neighborhood_distances);
			graph_lock.unlock();

			if (!neighborhood.empty())
			{
				/* For ZVariance computation */
				boost::accumulators::accumulator_set<double, boost::accumulators::features<boost::accumulators::tag::variance>> acc;

				/* Create neighborhood point cloud */
				pcl::PointCloud<pcl::PointXYZ>::Ptr neighborhood_cloud(new pcl::PointCloud<pcl::PointXYZ>);
				neighborhood_cloud->width = neighborhood.size();
				neighborhood_cloud->height = 1;
				neighborhood_cloud->is_dense = false;
				neighborhood_cloud->points.resize(neighborhood_cloud->width * neighborhood_cloud->height);
				for (int j = 0; j < neighborhood.size(); j++)
				{
					int neighbor_idx = neighborhood[j];
					neighborhood_cloud->points[j].x = cloud->points[neighbor_idx].x;
					neighborhood_cloud->points[j].y = cloud->points[neighbor_idx].y;
					neighborhood_cloud->points[j].z = cloud->points[neighbor_idx].z;
					acc(cloud->points[neighbor_idx].z);  /* for ZVariance computation */
				}

				/* Compute covariance matrix */
				Eigen::Matrix3d cov_matrix;
				pcl::computeCovarianceMatrix(*neighborhood_cloud, cov_matrix);

				/* Compute the eigen values and eigen vectors of the covariance matrix */
				Eigen::EigenSolver<Eigen::Matrix3d> solver(cov_matrix, true);
				Eigen::VectorXcd evalues = solver.eigenvalues().transpose();
				Eigen::MatrixXcd eigenvectors = solver.eigenvectors();
				Eigen::Vector3cd evector0 = eigenvectors.col(0);
				Eigen::Vector3cd evector1 = eigenvectors.col(1);
				Eigen::Vector3cd evector2 = eigenvectors.col(2);

				/* Specify the order of eigen values */
				int firstno, secno, thirdno;
				double max = -255.0;
				for (int j = 0; j < 3; j++)
				{
					if (evalues[j].real() > max)
					{
						firstno = j;
						max = evalues[j].real();
					}
				}
				max = -255.0;
				for (int j = 0; j < 3; j++)
				{
					if (j != firstno)
					{
						if (evalues[j].real() > max)
						{
							secno = j;
							max = evalues[j].real();
						}
					}
				}
				thirdno = 3 - firstno - secno;

				/* Calculate part of the features */
				double evqu0 = 0, evqu1 = 0, grav0 = 0, grav1 = 0;
				evqu0 = evalues[secno].real() / evalues[firstno].real();
				evqu1 = evalues[thirdno].real() / evalues[firstno].real();
				Eigen::Vector3cd g(0, -1.0, 0);
				Eigen::MatrixXcd gm0 = eigenvectors.col(firstno).transpose() * g;
				grav0 = gm0.data()[0].real();
				Eigen::MatrixXcd gm2 = eigenvectors.col(thirdno).transpose() * g;
				grav1 = gm2.data()[0].real();

				feature[(id - 1) * 4] = evqu0;
				feature[(id - 1) * 4 + 1] = evqu1;
				feature[(id - 1) * 4 + 2] = grav0;
				feature[(id - 1) * 4 + 3] = grav1;

				/* Compute ZVariance features */
				double zvariance = boost::accumulators::variance(acc);
				feature[20 + (id - 1)] = zvariance;
			}
			else  /* if there is no point in the neighborhood of current point */
			{
				feature[(id - 1) * 4] = 0;
				feature[(id - 1) * 4 + 1] = 0;
				feature[(id - 1) * 4 + 2] = 0;
				feature[(id - 1) * 4 + 3] = 0;
				feature[20 + (id - 1)] = 0;
			}

			if (id == 3)
			{
				/* Set absolute curvature features*/
				assert(abs_curvatures.size() == cloud->size());
				feature[26] = abs_curvatures[i];

				/* Compute average curvatures features */
				double avg_curv = abs_curvatures[i];
				for (int j = 0; j < neighborhood.size(); j++)
				{
					int n_idx = neighborhood[j];
					avg_curv += abs_curvatures[n_idx];
				}
				avg_curv /= (double)(neighborhood.size() + 1);

				/* Set average curvature*/
				feature[27] = avg_curv;
			}

			/* add the feature of the current point  to features matrix */
			features.row(i) = feature;
		}
	}
	/* If it is a thread computing height features and DistHist features */
	else if (id == 6)
	{
		/* Compute lowest point height */
		double ground = cloud->points[0].z;
		for (int i = 0; i < cloud->size(); i++)
			ground = std::min(ground, (double)cloud->points[i].z);

		for (int i = 0; i < cloud->size(); i++)
		{
			Eigen::VectorXd feature = Eigen::VectorXd::Zero(DIMEN);

			/* Compute distance histograms features */
			std::vector<double> distances;
			m_graph.shortest_paths(i, distances);
			int counter = 0;

			for (int j = 0; j < distances.size(); j++)
			{
				if (j != i)
				{
					double dist = distances[j];
					if (dist <= MAX_DISTANCE)
					{
						int bin = (int)(num_bins * dist / MAX_DISTANCE + 0.5);
						if (bin >= 0 && bin < num_bins)
						{
							feature[28 + bin] += 1.0;
							counter++;
						}
					}
				}
			}
			/* normalize histograms */
			if(counter > 0)
				feature /= (double)counter;

			/* Compute height features */
			feature[25] = cloud->points[i].z - ground;

			/* add the feature of the current point  to features matrix */
			features.row(i) = feature;
		}
	}

	promise.set_value(features);

	cout_mutex.lock();
	std::cout << "subthread_" << pid << "_" << id << " finished." << std::endl;
	cout_mutex.unlock();
}
