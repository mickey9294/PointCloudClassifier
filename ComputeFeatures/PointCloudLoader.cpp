#include "PointCloudLoader.h"

PointCloudLoader::PointCloudLoader()
{
}


PointCloudLoader::~PointCloudLoader()
{
}

bool PointCloudLoader::load_pointcloud(const std::string & file_path, pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud)
{
	bool is_mesh = false;
	boost::filesystem::path boost_file_path(file_path.c_str());
	std::string format_extension = boost_file_path.extension().string();

	if (format_extension.compare(".pts") == 0)
	{
		std::list<Eigen::Vector3f> points_list;

		Eigen::Vector3d min_coords(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
		Eigen::Vector3d max_coords(std::numeric_limits<double>::min(), std::numeric_limits<double>::min(), std::numeric_limits<double>::min());
		Eigen::Vector3d center(0, 0, 0);

		std::ifstream in(file_path.c_str());
		if (in.is_open())
		{
			std::string line;
			while (!in.eof())
			{
				std::getline(in, line);
				if (line.length() > 0)
				{
					std::vector<std::string> line_split;
					boost::split(line_split, line, boost::is_any_of(" "), boost::token_compress_on);
					float x = std::stof(line_split[0]);
					float y = std::stof(line_split[1]);
					float z = std::stof(line_split[2]);
					points_list.push_back(Eigen::Vector3f(x, y, z));

					if (x > max_coords[0])
						max_coords[0] = x;
					if (x < min_coords[0])
						min_coords[0] = x;
					if (y > max_coords[1])
						max_coords[1] = y;
					if (y < min_coords[1])
						min_coords[1] = y;
					if (z > max_coords[2])
						max_coords[2] = z;
					if (z < min_coords[2])
						min_coords[2] = z;
				}
			}

			double diameter = std::sqrt(std::pow(max_coords[0] - min_coords[0], 2)
				+ std::pow(max_coords[1] - min_coords[1], 2)
				+ std::pow(max_coords[2] - min_coords[2], 2));
			double scale = 1.0 / diameter;
			center = (max_coords + min_coords) / 2.0;
			center[2] -= 0.5 * (max_coords[2] - min_coords[2]);

			pointcloud->width = points_list.size();
			pointcloud->height = 1;
			pointcloud->is_dense = false;
			pointcloud->points.resize(pointcloud->width * pointcloud->height);
			int i = 0;
			if (std::abs(diameter - 1.0) < 1e-5 && center.norm() < 1e-5)
			{
				for (std::list<Eigen::Vector3f>::iterator it = points_list.begin(); it != points_list.end(); ++it, ++i)
				{
					pointcloud->points[i].x = it->x();
					pointcloud->points[i].y = it->y();
					pointcloud->points[i].z = it->z();
				}
			}
			else
			{
				for (std::list<Eigen::Vector3f>::iterator it = points_list.begin(); it != points_list.end(); ++it, ++i)
				{
					pointcloud->points[i].x = (it->x() - center[0]) * scale;
					pointcloud->points[i].y = (it->y() - center[1]) * scale;
					pointcloud->points[i].z = (it->z() - center[2]) * scale;
				}
			}

			in.close();

			is_mesh = false;
		}
		else
			std::cerr << "Cannot open point cloud file " << file_path << "." << std::endl;
	}
	else
	{
		MyMesh mesh;
		mesh.open_mesh(file_path.c_str(), false);

		if (mesh.n_faces() > 0)  /* If it is a training mesh model */
		{
			/* Sample points on the surface of the mesh */
			std::vector<int> face_index_map;
			std::vector<Eigen::Vector3f> bary_coords;
			const int num_samples = 3000;
			sample_on_mesh(mesh, num_samples, pointcloud, face_index_map, bary_coords);

			is_mesh = true;
		}
		else
		{
			pointcloud->width = mesh.n_vertices();
			pointcloud->height = 1;
			pointcloud->is_dense = false;
			pointcloud->points.resize(pointcloud->width * pointcloud->height);
			for (unsigned int i = 0; i < mesh.n_vertices(); i++)
			{
				MyMesh::VertexHandle vertex_handle(i);
				MyMesh::Point p = mesh.point(vertex_handle);

				pointcloud->points[i].x = p[0];
				pointcloud->points[i].y = p[1];
				pointcloud->points[i].z = p[2];
			}
			
			is_mesh = false;
		}
	}
	return is_mesh;
}

bool PointCloudLoader::load_pointcloud(const std::string & file_path, const std::string labels_dir, pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, 
	std::vector<int>& points_labels, std::vector<int> &face_index_map, std::vector<Eigen::Vector3f> &bary_coords)
{
	bool is_mesh = false;

	boost::filesystem::path boost_file_path(file_path.c_str());
	std::string format_extension = boost_file_path.extension().string();

	if (format_extension.compare(".pts") == 0)
	{
		std::list<Eigen::Vector3f> points_list;

		Eigen::Vector3d min_coords(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
		Eigen::Vector3d max_coords(std::numeric_limits<double>::min(), std::numeric_limits<double>::min(), std::numeric_limits<double>::min());
		Eigen::Vector3d center(0, 0, 0);

		std::ifstream in(file_path.c_str());
		if (in.is_open())
		{
			std::string line;
			while (!in.eof())
			{
				std::getline(in, line);
				if (line.length() > 0)
				{
					std::vector<std::string> line_split;
					boost::split(line_split, line, boost::is_any_of(" "), boost::token_compress_on);
					float x, y, z;
					if (line_split.size() < 4)
					{
						x = std::stof(line_split[0]);
						y = std::stof(line_split[1]);
						z = std::stof(line_split[2]);
					}
					else
					{
						x = std::stof(line_split[4]);
						y = std::stof(line_split[5]);
						z = std::stof(line_split[6]);
					}
					points_list.push_back(Eigen::Vector3f(x, y, z));

					if (x > max_coords[0])
						max_coords[0] = x;
					if (x < min_coords[0])
						min_coords[0] = x;
					if (y > max_coords[1])
						max_coords[1] = y;
					if (y < min_coords[1])
						min_coords[1] = y;
					if (z > max_coords[2])
						max_coords[2] = z;
					if (z < min_coords[2])
						min_coords[2] = z;
				}
			}

			double diameter = std::sqrt(std::pow(max_coords[0] - min_coords[0], 2)
				+ std::pow(max_coords[1] - min_coords[1], 2)
				+ std::pow(max_coords[2] - min_coords[2], 2));
			double scale = 1.0 / diameter;
			center = (max_coords + min_coords) / 2.0;
			center[2] -= 0.5 * (max_coords[2] - min_coords[2]);

			pointcloud->width = points_list.size();
			pointcloud->height = 1;
			pointcloud->is_dense = false;
			pointcloud->points.resize(pointcloud->width * pointcloud->height);
			points_labels.resize(pointcloud->size());
			int i = 0;
			if (std::abs(diameter - 1.0) < 1e-5 && center.norm() < 1e-5)
			{
				for (std::list<Eigen::Vector3f>::iterator it = points_list.begin(); it != points_list.end(); ++it, ++i)
				{
					pointcloud->points[i].x = it->x();
					pointcloud->points[i].y = it->y();
					pointcloud->points[i].z = it->z();
				}
			}
			else
			{
				for (std::list<Eigen::Vector3f>::iterator it = points_list.begin(); it != points_list.end(); ++it, ++i)
				{
					pointcloud->points[i].x = (it->x() - center[0]) * scale;
					pointcloud->points[i].y = (it->y() - center[1]) * scale;
					pointcloud->points[i].z = (it->z() - center[2]) * scale;
				}
			}

			/* Load the labels */
			std::string label_directory;
			if (labels_dir.empty())
			{
				std::vector<std::string> path_split;
				char separator = (file_path.find('\\') == std::string::npos) ? '/' : '\\';
				std::string separator_str(1, separator);
				boost::split(path_split, file_path, boost::is_any_of(separator_str), boost::token_compress_on);
				path_split.pop_back();
				path_split.pop_back();
				label_directory = boost::algorithm::join(path_split, separator_str);
				label_directory += "expert_verified" + separator_str + "points_label" + separator_str;
			}
			else
				label_directory = labels_dir;
			std::string shape_name = boost_file_path.stem().string();
			std::string seg_file_name = shape_name + ".seg";
			std::string seg_file_path = label_directory + "\\" + seg_file_name;
			std::ifstream seg_in(seg_file_path.c_str());
			if (seg_in.is_open())
			{
				for (int i = 0; i < pointcloud->size(); i++)
				{
					std::getline(seg_in, line);
					if (line.length() > 0)
					{
						int label = std::stoi(line);
						points_labels[i] = label;
					}
				}

				seg_in.close();
			}

			/* Generate default point_label_map and bary_coords */
			face_index_map.resize(pointcloud->size(), 0);
			Eigen::Vector3f default_bary(0.333, 0.333, 0.333);
			bary_coords.resize(pointcloud->size(), default_bary);

			in.close();

			is_mesh = false;
		}
		else
			std::cerr << "Cannot open point cloud file " << file_path << "." << std::endl;
	}
	else
	{
		MyMesh mesh;
		mesh.open_mesh(file_path.c_str());

		std::vector<std::string> path_list;
		boost::split(path_list, file_path, boost::is_any_of("\\"));
		path_list[path_list.size() - 2] = "off_normalized";
		std::string normalized_path = boost::algorithm::join(path_list, "\\");
		mesh.save_mesh_to_off(normalized_path.c_str());

		if (mesh.n_faces() > 0)  /* If it is a training mesh model */
		{
			is_mesh = true;
			/* Sample points on the surface of the mesh */
			const int num_samples = 3000;
			sample_on_mesh(mesh, num_samples, pointcloud, face_index_map, bary_coords);

			/* Load the labels */
			std::string shape_name = boost_file_path.stem().string();
			std::string seg_file_name = shape_name + ".seg";
			std::string seg_file_path;
			if (labels_dir.empty())
			{
				std::vector<std::string> path_split;
				char separator = (file_path.find('\\') == std::string::npos) ? '/' : '\\';
				std::string separator_str(1, separator);
				boost::split(path_split, file_path, boost::is_any_of(separator_str), boost::token_compress_on);
				path_split[path_split.size() - 2] = "gt";
				path_split[path_split.size() - 1] = seg_file_name;
				seg_file_path = boost::algorithm::join(path_split, separator_str);
			}
			else
				seg_file_path = labels_dir + "\\" + seg_file_name;

			points_labels.resize(pointcloud->size());
			std::vector<int> faces_labels(mesh.n_faces());
			std::ifstream seg_in(seg_file_path.c_str());
			if (seg_in.is_open())
			{
				for (int i = 0; i < mesh.n_faces(); i++)
				{
					std::string line;
					std::getline(seg_in, line);
					if(line.length() > 0)
						faces_labels[i] = stoi(line);
				}

				seg_in.close();

				int idx = 0;
				for (std::vector<int>::iterator it = face_index_map.begin(); it != face_index_map.end(); ++it, ++idx)
				{
					int face_index = *it;
					points_labels[idx] = faces_labels[face_index];
				}
			}
		}
		else
		{
			is_mesh = false;

			pointcloud->width = mesh.n_vertices();
			pointcloud->height = 1;
			pointcloud->is_dense = false;
			pointcloud->points.resize(pointcloud->width * pointcloud->height);
			for (unsigned int i = 0; i < mesh.n_vertices(); i++)
			{
				MyMesh::VertexHandle vertex_handle(i);
				MyMesh::Point p = mesh.point(vertex_handle);

				pointcloud->points[i].x = p[0];
				pointcloud->points[i].y = p[1];
				pointcloud->points[i].z = p[2];
			}

			/* Load the labels */
			std::string label_directory;
			if (labels_dir.empty())
			{
				std::vector<std::string> path_split;
				char separator = (file_path.find('\\') == std::string::npos) ? '/' : '\\';
				std::string separator_str(1, separator);
				boost::split(path_split, file_path, boost::is_any_of(separator_str), boost::token_compress_on);
				path_split.pop_back();
				path_split.pop_back();
				label_directory = boost::algorithm::join(path_split, separator_str);
				label_directory += "expert_verified" + separator_str + "points_label" + separator_str;
			}
			else
				label_directory = labels_dir;
			std::string shape_name = boost_file_path.stem().string();
			std::string seg_file_name = shape_name + ".seg";
			std::string seg_file_path = label_directory + "\\" + seg_file_name;
			std::ifstream seg_in(seg_file_path.c_str());
			if (seg_in.is_open())
			{
				std::string line;
				for (int i = 0; i < pointcloud->size(); i++)
				{
					std::getline(seg_in, line);
					int label = std::stoi(line);
					points_labels[i] = label;
				}

				seg_in.close();
			}

			/* Generate default point_label_map and bary_coords */
			face_index_map.resize(pointcloud->size(), 0);
			Eigen::Vector3f default_bary(0.333, 0.333, 0.333);
			bary_coords.resize(pointcloud->size(), default_bary);
		}
	}

	return is_mesh;
}

void PointCloudLoader::sample_on_mesh(const MyMesh & mesh, const int num_samples, pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, 
	std::vector<int>& face_index_map, std::vector<Eigen::Vector3f> &bary_coords)
{
	std::vector<FacetProb> faceProbs;
	setFaceProb(mesh, faceProbs);

	srand(time(NULL));

	pointcloud->width = num_samples;
	pointcloud->height = 1;
	pointcloud->is_dense = false;
	pointcloud->points.resize(pointcloud->width * pointcloud->height);
	face_index_map.resize(num_samples);
	bary_coords.resize(num_samples);

	for (int i = 0; i < num_samples; i++)
	{
		Eigen::Vector3f v;
		Eigen::Vector3f bary_coord;
		int face_index;
		getRandomPoint(mesh, faceProbs, v, bary_coord, face_index);

		pointcloud->points[i].x = v[0];
		pointcloud->points[i].y = v[1];
		pointcloud->points[i].z = v[2];
		face_index_map[i] = face_index;
		bary_coords[i] = bary_coord;
	}
}

void PointCloudLoader::setFaceProb(const MyMesh & mesh, std::vector<FacetProb>& faceProbs)
{
	//calculate the areas of triangle meshes
	faceProbs.clear();
	int i = 0;
	for (MyMesh::FaceIter fit = mesh.faces_begin(); fit != mesh.faces_end(); ++fit, i++)
	{
		FacetProb facetP;
		facetP.index = fit->idx(); // t the index of the triangle
		MyMesh::FaceHandle fh = *fit;
		MyMesh::ConstFaceVertexIter cfvit = mesh.cfv_iter(fh);
		MyMesh::Point point1 = mesh.point(*cfvit);
		MyMesh::Point point2 = mesh.point(*(++cfvit));
		MyMesh::Point point3 = mesh.point(*(++cfvit));
		Eigen::Vector3f vA(point1[0], point1[1], point1[2]);
		Eigen::Vector3f vB(point2[0], point2[1], point2[2]);
		Eigen::Vector3f vC(point3[0], point3[1], point3[2]);
		facetP.area = TriangleArea(vA, vB, vC);
		//insert area into areaArray and ensure the ascending order
		int k;
		for (k = 0; k<faceProbs.size(); k++)
		{
			if (facetP.area<faceProbs[k].area) break;
		}
		faceProbs.insert(faceProbs.begin() + k, facetP);
	}
	//area normalization
	double total_area = 0;
	for (int i = 0; i < faceProbs.size(); i++)
	{
		total_area += faceProbs[i].area;
	}
	if (total_area != 0)
	{
		for (int i = 0; i< faceProbs.size(); i++)
		{
			faceProbs[i].area /= total_area;
		}
	}
}

bool PointCloudLoader::getRandomPoint(const MyMesh & mesh, const std::vector<FacetProb>& faceProbs, 
	Eigen::Vector3f & v, Eigen::Vector3f & bary_coord, int & face_index)
{
	double d = (double)(rand() / (double)RAND_MAX);
	int i = 0;	/* "i" indicates the triangle in the ordered area list */
	while (d >= 0 && i < faceProbs.size())
	{
		d = d - faceProbs[i].area;
		i++;
	}
	if (d<0 || fabs(d) < 0.000001) i--;
	else
	{
		printf("d=%lf\nindex=%d\tarea_size=%d\n", d, i, faceProbs.size());
		return false;
	}

	face_index = faceProbs[i].index;	/* "index" indicates the selected triangle */
	MyMesh::FaceHandle fh(face_index);
	MyMesh::ConstFaceVertexIter cfvit = mesh.cfv_iter(fh);
	MyMesh::Point point1 = mesh.point(*cfvit);
	MyMesh::Point point2 = mesh.point(*(++cfvit));
	MyMesh::Point point3 = mesh.point(*(++cfvit));
	MyMesh::Point point = (point1 + point2 + point3) / 3.0;

	MyMesh::Point bary_point = (point1 + point2 + point3) / 3.0;
	bary_coord[0] = bary_point[0];
	bary_coord[1] = bary_point[1];
	bary_coord[2] = bary_point[2];

	//	srand((unsigned)time(NULL)); 
	double r1 = (double)(rand() / (double)RAND_MAX);
	double r2 = (double)(rand() / (double)RAND_MAX);
	v[0] = (1 - sqrt(r1)) * point1[0] + sqrt(r1)*(1 - r2) * point2[0] + sqrt(r1) * r2 * point3[0];
	v[1] = (1 - sqrt(r1)) * point1[1] + sqrt(r1)*(1 - r2) * point2[1] + sqrt(r1) * r2 * point3[1];
	v[2] = (1 - sqrt(r1)) * point1[2] + sqrt(r1)*(1 - r2) * point2[2] + sqrt(r1) * r2 * point3[2];

	return true;
}

double PointCloudLoader::TriangleArea(const Eigen::Vector3f & pA, const Eigen::Vector3f & pB, const Eigen::Vector3f & pC)
{
	//calculate the lengths of sides pA-pB and pA-pC
	Eigen::Vector3f v1 = pB - pA;
	double a = v1.norm();
	Eigen::Vector3f v2 = pC - pA;
	double b = v2.norm();
	//calculate the angle between sides pA-pB and pA-pC
	if (a == 0 || b == 0)
		return 0;
	double cos_angle = v1.dot(v2) / (a * b);

	//avoid accumulative error
	if (cos_angle>1)
		cos_angle = 1;
	else if (cos_angle<-1)
		cos_angle = -1;

	double angle = acos(cos_angle);
	double area = a*b*sin(angle) / 2;

	return area;
}
