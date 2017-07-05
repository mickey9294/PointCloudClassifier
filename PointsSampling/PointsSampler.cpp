#include "PointsSampler.h"



PointsSampler::PointsSampler()
{
}


PointsSampler::~PointsSampler()
{
}

void PointsSampler::batch_sample(const std::string shapes_list_file, const std::string samples_dir, 
	const std::string dense_samples_dir, const int num_samples, const int num_dense_samples, 
	const std::string modified_shapes_dir, const std::string labels_dir, const std::string output_labels_dir)
{
	std::cout << "Do batch sampling." << std::endl;

	std::list<std::string> shape_files_list;
	load_shapes_paths(shapes_list_file, shape_files_list);

	std::list<std::string> sample_files_list;

	for (std::list<std::string>::iterator shape_it = shape_files_list.begin(); 
		shape_it != shape_files_list.end(); ++shape_it)
	{

		sample_on_one_shape(*shape_it, samples_dir, dense_samples_dir, num_samples,
			num_dense_samples, modified_shapes_dir, labels_dir, output_labels_dir, sample_files_list);
	}

	std::string list_file_path = samples_dir + "\\..\\samples_list.txt";
	std::ofstream list_out(list_file_path.c_str());
	if (list_out.is_open())
	{
		for (std::list<std::string>::iterator file_it = sample_files_list.begin(); file_it != sample_files_list.end(); ++file_it)
			list_out << *file_it << std::endl;
	}
	else
	{
		std::cerr << "Error: cannot save sample files list to " << list_file_path << std::endl;
	}
	std::cout << "Sampling has all done." << std::endl;
}

void PointsSampler::sample_on_one_shape(const std::string shape_file_path, const std::string samples_dir, 
	const std::string dense_samples_dir, const int num_samples, const int num_dense_samples, 
	const std::string modified_shapes_dir, const std::string labels_dir, const std::string output_labels_dir,
	std::list<std::string> &sample_files_list)
{
	boost::filesystem::path boost_shape_path(shape_file_path);
	std::string format = boost_shape_path.extension().string();
	std::string shape_name = boost_shape_path.stem().string();

	std::cout << "Sample points on shape " << shape_name << "." << std::endl;

	if (format.compare(".off") == 0)
	{
		MyMesh mesh;
		mesh.open_mesh(shape_file_path.c_str());

		if (mesh.n_faces() > 0)  /* If it is a training mesh model */
		{
			/* Sample points on the surface of the mesh */
			std::vector<Eigen::Vector3f> samples, dense_samples, bary_coords, dense_bary_coords;
			std::vector<int> face_index_map, dense_face_index_map, sample_labels;

			sample_on_mesh(mesh, num_samples, samples, face_index_map, bary_coords);
			if (num_dense_samples > 0)
			{
				if (num_dense_samples == num_samples)
				{
					dense_samples.resize(samples.size());
					dense_face_index_map.resize(face_index_map.size());
					dense_bary_coords.resize(bary_coords.size());
					for (int i = 0; i < samples.size(); i++)
					{
						dense_samples[i] = samples[i];
						dense_face_index_map[i] = face_index_map[i];
						dense_bary_coords[i] = bary_coords[i];

					}
				}
				else
					sample_on_mesh(mesh, num_dense_samples, dense_samples, dense_face_index_map, dense_bary_coords);
			}

			/* Save the sample points */
			std::string samples_pts_file = samples_dir + "\\" + shape_name + ".pts";
			//std::string samples_off_file = samples_dir + "\\" + shape_name + ".off";
			std::string dense_samples_pts_file = dense_samples_dir + "\\" + shape_name + ".pts";
			//std::string dense_samples_off_file = dense_samples_dir + "\\" + shape_name + ".off";
			save_sample_points(samples_pts_file, samples, face_index_map, bary_coords);
			//save_sample_points(samples_off_file, samples, face_index_map, bary_coords);
			if(num_dense_samples > 0)
				save_sample_points(dense_samples_pts_file, dense_samples, dense_face_index_map, dense_bary_coords);
			//save_sample_points(dense_samples_off_file, dense_samples, dense_face_index_map, dense_bary_coords);
			sample_files_list.push_back(samples_pts_file);
			
			/* Save the labels of sample points */
			if (labels_dir.length() > 0)
			{
				std::string mesh_label_path = labels_dir + "\\" + shape_name + ".seg";
				load_samples_labels(mesh_label_path, mesh.n_faces(), face_index_map, sample_labels);

				std::string labels_out_path = output_labels_dir + "\\" + shape_name + ".seg";
				save_labels(labels_out_path, sample_labels);
			}

			/* Save the shape normalized */
			if (modified_shapes_dir.length() > 0)
			{
				std::string normalized_save_path = modified_shapes_dir + "\\" + shape_name + ".off";
				mesh.save_mesh_to_off(normalized_save_path.c_str());
			}
		}

		std::cout << "Sampling on shape " << shape_name << " has done." << std::endl;
	}
	else
	{
		std::cerr << "Error: File format error for shape " << boost::filesystem::basename(boost_shape_path) 
			<< ". The program does not take " << format << " file as input." << std::endl;
	}
}

int PointsSampler::load_shapes_paths(const std::string shapes_list_file, std::list<std::string>& shapes_list)
{
	std::ifstream in(shapes_list_file);
	if (in.is_open())
	{
		while (!in.eof())
		{
			std::string line;
			std::getline(in, line);
			if (line.length() > 0)
				shapes_list.push_back(line);
		}

		in.close();
	}
	else
	{
		std::cerr << "Error: cannot open shapes list file." << std::endl;
		return 0;
	}
	return 0;
}

void PointsSampler::sample_on_mesh(const MyMesh & mesh, const int num_samples, std::vector<Eigen::Vector3f> &samples,
	std::vector<int>& face_index_map, std::vector<Eigen::Vector3f> &bary_coords)
{
	std::vector<FacetProb> faceProbs;
	setFaceProb(mesh, faceProbs);

	srand(time(NULL));

	samples.resize(num_samples);
	face_index_map.resize(num_samples);
	bary_coords.resize(num_samples);

	for (int i = 0; i < num_samples; i++)
	{
		Eigen::Vector3f v;
		Eigen::Vector3f bary_coord;
		int face_index;
		getRandomPoint(mesh, faceProbs, v, bary_coord, face_index);

		samples[i][0] = v[0];
		samples[i][1] = v[1];
		samples[i][2] = v[2];
		face_index_map[i] = face_index;
		bary_coords[i] = bary_coord;
	}
}

void PointsSampler::setFaceProb(const MyMesh & mesh, std::vector<FacetProb>& faceProbs)
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

bool PointsSampler::getRandomPoint(const MyMesh & mesh, const std::vector<FacetProb>& faceProbs,
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

double PointsSampler::TriangleArea(const Eigen::Vector3f & pA, const Eigen::Vector3f & pB, const Eigen::Vector3f & pC)
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

void PointsSampler::save_sample_points(const std::string save_path, const std::vector<Eigen::Vector3f>& samples, 
	const std::vector<int>& face_index_map, const std::vector<Eigen::Vector3f>& bary_coords)
{
	std::ofstream out(save_path.c_str());

	if (out.is_open())
	{
		boost::filesystem::path boost_save_path(save_path);
		std::string format = boost_save_path.extension().string();

		if (format.compare(".pts") == 0)
		{
			std::vector<Eigen::Vector3f>::const_iterator sample_it, bary_it;
			std::vector<int>::const_iterator idx_it;

			for (sample_it = samples.begin(), bary_it = bary_coords.begin(), idx_it = face_index_map.begin();
				sample_it != samples.end() && bary_it != bary_coords.end() && idx_it != face_index_map.end();
				++sample_it, ++bary_it, ++idx_it)
			{
				out << *idx_it << " "
					<< bary_it->x() << " " << bary_it->y() << " " << bary_it->z() << " "
					<< sample_it->x() << " " << sample_it->y() << " " << sample_it->z() << std::endl;
			}
		}
		else if (format.compare(".off") == 0)
		{
			out << "OFF" << std::endl;
			out << samples.size() << " 0 0" << std::endl;

			for (std::vector<Eigen::Vector3f>::const_iterator sample_it = samples.begin();
				sample_it != samples.end(); ++sample_it)
				out << sample_it->x() << " " << sample_it->y() << " " << sample_it->z() << std::endl;
		}
		else if (format.compare(".ply") == 0)
		{
			out << "ply" << std::endl;
			out << "format ascii 1.0" << std::endl;
			out << "element vertex " << samples.size() << std::endl;
			out << "property float x" << std::endl;
			out << "property float y" << std::endl;
			out << "property float z" << std::endl;
			out << "element face 0" << std::endl;
			out << "property list uchar int vertex_indices" << std::endl;
			out << "end_header" << std::endl;

			for (std::vector<Eigen::Vector3f>::const_iterator sample_it = samples.begin();
				sample_it != samples.end(); ++sample_it)
				out << sample_it->x() << " " << sample_it->y() << " " << sample_it->z() << std::endl;
		}

		out.close();
	}
	else
	{
		std::cerr << "Error: cannot save sample points to " << save_path << std::endl;
	}
}

void PointsSampler::load_samples_labels(const std::string mesh_labels_path, const int nfaces, 
	const std::vector<int>& face_index_map, std::vector<int>& labels)
{
	std::ifstream in(mesh_labels_path.c_str());

	if (in.is_open())
	{
		labels.clear();
		labels.reserve(face_index_map.size());
		std::vector<int> face_labels;
		face_labels.reserve(nfaces);

		char buffer[3];
		while (!in.eof())
		{
			in.getline(buffer, 3);
			if (strlen(buffer) > 0)
			{
				int face_label = std::atoi(buffer);
				face_labels.push_back(face_label);
			}
		}

		for (std::vector<int>::const_iterator face_index_it = face_index_map.begin();
			face_index_it != face_index_map.end(); ++face_index_it)
		{
			int point_label = face_labels[*face_index_it];
			labels.push_back(point_label);
		}

		in.close();
	}
	else
	{
		std::cerr << "Error: cannot read mesh label file " << mesh_labels_path << std::endl;
	}
}

void PointsSampler::save_labels(const std::string save_path, const std::vector<int>& labels)
{
	std::ofstream out(save_path);
	if (out.is_open())
	{
		for (std::vector<int>::const_iterator it = labels.begin(); it != labels.end(); ++it)
			out << *it << std::endl;

		out.close();
	}
	else
	{
		std::cerr << "Error: cannot save label file " << save_path << std::endl;
	}
}
