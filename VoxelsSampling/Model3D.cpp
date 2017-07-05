#include "Model3D.h"

Model3D::Model3D()
{
}

void Model3D::preprocess(const string& obj_path, const string &shape_category)
{
	boost::filesystem::path boost_obj_path(obj_path);
	string format = boost_obj_path.extension().string();
	shape_name_ = boost_obj_path.stem().string();

	shape_category_ = shape_category;

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	bool ret;
	if (format.compare(".obj") == 0)
		ret = readFromOBJ(obj_path, V, F);
	else if (format.compare(".off") == 0)
		ret = igl::readOFF(obj_path, V, F);
	if (ret)
	{
		Eigen::MatrixXd surface_V;
		Eigen::MatrixXi surface_F;

		get_surface_faces(V, F, surface_V, surface_F);
		rotate(surface_V);
		
		sample_on_surface_mesh(surface_V, surface_F);
	}
}

bool Model3D::readFromOBJ(string filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F) const
{
	return igl::readOBJ(filename, V, F);
}

void Model3D::get_surface_faces(Eigen::MatrixXd & V, Eigen::MatrixXi & F,
	Eigen::MatrixXd &surface_V, Eigen::MatrixXi &surface_F)
{
	ANNkd_tree *ann_kd_tree;
	ANNpointArray ann_points = annAllocPts(F.rows(), 3);

	std::vector<Eigen::Vector3d> face_centers(F.rows());
	for (int i = 0; i < F.rows(); i++)
	{
		int vid1 = F(i, 0), vid2 = F(i, 1), vid3 = F(i, 2);
		Eigen::Vector3d v1 = V.row(vid1), v2 = V.row(vid2), v3 = V.row(vid3);
		Eigen::Vector3d center = (v1 + v2 + v3) / 3.0;

		face_centers[i] = center;
		ann_points[i][0] = center[0];
		ann_points[i][1] = center[1];
		ann_points[i][2] = center[2];
	}

	ann_kd_tree = new ANNkd_tree(ann_points, F.rows(), 3);

	std::list<int> surface_centers;
	std::list<Eigen::Vector3d> surface_verts;
	std::list<Eigen::Vector3i> surface_faces;
	std::unordered_map<int, int> ori_curr_map;
	
	ANNpoint search_point = annAllocPt(3);
	ANNidxArray nn_idx = new ANNidx[2];
	ANNdistArray dd = new ANNdist[2];
	for (int i = 0; i < face_centers.size(); i++)
	{
		Eigen::Vector3d &face_center = face_centers[i];
		
		for (int j = 0; j < 3; j++)
			search_point[j] = face_center[j];

		ann_kd_tree->annkSearch(search_point, 2, nn_idx, dd);

		double nearest_dist = dd[1];
		if (nearest_dist > 2.0e-5)
		{
			surface_centers.push_back(i);

			int vid1 = F(i, 0), vid2 = F(i, 1), vid3 = F(i, 2);
			Eigen::Vector3i face;

			if (ori_curr_map.find(vid1) == ori_curr_map.end())
			{
				ori_curr_map[vid1] = surface_verts.size();
				face[0] = surface_verts.size();
				surface_verts.push_back(V.row(vid1));
			}
			else
				face[0] = ori_curr_map[vid1];

			if (ori_curr_map.find(vid2) == ori_curr_map.end())
			{
				ori_curr_map[vid2] = surface_verts.size();
				face[1] = surface_verts.size();
				surface_verts.push_back(V.row(vid2));
			}
			else
				face[1] = ori_curr_map[vid2];

			if (ori_curr_map.find(vid3) == ori_curr_map.end())
			{
				ori_curr_map[vid3] = surface_verts.size();
				face[2] = surface_verts.size();
				surface_verts.push_back(V.row(vid3));
			}
			else
				face[2] = ori_curr_map[vid3];

			surface_faces.push_back(face);
		}
	}

	surface_V.resize(surface_verts.size(), 3);
	surface_F.resize(surface_faces.size(), 3);
	int idx = 0;
	for (std::list<Eigen::Vector3d>::iterator it = surface_verts.begin();
		it != surface_verts.end(); ++it, ++idx)
	{
		surface_V.row(idx) = *it;
	}
	idx = 0;
	for (std::list<Eigen::Vector3i>::iterator it = surface_faces.begin();
		it != surface_faces.end(); ++it, ++idx)
	{
		surface_F.row(idx) = *it;
	}
}

void Model3D::sample_on_surface_mesh(Eigen::MatrixXd & surface_V, Eigen::MatrixXi & surface_F)
{
	string shape_path = shape_name_ + ".off";
	igl::writeOFF(shape_path, surface_V, surface_F);

	std::string data_root = "D:\\Projects\\shape2pose\\data\\";
	//string off_path = data_root + "1_input\\" + shape_category + "\\off\\" + name + ".off";
	string modified_off_dir = data_root + "1_input\\" + shape_category_ + "\\off";
	string pts_dir = data_root + "2_analysis\\" + shape_category_ + "\\points\\even1000";
	string dense_pts_dir = data_root + "2_analysis\\" + shape_category_ + "\\points\\random100000";

	int num_samples = 2500;
	PointsSampler sampler;
	list<string> samples_file_list;
	sampler.sample_on_one_shape(shape_path, pts_dir, dense_pts_dir, num_samples, num_samples, modified_off_dir,
		"", "", samples_file_list);
}

void Model3D::rotate(Eigen::MatrixXd & surface_V)
{
	double rad_angle_z = -90.0 / 180.0 * M_PI, rad_angle_x = 90.0 / 180.0 * M_PI;
	Eigen::Vector3d rotate_axis_z(0, 1, 0), rotate_axis_x(1, 0, 0);
	Eigen::AngleAxisd rot_z(rad_angle_z, rotate_axis_z), rot_x(rad_angle_x, rotate_axis_x);
	Eigen::Affine3d rotation;
	rotation.setIdentity();

	rotation *= rot_x;
	rotation *= rot_z;

	for (int i = 0; i < surface_V.rows(); i++)
	{
		Eigen::Vector3d vert = surface_V.row(i);
		vert = rotation * vert;
		surface_V.row(i) = vert;
	}
}