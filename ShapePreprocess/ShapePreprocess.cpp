// ShapePreprocess.cpp : Defines the entry point for the console application.
//

#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <tchar.h>
#include <string>
#include <list>

#include <Eigen\Core>
#include <Eigen\Geometry>
#include <boost\filesystem.hpp>
#include <boost\algorithm\string.hpp>



void rotate(const std::string &obj_path);

int main(int argc, char **argv)
{
	std::string obj_dir(argv[1]);
	std::list<std::string> obj_path_list;
	std::list<std::string> obj_list;
	for (boost::filesystem::directory_iterator itr(obj_dir);
		itr != boost::filesystem::directory_iterator(); ++itr)
	{
		obj_path_list.push_back(itr->path().string());
		obj_list.push_back(itr->path().stem().string());
	}

	std::string shape_category = "coseg_chairs";
	std::string data_root = "D:\\Projects\\shape2pose\\data\\";
	std::string off_dir = data_root + "1_input\\" + shape_category + "\\off";
	std::string pts_dir = data_root + "2_analysis\\" + shape_category + "\\points\\even1000";
	std::string features_dir = data_root + "2_analysis\\" + shape_category + "\\features";

	std::ofstream list_out("test_list.txt");

	std::string cmd;

	std::list<std::string>::iterator oit;
	std::list<std::string>::iterator opit;
	for (oit = obj_list.begin(), opit = obj_path_list.begin();
		oit != obj_list.end() && opit != obj_path_list.end(); ++oit, ++opit)
	{
		std::string output_shape_path = *oit + ".off";

		cmd = "..\\X64\\Release\\VoxelsSampling " + *opit + " " + shape_category;
		system(cmd.c_str());

		cmd = "del " + output_shape_path;
		system(cmd.c_str());

		list_out << pts_dir + "\\" + *oit + ".pts" << std::endl;
		
	}

	list_out.close();

	cmd = "..\\x64\\Release\\ComputeFeatures " + shape_category + " test_list.txt " + features_dir;
	system(cmd.c_str());

	std::string test_features_list_file = features_dir + "\\" + shape_category + "_feats.txt";
	std::string models_dir = data_root + "3_trained\\classifier\\exp1_" + shape_category;
	std::string label_out_dir = data_root + "4_experiments\\exp1_" 
		+ shape_category + "\\1_prediction";
	cmd = "..\\x64\\Release\\PointClassifier batch_test " + shape_category + " " + test_features_list_file + " " + models_dir
		+ " " + label_out_dir;
	system(cmd.c_str());

    return 0;
}


void rotate(const std::string &obj_path)
{
	double rad_angle_z = -90.0 / 180.0 * M_PI, rad_angle_x = 90.0 / 180.0 * M_PI;
	Eigen::Vector3d rotate_axis_z(0, 1, 0), rotate_axis_x(1, 0, 0);
	Eigen::AngleAxisd rot_z(rad_angle_z, rotate_axis_z), rot_x(rad_angle_x, rotate_axis_x);
	Eigen::Affine3d rotation;
	rotation.setIdentity();
	
	rotation *= rot_x;
	rotation *= rot_z;

	std::ifstream in(obj_path.c_str());

	if (in.is_open())
	{
		boost::filesystem::path input_path(obj_path);
		std::string shape_name = input_path.stem().string();
		std::string format_extension = input_path.extension().string();

		std::string output_shape_path = shape_name + ".off";
		std::ofstream out(output_shape_path.c_str());

		/* used for normalize the point cloud */
		Eigen::Vector3d min_coords(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
		Eigen::Vector3d max_coords(std::numeric_limits<double>::min(), std::numeric_limits<double>::min(), std::numeric_limits<double>::min());
		Eigen::Vector3d center(0, 0, 0);

		if (format_extension.compare(".off") == 0)
		{
			std::string line;
			std::getline(in, line);
			std::getline(in, line);
			std::vector<std::string> line_split;
			boost::split(line_split, line, boost::is_any_of(" "), boost::token_compress_on);
			int num_vertices = std::stoi(line_split[0]);
			int num_faces = std::stoi(line_split[1]);

			if (out.is_open())
			{
				out << "OFF" << std::endl;
				out << num_vertices << " " << num_faces << " 0" << std::endl;
				
				std::list<Eigen::Vector3d> vertices;

				for (int i = 0; i < num_vertices; i++)
				{
					line_split.clear();
					std::getline(in, line);

					boost::split(line_split, line, boost::is_any_of(" "), boost::token_compress_on);
					double x = std::stod(line_split[0]);
					double y = std::stod(line_split[1]);
					double z = std::stod(line_split[2]);

					Eigen::Vector3d vert(x, y, z);

					vert = rotation * vert;

					//out << vert[0] << " " << vert[1] << " " << vert[2] << std::endl;
					if (vert[0] > max_coords[0])
						max_coords[0] = vert[0];
					if (vert[1] > max_coords[1])
						max_coords[1] = vert[1];
					if (vert[2] > max_coords[2])
						max_coords[2] = vert[2];
					if (vert[0] < min_coords[0])
						min_coords[0] = vert[0];
					if (vert[1] < min_coords[1])
						min_coords[1] = vert[1];
					if (vert[2] < min_coords[2])
						min_coords[2] = vert[2];

					vertices.push_back(vert);
				}

				center = (max_coords + min_coords) / 2.0;
				double diameter = (max_coords - min_coords).norm();
				double height = max_coords[2] - min_coords[2];
				Eigen::Vector3d move(0, 0, -(min_coords[2] - center[2]) / diameter);

				for (std::list<Eigen::Vector3d>::iterator vit = vertices.begin(); vit != vertices.end(); ++vit)
				{
					vit->operator-=(center);
					vit->operator/=(diameter);
					vit->operator+=(move);

					out << vit->x() << " " << vit->y() << " " << vit->z() << std::endl;
				}


				for (int i = 0; i < num_faces; i++)
				{
					std::getline(in, line);

					out << line << std::endl;
				}

				out.close();
			}
			else
			{
				std::cerr << "Error: cannot open output file " << output_shape_path << std::endl;
			}
		}
		else if (format_extension.compare(".obj") == 0)
		{
			std::string line;
			std::getline(in, line);
			std::getline(in, line);
			std::vector<std::string> line_split;
			boost::split(line_split, line, boost::is_any_of(" "), boost::token_compress_on);
			int num_vertices = std::stoi(line_split[1]);

			if (out.is_open())
			{
				out << "OFF" << std::endl;

				std::list<Eigen::Vector3d> vertices;

				for (int i = 0; i < num_vertices; i++)
				{
					line_split.clear();
					std::getline(in, line);

					boost::split(line_split, line, boost::is_any_of(" "), boost::token_compress_on);
					double x = std::stod(line_split[1]);
					double y = std::stod(line_split[2]);
					double z = std::stod(line_split[3]);

					Eigen::Vector3d vert(x, y, z);

					vert = rotation * vert;

					//out << vert[0] << " " << vert[1] << " " << vert[2] << std::endl;
					if (vert[0] > max_coords[0])
						max_coords[0] = vert[0];
					if (vert[1] > max_coords[1])
						max_coords[1] = vert[1];
					if (vert[2] > max_coords[2])
						max_coords[2] = vert[2];
					if (vert[0] < min_coords[0])
						min_coords[0] = vert[0];
					if (vert[1] < min_coords[1])
						min_coords[1] = vert[1];
					if (vert[2] < min_coords[2])
						min_coords[2] = vert[2];

					vertices.push_back(vert);
				}

				center = (max_coords + min_coords) / 2.0;
				double diameter = (max_coords - min_coords).norm();
				double height = max_coords[2] - min_coords[2];
				Eigen::Vector3d move(0, 0, -(min_coords[2] - center[2]) / diameter);

				for (std::list<Eigen::Vector3d>::iterator vit = vertices.begin(); vit != vertices.end(); ++vit)
				{
					vit->operator-=(center);
					vit->operator/=(diameter);
					vit->operator+=(move);

					//out << vit->x() << " " << vit->y() << " " << vit->z() << std::endl;
				}

				std::list<Eigen::Vector3i> faces;
				std::getline(in, line);
				boost::split(line_split, line, boost::is_any_of(" "), boost::token_compress_on);
				int num_faces = std::stoi(line_split[1]);
				for(int i = 0; i < num_faces; i++)
				{
					std::getline(in, line);
					boost::split(line_split, line, boost::is_any_of(" "), boost::token_compress_on);
					Eigen::Vector3i face(std::stoi(line_split[1]),
						std::stoi(line_split[2]), std::stoi(line_split[3]));
					faces.push_back(face);
				}

				out << vertices.size() << " " << faces.size() << " 0" << std::endl;
				for (std::list<Eigen::Vector3d>::iterator vit = vertices.begin(); vit != vertices.end(); ++vit)
					out << vit->x() << " " << vit->y() << " " << vit->z() << std::endl;
				for (std::list<Eigen::Vector3i>::iterator fit = faces.begin(); fit != faces.end(); ++fit)
					out << "3 " << fit->x()-1 << " " << fit->y()-1 << " " << fit->z()-1 << std::endl;

				out.close();
			}
			else
			{
				std::cerr << "Error: cannot open output file " << output_shape_path << std::endl;
			}
		}
		in.close();
	}
}