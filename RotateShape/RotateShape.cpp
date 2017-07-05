 // RotateShape.cpp : Defines the entry point for the console application.
//
#include <fstream>
#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#define PI 3.14159265358

void load_files_list(const std::string shapes_dir, std::list<std::string> &files_lsit);
void rotate_shape(const std::string input_shape_dir, const std::string shape_name, const std::string output_shape_dir,
	double rx, double ry, double rz, double angle);

int main(int argc, char **argv)
{
	std::string input_shapes_dir(argv[1]);
	std::string output_shapes_dir(argv[2]);
	double rx = std::atof(argv[3]);
	double ry = std::atof(argv[4]);
	double rz = std::atof(argv[5]);
	double angle = std::atof(argv[6]);

	std::list<std::string> shapes_list;
	load_files_list(input_shapes_dir, shapes_list);

	for (std::list<std::string>::iterator it = shapes_list.begin(); it != shapes_list.end(); ++it)
	{
		rotate_shape(input_shapes_dir, *it, output_shapes_dir, rx, ry, rz, angle);
	}

	system("PAUSE");
    return 0;
}

void load_files_list(const std::string shapes_dir, std::list<std::string> &files_lsit)
{
	boost::filesystem::path boost_shapes_dir(shapes_dir);

	boost::filesystem::directory_iterator end_it;

	if (boost::filesystem::exists(boost_shapes_dir) && boost::filesystem::is_directory(boost_shapes_dir))
	{
		for (boost::filesystem::directory_iterator dir_it(boost_shapes_dir);
			dir_it != end_it; ++dir_it)
		{
			std::string file_name = dir_it->path().filename().string();
			//std::cout << file_name <<  std::endl;
			files_lsit.push_back(file_name);
		}
	}
}

void rotate_shape(const std::string input_shape_dir, const std::string shape_name, const std::string output_shape_dir,
	double rx, double ry, double rz, double angle)
{
	std::string input_shape_path = input_shape_dir + "\\" + shape_name;

	boost::filesystem::path boost_file_path(input_shape_path.c_str());
	std::string format_extension = boost_file_path.extension().string();

	double rad_angle = angle / 180.0 * PI;
	Eigen::Vector3d rotate_axis(rx, ry, rz);
	Eigen::Affine3d rotation;
	rotation.setIdentity();
	Eigen::AngleAxisd rot(rad_angle, rotate_axis);
	rotation *= rot;

	std::ifstream in(input_shape_path.c_str());
	
	if (in.is_open())
	{
		boost::filesystem::path out_dir(output_shape_dir);
		if (!boost::filesystem::exists(out_dir))
			boost::filesystem::create_directories(out_dir);

		std::string output_shape_path = output_shape_dir + "\\" + shape_name;
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

					out << vert[0] << " " << vert[1] << " " << vert[2] << std::endl;
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
		else if (format_extension.compare(".pts") == 0)
		{
			if (out.is_open())
			{
				std::list<Eigen::Vector3d> vertices;
				std::list<int> face_indices;
				std::list<Eigen::Vector3d> barys;

				while (!in.eof())
				{
					std::string line;
					std::getline(in, line);
					if (line.length() > 0)
					{
						std::vector<std::string> line_split;
						boost::split(line_split, line, boost::is_any_of(" "), boost::token_compress_on);

						double x, y, z;
						Eigen::Vector3d vert;

						if (line_split.size() > 3)
						{
							int face_idx = std::stoi(line_split[0]);
							double bx = std::stod(line_split[1]);
							double by = std::stod(line_split[2]);
							double bz = std::stod(line_split[3]);
							vert[0] = std::stod(line_split[4]);
							vert[1] = std::stod(line_split[5]);
							vert[2] = std::stod(line_split[6]);

							face_indices.push_back(face_idx);
							Eigen::Vector3d bary_center(bx, by, bz);
							bary_center = rotation * bary_center;
							barys.push_back(bary_center);
						}
						else
						{
							vert[0] = std::stod(line_split[0]);
							vert[1] = std::stod(line_split[1]);
							vert[2] = std::stod(line_split[2]);
						}

						vert = rotation * vert;

						vertices.push_back(vert);

						if (vert[0] > max_coords[0])
							max_coords[0] = vert[0];
						if (vert[0] < min_coords[0])
							min_coords[0] = vert[0];
						if (vert[1] > max_coords[1])
							max_coords[1] = vert[1];
						if (vert[1] < min_coords[1])
							min_coords[1] = vert[1];
						if (vert[2] > max_coords[2])
							max_coords[2] = vert[2];
						if (vert[2] < min_coords[2])
							min_coords[2] = vert[2];
					}
				}

				double diameter = std::sqrt(std::pow(max_coords[0] - min_coords[0], 2)
					+ std::pow(max_coords[1] - min_coords[1], 2)
					+ std::pow(max_coords[2] - min_coords[2], 2));
				double scale = 1.0 / diameter;
				center = (max_coords + min_coords) / 2.0;
				center[2] -= 0.5 * (max_coords[2] - min_coords[2]);

				if (!face_indices.empty())
				{
					std::list<int>::iterator f_it;
					std::list<Eigen::Vector3d>::iterator b_it;
					std::list<Eigen::Vector3d>::iterator v_it;
					for (f_it = face_indices.begin(), b_it = barys.begin(), v_it = vertices.begin();
						f_it != face_indices.end() && b_it != barys.end() && v_it != vertices.end();
						++f_it, ++v_it, ++b_it)
					{
						out << *f_it << " ";
						*b_it = (*b_it - center) * scale;
						*v_it = (*v_it - center) * scale;

						out << b_it->x() << " " << b_it->y() << " " << b_it->z() << " "
							<< v_it->x() << " " << v_it->y() << " " << v_it->z() << std::endl;
					}
				}
				else
				{
					for (std::list<Eigen::Vector3d>::iterator v_it = vertices.begin();
						v_it != vertices.end(); ++v_it)
					{
						*v_it = (*v_it - center) * scale;
						
						out << v_it->x() << " " << v_it->y() << " " << v_it->z() << std::endl;
					}
				}

				out.close();
			}
			else
			{
				std::cerr << "Error: cannot open output file " << output_shape_path << std::endl;
			}
		}

		in.close();
	}
	else
	{
		std::cerr << "Error: cannot open input file " << input_shape_path << std::endl;
	}
}