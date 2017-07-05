#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <string>

void batch_modify(const std::string data_dir);
void modify_labels(const std::string labels_dir, const std::string new_labels_dir);

int main(int argc, char **argv)
{
	if (argc == 2)
	{
		std::string data_dir(argv[1]);

		batch_modify(data_dir);
	}
	else if (argc == 3)
	{
		std::string labels_dir(argv[1]);
		std::string new_labels_dir(argv[2]);

		modify_labels(labels_dir, new_labels_dir);
	}

	return 0;
}

void batch_modify(const std::string data_dir)
{
	boost::filesystem::path boost_data_dir(data_dir);

	boost::filesystem::directory_iterator end_it;

	for (boost::filesystem::directory_iterator dir_it(boost_data_dir); dir_it != end_it; ++dir_it)
	{
		std::string shape_name = boost::filesystem::basename(*dir_it);
		
		if (boost::filesystem::is_directory(*dir_it))
		{
			std::cout << "Processing " << shape_name << std::endl;
			std::string  shape_dir = dir_it->path().string();
			std::string labels_dir = shape_dir + "\\expert_verified\\points_label";
			std::string new_labels_dir = shape_dir + "\\gt_origin";

			modify_labels(labels_dir, new_labels_dir);

			std::cout << "done." << std::endl;
		}
	}

	std::cout << "all done." << std::endl;
}

void modify_labels(const std::string labels_dir, const std::string new_labels_dir)
{
	boost::filesystem::directory_iterator end_it;
	boost::filesystem::path boost_labels_dir(labels_dir);

	boost::filesystem::path boost_new_dir(new_labels_dir);
	if (!boost::filesystem::exists(boost_new_dir))
	{
		boost::filesystem::create_directories(boost_new_dir);
	}

	if (boost::filesystem::exists(boost_labels_dir) && boost::filesystem::is_directory(boost_labels_dir))
	{
		for (boost::filesystem::directory_iterator dir_it(boost_labels_dir);
			dir_it != end_it; ++dir_it)
		{
			std::string label_path = dir_it->path().string();
			std::string label_file = boost::filesystem::basename(dir_it->path()) + ".seg";

			std::ifstream in(label_path.c_str());
			std::string out_label_path = new_labels_dir + "\\" + label_file;
			std::ofstream out(out_label_path.c_str());
			if (in.is_open() && out.is_open())
			{
				char buffer[3];
				while (!in.eof())
				{
					in.getline(buffer, 3);
					if (std::strlen(buffer) > 0)
					{
						int label = std::atoi(buffer);
						label--;
						out << label << std::endl;
					}
				}

				in.close();
				out.close();
			}
		}
	}
}