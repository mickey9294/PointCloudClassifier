#include <iostream>
#include "PointCloudLoader.h"
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/ply_io.h>
#include "FeaturesExtractor.h"

void save_labels(const char * path, const std::vector<int> &labels)
{
	std::ofstream out(path);
	if (out.is_open())
	{
		for (std::vector<int>::const_iterator it = labels.begin(); it != labels.end(); ++it)
		{
			std::vector<std::string> label_line(8, "0");
			label_line[*it] = "1";
			std::string line = boost::algorithm::join(label_line, ",");
			out << line << std::endl;
		}

		out.close();
	}
}

int main(int argc, char **argv)
 {
	if (argc < 4)
	{
		std::cout << "Usage: ./" << argv[0] << " shape_category_name  shapes_list_file  output_directory  [labels_directory]\n";
	}
	else
	{
		std::string shape_category(argv[1]);
		std::string shapes_list_file(argv[2]);
		std::string output_directory(argv[3]);
		std::string labels_dir;
		if (argc == 5)
		{
			labels_dir = std::string(argv[4]);
		}

		FeaturesExtractor extractor(shape_category);
		extractor.extract_features(shapes_list_file, output_directory, labels_dir);
	}

	//system("PAUSE");
	return 0;
}