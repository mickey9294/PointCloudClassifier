#include <iostream>
#include "PartsSplitter.h"

int main(int argc, char **argv)
{
	if (argc < 5)
	{
		std::cout << "Usage: ./" << argv[0] << ":\nshape_category  shapes_list_file  labels_directory  new_labels_dir  order_directions  [force_match/no_less]\n";
		std::cout << "of" << std::endl;
		std::cout << "count_parts  shapes_list_file  num_labels  labels_directory" << std::endl;
	}
	else
	{
		if (strcmp(argv[1], "count_parts") != 0)
		{
			std::string shape_category(argv[1]);
			std::string shape_list_file(argv[2]);
			std::string labels_dir(argv[3]);
			std::string new_labels_dir(argv[4]);
			std::vector<Eigen::Vector3f> order_directions;
			int force_match = 0;
			if (argc < 11)
			{
				order_directions.resize(1);
				float x = std::atof(argv[5]);
				float y = std::atof(argv[6]);
				float z = std::atof(argv[7]);
				order_directions[0][0] = x;
				order_directions[0][1] = y;
				order_directions[0][2] = z;

				if (argc == 9)
				{
					std::string force(argv[8]);
					if (force.compare("force_match") == 0)
						force_match = 2;
					else if (force.compare("no_less") == 0)
						force_match = 1;
					else if (force.compare("no_more") == 0)
						force_match = 3;
				}
			}
			else if (argc >= 11)
			{
				order_directions.resize(2);
				float x1 = std::atof(argv[5]);
				float y1 = std::atof(argv[6]);
				float z1 = std::atof(argv[7]);
				float x2 = std::atof(argv[8]);
				float y2 = std::atof(argv[9]);
				float z2 = std::atof(argv[10]);
				order_directions[0][0] = x1;
				order_directions[0][1] = y1;
				order_directions[0][2] = z1;
				order_directions[1][0] = x2;
				order_directions[1][1] = y2;
				order_directions[1][2] = z2;

				if (argc == 12)
				{
					std::string force(argv[11]);
					if (force.compare("force_match") == 0)
						force_match = 2;
					else if (force.compare("no_less") == 0)
						force_match = 1;
					else if (force.compare("no_more") == 0)
						force_match = 3;
				}
			}

			boost::shared_ptr<PartsSplitter> splitter(new PartsSplitter(shape_category));
			splitter->split_parts(shape_list_file, labels_dir, new_labels_dir, order_directions, force_match);
		}
		else
		{
			std::string shapes_list_file(argv[2]);
			int num_parts = std::atoi(argv[3]);
			std::string labels_dir(argv[4]);

			boost::shared_ptr<PartsSplitter> splitter(new PartsSplitter());
			splitter->count_parts(shapes_list_file, num_parts, labels_dir);
		}
	}

	system("PAUSE");
	return 0;
}