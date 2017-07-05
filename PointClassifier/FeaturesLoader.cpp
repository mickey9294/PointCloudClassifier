#include "FeaturesLoader.h"



FeaturesLoader::FeaturesLoader()
{
}


FeaturesLoader::~FeaturesLoader()
{
}

void FeaturesLoader::load_features_from_files(const std::string features_list_file, 
	std::list<Eigen::VectorXd>& features, std::list<int>& labels, std::vector<int> &cluster_count)
{
	std::list<std::string> files_list;

	/* Load all features files */
	std::ifstream list_in(features_list_file.c_str());
	if (list_in.is_open())
	{
		std::string line;
		while (!list_in.eof())
		{
			std::getline(list_in, line);
			if (line.length() > 0)
			{
				files_list.push_back(line);
			}
		}

		list_in.close();
	}

	/* Load features from each file */
	for (std::list<std::string>::iterator file_it = files_list.begin(); file_it != files_list.end(); file_it++)
	{
		/* Read features */
		std::ifstream in(file_it->c_str());
		if (in.is_open())
		{
			std::string line;

			std::getline(in, line);
			if (line.length() > 0)
			{
				std::vector<std::string> line_list;

				boost::split(line_list, line, boost::is_any_of(","), boost::token_compress_on);
				Eigen::VectorXd feat(line_list.size());

				for (int i = 0; i < line_list.size(); i++)
					feat(i) = std::stod(line_list[i]);

				features.push_back(feat);
			}

			in.close();
		}

		/* Read labels */
		std::string seg_file = boost::filesystem::change_extension(*file_it, ".seg").string();
		std::ifstream seg_in(seg_file.c_str());
		if(seg_in.is_open())
		{
			char buffer[3];
			while (!seg_in.eof())
			{
				seg_in.getline(buffer, 3);
				if (strlen(buffer) > 0)
				{
					int label = std::atoi(buffer);
					labels.push_back(label);

					if (label < cluster_count.size() - 1)
						cluster_count[label]++;
					else
						cluster_count[cluster_count.size() - 1]++;
				}
			}

			seg_in.close();
		}
	}
}
