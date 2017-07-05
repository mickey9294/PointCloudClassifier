#include "ClassifierModel.h"



ClassifierModel::ClassifierModel()
{
}


ClassifierModel::~ClassifierModel()
{
}

bool ClassifierModel::load_label_info(std::string regions_file, bool _verbose)
{
	const char *_filename = regions_file.c_str();

	std::ifstream file(_filename);
	if (!file)
	{
		std::cerr << "Can't open file: \"" << _filename << "\"" << std::endl;
		return false;
	}

	if (_verbose)
		std::cout << "Loading " << _filename << "..." << std::endl;


	// NOTE:
	// All cuboids are also deleted.
	m_labels.clear();
	m_label_names.clear();


	std::string buffer;
	int new_label = 0;

	while (!file.eof())
	{
		std::getline(file, buffer);
		if (buffer == "") break;

		std::stringstream sstr(buffer);

		const unsigned int num_tokens = 3;
		std::string tokens[num_tokens];

		for (unsigned int i = 0; i < num_tokens; ++i)
		{
			if (sstr.eof())
			{
				std::cerr << "Error: Wrong file format: \"" << _filename << "\"" << std::endl;
				return false;
			}

			std::getline(sstr, tokens[i], ' ');
		}

		if (tokens[1] != "pnts" || tokens[2] != "1")
		{
			std::cerr << "Error: Wrong file format: \"" << _filename << "\"" << std::endl;
			return false;
		}

		// NOTE:
		// In this file format, labels are defined by the recorded order.
		m_labels.push_back(new_label);
		m_label_names.push_back(tokens[0]);
		++new_label;
	}

	m_null_label = new_label;

	file.close();
}

bool ClassifierModel::load_regions_symmetry(std::string regions_symmetry_file)
{
	std::ifstream in(regions_symmetry_file.c_str());
	if (in.is_open())
	{
		while (!in.eof())
		{
			std::string line;
			std::getline(in, line);

			if (line.length() > 0)
			{
				std::vector<std::string> line_list;
				boost::split(line_list, line, boost::is_any_of(" "), boost::token_compress_on);

				std::vector<int> sym_labels(line_list.size());
				for (int i = 0; i < line_list.size(); i++)
				{
					int label = get_label_from_label_name(line_list[i]);

					sym_labels[i] = label;
				}

				m_regions_symmetry.push_back(sym_labels);
			}
		}
	}
	else
		return false;

	return true;
}

int ClassifierModel::load_training_data(const std::string features_list_path, const std::string labels_dir,
	shark::Data<shark::RealVector>& dataset, std::vector<int>& data_labels)
{
	assert(!m_labels.empty());

	std::cout << "Loading training data." << std::endl;

	std::list<std::string> feat_files_list;

	/* Read features files list */
	std::ifstream list_in(features_list_path.c_str());
	if (list_in.is_open())
	{
		while (!list_in.eof())
		{
			std::string line;
			std::getline(list_in, line);
			if (line.length() > 0)
				feat_files_list.push_back(line);
		}
	}

	int total_num_points = 0;

	for (std::list<std::string>::iterator file_it = feat_files_list.begin(); file_it != feat_files_list.end(); ++file_it)
	{
		/* Read the features data */
		shark::Data<shark::RealVector> one_data;
		shark::importCSV(one_data, *file_it);
		std::vector<int> one_labels;
		one_labels.reserve(6000);

		int num_points = one_data.numberOfElements();
		total_num_points += num_points;

		/* Read the labels */
		boost::filesystem::path feat_path(*file_it);
		std::string shape_name = boost::filesystem::basename(feat_path);
		std::string label_file = labels_dir + "\\" + shape_name + ".seg";
		std::ifstream seg_in(label_file.c_str());
		if (seg_in.is_open())
		{
			char buffer[3];
			while (!seg_in.eof())
			{
				seg_in.getline(buffer, 3);
				if (strlen(buffer) > 0)
				{
					int label = std::atoi(buffer);
					if (label < m_labels.size() && label >= 0)
						one_labels.push_back(label);
					else
						one_labels.push_back(m_null_label);
				}
			}

			seg_in.close();
		}

		one_labels.shrink_to_fit();

		/* Append features and labels of this shape to dataset */
		dataset.append(one_data);
		data_labels.insert(data_labels.end(), one_labels.begin(), one_labels.end());
	}

	std::cout << "Loading training data done." << std::endl;

	return total_num_points;
}

int ClassifierModel::load_training_data(const std::string features_list_path, const std::string labels_dir, 
	std::vector<shark::RealVector>& dataset, std::vector<int>& data_labels)
{
	assert(!m_labels.empty());

	std::cout << "Loading training data." << std::endl;

	std::list<std::string> feat_files_list;

	/* Read features files list */
	std::ifstream list_in(features_list_path.c_str());
	if (list_in.is_open())
	{
		while (!list_in.eof())
		{
			std::string line;
			std::getline(list_in, line);
			if (line.length() > 0)
				feat_files_list.push_back(line);
		}
	}

	shark::Data<shark::RealVector> shark_data;

	for (std::list<std::string>::iterator file_it = feat_files_list.begin(); file_it != feat_files_list.end(); ++file_it)
	{
		/* Read the features data */
		shark::Data<shark::RealVector> one_data;
		shark::importCSV(one_data, *file_it);
		std::vector<int> one_labels;
		one_labels.reserve(6000);

		shark_data.append(one_data);

		/* Read the labels */
		boost::filesystem::path feat_path(*file_it);
		std::string shape_name = boost::filesystem::basename(feat_path);
		std::string label_file = labels_dir + "\\" + shape_name + ".seg";
		std::ifstream seg_in(label_file.c_str());
		if (seg_in.is_open())
		{
			char buffer[3];
			while (!seg_in.eof())
			{
				seg_in.getline(buffer, 3);
				if (strlen(buffer) > 0)
				{
					int label = std::atoi(buffer);
					if (label < m_labels.size() && label >= 0)
						one_labels.push_back(label);
					else
						one_labels.push_back(m_null_label);
				}
			}

			seg_in.close();
		}

		one_labels.shrink_to_fit();
		data_labels.insert(data_labels.end(), one_labels.begin(), one_labels.end());
		if (one_labels.size() != one_data.numberOfElements())
		{
			std::cout << "The number of data and labels do not match!" << std::endl;
		}
	}

	//shark::exportCSV(shark_data, "D:\\Projects\\shape2pose\\data\\2_analysis\\shapenet_lamps\\data.csv");

	/* copy the data to dataset vector */
	dataset.reserve(data_labels.size());
	shark::Data<shark::RealVector>::element_range elements = shark_data.elements();
	for (PREDICT_Elements::iterator pos = elements.begin(); pos != elements.end(); ++pos)
		dataset.push_back(*pos);

	std::cout << "Loading training data done." << std::endl;

	return dataset.size();
}

int ClassifierModel::load_testing_data(const std::string test_file_path, shark::Data<shark::RealVector>& test_data)
{
	std::cout << "Loading data from " << test_file_path << std::endl;

	shark::importCSV(test_data, test_file_path);

	std::cout << "Loading test data done." << std::endl;
	return test_data.numberOfElements();
}

int ClassifierModel::load_testing_data(const std::string test_file_path, shark::Data<shark::RealVector>& test_data, 
	const std::string ground_truth_path, std::vector<int> &ground_truth_labels)
{
	std::cout << "Loading data from " << test_file_path << std::endl;

	shark::importCSV(test_data, test_file_path);

	int num_points = test_data.numberOfElements();

	/* Load the ground truth labels */
	ground_truth_labels.reserve(num_points);
	std::ifstream seg_in(ground_truth_path.c_str());
	if (seg_in.is_open())
	{
		char buffer[3];

		while (!seg_in.eof())
		{
			seg_in.getline(buffer, 3);
			if (strlen(buffer) > 0)
			{
				int label = std::atoi(buffer);
				if (label >= 0 && label < m_labels.size())
					ground_truth_labels.push_back(label);
				else
					ground_truth_labels.push_back(m_null_label);
			}
		}

		seg_in.close();
	}
	else
		std::cerr << "Error: Labels file cannot be openned!" << std::endl;

	std::cout << "Loading test data and ground_truth done." << std::endl;
	return num_points;
}

bool ClassifierModel::is_symmetry(int label_0, int label_1)
{
	if (label_0 == label_1)
		return true;

	for (std::list<std::vector<int>>::iterator it = m_regions_symmetry.begin();
		it != m_regions_symmetry.end(); ++it)
	{
		if (it->size() < 2)
		{
			if (label_0 == it->front() || label_1 == it->front())
				return false;
			else
				continue;
		}
		else
		{
			if (std::find(it->begin(), it->end(), label_0) != it->end()
				&& std::find(it->begin(), it->end(), label_1) != it->end())
				return true;
			else
				continue;
		}
	}

	return false;
}

int ClassifierModel::get_label_from_label_name(std::string label_name)
{
	boost::trim(label_name);
	for (int i = 0; i < m_label_names.size(); i++)
	{
		if (label_name.compare(m_label_names[i]) == 0)
			return i;
	}

	return m_labels.size();
}
