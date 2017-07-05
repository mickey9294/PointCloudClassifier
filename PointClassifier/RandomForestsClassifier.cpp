#include "RandomForestsClassifier.h"

boost::mutex cout_mutex;

RandomForestsClassifier::RandomForestsClassifier(const std::string shape_category)
	: m_shape_category(shape_category)
{
	std::string body_dir = "D:\\Projects\\shape2pose\\data\\0_body\\";
	std::string regions_file = body_dir + shape_category + "\\regions.txt";
	load_label_info(regions_file);
	std::string regions_symmetry_file = body_dir + shape_category + "\\regions_symmetry.txt";
	load_regions_symmetry(regions_symmetry_file);
}


RandomForestsClassifier::~RandomForestsClassifier()
{
}

void RandomForestsClassifier::train(const std::string features_list_path, const std::string labels_dir, const std::string save_model_dir)
{
	m_classifiers.resize(num_classifiers(), NULL);

	//shark::Data<shark::RealVector> training_data;
	std::vector<shark::RealVector> training_data_vec;
	std::vector<int> original_labels;
	load_training_data(features_list_path, labels_dir, training_data_vec, original_labels);

	int idx = 0;
	for (std::list<std::vector<int>>::iterator group_it = m_regions_symmetry.begin();
		group_it != m_regions_symmetry.end(); ++group_it, ++idx)
	{
		int label = group_it->front();
		std::cout << "Training classifier of " << m_label_names[label] << std::endl;	

		/* Genrate labels vector for the current part */
		std::vector<unsigned int> labels_for_part(original_labels.size());
		for (int j = 0; j < original_labels.size(); j++)
		{
			if (is_symmetry(original_labels[j], label))
				labels_for_part[j] = 1;
			else
				labels_for_part[j] = 0;
		}

		/* Save the training data with labels for checking */
		//std::string data_out_path = "D:\\Projects\\shape2pose\\data\\3_trained\\classifier\\exp1_shapenet_lamps\\dataset_"
		//	+ m_label_names[label] + ".csv";
		//save_dataset(data_out_path, training_data_vec, labels_for_part);
		
		/* Save the labels for the part */
		//std::string label_path = "D:\\Projects\\shape2pose\\data\\2_analysis\\shapenet_lamps\\" + m_label_names[label] + ".seg";
		//std::ofstream label_out(label_path.c_str());
		//if (label_out.is_open())
		//{
		//	for (std::vector<unsigned int>::iterator l_it = labels_for_part.begin(); l_it != labels_for_part.end(); ++l_it)
		//		label_out << *l_it << std::endl;

		//	label_out.close();
		//}

		/* Form training labeled dataset */
		shark::ClassificationDataset one_training_data = shark::createLabeledDataFromRange(training_data_vec, labels_for_part);
		
		/* Training classifier model */
		shark::RFTrainer trainer;
		//shark::RFClassifier rf_model;
		m_classifiers[idx].reset(new shark::RFClassifier());
		trainer.train(*(m_classifiers[idx]), one_training_data);

		/* Save classifier model */
		std::cout << "Training done. Now save the classifier model." << std::endl;
		std::string save_model_path = save_model_dir + "\\" + m_shape_category + "_" + m_label_names[label] + ".model";
		std::ofstream model_out(save_model_path.c_str());
		if (model_out.is_open())
		{
			boost::archive::polymorphic_text_oarchive oa(model_out);
			m_classifiers[idx]->write(oa);
			model_out.close();
		}
		else
		{
			std::cerr << "Error: saving classifier model failed. Saving file cannot be opened." << std::endl;
		}

		std::cout << "Training of part " << m_label_names[idx] << " has all done." << std::endl;
	}

	/*Eigen::MatrixXd prob_distribution;
	test("D:\\Projects\\shape2pose\\data\\2_analysis\\shapenet_lamps\\features\\4.csv", "D:\\Projects\\shape2pose\\data\\3_trained\\classifier\\exp1_shapenet_lamps",
		"D:\\Projects\\shape2pose\\data\\4_experiments\\exp1_shapenet_lamps\\2_prediction\\4.arff", prob_distribution);*/
}

void RandomForestsClassifier::load_classifier_models(const std::string models_dir)
{
	assert(!m_labels.empty());

	classifier_mutex.lock();
	m_classifiers.resize(num_classifiers(), NULL);
	classifier_mutex.unlock();

	std::vector<boost::thread> threads(num_classifiers());
	std::vector<boost::function0<void>> functions(num_classifiers());
	
	/* Initialize loading threads */
	int idx = 0;
	for(std::list<std::vector<int>>::iterator group_it = m_regions_symmetry.begin(); 
		group_it != m_regions_symmetry.end(); ++group_it, ++idx)
	{
		int label = group_it->front();
		std::string label_name = m_label_names[label];
		std::string model_path = models_dir + "\\"  + m_shape_category + "_" + label_name + ".model";

		functions[idx] = boost::bind(&RandomForestsClassifier::load_one_classifier, this, idx, label, model_path);
		threads[idx] = boost::thread(functions[idx]);
	}
	/*Start the loading threads */
	for (int i = 0; i < num_classifiers(); i++)
		threads[i].join();

	/*std::string model_path = models_dir + "\\" + m_shape_category + "_body.model";
	load_one_classifier(0, 6, model_path);*/

	cout_mutex.lock();
	std::cout << "Classifier loading all done." << std::endl;
	cout_mutex.unlock();
}

void RandomForestsClassifier::test(const std::string test_file, const std::string models_dir, 
	const std::string label_output_path, Eigen::MatrixXd & prob_distribution)
{
	const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

	if (m_classifiers.empty())
		load_classifier_models(models_dir);

	shark::Data<shark::RealVector> test_data;
	int num_points = load_testing_data(test_file, test_data);

	if (num_points > 0)
	{
		std::cout << "Do classification." << std::endl;
		/* Create matrix stroring probability distribution of each point.
		* note that the number of columns is
		* 1 larger than label size, 'casue the last column represents null label*/
		prob_distribution = Eigen::MatrixXd::Zero(num_points, m_labels.size() + 1);

		int idx = 0;
		for (std::list<std::vector<int>>::iterator group_it = m_regions_symmetry.begin();
			group_it != m_regions_symmetry.end(); ++group_it, ++idx)
		{
			std::cout << idx << std::endl;
			shark::Data<shark::RealVector> prediction = m_classifiers[idx]->operator()(test_data);
			shark::Data<shark::RealVector>::element_range elements = prediction.elements();
			int point_idx = 0;
			for (PREDICT_Elements::iterator pos = elements.begin(); pos != elements.end(); ++pos, ++ point_idx)
			{
				shark::RealVector distribution = *pos;
				double negative_prob = distribution[0];
				double positive_prob = distribution[1];

				for (std::vector<int>::iterator sym_label_it = group_it->begin();
					sym_label_it != group_it->end(); ++sym_label_it)
					prob_distribution.row(point_idx)[*sym_label_it] = positive_prob;
			}
		}

		std::cout << "Classification done." << std::endl;

		/* Save the prediction result */
		std::ofstream predict_out(label_output_path.c_str());
		if (predict_out.is_open())
		{
			predict_out << prob_distribution.format(CSVFormat);
			predict_out.close();
		}
		else
			std::cerr << "Error: Saving predictions failed." << std::endl;
	}
	else
		std::cout << "Error: test dataset is empty." << std::endl;
}

void RandomForestsClassifier::test(const std::string test_file, const std::string models_dir, const std::string label_output_path, 
	Eigen::MatrixXd & prob_distribution, const std::string ground_truth_path, double & accuracy)
{
	const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

	if (m_classifiers.empty())
		load_classifier_models(models_dir);

	shark::Data<shark::RealVector> test_data;
	std::vector<int> ground_truth_labels;
	int num_points = load_testing_data(test_file, test_data, ground_truth_path, ground_truth_labels);

	if (num_points > 0)
	{
		std::cout << "Do classification." << std::endl;
		/* Create matrix stroring probability distribution of each point.
		* note that the number of columns is
		* 1 larger than label size, 'casue the last column represents null label*/
		prob_distribution = Eigen::MatrixXd::Zero(num_points, m_labels.size() + 1);

		int idx = 0;
		for (std::list<std::vector<int>>::iterator group_it = m_regions_symmetry.begin();
			group_it != m_regions_symmetry.end(); ++group_it, ++idx)
		{
			shark::Data<shark::RealVector> prediction = m_classifiers[idx]->operator()(test_data);
			shark::Data<shark::RealVector>::element_range elements = prediction.elements();
			int point_idx = 0;
			for (PREDICT_Elements::iterator pos = elements.begin(); pos != elements.end(); ++pos, ++point_idx)
			{
				shark::RealVector distribution = *pos;
				double negative_prob = distribution[0];
				double positive_prob = distribution[1];
				if (positive_prob < 1e-6)
					positive_prob = 0;

				for (std::vector<int>::iterator sym_label_it = group_it->begin();
					sym_label_it != group_it->end(); ++sym_label_it)
					prob_distribution.row(point_idx)[*sym_label_it] = positive_prob;
			}
		}

		std::cout << "Classification done." << std::endl;

		/* Validate the prediction result */
		accuracy = validate(prob_distribution, ground_truth_labels);
		std::cout << "Accuracy of the prediction is " << accuracy << std::endl;

		/* Save the prediction result */
		std::ofstream predict_out(label_output_path.c_str());
		if (predict_out.is_open())
		{
			predict_out << prob_distribution.format(CSVFormat);
			predict_out.close();
		}
		else
			std::cerr << "Error: Saving predictions failed." << std::endl;
	}
	else
		std::cout << "Error: test dataset is empty." << std::endl;
}

void RandomForestsClassifier::batch_test(const std::string test_list, const std::string models_dir, const std::string label_output_dir)
{
	const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

	if (m_classifiers.empty())
		load_classifier_models(models_dir);
	
	std::list<std::string> feats_file_list;
	std::ifstream list_in(test_list.c_str());
	if (list_in.is_open())
	{
		while (!list_in.eof())
		{
			std::string line;
			std::getline(list_in, line);
			if(line.length() > 0)
				feats_file_list.push_back(line);
		}

		list_in.close();
	}

	boost::filesystem::path boost_label_out_path(label_output_dir);
	if (!boost::filesystem::exists(boost_label_out_path))
	{
		boost::filesystem::create_directories(boost_label_out_path);
	}

	for (std::list<std::string>::iterator file_it = feats_file_list.begin(); file_it != feats_file_list.end(); ++file_it)
	{
		std::string feats_file_path = *file_it;

		boost::filesystem::path boost_file_path(feats_file_path);
		std::string shape_name = boost_file_path.stem().string();

		shark::Data<shark::RealVector> test_data;
		int num_points = load_testing_data(feats_file_path, test_data);

		if (num_points > 0)
		{
			std::cout << "Do classification for " <<m_shape_category << "_" << shape_name << std::endl;
			/* Create matrix stroring probability distribution of each point.
			* note that the number of columns is
			* 1 larger than label size, 'casue the last column represents null label*/
			Eigen::MatrixXd prob_distribution = Eigen::MatrixXd::Zero(num_points, m_labels.size() + 1);

			int idx = 0;
			for (std::list<std::vector<int>>::iterator group_it = m_regions_symmetry.begin();
				group_it != m_regions_symmetry.end(); ++group_it, ++idx)
			{
				shark::Data<shark::RealVector> prediction = m_classifiers[idx]->operator()(test_data);
				shark::Data<shark::RealVector>::element_range elements = prediction.elements();
				int point_idx = 0;
				for (PREDICT_Elements::iterator pos = elements.begin(); pos != elements.end(); ++pos, ++point_idx)
				{
					shark::RealVector distribution = *pos;
					double negative_prob = distribution[0];
					double positive_prob = distribution[1];
					if (positive_prob < 1e-6)
						positive_prob = 0;

					/*for (std::vector<int>::iterator sym_label_it = group_it->begin();
						sym_label_it != group_it->end(); ++sym_label_it)
						prob_distribution.row(point_idx)[*sym_label_it] = positive_prob;*/
					for (int sym_idx = 0; sym_idx < group_it->size(); ++sym_idx)
					{
						int sym_label = group_it->operator[](sym_idx);
						prob_distribution.row(point_idx)[sym_label] = positive_prob;
					}
				}
			}

			std::cout << "Classification for " << m_shape_category << "_" << shape_name << " has done." << std::endl;

			/* Save the prediction result */
			std::string label_output_path = label_output_dir + "\\" + shape_name + ".arff";
			std::ofstream predict_out(label_output_path.c_str());
			if (predict_out.is_open())
			{
				predict_out << prob_distribution.format(CSVFormat);
				predict_out.close();
			}
			else
				std::cerr << "Error: Saving predictions failed." << std::endl;
		}
	}
}

void RandomForestsClassifier::load_one_classifier(int id, int label, const std::string model_path)
{
	cout_mutex.lock();
	std::cout << "Loading random forests classifier for " << m_label_names[label] << std::endl;
	cout_mutex.unlock();

	/* get the pointer of corresponding classifier */
	classifier_mutex.lock();
	boost::shared_ptr<shark::RFClassifier> &classifier = m_classifiers[id];
	classifier_mutex.unlock();
	
	/* Load the classifier */
	if (!classifier)
		classifier.reset(new shark::RFClassifier());
	else
		classifier->clearModels();

	std::ifstream ifs(model_path.c_str());
	if (ifs.is_open())
	{
		boost::archive::polymorphic_text_iarchive ia(ifs);
		classifier->read(ia);
		ifs.close();
	}
	else
	{
		cout_mutex.lock();
		std::cerr << "Error: Loading classifier for " << m_label_names[label] << " failed!" << std::endl;
		cout_mutex.unlock();
	}

	cout_mutex.lock();
	std::cout << "Loading classifier for " << m_label_names[label] << " has done." << std::endl;
	cout_mutex.unlock();
}

double RandomForestsClassifier::validate(const Eigen::MatrixXd & prob_distribution, const std::vector<int>& ground_truth_labels)
{
	int correct_count = 0;

	for (int i = 0; i < prob_distribution.rows(); i++)
	{
		int predict_label;
		prob_distribution.row(i).maxCoeff(&predict_label);

		if (predict_label == ground_truth_labels[i])
			correct_count++;
	}

	double accuracy = (double)correct_count / (double)ground_truth_labels.size();

	return accuracy;
}

void RandomForestsClassifier::save_dataset(const std::string output_path, 
	std::vector<shark::RealVector>& data, std::vector<unsigned int>& labels)
{
	std::ofstream out(output_path.c_str());
	if (out.is_open())
	{
		std::vector<shark::RealVector>::iterator d_it;
		std::vector<unsigned int>::iterator l_it;
		for (d_it = data.begin(), l_it = labels.begin();
			d_it != data.end() && l_it != labels.end(); ++d_it, l_it++)
		{
			for (int i = 0; i < 36; i++)
				out << d_it->operator[](i) << ",";

			out << *l_it << std::endl;
		}
	
		out.close();
	}
}
