// PointClassifier.cpp : Defines the entry point for the console application.
#include <dlib/svm.h>
#include "RandomForestsClassifier.h"
#include <boost/shared_ptr.hpp>

int main(int argc, char **argv)
{
	if (argc < 6)
	{
		std::cout << "Usage: ./" << argv[0] << ":\ntrain shape_category_name features_list_file   labels_directory  saving_model_dir\n";
		std::cout << "\tor" << std::endl;
		std::cout << "test  shape_category  test_features_file  models_dir  label_output_path  [ground_truth_path]" << std::endl;
		std::cout << "\tor" << std::endl;
		std::cout << "batch_test  shape_category  test_list_file  models_dir  label_output_directory" << std::endl;
	}
	else
	{
		std::string action(argv[1]);
		std::string shape_category(argv[2]);
		boost::shared_ptr<ClassifierModel> classifier(new RandomForestsClassifier(shape_category));

		if (action.compare("train") == 0)
		{
			std::string features_list_file(argv[3]);
			std::string labels_dir(argv[4]);
			std::string saving_model_dir(argv[5]);

			classifier->train(features_list_file, labels_dir, saving_model_dir);
		}
		else if (action.compare("test") == 0)
		{
			std::string test_file(argv[3]);
			std::string models_dir(argv[4]);
			std::string label_output_path(argv[5]);
			
			Eigen::MatrixXd prob_distribution;

			if (argc == 6)
			{
				classifier->test(test_file, models_dir, label_output_path, prob_distribution);
			}
			else if (argc == 7)
			{
				std::string ground_truth_path(argv[6]);
				double accuracy = 0;
				classifier->test(test_file, models_dir, label_output_path, prob_distribution, ground_truth_path, accuracy);
			}
		}
		else if (action.compare("batch_test") == 0)
		{
			std::string test_list(argv[3]);
			std::string models_dir(argv[4]);
			std::string label_output_dir(argv[5]);

			classifier->batch_test(test_list, models_dir, label_output_dir);
		}
	}
	
    return 0;
}

