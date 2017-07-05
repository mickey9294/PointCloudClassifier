#ifndef RANDOMFORESTS_H
#define RANDOMFORESTS_H

#include "ClassifierModel.h"
#include <boost/thread.hpp>
#include <shark/Algorithms/Trainers/RFTrainer.h> //the random forest trainer

class RandomForestsClassifier :
	public ClassifierModel
{
public:
	RandomForestsClassifier(const std::string shape_category);
	~RandomForestsClassifier();

	boost::mutex classifier_mutex;

	void train(const std::string features_list_path, const std::string labels_dir, const std::string save_model_dir);

	virtual void load_classifier_models(const std::string models_dir);

	void test(const std::string test_file, const std::string models_dir, 
		const std::string label_output_path, Eigen::MatrixXd & prob_distribution);

	void test(const std::string test_file, const std::string models_dir, const std::string label_output_path, 
		Eigen::MatrixXd & prob_distribution, const std::string ground_truth_path, double &accuracy);

	void batch_test(const std::string test_list, const std::string models_dir, const std::string label_output_dir);

	void load_one_classifier(int id, int label, const std::string model_path);

private:
	std::string m_shape_category;
	std::vector<boost::shared_ptr<shark::RFClassifier>> m_classifiers;

	double validate(const Eigen::MatrixXd &prob_distribution, const std::vector<int> &ground_truth_labels);

	void save_dataset(const std::string output_path, std::vector<shark::RealVector> &data, std::vector<unsigned int> &labels);
};

#endif