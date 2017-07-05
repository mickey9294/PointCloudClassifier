#ifndef CLASSIFIERMODEL_H
#define CLASSIFIERMODEL_H

#include <shark/Data/Dataset.h>
#include <shark/Data/Csv.h>
#include "FeaturesLoader.h"
#include <assert.h>
#include <string>
#include <algorithm>

class ClassifierModel
{
public:
	ClassifierModel();
	virtual ~ClassifierModel();

	typedef shark::Data<shark::RealVector>::element_range PREDICT_Elements;
	typedef shark::Data<shark::RealVector>::const_element_reference PREDICT_ElementRef;

	static const int dimension = 36;

	virtual bool load_label_info(std::string regions_file, bool _verbose = false);

	virtual bool load_regions_symmetry(std::string regions_symmetry_file);

	virtual int load_training_data(const std::string features_list_path, const std::string labels_dir,
		shark::Data<shark::RealVector> &dataset, std::vector<int> &sample_labels);
	
	virtual int load_training_data(const std::string features_list_path, const std::string labels_dir,
		std::vector<shark::RealVector> &dataset, std::vector<int> &data_labels);

	virtual int load_testing_data(const std::string test_file_path, shark::Data<shark::RealVector> &test_data);

	virtual int load_testing_data(const std::string test_file_path, shark::Data<shark::RealVector> &test_data,
		const std::string ground_truth_path, std::vector<int> &ground_truth_labels);

	virtual void train(const std::string features_list_path, const std::string labels_dir, std::string save_model_dir) = 0;

	virtual void load_classifier_models(const std::string models_dir) = 0;

	virtual void test(const std::string test_file, const std::string models_dir, 
		const std::string label_output_path, Eigen::MatrixXd & prob_distribution) = 0;

	virtual void test(const std::string test_file, const std::string models_dir, const std::string label_output_path, 
		Eigen::MatrixXd & prob_distribution, const std::string ground_truth_path, double &accuracy) = 0;

	virtual void batch_test(const std::string test_list, const std::string models_dir, const std::string label_output_path) = 0;

	virtual int num_classifiers() { return m_regions_symmetry.size(); }

	virtual bool is_symmetry(int label_0, int label_1);

	std::vector<int> m_labels;
	std::vector<std::string> m_label_names;
	int m_null_label;
	std::list<std::vector<int>> m_regions_symmetry;

private:
	int get_label_from_label_name(std::string label_name);
};

#endif