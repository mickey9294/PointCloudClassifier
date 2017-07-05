#include <iostream>
#include "PointsSampler.h"

int main(int argc, char **argv)
{
	if (argc < 7)
	{
		std::cout << "Usage: ./" << argv[0] << " shapes_list_file  samples_directory  dense_samples_directory  num_of_samples\n";
		std::cout << "\t\tnum_of_dense_samples  modified_shapes_directory  [input_labels_directory]  [output_labels_directory]" << std::endl;
	}
	else
	{
		std::string shapes_list_file(argv[1]);
		std::string samples_dir(argv[2]);
		std::string dense_samples_dir(argv[3]);
		int num_samples = std::atoi(argv[4]);
		int num_dense_samples = std::atoi(argv[5]);
		std::string modified_shapes_dir(argv[6]);
		std::string labels_dir, out_labels_dir;
		if (argc == 9)
		{
			labels_dir = std::string(argv[7]);
			out_labels_dir = std::string(argv[8]);
		}

		PointsSampler sampler;
		sampler.batch_sample(shapes_list_file, samples_dir, dense_samples_dir, num_samples,
			num_dense_samples, modified_shapes_dir, labels_dir, out_labels_dir);
	}

	return 0;
}