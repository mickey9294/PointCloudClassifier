// VoxelsSampling.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


int main(int argc, char **argv)
{
	if (argc < 3)
	{
		std::cout << "Usage: ./" << argv[0] << ":\nvoxelized object file  shape_category\n";
	}
	else
	{
		std::string obj_path(argv[1]);
		std::string shape_category(argv[2]);

		Model3D m3d;
		std::cout << "Start to convert..." << std::endl;
		m3d.preprocess(obj_path, shape_category);
	}

	std::cout << "done." << std::endl;
	//system("PAUSE");
    return 0;
}

