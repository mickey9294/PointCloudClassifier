#pragma once

#include <Eigen/Core>
#include <Eigen\Geometry>
#include <fstream>
#include <cmath>
#include "igl/readOBJ.h"
#include "igl/writeOBJ.h"
#include "igl\readOFF.h"
#include "igl\writeOFF.h"
#include <set>
#include <string>
#include <vector>
#include "Voxel.h"
#include <boost\filesystem.hpp>
#include <ANN\ANN.h>
#include <unordered_map>
#include "PointsSampler.h"

//#define VOXEL_LENGTH 32
#define QUITE_SMALL 0.00001
#define ALMOST_EQUAL(x, y) ((x)-(y))<QUITE_SMALL

using namespace std;

class Model3D
{
public:
	Model3D();

	void preprocess(const string& filename, const string &shape_category);

private:
	bool readFromOBJ(string filenamem, Eigen::MatrixXd& V, Eigen::MatrixXi& F) const;

	void get_surface_faces(Eigen::MatrixXd &V, Eigen::MatrixXi &F, 
		Eigen::MatrixXd &surface_V, Eigen::MatrixXi &surface_F);
	void sample_on_surface_mesh(Eigen::MatrixXd &surface_V, Eigen::MatrixXi &surface_F);

	void rotate(Eigen::MatrixXd &surface_V);

	string shape_name_;
	string shape_category_;
};
