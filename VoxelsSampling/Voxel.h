#pragma once

enum VOXEL_LABEL{ OUTSIDE, BOUNDRY, INSIDE };

struct Voxel
{
	Voxel()
	{
		voxelLabel = OUTSIDE;
	}

	VOXEL_LABEL voxelLabel;
};