/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#define BLOCK_SIZE 512
#define SECTION_SIZE 1024  // define section size (size of subarray to be handled) to be twice the block size

// Define your kernels in this file you may use more than one kernel if you need to

// INSERT KERNEL(S) HERE
__global__ void work_efficient_inclusive_scan(float *X, float *Y, unsigned in_size) {
	__shared__ float XY[SECTION_SIZE];

	// Load elements from input into in-place array
	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * blockDim.x;

	// Each thread loads 2 elements, since section size is double block size
	if(t + start < in_size) {
		XY[t] = X[start + t];
	}

	if(t + start + BLOCK_SIZE < in_size) {
		XY[t + BLOCK_SIZE] = X[start + t + BLOCK_SIZE];
	}

	// Reduction tree step (increase stride size)
	for(unsigned int stride = 1; stride <= BLOCK_SIZE; stride *=2) {
		__syncthreads();
		unsigned int i = (threadIdx.x+1) * 2 * stride - 1;
		if(i < SECTION_SIZE) {
			XY[i] += XY[i-stride];
		}
	}

	// Distribution tree step (decrease stride size)
	for(unsigned int stride = SECTION_SIZE/4; stride>0; stride/=2) {
		__syncthreads();
		unsigned int i = (threadIdx.x+1) * 2 * stride - 1;
		if(i + stride < SECTION_SIZE) {
			XY[i + stride] += XY[i];
		}
	}

	__syncthreads();
	// Cp threads to output array
	t = threadIdx.x;

	if(t + start < in_size) {
		Y[start + t] = XY[t];
	}

	if(t + start < in_size && t + BLOCK_SIZE < in_size) {
		Y[start + t + BLOCK_SIZE] = XY[t + BLOCK_SIZE];
	}
}

/*
 * Note: this kernel is based off the assumption that the GRID_DIM is 1024, or exactly
 * twice the BLOCK_DIM. This way, one thread block in this prefix-scan stage will be able to
 * exactly handle all the block outputs from the previous prefix-scan stage.
 */
__global__ void work_efficient_inclusive_scan_2(float *X, float *Y, unsigned in_size) {
	__shared__ float XY[SECTION_SIZE];

	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
	// Each thread loads 2 elements, each element being the last element of every SECTION from last kernel
	if(SECTION_SIZE * (t+1) - 1 < in_size) {
		XY[t] = X[SECTION_SIZE * (t+1) - 1];
	}

	if(SECTION_SIZE * (t+BLOCK_SIZE+1) - 1 < in_size) {
		XY[t+BLOCK_SIZE] = X[SECTION_SIZE * (t+BLOCK_SIZE+1) - 1];
	}

	// Reduction tree step (increase stride size)
	for(unsigned int stride = 1; stride <= BLOCK_SIZE; stride *=2) {
		__syncthreads();
		unsigned int i = (threadIdx.x+1) * 2 * stride - 1;
		if(i < SECTION_SIZE) {
			XY[i] += XY[i-stride];
		}
	}

	// Distribution tree step (decrease stride size)
	for(unsigned int stride = SECTION_SIZE/4; stride>0; stride/=2) {
		__syncthreads();
		unsigned int i = (threadIdx.x+1) * 2 * stride - 1;
		if(i + stride < SECTION_SIZE) {
			XY[i + stride] += XY[i];
		}
	}

	__syncthreads();
	// Cp threads to output array
	t = threadIdx.x;
	if(t < in_size) {
		Y[t] = XY[t];
	}

	if(t+BLOCK_SIZE < in_size) {
		Y[t+BLOCK_SIZE] = XY[t+BLOCK_SIZE];
	}
}

__global__ void work_efficient_inclusive_scan_3(float *X2, float *X, float *Y, unsigned in_size) {

	unsigned int t = threadIdx.x;

	// Cp threads to output array (each thread copies 2 elements and add result from prev kernel
	if(start != 0) {  // Do for blocks 1 onwards
		if(start + t < in_size) {
			Y[start + t] = X2[start + t] + X[blockIdx.x - 1];
		}

		if(start + t + BLOCK_SIZE < in_size) {
			Y[start + t + BLOCK_SIZE] = X2[start + t + BLOCK_SIZE] + X[blockIdx.x - 1];
		}

	} else {
		if(start + t < in_size) {
			Y[start + t] = X2[start + t];
		}

		if(start + t + BLOCK_SIZE) {
			Y[start + t + BLOCK_SIZE] = X2[start + t + BLOCK_SIZE];
		}
	}

}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out2, float *in, unsigned in_size) {
    // INSERT CODE HERE
	dim3 DimGrid((in_size-1)/(BLOCK_SIZE*2)+1, 1, 1);
	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	work_efficient_inclusive_scan<<<DimGrid, DimBlock>>>(in, out2, in_size);
}

void preScan2(float *out3, float *out2, unsigned in_size) {
    // INSERT CODE HERE
	dim3 DimGrid(1, 1, 1);
	dim3 DimBlock(BLOCK_SIZE, 1, 1);

	work_efficient_inclusive_scan_2<<<DimGrid, DimBlock>>>(out2, out3, in_size);
}

void preScan3(float *out, float *out3, float *out2, unsigned in_size) {
    // INSERT CODE HERE
	dim3 DimGrid((in_size-1)/(BLOCK_SIZE*2)+1, 1, 1);
	dim3 DimBlock(BLOCK_SIZE, 1, 1);

	work_efficient_inclusive_scan_3<<<DimGrid, DimBlock>>>(out2, out3, out, in_size);
}
