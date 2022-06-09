// Definitions of cuda code for rotating image
// And definitions of wrapper functions that help
// bring cuda code into existing C++ application

// Include files for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_kernels.h"

#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <cmath>

void getError(cudaError_t err)
{
	if (err != cudaSuccess) {
		std::cout << "Error " << cudaGetErrorString(err) << std::endl;
	}
}

bool unifiedMemorySupported = false;


__global__ void blurMono8(uint8_t* input_image, uint8_t* output_image, int width, int height)
{
	const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int x = offset % width;
	int y = (offset - x) / width;
	int fsize = 10; // Filter size
	if (offset < width * height) {

		float output_gray = 0;
		int hits = 0;
		for (int ox = -fsize; ox < fsize + 1; ++ox) {
			for (int oy = -fsize; oy < fsize + 1; ++oy) {
				if ((x + ox) > -1 && (x + ox) < width && (y + oy) > -1 && (y + oy) < height) {
					const int currentoffset = (offset + ox + oy * width);
					output_gray += input_image[currentoffset];
					hits++;
				}
			}
		}
		output_image[offset] = output_gray / hits;
	}
}

// C++ wrapper functions
int CudaInitialize(int device)
{
	// check if cuda device exists
	cudaError_t cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << std::endl;
		return 1;
	}

	// see if unified memory is available
	int value = -1;
	cudaStatus = cudaDeviceGetAttribute(&value, cudaDevAttrManagedMemory, device);

	if (value == 1)
		unifiedMemorySupported = true;
	else if (value == 0)
		unifiedMemorySupported = false;
	else
	{
		std::cout << "Error reading cudaDevAttrManagedMemory: " << cudaStatus << std::endl;
		return 1;
	}

	if (unifiedMemorySupported)
		std::cout << "Unified Memory on GPU Supported" << std::endl;
	else
		std::cout << "Unified Memory on GPU Not Supported. Will use Pinned Memory on Host" << std::endl;

	return 0;
}

void CudaAllocateBuffer(void** pBuffer, int bufSize)
{
	if (unifiedMemorySupported)
		getError(cudaMallocManaged((void**)pBuffer, sizeof(uint8_t) * bufSize));
	else
		getError(cudaMallocHost((void**)pBuffer, sizeof(uint8_t) * bufSize));
}

void CudaFreeBuffer(void* pBuffer)
{
	if (unifiedMemorySupported)
		getError(cudaFree(pBuffer));
	else
		getError(cudaFreeHost(pBuffer));
}

void CudaBlurMono8(uint8_t* input, uint8_t* output, int width, int height)
{
	uint8_t* dev_out;
	getError(cudaMalloc((void**)&dev_out, sizeof(uint8_t) * width * height));

	dim3 blockDims(512, 1, 1);
	dim3 gridDims((unsigned int)ceil((double)(width * height / blockDims.x)), 1, 1);

	blurMono8 << <gridDims, blockDims >> > (input, dev_out, width, height);

	getError(cudaMemcpy(output, dev_out, sizeof(uint8_t) * width * height, cudaMemcpyDeviceToHost));

	cudaFree(dev_out);
}

// original sample image processing from https://github.com/madsravn/easyCuda/blob/master/kernels.cu
// if using this, mem sizes in wrapper functions and pBlurred in main() must be multipled by 3
/*
__global__ void blur(uint8_t* input_image, uint8_t* output_image, int width, int height)
{
	const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
	int x = offset % width;
	int y = (offset-x)/width;
	int fsize = 5; // Filter size
	if(offset < width*height) {

		float output_red = 0;
		float output_green = 0;
		float output_blue = 0;
		int hits = 0;
		for(int ox = -fsize; ox < fsize+1; ++ox) {
			for(int oy = -fsize; oy < fsize+1; ++oy) {
				if((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
					const int currentoffset = (offset+ox+oy*width)*3;
					output_red += input_image[currentoffset];
					output_green += input_image[currentoffset+1];
					output_blue += input_image[currentoffset+2];
					hits++;
				}
			}
		}
		output_image[offset] = output_red/hits;
		output_image[offset*3] = output_red/hits;
		output_image[offset*3+1] = output_green/hits;
		output_image[offset*3+2] = output_blue/hits;
		}
}
*/