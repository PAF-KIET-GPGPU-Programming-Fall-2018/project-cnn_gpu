
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main()
{
	const int arraysize = 121;
	float train[121][2];
	float trainOutput[121];
	//float Centers[][] = { { -1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1 }
	//{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-0.800000000000000,-0.800000000000000,-0.800000000000000,-0.800000000000000,-0.800000000000000,-0.800000000000000,-0.800000000000000,-0.800000000000000,-0.800000000000000,-0.800000000000000,-0.800000000000000,-0.600000000000000,-0.600000000000000,-0.600000000000000,-0.600000000000000,-0.600000000000000,-0.600000000000000,-0.600000000000000,-0.600000000000000,-0.600000000000000,-0.600000000000000,-0.600000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,0,0,0,0,0,0,0,0,0,0,0,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,1,1,1,1,1,1,1,1,1,1,1} };
	//float train[11];
	float data[11];
	data[0] = -1.0;
	for (int i = 1; i < 11; i++)
	{
		data[i] = data[i - 1] + 0.2;
		printf("%f\n", data[i]);
	}
	int count = 0;
	for (int i = 0; i < 11; i++)
	{
		for (int j = 0; j < 11; j++)
		{
			train[count][0] = data[j];
			train[count][1] = data[i];
			trainOutput[count] = exp(-pow(train[count][0], 2) - train[count][1]);
			printf("%.2f, %.2f %.2f\n", train[count][0], train[count][1], trainOutput[count]);
			count++;
		}
	}
	






	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
