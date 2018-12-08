#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif // !(__CUDACC_RTC__)

#include <device_functions.h>


__global__
void Kernels(float* x,float* y,float* CenterX, float* CenterY,float* outputKG,float* outputKC,float* MainOutput,int N,int sampleIndex,float alpha1,float alpha2)
{
	float intputsq = x[sampleIndex] * x[sampleIndex] + y[sampleIndex] * y[sampleIndex];
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	//printf("x= %f,  y=%f  \n", x[0], y[0]);
	if (i < N)
	{
		outputKG[i] = exp(-(pow((x[sampleIndex] - CenterX[i]), 2) + pow((y[sampleIndex] - CenterY[i]), 2)) / 0.04);
		outputKC[i] = (x[sampleIndex] * CenterX[i] + y[sampleIndex] * CenterY[i]) / (sqrt((pow(CenterX[i], 2) + pow(CenterY[i], 2))*intputsq) + 0.0000000000000001);
		MainOutput[i] = (alpha1*outputKC[i] + alpha2*outputKG[i]) / (alpha1 + alpha2);
	}
	
	
}
__global__
void OutputNeuron(float* Kernel,float* w, float* output,int N)
{
	__shared__ float temp[121];
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N)
	{
		temp[i] = w[i] * Kernel[i];
	//	printf("%d: w=%f,  K=%f,  Ans=%f\n", i,w[i],Kernel[i],temp[i]);
		__syncthreads();
		if (threadIdx.x == 0)
		{
			float sum = 0.0;
			for (int j = 0; j < N; j++)
			{
				sum += temp[j];
				
			}
			output[0] = sum;
			//printf("%f\n", output[0]);
		}
	}
	

}

__global__
void Multiplication(float* Kernel, float* w, float error, float learningRate, int N)
{
		
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	
	if (i < N)
	{
		w[i] = w[i] + (error*learningRate*Kernel[i]);
		//printf("%d w: %f\n", i, Kernel[i]);

	}
}
////////////////////////////////////////////////////////////////////////////
// First For Loop
__global__
void train_1(int n, float alp1, float alp2,  float *KC, float *KG, float *w)
{
   __shared__ float output[N];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n)
	{
	  	  
	RBF_out[i] = (alp1*KC[i] + alp2*KG[i])/(alp1+alp2);
	output[i] = RBF_out[i]*w[i];
	
  }
   __syncthreads();
	d_output = d_output+ output[i];
}

int N=121;
float *KC, *KG, *d_KC, *d_KG;
cudaMalloc(&d_KG, N * sizeof(float));
cudaMalloc(&d_KC, N * sizeof(float));

cudaMemcpy(d_KC, KC, N*sizeof(float), cudaMemcpyHostToDevice); // Copy Input
cudaMemcpy(d_KG, KG, N*sizeof(float), cudaMemcpyHostToDevice;// Copy Input

train_1<<<(N)/256, 256>>>(N, alp1, alp2, d_KC, d_KG); // Launch Statment

cudaMemcpy(y, d_output, N*sizeof(float), cudaMemcpyDeviceToHost); // Copy Output

__global__
void AlphaUpdate(float* KC, float* KG, float* w, float* updateAlpha1, float* updateAlpha2)
{
	__shared__ float temp1[121];
	__shared__ float temp2[121];

	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i < 121)
	{
		//printf("%d:  KC: %f,  KG:%f\n", i,KC[i], KG[i]);
		temp1[i] = w[i] * (KC[i] - KG[i]);
	
		temp2[i]= w[i] * (KG[i] - KC[i]);
		
		__syncthreads();

		if (threadIdx.x == 0)
		{
			float sum1 = 0.0;
			float sum2 = 0.0;
			for (int j = 0; j < 121; j++)
			{
				sum1 += temp1[j];
				sum2 += temp2[j];
			}
			updateAlpha1[0] = sum1;
			updateAlpha2[0] = sum2;
		//	printf("alpha1: %f\n", updateAlpha1[0]);
		}
		//if (threadIdx.x == 1)
		//{
		//	float sum2 = 0.0;
		//	for (int j = 0; j < 121; j++)
		//	{
		//		sum2 += temp2[j];
		//	}
		//	updateAlpha2[0] = sum2;
		//	//printf("alpha2: %f\n", updateAlpha2[0]);
		//}
		
	}

	
}




void CosinKernel(float x, float y, int CenterR, int CenterC, float Centers[][121],float* output);
void GaussianKernal(float x, float y, int CenterR, int CenterC, float Centers[][121], float* output);
int main()
{
	const int N = 121;
	size_t size = N * sizeof(float);

	//float train[121][2];
	float trainx[121];
	float trainy[121];
	float trainOutput[121];
	//float Centers[2][121] = { { -1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,-1,-0.8,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1 },
		//					  { -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.4,-0.4,-0.4,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,0,0,0,0,0,0,0,0,0,0,0,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,1,1,1,1,1,1,1,1,1,1,1 } };

	float Centerx[121] = { -1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,-1,-0.8,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1 };
	float Centery[121] = { -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.4,-0.4,-0.4,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,0,0,0,0,0,0,0,0,0,0,0,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,1,1,1,1,1,1,1,1,1,1,1 };

	float data[11];
	data[0] = -1.0;


	for (int i = 1; i < 11; i++)
	{
		data[i] = data[i - 1] + 0.2;
		//printf("%f\n", data[i]);
	}
	int count = 0;
	printf("Start ....! \n\n\n");
	for (int i = 0; i < 11; i++)
	{
		for (int j = 0; j < 11; j++)
		{
			trainx[count] = data[j];
			trainy[count] = data[i];
			trainOutput[count] = exp(-pow(trainx[count], 2) - trainy[count]);
			//printf("%.2f, %.2f %.2f\n", train[count][0], train[count][1], trainOutput[count]);
			count++;
		}
	}

	float alpha1 = 0.5;
	float alpha2 = 0.5;
	float outputNeuron = 1;

	float b = 0.0;
	float error;
	float learningRate = 0.001;
	int Epoch = 10000;
	float *h_alpha1Update, *h_alpha2Update;
	float *KC, *KG, *w, *RBFoutput, *h_output;
	KC = (float *)malloc(size);
	KG = (float *)malloc(size);
	w = (float *)malloc(size);
	RBFoutput = (float *)malloc(size);
	h_output = (float *)malloc(sizeof(float));
	h_alpha1Update = (float *)malloc(sizeof(float));
	h_alpha2Update = (float *)malloc(sizeof(float));
	for (int i = 0; i < N; i++)
		w[i] = 0;
	float finalOutput = 0.0;


	float *d_x, *d_y, *d_Cx, *d_Cy, *d1_o, *d2_o, *MainKout, *d_w, *d_output, *d_RBFin, *d_KC, *d_KG, *d_alpha1Update, *d_alpha2Update;
	cudaMalloc(&d_x, size);
	cudaMalloc(&d_y, size);
	cudaMalloc(&d_Cx, size);
	cudaMalloc(&d_Cy, size);
	cudaMalloc(&d1_o, size);
	cudaMalloc(&d2_o, size);
	cudaMalloc(&MainKout, size);

	cudaMalloc(&d_w, size);
	cudaMalloc(&d_RBFin, size);
	cudaMalloc(&d_output, sizeof(float));
	cudaMalloc(&d_KC, size);
	cudaMalloc(&d_KG, size);


	cudaMalloc(&d_alpha1Update, sizeof(float));
	cudaMalloc(&d_alpha2Update, sizeof(float));

	cudaMemcpy(d_x, trainx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, trainy, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Cx, Centerx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Cy, Centery, size, cudaMemcpyHostToDevice);

	dim3 DimGrid(1, 1, 1);
	dim3 DimBlock(1024, 1, 1);
	printf("launcing kernel\n");
	for (int Epoch = 0; Epoch <= 1; Epoch++)
	{
		for (int sample = 0; sample < 1; sample++)
		{
			Kernels << <DimGrid, DimBlock >> >(d_x, d_y, d_Cx, d_Cy, d1_o, d2_o, MainKout, N, sample, alpha1, alpha2);
		
			cudaMemcpy(KG, d1_o, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(KC, d2_o, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(RBFoutput, MainKout, size, cudaMemcpyDeviceToHost);



			cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_RBFin, RBFoutput, size, cudaMemcpyHostToDevice);
			OutputNeuron << <DimGrid, DimBlock >> >(d_RBFin, d_w, d_output, N);

			cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
			h_output[0] = h_output[0] + b;

			error = trainOutput[sample] - h_output[0];
			b = b + learningRate*error;


			/////////////////////////////////////////////////////////////////

			Multiplication << <DimGrid, DimBlock >> >(d_RBFin, d_w, error, learningRate, N);
			cudaMemcpy(w, d_w, size, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_KC, KC, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_KG, KG, size, cudaMemcpyHostToDevice);

			AlphaUpdate << <DimGrid, DimBlock >> >(d_KC, d_KG, d_w, d_alpha1Update, d_alpha2Update);

			cudaMemcpy(h_alpha1Update, d_alpha1Update, sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_alpha2Update, d_alpha2Update, sizeof(float), cudaMemcpyDeviceToHost);

			h_alpha1Update[0] = error*h_alpha1Update[0] * ((alpha1*alpha2) / (alpha1*pow(alpha1 + alpha2, 2)));
			h_alpha2Update[0] = error*h_alpha2Update[0] * ((alpha1*alpha2) / (alpha2*pow(alpha1 + alpha2, 2)));
			alpha1 = alpha1 + learningRate*h_alpha1Update[0];
			alpha2 = alpha2 + learningRate*h_alpha2Update[0];
			alpha1 = alpha1 / (alpha1 + alpha2);
			alpha2 = alpha2 / (alpha1 + alpha2);

			/*for (int i = 0; i < N; i++)
			{
				printf("W: %f\n", w[i]);
			}
			*/

			

		}
		printf("Epoch: %d\n", Epoch);
	}
	printf("Alpha1: %f , Alpha2: %f\n", alpha1, alpha2);

	
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_Cx); 
	cudaFree(d_Cy);
	cudaFree(d1_o);
	cudaFree(d2_o);
	cudaFree(MainKout);
	cudaFree(d_w);
	cudaFree(d_RBFin);
	cudaFree(d_output);
	return 0;
}



void GaussianKernal(float x, float y,  int CenterR, int CenterC, float Centers[][121],float* output)
{
	
	
//	printf("Gauss Kernel\n\n\n");
	for (int i = 0; i < 121; i++)
	{
		output[i] = exp(-(pow((x - Centers[0][i]), 2) + pow((y - Centers[1][i]), 2))/0.04);
	//	printf("%f\n", output[i]);
	}

}
void CosinKernel(float x,float y, int CenterR, int CenterC,float Centers[][121] , float* output)
{
//	printf("Cosine Kernel\n");
//	

	//float output[121];
	float sumCenter[121];
	float intputsq=x*x +y*y;
//	printf("\nMultiplication Kernel\n\n\n");
	
	for (int i = 0; i < 121; i++)
	{
		float sum = 0.0;
		sum = x * Centers[0][i]+ y * Centers[1][i];
		output[i] = sum;
		
		sumCenter[i] = sqrt((pow(Centers[0][i], 2) + pow(Centers[1][i], 2))*intputsq);
		output[i] = output[i] / (sumCenter[i]+0.0000000000000001);
		//printf("%f\n", output[i]);
	}
	
}
