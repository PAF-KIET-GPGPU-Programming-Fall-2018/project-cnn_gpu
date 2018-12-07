
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


__global__
void Gauss(float* x,float* y,float* CenterX, float* CenterY,float* output,int N)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	//printf("x= %f,  y=%f  \n", x[0], y[0]);
	if (i<N)
	output[i] = exp(-(pow((x[0]- CenterX[i]), 2) + pow((y[0] - CenterY[i]), 2)) / 0.04);
	printf("%d: %f\n", i, output[i]);
	
}
__global__
void Coss(float* x, float* y, float* CenterX, float* CenterY, float* output, int N)
{
	float sumCenter;
	float intputsq = x[0]*x[0] + y[0]*y[0];
		//printf("%f", intputsq);

	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N)
	{
		
		output[i] = (x[0] * CenterX[i] + y[0] * CenterY[i]) / (sqrt((pow(CenterX[i], 2) + pow(CenterY[i], 2))*intputsq) + 0.0000000000000001);
	
	}
}
////////////////////////////////////////////////////////////////////////////
// First For Loop
__global__
void train_1(int n, float alp1, float alp2,  float *KC, float *KG, float *w)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n)
	{
	  
	  
	RBF_out[i] = (alp1*KC[i] + alp2*KG[i])/(alp1+alp2);
	d_output += RBF_out[i]*w[i];
	
  }
}

int N=121;
float *KC, *KG, *d_KC, *d_KG;
cudaMemcpy(d_KC, KC, N*sizeof(float), cudaMemcpyHostToDevice); // Copy Input
cudaMemcpy(d_KG, KG, N*sizeof(float), cudaMemcpyHostToDevice;// Copy Input

train_1<<<(N)/256, 256>>>(N, alp1, alp2, d_KC, d_KG); // Launch Statment

cudaMemcpy(y, d_output, N*sizeof(float), cudaMemcpyDeviceToHost); // Copy Output

cudaFree(d_KC);
cudaFree(d_KG);
cudaFree(d_output);	  
////////////////////////////////////////////////////////////////////////////	   
// Kernel
__global__
void train2(int n, float l_rate, float error,  float *RBF_out, float *w)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n)
	{

	  
	w[i] = w[i] + l_rate*error*RBF_out[i];
	
  }
}
int N=121;
float *w, *RBFoutput, *d_RBFoutput, *d_w;

cudaMemcpy(d_KC, KC, N*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_KG, KG, N*sizeof(float), cudaMemcpyHostToDevice);
	   
train2<<<(N)/256, 256>>>(N, learningRate, error, d_RBFoutput, d_w);
	   
cudaMemcpy(w, d_w, N*sizeof(float), cudaMemcpyDeviceToHost);

	   
cudaFree(d_w);
cudaFree(d_RBF_output);
cudaFree(d_w);
////////////////////////////////////////////////////////////////////////////

__global__
void train3(int n, float alp1_new, float *KC,  float *KG, float *w)
{
	
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n)
	{

	  alp1_new += (w[i] * (KC[i] - KG[i]));
	
  	}
}

float *w, *d_KC, *d_KG, *d_w;

cudaMemcpy(d_KC, KC, N*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_KG, KG, N*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_w, w, N*sizeof(float), cudaMemcpyHostToDevice);
	   
train3<<<(N)/256, 256>>>(N, alp1_new, d_KC, d_KG, d_w);
	   
cudaMemcpy(alphaUpdate, d_w, N*sizeof(float), cudaMemcpyDeviceToHost);

cudaFree(d_w);
cudaFree(d_KG);
cudaFree(d_KC);
	   

	   
void training();
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

	float *KC, *KG,*w;
	KC = (float *)malloc(size);
	KG = (float *)malloc(size);
	w = (float *)malloc(size);
	float RBFoutput[121];
	float finalOutput = 0.0;
	float alpha1Update = 0.0, alpha2Update = 0.0;

	float *d_x, *d_y, *d_Cx, *d_Cy, *d_o;
	cudaMalloc(&d_x, size);
	cudaMalloc(&d_y, size);
	cudaMalloc(&d_Cx, size);
	cudaMalloc(&d_Cy, size);
	cudaMalloc(&d_o, size);

	cudaMemcpy(d_x, trainx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, trainy, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Cx, Centerx, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Cy, Centery, size, cudaMemcpyHostToDevice);
	dim3 DimGrid(1, 1, 1);
	dim3 DimBlock(256, 1, 1);
	printf("launcing kernel");

	Gauss <<<DimGrid, DimBlock >>>(d_x, d_y, d_Cx, d_Cy, d_o, N);
	cudaMemcpy(KG, d_o, size, cudaMemcpyDeviceToHost);
	Coss << <DimGrid, DimBlock >> >(d_x, d_y, d_Cx, d_Cy, d_o, N);
	cudaThreadSynchronize();
	cudaMemcpy(KC, d_o, size, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < 121; i++)
	{
		printf("Coss: %f,  Gauss: %f\n", KC[i],KG[i]);
	}
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_Cx); cudaFree(d_Cy); cudaFree(d_o);
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
