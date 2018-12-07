
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
void training();
void CosinKernel(float x, float y, int CenterR, int CenterC, float Centers[][121],float* output);
void GaussianKernal(float x, float y, int CenterR, int CenterC, float Centers[][121], float* output);
int main()
{
	const int N = 121;
	size_t size = N * sizeof(float);
	
	float train[121][2];
	float trainOutput[121];
	float Centers[2][121] = { { -1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,-1,-0.8,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1,-1,-0.800000000000000,-0.600000000000000,-0.400000000000000,-0.200000000000000,0,0.200000000000000,0.400000000000000,0.600000000000000,0.800000000000000,1 },
							  { -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.4,-0.4,-0.4,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.400000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,-0.200000000000000,0,0,0,0,0,0,0,0,0,0,0,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.200000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.400000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.600000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,0.800000000000000,1,1,1,1,1,1,1,1,1,1,1 } };
	float data[11];
	data[0] = -1.0;
	float alpha1 = 0.5;
	float alpha2 = 0.5;
	float outputNeuron = 1;
	float w[121];
	float b = 0.0;
	float error;
	float learningRate = 0.001;
	int Epoch = 10000;
	for (int i = 0; i < 121; i++)
	{
		w[i] = 0.0;
	}
	for (int i = 1; i < 11; i++)
	{
		data[i] = data[i - 1] + 0.2;
		//printf("%f\n", data[i]);
	}
	int count = 0;
	for (int i = 0; i < 11; i++)
	{
		for (int j = 0; j < 11; j++)
		{
			train[count][0] = data[j];
			train[count][1] = data[i];
			trainOutput[count] = exp(-pow(train[count][0], 2) - train[count][1]);
			//printf("%.2f, %.2f %.2f\n", train[count][0], train[count][1], trainOutput[count]);
			count++;
		}
	}
	
	float *KC,*KG;

	KC = (float *)malloc(size);
	KG = (float *)malloc(size);
	float RBFoutput[121];
	float finalOutput = 0.0;
	for (int Epoch = 0; Epoch < 10000; Epoch++)
	{
		for (int sample = 0; sample < 121; sample++)
		{
			CosinKernel(train[sample][0], train[sample][1], 2, 121, Centers, KC);
			GaussianKernal(train[sample][0], train[sample][1], 2, 121, Centers, KG);
			finalOutput = 0.0;
			for (int i = 0; i < 121; i++)
			{
				RBFoutput[i] = (alpha1*KC[i] + alpha2*KG[i]) / (alpha1 + alpha2);
				finalOutput += RBFoutput[i] * w[i] + b;

			}
		//	printf("Final: %f\n", finalOutput);
			error = trainOutput[sample] - finalOutput;
		//	printf("Error: %f\n", error);
			b = b + learningRate*error;
		//	printf("b: %f\n\n", b);
			for (int i = 0; i < 121; i++)
			{
				w[i] = w[i] + learningRate*error*RBFoutput[i];
				//printf("W: %f\n", w[i]);
			}
			float alpha1Update = 0.0, alpha2Update = 0.0;
			for (int i = 0; i < 121; i++)
			{
				alpha1Update += (w[i] * (KC[i] - KG[i]));//l*((alpha1*alpha2) / (alpha1*pow(alpha1 + alpha2, 2)));

			}
			//printf("Alpha1UP: %f\n", alpha1Update);
			for (int i = 0; i < 121; i++)
			{
				alpha2Update += (w[i] * (KG[i] - KC[i]));//l*((alpha1*alpha2) / (alpha1*pow(alpha1 + alpha2, 2)));

			}
			//printf("Alpha2UP: %f\n", alpha2Update);
			alpha1Update = error*alpha1Update*((alpha1*alpha2) / (alpha1*pow(alpha1 + alpha2, 2)));



			alpha2Update = error*alpha2Update*((alpha1*alpha2) / (alpha2*pow(alpha1 + alpha2, 2)));
			alpha1 = alpha1 + learningRate*alpha1Update;
			alpha2 = alpha2 + learningRate*alpha2Update;
			alpha1 = alpha1 / (alpha1 + alpha2);
			alpha2 = alpha2 / (alpha1 + alpha2);
			
		}
		printf("Alpha1: %f\n", alpha1);
		printf("Alpha2: %f\n", alpha2);
	}
	return 0;
}



void GaussianKernal(float x, float y,  int CenterR, int CenterC, float Centers[][121],float* output)
{
	
	float sigma = 0.04;
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