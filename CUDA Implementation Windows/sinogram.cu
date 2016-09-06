// To Compile: nvcc sinogram.cu -o sinogram.out
// To Run: ./sinogram.out numAngles theta.txt minR maxR numSensors sg.txt minX maxX numX minY maxY numY img.txt
// ./sinogram.out 360 theta.txt -53.2 53.2 2129 sg.txt -44.25 44.25 886 -29.5 29.5 591 img.txt

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define pi acos(-1)

// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Simple transformation kernel
__global__ void rotationKernel(float* d_sg, float dx, float minX, float dy, float minY, int numSensors, float dr, float minR, int numAngles, float* theta)
{
	// Calculate normalized texture coordinates
	unsigned int r_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int t_idx = blockIdx.y * blockDim.y + threadIdx.y;

	if (r_idx < numSensors && t_idx < numAngles) {

		float integral = 0;
		float r = r_idx * dr + minR;

		for (int z_idx = 0; z_idx < numSensors; z_idx++) {
			float z = z_idx * dr + minR;
			// Transform coordinates
			float tr = (r * cosf(theta[t_idx]) + z * sinf(theta[t_idx]) - minX)/dx + 0.5f;
			float tz = (z * cosf(theta[t_idx]) - r * sinf(theta[t_idx]) - minY)/dy + 0.5f;
			// Read from texture and write to global memory
			integral += tex2D(texRef, tr, tz);
		}

		d_sg[t_idx*numSensors + r_idx] = integral;
	}
}

// Host code
int main(int argc, char *argv[])
{
	// Get all values from inputs
	int numAngles = strtol(argv[1], NULL, 10);
	float minR = (float) atof(argv[3]);
	float maxR = (float) atof(argv[4]);
	int numSensors = strtol(argv[5], NULL, 10);
	float minX = (float) atof(argv[7]);
	float maxX = (float) atof(argv[8]);
	int numX = strtol(argv[9], NULL, 10);
	float minY = (float) atof(argv[10]);
	float maxY = (float) atof(argv[11]);
	int numY = strtol(argv[12], NULL, 10);

	// Calculate some other values based on those inputs
	float dr = (maxR - minR) / ((float)(numSensors - 1));
	float dx = (maxX - minX) / ((float)(numX - 1));
	float dy = (maxY - minY) / ((float)(numY - 1));

	// Read Image and Projection Angles from File
	float datfromfile, *h_img, *h_angles;
	h_img = (float *)malloc(numX*numY*sizeof(float));
	h_angles = (float *)malloc(numAngles*sizeof(float));
	FILE *in_img = fopen(argv[13], "r");
	FILE *in_theta = fopen(argv[2], "r");
	if (in_img == NULL) { fprintf(stderr, "Input file for image info has some issues. Please check."); exit(1); }
	for (int j = 0; j < numY; j++) {
		for (int i = 0; i < numX; i++) {
			fscanf(in_img, "%f", &datfromfile);
			h_img[i + j*numX] = datfromfile;
		}
	}
	for (int i = 0; i < numAngles; i++) {
		fscanf(in_theta, "%f", &datfromfile);
		h_angles[i] = datfromfile * pi / 180.0f;
	}
	float* d_angles;
	cudaMalloc(&d_angles, numAngles * sizeof(float));
	cudaMemcpy(d_angles, h_angles, numAngles * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, numX, numY);

	// Copy to device memory some data located at address h_img in host memory 
	cudaMemcpyToArray(cuArray, 0, 0, h_img, numX * numY * sizeof(float), cudaMemcpyHostToDevice);

	// Set texture reference parameters
	texRef.addressMode[0] = cudaAddressModeBorder;
	texRef.addressMode[1] = cudaAddressModeBorder;
	texRef.filterMode = cudaFilterModeLinear;
	texRef.normalized = false;

	// Bind the array to the texture reference
	cudaBindTextureToArray(texRef, cuArray, channelDesc);

	// Allocate result of transformation in device memory
	float* d_sg;
	cudaMalloc(&d_sg, numSensors * numAngles * sizeof(float));

	// Invoke kernel
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid((numSensors + dimBlock.x - 1) / dimBlock.x, (numAngles + dimBlock.y - 1) / dimBlock.y, 1);
	rotationKernel<<<dimGrid, dimBlock>>>(d_sg, dx, minX, dy, minY, numSensors, dr, minR, numAngles, d_angles);

	float *h_img_out;
	h_img_out = (float *)malloc(numSensors * numAngles * sizeof(float));
	cudaMemcpy(h_img_out, d_sg, numSensors * numAngles * sizeof(float), cudaMemcpyDeviceToHost);
	FILE *out = fopen(argv[6], "w");
	if (out == NULL) { printf("Error opening file!\n"); exit(1); }
	printf("numSensors = %d, numAngles = %d", numSensors, numAngles);
	for (int t_idx = 0; t_idx < numAngles; t_idx++) {
		for (int r_idx = 0; r_idx < numSensors; r_idx++) {
			//printf("Writing file!\n"); 
			fprintf(out, "%f\n", h_img_out[t_idx * numSensors + r_idx]);
		}
	}

	// Free device memory
	cudaFreeArray(cuArray);
	cudaFree(d_sg);

	return 0;
}
