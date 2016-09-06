// To Compile: nvcc filtBackproj.cu -o filtBackproj.out -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcufft
// To Run: ./filtBackproj.out numAngles theta.txt minR maxR numSensors sg.txt minX maxX numX minY maxY numY recon.txt
// ./filtBackproj.out 360 theta.txt -53.2 53.2 2129 sg.txt -44.25 44.25 886 -29.5 29.5 591 recon.txt

// Includes System
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Includes CUDA
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define pi acosf(-1.0f)
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

// 1D float textures
texture<float, cudaTextureType1DLayered, cudaReadModeElementType> texSinogram;

// 1D interpolation kernel: Should be very similar to what you get if you used 1D interpolation on MATLAB
__global__ void backprojKernel(int numAngles, float *d_theta,
	float xmin, float dx, int numx,
	float ymin, float dy, int numy,
	float rmin, float dr, float *d_output)
{
	unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
	if (x_idx < numx && y_idx < numy) {
		float x, y, r, r_idx;
		x = xmin + x_idx * dx;
		y = ymin + y_idx * dy;
		d_output[x_idx + numx * y_idx] = 0;
		for (int theta_idx = 0; theta_idx < numAngles; theta_idx++) {
			r = x*cosf(d_theta[theta_idx] * pi / 180.0f) + y*sinf(d_theta[theta_idx] * pi / 180.0f);
			r_idx = (r - rmin) / dr + 0.5f; // +0.5f is texture thing
			d_output[x_idx + numx * y_idx] += tex1DLayered(texSinogram, r_idx, theta_idx);
		}
		//printf("location=%d layer=%d loc2find=%f  result=%f \n", location_idx, layer, loc2find, d_output[location_idx]);
	}
}

// Ramp Filter Kernel
__global__ void rampFiltKernel(cufftComplex* sgFFT, int numSensors, int numAngles)
{
	// Calculate normalized texture coordinates
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int jj = blockIdx.y * blockDim.y + threadIdx.y;

	if (ii < numSensors && jj < numAngles) {
		sgFFT[ii + numSensors*jj].x *= MIN(ii, numSensors - ii) / (float)numSensors;
		sgFFT[ii + numSensors*jj].y *= MIN(ii, numSensors - ii) / (float)numSensors;
	}
}

// Get float array for real component of cufftComplex array
__global__ void getRealPart(float* dst, cufftComplex* src, int numVals)
{
	// Calculate normalized texture coordinates
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if (ii < numVals)  dst[ii] = src[ii].x;
}

// Host code
int main(int argc, char *argv[])
{
	// Get all integer values from inputs
	int numAngles = strtol(argv[1], NULL, 10);
	float minR = atof(argv[3]);
	float maxR = atof(argv[4]);
	int numSensors = strtol(argv[5], NULL, 10);
	float minX = atof(argv[7]);
	float maxX = atof(argv[8]);
	int numX = strtol(argv[9], NULL, 10);
	float minY = atof(argv[10]);
	float maxY = atof(argv[11]);
	int numY = strtol(argv[12], NULL, 10);

	// Calculate some other values based on those inputs
	float dr = (maxR - minR) / ((float)(numSensors - 1));
	float dx = (maxX - minX) / ((float)(numX - 1));
	float dy = (maxY - minY) / ((float)(numY - 1));

	// Read data from files to host arrays
	cufftComplex* sg;
	sg = (cufftComplex *)malloc(sizeof(cufftComplex) * numAngles * numSensors);
	float *h_theta, *sg_rf;
	h_theta = (float*)malloc(numAngles*sizeof(float));
	sg_rf = (float*)malloc(numAngles*numSensors*sizeof(float));
	FILE *in_sg = fopen(argv[6], "r");
	FILE *in_theta = fopen(argv[2], "r");
	if (in_sg == NULL)
	{
		fprintf(stderr, "Input file for sinogram has some issues. Please check."); exit(1);
	}
	if (in_theta == NULL)
	{
		fprintf(stderr, "Input file for angle info has some issues. Please check."); exit(1);
	}
	float datfromfile;
	for (int jj = 0; jj < numAngles; jj++) {
		for (int ii = 0; ii < numSensors; ii++) {
			fscanf(in_sg, "%f", &datfromfile);
			sg[ii + numSensors*jj].x = datfromfile;
			sg[ii + numSensors*jj].y = datfromfile;
		}
	}
	for (int kk = 0; kk < numAngles; kk++) {
		fscanf(in_theta, "%f", &datfromfile);
		h_theta[kk] = datfromfile;
	}
	float *d_theta, *d_sg_rf;
	cudaMalloc(&d_theta, numAngles * sizeof(float));
	cudaMalloc(&d_sg_rf, numAngles * numSensors * sizeof(float));
	cudaMemcpy(d_theta, h_theta, numAngles * sizeof(float), cudaMemcpyHostToDevice);

	// Ramp filter the sinogram before binding it to a texture
	// Setup device input data for FFT
	cufftComplex* dData;
	cudaMalloc((void **)&dData, sizeof(cufftComplex) * numAngles * numSensors);
	if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate\n"); return -1;
	}
	// Copy Host Array to Device Array
	cudaMemcpy(dData, sg, sizeof(cufftComplex)* numAngles * numSensors, cudaMemcpyHostToDevice);
	// Make FFT Plan
	cufftHandle plan;
	if (cufftPlan1d(&plan, numSensors, CUFFT_C2C, numAngles) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: Plan creation failed"); return -1;
	}
	// Execute FFT
	if (cufftExecC2C(plan, dData, dData, CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed"); return -1;
	}
	if (cudaThreadSynchronize() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to synchronize\n"); return -1;
	}
	// Now Ramp Filter the FFT
	dim3 dimBlockRF(16, 16, 1);
	dim3 dimGridRF((numSensors + dimBlockRF.x - 1) / dimBlockRF.x,
		(numAngles + dimBlockRF.y - 1) / dimBlockRF.y, 1);
	rampFiltKernel << <dimGridRF, dimBlockRF >> >(dData, numSensors, numAngles);
	// Do Inverse FFT
	if (cufftExecC2C(plan, dData, dData, CUFFT_INVERSE) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed"); return -1;
	}
	if (cudaThreadSynchronize() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to synchronize\n"); return -1;
	}
	// Write the real part of output as the ramp filtered sinogram
	int thdsPerBlk = 256;
	int blksPerGrid = (numSensors*numAngles + thdsPerBlk - 1) / thdsPerBlk;
	getRealPart << <blksPerGrid, thdsPerBlk >> >(d_sg_rf, dData, numSensors*numAngles);
	cudaMemcpy(sg_rf, d_sg_rf, sizeof(float)*numSensors*numAngles, cudaMemcpyDeviceToHost);

	// Set Up Texture for the Sinogram in Three Steps: 1), 2), and 3)
	// 1) Allocate CUDA array in device memory
	cudaExtent extentDesc = make_cudaExtent(numSensors, 0, numAngles);  // <-- 0 height required for 1Dlayered
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMemcpy3DParms mParams = { 0 };
	mParams.srcPtr = make_cudaPitchedPtr(sg_rf, numSensors*sizeof(float), numSensors, 1);
	mParams.kind = cudaMemcpyHostToDevice;
	mParams.extent = make_cudaExtent(numSensors, 1, numAngles);  // <<-- non zero height required for memcpy to do anything
	cudaArray* cuArray;
	cudaMalloc3DArray(&cuArray, &channelDesc, extentDesc, cudaArrayLayered);
	mParams.dstArray = cuArray;
	cudaMemcpy3D(&mParams);
	// 2) Set texture reference parameters
	texSinogram.addressMode[0] = cudaAddressModeBorder;
	texSinogram.filterMode = cudaFilterModeLinear;
	texSinogram.normalized = false;
	// 3) Bind the array to the texture reference
	cudaBindTextureToArray(texSinogram, cuArray, channelDesc);

	// Allocate result of backprojection in device memory
	float *d_output;
	cudaMalloc(&d_output, numX * numY * sizeof(float));
	float *h_output;
	h_output = (float*)malloc(numX*numY*sizeof(float));

	// Invoke kernel
	dim3 dimBlockBP(16, 16, 1);
	dim3 dimGridBP((numX + dimBlockBP.x - 1) / dimBlockBP.x,
		(numY + dimBlockBP.y - 1) / dimBlockBP.y, 1);
	backprojKernel << <dimGridBP, dimBlockBP >> >(numAngles, d_theta, minX, dx, numX, minY, dy, numY, minR, dr, d_output);

	// Now write the output to a text file
	cudaMemcpy(h_output, d_output, numX * numY * sizeof(float), cudaMemcpyDeviceToHost);
	FILE *out = fopen(argv[13], "w");
	if (out == NULL) { printf("Error opening file!\n"); exit(1); }
	for (int y_idx = 0; y_idx < numY; y_idx++) {
		for (int x_idx = 0; x_idx < numX; x_idx++) {
			fprintf(out, "%f\n", h_output[x_idx + numX * y_idx]);
		}
	}

	// Free device memory
	cudaFreeArray(cuArray);
	cudaFree(d_output);

	return 0;
}
