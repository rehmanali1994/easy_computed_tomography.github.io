// To Compile: nvcc filtBackproj.cu -o filtBackproj.out -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcufft
// To Run: ./filtBackproj.out numAngles theta.txt minR maxR numSensors sg.txt minX maxX numX minY maxY numY recon.txt
// ./filtBackproj.out 360 theta.txt -53.2 53.2 2129 sg.txt -44.25 44.25 886 -29.5 29.5 591 recon.txt

// Includes System
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mex.h>

// Includes CUDA
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>

#define pi 3.141592653589793238462643383279502884197169399375105820974f
#define CCE checkCudaErrors
#define HtoD cudaMemcpyHostToDevice
#define DtoH cudaMemcpyDeviceToHost
#define MIN(a,b) (((a)<(b))?(a):(b))
//#define MAX(a,b) (((a)>(b))?(a):(b))

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
		float x, y, r, r_idx, dt, integral = 0;
		x = xmin + x_idx * dx;
		y = ymin + y_idx * dy;
		for (int theta_idx = 0; theta_idx < numAngles; theta_idx++) {
			// Decide angular weight to assign to this backprojection
			if (theta_idx == 0) {
				dt = (pi / 180.0f)*(d_theta[1] - d_theta[0]);
			}
			else if (theta_idx == numAngles-1) {
				dt = (pi / 180.0f)*(d_theta[numAngles-1] - d_theta[numAngles-2]);
			}
			else {
				dt = (pi / 180.0f)*(d_theta[theta_idx+1] - d_theta[theta_idx-1])/2;
			}
			// Perform Backprojection by integration over angle
			r = x*cosf(d_theta[theta_idx] * pi / 180.0f) + y*sinf(d_theta[theta_idx] * pi / 180.0f);
			r_idx = (r - rmin) / dr + 0.5f; // +0.5f is texture thing
			integral += tex1DLayered(texSinogram, r_idx, theta_idx);
		}
		d_output[x_idx + numx * y_idx] = dt*integral/dr;
	}
}

// Ramp Filter Kernel
__global__ void rampFiltKernel(cufftComplex* sgFFT, int numSensors, int numAngles)
{
	// Calculate normalized texture coordinates
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int jj = blockIdx.y * blockDim.y + threadIdx.y;

	if (ii < numSensors && jj < numAngles) {
		sgFFT[ii + numSensors*jj].x *= ( MIN(ii, numSensors - ii) / ( (float) numSensors ) ) / ( (float) numSensors );
		sgFFT[ii + numSensors*jj].y *= ( MIN(ii, numSensors - ii) / ( (float) numSensors ) ) / ( (float) numSensors );
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
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Argument check
	if (nrhs != 5)	{ mexErrMsgTxt("Wrong number of inputs.\n"); }
  if (nlhs != 1)	{ mexErrMsgTxt("Wrong number of outputs.\n"); }

	// Gather values from inputs
	double *r = (double *)mxGetData(prhs[0]);
	int numSensors = (int) mxGetNumberOfElements(prhs[0]);
	float minR = (float) r[0];
	float maxR = (float) r[numSensors-1];
	float dr = (maxR - minR) / ((float)(numSensors - 1));
	double *theta_degrees = (double *)mxGetData(prhs[1]);
	int numAngles = (int) mxGetNumberOfElements(prhs[1]);
	double *sinogram = (double *)mxGetData(prhs[2]);
  double *x = (double *)mxGetData(prhs[3]);
	int numX = (int) mxGetNumberOfElements(prhs[3]);
	float minX = (float) x[0];
	float maxX = (float) x[numX-1];
	float dx = (maxX - minX) / ((float)(numX - 1));
	double *y = (double *)mxGetData(prhs[4]);
	int numY = (int) mxGetNumberOfElements(prhs[4]);
	float minY = (float) y[0];
	float maxY = (float) y[numY-1];
	float dy = (maxY - minY) / ((float)(numY - 1));

	// Read data from files to host arrays
	cufftComplex* sg;
	sg = (cufftComplex *)malloc(sizeof(cufftComplex) * numAngles * numSensors);
	float *h_theta, *sg_rf;
	h_theta = (float*)malloc(numAngles*sizeof(float));
	sg_rf = (float*)malloc(numAngles*numSensors*sizeof(float));
	for (int jj = 0; jj < numAngles; jj++) {
		for (int ii = 0; ii < numSensors; ii++) {
			sg[ii + numSensors*jj].x = (float)sinogram[ii + numSensors*jj];
			sg[ii + numSensors*jj].y = 0;
		}
	}
	for (int kk = 0; kk < numAngles; kk++) {
		h_theta[kk] = (float)theta_degrees[kk];
	}
	float *d_theta, *d_sg_rf;
	CCE(cudaMalloc(&d_theta, numAngles * sizeof(float)));
	CCE(cudaMalloc(&d_sg_rf, numAngles * numSensors * sizeof(float)));
	CCE(cudaMemcpy(d_theta, h_theta, numAngles * sizeof(float), cudaMemcpyHostToDevice));

	// Ramp filter the sinogram before binding it to a texture
	// Setup device input data for FFT
	cufftComplex* dData;
	CCE(cudaMalloc((void **)&dData, sizeof(cufftComplex) * numAngles * numSensors));
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
	}
	// Copy Host Array to Device Array
	CCE(cudaMemcpy(dData, sg, sizeof(cufftComplex)* numAngles * numSensors, cudaMemcpyHostToDevice));
	// Make FFT Plan
	cufftHandle plan;
	if (cufftPlan1d(&plan, numSensors, CUFFT_C2C, numAngles) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
	}
	// Execute FFT
	if (cufftExecC2C(plan, dData, dData, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
	}
	if (cudaThreadSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}
	// Now Ramp Filter the FFT
	dim3 dimBlockRF(16, 16, 1);
	dim3 dimGridRF((numSensors + dimBlockRF.x - 1) / dimBlockRF.x,
		(numAngles + dimBlockRF.y - 1) / dimBlockRF.y, 1);
	rampFiltKernel << <dimGridRF, dimBlockRF >> >(dData, numSensors, numAngles);
	// Do Inverse FFT
	if (cufftExecC2C(plan, dData, dData, CUFFT_INVERSE) != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
	}
	if (cudaThreadSynchronize() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}
	// Write the real part of output as the ramp filtered sinogram
	int thdsPerBlk = 256;
	int blksPerGrid = (numSensors*numAngles + thdsPerBlk - 1) / thdsPerBlk;
	getRealPart << <blksPerGrid, thdsPerBlk >> >(d_sg_rf, dData, numSensors*numAngles);
	CCE(cudaMemcpy(sg_rf, d_sg_rf, sizeof(float)*numSensors*numAngles, cudaMemcpyDeviceToHost));

	// Set Up Texture for the Sinogram in Three Steps: 1), 2), and 3)
	// 1) Allocate CUDA array in device memory
	cudaExtent extentDesc = make_cudaExtent(numSensors, 0, numAngles);  // <-- 0 height required for 1Dlayered
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMemcpy3DParms mParams = { 0 };
	mParams.srcPtr = make_cudaPitchedPtr(sg_rf, numSensors*sizeof(float), numSensors, 1);
	mParams.kind = cudaMemcpyHostToDevice;
	mParams.extent = make_cudaExtent(numSensors, 1, numAngles);  // <<-- non zero height required for memcpy to do anything
	cudaArray* cuArray;
	CCE(cudaMalloc3DArray(&cuArray, &channelDesc, extentDesc, cudaArrayLayered));
	mParams.dstArray = cuArray;
	CCE(cudaMemcpy3D(&mParams));
	// 2) Set texture reference parameters
	texSinogram.addressMode[0] = cudaAddressModeBorder;
	texSinogram.filterMode = cudaFilterModeLinear;
	texSinogram.normalized = false;
	// 3) Bind the array to the texture reference
	CCE(cudaBindTextureToArray(texSinogram, cuArray, channelDesc));

	// Allocate result of backprojection in device memory
	float *d_output;
	CCE(cudaMalloc(&d_output, numX * numY * sizeof(float)));
	float *h_output;
	h_output = (float*)malloc(numX*numY*sizeof(float));

	// Invoke kernel
	dim3 dimBlockBP(16, 16, 1);
	dim3 dimGridBP((numX + dimBlockBP.x - 1) / dimBlockBP.x,
		(numY + dimBlockBP.y - 1) / dimBlockBP.y, 1);
	backprojKernel << <dimGridBP, dimBlockBP >> >(numAngles, d_theta, minX, dx, numX, minY, dy, numY, minR, dr, d_output);

	// Now write the output to a MATLAB array
	CCE(cudaMemcpy(h_output, d_output, numX * numY * sizeof(float), cudaMemcpyDeviceToHost));
	plhs[0] = mxCreateDoubleMatrix(numX, numY, mxREAL);
	double *recon_out = (double *)mxGetPr(plhs[0]);
	for (int y_idx = 0; y_idx < numY; y_idx++) {
		for (int x_idx = 0; x_idx < numX; x_idx++) {
			recon_out[x_idx + numX * y_idx] = (float) h_output[x_idx + numX * y_idx];
		}
	}

	// Destroy all allocations and reset all state on the current device in the current process.
	CCE(cudaDeviceReset());
}
