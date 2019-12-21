// To Compile: nvcc sinogram.cu -o sinogram.out
// To Run: ./sinogram.out numAngles theta.txt minR maxR numSensors sg.txt minX maxX numX minY maxY numY img.txt
// ./sinogram.out 360 theta.txt -53.2 53.2 2129 sg.txt -44.25 44.25 886 -29.5 29.5 591 img.txt

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mex.h>

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>

#define pi 3.141592653589793238462643383279502884197169399375105820974f
#define CCE checkCudaErrors
#define HtoD cudaMemcpyHostToDevice
#define DtoH cudaMemcpyDeviceToHost
#define MIN(a,b) (((a)<(b))?(a):(b))
//#define MAX(a,b) (((a)>(b))?(a):(b))

// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Kernel Used to Calculate Sinogram By Rotating Object and Integrating Along Fixed Direction
__global__ void rotateIntegrateKernel(float* d_sg, float dx, float minX, float dy, float minY, int numSensors, float dr, float minR, int numAngles, float* theta)
{
	// Get Indices for Sensor-Line and Angle
	unsigned int r_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int t_idx = blockIdx.y * blockDim.y + threadIdx.y;

	// Make Sure to Limit Indices to the Number of Sensors and Angles
	if (r_idx < numSensors && t_idx < numAngles) {

		// Calculate Sensor-Line Coordinate [in Real World Units]
		float r = r_idx * dr + minR;

		// Perform Integration Over Line Given by Sensor and Angle
		float integral = 0;
		for (int z_idx = 0; z_idx < numSensors; z_idx++) {
			float z = z_idx * dr + minR;
			// Transform coordinates
			float tr = (r * cosf(theta[t_idx]) + z * sinf(theta[t_idx]) - minX)/dx + 0.5f;
			float tz = (z * cosf(theta[t_idx]) - r * sinf(theta[t_idx]) - minY)/dy + 0.5f;
			// Read from texture and write to global memory
			integral += tex2D(texRef, tr, tz);
		}

		// Output Integral onto Sinogram (Stored on Device)
		d_sg[t_idx*numSensors + r_idx] = integral*dr;
	}
}

// Host code
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Argument check
	if (nrhs != 4)	{ mexErrMsgTxt("Wrong number of inputs.\n"); }
  if (nlhs != 2)	{ mexErrMsgTxt("Wrong number of outputs.\n"); }

  // Gather values from inputs
  double *x = (double *)mxGetData(prhs[0]);
	int numX = (int) mxGetNumberOfElements(prhs[0]);
	float minX = (float) x[0];
	float maxX = (float) x[numX-1];
	float dx = (maxX - minX) / ((float)(numX - 1));
	double *y = (double *)mxGetData(prhs[1]);
	int numY = (int) mxGetNumberOfElements(prhs[1]);
	float minY = (float) y[0];
	float maxY = (float) y[numY-1];
	float dy = (maxY - minY) / ((float)(numY - 1));
	double *orig_object = (double *)mxGetData(prhs[2]);
	double *theta_degrees = (double *)mxGetData(prhs[3]);
	int numAngles = (int) mxGetNumberOfElements(prhs[3]);
	float dr = 1 / ( (1/dx) + (1/dy) );
	float maxR = (float) dr*ceil(sqrt(pow(MAX(abs(minX),abs(maxX)),2) + pow(MAX(abs(minY),abs(maxY)),2))/dr);
	float minR = -maxR;
	int numSensors = (int) ( 1 + ( (maxR - minR) / dr ) );

	// Write Double Arrays to Float Arrays
	float *h_img, *h_angles;
	h_img = (float *)malloc(numX*numY*sizeof(float));
	h_angles = (float *)malloc(numAngles*sizeof(float));
	for (int j = 0; j < numY; j++) {
		for (int i = 0; i < numX; i++) {
			h_img[i + j*numX] = (float) orig_object[i + j*numX];
		}
	}
	for (int i = 0; i < numAngles; i++) {
		h_angles[i] = (float)theta_degrees[i] * pi / 180.0f;
	}

	// Move Projection Angles from Host to Device
	float* d_angles;
	CCE(cudaMalloc(&d_angles, numAngles * sizeof(float)));
	CCE(cudaMemcpy(d_angles, h_angles, numAngles * sizeof(float), cudaMemcpyHostToDevice));

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray* cuArray;
	CCE(cudaMallocArray(&cuArray, &channelDesc, numX, numY));

	// Copy to device memory some data located at address h_img in host memory
	CCE(cudaMemcpyToArray(cuArray, 0, 0, h_img, numX * numY * sizeof(float), cudaMemcpyHostToDevice));

	// Set texture reference parameters
	texRef.addressMode[0] = cudaAddressModeBorder;
	texRef.addressMode[1] = cudaAddressModeBorder;
	texRef.filterMode = cudaFilterModeLinear;
	texRef.normalized = false;

	// Bind the array to the texture reference
	CCE(cudaBindTextureToArray(texRef, cuArray, channelDesc));

	// Allocate result of transformation in device memory
	float* d_sg;
	CCE(cudaMalloc(&d_sg, numSensors * numAngles * sizeof(float)));

	// Invoke kernel
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid((numSensors + dimBlock.x - 1) / dimBlock.x, (numAngles + dimBlock.y - 1) / dimBlock.y, 1);
	rotateIntegrateKernel<<<dimGrid, dimBlock>>>(d_sg, dx, minX, dy, minY, numSensors, dr, minR, numAngles, d_angles);

	// Write Sensor Coordiaate to Output MATLAB Array
	plhs[0] = mxCreateDoubleMatrix( 1, numSensors, mxREAL );
	double *r = (double *)mxGetPr(plhs[0]);
	for (int r_idx = 0; r_idx < numSensors; r_idx++) {
		r[r_idx] = (double) (minR + dr*r_idx);
	}

	// Move Sinogram from Device to Host and Copy to Output MATLAB Array
	float *h_img_out;
	h_img_out = (float *)malloc(numSensors * numAngles * sizeof(float));
	CCE(cudaMemcpy(h_img_out, d_sg, numSensors * numAngles * sizeof(float), cudaMemcpyDeviceToHost));
	plhs[1] = mxCreateDoubleMatrix(numSensors, numAngles, mxREAL);
	double *sg_out = (double *)mxGetPr(plhs[1]);
	for (int t_idx = 0; t_idx < numAngles; t_idx++) {
		for (int r_idx = 0; r_idx < numSensors; r_idx++) {
			sg_out[t_idx * numSensors + r_idx] = (double) h_img_out[t_idx * numSensors + r_idx];
		}
	}

	// Destroy all allocations and reset all state on the current device in the current process.
	CCE(cudaDeviceReset());

}
