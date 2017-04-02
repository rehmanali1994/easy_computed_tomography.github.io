# Why Did I Do This?
When I first learned how computed tomography worked, it was mostly through mathematics and I never implemented the algorithms to perform the image reconstruction myself. Instead, I used MATLAB built-in functions like radon and iradon to do things for me. The obvious downside to using these built-ins was that I never got to understand and implement the actual algorithms. The few resources I could find had very complicated implementations and did a poor job of helping me to understand how to implement these algorithms from scratch. 

Fortunately, after some time, I implemented a simple and easy-to-understand parallel-beam CT system in MATLAB without the use of radon and iradon. All the MATLAB files are short and well-commented, and I make extensive use of meshgrid, interp1, and interp2 functions to simplify many steps in the algorithms. 

The goal of this code is NOT to provide a comprehensive toolkit that enables you to reconstruct an image for an arbitrary CT system, but rather to provide key insights on how a simple parallel-beam CT system works. Once this simple example is understood, it should be much easier to simulate larger CT systems with different imaging geometries, because at that point, CT really just becomes a big geometry problem. 

# Elegant and Simple Derivation
## What does a CT system do?
A CT system effectively takes multiple projections of an object and tries to estimate what the object looks like at every point in space based on those projections. Here we say that the intensity of the object we are taking the projections of is f(x,y). In this parallel-beam geometry, in order to project f(x,y) onto a line, we must integrate f(x,y) along lines orthogonal to the line (...really plane) we are projecting onto. If we rotate this projection line, we can show how this projected profile changes as a function of angle: this is known as the sinogram or radon transform of f(x,y). 

![](https://cloud.githubusercontent.com/assets/10136046/18261138/5d015ace-73a9-11e6-9224-a74f51d615e8.png)

As it turns out, the projection of an object onto a plane can be analyzed using the Fourier-Slice (or Projection-Slice) Theorem, which basically says the Fourier transform of the projection of an object directly gives you values in the Fourier domain representation of the object. See the brief explanation of Fourier-Slice Theorem below:

![](https://cloud.githubusercontent.com/assets/10136046/18261140/5d285110-73a9-11e6-9edd-7ee35c30b88f.png)

Now that we understand how a CT system takes projections of an object at various angles and how Fourier Slice Theorem can be used to recover the Fourier spectrum of the object we are trying to reconstruct, we can now derive the process needed to reconstruct the object based on its projections: 

![](https://cloud.githubusercontent.com/assets/10136046/18261139/5d274a68-73a9-11e6-9a66-14fdcb298354.png)

Note that the result provides an exact algorithm on how to exactly recover the object function f(x,y). We first have to ramp filter the sinogram. This basically means multiplying the Fourier transform of each projection profile by the absolute value of frequency and then taking the inverse Fourier transform. We then backproject this ramp-filtered sinogram back into the image we are trying to reconstruct. The best way to visualize this is to pretend that the reconstructed image starts off as a blank canvas. Then, we take the ramp-filtered profile and smear it into the canvas at the angle the profile was initially projected at. **Example Demonstration** illustrates these steps.

## Example Demonstration
### Original Object Image
Here is the object I am trying to take projections of and reconstruct:
![](https://cloud.githubusercontent.com/assets/10136046/18263005/64185db4-73bb-11e6-88ac-d5acdaa310c1.png)
### Sinogram and Ramp Filtering
The image on the left shows the sinogram produced when simply projecting the image onto a line at various angles. The image on the right shows a ramp-filtered version of the same sinogram. Here ramp-filtering was applied to each row of the sinogram (each projection at a particular angle).
![](https://cloud.githubusercontent.com/assets/10136046/18263009/67fa0c16-73bb-11e6-915e-83f1830e990c.png)
### Backprojection Animation
Here you can see how each ramp-filtered projection gets smeared back into the reconstruction. As the “back-projections” accumulate, we start to see the original object.

![](https://cloud.githubusercontent.com/assets/10136046/18263016/73646e02-73bb-11e6-96ef-87d45e4cbcf6.gif)

# Implementation
## Simple Easy MATLAB
The **CT_ReconExample.m** script under the **Simple Intuitive MATLAB** directory of my repository runs all of the following functions and shows an animated reconstruction.
### Radon Transform / Sinogram Creation
The sinogram generating function sets the spacing and total span of detectors on the detector array based on the coordinates of pixels in the image. From this information we use meshgrid to generate **r_sensor** and **r_integration** which correspond to the coordinate along the detector array and the coordinate orthogonal to the detector array, respectively. We re-express **r_sensor** and **r_integration** in polar coordinates as **r_temp** and **theta_r** in order to easily rotate the detector and integration axes. The for loop iterates over rotation angle, uses interpolation to calculate the value of the image at points on the grid defined by the detector and integration axes, and sums along the integration axis;
```
function [r, sg] = sinogram(x, y, object, theta)
%SINOGRAM Generate a Sinogram of the Image
%   [r, sg] = sinogram(x, y, object, theta)
%   x -- x coordinates of object: numel(x) must be equal to size(object,2)
%   y -- y coordinates of object: numel(y) must be equal to size(object,1)
%   object -- intensity of object as a function of x and y
%   theta -- vector of angles (degrees) to take projections at

% 1/dr = 1/dx + 1/dy: just a way to make sure dr is smaller than dx and dy
dr = 1/(1/mean(diff(x))+ 1/mean(diff(y))); 

% get the number of elements on the sensor array
rmax = sqrt(max(abs(x))^2 + max(abs(y))^2);
rmax = dr*ceil(rmax/dr);

% give coordinate for each element relative to the center of sensor array
r = -rmax:dr:rmax; 
[r_sensor, r_integration] = meshgrid(r,r);
r_temp = sqrt(r_sensor.^2 + r_integration.^2);
theta_r = atan2(r_integration, r_sensor);

% get X, Y, R, and THETA for all points in object
[X, Y] = meshgrid(x,y);

% Set aside space for the sinogram
sg = zeros(numel(theta), numel(r));

% Now Construct the sinogram
for theta_idx = 1:numel(theta)
obj_rotated = interp2(X, Y, object, ...
r_temp.*cos(theta_r + deg2rad(theta(theta_idx))), ...
r_temp.*sin(theta_r + deg2rad(theta(theta_idx))), 'linear', 0);
sg(theta_idx,:) = sum(obj_rotated);
end

end
```

### Ramp-Filtering the Sinogram
Here we simply take an FFT of the sinogram along the detector coordinate. Then we multiply each projection spectrum with the ramp profile. Finally we take an inverse FFT along the detector coordinate to get the ramp-filtered sinogram.

```
function sg_rampfiltered = rampFilt(sg)
%RAMPFILT Apply Ramp Filter to Sinogram
%   sg_rampfiltered = rampFilt(sg)
%   Each row is a different angle of projection
%   Each column is a different sensor on the array

[numAngles, numSensors] = size(sg);
SG = fftshift(fft(sg')',2); % Take FFT of Sensor Data at Each Angle
ramp = ones(numAngles,1)*abs(-ceil((numSensors-1)/2):floor((numSensors-1)/2));
SG_rampfiltered = SG .* ramp; % Apply our ramp filter to the FFT
sg_rampfiltered = ifft(ifftshift(SG_rampfiltered,2)')';

end
```

### Backprojecting the Ramp-Filtered Sinogram
Recall the backprojection summation (integral) we derived:
![](https://cloud.githubusercontent.com/assets/10136046/18264599/0f887ada-73c6-11e6-8357-9b583aa3179d.png)
The function which performs the backprojection does pretty much the same thing: 
```
function recon_img = backproj(r_sensor, theta, sg, x, y, animation)
%BACKPROJ Backprojection Reconstruction Based on Sinogram
%   [x, y, recon_img] = backproj(r_sensor, theta, sg)
%   r_sensor -- positions of sensors on array
%   theta -- angles (in degrees) at which projections were taken
%   sg -- sinogram (sensor as columns; rows as degrees)
%   x -- x location of pixels in reconstruction (vector)
%   y -- y location of pixels in reconstruction (vector)
%   animation -- set this to true to show reconstruction
%   recon_img -- reconstructed image

% Set up coordinates for reconstructed image
[X, Y] = meshgrid(x,y);
recon_img = zeros(size(X));

for theta_idx = 1:numel(theta)
recon_img = recon_img + interp1(r_sensor, sg(theta_idx,:), ...
X*cosd(theta(theta_idx))+Y*sind(theta(theta_idx)), 'linear', 0);
end

end
```
## Optimizations for CUDA
### Disclaimer
I am not showing you any of the code needed to get the algorithm working on CUDA here. I would advise you to look at the CUDA code on my repository if you are interested in how to implement all of this in CUDA. Here I am only discussing the key features of CUDA that I used to implement and optimize various stages of the algorithm.
### Faster Sinogram Creation
For sinogram generation, I executed a kernel which ran separate threads for each point in the sinogram (each detector and projection angle). In each parallel thread, a for loop was used to iterate along the integration axis, and texture memory was used to interpolate image values on the grid defined by the detector and integration axes (just like in the MATLAB code). Summation along the integration direction was performed by the for loop executed by each thread. Since each point in the sinogram is calculated in parallel, the sinogram is computed much more quickly by the GPU.
### Ramp-Filtering by cuFFT
CUDA has a cuFFT library, which can perform a set of batched FFTs very quickly. In other words, this batch mode enables us to calculate the FFTs of the projections at each angle in parallel. In addition to the fact that individual FFTs are faster on a GPU, the batched FFT allows us to parallelize the calculation of each individual FFT. This batched FFT was used to perform the FFT step for ramp-filtering. A kernel was used to multiply the projection spectra by the ramp filter. Finally, a batched inverse FFT was performed in order to get the ramp-filtered sinogram. 
### Parallelized Backprojection
Each pixel in the reconstructed image was treated as a separate thread by the backprojection kernel. In each thread, the integration over projection angle was performed by a for loop in pretty much the same way that it was done in MATLAB. The 1D interpolation in MATLAB got replaced by a 1D layered texture where the projection angles served as the layers of the texture. 
