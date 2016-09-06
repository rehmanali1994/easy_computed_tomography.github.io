clear
clc

% Make image of original object
orig_object = flipud(rgb2gray(im2double(imread('../images/cholangioca.jpg'))));
[Ny, Nx] = size(orig_object);

% Now lets assign coordinates to points in the image
dx = 0.1; dy = 0.1; % spacing of points in image
% assume center of image is at (0,0)
x = (-(Nx-1)/2:(Nx-1)/2)*dx; 
y = (-(Ny-1)/2:(Ny-1)/2)*dy;

% Show image of original object
figure; imagesc(x,y,orig_object); axis xy equal tight; colormap gray;
xlabel('X coordinate'); ylabel('Y coordinate'); title('Original Object');

% Get Sinogram 
theta = 0.5:0.5:180; % Projection Angles in Degrees
tic; [r, sg] = sinogram(x, y, orig_object, theta); toc

% Now apply ramp filter to sinogram by using FFT
sg_rf = rampFilt(sg); toc

% Show sinogram of image and after ramp filtering
figure; subplot(1,2,1); imagesc(r,theta,sg); title('Simulated Sinogram of Object'); 
xlabel('Sensor Position'); ylabel('Angle of Projection (degrees)'); colormap gray;
subplot(1,2,2); imagesc(r,theta,sg_rf); title('Ramp Filtered Sinogram'); 
xlabel('Sensor Position'); ylabel('Angle of Projection (degrees)'); colormap gray;

% Reconstruct an image of the object
recon_img = backproj(r, theta, sg, x, y, true); toc
recon_img_rf = backproj(r, theta, sg_rf, x, y, true); toc

% Show image of the reconstructed object without ramp
figure; subplot(1,2,1); imagesc(x,y,recon_img); axis xy equal tight;
xlabel('X coordinate'); ylabel('Y coordinate'); colormap gray;
title('Reconstructed Object Without Ramp Filter');
subplot(1,2,2); imagesc(x,y,recon_img_rf); axis xy equal tight;
xlabel('X coordinate'); ylabel('Y coordinate'); colormap gray;
title('Reconstructed Object With Ramp Filter');