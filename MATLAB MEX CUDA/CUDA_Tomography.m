clear
clc

% Load Image and Give Pixels (x,y) Coordinates
orig_object = im2double(rgb2gray(imread('../images/cholangioca.jpg')));
%orig_object = phantom(128);
[Ny, Nx] = size(orig_object); 
dx = 0.1; dy = 0.1;
x = (-(Nx-1)/2:(Nx-1)/2) * dx;
y = (-(Ny-1)/2:(Ny-1)/2) * dy;

% Determining Sensor Array Element Spacing and Extent
dr = 1/(1/dx + 1/dy);

% Projection Angles for Computed Tomography
numAngles = 180; totalCoverage = 180;
theta = linspace(0,totalCoverage-totalCoverage/numAngles,numAngles);

% Create Sinogram
[r, sg] = sinogram(x, y, fliplr(orig_object'), theta(:));

% Perform Backprojection
%recon_img = filtBackproj(r, theta, sg, x, y)';
recon_img = backproj(r, theta, sg, x, y)';
dc_offset = mean(orig_object(:)) - mean(recon_img(:));
recon_img = recon_img + dc_offset;

% Show the reconstructed images
figure; imagesc(x, y, orig_object); colorbar();
colormap gray;
figure; imagesc(x, y, recon_img); colorbar();
colormap gray;


