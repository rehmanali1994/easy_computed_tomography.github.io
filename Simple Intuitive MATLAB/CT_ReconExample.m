clear
clc

% Make image of original object
orig_object = flipud(rgb2gray(im2double(imread('../images/transverseslice.jpg'))));
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
numAngles = 200; theta = linspace(0, 180, numAngles + 1); 
theta = theta(2:end); % Projection Angles in Degrees
tic; [r, sg] = sinogram(x, y, orig_object, theta); toc

% Now apply ramp filter to sinogram by using FFT
sg_rf = (1/mean(diff(r)))*rampFilt(sg); toc

% Show sinogram of image and after ramp filtering
figure; subplot(1,2,1); imagesc(r,theta,sg); title('Simulated Sinogram of Object'); 
xlabel('Sensor Position'); ylabel('Angle of Projection (degrees)'); colormap gray;
subplot(1,2,2); imagesc(r,theta,sg_rf); title('Ramp Filtered Sinogram'); 
xlabel('Sensor Position'); ylabel('Angle of Projection (degrees)'); colormap gray;

% Reconstruct an image of the object
recon_img = backproj(r, theta, sg, x, y, true); toc
recon_img_rf = backproj(r, theta, sg_rf, x, y, true); toc

% Calculate DC Offset Between Reconstructed and Original Object
dc_offset = mean(orig_object(:)) - mean(recon_img_rf(:));

% Show image of the reconstructed object without ramp
figure; imagesc(x,y,recon_img); axis xy equal tight;
xlabel('X coordinate'); ylabel('Y coordinate'); colorbar(); 
colormap gray; title('Reconstructed Object Without Ramp Filter');
figure; imagesc(x,y,recon_img_rf+dc_offset, [0,1]); axis xy equal tight; 
xlabel('X coordinate'); ylabel('Y coordinate'); colorbar();
colormap gray; title('Reconstructed Object With Ramp Filter');