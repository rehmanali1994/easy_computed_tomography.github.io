clear
clc

% Make image of original object
%orig_object = flipud(rgb2gray(im2double(imread('transverseslice.jpg'))));
orig_object = flipud(rgb2gray(im2double(imread('../images/cholangioca.jpg'))));
%orig_object = orig_object(1:5:end, 1:5:end);
%orig_object = phantom(128);
[Ny, Nx] = size(orig_object);

% Now lets assign coordinates to points in the image
dx = 0.1; dy = 0.1; % spacing of points in image
% assume center of image is at (0,0)
x = (-(Nx-1)/2:(Nx-1)/2)*dx; 
y = (-(Ny-1)/2:(Ny-1)/2)*dy;

% Get Sinogram 
theta = 0.1:0.1:180; % Projection Angles in Degrees
tic; [r, sg] = sinogram(x, y, fliplr(orig_object'), theta(:)); toc

% Show sinogram of image and after ramp filtering
figure; imagesc(r,theta,sg'); title('Simulated Sinogram of Object'); 
xlabel('Sensor Position'); ylabel('Angle of Projection (degrees)'); colormap gray;

% Iterative reconstruction of the object by ramp-filtered backprojection
recon_img_rf = filtBackproj(r, theta, sg, x, y)'; % Running Reconstruction
[r, sg_reproj] = sinogram(x, y, fliplr(recon_img_rf'), theta);
sg_err = sg - sg_reproj; % Error in Sinogram for Each Iteration
numIter = 10; % Number of Iterations
figure; M = moviein(numIter); 
for iter = 1:numIter
    % Calculate Filtered Backprojection Image
    tic; filt_backproj_img = filtBackproj(r, theta, sg_err, x, y)';
    [r, sg_reproj] = sinogram(x, y, fliplr(filt_backproj_img'), theta);
    % Update Reconstruction and Sinogram Error Using Reprojection
    recon_img_rf = recon_img_rf + filt_backproj_img;
    sg_err = sg_err - sg_reproj;
    % Plot Result of Next Iteration
    subplot(1,2,1); imagesc(x, y, recon_img_rf); 
    axis xy equal tight; colormap gray; colorbar();
    xlabel('X coordinate'); ylabel('Y coordinate'); 
    title(['FBP Reconstruction Iteration: ', num2str(iter)]);
    subplot(1,2,2); imagesc(x, y, filt_backproj_img); 
    axis xy equal tight; colormap gray; colorbar();
    xlabel('X coordinate'); ylabel('Y coordinate'); 
    title(['Scaled Increment Image Iteration: ', num2str(iter)]);
    M(iter) = getframe; toc; 
end

% Iterative Reconstruction by Exact Line-Search Gradient Descent
sg_err = sg; % Error in Sinogram for Each Iteration
recon_img = zeros(size(orig_object)); % Running Reconstruction
numIter = 100; % Number of Iterations
figure; M = moviein(numIter); 
for iter = 1:numIter
    % Calculate Gradient (Backprojection) Image
    tic; gradient_backproj_img = backproj(r, theta, sg_err, x, y)';
    [r, sg_reproj] = sinogram(x, y, fliplr(gradient_backproj_img'), theta(:));
    % Calculate Step Size for this Gradient by Exact Line Search
    reproj_scaling = (sg_reproj(:)'*sg_err(:)) / (sg_reproj(:)'*sg_reproj(:));
    % Update Reconstruction and Sinogram Error Using Calculated Step Size
    recon_img = recon_img + reproj_scaling * gradient_backproj_img;
    sg_err = sg_err - reproj_scaling * sg_reproj; % Like Gram=Schmidt Process
    % Plot Result of Next Iteration
    subplot(1,2,1); imagesc(x, y, recon_img); 
    axis xy equal tight; colormap gray; colorbar();
    xlabel('X coordinate'); ylabel('Y coordinate'); 
    title(['Gradient Descent Reconstruction Iteration: ', num2str(iter)]);
    subplot(1,2,2); imagesc(x, y, reproj_scaling * gradient_backproj_img); 
    axis xy equal tight; colormap gray; colorbar();
    xlabel('X coordinate'); ylabel('Y coordinate'); 
    title(['Scaled Gradient Increment Image Iteration: ', num2str(iter)]);
    M(iter) = getframe; toc; 
end

% Show image of original object and reconstructed images
figure; subplot(1,3,1); imagesc(x, y, orig_object, [0,1]); 
axis xy equal tight; colormap gray; title('Original Object');
xlabel('X coordinate'); ylabel('Y coordinate'); 
title('Image of Original Object'); colorbar();
subplot(1,3,2); imagesc(x, y, recon_img,[0,1]); axis xy equal tight;
xlabel('X coordinate'); ylabel('Y coordinate'); colormap gray;
title('Iterative Gradient Descent with Exact Line-Search'); colorbar();
subplot(1,3,3); imagesc(x, y, recon_img_rf,[0,1]); axis xy equal tight;
xlabel('X coordinate'); ylabel('Y coordinate'); colormap gray; colorbar();
title('Iterative Ramp-Filtered Backprojection Reconstructed Image'); 