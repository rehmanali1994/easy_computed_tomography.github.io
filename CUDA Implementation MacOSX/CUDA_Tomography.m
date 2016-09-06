clear
clc

% Load Image and Give Pixels (x,y) Coordinates
orig_object = im2double(rgb2gray(imread('../images/cholangioca.jpg')));
[Ny, Nx] = size(orig_object); 
dx = 0.1; dy = 0.1;
x = (-(Nx-1)/2:(Nx-1)/2) * dx;
y = (-(Ny-1)/2:(Ny-1)/2) * dy;

% Determining Sensor Array Element Spacing and Extent
dr = 1/(1/dx + 1/dy);
rmax = sqrt(max(abs(x))^2 + max(abs(y))^2);
rmax = dr*ceil(rmax/dr); 
Nr = round(2*rmax/dr + 1);

% Projection Angles for Computed Tomography
numAngles = 360; totalCoverage = 180;
theta = linspace(0,totalCoverage-totalCoverage/numAngles,numAngles);

% Save Important Values to .txt files before running CUDA
save_orig_object = orig_object'; % Just make row dimension the fast dimension
fid_img = fopen('img.txt','w');
fprintf(fid_img,'%f\n', save_orig_object(:));
fid_theta = fopen('theta.txt','w');
fprintf(fid_theta,'%f\n', theta(:));
fclose(fid_img); fclose(fid_theta);

% Compile then run the CUDA program that creates the sinogram
system(['./sinogram.out ', num2str(numAngles), ' theta.txt ', ...
	num2str(-rmax), ' ', num2str(rmax), ' ', num2str(Nr), ' sg.txt ', ...
	num2str(min(x)), ' ', num2str(max(x)), ' ', num2str(Nx), ' ', ...
	num2str(min(y)), ' ', num2str(max(y)), ' ', num2str(Ny), ' img.txt']);

% Compile then run the CUDA program that performs the filtered backprojection
system(['./filtBackproj.out ', num2str(numAngles), ' theta.txt ', ...
	num2str(-rmax), ' ', num2str(rmax), ' ', num2str(Nr), ' sg.txt ', ...
	num2str(min(x)), ' ', num2str(max(x)), ' ', num2str(Nx), ' ', ...
	num2str(min(y)), ' ', num2str(max(y)), ' ', num2str(Ny), ' recon.txt']);

% Load the reconstructed image
recon_img = textread('recon.txt');
recon_img = reshape(recon_img, Nx, Ny)';

% Delete all files that this process created
system('rm img.txt');
system('rm recon.txt');
system('rm sg.txt');
system('rm theta.txt');

% Show the reconstructed images
imagesc(x, y, recon_img);
colormap gray;
