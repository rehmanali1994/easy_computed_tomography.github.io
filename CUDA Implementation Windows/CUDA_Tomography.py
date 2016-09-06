import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os

# Load Image and Give Pixels (x,y) Coordinates
orig_object = misc.imread('../images/cholangioca.jpg', flatten = True);
Ny, Nx = orig_object.shape; 
dx = 0.1; dy = 0.1;
x = (np.arange(Nx) - (Nx-1)/2) * dx;
y = (np.arange(Ny) - (Ny-1)/2) * dy;

# Determining Sensor Array Element Spacing and Extent
dr = 1/(1/dx+ 1/dy);
rmax = np.sqrt(np.max(np.abs(x))**2 + np.max(np.abs(y))**2);
rmax = dr*np.ceil(rmax/dr); 
Nr = int(2*rmax/dr + 1);

# Projection Angles for Computed Tomography
numAngles = 360;
theta = np.linspace(0,180,numAngles,endpoint=False);

# Save Important Values to .txt files before running CUDA
np.savetxt('img.txt', orig_object.flatten());
np.savetxt('theta.txt', theta);

# Compile then run the CUDA program that creates the sinogram
os.system('sinogram.exe '+str(numAngles)+' theta.txt '+ \
	str(-rmax)+' '+str(rmax)+' '+str(Nr)+' sg.txt '+ \
	str(np.min(x))+' '+str(np.max(x))+' '+str(Nx)+' '+ \
	str(np.min(y))+' '+str(np.max(y))+' '+str(Ny)+' img.txt');

# Compile then run the CUDA program that performs the filtered backprojection
os.system('filtBackproj.exe '+str(numAngles)+' theta.txt '+ \
	str(-rmax)+' '+str(rmax)+' '+str(Nr)+' sg.txt '+ \
	str(np.min(x))+' '+str(np.max(x))+' '+str(Nx)+' '+ \
	str(np.min(y))+' '+str(np.max(y))+' '+str(Ny)+' recon.txt');

# Load the reconstructed image
recon_img = np.loadtxt('recon.txt');
recon_img = np.reshape(recon_img, (Ny,Nx))

# Delete all files that this process created
os.system('del img.txt');
os.system('del recon.txt');
os.system('del sg.txt');
os.system('del theta.txt');

# Show the reconstructed images
plt.imshow(recon_img, cmap='gray');
plt.show();
