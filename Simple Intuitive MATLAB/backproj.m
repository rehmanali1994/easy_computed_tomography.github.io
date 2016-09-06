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

if animation
    figure; M = moviein(numel(theta)); 
    for theta_idx = 1:numel(theta)
        recon_img = recon_img + interp1(r_sensor, sg(theta_idx,:), ...
            X*cosd(theta(theta_idx))+Y*sind(theta(theta_idx)), 'linear', 0);
        imagesc(x,y,recon_img); axis xy equal tight; colormap gray;
        xlabel('X coordinate'); ylabel('Y coordinate'); 
        title(['Reconstructed Object Upto Theta = ', ...
            num2str(theta(theta_idx)), ' degrees']);
        M(theta_idx) = getframe; 
    end
else
    for theta_idx = 1:numel(theta)
        recon_img = recon_img + interp1(r_sensor, sg(theta_idx,:), ...
            X*cosd(theta(theta_idx))+Y*sind(theta(theta_idx)), 'linear', 0);
    end
end

end

