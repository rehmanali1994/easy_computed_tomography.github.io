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
    sg(theta_idx,:) = sum(obj_rotated*dr);
end

end

