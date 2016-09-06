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

