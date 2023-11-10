% Define the parameters of the signal
f = 10;             % Frequency of the sinusoid (in Hz)
fs = 200;           % Sampling rate (in Hz)
t = 0:1/fs:1;        % Time vector

% Generate the sinusoidal signal
x = sin(2*pi*f*t);

% Plot the original signal
%figure(1)
subplot(3 1 1);
plot(t,x);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Signal');

% Sample the signal
Ts = 1/fs;           % Sampling interval (in seconds)
n = 0:Ts:1;          % Sampling instants
xn = sin(2*pi*f*n);  % Sampled signal

% Plot the sampled signal
%figure(2)
subplot(312);
stem(n,xn);
xlabel('Time (s)');
ylabel('Amplitude');
title('Sampled Signal');

% Reconstruct the analog signal using ideal reconstruction
xr = zeros(size(t));  % Initialize the reconstructed signal
for i = 1:length(n)
    xr = xr + xn(i)*sinc((t-(i-1)*Ts)/Ts);
end

% Plot the reconstructed signal
%figure(3)
subplot(313);
plot(t,xr);
xlabel('Time (s)');
ylabel('Amplitude');
title('Reconstructed Signal');




 