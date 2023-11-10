%Suppose out target is to pass all frequencies below 1200 Hz
clc;
close all;
fs=8000; % sampling frequency
n=50; % order of the filter
w=1200/ (fs/2); %normalized digital  frequecy 
b=fir1(n,w,'low'); % Zeros of the filter



freqz(b,1,128,8000); % Magnitude and Phase Plot of the filter
grid on;
title('Magnitude and Phase Plot of the low pass filter');

figure(2)

[h,w]=freqz(b,1,128,8000);
subplot(211);

plot(w,abs(h)); % Normalized Magnitude Plot
grid on;
title('Low pass Filter');

%figure(3)
subplot(212);

zplane(b,1);
title('Poles-zeros');

