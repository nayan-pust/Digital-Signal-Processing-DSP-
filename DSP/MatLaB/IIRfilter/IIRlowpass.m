%Suppose our target is to design a filter to pass all frequencies below 1200 Hz with pass band 
%ripples = 1 dB and minimum stop band attenuation of 50 dB at 1500 Hz. The sampling frequency 
%for the filter is 8000 Hz;
fs=8000;
[n,w]=buttord(1200/4000,1500/4000,1,50); % finding the order of the filter
[b,a]=butter(n,w); % finding zeros and poles for filter
figure(1)
freqz(b,a,512,8000);
grid on;
title('Magnitude and phase plot of IIR Low pass filter');


figure(2)
[h,q] = freqz(b,a,512,8000);
subplot(211);
plot(q,abs(h)); % Normalized Magnitude plot
grid
title('IIR Low pass filter');

subplot(212);
zplane(b,a);
grid on;
title('pole zero constellation diagram');

figure(3)
f=1200:2:1500;
freqz(b,a,f,8000)% plotting the Transition band
grid on;
title('Transition band of IIR filter');
