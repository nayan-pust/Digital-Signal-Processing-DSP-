%with pass band ripples = 1 dB and minimum stop band attenuation of 50 dB. The
%sampling frequency for the filter is 8000 Hz;
[n,w]=buttord([1200/4000,2800/4000],[400/4000, 3200/4000],1,50);
[b,a]=butter(n,w,'bandpass');
figure(1)
freqz(b,a,128,8000);
grid on;
title('Magnitude and phase plot of IIR Band pass filter');

figure(2)
[h,w]=freqz(b,a,128,8000);
subplot(211);
plot(w,abs(h))
title('IIR Band pass filter');
grid

subplot(212);
zplane(b,a);
grid on;
title('Pole-zero diagram');
figure(3)
f=600:2:1200;
freqz(b,a,f,8000); % Transition Band
figure(4)
f=2800:2:3200;
freqz(b,a,f,8000); % Transition Band
figure(5)
 