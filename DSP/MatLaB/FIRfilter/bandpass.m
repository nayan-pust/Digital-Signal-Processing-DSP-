fs=8000;
n=40;
b=fir1(n,[1200/4000 1800/4000],'bandpass');
freqz(b,1,128,8000)
title('Magnitude and Phase part of Bandpass filter');

figure(2)
[h,w]=freqz(b,1,128,8000);
subplot(211);
plot(w,abs(h)); % Normalized Magnitude Plot
grid on;
title('BandPass Filter');

subplot(212);
zplane(b,1);
grid on;
title('Pole_zero diagram');