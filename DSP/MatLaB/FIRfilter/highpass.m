%Now our target is to pass all frequencies above 1200 Hz
fs=8000;
n=50;
w=1200/ (fs/2); 

b=fir1(n,w,'high');
freqz(b,1,128,8000);
grid on;
title('Magnitude and phase part of high pass filter');

figure(2)

[h,w]=freqz(b,1,128,8000);
subplot(211);
plot(w,abs(h)); % Normalized Magnitude Plot
title('High pass Filter');
grid


subplot(212);
zplane(b,1);
title('Pole-zero diagram');
grid on;




