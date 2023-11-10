fs=8000;
n=40;
b=fir1(n,[1500/4000 1550/4000],'stop');
freqz(b,1,128,8000)
grid on;
title('Magnitude and phase of notch filter');

figure(2)
[h,w]=freqz(b,1,128,8000);
subplot(211);
plot(w,abs(h)); % Normalized Magnitude Plot
grid
title('Notch filter');

subplot(212);
zplane(b,1);
grid on;
title('pole-zero');



