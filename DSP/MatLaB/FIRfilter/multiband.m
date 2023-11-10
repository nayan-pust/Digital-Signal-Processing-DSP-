n=50;
w=[0.2 0.4 0.6];
b=fir1(n,w);
freqz(b,1,128,8000);
grid on;
title('Magnitude and phase of multiband filter');

figure(2)
[h,w]=freqz(b,1,128,8000);
subplot(211);
plot(w,abs(h)); % Normalized Magnitude Plot
grid on;
title('MultiBand Filter');

subplot(212);
zplane(b,1);
grid on;
title('Pole-zero');