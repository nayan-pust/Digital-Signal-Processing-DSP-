%We will consider same filter but our target now is to pass all frequencies above 1200 Hz
[n,w]=buttord(1200/5000,1500/5000,1,50);
[b,a]=butter(n,w,'high');
figure(1)
freqz(b,a,512,10000);
grid on;
title('Magnitude and phase plot of IIR High pass filter');

figure(2)
[h,q] = freqz(b,a,512,8000);
subplot(211);
plot(q,abs(h)); % Normalized Magnitude plot
grid
title('IIR High pass filter');
subplot(212);
zplane(b,a);
title('pole zero constellation diagram');

figure(3)
f=1200:2:1500;
freqz(b,a,f,10000);
title('Transition band of IIR filter');
