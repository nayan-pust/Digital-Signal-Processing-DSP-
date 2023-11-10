[n,w]=buttord([1200/4000,2800/4000],[400/4000, 3200/4000],1,50);
[b,a]=butter(n,w,'stop');
figure(1)
freqz(b,a,128,8000);
title('Magnitude and phase of IIR Band stop filter');
grid on;

figure(2)
[h,P]=freqz(b,a,128,8000);
subplot(211);
plot(P,abs(h));
grid on;
title('IIR Bandstop Filter');
subplot(212);
zplane(b,a);
grid on;
title('Pole-Zero');


figure(3)
f=600:2:1200;
freqz(b,a,f,8000); % Transition Band
figure(4)
f=2800:2:3200;
freqz(b,a,f,8000); % Transition Band

