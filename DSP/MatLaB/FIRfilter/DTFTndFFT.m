n=-1:3;
x=1:5;
k=0:500;
w=(pi/500)*k;
X=x*(exp(-j*pi/500))*(n'*k);
magX=abs(X);
angX=angle(X);
realX=real(X);
imagX=imag(X);

subplot(221);
plot(k/500,magX);
grid;
xlabel('Frequency in pi units ');
title('Magnitude part');


subplot(222);
plot(k/500,angX/pi);
grid;
xlabel('Frequency in pi units ');
title('Angle part');


subplot(223);
plot(k/500,realX);
grid;
xlabel('Frequency in pi units ');
title('Real part');


subplot(224);
plot(k/500,imagX);
grid;
xlabel('Frequency in pi units ');
title('Imaginary part');



