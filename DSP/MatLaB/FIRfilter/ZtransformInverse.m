syms z n
a=1/16^n;
ZTrans=ztrans(a);       %Z transform
disp(ZTrans);
InvrZ=iztrans(ZTrans);       %InverseZtransform
disp(InvrZ);


clc;clear all;close all;
B=[0 1 1];
A=[1 -2 3];
disp(roots(A));
disp(roots(B));
zplane(B,A);

syms Z n
A=ztrans(3*(-1)^n);
disp(A);
I=iztrans(3*Z/(Z+1));
II=iztrans(A);
disp(II);
disp(I);

clc;clear;
z=roots([1,0,2]);
p=roots([1,2,-1,1]);
disp(z);
disp(p);
zplane(z,p);

clc;clear;
z=roots([1,0,0,1]);
p=roots([1,0,2,0,1]);
disp(z);
disp(p);
zplane(z,p);

clc;clear;
ZZ=roots([4,8,10]);
PP=roots([2,8,18,20]);


clc;clear;
num=[1,0,0,1];
den=[1,0,2,0,1];
systf=tf(num,den);
disp(systf);
pzmap(systf);






