figure;clc;clear all;close all;
ws1=0.2*pi;wp1=0.4*pi;wp2=0.6*pi;ws2=0.8*pi;As=20;
tr_width=min((wp1-ws1),(ws2-wp2));
M=ceil(7*pi/tr_width)+1;
n=[0:1:M-1];
wc1=(ws1+wp1)/2;
wc2=(wp2+ws2)/2;
hd=ideal_lp(wc2,M)-ideal_lp(wc1,M);
w_bla=(blackman(M))';
h=hd.*w_bla;
[db,mag,pha,grd,w]=freqz_m(h,[1]);
subplot(2,2,1);stem(n,hd);axis([0 M-1 -0.4 0.5]);
title('Ideal Impulse Response');xlabel('n');ylabel('hd(n)');
subplot(2,2,2);stem(n,w_bla);axis([0 M-1 0 1.1]);
title('Blackman Window');xlabel('n');ylabel('w(n)');
subplot(2,2,3);stem(n,h);axis([0 M-1 -0.4 0.5]);
title('Actual Impulse Response');xlabel('n');ylabel('h(n)');
subplot(2,2,4);plot(w/pi,db);axis([0 1 -150 10]);grid;
title('Magnitude Response in dB');xlabel('frequency in piunits');ylabel('Decibels')