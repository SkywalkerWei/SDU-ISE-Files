h1=[1,-2,3,5,3,-2,1];
h2=[1,-2,3,3,-2,1];
h3=[1,-2,3,5,-3,2,-1];
h4=[1,-2,3,-3,2,-1];
figure;
[a1,w1,type1,tao1]=amp(h1);
subplot(221),plot(w1/pi,abs(a1));
xlabel('*pi');
title('h1 幅频特性');
[a2,w2,type2,tao2]=amp(h2);
subplot(222),plot(w2/pi,abs(a2));
title('h2 幅频特性');
xlabel('*pi');
[a3,w3,type3,tao3]=amp(h3);
subplot(223),plot(w3/pi,abs(a3));
title('h3 幅频特性');
xlabel('*pi');
[a4,w4,type4,tao4]=amp(h4);
subplot(224),plot(w4/pi,abs(a4));
title('h4 幅频特性');
xlabel('*pi');
figure;
subplot(221),zplane(h1,1);title('h1 零点');
subplot(222),zplane(h2,1);title('h2 零点');
subplot(223),zplane(h3,1);title('h3 零点');
subplot(224),zplane(h4,1);title('h4 零点');

function [A,w,type,tao]=amp(h)
N=length(h);
tao=(N-1)/2;
L=floor((N-1)/2);
n=1:L+1;
w=[0:500]*2*pi/500;
if all(abs(h(n)-h(N-n+1))<1e-10)
 A=2*h(n)*cos(((N+1)/2-n)'*w)-mod(N,2)*h(L+1);
 type=2-mod(N,2);
elseif all(abs(h(n)+h(N-n+1))<1e-10)&(h(L+1)*mod(N,2)==0)
 A=2*h(n)*sin(((N+1)/2-n)'*w);
 type=4-mod(N,2);
else disp('错误：这是非线性相位系统！')
 [A,m,w]=dtft(h);
 A=A.*exp(i*m);
 type='?';
 tao='?';
 
end
end