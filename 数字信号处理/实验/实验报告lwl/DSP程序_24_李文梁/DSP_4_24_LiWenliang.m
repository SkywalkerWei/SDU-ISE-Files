n=0:15;%定义n取值范围为0至15的整数
x1=cos(5*n*pi/16);%定义序列x1
subplot(3,2,1),stem(x1),title('x1(n)原序列');%绘制原序列x1
X1=fft(x1,16);%对x1做16点DFT变换
%绘制x1做16点DFT变换的图像
subplot(3,2,2);stem(abs(X1));title('x1(n)16点DFT的幅度谱')
%绘制对应的幅度谱
X2=fft(x1,32);%对x1做32点DFT变换
% 绘制x1做32点DFT变换的图像
subplot(3,2,3);stem(abs(X2)),title('x1(n)32点DFT的幅度谱')
%绘制对应的幅度谱
[H,w]=freqz(x1,512,'whole');%计算DTFT
H=abs(H);
subplot(3,2,4);plot(w/pi,H),grid on;title('x1(n) DTFT')%画出DTFT变换的图像

m=0:15;%定义m取值范围为0至15的整数
x3=4*cos(m*pi/2);%写出原序列表达式
X3=fft(x3,16);%对原序列做16点DFT变换
subplot(3,2,5);stem(x3),title('x2(n)原序列')%绘制原序列
subplot(3,2,6);stem(abs(X3)),title('x2(n)16点DFT的幅度谱');
%画出16点DFT变换的图像的幅度谱
