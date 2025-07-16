%%
wp=0.7*pi;%设定通带截止频率
ws=0.5*pi;%设定阻带截止频率
deltaw=wp-ws;%计算过渡带宽度
N=ceil(11*pi/deltaw);%使用ceil函数向上取整，由过渡带宽度计算滤波器的阶次N
wc=(wp+ws)/2;%计算wc
wn=wc/pi;%根据截止频率计算wn
n=0:N-1;%因为N=220,所以取从0至N-1的整数数列以满足阻带最小衰减的条件
h=fir1(N-1,wn,'high',blackman(N));
%使用fir1函数设计滤波器，得到滤波器的单位冲激响应h

[H,w]=freqz(h,1,1000,'whole');
%使用freqz函数计算滤波器频率响应，选择频率点数为1000，频率范围为（0到2π）
%所输出的结果H是频率响应的复数值，w是对应的频率值
H=(H(1:1:501))';w=(w(1:1:501))';
%为方便后续的画图这里只截取了频率响应的前半段（前501个值，对应0至π）并转置为列向量
mag=abs(H);%使用abs函数求绝对值求频率响应幅度谱
db=20*log10((mag+eps)/max(mag));
%将幅度谱转换为以分贝为单位的对数刻度，引入无穷小量eps防止出现零值导致报错
pha=angle(H);%使用angle函数求频率响应的相位谱
grd=grpdelay(h,1,w);%计算滤波器的组延迟
dw=2*pi/1000;%计算频率间隔，用于绘制后续的频率响应

%绘制相应图谱
figure
subplot(3,1,1)
stem(n,h);
title('布莱克曼窗');xlabel('n'); ylabel('w(n)');
axis([0,N,0,0.45]);grid on;

subplot(3,1,2)
plot(w/pi,db); 
title('幅度响应(dB)');xlabel('\omega/\pi'); ylabel('20log|H(e^j^\omega)|(dB)');
axis([0,1,-120,10]);grid on;
set(gca,'xtickmode','manual','xtick',[0,0.2,0.4,0.45,0.7,1.0]);%人为设置坐标点
set(gca,'ytickmode','manual','ytick',[-120,-90,-60,0,10]);

subplot(3,1,3)
plot(w/pi,pha);
title('相位响应');xlabel('\omega/\pi'); ylabel('arg|H(e^j^\omega)|');
axis([0,1,-4,4]);grid on;
%%
%%
wp=0.2*pi;%设定通带截止频率
ws=0.4*pi;%设定阻带截止频率
wc=(ws+wp)/2;%计算wc
wn=wc/pi;%根据截止频率计算wn
deltaw=ws-wp;%计算过渡带宽度
N0=ceil(6.6*pi/deltaw);%使用ceil函数向上取整，其中6.6π为哈明窗过渡带宽
N=N0+mod(N0+1,2);%为实现FIR类型1偶对称滤波器对N做奇数化处理
Windows=(hamming(N+1))';
hd=fir1(N,wc/(pi));
h=hd.*Windows;
%使用fir1函数设计滤波器，得到滤波器的单位冲激响应h
n=0:N;%取从0至N-1的整数数列满足阻带最小衰减的条件

[H,w]=freqz(h,1000,'whole');
%使用freqz函数计算滤波器频率响应，选择频率点数为1000，频率范围为（0到2π）
%所输出的结果H是频率响应的复数值，w是对应的频率值
H=(H(1:501))';w=(w(1:501))';
%为方便后续的画图这里只截取了频率响应的前半段（前501个值，对应0至π）并转置为列向量
mag=abs(H);%使用abs函数求绝对值求频率响应幅度谱
db=20*log10((mag+eps)/max(mag));
%将幅度谱转换为以分贝为单位的对数刻度，引入无穷小量eps防止出现零值导致报错
pha=angle(H);%使用angle函数求频率响应的相位谱

figure
subplot(3,1,1);
stem(n,h);
axis([0,N,0,0.5]);title('哈明窗');xlabel('n');ylabel('w(n)');grid on;

subplot(3,1,2);
plot(w/pi,db);
axis([0,1,-125,10]);title('幅度响应(dB)');xlabel('w/pi');ylabel('20log|H(e^j^\omega)|(dB)');
set(gca,'XTickMode','manual','XTick',[0,wp/pi,ws/pi,1]);
set(gca,'YTickMode','manual','YTick',[-50,-20,-3,0]);grid on;

subplot(3,1,3);
plot(w/pi,pha);
axis([0,1,-4,4]);title('相位频率响应');xlabel('w/pi');ylabel('arg|H(e^j^\omega)|');
set(gca,'XTickMode','manual','XTick',[0,wp/pi,ws/pi,1]);
set(gca,'YTickMode','manual','YTick',[-3.1416,0,3.1416,4]);grid on;
%%