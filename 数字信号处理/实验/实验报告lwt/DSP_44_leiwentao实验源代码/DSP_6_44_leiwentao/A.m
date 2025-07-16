ap = 1;%通带最大衰减
as = 15;%阻带最小衰减
Fs = 1;%抽样间隔
T = 1/Fs; 
wp=0.2*pi;
ws=0.3*pi;%考虑到预畸变
wap=2*Fs*tan(wp/2);
was=2*Fs*tan(ws/2);
[N,wac]=buttord(wap,was,ap,as,'s');% N为阶数，wac为3dB截止频率
[z,p,k]=buttap(N);% 创建巴特沃斯低通滤波器 z零点p极点k增益
[Bap,Aap]=zp2tf(z,p,k);% 由零极点和增益确定归一化Han(s)系数
[Bbs,Abs]=lp2lp(Bap,Aap,wac);% 低通到低通 计算去归一化Ha(s)
[B,A] = bilinear(Bbs,Abs,Fs); % 模拟域到数字域:双线性不变法
[H1,w] = freqz(B,A);% 根据H(z)求频率响应特性
figure(2);
f=w*Fs/(2*pi);
subplot(211);
plot(f,20*log10(abs(H1))); % 绘制幅度响应
title('双线性变换法——巴特沃斯BLPF(幅度)');
xlabel('频率/Hz');
ylabel('H1幅值/dB');
subplot(212);
plot(f,unwrap(angle(H1)));% 绘制相位响应
xlabel('频率/Hz');
ylabel('角度/Rad')
title('双线性变换法——巴特沃斯BLPF(相位)');