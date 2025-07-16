wp =0.2*pi; %设置通带截止角频率
ws =0.3*pi; %设置阻带截止角频率
AS=1; %设置通带最大衰减
AP=15; %设置阻带最小衰减
T = 1; % 采样周期
%计算模拟滤波器的阶数和截止频率
[N, Wc] = buttord(wp, ws, AS, AP, 's');%构造Butterworth模拟滤波器
[num, den] = butter(N, Wc, 's');%双线性变换实现模数转换
[num_1, den_1] = bilinear(num, den, 1/T);
f = 0:0.01:1; % 归一化频率范围为0到1
w = f * pi; %数字滤波器的频率（ωc）
H = freqz(num_1, den_1, w);
mag = abs(H);
phase = angle(H) * 180/pi; %将相位转换为度

%绘制幅度响应曲线
subplot(2,1,1);
plot(f, mag);
title('幅度响应');
xlabel('频率(\times\pi)');
ylabel('幅度');
grid on;

% 绘制相位响应曲线
subplot(2,1,2);
plot(f, phase);
title('相位响应');
xlabel('频率(\times\pi)');
ylabel('相位(°)');
grid on;
