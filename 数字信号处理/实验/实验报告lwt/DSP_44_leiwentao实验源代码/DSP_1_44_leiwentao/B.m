t1=-5:0.01:5;%定义区间-5到5 
x1=5*cos(2*pi*t1)-3*cos(3*pi*t1)+2*cos(6*pi*t1)+cos(8*pi*t1);%写出抽样信号表达式x1 
subplot(3,1,1);plot(t1,x1);%画出信号x1的图像
title("连续时间信号");%命名为"连续时间信号"
fs1=12;%抽样频率取12Hz 
ts=1/fs1;%抽样周期为频率的倒数 
t2=-5:ts:5;%以抽样周期为时间间隔 
x2=5*cos(2*pi*t2)-3*cos(3*pi*t2)+2*cos(6*pi*t2)+cos(8*pi*t2);%对信号进行频率为12Hz的抽样 
subplot(3,1,2);stem(t2,x2),grid on%画出抽样后的信号x2 
title("12Hz抽样后的信号");%命名为"12Hz抽样后的信号" 
fs2=20;%抽样频率取20Hz 
ts2=1/fs2;%抽样周期为频率倒数 
t3=-5:ts2:5;%以抽样周期为时间间隔
x3=5*cos(2*pi*t3)-3*cos(3*pi*t3)+2*cos(6*pi*t3)+cos(8*pi*t3);%对信号进频率为20Hz的抽样 
subplot(3,1,3);stem(t3,x3),grid on%画出抽样后的信号x3
title("20Hz抽样后的信号");%命名为"20Hz抽样后的信号" 