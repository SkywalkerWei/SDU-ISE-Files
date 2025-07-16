x1=zeros(1,5);%调用 zreos()函数
x1(1)=1;%给矩阵当中的 x1（1）赋值为1
x2=ones(1,5);%调用 ones() 函数
m=0:9;%定义 m 的范围 
x3=1*sin(2*pi*2*m/5+pi/2);%定义正弦序列表达式 x3=sin(4*pi*m/5+pi/2) 
x4=exp(j*pi*m);%定义复正弦序列 
x5=2.^m;%定义指数序列 
subplot(3,2,1);stem(x1);title("单位冲激");%输出单位冲激x1
subplot(3,2,2);stem(x2);title("单位阶跃");%输出单位阶跃x2
subplot(3,2,3);stem(x3);title("正弦序列");%输出正弦序列x3
subplot(3,2,4);stem(x4);title("复正弦序列");%输出复正弦序列x4
subplot(3,2,5);stem(x5);title("指数序列");%输出指数序列x5