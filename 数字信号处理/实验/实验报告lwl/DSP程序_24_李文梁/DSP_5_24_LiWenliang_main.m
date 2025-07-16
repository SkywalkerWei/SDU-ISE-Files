N=input('请输入阶数：');
char coefficient_Y(i) coefficient_X(i); 
for i=1:N %按序输入分子多项式的系数
coefficient_Y(i)=input('请输入分子多项式的各阶系数：');
end
coefficient_Y %显示输入结果
for j=1:N %按序输入分母多项式的系数
coefficient_X(j)=input('请输入分母多项式的各阶系数：');
end
coefficient_X %显示输入结果
E=DSP_5_24_LiWenliang_code1(coefficient_Y)%调用函数,求出分子多项式的结果
F=DSP_5_24_LiWenliang_code1(coefficient_X)%调用函数,求出分母多项式的结果
