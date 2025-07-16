%主程序， 调用函数DSP_5_hanshu_39.m
n = input('输入阶数：');
char coefficient_Y(i) coefficient_X(i);%按序输入Y的系数
for i=1:n
coefficient_Y(i)=input('输入Y的系数：')
end%按序输入X的系数
for j=1:n
coefficient_X(j)=input('输入X的系数：')
end
E=DSP_5_hanshu_39(coefficient_Y)%调用函数,求出Y的多项式结果
F=DSP_5_hanshu_39(coefficient_X)%调用函数,求出X的多项式结果