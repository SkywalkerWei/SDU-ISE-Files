function cal=TheCalledFunction(xxi)
N=length(xxi);%求多项式长度
root=roots(xxi);%求根
sort=cplxpair(root);%对求出的根排序
k=(N-1)/2;%计算级联个数
for J=1:k
sortpro=[sort(2*J-1) sort(2*J)];
cal(J,:)=poly(sortpro);
end%确定最终的多项式,poly将一对根转化为对应的多项式