h=[1,3,5,-3,-1]/9;
[N,Hk,wk]=dir2fs(h)
function[N,Hk,wk]=dir2fs(b);
N=length(b);
Hk=fft(b);
k=0:N-1;
wk=exp(2*pi*i/N).^k;
end