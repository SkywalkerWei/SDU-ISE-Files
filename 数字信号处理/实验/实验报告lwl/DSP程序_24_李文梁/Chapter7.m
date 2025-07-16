%%
wp=0.7*pi;%�趨ͨ����ֹƵ��
ws=0.5*pi;%�趨�����ֹƵ��
deltaw=wp-ws;%������ɴ����
N=ceil(11*pi/deltaw);%ʹ��ceil��������ȡ�����ɹ��ɴ���ȼ����˲����Ľ״�N
wc=(wp+ws)/2;%����wc
wn=wc/pi;%���ݽ�ֹƵ�ʼ���wn
n=0:N-1;%��ΪN=220,����ȡ��0��N-1���������������������С˥��������
h=fir1(N-1,wn,'high',blackman(N));
%ʹ��fir1��������˲������õ��˲����ĵ�λ�弤��Ӧh

[H,w]=freqz(h,1,1000,'whole');
%ʹ��freqz���������˲���Ƶ����Ӧ��ѡ��Ƶ�ʵ���Ϊ1000��Ƶ�ʷ�ΧΪ��0��2�У�
%������Ľ��H��Ƶ����Ӧ�ĸ���ֵ��w�Ƕ�Ӧ��Ƶ��ֵ
H=(H(1:1:501))';w=(w(1:1:501))';
%Ϊ��������Ļ�ͼ����ֻ��ȡ��Ƶ����Ӧ��ǰ��Σ�ǰ501��ֵ����Ӧ0���У���ת��Ϊ������
mag=abs(H);%ʹ��abs���������ֵ��Ƶ����Ӧ������
db=20*log10((mag+eps)/max(mag));
%��������ת��Ϊ�Էֱ�Ϊ��λ�Ķ����̶ȣ���������С��eps��ֹ������ֵ���±���
pha=angle(H);%ʹ��angle������Ƶ����Ӧ����λ��
grd=grpdelay(h,1,w);%�����˲��������ӳ�
dw=2*pi/1000;%����Ƶ�ʼ�������ڻ��ƺ�����Ƶ����Ӧ

%������Ӧͼ��
figure
subplot(3,1,1)
stem(n,h);
title('����������');xlabel('n'); ylabel('w(n)');
axis([0,N,0,0.45]);grid on;

subplot(3,1,2)
plot(w/pi,db); 
title('������Ӧ(dB)');xlabel('\omega/\pi'); ylabel('20log|H(e^j^\omega)|(dB)');
axis([0,1,-120,10]);grid on;
set(gca,'xtickmode','manual','xtick',[0,0.2,0.4,0.45,0.7,1.0]);%��Ϊ���������
set(gca,'ytickmode','manual','ytick',[-120,-90,-60,0,10]);

subplot(3,1,3)
plot(w/pi,pha);
title('��λ��Ӧ');xlabel('\omega/\pi'); ylabel('arg|H(e^j^\omega)|');
axis([0,1,-4,4]);grid on;
%%
%%
wp=0.2*pi;%�趨ͨ����ֹƵ��
ws=0.4*pi;%�趨�����ֹƵ��
wc=(ws+wp)/2;%����wc
wn=wc/pi;%���ݽ�ֹƵ�ʼ���wn
deltaw=ws-wp;%������ɴ����
N0=ceil(6.6*pi/deltaw);%ʹ��ceil��������ȡ��������6.6��Ϊ���������ɴ���
N=N0+mod(N0+1,2);%Ϊʵ��FIR����1ż�Գ��˲�����N������������
Windows=(hamming(N+1))';
hd=fir1(N,wc/(pi));
h=hd.*Windows;
%ʹ��fir1��������˲������õ��˲����ĵ�λ�弤��Ӧh
n=0:N;%ȡ��0��N-1�������������������С˥��������

[H,w]=freqz(h,1000,'whole');
%ʹ��freqz���������˲���Ƶ����Ӧ��ѡ��Ƶ�ʵ���Ϊ1000��Ƶ�ʷ�ΧΪ��0��2�У�
%������Ľ��H��Ƶ����Ӧ�ĸ���ֵ��w�Ƕ�Ӧ��Ƶ��ֵ
H=(H(1:501))';w=(w(1:501))';
%Ϊ��������Ļ�ͼ����ֻ��ȡ��Ƶ����Ӧ��ǰ��Σ�ǰ501��ֵ����Ӧ0���У���ת��Ϊ������
mag=abs(H);%ʹ��abs���������ֵ��Ƶ����Ӧ������
db=20*log10((mag+eps)/max(mag));
%��������ת��Ϊ�Էֱ�Ϊ��λ�Ķ����̶ȣ���������С��eps��ֹ������ֵ���±���
pha=angle(H);%ʹ��angle������Ƶ����Ӧ����λ��

figure
subplot(3,1,1);
stem(n,h);
axis([0,N,0,0.5]);title('������');xlabel('n');ylabel('w(n)');grid on;

subplot(3,1,2);
plot(w/pi,db);
axis([0,1,-125,10]);title('������Ӧ(dB)');xlabel('w/pi');ylabel('20log|H(e^j^\omega)|(dB)');
set(gca,'XTickMode','manual','XTick',[0,wp/pi,ws/pi,1]);
set(gca,'YTickMode','manual','YTick',[-50,-20,-3,0]);grid on;

subplot(3,1,3);
plot(w/pi,pha);
axis([0,1,-4,4]);title('��λƵ����Ӧ');xlabel('w/pi');ylabel('arg|H(e^j^\omega)|');
set(gca,'XTickMode','manual','XTick',[0,wp/pi,ws/pi,1]);
set(gca,'YTickMode','manual','YTick',[-3.1416,0,3.1416,4]);grid on;
%%