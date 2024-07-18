#include<stdio.h>
#include<ctype.h>//C5T4用到 / 另附有不使用库函数的做法
#include<math.h>//用于水仙花数立方 缩小找因数范围 小球反弹
#include<windows.h>//防止最后一个输出看不到 调用pause 清空终端缓冲区 便于显示
/*
    第四章课后题
    3.写出下面表达式的值
    1）0；
    2）1；
    3）1；
    4）0；
    5）1；
    <!--短路原则：上题即可用-->
    <!--条件表达式：第八题 printf("成绩是 %.1lf,相应的等级是 %c\n",s,69-(s>=60.0?((int)s==100?4:((int)(s-50.0))/10):0));//一步到位-->

    第五章
    <!--理解三种循环头区别：dowhile相比while至少做一次；for可以等效代替另两个 for(;;)死循环最方便-->
    <!--break：从循环里面往外退一层 / 结束case；continue：跳过本次循环该语句后面部分直接做下一次-->
*/
int CompareThree()//C4T4
{
    int a,b,c;
    printf("输入三个逗号分隔的数比大小：\n");
    scanf("%d,%d,%d",&a,&b,&c);
    a=a>b?a:b;
    a=a>c?a:c;
    printf("max=%d\n",a);
    
}
int FunctionSix()//C4T6
{
    int a;
    printf("输入x的值:\n");
    scanf("%d",&a);
    printf("x=%d, y=%d\n",a,a<1?a:(a<10?a*2-1:a*3-11));
}
int Trans()//C4T8
{
    double s;
    printf("输入学生成绩:\n");
    scanf("%lf",&s);
    printf("成绩是 %.1lf,相应的等级是 %c\n",s,69-(s>=60.0?((int)s==100?4:((int)(s-50.0))/10):0));//一步到位
    //if线
    if(s>=90.0){printf("A\n");}
    else if(s>=80.0){printf("B\n");}
    else if(s>=70.0){printf("C\n");}
    else if(s>=60.0){printf("D\n");}
    else{printf("E\n");}
    switch(((int)s)/10) //switch线
    {
        case 10:
        case 9:printf("A\n");break;
        case 8:printf("B\n");break;
        case 7:printf("C\n");break;
        case 6:printf("D\n");break;
        case 0 ... 5:printf("E\n");break;//该写法在通过MinGW部署的windows-msvc-gcc-x64环境可以编译运行
    }
}
void NumCount(int n)//位数
{
    int i=0;
    while(n>0){i++;n/=10;}
    printf("\n位数:%d\n",i);
}
void SplitNum(int n)//分割
{
    if(n>=10) SplitNum(n/10);
    printf("%d,",n%10);
}
void ReverseNum(int n)//逆序
{
    while(n>0){printf("%d",n%10);n/=10;}
}
int NumOptions()//C4T9
{
    int n1;
    printf("Number:");scanf("%d",&n1);
    NumCount(n1);
    printf("Split:");SplitNum(n1);
    printf("\n");
    printf("Reverse:");ReverseNum(n1);
    printf("\n");
}
int Interest()//C4T10
{
    double bonus;
    printf("\nThe interest is:");
    scanf("%lf",&bonus);
    if(bonus<=100000.0){printf("\n奖金:%.2lf\n",bonus*0.1);}
    else if(bonus<=200000){printf("\n奖金:%.2lf\n",bonus*0.075+2500.0);}
    else if(bonus<=400000){printf("\n奖金:%.2lf\n",(bonus-200000.0)*0.05+17500);}
    else if(bonus<=600000){printf("\n奖金:%.2lf\n",(bonus-400000.0)*0.03+27500);}
    else if(bonus<=1000000){printf("\n奖金:%.2lf\n",(bonus-600000.0)*0.015+33500);}
    else {printf("\n奖金:%.2lf\n",(bonus-1000000.0)*0.01+39500);}
    switch((int)bonus)//该写法在通过MinGW部署的windows-msvc-gcc-x64环境可以编译运行
    {
        case 0 ... 100000:
            printf("\n奖金:%.2lf\n",(double)bonus*0.1);
            break;
        case 100001 ... 200000:
            printf("\n奖金:%.2lf\n",(double)bonus*0.075+2500.0);
            break;
        case 200001 ... 400000:
            printf("\n奖金:%.2lf\n",((double)bonus-200000.0)*0.05+17500);
            break;
        case 400001 ... 600000:
            printf("\n奖金:%.2lf\n",((double)bonus-400000.0)*0.03+27500);
            break;
        case 600001 ... 1000000:
            printf("\n奖金:%.2lf\n",((double)bonus-600000.0)*0.015+33500);
            break;
        default:
            printf("\n奖金:%.2lf\n",((double)bonus-1000000.0)*0.01+39500);
    }
}
int SortInts()//C4T11
{
    int a[5];
    printf("\n排序:\n");
    for(int i=0;i<4;i++) scanf("%d",&a[i]);
    for(int i=0;i<4;i++)
    {
        int min=i;
        for(int j=i;j<4;j++){min=a[min]>a[j]?j:min;}
        int t=a[i];a[i]=a[min];a[min]=t;
    }
    for(int i=0;i<4;i++) printf("%d ",a[i]);
    printf("\n");
}
int Height()//C4T12
{
    double p[2][4]={{2.0,-2.0,-2.0,2.0},{2.0,2.0,-2.0,-2.0}},x,y;
    printf("Input point:\n");scanf("%lf,%lf",&x,&y);
    int ff=1;
    for(int i=0;i<4;i++){if(((p[0][i]-x)*(p[0][i]-x)+(p[1][i]-y)*(p[1][i]-y))<1){printf("\nHeight=10\n");ff=!ff;break;}}
    if(ff) printf("\nHeight=0\n");
}
int CharCount()//C5T4：没必要做 / 不使用库函数的做法见下文注释
{
    char ch,emp;
    int le=0,di=0,sp=0,ot=0;
    printf("Input a string:\n");
    emp=getchar();//吃掉上题回车
    while((ch=getchar())!='\n')
    {
        if(isupper(ch)||islower(ch)) le++;//ch>=65&&ch<=90 || ch>=97&&ch<=122
        else if(isdigit(ch)) di++;//ch>=48 && ch<=57
        else if(ch==' ') sp++;//isspcae会把所有空白字符都计入，会WA
        else ot++;
    }
    printf("\nletter:%d,digit:%d,space:%d,other:%d\n",le,di,sp,ot);
}
int SpSum()//C5T5
{
    int a,n;
    unsigned long long an=0,t=0;
    printf("Input a and n split with ,:\n");scanf("%d,%d",&a,&n);
    t=a;
    while(n>0)
    {
        an+=t;
        t=a+t*10;
        n--;
    }
    printf("\na+aa+aaa+...=%llu\n",an);
}
int Factorial()//C5T6
{
    //本题使用无符号__int64类型 长度足够存所有数值位 不需要double损失精度 int和long都有可能炸
    unsigned __int64 n=1,t=1;//__int64 在通过MinGW部署的windows-msvc-gcc-x64环境可以编译运行
    for(int i=2;i<=20;i++){t*=i;n+=t;}
    printf("\n1!+...+20!=%llu\n",n);//%l64u不识别
}
int ThreeSum()//C5T7
{
    double an=2.0;//平方和倒数的第一项加在这里
    an+=(1+100)*100/2;
    for(double i=2;i<=50;i++) an+=i*i;
    for(double i=2;i<=10;i++) an+=1/i;
    printf("\nSUM=%lf\n",an);
}
int Narcissus()//C5T8
{
    printf("\n水仙花数:");
    for(int i=100;i<=999;i++) if(i==pow(i/100,3)+pow(i%10,3)+pow(i/10%10,3)) printf("%d ",i);
    printf("\n");
}
int PerfectNumber()//C5T9
{
    for(int i=2;i<1000;i++)
    {
        int s=1;
        for(int j=2;j<=sqrt(i);j++){if(i%j==0) s+=j+i/j;}
        if(s==i)
        {
            printf("\n%d,its factors are",i);
            for(int j=1;j<i;j++){if(i%j==0) printf(" %d",j);}
        }
    }
    printf("\n");
}
int fib()//C5T10
{
    double a[25]={1.0,1.0},an;
    for(int i=2;i<25;i++) a[i]=a[i-1]+a[i-2];
    for(int i=1;i<=20;i++) an+=a[i+1]/a[i];
    printf("\nfib sum=%.12lf\n",an);//题目未告知保留几位小数
}
int Bounce()//C5T11
{
    //第一次100m 第二次（首项）100 第三次（第二项）50 第四次（第三项）25 ... 第十次（第九项）100*（1/2）^8.（第十次反弹高度为第十次距离的1/4）
    printf("\n10th Bounce back total:%.12lf\n10th bounce:%.12lf\n",100+100*(1.0-1/pow(2,9))/0.5,100/pow(2,10));
}
int LeapYearBissextile()//SP1
{
    printf("\n\n闰年\n");
    int count=0;
    for(int i=2000;i<=2500;i++)
    {
        if ((i%4==0 && i%100!=0) || i%400==0)
        {
            printf("%d ",i);count++;
            if(count%5==0) printf("\n");
        }
    }
}
int isPrime(int number)//SP2
{
    for(int i=2;i<=sqrt(number);i++){if(number%i==0) return 0;}
    return 1;
}
int FindPrime()//SP2
{
    printf("\n\n100~1000之间的素数：\n");
    int count=0;
    for(int i=100;i<=1000;i++)
    {
        if(isPrime(i)) 
        {
            printf("%d ",i);count++;
            if(count%7==0) printf("\n");
        }
    }
}
int main()
{
    
    CompareThree();//C4T4
    FunctionSix();//C4T6
    Trans();//C4T8
    NumOptions();//C4T9
    Interest();//C4T10
    SortInts();//C4T11
    Height();//C4T12
    CharCount();//C5T4
    SpSum();//C5T5
    system("cls");
    printf("后续部分为根据书上样例输出，按题意无输入\n");
    system("pause");
    Factorial();//C5T6
    ThreeSum();//C5T7
    Narcissus();//C5T8
    PerfectNumber();//C5T9
    fib();//C5T10
    Bounce();//C5T11
    LeapYearBissextile();//SP1
    FindPrime();//SP2
    system("pause");
}