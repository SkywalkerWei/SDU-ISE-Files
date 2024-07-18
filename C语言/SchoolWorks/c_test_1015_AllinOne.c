#include<stdio.h>
#include<windows.h>
int Test_Of_Sizeof()//sizeof测试
{
    printf("Size of Variables:\nint:%d\nlong:%d\nfloat:%d\ndouble:%d\n\n",sizeof(int),sizeof(long int),sizeof(float),sizeof(double));
    system("pause");
}
int DataOverflow()//变量溢出
{
    int a=2147483646;
    printf("Test of stack overflow:2147483646+2=%d\n\n",a+2);
    system("pause");
}
int Usage_Of_Operator()//连续运算：从右至左
{
    int b,bb;
    printf("Now is the test for Operators. You need to input an integer:");
    scanf("%d",&b);
    b+=b*=b/=3;
    bb=b+1;
    printf("b+=b*=b/=3 is %d ; And then b++ is %d .\n\n",b,bb);
    system("pause");
}
int rRound()//四舍五入
{
    double c;
    printf("Input an floating number and the program will output it's rounds:\n");
    scanf("%lf",&c);
    printf("1.Use the function Printf(): %.3lf\n",c);
    printf("2.Use mathematical methods: %f\n\n",(int)(c*1000+0.5)/1000.0);
    system("pause");
}
void PrintNum(int n)//分割
{
    if(n>=10) PrintNum(n/10);
    printf("%d ",n%10);
}
int SplitNum()//分割
{
    int n1;
    printf("Now you can split a number:");
    scanf("%d",&n1);
    PrintNum(n1);
    printf("\n");
    // printf("%d %d %d\n\n",n1/100,n1/10%10,n1%10);
    system("pause");
}
int Test_On_Book()
{
    /*
    第三章习题

    4.1：
    输出 c1=a,c2=b
    c1=97,c2=98
    4.2:
    输出 c1=<,c2== //ascii码60和61
    c1=-59,c2=-58 //占用符号位，补码变负数
    4.3:
    赋值97和98：与4.1相同
    赋值197和198：第一行输出超出ascii范围，视编译器输出不同；第二行c1=197,c2=198

    5.
    键盘输入：
    a=3 b=7
    8.5 71.82Aa

    6.
    */
    char c1='C',c2='h',c3='i',c4='n',c5='a';
    // int c6,c7;
    char c6,c7;
    c1+=4,c2+=4,c3+=4,c4+=4,c5+=4;
    printf("printf:%c%c%c%c%c\nputchar:",c1,c2,c3,c4,c5);
    putchar(c1);putchar(c2);putchar(c3);putchar(c4);putchar(c5);
    printf("\n\n");
    system("pause");
    //7.
    printf("Now is the No.7: The area and volume:两个值用空格隔开！\n");
    double rr,hh,p=3.14159;
    scanf("%lf %lf",&rr,&hh);
    printf("circumference:%.2lf\narea:%.2lf\nsuperficial area of ball:%.2lf\nvolume of ball:%.2lf\nvolume of columnar:%.2lf\n\n",2*p*rr,p*rr*rr,4*p*rr*rr,4.0/3.0*p*rr*rr*rr,p*rr*rr*hh);
    system("pause");
    //8.
    // system("cls");
    getchar();//存在读入的问题，上一行的回车
    printf("getchar test:需要连续两个数，不要空格\n");
    c6=getchar();c7=getchar();
    putchar(c6);putchar(c7);
    printf("\n%c %c\n",c6,c7);
    system("pause");
    /*
    8.1
    字符型：正常结果，可以用字符和数赋值
    整型：相当于记ASCII码，和char没有本质区别，可以用字符和数赋值
    8.2
    printf中不使用%c改用%d即可
    8.3
    不是。char为-128~127；unsigned char为0~255.超出范围的输入数值会发生截取，只取最低一个字节。
    int范围-2147483648~2147483647 4字节远大于char 但是由于ASCII码表有限 故实际上在输出字符方面功能未必比char好多少
    即只有大小在char范围内的可以互相替代
    */
}
int main()
{
    Test_Of_Sizeof();
    DataOverflow();
    Usage_Of_Operator();
    rRound();
    SplitNum();
    Test_On_Book();
    system("pause");
    return 0;
}