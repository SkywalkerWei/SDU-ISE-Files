#include<stdio.h>
#include<windows.h>
static void swap(int *a, int *b)//体会数组：交换变量
{
    int temp=*a;
    *a=*b;
    *b=temp;
}
static void input(int a[], int n)//体会数组：下标输入
{
    for(int i=0;i<n;i++) scanf("%d",&a[i]);
}
static void output(int a[], int n)//体会数组：指针输出
{
    int *p;
    for(p=a;p<a+n;p++) printf("%d ",*p);
    printf("\n");
}
//以上三个函数用于减少重复劳动
void SortTenChoose()//C6T2选择排序
{
    int arr[10],minIndex;
    printf("Input array:\n");
    input(arr,10);
    for(int i=0;i<9;i++)
    {
        minIndex=i;
        for(int j=i+1;j<10;j++) if(arr[j]<arr[minIndex]) minIndex=j;
        swap(&arr[i],&arr[minIndex]);
    }
    printf("\nSorted array:\n");
    output(arr,10);
    printf("\n");
    system("pause");
}
void SortTenBubble()//C6T2冒泡排序
{
    int arr[10];
    printf("Input array:\n");
    input(arr,10);
    for (int i=0;i<9;i++) for(int j=0;j<9-i;j++) if(arr[j]>arr[j+1]) swap(&arr[j+1],&arr[j]);
    printf("\nSorted array:\n");
    output(arr,10);
    printf("\n");
    system("pause");
}
void Matrix()//C6T3
{
    int matrix[3][3],sum1=0,sum2=0;
    printf("Input 3x3 integer matrix's elements:\n");
    for(int i=0;i<3;i++)
    {
        for (int j=0;j<3;j++)
        {
            printf("matrix[%d][%d]:",i,j);
            scanf("%d",&matrix[i][j]);
        }
    }
    for(int i=0;i<3;i++) {
        sum1+=matrix[i][i];  // 主对角线
        sum2+=matrix[i][2-i];  // 副对角线
    }
    printf("主对角%d,副对角%d\n\n",sum1,sum2);
    system("pause");
}
void Insert()//C6T4
{
    int arr[20]={1,3,5,7,9,11,13,15,17,19},n,temp,len=10;
    printf("Former array:\n");
    input(arr,len);
    printf("\nInput a number:\n");
    scanf("%d",&n);
    for(int i=0;i<len;i++)
    {
        if(n<arr[i])
        {
            for(int j=len;j>i;j--) arr[j]=arr[j-1];
            arr[i]=n;
            len++;
            break;
        }
        if(i==len-1)
        {
            arr[len]=n;
            len++;
            break;
        }
    }
    output(arr,len);
    printf("\n");
    system("pause");
}
void Reverse()//C6T5
{
    int n;
    printf("Input the length:\n");
    scanf("%d",&n);
    int arr[n];
    printf("Input elements\n");
    input(arr,n);
    printf("Former array:\n");
    output(arr,n);
    printf("\n");
    int l=0,r=n-1;
    while(l<r)
    {
        swap(&arr[l], &arr[r]);
        l++,r--;
    }
    printf("Reversed array:\n");
    output(arr,n);
    printf("\n");
    system("pause");
}
void Stars()//C6T11
{
    for(int i=0;i<5;i++)
    {
        printf("\n ");
        for(int j=1;j<=i;j++) printf(" ");
        printf("* * * * *");
    }
    printf("\n");
    system("pause");
}
void strCat()//C6T13
{
    char s1[100],s2[100];
    printf("Input str1:\n");
    gets(s1);
    printf("Input str2:\n");
    gets(s2);
    int i=strlen(s1),j=0;
    while(s2[j]!='\0')
    {
        s1[i]=s2[j];
        i++,j++;
    }
    s1[i]='\0';
    printf("Output string:\n");
    puts(s1);
    printf("\n");
    system("pause");
}
int ExpCharArray()//体会字符数组特性
{
    char str1[20] = "Hello"; //定义并初始化一个字符数组
    char str2[20]; //定义一个字符数组
    char str3[20]; //定义一个字符数组
    printf("请输入一个字符串：\n");
    scanf("%s", str2); //使用 scanf 函数输入一个字符串
    getchar(); //吸收回车符
    printf("请输入另一个字符串：\n");
    gets(str3); //使用 gets 函数输入一个字符串
    printf("str1: %s\n", str1); //使用 printf 函数输出一个字符串
    puts(str2); //使用 puts 函数输出一个字符串
    puts(str3); //使用 puts 函数输出一个字符串
    /*
    strlen()的本质是找'\0'符的位置，多个则取第一个出现的
    c没有直接的string数据类型，使用字符数组间接实现
    char字符串相当于多一个'\0'的字符数组
    直接定义字符串 char* str=" "
    用string给char数组赋初值会在末位的下一位加'\0'，直接定义char数组的每个元素则没有此情况
    */
}
int PrintChar()
{
    int n;char c;
    printf("The amount:\n");
    scanf("%d",&n);
    printf("What to output\n");
    scanf("%c",&c);
    for(int i=1;i<=n;i++) printf("%c",c);
    printf("\n");
    system("pause");
}
int main()
{
    SortTenChoose();
    SortTenBubble();
    Matrix();
    Insert();
    Reverse();
    Stars();
    strCat();
    ExpCharArray();
    PrintChar();
    system("pause");
    return 0;
}