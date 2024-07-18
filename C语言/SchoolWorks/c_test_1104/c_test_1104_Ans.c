void swap(int *a, int *b)//交换变量：换指针即可，把两个变量指针的内容，即其地址，换一下
{
    int temp=*a;
    *a=*b;
    *b=temp;
}
int isPrime(int number)//T1
{
    for(int i=2;i<=sqrt(number);i++){if(number%i==0) return 0;}
    return 1;
}
void TransScore(double score)//T2
{
    printf("The score is %.1lf, the level is %c\n",score,69-(score>=60.0?((int)score==100?4:((int)(score-50.0))/10):0));
}
// 输入数组值
void inputArray(int arr[],int length)
{
    printf("Input array's elements:\n");
    for(int i=0;i<length;i++) scanf("%d", &arr[i]);
}

// 输出数组值
void printArray(int arr[],int length)
{
    printf("Output array:");
    for(int i=0;i<length;i++) printf("%d ",arr[i]);
    printf("\n");
}

// 对数组排序
void sortArray(int arr[],int length)
{
    for(int i=0;i<length-1;i++) 
    for(int j=0;j<length-i-1;j++)
    if(arr[j]>arr[j+1]) swap(&arr[j],&arr[j+1]);
}

// 找最大值
int findMax(int arr[],int length)
{
    int max=arr[0];
    for(int i=1;i<length;i++) max=arr[i]>max?arr[i]:max;
    return max;
}

// 逆序
void reverseArray(int arr[],int length)
{
    for(int i=0;i<length/2;i++) swap(&arr[i],&arr[length-i-1]);
}