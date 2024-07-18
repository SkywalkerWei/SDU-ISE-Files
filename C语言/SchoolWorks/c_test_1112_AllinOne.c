#include<stdio.h>
#include<windows.h>
/*
int *p = int* p
表示p是一个指针，指向一个int类型
当*用于定义时，是标明该变量为指针类型
除此以外，*的作用是解引用
一个指针类型变量存储一个地址值，在指针量前加*调用即取出指针对应的地址中存储的值
在指针前面加 * 就是“取得这个指针指向的地址里的值”
int *p, p=&a,
定义一个指针p，p获得了a的地址，此时*p与a是等价的
*/

//减少重复劳动部分
void swap(int *a, int *b)
{
    int temp=*a;
    *a=*b;
    *b=temp;
}
void inputArray(int arr[],int length)//整个数组输入
{
    printf("Input the array's elements:\n");
    for(int i=0;i<length;i++) scanf("%d", &arr[i]);
    printf("\n");
}
void printArray(int arr[],int length)//整个数组输出
{
    printf("Output array:");
    for(int i=0;i<length;i++) printf("%d ",arr[i]);
    printf("\n");
}

//T1 全局变量实现
int T1max,T1min;//max和min会不会是保留字?
void T1global(int arr[],int length)
{
    //此处记录的是下标
    for(int i=1;i<length;i++) T1max=arr[T1max]>arr[i]?T1max:i,T1min=arr[T1min]<arr[i]?T1min:i;
}
//T1 指针实现
void T1pointer(int arr[],int length,int *mmax,int *mmin)
{
    *mmax=arr[0],*mmin=arr[0];
    for(int i=1;i<length;i++) *mmax=*mmax>arr[i]?*mmax:arr[i],*mmin=*mmin<arr[i]?*mmin:arr[i];
}
//对数组元素的引用
void referenceTest(int arr[],int length)
{
    int *p;
    /*形式1：指针 *(arr+i)形式*/
    for(int i=0;i<length;i++) printf("%d ",*(arr+i));	
    printf("\n");
    /*形式2：指针 *p形式*/
    for(p=arr;p<(arr+length);p++) printf("%d ",*p);
    printf("\n");		
    /*形式3：下标 arr[i]形式*/
    for(int i=0;i<length;i++) printf("%d ",arr[i]);		
    printf("\n");
    /*形式4：下标 p[i]形式*/
    p=arr;         //此时p与arr等价
    for(int i=0;i<length;i++) printf("%d ",p[i]);		
    printf("\n");
}
/*
向函数传递数组：
1.直接传指针：函数形如void f(int *a)；a相当于数组的首地址
2.一般方法，一个未定义大小的数组，形如void f(int arr[] <,int x>)；形参不会被检查出界问题
传递时使用数组名作为参数，形如f(arr,<x>)
定义一个*p=arr，传递p 与 直接传递arr 的效果是一样的
*/
void reverseArray(int arr[],int length)
{
    for(int i=0,j=length-1;i<j;i++,j--) swap(&arr[i],&arr[j]);
}
//以下用于实现位移，感觉做复杂了，没必要
typedef struct node 
{
    int data;
    struct node *next;//内含一个指向同类结构体的指针
} Node;
Node *createLinkedList(int arr[],int length)//返回值为指针类型
{
    Node *head=0,*tail=0;
    for(int i=0;i<length;i++)
    {
        Node *newNode=(Node*)malloc(sizeof(Node));//开一个指向结构体Node的指针newNode，指向一块新分配的内存，用来开新节点，同时设置为用于存储Node结构体
        newNode->data=arr[i];//调用newNode指向的地址的结构体的data词条
        if(!head) head=newNode;//如果是第一个，把头指针放这里
        else tail->next=newNode;//如果在中间，把尾指针指向这里然后后移尾指针 注意：tail->next等价于(*tail).next
        tail=newNode;
    }
    if(tail) tail->next=head;
    return head;//此时头指针的效果相当于数组传递参数只传递数组名，构造结束将其返回用于调用
}
void StepMoveList(Node **head, int step)
{
    if(!head) return;
    int len=1;
    for(Node *cur=*head;cur->next!=*head;cur=cur->next) len++;
    step%=len;
    if(!step) return;
    if(step>0) for(int i=0;i<step;i++) *head=(*head)->next;
    else 
    {
        step=-step;//表长n, 左移x相当于右移n-x
        for(int i=0;i<len-step;i++) *head=(*head)->next;
    }
}
void printLinkedList(Node *head)//传头的位置即可绕一圈把整个输出来
{
    if(!head)
    {
        printf("error: Empty linked list.\n");
        return ;
    }
    Node *cur=head;
    do//for走不动 原地tp
    {
        printf("%d ",cur->data);
        cur=cur->next;
    }while(cur!=head);
    printf("\n");
}
int main()
{
    int length;
    //T1 全局变量部分
    printf("Input the size of the array which needs to analyse:");
    scanf("%d", &length);
    printf("\n");
    int T1a[length];
    inputArray(T1a,length);
    T1global(T1a,length);
    printf("max=%d,min=%d\n",T1a[T1max],T1a[T1min]);
    printf("\n");
    system("pause");

    //T1 指针部分
    printf("Input the size of the array which needs to analyse:");
    scanf("%d", &length);
    printf("\n");
    int T1b[length];
    inputArray(T1b,length);
    int mmax,mmin;
    T1pointer(T1b,length,&mmax,&mmin);
    printf("max=%d,min=%d\n",mmax,mmin);
    printf("\n");
    system("pause");
    
    //T2
    printf("Input the size of the array for some of the next tests:\n");
    scanf("%d", &length);
    printf("\n");
    int T2[length];
    inputArray(T2,length);
    //测试两种引用方式
    referenceTest(T2,length);
    //反转基础功能实现
    reverseArray(T2,length);
    printf("\nThe array has been reversed:\n");
    printArray(T2,length);
    printf("\n");
    reverseArray(T2,length);//倒回去
    //int lengthTest=sizeof(T2)/sizeof(T2[0]);
    //printf("%d\n",lengthTest);
    system("pause");
    for(;;)
    {
        Node *head=createLinkedList(T2,length);
        printf("Input the movement's steps and direction:\nleft: <0\nright: >0\npress 0 to exit.\n");
        int steps;
        scanf("%d",&steps);
        if(!steps) break;
        StepMoveList(&head,steps);
        printf("\nLinked list after moving %d positions: ",steps);
        printLinkedList(head);
        system("pause");
        system("cls");
        Node *cur=head,*nnext;
        do
        {
            nnext=cur->next;
            free(cur);
            cur=nnext;
        }while(cur!=head);
        printf("Now the array has been reset:");
        printArray(T2,length);
    }
    return 0;
}