<<<<<<< HEAD
#include<stdio.h>
#include<windows.h>
void swap(int *a,int *b)
{
    int t;
    t=*a;*a=*b;*b=t;
}
void QuickSortUp(int a[],int begin,int end)
{
    if(begin>=end) return;
    int l=begin,r=end,key=begin;
    while(l<r)
    {
        while(l<r&&a[key]<=a[r]) r--;
        while(l<r&&a[key]>=a[l]) l++;
        swap(&a[l],&a[r]);//找左边比key大的和右边比key小的互换，直到相遇
    }//相遇之后把key换到相遇点
    swap(&a[key],&a[l]);
    key=l;
    QuickSortUp(a,begin,key-1);
    QuickSortUp(a,key+1,end);
}
void QuickSortDown(int a[],int begin,int end)
{
    if(begin>=end) return;
    int l=begin,r=end,key=begin;
    for(;l<r;)
    {
        for(;l<r&&a[r]<=a[key];r--);
        for(;l<r&&a[l]>=a[key];l++);
        swap(&a[l],&a[r]);
    }
    swap(&a[key],&a[r]);
    key=r;
    QuickSortDown(a,begin,key-1);
    QuickSortDown(a,key+1,end);
}
void QuickSortUp_Pro(int *a,int begin,int end)//快排优化：挖坑法
{
    if(begin>=end) return;
    int l=begin,r=end,key=*(a+begin),pit=begin;//把key取出来变成坑
    for(;l<r;)
    {
        for(;l<r&&a[r]>=key;r--);
        a[pit]=a[r];pit=r;
        for(;l<r&&a[l]<=key;l++);
        a[pit]=a[l];pit=l;
    }
    a[pit]=key;
    QuickSortUp_Pro(a,begin,pit-1);
    QuickSortUp_Pro(a,pit+1,end);
}
/*
void QuickSort_Pointer(int *a,int begin,int end)
{
    if(begin>=end) return;
    int key=begin,
}
*/
int main()
{
    int a[10];
    for(int i=0;i<10;i++) scanf("%d",&a[i]);
    QuickSortUp(a,0,9);
    for(int i=0;i<10;i++) printf("%d ",a[i]);
    printf("\n");
    QuickSortDown(a,0,9);
    for(int i=0;i<10;i++) printf("%d ",a[i]);
    printf("\n");
    QuickSortUp_Pro(a,0,9);
    for(int i=0;i<10;i++) printf("%d ",a[i]);
    system("pause");
=======
#include<stdio.h>
#include<windows.h>
void swap(int *a,int *b)
{
    int t;
    t=*a;*a=*b;*b=t;
}
void QuickSortUp(int a[],int begin,int end)
{
    if(begin>=end) return;
    int l=begin,r=end,key=begin;
    while(l<r)
    {
        while(l<r&&a[key]<=a[r]) r--;
        while(l<r&&a[key]>=a[l]) l++;
        swap(&a[l],&a[r]);//找左边比key大的和右边比key小的互换，直到相遇
    }//相遇之后把key换到相遇点
    swap(&a[key],&a[l]);
    key=l;
    QuickSortUp(a,begin,key-1);
    QuickSortUp(a,key+1,end);
}
void QuickSortDown(int a[],int begin,int end)
{
    if(begin>=end) return;
    int l=begin,r=end,key=begin;
    for(;l<r;)
    {
        for(;l<r&&a[r]<=a[key];r--);
        for(;l<r&&a[l]>=a[key];l++);
        swap(&a[l],&a[r]);
    }
    swap(&a[key],&a[r]);
    key=r;
    QuickSortDown(a,begin,key-1);
    QuickSortDown(a,key+1,end);
}
void QuickSortUp_Pro(int *a,int begin,int end)//快排优化：挖坑法
{
    if(begin>=end) return;
    int l=begin,r=end,key=*(a+begin),pit=begin;//把key取出来变成坑
    for(;l<r;)
    {
        for(;l<r&&a[r]>=key;r--);
        a[pit]=a[r];pit=r;
        for(;l<r&&a[l]<=key;l++);
        a[pit]=a[l];pit=l;
    }
    a[pit]=key;
    QuickSortUp_Pro(a,begin,pit-1);
    QuickSortUp_Pro(a,pit+1,end);
}
/*
void QuickSort_Pointer(int *a,int begin,int end)
{
    if(begin>=end) return;
    int key=begin,
}
*/
int main()
{
    int a[10];
    for(int i=0;i<10;i++) scanf("%d",&a[i]);
    QuickSortUp(a,0,9);
    for(int i=0;i<10;i++) printf("%d ",a[i]);
    printf("\n");
    QuickSortDown(a,0,9);
    for(int i=0;i<10;i++) printf("%d ",a[i]);
    printf("\n");
    QuickSortUp_Pro(a,0,9);
    for(int i=0;i<10;i++) printf("%d ",a[i]);
    system("pause");
>>>>>>> d9c4b397012ef527c9898b2e866606c209048376
}