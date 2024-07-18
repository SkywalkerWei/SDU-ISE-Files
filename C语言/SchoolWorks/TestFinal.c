#include<stdio.h>
int getString(char *a,char **b)
{
    char *p=a;b[0]=a;
    int n=0;
    for(;*p!='\0';p++) if(*p==' ') *p='\0',n++,b[n]=p+1;
    return n+1;
}
int main()
{
    char str[100005];
    char *strPtr[1005]={0};
    gets(str);
    int num,i;
    num=getString(str,strPtr);
    for(i=0;i<num;i++) puts(strPtr[i]); 
    return 0;
}