#include<stdio.h>
#include<windows.h>
struct Student
{
    char name[10];
    int number;
    double score[3];
    int total;
    struct Student *next;
}Stu,Stus[5];
typedef struct node 
{
    int data;
    struct node *next;
} Node;
Node *createLinkedList(int arr[],int length);
Node *createNode(int data);
void printLinkedList(Node *head);
void insertNode(Node **head,int data);
void deleteNode(Node **head,int data);
void sortList(Node **head);
void destroyList(Node **head);
void fileOperations();
int main()
{
    scanf("%d",&Stu.name);
    Stus[0].number=1,Stus[1].number=2;
    struct Student *pointer=&Stus[1];
    printf("\n%d\n%d\n%d\n",Stus[0].number,pointer->number,sizeof(struct Student));
    system("pause");
    return 0;
}
Node *createLinkedList(int arr[],int length)
{
    Node *head=0,*tail=0;
    for(int i=0;i<length;i++)
    {
        Node *newNode=(Node*)malloc(sizeof(Node));
        newNode->data=arr[i];
        if(!head) head=newNode;
        else tail->next=newNode;
        tail=newNode;
    }
    if(tail) tail->next=0;
    return head;
}
Node *createNode(int data)
{
    Node *newNode=(Node*)malloc(sizeof(Node));
    newNode->data=data;
    newNode->next=0;
    return newNode;
}
void printLinkedList(Node *head)
{
    if(!head)
    {
        printf("error: Empty linked list.\n");
        return ;
    }
    Node *cur=head;
    do
    {
        printf("%d ",cur->data);
        cur=cur->next;
    }while(cur!=head);
    printf("\n");
}
void insertNode(Node **head,int data)
{
    Node *newNode=createNode(data);
    if(!*head) *head=newNode;
    else
    {
        Node *temp=*head;
        while(temp->next) temp=temp->next;
        temp->next=newNode;
    }
}
void deleteNode(Node **head,int data)
{
    if(*head==0) return;
    Node *temp=*head,*prev=0;
    if(temp&&temp->data==data)
    {
        *head=temp->next;
        free(temp);
        return;
    }
    while(temp&&temp->data!=data)
    {
        prev=temp;
        temp=temp->next;
    }
    if(temp)
    {
        prev->next=temp->next;
        free(temp);
    }
    else return ;
}
void sortList(Node **head)
{
    if(!*head) return;
    Node *current=*head,*i=0;
    int temp;
    while(current)
    {
        i=current->next;
        while(i)
        {
            if(current->data>i->data)
            {
                temp=current->data;
                current->data=i->data;
                i->data=temp;
            }
            i=i->next;
        }
        current=current->next;
    }
}
void destroyList(Node **head)
{
    Node *current=*head,*next=0;
    while(current)
    {
        next=current->next;
        free(current);
        current=next;
    }
    *head=0;
}
void fileOperations()
{
    char filename[10],str1[10];
    scanf("%s",filename);
    FILE *file1=fopen(filename,"w+");
    fgetc(file1);
    fgets(str1,10,file1);
    printf("%s\n",str1);
    MessageBox(NULL,str1,"输出到显示器上",MB_OK);
    fputc("#",file1);
    fputs(str1,file1);
    fclose(file1);
}