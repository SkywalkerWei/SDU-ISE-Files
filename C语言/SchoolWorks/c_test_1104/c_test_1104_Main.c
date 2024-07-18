#include"c_test_1104_Ans.h"

/*T3
局部变量、静态局部变量、全局变量与静态全局变量:

普通的局部变量定义的时候直接定义或者在前面加上auto;
每次调用时，在进入函数后都会创造一个新的变量，调用结束时同时删除本次创造的这个变量。这就是局部变量的整个生命周期。
下次再调用该函数时，又会重新创造一个，经历整个程序运算，最终在函数运行完退出时再次被清除。

静态局部变量定义时前面加static关键字。
静态局部变量定义在函数内部，定义时加static关键字来标识，所在的函数在调用多次时，只有第一次才经历变量定义和初始化，以后多次在调用时不再定义和初始化，而是维持之前上一次调用时执行后这个变量的值。
本次接着来使用。在函数退出时它不释放，保持其值等待函数下一次被调用。
下次调用时不再重新创造和初始化该变量，而是直接用上一次留下的值为基础来进行操作。
静态局部变量的这种特性，和全局变量非常类似。它们的相同点是都创造和初始化一次，以后调用时值保持上次的不变。不同点在于作用域不同。

全局变量 定义在函数外面的变量。
普通全局变量 普通全局变量就是平时使用的，定义前不加任何修饰词。可以在各个文件中使用，可以在项目内别的.c文件中被看到，不能重名。
静态全局变量 静态全局变量用来解决重名问题，定义时在定义前加static关键字，只在本文件内使用，在别的文件中不使用。这样就不用担心重名问题。
所以静态全局变量用于缩小其可视范围
跨文件引用全局变量(extern)是说，在一个程序的多个.c源文件中，可以在一个.c文件中定义全局变量,并且可以在别的另一个.c文件中引用该变量（引用前要声明）
*/

/*T4
对头文件的编写和引用
当使用<>来指定包含的头文件时，编译器会从系统头文件库中进行查找，而使用""来包含的头文件，编译器将会从当前程序目录进行查找。
在include时，被包含文件可以是绝对路径，也可以是相对路径，总之，只要头文件的存放路径与当前源文件的关系正确即可。
include不仅仅能包含.h类型的头文件，可以包含任意类型的文件，例如包含一个.c文件。
实际上#include类似宏定义，编译时是把#include所包含的文件中的内容直接复制到#include所在的位置并替换#include语句
*/

int main()
{
    double s;
    int n,length;
    printf("Input the number:\n");
    scanf("%d",&n);
    printf("\n");
    printf("%d\n",isPrime(n));
    system("pause");
    printf("Input the score:\n");
    scanf("%lf",&s);
    printf("\n");
    TransScore(s);
    system("pause");
    printf("Input the length of array:");
    scanf("%d", &length);
    printf("\n");
    int arr[length];
    inputArray(arr,length);
    printArray(arr,length);
    system("pause");
    sortArray(arr,length);
    printf("Sorted:");printArray(arr,length);system("pause");
    printf("The maximum is:%d\n",findMax(arr,length));system("pause");
    reverseArray(arr,length);
    printf("Reversed:");printArray(arr,length);
    system("pause");
    return 0;
}