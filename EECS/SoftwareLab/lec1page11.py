# 迭代

num=float(input("input:\n"))
while num != 0 :
    g=1.0
    while abs(g*g-num) > 1e-4 :
        g=(g+num/g)/2
    print("%.6f\n" % g)
    num=float(input("input:\n"))

## 二分

def sqrt_(a):
    l,r=0,a
    while l<=r:
        m=(l+r)//2
        if m**2==a:
            return m
        elif m**2<a:
            l=m+1
        else:
            r=m-1
    return r
num=int(input())
print(sqrt_(num))