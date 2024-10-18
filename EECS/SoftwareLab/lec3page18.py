def cube_root(x,epsilon=1e-4,guess=1):
    if x==0:
        m=0.0
        print('num_guesses =%d\n' % guess)
        return m
    if x==1:
        m=1.0
        print('num_guesses =%d\n' % guess)
        return m
    if x<0:
        return -cube_root(-x)
    
    l=1
    r=x
    m=(l+r)/2.0
    while abs(x-m**3)>epsilon:
        guess+=1
        if m**3>x:
            r=m
        elif m**3<x:
            l=m
        else:
            break
        m=(l+r)/2.0
        print('%.4f\n' %m)

    print('num_guesses =%d\n' % guess)
    return m
       
while input("continue?[y/n]")!='n':
    cube=float(input("input num:) "))
    print('%.4fis close to the cube root of%.4f\n' % (cube_root(cube),cube))