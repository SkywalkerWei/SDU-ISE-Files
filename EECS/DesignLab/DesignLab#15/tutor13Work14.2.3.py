import lib601.util as util
class MyClass:
    def __init__(self, v):
        self.v = v

##def lotsOfClass(n, v):
##    result = []
##    for i in range(n):
##        result.append(MyClass(v))
##    return result
##
##class10 = lotsOfClass(10, 'oh') 
##class10[0].v = 'no' 

##def lotsOfClass(n, v):
##    def f(i):
##        return MyClass(v)
##    return util.makeVectorFill(n, f)
##
##class10 = lotsOfClass(10, 'oh') 
##class10[0].v = 'no'

def lotsOfClass(n, v):
    one = MyClass(v)
    result = []
    for i in range(n):
        result.append(one)
    return result
class10 = lotsOfClass(10, 'oh') 
class10[0].v = 'no'
print class10[3].v
        
