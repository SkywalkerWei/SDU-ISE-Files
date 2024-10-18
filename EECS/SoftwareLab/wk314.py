import lib601.sm as sm
import lib601.gfx as gfx
import lib601.util as util

# Task 1
class BA1(sm.SM):
    startState = 0
    def getNextValues(self, state, inp):
        if inp != 0:
            newState = state * 1.02 + inp - 100
        else:
            newState = state * 1.02
        return (newState, newState)

class BA2(sm.SM):
    startState = 0
    def getNextValues(self, state, inp):
        newState = state * 1.01 + inp
        return (newState, newState)

# 这也是3.1.3的部分
class PureFunction(sm.SM):
    def __init__(self, f):
        self.f = f
    
    def startState(self):
        return None
    
    def getNextValues(self, state, inp):
        return (state, self.f(inp))

ba1 = BA1()
ba2 = BA2()
combinedAccounts = sm.Parallel2(ba1, ba2) # 并行运行，同时给两个传状态，返回值为包含两个值的元组
maxBalance = PureFunction(lambda balances: max(balances)) # 匿名函数接受返回值并max
maxAccount = sm.Cascade(combinedAccounts, maxBalance) # 将并行的返回值作为比大小的输入

input_sequence = [1000,2000,4000,8000,-1000,-5000]
#maxAccount.transduce(input_sequence,verbose=True)

# Task 2
def splitAmount(inp):
    if abs(inp) > 3000:
        return (inp, 0)
    else:
        return (0, inp)

splitter = PureFunction(splitAmount)
splitAccounts = sm.Cascade(splitter, sm.Parallel2(ba1, ba2)) # 不同的是，先分配操作，再执行
sumBalance = PureFunction(lambda balances: sum(balances)) # 同上
switchAccount = sm.Cascade(splitAccounts, sumBalance) # 同上

input_sequence = [1000,2000,4000,8000,-1000,-5000]
#switchAccount.transduce(input_sequence,verbose=True)