# 3.1.3 纯函数状态机 
import lib601.sm as sm

class PureFunction(sm.SM):
    def __init__(self, f):
        # 初始化时传入一个函数 f
        self.f = f
        
    def startState(self):
        # 状态在题中没有给出
        return None
    
    def getNextValues(self, state, inp):
        # 将输入 inp 通过函数 f 处理，并返回状态迭代
        return (state, self.f(inp))