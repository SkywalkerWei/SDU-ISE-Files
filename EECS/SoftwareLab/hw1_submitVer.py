import string
import operator

class BinaryOp:
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __str__(self):
        return f'{self.opStr}({str(self.left)},{str(self.right)})'
    __repr__ = __str__
    
    def eval(self, env):
        try:
            left_val = self.left.eval(env) if hasattr(self.left, 'eval') else self.left # 这里为什么这么写：报错很多次Error: 'float' object has no attribute 'eval'，这样写可以笨办法解决问题
            right_val = self.right.eval(env) if hasattr(self.right, 'eval') else self.right
            
            if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)): # 这里这么写的原因类似，isinstance用来判断是不是已存在的变量
                return self.compute(left_val, right_val)
            return self.__class__(left_val, right_val)
        except RecursionError: # 防止无限递归
            return self

    def compute(self, left, right):
        raise NotImplementedError("Subclasses should implement this method.") # 这里为什么要多倒一次：因为operator直接操作无值变量报错，所以集中到父类里面处理

class Sum(BinaryOp):
    opStr = 'Sum'
    def compute(self, left, right):
        return operator.add(left, right)

class Prod(BinaryOp):
    opStr = 'Prod'
    def compute(self, left, right):
        return operator.mul(left, right)

class Quot(BinaryOp):
    opStr = 'Quot'
    def compute(self, left, right):
        return operator.truediv(left, right)

class Diff(BinaryOp):
    opStr = 'Diff'
    def compute(self, left, right):
        return operator.sub(left, right)

class Assign(BinaryOp):
    opStr = 'Assign'
    def eval(self, env):
        env[self.left.name] = self.right
        return self.right # 不return在处理无值变量时会出问题

class Number:
    def __init__(self, val):
        self.value = val
        
    def __str__(self):
        return f'Num({self.value})'
    
    __repr__ = __str__

    def eval(self, env):
        return self.value
    
class Variable:
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        return f'Var({self.name})'
    
    __repr__ = __str__

    def eval(self, env):
        if self.name not in env: # 在这里实现无值变量有值之后重新计算
            return self
        value = env[self.name]
        if isinstance(value, (int, float)):
            return value
        try:
            return value.eval(env)
        except RecursionError:
            return value
    
# class Sum(BinaryOp):
#     opStr = 'Sum'
#     def eval(self, env):
#         return operator.add(self.left.eval(env), self.right.eval(env))

# class Prod(BinaryOp):
#     opStr = 'Prod'
#     def eval(self, env):
#         return operator.mul(self.left.eval(env), self.right.eval(env))

# class Quot(BinaryOp):
#     opStr = 'Quot'
#     def eval(self, env):
#         return operator.truediv(self.left.eval(env), self.right.eval(env))

# class Diff(BinaryOp):
#     opStr = 'Diff'
#     def eval(self, env):
#         return operator.sub(self.left.eval(env), self.right.eval(env))

seps = ['(', ')', '+', '-', '*', '/', '=']

# 这是6.1

class StateMachine:
    def start(self):
        self.state = self.startState
    def step(self, _input):
        (_state, _output) = self.getNextValues(self.state, _input)
        self.state = _state
        return _output
    def transduce(self, inputs):
        self.start()
        return [self.step(_input) for _input in inputs]

class Tokenizer(StateMachine):
    def __init__(self):
        self.startState = ''

    def getNextValues(self, state, _input):
        if _input.isalnum() and self.state.isalnum():
            self.state += _input
            return self.state, self.startState
        return (self.startState, self.state) if _input == ' ' else (_input, self.state)

def tokenizeExpressions(string):
    tokenizer = Tokenizer()
    return [item for item in tokenizer.transduce(string) if item]

# 以上是6.1
# 以下是计算器本体

def tokenize(string):
    tokens, current_token = [], [] # 存储每个操作单元，current用于使多位数字保持一个整体
    for char in string.replace(" ", ""): # 去掉空格
        if char in seps: # 如果是运算符
            if current_token: # 如果已经有数字，让他出来进到tokens，清空current
                tokens.append(''.join(current_token))
                current_token = []
            tokens.append(char) # 把运算符加到tokens
        else:
            current_token.append(char) # 不是运算符，把数字缓存
    if current_token:
        tokens.append(''.join(current_token)) # 最后结尾把里面的最后一项进token
    return tokens

def parse(tokens):
    def parseExp(index):
        if index >= len(tokens): # 溢出特判
            return None, index
        token = tokens[index] # 当前位置拿出来
        if numberTok(token): # 数字，拿到数字，进下一个
            return Number(float(token)), index + 1
        if variableTok(token): # 变量，拿到变量值，进下一个
            return Variable(token), index + 1
        if token == '(': # 表达式，做运算
            left, index = parseExp(index + 1) # 这里在左括号上，所以要加一
            op, index = tokens[index], index + 1 # 在运算符上，所以加一
            right, index = parseExp(index) # 在括号左边的表达式上，这里算完了
            if tokens[index] != ')': # 缺右括号特判
                raise ValueError("Expected ')'") # 这是python自带的抛出错误类型，不会报未定义错误的
            return {'+': Sum, '-': Diff, '*': Prod, '/': Quot, '=': Assign}[op](left, right), index + 1 # 在右括号上，加一出来
        return None, index

    parsedExp, nextIndex = parseExp(0)
    if nextIndex != len(tokens):
        raise ValueError("Unexpected tokens remaining") # 这是python自带的抛出错误类型，不会报未定义错误的
    return parsedExp

def token_check(token, valid_set):
    return all(char in valid_set for char in token)

def numberTok(token):
    return token_check(token, string.digits + '.-')

def variableTok(token):
    return token[0] in string.ascii_letters

def calc():
    env = {}
    while True:
        e = input('% ')
        if e.lower() == 'exit':
            break
        try:
            parsed_exp = parse(tokenize(e))
            # print(parsed_exp)
            print(parsed_exp.eval(env))
            print('   env =', env)
        except Exception as ex:
            print(f"Error: {ex}")

if __name__ == "__main__":
    calc()