def evaluate_expression(expr, env, delayed_expressions):
    expr = expr.strip()

    # 如果表达式是数字，返回浮点数值；特判负数，负号会识别为减号导致错误；运算递归栈底是单个数字
    if expr.isdigit() or (expr[0] == '-' and expr[1:].isdigit()):
        return float(expr)

    # 如果表达式是变量，返回它在环境中的值
    if expr in env:
        return env[expr]

    # 如果表达式被括号包围，解析括号内的内容
    if expr[0] == '(' and expr[-1] == ')':
        expr = expr[1:-1].strip()

        # 处理赋值表达式
        if '=' in expr:
            var, sub_expr = expr.split('=', 1)
            var = var.strip() 
            try:
                value = evaluate_expression(sub_expr, env, delayed_expressions)
                if value is not None:
                    env[var] = value
                    update_delayed_expressions(env, delayed_expressions)
                else:
                    delayed_expressions[var] = sub_expr.strip()
                    print(f"延迟计算: {var} 依赖未定义的变量")
            except KeyError:
                delayed_expressions[var] = sub_expr.strip()
                print(f"延迟计算: {var} 依赖于未定义的变量，稍后计算")
            return None

        # 处理四则运算
        for op in ['+', '-', '*', '/']:  # 全部带括号，优先级无所谓
            if op in expr:
                left_expr, right_expr = expr.split(op, 1)
                left_value = evaluate_expression(left_expr, env, delayed_expressions)
                right_value = evaluate_expression(right_expr, env, delayed_expressions)
                if left_value is None or right_value is None:
                    return None
                if op == '+':
                    return left_value + right_value
                elif op == '-':
                    return left_value - right_value
                elif op == '*':
                    return left_value * right_value
                elif op == '/':
                    if right_value == 0:
                        raise ZeroDivisionError("除数不能为零")  # 这是python内置的抛出异常类型
                    return left_value / right_value

    raise ValueError(f"无法解析的表达式: {expr}")   # 这是python内置的抛出异常类型

def update_delayed_expressions(env, delayed_expressions):
    pending_expressions = delayed_expressions.copy()  # 避免修改字典时遍历
    for var, expr in pending_expressions.items():
        try:
            value = evaluate_expression(expr, env, delayed_expressions)
            if value is not None:
                env[var] = value
                print(f"延迟计算成功: {var} = {value}")
                del delayed_expressions[var]
        except KeyError:
            continue

def calc():
    env = {}
    delayed_expressions = {}
    while True:
        try:
            expr = input('% ')
            if expr.lower() == 'exit':  # 输入 'exit' 退出程序
                break
            result = evaluate_expression(expr, env, delayed_expressions)
            if result is not None:
                print(result)
            if result is None:
                print("None")
            print(f"\tenv = {env}")
        except Exception as e:  # exception是所有异常的基类，这里捕获在try中抛出的所有异常，命名为e，在下一行报错
            print(f"错误: {e}")

calc()