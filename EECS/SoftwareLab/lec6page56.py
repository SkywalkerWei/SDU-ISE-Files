memo=[0]*1001
memo[1]=1;memo[2]=1
length=0

def fibonacci(n):
    if memo[n]!=0:
        return memo[n]
    memo[n]=fibonacci(n-1)+fibonacci(n-2)
    return memo[n]

position = int(input("Enter the Fibonacci sequence position to find: "))
print(f"Fibonacci number at position {position}: {fibonacci(position)}")