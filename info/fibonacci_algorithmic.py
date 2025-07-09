# fibonacci_algorithmic.py

theory_content = {
    "Level 1": """

🟢 Level 1 – Writing Fibonacci with Basic Code

📘 **What is the algorithmic approach?**  
Here, we write simple code to compute Fibonacci numbers.

🔁 **Two common approaches**:
1. **Recursive function**:
   ```python
   def fib(n):
       if n == 0:
           return 0
       elif n == 1:
           return 1
       else:
           return fib(n-1) + fib(n-2)
Iterative using a loop:

a, b = 0, 1
for _ in range(n):
    a, b = b, a + b
return a
💡 When to use what:

Recursion is simple but slow for large n

Iteration is efficient and safe

🎯 This level focuses on helping you write correct, beginner-friendly implementations of Fibonacci logic.
""",
"Level 2": """
🟡 Level 2 – Improving Efficiency with Memoization and Iteration

💡 What's wrong with plain recursion?
It recalculates the same values repeatedly → slow.

🚀 Fix with memoization:
Use a cache (dictionary/array) to store previously computed results.

📘 Memoized version:


memo = {0: 0, 1: 1}
def fib(n):
    if n in memo:
        return memo[n]
    memo[n] = fib(n-1) + fib(n-2)
    return memo[n]
⚙️ Bottom-Up DP:


dp = [0, 1]
for i in range(2, n+1):
    dp.append(dp[i-1] + dp[i-2])
🔁 Two-variable method (space optimized):


a, b = 0, 1
for _ in range(n):
    a, b = b, a + b
🎯 This level helps you write faster and more memory-efficient code for Fibonacci problems using programming techniques like memoization and iteration.
""",

"Level 3": """
🔴 Level 3 – Optimizing Further with Logarithmic Time

⚡ Matrix Exponentiation:

Fibonacci can be computed in O(log n) time using:

| F(n+1)  F(n)   | = |1 1|^n
| F(n)    F(n-1) |   |1 0|
🧠 Fast Doubling:
A recursive trick to compute Fib(n) in O(log n) time:

def fib(n):
    def helper(n):
        if n == 0:
            return (0, 1)
        a, b = helper(n >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        return (d, c + d) if n & 1 else (c, d)
    return helper(n)[0]
🧮 Modular Arithmetic:
To avoid overflow for large n:

MOD = 10**9 + 7
🔄 Circular Buffer or Sliding Window:
Replace array with rotating variables for O(1) space.

🧪 Comparison of All Methods:

Naive Recursion: ❌ Slow (O(2ⁿ))

Memoization: ✅ Good (O(n))

Iteration: ✅ Faster (O(n), O(1) space)

Matrix/Double: 🚀 Best (O(log n))

🎯 This level equips you with high-performance algorithms suitable for competitive coding and real-world applications.
"""
}