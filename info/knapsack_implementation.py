# knapsack_implementation.py

theory_content = {
    "Level 1": """

🟢 Level 1 – Writing Basic Recursive and DP Code

📘 **How to start coding Knapsack**:
Start with a simple recursive function or initialize a DP table.

💡 **Recursive Version (without memoization)**:
```python
def knapsack(i, W):
    if i == 0 or W == 0:
        return 0
    if wt[i-1] > W:
        return knapsack(i-1, W)
    return max(knapsack(i-1, W), val[i-1] + knapsack(i-1, W - wt[i-1]))
⚙️ DP Table Setup (2D):

dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
🧪 What to learn here?

Basic function design

Zero-based vs one-based indexing

Understanding recursive thinking

🎯 This level introduces writing the core logic of recursive and tabular implementation.
""",

"Level 2": """
🟡 Level 2 – Full Memoization and Tabulation

⚙️ Memoized Recursive Approach:
Store answers to avoid recomputation.


from functools import lru_cache

@lru_cache(None)
def knapsack(i, w):
    if i == 0 or w == 0:
        return 0
    if wt[i-1] > w:
        return knapsack(i-1, w)
    return max(knapsack(i-1, w), val[i-1] + knapsack(i-1, w - wt[i-1]))
📘 Bottom-Up Tabulation:

for i in range(1, n+1):
    for w in range(1, W+1):
        if wt[i-1] <= w:
            dp[i][w] = max(dp[i-1][w], val[i-1] + dp[i-1][w - wt[i-1]])
        else:
            dp[i][w] = dp[i-1][w]
🧠 Enhancements:

Return selected items

Print maximum value from dp[n][W]

🎯 This level focuses on implementing full DP logic and understanding memory trade-offs.
""",

"Level 3": """
🔴 Level 3 – Space-Efficient & Advanced Code Structures

⚡ 1D Space Optimization:
Shrink DP space to a 1D array:

dp = [0] * (W + 1)
for i in range(n):
    for w in range(W, wt[i]-1, -1):
        dp[w] = max(dp[w], val[i] + dp[w - wt[i]])
📘 Track Items:
Backtrack from filled table to find the exact item set used in the optimal solution.

🧩 Advanced Implementations:

Handle large n or W (e.g., 10⁵ scale)

Class-based object-oriented design

File I/O for bulk item inputs

Handle conflicting item sets or custom constraints

🎯 This level enables professional-grade implementations ready for production or contests.
"""
}