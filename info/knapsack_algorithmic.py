# knapsack_algorithmic.py

theory_content = {
    "Level 1": """

🟢 Level 1 – Starting with Simple Choices

📘 **What is the algorithmic goal of Knapsack?**  
Select items with given weights and values to maximize the total value without exceeding the knapsack's capacity.

🔢 **Basic Inputs**:
- Capacity (W)
- List of items (weight, value)

🎯 **Simple decisions**:
- Can a single item fit?
- Which one of two items is better to pick?

💡 **How to decide?**
- Check if weight ≤ capacity
- Pick the one with the higher value (if only one fits)

🎓 **Example**:
Capacity = 5  
Item A: (weight=3, value=10)  
Item B: (weight=4, value=9)  
👉 Choose A (more value, both fit separately)

🎯 This level focuses on intuitive choices and comparisons of items.
""",

    "Level 2": """

🟡 Level 2 – Making Optimal Combinations

⚙️ **0/1 Knapsack Core Idea**:
We must **choose or skip** each item — not split it.

🧠 **Recursive Relation**:
```python
dp[i][w] = max(dp[i-1][w], dp[i-1][w - wt[i]] + val[i])
📘 Inclusion-Exclusion Logic:

Include item → Add value + check capacity left

Exclude item → Move to next without adding

💡 Algorithm Strategy:

Create a DP table of size (n+1) × (W+1)

Fill row by row based on the formula above

Answer is at dp[n][W]

🎯 This level builds logic for dynamic programming using item inclusion/exclusion.
""",
"Level 3": """
🔴 Level 3 – Advanced Optimization Techniques

⚡ Space Optimization:
Use a 1D DP array to reduce space from O(n×W) to O(W):

dp = [0] * (W + 1)
for i in range(n):
    for w in range(W, wt[i]-1, -1):
        dp[w] = max(dp[w], dp[w - wt[i]] + val[i])
🧠 Backtracking for Item List:
Once DP is filled, backtrack to find which items were selected.

💡 Variants:

Multi-Knapsack

Minimum item constraint

Conflict-based Knapsack

🎯 This level gives you tools to solve real-world, complex variations of Knapsack with efficient DP.
"""
}