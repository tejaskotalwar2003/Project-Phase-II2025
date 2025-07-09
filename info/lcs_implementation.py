# lcs_implementation.py

theory_content = {
    "Level 1": """

🟢 Level 1 – Step-by-Step LCS Using a 2D Table

💡 Goal: Find the length of the longest subsequence common to two strings A and B.

📘 Basic Idea:
We build a 2D table (DP matrix) to track LCS lengths between all prefixes of the strings.

🛠 Algorithm Steps:
1. Create a table of size (len(A)+1) × (len(B)+1)
2. Initialize first row and column with zeros
3. Loop through both strings:
   - If characters match → add 1 to the value from top-left diagonal
   - Else → take max of top and left cells
4. The last cell contains the LCS length
5. (Optional) Backtrack to get the actual LCS string

📌 Example:
A = "ABC", B = "AC"  
Table result:  
👉 LCS length = 2  
👉 Sequence = "AC"

🎯 This level walks you through the basic dynamic programming implementation of LCS.
""",

    "Level 2": """

🟡 Level 2 – Recursive + Memoized LCS

🧠 What’s wrong with plain recursion?
It recalculates the same subproblems again and again → exponential time.

💡 Solution: Use memoization to store answers to subproblems.

🛠 Steps:
1. Write a recursive function: lcs(i, j)
2. If A[i] == B[j]: → return 1 + lcs(i-1, j-1)
3. Else → return max(lcs(i-1, j), lcs(i, j-1))
4. Store results in a memo table (2D array or dictionary)

📌 Python Example:
```python
@lru_cache
def lcs(i, j):
    if i == 0 or j == 0:
        return 0
    if A[i-1] == B[j-1]:
        return 1 + lcs(i-1, j-1)
    return max(lcs(i-1, j), lcs(i, j-1))

🎯 This level shows how to improve time complexity from O(2^n) to O(n × m) using memoization.
""",
    "Level 3": """

🔴 Level 3 – Space-Efficient LCS + Reconstructing the Sequence

💥 Problem:
The full 2D table consumes O(n × m) space — too much for large inputs.

✅ Solution: Use only two 1D arrays → current and previous rows

🛠 Space Optimization Strategy:
1. Create two arrays of size m+1
2. For each row in A:
   - Fill current[] using values from previous[]
   - Swap current and previous at the end of the row
3. Final LCS length is in the last cell

📌 Python Sketch:
```python
def lcs_length(A, B):
    m, n = len(A), len(B)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if A[i - 1] == B[j - 1]:
                curr[j] = 1 + prev[j - 1]
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]

💬 What about the actual sequence?
If you need to reconstruct the full LCS string, you’ll still need the full 2D table and a backtracking step.

📦 Extra Insight:

This method is ideal when you're only interested in the length of LCS

In real-world systems where memory is constrained (e.g., mobile devices), this technique is highly efficient

🎯 This level prepares you to build efficient, production-grade LCS solutions for large-scale inputs.
"""
}