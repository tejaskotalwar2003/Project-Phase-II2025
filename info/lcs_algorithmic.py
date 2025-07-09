# lcs_algorithmic.py

theory_content = {
    "Level 1": """

🟢 Level 1 – Basic LCS Algorithm (2D Table Method)

🧠 What’s the Goal?  
Find the **length of the longest common subsequence** between two strings A and B.

📌 Method: Bottom-Up Dynamic Programming using a 2D Table

🛠 Steps to Implement:
1. Create a table `dp` of size (len(A)+1) × (len(B)+1)
2. Loop through both strings
3. If characters match → `dp[i][j] = 1 + dp[i-1][j-1]`
4. Else → take max of top and left: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`
5. The last cell contains the LCS length

📌 Example:
A = "ABC", B = "AC"  
👉 LCS = "AC", Length = 2

🔍 Visualization:
The table builds from top-left to bottom-right. Each cell stores the LCS length up to that point.

🎯 This level focuses on writing a clean, working algorithm to **compute the LCS length** using DP.
""",

    "Level 2": """

🟡 Level 2 – Recursive and Memoized LCS

⏱️ What’s the problem with recursion?  
It solves the same subproblems **again and again**, leading to **O(2^n)** complexity.

⚡ Memoization Fixes That:
- Store computed values using a 2D list or a dictionary
- Reduces complexity to **O(n × m)**

🧠 Recursive Formula:
- If A[i] == B[j] → `1 + LCS(i-1, j-1)`
- Else → `max(LCS(i-1, j), LCS(i, j-1))`

🛠 How to Memoize:
1. Write the basic recursive function
2. Add a memo table to remember answers
3. Check memo table before recursion

📌 Example:
```python
@lru_cache
def lcs(i, j):
    if i == 0 or j == 0: return 0
    if A[i-1] == B[j-1]: return 1 + lcs(i-1, j-1)
    return max(lcs(i-1, j), lcs(i, j-1))

🎯 This level teaches you how to optimize a recursive LCS solution for real-world usage.
"""
,
    "Level 3": """

🔴 Level 3 – Advanced LCS Algorithm and Optimizations

🧠 What’s the problem with space?
The classic DP solution builds a full table of size O(n × m), which consumes a lot of memory — especially with long strings.

💡 Can we reduce space?
Yes! Each row in the DP table depends only on the **previous row**, so we can keep just two 1D arrays.

✅ Optimization: Use Only Two Rows (or 1 if you alternate cleverly)
This reduces space from **O(n × m)** to **O(m)**.

🛠 Steps to Implement:
1. Initialize two arrays: `prev` and `curr`, each of length m+1
2. Loop through each character of string A (outer loop)
3. For each character in B (inner loop):
   - If A[i-1] == B[j-1]: `curr[j] = 1 + prev[j-1]`
   - Else: `curr[j] = max(prev[j], curr[j-1])`
4. After each row, copy `curr` into `prev`

📌 Final Result:
The LCS length is found in the **last cell of the last `curr` row** after processing all characters.

⚡ Extra Tip:
This method efficiently calculates the **length** of LCS.  
To get the actual **sequence**, you still need to use the full DP table and backtrack.

🎯 This level teaches you how to write optimized, memory-efficient LCS code that is scalable and used in real-world applications like version control, bioinformatics, and NLP.
"""
}

