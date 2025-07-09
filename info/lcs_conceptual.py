# lcs_conceptual.py

theory_content = {
    "Level 1": """

🟢 Level 1 – What is LCS?  
👀 LCS stands for **Longest Common Subsequence**.

📘 **What is a subsequence?**  
A subsequence is a group of characters that appear in the same order in both strings, but **not necessarily together**.

✅ **Difference from substring**:  
- Substring: Must be continuous (e.g., "ABC" in "ABCDEF")  
- Subsequence: Can skip characters (e.g., "ACE" in "ABCDE")

📌 **Example**:  
A = "ABCDEF", B = "AEBDF"  
👉 LCS = "ABDF"

🧩 **Why is LCS important?**  
- Helps measure **similarity** between two sequences  
- Used in **spell checkers**, **file comparison tools**, and **DNA sequence analysis**

🛠 **Beginner's Approach**:
- Build a 2D table of size (len(A)+1) × (len(B)+1)
- Fill it using simple comparison rules
- Last cell gives LCS length
- Backtrack to find the sequence

🎯 This level focuses on basic understanding and examples of what LCS means and why it matters.
""",

    "Level 2": """

🟡 Level 2 – Deep Dive into LCS as a DP Problem

💡 **Problem with recursion**:  
If you write a basic recursive solution to find LCS, it works — but only for small strings. Why? Because it **recomputes the same subproblems** again and again.

🧠 **Solution: Memoization**
- Store already-computed answers in a **dictionary** or a **2D array**
- Reuse values to save time

📌 **Example**:
A = "AXYT", B = "AYZX"  
👉 LCS = "AY"

🧩 **Dynamic Programming Insight**:
LCS is a classic **DP problem** because:
- It has **overlapping subproblems** (same i,j combinations come up again)
- It follows **optimal substructure** (LCS of A and B depends on LCS of subparts)

🛠 **How to implement with recursion + memo**:
1. If last characters match → 1 + LCS of remaining strings  
2. Else → max(LCS of A[:-1], B) and LCS of A, B[:-1])  
3. Store every result in a memo table

🧪 **Outcome**:
You reduce complexity from exponential to **O(n × m)**, where n and m are lengths of the two strings.

🎯 This level helps understand *how* LCS is solved efficiently using the principles of dynamic programming.
""",

    "Level 3": """

🔴 Level 3 – Optimizing and Extending the LCS Concept

🔍 **Why Optimize Further?**
The classic DP solution takes **O(n × m)** space. For **large inputs** (e.g., DNA, logs, essays), that’s too much memory!

⚡ **Trick**: We only need the **previous row** and the **current row** during table filling.

📌 **Space Optimization**:
- Use **two 1D arrays** instead of one 2D matrix
- Update values in-place while iterating
- Final value of LCS will be in the last cell of the last row

📌 **Example**:
A = "ABCBDAB", B = "BDCABA"  
👉 LCS could be "BDAB" or "BCBA" — both are correct

🧠 **Beyond the Basics**:
- **LCS isn’t always unique** — multiple sequences can be valid
- You can also modify LCS to:
  - Compare **three strings**
  - Be **case-insensitive**
  - Work with **fuzzy matches**

🔬 **Real-world Relevance**:
- LCS powers **Git diff**, **bioinformatics tools**, and **language processing**
- Advanced tools apply LCS logic to:
  - Detect similarity
  - Filter duplicates
  - Compare structure in intelligent ways

🎯 This level prepares you for real-world applications, performance tuning, and deeper algorithmic understanding of LCS.
"""
}
