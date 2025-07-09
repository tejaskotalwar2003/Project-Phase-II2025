# lcs_optimization.py

theory_content = {
    "Level 1": """

🟢 Level 1 – Why Optimize LCS?

🤔 Problem: The classic LCS algorithm runs in O(n × m) time and space.  
This is fine for small strings, but becomes costly when the strings are large (e.g., logs, essays, DNA sequences).

💥 Real-World Example:
Comparing two strings of 10,000 characters each would require a 10,000 × 10,000 DP table — that’s 100 million entries!

🎯 Goal: Optimize for time and/or memory.

💡 Optimization Motivation:
- Faster algorithms help when comparing large texts or files in real time.
- Memory-saving approaches make LCS practical on low-resource systems or mobile devices.

🛠 This level introduces why optimization matters before diving into how it's done.
""",

    "Level 2": """

🟡 Level 2 – Time Complexity Improvements

⏱️ Standard LCS time complexity: O(n × m)  
Can we do better? In some special cases, yes.

⚡ Optimization Techniques:

1. Pruning:
   - Skip unnecessary comparisons (e.g., early exit when one string is empty)
   - Useful in greedy variants and approximate LCS

2. Bounded LCS:
   - Limit max length of LCS for speed (e.g., early cutoff for real-time apps)

3. Bit-Parallel Algorithms:
   - Replace loops with bitwise operations (useful in low-level languages like C)
   - Example: The Myers Algorithm can find LCS in sublinear time in practice

📌 Optimization Consideration:
- Not all algorithms reduce worst-case time complexity, but many improve speed in practice.

📚 Real Applications:
- Fast LCS variants power search engines, bioinformatics scanners, and real-time auto-suggestion systems.

🎯 This level explores smarter algorithms that reduce runtime or respond faster under specific conditions.
""",

    "Level 3": """

🔴 Level 3 – Space Optimization & Real-World Scaling

🧠 Problem:
O(n × m) space is too much for long strings. Let’s reduce it.

✅ Classic Optimization: Use Two Rows Instead of Full DP Table
- At any time, each cell only needs values from the previous row
- So instead of a full 2D matrix, use two 1D arrays: prev[] and curr[]
- Final LCS length is in the last cell of the last row

📌 Memory Complexity becomes: O(m)  
Where m is the length of the shorter string

💡 Beyond Two Rows:
If you only need LCS length (not the sequence), you can use even one row and overwrite it cleverly

📚 Additional Techniques:
- Hirschberg’s Algorithm: Reduces space to O(n + m) and still finds the LCS string
- Divide & Conquer DP: Recursively split problem to manage memory better

🔬 Real-World Constraints:
- Mobile apps, embedded systems, or browsers can’t afford huge RAM usage
- Even cloud systems with multiple requests need optimization for scalability

🎯 This level gives you the tools to make LCS usable on large-scale inputs in performance-critical environments.
"""
}
