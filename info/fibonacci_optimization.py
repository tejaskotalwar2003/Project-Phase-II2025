# fibonacci_optimization.py

theory_content = {
    "Level 1": """

🟢 Level 1 – Why Optimize Fibonacci?

💡 **What’s the issue?**  
Naive recursion is slow because:
- It recalculates the same values.
- It has **exponential time complexity**.

🚀 **First Optimization Step**:
- Use **iteration** instead of recursion.
- Store results using **variables or arrays**.

📘 **Result**:  
- Much faster
- Avoids memory stack overflow

🎯 This level explains **why** optimization is necessary and introduces **basic techniques**.
""",

    "Level 2": """

🟡 Level 2 – Efficient Strategies with DP and Space Reduction

⚙️ **Two Optimization Paths**:
1. **Top-down (memoization)**: Cache results to avoid recomputation.
2. **Bottom-up (tabulation)**: Build the answer from the base up.

📉 **Reduce Space Complexity**:
- Replace array with just two variables (`a`, `b`)
- Results in **O(1)** space instead of O(n)

🧠 **Compare Approaches**:
- Memoization uses more memory (call stack + cache)
- Iterative methods are more space-efficient

🎯 This level helps you compare and choose the **best optimization method** for different scenarios.
""",

    "Level 3": """

🔴 Level 3 – Advanced Optimization Techniques

🧮 **Matrix Exponentiation**:
- Achieves **O(log n)** time using matrix powers

🧠 **Fast Doubling**:
- Divide-and-conquer recursive method  
- Computes large Fibonacci values efficiently

⚡ **Modular Arithmetic**:
- Useful in competitive coding or when numbers are huge

💾 **Memory Tricks**:
- Tail recursion
- Sliding window
- Precomputation tables

📦 **Real-World Constraints**:
- Use lightweight methods (e.g., two variables) for **embedded or low-memory systems**

🎯 This level gives you **cutting-edge tools** for solving Fibonacci with **maximum efficiency**.
"""
}
