# fibonacci_conceptual.py

theory_content = {
    "Level 1": """

🟢 Level 1 – What is the Fibonacci Sequence?

📘 **Definition**:  
The Fibonacci sequence is a series of numbers where each number is the **sum of the two preceding ones**, starting from 0 and 1.

📌 **Formula**:  
Fib(n) = Fib(n-1) + Fib(n-2)

👶 **Base cases**:
- Fib(0) = 0
- Fib(1) = 1

🧮 **First few numbers**:  
0, 1, 1, 2, 3, 5, 8, 13, 21, ...

💡 **Key Properties**:
- Each number grows based on the two before it.
- Simple rule, powerful growth pattern.

🔍 **Applications**:
- Found in nature: spirals, petals, pinecones.
- Useful for learning recursion and algorithms.

🎯 This level helps grasp the basic idea and structure of the Fibonacci sequence and its recursive nature.
""",

    "Level 2": """

🟡 Level 2 – Understanding Recursive Patterns and DP

💡 **Problem with naive recursion**:  
It recomputes the same values multiple times → **Exponential time complexity**.

🧠 **Memoization Fix**:
- Store computed values in a **dictionary or array**.
- Saves time by avoiding repeated work.

📘 **Time complexity**:
- Naive recursion: O(2^n)
- With memoization: O(n)

🔄 **Top-down vs Bottom-up**:
- Top-down: recursion + memoization
- Bottom-up: iteration + table

🔁 **Overlapping Subproblems**:
- Fib(n) calls Fib(n-1) and Fib(n-2), which call smaller Fib() again.

🛠️ **Iterative Solution**:
- Use two variables, update in a loop.

🎯 This level builds your understanding of how Fibonacci relates to **dynamic programming** and why optimizing recursive solutions is essential.
""",

    "Level 3": """

🔴 Level 3 – Advanced Optimizations and Mathematical Insights

⚡ **Matrix Exponentiation**:
- Fib(n) can be computed in **O(log n)** using 2x2 matrix exponentiation.

🧮 **Binet’s Formula**:
- A closed-form using the **Golden Ratio (φ)**:
  
  Fib(n) = (φⁿ - (−1/φ)ⁿ) / √5  
  (where φ ≈ 1.618...)

📉 **Fast Doubling Technique**:
- Efficient divide-and-conquer method for large n.

🔁 **Tail Recursion**:
- Prevents stack overflow by passing accumulated values.

🔍 **Space Optimizations**:
- Use just two variables instead of a full array.
- Achieve **O(1)** space and **O(n)** time.

🧪 **Real-world Implications**:
- Used in algorithm analysis (e.g., AVL trees).
- Appears in real-time systems, simulations, and biological models.

🎯 This level dives into the **math, performance tricks, and scalable algorithms** that make Fibonacci both elegant and efficient.
"""
}
