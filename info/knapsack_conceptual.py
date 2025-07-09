

## ✅ 3. `knapsack_conceptual.py`


theory_content = {
    "Level 1": """

🟢 Level 1 – What is the Knapsack Problem?

📘 **The Core Idea**:  
You have a bag (knapsack) with limited capacity and must choose items with weights and values to **maximize total value**.

🔢 **Inputs**:
- List of items with weight and value
- Knapsack capacity

💡 **Why 0/1?**  
You can either take an item **entirely** or not at all.

⚖️ **Key Concept**:
Find best combo of items without exceeding weight.

🎯 This level helps understand the **basic purpose and inputs** of the 0/1 Knapsack problem.
""",

    "Level 2": """

🟡 Level 2 – Inclusion/Exclusion and DP Thinking

📘 **Optimal Substructure**:
Every decision depends on subproblems:  
- Include the item  
- Exclude the item

🧠 **Overlapping Subproblems**:
Many sub-cases repeat — perfect for memoization.

🧮 **Why Greedy Fails**:
Choosing highest value or value/weight ratio **doesn't guarantee** optimal answer in 0/1 Knapsack.

🎯 This level develops the **conceptual logic** behind how dynamic programming solves the Knapsack.
""",

    "Level 3": """

🔴 Level 3 – Deep Theory and Generalizations

🧠 **Why NP-Complete?**  
No polynomial-time solution known for large input sizes (though DP helps in pseudo-polynomial time).

💡 **Generalizations**:
- Unbounded Knapsack
- Multi-knapsack
- Multi-objective (e.g., weight & volume)

📉 **Trade-Off Visualization**:
DP tables represent **value vs. capacity** trade-offs.

🔁 **Backtracking**:
Used to recover which items gave the optimal solution.

🎯 This level builds deep understanding of **problem complexity and generalizations**.
"""
}
