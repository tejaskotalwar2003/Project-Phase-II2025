mcq_questions = {

"Level 1": [
{
"question": "Which of the following is required to implement 0/1 Knapsack?",
"options": ["Weight, value arrays and capacity", "Only value array", "Only weight array", "Queue and stack"],
"answer": "Weight, value arrays and capacity"
},
{
"question": "Which programming construct is mainly used in iterative DP Knapsack?",
"options": ["Nested for-loops", "if-else", "while loop", "Switch case"],
"answer": "Nested for-loops"
},
{
"question": "What is the time complexity of a 2D DP Knapsack implementation?",
"options": ["O(nW)", "O(n)", "O(n^2)", "O(2^n)"],
"answer": "O(nW)"
},
{
"question": "Which condition must be checked before including an item?",
"options": ["wt[i] <= w", "val[i] >= 0", "i < w", "value is odd"],
"answer": "wt[i] <= w"
},
{
"question": "What is the return type of most Knapsack functions?",
"options": ["int", "string", "boolean", "tuple"],
"answer": "int"
},
{
"question": "Which array element holds the final result in 2D DP?",
"options": ["dp[n][W]", "dp[0][0]", "dp[1][1]", "dp[n-1][W-1]"],
"answer": "dp[n][W]"
},
{
"question": "How is the base case initialized in iterative DP?",
"options": ["dp[i][0] = 0 and dp[0][j] = 0", "All dp[i][j] = 1", "dp[1][0] = 1", "No base case"],
"answer": "dp[i][0] = 0 and dp[0][j] = 0"
},
{
"question": "What does the 'val[i]' represent in implementation?",
"options": ["Value of item i", "Weight of item i", "Index", "Capacity"],
"answer": "Value of item i"
},
{
"question": "Which Python structure is best for DP table?",
"options": ["List of lists", "Set", "Dictionary", "Tuple"],
"answer": "List of lists"
},
{
"question": "Knapsack function is usually called with how many parameters?",
"options": ["4", "1", "2", "6"],
"answer": "4"
},
{
"question": "Which condition defines the recursion stop?",
"options": ["n == 0 or W == 0", "i == j", "i < 0", "W < 0"],
"answer": "n == 0 or W == 0"
},
{
"question": "Knapsack solution is built using which logic?",
"options": ["Max of include vs exclude", "Only include", "Only exclude", "Min of two options"],
"answer": "Max of include vs exclude"
},
{
"question": "Which file would most likely contain knapsack code?",
"options": ["knapsack.py", "main.html", "style.css", "app.json"],
"answer": "knapsack.py"
},
{
"question": "Which statement adds an item to Knapsack value?",
"options": ["val[i-1] + knapsack(...)", "val[i-1] - W", "wt[i] = 0", "dp[i][j] = 0"],
"answer": "val[i-1] + knapsack(...)"
},
{
"question": "Which library helps add memoization to a function in Python?",
"options": ["functools", "math", "random", "json"],
"answer": "functools"
}
],

"Level 2": [
{
"question": "Which recursive call represents including an item?",
"options": ["val[i-1] + knapsack(n-1, W-wt[i-1])", "knapsack(n-1, W)", "val[i] + W", "W - wt[i]"],
"answer": "val[i-1] + knapsack(n-1, W-wt[i-1])"
},
{
"question": "What is the goal of the knapsack() recursive function?",
"options": ["Return max value", "Return weight", "Return list", "Print items"],
"answer": "Return max value"
},
{
"question": "Which decorator applies caching to recursive function?",
"options": ["@lru_cache", "@memoize", "@staticmethod", "@loop"],
"answer": "@lru_cache"
},
{
"question": "How many recursive calls are made per item?",
"options": ["2", "1", "3", "0"],
"answer": "2"
},
{
"question": "What happens if an item doesn't fit?",
"options": ["Skip it", "Add it anyway", "Double value", "End program"],
"answer": "Skip it"
},
{
"question": "Which line represents exclusion in recursion?",
"options": ["knapsack(n-1, W)", "knapsack(n, W-wt[i])", "return 0", "wt[i] + val[i]"],
"answer": "knapsack(n-1, W)"
},
{
"question": "Which variable reduces in both include/exclude?",
"options": ["n", "val", "wt", "W"],
"answer": "n"
},
{
"question": "How do you determine if recursion is correct?",
"options": ["Check base case and recurrence", "Loop count", "Memory size", "Print values only"],
"answer": "Check base case and recurrence"
},
{
"question": "Which of these causes infinite recursion?",
"options": ["Missing base case", "Memoization", "Correct return", "dp table"],
"answer": "Missing base case"
},
{
"question": "Which version is preferred for large constraints?",
"options": ["DP", "Recursion", "Greedy", "Search"],
"answer": "DP"
},
{
"question": "How is table filled in bottom-up DP?",
"options": ["Row-wise", "Column-wise", "Random", "Graph"],
"answer": "Row-wise"
},
{
"question": "Knapsack problem follows which structure?",
"options": ["Optimal substructure", "Greedy", "Tree", "Stack"],
"answer": "Optimal substructure"
},
{
"question": "Which loop nesting is used in bottom-up DP?",
"options": ["for i in range(n+1): for j in range(W+1)", "for j in range(W): for i in range(n)", "while loop", "if-else"],
"answer": "for i in range(n+1): for j in range(W+1)"
},
{
"question": "When using memoization, what stores results?",
"options": ["Dictionary or array", "Set", "List of strings", "File"],
"answer": "Dictionary or array"
},
{
"question": "How are recursive and DP solutions related?",
"options": ["DP is memoized recursion", "They are unrelated", "Recursion is faster", "DP uses greedy"],
"answer": "DP is memoized recursion"
}
],

"Level 3": [
{
"question": "What optimization reduces space from O(nW) to O(W)?",
"options": ["1D DP", "Memoization", "Sorting", "Recursion"],
"answer": "1D DP"
},
{
"question": "How do you update the DP array in 1D space optimization?",
"options": ["From right to left", "From left to right", "Top to bottom", "In-place"],
"answer": "From right to left"
},
{
"question": "Which problem property allows bottom-up DP implementation?",
"options": ["Overlapping subproblems", "Divide and conquer", "Parallelism", "Sorting"],
"answer": "Overlapping subproblems"
},
{
"question": "Which algorithm gives exact solution to 0/1 Knapsack?",
"options": ["DP", "Greedy", "DFS", "Brute-force"],
"answer": "DP"
},
{
"question": "Which form is best when memory is limited?",
"options": ["1D DP", "2D DP", "Recursive with memo", "Naive recursion"],
"answer": "1D DP"
},
{
"question": "Which language feature optimizes recursion in Python?",
"options": ["functools.lru_cache", "random.shuffle", "dict", "lambda"],
"answer": "functools.lru_cache"
},
{
"question": "Which array holds the final result in 1D DP?",
"options": ["dp[W]", "dp[0]", "dp[n]", "dp[n-1]"],
"answer": "dp[W]"
},
{
"question": "What type of traversal is required in 1D DP loop?",
"options": ["Reverse order", "Sorted order", "Ascending order", "Random order"],
"answer": "Reverse order"
},
{
"question": "Which method is fastest for large constraints?",
"options": ["1D DP", "2D DP", "Memoization", "Brute-force"],
"answer": "1D DP"
},
{
"question": "Which method may lead to stack overflow?",
"options": ["Recursive without memoization", "1D DP", "Tabulation", "Sorting"],
"answer": "Recursive without memoization"
},
{
"question": "What is the size of the DP array in 1D optimization?",
"options": ["W+1", "n+1", "nW", "n"],
"answer": "W+1"
},
{
"question": "What is the value stored at dp[w] after update?",
"options": ["Max profit for capacity w", "Weight", "Remaining space", "Loop index"],
"answer": "Max profit for capacity w"
},
{
"question": "Which property makes DP applicable?",
"options": ["Optimal substructure", "Backtracking", "Greedy", "Parallelism"],
"answer": "Optimal substructure"
},
{
"question": "Can 1D DP be used for tracking actual items?",
"options": ["Yes with extra data", "No", "Only in Java", "Only for sorted weights"],
"answer": "Yes with extra data"
},
{
"question": "Which method is fastest and most memory-efficient?",
"options": ["1D DP", "Recursive with memo", "2D DP", "Greedy"],
"answer": "1D DP"
}
]
}
tf_questions = {
"Level 1": [
{"question": "Knapsack implementation requires weight, value arrays and capacity.", "answer": True},
{"question": "Knapsack is typically solved using if-else conditions only.", "answer": False},
{"question": "The dp[n][W] cell contains the final result in 2D DP.", "answer": True},
{"question": "All dp[i][0] and dp[0][j] are initialized to 0 in iterative DP.", "answer": True},
{"question": "A single item can be used multiple times in 0/1 Knapsack.", "answer": False},
{"question": "Nested loops are commonly used in bottom-up implementation.", "answer": True},
{"question": "val[i] represents the weight of the ith item.", "answer": False},
{"question": "Knapsack can return maximum achievable value within capacity.", "answer": True},
{"question": "The Python function knapsack() usually takes 4 parameters.", "answer": True},
{"question": "The base case for recursion is W == 0 or n == 0.", "answer": True},
{"question": "All weights and values must be sorted before implementation.", "answer": False},
{"question": "DP uses a matrix to store intermediate values.", "answer": True},
{"question": "knapsack.py is a good name for a file containing the implementation.", "answer": True},
{"question": "Knapsack implementation is part of dynamic programming.", "answer": True},
{"question": "Knapsack returns the total weight of selected items.", "answer": False}
],

"Level 2": [
{"question": "In recursion, we must consider both include and exclude cases.", "answer": True},
{"question": "val[i-1] + knapsack(n-1, W-wt[i-1]) represents exclusion logic.", "answer": False},
{"question": "Memoization helps reduce recomputation in recursion.", "answer": True},
{"question": "@lru_cache in Python is used for caching function outputs.", "answer": True},
{"question": "Recursion always performs faster than DP.", "answer": False},
{"question": "An item is skipped if it doesn’t fit into remaining capacity.", "answer": True},
{"question": "Missing base cases can cause infinite recursion.", "answer": True},
{"question": "Bottom-up DP fills the table using nested loops.", "answer": True},
{"question": "Recursive implementation can exceed stack limit for large n.", "answer": True},
{"question": "In bottom-up, the DP table is filled randomly.", "answer": False},
{"question": "Knapsack solution uses optimal substructure.", "answer": True},
{"question": "Knapsack with DP is more efficient than naive recursion.", "answer": True},
{"question": "Memoization is used only in bottom-up approach.", "answer": False},
{"question": "wt[i-1] must be ≤ capacity W to include an item.", "answer": True},
{"question": "DP table can be used to trace back selected items.", "answer": True}
],

"Level 3": [
{"question": "1D DP reduces space complexity to O(W).", "answer": True},
{"question": "1D DP updates the same array from left to right.", "answer": False},
{"question": "Space optimization can cause incorrect results if not reversed.", "answer": True},
{"question": "The dp[W] cell stores the final result in 1D DP.", "answer": True},
{"question": "1D DP performs slower than 2D DP for all cases.", "answer": False},
{"question": "functools.lru_cache helps optimize recursion in Python.", "answer": True},
{"question": "1D DP overwrites useful values if updated incorrectly.", "answer": True},
{"question": "Top-down recursion with memoization is faster than brute-force.", "answer": True},
{"question": "Bottom-up DP builds from smallest subproblems to full capacity.", "answer": True},
{"question": "Knapsack's structure is not suitable for dynamic programming.", "answer": False},
{"question": "Greedy method gives exact results for 0/1 Knapsack.", "answer": False},
{"question": "Knapsack implementation often requires nested loops or recursion.", "answer": True},
{"question": "Space optimization should only be done if memory is a concern.", "answer": True},
{"question": "You can reconstruct the item choices from a 1D DP array alone.", "answer": False},
{"question": "Knapsack is a classic example of DP with space optimization.", "answer": True}
]
}