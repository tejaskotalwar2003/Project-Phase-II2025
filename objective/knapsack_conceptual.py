mcq_questions = {
"Level 1": [
{
"question": "What does the 0/1 in 0/1 Knapsack mean?",
"options": ["Take all or none of each item", "Use binary weights", "Use fractional items", "Sort items by index"],
"answer": "Take all or none of each item"
},
{
"question": "What is the goal of the 0/1 Knapsack problem?",
"options": ["Minimize total weight", "Maximize profit without exceeding weight", "Select every item", "Use minimum number of items"],
"answer": "Maximize profit without exceeding weight"
},
{
"question": "Which of the following is true about the Knapsack problem?",
"options": ["It is solved using recursion", "It is a greedy algorithm", "It is solved using graphs", "It does not need dynamic programming"],
"answer": "It is solved using recursion"
},
{
"question": "Which data structure is used in DP implementation of Knapsack?",
"options": ["Stack", "Queue", "2D Array", "Linked List"],
"answer": "2D Array"
},
{
"question": "Which item types are used in 0/1 Knapsack?",
"options": ["Continuous", "Fractional", "Whole items only", "Strings"],
"answer": "Whole items only"
},
{
"question": "What input values are required for each item in Knapsack?",
"options": ["Weight and Profit", "Height and Width", "Size only", "None"],
"answer": "Weight and Profit"
},
{
"question": "What is the primary constraint in the Knapsack problem?",
"options": ["Total profit", "Maximum weight capacity", "Minimum value", "Number of items"],
"answer": "Maximum weight capacity"
},
{
"question": "Which of these is a valid state in Knapsack DP?",
"options": ["dp[i][j]", "dp[i+j]", "dp[i*j]", "dp[j]"],
"answer": "dp[i][j]"
},
{
"question": "What does dp[i][j] represent in 0/1 Knapsack?",
"options": ["Minimum cost", "Maximum profit for first i items and weight j", "Item index", "Item weight"],
"answer": "Maximum profit for first i items and weight j"
},
{
"question": "Which technique is used in solving Knapsack efficiently?",
"options": ["Greedy", "Dynamic Programming", "Divide and Conquer", "Sorting"],
"answer": "Dynamic Programming"
},
{
"question": "Knapsack problem falls under which algorithm category?",
"options": ["Graph Theory", "Greedy", "Dynamic Programming", "Recursion only"],
"answer": "Dynamic Programming"
},
{
"question": "Is the 0/1 Knapsack problem NP-complete?",
"options": ["Yes", "No", "Only for small n", "Only when weights are prime"],
"answer": "Yes"
},
{
"question": "What happens if total item weight exceeds capacity?",
"options": ["Ignore the item", "Remove all items", "Split the item", "Double the item value"],
"answer": "Ignore the item"
},
{
"question": "Can an item be used more than once in 0/1 Knapsack?",
"options": ["Yes", "No", "Only if value is even", "Depends on input"],
"answer": "No"
},
{
"question": "Which of the following is NOT needed for solving Knapsack?",
"options": ["Weight array", "Value array", "Capacity value", "Binary tree"],
"answer": "Binary tree"
}
],

"Level 2": [
{
"question": "Which is a subproblem in the 0/1 Knapsack recursion?",
"options": ["dp[i-1][w]", "dp[i+1][w]", "dp[i][w+1]", "dp[0][0]"],
"answer": "dp[i-1][w]"
},
{
"question": "What does the recurrence relation for Knapsack involve?",
"options": ["Min value and max weight", "Max(dp[i-1][w], val[i] + dp[i-1][w-wt[i]])", "Sum of all weights", "Sorting items"],
"answer": "Max(dp[i-1][w], val[i] + dp[i-1][w-wt[i]])"
},
{
"question": "Which base case is used in the Knapsack DP table?",
"options": ["dp[0][j] = 0", "dp[i][0] = 0", "dp[0][0] = 0", "All of these"],
"answer": "All of these"
},
{
"question": "What does it mean when dp[i][j] = dp[i-1][j]?",
"options": ["Item was included", "Item was excluded", "Item had zero value", "Array overflow"],
"answer": "Item was excluded"
},
{
"question": "What is the time complexity of the DP solution?",
"options": ["O(n)", "O(nW)", "O(n log n)", "O(2^n)"],
"answer": "O(nW)"
},
{
"question": "Which condition checks if an item can fit in the knapsack?",
"options": ["wt[i] ≤ w", "val[i] > 0", "dp[i][j] > 0", "w == 0"],
"answer": "wt[i] ≤ w"
},
{
"question": "In recursion, which two cases are considered for each item?",
"options": ["Include and Exclude", "Add and Sort", "Multiply and Store", "Max and Min"],
"answer": "Include and Exclude"
},
{
"question": "Which of these methods improves Knapsack memory usage?",
"options": ["2D to 1D conversion", "Sorting weights", "Binary search", "Using stacks"],
"answer": "2D to 1D conversion"
},
{
"question": "Why is 0/1 Knapsack not solvable using greedy method?",
"options": ["Because weights vary", "Because values are unordered", "Because greedy is not optimal", "Because recursion is banned"],
"answer": "Because greedy is not optimal"
},
{
"question": "Which programming paradigm does Knapsack best represent?",
"options": ["Greedy", "Backtracking", "Dynamic Programming", "Simulation"],
"answer": "Dynamic Programming"
},
{
"question": "What is the size of the Knapsack DP table?",
"options": ["n x W", "W x n", "n x n", "W x W"],
"answer": "n x W"
},
{
"question": "What happens when all item weights exceed capacity?",
"options": ["All dp[i][j] = 0", "All items selected", "Profit maximized", "Stack overflow"],
"answer": "All dp[i][j] = 0"
},
{
"question": "What is returned from the final cell dp[n][W]?",
"options": ["Minimum weight", "Total capacity", "Maximum profit", "Index of max item"],
"answer": "Maximum profit"
},
{
"question": "In recursive solution, what is the stopping condition?",
"options": ["n == 0 or W == 0", "val[i] == wt[i]", "wt[i] == 0", "i == n"],
"answer": "n == 0 or W == 0"
},
{
"question": "What should you do if an item doesn't fit?",
"options": ["Skip it", "Add a dummy item", "Include it anyway", "Restart the process"],
"answer": "Skip it"
}
],


}


mcq_questions["Level 3"] = [
{
"question": "Which optimization reduces space from O(nW) to O(W)?",
"options": ["Space optimization", "Time shifting", "Greedy caching", "Matrix tiling"],
"answer": "Space optimization"
},
{
"question": "How does 1D DP optimization work in Knapsack?",
"options": ["Updates array from right to left", "Skips duplicate values", "Uses modulo index", "Ignores all recursion"],
"answer": "Updates array from right to left"
},
{
"question": "Which property makes Knapsack suitable for dynamic programming?",
"options": ["Optimal substructure", "Greedy choices", "Full search space", "Heuristic pruning"],
"answer": "Optimal substructure"
},
{
"question": "Which principle explains the choice of taking or skipping an item?",
"options": ["Divide and Conquer", "Recurrence relation", "Inclusion-Exclusion", "Greedy pairing"],
"answer": "Inclusion-Exclusion"
},
{
"question": "When is Knapsack problem best solved using brute-force?",
"options": ["For small n", "Always", "For sorted items", "For fractional inputs"],
"answer": "For small n"
},
{
"question": "Which advanced variant allows item reuse?",
"options": ["Unbounded Knapsack", "Bounded Knapsack", "0/1 Knapsack", "Grouped Knapsack"],
"answer": "Unbounded Knapsack"
},
{
"question": "Which algorithm is not ideal for 0/1 Knapsack?",
"options": ["Greedy", "Dynamic Programming", "Recursion", "Memoization"],
"answer": "Greedy"
},
{
"question": "What kind of table is used in iterative DP Knapsack?",
"options": ["Bottom-up", "Top-down", "Circular", "Graph"],
"answer": "Bottom-up"
},
{
"question": "What does dp[w] represent in space-optimized Knapsack?",
"options": ["Max profit for weight w", "Remaining weight", "Item index", "Count of items"],
"answer": "Max profit for weight w"
},
{
"question": "Why does 1D DP update backward?",
"options": ["To avoid overwriting states", "To simplify loops", "To use greedy method", "To sort values"],
"answer": "To avoid overwriting states"
},
{
"question": "How does the number of items affect time complexity?",
"options": ["Linearly", "Logarithmically", "Quadratically", "Exponentially"],
"answer": "Linearly"
},
{
"question": "What does W represent in O(nW) time complexity?",
"options": ["Knapsack capacity", "Number of items", "Total weight", "Total profit"],
"answer": "Knapsack capacity"
},
{
"question": "Knapsack can be solved using which search strategy?",
"options": ["DFS with pruning", "Breadth-first sort", "Trie hashing", "Topological sort"],
"answer": "DFS with pruning"
},
{
"question": "In a DP table, which corner holds the final result?",
"options": ["Bottom-right", "Top-left", "Center", "Bottom-left"],
"answer": "Bottom-right"
},
{
"question": "What kind of complexity class is Knapsack in?",
"options": ["NP-complete", "P", "NP-hard", "Linear"],
"answer": "NP-complete"
}
]



tf_questions = {
"Level 1": [
{"question": "0/1 Knapsack allows partial selection of items.", "answer": False},
{"question": "Knapsack problem is solved using recursion or DP.", "answer": True},
{"question": "In 0/1 Knapsack, each item is used at most once.", "answer": True},
{"question": "Knapsack requires only item values.", "answer": False},
{"question": "Knapsack aims to maximize value under weight constraint.", "answer": True},
{"question": "Each item has a fixed value and weight.", "answer": True},
{"question": "Knapsack can be solved without loops.", "answer": False},
{"question": "Knapsack can use 2D tables to store intermediate results.", "answer": True},
{"question": "Knapsack’s goal is to minimize weight.", "answer": False},
{"question": "All items must be selected in 0/1 Knapsack.", "answer": False},
{"question": "Knapsack problem is a type of optimization problem.", "answer": True},
{"question": "We can solve Knapsack using dynamic programming.", "answer": True},
{"question": "Knapsack’s input includes capacity limit.", "answer": True},
{"question": "Knapsack doesn’t use any recursion.", "answer": False},
{"question": "Each item in Knapsack has a profit and weight.", "answer": True}
],


"Level 2": [
    {"question": "dp[i][j] represents maximum value with first i items and capacity j.", "answer": True},
    {"question": "We include an item in Knapsack if it fits and gives better value.", "answer": True},
    {"question": "The problem has no overlapping subproblems.", "answer": False},
    {"question": "Memoization stores intermediate results for efficiency.", "answer": True},
    {"question": "In DP, each state depends on a smaller subproblem.", "answer": True},
    {"question": "In 0/1 Knapsack, we always include every item.", "answer": False},
    {"question": "Knapsack's time complexity is exponential with recursion.", "answer": True},
    {"question": "If an item doesn't fit, it is skipped.", "answer": True},
    {"question": "The value of dp[n][W] gives total weight.", "answer": False},
    {"question": "In recursion, base case is when n=0 or W=0.", "answer": True},
    {"question": "A 2D array is used in bottom-up DP for Knapsack.", "answer": True},
    {"question": "Knapsack can be solved using graphs.", "answer": False},
    {"question": "Items with lower weight are always selected first in Knapsack.", "answer": False},
    {"question": "Knapsack recursion solves by trying include and exclude choices.", "answer": True},
    {"question": "In DP table, first row and column are usually initialized to 0.", "answer": True}
],

"Level 3": [
    {"question": "Knapsack space optimization reduces 2D table to 1D.", "answer": True},
    {"question": "Space optimization may overwrite useful values if not careful.", "answer": True},
    {"question": "Knapsack has optimal substructure property.", "answer": True},
    {"question": "Greedy algorithm always gives optimal result for 0/1 Knapsack.", "answer": False},
    {"question": "Knapsack DP table can be filled from bottom up.", "answer": True},
    {"question": "Time complexity using DP is O(nW).", "answer": True},
    {"question": "Unbounded Knapsack allows multiple instances of an item.", "answer": True},
    {"question": "Space-optimized Knapsack always uses a stack.", "answer": False},
    {"question": "Knapsack problem is NP-complete.", "answer": True},
    {"question": "The value of dp[n][W] is the maximum achievable profit.", "answer": True},
    {"question": "0/1 Knapsack is solvable using DFS with pruning.", "answer": True},
    {"question": "BFS is the best approach for 0/1 Knapsack.", "answer": False},
    {"question": "Space optimization should update dp[w] in reverse order.", "answer": True},
    {"question": "Using 1D DP gives same result as 2D table in Knapsack.", "answer": True},
    {"question": "Knapsack always includes the most valuable item.", "answer": False}
]
}