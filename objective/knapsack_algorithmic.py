mcq_questions = {

"Level 1": [
{
"question": "What does the 0/1 in 0/1 Knapsack refer to?",
"options": ["Include or exclude each item", "Item index", "Binary weights", "Sorted values"],
"answer": "Include or exclude each item"
},
{
"question": "What data structure is typically used in basic Knapsack DP?",
"options": ["2D Array", "List", "Graph", "Tree"],
"answer": "2D Array"
},
{
"question": "Which of the following is a base case in recursive Knapsack?",
"options": ["n == 0 or W == 0", "n == 1 and W == 1", "n < W", "n == W"],
"answer": "n == 0 or W == 0"
},
{
"question": "What does dp[i][w] represent?",
"options": ["Maximum value for i items and capacity w", "Minimum items used", "Remaining space", "Item index"],
"answer": "Maximum value for i items and capacity w"
},
{
"question": "In recursion, which two cases are considered for each item?",
"options": ["Include and exclude", "Push and pop", "Sort and choose", "Swap and compare"],
"answer": "Include and exclude"
},
{
"question": "What input is required for 0/1 Knapsack?",
"options": ["Values, Weights, Capacity", "Names and values", "Sorted array", "Queue"],
"answer": "Values, Weights, Capacity"
},
{
"question": "What is the initial value of dp[i][0] or dp[0][w]?",
"options": ["0", "-1", "w", "i"],
"answer": "0"
},
{
"question": "Which loop structure is used in bottom-up Knapsack?",
"options": ["Nested for-loops", "While loop", "Recursion", "Binary loop"],
"answer": "Nested for-loops"
},
{
"question": "Which Python structure is most suitable for DP table?",
"options": ["List of lists", "Tuple", "Set", "Queue"],
"answer": "List of lists"
},
{
"question": "Which value is updated during DP computation?",
"options": ["dp[i][w]", "wt[i]", "val[i]", "index"],
"answer": "dp[i][w]"
},
{
"question": "What is checked before including an item in DP?",
"options": ["Weight ≤ current capacity", "Value is max", "Item is prime", "Weight > capacity"],
"answer": "Weight ≤ current capacity"
},
{
"question": "What is returned from knapsack(dp, wt, val, n, W)?",
"options": ["Maximum profit", "Total weight", "Sorted items", "All selected items"],
"answer": "Maximum profit"
},
{
"question": "How do you check if item is included in recursive Knapsack?",
"options": ["Compare include and exclude values", "Check loop index", "Check sorted order", "Check value % 2"],
"answer": "Compare include and exclude values"
},
{
"question": "How is value updated in recursion?",
"options": ["val[i-1] + recurse(n-1, W - wt[i-1])", "val[i] - W", "recurse(W - val[i])", "None"],
"answer": "val[i-1] + recurse(n-1, W - wt[i-1])"
},
{
"question": "What language feature helps memoization in Python?",
"options": ["@lru_cache", "@memoize", "set()", "range()"],
"answer": "@lru_cache"
}
],

"Level 2": [
{
"question": "What is the recurrence relation in 0/1 Knapsack?",
"options": ["dp[i][w] = max(dp[i-1][w], val[i] + dp[i-1][w-wt[i]])", "dp[i][w] = val[i]", "dp[i][w] = wt[i]", "dp[i][w] = dp[i][i]"],
"answer": "dp[i][w] = max(dp[i-1][w], val[i] + dp[i-1][w-wt[i]])"
},
{
"question": "In space-optimized Knapsack, the DP table becomes:",
"options": ["1D Array", "Tuple", "Set", "Queue"],
"answer": "1D Array"
},
{
"question": "Why is reverse traversal used in 1D DP Knapsack?",
"options": ["Prevent overwriting", "Faster lookup", "Greedy choice", "Sort values"],
"answer": "Prevent overwriting"
},
{
"question": "Which of the following avoids recomputation in recursion?",
"options": ["Memoization", "Iteration", "Looping", "Sorting"],
"answer": "Memoization"
},
{
"question": "What does 'W' typically represent in code?",
"options": ["Capacity", "Weight of one item", "Item index", "Minimum value"],
"answer": "Capacity"
},
{
"question": "Which Python structure is useful to store (weight, value) pairs?",
"options": ["List of tuples", "Queue", "Set", "Stack"],
"answer": "List of tuples"
},
{
"question": "When is an item included in recursion?",
"options": ["When wt[i] <= W", "If value > 100", "If index is odd", "Never"],
"answer": "When wt[i] <= W"
},
{
"question": "Which of the following improves time in top-down Knapsack?",
"options": ["Memoization", "List slicing", "Sorting", "Hashing"],
"answer": "Memoization"
},
{
"question": "What is the time complexity of DP Knapsack?",
"options": ["O(nW)", "O(2^n)", "O(log n)", "O(n^2)"],
"answer": "O(nW)"
},
{
"question": "What is the time complexity of recursive Knapsack without memoization?",
"options": ["O(2^n)", "O(nW)", "O(n^2)", "O(n log n)"],
"answer": "O(2^n)"
},
{
"question": "Which base case breaks recursion?",
"options": ["n == 0 or W == 0", "val[i] == 0", "i == j", "wt[i] > W"],
"answer": "n == 0 or W == 0"
},
{
"question": "What do you compare when building DP table?",
"options": ["Include vs exclude profit", "Weight vs index", "Value vs weight", "None"],
"answer": "Include vs exclude profit"
},
{
"question": "Which loop order is correct in bottom-up DP?",
"options": ["for i in range(n+1): for w in range(W+1):", "for w in range(W): for i in range(n):", "while i<n: while w<W:", "None"],
"answer": "for i in range(n+1): for w in range(W+1):"
},
{
"question": "How is dp[0][w] initialized in 0/1 Knapsack?",
"options": ["0", "1", "w", "None"],
"answer": "0"
},
{
"question": "Which of the following is NOT true?",
"options": ["DP is faster than recursion", "Greedy is always optimal", "Memoization saves time", "Knapsack is NP-complete"],
"answer": "Greedy is always optimal"
}
],

"Level 3": [
{
"question": "What does dp[w] = max(dp[w], val[i] + dp[w-wt[i]]) represent?",
"options": ["Space-optimized update", "Brute-force method", "Backtracking", "Greedy update"],
"answer": "Space-optimized update"
},
{
"question": "Which method provides O(nW) time and O(W) space?",
"options": ["1D DP", "Naive recursion", "2D DP", "Memoization"],
"answer": "1D DP"
},
{
"question": "When building 1D DP, why is right-to-left traversal important?",
"options": ["To avoid reuse in same iteration", "To reduce space", "To simplify logic", "No reason"],
"answer": "To avoid reuse in same iteration"
},
{
"question": "What’s the purpose of memoization in top-down approach?",
"options": ["Store intermediate results", "Speed up printing", "Increase memory", "Avoid loops"],
"answer": "Store intermediate results"
},
{
"question": "Which of the following statements is false?",
"options": ["Knapsack DP is always exact", "Greedy works for 0/1 Knapsack", "Top-down needs memoization", "1D DP uses array"],
"answer": "Greedy works for 0/1 Knapsack"
},
{
"question": "What does dp[i][j] depend on?",
"options": ["dp[i-1][j] and dp[i-1][j-wt[i-1]]", "dp[i+1][j]", "val[i]", "None"],
"answer": "dp[i-1][j] and dp[i-1][j-wt[i-1]]"
},
{
"question": "Which method is best for limited space and large n?",
"options": ["1D DP", "Memoization", "Greedy", "DFS"],
"answer": "1D DP"
},
{
"question": "Which structure is updated in place in 1D DP?",
"options": ["dp[w]", "dp[i][w]", "val[i]", "wt[i]"],
"answer": "dp[w]"
},
{
"question": "Which direction is used in building 1D Knapsack?",
"options": ["Reverse", "Forward", "Random", "Circular"],
"answer": "Reverse"
},
{
"question": "What happens if you update dp from left to right in 1D DP?",
"options": ["Overwrites correct states", "Works correctly", "Speeds up", "Uses more memory"],
"answer": "Overwrites correct states"
},
{
"question": "Which language provides built-in memoization decorators?",
"options": ["Python", "C", "Java", "HTML"],
"answer": "Python"
},
{
"question": "Which implementation is fastest in practice for 0/1 Knapsack?",
"options": ["1D DP", "2D DP", "Memoized recursion", "Brute-force"],
"answer": "1D DP"
},
{
"question": "If wt[i-1] > w, what is the result?",
"options": ["dp[i][w] = dp[i-1][w]", "Add value anyway", "Set to 0", "Skip all"],
"answer": "dp[i][w] = dp[i-1][w]"
},
{
"question": "What happens if value array is all 0?",
"options": ["Max profit is 0", "Error", "Max profit is W", "Recursion fails"],
"answer": "Max profit is 0"
},
{
"question": "In recursion, what causes stack overflow?",
"options": ["Too many recursive calls", "Low memory", "Invalid input", "Missing base case"],
"answer": "Too many recursive calls"
}
]
}
tf_questions = {
"Level 1": [
{"question": "0/1 Knapsack can be solved using recursion.", "answer": True},
{"question": "Each item can be taken multiple times in 0/1 Knapsack.", "answer": False},
{"question": "Knapsack problems deal with value and weight.", "answer": True},
{"question": "In Knapsack, capacity cannot be exceeded.", "answer": True},
{"question": "A 2D array is typically used in DP Knapsack.", "answer": True},
{"question": "Knapsack problems use greedy algorithm successfully.", "answer": False},
{"question": "You must sort items before solving Knapsack.", "answer": False},
{"question": "Knapsack problems are optimization problems.", "answer": True},
{"question": "The base case is when n=0 or W=0.", "answer": True},
{"question": "Recursive Knapsack always has better time complexity than DP.", "answer": False},
{"question": "Knapsack problems deal with integers only.", "answer": False},
{"question": "All Knapsack problems require dynamic programming.", "answer": False},
{"question": "Knapsack input includes weights, values, and capacity.", "answer": True},
{"question": "In recursion, the last item is always included.", "answer": False},
{"question": "You can visualize Knapsack using a DP table.", "answer": True}
],
"Level 2": [
{"question": "dp[i][j] stores max value for i items and capacity j.", "answer": True},
{"question": "Knapsack recurrence compares include vs exclude.", "answer": True},
{"question": "In top-down, memoization prevents recomputation.", "answer": True},
{"question": "dp[0][w] is initialized to zero.", "answer": True},
{"question": "Knapsack cannot be solved using memoization.", "answer": False},
{"question": "Recursive Knapsack without memoization is fast.", "answer": False},
{"question": "Knapsack can be solved both top-down and bottom-up.", "answer": True},
{"question": "dp[i][j] updates without comparing values.", "answer": False},
{"question": "Memoization uses a cache to store results.", "answer": True},
{"question": "Recursion base case prevents infinite calls.", "answer": True},
{"question": "DP table helps to visualize solution.", "answer": True},
{"question": "DP solution uses nested loops.", "answer": True},
{"question": "Knapsack is a classical DP problem.", "answer": True},
{"question": "Greedy approach always gives optimal Knapsack solution.", "answer": False},
{"question": "Knapsack recursion may fail on large inputs.", "answer": True}
],
"Level 3": [
{"question": "1D DP reduces space to O(W).", "answer": True},
{"question": "Right-to-left traversal in 1D DP avoids overwriting.", "answer": True},
{"question": "Memoization helps recursive Knapsack run faster.", "answer": True},
{"question": "In 1D DP, update must happen from left to right.", "answer": False},
{"question": "Greedy strategy is optimal for 0/1 Knapsack.", "answer": False},
{"question": "Recursive solution can lead to stack overflow.", "answer": True},
{"question": "Bigger value always means better item in Knapsack.", "answer": False},
{"question": "Knapsack is a type of NP-complete problem.", "answer": True},
{"question": "1D DP and 2D DP give same output for 0/1 Knapsack.", "answer": True},
{"question": "Knapsack table can be updated using just one row.", "answer": True},
{"question": "Knapsack recurrence uses max of include and exclude.", "answer": True},
{"question": "Knapsack can be solved using binary search.", "answer": False},
{"question": "The final answer lies at dp[n][W].", "answer": True},
{"question": "Knapsack can’t be solved without DP.", "answer": False},
{"question": "Knapsack allows reuse of items in 0/1 version.", "answer": False}
]
}