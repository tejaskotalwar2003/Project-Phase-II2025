mcq_questions = {

"Level 1": [
{
"question": "What is memoization?",
"options": [
"Storing function results to avoid recomputation",
"Encrypting memory",
"Adding delay to function",
"A type of loop"
],
"answer": "Storing function results to avoid recomputation"
},
{
"question": "Memoization is most commonly used with which type of function?",
"options": ["Recursive", "Iterative", "Linear", "Static"],
"answer": "Recursive"
},
{
"question": "What data structure is typically used for memoization?",
"options": ["Dictionary", "List", "Set", "Queue"],
"answer": "Dictionary"
},
{
"question": "Which of the following is a key feature of memoization?",
"options": [
"Caching",
"Recursion prevention",
"Loop unrolling",
"Sorting"
],
"answer": "Caching"
},
{
"question": "In which problem is memoization highly effective?",
"options": [
"Fibonacci calculation",
"Bubble sort",
"Linear search",
"Stack overflow"
],
"answer": "Fibonacci calculation"
},
{
"question": "What is avoided using memoization?",
"options": ["Repeated computation", "Recursion", "Base cases", "Loops"],
"answer": "Repeated computation"
},
{
"question": "Where are memoized results stored?",
"options": ["Cache table", "Stack", "Queue", "None"],
"answer": "Cache table"
},
{
"question": "Which decorator in Python is used for memoization?",
"options": [
"@functools.lru_cache",
"@staticmethod",
"@memo",
"@lambda"
],
"answer": "@functools.lru_cache"
},
{
"question": "Memoization is part of which programming paradigm?",
"options": ["Dynamic programming", "Greedy", "Backtracking", "Sorting"],
"answer": "Dynamic programming"
},
{
"question": "Which concept is similar to memoization?",
"options": ["Caching", "Looping", "Hashing", "Branching"],
"answer": "Caching"
},
{
"question": "Memoization stores values based on:",
"options": ["Function arguments", "Return values", "Loop index", "Memory size"],
"answer": "Function arguments"
},
{
"question": "What is the time benefit of memoization?",
"options": [
"Reduces exponential to polynomial time",
"Slows execution",
"Removes base cases",
"None"
],
"answer": "Reduces exponential to polynomial time"
},
{
"question": "Which problem type benefits least from memoization?",
"options": ["Non-overlapping subproblems", "Overlapping subproblems", "Recursive", "DP-based"],
"answer": "Non-overlapping subproblems"
},
{
"question": "What is the key requirement for memoization?",
"options": [
"Deterministic function output",
"Large data",
"Multithreading",
"Non-recursive logic"
],
"answer": "Deterministic function output"
},
{
"question": "Memoization works best when:",
"options": ["Function is called with same inputs repeatedly", "Function is linear", "Function is void", "Loop is large"],
"answer": "Function is called with same inputs repeatedly"
}
],

"Level 2": [
{
"question": "Which function is a good example of memoization in action?",
"options": ["Fibonacci", "Insertion Sort", "Factorial (iterative)", "Binary Search"],
"answer": "Fibonacci"
},
{
"question": "Which condition must be checked before recursive call in memoization?",
"options": ["If value is already in cache", "If input is sorted", "If list is empty", "If recursion depth is exceeded"],
"answer": "If value is already in cache"
},
{
"question": "How is the cache typically organized?",
"options": ["Key: input, Value: result", "Key: result, Value: input", "Unordered list", "Heap"],
"answer": "Key: input, Value: result"
},
{
"question": "What type of problems can memoization optimize?",
"options": ["Overlapping subproblems", "Brute-force", "Greedy", "Sorting"],
"answer": "Overlapping subproblems"
},
{
"question": "Memoization increases which resource usage?",
"options": ["Memory", "Time", "Threads", "CPUs"],
"answer": "Memory"
},
{
"question": "Which Python built-in function supports memoization?",
"options": ["functools.lru_cache", "itertools.memo", "cache.memo", "heapq"],
"answer": "functools.lru_cache"
},
{
"question": "In manual memoization, which structure is updated?",
"options": ["Dictionary", "List", "String", "Set"],
"answer": "Dictionary"
},
{
"question": "Which type of recursion benefits from memoization?",
"options": ["Top-down", "Bottom-up", "Linear", "Nested"],
"answer": "Top-down"
},
{
"question": "What is the best-case time complexity after memoization?",
"options": ["O(n)", "O(n^2)", "O(log n)", "O(2^n)"],
"answer": "O(n)"
},
{
"question": "Which of these is true about memoization?",
"options": ["Saves time, uses more memory", "Saves memory, uses more time", "Only saves memory", "Only uses recursion"],
"answer": "Saves time, uses more memory"
},
{
"question": "Which function type is memoization not ideal for?",
"options": ["Functions with random outputs", "Pure functions", "Recursive functions", "DP problems"],
"answer": "Functions with random outputs"
},
{
"question": "Memoization eliminates which type of computation?",
"options": ["Repeated", "Random", "Initial", "Terminal"],
"answer": "Repeated"
},
{
"question": "What happens if memo cache is not checked first?",
"options": ["Redundant calculations", "Faster result", "No effect", "Stack overflow"],
"answer": "Redundant calculations"
},
{
"question": "Memoization is considered:",
"options": ["Top-down DP", "Bottom-up DP", "Greedy method", "Sorting approach"],
"answer": "Top-down DP"
},
{
"question": "What is the default maximum size of functools.lru_cache?",
"options": ["128", "64", "256", "None"],
"answer": "128"
}
],

"Level 3": [
{
"question": "Which problem is most suited for memoization?",
"options": ["LCS (Longest Common Subsequence)", "Selection sort", "Binary Search", "Bubble sort"],
"answer": "LCS (Longest Common Subsequence)"
},
{
"question": "Which scenario requires custom cache in memoization?",
"options": ["Variable input arguments", "Fixed output", "Short recursion", "No recursion"],
"answer": "Variable input arguments"
},
{
"question": "What can you use to memoize functions with tuple inputs?",
"options": ["Dictionary with tuples as keys", "Set", "List", "Array"],
"answer": "Dictionary with tuples as keys"
},
{
"question": "How does Python’s lru_cache identify function inputs?",
"options": ["By hashing arguments", "By sorting arguments", "By looping", "By default order"],
"answer": "By hashing arguments"
},
{
"question": "What happens if cache size is exceeded in lru_cache?",
"options": ["Oldest values are discarded", "Error occurs", "All values are removed", "Nothing"],
"answer": "Oldest values are discarded"
},
{
"question": "Which implementation gives full control of memoization?",
"options": ["Manual cache using dictionary", "lru_cache", "Queue", "Tuple"],
"answer": "Manual cache using dictionary"
},
{
"question": "Memoization in functional programming relies on:",
"options": ["Pure functions", "Loops", "Mutable variables", "Classes"],
"answer": "Pure functions"
},
{
"question": "What is the downside of memoizing too many inputs?",
"options": ["High memory usage", "Less accuracy", "Longer recursion", "Fewer return values"],
"answer": "High memory usage"
},
{
"question": "Memoization can improve performance by avoiding:",
"options": ["Repeated subproblem calls", "Loop iteration", "Sorting", "Binary search"],
"answer": "Repeated subproblem calls"
},
{
"question": "Which of these problems will NOT benefit from memoization?",
"options": ["Quicksort", "Knapsack", "Fibonacci", "LCS"],
"answer": "Quicksort"
},
{
"question": "Recursive + memoization follows which paradigm?",
"options": ["Top-down DP", "Greedy", "BFS", "Backtracking"],
"answer": "Top-down DP"
},
{
"question": "Which feature improves performance of recursive DP functions?",
"options": ["Memoization", "Loop unrolling", "Recursion limit", "None"],
"answer": "Memoization"
},
{
"question": "Memoized solutions work best when subproblems are:",
"options": ["Repeated", "Independent", "Sorted", "Unrelated"],
"answer": "Repeated"
},
{
"question": "Which cache policy is used in lru_cache?",
"options": ["Least Recently Used", "First In First Out", "Most Recently Used", "Random"],
"answer": "Least Recently Used"
},
{
"question": "Memoization avoids stack overflow by:",
"options": ["Reducing recursion calls", "Stopping loops", "Adding delay", "Using sorting"],
"answer": "Reducing recursion calls"
}
]
}

tf_questions = {

"Level 1": [
{"question": "Memoization is a caching technique used in recursion.", "answer": True},
{"question": "Memoization increases execution time by repeating calculations.", "answer": False},
{"question": "Fibonacci sequence is a common example for memoization.", "answer": True},
{"question": "Memoization only works with loops.", "answer": False},
{"question": "Memoization stores function results to avoid recomputation.", "answer": True},
{"question": "Python has built-in support for memoization using decorators.", "answer": True},
{"question": "Memoization is useful when the same input is used multiple times.", "answer": True},
{"question": "Memoization is part of dynamic programming.", "answer": True},
{"question": "Memoization replaces all base cases in recursion.", "answer": False},
{"question": "Caching is another name for memoization.", "answer": True},
{"question": "Memoization is not used in LCS or Knapsack.", "answer": False},
{"question": "Memoization improves efficiency in recursive problems.", "answer": True},
{"question": "Dictionary is commonly used to store memoized values.", "answer": True},
{"question": "Memoization works only in Python.", "answer": False},
{"question": "Memoization is ideal for problems with overlapping subproblems.", "answer": True}
],

"Level 2": [
{"question": "Top-down DP is implemented using memoization.", "answer": True},
{"question": "Memoization increases time complexity.", "answer": False},
{"question": "functools.lru_cache is a decorator in Python for memoization.", "answer": True},
{"question": "Memoization saves previously computed results.", "answer": True},
{"question": "Overlapping subproblems benefit from memoization.", "answer": True},
{"question": "Memoization does not require function to be deterministic.", "answer": False},
{"question": "Random output functions should not be memoized.", "answer": True},
{"question": "Memoization is slower than recursion without memo.", "answer": False},
{"question": "Recursive functions with overlapping calls benefit from memoization.", "answer": True},
{"question": "Memoization requires more memory than plain recursion.", "answer": True},
{"question": "Memoization eliminates need for recursion.", "answer": False},
{"question": "DP and memoization solve same problems differently.", "answer": True},
{"question": "Memoization is suitable for problems with repeated calls.", "answer": True},
{"question": "You must check if result is cached before computing.", "answer": True},
{"question": "Using memoization incorrectly can cause performance drop.", "answer": True}
],

"Level 3": [
{"question": "lru_cache removes least recently used entries when full.", "answer": True},
{"question": "Memoization helps convert exponential time to polynomial time.", "answer": True},
{"question": "Pure functions are ideal for memoization.", "answer": True},
{"question": "Memoizing all input values can lead to high memory usage.", "answer": True},
{"question": "Memoization doesn’t reduce number of function calls.", "answer": False},
{"question": "Knapsack and LCS benefit from memoization.", "answer": True},
{"question": "Quicksort benefits from memoization.", "answer": False},
{"question": "Recursive calls with unique inputs still benefit from memoization.", "answer": False},
{"question": "Memoization avoids recomputing repeated subproblems.", "answer": True},
{"question": "lru_cache uses hashing to store and retrieve cache values.", "answer": True},
{"question": "Recursive + memoization is a form of top-down DP.", "answer": True},
{"question": "Memoization is less useful in problems with few repeated calls.", "answer": True},
{"question": "Manual cache control gives more flexibility than lru_cache.", "answer": True},
{"question": "Python's memoization requires external libraries.", "answer": False},
{"question": "Memoization avoids redundant computation in dynamic programming.", "answer": True}
]
}