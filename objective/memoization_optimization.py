mcq_questions = {

"Level 1": [
{
"question": "What is the main advantage of using memoization in optimization problems?",
"options": ["Avoid recomputation", "Slower loops", "More recursion", "Reduces memory"],
"answer": "Avoid recomputation"
},
{
"question": "Which of the following problems benefits from memoization optimization?",
"options": ["Fibonacci", "Binary Search", "Selection Sort", "Bubble Sort"],
"answer": "Fibonacci"
},
{
"question": "What is commonly used to store intermediate results in memoization?",
"options": ["Dictionary", "List", "Queue", "Tuple"],
"answer": "Dictionary"
},
{
"question": "In Python, which module supports memoization through decorators?",
"options": ["functools", "random", "math", "itertools"],
"answer": "functools"
},
{
"question": "What happens if you don’t use memoization in recursive optimization?",
"options": ["Redundant calculations", "Faster execution", "No recursion", "Stack reset"],
"answer": "Redundant calculations"
},
{
"question": "Which decorator in Python is used to apply memoization?",
"options": ["@lru_cache", "@memoize", "@staticmethod", "@dataclass"],
"answer": "@lru_cache"
},
{
"question": "Memoization converts which type of recursion to efficient code?",
"options": ["Exponential", "Linear", "Quadratic", "Constant"],
"answer": "Exponential"
},
{
"question": "What does LRU stand for in functools.lru_cache?",
"options": ["Least Recently Used", "Last Referenced Unit", "Light Recursion Utility", "Loop Reduction Unit"],
"answer": "Least Recently Used"
},
{
"question": "Memoization is part of which optimization strategy?",
"options": ["Dynamic Programming", "Greedy", "Brute Force", "Divide and Conquer"],
"answer": "Dynamic Programming"
},
{
"question": "What is one downside of memoization?",
"options": ["High memory usage", "Low CPU usage", "Slower speed", "Less recursion"],
"answer": "High memory usage"
},
{
"question": "Which data structure can be used for manual memoization in Python?",
"options": ["dict", "set", "list", "stack"],
"answer": "dict"
},
{
"question": "What kind of functions should be memoized?",
"options": ["Pure functions", "Functions with randomness", "I/O functions", "Threaded functions"],
"answer": "Pure functions"
},
{
"question": "Which cache strategy is used by functools.lru_cache?",
"options": ["LRU", "FIFO", "MRU", "LIFO"],
"answer": "LRU"
},
{
"question": "How does memoization optimize recursive algorithms?",
"options": ["Stores results of subproblems", "Increases recursion", "Reduces memory", "Creates multiple threads"],
"answer": "Stores results of subproblems"
},
{
"question": "Which problem is least suitable for memoization?",
"options": ["Random Number Generator", "Fibonacci", "Knapsack", "LCS"],
"answer": "Random Number Generator"
}
],

"Level 2": [
{
"question": "In recursive optimization, what must be checked before a new calculation?",
"options": ["If the result is cached", "If value is negative", "If input is sorted", "If input is even"],
"answer": "If the result is cached"
},
{
"question": "What is the effect of not checking the cache in memoization?",
"options": ["Repeated computation", "Faster output", "No output", "Stack reset"],
"answer": "Repeated computation"
},
{
"question": "What type of problems is memoization NOT ideal for?",
"options": ["Independent subproblems", "Overlapping subproblems", "Recursive with DP", "Fibonacci"],
"answer": "Independent subproblems"
},
{
"question": "Which optimization strategy uses memoization?",
"options": ["Top-down DP", "Backtracking", "Greedy", "Divide and Conquer"],
"answer": "Top-down DP"
},
{
"question": "In which language is @lru_cache commonly used for optimization?",
"options": ["Python", "C++", "Java", "HTML"],
"answer": "Python"
},
{
"question": "What happens when cache size exceeds limit in lru_cache?",
"options": ["Oldest entries are removed", "Newest entries are removed", "All cache is cleared", "Error is raised"],
"answer": "Oldest entries are removed"
},
{
"question": "Memoization is applied on which type of subproblem?",
"options": ["Repeated", "Unique", "Sorted", "Iterated"],
"answer": "Repeated"
},
{
"question": "Which function is a poor candidate for memoization?",
"options": ["Random.randint()", "Fibonacci", "Knapsack", "LCS"],
"answer": "Random.randint()"
},
{
"question": "Memoization improves performance by reducing:",
"options": ["Number of recursive calls", "Stack memory", "Loop count", "Recursion depth"],
"answer": "Number of recursive calls"
},
{
"question": "How does memoization optimize large DP problems?",
"options": ["Avoids solving same subproblem multiple times", "Sorts input", "Increases base cases", "Limits recursion depth"],
"answer": "Avoids solving same subproblem multiple times"
},
{
"question": "What is the space complexity impact of memoization?",
"options": ["Increased", "Decreased", "Constant", "None"],
"answer": "Increased"
},
{
"question": "What is the time complexity improvement of Fibonacci using memoization?",
"options": ["O(n)", "O(2^n)", "O(log n)", "O(n^2)"],
"answer": "O(n)"
},
{
"question": "Which built-in type is often used to simulate a memo cache?",
"options": ["dict", "tuple", "list", "int"],
"answer": "dict"
},
{
"question": "What do you use to uniquely identify inputs in memoization?",
"options": ["Function arguments", "Return values", "Indices", "Keys"],
"answer": "Function arguments"
},
{
"question": "Memoization prevents which issue in recursion-heavy problems?",
"options": ["Time limit exceeded", "Random results", "Syntax errors", "Variable shadowing"],
"answer": "Time limit exceeded"
}
],

"Level 3": [
{
"question": "Which cache policy is used in Python’s lru_cache?",
"options": ["Least Recently Used", "First In First Out", "Most Frequently Used", "Random"],
"answer": "Least Recently Used"
},
{
"question": "What is a downside of large memoization caches?",
"options": ["High memory usage", "Slower recursion", "No function calls", "Fewer arguments"],
"answer": "High memory usage"
},
{
"question": "What is a practical method to clear cache in Python?",
"options": ["function.cache_clear()", "reset()", "clear()", "flush()"],
"answer": "function.cache_clear()"
},
{
"question": "What makes a function suitable for memoization?",
"options": ["Deterministic outputs", "Randomness", "No parameters", "Side effects"],
"answer": "Deterministic outputs"
},
{
"question": "What makes memoization different from tabulation?",
"options": ["Top-down vs bottom-up", "Less memory", "No recursion", "Hashing"],
"answer": "Top-down vs bottom-up"
},
{
"question": "What happens if you use memoization in non-pure functions?",
"options": ["Incorrect caching", "Better results", "Faster output", "Nothing"],
"answer": "Incorrect caching"
},
{
"question": "Which feature can help limit cache memory in large applications?",
"options": ["maxsize in lru_cache", "print()", "while loop", "try-catch block"],
"answer": "maxsize in lru_cache"
},
{
"question": "What is a recommended cache size for large-scale problems?",
"options": ["Depends on problem size", "Always 128", "Always 256", "Maximum integer"],
"answer": "Depends on problem size"
},
{
"question": "Memoization improves runtime by reducing:",
"options": ["Recomputation", "Recursion", "Loop depth", "List operations"],
"answer": "Recomputation"
},
{
"question": "What structure allows use of tuple arguments as cache keys?",
"options": ["Dictionary", "List", "Set", "Queue"],
"answer": "Dictionary"
},
{
"question": "Which of these features ensures memoization safety?",
"options": ["No side effects", "Multiple threads", "Random input", "Infinite recursion"],
"answer": "No side effects"
},
{
"question": "Memoization in recursive DP is ideal for:",
"options": ["Repeated overlapping subproblems", "Sorting algorithms", "Greedy problems", "Binary trees"],
"answer": "Repeated overlapping subproblems"
},
{
"question": "Which of these benefits most from space-limited memoization?",
"options": ["Problems with limited unique subproblems", "Random function outputs", "Deep recursion trees", "Greedy algorithms"],
"answer": "Problems with limited unique subproblems"
},
{
"question": "Which alternative technique handles non-pure functions better?",
"options": ["Tabulation", "Memoization", "Recursion", "Stack unrolling"],
"answer": "Tabulation"
},
{
"question": "Python’s functools.lru_cache uses which default max size?",
"options": ["128", "256", "64", "Unlimited"],
"answer": "128"
}
]
}

tf_questions = {

"Level 1": [
{"question": "Memoization helps avoid repeated computations.", "answer": True},
{"question": "Memoization uses a cache to store previous results.", "answer": True},
{"question": "lru_cache is used in Java for memoization.", "answer": False},
{"question": "Pure functions are good candidates for memoization.", "answer": True},
{"question": "Memoization reduces the number of function calls.", "answer": True},
{"question": "Memoization increases memory usage.", "answer": True},
{"question": "Memoization is useful for optimizing recursive functions.", "answer": True},
{"question": "Memoization is only used in loops.", "answer": False},
{"question": "Knapsack problem benefits from memoization.", "answer": True},
{"question": "Memoization improves the efficiency of brute-force algorithms.", "answer": True},
{"question": "Memoization cannot be applied in Python.", "answer": False},
{"question": "Python’s @lru_cache helps with memoization.", "answer": True},
{"question": "Memoization is useless in problems with unique inputs.", "answer": True},
{"question": "The main goal of memoization is to reduce time complexity.", "answer": True},
{"question": "Memoization is the same as multithreading.", "answer": False}
],

"Level 2": [
{"question": "Memoization should only be used with deterministic functions.", "answer": True},
{"question": "Top-down DP is achieved using memoization.", "answer": True},
{"question": "Random outputs are ideal for memoization.", "answer": False},
{"question": "Memoization works best for overlapping subproblems.", "answer": True},
{"question": "Manual caching requires dictionary or map structure.", "answer": True},
{"question": "Memoization always reduces space usage.", "answer": False},
{"question": "Using memoization without checking cache is inefficient.", "answer": True},
{"question": "Fibonacci with memoization has O(n) time complexity.", "answer": True},
{"question": "Knapsack can be optimized using memoization.", "answer": True},
{"question": "Memoization is built-in using functools in Python.", "answer": True},
{"question": "Each subproblem must be solved only once with memoization.", "answer": True},
{"question": "Using memoization increases code complexity slightly.", "answer": True},
{"question": "Memoization helps in stack overflow scenarios.", "answer": True},
{"question": "Memoization uses additional memory to improve speed.", "answer": True},
{"question": "Memoization is part of greedy algorithm design.", "answer": False}
],

"Level 3": [
{"question": "lru_cache removes least recently used items when full.", "answer": True},
{"question": "Too much memoization can lead to high memory consumption.", "answer": True},
{"question": "Clearing cache can be done using cache_clear().", "answer": True},
{"question": "Non-pure functions may cause incorrect results in memoization.", "answer": True},
{"question": "Memoization is more effective in top-down recursion.", "answer": True},
{"question": "Memoization reduces runtime by avoiding recomputation.", "answer": True},
{"question": "Randomized outputs work well with memoization.", "answer": False},
{"question": "Hashable keys are required in memoization cache.", "answer": True},
{"question": "Pure functions return consistent outputs for same inputs.", "answer": True},
{"question": "Memoization helps optimize recursive DP problems.", "answer": True},
{"question": "functools.lru_cache has a default maxsize of 128.", "answer": True},
{"question": "Memory leaks can happen with careless memoization.", "answer": True},
{"question": "Tabulation is more suitable than memoization for non-recursive optimization.", "answer": True},
{"question": "Memoization can be disabled in lru_cache using maxsize=None.", "answer": True},
{"question": "Memoization works equally well with random inputs.", "answer": False}
]
}