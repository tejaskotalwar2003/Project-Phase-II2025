mcq_questions = {
"Level 1": [
{
"question": "What are the first five numbers in the Fibonacci sequence?",
"options": ["0 1 1 2 3", "1 1 2 3 5", "0 1 2 3 5", "1 2 3 5 8"],
"answer": "0 1 1 2 3"
},
{
"question": "What is the value of Fibonacci(0)?",
"options": ["1", "0", "2", "Undefined"],
"answer": "0"
},
{
"question": "What is the value of Fibonacci(1)?",
"options": ["0", "1", "2", "3"],
"answer": "1"
},
{
"question": "What is the recursive formula for Fibonacci?",
"options": ["F(n) = F(n−1) + F(n−2)", "F(n) = F(n−1) × F(n−2)", "F(n) = F(n−2) − F(n−1)", "F(n) = 2 × F(n−1)"],
"answer": "F(n) = F(n−1) + F(n−2)"
},
{
"question": "Which number comes next: 0, 1, 1, 2, 3, 5, ?",
"options": ["7", "8", "6", "10"],
"answer": "8"
},
{
"question": "Fibonacci(5) equals:",
"options": ["3", "5", "8", "13"],
"answer": "5"
},
{
"question": "How many base cases are there in the Fibonacci recursion?",
"options": ["0", "1", "2", "3"],
"answer": "2"
},
{
"question": "Which of the following is NOT a Fibonacci number?",
"options": ["13", "21", "34", "50"],
"answer": "50"
},
{
"question": "Which statement is TRUE about the Fibonacci sequence?",
"options": [
"Every third number is divisible by 3",
"It always increases by 2",
"It starts at 1",
"Only odd numbers appear"
],
"answer": "Every third number is divisible by 3"
},
{
"question": "Which method gives the most intuitive understanding of Fibonacci?",
"options": ["Stack", "Recursion", "Hashing", "Sorting"],
"answer": "Recursion"
},
{
"question": "What is the time complexity of naive recursive Fibonacci?",
"options": ["O(n)", "O(n^2)", "O(log n)", "O(2^n)"],
"answer": "O(2^n)"
},
{
"question": "In Fibonacci, F(3) equals:",
"options": ["2", "3", "1", "4"],
"answer": "2"
},
{
"question": "Which technique is used to improve recursive Fibonacci?",
"options": ["Sorting", "Backtracking", "Memoization", "Simulation"],
"answer": "Memoization"
},
{
"question": "What is the output of Fibonacci(7)?",
"options": ["8", "11", "13", "21"],
"answer": "13"
},
{
"question": "Which of the following tools is best to visualize Fibonacci calculations?",
"options": ["Bar chart", "Stack diagram", "Line graph", "Recursion tree"],
"answer": "Recursion tree"
}
]
,

"Level 2" :[
{
"question": "What is memoization in the context of Fibonacci?",
"options": [
"Repeating calculations for accuracy",
"Storing results to avoid recomputation",
"Visualizing recursion",
"Forcing the use of stack memory"
],
"answer": "Storing results to avoid recomputation"
},
{
"question": "Which data structure is commonly used for memoization in Python?",
"options": ["List", "Set", "Dictionary", "Queue"],
"answer": "Dictionary"
},
{
"question": "How many function calls occur in naive recursion for F(5)?",
"options": ["5", "10", "15", "25"],
"answer": "15"
},
{
"question": "Which Python decorator is used for automatic memoization?",
"options": ["@cached", "@remember", "@functools.lru_cache", "@optimize"],
"answer": "@functools.lru_cache"
},
{
"question": "What is the main drawback of using naive recursion for Fibonacci?",
"options": [
"It cannot compute values above 10",
"It uses excessive memory",
"It repeats subproblems unnecessarily",
"It gives wrong answers"
],
"answer": "It repeats subproblems unnecessarily"
},
{
"question": "What is the time complexity of recursive Fibonacci with memoization?",
"options": ["O(n^2)", "O(2^n)", "O(n)", "O(n log n)"],
"answer": "O(n)"
},
{
"question": "Which problem-solving technique does memoization belong to?",
"options": ["Backtracking", "Greedy", "Divide and Conquer", "Dynamic Programming"],
"answer": "Dynamic Programming"
},
{
"question": "Which condition is checked in a memoized recursive function?",
"options": [
"if result in memo: return it",
"if result == 0: break",
"if input is empty",
"if depth == limit"
],
"answer": "if result in memo: return it"
},
{
"question": "What does the memoization table store in Fibonacci?",
"options": [
"All prime numbers",
"Previously computed Fibonacci values",
"Only even Fibonacci values",
"Only base cases"
],
"answer": "Previously computed Fibonacci values"
},
{
"question": "In recursion, what triggers the base case in Fibonacci?",
"options": ["F(n) = 1", "n == 1", "n == 0 or n == 1", "n == 10"],
"answer": "n == 0 or n == 1"
},
{
"question": "Which of the following is a correct base case for a recursive Fibonacci function?",
"options": ["if n == 2: return 2", "if n <= 1: return n", "if n == 1: return 0", "if n == 3: return 1"],
"answer": "if n <= 1: return n"
},
{
"question": "What is the output of memoized Fibonacci(10)?",
"options": ["34", "55", "89", "21"],
"answer": "55"
},
{
"question": "How many recursive calls are saved when computing Fibonacci(6) with memoization?",
"options": ["All except base cases", "None", "Only even numbers", "Only one call"],
"answer": "All except base cases"
},
{
"question": "What happens if you forget to memoize the result in a recursive function?",
"options": [
"Fibonacci will return random values",
"Program will run faster",
"Same values will be recomputed repeatedly",
"It will switch to iteration"
],
"answer": "Same values will be recomputed repeatedly"
},
{
"question": "What is a key benefit of top-down DP (recursion with memoization)?",
"options": [
"Easier to write and understand",
"Requires matrix multiplication",
"Uses binary search",
"Avoids base cases"
],
"answer": "Easier to write and understand"
}
],

"Level 3":[
{
"question": "What is the time complexity of bottom-up (iterative) Fibonacci?",
"options": ["O(n^2)", "O(2^n)", "O(n)", "O(log n)"],
"answer": "O(n)"
},
{
"question": "What is the space complexity of the optimized iterative Fibonacci?",
"options": ["O(n)", "O(1)", "O(log n)", "O(2^n)"],
"answer": "O(1)"
},
{
"question": "Which values are stored in space-optimized Fibonacci?",
"options": [
"Only odd Fibonacci values",
"Last two Fibonacci values",
"Complete list from 0 to n",
"Only even Fibonacci values"
],
"answer": "Last two Fibonacci values"
},
{
"question": "Which of the following is most memory-efficient?",
"options": [
"Naive recursion",
"Memoized recursion",
"Full DP table",
"Iterative with two variables"
],
"answer": "Iterative with two variables"
},
{
"question": "What is the value of Fibonacci(25) using bottom-up DP?",
"options": ["75025", "121393", "46368", "28657"],
"answer": "75025"
},
{
"question": "Which method is best for very large n (e.g., Fibonacci(100000))?",
"options": [
"Naive recursion",
"Memoized recursion",
"Iterative space-optimized",
"Brute force"
],
"answer": "Iterative space-optimized"
},
{
"question": "How many variables are needed for the most space-optimized Fibonacci?",
"options": ["1", "2", "3", "Depends on n"],
"answer": "2"
},
{
"question": "Which of the following can overflow memory for large n?",
"options": [
"Space-optimized iterative",
"Memoized recursion with large dictionary",
"Naive recursion with base cases",
"for-loop implementation"
],
"answer": "Memoized recursion with large dictionary"
},
{
"question": "Which version of Fibonacci supports parallelization best?",
"options": [
"Iterative",
"Memoized recursion",
"Matrix exponentiation",
"Backtracking"
],
"answer": "Matrix exponentiation"
},
{
"question": "What is the time complexity of matrix-exponentiation-based Fibonacci?",
"options": ["O(n)", "O(log n)", "O(n log n)", "O(1)"],
"answer": "O(log n)"
},
{
"question": "Which technique is used in matrix-based Fibonacci?",
"options": ["Greedy", "Matrix power", "Backtracking", "DFS"],
"answer": "Matrix power"
},
{
"question": "What is the smallest n for which naive recursion becomes impractical?",
"options": ["n = 10", "n = 20", "n = 30", "n = 5"],
"answer": "n = 30"
},
{
"question": "Which method avoids stack overflow completely?",
"options": [
"Naive recursion",
"Memoized recursion",
"Bottom-up iterative",
"Tail recursion without base case"
],
"answer": "Bottom-up iterative"
},
{
"question": "Which is fastest for computing large Fibonacci numbers accurately?",
"options": [
"Naive recursion",
"Matrix exponentiation",
"Memoization with dictionary",
"Using math.log"
],
"answer": "Matrix exponentiation"
},
{
"question": "Which technique is typically not used in Fibonacci optimization?",
"options": [
"Greedy approach",
"Matrix exponentiation",
"Space optimization",
"Memoization"
],
"answer": "Greedy approach"
}
]
}
tf_questions = {

"Level 1": [
{"question": "The Fibonacci sequence starts with 0 and 1.", "answer": True},
{"question": "Each Fibonacci number is the sum of the two previous numbers.", "answer": True},
{"question": "Fibonacci(2) equals 3.", "answer": False},
{"question": "The sequence 0, 1, 1, 2, 3, 5 is part of Fibonacci.", "answer": True},
{"question": "Fibonacci(0) = 1.", "answer": False},
{"question": "The Fibonacci sequence is an example of a recursive series.", "answer": True},
{"question": "Fibonacci numbers are always even.", "answer": False},
{"question": "The third Fibonacci number is 2.", "answer": False},
{"question": "Fibonacci(4) = 3.", "answer": True},
{"question": "Fibonacci can be used to model population growth.", "answer": True},
{"question": "The Fibonacci sequence is used in nature and spirals.", "answer": True},
{"question": "Fibonacci(6) = 13.", "answer": False},
{"question": "The Fibonacci sequence is only useful for math problems.", "answer": False},
{"question": "In Fibonacci, F(1) = 1 and F(2) = 1.", "answer": True},
{"question": "The Fibonacci sequence is infinite.", "answer": True}
],

"Level 2": [
{"question": "Memoization helps improve Fibonacci recursion.", "answer": True},
{"question": "Using memoization increases the number of recursive calls.", "answer": False},
{"question": "The memoization technique uses a dictionary or array.", "answer": True},
{"question": "Recursive Fibonacci without memoization is inefficient.", "answer": True},
{"question": "Fibonacci(10) equals 34.", "answer": False},
{"question": "Base cases in Fibonacci recursion are F(0) and F(1).", "answer": True},
{"question": "Memoized Fibonacci has exponential time complexity.", "answer": False},
{"question": "Memoization avoids recalculating subproblems.", "answer": True},
{"question": "Storing computed Fibonacci values reduces stack usage.", "answer": True},
{"question": "Fibonacci recursion with memoization uses more memory than naive recursion.", "answer": True},
{"question": "Fibonacci(7) with memoization and naive recursion will give different results.", "answer": False},
{"question": "Fibonacci is a classic example of a divide and conquer problem.", "answer": False},
{"question": "Memoization improves performance at the cost of space.", "answer": True},
{"question": "The recursive tree of Fibonacci without memoization has repeated branches.", "answer": True},
{"question": "Dynamic Programming is not applicable to Fibonacci problems.", "answer": False}
],

"Level 3": [
{"question": "Space-optimized Fibonacci uses only two variables.", "answer": True},
{"question": "Time complexity of iterative Fibonacci is O(n).", "answer": True},
{"question": "Space complexity of optimized Fibonacci is O(n).", "answer": False},
{"question": "Bottom-up Fibonacci avoids recursion completely.", "answer": True},
{"question": "Matrix exponentiation can compute Fibonacci in O(log n) time.", "answer": True},
{"question": "Iterative Fibonacci and recursive Fibonacci give different results.", "answer": False},
{"question": "Fibonacci(25) = 75025.", "answer": True},
{"question": "Matrix method uses exponentiation of a 2×2 matrix.", "answer": True},
{"question": "For very large n, naive recursion is more efficient.", "answer": False},
{"question": "Space optimization reduces memory but increases time.", "answer": False},
{"question": "Fibonacci using dynamic programming avoids stack overflow.", "answer": True},
{"question": "In optimized approach, we only need to keep track of the last 5 values.", "answer": False},
{"question": "Fibonacci numbers can grow very large quickly.", "answer": True},
{"question": "You can compute Fibonacci(10^6) efficiently using recursion with memoization.", "answer": False},
{"question": "Matrix-based Fibonacci is ideal for performance-critical applications.", "answer": True}
]

}