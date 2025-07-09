mcq_questions = {
"Level 1": [
{
"question": "Which of the following defines the Fibonacci sequence?",
"options": ["F(n) = F(n-1) + F(n-2)", "F(n) = F(n-1) * F(n-2)", "F(n) = F(n-1) - F(n-2)", "F(n) = F(n-1) / F(n-2)"],
"answer": "F(n) = F(n-1) + F(n-2)"
},
{
"question": "What are the first five Fibonacci numbers?",
"options": ["0, 1, 1, 2, 3", "1, 2, 3, 4, 5", "0, 1, 2, 3, 5", "1, 1, 2, 4, 8"],
"answer": "0, 1, 1, 2, 3"
},
{
"question": "Which number does Fibonacci(0) return?",
"options": ["1", "0", "2", "Undefined"],
"answer": "0"
},
{
"question": "What is the value of Fibonacci(3)?",
"options": ["1", "2", "3", "4"],
"answer": "2"
},
{
"question": "Which is NOT a Fibonacci number?",
"options": ["13", "21", "34", "40"],
"answer": "40"
},
{
"question": "What does Fibonacci(6) equal?",
"options": ["5", "8", "13", "21"],
"answer": "8"
},
{
"question": "How many base cases are used in Fibonacci recursion?",
"options": ["1", "2", "3", "4"],
"answer": "2"
},
{
"question": "Which of the following is true about Fibonacci numbers?",
"options": ["All are even", "All are prime", "They form a sequence", "They are negative"],
"answer": "They form a sequence"
},
{
"question": "The Fibonacci sequence starts with:",
"options": ["1, 1", "0, 1", "1, 2", "0, 0"],
"answer": "0, 1"
},
{
"question": "Fibonacci(5) returns:",
"options": ["3", "5", "8", "13"],
"answer": "5"
},
{
"question": "Which language has built-in support for memoization in Fibonacci?",
"options": ["Python", "HTML", "CSS", "SQL"],
"answer": "Python"
},
{
"question": "The Fibonacci sequence grows:",
"options": ["Linearly", "Randomly", "Exponentially", "Logarithmically"],
"answer": "Exponentially"
},
{
"question": "What is the sum of the first four Fibonacci numbers?",
"options": ["5", "7", "4", "8"],
"answer": "4"
},
{
"question": "Which of the following is the correct 7th Fibonacci number?",
"options": ["8", "13", "21", "34"],
"answer": "13"
},
{
"question": "What is the index of Fibonacci number 21?",
"options": ["6", "7", "8", "9"],
"answer": "8"
}
],
"Level 2": [
{
"question": "What is the time complexity of naive recursive Fibonacci?",
"options": ["O(n)", "O(n^2)", "O(log n)", "O(2^n)"],
"answer": "O(2^n)"
},
{
"question": "Which technique improves the time complexity of Fibonacci?",
"options": ["Memoization", "Greedy", "Sorting", "Binary Search"],
"answer": "Memoization"
},
{
"question": "Fibonacci(10) equals:",
"options": ["34", "55", "89", "21"],
"answer": "55"
},
{
"question": "What is used to store intermediate Fibonacci results?",
"options": ["List", "Stack", "Dictionary", "Queue"],
"answer": "Dictionary"
},
{
"question": "Which statement best defines dynamic programming?",
"options": ["Breaking into independent problems", "Optimizing using stored subproblems", "Random guessing", "Recursive looping"],
"answer": "Optimizing using stored subproblems"
},
{
"question": "What is the output of Fibonacci(7)?",
"options": ["8", "11", "13", "21"],
"answer": "13"
},
{
"question": "Which of the following prevents recomputation in recursion?",
"options": ["Memoization", "Loop", "List", "Break statement"],
"answer": "Memoization"
},
{
"question": "In memoized Fibonacci, how are values saved?",
"options": ["By index", "By key-value", "By priority", "By weight"],
"answer": "By key-value"
},
{
"question": "How many calls are saved using memoization for Fibonacci(6)?",
"options": ["None", "All except base", "Only last 2", "All"],
"answer": "All except base"
},
{
"question": "What is the return type of a recursive Fibonacci function?",
"options": ["String", "Boolean", "Integer", "Float"],
"answer": "Integer"
},
{
"question": "Which of the following terms applies to Fibonacci recursion?",
"options": ["Top-down", "Bottom-up", "Linear", "Parallel"],
"answer": "Top-down"
},
{
"question": "Which recursive step is correct for Fibonacci?",
"options": ["F(n) = F(n) + F(n)", "F(n) = F(n-1) + F(n-2)", "F(n) = F(n-2) - F(n-1)", "F(n) = F(n+1)"],
"answer": "F(n) = F(n-1) + F(n-2)"
},
{
"question": "Which of these helps visualize recursion calls?",
"options": ["Tree diagram", "Bar graph", "Pie chart", "Linked list"],
"answer": "Tree diagram"
},
{
"question": "Which tool visualizes stack calls of Fibonacci recursion?",
"options": ["Debugger", "Editor", "Shell", "Console"],
"answer": "Debugger"
},
{
"question": "What is used in Python to implement memoization quickly?",
"options": ["@memoize", "@cache", "@functools.lru_cache", "@return"],
"answer": "@functools.lru_cache"
}
],
"Level 3": [
{
"question": "What is the space complexity of space-optimized Fibonacci?",
"options": ["O(n)", "O(1)", "O(n^2)", "O(log n)"],
"answer": "O(1)"
},
{
"question": "Which Fibonacci implementation is best for large n?",
"options": ["Memoized recursion", "Naive recursion", "Iterative with 2 vars", "Loop with full list"],
"answer": "Iterative with 2 vars"
},
{
"question": "What is Binet’s formula used for?",
"options": ["Exact Fibonacci computation", "Approximation only", "Matrix inversion", "Optimization"],
"answer": "Exact Fibonacci computation"
},
{
"question": "Which method achieves O(log n) time for Fibonacci?",
"options": ["Matrix exponentiation", "Loop", "Memoization", "Brute force"],
"answer": "Matrix exponentiation"
},
{
"question": "What is required for Binet’s formula?",
"options": ["Golden ratio", "Modulo", "Random seed", "Constant"],
"answer": "Golden ratio"
},
{
"question": "Matrix-based Fibonacci uses how many variables?",
"options": ["2", "3", "4", "None"],
"answer": "4"
},
{
"question": "Which method uses matrix multiplication?",
"options": ["Matrix Fibonacci", "Loop", "Recursive", "Hashing"],
"answer": "Matrix Fibonacci"
},
{
"question": "What is the Fibonacci(50) value using matrix exponentiation?",
"options": ["12586269025", "102334155", "832040", "144"],
"answer": "12586269025"
},
{
"question": "What is the fastest known method for Fibonacci?",
"options": ["Matrix", "Loop", "Recursive", "Naive"],
"answer": "Matrix"
},
{
"question": "Which of these methods gives exact results for large n?",
"options": ["Matrix", "Memoization", "Loop", "Binet’s"],
"answer": "Matrix"
},
{
"question": "What happens with recursive Fibonacci for large n?",
"options": ["Stack overflow", "Fast result", "Low memory", "Returns 0"],
"answer": "Stack overflow"
},
{
"question": "What is the best case for using matrix method?",
"options": ["Huge input", "Small input", "Sorted list", "Floating points"],
"answer": "Huge input"
},
{
"question": "Which method is most suitable in embedded systems?",
"options": ["Loop with 2 vars", "Matrix", "Recursive", "Memoized"],
"answer": "Loop with 2 vars"
},
{
"question": "Is Binet's formula accurate for large n in float?",
"options": ["No", "Yes, with errors", "Perfectly", "Only small n"],
"answer": "Yes, with errors"
},
{
"question": "Can Fibonacci be computed using bitwise operators?",
"options": ["No", "Yes", "Partially", "Only in C++"],
"answer": "No"
}
]
}

tf_questions = {
"Level 1": [
{"question": "The Fibonacci sequence starts with 0 and 1.", "answer": True},
{"question": "Fibonacci(2) is equal to 2.", "answer": False},
{"question": "Fibonacci(5) = 5.", "answer": True},
{"question": "Every Fibonacci number is a multiple of 2.", "answer": False},
{"question": "Fibonacci numbers form an increasing sequence.", "answer": True},
{"question": "Fibonacci(0) = 0 and Fibonacci(1) = 1.", "answer": True},
{"question": "Fibonacci numbers follow the formula F(n) = F(n-1) + F(n-2).", "answer": True},
{"question": "The 4th Fibonacci number is 4.", "answer": False},
{"question": "Fibonacci numbers are related to nature and spirals.", "answer": True},
{"question": "The Fibonacci sequence grows linearly.", "answer": False},
{"question": "Fibonacci(6) = 8.", "answer": True},
{"question": "Fibonacci can be used to model population growth.", "answer": True},
{"question": "Fibonacci numbers are only defined for even indices.", "answer": False},
{"question": "Fibonacci(3) = 2.", "answer": True},
{"question": "The Fibonacci sequence is infinite.", "answer": True}
],
"Level 2": [
{"question": "Memoization improves the performance of Fibonacci recursion.", "answer": True},
{"question": "Naive recursive Fibonacci has exponential time complexity.", "answer": True},
{"question": "Dynamic programming is used to avoid recomputation.", "answer": True},
{"question": "Fibonacci(7) = 13.", "answer": True},
{"question": "Using a dictionary helps in memoization.", "answer": True},
{"question": "The recursive formula for Fibonacci is F(n) = F(n-2) - F(n-1).", "answer": False},
{"question": "Top-down approach uses recursion and memoization.", "answer": True},
{"question": "Memoization increases memory usage to improve time.", "answer": True},
{"question": "Fibonacci(10) = 34.", "answer": False},
{"question": "Recursive Fibonacci without memoization is faster than iteration.", "answer": False},
{"question": "Dynamic programming uses overlapping subproblems.", "answer": True},
{"question": "Fibonacci recursion creates a tree of calls.", "answer": True},
{"question": "You can visualize recursive calls using a tree diagram.", "answer": True},
{"question": "The lru_cache decorator can memoize Fibonacci in Python.", "answer": True},
{"question": "In memoization, base cases are still computed multiple times.", "answer": False}
],
"Level 3": [
{"question": "Matrix exponentiation computes Fibonacci in O(log n) time.", "answer": True},
{"question": "Binet’s formula is based on the Golden Ratio.", "answer": True},
{"question": "Space-optimized Fibonacci uses only two variables.", "answer": True},
{"question": "Matrix-based Fibonacci is inaccurate for large n.", "answer": False},
{"question": "Fibonacci(50) is a very large number.", "answer": True},
{"question": "Recursive Fibonacci is ideal for very large inputs.", "answer": False},
{"question": "Matrix-based Fibonacci is faster than iterative methods for large n.", "answer": True},
{"question": "Fibonacci can be expressed using closed-form formula.", "answer": True},
{"question": "Fibonacci numbers follow O(2^n) growth in recursion.", "answer": True},
{"question": "Binet’s formula always returns an integer.", "answer": False},
{"question": "Matrix exponentiation can be implemented recursively or iteratively.", "answer": True},
{"question": "Fibonacci(100000) can be computed instantly using naive recursion.", "answer": False},
{"question": "Fibonacci growth is exponential.", "answer": True},
{"question": "Space optimization increases memory usage.", "answer": False},
{"question": "Matrix method uses matrix [[1,1],[1,0]] raised to power n.", "answer": True}
]
}