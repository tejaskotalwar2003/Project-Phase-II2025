mcq_questions = {
"Level 1": [
{
"question": "What does Fibonacci optimization aim to improve?",
"options": ["Only output formatting", "Time and space efficiency", "Sorting speed", "Loop structures"],
"answer": "Time and space efficiency"
},
{
"question": "Which of the following stores only the last two Fibonacci numbers?",
"options": ["Memoization", "Space optimization", "Backtracking", "Naive recursion"],
"answer": "Space optimization"
},
{
"question": "What is the space complexity of a space-optimized Fibonacci solution?",
"options": ["O(n)", "O(log n)", "O(1)", "O(n^2)"],
"answer": "O(1)"
},
{
"question": "Which approach avoids the use of recursion completely?",
"options": ["Memoization", "Backtracking", "Bottom-up Iterative", "Naive recursion"],
"answer": "Bottom-up Iterative"
},
{
"question": "What is the time complexity of iterative Fibonacci?",
"options": ["O(n)", "O(1)", "O(n log n)", "O(n^2)"],
"answer": "O(n)"
},
{
"question": "What are the first two values in space-optimized Fibonacci?",
"options": ["F(1) and F(2)", "0 and 1", "1 and 2", "F(n−1) and F(n−2)"],
"answer": "0 and 1"
},
{
"question": "Which of these is used in iterative Fibonacci?",
"options": ["Recursion", "Call stack", "Loop", "Queue"],
"answer": "Loop"
},
{
"question": "Which of these variables are typically updated in a loop?",
"options": ["prev and curr", "high and low", "sum and count", "n and m"],
"answer": "prev and curr"
},
{
"question": "Which statement is TRUE about space optimization?",
"options": ["It increases time complexity", "It reduces memory usage", "It stores the full sequence", "It requires matrix input"],
"answer": "It reduces memory usage"
},
{
"question": "How many variables are needed in optimized iterative Fibonacci?",
"options": ["3", "4", "2", "n"],
"answer": "2"
},
{
"question": "What is the main benefit of space optimization?",
"options": ["Code readability", "Better output", "Lower memory usage", "Faster printing"],
"answer": "Lower memory usage"
},
{
"question": "Does space optimization affect accuracy of Fibonacci results?",
"options": ["Yes", "No", "Only for odd values", "Only for large n"],
"answer": "No"
},
{
"question": "What loop type is commonly used for iterative Fibonacci?",
"options": ["for", "while", "do-while", "repeat-until"],
"answer": "for"
},
{
"question": "Fibonacci(5) using iterative optimization equals:",
"options": ["3", "5", "8", "13"],
"answer": "5"
},
{
"question": "Can space-optimized Fibonacci handle large values of n?",
"options": ["Yes", "No", "Only up to 100", "Only even numbers"],
"answer": "Yes"
}
],


"Level 2": [
    {
        "question": "In space-optimized Fibonacci, what is updated at each iteration?",
        "options": ["The full DP table", "Last two Fibonacci values", "Fibonacci of all previous values", "Stack frames"],
        "answer": "Last two Fibonacci values"
    },
    {
        "question": "What is a drawback of space optimization?",
        "options": ["Slower execution", "Can't reconstruct full sequence", "More memory usage", "Cannot compute base cases"],
        "answer": "Can't reconstruct full sequence"
    },
    {
        "question": "Which of these is most memory-efficient for large n?",
        "options": ["Recursive with memoization", "Space-optimized iterative", "Backtracking", "Tabulation with full DP table"],
        "answer": "Space-optimized iterative"
    },
    {
        "question": "What are the two key variables used in space optimization?",
        "options": ["i and j", "prev and curr", "left and right", "x and y"],
        "answer": "prev and curr"
    },
    {
        "question": "When computing Fibonacci(n), how many variables do we really need?",
        "options": ["n", "log n", "2", "n-2"],
        "answer": "2"
    },
    {
        "question": "Can you track the full sequence with space optimization?",
        "options": ["Yes", "No", "Only for n ≤ 10", "Only if n is even"],
        "answer": "No"
    },
    {
        "question": "Which loop control structure is ideal for iterative Fibonacci?",
        "options": ["for-loop", "if-else", "recursion", "goto"],
        "answer": "for-loop"
    },
    {
        "question": "Which operation is repeated in space-optimized Fibonacci?",
        "options": ["Multiplication", "Addition", "Subtraction", "Division"],
        "answer": "Addition"
    },
    {
        "question": "Which of the following is most commonly avoided in space optimization?",
        "options": ["Lists", "Recursion", "Loops", "Print statements"],
        "answer": "Lists"
    },
    {
        "question": "Space optimization is a form of:",
        "options": ["Greedy algorithm", "Recursion", "Dynamic Programming", "Sorting"],
        "answer": "Dynamic Programming"
    },
    {
        "question": "What value is returned by optimized Fibonacci(10)?",
        "options": ["34", "55", "89", "13"],
        "answer": "55"
    },
    {
        "question": "Which of the following would consume more space?",
        "options": ["Full DP table", "Recursion", "Space-optimized loop", "for-loop with 2 variables"],
        "answer": "Full DP table"
    },
    {
        "question": "Space optimization eliminates:",
        "options": ["Code complexity", "Memoization", "Full DP array", "Iteration"],
        "answer": "Full DP array"
    },
    {
        "question": "You can reverse-calculate the entire sequence from a space-optimized method.",
        "options": ["Yes", "No", "Only up to n=5", "Only odd-index values"],
        "answer": "No"
    },
    {
        "question": "What is required to convert iterative DP into space-optimized form?",
        "options": ["Extra memory", "Tracking just two values", "Sorting step", "Hash maps"],
        "answer": "Tracking just two values"
    }
],

"Level 3": [
    {
        "question": "What is the time complexity of matrix exponentiation for Fibonacci?",
        "options": ["O(log n)", "O(n)", "O(n^2)", "O(n log n)"],
        "answer": "O(log n)"
    },
    {
        "question": "Which matrix is used in matrix exponentiation?",
        "options": [
            "[[1, 1], [1, 0]]", "[[2, 0], [0, 2]]", "[[0, 1], [1, 0]]", "[[1, 1], [2, 1]]"
        ],
        "answer": "[[1, 1], [1, 0]]"
    },
    {
        "question": "Which method is best for computing Fibonacci(10^6)?",
        "options": ["Naive recursion", "Recursive with memoization", "Matrix exponentiation", "Full table DP"],
        "answer": "Matrix exponentiation"
    },
    {
        "question": "Matrix exponentiation for Fibonacci involves:",
        "options": ["Matrix sorting", "Matrix inversion", "Binary exponentiation", "Modular hashing"],
        "answer": "Binary exponentiation"
    },
    {
        "question": "Which method is the fastest for huge n (like 10^12)?",
        "options": ["Memoization", "Backtracking", "Space optimization", "Matrix method"],
        "answer": "Matrix method"
    },
    {
        "question": "What is the space complexity of matrix exponentiation?",
        "options": ["O(n)", "O(1)", "O(log n)", "O(n^2)"],
        "answer": "O(1)"
    },
    {
        "question": "What kind of multiplication is performed in matrix Fibonacci?",
        "options": ["Element-wise", "Scalar", "Matrix × Matrix", "Transpose"],
        "answer": "Matrix × Matrix"
    },
    {
        "question": "Can matrix-based Fibonacci be parallelized for speed?",
        "options": ["Yes", "No", "Only on GPUs", "Only for even Fibonacci"],
        "answer": "Yes"
    },
    {
        "question": "Matrix Fibonacci is part of which category?",
        "options": ["Graph algorithms", "Linear algebra-based DP", "Sorting methods", "Graph traversal"],
        "answer": "Linear algebra-based DP"
    },
    {
        "question": "Matrix exponentiation returns exact Fibonacci values.",
        "options": ["True", "False", "Only up to n=100", "Only odd n"],
        "answer": "True"
    },
    {
        "question": "Does matrix method improve space complexity over tabulation?",
        "options": ["Yes", "No", "Only sometimes", "Depends on input"],
        "answer": "Yes"
    },
    {
        "question": "Matrix exponentiation is only suitable for small input sizes.",
        "options": ["True", "False", "Only for prime n", "Only for F(10) to F(20)"],
        "answer": "False"
    },
    {
        "question": "What happens if you try to use naive recursion for n = 1000000?",
        "options": ["It fails or crashes", "It works instantly", "It produces wrong values", "It returns 0"],
        "answer": "It fails or crashes"
    },
    {
        "question": "Can matrix exponentiation be implemented using recursion?",
        "options": ["Yes", "No", "Only for even n", "Only if memoized"],
        "answer": "Yes"
    },
    {
        "question": "Matrix-based Fibonacci helps in cryptographic applications.",
        "options": ["Yes", "No", "Only in RSA", "Only with graphs"],
        "answer": "Yes"
    }
]
}

tf_questions = {
"Level 1": [
{"question": "Fibonacci optimization improves both time and space usage.", "answer": True},
{"question": "In space optimization, we store all previous Fibonacci numbers.", "answer": False},
{"question": "Bottom-up approach uses iteration instead of recursion.", "answer": True},
{"question": "Optimized Fibonacci needs more than two variables to run.", "answer": False},
{"question": "Space-optimized Fibonacci has constant space complexity.", "answer": True},
{"question": "Iterative Fibonacci is faster than naive recursion.", "answer": True},
{"question": "Space-optimized Fibonacci still uses a full DP array.", "answer": False},
{"question": "Fibonacci(0) = 0 and Fibonacci(1) = 1.", "answer": True},
{"question": "Space optimization is a form of brute force.", "answer": False},
{"question": "In bottom-up approach, values are built from F(0) upward.", "answer": True},
{"question": "DP always refers to Divide and Print.", "answer": False},
{"question": "You can compute Fibonacci(100000) using space optimization.", "answer": True},
{"question": "Space-optimized DP requires a full matrix.", "answer": False},
{"question": "Bottom-up DP eliminates recursion.", "answer": True},
{"question": "The values in Fibonacci grow very slowly.", "answer": False}
],
"Level 2": [
{"question": "In space-optimized Fibonacci, only the last two values are needed.", "answer": True},
{"question": "Space optimization allows you to print the entire sequence easily.", "answer": False},
{"question": "Fibonacci(50) can be computed quickly using a loop.", "answer": True},
{"question": "Using an array is necessary for Fibonacci optimization.", "answer": False},
{"question": "Space-optimized Fibonacci uses a loop to update values.", "answer": True},
{"question": "Optimized Fibonacci reduces memory usage.", "answer": True},
{"question": "Space optimization increases code complexity a little.", "answer": True},
{"question": "You cannot compute Fibonacci(1000000) using space optimization.", "answer": False},
{"question": "Space optimization is used to reduce memory.", "answer": True},
{"question": "Storing only last two values helps in space efficiency.", "answer": True},
{"question": "Iterative Fibonacci uses less memory than recursive Fibonacci.", "answer": True},
{"question": "Fibonacci optimization only works for even indices.", "answer": False},
{"question": "DP is short for Dynamic Programming.", "answer": True},
{"question": "Space optimization is not applicable to all DP problems.", "answer": True},
{"question": "Space-optimized code requires matrix operations.", "answer": False}
],
"Level 3": [
{"question": "Matrix exponentiation can compute Fibonacci numbers in log time.", "answer": True},
{"question": "Matrix-based Fibonacci uses a 2x2 matrix.", "answer": True},
{"question": "Naive recursion is faster than matrix exponentiation.", "answer": False},
{"question": "Matrix exponentiation gives exact results for large n.", "answer": True},
{"question": "Matrix exponentiation is only useful for sorting problems.", "answer": False},
{"question": "Binary exponentiation helps reduce computation time.", "answer": True},
{"question": "Matrix exponentiation requires the use of recursion.", "answer": False},
{"question": "The identity matrix is used in Fibonacci matrix exponentiation.", "answer": True},
{"question": "Matrix exponentiation helps compute Fibonacci efficiently.", "answer": True},
{"question": "Matrix exponentiation consumes more space than tabulation.", "answer": False},
{"question": "You can compute Fibonacci(10^18) with matrix exponentiation.", "answer": True},
{"question": "Matrix exponentiation is part of dynamic programming.", "answer": False},
{"question": "Matrix exponentiation is language-dependent.", "answer": False},
{"question": "Matrix exponentiation is only theoretical and not used in practice.", "answer": False},
{"question": "Matrix exponentiation can be implemented iteratively or recursively.", "answer": True}
]
}