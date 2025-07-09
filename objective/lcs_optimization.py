mcq_questions = {
    "Level 1": [
                {
            "question": "Which technique is used to reduce memory usage in LCS?",
            "options": ["Stack-based recursion", "Loop unrolling", "Space optimization", "Bit masking"],
            "answer": "Space optimization"
        },
        {
            "question": "What is the space complexity of optimized LCS?",
            "options": ["O(n*m)", "O(n)", "O(n+n)", "O(1)"],
            "answer": "O(n)"
        },
        {
            "question": "Which DP technique allows reducing 2D matrix to 2 rows?",
            "options": ["Memoization", "Greedy", "Row-swapping", "Bottom-up rolling"],
            "answer": "Bottom-up rolling"
        },
        {
            "question": "Which version of LCS is better for limited memory systems?",
            "options": ["Recursive", "Backtracking", "Space-optimized", "Full table"],
            "answer": "Space-optimized"
        },
        {
            "question": "What is the benefit of space optimization in LCS?",
            "options": ["Faster loops", "Lower memory usage", "Fewer matches", "Recursive simplification"],
            "answer": "Lower memory usage"
        },
        {
            "question": "What do we avoid in space-optimized LCS?",
            "options": ["Looping", "Recursion", "2D array allocation", "Base cases"],
            "answer": "2D array allocation"
        },
        {
            "question": "Which data structure can help track only the required rows in LCS?",
            "options": ["Set", "Deque", "List of 2 rows", "Hashmap"],
            "answer": "List of 2 rows"
        },
        {
            "question": "In space-optimized LCS, which row is updated repeatedly?",
            "options": ["Last row", "Current and previous", "Middle row", "Diagonal only"],
            "answer": "Current and previous"
        },
        {
            "question": "What is reduced in space-optimized LCS?",
            "options": ["Time", "Input size", "Matrix storage", "Output"],
            "answer": "Matrix storage"
        },
        {
            "question": "What is the space usage of recursive LCS with memoization?",
            "options": ["O(1)", "O(n)", "O(n*m)", "O(log n)"],
            "answer": "O(n*m)"
        },
        {
            "question": "Can we retrieve LCS string from space-optimized version?",
            "options": ["Yes, easily", "No", "Only length", "Only last character"],
            "answer": "Only length"
        },
        {
            "question": "Which approach avoids storing full matrix?",
            "options": ["Greedy", "Backtracking", "Rolling row", "Divide and conquer"],
            "answer": "Rolling row"
        },
        {
            "question": "Which is NOT a space-saving strategy in LCS?",
            "options": ["Row reuse", "Partial storage", "Diagonal processing", "Storing full table"],
            "answer": "Storing full table"
        },
        {
            "question": "Which language feature helps write space-optimized LCS in Python?",
            "options": ["List slicing", "Dictionary indexing", "Comprehension", "Loop nesting"],
            "answer": "List slicing"
        },
        {
            "question": "What is a trade-off in space optimization of LCS?",
            "options": ["Increased recursion", "Slower time", "Loss of sequence", "Bigger table"],
            "answer": "Loss of sequence"
        }
    ],
    "level 2":[
        {
            "question": "Which LCS version allows space reduction using only two rows?",
            "options": ["Recursive", "Memoized", "Rolling array", "Greedy"],
            "answer": "Rolling array"
        },
        {
            "question": "Which operation is not required in space-optimized LCS?",
            "options": ["Full matrix allocation", "String traversal", "Row updating", "Length tracking"],
            "answer": "Full matrix allocation"
        },
        {
            "question": "How many rows are actively maintained in space-optimized LCS?",
            "options": ["1", "2", "n", "m"],
            "answer": "2"
        },
        {
            "question": "Which part of the DP table is reused in space-optimized LCS?",
            "options": ["Top-left cell", "Diagonal block", "Only last two rows", "Full matrix"],
            "answer": "Only last two rows"
        },
        {
            "question": "What gets compromised in space-optimized LCS?",
            "options": ["Time", "Result correctness", "Sequence recovery", "Subproblem definition"],
            "answer": "Sequence recovery"
        },
        {
            "question": "Which data structure helps simulate a 2-row matrix in Python?",
            "options": ["Deque", "Tuple", "List of Lists", "Set"],
            "answer": "List of Lists"
        },
        {
            "question": "What is the key advantage of space-optimized LCS?",
            "options": ["More matches", "Faster indexing", "Less memory usage", "Higher accuracy"],
            "answer": "Less memory usage"
        },
        {
            "question": "Which of these cannot reconstruct the LCS string?",
            "options": ["Full matrix", "Backtracking with directions", "Space-optimized method", "Standard DP"],
            "answer": "Space-optimized method"
        },
        {
            "question": "Which optimization improves both time and space in recursive LCS?",
            "options": ["Tabulation", "Memoization", "Greedy recursion", "Sorting"],
            "answer": "Memoization"
        },
        {
            "question": "Which of the following is true for memory-efficient LCS?",
            "options": ["LCS string is always recoverable", "Only length is computed", "Time increases drastically", "It’s only used for short strings"],
            "answer": "Only length is computed"
        },
        {
            "question": "Which is required to compute LCS length efficiently?",
            "options": ["Full traceback matrix", "Sorted strings", "Only previous row", "Character positions"],
            "answer": "Only previous row"
        },
        {
            "question": "Which memory-saving technique is not applicable to LCS?",
            "options": ["Compressing matrix", "Row recycling", "Diagonal skipping", "Rolling buffer"],
            "answer": "Diagonal skipping"
        },
        {
            "question": "Which Python trick is often used in space-optimized LCS loops?",
            "options": ["List slicing", "Heapify", "Lambda sorting", "Threading"],
            "answer": "List slicing"
        },
        {
            "question": "Why is LCS length not affected by space optimization?",
            "options": ["It skips rows", "It skips characters", "Only current values are needed", "It uses hash tables"],
            "answer": "Only current values are needed"
        },
        {
            "question": "In memory-limited environments, which LCS approach is preferred?",
            "options": ["Greedy", "Recursive", "Space-optimized", "Hashing"],
            "answer": "Space-optimized"
        }

    ],
    "level 3":[
        {
            "question": "Which optimization is used in LCS when only length is required?",
            "options": ["Tabulation with traceback", "Greedy choice", "Two-row dynamic programming", "String hashing"],
            "answer": "Two-row dynamic programming"
        },
        {
            "question": "Which structure is reused in space-efficient LCS?",
            "options": ["Full matrix", "Only first row", "Previous and current row", "Diagonal values"],
            "answer": "Previous and current row"
        },
        {
            "question": "What is the primary limitation of space-optimized LCS?",
            "options": ["Slower runtime", "Loss of output length", "Cannot retrieve sequence", "More memory"],
            "answer": "Cannot retrieve sequence"
        },
        {
            "question": "Which optimization improves both time and space in recursion?",
            "options": ["Memoization", "Tabulation", "Binary search", "Sorting"],
            "answer": "Memoization"
        },
        {
            "question": "How can LCS be computed with O(n) space?",
            "options": ["By storing diagonals only", "Using greedy matching", "Using 2 rolling arrays", "Using recursion"],
            "answer": "Using 2 rolling arrays"
        },
        {
            "question": "Which approach is ideal when only LCS length is needed for long strings?",
            "options": ["Memoized recursion", "Backtracking", "Space-efficient DP", "Divide and conquer"],
            "answer": "Space-efficient DP"
        },
        {
            "question": "What is the biggest downside of optimizing space in LCS?",
            "options": ["Speed drop", "Loss of accuracy", "Loss of traceback info", "Stack overflow"],
            "answer": "Loss of traceback info"
        },
        {
            "question": "Which form of LCS uses O(n*m) space?",
            "options": ["Space-optimized", "Top-down memoized", "Bottom-up full table", "Greedy"],
            "answer": "Bottom-up full table"
        },
        {
            "question": "Which scenario best suits space optimization?",
            "options": ["Short strings", "Identical inputs", "Huge inputs", "Zero matches"],
            "answer": "Huge inputs"
        },
        {
            "question": "Which optimization does not apply to LCS?",
            "options": ["Diagonal prefetch", "Rolling row", "Memoization", "Bottom-up tabulation"],
            "answer": "Diagonal prefetch"
        },
        {
            "question": "Which of these is required to build the actual LCS sequence?",
            "options": ["Current row only", "Full DP table", "Greedy reconstruction", "List of lengths"],
            "answer": "Full DP table"
        },
        {
            "question": "Which is better for mobile applications with limited memory?",
            "options": ["Recursive LCS", "Full matrix", "Space-optimized LCS", "Brute force LCS"],
            "answer": "Space-optimized LCS"
        },
        {
            "question": "Which language feature is best for writing rolling arrays in Python?",
            "options": ["Lambda", "Generators", "List slicing", "Multithreading"],
            "answer": "List slicing"
        },
        {
            "question": "In which case is full matrix allocation mandatory?",
            "options": ["Length only", "Backtracking result", "Greedy search", "Fastest runtime"],
            "answer": "Backtracking result"
        },
        {
            "question": "Space-optimized LCS is ideal when:",
            "options": ["We need the exact sequence", "We need only LCS length", "Time is critical", "Space is abundant"],
            "answer": "We need only LCS length"
        }

    ]
}
tf_questions = {

"Level 1": [        {"question": "Space-optimized LCS saves memory by reusing rows.", "answer": True},
        {"question": "We can always recover the LCS string using space-optimized method.", "answer": False},
        {"question": "Space optimization does not affect the LCS length.", "answer": True},
        {"question": "The full DP table is required to get LCS length.", "answer": False},
        {"question": "Rolling row technique can reduce memory from O(n*m) to O(n).", "answer": True},
        {"question": "Only one row is enough to compute LCS length.", "answer": False},
        {"question": "Memoization reduces both time and space.", "answer": False},
        {"question": "LCS with full DP matrix uses O(n*m) space.", "answer": True},
        {"question": "You must store the entire LCS matrix to compute the length.", "answer": False},
        {"question": "Using only two rows is sufficient to store intermediate results in LCS.", "answer": True},
        {"question": "Memory-optimized LCS is faster than standard LCS in all cases.", "answer": False},
        {"question": "In Python, lists can be reused to store rolling rows.", "answer": True},
        {"question": "Space optimization helps in computing all possible LCS sequences.", "answer": False},
        {"question": "If only the LCS length is needed, space optimization is ideal.", "answer": True},
        {"question": "Storing fewer rows affects LCS correctness.", "answer": False}
],
"level 2":[
            {"question": "Space-optimized LCS can only compute the length, not the sequence.", "answer": True},
        {"question": "The LCS matrix must be fully stored in space-efficient versions.", "answer": False},
        {"question": "Rolling arrays require only two rows at a time.", "answer": True},
        {"question": "Space optimization reduces both memory and accuracy.", "answer": False},
        {"question": "List slicing is useful in space-efficient Python implementations.", "answer": True},
        {"question": "Only the last computed row is needed to compute the next row.", "answer": True},
        {"question": "You can retrieve the full LCS string from a 2-row optimized table.", "answer": False},
        {"question": "Memory-efficient LCS is ideal when comparing huge text files.", "answer": True},
        {"question": "Using 1D array instead of 2D in LCS is possible for length computation.", "answer": True},
        {"question": "Memoization is used to save time, not space.", "answer": True},
        {"question": "Using greedy techniques guarantees LCS optimization.", "answer": False},
        {"question": "In space-optimized LCS, directions (↖, ↑, ←) are not stored.", "answer": True},
        {"question": "Space-optimized LCS must use bottom-up approach.", "answer": True},
        {"question": "The entire input strings must be reversed in space-efficient LCS.", "answer": False},
        {"question": "Rolling row technique applies to both row-major and column-major traversals.", "answer": True}

],
"level 3":[
        {"question": "Space optimization in LCS helps save memory for long strings.", "answer": True},
        {"question": "You can always retrieve the LCS sequence in a space-optimized version.", "answer": False},
        {"question": "Only two rows are needed to calculate LCS length using DP.", "answer": True},
        {"question": "List slicing in Python supports space optimization in LCS.", "answer": True},
        {"question": "Loss of traceback is a trade-off in space-efficient LCS.", "answer": True},
        {"question": "Memoization increases memory usage but improves speed.", "answer": True},
        {"question": "Tabulation always uses less space than recursion.", "answer": False},
        {"question": "Space-efficient DP cannot work if strings are identical.", "answer": False},
        {"question": "Storing full matrix is required to recover the actual sequence.", "answer": True},
        {"question": "Greedy algorithms can replace DP in space-optimized LCS.", "answer": False},
        {"question": "Memory-efficient LCS uses loop nesting and row reuse.", "answer": True},
        {"question": "Rolling arrays are ideal when LCS length is sufficient.", "answer": True},
        {"question": "Space optimization in LCS compromises on accuracy.", "answer": False},
        {"question": "It is impossible to do LCS with less than O(n*m) memory.", "answer": False},
        {"question": "Using 2 rows in space-optimized LCS is a form of dynamic programming.", "answer": True}

]
}