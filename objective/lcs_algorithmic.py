mcq_questions = {
    "Level 1": [        {
            "question": "What is the main approach used in solving LCS?",
            "options": ["Greedy", "Dynamic Programming", "Depth-First Search", "Divide and Conquer"],
            "answer": "Dynamic Programming"
        },
        {
            "question": "What is the base case in the LCS DP table?",
            "options": ["All values are 1", "First row and column are 0", "Diagonal values are 1", "No base case"],
            "answer": "First row and column are 0"
        },
        {
            "question": "Which of the following is true if str1[i-1] == str2[j-1]?",
            "options": ["dp[i][j] = dp[i-1][j] + 1", "dp[i][j] = dp[i][j-1] + 1", "dp[i][j] = dp[i-1][j-1] + 1", "dp[i][j] = 1"],
            "answer": "dp[i][j] = dp[i-1][j-1] + 1"
        },
        {
            "question": "If characters do not match, which formula is used?",
            "options": ["dp[i][j] = 0", "dp[i][j] = max(dp[i-1][j], dp[i][j-1])", "dp[i][j] = dp[i-1][j-1]", "dp[i][j] = min(dp[i-1][j], dp[i][j-1])"],
            "answer": "dp[i][j] = max(dp[i-1][j], dp[i][j-1])"
        },
        {
            "question": "What is the time complexity of LCS using Dynamic Programming?",
            "options": ["O(n)", "O(m)", "O(n*m)", "O(log n)"],
            "answer": "O(n*m)"
        },
        {
            "question": "What is the LCS of 'abc' and 'abc'?",
            "options": ["abc", "ab", "bc", "c"],
            "answer": "abc"
        },
        {
            "question": "Which matrix dimension is used in bottom-up LCS?",
            "options": ["len1 x len2", "(len1+1) x (len2+1)", "len1+len2", "2 x n"],
            "answer": "(len1+1) x (len2+1)"
        },
        {
            "question": "LCS follows which of the following principles?",
            "options": ["Greedy Choice", "Overlapping Subproblems", "Randomization", "In-place swap"],
            "answer": "Overlapping Subproblems"
        },
        {
            "question": "What is returned from LCS(str1, str2)?",
            "options": ["A number", "A character", "Length of LCS", "None"],
            "answer": "Length of LCS"
        },
        {
            "question": "Is LCS solvable using top-down and bottom-up DP?",
            "options": ["Only top-down", "Only bottom-up", "Yes, both", "No"],
            "answer": "Yes, both"
        },
        {
            "question": "What happens when one string is empty?",
            "options": ["Return -1", "Return 0", "Return 1", "Return str2"],
            "answer": "Return 0"
        },
        {
            "question": "What kind of problem does LCS represent?",
            "options": ["Optimization", "Search", "Enumeration", "Sorting"],
            "answer": "Optimization"
        },
        {
            "question": "Which step follows initialization of the LCS matrix?",
            "options": ["Backtracking", "Recursion", "Filling the table", "Sorting characters"],
            "answer": "Filling the table"
        },
        {
            "question": "How is the LCS length finally obtained?",
            "options": ["From dp[0][0]", "From max of matrix", "From dp[len1][len2]", "From sum of diagonals"],
            "answer": "From dp[len1][len2]"
        },
        {
            "question": "Which structure best describes the solution space of LCS?",
            "options": ["Array", "Tree", "Graph", "Matrix"],
            "answer": "Matrix"
        }
    ],
    "Level 2": [        
         {
            "question": "Which approach avoids stack overflow in LCS computation?",
            "options": ["Top-down recursion", "Bottom-up dynamic programming", "Pure backtracking", "Greedy recursion"],
            "answer": "Bottom-up dynamic programming"
        },
        {
            "question": "Which condition causes a move diagonally in the LCS DP table?",
            "options": ["Characters match", "Characters differ", "First column", "Base case"],
            "answer": "Characters match"
        },
        {
            "question": "Which loop structure is used to fill the LCS matrix?",
            "options": ["Nested for-loops", "Recursive call", "While loop", "Do-while"],
            "answer": "Nested for-loops"
        },
        {
            "question": "When characters mismatch, which formula is applied?",
            "options": ["dp[i][j] = 0", "dp[i][j] = max(dp[i-1][j], dp[i][j-1])", "dp[i][j] = dp[i-1][j-1]", "dp[i][j] = -1"],
            "answer": "dp[i][j] = max(dp[i-1][j], dp[i][j-1])"
        },
        {
            "question": "Which structure is most efficient for storing the LCS table?",
            "options": ["2D array", "HashMap", "List", "Queue"],
            "answer": "2D array"
        },
        {
            "question": "Which statement is true about LCS recursive implementation?",
            "options": ["It has linear time complexity", "It always uses constant memory", "It’s slower than bottom-up", "It doesn’t require base case"],
            "answer": "It’s slower than bottom-up"
        },
        {
            "question": "What do you store in dp[i][j] if str1[i-1] == str2[j-1]?",
            "options": ["0", "1", "dp[i-1][j-1] + 1", "dp[i][j-1] + 1"],
            "answer": "dp[i-1][j-1] + 1"
        },
        {
            "question": "Which technique can avoid recomputation in LCS recursion?",
            "options": ["Loop unrolling", "Memoization", "Lazy evaluation", "Stack reuse"],
            "answer": "Memoization"
        },
        {
            "question": "Which value indicates LCS length when the DP matrix is filled?",
            "options": ["dp[0][0]", "dp[-1][-1]", "dp[len1][len2]", "dp[1][1]"],
            "answer": "dp[len1][len2]"
        },
        {
            "question": "Which part of the LCS table is required for sequence reconstruction?",
            "options": ["First row", "Entire matrix", "Last column", "Only diagonal"],
            "answer": "Entire matrix"
        },
        {
            "question": "Which case will always result in LCS length 0?",
            "options": ["Same strings", "One empty string", "Two substrings", "Repeating characters"],
            "answer": "One empty string"
        },
        {
            "question": "Which concept does LCS not use?",
            "options": ["Overlapping subproblems", "Optimal substructure", "Greedy choice property", "Recursion"],
            "answer": "Greedy choice property"
        },
        {
            "question": "What causes exponential time in plain recursive LCS?",
            "options": ["Matrix creation", "Sorting strings", "Repeated subproblems", "Greedy branching"],
            "answer": "Repeated subproblems"
        },
        {
            "question": "Which version of LCS reduces time by avoiding recomputation?",
            "options": ["Memoized recursion", "Iterative backtracking", "Brute force", "Sorted LCS"],
            "answer": "Memoized recursion"
        },
        {
            "question": "If str1 and str2 are reversed, what happens to LCS length?",
            "options": ["Same", "Always zero", "Length doubles", "Depends on content"],
            "answer": "Same"
        }],
        "level 3":[
        {
            "question": "What is the space complexity of the optimized LCS approach?",
            "options": ["O(n*m)", "O(n)", "O(1)", "O(log n)"],
            "answer": "O(n)"
        },
        {
            "question": "Which condition is used to move diagonally in the LCS table?",
            "options": ["Characters do not match", "End of string", "Characters match", "Base case"],
            "answer": "Characters match"
        },
        {
            "question": "Which of the following is required to reconstruct the LCS string?",
            "options": ["Only last row", "First column", "Full matrix", "Length of LCS"],
            "answer": "Full matrix"
        },
        {
            "question": "What causes high time complexity in naive recursive LCS?",
            "options": ["Use of extra arrays", "Excessive memory", "Repeated subproblems", "Greedy steps"],
            "answer": "Repeated subproblems"
        },
        {
            "question": "Which direction is followed in backtracking LCS from the matrix?",
            "options": ["Right", "Down", "Diagonal", "Up"],
            "answer": "Diagonal"
        },
        {
            "question": "Which technique helps avoid recomputation in recursive LCS?",
            "options": ["DFS", "Greedy approach", "Memoization", "Binary search"],
            "answer": "Memoization"
        },
        {
            "question": "Which problem type does LCS most closely resemble?",
            "options": ["Graph traversal", "Matrix chain multiplication", "Edit distance", "Knapsack"],
            "answer": "Edit distance"
        },
        {
            "question": "How many recursive calls are made in plain recursive LCS (worst case)?",
            "options": ["Linear", "Polynomial", "Exponential", "Logarithmic"],
            "answer": "Exponential"
        },
        {
            "question": "What causes optimal substructure in LCS?",
            "options": ["Backtracking", "Using prefix combinations", "Divide and conquer", "Reuse of past results"],
            "answer": "Reuse of past results"
        },
        {
            "question": "Which one is NOT a valid optimization in LCS?",
            "options": ["Memoization", "Two-row matrix", "Greedy shortcut", "Tabulation"],
            "answer": "Greedy shortcut"
        },
        {
            "question": "Which recurrence defines LCS when characters match?",
            "options": ["dp[i][j] = 0", "dp[i][j] = dp[i-1][j-1] + 1", "dp[i][j] = max(i, j)", "dp[i][j] = i + j"],
            "answer": "dp[i][j] = dp[i-1][j-1] + 1"
        },
        {
            "question": "Which Python structure suits a 2D LCS matrix?",
            "options": ["Dictionary", "List of lists", "Set", "Tuple"],
            "answer": "List of lists"
        },
        {
            "question": "What is the minimum space required for LCS computation?",
            "options": ["1 row", "2 rows", "Entire matrix", "No storage"],
            "answer": "2 rows"
        },
        {
            "question": "LCS on strings of length n and m will fill how many DP cells?",
            "options": ["n*m", "n+m", "n^2", "m^2"],
            "answer": "n*m"
        },
        {
            "question": "What property allows us to skip recomputing dp[i][j]?",
            "options": ["Memoization", "Caching prefix sums", "Sorting strings", "Dynamic slicing"],
            "answer": "Memoization"
        }

        ]

    }

tf_questions = {
    "Level 1": [
                {"question": "The LCS algorithm is an example of dynamic programming.", "answer": True},
        {"question": "LCS always uses a recursive function internally.", "answer": False},
        {"question": "The LCS matrix grows with the product of input sizes.", "answer": True},
        {"question": "Diagonal movement in LCS matrix implies a match.", "answer": True},
        {"question": "Top-down and bottom-up DP give different LCS results.", "answer": False},
        {"question": "If two strings are identical, their LCS is their full length.", "answer": True},
        {"question": "LCS cannot be solved using tabulation.", "answer": False},
        {"question": "The final cell in the matrix contains the length of LCS.", "answer": True},
        {"question": "Memoization is used in bottom-up LCS.", "answer": False},
        {"question": "In LCS, we compare str1[i] with str2[j] directly in loops.", "answer": False},
        {"question": "LCS assumes input strings are sorted.", "answer": False},
        {"question": "LCS matrix is always square.", "answer": False},
        {"question": "Recursive LCS has exponential time without memoization.", "answer": True},
        {"question": "Filling the DP table takes linear time.", "answer": False},
        {"question": "Initialization of the matrix includes setting first row and column to zero.", "answer": True}
    ],
    "level 2":[
        {"question": "LCS follows the principle of optimal substructure.", "answer": True},
        {"question": "Memoization stores the results of already solved subproblems.", "answer": True},
        {"question": "The bottom-up LCS approach uses recursion.", "answer": False},
        {"question": "A full DP table is needed to calculate LCS length.", "answer": False},
        {"question": "If characters match, we move diagonally in the table.", "answer": True},
        {"question": "In recursive LCS, overlapping subproblems occur.", "answer": True},
        {"question": "A greedy approach always gives optimal LCS.", "answer": False},
        {"question": "LCS is based on dynamic programming principles.", "answer": True},
        {"question": "LCS matrix dimensions are (len1+1)x(len2+1).", "answer": True},
        {"question": "Backtracking is necessary to extract LCS string.", "answer": True},
        {"question": "LCS cannot be implemented using recursion.", "answer": False},
        {"question": "Memoization increases time complexity.", "answer": False},
        {"question": "Each DP cell dp[i][j] represents the LCS length of prefixes.", "answer": True},
        {"question": "Recursive LCS with memoization is faster than bottom-up.", "answer": False},
        {"question": "All diagonal values in the LCS table are always equal.", "answer": False}

    ],
    "level 3":[
        {"question": "Recursive LCS without memoization is exponential in time.", "answer": True},
        {"question": "DP table size depends on the lengths of input strings.", "answer": True},
        {"question": "All LCS problems require full matrix reconstruction.", "answer": False},
        {"question": "LCS uses a top-down approach only.", "answer": False},
        {"question": "Characters that match in LCS must appear in the same order.", "answer": True},
        {"question": "Memoization is a top-down optimization.", "answer": True},
        {"question": "Tabulation is slower than recursion.", "answer": False},
        {"question": "The cell dp[i][j] contains the LCS length of prefixes str1[0:i] and str2[0:j].", "answer": True},
        {"question": "LCS can be solved in logarithmic space.", "answer": False},
        {"question": "Overlapping subproblems are common in LCS.", "answer": True},
        {"question": "LCS cannot be implemented without backtracking.", "answer": False},
        {"question": "Diagonal movement in LCS matrix means a character match.", "answer": True},
        {"question": "If two characters don't match, we use the maximum of left and top.", "answer": True},
        {"question": "The entire matrix must be stored for LCS length only.", "answer": False},
        {"question": "In bottom-up DP, previous row values are reused.", "answer": True}

    ]
}