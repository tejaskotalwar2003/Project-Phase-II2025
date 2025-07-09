mcq_questions = {
    "Level 1": [
        {
            "question": "What does LCS stand for in dynamic programming?",
            "options": ["Longest Common Subsequence", "Least Common Subset", "Longest Constant Substring", "Last Common Subsequence"],
            "answer": "Longest Common Subsequence"
        },
        {
            "question": "Which technique is primarily used to solve LCS?",
            "options": ["Recursion", "Greedy", "Dynamic Programming", "Backtracking"],
            "answer": "Dynamic Programming"
        },
        {
            "question": "What does the LCS of two strings represent?",
            "options": ["The maximum matching substring", "The longest sequence common to both strings in order", "The number of edits required", "The reversed common substring"],
            "answer": "The longest sequence common to both strings in order"
        },
        {
            "question": "LCS allows characters to be:",
            "options": ["Repeated", "Skipped", "Sorted", "Hashed"],
            "answer": "Skipped"
        },
        {
            "question": "Which of the following is TRUE about LCS?",
            "options": ["It only works for numbers", "It considers order of characters", "It returns length of longest substring", "It needs strings of equal length"],
            "answer": "It considers order of characters"
        },
        {
            "question": "The base case of LCS when any string is empty is:",
            "options": ["1", "Length of other string", "0", "Infinity"],
            "answer": "0"
        },
        {
            "question": "LCS is useful in which real-world application?",
            "options": ["Sorting arrays", "Syntax error detection", "Version control diff", "Binary search"],
            "answer": "Version control diff"
        },
        {
            "question": "The LCS of 'abc' and 'def' is:",
            "options": ["ab", "a", "", "abc"],
            "answer": ""
        },
        {
            "question": "Which of the following is NOT required for bottom-up LCS?",
            "options": ["DP table", "Iteration", "Memoization", "String lengths"],
            "answer": "Memoization"
        },
        {
            "question": "Which step is performed when characters match during LCS computation?",
            "options": ["Skip both", "Add 1 and move diagonally", "Subtract 1", "Move right"],
            "answer": "Add 1 and move diagonally"
        },
        {
            "question": "If two strings are identical, their LCS will be:",
            "options": ["One of them", "Empty string", "Only vowels", "Reversed string"],
            "answer": "One of them"
        },
        {
            "question": "How many subproblems are there in classic LCS DP?",
            "options": ["n", "m", "n*m", "n+m"],
            "answer": "n*m"
        },
        {
            "question": "What is returned by classic LCS algorithm?",
            "options": ["The LCS string", "The LCS length", "Both", "None"],
            "answer": "The LCS length"
        },
        {
            "question": "Which statement is FALSE about LCS?",
            "options": ["It preserves character order", "It skips unmatched characters", "It finds longest repeated string", "It uses overlapping subproblems"],
            "answer": "It finds longest repeated string"
        },
        {
            "question": "What is the time complexity of bottom-up LCS?",
            "options": ["O(n)", "O(m+n)", "O(n*m)", "O(n^2*m)"],
            "answer": "O(n*m)"
        }

    ],
    "level 2":[
        {
            "question": "Which property allows LCS to be solved using dynamic programming?",
            "options": ["Greedy choice", "Backtracking nature", "Optimal substructure", "Non-determinism"],
            "answer": "Optimal substructure"
        },
        {
            "question": "What does overlapping subproblems mean in LCS?",
            "options": ["No base cases exist", "Same subproblem is solved multiple times", "Only greedy is used", "One solution leads to all"],
            "answer": "Same subproblem is solved multiple times"
        },
        {
            "question": "In LCS, if str1[i-1] == str2[j-1], what is the update to dp[i][j]?",
            "options": ["dp[i-1][j] + 1", "dp[i-1][j-1] + 1", "dp[i][j-1] + 1", "dp[i][j] = 1"],
            "answer": "dp[i-1][j-1] + 1"
        },
        {
            "question": "What is the recurrence relation for LCS when characters mismatch?",
            "options": ["dp[i][j] = 0", "dp[i][j] = 1", "dp[i][j] = max(dp[i-1][j], dp[i][j-1])", "dp[i][j] = dp[i-1][j-1]"],
            "answer": "dp[i][j] = max(dp[i-1][j], dp[i][j-1])"
        },
        {
            "question": "What is the first step in bottom-up LCS?",
            "options": ["Initialize full matrix with 1", "Initialize dp[0][j] and dp[i][0] as 0", "Recursively call subproblems", "Print strings"],
            "answer": "Initialize dp[0][j] and dp[i][0] as 0"
        },
        {
            "question": "What does dp[i][j] represent in LCS DP table?",
            "options": ["LCS of str1[0:i-1] and str2[0:j-1]", "Longest string from i to j", "Characters that match", "Length of str1"],
            "answer": "LCS of str1[0:i-1] and str2[0:j-1]"
        },
        {
            "question": "Which part of the DP matrix gives the final answer in LCS?",
            "options": ["dp[0][0]", "dp[len1][len2]", "dp[1][1]", "dp[len2][len1]"],
            "answer": "dp[len1][len2]"
        },
        {
            "question": "How many directions are possible when tracing LCS string?",
            "options": ["2", "1", "3", "4"],
            "answer": "3"
        },
        {
            "question": "Which direction in DP table means 'match' in LCS?",
            "options": ["Left", "Up", "Diagonal", "Right"],
            "answer": "Diagonal"
        },
        {
            "question": "What does LCS focus on?",
            "options": ["Substrings", "Permutations", "Subsequences", "Prefixes"],
            "answer": "Subsequences"
        },
        {
            "question": "If one of the input strings is empty, what will LCS length be?",
            "options": ["Length of other string", "Infinity", "0", "1"],
            "answer": "0"
        },
        {
            "question": "Which of the following helps reconstruct the actual LCS string?",
            "options": ["Just length", "Backtracking from DP table", "Top-down recursion", "Greedy guess"],
            "answer": "Backtracking from DP table"
        },
        {
            "question": "What type of problem does LCS best represent?",
            "options": ["String hashing", "Sequence alignment", "Randomized search", "Binary search"],
            "answer": "Sequence alignment"
        },
        {
            "question": "What is the value in cell dp[i][j] if str1[i-1] != str2[j-1]?",
            "options": ["dp[i-1][j-1] + 1", "dp[i][j-1] + dp[i-1][j]", "max(dp[i-1][j], dp[i][j-1])", "1"],
            "answer": "max(dp[i-1][j], dp[i][j-1])"
        },
        {
            "question": "Which of the following is a correct use case for LCS?",
            "options": ["Finding minimum spanning tree", "Comparing versions of files", "Encrypting passwords", "Finding shortest path"],
            "answer": "Comparing versions of files"
        }

    ],
    "level 3":[
        {
            "question": "Why is LCS considered a classic dynamic programming problem?",
            "options": ["Because it uses sorting", "Due to its optimal substructure and overlapping subproblems", "It’s recursive only", "It uses divide and conquer only"],
            "answer": "Due to its optimal substructure and overlapping subproblems"
        },
        {
            "question": "What type of sequences does LCS operate on?",
            "options": ["Permutations", "Combinations", "Ordered subsequences", "Random characters"],
            "answer": "Ordered subsequences"
        },
        {
            "question": "Which property of LCS makes memoization useful?",
            "options": ["No base case", "Repeated subproblems", "Binary recursion", "Sorted input"],
            "answer": "Repeated subproblems"
        },
        {
            "question": "What is a limitation of LCS in semantic similarity?",
            "options": ["It uses too much memory", "It ignores character meaning", "It needs sorting", "It returns unordered output"],
            "answer": "It ignores character meaning"
        },
        {
            "question": "In the DP matrix, a match results in which direction?",
            "options": ["Down", "Right", "Diagonal", "Up"],
            "answer": "Diagonal"
        },
        {
            "question": "Which type of input pair will yield empty LCS?",
            "options": ["Equal strings", "Partially similar", "Completely different", "Palindromes"],
            "answer": "Completely different"
        },
        {
            "question": "Which technique is used to reduce LCS space complexity?",
            "options": ["Recursion", "Memoization", "Rolling array", "Greedy approach"],
            "answer": "Rolling array"
        },
        {
            "question": "Which of the following is most likely to have more than one valid LCS?",
            "options": ["ABC and DEF", "ABC and ACB", "AB and AB", "AAA and AA"],
            "answer": "ABC and ACB"
        },
        {
            "question": "How is LCS different from Edit Distance?",
            "options": ["It doesn’t allow gaps", "It returns a sequence, not a number", "It is not a DP problem", "It changes input"],
            "answer": "It returns a sequence, not a number"
        },
        {
            "question": "Which part of the matrix is used to reconstruct the full LCS string?",
            "options": ["First row", "Main diagonal", "Full matrix", "Only first column"],
            "answer": "Full matrix"
        },
        {
            "question": "What happens to LCS length if one character is removed from both strings?",
            "options": ["Always decreases", "Never changes", "Depends on character removed", "Always increases"],
            "answer": "Depends on character removed"
        },
        {
            "question": "Why is dynamic programming better than recursion in LCS?",
            "options": ["It uses more memory", "It avoids recomputation", "It sorts strings", "It’s approximate"],
            "answer": "It avoids recomputation"
        },
        {
            "question": "Which variation of LCS allows matching multiple LCS sequences?",
            "options": ["Backtracking", "Greedy LCS", "All-path tracing", "Weighted LCS"],
            "answer": "All-path tracing"
        },
        {
            "question": "Which metric is closest in nature to LCS?",
            "options": ["Jaccard Similarity", "Edit Distance", "Hamming Distance", "Substring search"],
            "answer": "Edit Distance"
        },
        {
            "question": "Which approach is not helpful for LCS reconstruction?",
            "options": ["Full DP table", "Backtracking", "Memoization-only", "Diagonal walk"],
            "answer": "Memoization-only"
        }

    ]
}
tf_questions = {
   "Level 1": [
        {"question": "LCS considers the relative order of characters.", "answer": True},
        {"question": "LCS requires that characters be contiguous.", "answer": False},
        {"question": "An empty string has LCS length 0 with any string.", "answer": True},
        {"question": "LCS can be used for file comparison in version control systems.", "answer": True},
        {"question": "The LCS of two completely different strings is empty.", "answer": True},
        {"question": "Greedy algorithm is optimal for solving LCS.", "answer": False},
        {"question": "The result of LCS is always unique.", "answer": False},
        {"question": "LCS can be implemented both recursively and iteratively.", "answer": True},
        {"question": "LCS ignores the order of characters in strings.", "answer": False},
        {"question": "Memoization is useful for top-down LCS.", "answer": True},
        {"question": "Bottom-up LCS uses a 2D table to store subproblem results.", "answer": True},
        {"question": "The LCS of a string with itself is the string itself.", "answer": True},
        {"question": "LCS does not support Unicode strings.", "answer": False},
        {"question": "The LCS string is always a substring of both input strings.", "answer": False},
        {"question": "The length of the LCS is always less than or equal to the length of the shorter string.", "answer": True}

    ],
    "level 2":[
        {"question": "LCS exhibits optimal substructure and overlapping subproblems.", "answer": True},
        {"question": "LCS can only be implemented using recursion.", "answer": False},
        {"question": "In bottom-up LCS, a match leads to a diagonal move.", "answer": True},
        {"question": "Memoization is used in top-down LCS to avoid recomputation.", "answer": True},
        {"question": "Backtracking is used to reconstruct the LCS string from the table.", "answer": True},
        {"question": "If two characters match, LCS length is decreased.", "answer": False},
        {"question": "The DP matrix for LCS must be square.", "answer": False},
        {"question": "Every LCS implementation gives only the length by default.", "answer": True},
        {"question": "The character order matters in LCS.", "answer": True},
        {"question": "LCS ignores unmatched characters while computing length.", "answer": True},
        {"question": "Base cases in LCS are always set to -1.", "answer": False},
        {"question": "Top-down LCS uses recursion with memoization.", "answer": True},
        {"question": "In bottom-up LCS, we fill the table from left to right and top to bottom.", "answer": True},
        {"question": "Diagonal cell dp[i-1][j-1] is used when characters match.", "answer": True},
        {"question": "In LCS, order of matching characters must be preserved.", "answer": True}

    ],
    "level 3":[
        {"question": "LCS can be used to find similarities between different languages.", "answer": False},
        {"question": "Multiple valid LCS sequences may exist for the same input.", "answer": True},
        {"question": "LCS treats characters semantically while comparing.", "answer": False},
        {"question": "LCS assumes characters are case-sensitive by default.", "answer": True},
        {"question": "In LCS, two different inputs can yield the same LCS length.", "answer": True},
        {"question": "LCS can be extended to handle three or more strings.", "answer": True},
        {"question": "The longer the input strings, the more likely the LCS is empty.", "answer": False},
        {"question": "LCS requires inputs to be sorted for correct computation.", "answer": False},
        {"question": "Space-optimized LCS sacrifices traceback ability.", "answer": True},
        {"question": "The length of the LCS is always ≤ min(len(str1), len(str2)).", "answer": True},
        {"question": "Dynamic programming always stores full matrix for LCS.", "answer": False},
        {"question": "LCS matrix grows linearly with string length.", "answer": False},
        {"question": "LCS uses dynamic programming due to overlapping subproblems.", "answer": True},
        {"question": "The order of characters is not relevant in LCS.", "answer": False},
        {"question": "You cannot get the actual LCS string using only memoized recursion.", "answer": True}

    ] 
}