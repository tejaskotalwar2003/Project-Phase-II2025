import re

def generate_answer(category, level, question):
    if category.lower() == "conceptual":
        return "Answer generation for conceptual questions is not implemented yet."

    if category == "Algorithmic":
        return answer_algorithmic(level, question)
    elif category == "Application":
        return answer_application(level, question)
    elif category == "Optimization":
        return answer_optimization(level, question)

    return "No answer available for this category."

def answer_algorithmic(level, question):
    if level == "Level 1":
        return answer_algo_lvl1(question)
    elif level == "Level 2":
        return answer_algo_lvl2(question)
    elif level == "Level 3":
        return answer_algo_lvl3(question)
    return "No answer for this level."

def answer_application(level, question):
    return f"(Application-based answer for {level} will be added soon.)"

def answer_optimization(level, question):
    return f"(Optimization-based answer for {level} will be added soon.)"

def answer_algo_lvl1(question):
    q = question.lower()

    if "recursive function" in q and "length of the lcs" in q:
        match = re.search(r"lcs.*between ['\"](.*?)['\"] and ['\"](.*?)['\"]", question, re.IGNORECASE)
        if match:
            s1 = match.group(1)
            s2 = match.group(2)
            m, n = len(s1), len(s2)
            return (
                f"To solve the LCS problem recursively for the strings '{s1}' and '{s2}':\n\n"
                "🧠 Step-by-step explanation:\n"
                "1. Base case: If either string is empty, LCS is 0.\n"
                "2. If the last characters match, we add 1 to the result of the smaller subproblem.\n"
                "3. Otherwise, we take the maximum of:\n"
                "   - LCS of first string shortened by 1\n"
                "   - LCS of second string shortened by 1\n\n"
                "🔁 Recursive function:\n"
                "```python\n"
                "def lcs_recursive(s1, s2, m, n):\n"
                "    if m == 0 or n == 0:\n"
                "        return 0\n"
                "    if s1[m-1] == s2[n-1]:\n"
                "        return 1 + lcs_recursive(s1, s2, m-1, n-1)\n"
                "    return max(lcs_recursive(s1, s2, m-1, n), lcs_recursive(s1, s2, m, n-1))\n"
                "```\n\n"
                f"📥 Inputs:\n"
                f"  s1 = '{s1}', s2 = '{s2}'\n"
                f"  m = {m}, n = {n}\n\n"
                f"📤 Call: lcs_recursive('{s1}', '{s2}', {m}, {n})\n"
                f"🕒 Time Complexity: O(2^{min(m, n)}) (exponential)\n"
            )
        else:
            return "Sorry, I couldn't extract the input strings to answer this recursive LCS question."

    if ("construct" in q or "create" in q or "build" in q) and ("dp table" in q or "2d table" in q):
        match = re.search(r"for ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "Could not extract input strings for DP table construction."

        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        steps = []

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    steps.append(f"dp[{i}][{j}] (chars '{s1[i - 1]}' == '{s2[j - 1]}'): set to dp[{i - 1}][{j - 1}] + 1 = {dp[i][j]}")
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    steps.append(f"dp[{i}][{j}] (chars '{s1[i - 1]}' != '{s2[j - 1]}'): max(dp[{i - 1}][{j}]={dp[i - 1][j]}, dp[{i}][{j - 1}]={dp[i][j - 1]}) = {dp[i][j]}")

        header = "    " + "  ".join(s2)
        rows = []
        for i in range(m + 1):
            label = s1[i - 1] if i > 0 else " "
            row_str = "  ".join(str(dp[i][j]) for j in range(n + 1))
            rows.append(f"{label} {row_str}")

        dp_table_str = header + "\n" + "\n".join(rows)

        answer = f"DP table construction for '{s1}' and '{s2}':\n\n"
        answer += "Step-by-step updates:\n"
        for step in steps:
            answer += f"- {step}\n"
        answer += "\nFinal DP Table:\n" + dp_table_str + "\n"
        answer += f"LCS length is in dp[{m}][{n}] = {dp[m][n]}.\n"

        return answer
    
    # Check if the question is about initializing the DP table
    if "initialize" in q and ("first row" in q or "first column" in q) and "dp table" in q:
        match = re.search(r"for ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "Could not extract input strings for initialization explanation."

        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        return (
            f"To initialize the first row and first column of the DP table for LCS between '{s1}' and '{s2}':\n\n"
            "🧱 Initialization Steps:\n"
            "1. Create a table `dp` of size (m+1) x (n+1) filled with zeros.\n"
            "2. The first row and first column represent comparisons with an empty string.\n"
            "3. Since the LCS with an empty string is 0, initialize:\n"
            "   - dp[0][j] = 0 for all j\n"
            "   - dp[i][0] = 0 for all i\n\n"
            "📌 Example code:\n"
            "```python\n"
            f"m, n = {m}, {n}\n"
            "dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]\n"
            "# First row and first column are already 0 by initialization\n"
            "```\n"
            "✅ This sets up the base cases needed for the dynamic programming approach.\n"
        )
#..........................................
    # Check if the question is about updating the DP table when characters do not match
    if "dp" in q and "when characters do not match" in q and ("dp[" in q or "cell" in q):
        match = re.search(r"dp\[(\d+)]\[(\d+)]", question)
        if not match:
            return "Could not identify the dp cell indices from the question."

        i, j = int(match.group(1)), int(match.group(2))
        return (
            f"To update the cell `dp[{i}][{j}]` when characters at those positions do not match:\n\n"
            "📘 Rule:\n"
            "When characters from the two strings at current indices do **not** match,\n"
            "we choose the maximum from the **left** and **top** cells in the DP table.\n\n"
            "🔁 Formula:\n"
            f"`dp[{i}][{j}] = max(dp[{i-1}][{j}], dp[{i}][{j-1}])`\n\n"
            "📌 Reason:\n"
            "- dp[i-1][j] → Exclude character from the first string.\n"
            "- dp[i][j-1] → Exclude character from the second string.\n"
            "- We take the best result from both possibilities.\n\n"
            "✅ This ensures the LCS is correctly built even when characters differ."
        )
#..........................................
    # Check if the question is about updating the DP table when characters match
    if "base recursive formula" in q and "matches" in q and "position" in q:
        match = re.search(
        r"['\"]?(?P<char1>[A-Za-z])['\"]? string's value at position (\d+) matches ['\"]?(?P<char2>[A-Za-z])['\"]? string's value at position (\d+)",
        question
        )

        if not match:
            return "Could not extract matching characters and their positions from the question."

        char1 = match.group("char1")
        char2 = match.group("char2")
        i = int(match.group(2))
        j = int(match.group(4))

        return (
            f"When characters match — '{char1}' at position {i} and '{char2}' at position {j}:\n\n"
            "📘 Recursive Formula:\n"
            "```python\n"
            "if s1[i] == s2[j]:\n"
            "    return 1 + lcs(s1, s2, i - 1, j - 1)\n"
            "```\n\n"
            "🧠 Explanation:\n"
            "- Since the characters match, we include this match in our result (hence `+1`).\n"
            "- Then we solve the subproblem excluding both matched characters (i.e., go diagonally back).\n\n"
            f"✅ This forms the key recurrence relation when '{char1}' and '{char2}' match at indices {i} and {j}, respectively."
        )

#..........................................
    # Check if the question is about a dry run of the bottom-up LCS algorithm
    if "dry run" in q and "bottom-up" in q and "lcs" in q:
        match = re.search(r"on strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "Could not extract input strings for bottom-up dry run."

        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        steps = []

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    steps.append(f"dp[{i}][{j}] (chars '{s1[i - 1]}' == '{s2[j - 1]}') → dp[{i - 1}][{j - 1}] + 1 = {dp[i][j]}")
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    steps.append(f"dp[{i}][{j}] (chars '{s1[i - 1]}' != '{s2[j - 1]}') → max(dp[{i - 1}][{j}]={dp[i - 1][j]}, dp[{i}][{j - 1}]={dp[i][j - 1]}) = {dp[i][j]}")

        header = "      " + " ".join(s2)  # 5 spaces before header labels

        rows = []
        for i in range(m + 1):
            label = s1[i - 1] if i > 0 else " "
            row_str = "  ".join(str(dp[i][j]) for j in range(n + 1))
            rows.append(f"{label}   {row_str}")  # 3 spaces after label for alignment

        dp_table_str = header + "\n" + "\n".join(rows)

        answer = f"🔍 Dry run of bottom-up LCS for strings '{s1}' and '{s2}':\n\n"
        answer += "📘 Step-by-step updates to the DP table:\n"
        for step in steps:
            answer += f"- {step}\n"
        answer += "\n🧾 Final DP Table:\n" + dp_table_str + "\n"
        answer += f"✅ LCS length is dp[{m}][{n}] = {dp[m][n]}.\n"

        return answer

#..........................................
    # Check if the question is about a specific function to compute LCS length
    if "function" in q and "computes the lcs length" in q and "bottom-up" in q:
        match = re.search(r"function ['\"]?(\w+)['\"]?", question)
        if not match:
            return "Could not extract the function name from the question."

        func_name = match.group(1)
        return (
            f"Here's a bottom-up dynamic programming implementation of the function `{func_name}` to compute the LCS length:\n\n"
            "🧠 Explanation:\n"
            "- We build a DP table of size (m+1) x (n+1), where `m` and `n` are the lengths of the input strings.\n"
            "- We fill the table row by row comparing characters.\n"
            "- If characters match, we take diagonal +1. If not, we take the max of left and top cells.\n\n"
            "```python\n"
            f"def {func_name}(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]\n"
            "    \n"
            "    for i in range(1, m + 1):\n"
            "        for j in range(1, n + 1):\n"
            "            if s1[i - 1] == s2[j - 1]:\n"
            "                dp[i][j] = dp[i - 1][j - 1] + 1\n"
            "            else:\n"
            "                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])\n"
            "    \n"
            "    return dp[m][n]\n"
            "```\n\n"
            "📥 Example:\n"
            f"`{func_name}('ABC', 'AC')` → returns `2`\n"
            "✅ This function computes the length of the Longest Common Subsequence using bottom-up DP."
        )
#..........................................
    # Check if the question is about the first row of the DP table
    if "first row" in q and "dp table" in q:
        match = re.search(r"for the strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "Could not extract input strings for the DP row explanation."

        s1, s2 = match.group(1), match.group(2)
        n = len(s2)
        row = [0] * (n + 1)
        return (
            f"In the DP table for strings '{s1}' and '{s2}', the first row corresponds to comparing an empty first string with the second string.\n\n"
            f"🔢 First Row Values (dp[0][j] for j = 0 to {n}):\n"
            f"{row}\n\n"
            "📘 Reason:\n"
            "- An empty string has no subsequences, so the LCS length is 0 when compared to any prefix of the second string.\n"
            "- Therefore, the entire first row is filled with zeros."
        )


    #...........................................    

    if "how many comparisons" in q and "recursively" in q:
        match = re.search(r"strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "Could not extract strings to estimate recursive comparisons."

        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        min_len = min(m, n)
        return (
            f"To compute LCS recursively for '{s1}' and '{s2}' (lengths {m} and {n}):\n\n"
            "🔁 The recursive solution explores all combinations of characters, making repeated subproblem calls.\n"
            f"⏱️ Approximate comparisons: O(2^{min_len}) in worst case.\n"
            f"🧮 For these lengths, that’s roughly 2^{min_len} = {2 ** min_len} comparisons.\n\n"
            "📌 Note: This is due to overlapping subproblems and exponential branching when characters don't match."
        )

#..........................................
    # Check if the question is about creating a 2D table for LCS
    # Check if the question is about creating a 2D table for LCS
    if "create a 2d table" in q and "lcs" in q:
    # Match any two quoted strings joined by "and", flexible and reliable
        match = re.search(r"['\"]([^'\"]+)['\"]\s+and\s+['\"]([^'\"]+)['\"]", question)
        
        if not match:
            return "❌ Could not extract input strings for DP table construction."

        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)

        return (
            f"To calculate the LCS for '{s1}' and '{s2}', create a 2D table `dp` of size ({m+1}) x ({n+1}):\n\n"
            "📐 Table Dimensions:\n"
            f"- Rows: {m+1} (length of first string + 1)\n"
            f"- Columns: {n+1} (length of second string + 1)\n\n"
            "📘 Reason:\n"
            "- The extra row and column handle the base case of empty prefixes.\n"
            "- Every cell `dp[i][j]` stores the length of the LCS of `s1[0..i-1]` and `s2[0..j-1]`.\n\n"
            "🔧 Initialization Code:\n"
            "```python\n"
            f"dp = [[0 for _ in range({n+1})] for _ in range({m+1})]\n"
            "```"
        )





#..........................................
    # Check if the question is about the LCS length when no characters match
    if "lcs length" in q and "no characters match" in q:
        match = re.search(r"strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "Could not extract strings for LCS edge case answer."

        s1, s2 = match.group(1), match.group(2)
        return (
            f"If none of the characters in '{s1}' and '{s2}' match:\n\n"
            "📉 Result:\n"
            "The LCS length will be 0.\n\n"
            "📘 Reason:\n"
            "- A common subsequence requires at least one shared character.\n"
            "- With no matches, there's no common subsequence → length = 0.\n\n"
            "✅ Final Answer: **0**"
        )

#..........................................
    # Check if the question is about the LCS length when no characters match
    if "lcs length" in q and "no characters match" in q:
        match = re.search(r"strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "Could not extract strings for LCS edge case answer."

        s1, s2 = match.group(1), match.group(2)
        return (
            f"If none of the characters in '{s1}' and '{s2}' match:\n\n"
            "📉 Result:\n"
            "The LCS length will be 0.\n\n"
            "📘 Reason:\n"
            "- A common subsequence requires at least one shared character.\n"
            "- With no matches, there's no common subsequence → length = 0.\n\n"
            "✅ Final Answer: **0**"
        )

#..........................................
    # Check if the question is about filling in the first few cells of the DP table
    if "fill in the first few cells" in q and "dp table" in q:
        match = re.search(r"strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "Could not extract input strings for filling DP cells."

        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(1, min(m + 1, 3)):
            for j in range(1, min(n + 1, 3)):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        partial = "\n".join(" ".join(map(str, row[:3])) for row in dp[:3])
        return (
            f"Here are the first few DP table cells for strings '{s1}' and '{s2}':\n\n"
            f"{partial}\n\n"
            "📘 Only partial initialization is shown to illustrate the early cell updates."
        )

#..........................................

    if "time complexity" in q and "iterate" in q and "loop" in q:
        return (
            "To compare each index of one string with each index of another in nested loops:\n\n"
            "📐 For strings of length `m` and `n`, this leads to:\n"
            "```python\n"
            "for i in range(m):\n"
            "    for j in range(n):\n"
            "        compare(s1[i], s2[j])\n"
            "```\n"
            "⏱️ Time Complexity: **O(m × n)**\n"
            "✅ This is standard for bottom-up dynamic programming LCS."
        )

#..........................................
    # Check if the question is about the DP table when characters do not match
    if "do not match" in q and "dp" in q and "values are compared" in q:
        match = re.search(r"positions (\d+) and (\d+)", question)
        if not match:
            return "Could not extract index positions."

        i, j = int(match.group(1)), int(match.group(2))
        return (
            f"When characters at positions {i} and {j} do not match, we use the rule:\n\n"
            f"📘 `dp[{i}][{j}] = max(dp[{i-1}][{j}], dp[{i}][{j-1}])`\n\n"
            "✅ This ensures we carry forward the best LCS length without including the mismatch."
        )

    # Check if the question is about the number of comparisons in a recursive LCS
    if "purpose" in q and ("dp[i-1][j]" in q or "dp[i][j-1]" in q):
        return (
            "In the LCS DP recurrence, if characters don't match, we choose the longer of:\n"
            "- `dp[i-1][j]`: Exclude current character from first string.\n"
            "- `dp[i][j-1]`: Exclude current character from second string.\n\n"
            "📘 Comparing both ensures we don't miss the optimal substructure."
        )

#..........................................
    # Check if the question is about the steps to compute LCS using bottom-up DP
    if "steps" in q and "bottom-up" in q and "lcs" in q:
        return (
            "🪜 Steps to compute LCS using bottom-up DP:\n"
            "1. Initialize a 2D DP table of size (m+1) x (n+1).\n"
            "2. Fill the first row and column with 0 (empty string case).\n"
            "3. Loop i from 1 to m, and j from 1 to n:\n"
            "   - If s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1] + 1\n"
            "   - Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
            "4. The result is in dp[m][n]."
        )

#..........................................
    # Check if the question is about the best data structure for storing intermediate values
    if "data structure" in q and "intermediate values" in q:
        return (
            "📦 The best data structure to store intermediate values in LCS is a **2D array (matrix)**.\n"
            "- It allows O(1) access to subproblem results.\n"
            "- Size: (m+1) x (n+1), where m and n are the lengths of the strings."
        )


#.........  

    if "loop structure" in q and "fill the dp table" in q:
        return (
            "📘 Loop structure for filling the LCS DP table from bottom-up:\n"
            "```python\n"
            "for i in range(1, m + 1):\n"
            "    for j in range(1, n + 1):\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            dp[i][j] = dp[i-1][j-1] + 1\n"
            "        else:\n"
            "            dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
            "```"
        )


#...............

    if "final value" in q and "dp[" in q and "represent" in q:
        match = re.search(r"dp\[(\d+)]\[(\d+)]", question)
        if not match:
            return "Could not extract the final cell coordinates."

        i, j = match.group(1), match.group(2)
        return (
            f"The final value in `dp[{i}][{j}]` represents:\n"
            "- The **length of the Longest Common Subsequence** of the two full input strings.\n"
            "✅ It is the result of the bottom-up DP algorithm."
        )

#....................

    if "pseudocode" in q and "nested for-loop" in q and "dp table" in q:
        return (
            "🧠 Pseudocode for computing LCS using bottom-up DP:\n"
            "```\n"
            "function LCS(s1, s2):\n"
            "    m = length of s1\n"
            "    n = length of s2\n"
            "    create dp[0..m][0..n] and fill with 0\n\n"
            "    for i from 1 to m:\n"
            "        for j from 1 to n:\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                dp[i][j] = dp[i-1][j-1] + 1\n"
            "            else:\n"
            "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n\n"
            "    return dp[m][n]\n"
            "```"
        )


    # Default fallback answer for unrecognized questions
    return "Answer generation for this question is not implemented yet."

def answer_algo_lvl2(question):
    q = question.lower()

    # 1️⃣ Memoized Recursive Function
    if "memoized recursive" in q and "lcs length" in q:
        match = re.search(r"for ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.group(1), match.group(2)
            m, n = len(s1), len(s2)
            return (
                f"📌 Goal: Compute LCS length between '{s1}' and '{s2}' using **memoized recursion**.\n\n"
                "🧠 Why Memoization?\n"
                "- Naive recursion recomputes the same subproblems multiple times.\n"
                "- Memoization caches results → avoids redundancy → boosts efficiency.\n\n"
                "🔁 Recursive Strategy:\n"
                "1. If either string is empty → return 0.\n"
                "2. If characters match → 1 + LCS of prefixes.\n"
                "3. Else → max of two smaller subproblems.\n\n"
                "📦 Memoization Tool: `@lru_cache` from `functools`\n\n"
                "```python\n"
                "from functools import lru_cache\n\n"
                f"s1 = '{s1}'\n"
                f"s2 = '{s2}'\n\n"
                "@lru_cache(None)\n"
                "def lcs(i, j):\n"
                "    if i == 0 or j == 0:\n"
                "        return 0\n"
                "    if s1[i-1] == s2[j-1]:\n"
                "        return 1 + lcs(i-1, j-1)\n"
                "    return max(lcs(i-1, j), lcs(i, j-1))\n\n"
                f"print('LCS Length:', lcs({m}, {n}))\n"
                "```\n\n"
                f"📥 Inputs: '{s1}', '{s2}' → lengths: {m}, {n}\n"
                "🕒 Time Complexity: O(m × n)\n"
                "🧠 Space: O(m × n) due to recursion + memo table\n"
                "✅ This version avoids recomputation and is efficient even for long strings!"
            )
    # 2️⃣ Build and Return the LCS String
    if "builds and returns the lcs string" in q:
        match = re.search(r"for ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.group(1), match.group(2)
            return (
                f"📌 Task: Build and return the **actual LCS string** between '{s1}' and '{s2}'.\n\n"
                "🧠 Approach:\n"
                "- First, fill the DP table as usual.\n"
                "- Then, backtrack from dp[m][n] to reconstruct the sequence.\n\n"
                "```python\n"
                "def lcs_string(s1, s2):\n"
                "    m, n = len(s1), len(s2)\n"
                "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
                "    for i in range(1, m+1):\n"
                "        for j in range(1, n+1):\n"
                "            if s1[i-1] == s2[j-1]:\n"
                "                dp[i][j] = dp[i-1][j-1] + 1\n"
                "            else:\n"
                "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
                "    i, j = m, n\n"
                "    lcs = []\n"
                "    while i > 0 and j > 0:\n"
                "        if s1[i-1] == s2[j-1]:\n"
                "            lcs.append(s1[i-1])\n"
                "            i -= 1\n"
                "            j -= 1\n"
                "        elif dp[i-1][j] > dp[i][j-1]:\n"
                "            i -= 1\n"
                "        else:\n"
                "            j -= 1\n"
                "    return ''.join(reversed(lcs))\n"
                "```\n\n"
                "✅ This gives both the LCS length and one possible LCS sequence."
            )

    # 3️⃣ Convert Top-Down to Bottom-Up
    if "convert a top-down" in q and "bottom-up" in q:
        return (
            "🔁 Convert top-down LCS to bottom-up:\n\n"
            "📘 In top-down:\n"
            "- You use recursion + memoization.\n\n"
            "📘 In bottom-up:\n"
            "- You create a DP table and fill it iteratively.\n\n"
            "🧱 Steps:\n"
            "1. Create a 2D table `dp[m+1][n+1]`.\n"
            "2. Loop through each i, j.\n"
            "3. Apply recurrence:\n"
            "   - If match: `dp[i][j] = dp[i-1][j-1] + 1`\n"
            "   - Else: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`\n\n"
            "✅ Bottom-up removes stack overhead and is iterative in nature."
        )

    # 4️⃣ Space Complexity Optimization
    if "space complexity" in q:
        return (
            "🧠 Normal bottom-up LCS uses **O(m × n)** space.\n\n"
            "💡 But you can optimize space to **O(min(m, n))** by using only two rows:\n\n"
            "```python\n"
            "prev = [0] * (n + 1)\n"
            "curr = [0] * (n + 1)\n"
            "for i in range(1, m + 1):\n"
            "    for j in range(1, n + 1):\n"
            "        if s1[i - 1] == s2[j - 1]:\n"
            "            curr[j] = prev[j - 1] + 1\n"
            "        else:\n"
            "            curr[j] = max(prev[j], curr[j - 1])\n"
            "    prev, curr = curr, [0] * (n + 1)\n"
            "```\n\n"
            "✅ This saves memory and works well for large inputs."
        )

    # 5️⃣ Reconstruct All LCS Strings
    if "all possible lcs" in q:
        return (
            "🔄 To reconstruct **all** LCS sequences:\n"
            "1. Build the full DP table.\n"
            "2. Use DFS or backtracking to explore all matching paths.\n"
            "3. Use memoization to avoid recomputation.\n\n"
            "💡 Multiple LCS strings may exist due to equal choices during backtracking.\n"
            "✅ This is helpful when exact LCS structure matters, like in diff tools."
        )

    # 6️⃣ Minimal Space Row-by-Row
    if "row by row" in q and "minimal space" in q:
        return (
            "🧮 Row-by-row LCS using minimal space (2 rows):\n"
            "```python\n"
            "prev = [0] * (n + 1)\n"
            "curr = [0] * (n + 1)\n"
            "for i in range(1, m + 1):\n"
            "    for j in range(1, n + 1):\n"
            "        if s1[i - 1] == s2[j - 1]:\n"
            "            curr[j] = prev[j - 1] + 1\n"
            "        else:\n"
            "            curr[j] = max(prev[j], curr[j - 1])\n"
            "    prev, curr = curr, [0] * (n + 1)\n"
            "```\n"
            "✅ Efficient space usage with the same LCS result."
        )
    # 7️⃣ Single-Dimensional Array
    if "single-dimensional array" in q:
        return (
            "📏 LCS with single 1D array optimization:\n\n"
            "🧠 Use a single array and `prev` variable to simulate row-wise update.\n"
            "⚠️ Carefully track previous values since the current value depends on both left and diagonal entries.\n\n"
            "```python\n"
            "dp = [0] * (n + 1)\n"
            "for i in range(1, m + 1):\n"
            "    prev = 0\n"
            "    for j in range(1, n + 1):\n"
            "        temp = dp[j]\n"
            "        if s1[i - 1] == s2[j - 1]:\n"
            "            dp[j] = prev + 1\n"
            "        else:\n"
            "            dp[j] = max(dp[j], dp[j - 1])\n"
            "        prev = temp\n"
            "```\n\n"
            "✅ Saves even more memory — just one array."
        )

    # 8️⃣ Return Both LCS Length and Sequence
    if "returns both the lcs length" in q.lower():
        match = re.search(r"['\"]([A-Z]+)['\"].*?['\"]([A-Z]+)['\"]", q, flags=re.IGNORECASE)
        
        if match:
            s1, s2 = match.groups()
            
            return (
                f"🧾 Here's a Python function to return both the LCS length and one valid subsequence for '{s1}' and '{s2}':\n\n"
                "```python\n"
                "def lcs_with_length(s1, s2):\n"
                "    m, n = len(s1), len(s2)\n"
                "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n\n"
                "    # Build the DP table\n"
                "    for i in range(1, m + 1):\n"
                "        for j in range(1, n + 1):\n"
                "            if s1[i - 1] == s2[j - 1]:\n"
                "                dp[i][j] = dp[i - 1][j - 1] + 1\n"
                "            else:\n"
                "                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])\n\n"
                "    # Backtrack to find one valid LCS\n"
                "    i, j = m, n\n"
                "    lcs = []\n"
                "    while i > 0 and j > 0:\n"
                "        if s1[i - 1] == s2[j - 1]:\n"
                "            lcs.append(s1[i - 1])\n"
                "            i -= 1\n"
                "            j -= 1\n"
                "        elif dp[i - 1][j] >= dp[i][j - 1]:\n"
                "            i -= 1\n"
                "        else:\n"
                "            j -= 1\n"
                "    lcs.reverse()\n"
                "    return dp[m][n], ''.join(lcs)\n\n"
                f"# Example usage:\n"
                f"length, subseq = lcs_with_length('{s1}', '{s2}')\n"
                "print('LCS Length:', length)\n"
                "print('LCS:', subseq)\n"
                "```"
            )
        else:
            return "❌ Couldn't extract strings. Please ask with proper quoted inputs like `'PINTER'` and `'PAINTER'`."


    if "recursion tree" in q and "without using memoization" in q:
        match = re.search(r"for ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if match:
            s1, s2 = match.group(1), match.group(2)
            m, n = len(s1), len(s2)
            return (
                f"🌲 Let’s analyze the recursion tree for computing LCS('{s1}', '{s2}') without memoization.\n\n"
                "🧠 Base Idea:\n"
                "- At every step, we compare s1[i-1] and s2[j-1]\n"
                "- If they match → move diagonally (i-1, j-1)\n"
                "- Else → branch into two recursive calls:\n"
                "   • LCS(i-1, j)\n"
                "   • LCS(i, j-1)\n\n"
                "🔁 This creates a binary recursion tree with exponential growth.\n"
                f"💥 For '{s1}' and '{s2}', total recursive calls ≈ O(2^{min(m, n)}) without memoization.\n\n"
                "🔂 Duplicate Subproblems Example:\n"
                "- You may compute LCS(3, 2) multiple times via different paths.\n"
                "- No value is reused, leading to inefficiency.\n\n"
                "📦 Solution: Memoization\n"
                "- Use a cache (e.g., @lru_cache or a dict) to store results of LCS(i, j).\n"
                "- Avoid recomputing → prune the recursion tree drastically.\n\n"
                "✅ With memoization, time complexity drops to O(m × n), and recursion tree becomes a DAG.\n"
                "➡️ Much faster and scalable for large inputs."
            )


    # 🔟 Identify Duplicate Subproblems
    if "duplicate subproblems" in q.lower():
    # Extract input strings from the question
        match = re.search(r"LCS.*?['\"]([A-Z]+)['\"].*?['\"]([A-Z]+)['\"]", q, flags=re.IGNORECASE)
        
        if match:
            s1, s2 = match.groups()
            m, n = len(s1), len(s2)

            # Choose two common repeated subproblem indices
            i1, j1 = m - 1, n - 1
            i2, j2 = m - 2, n - 2

            # Get the substrings corresponding to those indices
            sub1_i1, sub2_j1 = s1[:i1], s2[:j1]
            sub1_i2, sub2_j2 = s1[:i2], s2[:j2]

            return (
                f"🔍 Comparing '{s1}' and '{s2}':\n\n"
                "🌀 In naive recursion, many subproblems like `LCS(i, j)` are **repeated**.\n"
                f"For example:\n"
                f"• `LCS({i1}, {j1})` corresponds to '{sub1_i1}' and '{sub2_j1}'\n"
                f"• `LCS({i2}, {j2})` corresponds to '{sub1_i2}' and '{sub2_j2}'\n"
                "These calls may occur again and again from different branches of the recursion tree.\n\n"
                "💡 **Memoization** stores results of already computed subproblems `(i, j)` in a table.\n"
                "✅ This avoids recomputation and reduces time complexity from **exponential** to **O(m × n)**.\n"
            )
        else:
            return (
                "🌀 Naive LCS recursion **repeats many subproblems**, e.g., `LCS(i, j)` is called multiple times.\n\n"
                "💡 Memoization caches results of `(i, j)` so each subproblem is solved only once.\n"
                "✅ This reduces time complexity from exponential to polynomial (O(m × n))."
            )
    # 1️⃣1️⃣ Size of DP Table
    if "size of the dp table" in q.lower() and "number of cells" in q.lower():
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if match:
            s1, s2 = match.group(1), match.group(2)
            m, n = len(s1), len(s2)
            return (
                f"📏 For strings '{s1}' and '{s2}' (lengths {m}, {n}):\n\n"
                f"🧮 DP Table Size = ({m + 1} × {n + 1}) = **{(m + 1) * (n + 1)} cells**\n"
                "📘 Extra row and column handle empty prefixes (base case)."
            )
        else:
            return (
                "❌ Could not extract the input strings. Please format the question like:\n"
                "`Explain size of DP table for 'STRING1' and 'STRING2'`"
            )
    # 1️⃣2️⃣ Print Memo Table
    if "print the memo table" in q:
        return (
            "🧠 In a memoized recursive LCS, we usually use `@lru_cache` or a dictionary.\n\n"
            "📋 To inspect the memoized values:\n"
            "- Use a custom dictionary instead of `@lru_cache`.\n"
            "- Print the dictionary after computation.\n\n"
            "💡 Example:\n"
            "```python\n"
            "memo = {}\n"
            "def lcs(i, j):\n"
            "    if i == 0 or j == 0:\n"
            "        return 0\n"
            "    if (i, j) in memo:\n"
            "        return memo[(i, j)]\n"
            "    if s1[i-1] == s2[j-1]:\n"
            "        memo[(i, j)] = 1 + lcs(i-1, j-1)\n"
            "    else:\n"
            "        memo[(i, j)] = max(lcs(i-1, j), lcs(i, j-1))\n"
            "    return memo[(i, j)]\n"
            "```"
        )

    # 1️⃣3️⃣ Diagonal Movement
    if "diagonal movement" in q.lower():
        return (
            "↘️ Diagonal movement in the LCS DP table refers to the case when characters match:\n\n"
            "🧠 If s1[i-1] == s2[j-1], then we set:\n"
            "`dp[i][j] = dp[i-1][j-1] + 1`\n\n"
            "✅ This means the current characters contribute to the LCS, so we:\n"
            "1. Add 1 to the LCS length.\n"
            "2. Move diagonally up-left (↖️) to continue checking remaining substrings.\n\n"
            "📌 For example:\n"
            "  If s1 = 'PINTER' and s2 = 'PAINTER',\n"
            "  at position i=2 (s1[1]='I') and j=3 (s2[2]='I'), characters match,\n"
            "  so we use the diagonal cell and add 1:\n"
            "  `dp[2][3] = dp[1][2] + 1`\n\n"
            "💡 Diagonal steps are what form the actual LCS characters during backtracking."
        )

    if "backtrack through the dp table" in q:
        match = re.search(r"for ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if match:
            s1, s2 = match.group(1), match.group(2)
            m, n = len(s1), len(s2)
            return (
                f"📌 Goal: Reconstruct the actual LCS from the computed DP table for:\n"
                f"  - String 1: '{s1}'\n"
                f"  - String 2: '{s2}'\n\n"
                "🔍 Approach:\n"
                "1. Start from the bottom-right of the DP table (i = m, j = n).\n"
                "2. If characters match: move diagonally ↖️ and add the character.\n"
                "3. Else: move to the cell with the larger value (↑ or ←).\n"
                "4. Collect characters, then reverse to get the correct order.\n\n"
                "🧠 This prints one valid LCS sequence.\n\n"
                "✅ Implementation:\n"
                "```python\n"
                f"s1 = '{s1}'\n"
                f"s2 = '{s2}'\n"
                "m, n = len(s1), len(s2)\n\n"
                "dp = [[0] * (n+1) for _ in range(m+1)]\n"
                "for i in range(1, m+1):\n"
                "    for j in range(1, n+1):\n"
                "        if s1[i-1] == s2[j-1]:\n"
                "            dp[i][j] = dp[i-1][j-1] + 1\n"
                "        else:\n"
                "            dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n\n"
                "# 🔁 Backtrack to build the LCS\n"
                "i, j = m, n\n"
                "lcs = []\n"
                "while i > 0 and j > 0:\n"
                "    if s1[i-1] == s2[j-1]:\n"
                "        lcs.append(s1[i-1])\n"
                "        i -= 1\n"
                "        j -= 1\n"
                "    elif dp[i-1][j] > dp[i][j-1]:\n"
                "        i -= 1\n"
                "    else:\n"
                "        j -= 1\n"
                "lcs = ''.join(reversed(lcs))\n"
                "print('LCS:', lcs)\n"
                "```\n\n"
                f"🧪 Example Input: '{s1}' and '{s2}' → LCS = printed from above code\n"
                "🧠 Time Complexity: O(m × n)\n"
                "🧠 Space Complexity: O(m × n)"
            )


    # 1️⃣5️⃣ One String is Empty
    if "one of the strings is empty" in q:
        return (
            "📉 If either input string is empty:\n"
            "- There can be no common subsequence.\n"
            "- Hence, the LCS length is **0**.\n\n"
            "✅ Base case for LCS recurrence."
        )

    # 1️⃣6️⃣ Ignore Spaces & Punctuation
    if "ignore spaces and punctuation" in q:
        return (
            "✂️ To ignore spaces and punctuation in LCS:\n\n"
            "📘 Preprocess input strings:\n"
            "```python\n"
            "import re\n"
            "s1 = re.sub(r'[^a-zA-Z0-9]', '', s1)\n"
            "s2 = re.sub(r'[^a-zA-Z0-9]', '', s2)\n"
            "```\n"
            "✅ Now run standard LCS algorithm on cleaned strings."
        )
    # 1️⃣7️⃣ Min Insertions/Deletions using LCS
    if "insertions and deletions" in q:
        return (
            "✏️ To convert one string into another using insertions and deletions:\n\n"
            "📘 Steps:\n"
            "1. Compute LCS length `L` between string1 and string2\n"
            "2. Deletions = len(string1) - L\n"
            "3. Insertions = len(string2) - L\n\n"
            "✅ Total operations = Insertions + Deletions"
        )

    # 1️⃣8️⃣ Full DP Table with LCS Path Highlight
   
    if "highlight the lcs path" in q.lower():
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if match:
            X, Y = match.group(1), match.group(2)
            m, n = len(X), len(Y)

            # Step 1: Build DP table
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i - 1] == Y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            # Step 2: Backtrack to find LCS path
            path = set()
            i, j = m, n
            while i > 0 and j > 0:
                if X[i - 1] == Y[j - 1]:
                    path.add((i, j))
                    i -= 1
                    j -= 1
                elif dp[i - 1][j] >= dp[i][j - 1]:
                    i -= 1
                else:
                    j -= 1

            # Step 3: Build visual table with highlights
            visual = "📊 DP Table with LCS Path Highlighted (`*` marks the LCS cells):\n\n"
            header = "    " + "  ".join(f"{ch}" for ch in " " + Y) + "\n"
            visual += header
            for i in range(m + 1):
                row = f"{X[i - 1] if i > 0 else ' '} "
                for j in range(n + 1):
                    val = f"{dp[i][j]}"
                    if (i, j) in path:
                        row += f"*{val}* "
                    else:
                        row += f" {val}  "
                visual += row.rstrip() + "\n"

            return visual
        else:
            return (
                "❌ Could not extract strings. Format your question like:\n"
                "`Print the full DP table after computing LCS between 'STRING1' and 'STRING2', and highlight the LCS path.`"
            )

    # 1️⃣9️⃣ Time and Space Complexity
    if "time and space complexity" in q:
        return (
            "📈 **Time Complexity:**\n"
            "- O(m × n) — where `m` and `n` are the lengths of the two input strings.\n\n"
            "📦 **Space Complexity:**\n"
            "- O(m × n) for full DP table.\n"
            "- O(n) if optimized with 2 rows or 1D array.\n\n"
            "✅ Efficient for most practical cases."
        )

    # 🔚 Default fallback for unhandled questions
    return "Answer generation for this Level 2 question is not implemented yet."

import re
from functools import lru_cache

def answer_algo_lvl3(question):
    q = question.lower()

    # 1️⃣ Reconstruct all possible LCS sequences
    if "all possible lcs sequences" in q or "reconstruct all possible lcs" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return f"""To reconstruct all LCS sequences for '{s1}' and '{s2}':
    1. Build a standard DP table using bottom-up dynamic programming.
    2. Backtrack from dp[m][n] using memoized recursion:
    - If characters match: go diagonally and append.
    - If not: explore all optimal branches (↑ and ←) when values are equal.
    3. Use a set to avoid storing duplicate subsequences.

    def all_lcs(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Build the DP table for LCS length
        for i in range(m):
            for j in range(n):
                if s1[i] == s2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

        from functools import lru_cache

        @lru_cache(None)
        def backtrack(i, j):
            if i == 0 or j == 0:
                return {{""}}
            if s1[i - 1] == s2[j - 1]:
                return {{seq + s1[i - 1] for seq in backtrack(i - 1, j - 1)}}
            
            res = set()
            if dp[i - 1][j] >= dp[i][j - 1]:
                res |= backtrack(i - 1, j)
            if dp[i][j - 1] >= dp[i - 1][j]:
                res |= backtrack(i, j - 1)
            return res

        # All sequences are built backwards, so reverse each sequence
        return {{seq[::-1] for seq in backtrack(m, n)}}

        
    🧠 Challenges involved:

    Multiple paths can give same LCS → need to explore all.

    Recursion may branch heavily, especially with ties.

    Duplicate LCS strings must be avoided using a set.

    Memory grows with the number of possible sequences.

    ✅ This method ensures all valid and unique LCS subsequences are generated.
    """

    
      # 2️⃣ Count the number of distinct LCS sequences
    if "number of distinct lcs sequences" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return f"""To count distinct LCS sequences for '{s1}' and '{s2}':
1. Build the DP table as in standard LCS.
2. Use memoized DFS to explore all LCS paths without storing actual strings.
3. At each step, return total counts from all valid directions.
4. Ensure overlapping subproblems are cached.

```python
def count_lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    from functools import lru_cache
    @lru_cache(None)
    def count(i, j):
        if i == 0 or j == 0:
            return 1
        if s1[i-1] == s2[j-1]:
            return count(i-1, j-1)
        res = 0
        if dp[i-1][j] == dp[i][j]:
            res += count(i-1, j)
        if dp[i][j-1] == dp[i][j]:
            res += count(i, j-1)
        if dp[i-1][j] == dp[i][j] and dp[i][j-1] == dp[i][j]:
            res -= count(i-1, j-1)  # Avoid double count
        return res

    return count(m, n)
🧠 Notes:

Only paths matching LCS length are counted.

Set subtraction avoids counting same path from two sides.

Returns a count, not the actual sequences.

✅ Efficiently gives number of unique LCS strings.
"""
        # 3️⃣ Compare LCS strategies: recursive, memoized, bottom-up, space-optimized
    if "compare" in q and "recursive" in q and "bottom-up" in q:
        return """📊 Comparing LCS Approaches:

1. **Recursive (Naive)**
   - No caching; explores all paths.
   - Time: O(2^min(m, n))
   - Space: O(m + n) (call stack)
   - ❌ Exponential — impractical for large inputs.

2. **Memoized (Top-Down)**
   - Uses recursion + caching via `@lru_cache`.
   - Time: O(m × n)
   - Space: O(m × n) + recursion depth
   - ✅ Efficient, but may hit recursion limits.

3. **Bottom-Up (Tabulation)**
   - Iterative 2D DP table.
   - Time: O(m × n)
   - Space: O(m × n)
   - ✅ Stable, readable, suitable for large inputs.

4. **Space-Optimized**
   - Uses two 1D rows instead of full table.
   - Time: O(m × n)
   - Space: O(n)
   - ✅ Ideal when memory is limited.

🧠 Summary:
- Prefer **bottom-up** for clarity, **space-optimized** for memory-critical tasks.
"""


       # 5️⃣ Parallelize the LCS DP table filling
    if "parallelize" in q and "dp table" in q:
        return """⚙️ How to Parallelize LCS DP Table Filling:

The LCS DP table has data dependencies: each dp[i][j] depends on dp[i−1][j], dp[i][j−1], and dp[i−1][j−1].  
To parallelize computation:

1. Process the table anti-diagonally — that is, all cells where i + j = constant can be computed in parallel.
2. Each diagonal forms a wavefront. At step k, compute all cells where i + j = k.

Benefits:
- Enables safe parallelization since all dependencies from previous diagonals are resolved.
- Can use threads or GPU blocks to process each diagonal in parallel.

✅ Ideal for performance improvement on large strings using multicore or CUDA.
"""

   
       # 6️⃣ Common prefix, suffix, and subsequence
    if "common prefix" in q and "suffix" in q and "subsequence" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return f"""To compute common prefix, suffix, and LCS between '{s1}' and '{s2}':

1. Common Prefix:
   Compare characters from start until mismatch.

2. Common Suffix:
   Compare characters from end until mismatch.

3. Longest Common Subsequence (LCS):
   Use standard DP-based LCS algorithm.

```python
def common_parts(s1, s2):
    # Prefix
    prefix = []
    for a, b in zip(s1, s2):
        if a == b:
            prefix.append(a)
        else:
            break
    prefix = ''.join(prefix)

    # Suffix
    suffix = []
    for a, b in zip(reversed(s1), reversed(s2)):
        if a == b:
            suffix.append(a)
        else:
            break
    suffix = ''.join(reversed(suffix))

    # LCS using DP
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    i, j = m, n
    lcs = []
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            lcs.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    lcs = ''.join(reversed(lcs))

    return prefix, suffix, lcs
✅ This returns the prefix, suffix, and LCS as three separate results.
"""
       # 7️⃣ Diagonal DP table generation for LCS
    if "generate" in q and "table diagonally" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return f"""To fill the LCS DP table diagonally for '{s1}' and '{s2}':

1. Process all table cells along anti-diagonals (i + j = constant).
2. Each anti-diagonal can be filled in a single pass, respecting dependencies.

This helps optimize memory locality and enables parallelization.

```python
def diagonal_lcs_dp(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for diag in range(2, m + n + 1):
        for i in range(1, m + 1):
            j = diag - i
            if 1 <= j <= n:
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
🧠 Pros:

Enables parallel execution along diagonals.

Better cache performance due to data locality.

⚠️ Cons:

Slightly more complex indexing logic.

Harder to debug for beginners.

✅ Use diagonal strategy for optimization and parallel-friendly LCS algorithms.
"""
       # 8️⃣ Analyze impact of shuffling on LCS length
    if "lcs length changes" in q and "shuffled" in q:
        match = re.search(r"['\"](.*?)['\"] or ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return f"""🎲 Analyzing effect of shuffling characters in '{s1}' or '{s2}' on LCS length:

- LCS depends on the relative order of characters.
- Shuffling breaks the original ordering → LCS length is likely to drop.

Example:

Original:
  s1 = "abcdef"
  s2 = "abcxyz"
  → LCS = "abc" → length = 3

Shuffled:
  s1 = "fedcba"
  s2 = "zyxcba"
  → LCS = "cba" or similar → length = still 3 or less, but ordering now matters.

🧠 Prediction:
- If both strings are shuffles of the same multiset of characters, LCS ≈ frequency overlap.
- If one string is fixed and the other is randomly shuffled, LCS tends to decrease.

✅ Shuffling generally reduces LCS length unless characters accidentally align.
"""

       # 9️⃣ Memory-efficient LCS using two rows
    if "memory-efficient" in q and "two rows" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return f"""To compute LCS between '{s1}' and '{s2}' using only two rows:

1. Full DP table uses O(m × n) space.
2. But each row depends only on the previous → use two 1D arrays.

This saves memory, especially for long strings.

```python
def space_optimized_lcs(s1, s2):
    m, n = len(s1), len(s2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]
✅ Time: O(m × n), Space: O(n)
✅ Ideal for large inputs where memory is constrained.
"""

       # 🔟 LCS for three strings
    if "three strings" in q or "3 strings" in q:
        match = re.search(r"['\"](.*?)['\"], ['\"](.*?)['\"](?:,| and) ['\"](.*?)['\"]", question)
        if match:
            s1, s2, s3 = match.groups()
            return f"""To find LCS among three strings '{s1}', '{s2}', and '{s3}':

1. Use 3D Dynamic Programming:
   dp[i][j][k] = length of LCS of s1[0..i-1], s2[0..j-1], s3[0..k-1]

2. Transition:
   - If characters match at i−1, j−1, k−1 → dp[i][j][k] = 1 + dp[i−1][j−1][k−1]
   - Else → max of excluding one character from any string.

```python
def lcs_3_strings(s1, s2, s3):
    m, n, o = len(s1), len(s2), len(s3)
    dp = [[[0]*(o+1) for _ in range(n+1)] for _ in range(m+1)]

    for i in range(1, m+1):
        for j in range(1, n+1):
            for k in range(1, o+1):
                if s1[i-1] == s2[j-1] == s3[k-1]:
                    dp[i][j][k] = dp[i-1][j-1][k-1] + 1
                else:
                    dp[i][j][k] = max(dp[i-1][j][k], dp[i][j-1][k], dp[i][j][k-1])

    return dp[m][n][o]
🧠 Complexity:

Time: O(m × n × o)

Space: O(m × n × o)

✅ Extends classic LCS by adding a third dimension to handle one more string.
"""
        # 1️⃣1️⃣ Create test cases for LCS edge cases
    if "test cases" in q and "edge cases" in q:
        return """🧪 Test Cases to Validate LCS Function

Here are common edge cases you should test for:

1. Empty Strings
```python
    assert lcs("", "") == ""
    assert lcs("abc", "") == ""
    assert lcs("", "xyz") == ""

2. Identical Strings

   assert lcs("abc", "abc") == "abc"

3. No Common Characters

   assert lcs("abc", "xyz") == ""
   
4. Repeated Characters


    assert lcs("aabba", "ababa") in ["abba", "aaba", "abaa"]
5. One Character Match


    assert lcs("x", "x") == "x"
    assert lcs("x", "y") == ""
6. Long Inputs

    # Optional performance test — doesn't assert, but runs
    lcs("a"*1000, "a"*1000)
✅ These test cases ensure correctness across minimal, typical, and large inputs.
"""
    # 1️⃣2️⃣ Iterative LCS with no recursion and constant space
    if "recursive-free" in q or ("iterative" in q and "constant space" in q):
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return f"""To compute LCS between '{s1}' and '{s2}' without recursion and using constant space:

1. Use two 1D arrays to track previous and current LCS values.
2. Update values iteratively using nested loops.
3. No recursion or full table storage required.

```python
def lcs_iterative_space_optimized(s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1  # Ensure s2 is shorter (to minimize space)
    m, n = len(s1), len(s2)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[n]

✅ Time: O(m × n)
✅ Space: O(n)
✅ Uses pure iteration — no recursion or full DP table.
"""

    # 1️⃣3️⃣ LCS variant allowing one unmatched character skip
    if "skipping one unmatched character" in q or "skip one unmatched" in q:
        match = re.search(r"['\"](.*?)['\"] or ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return f"""To compute LCS for '{s1}' and '{s2}' allowing one skip:

🔁 Idea:
- Standard LCS allows only exact matches.
- This variant allows skipping one unmatched character from either string once.

🧠 Approach:
- Extend DP state to track whether the skip has been used: dp[i][j][used_skip]
- Try skipping one character from s1 or s2 when mismatch occurs.

Example Implementation:

```python
def lcs_with_one_skip(s1, s2):
    from functools import lru_cache
    @lru_cache(None)
    def dp(i, j, used):
        if i == len(s1) or j == len(s2):
            return 0
        if s1[i] == s2[j]:
            return 1 + dp(i+1, j+1, used)
        skip_options = []
        if not used:
            skip_options = [dp(i+1, j, True), dp(i, j+1, True)]
        return max(dp(i+1, j, used), dp(i, j+1, used), *skip_options)

    return dp(0, 0, False)


    ✅ Returns the LCS length, allowing one character to be skipped from either string.
    ✅ Time: O(m × n × 2), Space: O(m × n × 2) with memoization.
    """

    # 1️⃣4️⃣ LCS with wildcards (e.g., '?', '*')

    elif "alternating subsequence" in q and "between" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return f"""🔀 Longest Common Alternating Subsequence (LCAS) between '{{s1}}' and '{{s2}}':

    🧠 Definition:
    - A subsequence that alternates between two or more characters. Example: 'a-b-a' or '1-2-1'.
    - It must also be a **common subsequence** of both strings.

    🆚 Difference from Standard LCS:
    - LCS only requires matching order of characters.
    - LCAS requires **alternation** (no two same characters adjacent in the result).

    👨‍💻 Approach:
    1. Generate all common subsequences (like in LCS).
    2. Filter those where adjacent characters alternate.
    3. Track the longest one among those.

    ```python
    def is_alternating(seq):
        return all(seq[i] != seq[i+1] for i in range(len(seq)-1))

    def lcas(s1, s2):
        from functools import lru_cache

        @lru_cache(None)
        def dp(i, j, prev):
            if i == len(s1) or j == len(s2):
                return ""
            res = ""
            if s1[i] == s2[j] and s1[i] != prev:
                take = s1[i] + dp(i+1, j+1, s1[i])
                if len(take) > len(res):
                    res = take
            skip1 = dp(i+1, j, prev)
            skip2 = dp(i, j+1, prev)
            return max(res, skip1, skip2, key=len)

        return dp(0, 0, "")
    📌 Example:

    s1 = "{s1}"
    s2 = "{s2}"
    print(lcas(s1, s2))
    ✅ This function will return the Longest Common Alternating Subsequence.
    """
    # 15️⃣ LCS on streams of characters instead of complete strings
    elif "streams of characters" in q and "instead of complete strings" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return """🔁 Adapting LCS to work on character **streams** like '{}' and '{}' :

    🧠 Problem:
    - LCS normally requires full strings in memory.
    - But streams (like file lines, data from sensors) can't be fully stored due to memory or time limits.

    💡 Solution Strategy:
    1. Use **online LCS algorithms** with limited memory.
    2. Maintain a sliding window or chunk buffer.
    3. Use memory-efficient DP (only two rows at a time).

    ⚙️ Approach:

    ```python
    def lcs_streaming(stream1, stream2, buffer_size=100):
        from collections import deque

        def process_chunk(chunk1, chunk2):
            m, n = len(chunk1), len(chunk2)
            prev = [0] * (n + 1)
            for i in range(m):
                curr = [0] * (n + 1)
                for j in range(n):
                    if chunk1[i] == chunk2[j]:
                        curr[j+1] = prev[j] + 1
                    else:
                        curr[j+1] = max(prev[j+1], curr[j])
                prev = curr
            return prev[-1]

        buffer1 = deque()
        buffer2 = deque()
        lcs_len = 0

        for c1, c2 in zip(stream1, stream2):
            buffer1.append(c1)
            buffer2.append(c2)
            if len(buffer1) == buffer_size:
                lcs_len += process_chunk(list(buffer1), list(buffer2))
                buffer1.clear()
                buffer2.clear()

        if buffer1:
            lcs_len += process_chunk(list(buffer1), list(buffer2))

        return lcs_len
    ✅ Use this when full input is not known ahead of time.
    ✅ Time: O(chunk_size² × chunks), Space: O(chunk_size)
    ✅ Works with generators, file streams, etc.
    """.format(s1, s2)

    # 1️⃣6️⃣ 
    elif "extra weight to vowels" in q or "capital letters" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return (
                "🌟 Weighted LCS for '{}' and '{}':\n\n"
                "🧠 Idea:\n"
                "- Regular LCS treats all characters equally (weight = 1).\n"
                "- We want to give **vowels and capital letters** more importance.\n\n"
                "📌 Weight Rule:\n"
                "- Vowels (a, e, i, o, u): 2 points\n"
                "- Capital letters: 2 points\n"
                "- Other characters: 1 point\n\n"
                "⚙️ Modified DP Algorithm:\n"
                "```python\n"
                "def weighted_lcs(s1, s2):\n"
                "    def weight(ch):\n"
                "        if ch.lower() in 'aeiou':\n"
                "            return 2\n"
                "        elif ch.isupper():\n"
                "            return 2\n"
                "        return 1\n\n"
                "    m, n = len(s1), len(s2)\n"
                "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
                "    for i in range(m):\n"
                "        for j in range(n):\n"
                "            if s1[i] == s2[j]:\n"
                "                dp[i+1][j+1] = dp[i][j] + weight(s1[i])\n"
                "            else:\n"
                "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n"
                "    return dp[m][n]\n"
                "```\n\n"
                "✅ Suitable for NLP, pattern matching with emphasis on important symbols.\n"
            ).format(s1, s2)

#    # 1️⃣7️⃣ 
    elif "special characters" in q or "digits" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return (
                "🔎 Exploring **LCS behavior with digits and special characters** in '{}' and '{}':\n\n"
                "🧪 By default:\n"
                "- LCS treats all characters (letters, digits, symbols) **equally**.\n"
                "- So 'A' and '1' are matched only if **identical**.\n\n"
                "🔧 Potential Issues:\n"
                "1. You may want to **ignore punctuation** or non-alphabetic symbols.\n"
                "2. Or treat digits as a special group (e.g., '5' and '7' being similar).\n\n"
                "💡 Suggested Improvements:\n"
                "- Preprocess strings to **clean unwanted characters**.\n"
                "- Use custom similarity rules (e.g., regex-based match).\n\n"
                "⚙️ Preprocessing Example:\n"
                "```python\n"
                "import re\n"
                "def clean_string(s):\n"
                "    return re.sub(r'[^a-zA-Z0-9]', '', s)\n"
                "\n"
                "def lcs_cleaned(s1, s2):\n"
                "    s1 = clean_string(s1)\n"
                "    s2 = clean_string(s2)\n"
                "    m, n = len(s1), len(s2)\n"
                "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
                "    for i in range(m):\n"
                "        for j in range(n):\n"
                "            if s1[i] == s2[j]:\n"
                "                dp[i+1][j+1] = dp[i][j] + 1\n"
                "            else:\n"
                "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n"
                "    return dp[m][n]\n"
                "```\n\n"
                "✅ Customization allows you to:\n"
                "- Ignore emojis, symbols, case, or even whitespace.\n"
                "- Focus only on alphabetic/digit patterns.\n"
            ).format(s1, s2)
 
#   # 1️⃣8️⃣
    elif "return the positions" in q and "LCS characters" in q:

        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()

            # Function to compute LCS indices
            def lcs_indices(s1, s2):
                m, n = len(s1), len(s2)
                dp = [[0]*(n+1) for _ in range(m+1)]
                for i in range(m):
                    for j in range(n):
                        if s1[i] == s2[j]:
                            dp[i+1][j+1] = dp[i][j] + 1
                        else:
                            dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

                i, j = m, n
                indices_s1, indices_s2 = [], []
                while i > 0 and j > 0:
                    if s1[i-1] == s2[j-1]:
                        indices_s1.append(i-1)
                        indices_s2.append(j-1)
                        i -= 1
                        j -= 1
                    elif dp[i-1][j] >= dp[i][j-1]:
                        i -= 1
                    else:
                        j -= 1

                return list(reversed(indices_s1)), list(reversed(indices_s2))

            # Get indices for the matched strings
            idx_s1, idx_s2 = lcs_indices(s1, s2)

            # Format the answer with explanation, code, and computed indices
            answer = (
                f"📌 Extracting **indices of LCS characters** from '{s1}' and '{s2}':\n\n"
                "🎯 Goal:\n"
                "- Identify exact index positions where LCS characters appear in both strings.\n\n"
                "🔍 Example:\n"
                "- s1 = 'ABCBDAB', s2 = 'BDCABA'\n"
                "- LCS = 'BCBA'\n"
                "- Indices in s1 = [1, 2, 4, 6], in s2 = [0, 1, 3, 5]\n\n"
                "⚙️ Code:\n"
                "```python\n"
                "def lcs_indices(s1, s2):\n"
                "    m, n = len(s1), len(s2)\n"
                "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
                "    for i in range(m):\n"
                "        for j in range(n):\n"
                "            if s1[i] == s2[j]:\n"
                "                dp[i+1][j+1] = dp[i][j] + 1\n"
                "            else:\n"
                "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n\n"
                "    i, j = m, n\n"
                "    indices_s1, indices_s2 = [], []\n"
                "    while i > 0 and j > 0:\n"
                "        if s1[i-1] == s2[j-1]:\n"
                "            indices_s1.append(i-1)\n"
                "            indices_s2.append(j-1)\n"
                "            i -= 1\n"
                "            j -= 1\n"
                "        elif dp[i-1][j] >= dp[i][j-1]:\n"
                "            i -= 1\n"
                "        else:\n"
                "            j -= 1\n"
                "    return list(reversed(indices_s1)), list(reversed(indices_s2))\n"
                "```\n\n"
                "✅ Use this when you want to **highlight or mark** LCS characters in original strings.\n\n"
                f"🧮 Computed indices:\n- Indices in s1: {idx_s1}\n- Indices in s2: {idx_s2}"
            )
            return answer


#    # 1️⃣9️⃣

    elif "LCS is exactly half the length" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return (
                "🎯 Task: Create strings '{}' and '{}' such that their **LCS is half the length of the shorter string**.\n\n"
                "📌 Strategy:\n"
                "1. Choose a common subsequence of length `k`.\n"
                "2. Add different characters in between to make strings longer.\n\n"
                "🧪 Example:\n"
                "- Let common = 'ACE'\n"
                "- s1 = 'A1C2E3', s2 = '0A9C8E7'\n"
                "- LCS = 'ACE' of length 3\n"
                "- Shorter string length = 6, and 3 = 6 // 2 ✅\n\n"
                "🔢 Code to verify:\n"
                "```python\n"
                "def lcs_length(s1, s2):\n"
                "    m, n = len(s1), len(s2)\n"
                "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
                "    for i in range(m):\n"
                "        for j in range(n):\n"
                "            if s1[i] == s2[j]:\n"
                "                dp[i+1][j+1] = dp[i][j] + 1\n"
                "            else:\n"
                "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n"
                "    return dp[m][n]\n"
                "\n"
                "# Example validation\n"
                "s1 = 'A1C2E3'\n"
                "s2 = '0A9C8E7'\n"
                "print(lcs_length(s1, s2))  # Should print 3\n"
                "```\n\n"
                "✅ You can generate such examples programmatically or use this template to test your LCS."
            ).format(s1, s2)

#   # 2️⃣0️⃣
    elif "case-insensitive" in q and "ignore spaces" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if match:
            s1, s2 = match.groups()
            return (
                "🔤 Modified LCS between '{}' and '{}' (case-insensitive, spaces ignored):\n\n"
                "🛠️ Preprocessing:\n"
                "- Convert both strings to lowercase.\n"
                "- Remove all whitespace characters.\n\n"
                "⚙️ Code:\n"
                "```python\n"
                "def lcs_case_insensitive_ignore_spaces(a, b):\n"
                "    a = ''.join(a.lower().split())\n"
                "    b = ''.join(b.lower().split())\n"
                "    m, n = len(a), len(b)\n"
                "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
                "    for i in range(m):\n"
                "        for j in range(n):\n"
                "            if a[i] == b[j]:\n"
                "                dp[i+1][j+1] = dp[i][j] + 1\n"
                "            else:\n"
                "                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])\n"
                "    return dp[m][n]\n"
                "```\n\n"
                "✅ This LCS version is **whitespace-agnostic** and **case-tolerant**, ideal for comparing user inputs or free-text documents."
            ).format(s1, s2)

    return "Answer generation for this Level 3 question is not implemented yet."

import re

def answer_app_lvl1(question):
    q = question.lower()

    # Helper: LCS length and similarity calculator
    def compute_lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        steps = []
        for i in range(m):
            for j in range(n):
                if a[i] == b[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                    steps.append(f"dp[{i+1}][{j+1}] = dp[{i}][{j}] + 1 = {dp[i+1][j+1]} (match '{a[i]}')")
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
                    steps.append(f"dp[{i+1}][{j+1}] = max(dp[{i}][{j+1}]={dp[i][j+1]}, dp[{i+1}][{j}]={dp[i+1][j]}) = {dp[i+1][j+1]}")
        lcs_len = dp[m][n]
        sim = (lcs_len / max(m, n)) * 100 if max(m, n) > 0 else 0
        return lcs_len, sim, steps

    # 1. Spell checking: typed_word vs correct_word
    if "spell checking" in q or ("typed input" in q and "correct word" in q):
        match = re.search(r"typed input ['\"](.*?)['\"].*correct word ['\"](.*?)['\"]", question)
        if not match:
            match = re.search(r"typed word ['\"](.*?)['\"].*correct word ['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract 'typed_word' and 'correct_word'."
        typed, correct = match.group(1), match.group(2)
        lcs_len, sim, steps = compute_lcs(typed, correct)
        return (
            f"📝 Comparing typed input '{typed}' with correct word '{correct}':\n"
            f"🔹 LCS length = {lcs_len}\n"
            f"🔹 Similarity = {sim:.1f}%\n\n"
            "📘 LCS helps detect spelling errors by comparing common subsequences.\n"
            "If the similarity is high, we can suggest the correct word.\n"
            + "\n".join(steps)
        )

    # Match any two inputs in quotes (generic fallback)
    match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
    if match:
        a, b = match.group(1), match.group(2)
        lcs_len, sim, steps = compute_lcs(a, b)

        response = f"📊 Comparing '{a}' with '{b}':\n"
        response += f"🔹 LCS length = {lcs_len}\n"
        response += f"🔹 Similarity = {sim:.1f}%\n\n"

        # Use-case-based explanation
        if "query" in q:
            response += "🔍 Application: Suggest relevant search completions by comparing user input with stored queries.\n"
        elif "name" in q or "contact" in q:
            response += "📇 Application: Detect spelling similarity between names to identify duplicates or matches.\n"
        elif "phrase" in q:
            response += "🧩 Application: Measure commonality between phrases for overlap detection.\n"
        elif "search" in q:
            response += "🔎 Application: Improve search accuracy by comparing keyword similarity.\n"
        elif "password" in q:
            response += "🔐 Application: Detect similarity between entered and stored passwords for security suggestions.\n"
        elif "command" in q:
            response += "💻 Application: Recommend closest valid command when a user mistypes.\n"
        elif "username" in q:
            response += "👤 Application: Suggest similar usernames if a chosen one is unavailable.\n"
        elif "input" in q and "target" in q:
            response += "📥 Application: Compare typed input with expected target to measure match confidence.\n"
        else:
            response += "ℹ️ General string similarity check using LCS.\n"

        response += "\n🧮 Step-by-step LCS computation:\n" + "\n".join(steps)
        return response

    return "ℹ️ No specific application match found in Level 1."
import re

def answer_app_lvl2(question):
    q = question.lower()

    def compute_lcs_steps(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        steps = []
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match '{s1[i-1]}')")
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    steps.append(f"dp[{i}][{j}] = max(dp[{i-1}][{j}]={dp[i-1][j]}, dp[{i}][{j-1}]={dp[i][j-1]}) = {dp[i][j]}")
        return dp[m][n], steps

    # Try to extract any two quoted strings from the question
    match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
    if not match:
        return "❌ Could not extract comparison strings."

    s1, s2 = match.group(1), match.group(2)
    lcs_len, steps = compute_lcs_steps(s1, s2)
    similarity = (lcs_len / max(len(s1), len(s2))) * 100 if max(len(s1), len(s2)) > 0 else 0

    header = f"📥 Comparing:\n- Input 1: '{s1}'\n- Input 2: '{s2}'\n"
    summary = f"\n📊 LCS length = {lcs_len}\n🔍 Similarity = {similarity:.1f}%\n"

    # Application-specific explanation
    explanation = "\n📘 Application:\n"
    if "student answer" in q:
        explanation += "- Used to detect similar answer patterns in exams for academic integrity checks.\n"
    elif "plagiarism" in q or "code segments" in q:
        explanation += "- Used in plagiarism detection tools to find reused code or text.\n"
    elif "dna" in q:
        explanation += "- Used in bioinformatics to find matching gene sequences.\n"
    elif "chat" in q:
        explanation += "- Used to match user messages to predefined chatbot responses.\n"
    elif "melody" in q or "music" in q:
        explanation += "- Used to compare patterns in melody or rhythm for similarity.\n"
    elif "document" in q or "paragraph" in q:
        explanation += "- Used to find overlapping sections in legal or academic documents.\n"
    elif "auto-complete" in q or "user_input" in q:
        explanation += "- Used to recommend completions based on dictionary match strength.\n"
    elif "diff utility" in q or "highlight differences" in q:
        explanation += "- Used to visually compare file versions and revisions.\n"
    elif "paper" in q or "citation" in q:
        explanation += "- Used to find overlapping references or quotes in research papers.\n"
    elif "project" in q or "reuse of code" in q:
        explanation += "- Used to detect shared code blocks across repositories.\n"
    elif "resume" in q or "skills" in q:
        explanation += "- Used to match a candidate's resume with job descriptions.\n"
    elif "essay" in q or "source" in q:
        explanation += "- Used to trace phrases back to their original sources.\n"
    elif "chatbot" in q or "faq" in q:
        explanation += "- Used to match queries to the most relevant FAQ entry.\n"
    elif "ocr" in q or "scan" in q:
        explanation += "- Used to compare scanned text with a clean reference to spot errors.\n"
    elif "email subject" in q:
        explanation += "- Used to cluster similar email campaigns.\n"
    elif "contract" in q:
        explanation += "- Used to detect reused clauses across contract versions.\n"
    elif "search" in q and "product" in q:
        explanation += "- Used to match search queries with product titles for relevant suggestions.\n"
    else:
        explanation += "- General string similarity scoring using LCS.\n"

    trace = "\n🧮 Step-by-step DP Table Construction:\n" + "\n".join(steps)
    return header + summary + explanation + trace


import re

def answer_app_lvl3(question: str) -> str:
    q = question.lower()

    def compute_lcs_steps(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        steps = []
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match '{s1[i-1]}')")
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    steps.append(f"dp[{i}][{j}] = max(dp[{i-1}][{j}]={dp[i-1][j]}, dp[{i}][{j-1}]={dp[i][j-1]}) = {dp[i][j]}")
        return dp[m][n], steps

    # Q1 — Smart merging of code_version1 and code_version2
    if "smart merging" in q or ("code_version1" in q and "code_version2" in q):
        match = re.search(r"'(.*?)'.*?'(.*?)'", question)
        if not match:
            return "❌ Could not extract 'code_version1' and 'code_version2'."
        version1, version2 = match.group(1), match.group(2)

        # Treat each line as a character in LCS
        lines1 = version1.strip().split("\\n")
        lines2 = version2.strip().split("\\n")

        lcs_len, trace_steps = compute_lcs_steps(lines1, lines2)
        similarity = (lcs_len / max(len(lines1), len(lines2))) * 100 if max(len(lines1), len(lines2)) > 0 else 0

        output = f"🧠 LCS-Based Smart Merge Strategy\n"
        output += f"\n📥 Comparing code_version1 and code_version2 (line-by-line):"
        output += f"\n- Version 1: {len(lines1)} lines\n- Version 2: {len(lines2)} lines\n"
        output += f"\n📊 LCS length (common lines) = {lcs_len}"
        output += f"\n🔍 Similarity = {similarity:.1f}%\n"

        output += "\n📘 Application:\n"
        output += "- By computing the LCS of lines between two code versions, we identify the stable (unchanged) lines.\n"
        output += "- During merge:\n"
        output += "  • LCS lines are kept as-is.\n"
        output += "  • Differences between LCS chunks are resolved by comparing change blocks.\n"
        output += "- This reduces false conflicts and improves 2-way merging without requiring a base version.\n"

        output += "\n🧮 Step-by-step DP Table Construction (Line-Based):\n"
        output += "\n".join(trace_steps)
        return output

    # Q2 — Speech recognition: spoken_text vs expected_text
    elif "speech recognition" in q and "spoken" in q and "expected" in q:
        match = re.search(r"'(.*?)'.*?'(.*?)'", question)
        if not match:
            return "❌ Could not extract 'spoken_text' and 'expected_text'."
        spoken, expected = match.group(1), match.group(2)

        # Compute LCS
        def compute_lcs_steps(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            steps = []
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match '{s1[i-1]}')")
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                        steps.append(f"dp[{i}][{j}] = max(dp[{i-1}][{j}]={dp[i-1][j]}, dp[{i}][{j-1}]={dp[i][j-1]}) = {dp[i][j]}")
            return dp[m][n], steps

        lcs_len, trace_steps = compute_lcs_steps(spoken, expected)
        similarity = (lcs_len / max(len(spoken), len(expected))) * 100 if max(len(spoken), len(expected)) > 0 else 0

        output = f"🎙️ LCS-Based Evaluation for Speech Recognition\n"
        output += f"\n🗣️ Spoken input: '{spoken}'"
        output += f"\n📖 Expected text: '{expected}'\n"
        output += f"\n📊 LCS length = {lcs_len}"
        output += f"\n🔍 Similarity = {similarity:.1f}%\n"

        output += "\n📘 Application:\n"
        output += "- In speech recognition, LCS helps compare recognized speech with ground truth.\n"
        output += "- It measures how much of the original phrase was correctly captured in the right order.\n"
        output += "- Can be used to:\n"
        output += "  • Evaluate ASR (Automatic Speech Recognition) system accuracy\n"
        output += "  • Provide feedback on speaking exercises (e.g., language learning)\n"

        output += "\n🧮 Step-by-step DP Table Construction (Character-Based):\n"
        output += "\n".join(trace_steps)
        return output
    # Q3 — LCS-based product recommendation via feature string comparison
    elif "recommendation engine" in q and "feature string" in q:
        match = re.search(r"'(.*?)'.*?'(.*?)'", question)
        if not match:
            return "❌ Could not extract feature strings."
        feature1, feature2 = match.group(1), match.group(2)

        # Compute LCS between features
        def compute_lcs_steps(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            steps = []
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match '{s1[i-1]}')")
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                        steps.append(f"dp[{i}][{j}] = max(dp[{i-1}][{j}]={dp[i-1][j]}, dp[{i}][{j-1}]={dp[i][j-1]}) = {dp[i][j]}")
            return dp[m][n], steps

        lcs_len, trace_steps = compute_lcs_steps(feature1, feature2)
        similarity = (lcs_len / max(len(feature1), len(feature2))) * 100 if max(len(feature1), len(feature2)) > 0 else 0

        output = f"🛒 LCS-Based Product Recommendation via Feature Comparison\n"
        output += f"\n🧾 Feature Set A: '{feature1}'"
        output += f"\n🧾 Feature Set B: '{feature2}'\n"
        output += f"\n📊 LCS length = {lcs_len}"
        output += f"\n🔍 Similarity = {similarity:.1f}%\n"

        output += "\n📘 Application:\n"
        output += "- LCS is used to compare product features by sequence alignment.\n"
        output += "- This helps recommend products that have similar functional descriptions or tags.\n"
        output += "- Used in:\n"
        output += "  • Product recommendation engines\n"
        output += "  • Catalog de-duplication\n"
        output += "  • User-preference matching based on selected features\n"

        output += "\n🧮 Step-by-step DP Table Construction (Character-Based):\n"
        output += "\n".join(trace_steps)
        return output
    # Q4 — Legal document clause comparison using LCS
    elif "legal document" in q or ("doc_version1" in q and "doc_version2" in q):
        match = re.search(r"'(.*?)'.*?'(.*?)'", question)
        if not match:
            return "❌ Could not extract doc_version1 and doc_version2."
        doc1, doc2 = match.group(1), match.group(2)

        # Treat each clause/line as a token — split by '\n' or '. '
        clauses1 = [line.strip() for line in re.split(r'\n|\.\s+', doc1.strip()) if line.strip()]
        clauses2 = [line.strip() for line in re.split(r'\n|\.\s+', doc2.strip()) if line.strip()]

        def compute_lcs_steps(seq1, seq2):
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            steps = []
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i - 1] == seq2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match clause '{seq1[i-1]}')")
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                        steps.append(f"dp[{i}][{j}] = max(dp[{i-1}][{j}]={dp[i-1][j]}, dp[{i}][{j-1}]={dp[i][j-1]}) = {dp[i][j]}")
            return dp[m][n], steps

        lcs_len, trace_steps = compute_lcs_steps(clauses1, clauses2)
        similarity = (lcs_len / max(len(clauses1), len(clauses2))) * 100 if max(len(clauses1), len(clauses2)) > 0 else 0

        output = f"📄 Legal Document Clause Comparison using LCS\n"
        output += f"\n📘 Input Documents:\n- Document 1: {len(clauses1)} clauses\n- Document 2: {len(clauses2)} clauses\n"
        output += f"\n📊 LCS length (matching clauses) = {lcs_len}"
        output += f"\n🔍 Similarity = {similarity:.1f}%\n"

        output += "\n📘 Application:\n"
        output += "- LCS helps preserve the order of clauses while identifying overlap.\n"
        output += "- Especially useful in:\n"
        output += "  • Redlining contract versions\n"
        output += "  • Auditing changes in policy documents\n"
        output += "  • Highlighting reused boilerplate or copied terms\n"

        output += "\n🧮 Step-by-step DP Table Construction (Clause-Based):\n"
        output += "\n".join(trace_steps)
        return output
    elif "diff viewer" in q or ("code_version1" in q and "code_version2" in q):
        match = re.search(r"'(.*?)'.*?'(.*?)'", question)
        if not match:
            return "❌ Could not extract code versions."
        code1, code2 = match.group(1), match.group(2)
        lines1 = code1.strip().split("\\n")
        lines2 = code2.strip().split("\\n")

        def compute_lcs_steps(seq1, seq2):
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            steps = []
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i - 1] == seq2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match line '{seq1[i-1]}')")
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                        steps.append(f"dp[{i}][{j}] = max({dp[i-1][j]}, {dp[i][j-1]}) = {dp[i][j]}")
            return dp[m][n], steps

        lcs_len, steps = compute_lcs_steps(lines1, lines2)
        similarity = (lcs_len / max(len(lines1), len(lines2))) * 100 if max(len(lines1), len(lines2)) > 0 else 0

        return (
            f"🧾 LCS Diff Viewer for Version Control:\n"
            f"📄 Code 1: {len(lines1)} lines\n📄 Code 2: {len(lines2)} lines\n"
            f"📊 Common Lines (LCS) = {lcs_len}\n🔍 Similarity = {similarity:.1f}%\n\n"
            f"📘 Application:\n"
            f"- Identifies common and differing lines using LCS.\n"
            f"- Perfect for visual diff tools in Git or SVN.\n"
            f"- Common lines = LCS. Lines outside LCS = additions/deletions.\n\n"
            f"🧮 Step-by-step DP Table Construction (Line-Based):\n" +
            "\n".join(steps)
        )
    elif "dna comparison" in q or "approximate match" in q:
        match = re.search(r"'(.*?)'.*?'(.*?)'", question)
        if not match:
            return "❌ Could not extract DNA sequences."
        s1, s2 = match.group(1), match.group(2)

        def compute_approx_lcs(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0]*(n+1) for _ in range(m+1)]
            steps = []
            for i in range(1, m+1):
                for j in range(1, n+1):
                    # Allow approximate match if same or purine/pyrimidine match
                    if s1[i-1] == s2[j-1] or (s1[i-1] in 'AG' and s2[j-1] in 'AG') or (s1[i-1] in 'CT' and s2[j-1] in 'CT'):
                        dp[i][j] = dp[i-1][j-1] + 1
                        steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match '{s1[i-1]}' ~ '{s2[j-1]}')")
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                        steps.append(f"dp[{i}][{j}] = max({dp[i-1][j]}, {dp[i][j-1]}) = {dp[i][j]}")
            return dp[m][n], steps

        lcs_len, steps = compute_approx_lcs(s1, s2)
        similarity = (lcs_len / max(len(s1), len(s2))) * 100 if max(len(s1), len(s2)) > 0 else 0

        return (
            f"🧬 Approximate LCS for DNA Sequences:\n"
            f"DNA1: '{s1}'\nDNA2: '{s2}'\n"
            f"📊 Similarity = {similarity:.1f}% (allowing purine/pyrimidine mismatch)\n\n"
            f"📘 Application:\n"
            f"- Finds overlapping segments even with minor biological variance.\n"
            f"- Tolerance can be tuned to domain knowledge.\n\n"
            f"🧮 Step-by-step DP Table Construction (Nucleotide-Based):\n" +
            "\n".join(steps)
        )
    elif "chatbot" in q and "faqs" in q:
        match = re.search(r"input to (.*?) and ranks", question)
        if not match:
            return "❌ No input detected."
        user_input = "user asked: how to reset my password?"

        faqs = [
            "how can I reset my password",
            "what is the refund policy",
            "how to change my email address"
        ]

        def lcs_len(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        scores = []
        for faq in faqs:
            lcs_score = lcs_len(user_input, faq)
            sim = (lcs_score / max(len(user_input), len(faq))) * 100
            scores.append((faq, lcs_score, sim))

        scores.sort(key=lambda x: -x[2])

        output = "🤖 Chatbot FAQ Matching using LCS\n"
        output += f"User Input: '{user_input}'\n\n"
        output += "📊 Ranked Matches:\n"
        for faq, score, sim in scores:
            output += f"- FAQ: '{faq}' → LCS={score}, Similarity={sim:.1f}%\n"

        output += "\n📘 Application:\n"
        output += "- Compare user input with all FAQ entries.\n"
        output += "- Rank by LCS-based similarity.\n"
        output += "- Return top-ranked answer as chatbot reply.\n"

        return output
    elif "text summarization" in q and "multiple reports" in q:
        match = re.findall(r"'(.*?)'", question)
        if len(match) < 3:
            return "❌ Could not extract three reports."

        report1, report2, report3 = match[0], match[1], match[2]

        def lcs_pair(a, b):
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    if a[i] == b[j]:
                        dp[i + 1][j + 1] = dp[i][j] + 1
                    else:
                        dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

            # Backtrack LCS
            i, j = m, n
            lcs = []
            while i > 0 and j > 0:
                if a[i - 1] == b[j - 1]:
                    lcs.append(a[i - 1])
                    i -= 1
                    j -= 1
                elif dp[i - 1][j] > dp[i][j - 1]:
                    i -= 1
                else:
                    j -= 1
            return ''.join(reversed(lcs))

        r1_r2 = lcs_pair(report1, report2)
        common_lcs = lcs_pair(r1_r2, report3)

        return (
            f"🧾 LCS-Based Narrative Extraction from 3 Reports:\n"
            f"- Report 1 length: {len(report1)}\n"
            f"- Report 2 length: {len(report2)}\n"
            f"- Report 3 length: {len(report3)}\n\n"
            f"📊 Longest Common Narrative Thread (LCS across all 3):\n"
            f"\"{common_lcs}\"\n\n"
            f"📘 Application:\n"
            f"- Extracts shared storyline between multiple documents.\n"
            f"- Useful in summarizing incident reports, meeting notes, or duplicate writeups.\n"
        )
    elif "historical linguistics" in q:
        match = re.search(r"'(.*?)'.*?'(.*?)'", question)
        if not match:
            return "❌ Could not extract two language forms."
        word1, word2 = match.group(1), match.group(2)

        def compute_lcs_steps(a, b):
            m, n = len(a), len(b)
            dp = [[0]*(n+1) for _ in range(m+1)]
            steps = []
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                        steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match '{a[i-1]}')")
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                        steps.append(f"dp[{i}][{j}] = max({dp[i-1][j]}, {dp[i][j-1]}) = {dp[i][j]}")
            return dp[m][n], steps

        lcs_len, steps = compute_lcs_steps(word1, word2)
        similarity = (lcs_len / max(len(word1), len(word2))) * 100

        return (
            f"🌍 LCS for Language Evolution:\n"
            f"Word 1: '{word1}'\nWord 2: '{word2}'\n"
            f"📊 Shared root LCS = {lcs_len}, Similarity = {similarity:.1f}%\n\n"
            f"📘 Application:\n"
            f"- Identifies common subsequence across historical word forms.\n"
            f"- Useful for tracking how words diverge over time.\n\n"
            f"🧮 Step-by-step LCS DP Construction:\n" +
            "\n".join(steps)
        )
    elif "navigation path" in q:
        match = re.search(r"'(.*?)'.*?'(.*?)'", question)
        if not match:
            return "❌ Could not extract navigation paths."
        path1, path2 = match.group(1), match.group(2)
        nav1 = path1.strip().split(">")
        nav2 = path2.strip().split(">")

        def compute_lcs_steps(a, b):
            m, n = len(a), len(b)
            dp = [[0]*(n+1) for _ in range(m+1)]
            steps = []
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                        steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match '{a[i-1]}')")
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                        steps.append(f"dp[{i}][{j}] = max({dp[i-1][j]}, {dp[i][j-1]}) = {dp[i][j]}")
            return dp[m][n], steps

        lcs_len, steps = compute_lcs_steps(nav1, nav2)
        similarity = (lcs_len / max(len(nav1), len(nav2))) * 100

        return (
            f"📈 LCS-Based User Journey Analysis\n"
            f"Path 1: {nav1}\nPath 2: {nav2}\n"
            f"📊 Common sequence length: {lcs_len}, Similarity: {similarity:.1f}%\n\n"
            f"📘 Application:\n"
            f"- Tracks recurring journey patterns across users.\n"
            f"- Improves personalization by understanding intent.\n\n"
            f"🧮 Step-by-step LCS DP Construction:\n" +
            "\n".join(steps)
        )
    elif "software documentation" in q or ("doca" in q and "docb" in q):
        match = re.search(r"'(.*?)'.*?'(.*?)'", question)
        if not match:
            return "❌ Could not extract docA and docB."
        doc1, doc2 = match.group(1), match.group(2)

        lines1 = [line.strip() for line in doc1.strip().split("\\n") if line.strip()]
        lines2 = [line.strip() for line in doc2.strip().split("\\n") if line.strip()]

        def compute_lcs_steps(seq1, seq2):
            m, n = len(seq1), len(seq2)
            dp = [[0]*(n+1) for _ in range(m+1)]
            steps = []
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                        steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match line '{seq1[i-1]}')")
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                        steps.append(f"dp[{i}][{j}] = max({dp[i-1][j]}, {dp[i][j-1]}) = {dp[i][j]}")
            return dp[m][n], steps

        lcs_len, steps = compute_lcs_steps(lines1, lines2)
        similarity = (lcs_len / max(len(lines1), len(lines2))) * 100 if max(len(lines1), len(lines2)) > 0 else 0

        return (
            f"📘 LCS-Based Documentation Block Comparison\n"
            f"Doc A: {len(lines1)} lines\nDoc B: {len(lines2)} lines\n"
            f"📊 Common lines (LCS): {lcs_len}, Similarity = {similarity:.1f}%\n\n"
            f"🧠 Application:\n"
            f"- Highlights reused or duplicated content in tech docs.\n"
            f"- Supports version tracking, reuse auditing, and knowledge base pruning.\n\n"
            f"🧮 Step-by-step DP Table Construction (Line-Based):\n" + "\n".join(steps)
        )
    elif "adaptive learning" in q and "student answer" in q:
        match = re.search(r"'(.*?)'.*?'(.*?)'", question)
        if not match:
            return "❌ Could not extract student_input and expected_answer."
        student, expected = match.group(1), match.group(2)

        def compute_lcs_steps(a, b):
            m, n = len(a), len(b)
            dp = [[0]*(n+1) for _ in range(m+1)]
            steps = []
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                        steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match '{a[i-1]}')")
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                        steps.append(f"dp[{i}][{j}] = max({dp[i-1][j]}, {dp[i][j-1]}) = {dp[i][j]}")
            return dp[m][n], steps

        lcs_len, steps = compute_lcs_steps(student, expected)
        similarity = (lcs_len / max(len(student), len(expected))) * 100

        return (
            f"🧑‍🎓 Adaptive Learning: LCS-Based Answer Matching\n"
            f"Student Answer: '{student}'\nModel Answer:   '{expected}'\n"
            f"📊 LCS length = {lcs_len}, Similarity = {similarity:.1f}%\n\n"
            f"📘 Application:\n"
            f"- Useful for automated grading with flexibility in word order or phrasing.\n"
            f"- Helps group similar learning patterns and misconceptions.\n\n"
            f"🧮 Step-by-step DP Table Construction:\n" + "\n".join(steps)
        )
    elif "real-time chat monitoring" in q or "repeated sensitive pattern" in q:
        messages = [
            "We need to ship the order under the table.",
            "Can we pay cash under the table tomorrow?",
            "Just do it quietly, no record needed."
        ]

        def lcs(a, b):
            m, n = len(a), len(b)
            dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(m):
                for j in range(n):
                    if a[i] == b[j]:
                        dp[i+1][j+1] = dp[i][j] + 1
                    else:
                        dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
            i, j = m, n
            result = []
            while i > 0 and j > 0:
                if a[i-1] == b[j-1]:
                    result.append(a[i-1])
                    i -= 1
                    j -= 1
                elif dp[i-1][j] > dp[i][j-1]:
                    i -= 1
                else:
                    j -= 1
            return ''.join(reversed(result))

        common12 = lcs(messages[0], messages[1])
        common_all = lcs(common12, messages[2])

        return (
            f"🔍 LCS-Based Chat Pattern Detection\n"
            f"Message 1: {messages[0]}\n"
            f"Message 2: {messages[1]}\n"
            f"Message 3: {messages[2]}\n\n"
            f"📊 Shared pattern: '{common_all}'\n"
            f"📘 Application:\n"
            f"- Detects suspicious or repeated intent across different chats.\n"
            f"- Helps flag collusion, fraud, or covert behavior across messages.\n"
        )
    elif "authors" in q or ("chapter1" in q and "chapter2" in q):
        match = re.search(r"'(.*?)'.*?'(.*?)'", question)
        if not match:
            return "❌ Could not extract chapter1 and chapter2."
        ch1, ch2 = match.group(1), match.group(2)
        lines1 = ch1.split("\\n")
        lines2 = ch2.split("\\n")

        def compute_lcs_steps(a, b):
            m, n = len(a), len(b)
            dp = [[0]*(n+1) for _ in range(m+1)]
            steps = []
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                        steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match line '{a[i-1]}')")
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                        steps.append(f"dp[{i}][{j}] = max({dp[i-1][j]}, {dp[i][j-1]}) = {dp[i][j]}")
            return dp[m][n], steps

        lcs_len, steps = compute_lcs_steps(lines1, lines2)
        similarity = (lcs_len / max(len(lines1), len(lines2))) * 100

        return (
            f"📝 Chapter Comparison Using LCS\n"
            f"📄 Chapter 1: {len(lines1)} lines\n📄 Chapter 2: {len(lines2)} lines\n"
            f"📊 Overlapping lines (LCS): {lcs_len}, Similarity = {similarity:.1f}%\n\n"
            f"📘 Application:\n"
            f"- Detects reused content across book chapters or articles.\n"
            f"- Helps authors avoid redundancy and revise for uniqueness.\n\n"
            f"🧮 Step-by-step DP Table Construction (Line-Based):\n" + "\n".join(steps)
        )
    elif "code tutoring" in q or ("student_code" in q and "reference_code" in q):
        match = re.search(r"'(.*?)'.*?'(.*?)'", question)
        if not match:
            return "❌ Could not extract student_code and reference_code."
        s_code, r_code = match.group(1), match.group(2)
        s_lines = s_code.strip().split("\\n")
        r_lines = r_code.strip().split("\\n")

        def compute_lcs_steps(a, b):
            m, n = len(a), len(b)
            dp = [[0]*(n+1) for _ in range(m+1)]
            steps = []
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                        steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match line '{a[i-1]}')")
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                        steps.append(f"dp[{i}][{j}] = max({dp[i-1][j]}, {dp[i][j-1]}) = {dp[i][j]}")
            return dp[m][n], steps

        lcs_len, steps = compute_lcs_steps(s_lines, r_lines)
        similarity = (lcs_len / max(len(s_lines), len(r_lines))) * 100

        return (
            f"👨‍💻 Adaptive Code Tutoring via LCS\n"
            f"Student Code: {len(s_lines)} lines\nReference Code: {len(r_lines)} lines\n"
            f"📊 Matching lines = {lcs_len}, Similarity = {similarity:.1f}%\n\n"
            f"📘 Application:\n"
            f"- Highlights which logic blocks are present/missing.\n"
            f"- Helps instructors and auto-graders understand student’s approach.\n\n"
            f"🧮 Step-by-step DP Table Construction (Line-Based):\n" + "\n".join(steps)
        )
    elif "document sync" in q or "shared sources" in q:
        doc1 = "Company policy requires all reports to be submitted weekly."
        doc2 = "All reports must be submitted weekly per company policy."

        def compute_lcs_steps(a, b):
            m, n = len(a), len(b)
            dp = [[0]*(n+1) for _ in range(m+1)]
            steps = []
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                        steps.append(f"dp[{i}][{j}] = dp[{i-1}][{j-1}] + 1 = {dp[i][j]} (match '{a[i-1]}')")
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                        steps.append(f"dp[{i}][{j}] = max({dp[i-1][j]}, {dp[i][j-1]}) = {dp[i][j]}")
            return dp[m][n], steps

        lcs_len, steps = compute_lcs_steps(doc1, doc2)
        similarity = (lcs_len / max(len(doc1), len(doc2))) * 100

        return (
            f"🔁 LCS-Based Document Sync System\n"
            f"📄 Doc A: '{doc1}'\n📄 Doc B: '{doc2}'\n"
            f"📊 Overlap LCS length = {lcs_len}, Similarity = {similarity:.1f}%\n\n"
            f"📘 Application:\n"
            f"- Identifies common sections between distributed documents.\n"
            f"- Supports auto-merging and highlighting sync differences.\n\n"
            f"🧮 Step-by-step DP Table Construction (Char-Based):\n" + "\n".join(steps)
        )

    return "❌ This Level 3 question is not yet implemented."


def answer_implementation_lvl1(question):
    import re
    q = question.lower()

    if "initialize" in q and "dp table" in q and "lcs" in q:
        match = re.search(r"strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        return (
            f"👨‍🏫 To compute the Longest Common Subsequence (LCS), we use dynamic programming.\n"
            f"Step 1 is to initialize a 2D table (called DP table) of size ({m+1}) x ({n+1}) for strings '{s1}' and '{s2}'.\n"
            f"This table stores the length of LCS for every prefix of the two strings.\n\n"
            f"🛠 Here's the Python code to do that:\n"
            "```python\n"
            f"dp = [[0 for _ in range({n+1})] for _ in range({m+1})]\n"
            "```\n"
            "🎯 All cells are initialized to 0 because the LCS of any string with an empty string is 0."
        )


    # 2. Recursive LCS function
    elif "recursive" in q and ("length of lcs" in q or "returns the length" in q):
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 To compute the LCS of '{s1}' and '{s2}' using recursion, we explore two main ideas:\n"
            f"1. If either string is empty, LCS is 0.\n"
            f"2. If the last characters match, include that character and recurse on remaining strings.\n"
            f"3. If they don't match, compute LCS by skipping one character from each string (two options) and take the maximum.\n\n"
            f"🧠 Recursive function code:\n"
            "```python\n"
            "def lcs_recursive(s1, s2, m, n):\n"
            "    if m == 0 or n == 0:\n"
            "        return 0\n"
            "    if s1[m-1] == s2[n-1]:\n"
            "        return 1 + lcs_recursive(s1, s2, m-1, n-1)\n"
            "    return max(lcs_recursive(s1, s2, m-1, n), lcs_recursive(s1, s2, m, n-1))\n"
            "```\n"
            "⚠️ Note: This is not optimized yet — it recalculates many overlapping subproblems."
        )

    # 3. Base case for recursive function
    elif "base case" in q and "recursive" in q:
        return (
            "👨‍🏫 In any recursive function, a base case defines when to stop recursion.\n"
            "For LCS, if either of the strings becomes empty (i.e., length 0), the LCS is clearly 0.\n\n"
            "✅ Here's how we write the base case in code:\n"
            "```python\n"
            "if m == 0 or n == 0:\n"
            "    return 0\n"
            "```\n"
            "🎯 This ensures recursion stops when one of the strings is fully processed."
        )


    # 4. Fill only first row and first column of DP table
    elif "fill only the first row" in q or "first column" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        return (
            f"👨‍🏫 When initializing the LCS DP table for strings '{s1}' and '{s2}',\n"
            f"we set the entire first row and first column to 0. This is because any string compared with an empty string has LCS = 0.\n\n"
            f"🧱 Table creation and initialization:\n"
            "```python\n"
            f"dp = [[0 for _ in range({n+1})] for _ in range({m+1})]\n"
            "# First row and first column are already 0 from initialization\n"
            "```\n"
            "📌 No need to manually set them again unless using a different structure."
        )

    # 5. Compare characters and print matching pairs
    elif "compare" in q and "matching pairs" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        matches = []
        for i, ch1 in enumerate(s1):
            for j, ch2 in enumerate(s2):
                if ch1 == ch2:
                    matches.append(f"✅ Match: '{ch1}' at s1[{i}] and s2[{j}]")
        result = "\n".join(matches)
        return (
            f"👨‍🏫 To find character matches between '{s1}' and '{s2}',\n"
            f"we use two nested loops — one for each string — and compare each character.\n\n"
            "💡 Here's how it works:\n"
            "```python\n"
            "for i in range(len(s1)):\n"
            "    for j in range(len(s2)):\n"
            "        if s1[i] == s2[j]:\n"
            "            print(f\"Match at s1[{i}] and s2[{j}]: {s1[i]}\")\n"
            "```\n"
            f"{result if result else '⚠️ No matching characters found.'}"
        )


    # 6. Print partially computed LCS DP table
    elif "print the dp table" in q and "partially" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, min(m + 1, 3)):
            for j in range(1, min(n + 1, 3)):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        table = "\n".join(" ".join(map(str, row[:3])) for row in dp[:3])

        return (
            f"👨‍🏫 We begin filling the DP table based on LCS logic. Here's a partial computation for strings '{s1}' and '{s2}'.\n"
            f"🧠 This example only fills the first few cells to demonstrate the pattern.\n\n"
            "💡 Sample partially filled DP table (first 3 rows and 3 columns):\n"
            f"{table}\n\n"
            "📝 Note: dp[i][j] stores the length of the LCS between s1[0...i-1] and s2[0...j-1]."
        )

    # 7. Create 2D matrix of size (LEN1+1) x (LEN2+1)
    elif "create a 2d matrix" in q and "initialized with zeros" in q:
        match = re.search(r"size\s*\(?(\d+)\+1\)?\s*x\s*\(?(\d+)\+1\)?", q)
        if not match:
            return "❌ Could not extract matrix size."
        len1, len2 = int(match.group(1)), int(match.group(2))
        return (
            f"👨‍🏫 A 2D matrix of size ({len1+1} x {len2+1}) is created to store LCS values.\n"
            f"Each entry dp[i][j] represents the LCS of the first i characters of one string and first j characters of the other.\n\n"
            "🛠 Here's how you initialize it with 0s:\n"
            "```python\n"
            f"dp = [[0 for _ in range({len2+1})] for _ in range({len1+1})]\n"
            "```\n"
            "🎯 This is the base setup before applying LCS logic."
        )


    # 8. Print each cell value of a small DP table
    elif "print each cell value" in q and "dp table" in q:
        return (
            "👨‍🏫 To display the contents of a DP table (2D list), we use nested loops.\n"
            "Each row is printed line by line, and each cell is printed with a space separator.\n\n"
            "💡 Example code:\n"
            "```python\n"
            "for row in dp:\n"
            "    for cell in row:\n"
            "        print(cell, end=' ')\n"
            "    print()\n"
            "```\n"
            "📝 This helps visually verify how the LCS table is filled."
        )


    # 9. Base case when one string is empty
    elif "recursive" in q and "base case" in q and ("one of" in q or "empty" in q):
        return (
            "👨‍🏫 In LCS recursion, when either string becomes empty, there's no common subsequence left.\n"
            "Hence, the result is 0 — this is the base case of the recursion.\n\n"
            "✅ Python code:\n"
            "```python\n"
            "if m == 0 or n == 0:\n"
            "    return 0\n"
            "```\n"
            "🎯 This prevents unnecessary further recursion and acts as the stopping condition."
        )


    # 10. Loop through characters of STR1
    elif "loop" in q and "iterate through characters of" in q:
        match = re.search(r"characters of ['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract string."
        s = match.group(1)
        return (
            f"👨‍🏫 To prepare for comparing characters in LCS, we often need to loop through each character of string '{s}'.\n"
            "💡 Here's how you do it:\n"
            "```python\n"
            f"for ch in '{s}':\n"
            "    print(ch)\n"
            "```\n"
            "📌 This helps in debugging or manually comparing characters between two strings."
        )

     # 11. Print function for LCS table
    elif "print function" in q and "lcs table" in q:
        return (
                "👨‍🏫 To understand how LCS table is filled, it's useful to have a function that prints it in grid format.\n"
                "This helps in visualizing how values evolve row by row.\n\n"
                "💡 Here's a simple function to print the DP table:\n"
                "```python\n"
                "def print_dp_table(dp):\n"
                "    for row in dp:\n"
                "        print(' '.join(map(str, row)))\n"
                "```\n"
                "📝 Each row is printed on a new line with space-separated values."
            )
 

    # 12. Function that returns max of two values
    elif "max between two values" in q:
        return (
            "👨‍🏫 In the LCS recurrence relation, we often choose the maximum between two subproblem results.\n"
            "This function helps keep code clean and readable.\n\n"
            "💡 Here's a basic function to return the max of two values:\n"
            "```python\n"
            "def max_of_two(a, b):\n"
            "    return a if a > b else b\n"
            "```\n"
            "🎯 Alternatively, Python has a built-in max():\n"
            "```python\n"
            "return max(a, b)\n"
            "```"
        )


    # 13. If-else block for matching characters
    elif "if-else" in q and "position i and j" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 When comparing characters at position i and j of '{s1}' and '{s2}', we apply the LCS recurrence rule:\n"
            "1. If s1[i] == s2[j], then dp[i+1][j+1] = dp[i][j] + 1\n"
            "2. Else, take max of left and top values.\n\n"
            "💡 Here's the if-else block:\n"
            "```python\n"
            "if s1[i] == s2[j]:\n"
            "    dp[i+1][j+1] = dp[i][j] + 1\n"
            "else:\n"
            "    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n"
            "```\n"
            "🎯 This is the heart of the iterative LCS logic."
        )

    # 14. Memoization table using -1
    elif "memoization" in q and "initialized with -1" in q:
        match = re.search(r"lengths (\d+) and (\d+)", q)
        if not match:
            return "❌ Could not extract lengths."
        len1, len2 = int(match.group(1)), int(match.group(2))
        return (
            f"👨‍🏫 In top-down LCS with memoization, we use a 2D list (dp) to store intermediate results.\n"
            f"Each cell dp[i][j] is initialized with -1 to mark it uncomputed.\n\n"
            f"💡 Initialize memoization table of size ({len1+1} x {len2+1}):\n"
            "```python\n"
            f"dp = [[-1 for _ in range({len2+1})] for _ in range({len1+1})]\n"
            "```\n"
            "📝 Later, we check: if dp[i][j] != -1, use that value instead of recomputing."
        )

    # 15. Nested loop to fill DP using recurrence
    elif "nested loop" in q and "recurrence" in q:
        return (
            "👨‍🏫 To solve LCS iteratively, we use a nested loop and fill each cell based on the characters of s1 and s2.\n"
            "The logic inside the loop checks if characters match — if they do, we add 1 to the diagonal; else take max of left/top.\n\n"
            "💡 Here's the standard loop structure:\n"
            "```python\n"
            "for i in range(1, len(s1)+1):\n"
            "    for j in range(1, len(s2)+1):\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            dp[i][j] = dp[i-1][j-1] + 1\n"
            "        else:\n"
            "            dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
            "```\n"
            "🎯 This is the bottom-up (tabulation) version of LCS."
        )


    # 16. Loop comparing i and j positions
    elif ("loop" in q and "compare characters at i and j" in q) or ("loop that compares" in q and "at i and j" in q):
        match = re.search(r"strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract input strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 To compare characters at positions i and j in '{s1}' and '{s2}', we use nested loops:\n"
            f"This is a preliminary step often used before applying LCS logic.\n\n"
            "💡 Code to compare and print character positions:\n"
            "```python\n"
            f"s1 = '{s1}'\n"
            f"s2 = '{s2}'\n"
            "for i in range(len(s1)):\n"
            "    for j in range(len(s2)):\n"
            "        print(f's1[{i}] = {s1[i]}, s2[{j}] = {s2[j]}')\n"
            "```\n"
            "🎯 This helps in debugging character-by-character comparisons."
        )


    # 17. Top-down memoized LCS
    elif "top-down" in q and "memoization" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        return (
            f"👨‍🏫 To optimize the recursive solution of LCS for '{s1}' and '{s2}', we use top-down memoization.\n"
            f"This avoids redundant computation by storing results of subproblems.\n\n"
            f"💡 Here's the full function with memoization:\n"
            "```python\n"
            "def lcs_memo(s1, s2, m, n, dp):\n"
            "    if m == 0 or n == 0:\n"
            "        return 0\n"
            "    if dp[m][n] != -1:\n"
            "        return dp[m][n]\n"
            "    if s1[m-1] == s2[n-1]:\n"
            "        dp[m][n] = 1 + lcs_memo(s1, s2, m-1, n-1, dp)\n"
            "    else:\n"
            "        dp[m][n] = max(lcs_memo(s1, s2, m-1, n, dp), lcs_memo(s1, s2, m, n-1, dp))\n"
            "    return dp[m][n]\n"
            "```\n"
            "📝 Be sure to initialize dp as a 2D array with -1s before using this."
        )


    # 18. Fill DP table up to [i][j]
    elif "fill a dp table" in q and "up to index" in q:
        match = re.search(r"between ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 To fill the LCS table up to a particular index [i][j], we use LCS recurrence logic:\n"
            "1. If s1[i-1] == s2[j-1] → dp[i][j] = dp[i-1][j-1] + 1\n"
            "2. Else, take max(dp[i-1][j], dp[i][j-1])\n\n"
            "💡 Code snippet:\n"
            "```python\n"
            "if s1[i-1] == s2[j-1]:\n"
            "    dp[i][j] = dp[i-1][j-1] + 1\n"
            "else:\n"
            "    dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
            "```\n"
            "🎯 This is the transition formula used in bottom-up DP."
        )


        # 19. Initialize and print initial DP state
    elif "initialize a dp table" in q and "print its initial state" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        table_str = "\n".join(str(row) for row in dp)
        return (
            f"👨‍🏫 Here's how you initialize a DP table for strings '{s1}' and '{s2}' and print its initial state:\n\n"
            "💡 Code:\n"
            "```python\n"
            f"dp = [[0 for _ in range({n+1})] for _ in range({m+1})]\n"
            "for row in dp:\n"
            "    print(row)\n"
            "```\n"
            "📋 Initial state of DP table:\n"
            f"{table_str}"
        )


    # 20. Nested loops, only size given
    elif "simulate lcs table filling" in q and "only size" in q:
        match = re.search(r"size (\d+)x(\d+)", q)
        if not match:
            return "❌ Could not extract matrix size."
        m, n = int(match.group(1)), int(match.group(2))
        return (
            f"👨‍🏫 Sometimes we simulate LCS logic for practice or testing by filling a table of size {m}x{n} with dummy values.\n"
            "Here, we won't use any actual string — just apply LCS-style filling with made-up values.\n\n"
            "💡 Sample simulation:\n"
            "```python\n"
            f"dp = [[0 for _ in range({n+1})] for _ in range({m+1})]\n"
            "for i in range(1, m+1):\n"
            "    for j in range(1, n+1):\n"
            "        dp[i][j] = i + j  # dummy simulation logic\n"
            "```\n"
            "🧪 You can modify the formula to mimic LCS filling patterns."
        )



    return "❌ No matching implementation-level Level 1 pattern found in the question."

def answer_implementation_lvl2(question):
    import re
    q = question.lower()

    # 1. Bottom-up LCS
    if "bottom-up" in q and "lcs" in q and "return the length" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        return (
            f"👨‍🏫 To compute the Longest Common Subsequence (LCS) using the bottom-up dynamic programming approach, "
            f"we build a 2D table for strings '{s1}' and '{s2}' where each cell dp[i][j] stores the LCS length of the first i characters of '{s1}' and first j characters of '{s2}'.\n\n"
            f"🪜 Steps to follow:\n"
            f"1. Create a DP table of size ({m+1} x {n+1}) initialized with 0.\n"
            f"2. Loop through each character of '{s1}' and '{s2}'.\n"
            f"3. If characters match, update the current cell with diagonal value +1.\n"
            f"4. If not, take the maximum from the left or top cell.\n\n"
            f"🛠 Code to compute the LCS length:\n"
            "```python\n"
            f"def lcs_bottom_up(s1, s2):\n"
            f"    m, n = len(s1), len(s2)\n"
            f"    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]\n\n"
            f"    for i in range(1, m+1):\n"
            f"        for j in range(1, n+1):\n"
            f"            if s1[i-1] == s2[j-1]:\n"
            f"                dp[i][j] = dp[i-1][j-1] + 1\n"
            f"            else:\n"
            f"                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n\n"
            f"    return dp[m][n]\n\n"
            f"# Example:\n"
            f"print(lcs_bottom_up('{s1}', '{s2}'))  # Output: length of LCS\n"
            "```\n\n"
            "🎯 The final value `dp[m][n]` gives the length of the LCS.\n"
            "📌 This approach avoids recursion and computes LCS efficiently using tabulation."
        )

    # 2. Recursive LCS with Memoization
    elif "recursive" in q and "memoization" in q and "lcs" in q:
        match = re.search(r"strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        return (
            f"👨‍🏫 To compute the Longest Common Subsequence (LCS) between '{s1}' and '{s2}' using recursion, "
            f"we can apply memoization to avoid recalculating the same subproblems repeatedly.\n\n"
            f"🧠 Memoization is a top-down technique where we store previously computed results in a 2D table.\n"
            f"This greatly improves efficiency over plain recursion.\n\n"
            f"🪜 Steps:\n"
            f"1. Create a 2D table `dp` of size ({m+1} x {n+1}) initialized with -1.\n"
            f"2. Define a recursive function that returns LCS length for `s1[0..i-1]` and `s2[0..j-1]`.\n"
            f"3. If `dp[i][j]` already has a value, return it directly.\n"
            f"4. If characters match, store `1 + dp[i-1][j-1]`.\n"
            f"5. Else, store `max(dp[i-1][j], dp[i][j-1])`.\n\n"
            f"🛠 Python code for recursive LCS with memoization:\n"
            "```python\n"
            "def lcs_recursive_memo(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[-1 for _ in range(n+1)] for _ in range(m+1)]\n\n"
            "    def helper(i, j):\n"
            "        if i == 0 or j == 0:\n"
            "            return 0\n"
            "        if dp[i][j] != -1:\n"
            "            return dp[i][j]\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            dp[i][j] = 1 + helper(i-1, j-1)\n"
            "        else:\n"
            "            dp[i][j] = max(helper(i-1, j), helper(i, j-1))\n"
            "        return dp[i][j]\n\n"
            "    return helper(m, n)\n\n"
            f"# Example:\n"
            f"print(lcs_recursive_memo('{s1}', '{s2}'))  # Output: length of LCS\n"
            "```\n\n"
            "🎯 This avoids recomputation and is much faster than plain recursion.\n"
            "📌 Always initialize the memo table (`dp`) with -1 and pass correct indices (m, n) to start."
        )
    # 3. Create a class with a method to compute LCS using memoized approach
    elif "class" in q and "method" in q and "memoized" in q:
        match = re.search(r"compute lcs of ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        return (
            f"👨‍🏫 We can organize the memoized LCS approach in an object-oriented way by using a class.\n"
            f"Here’s how you can write a class with a method to compute LCS of '{s1}' and '{s2}'.\n\n"
            f"🧠 This approach makes code modular, reusable, and clean.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "class LCS:\n"
            "    def __init__(self, s1, s2):\n"
            "        self.s1 = s1\n"
            "        self.s2 = s2\n"
            "        self.dp = [[-1 for _ in range(len(s2)+1)] for _ in range(len(s1)+1)]\n\n"
            "    def compute(self, i, j):\n"
            "        if i == 0 or j == 0:\n"
            "            return 0\n"
            "        if self.dp[i][j] != -1:\n"
            "            return self.dp[i][j]\n"
            "        if self.s1[i-1] == self.s2[j-1]:\n"
            "            self.dp[i][j] = 1 + self.compute(i-1, j-1)\n"
            "        else:\n"
            "            self.dp[i][j] = max(self.compute(i-1, j), self.compute(i, j-1))\n"
            "        return self.dp[i][j]\n\n"
            "# Example usage:\n"
            f"obj = LCS('{s1}', '{s2}')\n"
            "print(obj.compute(len(obj.s1), len(obj.s2)))\n"
            "```\n\n"
            "🎯 This structure encapsulates everything inside a class.\n"
            "📌 Makes future extension (e.g., reconstructing the sequence) easier."
        )

    # 4. Fill a 2D DP table and return both table and LCS length
    elif "fill a 2d dp table" in q and "return both the table and lcs length" in q:
        match = re.search(r"strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        return (
            f"👨‍🏫 To return both the LCS length and the filled DP table for strings '{s1}' and '{s2}', "
            f"we simply construct the standard bottom-up LCS table and return it along with dp[m][n].\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_with_table(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]\n\n"
            "    for i in range(1, m+1):\n"
            "        for j in range(1, n+1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                dp[i][j] = dp[i-1][j-1] + 1\n"
            "            else:\n"
            "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n\n"
            "    return dp, dp[m][n]\n\n"
            f"# Example:\n"
            f"table, length = lcs_with_table('{s1}', '{s2}')\n"
            f"print('LCS Length:', length)\n"
            f"for row in table:\n"
            f"    print(row)\n"
            "```\n\n"
            "🎯 This helps in visualizing the table and also getting the result.\n"
            "📌 Useful when we also want to reconstruct the LCS string later."
        )

    # 5. Print the filled LCS table after computing it
    elif "print" in q and "filled lcs table" in q and "after computing" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        m, n = len(s1), len(s2)
        return (
            f"👨‍🏫 After computing the LCS table for '{s1}' and '{s2}', we can print it to observe how values were filled.\n"
            f"This helps in debugging and understanding how the algorithm builds solutions.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def print_lcs_table(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]\n\n"
            "    for i in range(1, m+1):\n"
            "        for j in range(1, n+1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                dp[i][j] = dp[i-1][j-1] + 1\n"
            "            else:\n"
            "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n\n"
            "    for row in dp:\n"
            "        print(' '.join(map(str, row)))\n"
            "```\n\n"
            f"# Output for strings '{s1}' and '{s2}':\n"
            f"print_lcs_table('{s1}', '{s2}')\n\n"
            "🎯 This helps verify that DP table values are filled as expected."
        )

    # 6. Extract and return the LCS string (not just length)
    elif "return the lcs string" in q and "not just length" in q:
        match = re.search(r"['\"](.*?)['\"] and ['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 To return the actual LCS string between '{s1}' and '{s2}', "
            f"we first compute the full LCS table and then backtrack from the bottom-right corner.\n\n"
            f"🪜 Steps:\n"
            f"1. Build DP table using standard LCS logic.\n"
            f"2. Start from bottom-right and move diagonally if characters match.\n"
            f"3. Else move in direction of the larger value (left or up).\n"
            f"4. Collect characters in reverse order.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def get_lcs_string(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]\n\n"
            "    for i in range(1, m+1):\n"
            "        for j in range(1, n+1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                dp[i][j] = dp[i-1][j-1] + 1\n"
            "            else:\n"
            "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n\n"
            "    # Backtrack to build the LCS string\n"
            "    i, j = m, n\n"
            "    lcs = []\n"
            "    while i > 0 and j > 0:\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            lcs.append(s1[i-1])\n"
            "            i -= 1\n"
            "            j -= 1\n"
            "        elif dp[i-1][j] > dp[i][j-1]:\n"
            "            i -= 1\n"
            "        else:\n"
            "            j -= 1\n"
            "    return ''.join(reversed(lcs))\n"
            "```\n\n"
            f"# Example:\n"
            f"print(get_lcs_string('{s1}', '{s2}'))  # Output: actual LCS string\n\n"
            "🎯 This gives the actual sequence, not just the length.\n"
            "📌 Useful when you need to know *what* the common subsequence is."
        )

    # 7. Build the LCS table and track operations at each cell
    elif "build the lcs table" in q and "track the operations" in q:
        match = re.search(r"comparisons in ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 To track how each cell in the LCS DP table is filled for strings '{s1}' and '{s2}', "
            f"we can print the decision made at each step.\n\n"
            f"📋 Either we:\n"
            f"• Add 1 (match found), or\n"
            f"• Take max from top/left (no match)\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_tracking_operations(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]\n\n"
            "    for i in range(1, m+1):\n"
            "        for j in range(1, n+1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                dp[i][j] = dp[i-1][j-1] + 1\n"
            "                print(f\"Match at s1[{i-1}] and s2[{j-1}] → '{s1[i-1]}' → dp[{i}][{j}] = {dp[i][j]}\")\n"
            "            else:\n"
            "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
            "                print(f\"No match at s1[{i-1}] and s2[{j-1}] → dp[{i}][{j}] = {dp[i][j]}\")\n"
            "```\n\n"
            f"# Example:\n"
            f"lcs_tracking_operations('{s1}', '{s2}')\n\n"
            "🎯 This gives a cell-by-cell trace of how the LCS table is formed."
        )
    # 8. Top-down memoization with input strings
    elif "handle input strings" in q and "top-down memoization" in q:
        match = re.search(r"strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This function reads two input strings '{s1}' and '{s2}' and returns the LCS length using top-down memoization.\n\n"
            f"🧠 Memoization avoids repeated calculations by storing intermediate results in a dictionary.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_top_down(s1, s2):\n"
            "    memo = {}\n\n"
            "    def dp(i, j):\n"
            "        if i == len(s1) or j == len(s2):\n"
            "            return 0\n"
            "        if (i, j) in memo:\n"
            "            return memo[(i, j)]\n"
            "        if s1[i] == s2[j]:\n"
            "            memo[(i, j)] = 1 + dp(i + 1, j + 1)\n"
            "        else:\n"
            "            memo[(i, j)] = max(dp(i + 1, j), dp(i, j + 1))\n"
            "        return memo[(i, j)]\n\n"
            "    return dp(0, 0)\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_top_down('{s1}', '{s2}'))\n\n"
            "🎯 This uses recursion but avoids recomputation via memoization."
        )

    # 9. Print match or max decision at each DP table cell
    elif "print the decision" in q and "each cell in the dp table" in q:
        match = re.search(r"for ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This function prints whether a match was found or a max was taken for each cell in the LCS DP table for '{s1}' and '{s2}'.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_decision_trace(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n\n"
            "    for i in range(1, m + 1):\n"
            "        for j in range(1, n + 1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                dp[i][j] = dp[i-1][j-1] + 1\n"
            "                print(f\"Match at dp[{i}][{j}] = {dp[i][j]} (chars: '{s1[i-1]}')\")\n"
            "            else:\n"
            "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
            "                print(f\"No match at dp[{i}][{j}] → max of top {dp[i-1][j]} and left {dp[i][j-1]} = {dp[i][j]}\")\n"
            "```\n\n"
            f"# Example:\n"
            f"lcs_decision_trace('{s1}', '{s2}')\n\n"
            "🎯 This is useful for debugging and understanding how the LCS table is built."
        )

    # 10. Complete recursive function stub
    elif "complete the logic" in q and "compute lcs recursively" in q:
        match = re.search(r"for ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This answer completes the recursive LCS function logic for strings '{s1}' and '{s2}'.\n"
            f"No memoization used here — just plain recursion.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_recursive(i, j, s1, s2):\n"
            "    if i == len(s1) or j == len(s2):\n"
            "        return 0\n"
            "    if s1[i] == s2[j]:\n"
            "        return 1 + lcs_recursive(i + 1, j + 1, s1, s2)\n"
            "    else:\n"
            "        return max(lcs_recursive(i + 1, j, s1, s2), lcs_recursive(i, j + 1, s1, s2))\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_recursive(0, 0, '{s1}', '{s2}'))\n\n"
            "🎯 Works well for small strings, but inefficient for long inputs."
        )

    # 11. Add memoization to existing recursive LCS
    elif "add memoization" in q and "existing recursive lcs" in q:
        match = re.search(r"for ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 We improve the plain recursive LCS function by adding memoization to avoid recomputing overlapping subproblems.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_recursive_memo(i, j, s1, s2, memo):\n"
            "    if i == len(s1) or j == len(s2):\n"
            "        return 0\n"
            "    if (i, j) in memo:\n"
            "        return memo[(i, j)]\n"
            "    if s1[i] == s2[j]:\n"
            "        memo[(i, j)] = 1 + lcs_recursive_memo(i + 1, j + 1, s1, s2, memo)\n"
            "    else:\n"
            "        memo[(i, j)] = max(\n"
            "            lcs_recursive_memo(i + 1, j, s1, s2, memo),\n"
            "            lcs_recursive_memo(i, j + 1, s1, s2, memo)\n"
            "        )\n"
            "    return memo[(i, j)]\n"
            "```\n\n"
            f"# Example:\n"
            f"memo = {{}}\n"
            f"print(lcs_recursive_memo(0, 0, '{s1}', '{s2}', memo))\n\n"
            "🎯 Efficient and clean — avoids repeated recursive calls."
        )

    # 12. Store intermediate results in dictionary instead of 2D list
    elif "stores intermediate results in a dictionary" in q and "instead of a 2d list" in q:
        match = re.search(r"for ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 Instead of using a 2D list to store DP values, we can use a dictionary.\n"
            f"This is memory-efficient especially when table is sparse.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_dict_table(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = {}\n"
            "    for i in range(m + 1):\n"
            "        for j in range(n + 1):\n"
            "            if i == 0 or j == 0:\n"
            "                dp[(i, j)] = 0\n"
            "            elif s1[i - 1] == s2[j - 1]:\n"
            "                dp[(i, j)] = dp[(i - 1, j - 1)] + 1\n"
            "            else:\n"
            "                dp[(i, j)] = max(dp[(i - 1, j)], dp[(i, j - 1)])\n"
            "    return dp[(m, n)]\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_dict_table('{s1}', '{s2}'))\n\n"
            "🎯 Uses hash map (dictionary) instead of 2D matrix for flexibility."
        )
    # 13. Print both the LCS length and LCS string
    elif "prints both the lcs length and lcs string" in q:
        match = re.search(r"strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This function computes both the LCS length and the actual LCS string for '{s1}' and '{s2}'.\n"
            f"We first build the DP table and then backtrack to form the string.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_length_and_string(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n\n"
            "    # Backtracking\n"
            "    lcs = []\n"
            "    i, j = m, n\n"
            "    while i > 0 and j > 0:\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            lcs.append(s1[i-1])\n"
            "            i -= 1\n"
            "            j -= 1\n"
            "        elif dp[i-1][j] > dp[i][j-1]:\n"
            "            i -= 1\n"
            "        else:\n"
            "            j -= 1\n"
            "    return dp[m][n], ''.join(reversed(lcs))\n"
            "```\n\n"
            f"# Example:\n"
            f"length, seq = lcs_length_and_string('{s1}', '{s2}')\n"
            f"print('Length:', length)\n"
            f"print('LCS:', seq)\n\n"
            "🎯 Very useful when both the score and actual sequence are required."
        )

    # 14. Complete a partially implemented LCS bottom-up function
    elif "complete a partially implemented lcs bottom-up function" in q:
        match = re.search(r"using ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This completes a partially written LCS bottom-up function for '{s1}' and '{s2}'.\n"
            f"The main idea is to fill the DP table row by row.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def complete_lcs_bottom_up(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]\n\n"
            "    for i in range(1, m+1):\n"
            "        for j in range(1, n+1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                dp[i][j] = dp[i-1][j-1] + 1\n"
            "            else:\n"
            "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n\n"
            "    return dp[m][n]\n"
            "```\n\n"
            f"# Example:\n"
            f"print(complete_lcs_bottom_up('{s1}', '{s2}'))\n\n"
            "🎯 Clean and efficient tabulation logic."
        )

    # 15. Optimize LCS to use two rows instead of full table
    elif "optimize an existing lcs table-filling loop" in q and "only two rows of memory" in q:
        match = re.search(r"using ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 We can optimize the LCS DP solution to use only two rows instead of the full table.\n"
            f"This saves space while still giving correct result.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_optimized_space(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    prev = [0] * (n + 1)\n"
            "    curr = [0] * (n + 1)\n\n"
            "    for i in range(1, m + 1):\n"
            "        for j in range(1, n + 1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                curr[j] = prev[j-1] + 1\n"
            "            else:\n"
            "                curr[j] = max(prev[j], curr[j-1])\n"
            "        prev, curr = curr, [0] * (n + 1)\n\n"
            "    return prev[n]\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_optimized_space('{s1}', '{s2}'))\n\n"
            "🎯 Useful when working with large input strings to reduce memory usage."
        )

    # 16. Create a helper function to trace back LCS from DP table
    elif "helper function to trace back the lcs string" in q:
        match = re.search(r"for ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 After building the LCS table, we can use a helper function to trace back the actual LCS string.\n"
            f"This works by starting from bottom-right and following the logic.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def trace_lcs(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n\n"
            "    def build(i, j):\n"
            "        if i == 0 or j == 0:\n"
            "            return ''\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            return build(i-1, j-1) + s1[i-1]\n"
            "        elif dp[i-1][j] > dp[i][j-1]:\n"
            "            return build(i-1, j)\n"
            "        else:\n"
            "            return build(i, j-1)\n\n"
            "    return build(m, n)\n"
            "```\n\n"
            f"# Example:\n"
            f"print(trace_lcs('{s1}', '{s2}'))\n\n"
            "🎯 This is useful when you already have the table and just want the sequence."
        )

    # 17. Return both LCS length and sequence in a full function
    elif "return both lcs length and the sequence" in q and "input strings" in q:
        match = re.search(r"strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This function returns both the LCS length and the actual LCS sequence for strings '{s1}' and '{s2}'.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_length_and_sequence(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n\n"
            "    i, j = m, n\n"
            "    lcs = []\n"
            "    while i > 0 and j > 0:\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            lcs.append(s1[i-1])\n"
            "            i -= 1\n"
            "            j -= 1\n"
            "        elif dp[i-1][j] > dp[i][j-1]:\n"
            "            i -= 1\n"
            "        else:\n"
            "            j -= 1\n"
            "    return dp[m][n], ''.join(reversed(lcs))\n"
            "```\n\n"
            f"# Example:\n"
            f"length, sequence = lcs_length_and_sequence('{s1}', '{s2}')\n"
            f"print('Length:', length)\n"
            f"print('Sequence:', sequence)\n\n"
            "🎯 Great when both info is needed in one step."
        )
    # 18. Check if computed LCS is a subsequence of both strings
    elif "check and return whether the computed lcs string is a subsequence" in q:
        match = re.search(r"of both ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 Once we compute the LCS string, we can verify if it’s a subsequence of both '{s1}' and '{s2}'.\n\n"
            f"🧪 A string x is a subsequence of y if all characters of x appear in y in order (not necessarily contiguous).\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def is_subsequence(sub, full):\n"
            "    i = 0\n"
            "    for char in full:\n"
            "        if i < len(sub) and sub[i] == char:\n"
            "            i += 1\n"
            "    return i == len(sub)\n\n"
            "def lcs_and_check_subsequence(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n\n"
            "    i, j = m, n\n"
            "    lcs = []\n"
            "    while i > 0 and j > 0:\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            lcs.append(s1[i-1])\n"
            "            i -= 1\n"
            "            j -= 1\n"
            "        elif dp[i-1][j] > dp[i][j-1]:\n"
            "            i -= 1\n"
            "        else:\n"
            "            j -= 1\n"
            "    lcs_str = ''.join(reversed(lcs))\n"
            "    return is_subsequence(lcs_str, s1) and is_subsequence(lcs_str, s2)\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_and_check_subsequence('{s1}', '{s2}'))\n\n"
            "🎯 Output will be True if LCS is a subsequence of both strings."
        )

    # 19. Given filled DP table, trace one valid LCS path
    elif "given a filled dp table" in q and "trace and return one valid lcs path" in q:
        match = re.search(r"for ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This function assumes the LCS table is filled and backtracks to return one valid LCS sequence.\n"
            f"It moves diagonally when characters match and otherwise follows the direction of the greater value.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def trace_one_lcs_path(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n\n"
            "    i, j = m, n\n"
            "    lcs = []\n"
            "    while i > 0 and j > 0:\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            lcs.append(s1[i-1])\n"
            "            i -= 1\n"
            "            j -= 1\n"
            "        elif dp[i-1][j] > dp[i][j-1]:\n"
            "            i -= 1\n"
            "        else:\n"
            "            j -= 1\n"
            "    return ''.join(reversed(lcs))\n"
            "```\n\n"
            f"# Example:\n"
            f"print(trace_one_lcs_path('{s1}', '{s2}'))\n\n"
            "🎯 This is useful for understanding the path reconstruction process."
        )

    # 20. Wrapper function to read input, compute LCS and return result
    elif "wrapper function" in q and "computes lcs length" in q:
        match = re.search(r"strings ['\"](.*?)['\"] and ['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 A wrapper function simply packages the logic: takes input strings, calls LCS logic, and returns output.\n"
            f"Useful in API or main function context.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def compute_lcs_wrapper(s1, s2):\n"
            "    def lcs(s1, s2):\n"
            "        m, n = len(s1), len(s2)\n"
            "        dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "        for i in range(m):\n"
            "            for j in range(n):\n"
            "                if s1[i] == s2[j]:\n"
            "                    dp[i+1][j+1] = dp[i][j] + 1\n"
            "                else:\n"
            "                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n"
            "        return dp[m][n]\n"
            "    return lcs(s1, s2)\n"
            "```\n\n"
            f"# Example:\n"
            f"print(compute_lcs_wrapper('{s1}', '{s2}'))\n\n"
            "🎯 Clean main-level logic, useful for integrating in projects."
        )

def answer_implementation_lvl3(question):
    import re
    q = question.lower()
# 1. Space-Optimized LCS using Two 1D Arrays
    if "space-optimized" in q and "two 1d arrays" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 To reduce space usage in LCS, we can use just two 1D arrays instead of a full 2D table.\n"
            f"This reduces the space complexity from O(m×n) to O(2×n).\n"
            f"We compute the table row-by-row using two arrays: previous and current.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_space_optimized(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    if n > m:\n"
            "        s1, s2 = s2, s1  # Ensure s2 is shorter\n"
            "        m, n = n, m\n"
            "    prev = [0] * (n + 1)\n"
            "    curr = [0] * (n + 1)\n\n"
            "    for i in range(1, m + 1):\n"
            "        for j in range(1, n + 1):\n"
            "            if s1[i - 1] == s2[j - 1]:\n"
            "                curr[j] = prev[j - 1] + 1\n"
            "            else:\n"
            "                curr[j] = max(prev[j], curr[j - 1])\n"
            "        prev, curr = curr, [0] * (n + 1)\n"
            "    return prev[n]\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_space_optimized('{s1}', '{s2}'))\n\n"
            "🎯 Efficient when memory is limited — uses only two rows."
        )

# 2. Return all possible LCS sequences
    elif "return all possible lcs sequences" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This function returns all possible LCS sequences between '{s1}' and '{s2}' using recursion with memoization.\n"
            f"🔁 We first fill the DP table, then use a set-based backtracking approach to collect all sequences.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def all_lcs(s1, s2):\n"
            "    from functools import lru_cache\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n\n"
            "    @lru_cache(None)\n"
            "    def backtrack(i, j):\n"
            "        if i == 0 or j == 0:\n"
            "            return set([''])\n"
            "        elif s1[i-1] == s2[j-1]:\n"
            "            return {x + s1[i-1] for x in backtrack(i-1, j-1)}\n"
            "        else:\n"
            "            result = set()\n"
            "            if dp[i-1][j] >= dp[i][j-1]:\n"
            "                result.update(backtrack(i-1, j))\n"
            "            if dp[i][j-1] >= dp[i-1][j]:\n"
            "                result.update(backtrack(i, j-1))\n"
            "            return result\n\n"
            "    return list(backtrack(m, n))\n"
            "```\n\n"
            f"# Example:\n"
            f"print(all_lcs('{s1}', '{s2}'))\n\n"
            "🎯 This handles multiple LCS variants — great for debugging or path enumeration."
        )

    # 3. Return indices of matching characters
    elif "indices of matching characters" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", question)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This enhanced version of LCS also returns the matching indices in both strings.\n"
            f"🎯 Very useful for tasks like diff tools or highlighting matches.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_with_indices(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n\n"
            "    i, j = m, n\n"
            "    indices = []\n"
            "    while i > 0 and j > 0:\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            indices.append((i-1, j-1))\n"
            "            i -= 1\n"
            "            j -= 1\n"
            "        elif dp[i-1][j] > dp[i][j-1]:\n"
            "            i -= 1\n"
            "        else:\n"
            "            j -= 1\n"
            "    return list(reversed(indices))\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_with_indices('{s1}', '{s2}'))\n\n"
            "🎯 Output will be list of index pairs where characters matched."
        )
    
    
    elif "count the number of distinct lcs sequences" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 To count all distinct LCS sequences for '{s1}' and '{s2}', we use DP to fill the LCS table\n"
            f"and then recursively count all distinct paths from the bottom-right cell.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "from functools import lru_cache\n"
            "def count_all_lcs(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0] * (n+1) for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n\n"
            "    @lru_cache(None)\n"
            "    def count(i, j):\n"
            "        if i == 0 or j == 0:\n"
            "            return 1\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            return count(i-1, j-1)\n"
            "        total = 0\n"
            "        if dp[i-1][j] == dp[i][j]:\n"
            "            total += count(i-1, j)\n"
            "        if dp[i][j-1] == dp[i][j]:\n"
            "            total += count(i, j-1)\n"
            "        if dp[i-1][j] == dp[i][j] and dp[i][j-1] == dp[i][j]:\n"
            "            total -= count(i-1, j-1)  # Avoid overcounting\n"
            "        return total\n\n"
            "    return count(m, n)\n"
            "```\n\n"
            f"# Example:\n"
            f"print(count_all_lcs('{s1}', '{s2}'))\n\n"
            "🎯 This gives the total number of unique LCS paths."
        )

    elif "traceback and visualize the path" in q.lower() and "dp table" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 Traceback lets us visualize how we reach the LCS in the DP table.\n"
            f"We mark each step taken to construct the sequence: diagonal for match, up/left for max path.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_trace_visual(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n\n"
            "    path = []\n"
            "    i, j = m, n\n"
            "    while i > 0 and j > 0:\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            path.append((i-1, j-1))  # Match\n"
            "            i -= 1\n"
            "            j -= 1\n"
            "        elif dp[i-1][j] >= dp[i][j-1]:\n"
            "            i -= 1\n"
            "        else:\n"
            "            j -= 1\n"
            "    for idx in reversed(path):\n"
            "        print(f\"Match at s1[{idx[0]}] == s2[{idx[1]}] → '{s1[idx[0]]}'\")\n"
            "```\n\n"
            f"# Example:\n"
            f"lcs_trace_visual('{s1}', '{s2}')\n\n"
            "🎯 Helpful to visually trace character matches and LCS path."
        )

    elif "iterative tabular approach with backtracking" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This approach avoids recursion and instead uses tabulation + backtracking to find the LCS.\n"
            f"It is efficient and safe from stack overflow.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_iterative(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n\n"
            "    i, j = m, n\n"
            "    lcs = []\n"
            "    while i > 0 and j > 0:\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            lcs.append(s1[i-1])\n"
            "            i -= 1\n"
            "            j -= 1\n"
            "        elif dp[i-1][j] > dp[i][j-1]:\n"
            "            i -= 1\n"
            "        else:\n"
            "            j -= 1\n"
            "    return ''.join(reversed(lcs))\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_iterative('{s1}', '{s2}'))\n\n"
            "🎯 Simple and efficient without recursion."
        )


    elif "handle a list of strings" in q.lower() or ("generic lcs method" in q.lower() and "list of strings" in q.lower()):
        match = re.findall(r"['\"](.*?)['\"]", q)
        if not match or len(match) < 3:
            return "❌ Could not extract strings."
        s1, s2, s3 = match[0], match[1], match[2]
        return (
            f"👨‍🏫 This function computes the LCS of more than two strings — here, '{s1}', '{s2}', and '{s3}'.\n"
            f"We use a 3D DP table to generalize the LCS algorithm to three strings.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_of_three(s1, s2, s3):\n"
            "    m, n, o = len(s1), len(s2), len(s3)\n"
            "    dp = [[[0]*(o+1) for _ in range(n+1)] for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            for k in range(o):\n"
            "                if s1[i] == s2[j] == s3[k]:\n"
            "                    dp[i+1][j+1][k+1] = dp[i][j][k] + 1\n"
            "                else:\n"
            "                    dp[i+1][j+1][k+1] = max(dp[i][j+1][k+1], dp[i+1][j][k+1], dp[i+1][j+1][k])\n"
            "    return dp[m][n][o]\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_of_three('{s1}', '{s2}', '{s3}'))\n\n"
            "🎯 Handles multiple strings — perfect for sequence analysis across versions."
        )

    elif "custom data structure" in q.lower() and "intermediate states" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 Here, we use a custom class to store intermediate states of LCS computation.\n"
            f"This allows flexibility for extension, debugging, or logging internal steps.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "class LCSState:\n"
            "    def __init__(self, s1, s2):\n"
            "        self.s1 = s1\n"
            "        self.s2 = s2\n"
            "        self.dp = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]\n"
            "    def compute(self):\n"
            "        for i in range(len(self.s1)):\n"
            "            for j in range(len(self.s2)):\n"
            "                if self.s1[i] == self.s2[j]:\n"
            "                    self.dp[i+1][j+1] = self.dp[i][j] + 1\n"
            "                else:\n"
            "                    self.dp[i+1][j+1] = max(self.dp[i][j+1], self.dp[i+1][j])\n"
            "        return self.dp[-1][-1]\n"
            "```\n\n"
            f"# Example:\n"
            f"state = LCSState('{s1}', '{s2}')\nprint(state.compute())\n\n"
            "🎯 Ideal for OOP-based LCS modules where DP table may be reused or analyzed."
        )

    elif "edit distance results" in q.lower() and "in a single pass" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This function computes both LCS and edit distance in one pass using DP.\n"
            f"LCS gives longest common subsequence, edit distance counts operations.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_and_edit_distance(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n"
            "    lcs_len = dp[m][n]\n"
            "    edit_distance = (m + n) - 2 * lcs_len\n"
            "    return lcs_len, edit_distance\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_and_edit_distance('{s1}', '{s2}'))\n\n"
            "🎯 Useful when both similarity and change measure are needed."
        )


    elif "diff-like output" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This function uses LCS to generate a diff-like output showing additions (+), deletions (-), and matches ( ).\n"
            f"Helpful for text comparison and visualizing changes.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_diff_output(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n\n"
            "    i, j = m, n\n"
            "    result = []\n"
            "    while i > 0 and j > 0:\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            result.append('  ' + s1[i-1])\n"
            "            i -= 1\n"
            "            j -= 1\n"
            "        elif dp[i-1][j] > dp[i][j-1]:\n"
            "            result.append('- ' + s1[i-1])\n"
            "            i -= 1\n"
            "        else:\n"
            "            result.append('+ ' + s2[j-1])\n"
            "            j -= 1\n"
            "    while i > 0:\n"
            "        result.append('- ' + s1[i-1])\n"
            "        i -= 1\n"
            "    while j > 0:\n"
            "        result.append('+ ' + s2[j-1])\n"
            "        j -= 1\n"
            "    return '\n'.join(reversed(result))\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_diff_output('{s1}', '{s2}'))\n\n"
            "🎯 Useful for version control or displaying text changes."
        )

    elif "generator function" in q.lower() and "step-by-step" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This version of LCS is a generator function that yields the DP matrix row-by-row.\n"
            f"It helps visualize or debug the computation step-by-step.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_generator(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(1, m+1):\n"
            "        for j in range(1, n+1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                dp[i][j] = dp[i-1][j-1] + 1\n"
            "            else:\n"
            "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
            "        yield dp[i][:]  # Yield a copy of the current row\n"
            "```\n\n"
            f"# Example usage:\n"
            f"for row in lcs_generator('{s1}', '{s2}'):\n    print(row)\n\n"
            "🎯 Handy for visualizing each stage of DP table filling."
        )

    elif "multithreaded implementation" in q.lower() and "parallel" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 Multithreading can speed up DP table filling by processing each row or diagonal in parallel.\n"
            f"Note: Python's GIL limits CPU-bound speedups unless multiprocessing is used.\n\n"
            f"🛠 Simplified Example Using Threads per Row:\n"
            "```python\n"
            "import threading\n"
            "def lcs_parallel(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    def fill_row(i):\n"
            "        for j in range(1, n+1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                dp[i][j] = dp[i-1][j-1] + 1\n"
            "            else:\n"
            "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
            "    threads = []\n"
            "    for i in range(1, m+1):\n"
            "        t = threading.Thread(target=fill_row, args=(i,))\n"
            "        threads.append(t)\n"
            "        t.start()\n"
            "    for t in threads:\n"
            "        t.join()\n"
            "    return dp[m][n]\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_parallel('{s1}', '{s2}'))\n\n"
            "🎯 Demo of threading logic — best for large inputs + custom interpreters."
        )

    elif "custom cache strategy" in q.lower() and "memoization" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 Instead of @lru_cache, we use a manual dictionary to store LCS subproblem results for better control.\n"
            f"This allows you to implement custom caching strategies.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_custom_cache(s1, s2):\n"
            "    cache = {}\n"
            "    def helper(i, j):\n"
            "        if i == 0 or j == 0:\n"
            "            return 0\n"
            "        if (i, j) in cache:\n"
            "            return cache[(i, j)]\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            cache[(i, j)] = 1 + helper(i-1, j-1)\n"
            "        else:\n"
            "            cache[(i, j)] = max(helper(i-1, j), helper(i, j-1))\n"
            "        return cache[(i, j)]\n"
            "    return helper(len(s1), len(s2))\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_custom_cache('{s1}', '{s2}'))\n\n"
            "🎯 Lets you control what and how subproblems are cached."
        )

    elif "ignores case" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This LCS version ignores case by converting all characters to the same case before comparison.\n"
            f"Useful for case-insensitive text analysis.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_ignore_case(s1, s2):\n"
            "    s1, s2 = s1.lower(), s2.lower()\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n"
            "    return dp[m][n]\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_ignore_case('{s1}', '{s2}'))\n\n"
            "🎯 Ignores upper/lowercase mismatch while computing LCS."
        )

    elif "non-empty and consist of uppercase letters only" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This wrapper ensures both strings are non-empty and fully uppercase before calling LCS.\n"
            f"Acts as a validation layer before actual computation.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def safe_lcs(s1, s2):\n"
            "    if not s1 or not s2:\n"
            "        return 0\n"
            "    if not (s1.isupper() and s2.isupper()):\n"
            "        raise ValueError('Strings must be uppercase only')\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n"
            "    return dp[m][n]\n"
            "```\n\n"
            f"# Example:\n"
            f"print(safe_lcs('{s1}', '{s2}'))\n\n"
            "🎯 Adds a safety layer for uppercase constraints."
        )

    elif "test cases to validate your lcs implementation" in q.lower():
        match = re.search(r"['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract string."
        s = match.group(1)
        return (
            f"👨‍🏫 Test cases are crucial to verify correctness and handle edge conditions in LCS.\n"
            f"Here, we define a variety of tests including empty strings and identical inputs like '{s}'.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def test_lcs():\n"
            "    def lcs_length(a, b):\n"
            "        m, n = len(a), len(b)\n"
            "        dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "        for i in range(m):\n"
            "            for j in range(n):\n"
            "                if a[i] == b[j]:\n"
            "                    dp[i+1][j+1] = dp[i][j] + 1\n"
            "                else:\n"
            "                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n"
            "        return dp[m][n]\n"
            "    assert lcs_length('', '') == 0\n"
            "    assert lcs_length('ABC', '') == 0\n"
            "    assert lcs_length('', 'XYZ') == 0\n"
            "    assert lcs_length('ABC', 'ABC') == 3\n"
            "    assert lcs_length('{s}', '{s}') == len('{s}')\n"
            "    print('✅ All test cases passed!')\n"
            "```\n\n"
            "🎯 Run test_lcs() to validate LCS correctness across edge cases."
        )
    

    elif "functional programming style" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 Functional programming avoids mutable state — we implement LCS using recursion and memoization.\n"
            f"We avoid any external or global variables.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "from functools import lru_cache\n"
            "def lcs_functional(s1, s2):\n"
            "    @lru_cache(None)\n"
            "    def lcs(i, j):\n"
            "        if i == 0 or j == 0:\n"
            "            return 0\n"
            "        if s1[i-1] == s2[j-1]:\n"
            "            return 1 + lcs(i-1, j-1)\n"
            "        return max(lcs(i-1, j), lcs(i, j-1))\n"
            "    return lcs(len(s1), len(s2))\n"
            "```\n\n"
            f"# Example:\n"
            f"print(lcs_functional('{s1}', '{s2}'))\n\n"
            "🎯 Clean and pure LCS with no side effects."
        )

    elif "cli tool" in q.lower() and "computation table" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This CLI tool accepts input and prints both LCS and the computation table.\n"
            f"Great for debugging or use in command-line apps.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_cli(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n"
            "    for row in dp:\n"
            "        print(row)\n"
            "    print('LCS Length:', dp[m][n])\n"
            "```\n\n"
            f"# Example:\n"
            f"lcs_cli('{s1}', '{s2}')\n\n"
            "🎯 Simple tool to visualize matrix and LCS."
        )


    elif "cache previously computed string pairs" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 We create a utility that caches previously computed LCS results to save recomputation.\n"
            f"It uses a dictionary for memoization keyed by (s1, s2).\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "lcs_cache = {}\n"
            "def cached_lcs(s1, s2):\n"
            "    key = (s1, s2)\n"
            "    if key in lcs_cache:\n"
            "        return lcs_cache[key]\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n"
            "    lcs_cache[key] = dp[m][n]\n"
            "    return dp[m][n]\n"
            "```\n\n"
            f"# Example:\n"
            f"print(cached_lcs('{s1}', '{s2}'))\n\n"
            "🎯 Boosts performance when same string pairs are queried repeatedly."
        )

    elif "web api endpoint" in q.lower() and "query params" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"👨‍🏫 This Flask-based API accepts s1 and s2 as query parameters and returns the LCS result in JSON.\n"
            f"Ideal for building web-based string analysis tools.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "from flask import Flask, request, jsonify\n"
            "app = Flask(__name__)\n\n"
            "def lcs(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if s1[i] == s2[j]:\n"
            "                dp[i+1][j+1] = dp[i][j] + 1\n"
            "            else:\n"
            "                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n"
            "    return dp[m][n]\n\n"
            "@app.route('/lcs')\n"
            "def lcs_endpoint():\n"
            "    s1 = request.args.get('s1', '')\n"
            "    s2 = request.args.get('s2', '')\n"
            "    result = lcs(s1, s2)\n"
            "    return jsonify({'s1': s1, 's2': s2, 'lcs_length': result})\n\n"
            "# Example: Run and visit http://localhost:5000/lcs?s1=ABC&s2=AC\n"
            "```\n\n"
            "🎯 Integrate LCS as a service — ideal for apps or platforms."
        )

#for level 1 optimization

# 🚀 Optimization Category – LCS Level 1 Answers

def answer_optimization_lvl1(q):
    import re
    q = q.lower()

    if "inefficient to solve lcs using plain recursion" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"🔁 Plain recursion for LCS (like for '{s1}' and '{s2}') is inefficient because:\n"
            "- It recalculates the same subproblems multiple times\n"
            "- Time complexity is exponential: O(2^n)\n"
            "- It lacks memory to store intermediate results\n\n"
            "🧠 Result: Very slow for even moderately long strings."
        )

    elif "avoid recalculating the same subproblems" in q:
        return (
            "💡 We avoid recalculating LCS subproblems by using:\n\n"
            "✅ Memoization: Store already computed results\n"
            "✅ Dynamic Programming: Bottom-up table filling\n\n"
            "🎯 Both ensure each subproblem is solved only once."
        )

    elif "what is memoization" in q:
        return (
            "📚 Memoization is an optimization where results of expensive function calls are stored.\n\n"
            "✅ In LCS recursion, we save results for (i, j) states.\n"
            "🔁 So if the same (i, j) comes again, we just look up instead of recalculating.\n\n"
            "🎯 Converts exponential time → polynomial time."
        )

    elif "how dynamic programming improves the performance of lcs" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract inputs."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"🧠 Dynamic Programming improves LCS performance for '{s1}' and '{s2}' by:\n"
            "1️⃣ Storing previous results in a DP table\n"
            "2️⃣ Avoiding recursion and redundant calls\n"
            "3️⃣ Solving subproblems in increasing order\n\n"
            "🎯 Time complexity drops to O(m×n) from O(2^n)"
        )

    elif "what kind of problem pattern makes lcs suitable for optimization" in q:
        return (
            "🔍 LCS fits DP optimization because:\n\n"
            "✔️ It has overlapping subproblems\n"
            "✔️ It has optimal substructure\n"
            "✔️ The solution depends on smaller inputs\n\n"
            "🎯 Classic case for applying memoization or tabulation."
        )

    elif "overlapping subproblems in lcs suggest that optimization is needed" in q:
        return (    
            "🔁 In LCS, you often face the same subproblem multiple times — like computing LCS(i, j) over and over.\n\n"
            "💡 This repetition wastes time and leads to exponential complexity in plain recursion.\n\n"
            "📦 Optimization techniques like memoization or tabulation help by storing results and avoiding recomputation.\n\n"
            "🎯 That's why overlapping subproblems are a key sign that optimization is essential in LCS."
        )

    elif "how does tabulation help in optimizing lcs" in q:
        return (
            "📊 Tabulation helps by building the LCS solution from the bottom up using loops, not recursion.\n\n"
            "🧠 Here's how it works:\n"
            "1️⃣ Create a 2D table of size (m+1) x (n+1) where m and n are the lengths of the strings.\n"
            "2️⃣ Fill the table row by row based on matching characters.\n"
            "3️⃣ Use already filled values to compute new ones — no need to re-calculate anything!\n\n"
            "🎯 This approach eliminates recursion and reduces time from exponential to polynomial."
        )

    elif "compare the time complexity of recursive lcs vs. optimized lcs" in q:
        return (
            "⏱️ Let's compare the time complexity of two LCS methods:\n\n"
            "❌ Plain Recursive LCS:\n"
            "- Solves overlapping subproblems again and again.\n"
            "- Time complexity: O(2^n), which grows very fast.\n\n"
            "✅ Optimized LCS using Dynamic Programming:\n"
            "- Each subproblem solved once and stored.\n"
            "- Time complexity: O(m × n), where m and n are lengths of strings.\n\n"
            "🎯 Clearly, optimization makes LCS practical for larger inputs."
        )

    elif "optimization important in lcs when dealing with slightly longer strings" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract input strings."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"📈 Even slightly longer strings like '{s1}' and '{s2}' make plain recursion very slow due to repeated calculations.\n\n"
            "🔍 A problem that seems fast for 4-5 characters becomes extremely slow for 10+ characters.\n"
            "That’s exponential growth.\n\n"
            "✅ Optimization using memoization or tabulation allows the same problem to run much faster — by reducing repetition.\n\n"
            "🎯 For anything more than small examples, optimization is not optional — it's necessary!"
        )

    elif "key difference between an unoptimized recursive lcs and a memoized lcs" in q:
        return (
            "⚡ Let’s break down the key difference:\n\n"
            "❌ Unoptimized Recursive LCS:\n"
            "- Calls the same function for (i, j) multiple times.\n"
            "- Memory inefficient — doesn't store results.\n"
            "- Time complexity: O(2^n)\n\n"
            "✅ Memoized Recursive LCS:\n"
            "- Uses a dictionary or 2D array to cache results.\n"
            "- Each subproblem solved once.\n"
            "- Time complexity drops to O(m × n)\n\n"
            "🎯 Memoization turns brute-force recursion into efficient computation."
        )


# 🚀 Optimization Category – LCS Level 2 Answers

def answer_optimization_lvl2(q):
    import re
    q = q.lower()

    if "bottom-up dynamic programming" in q and "for the strings" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        str1, str2 = match.group(1), match.group(2)
        return (
            f"👨‍💻 The bottom-up approach to LCS uses tabulation (DP) to avoid redundant subproblems.\n"
            f"We fill a 2D DP table where each cell represents LCS length at that subproblem.\n"
            f"This method has time complexity O(m×n) and space complexity O(m×n).\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_bottom_up(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
            "    for i in range(1, m+1):\n"
            "        for j in range(1, n+1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                dp[i][j] = dp[i-1][j-1] + 1\n"
            "            else:\n"
            "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
            "    return dp[m][n]\n"
            "```\n\n"
            f"# Example:\nprint(lcs_bottom_up('{str1}', '{str2}'))\n\n"
            "🎯 This is the most stable and widely-used implementation for LCS."
        )

    elif "tabulation reduces the overhead" in q:
        return (
            "🧠 Tabulation avoids recursion by filling the DP table iteratively.\n"
            "It eliminates function call overhead and avoids stack overflow issues for large inputs.\n\n"
            "🔍 Compared to top-down recursion with memoization:\n"
            "- No recursive stack buildup\n"
            "- Better cache locality (memory-efficient)\n"
            "- Faster in practice for long strings\n\n"
            "✅ Use tabulation when input size is large or stack usage is a concern."
        )

    elif "use only two rows" in q or "optimize the lcs algorithm to use only two rows" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        a, b = match.group(1), match.group(2)
        return (
            f"📉 We can optimize LCS space usage by keeping just 2 rows instead of the entire 2D table.\n"
            f"This works because each cell only depends on the previous row.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_two_rows(a, b):\n"
            "    m, n = len(a), len(b)\n"
            "    prev = [0] * (n+1)\n"
            "    curr = [0] * (n+1)\n"
            "    for i in range(1, m+1):\n"
            "        for j in range(1, n+1):\n"
            "            if a[i-1] == b[j-1]:\n"
            "                curr[j] = prev[j-1] + 1\n"
            "            else:\n"
            "                curr[j] = max(prev[j], curr[j-1])\n"
            "        prev, curr = curr, [0]*(n+1)\n"
            "    return prev[n]\n"
            "```\n\n"
            f"# Example:\nprint(lcs_two_rows('{a}', '{b}'))\n\n"
            "🎯 Space reduced from O(m×n) to O(2×n). Useful for large strings."
        )

    elif "space and time trade-offs" in q and "recursive" in q:
        return (
            "📊 Trade-offs between recursive memoization and iterative tabulation:\n\n"
            "🧵 Recursive Memoization:\n"
            "- Easier to write\n"
            "- Uses call stack (risk of overflow)\n"
            "- May be slower due to function overhead\n"
            "- Good for sparse DP tables\n\n"
            "📋 Iterative Tabulation:\n"
            "- Uses loops, no stack\n"
            "- Usually faster and more memory-efficient\n"
            "- Fills entire DP table, even unused cells\n"
            "- Preferred for dense problems like full LCS\n\n"
            "🎯 Choose based on input size and recurrence pattern."
        )

    elif "convert a recursive lcs function into an iterative one" in q:
        return (
            "🔁 Recursive → Iterative conversion involves replacing function calls with nested loops.\n\n"
            "📚 Recursive:\n"
            "    lcs(i, j) = 1 + lcs(i-1, j-1) if match\n"
            "               max(lcs(i-1, j), lcs(i, j-1)) otherwise\n\n"
            "🛠 Replace with bottom-up table building:\n"
            "- Build a table dp[i][j] using loops\n"
            "- Initialize dp[0][*] and dp[*][0] = 0\n"
            "- Fill from smallest subproblems up\n\n"
            "🎯 Result: faster, no recursion, more control over memory."
        )


    elif "space optimization to solve the lcs problem" in q.lower() and "input strings" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        input1, input2 = match.group(1), match.group(2)
        return (
            f"📉 Space optimization for LCS involves using just two 1D arrays.\n"
            f"We update current row using values from the previous row.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_space_optimized(input1, input2):\n"
            "    m, n = len(input1), len(input2)\n"
            "    prev = [0]*(n+1)\n"
            "    curr = [0]*(n+1)\n"
            "    for i in range(1, m+1):\n"
            "        for j in range(1, n+1):\n"
            "            if input1[i-1] == input2[j-1]:\n"
            "                curr[j] = prev[j-1] + 1\n"
            "            else:\n"
            "                curr[j] = max(prev[j], curr[j-1])\n"
            "        prev, curr = curr, [0]*(n+1)\n"
            "    return prev[n]\n"
            "```\n\n"
            f"# Example:\nprint(lcs_space_optimized('{input1}', '{input2}'))\n\n"
            "🎯 Efficient for large input when full table is not needed."
        )

    elif "compare the space complexity" in q.lower():
        return (
            "📊 Space Comparison in LCS:\n\n"
            "🔷 Standard DP: O(m × n)\n"
            "  - Stores all subproblem results\n"
            "  - Useful if we need full traceback or table\n\n"
            "🔶 Space-Optimized: O(2 × n)\n"
            "  - Only previous and current row maintained\n"
            "  - Enough for computing LCS length\n\n"
            "🎯 Choose based on whether full sequence reconstruction is required."
        )

    elif "why does the space-optimized lcs algorithm still work" in q.lower():
        return (
            "🤔 Why it works despite reducing DP size:\n\n"
            "➡️ In LCS, dp[i][j] depends only on dp[i-1][j], dp[i][j-1], and dp[i-1][j-1].\n"
            "This means only previous row is required to compute current row.\n\n"
            "✅ No need to store full table\n"
            "✅ Perfectly valid for computing LCS length\n"
            "❌ But you cannot reconstruct the full sequence unless extra tracking is added."
        )

    elif "reduce memory usage for large strings" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract strings."
        long1, long2 = match.group(1), match.group(2)
        return (
            f"💾 To reduce memory usage with large strings like '{long1}' and '{long2}', use:\n"
            "- Space-optimized 1D DP (only two rows)\n"
            "- Avoid traceback unless needed\n"
            "- Consider streaming character-by-character\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_memory_reduced(s1, s2):\n"
            "    m, n = len(s1), len(s2)\n"
            "    if n > m:\n"
            "        s1, s2 = s2, s1\n"
            "        m, n = n, m\n"
            "    prev = [0] * (n+1)\n"
            "    curr = [0] * (n+1)\n"
            "    for i in range(1, m+1):\n"
            "        for j in range(1, n+1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                curr[j] = prev[j-1] + 1\n"
            "            else:\n"
            "                curr[j] = max(prev[j], curr[j-1])\n"
            "        prev, curr = curr, [0]*(n+1)\n"
            "    return prev[n]\n"
            "```\n\n"
            f"# Example:\nprint(lcs_memory_reduced('{long1}', '{long2}'))\n\n"
            "🎯 Works even with strings of length > 10⁵."
        )

    elif "modifications are needed to recover the actual lcs sequence from a space-optimized" in q.lower():
        return (
            "🔁 Recovering sequence from space-optimized LCS requires:\n\n"
            "1️⃣ Extra array to track choices (match / direction)\n"
            "2️⃣ Store more than just length (backpointers or traceback logic)\n"
            "3️⃣ Alternative: use 2D table for traceback, 1D for computing length\n\n"
            "✅ Trade memory for reconstructing sequence\n"
            "❌ Pure 1D array gives only LCS length, not path."
        )


    elif "lcs with both memoization and tabulation" in q.lower():
        return (
            "🧠 Combining Memoization and Tabulation?\n\n"
            "👉 Actually, they are alternative strategies:\n"
            "- Memoization (Top-Down): Uses recursion + cache\n"
            "- Tabulation (Bottom-Up): Uses iterative loops\n\n"
            "🔍 You can compare their runtime and memory by timing both versions:\n"
            "- Memoization may skip unnecessary cells\n"
            "- Tabulation computes full table but faster loop\n"
            "🎯 Benchmark both using time.time() and memory_profiler."
        )

    elif "reusable function for lcs that uses minimal space" in q.lower() and "input" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract stream inputs."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"🔁 This is a reusable space-optimized LCS function suitable for streaming or real-time systems.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_minimal_space(s1, s2):\n"
            "    prev, curr = [0]*(len(s2)+1), [0]*(len(s2)+1)\n"
            "    for i in range(1, len(s1)+1):\n"
            "        for j in range(1, len(s2)+1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                curr[j] = prev[j-1] + 1\n"
            "            else:\n"
            "                curr[j] = max(prev[j], curr[j-1])\n"
            "        prev, curr = curr, [0]*(len(s2)+1)\n"
            "    return prev[-1]\n"
            "```\n\n"
            f"# Example:\nprint(lcs_minimal_space('{s1}', '{s2}'))\n\n"
            "🎯 Perfect for embedded or live systems."
        )

    elif "comparing multiple strings to a common reference" in q.lower():
        return (
            "📈 If comparing multiple strings (S1, S2, ..., Sn) to a reference string R:\n\n"
            "🚀 Optimization Tips:\n"
            "- Precompute LCS of R with each string\n"
            "- Use space-optimized function\n"
            "- If many strings are similar, cache DP rows\n"
            "- Parallelize computations if independent\n\n"
            "🎯 This reduces repeated work and improves throughput."
        )

    elif "memoization preferred over tabulation" in q.lower():
        return (
            "🧠 Memoization is preferred over tabulation when:\n\n"
            "✔️ The DP table is sparse (not all entries needed)\n"
            "✔️ Problem has deep recursion but few unique subproblems\n"
            "✔️ You want quick prototyping\n\n"
            "❌ Avoid memoization if:\n"
            "- Input is large and causes stack overflow\n"
            "- Function call overhead slows down performance\n\n"
            "🎯 For full LCS, tabulation is usually better; for sparse or tree-like paths, memoization shines."
        )

    elif "limit memory usage while comparing large-scale input pairs from files" in q.lower():
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract file placeholders."
        fileA, fileB = match.group(1), match.group(2)
        return (
            f"🗂️ To limit memory while comparing large inputs from files '{fileA}' and '{fileB}':\n\n"
            "1️⃣ Stream file line-by-line if possible\n"
            "2️⃣ Use two-row DP version (space O(n))\n"
            "3️⃣ Avoid storing the entire LCS table\n"
            "4️⃣ Use generators or buffers if working with long sequences\n\n"
            "🛠 Python Skeleton:\n"
            "```python\n"
            "def lcs_file_memory_optimized(a, b):\n"
            "    prev, curr = [0]*(len(b)+1), [0]*(len(b)+1)\n"
            "    for i in range(1, len(a)+1):\n"
            "        for j in range(1, len(b)+1):\n"
            "            if a[i-1] == b[j-1]:\n"
            "                curr[j] = prev[j-1] + 1\n"
            "            else:\n"
            "                curr[j] = max(prev[j], curr[j-1])\n"
            "        prev, curr = curr, [0]*(len(b)+1)\n"
            "    return prev[-1]\n"
            "```\n\n"
            f"🎯 Combine this with file streaming for low memory LCS between {fileA} and {fileB}."
        )
# 🚀 Optimization Category – LCS Level 3 Answers

def answer_optimization_lvl3(q):
    import re
    q = q.lower()

    if "fully space-optimized lcs algorithm using only one row" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract text inputs."
        text1, text2 = match.group(1), match.group(2)
        return (
            f"💡 This is the most compact version of LCS — using only one row and a rolling variable.\n"
            f"Perfect when you only need the LCS length and must reduce memory to O(n).\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_one_row(t1, t2):\n"
            "    m, n = len(t1), len(t2)\n"
            "    if n > m:\n"
            "        t1, t2 = t2, t1\n"
            "        m, n = n, m\n"
            "    dp = [0] * (n + 1)\n"
            "    for i in range(1, m + 1):\n"
            "        prev = 0\n"
            "        for j in range(1, n + 1):\n"
            "            temp = dp[j]\n"
            "            if t1[i-1] == t2[j-1]:\n"
            "                dp[j] = prev + 1\n"
            "            else:\n"
            "                dp[j] = max(dp[j], dp[j-1])\n"
            "            prev = temp\n"
            "    return dp[n]\n"
            "```\n\n"
            f"# Example:\nprint(lcs_one_row('{text1}', '{text2}'))\n\n"
            "🎯 Minimal space, only 1 row and a variable."
        )

    elif "streaming environment with memory constraints" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract stream chunks."
        s1, s2 = match.group(1), match.group(2)
        return (
            f"📦 In streaming environments, you can’t load the full string into memory.\n"
            f"We use space-efficient rolling arrays and process in small chunks like '{s1}' and '{s2}'.\n\n"
            f"🛠 Skeleton Code for Streamed LCS:\n"
            "```python\n"
            "def lcs_streaming_chunked(stream1, stream2):\n"
            "    n = len(stream2)\n"
            "    dp = [0] * (n + 1)\n"
            "    for chunk in stream1:  # e.g., character-wise or buffered\n"
            "        prev = 0\n"
            "        for j in range(1, n + 1):\n"
            "            temp = dp[j]\n"
            "            if chunk == stream2[j - 1]:\n"
            "                dp[j] = prev + 1\n"
            "            else:\n"
            "                dp[j] = max(dp[j], dp[j - 1])\n"
            "            prev = temp\n"
            "    return dp[n]\n"
            "```\n\n"
            "🎯 Works for live input or memory-limited systems."
        )

    elif "scalable lcs solution optimized for extremely large sequences" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract genome data filenames."
        g1, g2 = match.group(1), match.group(2)
        return (
            f"🧬 For sequences like genome data from '{g1}' and '{g2}', use block-wise LCS and disk buffering.\n"
            f"This avoids loading entire sequences into RAM.\n\n"
            f"🛠 Strategy:\n"
            "1. Stream data from file line-by-line\n"
            "2. Use two-row DP buffer\n"
            "3. Optionally use disk-based NumPy array or memory-mapped files\n\n"
            f"🎯 This design scales for gigabyte-level genomic comparisons."
        )

    elif "return only the length of lcs without building the full dp table" in q:
        return (
            "📉 To compute only the LCS length with minimal space:\n\n"
            "✅ Use 1D array\n"
            "✅ Track previous values\n"
            "❌ Do not store backtracking information\n\n"
            "Result: O(n) space, O(mn) time\n\n"
            "🧠 Already implemented in the one-row approach shown earlier."
        )

    elif "lcs variant that works with three input strings" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract three input strings."
        s1, s2, s3 = match.group(1), match.group(2), match.group(3)
        return (
            f"🧠 LCS for 3 strings '{s1}', '{s2}', and '{s3}' needs a 3D table: dp[i][j][k].\n"
            f"For space optimization, use 2 layers instead of full cube.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def lcs_3strings_optimized(s1, s2, s3):\n"
            "    m, n, o = len(s1), len(s2), len(s3)\n"
            "    prev = [[0]*(o+1) for _ in range(n+1)]\n"
            "    curr = [[0]*(o+1) for _ in range(n+1)]\n"
            "    for i in range(1, m+1):\n"
            "        for j in range(1, n+1):\n"
            "            for k in range(1, o+1):\n"
            "                if s1[i-1] == s2[j-1] == s3[k-1]:\n"
            "                    curr[j][k] = prev[j-1][k-1] + 1\n"
            "                else:\n"
            "                    curr[j][k] = max(prev[j][k], curr[j-1][k], curr[j][k-1])\n"
            "        prev, curr = curr, [[0]*(o+1) for _ in range(n+1)]\n"
            "    return prev[n][o]\n"
            "```\n\n"
            f"# Example:\nprint(lcs_3strings_optimized('{s1}', '{s2}', '{s3}'))\n\n"
            "🎯 Memory reduced from O(m×n×o) to O(2×n×o)."
        )
    elif "parallelize the lcs computation" in q and "multi-core" in q:
        return (
            "⚙️ Parallelizing LCS is tricky due to data dependencies.\n\n"
            "🔹 But we can parallelize:\n"
            "1️⃣ Diagonal waves (wavefront parallelism)\n"
            "2️⃣ Independent block-wise LCS\n"
            "3️⃣ Large batch comparisons\n\n"
            "🧠 For correctness:\n"
            "- Diagonal cells can be updated in parallel\n"
            "- Rows and cols can't, due to dependency\n\n"
            "✅ Use OpenMP (C++), multiprocessing/threading (Python)\n"
            "🎯 Best used for very large strings or multi-document workloads."
        )

    elif "cache-aware version of lcs" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract large string names."
        big_str1, big_str2 = match.group(1), match.group(2)
        return (
            f"🧠 Cache-aware LCS improves locality when processing large strings like '{big_str1}' and '{big_str2}'.\n\n"
            "🔹 Techniques:\n"
            "- Row blocking / tiling (divide DP into blocks)\n"
            "- Loop unrolling for tight inner loops\n"
            "- Avoid cache thrashing via aligned memory\n\n"
            "🎯 These tricks reduce cache misses and boost throughput on CPU."
        )

    elif "integrate time and space optimizations" in q and "embedded systems" in q:
        return (
            "📦 For embedded systems, memory and CPU cycles are limited.\n\n"
            "🛠 Strategy:\n"
            "- Use 1D DP (space optimized)\n"
            "- Avoid recursion\n"
            "- Pre-allocate all arrays\n"
            "- Use C/embedded Python if possible\n\n"
            "✅ Keep RAM < 100KB\n"
            "✅ Constant memory footprint\n"
            "✅ Use integer-only computation\n\n"
            "🎯 Ideal for real-time LCS modules on microcontrollers."
        )

    elif "computing lcs between a small string and multiple large documents" in q:
        return (
            "📁 When comparing a short string against many large docs:\n\n"
            "🔹 Optimizations:\n"
            "1️⃣ Reuse same DP row for each doc\n"
            "2️⃣ Cache compiled pattern for short string\n"
            "3️⃣ Chunk large docs and reuse buffer\n"
            "4️⃣ Parallel scan with thread pool\n\n"
            "🎯 Example use case: Plagiarism checker or DNA pattern scanning."
        )

    elif "limits both memory and time to meet constraints of an online judge" in q:
        return (
            "⏱️ For online judges, you must optimize both space and runtime.\n\n"
            "📦 Best strategy:\n"
            "- Use two-row LCS (space O(n))\n"
            "- Tight inner loop\n"
            "- Avoid unnecessary prints / logging\n"
            "- Use fast I/O (e.g., sys.stdin.readline in Python)\n\n"
            "🎯 Example:\n"
            "```python\n"
            "import sys\n"
            "def fast_lcs(a, b):\n"
            "    prev, curr = [0]*(len(b)+1), [0]*(len(b)+1)\n"
            "    for i in range(1, len(a)+1):\n"
            "        for j in range(1, len(b)+1):\n"
            "            if a[i-1] == b[j-1]:\n"
            "                curr[j] = prev[j-1] + 1\n"
            "            else:\n"
            "                curr[j] = max(prev[j], curr[j-1])\n"
            "        prev, curr = curr, [0]*(len(b)+1)\n"
            "    return prev[-1]\n"
            "```\n"
            "✅ Passes memory/time limits for strings up to 10⁵."
        )
    elif "bit-parallelism" in q and "binary strings" in q:
            return (
                "⚡ Bit-parallel LCS optimization applies when input strings are binary and of limited alphabet size.\n\n"
                "🧠 Use bitwise operations instead of nested loops:\n"
                "- Encode presence of chars in bit vectors\n"
                "- Shift bits and mask updates in parallel\n"
                "- Used in Baeza-Yates-Gonnet LCS variant\n\n"
                "🎯 Ideal for alphabets ≤ 64 chars and SIMD-friendly environments."
            )

    elif "memory pooling to avoid repeated allocations" in q:
        return (
            "🧪 For batch comparisons, frequent allocations slow down performance.\n\n"
            "💡 Use memory pooling:\n"
            "- Pre-allocate arrays once\n"
            "- Reuse buffers for each pair of inputs\n"
            "- Avoids garbage collector or malloc overhead\n\n"
            "🎯 Use numpy arrays, memoryviews or pools in C/C++\n"
            "📦 Efficient for NLP, text diff, and classification tools."
        )

    elif "hybrid lcs approach" in q and "memoization and tabulation" in q:
        return (
            "🔁 Hybrid LCS selects between recursion (memoization) and iteration (tabulation) dynamically.\n\n"
            "🧠 Strategy:\n"
            "- If input size < threshold → recursion + memoization\n"
            "- If input size > threshold → tabulated DP\n\n"
            "🧪 Detect structure (e.g. one string mostly repeated)\n"
            "🎯 Build a wrapper function that benchmarks or heuristically selects."
        )

    elif "scale for datasets like" in q and "in linear memory" in q:
        match = re.search(r"['\"](.*?)['\"].*?['\"](.*?)['\"]", q)
        if not match:
            return "❌ Could not extract dataset names."
        ds1, ds2 = match.group(1), match.group(2)
        return (
            f"📊 To scale LCS for large datasets like '{ds1}' and '{ds2}' in linear memory:\n\n"
            "💡 Use Hirschberg’s Algorithm:\n"
            "- Splits problem recursively\n"
            "- Uses only O(n) space\n"
            "- Recovers actual LCS\n\n"
            "🧠 Time: O(m×n), Space: O(min(m,n))\n"
            "🎯 Gold standard for space-efficient sequence alignment."
        )

    elif "reduce both memory and recomputation when performing multiple lcs checks" in q:
        return (
            "🧠 When doing many LCS calls (e.g., comparing a query against multiple targets):\n\n"
            "🔁 Techniques:\n"
            "- Reuse DP rows\n"
            "- Batch pre-process common string\n"
            "- Parallelize across cores\n"
            "- Use trie/suffix tree for suffix-based reuse\n\n"
            "🎯 Example: autocomplete engines, bioinformatics sweeps, large scale search systems."
        )


def answer_app_lcs(level, question):
    if level == "Level 1":
        return answer_app_lvl1(question)
    elif level == "Level 2":
        return answer_app_lvl2(question)
    elif level == "Level 3":
        return answer_app_lvl3(question)
    else:
        return f"⚠️ Unsupported level: {level}"
    
def answer_impl_lcs(level, question):
    if level == "Level 1":
        return answer_implementation_lvl1(question)
    elif level == "Level 2":
        return answer_implementation_lvl2(question)
    elif level == "Level 3":
        return answer_implementation_lvl3(question)
    else:
        return f"⚠️ Unsupported level: {level}"

def answer_opt_lcs(level, question):
    if level == "Level 1":
        return answer_optimization_lvl1(question)
    elif level == "Level 2":
        return answer_optimization_lvl2(question)
    elif level == "Level 3":
        return answer_optimization_lvl3(question)
    else:
        return f"⚠️ Unsupported level: {level}"    

# Manual test block
if __name__ == "__main__":
    print("Test started")
    questionst= [
         "Write a recursive function to find the length of the LCS between 'ABCD' and 'ACBD'.",
        "Construct the DP table to compute the LCS length for 'ABC' and 'ACB'.",
        "How do you initialize the first row and column of the DP table when solving the LCS for 'AXY' and 'AYZ'?",
        "Describe how to update the cell dp[3][2] in the LCS DP table when characters do not match.",
        "Perform a dry run of the LCS bottom-up algorithm on strings 'AGGTAB' and 'GXTXAYB' and list the filled DP table.",
        "Write a function 'lcs_bottom_up' that computes the LCS length using bottom-up DP for two strings.",
        "What values will be stored in the first row of the DP table for the strings 'MNO' and 'NOP'?",
        "Given two strings 'HELLO' and 'WORLD', how many comparisons are needed to check each character recursively?",
        "Write a base recursive formula for LCS when 'C' at position 2 matches 'C' at position 1.",
        "Create a 2D table to calculate the LCS of 'DYNAMIC' and 'PROGRAM' with dimensions based on their lengths.",
        "What will the LCS length be for strings 'DOG' and 'CAT' if no characters match?",
        "Fill in the first few cells of a DP table for the strings 'AB' and 'AC'.",
        "Using a loop, iterate over each index of 'ABC' and compare with each index of 'ACB'. What would the time complexity be?",
        "If characters at positions 2 and 3 do not match, what values are compared in the LCS DP formula?",
        "Explain the purpose of checking both 'dp[i-1][j]' and 'dp[i][j-1]' in the LCS implementation.",
        "List the steps needed to compute the length of LCS using bottom-up dynamic programming for 'AGGT' and 'GXTX'.",
        "Which data structure is best suited to store the intermediate values while calculating LCS for 'HELLO' and 'WORLD'?",
        "Describe the loop structure needed to fill the DP table for 'DYNAMIC' and 'PROGRAM' from bottom up.",
        "What would the final value in dp[7][7] represent in the LCS table of 'DYNAMIC' and 'PROGRAM'?",
        "Write pseudocode for computing LCS using a nested for-loop and a DP table for strings 'ABCDEF' and 'FBDAMN'."


    ]
    # question1 = "Perform a dry run of the LCS bottom-up algorithm on strings 'ACD' and 'ABC'."
    # answer = answer_algo_lvl1(question1)
    # print(answer)
#     for question in questionst:
#         answer = answer_algo_lvl1(question)
#         print(f"Question: {question}\nAnswer: {answer}\n")
# if __name__ == "__main__":
#     print("Test started")

#     question6 = "Construct the DP table to compute the LCS length for 'Soham' and 'oham'."
#     answer = answer_algo_lvl1(question6)
#     print(answer)

# if __name__ == "__main__":
#     print("\n--- Testing Level 2 Algorithmic Answers ---\n")
#     questions_lvl2 = [
#         "Implement a memoized recursive function to compute the LCS length for 'ABCD' and 'ACBD'.",
#         "Explain how memoization improves the naive recursive LCS solution for inputs like 'ABCD' and 'ACBD'.",
#         "Construct a function that builds and returns the LCS string (not just its length) for 'ABC' and 'ACB'.",
#         "Convert a top-down LCS solution for 'ABC' and 'ACB' to an equivalent bottom-up implementation.",
#         "What is the space complexity of your bottom-up LCS algorithm for 'ABC' and 'ACB', and how can it be optimized?",
#         "Modify your DP algorithm to reconstruct all possible LCS strings between 'AXY' and 'AYZ'.",
#         "Write code to fill the LCS DP table row by row using minimal space for 'AGGTAB' and 'GXTXAYB'.",
#         "Use a single-dimensional array to compute the LCS length between 'AGGTAB' and 'GXTXAYB'. Describe your approach.",
#         "Write a Python function that returns both the LCS length and one valid subsequence for 'ABC' and 'ACB'.",
#         "Identify duplicate subproblems in the recursion tree of LCS for 'HELLO' and 'WORLD', and explain how memoization helps.",
#         "Explain how the size of the DP table depends on the lengths of 'HELLO' and 'WORLD', and compute the number of cells.",
#         "Write a function to find the LCS length using memoization and print the memo table after processing 'HELLO' and 'WORLD'.",
#         "Explain the significance of diagonal movement in the DP table when characters match in 'AXY' and 'AYZ'.",
#         "Given a correct LCS length, backtrack through the DP table to print the actual LCS for 'ABCDEF' and 'FBDAMN'.",
#         "What will happen in the LCS logic if one of the strings is empty, like 'ABC' and ''?",
#         "Modify the LCS algorithm to ignore spaces and punctuation in 'A,B.C!' and 'ABC'.",
#         "Find the minimum number of insertions and deletions needed to convert 'ABC' into 'ACB' using LCS length.",
#         "Print the full DP table after computing LCS between 'AGGTAB' and 'GXTXAYB', and highlight the LCS path.",
#         "Break down the time and space complexity of your LCS implementation for 'HELLO' and 'WORLD'."
#     ]

#     for question in questions_lvl2:
#         answer = answer_algo_lvl2(question)
#         print(f"Q: {question}\nA: {answer}\n{'-'*80}\n")


if __name__ == "__main__":
    print("\n--- Testing Level 3 Algorithmic Answers (Q1–Q15) ---\n")

    questions_lvl3 = [
        "Reconstruct all possible LCS sequences for 'ABCBDAB' and 'BDCAB' by backtracking the DP table. What challenges arise?",
        "Modify the standard LCS algorithm to handle wildcard characters like '?' in 'AB?D' or 'A?BD' that can match any character.",
        "Write a function that returns the number of distinct LCS sequences for strings 'AGGTAB' and 'GXTXAYB'.",
        "Compare recursive, memoized, bottom-up, and space-optimized LCS approaches on inputs like 'ABC' and 'ACB' in terms of performance.",
        "Determine the minimum number of insertions and deletions to convert 'ABC' into 'ACB' using their LCS.",
        "How can you parallelize the filling of the LCS DP table for 'AGGTAB' and 'GXTXAYB'? Discuss data dependencies.",
        "Build a function that returns the longest common prefix, suffix, and subsequence between 'PRETEST' and 'PREFIX'.",
        "Write an iterative function that generates the LCS table diagonally for 'AGGTAB' and 'GXTXAYB'. What are the pros and cons?",
        "Analyze how the LCS length changes when characters in 'ABCDE' or 'AECBD' are shuffled. Can it be predicted?",
        "Implement a memory-efficient LCS algorithm for 'ABCDEF' and 'FBDAMN' using only two rows of the DP table.",
        "Adapt the LCS algorithm to handle three strings: 'abcd', 'bcda', and 'cdab'. Describe the changes in approach.",
        "Create test cases to validate your LCS function for edge cases like empty strings, repeats, and identical inputs.",
        "Design a recursive-free LCS algorithm that uses iterative loops and constant space for 'ABC' and 'ACB'.",
        "Modify the LCS logic to allow skipping one unmatched character from either 'ABC' or 'ACB'. Implement this variant.",
        "Find the longest common alternating subsequence (e.g., a-b-a) between 'abacab' and 'acabab'. How does it differ from standard LCS?"
    ]

    for i, question in enumerate(questions_lvl3, 1):
        print(f"Q{i}: {question}")
        try:
            answer = answer_algo_lvl3(question)
        except Exception as e:
            answer = f"❌ Error occurred: {str(e)}"
        print(f"A{i}: {answer}\n{'-'*80}\n")

# sample_questions = [
#     "Initialize a 2D DP table for LCS computation for strings 'abc' and 'acd'.",
#     "Implement the base case for a recursive LCS function with strings 'a' and 'b'.",
#     "Create a function to fill only the first row and first column of an LCS table for 'abc' and 'def'.",
#     "Write a loop that compares each character of 'abc' with 'bcd' and prints matching pairs.",
#     "Print the DP table after partially computing the LCS table for 'abc' and 'abd'.",
#     "Create a 2D matrix of size (3+1) x (2+1) initialized with zeros.",
#     "Write code to print each cell value of a small LCS DP table for 'a' and 'b'.",
#     "Write the recursive LCS function for base case when one of 'ab' or 'cd' is empty.",
#     "Build a loop to iterate through characters of 'abc' to prepare for LCS matching.",
#     "Implement a simple 2D array print function for displaying the LCS table.",
#     "Write a function that returns the max between two values as used in LCS logic.",
#     "Write an if-else block to handle when characters at position i and j are equal in 'abc' and 'adc'.",
#     "Implement a memoization table using a 2D list initialized with -1 for lengths 3 and 4.",
#     "Write a nested loop to fill values in the DP table using the LCS recurrence relation.",
#     "Create a loop that compares characters at i and j for strings 'abc' and 'abc'.",
#     "Implement a function to return the length of LCS using a top-down memoization approach for 'abc' and 'abd'.",
#     "Fill a DP table up to index [i][j] based on character matches between 'abc' and 'abc'.",
#     "Use nested loops to simulate LCS table filling without using actual strings — only size 3x4."
# ]

# # ✅ Test loop
# for idx, q in enumerate(sample_questions, 1):
#     print(f"\n🔹 Test {idx}: {q}")
#     print(answer_implementation_lvl1(q))


# ✅ Manual test block for Level 2 implementation questions
# if __name__ == "__main__":
#     print("🚀 Level 2 LCS Implementation Tests Starting...\n")
    
#     test_questions_lvl2 = [
#         "Write a recursive LCS function with memoization for strings 'AGGTAB' and 'GXTXAYB'.",
#         "Create a class with a method to compute LCS of 'ABCBDAB' and 'BDCAB' using a memoized approach.",
#         "Fill a 2D DP table for strings 'ABCDEF' and 'FBDAMN' and return both the table and LCS length.",
#         "Implement a function to print the filled LCS table for 'DYNAMIC' and 'PROGRAM' after computing it.",
#         "Write code to extract and return the LCS string (not just length) from the DP table for 'HELLO' and 'YELLOW'.",
#         "Build the LCS table using nested loops and track the operations for each cell based on character comparisons in 'ABCDGH' and 'AEDFHR'.",
#         "Write code to handle input strings 'AXYT' and 'AYZX' and return the LCS using top-down memoization.",
#         "Implement the LCS function and print the decision taken (match or max) for each cell in the DP table for 'ABC' and 'ACB'.",
#         "Given a function stub, complete the logic to compute LCS recursively for 'DOG' and 'CAT'.",
#         "Add memoization to an existing recursive LCS function to improve performance for 'ABCDE' and 'AECBD'.",
#         "Implement a version of the LCS algorithm that stores intermediate results in a dictionary instead of a 2D list for 'ABCD' and 'ACBAD'.",
#         "Write a function that takes strings 'GEEKSFORGEEKS' and 'GEEKSQUIZ' and prints both the LCS length and LCS string.",
#         "Complete a partially implemented LCS bottom-up function with a loop for DP table computation using 'XMJYAUZ' and 'MZJAWXU'.",
#         "Optimize an existing LCS table-filling loop to use only two rows of memory instead of the full DP table for 'ABC' and 'DEF'.",
#         "Create a helper function to trace back the LCS string from the DP table filled for 'ABAZDC' and 'BACBAD'.",
#         "Write the full function to return both LCS length and the sequence for input strings 'ABCDEF' and 'FBDAMN'.",
#         "Implement a function to check and return whether the computed LCS string is a subsequence of both 'HELLO' and 'WORLD'.",
#         "Given a filled DP table for 'ABCD' and 'ABED', write code to trace and return one valid LCS path.",
#         "Design a wrapper function that reads strings 'LCSALGORITHM' and 'DYNAMICPROGRAMMING', computes LCS length, and returns the result."
#     ]

    # # Run each test question
    # for idx, question in enumerate(test_questions_lvl2, start=1):
    #     print(f"🧪 Test {idx}: {question}")
    #     try:
    #         answer = answer_implementation_lvl2(question)
    #         print("✅ Answer:\n", answer)
    #     except Exception as e:
    #         print("❌ Error:", e)
    #     print("\n" + "-"*80 + "\n")


# # ✅ Level 3 LCS Testing Block
# def test_lcs_level3_answers():
#     questions = [
#         "Implement space-optimized LCS for 'ABCDEF' and 'AEBDF'.",
#         "Write a function to return all possible LCS sequences for 'ABCBDAB' and 'BDCAB'.",
#         "Modify the standard LCS algorithm to also return the indices of matching characters in 'XMJYAUZ' and 'MZJAWXU'.",
#         "Extend the LCS implementation to count the number of distinct LCS sequences between 'AGTGATG' and 'GTTAG'.",
#         "Implement LCS with traceback and visualize the path in the DP table for 'ABCBDAB' and 'BDCAB'.",
#         "Write an LCS function that avoids recursion and uses an iterative tabular approach with backtracking for 'ABCDEF' and 'AEDFHR'.",
#         "Implement a generic LCS method that can handle a list of strings instead of just two (e.g., 'AGGT12', '12TXAYB', and '12XBA').",
#         "Design a custom data structure to efficiently store intermediate states during LCS computation between 'ABCBDAB' and 'BDCAB'.",
#         "Implement LCS computation and store both LCS and edit distance results for 'ABCDEF' and 'FBDAMN' in a single pass.",
#         "Write code to generate a diff-like output between 'ABCDEF' and 'ABDF'.",
#         "Convert a bottom-up LCS algorithm into a generator function that yields results step-by-step for 'AXYT' and 'AYZX'.",
#         "Write a multithreaded implementation to fill the LCS DP table in parallel for large strings like 'AGGTAB' and 'GXTXAYB'.",
#         "Implement memoization for LCS using a custom cache strategy instead of built-in decorators for 'ABCD' and 'ACBAD'.",
#         "Implement a modified LCS that ignores case while comparing characters in 'AbCdEf' and 'abcdef'.",
#         "Write a wrapper that computes LCS only if 'HELLO' and 'WORLD' are non-empty and consist of uppercase letters only.",
#         "Write test cases to validate your LCS implementation for various edge cases, including empty strings and identical inputs like 'MANGO'.",
#         "Implement the LCS algorithm using functional programming style for strings 'ABCBDAB' and 'BDCAB'.",
#         "Build a CLI tool to accept 'XMJYAUZ' and 'MZJAWXU' and print their LCS along with the computation table.",
#         "Create an LCS utility that caches previously computed string pairs like 'ABCDEF' and 'ABDF' to improve performance.",
#         "Implement a web API endpoint to accept 'ABC' and 'AC' as query params and return the LCS string and length."
#     ]

#     print("🧪 Running Level 3 LCS Tests:\n")
#     for i, q in enumerate(questions, start=1):
#         print(f"\n🔹 Test {i}: {q}")
#         try:
#             response = answer_implementation_lvl3(q)
#             print(response)
#         except Exception as e:
#             print(f"❌ Error in test {i}: {e}")

# # Example call
# if __name__ == "__main__":
#     test_lcs_level3_answers()


# ✅ Level 2 Optimization LCS Testing Function
# def test_lcs_optimization_lvl2():
#     questions = [
#         "Implement the LCS algorithm using bottom-up dynamic programming for the strings 'ABCBDAB' and 'BDCAB'.",
#         "Explain how tabulation reduces the overhead in LCS computation compared to top-down recursion with memoization.",
#         "Optimize the LCS algorithm to use only two rows of the DP table instead of a full 2D table for strings 'AGGTAB' and 'GXTXAYB'.",
#         "What are the space and time trade-offs when choosing between recursive memoization and iterative tabulation in LCS?",
#         "How can you convert a recursive LCS function into an iterative one to reduce function call overhead?",
#         "Use space optimization to solve the LCS problem for two input strings 'AXYT' and 'AYZX' efficiently.",
#         "Compare the space complexity of a standard LCS DP table with a space-optimized implementation.",
#         "Why does the space-optimized LCS algorithm still work correctly despite reducing the DP table size?",
#         "Optimize the LCS algorithm to reduce memory usage for large strings like 'A' * 10000 and 'A' * 9000.",
#         "What modifications are needed to recover the actual LCS sequence from a space-optimized DP solution?",
#         "Use LCS with both memoization and tabulation, then compare the runtime and memory usage.",
#         "Create a reusable function for LCS that uses minimal space and can be called in real-time systems with input 'HELLO' and 'WORLD'.",
#         "When comparing multiple strings to a common reference, how can you optimize repeated LCS calculations?",
#         "In what cases is memoization preferred over tabulation for solving LCS problems?",
#         "Modify an LCS solution to limit memory usage while comparing large-scale input pairs from files 'fileA.txt' and 'fileB.txt'."
#     ]

#     print("🧪 Running LCS Level 2 Optimization Tests:")
#     for i, q in enumerate(questions, start=1):
#         print(f"\n🔹 Test {i}: {q}")
#         try:
#             response = answer_optimization_lvl2(q)
#             print(response)
#         except Exception as e:
#             print(f"❌ Error in test {i}: {e}")

# # Run the test block
# if __name__ == "__main__":
#     test_lcs_optimization_lvl2()

# # ✅ Level 3 Optimization LCS Testing Function
# def test_lcs_optimization_lvl3():
#     questions = [
#         "Implement a fully space-optimized LCS algorithm using only one row and a previous value tracker for inputs 'ABCDEF' and 'AEBDF'.",
#         "How can LCS be computed in a streaming environment with memory constraints for inputs 'A', 'B'?",
#         "Design a scalable LCS solution optimized for extremely large sequences like genome data 'genomeA.txt' and 'genomeB.txt' stored in files.",
#         "Modify the standard LCS algorithm to return only the length of LCS without building the full DP table, using minimal space.",
#         "Apply space optimization techniques to an LCS variant that works with three input strings: 'ABC', 'BDC', and 'BAC'.",
#         "How can you parallelize the LCS computation while preserving correctness and optimizing time on multi-core systems?",
#         "Design a cache-aware version of LCS that minimizes cache misses when processing large strings 'BIGSTRING1' and 'BIGSTRING2'.",
#         "Integrate time and space optimizations to create an LCS library function usable in resource-constrained embedded systems.",
#         "When computing LCS between a small string and multiple large documents, how can pre-processing and memory reuse improve performance?",
#         "Create an LCS solution that limits both memory and time to meet the constraints of an online judge with large input limits.",
#         "Explain how bit-parallelism could be applied to optimize the LCS computation for alphabet-limited binary strings.",
#         "Build an LCS solution that uses memory pooling to avoid repeated allocations during batch comparisons of text samples.",
#         "Implement a hybrid LCS approach that chooses between memoization and tabulation based on input size and pattern structure.",
#         "What algorithmic improvements allow LCS computation to scale for datasets like 'datasetA' and 'datasetB' in linear memory?",
#         "Describe the techniques used to reduce both memory and recomputation when performing multiple LCS checks in succession."
#     ]

#     print("🧪 Running Level 3 Optimization LCS Tests:")
#     for i, q in enumerate(questions, start=1):
#         print(f"\n🔹 Test {i}: {q}")
#         try:
#             response = answer_optimization_lvl3(q)
#             print(response)
#         except Exception as e:
#             print(f"❌ Error in test {i}: {e}")

# # ✅ Call the test function
# if __name__ == "__main__":
#     test_lcs_optimization_lvl3()


# def test_lcs_optimization_lvl1():
#     questions = [
#         "Why is it inefficient to solve LCS using plain recursion for input strings like '{{string1}}' and '{{string2}}'?",
#         "How can we avoid recalculating the same subproblems in an LCS recursive solution?",
#         "What is memoization, and how can it help optimize a naive LCS recursive implementation?",
#         "Explain how dynamic programming improves the performance of LCS calculation for inputs like '{{s1}}' and '{{s2}}'.",
#         "What kind of problem pattern makes LCS suitable for optimization using dynamic programming?",
#         "Why do overlapping subproblems in LCS suggest that optimization is needed?",
#         "How does tabulation help in optimizing LCS, and what are its basic steps?",
#         "Compare the time complexity of recursive LCS vs. optimized LCS using DP.",
#         "Why is optimization important in LCS when dealing with slightly longer strings like '{{longer_str1}}' and '{{longer_str2}}'?",
#         "What is the key difference between an unoptimized recursive LCS and a memoized LCS in terms of performance?"
#     ]

#     print("🧪 Running Level 1 Optimization LCS Tests:")
#     for i, q in enumerate(questions, start=1):
#         print(f"\n🔹 Test {i}: {q}")
#         try:
#             response = answer_optimization_lvl1(q)
#             print(response)
#         except Exception as e:
#             print(f"❌ Error in test {i}: {e}")

# # ✅ Call the test function
# if __name__ == "__main__":
#     test_lcs_optimization_lvl1()
