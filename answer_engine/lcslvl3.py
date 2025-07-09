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
