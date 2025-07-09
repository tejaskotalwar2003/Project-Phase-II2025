import re
from functools import lru_cache
from mcm_answer import print_optimal_parenthesization


# --- HELPER FUNCTIONS (START) ---
# These are general helper functions that might be used by the answer generation logic.

# Simple iterative Fibonacci (for comparison/demonstration)
def fib_iterative_basic(n):
    if n < 0: return None
    if n <= 1: return n
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# Simple recursive Fibonacci (unmemoized, for comparison)
def fib_recursive_basic(n):
    if n < 0: return None
    if n <= 1: return n
    return fib_recursive_basic(n - 1) + fib_recursive_basic(n - 2)

# Factorial function (for demonstration)
def factorial_basic(n):
    if n < 0: return None
    if n == 0: return 1
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res

# --- HELPER FUNCTIONS (END) ---


# --- ANSWER GENERATION FUNCTION FOR MEMOIZATION APPLICATION LEVEL 1 (START) ---

def answer_memoization_application_lvl1(question):
    q = question.lower()

    # 1. 
    if "fibonacci number using memoization" in q:
        import re
        match = re.search(r"compute the (\d+)(st|nd|rd|th)? fibonacci", q)
        n = int(match.group(1)) if match else 10

        def fib(n, memo={}):
            if n in memo:
                return memo[n]
            if n <= 1:
                return n
            memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
            return memo[n]

        result = fib(n)

        return (
            f"🧠 **Problem:** Compute the {n}th Fibonacci number efficiently using memoization.\n\n"
            f"💡 **Why memoization?**\n"
            f"Without memoization, Fibonacci has exponential time due to repeated calls. Memoization caches results of previous calls.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"def fib(n, memo={{}}):\n"
            f"    if n in memo:\n"
            f"        return memo[n]\n"
            f"    if n <= 1:\n"
            f"        return n\n"
            f"    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)\n"
            f"    return memo[n]\n\n"
            f"result = fib({n})\n"
            f"print('Fibonacci({n}) =', result)\n"
            f"```\n"
            f"📊 **Computed Output:** `Fibonacci({n}) = {result}`\n"
            f"✅ **Time Complexity:** O(n)\n"
            f"✅ **Space Complexity:** O(n)"
        )

# 2.
    elif "factorial" in q and "memoization" in q:
        import re
        match = re.search(r"factorial of (\d+)", q)
        n = int(match.group(1)) if match else 6

        def fact(n, memo={}):
            if n in memo:
                return memo[n]
            if n <= 1:
                return 1
            memo[n] = n * fact(n - 1, memo)
            return memo[n]

        result = fact(n)

        return (
            f"🧠 **Problem:** Compute factorial of {n} using memoization.\n\n"
            f"💡 Avoids recalculating intermediate factorial values.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"def fact(n, memo={{}}):\n"
            f"    if n in memo:\n"
            f"        return memo[n]\n"
            f"    if n <= 1:\n"
            f"        return 1\n"
            f"    memo[n] = n * fact(n - 1, memo)\n"
            f"    return memo[n]\n\n"
            f"print(fact({n}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n), Space: O(n)"
        )



    # 3. 
    elif "climb" in q and "memoization" in q:
        import re
        match = re.search(r"climb (\d+) stairs", q)
        n = int(match.group(1)) if match else 5

        def climb(n, memo={}):
            if n in memo:
                return memo[n]
            if n <= 2:
                return n
            memo[n] = climb(n - 1, memo) + climb(n - 2, memo)
            return memo[n]

        result = climb(n)

        return (
            f"🧠 **Problem:** Count ways to climb {n} stairs (1 or 2 steps) using memoization.\n\n"
            f"💡 Caches intermediate stair counts.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"def climb(n, memo={{}}):\n"
            f"    if n in memo:\n"
            f"        return memo[n]\n"
            f"    if n <= 2:\n"
            f"        return n\n"
            f"    memo[n] = climb(n - 1, memo) + climb(n - 2, memo)\n"
            f"    return memo[n]\n\n"
            f"print(climb({n}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n), Space: O(n)"
        )



    # 4. 
    elif "nth value of the" in q and "sequence" in q:
        import re
        match_n = re.search(r"number (\d+)", q)
        match_seq = re.search(r"of the (.*?) sequence", q)
        n = int(match_n.group(1)) if match_n else 6
        seq_name = match_seq.group(1).strip().title() if match_seq else "Tribonacci"

        def trib(n, memo={}):
            if n in memo:
                return memo[n]
            if n == 0:
                return 0
            if n == 1 or n == 2:
                return 1
            memo[n] = trib(n - 1, memo) + trib(n - 2, memo) + trib(n - 3, memo)
            return memo[n]

        result = trib(n)

        return (
            f"🧠 **Problem:** Compute the {n}th term of the {seq_name} sequence using memoization.\n\n"
            f"💡 Memoization avoids recomputation of overlapping subproblems.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"def trib(n, memo={{}}):\n"
            f"    if n in memo:\n"
            f"        return memo[n]\n"
            f"    if n == 0:\n"
            f"        return 0\n"
            f"    if n == 1 or n == 2:\n"
            f"        return 1\n"
            f"    memo[n] = trib(n - 1, memo) + trib(n - 2, memo) + trib(n - 3, memo)\n"
            f"    return memo[n]\n\n"
            f"print(trib({n}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n), Space: O(n)"
        )



    # 5. 
    elif "reach the end of an array" in q:
        import re
        match = re.search(r"array of length (\d+)", q)
        n = int(match.group(1)) if match else 7

        def jumps(n, memo={}):
            if n in memo:
                return memo[n]
            if n <= 1:
                return 1
            memo[n] = jumps(n - 1, memo) + jumps(n - 2, memo)
            return memo[n]

        result = jumps(n)

        return (
            f"🧠 **Problem:** Count ways to reach the end of an array of length {n} using 1 or 2 steps.\n\n"
            f"💡 Use memoization to avoid recalculating repeated subpaths.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"def jumps(n, memo={{}}):\n"
            f"    if n in memo:\n"
            f"        return memo[n]\n"
            f"    if n <= 1:\n"
            f"        return 1\n"
            f"    memo[n] = jumps(n - 1, memo) + jumps(n - 2, memo)\n"
            f"    return memo[n]\n\n"
            f"print(jumps({n}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n), Space: O(n)"
        )



    # 6. Modify a recursive function for climbing {{n}} stairs to use memoization.
    elif "convert the recursive solution for finding the nth term of" in q:
        import re
        match_n = re.search(r"nth term of (.*?) into", q)
        match_idx = re.search(r"finding the (\d+)(st|nd|rd|th)?", q)
        seq_name = match_n.group(1).strip() if match_n else "custom sequence"
        n = int(match_idx.group(1)) if match_idx else 6

        def custom_seq(n, memo={}):
            if n in memo:
                return memo[n]
            if n <= 1:
                return 1
            memo[n] = custom_seq(n - 1, memo) + 2 * custom_seq(n - 2, memo)
            return memo[n]

        result = custom_seq(n)

        return (
            f"🧠 **Problem:** Convert a recursive solution to compute the {n}th term of `{seq_name}` into a memoized version.\n\n"
            f"💡 **Why memoization?** Recursive calls recompute overlapping subproblems. Memoization caches intermediate results.\n\n"
            f"🔧 **Memoized Example Code:**\n"
            f"```python\n"
            f"def custom_seq(n, memo={{}}):\n"
            f"    if n in memo:\n"
            f"        return memo[n]\n"
            f"    if n <= 1:\n"
            f"        return 1\n"
            f"    memo[n] = custom_seq(n - 1, memo) + 2 * custom_seq(n - 2, memo)\n"
            f"    return memo[n]\n\n"
            f"print(custom_seq({n}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n), Space: O(n)"
        )


    # 7. Use a dictionary to cache the output of {{function_name}} calls.
    elif "recursive function that checks if" in q and "apply memoization" in q:
        import re
        match_prop = re.search(r"checks if (.*?),", q)
        number_property = match_prop.group(1).strip() if match_prop else "a number is even"

        memo = {}
        def check_property(n):
            if n in memo:
                return memo[n]
            if number_property == "a number is even":
                result = (n % 2 == 0)
            elif number_property == "a number is prime":
                result = n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))
            elif number_property == "a number is a power of 2":
                result = (n & (n - 1) == 0 and n > 0)
            else:
                result = False
            memo[n] = result
            return result

        sample_input = 16
        result = check_property(sample_input)

        return (
            f"🧠 **Problem:** Apply memoization to a recursive function that checks whether `{number_property}`.\n\n"
            f"💡 **Why memoization?** If the function is called repeatedly with the same input, caching saves computation.\n\n"
            f"🔧 **Example Memoized Function:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def check_property(n):\n"
            f"    if n in memo:\n"
            f"        return memo[n]\n"
            f"    # Example: check if n is power of 2\n"
            f"    result = (n & (n - 1) == 0 and n > 0)\n"
            f"    memo[n] = result\n"
            f"    return result\n\n"
            f"print(check_property({sample_input}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(1) per call after caching\n"
            f"✅ Space: O(N) for memo dictionary"
        )


    # 8. Transform a recursive function for calculating the {{term_name}} into a memoized version.
    elif "check if the string" in q and "is a palindrome" in q:
        import re
        match = re.search(r"the string '(.+?)'", q)
        s = match.group(1) if match else "madam"

        memo = {}
        def is_palindrome(l, r):
            if (l, r) in memo:
                return memo[(l, r)]
            if l >= r:
                return True
            if s[l] != s[r]:
                memo[(l, r)] = False
                return False
            memo[(l, r)] = is_palindrome(l + 1, r - 1)
            return memo[(l, r)]

        result = is_palindrome(0, len(s) - 1)

        return (
            f"🧠 **Problem:** Write a memoized function to check if the string '{s}' is a palindrome.\n\n"
            f"💡 Uses recursion with two pointers and memoizes (l, r) results to avoid redundant substring checks.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def is_palindrome(l, r):\n"
            f"    if (l, r) in memo:\n"
            f"        return memo[(l, r)]\n"
            f"    if l >= r:\n"
            f"        return True\n"
            f"    if s[l] != s[r]:\n"
            f"        memo[(l, r)] = False\n"
            f"        return False\n"
            f"    memo[(l, r)] = is_palindrome(l + 1, r - 1)\n"
            f"    return memo[(l, r)]\n\n"
            f"s = '{s}'\n"
            f"print(is_palindrome(0, len(s) - 1))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n²), Space: O(n²) for memoization"
        )


    # 9. Write a top-down memoized function for the {{series_name}} sequence.
    elif "calculating" in q and "recursively" in q:
        import re
        match = re.search(r"calculating (\w+)\s+recursively", q)
        func_name = match.group(1) if match else "power"

        def power(x, n, memo={}):
            if (x, n) in memo:
                return memo[(x, n)]
            if n == 0:
                return 1
            if n % 2 == 0:
                half = power(x, n // 2, memo)
                memo[(x, n)] = half * half
            else:
                memo[(x, n)] = x * power(x, n - 1, memo)
            return memo[(x, n)]

        x, n = 2, 10
        result = power(x, n)

        return (
            f"🧠 **Problem:** Use a dictionary to memoize the recursive calculation of `{func_name}`.\n\n"
            f"💡 For example, memoizing `power(x, n)` saves exponential recomputation of powers.\n\n"
            f"🔧 **Memoized Code Example:**\n"
            f"```python\n"
            f"def power(x, n, memo={{}}):\n"
            f"    if (x, n) in memo:\n"
            f"        return memo[(x, n)]\n"
            f"    if n == 0:\n"
            f"        return 1\n"
            f"    if n % 2 == 0:\n"
            f"        half = power(x, n // 2, memo)\n"
            f"        memo[(x, n)] = half * half\n"
            f"    else:\n"
            f"        memo[(x, n)] = x * power(x, n - 1, memo)\n"
            f"    return memo[(x, n)]\n\n"
            f"print(power({x}, {n}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(log n), Space: O(log n)"
        )


    # 10. Create a memoized wrapper function for computing factorial values.
    elif "binomial coefficient" in q or "computes" in q and "ncr" in q.lower():
        import re
        match_n = re.search(r"\(?(\d+)[cC](\d+)\)?", q)
        n = int(match_n.group(1)) if match_n else 6
        r = int(match_n.group(2)) if match_n else 2

        memo = {}
        def nCr(n, r):
            if r == 0 or r == n:
                return 1
            if (n, r) in memo:
                return memo[(n, r)]
            memo[(n, r)] = nCr(n - 1, r - 1) + nCr(n - 1, r)
            return memo[(n, r)]

        result = nCr(n, r)

        return (
            f"🧠 **Problem:** Use memoization to compute binomial coefficient {n}C{r}.\n\n"
            f"💡 Recursive formula: `C(n, r) = C(n-1, r-1) + C(n-1, r)` with base cases `C(n, 0) = C(n, n) = 1`.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def nCr(n, r):\n"
            f"    if r == 0 or r == n:\n"
            f"        return 1\n"
            f"    if (n, r) in memo:\n"
            f"        return memo[(n, r)]\n"
            f"    memo[(n, r)] = nCr(n - 1, r - 1) + nCr(n - 1, r)\n"
            f"    return memo[(n, r)]\n\n"
            f"print(nCr({n}, {r}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n*r), Space: O(n*r)"
        )


    # 11. Use memoization to return the nth term of the {{sequence_type}} using base cases.
    elif "number of paths from the top-left to bottom-right" in q and "grid" in q:
        import re
        match = re.search(r"(\d+)[x×](\d+)", q)
        m = int(match.group(1)) if match else 3
        n = int(match.group(2)) if match else 3

        memo = {}
        def unique_paths(i, j):
            if i == m - 1 or j == n - 1:
                return 1
            if (i, j) in memo:
                return memo[(i, j)]
            memo[(i, j)] = unique_paths(i + 1, j) + unique_paths(i, j + 1)
            return memo[(i, j)]

        result = unique_paths(0, 0)

        return (
            f"🧠 **Problem:** Compute the number of unique paths in a {m}x{n} grid from top-left to bottom-right.\n\n"
            f"💡 Use memoization to cache overlapping recursive calls for each (i, j) cell.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def unique_paths(i, j):\n"
            f"    if i == {m - 1} or j == {n - 1}:\n"
            f"        return 1\n"
            f"    if (i, j) in memo:\n"
            f"        return memo[(i, j)]\n"
            f"    memo[(i, j)] = unique_paths(i + 1, j) + unique_paths(i, j + 1)\n"
            f"    return memo[(i, j)]\n\n"
            f"print(unique_paths(0, 0))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(m × n), Space: O(m × n)"
        )


    # 12. Write a memoized function to solve the problem of computing {{problem_name}} efficiently.
    elif "calculate the number of ways to represent it as the sum of" in q:
        import re
        match_n = re.search(r"number (\d+)", q)
        match_steps = re.search(r"sum of \[(.*?)\]", q)
        n = int(match_n.group(1)) if match_n else 5
        step_sizes = list(map(int, match_steps.group(1).split(','))) if match_steps else [1, 3, 4]

        memo = {}
        def count_ways(target):
            if target == 0:
                return 1
            if target < 0:
                return 0
            if target in memo:
                return memo[target]
            memo[target] = sum(count_ways(target - step) for step in step_sizes)
            return memo[target]

        result = count_ways(n)

        return (
            f"🧠 **Problem:** Count ways to represent {n} as the sum of steps {step_sizes} using memoization.\n\n"
            f"💡 We break down the number recursively and cache intermediate results.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"step_sizes = {step_sizes}\n"
            f"memo = {{}}\n"
            f"def count_ways(target):\n"
            f"    if target == 0:\n"
            f"        return 1\n"
            f"    if target < 0:\n"
            f"        return 0\n"
            f"    if target in memo:\n"
            f"        return memo[target]\n"
            f"    memo[target] = sum(count_ways(target - step) for step in step_sizes)\n"
            f"    return memo[target]\n\n"
            f"print(count_ways({n}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n × k), where k = number of step sizes\n"
            f"✅ Space: O(n)"
        )


    # 13. Demonstrate how memoization reduces the complexity of {{problem_name}}.
    elif "number of valid combinations of" in q:
        import re
        match_struct = re.search(r"combinations of (.+?)\.", q)
        struct = match_struct.group(1) if match_struct else "balanced parentheses"
        n = 4  # assuming 4 pairs (can also extract if given)

        memo = {}
        def catalan(n):
            if n <= 1:
                return 1
            if n in memo:
                return memo[n]
            memo[n] = sum(catalan(i) * catalan(n - i - 1) for i in range(n))
            return memo[n]

        result = catalan(n)

        return (
            f"🧠 **Problem:** Use memoization to compute number of valid combinations of `{struct}` (e.g., for {n} pairs).\n\n"
            f"💡 Catalan numbers count valid structures like parentheses, trees, and bracket expressions.\n\n"
            f"🔧 **Memoized Catalan Function:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def catalan(n):\n"
            f"    if n <= 1:\n"
            f"        return 1\n"
            f"    if n in memo:\n"
            f"        return memo[n]\n"
            f"    memo[n] = sum(catalan(i) * catalan(n - i - 1) for i in range(n))\n"
            f"    return memo[n]\n\n"
            f"print(catalan({n}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n²), Space: O(n)"
        )


    # 14. Apply memoization to return the nth term of the {{series_name}} using base cases.
    elif "computes the" in q and "number recursively" in q:
        import re
        match_n = re.search(r"computes the (\d+)(st|nd|rd|th)?", q)
        match_name = re.search(r"\d+\w*\s+(.*?) number", q)
        n = int(match_n.group(1)) if match_n else 5
        name = match_name.group(1).strip().title() if match_name else "Catalan"

        memo = {}
        def catalan(n):
            if n <= 1:
                return 1
            if n in memo:
                return memo[n]
            memo[n] = sum(catalan(i) * catalan(n - i - 1) for i in range(n))
            return memo[n]

        result = catalan(n)

        return (
            f"🧠 **Problem:** Compute the {n}th {name} number using memoization.\n\n"
            f"💡 Example: Catalan numbers are used in dynamic programming for combinatorial problems.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def catalan(n):\n"
            f"    if n <= 1:\n"
            f"        return 1\n"
            f"    if n in memo:\n"
            f"        return memo[n]\n"
            f"    memo[n] = sum(catalan(i) * catalan(n - i - 1) for i in range(n))\n"
            f"    return memo[n]\n\n"
            f"print(catalan({n}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n²), Space: O(n)"
        )


    # 15. Write a function using memoization to compute {{recursive_definition}}.
    elif "minimum number of coins needed to make change" in q:
        import re
        match_amount = re.search(r"change for (\d+)", q)
        match_coins = re.search(r"denominations \[(.*?)\]", q)
        amount = int(match_amount.group(1)) if match_amount else 11
        coins = list(map(int, match_coins.group(1).split(','))) if match_coins else [1, 2, 5]

        memo = {}
        def min_coins(n):
            if n == 0:
                return 0
            if n in memo:
                return memo[n]
            res = float('inf')
            for coin in coins:
                if n - coin >= 0:
                    res = min(res, 1 + min_coins(n - coin))
            memo[n] = res
            return res

        result = min_coins(amount)
        final = result if result != float('inf') else "No solution"

        return (
            f"🧠 **Problem:** Find the minimum number of coins needed to make change for {amount} using denominations {coins}.\n\n"
            f"💡 Uses top-down recursion with memoization to store minimal coins for each value.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"coins = {coins}\n"
            f"memo = {{}}\n"
            f"def min_coins(n):\n"
            f"    if n == 0:\n"
            f"        return 0\n"
            f"    if n in memo:\n"
            f"        return memo[n]\n"
            f"    res = float('inf')\n"
            f"    for coin in coins:\n"
            f"        if n - coin >= 0:\n"
            f"            res = min(res, 1 + min_coins(n - coin))\n"
            f"    memo[n] = res\n"
            f"    return res\n\n"
            f"print(min_coins({amount}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{final}`\n"
            f"✅ Time: O(amount × len(coins)), Space: O(amount)"
        )


   
    

# --- ANSWER GENERATION FUNCTION FOR MEMOIZATION APPLICATION LEVEL 2 (START) ---

def answer_memoization_application_lvl2(question):
    q = question.lower()

    # Helper function to parse coin/array/weights/values strings
    def parse_list_str(s):
        return list(map(int, re.findall(r'\d+', s)))

    def parse_weights_values_m(weights_str, values_str):
        weights = list(map(int, re.findall(r'\d+', weights_str)))
        values = list(map(int, re.findall(r'\d+', values_str)))
        items = []
        for i in range(len(weights)):
            items.append({'name': f"Item{i+1}", 'weight': weights[i], 'value': values[i]})
        return items

    # 1. Solve the {{problem}} using memoization and explain how it improves efficiency.
    if "number of ways to make a sum of" in q and "coins" in q:
        import re
        match_target = re.search(r"sum of (\d+)", q)
        match_coins = re.search(r"coins \[(.*?)\]", q)
        target = int(match_target.group(1)) if match_target else 7
        coins = list(map(int, match_coins.group(1).split(','))) if match_coins else [1, 2, 5]

        memo = {}
        def count_ways(n, idx):
            if n == 0:
                return 1
            if n < 0 or idx == len(coins):
                return 0
            key = (n, idx)
            if key in memo:
                return memo[key]
            # include or exclude current coin
            memo[key] = count_ways(n, idx + 1) + count_ways(n - coins[idx], idx)
            return memo[key]

        result = count_ways(target, 0)

        return (
            f"🧠 **Problem:** Find the number of ways to make a sum of {target} using coins {coins}.\n\n"
            f"💡 We recursively try including and excluding each coin, and memoize intermediate states.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def count_ways(n, idx):\n"
            f"    if n == 0:\n"
            f"        return 1\n"
            f"    if n < 0 or idx == len(coins):\n"
            f"        return 0\n"
            f"    key = (n, idx)\n"
            f"    if key in memo:\n"
            f"        return memo[key]\n"
            f"    memo[key] = count_ways(n, idx + 1) + count_ways(n - coins[idx], idx)\n"
            f"    return memo[key]\n\n"
            f"coins = {coins}\n"
            f"print(count_ways({target}, 0))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n × m), Space: O(n × m), where n = target, m = number of coins"
        )


    # 2. Use memoization to solve the problem of counting all ways to reach sum {{target}} using coins {{coins}}.
    elif "ways to decode the string" in q:
        import re
        match = re.search(r"decode the string '(\d+)'", q)
        s = match.group(1) if match else "1234"

        memo = {}
        def decode(i):
            if i in memo:
                return memo[i]
            if i == len(s):
                return 1
            if s[i] == '0':
                return 0
            res = decode(i + 1)
            if i + 1 < len(s) and int(s[i:i+2]) <= 26:
                res += decode(i + 2)
            memo[i] = res
            return res

        result = decode(0)

        return (
            f"🧠 **Problem:** Count the number of ways to decode the digit string '{s}' where A=1 to Z=26.\n\n"
            f"💡 We explore all valid single or two-digit decodings and memoize results from each position.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def decode(i):\n"
            f"    if i in memo:\n"
            f"        return memo[i]\n"
            f"    if i == len(s):\n"
            f"        return 1\n"
            f"    if s[i] == '0':\n"
            f"        return 0\n"
            f"    res = decode(i + 1)\n"
            f"    if i + 1 < len(s) and int(s[i:i+2]) <= 26:\n"
            f"        res += decode(i + 2)\n"
            f"    memo[i] = res\n"
            f"    return res\n\n"
            f"s = '{s}'\n"
            f"print(decode(0))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n), Space: O(n)"
        )

    
    # 3. Apply memoization to find the number of ways to decode the string '{{digits}}'.
    elif "subset of" in q and "sum to" in q:
        import re
        match_arr = re.search(r"subset of \[(.*?)\]", q)
        match_target = re.search(r"sum to (\d+)", q)
        arr = list(map(int, match_arr.group(1).split(','))) if match_arr else [3, 1, 5, 9]
        target = int(match_target.group(1)) if match_target else 10

        memo = {}
        def subset_sum(i, t):
            if t == 0:
                return True
            if i == len(arr) or t < 0:
                return False
            key = (i, t)
            if key in memo:
                return memo[key]
            memo[key] = subset_sum(i + 1, t) or subset_sum(i + 1, t - arr[i])
            return memo[key]

        result = subset_sum(0, target)

        return (
            f"🧠 **Problem:** Check if a subset of {arr} sums to {target} using memoization.\n\n"
            f"💡 Try both including and excluding each element, and memoize state (i, target).\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def subset_sum(i, t):\n"
            f"    if t == 0:\n"
            f"        return True\n"
            f"    if i == len(arr) or t < 0:\n"
            f"        return False\n"
            f"    key = (i, t)\n"
            f"    if key in memo:\n"
            f"        return memo[key]\n"
            f"    memo[key] = subset_sum(i + 1, t) or subset_sum(i + 1, t - arr[i])\n"
            f"    return memo[key]\n\n"
            f"arr = {arr}\n"
            f"print(subset_sum(0, {target}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n × target), Space: O(n × target)"
        )


    # 4. Implement a memoized version of the subset sum problem for array {{arr}} and target {{target}}.
    elif "0/1 knapsack" in q and "capacity" in q:
        import re
        w_match = re.search(r"weights=\[(.*?)\]", q)
        v_match = re.search(r"values=\[(.*?)\]", q)
        c_match = re.search(r"capacity=(\d+)", q)
        weights = list(map(int, w_match.group(1).split(','))) if w_match else [2, 3, 4, 5]
        values = list(map(int, v_match.group(1).split(','))) if v_match else [3, 4, 5, 6]
        cap = int(c_match.group(1)) if c_match else 5

        memo = {}
        def knapsack(i, rem):
            if i == len(weights) or rem <= 0:
                return 0
            key = (i, rem)
            if key in memo:
                return memo[key]
            if weights[i] > rem:
                memo[key] = knapsack(i + 1, rem)
            else:
                memo[key] = max(
                    knapsack(i + 1, rem),
                    values[i] + knapsack(i + 1, rem - weights[i])
                )
            return memo[key]

        result = knapsack(0, cap)

        return (
            f"🧠 **Problem:** Solve 0/1 Knapsack with weights={weights}, values={values}, and capacity={cap}.\n\n"
            f"💡 At each step, decide whether to include or exclude the current item. Memoize (i, remaining capacity).\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def knapsack(i, rem):\n"
            f"    if i == len(weights) or rem <= 0:\n"
            f"        return 0\n"
            f"    key = (i, rem)\n"
            f"    if key in memo:\n"
            f"        return memo[key]\n"
            f"    if weights[i] > rem:\n"
            f"        memo[key] = knapsack(i + 1, rem)\n"
            f"    else:\n"
            f"        memo[key] = max(\n"
            f"            knapsack(i + 1, rem),\n"
            f"            values[i] + knapsack(i + 1, rem - weights[i])\n"
            f"        )\n"
            f"    return memo[key]\n\n"
            f"weights = {weights}\n"
            f"values = {values}\n"
            f"print(knapsack(0, {cap}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n × capacity), Space: O(n × capacity)"
        )


    # 5. Write a memoized function to solve 0/1 Knapsack with weights={{weights}}, values={{values}}, and capacity={{capacity}}.
    elif "edit distance" in q and "between the strings" in q:
        import re
        w1 = re.search(r"between the strings '(\w+)'", q)
        w2 = re.search(r"and '(\w+)'", q)
        a = w1.group(1) if w1 else "kitten"
        b = w2.group(1) if w2 else "sitting"

        memo = {}
        def edit(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            if i == len(a):
                return len(b) - j
            if j == len(b):
                return len(a) - i
            if a[i] == b[j]:
                memo[(i, j)] = edit(i + 1, j + 1)
            else:
                insert = 1 + edit(i, j + 1)
                delete = 1 + edit(i + 1, j)
                replace = 1 + edit(i + 1, j + 1)
                memo[(i, j)] = min(insert, delete, replace)
            return memo[(i, j)]

        result = edit(0, 0)

        return (
            f"🧠 **Problem:** Compute the edit distance between '{a}' and '{b}' using memoization.\n\n"
            f"💡 Recursively compare characters and explore insert, delete, replace. Memoize (i, j) pairs.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def edit(i, j):\n"
            f"    if (i, j) in memo:\n"
            f"        return memo[(i, j)]\n"
            f"    if i == len(a):\n"
            f"        return len(b) - j\n"
            f"    if j == len(b):\n"
            f"        return len(a) - i\n"
            f"    if a[i] == b[j]:\n"
            f"        memo[(i, j)] = edit(i + 1, j + 1)\n"
            f"    else:\n"
            f"        insert = 1 + edit(i, j + 1)\n"
            f"        delete = 1 + edit(i + 1, j)\n"
            f"        replace = 1 + edit(i + 1, j + 1)\n"
            f"        memo[(i, j)] = min(insert, delete, replace)\n"
            f"    return memo[(i, j)]\n\n"
            f"a = '{a}'\n"
            f"b = '{b}'\n"
            f"print(edit(0, 0))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n × m), Space: O(n × m)"
        )


    # 6. Apply memoization to solve the edit distance between '{{word1}}' and '{{word2}}'.
    elif "longest common subsequence" in q:
        import re
        w1 = re.search(r"between '(\w+)'", q)
        w2 = re.search(r"and '(\w+)'", q)
        a = w1.group(1) if w1 else "AGGTAB"
        b = w2.group(1) if w2 else "GXTXAYB"

        memo = {}
        def lcs(i, j):
            if i == len(a) or j == len(b):
                return 0
            if (i, j) in memo:
                return memo[(i, j)]
            if a[i] == b[j]:
                memo[(i, j)] = 1 + lcs(i + 1, j + 1)
            else:
                memo[(i, j)] = max(lcs(i + 1, j), lcs(i, j + 1))
            return memo[(i, j)]

        result = lcs(0, 0)

        return (
            f"🧠 **Problem:** Calculate the length of the longest common subsequence (LCS) between '{a}' and '{b}' using memoization.\n\n"
            f"💡 We recursively compare characters of both strings. If they match, move both indices forward and add 1; else, explore skipping one character in either string. Memoize subproblems (i, j).\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def lcs(i, j):\n"
            f"    if i == len(a) or j == len(b):\n"
            f"        return 0\n"
            f"    if (i, j) in memo:\n"
            f"        return memo[(i, j)]\n"
            f"    if a[i] == b[j]:\n"
            f"        memo[(i, j)] = 1 + lcs(i + 1, j + 1)\n"
            f"    else:\n"
            f"        memo[(i, j)] = max(lcs(i + 1, j), lcs(i, j + 1))\n"
            f"    return memo[(i, j)]\n\n"
            f"a = '{a}'\n"
            f"b = '{b}'\n"
            f"print(lcs(0, 0))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n × m), Space: O(n × m), where n = len(a), m = len(b)"
        )


    # 7. Memoize a function to find LCS length between '{{string1}}' and '{{string2}}'.
    elif "count the number of ways to climb" in q:
        import re
        n_match = re.search(r"climb (\d+) stairs", q)
        options_match = re.search(r"step sizes are \[(.*?)\]", q)
        n = int(n_match.group(1)) if n_match else 5
        options = list(map(int, options_match.group(1).split(','))) if options_match else [1, 2]

        memo = {}
        def climb(steps):
            if steps == 0:
                return 1
            if steps < 0:
                return 0
            if steps in memo:
                return memo[steps]
            ways = 0
            for opt in options:
                ways += climb(steps - opt)
            memo[steps] = ways
            return ways

        result = climb(n)

        return (
            f"🧠 **Problem:** Count the number of ways to climb {n} stairs when allowed step sizes are {options}, using memoization.\n\n"
            f"💡 For each step count, recursively sum ways to climb by subtracting each allowed step size. Memoize intermediate results to avoid recomputation.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def climb(steps):\n"
            f"    if steps == 0:\n"
            f"        return 1\n"
            f"    if steps < 0:\n"
            f"        return 0\n"
            f"    if steps in memo:\n"
            f"        return memo[steps]\n"
            f"    ways = 0\n"
            f"    for opt in {options}:\n"
            f"        ways += climb(steps - opt)\n"
            f"    memo[steps] = ways\n"
            f"    return ways\n\n"
            f"n = {n}\n"
            f"print(climb(n))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n × k), Space: O(n), where n = number of stairs, k = number of step options"
        )


    # 8. Write a memoized function for climbing {{n}} steps with step options {{options}}.
    elif "wildcard pattern matching" in q:
        import re
        t_match = re.search(r"text '(.+?)'", q)
        p_match = re.search(r"pattern '(.+?)'", q)
        text = t_match.group(1) if t_match else "aab"
        pattern = p_match.group(1) if p_match else "a*b"

        memo = {}
        def is_match(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            if j == len(pattern):
                return i == len(text)
            if i == len(text):
                # pattern remaining must be all '*'
                for x in pattern[j:]:
                    if x != '*':
                        memo[(i, j)] = False
                        return False
                memo[(i, j)] = True
                return True

            if pattern[j] == '*':
                # * matches zero or more chars
                memo[(i, j)] = is_match(i, j + 1) or is_match(i + 1, j)
            elif pattern[j] == '?' or pattern[j] == text[i]:
                memo[(i, j)] = is_match(i + 1, j + 1)
            else:
                memo[(i, j)] = False
            return memo[(i, j)]

        result = is_match(0, 0)

        return (
            f"🧠 **Problem:** Determine if text '{text}' matches pattern '{pattern}' with wildcards '*' and '?' using memoization.\n\n"
            f"💡 Recursively check matches: '*' can match zero or more chars, '?' matches any single char. Memoize (i, j) states.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def is_match(i, j):\n"
            f"    if (i, j) in memo:\n"
            f"        return memo[(i, j)]\n"
            f"    if j == len(pattern):\n"
            f"        return i == len(text)\n"
            f"    if i == len(text):\n"
            f"        for x in pattern[j:]:\n"
            f"            if x != '*':\n"
            f"                memo[(i, j)] = False\n"
            f"                return False\n"
            f"        memo[(i, j)] = True\n"
            f"        return True\n"
            f"    if pattern[j] == '*':\n"
            f"        memo[(i, j)] = is_match(i, j + 1) or is_match(i + 1, j)\n"
            f"    elif pattern[j] == '?' or pattern[j] == text[i]:\n"
            f"        memo[(i, j)] = is_match(i + 1, j + 1)\n"
            f"    else:\n"
            f"        memo[(i, j)] = False\n"
            f"    return memo[(i, j)]\n\n"
            f"text = '{text}'\n"
            f"pattern = '{pattern}'\n"
            f"print(is_match(0, 0))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n × m), Space: O(n × m), where n = len(text), m = len(pattern)"
        )


    # 9. Use memoization to solve wildcard pattern matching between '{{text}}' and pattern '{{pattern}}'.
    elif "minimum number of coins needed" in q:
        import re
        amount_match = re.search(r"amount (\d+)", q)
        coins_match = re.search(r"denominations \[(.*?)\]", q)
        amount = int(amount_match.group(1)) if amount_match else 11
        coins = list(map(int, coins_match.group(1).split(','))) if coins_match else [1, 2, 5]

        memo = {}
        def min_coins(amt):
            if amt == 0:
                return 0
            if amt < 0:
                return float('inf')
            if amt in memo:
                return memo[amt]
            res = float('inf')
            for c in coins:
                sub_res = min_coins(amt - c)
                if sub_res != float('inf'):
                    res = min(res, 1 + sub_res)
            memo[amt] = res
            return res

        result = min_coins(amount)
        final_result = result if result != float('inf') else -1

        return (
            f"🧠 **Problem:** Find the minimum number of coins needed to make amount {amount} from denominations {coins} using memoization.\n\n"
            f"💡 Recursively try each coin and subtract it from amount. Memoize the minimum coins needed for each sub-amount.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def min_coins(amt):\n"
            f"    if amt == 0:\n"
            f"        return 0\n"
            f"    if amt < 0:\n"
            f"        return float('inf')\n"
            f"    if amt in memo:\n"
            f"        return memo[amt]\n"
            f"    res = float('inf')\n"
            f"    for c in {coins}:\n"
            f"        sub_res = min_coins(amt - c)\n"
            f"        if sub_res != float('inf'):\n"
            f"            res = min(res, 1 + sub_res)\n"
            f"    memo[amt] = res\n"
            f"    return res\n\n"
            f"amount = {amount}\n"
            f"print(min_coins(amount) if min_coins(amount) != float('inf') else -1)\n"
            f"```\n"
            f"📊 **Computed Output:** `{final_result}`\n"
            f"✅ Time: O(amount × k), Space: O(amount), where k = number of coin denominations"
        )


    # 10. Write a memoized version of the minimum coin change problem for coins {{coins}} and amount {{amount}}.
    elif "number of unique binary search trees" in q:
        import re
        n_match = re.search(r"with (\d+) nodes", q)
        n = int(n_match.group(1)) if n_match else 3

        memo = {}
        def num_trees(start, end):
            if start > end:
                return 1
            if (start, end) in memo:
                return memo[(start, end)]
            total = 0
            for root in range(start, end + 1):
                left = num_trees(start, root - 1)
                right = num_trees(root + 1, end)
                total += left * right
            memo[(start, end)] = total
            return total

        result = num_trees(1, n)

        return (
            f"🧠 **Problem:** Compute the number of unique binary search trees that can be formed with {n} nodes using memoization.\n\n"
            f"💡 For each number from start to end as root, recursively compute number of unique BSTs in left and right subtrees. Memoize results for intervals (start, end).\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def num_trees(start, end):\n"
            f"    if start > end:\n"
            f"        return 1\n"
            f"    if (start, end) in memo:\n"
            f"        return memo[(start, end)]\n"
            f"    total = 0\n"
            f"    for root in range(start, end + 1):\n"
            f"        left = num_trees(start, root - 1)\n"
            f"        right = num_trees(root + 1, end)\n"
            f"        total += left * right\n"
            f"    memo[(start, end)] = total\n"
            f"    return total\n\n"
            f"n = {n}\n"
            f"print(num_trees(1, n))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n³), Space: O(n²), where n = number of nodes"
        )


    # 11. Memoize a recursive function that returns the number of unique BSTs with {{n}} nodes.
    elif "palindromic substrings" in q:
        import re
        s_match = re.search(r"string '(.+?)'", q)
        s = s_match.group(1) if s_match else "ababa"

        memo = {}
        def is_palindrome(i, j):
            if i >= j:
                return True
            if (i, j) in memo:
                return memo[(i, j)]
            memo[(i, j)] = (s[i] == s[j]) and is_palindrome(i + 1, j - 1)
            return memo[(i, j)]

        count = 0
        for start in range(len(s)):
            for end in range(start, len(s)):
                if is_palindrome(start, end):
                    count += 1

        return (
            f"🧠 **Problem:** Count the number of palindromic substrings in '{s}' using memoization.\n\n"
            f"💡 We check each substring if it is a palindrome by comparing characters at start and end indices recursively. We memoize results for substring indices to avoid repeated checks.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def is_palindrome(i, j):\n"
            f"    if i >= j:\n"
            f"        return True\n"
            f"    if (i, j) in memo:\n"
            f"        return memo[(i, j)]\n"
            f"    memo[(i, j)] = (s[i] == s[j]) and is_palindrome(i + 1, j - 1)\n"
            f"    return memo[(i, j)]\n\n"
            f"s = '{s}'\n"
            f"count = 0\n"
            f"for start in range(len(s)):\n"
            f"    for end in range(start, len(s)):\n"
            f"        if is_palindrome(start, end):\n"
            f"            count += 1\n"
            f"print(count)\n"
            f"```\n"
            f"📊 **Computed Output:** `{count}`\n"
            f"✅ Time: O(n²), Space: O(n²), where n = len(s)"
        )


    # 12. Use memoization to compute the number of palindromic substrings in '{{s}}'.
    elif "maximum sum of non-adjacent elements" in q:
        import re
        arr_match = re.search(r"array \[(.*?)\]", q)
        arr = list(map(int, arr_match.group(1).split(','))) if arr_match else [3, 2, 7, 10]

        memo = {}
        def max_sum(i):
            if i >= len(arr):
                return 0
            if i in memo:
                return memo[i]
            # Include current element and skip next
            include = arr[i] + max_sum(i + 2)
            # Exclude current element
            exclude = max_sum(i + 1)
            memo[i] = max(include, exclude)
            return memo[i]

        result = max_sum(0)

        return (
            f"🧠 **Problem:** Find the maximum sum of non-adjacent elements in array {arr} using memoization.\n\n"
            f"💡 For each index, recursively decide to include the current element and skip the next, or exclude it and consider the next. Memoize results by index.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def max_sum(i):\n"
            f"    if i >= len(arr):\n"
            f"        return 0\n"
            f"    if i in memo:\n"
            f"        return memo[i]\n"
            f"    include = arr[i] + max_sum(i + 2)\n"
            f"    exclude = max_sum(i + 1)\n"
            f"    memo[i] = max(include, exclude)\n"
            f"    return memo[i]\n\n"
            f"arr = {arr}\n"
            f"print(max_sum(0))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n), Space: O(n), where n = len(arr)"
        )


    # 13. Apply memoization to solve the maximum sum of non-adjacent elements in array {{arr}}.
    elif "boolean parenthesization" in q:
        import re
        expr_match = re.search(r"expression '(.+?)'", q)
        expr = expr_match.group(1) if expr_match else "T|F&T^F"

        memo = {}
        def count_ways(i, j, is_true):
            if i > j:
                return 0
            if i == j:
                if is_true:
                    return 1 if expr[i] == 'T' else 0
                else:
                    return 1 if expr[i] == 'F' else 0
            if (i, j, is_true) in memo:
                return memo[(i, j, is_true)]

            ways = 0
            for k in range(i + 1, j, 2):
                op = expr[k]
                left_true = count_ways(i, k - 1, True)
                left_false = count_ways(i, k - 1, False)
                right_true = count_ways(k + 1, j, True)
                right_false = count_ways(k + 1, j, False)

                if op == '&':
                    ways += left_true * right_true if is_true else (left_true * right_false + left_false * right_true + left_false * right_false)
                elif op == '|':
                    ways += (left_true * right_true + left_true * right_false + left_false * right_true) if is_true else (left_false * right_false)
                elif op == '^':
                    ways += (left_true * right_false + left_false * right_true) if is_true else (left_true * right_true + left_false * right_false)
            memo[(i, j, is_true)] = ways
            return ways

        result = count_ways(0, len(expr) - 1, True)

        return (
            f"🧠 **Problem:** Count the number of ways to parenthesize the boolean expression '{expr}' so that it evaluates to True, using memoization.\n\n"
            f"💡 Recursively partition the expression around each operator, counting ways to get True/False on left and right subexpressions, combining results per operator logic. Memoize subproblems by (i, j, is_true).\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def count_ways(i, j, is_true):\n"
            f"    if i > j:\n"
            f"        return 0\n"
            f"    if i == j:\n"
            f"        if is_true:\n"
            f"            return 1 if expr[i] == 'T' else 0\n"
            f"        else:\n"
            f"            return 1 if expr[i] == 'F' else 0\n"
            f"    if (i, j, is_true) in memo:\n"
            f"        return memo[(i, j, is_true)]\n"
            f"    ways = 0\n"
            f"    for k in range(i + 1, j, 2):\n"
            f"        op = expr[k]\n"
            f"        left_true = count_ways(i, k - 1, True)\n"
            f"        left_false = count_ways(i, k - 1, False)\n"
            f"        right_true = count_ways(k + 1, j, True)\n"
            f"        right_false = count_ways(k + 1, j, False)\n"
            f"        if op == '&':\n"
            f"            ways += left_true * right_true if is_true else (left_true * right_false + left_false * right_true + left_false * right_false)\n"
            f"        elif op == '|':\n"
            f"            ways += (left_true * right_true + left_true * right_false + left_false * right_true) if is_true else (left_false * right_false)\n"
            f"        elif op == '^':\n"
            f"            ways += (left_true * right_false + left_false * right_true) if is_true else (left_true * right_true + left_false * right_false)\n"
            f"    memo[(i, j, is_true)] = ways\n"
            f"    return ways\n\n"
            f"expr = '{expr}'\n"
            f"print(count_ways(0, len(expr) - 1, True))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n³), Space: O(n²), where n = length of expression"
        )

            
    # 14. Write a memoized function to solve 0/1 Knapsack with weights={{weights}}, values={{values}}, and capacity={{capacity}}. (Duplicate of Q5, so providing similar answer)
    elif "rod cutting" in q:
        import re
        length_match = re.search(r"rod of length (\d+)", q)
        prices_match = re.search(r"prices \[(.*?)\]", q)
        length = int(length_match.group(1)) if length_match else 4
        prices = list(map(int, prices_match.group(1).split(','))) if prices_match else [2, 5, 7, 8]

        memo = {}
        def cut_rod(n):
            if n == 0:
                return 0
            if n in memo:
                return memo[n]
            max_val = float('-inf')
            for i in range(1, n + 1):
                if i <= len(prices):
                    max_val = max(max_val, prices[i - 1] + cut_rod(n - i))
            memo[n] = max_val
            return max_val

        result = cut_rod(length)

        return (
            f"🧠 **Problem:** Solve the rod cutting problem for rod length {length} with prices {prices} using memoization.\n\n"
            f"💡 For each length i, recursively try cutting the rod into pieces and sum prices. Memoize maximum revenue for each sub-length.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def cut_rod(n):\n"
            f"    if n == 0:\n"
            f"        return 0\n"
            f"    if n in memo:\n"
            f"        return memo[n]\n"
            f"    max_val = float('-inf')\n"
            f"    for i in range(1, n + 1):\n"
            f"        if i <= len(prices):\n"
            f"            max_val = max(max_val, prices[i - 1] + cut_rod(n - i))\n"
            f"    memo[n] = max_val\n"
            f"    return max_val\n\n"
            f"length = {length}\n"
            f"prices = {prices}\n"
            f"print(cut_rod(length))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n²), Space: O(n), where n = rod length"
        )


    # 15. Use memoization to solve Boolean Parenthesization for expression '{{expression}}'.
    elif "tiling problem" in q:
        import re
        n_match = re.search(r"2 x (\d+)", q)
        n = int(n_match.group(1)) if n_match else 5

        memo = {}
        def tile_ways(x):
            if x == 0 or x == 1:
                return 1
            if x in memo:
                return memo[x]
            memo[x] = tile_ways(x - 1) + tile_ways(x - 2)
            return memo[x]

        result = tile_ways(n)

        return (
            f"🧠 **Problem:** Calculate the number of ways to tile a 2 x {n} board using 2x1 tiles with memoization.\n\n"
            f"💡 The problem breaks down to either placing a tile vertically (reducing board width by 1) or horizontally (reducing by 2). Use recursion with memoization to count ways.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def tile_ways(x):\n"
            f"    if x == 0 or x == 1:\n"
            f"        return 1\n"
            f"    if x in memo:\n"
            f"        return memo[x]\n"
            f"    memo[x] = tile_ways(x - 1) + tile_ways(x - 2)\n"
            f"    return memo[x]\n\n"
            f"n = {n}\n"
            f"print(tile_ways(n))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n), Space: O(n)"
        )



# --- ANSWER GENERATION FUNCTION FOR MEMOIZATION APPLICATION LEVEL 3 (START) ---

def answer_memoization_application_lvl3(question):
    q = question.lower()

    # Helper function to parse list strings (already defined globally)
    def parse_list_str(s):
        return list(map(int, re.findall(r'\d+', s)))

    # Helper for parsing weights/values (already defined globally)
    def parse_weights_values_m(weights_str, values_str):
        weights = list(map(int, re.findall(r'\d+', weights_str)))
        values = list(map(int, re.findall(r'\d+', values_str)))
        items = []
        for i in range(len(weights)):
            items.append({'name': f"Item{i+1}", 'weight': weights[i], 'value': values[i]})
        return items

    # Helper for scalar_mult_cost (from MCM, potentially needed for matrix dims)
    def scalar_mult_cost_memo(p_i, p_k, p_j):
        return p_i * p_k * p_j

    # Helper for parsing dimensions string (from MCM, potentially needed for matrix dims)
    def parse_dims_string_memo(dims_str):
        array_match = re.search(r'\[(\d+(?:,\s*\d+)*)\]', dims_str)
        if array_match:
            return list(map(int, array_match.group(1).split(',')))
        dims = []
        pairs = dims_str.split(',')
        first_pair_match = re.search(r'(\d+)x(\d+)', pairs[0])
        if first_pair_match:
            dims.append(int(first_pair_match.group(1)))
            dims.append(int(first_pair_match.group(2)))
        for i in range(1, len(pairs)):
            next_pair_match = re.search(r'\d+x(\d+)', pairs[i])
            if next_pair_match:
                dims.append(int(next_pair_match.group(1)))
        return dims

    # 1. Apply memoization to solve the {{complex_problem}} under constraints {{constraint}}.
    if "apply memoization to solve the" in q and "under constraints" in q:
        match = re.search(r"solve the (.+) under constraints (.+)\.", q)
        problem_name = match.group(1).strip() if match else "a complex optimization"
        constraint = match.group(2).strip() if match else "specific conditions"

        # This is a generic question, so provide a general explanation.
        return (
            f"To apply memoization to solve the {problem_name} problem under constraints {constraint}:\n\n"
            f"For complex problems with constraints in applications (e.g., resource allocation, scheduling, combinatorial optimization), a direct recursive solution often leads to recomputing the same subproblems many times. Memoization is applied to efficiently handle these overlapping subproblems [cite: memoization.json].\n\n"
            f"**Approach:**\n"
            f"1.  **Identify Recursive Structure:** Break down the complex problem into smaller, identical subproblems that can be solved recursively.\n"
            f"2.  **Define State:** Determine the minimal set of parameters that uniquely define each subproblem (e.g., `(current_index, remaining_capacity, current_constraint_value)`).\n"
            f"3.  **Create Memoization Table:** Use a dictionary or multi-dimensional array (e.g., `memo[(index, capacity, constraint)]`) to store the results of solved subproblems [cite: memoization.json].\n"
            f"4.  **Implement Memoized Recursion:**\n"
            f"    * Base Cases: Define the solutions for the simplest subproblems.\n"
            f"    * Memo Check: At the start of the recursive function, check if the current state's result is in the memo. If yes, return it.\n"
            f"    * Recursive Step: Compute the solution for the current state by making recursive calls to smaller subproblems.\n"
            f"    * Memo Store: Before returning, store the computed result in the memo [cite: memoization.json].\n"
            f"This approach transforms the exponential time complexity of naive recursion into a polynomial one (e.g., O(N*C*K) if N, C, K are state parameters), making complex applications feasible."
        )

    # 2. Write a recursive + memoized solution for the Traveling Salesman Problem for {{locations}}.
    elif "write a recursive + memoized solution for the traveling salesman problem" in q:
        match = re.search(r"for \{(\d+(?:,\s*\d+)*)\}", q)
        if match:
            locations_str = match.group(1)
            # For TSP, locations usually imply a number of cities, and a distance matrix.
            # Let's assume locations_str gives the number of cities, and we'll use a placeholder distance matrix.
            num_cities = len(parse_list_str(locations_str)) # Or just int(locations_str) if it's a single number

            # Placeholder for a distance graph (adjacency matrix)
            # For a small number of cities, e.g., 4 cities, a 4x4 matrix
            # dist[i][j] = distance from city i to city j
            # Example: 4 cities
            # dist = [[0, 10, 15, 20],
            #         [10, 0, 35, 25],
            #         [15, 35, 0, 30],
            #         [20, 25, 30, 0]]
            
            # For a generic solution, we'll use a conceptual distance matrix.
            # TSP with memoization typically uses bitmask DP.
            # State: (mask, current_city) -> min_cost
            # mask: a bitmask representing visited cities
            # current_city: the last city visited in the path

            # Example distance matrix for num_cities
            # (Create a dummy one for calculation)
            dummy_dist_matrix = [[0] * num_cities for _ in range(num_cities)]
            for i in range(num_cities):
                for j in range(num_cities):
                    if i != j:
                        dummy_dist_matrix[i][j] = (i + j + 1) * 5 # Arbitrary non-zero distance
            
            memo_tsp = {} # (mask, current_city) -> min_cost

            def tsp_memo(mask, current_city):
                if mask == (1 << num_cities) - 1: # All cities visited
                    return dummy_dist_matrix[current_city][0] # Return to start (city 0)
                if (mask, current_city) in memo_tsp:
                    return memo_tsp[(mask, current_city)]

                min_path = float('inf')
                for next_city in range(num_cities):
                    if not (mask & (1 << next_city)): # If next_city not visited
                        cost = dummy_dist_matrix[current_city][next_city] + tsp_memo(mask | (1 << next_city), next_city)
                        min_path = min(min_path, cost)
                
                memo_tsp[(mask, current_city)] = min_path
                return min_path
            
            # Start from city 0, with only city 0 visited
            min_tsp_cost = tsp_memo(1, 0) # Mask 1 (0001) means city 0 visited

            return (
                f"To write a recursive + memoized solution for the Traveling Salesman Problem (TSP) for {num_cities} locations:\n\n"
                f"TSP is an NP-hard problem. A recursive solution with memoization (using bitmask dynamic programming) can solve it for a small number of locations (N < ~20-25) by caching the minimum cost to visit a subset of cities ending at a particular city [cite: memoization.json].\n\n"
                f"**State:** `dp[mask][last_city]` = minimum cost to visit cities represented by `mask`, ending at `last_city`.\n"
                f"**Bitmask:** A bitmask is used to represent the set of visited cities (e.g., if the 0th bit is set, city 0 is visited).\n\n"
                f"```python\n"
                f"tsp_memo_cache = {{}} # Key: (mask, current_city)\n"
                f"# dist_matrix = [...] # Adjacency matrix of distances between cities\n"
                f"# num_cities = ... # Number of cities\n"
                f"def solve_tsp_memo(mask, current_city, num_cities_val, dist_matrix_val):\n"
                f"    # Base case: If all cities visited (mask has all bits set)\n"
                f"    if mask == (1 << num_cities_val) - 1:\n"
                f"        return dist_matrix_val[current_city][0] # Return to starting city (city 0)\n"
                f"    \n"
                f"    if (mask, current_city) in tsp_memo_cache:\n"
                f"        return tsp_memo_cache[(mask, current_city)]\n"
                f"    \n"
                f"    min_path_cost = float('inf')\n"
                f"    for next_city in range(num_cities_val):\n"
                f"        # If next_city has not been visited yet (bit not set in mask)\n"
                f"        if not (mask & (1 << next_city)):\n"
                f"            cost_to_next = dist_matrix_val[current_city][next_city]\n"
                f"            # Recursively find path from next_city with updated mask\n"
                f"            remaining_path_cost = solve_tsp_memo(mask | (1 << next_city), next_city, num_cities_val, dist_matrix_val)\n"
                f"            min_path_cost = min(min_path_cost, cost_to_next + remaining_path_cost)\n"
                f"    \n"
                f"    tsp_memo_cache[(mask, current_city)] = min_path_cost\n"
                f"    return min_path_cost\n"
                f"```\n\n"
                f"For {num_cities} locations (using a dummy distance matrix), the minimum TSP cost is: **{min_tsp_cost}**."
            )

    # 3. Memoize a recursive solution for computing number of ways to reach bottom-right of a {{m}}x{{n}} grid with obstacles {{grid}}.
    elif "memoize a recursive solution for computing number of ways to reach bottom-right of a" in q:
        match = re.search(r"(\d+)x(\d+) grid with obstacles \{([\d,\s]+)\}", q)
        if match:
            m_rows = int(match.group(1))
            n_cols = int(match.group(2))
            obstacles_str = match.group(3)
            
            # Obstacles are typically (row, col) pairs.
            # Example: obstacles = [(0,1), (1,1)]
            # For simplicity, let's parse as flat list and assume (row,col) pairs
            # For the example, let's assume obstacles are coordinates like (r,c)
            # The input format {{grid}} is ambiguous, so we'll use a simple list of 0s and 1s
            # where 1 is obstacle, 0 is clear.
            # Let's create a dummy grid for the example
            dummy_grid = [[0 for _ in range(n_cols)] for _ in range(m_rows)]
            
            # Populate dummy obstacles based on example string (e.g., "0,1,1,1" means (0,1) and (1,1) are obstacles)
            obstacle_coords = []
            flat_obstacles = parse_list_str(obstacles_str)
            for i in range(0, len(flat_obstacles), 2):
                if i+1 < len(flat_obstacles):
                    r_obs, c_obs = flat_obstacles[i], flat_obstacles[i+1]
                    if 0 <= r_obs < m_rows and 0 <= c_obs < n_cols:
                        dummy_grid[r_obs][c_obs] = 1 # Mark as obstacle
                        obstacle_coords.append(f"({r_obs},{c_obs})")

            memo_grid_paths = {}
            def count_paths_memo(r, c):
                if r < 0 or c < 0: return 0 # Out of bounds
                if dummy_grid[r][c] == 1: return 0 # Obstacle
                if r == 0 and c == 0: return 1 # Base case: starting point
                if (r, c) in memo_grid_paths: return memo_grid_paths[(r, c)]

                # Can only move right or down
                ways = count_paths_memo(r - 1, c) + count_paths_memo(r, c - 1)
                
                memo_grid_paths[(r, c)] = ways
                return ways
            
            num_ways = count_paths_memo(m_rows - 1, n_cols - 1) # Start from (0,0) to (m-1, n-1)

            return (
                f"To compute the number of ways to reach the bottom-right of a {m_rows}x{n_cols} grid with obstacles {obstacle_coords} using memoization:\n\n"
                f"This is a classic grid pathfinding problem with obstacles. Memoization caches the number of ways to reach each cell `(r, c)`, avoiding redundant calculations for overlapping paths [cite: memoization.json].\n\n"
                f"**Grid:** {m_rows}x{n_cols}, Obstacles at: {obstacle_coords}\n"
                f"```python\n"
                f"grid_paths_memo_cache = {{}}\n"
                f"# grid_val = [...] # 2D list representing the grid (0 for clear, 1 for obstacle)\n"
                f"def count_grid_paths_memo(r_idx, c_idx, grid_val):\n"
                f"    if r_idx < 0 or c_idx < 0: return 0\n"
                f"    if grid_val[r_idx][c_idx] == 1: return 0 # Obstacle\n"
                f"    if r_idx == 0 and c_idx == 0: return 1 # Starting point\n"
                f"    if (r_idx, c_idx) in grid_paths_memo_cache: return grid_paths_memo_cache[(r_idx, c_idx)]\n"
                f"    \n"
                f"    ways = count_grid_paths_memo(r_idx - 1, c_idx, grid_val) + \\\n"
                f"           count_grid_paths_memo(r_idx, c_idx - 1, grid_val)\n"
                f"    \n"
                f"    grid_paths_memo_cache[(r_idx, c_idx)] = ways\n"
                f"    return ways\n"
                f"```\n\n"
                f"Number of ways to reach bottom-right: **{num_ways}**."
            )

    # 4. Use memoization to solve the minimum edit distance between strings '{{str1}}' and '{{str2}}'.
    elif "use memoization to solve the minimum edit distance between strings" in q:
        match = re.search(r"between strings '(\w+)' and '(\w+)'", q)
        if match:
            str1 = match.group(1)
            str2 = match.group(2)

            memo_edit_dist = {}
            def edit_distance_memo(i, j):
                if i == 0: return j # Base case: str1 is empty, need j insertions
                if j == 0: return i # Base case: str2 is empty, need i deletions
                if (i, j) in memo_edit_dist: return memo_edit_dist[(i, j)]

                cost_match = 0 if str1[i-1] == str2[j-1] else 1
                
                val_match_replace = edit_distance_memo(i - 1, j - 1) + cost_match
                val_delete = edit_distance_memo(i - 1, j) + 1
                val_insert = edit_distance_memo(i, j - 1) + 1
                
                result = min(val_match_replace, val_delete, val_insert)
                memo_edit_dist[(i, j)] = result
                return result
            
            distance = edit_distance_memo(len(str1), len(str2))

            return (
                f"To solve the minimum edit distance between strings '{str1}' and '{str2}' using memoization:\n\n"
                f"Edit distance (Levenshtein distance) calculates the minimum edits (insertions, deletions, substitutions) to transform one string into another. Its recursive solution has overlapping subproblems, making memoization highly effective for optimizing string comparison in applications like spell checkers or bioinformatics [cite: memoization.json].\n\n"
                f"```python\n"
                f"edit_dist_memo_cache = {{}}\n"
                f"def calculate_min_edit_distance_memo(i_idx, j_idx, s1_val, s2_val):\n"
                f"    if i_idx == 0: return j_idx\n"
                f"    if j_idx == 0: return i_idx\n"
                f"    if (i_idx, j_idx) in edit_dist_memo_cache: return edit_dist_memo_cache[(i_idx, j_idx)]\n\n"
                f"    cost_sub = 0 if s1_val[i_idx-1] == s2_val[j_idx-1] else 1\n"
                f"    \n"
                f"    val_match_sub = calculate_min_edit_distance_memo(i_idx - 1, j_idx - 1, s1_val, s2_val) + cost_sub\n"
                f"    val_delete = calculate_min_edit_distance_memo(i_idx - 1, j_idx, s1_val, s2_val) + 1\n"
                f"    val_insert = calculate_min_edit_distance_memo(i_idx, j_idx - 1, s1_val, s2_val) + 1\n"
                f"    \n"
                f"    result = min(val_match_sub, val_delete, val_insert)\n"
                f"    edit_dist_memo_cache[(i_idx, j_idx)] = result\n"
                f"    return result\n"
                f"```\n\n"
                f"String 1: '{str1}', String 2: '{str2}'\n"
                f"Minimum Edit Distance: **{distance}**."
            )

    # 5. Write a memoized recursive function for longest common subsequence of '{{str1}}' and '{{str2}}'.
    elif "write a memoized recursive function for longest common subsequence of" in q:
        match = re.search(r"of '(\w+)' and '(\w+)'", q)
        if match:
            s1 = match.group(1)
            s2 = match.group(2)

            memo_lcs_len = {}
            def lcs_length_memo(i, j):
                if i == 0 or j == 0: return 0
                if (i, j) in memo_lcs_len: return memo_lcs_len[(i, j)]

                if s1[i-1] == s2[j-1]: # Characters match
                    result = 1 + lcs_length_memo(i - 1, j - 1)
                else: # Characters don't match
                    result = max(lcs_length_memo(i - 1, j), lcs_length_memo(i, j - 1))
                
                memo_lcs_len[(i, j)] = result
                return result
            
            length = lcs_length_memo(len(s1), len(s2))

            return (
                f"To find the Longest Common Subsequence (LCS) length between '{s1}' and '{s2}' using a memoized recursive function:\n\n"
                f"The LCS problem is a classic dynamic programming problem with optimal substructure and overlapping subproblems [cite: lcs.json, memoization.json]. Memoization caches the LCS length for all substrings, preventing redundant calculations, which is useful in applications like version control systems or bioinformatics [cite: memoization.json].\n\n"
                f"```python\n"
                f"lcs_len_memo_cache = {{}}\n"
                f"def calculate_lcs_length_memo_recursive(i_idx, j_idx, s1_val, s2_val):\n"
                f"    if i_idx == 0 or j_idx == 0: return 0\n"
                f"    if (i_idx, j_idx) in lcs_len_memo_cache: return lcs_len_memo_cache[(i_idx, j_idx)]\n\n"
                f"    if s1_val[i_idx-1] == s2_val[j_idx-1]:\n"
                f"        result = 1 + calculate_lcs_length_memo_recursive(i_idx - 1, j_idx - 1, s1_val, s2_val)\n"
                f"    else:\n"
                f"        result = max(calculate_lcs_length_memo_recursive(i_idx - 1, j_idx, s1_val, s2_val),\n"
                f"                     calculate_lcs_length_memo_recursive(i_idx, j_idx - 1, s1_val, s2_val))\n"
                f"    \n"
                f"    lcs_len_memo_cache[(i_idx, j_idx)] = result\n"
                f"    return result\n"
                f"```\n\n"
                f"String 1: '{s1}', String 2: '{s2}'\n"
                f"LCS Length: **{length}**."
            )

    # 6. Memoize a solution to find the number of binary search trees with {{n}} nodes.
    elif "memoize a solution to find the number of binary search trees with" in q:
        match = re.search(r"with (\d+) nodes", q)
        if match:
            n_nodes = int(match.group(1))

            memo_bst_count = {}
            def num_unique_bsts_memo_app(n_val):
                if n_val <= 1: return 1 # Base case: 0 nodes or 1 node has 1 unique BST
                if n_val in memo_bst_count: return memo_bst_count[n_val]

                count = 0
                for i in range(1, n_val + 1): # i is the root node
                    left_subtrees = num_unique_bsts_memo_app(i - 1)
                    right_subtrees = num_unique_bsts_memo_app(n_val - i)
                    count += left_subtrees * right_subtrees
                
                memo_bst_count[n_val] = count
                return count
            
            num_bsts = num_unique_bsts_memo_app(n_nodes)

            return (
                f"To find the number of unique Binary Search Trees (BSTs) with {n_nodes} nodes using memoization:\n\n"
                f"This problem (related to Catalan numbers) has overlapping subproblems: the number of BSTs for `n` nodes depends on the number of BSTs for `0` to `n-1` nodes. Memoization efficiently caches these results, which is useful in compiler optimization or database indexing applications [cite: memoization.json].\n\n"
                f"```python\n"
                f"bst_count_memo_cache = {{}}\n"
                f"def count_unique_bsts_memoized_app(n_nodes_val):\n"
                f"    if n_nodes_val <= 1: return 1\n"
                f"    if n_nodes_val in bst_count_memo_cache: return bst_count_memo_cache[n_nodes_val]\n\n"
                f"    count = 0\n"
                f"    for i in range(1, n_nodes_val + 1): # Try each node as root\n"
                f"        left_subtrees = count_unique_bsts_memoized_app(i - 1)\n"
                f"        right_subtrees = count_unique_bsts_memoized_app(n_nodes_val - i)\n"
                f"        count += left_subtrees * right_subtrees\n"
                f"    \n"
                f"    bst_count_memo_cache[n_nodes_val] = count\n"
                f"    return count\n"
                f"```\n\n"
                f"Number of nodes: {n_nodes}\n"
                f"Number of unique BSTs: **{num_bsts}**."
            )

    # 7. Solve the N-Queens problem with memoization and return number of valid configurations on {{n}}x{{n}} board.
    elif "solve the n-queens problem with memoization and return number of valid configurations" in q:
        match = re.search(r"on (\d+)x(\d+) board", q)
        if match:
            n_board = int(match.group(1))

            # N-Queens is typically solved with backtracking. Memoization is less directly applicable
            # for the *number of solutions* unless you frame it as counting solutions for sub-boards.
            # However, if we think of it as "how many ways to place queens from row `r` onwards given `col_mask`, `diag1_mask`, `diag2_mask`"
            # then that state can be memoized.
            
            memo_nqueens = {} # (row, col_mask, diag1_mask, diag2_mask) -> count

            def solve_nqueens_memo(row, col_mask, diag1_mask, diag2_mask):
                if row == n_board: # All queens placed
                    return 1
                
                state_key = (row, col_mask, diag1_mask, diag2_mask)
                if state_key in memo_nqueens:
                    return memo_nqueens[state_key]

                count = 0
                # Iterate through columns in the current row
                for col in range(n_board):
                    # Check if current position (row, col) is safe
                    # col_mask: bit set if column is occupied
                    # diag1_mask: bit set if diagonal (row - col) is occupied (offset by N-1 to be non-negative)
                    # diag2_mask: bit set if diagonal (row + col) is occupied
                    
                    is_col_safe = not (col_mask & (1 << col))
                    is_diag1_safe = not (diag1_mask & (1 << (row - col + n_board - 1)))
                    is_diag2_safe = not (diag2_mask & (1 << (row + col)))

                    if is_col_safe and is_diag1_safe and is_diag2_safe:
                        # Place queen, recurse for next row
                        new_col_mask = col_mask | (1 << col)
                        new_diag1_mask = diag1_mask | (1 << (row - col + n_board - 1))
                        new_diag2_mask = diag2_mask | (1 << (row + col))
                        
                        count += solve_nqueens_memo(row + 1, new_col_mask, new_diag1_mask, new_diag2_mask)
                
                memo_nqueens[state_key] = count
                return count
            
            num_configs = solve_nqueens_memo(0, 0, 0, 0) # Start from row 0, all masks empty

            return (
                f"To solve the N-Queens problem for a {n_board}x{n_board} board using memoization (bitmask DP):\n\n"
                f"The N-Queens problem involves placing N queens on an NxN chessboard such that no two queens attack each other. A recursive backtracking solution can be memoized by caching results for subproblems defined by the current row and the occupied columns and diagonals (represented by bitmasks) [cite: memoization.json].\n\n"
                f"```python\n"
                f"nqueens_memo_cache = {{}} # Key: (row, col_mask, diag1_mask, diag2_mask)\n"
                f"def solve_nqueens_memoized(row_idx, col_mask_val, diag1_mask_val, diag2_mask_val, board_size):\n"
                f"    if row_idx == board_size: return 1 # All queens placed, found a valid configuration\n"
                f"    \n"
                f"    state_key = (row_idx, col_mask_val, diag1_mask_val, diag2_mask_val)\n"
                f"    if state_key in nqueens_memo_cache: return nqueens_memo_cache[state_key]\n"
                f"    \n"
                f"    count = 0\n"
                f"    for col_idx in range(board_size):\n"
                f"        # Check if current position (row_idx, col_idx) is safe\n"
                f"        is_col_safe = not (col_mask_val & (1 << col_idx))\n"
                f"        is_diag1_safe = not (diag1_mask_val & (1 << (row_idx - col_idx + board_size - 1)))\n"
                f"        is_diag2_safe = not (diag2_mask_val & (1 << (row_idx + col_idx)))\n"
                f"        \n"
                f"        if is_col_safe and is_diag1_safe and is_diag2_safe:\n"
                f"            # Place queen, recurse for next row with updated masks\n"
                f"            new_col_mask = col_mask_val | (1 << col_idx)\n"
                f"            new_diag1_mask = diag1_mask_val | (1 << (row_idx - col_idx + board_size - 1))\n"
                f"            new_diag2_mask = diag2_mask_val | (1 << (row_idx + col_idx))\n"
                f"            \n"
                f"            count += solve_nqueens_memoized(row_idx + 1, new_col_mask, new_diag1_mask, new_diag2_mask, board_size)\n"
                f"    \n"
                f"    nqueens_memo_cache[state_key] = count\n"
                f"    return count\n"
                f"```\n\n"
                f"Board size: {n_board}x{n_board}\n"
                f"Number of valid configurations: **{num_configs}**."
            )

    # 8. Use memoization in solving maximum profit problem in stock prices array {{prices}}.
    elif "use memoization in solving maximum profit problem in stock prices array" in q:
        match = re.search(r"array \{(\d+(?:,\s*\d+)*)\}", q)
        if match:
            prices = parse_list_str(match.group(1))

            # This is a classic DP problem: Buy/Sell Stock (at most one transaction)
            # Or multiple transactions, or k transactions.
            # Let's assume "at most one transaction" as it's the simplest.
            # For multiple transactions (unlimited), it's also DP.
            # For "maximum profit problem", it usually implies unlimited transactions.
            
            # Max profit with at most one transaction: iterate and keep track of min price seen so far.
            # This is iterative O(N), not typically memoized recursive.
            # If it's "max profit with K transactions", it's 3D DP: dp[k][i][holding_stock]
            
            # Let's do max profit with *unlimited* transactions (buy and sell any number of times)
            # This can be solved by summing all positive differences.
            # This is also O(N) iterative, no memoization needed.
            
            # If the question implies a recursive solution with memoization, it's likely a more complex variation.
            # Let's assume "max profit with at most one transaction" but frame a recursive solution with memoization.
            # State: (index, can_buy) -> max_profit
            # can_buy: True if we can buy, False if we must sell (already holding stock)

            memo_stock_profit = {} # (index, can_buy_flag) -> max_profit

            def max_profit_memo(index, can_buy):
                if index == len(prices): return 0 # Base case: end of prices
                if (index, can_buy) in memo_stock_profit: return memo_stock_profit[(index, can_buy)]

                profit = 0
                if can_buy:
                    # Option 1: Buy stock at current price
                    buy_profit = -prices[index] + max_profit_memo(index + 1, False) # Cost to buy, then must sell next
                    # Option 2: Don't buy stock at current price
                    no_buy_profit = max_profit_memo(index + 1, True) # Still can buy later
                    profit = max(buy_profit, no_buy_profit)
                else: # Must sell
                    # Option 1: Sell stock at current price
                    sell_profit = prices[index] + max_profit_memo(index + 1, True) # Gain from selling, then can buy again
                    # Option 2: Don't sell stock at current price (hold it)
                    no_sell_profit = max_profit_memo(index + 1, False) # Still holding, must sell later
                    profit = max(sell_profit, no_sell_profit)
                
                memo_stock_profit[(index, can_buy)] = profit
                return profit
            
            # Start with ability to buy (True) from index 0
            max_profit = max_profit_memo(0, True)

            return (
                f"To solve the maximum profit problem in stock prices array {prices} using memoization (for unlimited transactions):\n\n"
                f"This problem can be framed recursively, where at each day, you either buy, sell, or do nothing. Memoization caches the maximum profit for each state (current day, whether you hold a stock or not) [cite: memoization.json].\n\n"
                f"```python\n"
                f"stock_profit_memo_cache = {{}} # Key: (index, holding_stock_boolean)\n"
                f"# prices_list = [...] # Array of stock prices\n"
                f"def calculate_max_profit_memo(index_val, holding_stock_flag, prices_list_val):\n"
                f"    if index_val == len(prices_list_val): return 0 # Base case: end of prices\n"
                f"    \n"
                f"    state_key = (index_val, holding_stock_flag)\n"
                f"    if state_key in stock_profit_memo_cache: return stock_profit_memo_cache[state_key]\n"
                f"    \n"
                f"    profit_if_do_nothing = calculate_max_profit_memo(index_val + 1, holding_stock_flag, prices_list_val)\n"
                f"    \n"
                f"    if holding_stock_flag: # If currently holding stock, option to SELL\n"
                f"        # Sell: current price + profit from next state (can buy again)\n"
                f"        profit_if_action = prices_list_val[index_val] + calculate_max_profit_memo(index_val + 1, False, prices_list_val)\n"
                f"    else: # If not holding stock, option to BUY\n"
                f"        # Buy: -current price + profit from next state (now holding)\n"
                f"        profit_if_action = -prices_list_val[index_val] + calculate_max_profit_memo(index_val + 1, True, prices_list_val)\n"
                f"        \n"
                f"    result = max(profit_if_do_nothing, profit_if_action)\n"
                f"    stock_profit_memo_cache[state_key] = result\n"
                f"    return result\n"
                f"```\n\n"
                f"Stock Prices: {prices}\n"
                f"Maximum Profit (unlimited transactions): **{max_profit}**."
            )

    # 9. Memoize solution for the shortest path from node {{start}} to node {{end}} in graph {{graph}}.
    elif "memoize solution for the shortest path from node" in q and "to node" in q and "in graph" in q:
        match = re.search(r"node (\w+) to node (\w+) in graph (.+)\.", q)
        if match:
            start_node = match.group(1)
            end_node = match.group(2)
            graph_str = match.group(3)
            
            # Graph representation is ambiguous. Let's assume adjacency list with weights.
            # Example: "A:B=1,C=4; B:C=2,D=5; C:D=1; D:E=3"
            # Parse graph into a dictionary: {node: {neighbor: weight}}
            graph = {}
            nodes_in_graph = set()
            for part in graph_str.split(';'):
                if ':' in part:
                    node, connections = part.split(':')
                    node = node.strip()
                    graph[node] = {}
                    nodes_in_graph.add(node)
                    for conn in connections.split(','):
                        if '=' in conn:
                            neighbor, weight = conn.split('=')
                            graph[node][neighbor.strip()] = int(weight)
                            nodes_in_graph.add(neighbor.strip())
            
            # Shortest path in general graph (with potentially negative cycles) is Bellman-Ford.
            # If it's DAG or no negative cycles, Dijkstra's is faster but iterative.
            # Memoized recursive usually implies Bellman-Ford like approach for general graphs, or simple DP for DAGs.
            
            memo_shortest_path = {} # (current_node) -> min_distance_to_end

            # This recursive approach computes shortest path from current_node TO end_node
            # This is usually done with Bellman-Ford or Dijkstra iteratively.
            # A recursive memoized version for shortest path is more common for DAGs.
            # For general graphs, it can lead to infinite recursion if cycles are not handled.
            # Let's assume a DAG or that we are looking for simple paths (no repeated nodes).
            # For simplicity, let's implement a recursive Dijkstra-like approach for positive weights.
            
            # This is a recursive version of Dijkstra/Bellman-Ford, but it's usually less efficient than iterative
            # due to recursion overhead and potentially re-visiting nodes if not careful with state.
            
            # A more common memoized shortest path:
            # memo[node] = shortest_distance_from_start_to_node
            # This is filled iteratively (Bellman-Ford/Dijkstra).
            # If the question explicitly asks for *recursive* memoized, it's typically for DAGs.
            
            # Let's use a recursive Bellman-Ford like approach for positive weights for simplicity.
            # This will find shortest path from `start_node` to `current_node`.
            # We need shortest path from `current_node` to `end_node`.
            
            # Standard recursive shortest path (from start to current node):
            # memo[u] = shortest_dist from start to u
            # dist(u) = min(dist(v) + weight(v,u)) for all v -> u
            
            # Let's define a memoized function that computes shortest path from `u` to `end_node`.
            # This is more natural for recursive DP.
            
            memo_shortest_path_to_end = {} # (current_node) -> min_dist_from_current_to_end

            def get_shortest_path_to_end_memo(current_node):
                if current_node == end_node: return 0 # Base case: already at end
                if current_node not in graph: return float('inf') # No outgoing edges
                if current_node in memo_shortest_path_to_end: return memo_shortest_path_to_end[current_node]

                min_dist = float('inf')
                for neighbor, weight in graph[current_node].items():
                    dist_from_neighbor_to_end = get_shortest_path_to_end_memo(neighbor)
                    if dist_from_neighbor_to_end != float('inf'):
                        min_dist = min(min_dist, weight + dist_from_neighbor_to_end)
                
                memo_shortest_path_to_end[current_node] = min_dist
                return min_dist
            
            # Check if start_node and end_node exist in the graph
            if start_node not in nodes_in_graph or end_node not in nodes_in_graph:
                shortest_dist = "Nodes not found in graph."
            else:
                shortest_dist = get_shortest_path_to_end_memo(start_node)
                if shortest_dist == float('inf'):
                    shortest_dist = "No path"

            return (
                f"To find the shortest path from node '{start_node}' to node '{end_node}' in graph '{graph_str}' using a memoized recursive solution:\n\n"
                f"This problem can be solved with a recursive dynamic programming approach, particularly suitable for Directed Acyclic Graphs (DAGs) or graphs with positive edge weights where cycles are handled by memoization's caching. Memoization stores the shortest path from each intermediate node to the `end_node` [cite: memoization.json].\n\n"
                f"**Graph (Adjacency List):** {graph}\n"
                f"```python\n"
                f"shortest_path_memo_cache = {{}}\n"
                f"# graph_adj_list = {{...}} # Adjacency list with weights\n"
                f"# end_node_val = '...' # Target end node\n"
                f"def find_shortest_path_memo(current_node_val, end_node_val, graph_adj_list_val):\n"
                f"    if current_node_val == end_node_val: return 0\n"
                f"    if current_node_val not in graph_adj_list_val: return float('inf') # No outgoing edges\n"
                f"    if current_node_val in shortest_path_memo_cache: return shortest_path_memo_cache[current_node_val]\n"
                f"    \n"
                f"    min_dist = float('inf')\n"
                f"    for neighbor, weight in graph_adj_list_val[current_node_val].items():\n"
                f"        dist_from_neighbor_to_end = find_shortest_path_memo(neighbor, end_node_val, graph_adj_list_val)\n"
                f"        if dist_from_neighbor_to_end != float('inf'):\n"
                f"            min_dist = min(min_dist, weight + dist_from_neighbor_to_end)\n"
                f"    \n"
                f"    shortest_path_memo_cache[current_node_val] = min_dist\n"
                f"    return min_dist\n"
                f"```\n\n"
                f"Start Node: '{start_node}', End Node: '{end_node}'\n"
                f"Shortest Path Distance: **{shortest_dist}**."
            )

    # 10. Implement a hybrid top-down memoization and bottom-up DP solution for {{dp_problem}}.
    elif "implement a hybrid top-down memoization and bottom-up dp solution for" in q:
        match = re.search(r"solution for (.+)\.", q)
        dp_problem_name = match.group(1).strip() if match else "Fibonacci"

        # This is a conceptual explanation of a hybrid approach.
        # A common hybrid is: use top-down recursion for general calls, but if recursion depth gets too high
        # or a subproblem is large, switch to an iterative (bottom-up) approach for that subproblem.
        # Or, fill up a DP table bottom-up, but if a specific value is needed, use a top-down call.

        return (
            f"To implement a hybrid top-down memoization and bottom-up DP solution for the {dp_problem_name} problem:\n\n"
            f"A hybrid approach combines the intuitive top-down (recursive with memoization) style with the iterative efficiency of bottom-up (tabulation) DP. This can be beneficial for problems where some subproblems are naturally solved recursively, while others are better handled iteratively, or to prevent recursion depth limits [cite: memoization.json].\n\n"
            f"**Hybrid Strategy (Conceptual):**\n"
            f"1.  **Top-Down Interface:** Define a recursive function `solve(state)` that uses memoization. This function is the primary entry point.\n"
            f"2.  **Memoization Cache:** Maintain a global or class-level cache `memo`.\n"
            f"3.  **Base Cases:** Handle the simplest `state` values directly.\n"
            f"4.  **Hybrid Logic:** Inside `solve(state)`:\n"
            f"    * If `state` is in `memo`, return `memo[state]`.\n"
            f"    * If `state` is a 'large' subproblem (e.g., `n` is above a threshold, or a complex sub-structure), call an **iterative (bottom-up) helper function** to compute `solve(state)` and potentially fill a small local DP table for its immediate dependencies. Store this result in `memo`.\n"
            f"    * Otherwise (for 'smaller' or 'recursive-friendly' subproblems), make standard recursive calls to `solve(smaller_state_1)` and `solve(smaller_state_2)`, combining their results.\n"
            f"    * Store the final `result` in `memo[state]`.\n"
            f"**Benefits:** Combines the natural recursive structure for problem decomposition with the efficiency of iteration for specific subproblems, potentially preventing stack overflow and optimizing for different types of subproblems."
        )

    # 11. Apply memoization to find number of integer partitions of {{target}} using integers in {{nums}}.
    elif "apply memoization to find number of integer partitions of" in q and "using integers in" in q:
        match = re.search(r"partitions of (\d+) using integers in \{(\d+(?:,\s*\d+)*)\}", q)
        if match:
            target = int(match.group(1))
            nums = parse_list_str(match.group(2))

            memo_partitions = {} # (current_target, current_num_index) -> count

            def count_partitions_memo(current_target, num_idx):
                if current_target == 0: return 1 # Base case: one way to make 0 (empty set)
                if current_target < 0: return 0 # Cannot make negative sum
                if num_idx == 0: return 0 # No numbers left to use

                if (current_target, num_idx) in memo_partitions: return memo_partitions[(current_target, num_idx)]

                # Option 1: Exclude current number (nums[num_idx-1])
                ways_exclude = count_partitions_memo(current_target, num_idx - 1)
                
                # Option 2: Include current number (nums[num_idx-1])
                # We can use the current number multiple times (if not specified 0/1)
                # If 0/1 (each number used once): ways_include = count_partitions_memo(current_target - nums[num_idx-1], num_idx - 1)
                # If unlimited (like coin change count): ways_include = count_partitions_memo(current_target - nums[num_idx-1], num_idx)
                # Assuming 0/1 for "integer partitions" from a given set of integers.
                ways_include = count_partitions_memo(current_target - nums[num_idx-1], num_idx - 1)
                
                result = ways_exclude + ways_include
                memo_partitions[(current_target, num_idx)] = result
                return result
            
            # Sort numbers for consistent behavior if needed, or handle duplicates.
            # For "integer partitions", numbers are usually unique and used once.
            # If numbers can be repeated, it's a variation of Coin Change (ways to make sum).
            # Let's assume 0/1 usage of each number in `nums`.
            
            num_ways = count_partitions_memo(target, len(nums))

            return (
                f"To find the number of integer partitions of {target} using integers in {nums} using memoization:\n\n"
                f"This problem is a variation of the Subset Sum Count problem. Memoization caches the number of ways to form a specific sum using a subset of the available integers, avoiding redundant calculations for overlapping subproblems [cite: memoization.json].\n\n"
                f"```python\n"
                f"integer_partitions_memo_cache = {{}}\n"
                f"# nums_list = [...] # List of integers available\n"
                f"def count_integer_partitions_memo(current_target_val, num_idx, nums_list_val):\n"
                f"    if current_target_val == 0: return 1\n"
                f"    if current_target_val < 0: return 0\n"
                f"    if num_idx == 0: return 0 # No numbers left\n"
                f"    if (current_target_val, num_idx) in integer_partitions_memo_cache: return integer_partitions_memo_cache[(current_target_val, num_idx)]\n\n"
                f"    # Option 1: Exclude current number (nums_list_val[num_idx-1])\n"
                f"    ways_exclude = count_integer_partitions_memo(current_target_val, num_idx - 1, nums_list_val)\n"
                f"    \n"
                f"    # Option 2: Include current number (nums_list_val[num_idx-1])\n"
                f"    ways_include = count_integer_partitions_memo(current_target_val - nums_list_val[num_idx-1], num_idx - 1, nums_list_val)\n"
                f"    \n"
                f"    result = ways_exclude + ways_include\n"
                f"    integer_partitions_memo_cache[(current_target_val, num_idx)] = result\n"
                f"    return result\n"
                f"```\n\n"
                f"Target: {target}, Integers: {nums}\n"
                f"Number of integer partitions: **{num_ways}**."
            )

    # 12. Memoize recursive calls in bitmask-based solution for visiting all cities in {{n}} steps.
    elif "memoize recursive calls in bitmask-based solution for visiting all cities in" in q and "steps" in q:
        match = re.search(r"visiting all cities in (\d+) steps", q)
        if match:
            n_steps = int(match.group(1)) # This implies a fixed number of steps, not just min steps.
            # This is a variation of TSP or Hamiltonian Path/Cycle.
            # If "n steps" means exactly N steps, it's a constrained path.
            # Let's assume a simple graph, and we need to visit all cities in exactly N steps.
            # This is usually a DP state: dp[mask][last_city][steps_taken]
            
            # For simplicity, let's re-use the TSP bitmask DP concept, but for "exactly N steps".
            # This is more complex than standard TSP.
            # Let's re-frame to standard TSP (visiting all cities in minimum steps) but emphasize bitmask + memoization.
            # If "n steps" is the number of cities, it's standard TSP (N cities, N-1 steps to visit all, then 1 to return).
            
            # Let's assume `n` refers to the number of cities for a standard TSP problem.
            num_cities_tsp_steps = n_steps # Interpreting 'n steps' as 'n cities' for standard TSP
            
            # Dummy distance matrix
            dummy_dist_matrix_tsp = [[0] * num_cities_tsp_steps for _ in range(num_cities_tsp_steps)]
            for i in range(num_cities_tsp_steps):
                for j in range(num_cities_tsp_steps):
                    if i != j:
                        dummy_dist_matrix_tsp[i][j] = (i + j + 1) * 5 # Arbitrary distance
            
            memo_tsp_bitmask_steps = {} # (mask, current_city) -> min_cost

            def tsp_bitmask_memo(mask, current_city):
                if mask == (1 << num_cities_tsp_steps) - 1: # All cities visited
                    return dummy_dist_matrix_tsp[current_city][0] # Return to start (city 0)
                if (mask, current_city) in memo_tsp_bitmask_steps:
                    return memo_tsp_bitmask_steps[(mask, current_city)]

                min_path = float('inf')
                for next_city in range(num_cities_tsp_steps):
                    if not (mask & (1 << next_city)): # If next_city not visited
                        cost = dummy_dist_matrix_tsp[current_city][next_city] + tsp_bitmask_memo(mask | (1 << next_city), next_city)
                        min_path = min(min_path, cost)
                
                memo_tsp_bitmask_steps[(mask, current_city)] = min_path
                return min_path
            
            # Start from city 0, with only city 0 visited
            min_tsp_cost_steps = tsp_bitmask_memo(1, 0) # Mask 1 (0001) means city 0 visited

            return (
                f"To memoize recursive calls in a bitmask-based solution for visiting all cities (Traveling Salesman Problem) for {n_steps} cities:\n\n"
                f"The Traveling Salesman Problem (TSP) seeks the shortest possible route that visits each city exactly once and returns to the origin. For a small number of cities (N < ~20-25), a recursive solution with memoization using bitmasks is feasible [cite: memoization.json].\n"
                f"**State:** `dp[mask][last_city]` = minimum cost to visit cities represented by `mask`, ending at `last_city`.\n"
                f"**Bitmask:** A bitmask effectively represents the set of visited cities. Each bit corresponds to a city; if the bit is set, the city has been visited.\n\n"
                f"```python\n"
                f"tsp_bitmask_memo_cache = {{}} # Key: (mask, current_city)\n"
                f"# dist_matrix = [...] # Adjacency matrix of distances between cities\n"
                f"# num_cities = ... # Number of cities\n"
                f"def solve_tsp_bitmask_memo(mask, current_city, num_cities_val, dist_matrix_val):\n"
                f"    if mask == (1 << num_cities_val) - 1: # All cities visited\n"
                f"        return dist_matrix_val[current_city][0] # Return to starting city (city 0)\n"
                f"    \n"
                f"    if (mask, current_city) in tsp_bitmask_memo_cache:\n"
                f"        return tsp_bitmask_memo_cache[(mask, current_city)]\n"
                f"    \n"
                f"    min_cost = float('inf')\n"
                f"    for next_city in range(num_cities_val):\n"
                f"        if not (mask & (1 << next_city)): # If next_city not visited\n"
                f"            cost_to_next = dist_matrix_val[current_city][next_city]\n"
                f"            remaining_cost = solve_tsp_bitmask_memo(mask | (1 << next_city), next_city, num_cities_val, dist_matrix_val)\n"
                f"            min_cost = min(min_cost, cost_to_next + remaining_cost)\n"
                f"    \n"
                f"    tsp_bitmask_memo_cache[(mask, current_city)] = min_cost\n"
                f"    return min_cost\n"
                f"```\n\n"
                f"For {n_steps} cities (using a dummy distance matrix), the minimum cost to visit all cities is: **{min_tsp_cost_steps}**."
            )

    # 13. Use memoization to track optimal matrix multiplication order for matrix dimensions {{dims}}.
    elif "use memoization to track optimal matrix multiplication order for matrix dimensions" in q:
        match = re.search(r"dimensions\s+(\[.*?\])\.?", q)
        if match:
            dims_str = match.group(1)
            p_dims = parse_dims_string_memo(dims_str)
            n = len(p_dims) - 1

            # Reuse MCM DP logic (which is bottom-up tabulation, but the concept is memoization)
            # The question asks to "track optimal order", which is done by the 's' table.
            
            # MCM DP (bottom-up) already builds memoized tables (m for cost, s for split)
            # We can also show the top-down memoized version for order tracking.
            
            memo_mcm_cost_track = {} # (i, j) -> min_cost
            memo_mcm_split_track = {} # (i, j) -> k_split

            def mcm_memoized_track(i, j):
                if i == j: return 0
                if (i, j) in memo_mcm_cost_track: return memo_mcm_cost_track[(i, j)]
                
                min_q = float('inf')
                best_k = -1

                for k in range(i, j):
                    cost = mcm_memoized_track(i, k) + mcm_memoized_track(k + 1, j) + scalar_mult_cost_memo(p_dims[i-1], p_dims[k], p_dims[j])
                    if cost < min_q:
                        min_q = cost
                        best_k = k
                
                memo_mcm_cost_track[(i, j)] = min_q
                memo_mcm_split_track[(i, j)] = best_k
                return min_q
            
            optimal_cost = mcm_memoized_track(1, n)
            optimal_order_str = print_optimal_parenthesization(memo_mcm_split_track, 1, n)

            return (
                f"To use memoization to track optimal matrix multiplication order for matrix dimensions {dims_str} (P={p_dims}):\n\n"
                f"The Matrix Chain Multiplication (MCM) problem is a classic example of dynamic programming where memoization is used to store both the minimum cost (`m[i][j]`) and the optimal split point (`s[i][j]`) for each subproblem [cite: mcm.json, memoization.json]. The `s` table then allows reconstruction of the optimal parenthesization.\n\n"
                f"**Memoized Recursive Approach:**\n"
                f"```python\n"
                f"mcm_cost_memo_track = {{}} # Stores min cost for (i, j)\n"
                f"mcm_split_memo_track = {{}} # Stores optimal split k for (i, j)\n"
                f"# p_dims_val = [...] # Matrix dimensions\n"
                f"def calculate_mcm_memo_track(i_idx, j_idx, p_dims_val):\n"
                f"    if i_idx == j_idx: return 0\n"
                f"    if (i_idx, j_idx) in mcm_cost_memo_track: return mcm_cost_memo_track[(i_idx, j_idx)]\n"
                f"    \n"
                f"    min_q = float('inf')\n"
                f"    best_k = -1\n"
                f"    for k in range(i_idx, j_idx):\n"
                f"        cost = calculate_mcm_memo_track(i_idx, k, p_dims_val) + \\\n"
                f"               calculate_mcm_memo_track(k + 1, j_idx, p_dims_val) + \\\n"
                f"               (p_dims_val[i_idx-1] * p_dims_val[k] * p_dims_val[j_idx])\n"
                f"        if cost < min_q:\n"
                f"            min_q = cost\n"
                f"            best_k = k\n"
                f"    \n"
                f"    mcm_cost_memo_track[(i_idx, j_idx)] = min_q\n"
                f"    mcm_split_memo_track[(i_idx, j_idx)] = best_k\n"
                f"    return min_q\n"
                f"```\n\n"
                f"Minimum Cost: **{optimal_cost}**\n"
                f"Optimal Multiplication Order: **{optimal_order_str}**."
            )

    # 14. Apply memoization to solve recursive logic in subset XOR sum problem for array {{arr}}.
    elif "apply memoization to solve recursive logic in subset xor sum problem for array" in q:
        match = re.search(r"array \{(\d+(?:,\s*\d+)*)\}", q)
        if match:
            arr = parse_list_str(match.group(1))

            # This is a variation of subset sum.
            # dp[index][current_xor_sum] = True/False (or count)
            
            memo_xor_sum = {} # (index, current_xor_sum) -> count (or boolean)

            # Let's count the number of subsets that achieve a certain XOR sum.
            # Or, find if a target XOR sum is possible.
            # The question is "subset XOR sum problem", usually implying finding all possible XOR sums or a specific one.
            # Let's find the XOR sum of all possible subsets.
            
            # This is usually done iteratively (tabulation)
            # dp[xor_sum] = True if xor_sum is achievable
            
            # For a recursive memoized version:
            # Function to find if a target_xor_sum is achievable from index 'idx' onwards.
            
            # Let's find the XOR sum of all possible subsets (return a set of all possible XOR sums).
            
            # This can be done by building up a set of all possible XOR sums.
            # memo[idx] = set of all XOR sums achievable from arr[idx:]
            
            memo_xor_subset_sums = {} # (index, current_xor_val) -> boolean (can reach end with 0)

            def can_reach_xor_sum_memo(index, current_xor):
                if index == len(arr):
                    return current_xor == 0 # Base case: if we reached end and XOR sum is 0
                if (index, current_xor) in memo_xor_subset_sums:
                    return memo_xor_subset_sums[(index, current_xor)]

                # Option 1: Exclude current element
                exclude = can_reach_xor_sum_memo(index + 1, current_xor)
                
                # Option 2: Include current element
                include = can_reach_xor_sum_memo(index + 1, current_xor ^ arr[index])
                
                result = exclude or include
                memo_xor_subset_sums[(index, current_xor)] = result
                return result
            
            # To find all possible XOR sums, we need a different memoized function.
            # Let's return the set of all possible XOR sums.
            
            memo_all_xor_sums = {} # (index) -> set of all XOR sums from arr[index:]

            def get_all_xor_sums_memo(index):
                if index == len(arr):
                    return {0} # Base case: empty set has XOR sum 0
                if index in memo_all_xor_sums:
                    return memo_all_xor_sums[index]

                # Get all XOR sums from remaining array
                sums_from_rest = get_all_xor_sums_memo(index + 1)
                
                # Add current element XORed with each sum from rest
                current_sums = set(sums_from_rest)
                for s in sums_from_rest:
                    current_sums.add(s ^ arr[index])
                
                memo_all_xor_sums[index] = current_sums
                return current_sums
            
            all_possible_xor_sums = sorted(list(get_all_xor_sums_memo(0)))

            return (
                f"To solve recursive logic in the subset XOR sum problem for array {arr} using memoization:\n\n"
                f"The subset XOR sum problem involves finding all possible XOR sums of subsets of a given array. Memoization efficiently caches the set of XOR sums achievable from suffixes of the array, preventing redundant computations [cite: memoization.json].\n\n"
                f"```python\n"
                f"xor_sums_memo_cache = {{}} # Key: (index) -> set of XOR sums\n"
                f"# arr_val = [...] # Input array\n"
                f"def get_all_subset_xor_sums_memo(index_val, arr_val):\n"
                f"    if index_val == len(arr_val):\n"
                f"        return {{0}} # Base case: empty set has XOR sum 0\n"
                f"    if index_val in xor_sums_memo_cache:\n"
                f"        return xor_sums_memo_cache[index_val]\n\n"
                f"    # Get all XOR sums from the rest of the array\n"
                f"    sums_from_rest = get_all_subset_xor_sums_memo(index_val + 1, arr_val)\n"
                f"    \n"
                f"    # Create new sums by XORing current element with sums from rest\n"
                f"    current_element = arr_val[index_val]\n"
                f"    new_sums = set(sums_from_rest)\n"
                f"    for s in sums_from_rest:\n"
                f"        new_sums.add(s ^ current_element)\n"
                f"    \n"
                f"    xor_sums_memo_cache[index_val] = new_sums\n"
                f"    return new_sums\n"
                f"```\n\n"
                f"Array: {arr}\n"
                f"All possible subset XOR sums: **{all_possible_xor_sums}**."
            )

    # 15. Design memoized approach to solve DP with two changing parameters like f(i, j) for {{dp_problem}}.
    elif "design memoized approach to solve dp with two changing parameters like f(i, j)" in q:
        match = re.search(r"for (.+)\.", q)
        dp_problem_name = match.group(1).strip() if match else "Longest Common Subsequence"

        # Use LCS as a classic example of f(i,j) DP
        return (
            f"To de sign a memoized approach to solve a dynamic programming problem with two changing parameters like `f(i, j)` (e.g., for {dp_problem_name}):\n\n"
            f"Many DP problems involve two dimensions of state (e.g., `i` and `j` representing indices in two sequences, or items and capacity). Memoization effectively transforms the recursive solution's exponential complexity into a polynomial one (typically O(N*M) if parameters range up to N and M) [cite: memoization.json].\n\n"
            f"**Design Steps:**\n"
            f"1.  **Define Recursive Relation:** Express `f(i, j)` in terms of smaller subproblems `f(i', j')`.\n"
            f"2.  **Identify Base Cases:** Define the simplest `f(i, j)` values (e.g., when `i` or `j` is 0).\n"
           f"3.  **Memoization Table:** Create a 2D cache (e.g., a dictionary `memo = {{}}` with `(i, j)` tuples as keys, or a 2D list/array `memo[i][j]`) [cite: memoization.json].\n"
            f"4.  **Implement Memoized Function:**\n"
            f"    * At the start, check `if (i, j) in memo: return memo[(i, j)]`.\n"
            f"    * Implement the recursive relation, making calls to `f(i', j')`.\n"
            f"    * Store the result: `memo[(i, j)] = result` before returning [cite: memoization.json].\n"
            f"**Example: Longest Common Subsequence (LCS)**\n"
            f"LCS length `LCS(i, j)` (for `str1[0...i-1]` and `str2[0...j-1]`) is defined as:\n"
            f"If `str1[i-1] == str2[j-1]`: `1 + LCS(i-1, j-1)`\n"
            f"Else: `max(LCS(i-1, j), LCS(i, j-1))`\n"
            f"Base cases: `LCS(0, j) = 0`, `LCS(i, 0) = 0`.\n"
            f"```python\n"
            f"lcs_2d_memo = {{}}\n"
            f"def solve_lcs_memo(i, j, s1, s2):\n"
            f"    if i == 0 or j == 0: return 0\n"
            f"    if (i, j) in lcs_2d_memo: return lcs_2d_memo[(i, j)]\n"
            f"    \n"
            f"    if s1[i-1] == s2[j-1]:\n"
            f"        result = 1 + solve_lcs_memo(i - 1, j - 1, s1, s2)\n"
            f"    else:\n"
            f"        result = max(solve_lcs_memo(i - 1, j, s1, s2), solve_lcs_memo(i, j - 1, s1, s2))\n"
            f"    \n"
            f"    lcs_2d_memo[(i, j)] = result\n"
            f"    return result\n"
            f"```"
        )

    # 16. Implement memoization in recursive DP for counting valid parentheses combinations of size {{n}}.
    elif "implement memoization in recursive dp for counting valid parentheses combinations of size" in q:
        match = re.search(r"size (\d+)", q)
        if match:
            n_pairs = int(match.group(1))

            # This problem is equivalent to Catalan numbers.
            # C_n = sum(C_i * C_{n-1-i}) for i from 0 to n-1.
            # C_0 = 1 (empty string), C_1 = 1 (())
            
            memo_parens = {}
            def count_valid_parens_memo(n_val):
                if n_val == 0: return 1 # Base case: 1 way for 0 pairs (empty string)
                if n_val < 0: return 0
                if n_val in memo_parens: return memo_parens[n_val]

                count = 0
                for i in range(n_val): # i represents number of pairs in first part
                    count += count_valid_parens_memo(i) * count_valid_parens_memo(n_val - 1 - i)
                
                memo_parens[n_val] = count
                return count
            
            num_combinations = count_valid_parens_memo(n_pairs)

            return (
                f"To count valid parentheses combinations of size {n_pairs} using memoization in recursive DP:\n\n"
                f"This problem is a classic application of dynamic programming, often solved using Catalan numbers. Memoization efficiently caches the number of valid combinations for smaller sizes, preventing redundant calculations for overlapping subproblems [cite: memoization.json].\n\n"
                f"```python\n"
                f"valid_parens_memo_cache = {{}}\n"
                f"def count_valid_parentheses_memo(n_val):\n"
                f"    if n_val == 0: return 1\n"
                f"    if n_val < 0: return 0\n"
                f"    if n_val in valid_parens_memo_cache: return valid_parens_memo_cache[n_val]\n\n"
                f"    count = 0\n"
                f"    for i in range(n_val): # i represents the number of pairs in the first part\n"
                f"        # ( (i pairs) ) ( (n_val - 1 - i) pairs )\n"
                f"        count += count_valid_parentheses_memo(i) * count_valid_parentheses_memo(n_val - 1 - i)\n"
                f"    \n"
                f"    valid_parens_memo_cache[n_val] = count\n"
                f"    return count\n"
                f"```\n\n"
                f"Size: {n_pairs}\n"
                f"Number of valid parentheses combinations: **{num_combinations}**."
            )

    # 17. Solve problem with recursive state: f(i, k, t) using memoization in context of {{problem_name}}.
    elif "solve problem with recursive state: f(i, k, t) using memoization in context of" in q:
        match = re.search(r"context of (.+)\.", q)
        problem_name = match.group(1).strip() if match else "a 3D DP problem"

        # This is a generic 3D DP state. Provide a conceptual example.
        # Example: 0/1 Knapsack with a third constraint (e.g., max_items)
        return (
            f"To solve a problem with recursive state `f(i, k, t)` using memoization in the context of {problem_name}:\n\n"
            f"When a dynamic programming problem has a recursive state defined by three changing parameters (`i`, `k`, `t`), memoization uses a 3D cache (e.g., a dictionary with `(i, k, t)` tuples as keys, or a 3D array `memo[i][k][t]`) to store and retrieve results of subproblems [cite: memoization.json].\n\n"
            f"**Conceptual Example (e.g., Knapsack with Max Items Constraint):**\n"
            f"`f(item_idx, current_capacity, items_taken)` = max value considering items up to `item_idx`, with `current_capacity`, and having taken `items_taken`.\n"
            f"```python\n"
            f"three_param_memo = {{}}\n"
            f"def solve_f_i_k_t_memo(i, k, t, problem_params):\n"
            f"    # Base cases for i, k, t\n"
            f"    if i < 0 or k < 0 or t < 0: return 0 # Adjust base cases based on problem\n"
            f"    if (i, k, t) in three_param_memo: return three_param_memo[(i, k, t)]\n"
            f"    \n"
            f"    # Recursive calls, combining results from smaller states\n"
            f"    # Example: result = solve_f_i_k_t_memo(i-1, k, t, ...) + solve_f_i_k_t_memo(i, k-1, t, ...)\n"
            f"    result = ... # Complex computation based on problem_params and recursive calls\n"
            f"    \n"
            f"    three_param_memo[(i, k, t)] = result\n"
            f"    return result\n"
            f"```\n\n"
            f"This approach reduces the time complexity from exponential to polynomial (O(I*K*T)) by ensuring each unique state `(i, k, t)` is computed only once [cite: memoization.json]."
        )

    # 18. Write a memoized function for recursive DAG traversal with caching at node level for graph {{graph}}.
    elif "write a memoized function for recursive dag traversal with caching at node level for graph" in q:
        match = re.search(r"for graph (.+)\.", q)
        if match:
            graph_str = match.group(1)
            
            # Parse graph into adjacency list. Assume nodes are simple strings/integers.
            # Example: "A:B,C; B:D; C:D; D:"
            graph_adj = {}
            for part in graph_str.split(';'):
                if ':' in part:
                    node, neighbors_str = part.split(':')
                    node = node.strip()
                    graph_adj[node] = []
                    if neighbors_str:
                        for neighbor in neighbors_str.split(','):
                            graph_adj[node].append(neighbor.strip())
            
            # Example: Find longest path in DAG
            memo_dag_path = {} # (node) -> longest_path_from_node

            def longest_path_dag_memo(node):
                if node not in graph_adj or not graph_adj[node]: # Base case: no outgoing edges
                    return 0 # Path length from this node is 0
                if node in memo_dag_path:
                    return memo_dag_path[node]

                max_len = 0
                for neighbor in graph_adj[node]:
                    max_len = max(max_len, 1 + longest_path_dag_memo(neighbor)) # 1 for edge + path from neighbor
                
                memo_dag_path[node] = max_len
                return max_len
            
            # Find overall longest path by iterating through all nodes as potential start nodes
            overall_longest_path = 0
            if graph_adj:
                for node in graph_adj:
                    overall_longest_path = max(overall_longest_path, longest_path_dag_memo(node))

            return (
                f"To write a memoized function for recursive DAG (Directed Acyclic Graph) traversal with caching at the node level for graph '{graph_str}':\n\n"
                f"In DAG traversals (e.g., for finding longest path, shortest path, or topological sorting dependent calculations), memoization is used to cache results computed for each node. This prevents redundant re-traversals of subgraphs, optimizing performance [cite: memoization.json].\n\n"
                f"**Graph (Adjacency List):** {graph_adj}\n"
                f"**Example (Longest Path in DAG):**\n"
                f"```python\n"
                f"dag_path_memo_cache = {{}} # Key: node -> result\n"
                f"# graph_adj_list = {{...}} # Adjacency list of the DAG\n"
                f"def find_longest_path_dag_memo(node_val, graph_adj_list_val):\n"
                f"    if node_val not in graph_adj_list_val or not graph_adj_list_val[node_val]:\n"
                f"        return 0 # Base case: no outgoing edges, path length is 0\n"
                f"    if node_val in dag_path_memo_cache:\n"
                f"        return dag_path_memo_cache[node_val]\n"
                f"    \n"
                f"    max_len = 0\n"
                f"    for neighbor in graph_adj_list_val[node_val]:\n"
                f"        max_len = max(max_len, 1 + find_longest_path_dag_memo(neighbor, graph_adj_list_val))\n"
                f"    \n"
                f"    dag_path_memo_cache[node_val] = max_len\n"
                f"    return max_len\n"
                f"```\n\n"
                f"Overall Longest Path in DAG: **{overall_longest_path}**."
            )

    # 19. Use memoization to solve recursive grammar parsing problem for string '{{s}}' and rule set {{rules}}.
    elif "use memoization to solve recursive grammar parsing problem for string" in q:
        match = re.search(r"string '(\w+)' and rule set (.+)\.", q)
        if match:
            s = match.group(1)
            rules_str = match.group(2)
            
            # Grammar parsing is complex. Let's use a simplified example:
            # Can string s[i...j] be parsed as a valid expression based on simple rules.
            # Rules: "S -> AB", "A -> a", "B -> b"
            # Or, for "valid parentheses" type grammar.
            
            # Let's use a simplified "word break" problem as a proxy for grammar parsing.
            # Given a string and a dictionary of words, can the string be segmented into words from the dictionary?
            # memo[index] = True if s[index:] can be segmented.
            
            # Assume rules_str is a comma-separated list of words in the dictionary.
            word_dict = set(rules_str.replace("{", "").replace("}", "").replace("'", "").split(','))
            word_dict = {word.strip() for word in word_dict if word.strip()} # Clean up and make a set
            
            memo_word_break = {} # (index) -> boolean

            def word_break_memo(start_idx):
                if start_idx == len(s): return True # Base case: reached end of string
                if start_idx in memo_word_break: return memo_word_break[start_idx]

                for end_idx in range(start_idx + 1, len(s) + 1):
                    word = s[start_idx:end_idx]
                    if word in word_dict:
                        if word_break_memo(end_idx):
                            memo_word_break[start_idx] = True
                            return True
                
                memo_word_break[start_idx] = False
                return False
            
            can_parse = word_break_memo(0)

            return (
                f"To solve a recursive grammar parsing problem for string '{s}' and rule set {rules_str} using memoization:\n\n"
                f"Recursive grammar parsing (e.g., for compilers, natural language processing, or query parsers) often involves checking if substrings conform to grammar rules. This leads to overlapping subproblems (e.g., parsing a phrase might involve parsing the same sub-phrases multiple times). Memoization caches the parsing results for substrings, drastically improving efficiency [cite: memoization.json].\n\n"
                f"**Example (Word Break Problem as a proxy for parsing):**\n"
                f"Given string '{s}' and dictionary {word_dict}:\n"
                f"```python\n"
                f"word_break_memo_cache = {{}}\n"
                f"# word_dictionary = {{...}} # Set of valid words/tokens\n"
                f"def solve_grammar_parsing_memo(start_idx, input_string, word_dictionary_val):\n"
                f"    if start_idx == len(input_string): return True\n"
                f"    if start_idx in word_break_memo_cache: return word_break_memo_cache[start_idx]\n"
                f"    \n"
                f"    for end_idx in range(start_idx + 1, len(input_string) + 1):\n"
                f"        word = input_string[start_idx:end_idx]\n"
                f"        if word in word_dictionary_val:\n"
                f"            if solve_grammar_parsing_memo(end_idx, input_string, word_dictionary_val):\n"
                f"                word_break_memo_cache[start_idx] = True\n"
                f"                return True\n"
                f"    \n"
                f"    word_break_memo_cache[start_idx] = False\n"
                f"    return False\n"
                f"```\n\n"
                f"Can string '{s}' be parsed/segmented: **{can_parse}**."
            )

    # 20. Apply memoization to optimize computation of multivariate recurrence function f(i, j, k).
    elif "apply memoization to optimize computation of multivariate recurrence function f(i, j, k)" in q:
        # This is a generic multivariate recurrence. Provide a conceptual example.
        # Example: A 3D DP problem like a 3D knapsack, or a path in a 3D grid.
        return (
            f"To apply memoization to optimize the computation of a multivariate recurrence function `f(i, j, k)`:\n\n"
            f"When a recursive function's state depends on multiple changing parameters (e.g., `i`, `j`, and `k`), it often leads to a large number of overlapping subproblems. Memoization is essential for optimizing such functions, transforming their exponential complexity into polynomial time [cite: memoization.json].\n\n"
            f"**Design Steps:**\n"
            f"1.  **Define Recursive Relation:** Express `f(i, j, k)` in terms of smaller subproblems `f(i', j', k')`.\n"
            f"2.  **Identify Base Cases:** Define the simplest `f(i, j, k)` values.\n"
            f"3.  **Memoization Table:** Create a multi-dimensional cache (e.g., a dictionary `memo = {{}}` with `(i, j, k)` tuples as keys, or a 3D array `memo[i][j][k]`) [cite: memoization.json].\n"
            f"4.  **Implement Memoized Function:**\n"
            f"    * At the start, check `if (i, j, k) in memo: return memo[(i, j, k)]`.\n"
            f"    * Implement the recursive relation, making calls to `f(i', j', k')`.\n"
            f"    * Store the result: `memo[(i, j, k)] = result` before returning [cite: memoization.json].\n"
            f"**Example (Conceptual):**\n"
            f"Consider a problem like finding the number of ways to reach a cell `(i, j, k)` in a 3D grid, where you can only move along axes. `f(i, j, k) = f(i-1, j, k) + f(i, j-1, k) + f(i, j, k-1)`.\n"
            f"```python\n"
            f"multivariate_memo = {{}}\n"
            f"def solve_multivariate_memo(i, j, k):\n"
            f"    if i < 0 or j < 0 or k < 0: return 0 # Base case: out of bounds\n"
            f"    if i == 0 and j == 0 and k == 0: return 1 # Base case: starting point\n"
            f"    if (i, j, k) in multivariate_memo: return multivariate_memo[(i, j, k)]\n"
            f"    \n"
            f"    result = solve_multivariate_memo(i - 1, j, k) + \\\n"
            f"             solve_multivariate_memo(i, j - 1, k) + \\\n"
            f"             solve_multivariate_memo(i, j, k - 1)\n"
            f"    \n"
            f"    multivariate_memo[(i, j, k)] = result\n"
            f"    return result\n"
            f"```\n"
            f"This approach makes the computation of such complex recurrences feasible and efficient."
        )

    return "Answer generation for this Memoization Level 3 Application question is not implemented yet."

# --- ANSWER GENERATION FUNCTION FOR MEMOIZATION APPLICATION LEVEL 1 (END) ---

#---- quantitative
def answer_memoization_quantitative_lvl1(question):
    import re
    q = question.lower()

# q1 . "Using memoization, determine the maximum sum of non-adjacent elements in the array {{arr}}.",
    if "maximum sum of non-adjacent elements" in q:
        # extract array from question
        match = re.search(r"array\s*\[([^\]]+)\]", question)
        if match:
            arr_str = match.group(1)
            try:
                arr = [int(x.strip()) for x in arr_str.split(",")]
            except:
                return "⚠️ Could not parse array values."
        else:
            return "⚠️ No array found in question."

        # define memoization function
        def max_non_adj_sum(arr):
            memo = {}
            def helper(i):
                if i >= len(arr): return 0
                if i in memo: return memo[i]
                include = arr[i] + helper(i + 2)
                exclude = helper(i + 1)
                memo[i] = max(include, exclude)
                return memo[i]
            return helper(0)

        result = max_non_adj_sum(arr)

        return (
            f"👨‍🏫 To find the maximum sum of non-adjacent elements in an array like {arr}, we use memoization to store intermediate results.\n\n"
            f"🪜 Logic:\n"
            f"- Either include arr[i] and skip arr[i+1], or skip arr[i]\n"
            f"- State: dp(i) = max(arr[i] + dp(i+2), dp(i+1))\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def max_non_adj_sum(arr):\n"
            "    memo = {}\n"
            "    def helper(i):\n"
            "        if i >= len(arr): return 0\n"
            "        if i in memo: return memo[i]\n"
            "        include = arr[i] + helper(i + 2)\n"
            "        exclude = helper(i + 1)\n"
            "        memo[i] = max(include, exclude)\n"
            "        return memo[i]\n"
            "    return helper(0)\n"
            "\n"
            f"# Input array:\n"
            f"arr = {arr}\n"
            f"print('Maximum sum of non-adjacent elements:', max_non_adj_sum(arr))\n"
            "```\n\n"
            f"✅ Final Answer: {result}"
        )
    
    # --- Question 1: Climb a staircase with 1, 2, or 3 steps ---
    elif "climb a staircase with" in q and "steps, where you can take either 1, 2, or 3 steps at a time" in q:
        match = re.search(r"climb a staircase with (\d+) steps", q)
        if match:
            n = int(match.group(1))
        else:
            return "⚠️ Could not parse the number of steps (n) from the question."

        memo = {} # Initialize a dictionary for memoization

        def dp(i):
            # Base cases
            if i < 0:
                return 0
            if i == 0:
                return 1
            
            # Check if the result is already memoized
            if i in memo:
                return memo[i]
            
            # Compute the result using the recurrence relation
            # Sum ways from taking 1, 2, or 3 steps from previous positions
            memo[i] = dp(i-1) + dp(i-2) + dp(i-3)
            return memo[i]

        result = dp(n)
        
        return (
            f"👨‍🏫 To calculate the total number of unique ways to climb a staircase with {n} steps, where you can take either 1, 2, or 3 steps at a time, we use memoization.\n\n"
            f"🪜 Logic:\n"
            f"- `dp(i) = dp(i-1) + dp(i-2) + dp(i-3)`\n"
            f"- Base cases: `dp(0) = 1`, `dp(negative) = 0`\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "memo = {} # Initialize a dictionary for memoization\n\n"
            "def dp(i):\n"
            "    # Base cases\n"
            "    if i < 0:\n"
            "        return 0\n"
            "    if i == 0:\n"
            "        return 1\n"
            "    \n"
            "    # Check if the result is already memoized\n"
            "    if i in memo:\n"
            "        return memo[i]\n"
            "    \n"
            "    # Compute the result using the recurrence relation\n"
            "    memo[i] = dp(i-1) + dp(i-2) + dp(i-3)\n"
            "    return memo[i]\n\n"
            f"# Input for this specific question:\n"
            f"n = {n}\n"
            f"result = dp(n)\n"
            f"print(f'Total unique ways to climb {{n}} steps:', result)\n"
            "```\n\n"
            f"✅ Final Answer: {result}"
        )
    
     # --- Question 2: Minimum coins for a given amount ---
    elif "minimum number of coins required to make a total of" in q and "given the denominations" in q:
        match_amount = re.search(r"total of (\d+)", q)
        match_denominations = re.search(r"denominations \[([\d,\s]+)\]", q)
        
        if match_amount and match_denominations:
            amount = int(match_amount.group(1))
            coins_str = match_denominations.group(1)
            coins = [int(c.strip()) for c in coins_str.split(',')]
        else:
            return "⚠️ Could not parse the total amount or denominations from the question."

        memo = {} # Initialize a dictionary for memoization

        def dp(rem):
            # Base cases
            if rem == 0:
                return 0
            if rem < 0:
                return float('inf') # Represent impossibility with infinity
            
            # Check if the result is already memoized
            if rem in memo:
                return memo[rem]
            
            min_count = float('inf')
            # Iterate through each coin denomination
            for coin in coins:
                # Recursively find the minimum for the remaining amount
                res = dp(rem - coin)
                # If the subproblem is solvable (not infinity), consider it
                if res != float('inf'):
                    min_count = min(min_count, 1 + res)
            
            # Memoize the result
            memo[rem] = min_count
            return min_count
        
        # Start the DP process with the target amount
        result_raw = dp(amount)
        # Return -1 if the amount cannot be made, otherwise return the actual count
        result = result_raw if result_raw != float('inf') else -1
        
        return (
            f"👨‍🏫 To find the minimum number of coins required to make a total of {amount}, given the denominations {coins}, we use memoization.\n\n"
            f"🪜 Logic:\n"
            f"- `dp(rem) = 1 + min(dp(rem - c))` for all `c` in `coins`.\n"
            f"- Base cases: `dp(0) = 0`, `dp(negative_amount) = infinity`\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "memo = {} # Initialize a dictionary for memoization\n\n"
            "def dp(rem):\n"
            "    # Base cases\n"
            "    if rem == 0:\n"
            "        return 0\n"
            "    if rem < 0:\n"
            "        return float('inf')\n"
            "    \n"
            "    # Check if the result is already memoized\n"
            "    if rem in memo:\n"
            "        return memo[rem]\n"
            "    \n"
            "    min_count = float('inf')\n"
            "    for coin in coins:\n" # 'coins' is accessible from outer scope
            "        res = dp(rem - coin)\n"
            "        if res != float('inf'):\n"
            "            min_count = min(min_count, 1 + res)\n"
            "    \n"
            "    memo[rem] = min_count\n"
            "    return min_count\n\n"
            f"# Input for this specific question:\n"
            f"amount = {amount}\n"
            f"coins = {coins}\n"
            f"result_raw = dp(amount)\n"
            f"result_display = result_raw if result_raw != float('inf') else -1\n"
            f"print(f'Minimum number of coins required to make {{amount}} with denominations {{coins}}: {{result_display}}')\n"
            "```\n\n"
            f"✅ Final Answer: {result}"
        )


    # --- Question 3: Number of ways to partition an integer ---
    elif "number of ways to partition the number" in q and "into the sum of positive integers using memoization" in q:
        match = re.search(r"partition the number (\d+)", q)
        if match:
            n = int(match.group(1))
        else:
            return "⚠️ Could not parse the number (n) from the question."

        memo = {} # Initialize a dictionary for memoization

        def dp(current_n, max_allowed_num):
            # Base cases
            if current_n == 0:
                return 1
            if current_n < 0 or max_allowed_num == 0:
                return 0
            
            # Check if the result is already memoized
            state = (current_n, max_allowed_num)
            if state in memo:
                return memo[state]
            
            # Compute the result using the recurrence relation
            # Option 1: Don't use the current max_allowed_num
            ways_without_max = dp(current_n, max_allowed_num - 1)
            
            # Option 2: Use the current max_allowed_num
            # Note: We can use max_allowed_num multiple times, so the second recursive call
            # keeps max_allowed_num the same.
            ways_with_max = dp(current_n - max_allowed_num, max_allowed_num)
            
            # Memoize the result
            memo[state] = ways_without_max + ways_with_max
            return memo[state]
        
        result = dp(n, n) # Start the DP process. The largest number we can use in the partition is 'n' itself.
        
        return (
            f"👨‍🏫 To find the number of ways to partition the number {n} into the sum of positive integers using memoization (order does not matter).\n\n"
            f"🪜 Logic:\n"
            f"- `dp(current_n, max_allowed_num) = dp(current_n, max_allowed_num - 1) + dp(current_n - max_allowed_num, max_allowed_num)`\n"
            f"- Base cases: `dp(0, max_allowed_num) = 1`, `dp(negative_n, max_allowed_num) = 0`, `dp(positive_n, 0) = 0`\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "memo = {} # Initialize a dictionary for memoization\n\n"
            "def dp(current_n, max_allowed_num):\n"
            "    # Base cases\n"
            "    if current_n == 0:\n"
            "        return 1\n"
            "    if current_n < 0 or max_allowed_num == 0:\n"
            "        return 0\n"
            "    \n"
            "    # Check if the result is already memoized\n"
            "    state = (current_n, max_allowed_num)\n"
            "    if state in memo:\n"
            "        return memo[state]\n"
            "    \n"
            "    # Compute the result using the recurrence relation\n"
            "    ways_without_max = dp(current_n, max_allowed_num - 1)\n"
            "    ways_with_max = dp(current_n - max_allowed_num, max_allowed_num)\n"
            "    \n"
            "    memo[state] = ways_without_max + ways_with_max\n"
            "    return memo[state]\n\n"
            f"# Input for this specific question:\n"
            f"n = {n}\n"
            f"result = dp(n, n)\n"
            f"print(f'Number of ways to partition {{n}} into positive integers: {{result}}')\n"
            "```\n\n"
            f"✅ Final Answer: {result}"
        )

     # --- Question 4: Nth term of the Fibonacci sequence ---
    elif "nth term of the fibonacci sequence where" in q and "is the given input" in q:
        match = re.search(r"where (\d+) is the given input", q)
        if match:
            n = int(match.group(1))
        else:
            return "⚠️ Could not parse the value of n from the question."

        memo = {} # Initialize a dictionary for memoization

        def f(i):
            # Base cases
            if i <= 1:
                return i
            
            # Check if the result is already memoized
            if i in memo:
                return memo[i]
            
            # Compute the result using the recurrence relation
            memo[i] = f(i - 1) + f(i - 2)
            return memo[i]

        result = f(n)
        
        return (
            f"👨‍🏫 To find the {n}th term of the Fibonacci sequence using memoization.\n\n"
            f"🪜 Logic:\n"
            f"- `F(n) = F(n-1) + F(n-2)`\n"
            f"- Base cases: `F(0) = 0`, `F(1) = 1`\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "memo = {} # Initialize a dictionary for memoization\n\n"
            "def f(i):\n"
            "    # Base cases\n"
            "    if i <= 1:\n"
            "        return i\n"
            "    \n"
            "    # Check if the result is already memoized\n"
            "    if i in memo:\n"
            "        return memo[i]\n"
            "    \n"
            "    # Compute the result using the recurrence relation\n"
            "    memo[i] = f(i - 1) + f(i - 2)\n"
            "    return memo[i]\n\n"
            f"# Input for this specific question:\n"
            f"n = {n}\n"
            f"result = f(n)\n"
            f"print(f'The {{n}}th term of the Fibonacci sequence is: {{result}}')\n"
            "```\n\n"
            f"✅ Final Answer: {result}"
        )

     # --- Question 1: Unique Ways to Reach Top-Right of an m x n Grid ---
    elif "calculate the number of ways to reach the top of a" in q and "grid from the bottom-left corner, only moving up or right" in q:
        match = re.search(r"a (\d+)x(\d+) grid", q)
        if match:
            m = int(match.group(1))
            n = int(match.group(2))
        else:
            return "⚠️ Could not parse grid dimensions (m x n) from the question."

        memo = {} # Initialize a dictionary for memoization

        def dp(i, j):
            # Base case: If we reach the top-right corner (0, n-1)
            if i == 0 and j == n - 1:
                return 1
            # Base case: If we go out of bounds
            # For bottom-left to top-right, moving UP (i-1) and RIGHT (j+1) from original code.
            # My logic in explanation is from TOP-LEFT to BOTTOM-RIGHT.
            # Let's adjust dp(i,j) to mean ways to reach (i,j) from (0,0) or ways to reach (m-1,n-1) from (i,j).
            # The current question asks from BOTTOM-LEFT (m-1,0) to TOP-RIGHT (0,n-1) moving UP or RIGHT.
            # So, from (i,j), next moves are (i-1,j) or (i,j+1).
            if i < 0 or j >= n or i >= m or j < 0:
                return 0
            
            # Check if the result is already memoized
            state = (i, j)
            if state in memo:
                return memo[state]
            
            # Recursive calls: sum ways from moving up or right
            # From current (i,j), we can go up to (i-1, j) or right to (i, j+1)
            paths_from_up = dp(i - 1, j)
            paths_from_right = dp(i, j + 1)
            
            # Memoize the result
            memo[state] = paths_from_up + paths_from_right
            return memo[state]

        # Start the DP from the bottom-left corner (m-1, 0)
        result = dp(m - 1, 0)
        
        return (
            f"👨‍🏫 To calculate the number of ways to reach the top of a {m}x{n} grid from the bottom-left corner, moving only up or right, we use memoization.\n\n"
            f"🪜 Logic:\n"
            f"- `dp(i, j) = dp(i-1, j) + dp(i, j+1)` (from current cell, sum ways from moving up or right)\n"
            f"- Base cases: `dp(0, n-1) = 1` (destination), `dp(out_of_bounds) = 0`\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "memo = {} # Initialize a dictionary for memoization\n\n"
            "def dp(i, j):\n"
            "    # Base case: If we reach the top-right corner (0, n-1)\n"
            "    if i == 0 and j == n - 1:\n"
            "        return 1\n"
            "    # Base case: If we go out of bounds (check all four boundaries)\n"
            "    if i < 0 or j >= n or i >= m or j < 0:\n" # Uses m and n from outer scope
            "        return 0\n"
            "    \n"
            "    # Check if the result is already memoized\n"
            "    state = (i, j)\n"
            "    if state in memo:\n"
            "        return memo[state]\n"
            "    \n"
            "    # Recursive calls: sum ways from moving up or right\n"
            "    paths_from_up = dp(i - 1, j)\n"
            "    paths_from_right = dp(i, j + 1)\n"
            "    \n"
            "    # Memoize the result\n"
            "    memo[state] = paths_from_up + paths_from_right\n"
            "    return memo[state]\n\n"
            f"# Input for this specific question:\n"
            f"m = {m}\n"
            f"n = {n}\n"
            f"result = dp(m - 1, 0) # Start from bottom-left corner (m-1, 0)\n"
            f"print(f'Number of ways to reach top of {{m}}x{{n}} grid from bottom-left (up/right only): {{result}}')\n"
            "```\n\n"
            f"✅ Final Answer: {result}"
        )

    # --- Question 2: Distinct Paths in a Grid with Obstacles ---
    elif "determine the total number of distinct paths to traverse a grid with obstacles" in q and "represented by a" in q and "grid of 0s and 1s" in q:
        # For simplicity, we'll assume the m and n are part of the direct question,
        # and the grid itself is not provided as a literal string in the question.
        # If the grid is to be parsed, a more complex regex and grid reconstruction logic is needed.
        # Let's assume a default grid for computation if not provided, or extract m, n for conceptual code.
        match = re.search(r"by a (\d+)x(\d+) grid", q)
        if match:
            m = int(match.group(1))
            n = int(match.group(2))
        else:
            # Default example grid if dimensions are not explicitly given
            m, n = 3, 3 
        
        # Default example grid with obstacles for computation, assuming 0 for clear, 1 for obstacle
        # This part would typically be parsed from a more complex question string if the grid content changes.
        # For this example, we'll use a fixed example grid for computation.
        example_grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
        # Adjust m, n to match example_grid if a default is used
        if m != len(example_grid) or n != len(example_grid[0]):
             m, n = len(example_grid), len(example_grid[0])


        memo = {} # Initialize a dictionary for memoization

        def dp(r, c):
            # Base case: If we go out of bounds or hit an obstacle
            if r >= m or c >= n or example_grid[r][c] == 1:
                return 0
            # Base case: If we reach the bottom-right corner
            if r == m - 1 and c == n - 1:
                return 1
            
            # Check if the result is already memoized
            state = (r, c)
            if state in memo:
                return memo[state]
            
            # Recursive calls: Sum ways from moving down or right
            paths_from_down = dp(r + 1, c)
            paths_from_right = dp(r, c + 1)
            
            # Memoize the result
            memo[state] = paths_from_down + paths_from_right
            return memo[state]

        # Start the DP from the top-left corner (0, 0)
        result = dp(0, 0)
        
        return (
            f"👨‍🏫 To determine the total number of distinct paths to traverse a {m}x{n} grid with obstacles (where 1s represent obstacles), we use memoization.\n"
            f"This problem extends the classic grid pathfinding by introducing blocked cells that cannot be traversed. We move from the top-left `(0,0)` to the bottom-right `(m-1, n-1)`, moving only `down` or `right`.\n\n"
            f"🪜 Logic:\n"
            f"- `dp(r, c) = dp(r+1, c) + dp(r, c+1)` (sum ways from moving down or right)\n"
            f"- Base cases: `dp(m-1, n-1) = 1` (destination), `dp(out_of_bounds_or_obstacle) = 0`\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "memo = {} # Initialize a dictionary for memoization\n\n"
            "def dp(r, c):\n"
            "    # Base case: If we go out of bounds or hit an obstacle\n"
            f"    # m and n are derived from the grid dimensions\n"
            f"    # example_grid is the actual grid being used for computation\n"
            "    if r >= len(example_grid) or c >= len(example_grid[0]) or example_grid[r][c] == 1:\n"
            "        return 0\n"
            "    # Base case: If we reach the bottom-right corner\n"
            "    if r == len(example_grid) - 1 and c == len(example_grid[0]) - 1:\n"
            "        return 1\n"
            "    \n"
            "    # Check if the result is already memoized\n"
            "    state = (r, c)\n"
            "    if state in memo:\n"
            "        return memo[state]\n"
            "    \n"
            "    # Recursive calls: Sum ways from moving down or right\n"
            "    paths_from_down = dp(r + 1, c)\n"
            "    paths_from_right = dp(r, c + 1)\n"
            "    \n"
            "    # Memoize the result\n"
            "    memo[state] = paths_from_down + paths_from_right\n"
            "    return memo[state]\n\n"
            f"# Input for this specific question (using an example grid for computation):\n"
            f"example_grid = [\n"
            f"    [0, 0, 0],\n"
            f"    [0, 1, 0],\n"
            f"    [0, 0, 0]\n"
            f"]\n"
            f"m, n = len(example_grid), len(example_grid[0]) # Dimensions for conceptual clarity\n"
            f"result = dp(0, 0) # Start from top-left corner (0, 0)\n"
            f"print(f'Total number of distinct paths in a {{m}}x{{n}} grid with obstacles: {{result}}')\n"
            "```\n\n"
            f"✅ Final Answer: {result}"
        )

     # --- Question 3: Maximum profit from buying and selling stock ---
    elif "maximum profit that can be obtained from buying and selling a stock with prices given in an array" in q:
        match = re.search(r"array \[([\d,\s]+)\]", q)
        if match:
            prices_str = match.group(1)
            prices = [int(p.strip()) for p in prices_str.split(',')]
        else:
            # Default example prices if not explicitly given in the question
            prices = [7, 1, 5, 3, 6, 4] 
        
        memo = {} # Initialize a dictionary for memoization

        def dp(i, holding_stock_flag):
            # Base Case: If we've processed all days
            if i == len(prices):
                return 0
            
            # Check if the result is already memoized
            state = (i, holding_stock_flag)
            if state in memo:
                return memo[state]
            
            # Option 1: Do nothing on the current day
            profit_if_do_nothing = dp(i + 1, holding_stock_flag)
            
            # Option 2: Perform an action (buy or sell)
            profit_if_action = 0
            if holding_stock_flag: # If we are holding a stock, we can sell it
                # Selling: Current price + max profit from next day (now we can buy again)
                profit_if_action = prices[i] + dp(i + 1, False)
            else: # If we are not holding a stock, we can buy it
                # Buying: -Current price + max profit from next day (now we are holding)
                profit_if_action = -prices[i] + dp(i + 1, True)
            
            # Memoize the maximum profit for the current state
            memo[state] = max(profit_if_do_nothing, profit_if_action)
            return memo[state]

        # Start the DP process: on day 0, we are not holding any stock (False)
        result = dp(0, False)
        
        return (
            f"👨‍🏫 To determine the maximum profit that can be obtained from buying and selling a stock with prices given in an array {prices}, we use memoization.\n"
            f"This problem assumes unlimited transactions (you can buy and sell multiple times), but you must sell a stock before buying another. We track the current day and whether we are holding a stock.\n\n"
            f"🪜 Logic:\n"
            f"- `dp(i, holding)`: Max profit from day `i` onwards, given `holding` status.\n"
            f"- If `holding` is True (sell): `max(prices[i] + dp(i+1, False), dp(i+1, True))`\n"
            f"- If `holding` is False (buy): `max(-prices[i] + dp(i+1, True), dp(i+1, False))`\n"
            f"- Base case: `dp(len(prices), any_holding_status) = 0`\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "memo = {} # Initialize a dictionary for memoization\n\n"
            "def dp(i, holding_stock_flag):\n"
            "    # Base Case: If we've processed all days\n"
            "    if i == len(prices):\n" # Uses 'prices' from outer scope
            "        return 0\n"
            "    \n"
            "    # Check if the result is already memoized\n"
            "    state = (i, holding_stock_flag)\n"
            "    if state in memo:\n"
            "        return memo[state]\n"
            "    \n"
            "    # Option 1: Do nothing on the current day\n"
            "    profit_if_do_nothing = dp(i + 1, holding_stock_flag)\n"
            "    \n"
            "    # Option 2: Perform an action (buy or sell)\n"
            "    profit_if_action = 0\n"
            "    if holding_stock_flag: # If we are holding a stock, we can sell it\n"
            "        profit_if_action = prices[i] + dp(i + 1, False)\n" # Uses 'prices[i]'
            "    else: # If we are not holding a stock, we can buy it\n"
            "        profit_if_action = -prices[i] + dp(i + 1, True)\n" # Uses 'prices[i]'
            "    \n"
            "    # Memoize the maximum profit for the current state\n"
            "    memo[state] = max(profit_if_do_nothing, profit_if_action)\n"
            "    return memo[state]\n\n"
            f"# Input for this specific question:\n"
            f"prices = {prices}\n"
            f"result = dp(0, False) # Start on day 0, not holding any stock\n"
            f"print(f'Maximum profit from buying and selling stock with prices {{prices}}: {{result}}')\n"
            "```\n\n"
            f"✅ Final Answer: {result}"
        )

     # --- Question 4: Number of unique binary search trees ---
    elif "calculate the number of unique binary search trees that can be constructed using" in q and "nodes" in q:
        match = re.search(r"using (\d+) nodes", q)
        if match:
            n = int(match.group(1))
        else:
            # Default example value if not explicitly given
            n = 3
        
        memo = {} # Initialize a dictionary for memoization

        def dp(num_nodes_val):
            # Base cases: 0 nodes or 1 node each have 1 unique BST (empty tree or single node tree)
            if num_nodes_val <= 1:
                return 1
            
            # Check if the result is already memoized
            if num_nodes_val in memo:
                return memo[num_nodes_val]
            
            total = 0
            # Iterate through all possible root nodes: 'i' represents the number of nodes
            # in the left subtree (from 0 to num_nodes_val - 1).
            # The right subtree will then have (num_nodes_val - 1 - i) nodes.
            for i in range(num_nodes_val): 
                left_subtrees_ways = dp(i)
                right_subtrees_ways = dp(num_nodes_val - 1 - i)
                total += left_subtrees_ways * right_subtrees_ways
            
            # Memoize the result
            memo[num_nodes_val] = total
            return total

        result = dp(n)
        
        return (
            f"👨‍🏫 To calculate the number of unique binary search trees that can be constructed using {n} nodes, we use memoization.\n"
            f"This problem is a classic application of dynamic programming, famously solved by Catalan numbers. The approach involves considering each node as a potential root and recursively calculating the combinations of left and right subtrees.\n\n"
            f"🪜 Logic:\n"
            f"- `dp(k) = Σ (dp(i) * dp(k-1-i))` for `i` from `0` to `k-1`\n"
            f"- Base cases: `dp(0) = 1`, `dp(1) = 1`\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "memo = {} # Initialize a dictionary for memoization\n\n"
            "def dp(num_nodes_val):\n"
            "    # Base cases: 0 nodes or 1 node each have 1 unique BST\n"
            "    if num_nodes_val <= 1:\n"
            "        return 1\n"
            "    \n"
            "    # Check if the result is already memoized\n"
            "    if num_nodes_val in memo:\n"
            "        return memo[num_nodes_val]\n"
            "    \n"
            "    total = 0\n"
            "    # Iterate through all possible root nodes: 'i' is the number of nodes in left subtree\n"
            "    for i in range(num_nodes_val): \n"
            "        left_subtrees_ways = dp(i)\n"
            "        right_subtrees_ways = dp(num_nodes_val - 1 - i)\n"
            "        total += left_subtrees_ways * right_subtrees_ways\n"
            "    \n"
            "    # Memoize the result\n"
            "    memo[num_nodes_val] = total\n"
            "    return total\n\n"
            f"# Input for this specific question:\n"
            f"n = {n}\n"
            f"result = dp(n)\n"
            f"print(f'Number of unique binary search trees for {{n}} nodes: {{result}}')\n"
            "```\n\n"
            f"✅ Final Answer: {result}"
        )

    # --- Question 5: Minimum operations to convert string (Edit Distance) ---
    elif "find the minimum number of operations required to convert string" in q and "to string" in q and "using insertion, deletion, and substitution" in q:
        match = re.search(r"convert string (\w+) to string (\w+)", q)
        if match:
            str1 = match.group(1)
            str2 = match.group(2)
        else:
            # Default example values if not explicitly given
            str1, str2 = "horse", "ros"
        
        memo = {} # Initialize a dictionary for memoization

        def dp(i, j):
            # Base Case 1: If str1 is exhausted, we need to insert remaining characters of str2
            if i == len(str1): # Uses 'str1' from outer scope
                return len(str2) - j # Uses 'str2' from outer scope
            # Base Case 2: If str2 is exhausted, we need to delete remaining characters of str1
            if j == len(str2): # Uses 'str2' from outer scope
                return len(str1) - i # Uses 'str1' from outer scope
            
            # Check if the result is already memoized
            state = (i, j)
            if state in memo:
                return memo[state]
            
            cost = 0
            if str1[i] == str2[j]: # Compare characters
                # Characters match, no operation needed, move to next
                cost = dp(i + 1, j + 1)
            else:
                # Characters mismatch, consider all 3 operations + 1 cost
                insert_op = 1 + dp(i, j + 1)      # Insert s2[j] into s1 (effectively delete from s2)
                delete_op = 1 + dp(i + 1, j)      # Delete s1[i] from s1
                replace_op = 1 + dp(i + 1, j + 1) # Replace s1[i] with s2[j]
                cost = min(insert_op, delete_op, replace_op)
            
            # Memoize the result
            memo[state] = cost
            return cost

        result = dp(0, 0) # Start from the beginning of both strings (index 0, 0)
        
        return (
            f"👨‍🏫 To find the minimum number of operations (insertions, deletions, or substitutions) required to convert string '{str1}' to string '{str2}', we use memoization. This is known as the Edit Distance (or Levenshtein Distance) problem.\n"
            f"It involves recursively comparing characters and choosing the operation that leads to the minimum total cost. Memoization efficiently stores the minimum edit distances for all pairs of substrings.\n\n"
            f"🪜 Logic:\n"
            f"- `dp(i, j)`: Minimum edits to convert `str1[i:]` to `str2[j:]`.\n"
            f"- If `str1[i] == str2[j]`: `dp(i+1, j+1)`\n"
            f"- Else: `1 + min(dp(i, j+1) (insert), dp(i+1, j) (delete), dp(i+1, j+1) (substitute))`\n"
            f"- Base cases: If `i == len(str1)`, return `len(str2) - j`; if `j == len(str2)`, return `len(str1) - i`.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "memo = {} # Initialize a dictionary for memoization\n\n"
            "def dp(i, j):\n"
            "    # Base Case 1: If str1 is exhausted, we need to insert remaining characters of str2\n"
            "    if i == len(str1):\n" # Uses 'str1' from outer scope
            "        return len(str2) - j # Uses 'str2' from outer scope\n"
            "    # Base Case 2: If str2 is exhausted, we need to delete remaining characters of str1\n"
            "    if j == len(str2):\n" # Uses 'str2' from outer scope
            "        return len(str1) - i # Uses 'str1' from outer scope\n"
            "    \n"
            "    # Check if the result is already memoized\n"
            "    state = (i, j)\n"
            "    if state in memo:\n"
            "        return memo[state]\n"
            "    \n"
            "    cost = 0\n"
            "    if str1[i] == str2[j]: # Compare characters\n"
            "        cost = dp(i + 1, j + 1)\n"
            "    else:\n"
            "        insert_op = 1 + dp(i, j + 1)\n"
            "        delete_op = 1 + dp(i + 1, j)\n"
            "        replace_op = 1 + dp(i + 1, j + 1)\n"
            "        cost = min(insert_op, delete_op, replace_op)\n"
            "    \n"
            "    # Memoize the result\n"
            "    memo[state] = cost\n"
            "    return cost\n\n"
            f"# Input for this specific question:\n"
            f"str1 = '{str1}'\n"
            f"str2 = '{str2}'\n"
            f"result = dp(0, 0) # Start from the beginning of both strings\n"
            f"print(f'Minimum number of operations to convert \"{{str1}}\" to \"{{str2}}\": {{result}}')\n"
            "```\n\n"
            f"✅ Final Answer: {result}"
        )

    elif "climb a staircase with" in q and "steps, where you can take up to" in q and "steps at a time" in q:
        match = re.search(r"climb a staircase with (\d+) steps, where you can take up to (\d+) steps at a time", q)
        if match:
            n = int(match.group(1))
            max_steps = int(match.group(2))
        else:
            # Fallback if parsing fails. In a real system, you might handle this more robustly.
            n, max_steps = 5, 3 

        # --- Formatted Explanation (as comments) ---
        # 👨‍🏫 Explanation:
        # This problem is a generalized version of the classic staircase problem. We need to find the total number of unique ways to climb `n` steps, given that we can take any number of steps from 1 up to `max_steps` at a time. The solution uses a recursive approach with memoization. To find the number of ways to reach step `i`, we sum the number of ways to reach `i-1`, `i-2`, ..., up to `i-max_steps`. Memoization is crucial here as it stores the computed number of ways for each intermediate step, preventing redundant calculations and significantly improving performance.

        # 🪜 Logic (as comments):
        # The recurrence relation for the number of ways to reach step `i` is:
        # * `dp(i) = Σ dp(i - k)` for `k` from `1` to `max_steps`
        # * Base cases:
        #     * `dp(0) = 1` (There's one way to be at step 0: do nothing or you are already there).
        #     * `dp(negative_i) = 0` (You cannot reach a negative step).

        # --- Actual Code and Computation ---
        memo = {} # Initialize a dictionary for memoization

        def dp(i):
            # Base cases
            if i == 0:
                return 1
            if i < 0:
                return 0
            
            # Check if the result is already memoized
            if i in memo:
                return memo[i]
            
            total_ways = 0
            # Iterate through each possible step size from 1 up to max_steps
            # (max_steps is accessible from the outer scope as a captured variable)
            for step_size in range(1, max_steps + 1):
                total_ways += dp(i - step_size)
            
            # Memoize the result
            memo[i] = total_ways
            return total_ways
            
        
        result = dp(n) # Compute the result for the given n

        # --- Final Answer Formatting (as comments and print statement) ---
        response_output = (
            f"👨‍🏫 Explanation:\n"
            f"This problem is a generalized version of the classic staircase problem. We need to find the total number of unique ways to climb {n} steps, given that we can take any number of steps from 1 up to {max_steps} at a time. The solution uses a recursive approach with memoization. To find the number of ways to reach step `i`, we sum the number of ways to reach `i-1`, `i-2`, ..., up to `i-max_steps`. Memoization is crucial here as it stores the computed number of ways for each intermediate step, preventing redundant calculations and significantly improving performance.\n\n"
            f"🪜 Logic:\n"
            f"- `dp(i) = Σ dp(i - k)` for `k` from `1` to `{max_steps}`\n"
            f"- Base cases: `dp(0) = 1`, `dp(negative_i) = 0`\n\n"
            f"🛠 Code:\n"
            f"```python\n" # Start of inner code block
            f"import re\n\n"
            f"# This code block is designed to be self-contained and runnable.\n"
            f"# It parses the required 'n' and 'max_steps' from a sample question string for demonstration.\n"
            f"# In your integrated system, these values might be directly available as variables.\n\n"
            f"question_text_internal = \"Using memoization, calculate the total number of unique ways to climb a staircase with {n} steps, where you can take up to {max_steps} steps at a time.\"\n"
            f"q_internal = question_text_internal.lower()\n\n"
            f"match_internal = re.search(r\"climb a staircase with (\\d+) steps, where you can take up to (\\d+) steps at a time\", q_internal)\n"
            f"if match_internal:\n"
            f"    n_computed = int(match_internal.group(1))\n"
            f"    max_steps_computed = int(match_internal.group(2))\n"
            f"else:\n"
            f"    n_computed, max_steps_computed = {n}, {max_steps} # Fallback to already parsed values\n\n"
            f"memo = {{}} # Initialize a dictionary for memoization\n\n"
            f"def dp(i):\n"
            f"    if i == 0:\n"
            f"        return 1\n"
            f"    if i < 0:\n"
            f"        return 0\n"
            f"    if i in memo:\n"
            f"        return memo[i]\n"
            f"    \n"
            f"    total_ways = 0\n"
            f"    for step_size in range(1, max_steps_computed + 1):\n"
            f"        total_ways += dp(i - step_size)\n"
            f"    \n"
            f"    memo[i] = total_ways\n"
            f"    return total_ways\n\n"
            f"result_computed = dp(n_computed)\n"
            f"print(f'Total unique ways to climb {{n_computed}} steps (up to {{max_steps_computed}} steps at a time): {{result_computed}}')\n"
            f"```\n\n" # End of inner code block
            f"✅ Final Answer: {result}"
        )
        return response_output

    elif "maximum sum of non-adjacent elements" in q:
        # Step 3: Extract input values (like array)
        import re
        match = re.search(r"\[([0-9,\s-]+)\]", q)
        if match:
            arr_str = match.group(1)
            try:
                arr = [int(x.strip()) for x in arr_str.split(",")]
            except:
                return "⚠️ Could not parse array values."
        else:
            return "⚠️ No array found in question."

        # Step 4: Run memoized DP logic
        def max_non_adj_sum(arr):
            memo = {}
            def helper(i):
                if i >= len(arr): return 0
                if i in memo: return memo[i]
                include = arr[i] + helper(i + 2)
                exclude = helper(i + 1)
                memo[i] = max(include, exclude)
                return memo[i]
            return helper(0)

        result = max_non_adj_sum(arr)

        # Step 5: Return explanation + result
        return (
            f"👨‍🏫 To find the maximum sum of non-adjacent elements in an array like {arr}, we use memoization.\n\n"
            f"🪜 Logic:\n"
            f"- Include current element and skip next\n"
            f"- Or skip current element\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"def max_non_adj_sum(arr):\n"
            f"    memo = {{}}\n"
            f"    def helper(i):\n"
            f"        if i >= len(arr): return 0\n"
            f"        if i in memo: return memo[i]\n"
            f"        include = arr[i] + helper(i + 2)\n"
            f"        exclude = helper(i + 1)\n"
            f"        memo[i] = max(include, exclude)\n"
            f"        return memo[i]\n"
            f"    return helper(0)\n\n"
            f"arr = {arr}\n"
            f"print(max_non_adj_sum(arr))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )


    elif "Fibonacci number where" in q and "is given" in q:
        match = re.search(r"where (\d+) is given", q)
        if match:
            n = int(match.group(1))
        else:
            return "⚠️ Could not extract 'n' value for Fibonacci from the question."
        
        memo = {}

        def fib(i):
            if i <= 1:
                return i
            if i in memo:
                return memo[i]
            memo[i] = fib(i - 1) + fib(i - 2)
            return memo[i]

        result = fib(n)

        return (
            f"👨‍🏫 Explanation:\n"
            f"The Fibonacci sequence is defined as F(n) = F(n-1) + F(n-2) with base cases F(0)=0 and F(1)=1.\n"
            f"To compute F({n}), we use memoization to store previously computed values and avoid redundant calculations.\n\n"
            f"🪜 Logic:\n"
            f"- F(n) = F(n-1) + F(n-2)\n"
            f"- Base cases:\n"
            f"  - F(0) = 0\n"
            f"  - F(1) = 1\n"
            f"- Use a dictionary to memoize the results.\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def fib(n):\n"
            f"    if n <= 1:\n"
            f"        return n\n"
            f"    if n in memo:\n"
            f"        return memo[n]\n"
            f"    memo[n] = fib(n-1) + fib(n-2)\n"
            f"    return memo[n]\n"
            f"\n"
            f"print(fib({n}))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )


    elif "number of distinct paths to reach the top of a grid" in q and "bottom-left" in q:
        match = re.search(r"dimensions\s*(\d+)\s*[x×]\s*(\d+)", q)
        if match:
            m, n = int(match.group(1)), int(match.group(2))
        else:
            return "⚠️ Could not parse grid dimensions (m x n)."

        memo = {}

        def dp(i, j):
            if i >= m or j >= n:
                return 0
            if i == m - 1 and j == n - 1:
                return 1
            if (i, j) in memo:
                return memo[(i, j)]
            memo[(i, j)] = dp(i + 1, j) + dp(i, j + 1)
            return memo[(i, j)]

        result = dp(0, 0)

        return (
            f"👨‍🏫 Explanation:\n"
            f"We need to move from the bottom-left (0,0) to the top-right ({m-1},{n-1}) only moving right or up.\n"
            f"We use memoization to store results of each cell to avoid recomputation.\n\n"
            f"🪜 Logic:\n"
            f"- dp(i,j) = dp(i+1,j) + dp(i,j+1)\n"
            f"- Stop when reaching the bottom-right (goal)\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def dp(i, j):\n"
            f"    if i >= {m} or j >= {n}: return 0\n"
            f"    if i == {m-1} and j == {n-1}: return 1\n"
            f"    if (i, j) in memo: return memo[(i, j)]\n"
            f"    memo[(i, j)] = dp(i+1, j) + dp(i, j+1)\n"
            f"    return memo[(i, j)]\n"
            f"print(dp(0, 0))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )


    elif "grid" in q and "obstacles" in q and "from the top-left to the bottom-right" in q:
    # Sample 3x3 grid with obstacles (0: free, 1: obstacle)
        grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
        m, n = len(grid), len(grid[0])
        memo = {}

        def count_paths(i, j):
            if i >= m or j >= n:
                return 0
            if grid[i][j] == 1:
                return 0
            if i == m - 1 and j == n - 1:
                return 1
            if (i, j) in memo:
                return memo[(i, j)]
            memo[(i, j)] = count_paths(i + 1, j) + count_paths(i, j + 1)
            return memo[(i, j)]

        result = count_paths(0, 0)

        return (
            f"👨‍🏫 Explanation:\n"
            f"You're given a grid of 0s (free) and 1s (obstacles). The goal is to count the number of distinct paths from the top-left (0,0) to the bottom-right ({m-1},{n-1}) using memoization.\n"
            f"You can only move right or down, and cannot pass through cells with a 1.\n\n"
            f"🪜 Logic:\n"
            f"- dp(i,j) = 0 if out of bounds or obstacle\n"
            f"- If at destination: return 1\n"
            f"- Else: dp(i,j) = dp(i+1,j) + dp(i,j+1)\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"grid = {grid}\n"
            f"memo = {{}}\n"
            f"def count_paths(i, j):\n"
            f"    if i >= {m} or j >= {n}:\n"
            f"        return 0\n"
            f"    if grid[i][j] == 1:\n"
            f"        return 0\n"
            f"    if i == {m - 1} and j == {n - 1}:\n"
            f"        return 1\n"
            f"    if (i, j) in memo:\n"
            f"        return memo[(i, j)]\n"
            f"    memo[(i, j)] = count_paths(i + 1, j) + count_paths(i, j + 1)\n"
            f"    return memo[(i, j)]\n"
            f"print(count_paths(0, 0))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )


# 
    # --- Question 1: Unique ways to partition a number into a sum of integers ---
    elif "partition the number" in q and "into" in q and "sum of" in q:
        match = re.search(r"partition the number\s+(\d+)", q)
        if match:
            n = int(match.group(1))
        else:
            return "⚠️ Could not parse the value of n for partition problem."

        memo = {}

        def count_partitions(n, max_num):
            if n == 0:
                return 1
            if n < 0 or max_num == 0:
                return 0
            if (n, max_num) in memo:
                return memo[(n, max_num)]
            include = count_partitions(n - max_num, max_num)
            exclude = count_partitions(n, max_num - 1)
            memo[(n, max_num)] = include + exclude
            return memo[(n, max_num)]

        result = count_partitions(n, n)

        return (
            f"👨‍🏫 Explanation:\n"
            f"We are asked to count the number of integer partitions of {n} using memoization.\n"
            f"This problem breaks down into including and excluding the largest number allowed at each step.\n\n"
            f"🪜 Logic:\n"
            f"- dp(n, max) = dp(n - max, max) + dp(n, max - 1)\n"
            f"- Base cases:\n"
            f"  - dp(0, _) = 1 (only one way to partition 0)\n"
            f"  - dp(n < 0, _) or dp(_, max == 0) = 0\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def count_partitions(n, max_num):\n"
            f"    if n == 0:\n"
            f"        return 1\n"
            f"    if n < 0 or max_num == 0:\n"
            f"        return 0\n"
            f"    if (n, max_num) in memo:\n"
            f"        return memo[(n, max_num)]\n"
            f"    include = count_partitions(n - max_num, max_num)\n"
            f"    exclude = count_partitions(n, max_num - 1)\n"
            f"    memo[(n, max_num)] = include + exclude\n"
            f"    return memo[(n, max_num)]\n"
            f"\nprint(count_partitions({n}, {n}))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )


    elif "climb a staircase" in q and "you can take 1 or 2 steps at a time" in q:
        match = re.search(r"climb a staircase with (\d+)", q)
        if match:
            n = int(match.group(1))
        else:
            return "⚠️ Could not parse number of steps."

        memo = {}
        def dp(i):
            if i == 0:
                return 1
            if i < 0:
                return 0
            if i in memo:
                return memo[i]
            memo[i] = dp(i - 1) + dp(i - 2)
            return memo[i]

        result = dp(n)

        return (
            f"👨‍🏫 Explanation:\n"
            f"This is a staircase problem where you're allowed to take 1 or 2 steps at a time. We solve it using memoization to avoid recomputation.\n\n"
            f"🪜 Logic:\n"
            f"- dp(i) = dp(i-1) + dp(i-2)\n"
            f"- Base cases:\n"
            f"  - dp(0) = 1 (1 way to stay at the bottom)\n"
            f"  - dp(i<0) = 0\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def dp(i):\n"
            f"    if i == 0:\n"
            f"        return 1\n"
            f"    if i < 0:\n"
            f"        return 0\n"
            f"    if i in memo:\n"
            f"        return memo[i]\n"
            f"    memo[i] = dp(i - 1) + dp(i - 2)\n"
            f"    return memo[i]\n"
            f"print(dp({n}))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )



# refined question
    elif "maximum profit" in q and "at most" in q and "times" in q:
        # Extract array
        match_arr = re.search(r"array of integers\s*\[([^\]]+)\]", question)
        match_k = re.search(r"at most (\d+) times", question)
        if match_arr and match_k:
            arr_str = match_arr.group(1)
            try:
                prices = [int(x.strip()) for x in arr_str.split(",")]
            except:
                return "⚠️ Could not parse array values."
            k = int(match_k.group(1))
        else:
            return "⚠️ Could not extract array or k from question."

        # Memoized trading function with up to k transactions
        def max_profit_k_times(prices, k):
            memo = {}
            def dp(i, holding, transactions_left):
                if i == len(prices) or transactions_left == 0:
                    return 0
                if (i, holding, transactions_left) in memo:
                    return memo[(i, holding, transactions_left)]
                skip = dp(i + 1, holding, transactions_left)
                if holding:
                    # sell now
                    sell = prices[i] + dp(i + 1, False, transactions_left - 1)
                    memo[(i, holding, transactions_left)] = max(sell, skip)
                else:
                    # buy now
                    buy = -prices[i] + dp(i + 1, True, transactions_left)
                    memo[(i, holding, transactions_left)] = max(buy, skip)
                return memo[(i, holding, transactions_left)]
            return dp(0, False, k)

        result = max_profit_k_times(prices, k)

        return (
            f"👨‍🏫 This is a stock trading problem where you're allowed to make at most {k} transactions.\n"
            f"Goal: Maximize profit using at most {k} buy+sell pairs.\n\n"
            f"🪜 Idea:\n"
            f"- At each day, you can buy, sell, or skip.\n"
            f"- State: (day index, holding?, transactions left)\n"
            f"- Use memoization to cache overlapping subproblems.\n\n"
            f"🛠 Python Code:\n"
            "```python\n"
            "def max_profit_k_times(prices, k):\n"
            "    memo = {}\n"
            "    def dp(i, holding, transactions_left):\n"
            "        if i == len(prices) or transactions_left == 0:\n"
            "            return 0\n"
            "        if (i, holding, transactions_left) in memo:\n"
            "            return memo[(i, holding, transactions_left)]\n"
            "        skip = dp(i + 1, holding, transactions_left)\n"
            "        if holding:\n"
            "            sell = prices[i] + dp(i + 1, False, transactions_left - 1)\n"
            "            memo[(i, holding, transactions_left)] = max(sell, skip)\n"
            "        else:\n"
            "            buy = -prices[i] + dp(i + 1, True, transactions_left)\n"
            "            memo[(i, holding, transactions_left)] = max(buy, skip)\n"
            "        return memo[(i, holding, transactions_left)]\n"
            "    return dp(0, False, k)\n\n"
            f"prices = {prices}\n"
            f"k = {k}\n"
            "print('Maximum profit with at most', k, 'transactions:', max_profit_k_times(prices, k))\n"
            "```\n\n"
            f"📌 Example: For prices = {prices}, with at most {k} transactions → ✅ Final Answer: {result}"
        )

    elif (
    "minimum number of edits required to convert string" in q
    and "into string" in q
    and "insertion" in q
    and "deletion" in q
    and "substitution" in q
):
        match = re.search(r"convert string (\w+) into string (\w+)", q)
        if match:
            str1, str2 = match.group(1), match.group(2)
        else:
            return "⚠️ Could not parse the input strings."

        memo = {}
        def dp(i, j):
            if i == 0: return j
            if j == 0: return i
            if (i, j) in memo: return memo[(i, j)]

            if str1[i - 1] == str2[j - 1]:
                memo[(i, j)] = dp(i - 1, j - 1)
            else:
                insert = dp(i, j - 1)
                delete = dp(i - 1, j)
                substitute = dp(i - 1, j - 1)
                memo[(i, j)] = 1 + min(insert, delete, substitute)
            return memo[(i, j)]

        result = dp(len(str1), len(str2))

        return (
            f"👨‍🏫 Explanation:\n"
            f"We use memoization to compute the minimum edit distance between two strings: '{str1}' and '{str2}'.\n"
            f"Allowed operations are:\n"
            f"1. Insertion\n"
            f"2. Deletion\n"
            f"3. Substitution\n\n"
            f"🪜 Logic:\n"
            f"- dp(i,j) = minimum of:\n"
            f"    - dp(i-1, j) + 1   (deletion)\n"
            f"    - dp(i, j-1) + 1   (insertion)\n"
            f"    - dp(i-1, j-1) + (1 if characters differ else 0)\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def dp(i, j):\n"
            f"    if i == 0: return j\n"
            f"    if j == 0: return i\n"
            f"    if (i, j) in memo: return memo[(i, j)]\n"
            f"    if str1[i - 1] == str2[j - 1]:\n"
            f"        memo[(i, j)] = dp(i - 1, j - 1)\n"
            f"    else:\n"
            f"        insert = dp(i, j - 1)\n"
            f"        delete = dp(i - 1, j)\n"
            f"        substitute = dp(i - 1, j - 1)\n"
            f"        memo[(i, j)] = 1 + min(insert, delete, substitute)\n"
            f"    return memo[(i, j)]\n\n"
            f"str1 = '{str1}'\n"
            f"str2 = '{str2}'\n"
            f"print(dp(len(str1), len(str2)))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )


def answer_memoization_quantitative_lvl2(question):
    import re
    import math
    q = question.lower()

# 🔢 Q1. Climb stairs with rest after every k steps

    if "rest after every" in q and "staircase" in q:
        match = re.search(r"staircase with (\d+) steps.*?every (\d+) steps", q)
        if not match:
            return "❌ Couldn't extract step and rest values."
        n, k = int(match.group(1)), int(match.group(2))
        return (
            f"👨‍🏫 You must climb {n} steps taking up to some max_steps at a time and rest after every {k} consecutive steps.\n"
            f"We use an extra state rest = count of consecutive steps since last rest.\n\n"
            f"🪜 Base Case: If rest == k, rest is forced (reset rest).\n"
            f"State: dp(pos, rest) = Σ dp(pos - step, rest + step) for step in 1..max_steps\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def climb_with_rest(n, max_steps, k):\n"
            "    memo = {}\n"
            "    def dp(i, rest):\n"
            "        if i == 0: return 1\n"
            "        if rest == k: rest = 0\n"
            "        if (i, rest) in memo: return memo[(i, rest)]\n"
            "        total = 0\n"
            "        for step in range(1, max_steps+1):\n"
            "            if i - step >= 0:\n"
            "                total += dp(i - step, rest + step)\n"
            "        memo[(i, rest)] = total\n"
            "        return total\n"
            "    return dp(n, 0)\n"
            "```"
        )

# 🔢 Q2. Min deletions to make array strictly increasing

    elif "minimum number of deletions required to make the array strictly increasing" in q:
        match = re.search(r"\[([0-9,\s-]+)\]", question)
        if not match:
            return "⚠️ Please include the actual array in the question like: [3, 2, 5, 1, 4]"
        arr = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]

        memo = {}
        def lis(i, prev):
            if i == len(arr):
                return 0
            key = (i, prev)
            if key in memo:
                return memo[key]
            take = 0
            if prev == -1 or arr[i] > arr[prev]:
                take = 1 + lis(i + 1, i)
            not_take = lis(i + 1, prev)
            memo[key] = max(take, not_take)
            return memo[key]

        lis_len = lis(0, -1)
        deletions = len(arr) - lis_len

        return (
            f"👨‍🏫 To make the array strictly increasing, we calculate the Longest Increasing Subsequence (LIS), then subtract its length from total elements.\n\n"
            f"Array: {arr}\n"
            f"LIS Length: {lis_len}\n"
            f"Deletions Needed: {deletions}\n\n"
            f"🪜 Logic:\n"
            f"- Use memoized recursion to compute LIS.\n"
            f"- Minimum deletions = len(arr) - LIS length\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"arr = {arr}\n"
            f"memo = {{}}\n"
            f"def lis(i, prev):\n"
            f"    if i == len(arr): return 0\n"
            f"    key = (i, prev)\n"
            f"    if key in memo: return memo[key]\n"
            f"    take = 0\n"
            f"    if prev == -1 or arr[i] > arr[prev]:\n"
            f"        take = 1 + lis(i + 1, i)\n"
            f"    not_take = lis(i + 1, prev)\n"
            f"    memo[key] = max(take, not_take)\n"
            f"    return memo[key]\n"
            f"\n"
            f"lis_len = lis(0, -1)\n"
            f"deletions = len(arr) - lis_len\n"
            f"print(deletions)\n"
            f"```\n\n"
            f"✅ Final Answer: {deletions}"
        )



# 🔢 Q3. Max sum of non-adjacent elements (can be negative)

    elif "maximum sum of non-adjacent elements" in q:
        match = re.search(r"\[([0-9,\s-]+)\]", question)
        if not match:
            return "⚠️ Could not find array in the question."
        arr = [int(x.strip()) for x in match.group(1).split(",")]

        memo = {}

        def dp(i):
            if i >= len(arr):
                return 0
            if i in memo:
                return memo[i]
            include = arr[i] + dp(i + 2)
            exclude = dp(i + 1)
            memo[i] = max(include, exclude)
            return memo[i]

        result = dp(0)

        return (
            f"👨‍🏫 To find the maximum sum of non-adjacent elements (including negative numbers), we use recursion with memoization.\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def dp(i):\n"
            f"    if i >= len(arr): return 0\n"
            f"    if i in memo: return memo[i]\n"
            f"    include = arr[i] + dp(i + 2)\n"
            f"    exclude = dp(i + 1)\n"
            f"    memo[i] = max(include, exclude)\n"
            f"    return memo[i]\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )


# 🔢 Q4. Min operations to make array non-decreasing

    elif "minimum number of operations to convert the array into a non-decreasing order" in q:
        match = re.search(r"\[([0-9,\s-]+)\]", question)
        if not match:
            return "⚠️ Could not extract array."
        arr = [int(x.strip()) for x in match.group(1).split(",")]

        memo = {}

        def dp(i, prev):
            if i == len(arr):
                return 0
            key = (i, prev)
            if key in memo:
                return memo[key]
            if arr[i] >= prev:
                memo[key] = dp(i + 1, arr[i])
            else:
                # Either delete this or change it to prev
                delete = 1 + dp(i + 1, prev)
                change = 1 + dp(i + 1, prev)
                memo[key] = min(delete, change)
            return memo[key]

        result = dp(0, float('-inf'))

        return (
            f"👨‍🏫 We are computing the minimum operations to make an array non-decreasing.\n"
            f"Options: delete or replace the current value if it violates order.\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def dp(i, prev):\n"
            f"    if i == len(arr): return 0\n"
            f"    key = (i, prev)\n"
            f"    if key in memo: return memo[key]\n"
            f"    if arr[i] >= prev:\n"
            f"        memo[key] = dp(i + 1, arr[i])\n"
            f"    else:\n"
            f"        delete = 1 + dp(i + 1, prev)\n"
            f"        change = 1 + dp(i + 1, prev)\n"
            f"        memo[key] = min(delete, change)\n"
            f"    return memo[key]\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
    )

    # 🔢 Question 5: Distinct arrangements with identical items
    elif "arrange" in q and "objects from a set of" in q:
        match = re.search(r"arrange (\d+) objects from a set of (\d+)", q)
        if not match:
            return "⚠️ Could not parse k and n."
        k = int(match.group(1))
        n = int(match.group(2))

        from math import comb
        result = comb(n, k)

        return (
            f"👨‍🏫 To arrange {k} objects from {n} total (assuming some are identical), we consider combinations if objects are identical.\n\n"
            f"🪜 Logic: Use combinations C(n, k) = n! / (k!(n-k)!)\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"from math import comb\n"
            f"print(comb({n}, {k}))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )

# 🔢 Question 6: Number of unique binary search trees with n distinct nodes
    elif "unique binary search trees" in q:
        match = re.search(r"formed using (\d+) distinct nodes", q)
        if not match:
            return "⚠️ Could not find value of n."
        n = int(match.group(1))

        memo = {}

        def dp(n):
            if n <= 1:
                return 1
            if n in memo:
                return memo[n]
            total = 0
            for i in range(n):
                total += dp(i) * dp(n - i - 1)
            memo[n] = total
            return total

        result = dp(n)

        return (
            f"👨‍🏫 The number of unique Binary Search Trees (BSTs) formed from {n} distinct nodes is the nth Catalan number.\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def dp(n):\n"
            f"    if n <= 1: return 1\n"
            f"    if n in memo: return memo[n]\n"
            f"    total = 0\n"
            f"    for i in range(n):\n"
            f"        total += dp(i) * dp(n - i - 1)\n"
            f"    memo[n] = total\n"
            f"    return total\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )
# 🔢 Question 7: Ways to make the sum with coin denominations
    elif "ways to make the sum using the coins" in q:
        match = re.search(r"coin denominations\s*\[([0-9,\s]+)\].*?sum (\d+)", question)
        if not match:
            return "⚠️ Could not extract coins or target sum."
        coins = [int(x.strip()) for x in match.group(1).split(",")]
        target = int(match.group(2))

        memo = {}

        def dp(i, t):
            if t == 0: return 1
            if i == len(coins) or t < 0: return 0
            if (i, t) in memo: return memo[(i, t)]
            memo[(i, t)] = dp(i, t - coins[i]) + dp(i + 1, t)
            return memo[(i, t)]

        result = dp(0, target)

        return (
            f"👨‍🏫 We are computing the number of ways to make sum {target} using coins {coins}.\n\n"
            f"🪜 Logic:\n"
            f"- At each step, include or exclude the current coin.\n"
            f"- Use memoization with (i, t) = (coin index, remaining sum).\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"coins = {coins}\n"
            f"target = {target}\n"
            f"memo = {{}}\n"
            f"def dp(i, t):\n"
            f"    if t == 0: return 1\n"
            f"    if i == len(coins) or t < 0: return 0\n"
            f"    if (i, t) in memo: return memo[(i, t)]\n"
            f"    memo[(i, t)] = dp(i, t - coins[i]) + dp(i + 1, t)\n"
            f"    return memo[(i, t)]\n"
            f"\n"
            f"print(dp(0, target))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )


# 🔢 Question 8: Combinations to reach the target sum from a set
    elif "number of possible combinations to reach the target sum" in q:
        match = re.search(r"target sum (\d+).*?\[([0-9,\s-]+)\]", question)
        if not match:
            return "⚠️ Could not extract target or array."
        target = int(match.group(1))
        arr = [int(x.strip()) for x in match.group(2).split(",")]

        memo = {}
        def dp(t):
            if t == 0: return 1
            if t < 0: return 0
            if t in memo: return memo[t]
            total = 0
            for num in arr:
                total += dp(t - num)
            memo[t] = total
            return total

        result = dp(target)

        return (
            f"👨‍🏫 To calculate the number of possible combinations to reach the target sum {target} using elements from {arr}, "
            f"we use a recursive approach with memoization to avoid redundant calculations.\n\n"
            f"🪜 Logic:\n"
            f"- Try every number from the array and reduce the target accordingly.\n"
            f"- Use memoization to cache results for a given remaining target.\n"
            f"- Base case: dp(0) = 1 (one way to make sum 0), dp(<0) = 0\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"arr = {arr}\n"
            f"target = {target}\n"
            f"memo = {{}}\n"
            f"def dp(t):\n"
            f"    if t == 0: return 1\n"
            f"    if t < 0: return 0\n"
            f"    if t in memo: return memo[t]\n"
            f"    total = 0\n"
            f"    for num in arr:\n"
            f"        total += dp(t - num)\n"
            f"    memo[t] = total\n"
            f"    return total\n"
            f"\n"
            f"print(dp(target))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )
    
#  Question 9: Grid with blocked cells to top-right
    elif "grid of size" in q and "list of blocked cells" in q:
        match = re.search(r"grid of size (\d+)x(\d+).*?\[([0-9,\(\)\s]+)\]", question)
        if not match:
            return "⚠️ Could not extract grid size or blocked cells."
        m, n = int(match.group(1)), int(match.group(2))
        blocked_str = match.group(3)
        blocked = eval(f"[{blocked_str}]")

        blocked_set = set(blocked)
        memo = {}

        def dp(i, j):
            if (i, j) in blocked_set: return 0
            if i >= m or j >= n: return 0
            if (i, j) == (m - 1, n - 1): return 1
            if (i, j) in memo: return memo[(i, j)]
            memo[(i, j)] = dp(i + 1, j) + dp(i, j + 1)
            return memo[(i, j)]

        result = dp(0, 0)

        return (
            f"👨‍🏫 We are finding the number of distinct paths in a {m}x{n} grid avoiding blocked cells: {blocked}.\n\n"
            f"🪜 Logic:\n"
            f"- Start at (0, 0) and only move right or down.\n"
            f"- If a cell is blocked, it contributes 0 paths.\n"
            f"- Use memoization to store results for each (i, j).\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"m, n = {m}, {n}\n"
            f"blocked = {blocked}\n"
            f"blocked_set = set(blocked)\n"
            f"memo = {{}}\n"
            f"def dp(i, j):\n"
            f"    if (i, j) in blocked_set: return 0\n"
            f"    if i >= m or j >= n: return 0\n"
            f"    if (i, j) == (m - 1, n - 1): return 1\n"
            f"    if (i, j) in memo: return memo[(i, j)]\n"
            f"    memo[(i, j)] = dp(i + 1, j) + dp(i, j + 1)\n"
            f"    return memo[(i, j)]\n"
            f"\n"
            f"print(dp(0, 0))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )
    

    # 🔢 Question 10: Minimum number of steps in a maze with obstacles
    elif "minimum number of steps" in q and "matrix with obstacles" in q:
        # Step 1: Extract the grid (matrix)
        match = re.search(r"matrix.*(\[\[.*\]\])", question)
        if not match:
            return "⚠️ Could not extract the obstacle matrix. Please format like: matrix [[0,0,0],[0,1,0],[0,0,0]]"
        try:
            grid = eval(match.group(1))
        except:
            return "⚠️ Grid format could not be evaluated. Ensure it's a valid 2D list like [[0,1],[1,0]]"

        m, n = len(grid), len(grid[0])
        memo = {}

        def dp(i, j):
            if i >= m or j >= n or grid[i][j] == 1:
                return float('inf')
            if i == m - 1 and j == n - 1:
                return 0
            if (i, j) in memo:
                return memo[(i, j)]
            memo[(i, j)] = 1 + min(dp(i + 1, j), dp(i, j + 1))
            return memo[(i, j)]

        steps = dp(0, 0)
        result = steps if steps != float('inf') else -1

        return (
            f"👨‍🏫 We are computing the minimum number of steps from top-left to bottom-right of a grid with obstacles.\n\n"
            f"🧱 Grid:\n"
            f"{grid}\n\n"
            f"🪜 Logic:\n"
            f"- At each cell, recursively try going right or down.\n"
            f"- If a cell has a 1 (obstacle), it's blocked.\n"
            f"- Use memoization to cache (i,j) paths.\n"
            f"- Return ∞ if stuck, 0 if at destination.\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"grid = {grid}\n"
            f"m, n = len(grid), len(grid[0])\n"
            f"memo = {{}}\n"
            f"def dp(i, j):\n"
            f"    if i >= m or j >= n or grid[i][j] == 1: return float('inf')\n"
            f"    if i == m-1 and j == n-1: return 0\n"
            f"    if (i, j) in memo: return memo[(i, j)]\n"
            f"    memo[(i, j)] = 1 + min(dp(i+1, j), dp(i, j+1))\n"
            f"    return memo[(i, j)]\n"
            f"\n"
            f"steps = dp(0, 0)\n"
            f"print(steps if steps != float('inf') else -1)\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )


#  Question 11: Partition set into two equal-sum subsets
    elif "partition a set of" in q and "two subsets" in q and "same" in q:
        import re
        n_match = re.search(r"set of (\d+)", q)
        n = int(n_match.group(1)) if n_match else 5

        total_sum = n * (n + 1) // 2
        if total_sum % 2 != 0:
            return (
                f"🧠 **Problem:** Partition a set of {n} distinct integers into two subsets with equal sum using memoization.\n\n"
                f"🚫 Total sum = {total_sum} is **odd**, so it's **impossible** to partition equally.\n"
                f"✅ Answer: `0`"
            )

        target = total_sum // 2
        memo = {}

        def count_subsets(i, curr_sum):
            if curr_sum == 0:
                return 1
            if i == 0 or curr_sum < 0:
                return 0
            if (i, curr_sum) in memo:
                return memo[(i, curr_sum)]

            # include i or not
            memo[(i, curr_sum)] = count_subsets(i - 1, curr_sum) + count_subsets(i - 1, curr_sum - i)
            return memo[(i, curr_sum)]

        total_ways = count_subsets(n, target)
        result = total_ways // 2  # since each partition pair is counted twice

        return (
            f"🧠 **Problem:** Partition a set of {n} distinct integers into two subsets with equal sum using memoization.\n\n"
            f"💡 We check if the total sum = {total_sum} is even. Then we count the number of subsets that sum to {target} = total_sum / 2.\n"
            f"Each valid subset implies the other one automatically. But we divide the count by 2 to avoid double-counting.\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"total_sum = n * (n + 1) // 2\n"
            f"if total_sum % 2 != 0:\n"
            f"    return 0\n"
            f"target = total_sum // 2\n"
            f"memo = {{}}\n"
            f"def count_subsets(i, curr_sum):\n"
            f"    if curr_sum == 0:\n"
            f"        return 1\n"
            f"    if i == 0 or curr_sum < 0:\n"
            f"        return 0\n"
            f"    if (i, curr_sum) in memo:\n"
            f"        return memo[(i, curr_sum)]\n"
            f"    memo[(i, curr_sum)] = count_subsets(i - 1, curr_sum) + count_subsets(i - 1, curr_sum - i)\n"
            f"    return memo[(i, curr_sum)]\n\n"
            f"print(count_subsets({n}, {target}) // 2)\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n·target), Space: O(n·target)"
        )


# Question 12: Min cost to cut rod with prices
    elif "minimum cost of cutting a rod" in q:
        match_len = re.search(r"rod of length (\d+)", q)
        match_arr = re.search(r"\[([0-9,\s]+)\]", q)

        if not match_len or not match_arr:
            return "⚠️ Could not extract rod length or prices."

        n = int(match_len.group(1))
        price = [int(x.strip()) for x in match_arr.group(1).split(",")]

        memo = {}
        def dp(length):
            if length == 0:
                return 0
            if length in memo:
                return memo[length]
            max_val = float("-inf")
            for i in range(1, min(length, len(price)) + 1):
                max_val = max(max_val, price[i - 1] + dp(length - i))
            memo[length] = max_val
            return max_val

        result = dp(n)

        return (
            f"👨‍🏫 We are cutting a rod of length {n} into smaller pieces to maximize total price based on the price list: {price}.\n\n"
            f"🪜 Logic:\n"
            f"- Try cutting at every length i (1 to n) and compute: price[i-1] + dp(n - i).\n"
            f"- Use memoization to avoid recomputation.\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"price = {price}\n"
            f"n = {n}\n"
            f"memo = {{}}\n"
            f"def dp(length):\n"
            f"    if length == 0: return 0\n"
            f"    if length in memo: return memo[length]\n"
            f"    max_val = float('-inf')\n"
            f"    for i in range(1, min(length, len(price)) + 1):\n"
            f"        max_val = max(max_val, price[i - 1] + dp(length - i))\n"
            f"    memo[length] = max_val\n"
            f"    return max_val\n"
            f"\n"
            f"print(dp(n))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )

    
# Question 13: Select k from n (with duplicates allowed)
    elif "select" in q and "order does not matter" in q:
        import re
        k_match = re.search(r"select (\d+)", q)
        n_match = re.search(r"list of (\d+)", q)
        k = int(k_match.group(1)) if k_match else 3
        n = int(n_match.group(1)) if n_match else 5

        memo = {}
        def choose(n, k):
            if k == 0 or k == n:
                return 1
            if k > n:
                return 0
            if (n, k) in memo:
                return memo[(n, k)]
            memo[(n, k)] = choose(n - 1, k - 1) + choose(n - 1, k)
            return memo[(n, k)]

        result = choose(n, k)

        return (
            f"🧠 **Problem:** Using memoization, find the number of ways to choose {k} items from {n}, where order doesn't matter and some items might be identical.\n\n"
            f"💡 Even if items are identical, the count of unique combinations is determined using binomial coefficients C(n, k). We memoize results for subproblems (n, k).\n\n"
            f"🔧 **Memoized Code:**\n"
            f"```python\n"
            f"memo = {{}}\n"
            f"def choose(n, k):\n"
            f"    if k == 0 or k == n:\n"
            f"        return 1\n"
            f"    if k > n:\n"
            f"        return 0\n"
            f"    if (n, k) in memo:\n"
            f"        return memo[(n, k)]\n"
            f"    memo[(n, k)] = choose(n - 1, k - 1) + choose(n - 1, k)\n"
            f"    return memo[(n, k)]\n\n"
            f"print(choose({n}, {k}))\n"
            f"```\n"
            f"📊 **Computed Output:** `{result}`\n"
            f"✅ Time: O(n·k), Space: O(n·k)"
        )



# Question 14: Longest increasing subsequence in array
    elif "longest increasing subsequence" in q:
        match = re.search(r"([0-9,\s-]+)", question)
        if not match:
            return "⚠️ Could not extract array."
        arr = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
        memo = {}
        def lis(i, prev):
            if i == len(arr):
                return 0
            key = (i, prev)
            if key in memo:
                return memo[key]
            take = 0
            if prev == -1 or arr[i] > arr[prev]:
                take = 1 + lis(i + 1, i)
            not_take = lis(i + 1, prev)
            memo[key] = max(take, not_take)
            return memo[key]

        result = lis(0, -1)
        return (
            f"👨‍🏫 We are computing the Longest Increasing Subsequence (LIS) of array: {arr}.\n\n"
            f"🪜 Logic:\n"
            f"- At each index, either take the element (if increasing) or skip.\n"
            f"- Use memoization on (index, prev_index) pairs.\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"arr = {arr}\n"
            f"memo = {{}}\n"
            f"def lis(i, prev):\n"
            f"    if i == len(arr): return 0\n"
            f"    key = (i, prev)\n"
            f"    if key in memo: return memo[key]\n"
            f"    take = 0\n"
            f"    if prev == -1 or arr[i] > arr[prev]:\n"
            f"        take = 1 + lis(i + 1, i)\n"
            f"    not_take = lis(i + 1, prev)\n"
            f"    memo[key] = max(take, not_take)\n"
            f"    return memo[key]\n"
            f"\n"
            f"print(lis(0, -1))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )

# Question 15: Climbing stairs with steps of 1, 2, or 3
    elif "climb a staircase of" in q and "1" in q and "2" in q and "3" in q:
        match = re.search(r"staircase of (\d+)", q)
        if not match:
            return "⚠️ Could not extract staircase size."
        n = int(match.group(1))

        memo = {}
        def dp(i):
            if i < 0: return 0
            if i == 0: return 1
            if i in memo: return memo[i]
            memo[i] = dp(i - 1) + dp(i - 2) + dp(i - 3)
            return memo[i]

        result = dp(n)

        return (
            f"👨‍🏫 We are counting how many ways to climb {n} steps using steps of 1, 2, or 3 at a time.\n\n"
            f"🪜 Logic:\n"
            f"- dp(n) = dp(n-1) + dp(n-2) + dp(n-3)\n"
            f"- Base: dp(0)=1 (1 way to stand still), dp(<0)=0\n"
            f"- Use memoization to cache subproblems.\n\n"
            f"🛠 Code:\n"
            f"```python\n"
            f"n = {n}\n"
            f"memo = {{}}\n"
            f"def dp(i):\n"
            f"    if i < 0: return 0\n"
            f"    if i == 0: return 1\n"
            f"    if i in memo: return memo[i]\n"
            f"    memo[i] = dp(i - 1) + dp(i - 2) + dp(i - 3)\n"
            f"    return memo[i]\n"
            f"\n"
            f"print(dp(n))\n"
            f"```\n\n"
            f"✅ Final Answer: {result}"
        )


def answer_memoization_quantitative_lvl3(question):
    import re
    q = question.lower()

#  Q1: Min cost to reach target in graph with weighted edges and obstacles
    if "minimum cost to reach the target" in q and "graph" in q:
        return (
            f"👨‍🏫 You are given a weighted graph with obstacles. Goal: find minimum cost path to target.\n"
            f"We use memoization + DFS from source, skipping blocked nodes.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def min_cost_graph(n, edges, start, end, obstacles):\n"
            "    graph = {i: [] for i in range(n)}\n"
            "    for u, v, w in edges:\n"
            "        graph[u].append((v, w))\n"
            "        graph[v].append((u, w))\n"
            "    blocked = set(obstacles)\n"
            "    memo = {}\n"
            "    def dfs(u):\n"
            "        if u in blocked: return float('inf')\n"
            "        if u == end: return 0\n"
            "        if u in memo: return memo[u]\n"
            "        min_cost = float('inf')\n"
            "        for v, w in graph[u]:\n"
            "            min_cost = min(min_cost, w + dfs(v))\n"
            "        memo[u] = min_cost\n"
            "        return min_cost\n"
            "    res = dfs(start)\n"
            "    return res if res != float('inf') else -1\n"
            "```"
        )

# 🔢 Q2: Max path sum in matrix with negative values
    elif "maximum path sum" in q and "matrix" in q and "negative" in q:
        return (
            f"👨‍🏫 Goal: Find max path sum in m×n matrix from top-left to bottom-right, only down/right allowed.\n"
            f"Cells can have negative values, so we cannot just take max of neighbors blindly.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def max_path_sum(grid):\n"
            "    m, n = len(grid), len(grid[0])\n"
            "    memo = {}\n"
            "    def dp(i, j):\n"
            "        if i >= m or j >= n:\n"
            "            return float('-inf')\n"
            "        if i == m-1 and j == n-1:\n"
            "            return grid[i][j]\n"
            "        if (i, j) in memo:\n"
            "            return memo[(i, j)]\n"
            "        right = dp(i, j+1)\n"
            "        down = dp(i+1, j)\n"
            "        memo[(i, j)] = grid[i][j] + max(right, down)\n"
            "        return memo[(i, j)]\n"
            "    return dp(0, 0)\n"
            "```"
        )


# 🔢 Q3: Longest path in DAG with weights
    elif "longest path" in q and "dag" in q:
        return (
            f"👨‍🏫 In a DAG (Directed Acyclic Graph), we want the longest weighted path.\n"
            f"Use memoized DFS + topological order if needed.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def longest_path_dag(n, edges):\n"
            "    from collections import defaultdict\n"
            "    graph = defaultdict(list)\n"
            "    for u, v, w in edges:\n"
            "        graph[u].append((v, w))\n"
            "    memo = {}\n"
            "    def dfs(u):\n"
            "        if u in memo:\n"
            "            return memo[u]\n"
            "        max_len = 0\n"
            "        for v, w in graph[u]:\n"
            "            max_len = max(max_len, w + dfs(v))\n"
            "        memo[u] = max_len\n"
            "        return max_len\n"
            "    return max(dfs(i) for i in range(n))\n"
            "```"
        )

# 🔢 Q4: Distinct paths in 3D grid with 6-directional movement
    elif "6 directions" in q and "blocked" in q and "grid" in q:
        return (
            f"👨‍🏫 In a 3D grid (x×y×z), you can move in 6 directions. Avoid blocked cells and count distinct paths.\n"
            f"Use memoization on (x, y, z) to avoid recomputation.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def count_3d_paths(x, y, z, blocked):\n"
            "    from functools import lru_cache\n"
            "    blocked = set(tuple(b) for b in blocked)\n"
            "    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]\n"
            "    @lru_cache(None)\n"
            "    def dp(i,j,k):\n"
            "        if (i,j,k) == (x-1,y-1,z-1): return 1\n"
            "        if not(0 <= i < x and 0 <= j < y and 0 <= k < z): return 0\n"
            "        if (i,j,k) in blocked: return 0\n"
            "        return sum(dp(i+di, j+dj, k+dk) for di,dj,dk in directions)\n"
            "    return dp(0,0,0)\n"
            "```"
        )

# 🔢 Q5: Number of ways to split array into two equal sum subsets
    elif "split the array into two subsets" in q and "equal" in q:
        return (
            f"👨‍🏫 Problem is equivalent to subset sum count: find number of subsets that sum to total_sum // 2\n"
            f"Use memoization on index and remaining sum.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def count_equal_split(arr):\n"
            "    total = sum(arr)\n"
            "    if total % 2 != 0: return 0\n"
            "    target = total // 2\n"
            "    memo = {}\n"
            "    def dp(i, t):\n"
            "        if t == 0: return 1\n"
            "        if i == len(arr) or t < 0: return 0\n"
            "        if (i, t) in memo: return memo[(i, t)]\n"
            "        memo[(i, t)] = dp(i+1, t-arr[i]) + dp(i+1, t)\n"
            "        return memo[(i, t)]\n"
            "    return dp(0, target)\n"
            "```"
        )

    elif "distribute" in q and "identical items" in q and "distinct bins" in q:
        match = re.search(r"distribute\s*\{\{?(.*?)\}?\}.*?into\s*\{\{?(.*?)\}?\}.*?maximum of\s*\{\{?(.*?)\}?\}", q)
        n = int(match.group(1)) if match else 5
        k = int(match.group(2)) if match else 3
        max_each = int(match.group(3)) if match else 2
        return (
            f"👨‍🏫 You must distribute {n} identical items into {k} bins, with max {max_each} items per bin.\n"
            f"This is a variation of bounded integer partition with memoization.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def distribute_items(n, k, max_each):\n"
            "    memo = {}\n"
            "    def dp(i, total):\n"
            "        if i == k:\n"
            "            return 1 if total == n else 0\n"
            "        if (i, total) in memo:\n"
            "            return memo[(i, total)]\n"
            "        count = 0\n"
            "        for x in range(0, min(max_each, n - total) + 1):\n"
            "            count += dp(i + 1, total + x)\n"
            "        memo[(i, total)] = count\n"
            "        return count\n"
            "    return dp(0, 0)\n\n"
            f"# Example:\n"
            f"print(distribute_items({n}, {k}, {max_each}))\n"
            "```"
        )

    elif "movement up, down, left, right" in q and "obstacles" in q:
        return (
            f"👨‍🏫 In a 2D grid, count the number of unique paths from top-left to bottom-right with movement in all 4 directions.\n"
            f"Use DFS + memo + visited set to avoid cycles and blocked cells.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def unique_paths_4dir(grid, obstacles):\n"
            "    m, n = len(grid), len(grid[0])\n"
            "    blocked = set(tuple(ob) for ob in obstacles)\n"
            "    directions = [(-1,0),(1,0),(0,-1),(0,1)]\n"
            "    memo = {}\n"
            "    def dfs(i,j,visited):\n"
            "        if (i,j) == (m-1,n-1): return 1\n"
            "        if not(0 <= i < m and 0 <= j < n): return 0\n"
            "        if (i,j) in blocked or (i,j) in visited: return 0\n"
            "        key = (i,j,frozenset(visited))\n"
            "        if key in memo: return memo[key]\n"
            "        visited.add((i,j))\n"
            "        total = sum(dfs(i+di,j+dj,visited) for di,dj in directions)\n"
            "        visited.remove((i,j))\n"
            "        memo[key] = total\n"
            "        return total\n"
            "    return dfs(0,0,set())\n"
            "```"
        )

    elif "maximum value obtainable" in q and "limited number" in q:
        return (
            f"👨‍🏫 You’re given coins with limits. You must reach exact target value using each coin at most limit[i] times.\n"
            f"Use memoization with (index, remaining target).\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def coin_limited(coins, limits, target):\n"
            "    memo = {}\n"
            "    def dp(i, t):\n"
            "        if t == 0: return 1\n"
            "        if i == len(coins) or t < 0: return 0\n"
            "        if (i, t) in memo: return memo[(i, t)]\n"
            "        ways = 0\n"
            "        for cnt in range(0, limits[i]+1):\n"
            "            if cnt * coins[i] <= t:\n"
            "                ways += dp(i+1, t - cnt * coins[i])\n"
            "        memo[(i, t)] = ways\n"
            "        return ways\n"
            "    return dp(0, target)\n"
            "```"
        )

    elif "maximum profit" in q and "non-overlapping jobs" in q:
        return (
            f"👨‍🏫 You’re given jobs with start, end, profit. You must select max profit set of non-overlapping jobs.\n"
            f"Sort by end time and memoize with index.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def job_scheduling(jobs):\n"
            "    jobs.sort(key=lambda x: x[1])\n"
            "    from bisect import bisect_right\n"
            "    starts = [job[0] for job in jobs]\n"
            "    memo = {}\n"
            "    def dp(i):\n"
            "        if i == len(jobs): return 0\n"
            "        if i in memo: return memo[i]\n"
            "        next_idx = bisect_right(starts, jobs[i][1])\n"
            "        take = jobs[i][2] + dp(next_idx)\n"
            "        skip = dp(i+1)\n"
            "        memo[i] = max(take, skip)\n"
            "        return memo[i]\n"
            "    return dp(0)\n"
            "```"
        )

# 🔢 Q10: Tiling n×m grid with 1×2 tiles
    elif "tile" in q and "1x2" in q and "grid" in q:
        return (
            f"👨‍🏫 You’re asked to count how many ways to tile an n×m grid using 1×2 tiles (domino).\n"
            f"This is solved using recursion with memoization on (row, mask).\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def count_tilings(n, m):\n"
            "    from functools import lru_cache\n"
            "    @lru_cache(None)\n"
            "    def dp(row, mask):\n"
            "        if row == n:\n"
            "            return 1 if mask == 0 else 0\n"
            "        def helper(col, curr_mask, next_mask):\n"
            "            if col == m:\n"
            "                return dp(row + 1, next_mask)\n"
            "            if (curr_mask >> col) & 1:\n"
            "                return helper(col + 1, curr_mask, next_mask)\n"
            "            res = 0\n"
            "            res += helper(col + 1, curr_mask, next_mask | (1 << col))\n"
            "            if col + 1 < m and not ((curr_mask >> (col + 1)) & 1):\n"
            "                res += helper(col + 2, curr_mask, next_mask)\n"
            "            return res\n"
            "        return helper(0, mask, 0)\n"
            "    return dp(0, 0)\n"
            "```"
        )

    elif "partition the array into two subsets" in q and "absolute difference" in q:
        return (
            f"👨‍🏫 Goal: Partition array into two subsets so that |sum1 - sum2| is minimized.\n"
            f"Use subset sum DP to find the closest sum to total // 2.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def min_subset_diff(arr):\n"
            "    total = sum(arr)\n"
            "    n = len(arr)\n"
            "    memo = {}\n"
            "    def dp(i, curr):\n"
            "        if i == n:\n"
            "            return abs((total - curr) - curr)\n"
            "        if (i, curr) in memo:\n"
            "            return memo[(i, curr)]\n"
            "        take = dp(i+1, curr + arr[i])\n"
            "        skip = dp(i+1, curr)\n"
            "        memo[(i, curr)] = min(take, skip)\n"
            "        return memo[(i, curr)]\n"
            "    return dp(0, 0)\n"
            "```"
        )

    elif "arrange" in q and "sum of the positions" in q and "even" in q:
        return (
            f"👨‍🏫 We want to arrange items such that sum of positions of any two consecutive items is even.\n"
            f"This implies both positions must be even-even or odd-odd ⇒ count even permutations.\n"
            f"Use DP with memoization to count valid permutations.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "from math import comb\n"
            "def count_even_pos_sequences(n):\n"
            "    if n == 0: return 0\n"
            "    even = (n + 1) // 2\n"
            "    odd = n // 2\n"
            "    return comb(n, even)  # Approximate placeholder\n"
            "```"
        )


    elif "maximum number of distinct subsequences" in q:
        return (
            f"👨‍🏫 Count all possible subsequences — this is 2^n for string of length n.\n"
            f"But to count distinct ones, use memo + visited substrings.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def count_distinct_subseq(s):\n"
            "    n = len(s)\n"
            "    memo = {}\n"
            "    mod = 10**9 + 7\n"
            "    def dp(i):\n"
            "        if i == n: return 1\n"
            "        if i in memo: return memo[i]\n"
            "        total = dp(i+1)\n"
            "        for j in range(i+1, n):\n"
            "            if s[j] == s[i]:\n"
            "                total += dp(j+1)\n"
            "                break\n"
            "        memo[i] = total % mod\n"
            "        return memo[i]\n"
            "    return dp(0) - 1\n"
            "```"
        )


    elif "maximum number of non-overlapping intervals" in q:
        return (
            f"👨‍🏫 Greedy approach + memoization: Sort by end time and find max compatible subset.\n"
            f"Use memo[index] to store best answer from that index onward.\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def max_non_overlap(intervals):\n"
            "    intervals.sort(key=lambda x: x[1])\n"
            "    from bisect import bisect_right\n"
            "    starts = [i[0] for i in intervals]\n"
            "    memo = {}\n"
            "    def dp(i):\n"
            "        if i == len(intervals): return 0\n"
            "        if i in memo: return memo[i]\n"
            "        next_i = bisect_right(starts, intervals[i][1])\n"
            "        take = 1 + dp(next_i)\n"
            "        skip = dp(i+1)\n"
            "        memo[i] = max(take, skip)\n"
            "        return memo[i]\n"
            "    return dp(0)\n"
            "```"
        )

    elif "maximum number of points" in q and "penalty" in q:
        return (
            f"👨‍🏫 Standard max path sum in m×n grid with penalties: you can only go right/down.\n"
            f"Use memo[(i,j)] = max(grid[i][j] + max(right, down))\n\n"
            f"🛠 Code:\n"
            "```python\n"
            "def max_points(grid):\n"
            "    m, n = len(grid), len(grid[0])\n"
            "    memo = {}\n"
            "    def dp(i, j):\n"
            "        if i >= m or j >= n: return float('-inf')\n"
            "        if (i,j) == (m-1,n-1): return grid[i][j]\n"
            "        if (i,j) in memo: return memo[(i,j)]\n"
            "        memo[(i,j)] = grid[i][j] + max(dp(i+1, j), dp(i, j+1))\n"
            "        return memo[(i,j)]\n"
            "    return dp(0, 0)\n"
            "```"
        )

def answer_memoization_quantitative(level, question):
    if level == "Level 1":
        return answer_memoization_quantitative_lvl1(question)
    elif level == "Level 2":
        return answer_memoization_quantitative_lvl2(question)
    elif level == "Level 3":
        return answer_memoization_quantitative_lvl3(question)
    else:
        return "No answer for this level."

def answer_memoization_application(level, question):
    if level == "Level 1":
        return answer_memoization_application_lvl1(question)
    elif level == "Level 2":
        return answer_memoization_application_lvl2(question)
    elif level == "Level 3":
        return answer_memoization_application_lvl3(question)
    else:
        return "No answer for this level."
# --- TEST BLOCKS (START) ---

def test_answer_memoization_lvl1_conceptual():
    print("\n--- Testing Memoization Level 1 Conceptual Answers ---\n")
    questions_memo_lvl1_conceptual = [
        "Use memoization to optimize the Fibonacci problem.",
        "Convert the recursive implementation of Factorial into a memoized solution.",
        "Write a simple memoized function to calculate the 10th Fibonacci number.",
        "Demonstrate how memoization avoids repeated calls in Fibonacci.",
        "Implement memoization to solve a basic recursive function like factorial or Fibonacci.",
        "Modify a recursive function for climbing 5 stairs to use memoization.",
        "Use a dictionary to cache the output of my_custom_function calls.",
        "Transform a recursive function for calculating the Nth Power (e.g., base^exp) into a memoized version.",
        "Write a top-down memoized function for the Tribonacci sequence.",
        "Create a memoized wrapper function for computing factorial values.",
        "Use memoization to return the 8th term of the Fibonacci sequence using base cases.",
        "Write a memoized function to solve the problem of computing permutations efficiently.",
        "Demonstrate how memoization reduces the complexity of binomial coefficients.",
        "Apply memoization to return the 7th term of the Pascal's Triangle row sequence using base cases.",
        "Write a function using memoization to compute C(n, k) (combinations) recursively.",
        "Memoize a recursive function that checks whether a string is a palindrome efficiently.",
        "Create a memoized recursive solution to test if a given amount can be made with a set of coins repeatedly.",
        "Demonstrate lookup-table based memoization for computing nth Catalan number."
    ]

    for i, question in enumerate(questions_memo_lvl1_conceptual, 1):
        print(f"Test Case {i}:")
        print(f"Question: {question}")
        answer = answer_memoization_lvl1(question)
        print(f"Answer:\n{answer}\n{'-'*50}\n")


def test_answer_memoization_application_lvl1():
    print("\n--- Testing Memoization Level 1 Application Answers ---\n")
    questions_memo_lvl1_application = [
        "Use memoization to optimize the Fibonacci problem in a simulation.",
        "Convert the recursive implementation of population_growth_model into a memoized solution.",
        "Write a simple memoized function to calculate the 15th Fibonacci number for a data generation module.",
        "Demonstrate how memoization avoids repeated calls in route_optimization for a delivery app.",
        "Implement memoization to solve a basic recursive function like calculate_game_score or build_combinatorial_tree.",
        "Modify a recursive function for climbing 10 stairs in a game to use memoization.",
        "Use a dictionary to cache the output of get_user_preferences calls in a recommendation engine.",
        "Transform a recursive function for calculating the Nth term of a financial_series into a memoized version.",
        "Write a top-down memoized function for the protein_sequence_alignment.",
        "Create a memoized wrapper function for computing probability_distributions in a statistical analysis tool.",
        "Use memoization to return the 12th term of the animal_population_sequence using base cases.",
        "Write a memoized function to solve the problem of computing resource_allocation_cost efficiently.",
        "Demonstrate how memoization reduces the complexity of parsing_grammar in a natural language processing tool.",
        "Apply memoization to return the 9th term of the manufacturing_defect_pattern using base cases.",
        "Write a function using memoization to compute recurrence_relation(n) for a scientific model.",
        "Memoize a recursive function that checks whether a given_state_is_valid in an AI planning system.",
        "Create a memoized recursive solution to test if a customer_segment_matches_criteria repeatedly in a CRM.",
        "Demonstrate lookup-table based memoization for computing product_pricing_tiers in an e-commerce platform."
    ]

    for i, question in enumerate(questions_memo_lvl1_application, 1):
        print(f"Test Case {i}:")
        print(f"Question: {question}")
        answer = answer_memoization_application_lvl1(question)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

def test_answer_memoization_application_lvl2():
    print("\n--- Testing Memoization Level 2 Application Answers ---\n")
    questions_memo_lvl2_application = [
        "Solve the Fibonacci problem using memoization and explain how it improves efficiency.",
        "Use memoization to solve the problem of counting all ways to reach sum 7 using coins {1,2,3}.",
        "Apply memoization to find the number of ways to decode the string '12345'.",
        "Implement a memoized version of the subset sum problem for array {2,3,7,8,10} and target 11.",
        "Write a memoized function to solve 0/1 Knapsack with weights={1,3,4,5}, values={10,40,50,70}, and capacity=8.",
        "Apply memoization to solve the edit distance between 'kitten' and 'sitting'.",
        "Memoize a function to find LCS length between 'ABCBDAB' and 'BDCABA'.",
        "Write a memoized function for climbing 4 steps with step options {1,2}.",
        "Use memoization to solve wildcard pattern matching between 'abctest' and pattern 'abc*st'.",
        "Write a memoized version of the minimum coin change problem for coins {1,2,5} and amount 11.",
        "Memoize a recursive function that returns the number of unique BSTs with 3 nodes.",
        "Use memoization to compute the number of palindromic substrings in 'abaacada'.",
        "Apply memoization to solve the maximum sum of non-adjacent elements in array {3,2,5,1,4}.",
        "Write a memoized function to solve 0/1 Knapsack with weights={1,3,4,5}, values={10,40,50,70}, and capacity=8.", # Duplicate for testing
        "Use memoization to solve Boolean Parenthesization for expression 'T|F&T^F'.",
        "Memoize a recursive solution for rod cutting given length 4 and prices {1,5,8,9}.",
        "Solve tiling problem using memoization for 2 x 4 board with tiles of size 2 x 1.",
        "Write a memoized function to count number of binary strings of length 5 without consecutive 1s.",
        "Use memoization to solve longest increasing subsequence for array {10,22,9,33,21,50,41,60}.",
        "Implement memoization in recursive solution to count number of partitions of array {1,5,11,5} into equal subsets."
    ]

    for i, question in enumerate(questions_memo_lvl2_application, 1):
        print(f"Test Case {i}:")
        print(f"Question: {question}")
        answer = answer_memoization_application_lvl2(question)
        print(f"Answer:\n{answer}\n{'-'*50}\n")


def test_answer_memoization_application_lvl3():
    print("\n--- Testing Memoization Level 3 Application Answers ---\n")
    questions_memo_lvl3_application = [
        "Apply memoization to solve the complex_optimization problem under constraints resource_limit.",
        "Write a recursive + memoized solution for the Traveling Salesman Problem for {4} locations.",
        "Memoize a recursive solution for computing number of ways to reach bottom-right of a 3x3 grid with obstacles {0,1,1,1}.", # Obstacles at (0,1) and (1,1)
        "Use memoization to solve the minimum edit distance between strings 'algorithm' and 'altruistic'.",
        "Write a memoized recursive function for longest common subsequence of 'ABCBDAB' and 'BDCABA'.",
        "Memoize a solution to find the number of binary search trees with 4 nodes.",
        "Solve the N-Queens problem with memoization and return number of valid configurations on 4x4 board.",
        "Use memoization in solving maximum profit problem in stock prices array {7,1,5,3,6,4}.",
        "Memoize solution for the shortest path from node A to node E in graph A:B=1,C=4;B:C=2,D=5;C:D=1;D:E=3;E:.",
        "Implement a hybrid top-down memoization and bottom-up DP solution for knapsack_problem.",
        "Apply memoization to find number of integer partitions of 5 using integers in {1,2,3}.",
        "Memoize recursive calls in bitmask-based solution for visiting all cities in 4 steps.",
        "Use memoization to track optimal matrix multiplication order for matrix dimensions [10, 20, 5, 30].",
        "Apply memoization to solve recursive logic in subset XOR sum problem for array {1,2,4,8}.",
        "Design memoized approach to solve DP with two changing parameters like f(i, j) for longest_common_subsequence.",
        "Implement memoization in recursive DP for counting valid parentheses combinations of size 3.",
        "Solve problem with recursive state: f(i, k, t) using memoization in context of a_custom_knapsack_variant.",
        "Write a memoized function for recursive DAG traversal with caching at node level for graph A:B,C;B:D;C:D;D:.",
        "Use memoization to solve recursive grammar parsing problem for string 'ababa' and rule set {a,b,aba}.",
        "Apply memoization to optimize computation of multivariate recurrence function f(3, 2, 1)."
    ]

    for i, question in enumerate(questions_memo_lvl3_application, 1):
        print(f"Test Case {i}:")
        print(f"Question: {question}")
        answer = answer_memoization_application_lvl3(question)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

def test_answer_memoization_quantitative_lvl1():
    print("\n--- Testing Memoization Level 1 Quantitative Answers ---\n")
    questions_memo_lvl1_quantitative = [
        "Using memoization, determine the maximum sum of non-adjacent elements in the array [3, 2, 5, 10]."
,
         # 1. Staircase with 1, 2, 3 steps
    "Using memoization, calculate the total number of unique ways to climb a staircase with 5 steps, where you can take either 1, 2, or 3 steps at a time.",

    # 2. Minimum number of coins
    "Using memoization, find the minimum number of coins required to make a total of 11, given the denominations [1, 5, 6, 8].",

    # 3. Partition number
    "Find the number of ways to partition the number 5 into the sum of positive integers using memoization.",

    # 4. Fibonacci nth term
    "Using memoization, find the nth term of the Fibonacci sequence where 8 is the given input.",

    # 5. Grid path (bottom-left to top-right)
    "Using memoization, calculate the number of ways to reach the top of a 3x3 grid from the bottom-left corner, only moving up or right.",

    # 6. Grid with obstacles
    "Using memoization, determine the total number of distinct paths to traverse a grid with obstacles, represented by a 3x3 grid of 0s and 1s, where 1s represent obstacles.",

    # 7. Stock buy-sell profit (unlimited)
    "Using memoization, determine the maximum profit that can be obtained from buying and selling a stock with prices given in an array [7, 1, 5, 3, 6, 4].",

    # 8. Unique binary search trees
    "Using memoization, calculate the number of unique binary search trees that can be constructed using 4 nodes.",

    # 9. Edit distance between two strings
    "Using memoization, find the minimum number of operations required to convert string horse to string ros using insertion, deletion, and substitution.",

    # 10. General staircase (up to k steps)
    "Using memoization, calculate the total number of unique ways to climb a staircase with 6 steps, where you can take up to 2 steps at a time.",


    # 13. Grid top-right path again
    "Using memoization, calculate the number of distinct paths to reach the top of a grid with dimensions 3x3 from the bottom-left corner, moving only right or up.",

    # 14. Obstacle grid alternate phrasing
    "Given a grid of size 3x3 with obstacles (represented as 0s and 1s), using memoization, find the number of distinct paths from the top-left to the bottom-right corner.",

    # 15. Integer partition alternate phrasing
    "Using memoization, calculate the number of unique ways to partition the number 5 into a sum of integers.",

    # 16. Staircase with 1 or 2 steps
    "Using memoization, find the number of ways to climb a staircase with 6 steps when you can take 1 or 2 steps at a time.",

    # 17. Max profit with k transactions (not yet implemented in code)
    "Given an array of integers [3, 2, 6, 5, 0, 3], using memoization, determine the maximum profit that can be obtained from buying and selling stock at most 2 times.",

    # 18. Edit distance alternate phrasing
    "Using memoization, calculate the minimum number of edits required to convert string kitten into string sitting, where only insertions, deletions, and substitutions are allowed."
    ]

    for i, question in enumerate(questions_memo_lvl1_quantitative, 1):
        print(f"Test Case {i}:")
        print(f"Question: {question}")
        answer = answer_memoization_quantitative_lvl1(question)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

def testing_answer_memoization_quantitative_lvl2():
    print("\n--- Testing Memoization Level 2 Quantitative Answers ---\n")
    questions_memo_lvl2_quantitative = [
       "Using memoization, calculate the total number of ways to climb a staircase with 10 steps, where you can take up to 3 steps at a time, but you must rest after every 4 steps.",
    "Given an array of size 7, using memoization, calculate the minimum number of deletions required to make the array strictly increasing.",
    "Using memoization, find the maximum sum of non-adjacent elements from an array of size 6 with negative and positive integers [3, 2, 5, 10, 7, -2].",
    "Given an array of integers [5, 1, 3, 2, 6], using memoization, determine the minimum number of operations to convert the array into a non-decreasing order.",
    "Using memoization, calculate the number of distinct ways to arrange 4 objects from a set of 6 objects, where some objects are identical.",
    "Using memoization, find the number of unique binary search trees that can be formed using 5 distinct nodes.",
    "Given a list of coin denominations [1, 2, 5] and a target sum 11, using memoization, calculate the number of ways to make the sum using the coins.",
    "Using memoization, calculate the number of possible combinations to reach the target sum 8 from a set of integers [2, 3, 5].",
    "Given a grid of size 3x3 and a list of blocked cells [(1,1)], using memoization, find the number of distinct paths to the top-right corner.",
    "Using memoization, calculate the minimum number of steps required to reach from the top-left to the bottom-right corner of a matrix with obstacles: [[0,0,0],[0,1,0],[0,0,0]]"
    "Using memoization, calculate the number of ways to partition a set of 6 distinct integers into two subsets such that the sum of the integers in each subset is the same.",
    "Given a list of integers [1, 2, 3, 4, 5], using memoization, determine the minimum cost of cutting a rod of length 5 into smaller pieces, where the price of a rod of length i is price[i].",
    "Using memoization, find the number of ways to select 3 items from a list of 5 items where the order does not matter, but some items are identical.",
    "Using memoization, calculate the number of ways to assign 4 tasks to 3 workers, where each worker can handle multiple tasks and each task requires a certain amount of time from each worker.",
    "Given a list of integers [10, 22, 9, 33, 21, 50, 41, 60], using memoization, find the length of the longest increasing subsequence in the array.",
    "Using memoization, calculate the number of distinct ways to climb a staircase of 7 steps, but now with the possibility of taking 1, 2, or 3 steps at a time."
    ]

    for i, question in enumerate(questions_memo_lvl2_quantitative, 1):
        print(f"Test Case {i}:")
        print(f"Question: {question}")
        answer = answer_memoization_quantitative_lvl2(question)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

# Main execution block to run tests
if __name__ == "__main__":
    # Uncomment the test function calls below to run specific test suites.
    
    # test_answer_memoization_lvl1_conceptual() # Run Conceptual Level 1 tests
    # test_answer_memoization_application_lvl1() # Run Application Level 1 tests
    # test_answer_memoization_application_lvl2() # Run Application Level 2 tests
    # test_answer_memoization_application_lvl3() # Run Application Level 3 tests
    # test_answer_memoization_quantitative_lvl1() # Run Quantitative Level 1 tests
    testing_answer_memoization_quantitative_lvl2() # Run Quantitative Level 2 tests
