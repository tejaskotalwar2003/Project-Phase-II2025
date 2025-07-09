import re
import math
import time

# Helper functions for Fibonacci calculations, moved to global scope
# These are used by various answer_algo_lvlX_fibonacci functions

# Placeholder for fib_two_vars_global - ensure this is defined globally in your file
def fib_two_vars_global(n_val):
    if n_val < 0: return 0 # Handle negative n
    if n_val <= 1: return n_val
    a, b = 0, 1
    for _ in range(2, n_val + 1):
        a, b = b, a + b
    return b

# Placeholder for sum_fib_seq_global - ensure this is defined globally in your file
def sum_fib_seq_global(n_count):
    if n_count <= 0: return 0
    a, b = 0, 1
    total_sum = 0
    for _ in range(n_count):
        total_sum += a
        a, b = b, a + b
    return total_sum

# Placeholder for fibonacci_matrix_exponentiation - ensure this is defined globally in your file
def fibonacci_matrix_exponentiation(n, mod=None):
    if n <= 1:
        return n
    # This is a simplified 2x2 matrix multiplication and power function for Fib.
    # Ensure the full, correct versions of multiply_matrices and power (with optional mod)
    # are present globally for this to work for actual calculations.
    # For this example, we'll use a simple iterative approach to get the value.
    if mod is None:
        return fib_two_vars_global(n)
    else:
        # For actual matrix exponentiation with mod, you need the full power(matrix, n-1, mod) setup.
        # This is a basic fallback for modular results if matrix functions aren't fully integrated yet.
        return fib_two_vars_global(n) % mod

# Placeholder for get_all_fib_less_than_x_global
def get_all_fib_less_than_x_global(x_limit):
    fib_numbers = []
    a, b = 0, 1
    while a < x_limit:
        fib_numbers.append(a)
        a, b = b, a + b
    return fib_numbers



def fib_memo_global(n_val, memo=None):
    """Global memoized Fibonacci for internal calculations."""
    if memo is None:
        memo = {}
    if n_val in memo:
        return memo[n_val]
    if n_val <= 1:
        return n_val
    memo[n_val] = fib_memo_global(n_val - 1, memo) + fib_memo_global(n_val - 2, memo)
    return memo[n_val]

def fib_bottom_up_global(k_val):
    """Global bottom-up Fibonacci for internal calculations."""
    if k_val <= 1:
        return k_val
    dp = [0] * (k_val + 1)
    dp[0] = 0
    dp[1] = 1
    for i in range(2, k_val + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[k_val]

def fib_two_vars_global(n_val):
    """Global iterative Fibonacci using two variables for internal calculations."""
    if n_val <= 1:
        return n_val
    a, b = 0, 1
    for _ in range(2, n_val + 1):
        a, b = b, a + b
    return b

def sum_fib_seq_global(n_count):
    """Global function to sum first n Fibonacci numbers for internal calculations."""
    if n_count <= 0:
        return 0
    # Note: F0=0, F1=1, F2=1, ...
    # Sum of first n terms (F0 to F(n-1)) is F(n+1) - 1.
    # Sum of first n numbers (F1 to Fn) is F(n+2) - 1.
    # The question implies first 'count' numbers, which usually means F0, F1, ..., F(count-1)
    # The current iterative sum is correct for F0 to F(count-1)
    a, b = 0, 1
    total_sum = 0
    for _ in range(n_count):
        total_sum += a
        a, b = b, a + b
    return total_sum

def get_even_fibonacci_global(n_count):
    """Global function to get first n even Fibonacci numbers for internal calculations."""
    even_fib_list = []
    a, b = 0, 1
    # Fibonacci numbers repeat parity: O, O, E, O, O, E, ... (0, 1, 1, 2, 3, 5, 8, ...)
    # Every 3rd Fibonacci number is even.
    while len(even_fib_list) < n_count:
        if a % 2 == 0:
            even_fib_list.append(a)
        a, b = b, a + b
    return even_fib_list

def fib_tabulation_array_func_global(idx_val):
    """Global tabulation with array Fibonacci for internal calculations."""
    if idx_val <= 1:
        return idx_val
    dp = [0] * (idx_val + 1)
    dp[0] = 0
    dp[1] = 1
    for i in range(2, idx_val + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[idx_val]

def fib_tail_recursive_func_global(n_arg, a=0, b=1):
    """Global tail recursive Fibonacci for internal calculations."""
    if n_arg == 0:
        return a
    if n_arg == 1:
        return b
    return fib_tail_recursive_func_global(n_arg - 1, b, a + b)

def largest_fib_smaller_than_limit_global(limit_val):
    """Global function to find largest Fibonacci number smaller than limit for internal calculations."""
    a, b = 0, 1
    largest_fib = 0
    while a < limit_val:
        largest_fib = a
        a, b = b, a + b
    return largest_fib

def get_all_fib_less_than_x_global(x_limit):
    """Global function to get all Fibonacci numbers less than x for internal calculations."""
    fib_numbers = []
    a, b = 0, 1
    while a < x_limit:
        fib_numbers.append(a)
        a, b = b, a + b
    return fib_numbers

# Helper for is_fibonacci_number
def is_perfect_square_global(k):
    """Global helper to check if a number is a perfect square."""
    if k < 0: return False
    s = int(math.sqrt(k))
    return s * s == k

def is_fibonacci_num_global(n_val):
    """Global function to check if a number is Fibonacci for internal calculations."""
    # A positive integer 'n' is a Fibonacci number if and only if one or both of
    # (5*n^2 + 4) or (5*n^2 - 4) is a perfect square.
    return is_perfect_square_global(5 * n_val * n_val + 4) or is_perfect_square_global(5 * n_val * n_val - 4)

def print_fibonacci_between_positions_global(start_pos, end_pos):
    """Global function to print Fibonacci numbers between positions for internal calculations."""
    fib_sequence_range = []
    a, b = 0, 1
    for i in range(end_pos + 1):
        if i >= start_pos:
            fib_sequence_range.append(a)
        a, b = b, a + b
    return fib_sequence_range


# Matrix exponentiation helper functions for optimization question
def multiply_matrices(A, B, mod=None):
    """Multiplies two 2x2 matrices. Supports modular arithmetic if mod is provided."""
    x = A[0][0] * B[0][0] + A[0][1] * B[1][0]
    y = A[0][0] * B[0][1] + A[0][1] * B[1][1]
    z = A[1][0] * B[0][0] + A[1][1] * B[1][0]
    w = A[1][0] * B[0][1] + A[1][1] * B[1][1]

    if mod is not None:
        return [[x % mod, y % mod], [z % mod, w % mod]]
    return [[x, y], [z, w]]

def power(M, n, mod=None):
    """Raises a 2x2 matrix to the power n using binary exponentiation. Supports modular arithmetic."""
    result = [[1, 0], [0, 1]]  # Identity matrix
    M_copy = M
    while n > 0:
        if n % 2 == 1:
            result = multiply_matrices(result, M_copy, mod)
        M_copy = multiply_matrices(M_copy, M_copy, mod)
        n //= 2
    return result

def fibonacci_matrix_exponentiation(n, mod=None):
    """Computes Fibonacci(n) using matrix exponentiation."""
    if n <= 1:
        return n
    # Base matrix: [[1, 1], [1, 0]]
    matrix = [[1, 1], [1, 0]]
    result_matrix = power(matrix, n - 1, mod)
    return result_matrix[0][0] if mod is None else result_matrix[0][0] % mod

def fibonacci_optimized_iterative(n):
    """Iterative Fibonacci with O(1) space, for optimization comparison."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def get_intermediate_fib_values(n_val):
    """Helper to get all intermediate Fibonacci values."""
    intermediate_values = []
    if n_val < 0: return []
    if n_val <= 1:
        intermediate_values.append(n_val)
    else:
        a, b = 0, 1
        for i in range(n_val + 1):
            intermediate_values.append(a)
            a, b = b, a + b
    return intermediate_values

# Fast Doubling helper functions
def fib_fast_doubling(n, mod=None):
    """Computes Fibonacci(n) using the fast doubling method."""
    if n == 0:
        return (0, 1) if mod is None else (0, 1) # F(0), F(1)
    
    a, b = fib_fast_doubling(n // 2, mod) # (F(k), F(k+1)) where k = n // 2
    
    # F(2k) = F(k) * (2*F(k+1) - F(k))
    # F(2k+1) = F(k+1)^2 + F(k)^2
    
    c = a * (2 * b - a)
    d = b * b + a * a
    
    if mod is not None:
        c %= mod
        d %= mod

    if n % 2 == 0: # If n is even, n = 2k
        return (c, d) # (F(n), F(n+1))
    else: # If n is odd, n = 2k+1
        return (d, c + d) if mod is None else (d, (c + d) % mod) # (F(n), F(n+1))

# Sum of first n Fibonacci numbers (S_n = F_{n+2} - 1)
def sum_fib_numbers_formula(n, mod=None):
    """Computes the sum of the first n Fibonacci numbers (F_0 to F_{n-1}) using the formula S_n = F_{n+2} - 1."""
    if n == 0: return 0
    fib_n_plus_2 = fibonacci_matrix_exponentiation(n + 2, mod)
    if mod is None:
        return fib_n_plus_2 - 1
    else:
        return (fib_n_plus_2 - 1 + mod) % mod # Add mod to handle potential negative result from -1

# Sum of first n even Fibonacci numbers (E_n = F_{3k+2} - 1 for F_0, F_2, F_4, ...)
# More simply: sum of even Fibonacci numbers up to F_n is F_{n+1} - 1 if F_n is even, F_{n+1} is odd, F_{n+2} is odd
# Sum of even terms: E_n = (F_{k+1} - 1) / 2 where k is the index of the last even fib, or just F_{last_even_index + 2} - 1
# This is a known identity: Sum of even Fibonacci numbers up to F_n is F_{n+1} - 1 if n is multiple of 3 (F_n is even)
# A more general identity: Sum of F_i for i=0 to n where i is even is F_n+1 - 1 if n is odd, F_n+1 if n is even.
# Let's use the identity: Sum of even Fibonacci numbers up to F_N = F_{N+1} - 1 (if N is odd, as the last F_N is included if even)
# Or for F0, F2, F4, ... F(3k) is the formula: Sum of F(3i) for i = 0 to k is F(3k+2) - 1.
# The user wants sum of even Fibonacci numbers NOT EXCEEDING limit. This requires iteration.


def answer_algo_lvl1_fibonacci(question):
    q = question.lower()

    # 1. What is the Fibonacci number at position {{n}}?
    if "fibonacci number at position" in q:
        match = re.search(r"position (\d+)", q)
        if match:
            n = int(match.group(1))
            # Reusing global helper
            calculated_fib = fib_two_vars_global(n)
            return (
                f"To compute the Fibonacci number at position {n} using iteration:\n\n"
                "```python\n"
                "def fibonacci(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
                "```\n\n"
                f" Input: {n}\n"
                f" Output: {calculated_fib}"
            )

    # 2. Recursive function to calculate Fibonacci number at position {{k}}
    elif "recursive function" in q and "fibonacci number" in q:
        match = re.search(r"position (\d+)", q)
        if match:
            k = int(match.group(1))
            # Define recursive fib locally to avoid global state from memoization or tail recursion issues in tests
            def fib_recursive_local(n_val):
                if n_val <= 1:
                    return n_val
                return fib_recursive_local(n_val - 1) + fib_recursive_local(n_val - 2)
            return (
                f"To compute Fibonacci({k}) using recursion:\n\n"
                "```python\n"
                "def fibonacci_recursive(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)\n"
                "```\n\n"
                f" Input: {k}\n"
                f" Output: {fib_recursive_local(k)}\n"
                " Time Complexity: O(2^n)"
            )

    # 3. For-loop to print first {{count}} Fibonacci numbers
    elif "for-loop" in q and "first" in q and "fibonacci numbers" in q:
        match = re.search(r"first (\d+)", q)
        if match:
            count = int(match.group(1))
            seq = []
            a, b = 0, 1
            for _ in range(count):
                seq.append(a)
                a, b = b, a + b
            return (
                f"To print the first {count} Fibonacci numbers using a loop:\n\n"
                "```python\n"
                "def print_fibonacci(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        print(a, end=' ')\n"
                "        a, b = b, a + b\n"
                "```\n\n"
                f" Input: {count}\n"
                f" Output: {' '.join(map(str, seq))}"
            )

    # 4. Write a program to print the first {{n}} Fibonacci numbers
    elif "program to print" in q and "first" in q and "fibonacci numbers" in q:
        match = re.search(r"first (\d+)", q)
        if match:
            n = int(match.group(1))
            seq = []
            a, b = 0, 1
            for _ in range(n):
                seq.append(a)
                a, b = b, a + b
            return (
                f"Here’s a simple program to print the first {n} Fibonacci numbers:\n\n"
                "```python\n"
                "def fibonacci_series(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        print(a, end=' ')\n"
                "        a, b = b, a + b\n"
                "```\n\n"
                f" Output: {' '.join(map(str, seq))}"
            )

    # 5. Compute the {{k}}th Fibonacci number using a loop
    elif "compute the" in q and "fibonacci number" in q and "using a loop" in q:
        match = re.search(r"compute the (\d+)", q)
        if match:
            k = int(match.group(1))
            calculated_fib = fib_two_vars_global(k)
            return (
                f"To compute the {k}th Fibonacci number using a loop:\n\n"
                "```python\n"
                "def fibonacci(k):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(k):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
                "```\n\n"
                f" Output: {calculated_fib}"
            )

    # 6. Return the {{x}}th Fibonacci number using recursion
    elif "return the" in q and "fibonacci number" in q and "using recursion" in q:
        match = re.search(r"return the (\d+)", q)
        if match:
            x = int(match.group(1))
            # Define recursive fib locally
            def fib_recursive_local(n_val):
                if n_val <= 1:
                    return n_val
                return fib_recursive_local(n_val - 1) + fib_recursive_local(n_val - 2)
            return (
                f"To return the {x}th Fibonacci number using recursion:\n\n"
                "```python\n"
                "def fibonacci_recursive(x):\n"
                "    if x <= 1:\n"
                "        return x\n"
                "    return fibonacci_recursive(x - 1) + fibonacci_recursive(x - 2)\n"
                "```\n\n"
                f" Output: {fib_recursive_local(x)}"
            )

    # 7. Generate a Fibonacci series of length {{length}} and display it
    elif "generate a fibonacci series" in q and "length" in q:
        match = re.search(r"length (\d+)", q)
        if match:
            length = int(match.group(1))
            a, b = 0, 1
            series = []
            for _ in range(length):
                series.append(a)
                a, b = b, a + b
            return (
                f"To generate a Fibonacci series of length {length}:\n\n"
                "```python\n"
                "def generate_series(n):\n"
                "    a, b = 0, 1\n"
                "    result = []\n"
                "    for _ in range(n):\n"
                "        result.append(a)\n"
                "        a, b = b, a + b\n"
                "    return result\n"
                "```\n\n"
                f" Output: {series}"
            )

    elif "loop-based algorithm" in q and "fibonacci number at position" in q:
        match = re.search(r"position (\d+)", q)
        if match:
            pos = int(match.group(1))
            calculated_fib = fib_two_vars_global(pos)
            return (
                f"Loop-based algorithm to find Fibonacci({pos}):\n\n"
                "```python\n"
                "def fibonacci(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
                "```\n\n"
                f"[Input]: {pos}\n"
                f"[Output]: {calculated_fib}"
            )


    elif "recursive function" in q and "fibonacci" in q and "where n" in q:
        match = re.search(r"n\s*=\s*(\d+)", q)
        if match:
            val = int(match.group(1))
            # Define recursive fib locally
            def fib_recursive_local(n_val):
                if n_val <= 1:
                    return n_val
                return fib_recursive_local(n_val - 1) + fib_recursive_local(n_val - 2)
            return (
                f"Recursive function to calculate Fibonacci({val}):\n\n"
                "```python\n"
                "def fibonacci_recursive(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)\n"
                "```\n\n"
                f"[Input]: {val}\n"
                f"[Output]: {fib_recursive_local(val)}"
            )


    elif "print the fibonacci series up to the" in q:
        match = re.search(r"up to the (\d+)", q)
        if match:
            limit = int(match.group(1))
            a, b = 0, 1
            series = []
            for _ in range(limit + 1):
                series.append(a)
                a, b = b, a + b
            return (
                f"Program to print Fibonacci series up to the {limit}th term:\n\n"
                "```python\n"
                "def fibonacci_series(limit):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(limit + 1):\n"
                "        print(a, end=' ')\n"
                "        a, b = b, a + b\n"
                "```\n\n"
                f"[Output]: {' '.join(map(str, series))}"
        )


    elif "display the fibonacci numbers from position 0 to" in q:
        match = re.search(r"to (\d+)", q)
        if match:
            end = int(match.group(1))
            a, b = 0, 1
            sequence = []
            for _ in range(end + 1):
                sequence.append(a)
                a, b = b, a + b
            return (
                f"Fibonacci numbers from position 0 to {end}:\n\n"
                "```python\n"
                "def fibonacci_range(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n + 1):\n"
                "        print(a, end=' ')\n"
                "        a, b = b, a + b\n"
                "```\n\n"
                f"[Output]: {' '.join(map(str, sequence))}"
            )


    elif "return a list of the first" in q and "fibonacci numbers" in q:
        match = re.search(r"first (\d+)", q)
        if match:
            count = int(match.group(1))
            def fib_list(n_val):
                result = []
                a, b = 0, 1
                for _ in range(n_val):
                    result.append(a)
                    a, b = b, a + b
                return result
            series = fib_list(count)
            return (
                f"Function to return list of first {count} Fibonacci numbers:\n\n"
                "```python\n"
                "def get_fibonacci_list(n):\n"
                "    a, b = 0, 1\n"
                "    result = []\n"
                "    for _ in range(n):\n"
                "        result.append(a)\n"
                "        a, b = b, a + b\n"
                "    return result\n"
                "```\n\n"
                f"[Output]: {series}"
            )

    elif "simple recursive function" in q and "fibonacci" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            n = int(match.group(1))
            # Define recursive fib locally
            def fib_recursive_local(n_val):
                if n_val <= 1:
                    return n_val
                return fib_recursive_local(n_val - 1) + fib_recursive_local(n_val - 2)
            return (
                f"Simple recursive function to calculate Fibonacci({n}):\n\n"
                "```python\n"
                "def fibonacci_recursive(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)\n"
                "```\n\n"
                f"[Output]: {fib_recursive_local(n)}"
            )

    elif "compute and print the 0th to" in q:
        match = re.search(r"to (\d+)", q)
        if match:
            last = int(match.group(1))
            sequence = []
            a, b = 0, 1
            for _ in range(last + 1):
                sequence.append(a)
                a, b = b, a + b
            return (
                f"Compute and print Fibonacci numbers from 0th to {last}th:\n\n"
                "```python\n"
                "def print_fibonacci_range(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n + 1):\n"
                "        print(a, end=' ')\n"
                "        a, b = b, a + b\n"
                "```\n\n"
                f"[Output]: {' '.join(map(str, sequence))}"
            )


    elif "print the first" in q and "elements of the fibonacci sequence" in q:
        match = re.search(r"first (\d+)", q)
        if match:
            k = int(match.group(1))
            a, b = 0, 1
            elements = []
            for _ in range(k):
                elements.append(a)
                a, b = b, a + b
            return (
                f"First {k} elements of the Fibonacci sequence:\n\n"
                "```python\n"
                "def print_fibonacci(k):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(k):\n"
                "        print(a, end=' ')\n"
                "        a, b = b, a + b\n"
                "```\n\n"
                f"[Output]: {' '.join(map(str, elements))}"
            )


    return "Answer generation for this Fibonacci question is not implemented yet."


# --- Level 2 Algorithmic Questions ---
def answer_algo_lvl2_fibonacci(question):
    q = question.lower()

    # 1. Implement Fibonacci({{n}}) using memoization to avoid redundant calculations.
    if "fibonacci" in q and "memoization to avoid redundant calculations" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            n = int(match.group(1))
            # Create a new memo for each call to avoid state pollution between test cases
            _memo_instance = {}
            def fib_memo_local(n_val):
                if n_val in _memo_instance: #
                    return _memo_instance[n_val]
                if n_val <= 1: #
                    return n_val
                _memo_instance[n_val] = fib_memo_local(n_val - 1) + fib_memo_local(n_val - 2) #
                return _memo_instance[n_val]

            return (
                f"To implement Fibonacci({n}) using memoization to avoid redundant calculations:\n\n"
                "Memoization stores the results of expensive function calls and returns the cached result when the same inputs occur again.\n\n"
                "```python\n"
                "memo = {}\n"
                "def fib_memo(n):\n"
                "    if n in memo:\n"
                "        return memo[n]\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    memo[n] = fib_memo(n - 1) + fib_memo(n - 2)\n"
                "    return memo[n]\n"
                "```\n\n"
                f"Input: {n}\n"
                f"Output: {fib_memo_local(n)}\n"
                "Time Complexity: O(n)\n"
                "Space Complexity: O(n) for the memoization table"
            )

    # 2. Write a bottom-up dynamic programming approach to compute Fibonacci({{k}}).
    elif "bottom-up dynamic programming approach to compute fibonacci" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            k = int(match.group(1))
            # Use global helper
            calculated_fib = fib_bottom_up_global(k)
            return (
                f"To compute Fibonacci({k}) using a bottom-up dynamic programming approach:\n\n"
                "Bottom-up DP (tabulation) builds the solution from the base cases up.\n\n"
                "```python\n"
                "def fib_bottom_up(k):\n"
                "    if k <= 1:\n"
                "        return k\n"
                "    dp = [0] * (k + 1)\n"
                "    dp[0] = 0\n"
                "    dp[1] = 1\n"
                "    for i in range(2, k + 1):\n"
                "        dp[i] = dp[i - 1] + dp[i - 2]\n"
                "    return dp[k]\n"
                "```\n\n"
                f"Input: {k}\n"
                f"Output: {calculated_fib}\n"
                "Time Complexity: O(n)\n"
                "Space Complexity: O(n) for the DP table"
            )

    # 3. Modify the Fibonacci function to also return the total number of function calls made for n = {{val}}.
    elif "return the total number of function calls made for n" in q:
        match = re.search(r"n\s*=\s*(\d+)", q)
        if match:
            val = int(match.group(1))
            # Naive recursive Fibonacci with call counting (local scope)
            _call_count_naive = 0
            def fib_naive_count_local(n_val_local):
                nonlocal _call_count_naive
                _call_count_naive += 1
                if n_val_local <= 1:
                    return n_val_local
                return fib_naive_count_local(n_val_local - 1) + fib_naive_count_local(n_val_local - 2)

            fib_value_naive = fib_naive_count_local(val)

            # Memoized Fibonacci with call counting (local scope)
            _call_count_memo = 0
            _memo_for_counts_local = {}

            def fib_memo_count_local(n_val_local):
                nonlocal _call_count_memo
                _call_count_memo += 1
                if n_val_local in _memo_for_counts_local:
                    return _memo_for_counts_local[n_val_local]
                if n_val_local <= 1:
                    result = n_val_local
                else:
                    result = fib_memo_count_local(n_val_local - 1) + fib_memo_count_local(n_val_local - 2)
                _memo_for_counts_local[n_val_local] = result
                return result

            fib_value_memo = fib_memo_count_local(val)

            return (
                f"To modify the Fibonacci function to return the total number of function calls made for n = {val}:\n\n"
                "**1. Naive Recursive Approach (for comparison of calls):**\n"
                "```python\n"
                "call_count_naive = 0\n"
                "def fib_naive_count(n):\n"
                "    global call_count_naive \n"
                "    call_count_naive += 1\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    return fib_naive_count(n - 1) + fib_naive_count(n - 2)\n"
                "```\n"
                f"For n = {val}, Fibonacci({val}) = {fib_value_naive}, Naive Recursive Calls: {_call_count_naive}\n\n"
                "**2. Memoized Recursive Approach (optimized calls):**\n"
                "```python\n"
                "memo = {}\n"
                "call_count_memo = 0\n"
                "def fib_memo_count(n):\n"
                "    global call_count_memo\n"
                "    call_count_memo += 1\n"
                "    if n in memo:\n"
                "        return memo[n]\n"
                "    if n <= 1:\n"
                "        result = n\n"
                "    else:\n"
                "        result = fib_memo_count(n - 1) + fib_memo_count(n - 2)\n"
                "    memo[n] = result\n"
                "    return result\n"
                "```\n"
                f"For n = {val}, Fibonacci({val}) = {fib_value_memo}, Memoized Recursive Calls: {_call_count_memo}\n"
                "The memoized version significantly reduces redundant calls."
            )

    # 4. Implement a Fibonacci function that uses memoization with a dictionary for n = {{n}}.
    elif "memoization with a dictionary for n" in q:
        match = re.search(r"n\s*=\s*(\d+)", q)
        if match:
            n_val = int(match.group(1))
            # Create a new memo for each call to avoid state pollution between test cases
            _fib_cache_dict_instance = {} #
            def fibonacci_dict_memo_local(n_arg):
                if n_arg in _fib_cache_dict_instance: #
                    return _fib_cache_dict_instance[n_arg]
                if n_arg <= 1:
                    result = n_arg
                else:
                    result = fibonacci_dict_memo_local(n_arg - 1) + fibonacci_dict_memo_local(n_arg - 2)
                _fib_cache_dict_instance[n_arg] = result #
                return result

            return (
                f"To implement a Fibonacci function that uses memoization with a dictionary for n = {n_val}:\n\n"
                "A dictionary acts as a cache to store already computed Fibonacci values, using 'n' as the key.\n\n"
                "```python\n"
                "fib_cache = {}\n"
                "def fibonacci_dict_memo(n):\n"
                "    if n in fib_cache:\n"
                "        return fib_cache[n]\n"
                "    if n <= 1:\n"
                "        result = n\n"
                "    else:\n"
                "        result = fibonacci_dict_memo(n - 1) + fibonacci_dict_memo(n - 2)\n"
                "    fib_cache[n] = result\n"
                "    return result\n"
                "```\n\n"
                f"Input: {n_val}\n"
                f"Output: {fibonacci_dict_memo_local(n_val)}\n"
                "Time Complexity: O(n)\n"
                "Space Complexity: O(n)"
            )

    # 5. Create an iterative approach for Fibonacci({{x}}) using only two variables.
    elif "iterative approach for fibonacci" in q and "using only two variables" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            x_val = int(match.group(1))
            # Use global helper
            calculated_fib = fib_two_vars_global(x_val)

            return (
                f"To create an iterative approach for Fibonacci({x_val}) using only two variables:\n\n"
                "This method optimizes space complexity to O(1) by only keeping track of the two previous Fibonacci numbers.\n\n"
                "```python\n"
                "def fib_two_variables(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b # Update 'a' to old 'b', 'b' to new sum\n"
                "    return b\n"
                "```\n\n"
                f"Input: {x_val}\n"
                f"Output: {calculated_fib}\n"
                "Time Complexity: O(n)\n"
                "Space Complexity: O(1)"
            )

    # 6. Write a function that returns the sum of the first {{count}} Fibonacci numbers.
    elif "sum of the first" in q and "fibonacci numbers" in q:
        match = re.search(r"first (\d+)", q)
        if match:
            count = int(match.group(1))
            # Use global helper
            calculated_sum = sum_fib_seq_global(count)
            return (
                f"To write a function that returns the sum of the first {count} Fibonacci numbers:\n\n"
                "```python\n"
                "def sum_first_fibonacci(count):\n"
                "    if count <= 0:\n"
                "        return 0\n"
                "    a, b = 0, 1\n"
                "    total_sum = 0\n"
                "    for _ in range(count):\n"
                "        total_sum += a\n"
                "        a, b = b, a + b\n"
                "    return total_sum\n"
                "```\n\n"
                f"Input: {count}\n"
                f"Output: {calculated_sum}"
            )

    # 7. Implement a function to generate the first {{n}} even Fibonacci numbers.
    elif "generate the first" in q and "even fibonacci numbers" in q:
        match = re.search(r"first (\d+)", q)
        if match:
            n_even = int(match.group(1))
            # Use global helper
            even_fibs = get_even_fibonacci_global(n_even)

            return (
                f"To implement a function to generate the first {n_even} even Fibonacci numbers:\n\n"
                "```python\n"
                "def generate_even_fibonacci(n_count):\n"
                "    even_fib_list = []\n"
                "    a, b = 0, 1\n"
                "    while len(even_fib_list) < n_count:\n"
                "        if a % 2 == 0:\n"
                "            even_fib_list.append(a)\n"
                "        a, b = b, a + b\n"
                "    return even_fib_list\n"
                "```\n\n"
                f"Input: {n_even}\n"
                f"Output: {even_fibs}"
            )

    # 8. Compute the Fibonacci number at index {{idx}} using tabulation with an array.
    elif "fibonacci number at index" in q and "tabulation with an array" in q:
        match = re.search(r"index (\d+)", q)
        if match:
            idx = int(match.group(1))
            # Use global helper
            calculated_fib = fib_tabulation_array_func_global(idx)
            return (
                f"To compute the Fibonacci number at index {idx} using tabulation with an array:\n\n"
                "Tabulation is a bottom-up dynamic programming approach that fills a table (array) with computed values from base cases to the desired result.\n\n"
                "```python\n"
                "def fib_tabulation_array(idx):\n"
                "    if idx <= 1:\n"
                "        return idx\n"
                "    dp = [0] * (idx + 1)\n"
                "    dp[0] = 0\n"
                "    dp[1] = 1\n"
                "    for i in range(2, idx + 1):\n"
                "        dp[i] = dp[i - 1] + dp[i - 2]\n"
                "    return dp[idx]\n"
                "```\n\n"
                f"Input: {idx}\n"
                f"Output: {calculated_fib}\n"
                "Time Complexity: O(n)\n"
                "Space Complexity: O(n)"
            )

    # 9. Write a program to compute Fibonacci({{n}}) using tail recursion.
    elif "fibonacci" in q and "using tail recursion" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            n_val = int(match.group(1))
            # Use global helper
            calculated_fib = fib_tail_recursive_func_global(n_val)

            return (
                f"To compute Fibonacci({n_val}) using tail recursion:\n\n"
                "Tail recursion can sometimes be optimized by compilers to iterative loops, preventing stack overflow for large 'n'.\n\n"
                "```python\n"
                "def fib_tail_recursive(n, a=0, b=1):\n"
                "    if n == 0:\n"
                "        return a\n"
                "    if n == 1:\n"
                "        return b\n"
                "    return fib_tail_recursive(n - 1, b, a + b)\n"
                "```\n\n"
                f"Input: {n_val}\n"
                f"Output: {calculated_fib}\n"
                "Time Complexity: O(n)\n"
                "Space Complexity: O(n) for stack frames (can be O(1) if optimized by compiler)"
            )

    # 10. Implement a function that finds the largest Fibonacci number smaller than {{limit}}.
    elif "largest fibonacci number smaller than" in q:
        match = re.search(r"smaller than (\d+)", q)
        if match:
            limit = int(match.group(1))
            # Use global helper
            largest_fib = largest_fib_smaller_than_limit_global(limit)

            return (
                f"To implement a function that finds the largest Fibonacci number smaller than {limit}:\n\n"
                "```python\n"
                "def find_largest_fib_smaller(limit):\n"
                "    a, b = 0, 1\n"
                "    largest_fib = 0\n"
                "    while a < limit:\n"
                "        largest_fib = a\n"
                "        a, b = b, a + b\n"
                "    return largest_fib\n"
                "```\n\n"
                f"Input: {limit}\n"
                f"Output: {largest_fib}\n"
                "Time Complexity: O(log(limit)) because Fibonacci numbers grow exponentially."
            )

    # 11. Create a function that returns all Fibonacci numbers less than {{x}}.
    elif "returns all fibonacci numbers less than" in q:
        match = re.search(r"less than (\d+)", q)
        if match:
            x_val = int(match.group(1))
            # Use global helper
            fib_numbers = get_all_fib_less_than_x_global(x_val)

            return (
                f"To create a function that returns all Fibonacci numbers less than {x_val}:\n\n"
                "```python\n"
                "def get_fibonacci_numbers_less_than(x):\n"
                "    fib_numbers = []\n"
                "    a, b = 0, 1\n"
                "    while a < x:\n"
                "        fib_numbers.append(a)\n"
                "        a, b = b, a + b\n"
                "    return fib_numbers\n"
                "```\n\n"
                f"Input: {x_val}\n"
                f"Output: {fib_numbers}"
            )

    # 12. Using memoization, calculate Fibonacci({{val}}) with a cache to track reused subproblems.
    elif "fibonacci" in q and "with a cache to track reused subproblems" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            val = int(match.group(1))
            # Create a new memo and reuse counter for each call to avoid state pollution
            _fib_cache_tracking_instance = {}
            _reused_count_instance = 0

            def fib_with_tracking_local(n_val_local):
                nonlocal _reused_count_instance
                if n_val_local in _fib_cache_tracking_instance:
                    _reused_count_instance += 1
                    return _fib_cache_tracking_instance[n_val_local]
                if n_val_local <= 1:
                    result = n_val_local
                else:
                    result = fib_with_tracking_local(n_val_local - 1) + fib_with_tracking_local(n_val_local - 2)
                _fib_cache_tracking_instance[n_val_local] = result
                return result

            fib_result = fib_with_tracking_local(val)

            return (
                f"To calculate Fibonacci({val}) using memoization with a cache to track reused subproblems:\n\n"
                "The cache (dictionary) stores results, and a counter tracks how many times a stored result is reused, demonstrating the efficiency gain from avoiding redundant calculations.\n\n"
                "```python\n"
                "fib_cache = {}\n"
                "reused_count = 0\n"
                "def fib_with_tracking(n):\n"
                "    global reused_count # For demonstration, better encapsulated in a class\n"
                "    if n in fib_cache:\n"
                "        reused_count += 1 # Increment if subproblem is reused\n"
                "        return fib_cache[n]\n"
                "    if n <= 1:\n"
                "        result = n\n"
                "    else:\n"
                "        result = fib_with_tracking(n - 1) + fib_with_tracking(n - 2)\n"
                "    fib_cache[n] = result\n"
                "    return result\n"
                "```\n\n"
                f"Input: {val}\n"
                f"Output: Fibonacci({val}) = {fib_result}\n"
                f"Reused Subproblems from Cache: {_reused_count_instance}"
            )

    # 13. Modify your iterative Fibonacci algorithm to print only odd Fibonacci numbers up to the {{n}}th term.
    elif "print only odd fibonacci numbers up to the" in q:
        match = re.search(r"up to the (\d+)th term", q)
        if match:
            n_term = int(match.group(1))
            odd_fibs = []
            a, b = 0, 1
            for i in range(n_term + 1):
                if a % 2 != 0:
                    odd_fibs.append(a)
                a, b = b, a + b
            
            return (
                f"To modify the iterative Fibonacci algorithm to print only odd Fibonacci numbers up to the {n_term}th term:\n\n"
                "This iterative approach generates Fibonacci numbers and checks for odd parity before adding them to the list.\n\n"
                "```python\n"
                "def print_odd_fibonacci_up_to_n(n_term):\n"
                "    odd_fibs = []\n"
                "    a, b = 0, 1\n"
                "    for i in range(n_term + 1):\n"
                "        if a % 2 != 0:\n"
                "            odd_fibs.append(a)\n"
                "        a, b = b, a + b\n"
                "    return odd_fibs\n"
                "```\n\n"
                f"Input: {n_term}\n"
                f"Output: {odd_fibs}"
            )

    # 14. Write a function that returns True if a number {{num}} is a Fibonacci number.
    elif "returns true if a number" in q and "is a fibonacci number" in q:
        match = re.search(r"number (\d+)", q)
        if match:
            num_check = int(match.group(1))
            # Use global helper
            is_fib = is_fibonacci_num_global(num_check)

            return (
                f"To write a function that returns True if a number {num_check} is a Fibonacci number:\n\n"
                "This function uses a mathematical property of Fibonacci numbers: a positive integer 'n' is a Fibonacci number if and only if `(5*n^2 + 4)` or `(5*n^2 - 4)` is a perfect square. This property is related to Binet's formula for the nth Fibonacci number.\n\n"
                "```python\n"
                "import math\n\n"
                "def is_perfect_square(k):\n"
                "    s = int(math.sqrt(k))\n"
                "    return s * s == k\n\n"
                "def is_fibonacci_number(num):\n"
                "    return is_perfect_square(5 * num * num + 4) or \\\n"
                "           is_perfect_square(5 * num * num - 4)\n"
                "```\n\n"
                f"Input: {num_check}\n"
                f"Output: {is_fib}"
            )

    # 15. Print Fibonacci numbers between two given positions: start = {{start}}, end = {{end}}.
    elif "print fibonacci numbers between two given positions" in q:
        match = re.search(r"start = (\d+), end = (\d+)", q)
        if match:
            start_pos = int(match.group(1))
            end_pos = int(match.group(2))
            
            # Use global helper
            fib_sequence_range = print_fibonacci_between_positions_global(start_pos, end_pos)

            return (
                f"To print Fibonacci numbers between two given positions: start = {start_pos}, end = {end_pos}:\n\n"
                "This function iteratively generates Fibonacci numbers and collects those within the specified index range.\n\n"
                "```python\n"
                "def print_fibonacci_between_positions(start, end):\n"
                "    fib_nums = []\n"
                "    a, b = 0, 1\n"
                "    for i in range(end + 1):\n"
                "        if i >= start:\n"
                "            fib_nums.append(a)\n"
                "        a, b = b, a + b\n"
                "    return fib_nums\n"
                "```\n\n"
                f"Input: start = {start_pos}, end = {end_pos}\n"
                f"Output: {fib_sequence_range}"
            )

    # 16. Optimize your Fibonacci algorithm to handle inputs up to n = {{maxn}} efficiently.
    elif "optimize your fibonacci algorithm to handle inputs up to n" in q and "efficiently" in q:
        match = re.search(r"n = (\d+)", q)
        if match:
            maxn = int(match.group(1))
            return (
                f"To optimize the Fibonacci algorithm to handle inputs up to n = {maxn} efficiently:\n\n"
                "For large 'n', iterative (bottom-up DP with O(1) space) or matrix exponentiation (O(log n) time) are the most efficient approaches. Naive recursion is highly inefficient due to redundant calculations.\n\n"
                "**1. Space-Optimized Iterative Approach (O(n) time, O(1) space):**\n"
                "This is generally preferred for its simplicity and efficiency for inputs up to a moderately large 'n' (e.g., up to ~10^6).\n"
                "```python\n"
                "def fibonacci_optimized_iterative(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
                "```\n\n"
                "**2. Matrix Exponentiation (O(log n) time):**\n"
                "For extremely large 'n' (e.g., up to 10^18), matrix exponentiation is the most efficient. It leverages the property that F(n) can be found by raising a specific matrix to the power of n.\n"
                "```python\n"
                "def multiply_matrices(A, B):\n"
                "    x = A[0][0] * B[0][0] + A[0][1] * B[1][0]\n"
                "    y = A[0][0] * B[0][1] + A[0][1] * B[1][1]\n"
                "    z = A[1][0] * B[0][0] + A[1][1] * B[1][0]\n"
                "    w = A[1][0] * B[0][1] + A[1][1] * B[1][1]\n"
                "    return [[x, y], [z, w]]\n\n"
                "def power(M, n):\n"
                "    result = [[1, 0], [0, 1]] # Identity matrix\n"
                "    M_copy = M\n"
                "    while n > 0:\n"
                "        if n % 2 == 1:\n"
                "            result = multiply_matrices(result, M_copy)\n"
                "        M_copy = multiply_matrices(M_copy, M_copy)\n"
                "        n //= 2\n"
                "    return result\n\n"
                "def fibonacci_matrix_exponentiation(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    matrix = [[1, 1], [1, 0]]\n"
                "    result_matrix = power(matrix, n - 1)\n"
                "    return result_matrix[0][0]\n"
                "```\n\n"
                f"For n = {maxn}:\n"
                f"  Iterative (O(1) space): {fibonacci_optimized_iterative(maxn)}\n"
                f"  Matrix Exponentiation (O(log n)): {fibonacci_matrix_exponentiation(maxn)}\n"
                "The choice depends on the maximum value of 'n' and specific system constraints."
            )

    # 17. Write a function that returns the average of the first {{n}} Fibonacci numbers.
    elif "average of the first" in q and "fibonacci numbers" in q:
        match = re.search(r"first (\d+)", q)
        if match:
            n_avg = int(match.group(1))
            # Use global helper
            calculated_average = sum_fib_seq_global(n_avg) / n_avg if n_avg > 0 else 0.0

            return (
                f"To write a function that returns the average of the first {n_avg} Fibonacci numbers:\n\n"
                "This function calculates the sum of the first 'n' Fibonacci numbers and then divides by 'n' to find the average.\n\n"
                "```python\n"
                "def get_average_of_first_fibonacci(n):\n"
                "    if n <= 0:\n"
                "        return 0.0\n"
                "    a, b = 0, 1\n"
                "    total_sum = 0\n"
                "    for _ in range(n):\n"
                "        total_sum += a\n"
                "        a, b = b, a + b\n"
                "    return total_sum / n\n"
                "```\n\n"
                f"Input: {n_avg}\n"
                f"Output: {calculated_average}"
            )

    # 18. Implement a program that computes Fibonacci({{n}}) and prints all intermediate values.
    elif "prints all intermediate values" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            n_val = int(match.group(1))
            # Use global helper
            intermediate_values = get_intermediate_fib_values(n_val)
            
            # For printing the intermediate values as part of the output string
            intermediate_output_str = ""
            for i, val in enumerate(intermediate_values):
                intermediate_output_str += f"Fib({i}): {val}\n"

            final_fib_val = intermediate_values[n_val] if n_val < len(intermediate_values) else 'Not computed'

            return (
                f"To implement a program that computes Fibonacci({n_val}) and prints all intermediate values:\n\n"
                "This iterative program calculates Fibonacci numbers sequentially and stores each intermediate value. This is typically done with a bottom-up (tabulation) approach or a simple iterative loop.\n\n"
                "```python\n"
                "def fibonacci_with_intermediate_values(n):\n"
                "    intermediate_values = []\n"
                "    if n <= 1:\n"
                "        intermediate_values.append(n)\n"
                "        print(f'Fib({n}): {n}')\n"
                "        return n, intermediate_values\n"
                "    \n"
                "    a, b = 0, 1\n"
                "    for i in range(n + 1):\n"
                "        intermediate_values.append(a)\n"
                "        print(f'Fib({i}): {a}') # Print intermediate value\n"
                "        a, b = b, a + b\n"
                "    return intermediate_values[-1], intermediate_values # Return Fib(n) and all values\n"
                "```\n\n"
                f"Input: {n_val}\n"
                f"Intermediate Values computed (Fib(0) to Fib({n_val})): \n{intermediate_output_str}"
                f"Final Fibonacci({n_val}): {final_fib_val}"
            )

    return "Answer generation for this Level 2 Fibonacci question is not implemented yet."


# --- Level 3 Algorithmic Questions ---
def answer_algo_lvl3_fibonacci(question):
    q = question.lower()

    # 1. Implement Fibonacci({{n}}) using matrix exponentiation to achieve O(log n) time.
    if "fibonacci" in q and "using matrix exponentiation to achieve o(log n) time" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            n = int(match.group(1))
            calculated_fib = fibonacci_matrix_exponentiation(n)
            return (
                f"To implement Fibonacci({n}) using matrix exponentiation to achieve O(log n) time:\n\n"
                "This method leverages the property that the nth Fibonacci number can be found by raising a specific 2x2 matrix ([[1, 1], [1, 0]]) to the power of (n-1). Matrix exponentiation itself uses binary exponentiation (repeated squaring), leading to O(log n) time complexity.\n\n"
                "```python\n"
                "def multiply_matrices(A, B):\n"
                "    x = A[0][0] * B[0][0] + A[0][1] * B[1][0]\n"
                "    y = A[0][0] * B[0][1] + A[0][1] * B[1][1]\n"
                "    z = A[1][0] * B[0][0] + A[1][1] * B[1][0]\n"
                "    w = A[1][0] * B[0][1] + A[1][1] * B[1][1]\n"
                "    return [[x, y], [z, w]]\n\n"
                "def power(M, n):\n"
                "    result = [[1, 0], [0, 1]] # Identity matrix\n"
                "    M_copy = M\n"
                "    while n > 0:\n"
                "        if n % 2 == 1:\n"
                "            result = multiply_matrices(result, M_copy)\n"
                "        M_copy = multiply_matrices(M_copy, M_copy)\n"
                "        n //= 2\n"
                "    return result\n\n"
                "def fibonacci_matrix_exponentiation(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    matrix = [[1, 1], [1, 0]]\n"
                "    result_matrix = power(matrix, n - 1)\n"
                "    return result_matrix[0][0]\n"
                "```\n\n"
                f"Input: {n}\n"
                f"Output: {calculated_fib}\n"
                "Time Complexity: O(log n)\n"
                "Space Complexity: O(log n) for recursion stack in `power` function (can be O(1) iterative)"
            )

    # 2. Write a program to compute the {{n}}th Fibonacci number modulo {{mod}}.
    elif "compute the" in q and "fibonacci number modulo" in q:
        match = re.search(r"the (\d+)th Fibonacci number modulo (\d+)", q)
        if match:
            n = int(match.group(1))
            mod = int(match.group(2))
            calculated_fib_mod = fibonacci_matrix_exponentiation(n, mod)
            return (
                f"To compute the {n}th Fibonacci number modulo {mod}:\n\n"
                "Matrix exponentiation is ideal for modular arithmetic with Fibonacci numbers, as all intermediate multiplication results can be taken modulo `mod`. This prevents numbers from becoming excessively large while maintaining correctness.\n\n"
                "```python\n"
                "def multiply_matrices_mod(A, B, mod):\n"
                "    x = (A[0][0] * B[0][0] + A[0][1] * B[1][0]) % mod\n"
                "    y = (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % mod\n"
                "    z = (A[1][0] * B[0][0] + A[1][1] * B[1][0]) % mod\n"
                "    w = (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % mod\n"
                "    return [[x, y], [z, w]]\n\n"
                "def power_mod(M, n, mod):\n"
                "    result = [[1, 0], [0, 1]] # Identity matrix\n"
                "    M_copy = M\n"
                "    while n > 0:\n"
                "        if n % 2 == 1:\n"
                "            result = multiply_matrices_mod(result, M_copy, mod)\n"
                "        M_copy = multiply_matrices_mod(M_copy, M_copy, mod)\n"
                "        n //= 2\n"
                "    return result\n\n"
                "def fibonacci_matrix_mod(n, mod):\n"
                "    if n <= 1:\n"
                "        return n % mod # Ensure base cases are also modulo mod\n"
                "    matrix = [[1, 1], [1, 0]]\n"
                "    result_matrix = power_mod(matrix, n - 1, mod)\n"
                "    return result_matrix[0][0] % mod\n"
                "```\n\n"
                f"Input: n = {n}, mod = {mod}\n"
                f"Output: {calculated_fib_mod}\n"
                "Time Complexity: O(log n)\n"
                "Space Complexity: O(log n)"
            )

    # 3. Create a function to compute large Fibonacci numbers (e.g., n = {{n}}) using fast doubling technique.
    elif "compute large fibonacci numbers" in q and "using fast doubling technique" in q:
        match = re.search(r"n = (\d+)", q)
        if match:
            n = int(match.group(1))
            calculated_fib, _ = fib_fast_doubling(n) # fast doubling returns (F(n), F(n+1))
            return (
                f"To compute large Fibonacci numbers (e.g., n = {n}) using the fast doubling technique:\n\n"
                "The fast doubling method computes F(2k) and F(2k+1) efficiently from F(k) and F(k+1). This recursive approach effectively halves the problem size at each step, similar to matrix exponentiation, achieving O(log n) time complexity. It can be more efficient in practice for large numbers as it avoids general matrix multiplication overhead.\n\n"
                "```python\n"
                "def fib_fast_doubling(n):\n"
                "    if n == 0:\n"
                "        return (0, 1) # Returns (F(0), F(1))\n"
                "    \n"
                "    a, b = fib_fast_doubling(n // 2) # (F(k), F(k+1)) where k = n // 2\n"
                "    \n"
                "    # Identities:\n"
                "    # F(2k) = F(k) * (2*F(k+1) - F(k))\n"
                "    # F(2k+1) = F(k+1)^2 + F(k)^2\n"
                "    \n"
                "    c = a * (2 * b - a) # F(2k)\n"
                "    d = b * b + a * a   # F(2k+1)\n"
                "    \n"
                "    if n % 2 == 0: # If n is even, n = 2k\n"
                "        return (c, d) # Returns (F(n), F(n+1))\n"
                "    else: # If n is odd, n = 2k+1\n"
                "        return (d, c + d) # Returns (F(n), F(n+1))\n"
                "```\n\n"
                f"Input: n = {n}\n"
                f"Output: {calculated_fib}\n"
                "Time Complexity: O(log n)\n"
                "Space Complexity: O(log n) for recursion stack"
            )

    # 4. Implement a space-optimized Fibonacci algorithm that computes Fibonacci({{n}}) using O(1) space.
    elif "space-optimized fibonacci algorithm that computes fibonacci" in q and "using o(1) space" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            n = int(match.group(1))
            calculated_fib = fib_two_vars_global(n)
            return (
                f"To implement a space-optimized Fibonacci algorithm that computes Fibonacci({n}) using O(1) space:\n\n"
                "This iterative approach only stores the two most recent Fibonacci numbers at any given time, thus requiring a constant amount of memory regardless of 'n'. This is the most memory-efficient method for calculating a single Fibonacci number up to a practical limit of integer size.\n\n"
                "```python\n"
                "def fibonacci_o1_space(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
                "```\n\n"
                f"Input: {n}\n"
                f"Output: {calculated_fib}\n"
                "Time Complexity: O(n)\n"
                "Space Complexity: O(1)"
            )

    # 5. Write an efficient algorithm to compute the sum of the first {{n}} Fibonacci numbers modulo {{mod}}.
    elif "compute the sum of the first" in q and "fibonacci numbers modulo" in q:
        match = re.search(r"first (\d+) Fibonacci numbers modulo (\d+)", q)
        if match:
            n = int(match.group(1))
            mod = int(match.group(2))
            
            # Identity: Sum(F_i for i=0 to k) = F_{k+2} - 1
            # Here, we want sum of first 'n' numbers, which means F_0 to F_{n-1}.
            # So, k = n-1. The sum is F_{(n-1)+2} - 1 = F_{n+1} - 1.
            # No, the problem asks for the sum of the first N Fibonacci numbers (F_0 to F_N-1).
            # So, S_N = F_{N+2} - 1
            # The identity is Sum_{i=0 to n} F_i = F_{n+2} - 1.
            # If the user means first 'n' *terms* (F_0, F_1, ..., F_{n-1}), then use (n-1)+2 = n+1.
            # So sum is F_{n+1} - 1.

            # Example: sum of first 3 fib numbers (0, 1, 1) = 2. F(3+1)-1 = F(4)-1 = 3-1 = 2.
            # So, F_{n+1} - 1.
            
            sum_fib_mod = sum_fib_numbers_formula(n, mod)

            return (
                f"To compute the sum of the first {n} Fibonacci numbers modulo {mod}:\n\n"
                "The sum of the first 'N' Fibonacci numbers (from F0 to F(N-1)) can be efficiently computed using the identity: `Sum(F_i for i=0 to N-1) = F_{N+1} - 1`. We can compute F_{N+1} using matrix exponentiation with modular arithmetic, and then subtract 1 (taking care of negative results in modulo).\n\n"
                "```python\n"
                "# (Requires multiply_matrices_mod, power_mod, fibonacci_matrix_mod from above)\n"
                "def sum_first_fibonacci_mod(n, mod):\n"
                "    if n == 0:\n"
                "        return 0\n"
                "    # Sum of F_0 to F_{n-1} is F_{n+1} - 1\n"
                "    fib_n_plus_1 = fibonacci_matrix_mod(n + 1, mod) # Reusing helper function\n"
                "    return (fib_n_plus_1 - 1 + mod) % mod\n"
                "```\n\n"
                f"Input: n = {n}, mod = {mod}\n"
                f"Output: {sum_fib_mod}\n"
                "Time Complexity: O(log n)\n"
                "Space Complexity: O(log n)"
            )

    # 6. Compare performance of recursive, iterative, and matrix-based Fibonacci implementations for n = {{n}}.
    elif "compare performance of recursive, iterative, and matrix-based fibonacci implementations" in q:
        match = re.search(r"n = (\d+)", q)
        if match:
            n = int(match.group(1))

            results = {}

            # Recursive (naive)
            def fib_naive_compare(n_val):
                if n_val <= 1: return n_val
                return fib_naive_compare(n_val - 1) + fib_naive_compare(n_val - 2)
            
            start_time = time.perf_counter()
            try:
                fib_recursive_val = fib_naive_compare(n)
                end_time = time.perf_counter()
                results['Recursive (Naive)'] = {'value': fib_recursive_val, 'time': (end_time - start_time) * 1000}
            except RecursionError:
                results['Recursive (Naive)'] = {'value': 'RecursionError (n too large)', 'time': 'N/A'}


            # Iterative (O(1) space)
            start_time = time.perf_counter()
            fib_iterative_val = fibonacci_optimized_iterative(n)
            end_time = time.perf_counter()
            results['Iterative (O(1) Space)'] = {'value': fib_iterative_val, 'time': (end_time - start_time) * 1000}

            # Matrix-based (O(log n))
            start_time = time.perf_counter()
            fib_matrix_val = fibonacci_matrix_exponentiation(n)
            end_time = time.perf_counter()
            results['Matrix Exponentiation (O(log n))'] = {'value': fib_matrix_val, 'time': (end_time - start_time) * 1000}
            
            comparison_str = f"Performance comparison for Fibonacci({n}):\n\n"
            comparison_str += "| Approach               | Fibonacci Value | Time (ms)        | Time Complexity |\n"
            comparison_str += "|------------------------|-----------------|------------------|-----------------|\n"
            for name, data in results.items():
                time_str = f"{data['time']:.4f}" if isinstance(data['time'], float) else data['time']
                comparison_str += f"| {name:<22} | {str(data['value']):<15} | {time_str:<16} | {'O(2^n)' if 'Recursive' in name else 'O(n)' if 'Iterative' in name else 'O(log n)'} |\n"
            
            comparison_str += "\nObservations:\n"
            comparison_str += "* **Recursive (Naive)**: Exponential time complexity, becoming impractical for even moderate 'n' due to redundant calculations.\n"
            comparison_str += "* **Iterative (O(1) Space)**: Linear time complexity (O(n)), highly efficient for practical 'n' values, and uses minimal memory.\n"
            comparison_str += "* **Matrix Exponentiation (O(log n))**: Logarithmic time complexity, making it the fastest for very large 'n' (e.g., beyond millions)."

            return comparison_str

    # 7. Design an algorithm to return the last digit of the {{n}}th Fibonacci number efficiently.
    elif "return the last digit of the" in q and "fibonacci number efficiently" in q:
        match = re.search(r"the (\d+)th Fibonacci number efficiently", q)
        if match:
            n = int(match.group(1))
            # The last digit of Fibonacci numbers repeats with a Pisano period of 60 for modulo 10.
            # This means F(n) % 10 is the same as F(n % 60) % 10.
            pisano_period_10 = 60
            effective_n = n % pisano_period_10

            # Calculate F(effective_n) and take modulo 10
            # Use space-optimized iterative for efficiency
            last_digit = fib_two_vars_global(effective_n) % 10

            return (
                f"To design an algorithm to return the last digit of the {n}th Fibonacci number efficiently:\n\n"
                "The last digit of Fibonacci numbers repeats in a cycle (called the Pisano period). For modulo 10 (i.e., the last digit), the Pisano period is 60. This means `F(n) % 10` is equivalent to `F(n % 60) % 10`. We can compute a Fibonacci number for a much smaller `n` using an efficient iterative approach and then take its last digit.\n\n"
                "```python\n"
                "def get_last_digit_fibonacci(n):\n"
                "    if n < 0: return None # Or handle error\n"
                "    if n <= 1: return n\n"
                "    \n"
                "    # Pisano period for modulo 10 is 60\n"
                "    effective_n = n % 60 \n"
                "    \n"
                "    if effective_n == 0: return 0 # F(60)%10 is 0, F(0)%10 is 0\n"
                "    \n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, effective_n + 1):\n"
                "        a, b = b, (a + b) % 10 # Only keep the last digit\n"
                "    return b\n"
                "```\n\n"
                f"Input: {n}\n"
                f"Output (last digit): {last_digit}\n"
                "Time Complexity: O(1) (effectively constant time due to Pisano period)\n"
                "Space Complexity: O(1)"
            )

    # 8. Develop a function to compute the sum of even Fibonacci numbers not exceeding {{limit}}.
    elif "sum of even fibonacci numbers not exceeding" in q:
        match = re.search(r"not exceeding (\d+)", q)
        if match:
            limit = int(match.group(1))
            
            sum_even_fibs = 0
            a, b = 0, 1
            while a <= limit:
                if a % 2 == 0:
                    sum_even_fibs += a
                a, b = b, a + b
            
            return (
                f"To develop a function to compute the sum of even Fibonacci numbers not exceeding {limit}:\n\n"
                "This function iteratively generates Fibonacci numbers and adds them to a running sum if they are even and do not exceed the given limit. This approach is straightforward and efficient for practical limits.\n\n"
                "```python\n"
                "def sum_even_fibonacci_up_to_limit(limit):\n"
                "    total_sum = 0\n"
                "    a, b = 0, 1\n"
                "    while a <= limit:\n"
                "        if a % 2 == 0:\n"
                "            total_sum += a\n"
                "        a, b = b, a + b\n"
                "    return total_sum\n"
                "```\n\n"
                f"Input: {limit}\n"
                f"Output: {sum_even_fibs}\n"
                "Time Complexity: O(log(limit)) (since Fibonacci numbers grow exponentially)\n"
                "Space Complexity: O(1)"
            )

    # 9. Compute Fibonacci({{n}}) using matrix exponentiation with modular arithmetic to handle large results.
    elif "fibonacci" in q and "using matrix exponentiation with modular arithmetic to handle large results" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            n = int(match.group(1))
            # Choose a large modulus, e.g., 10^9 + 7, common in competitive programming
            mod = 10**9 + 7
            calculated_fib_mod = fibonacci_matrix_exponentiation(n, mod)
            return (
                f"To compute Fibonacci({n}) using matrix exponentiation with modular arithmetic to handle large results:\n\n"
                "When Fibonacci numbers become very large (exceeding standard integer types), computing them modulo a specific number is often required. Matrix exponentiation allows applying the modulo operation at each step of matrix multiplication, preventing intermediate values from overflowing while preserving the correct modular result. This is crucial for problems requiring F(n) % M for very large 'n'.\n\n"
                "```python\n"
                "def multiply_matrices_mod(A, B, mod):\n"
                "    x = (A[0][0] * B[0][0] + A[0][1] * B[1][0]) % mod\n"
                "    y = (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % mod\n"
                "    z = (A[1][0] * B[0][0] + A[1][1] * B[1][0]) % mod\n"
                "    w = (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % mod\n"
                "    return [[x, y], [z, w]]\n\n"
                "def power_mod(M, n, mod):\n"
                "    result = [[1, 0], [0, 1]] # Identity matrix\n"
                "    M_copy = M\n"
                "    while n > 0:\n"
                "        if n % 2 == 1:\n"
                "            result = multiply_matrices_mod(result, M_copy, mod)\n"
                "        M_copy = multiply_matrices_mod(M_copy, M_copy, mod)\n"
                "        n //= 2\n"
                "    return result\n\n"
                "def fibonacci_matrix_mod(n, mod):\n"
                "    if n <= 1:\n"
                "        return n % mod\n"
                "    matrix = [[1, 1], [1, 0]]\n"
                "    result_matrix = power_mod(matrix, n - 1, mod)\n"
                "    return result_matrix[0][0] % mod\n"
                "```\n\n"
                f"Input: n = {n}, mod = {mod}\n"
                f"Output: {calculated_fib_mod}\n"
                "Time Complexity: O(log n)\n"
                "Space Complexity: O(log n)"
            )

    # 10. Use memoization with a decorator pattern to compute Fibonacci({{n}}) efficiently.
    elif "memoization with a decorator pattern to compute fibonacci" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            n = int(match.group(1))

            # Using Python's built-in functools.lru_cache as a decorator
            from functools import lru_cache

            @lru_cache(maxsize=None) # maxsize=None means unlimited cache size
            def fib_decorator(n_val):
                if n_val <= 1:
                    return n_val
                return fib_decorator(n_val - 1) + fib_decorator(n_val - 2)

            calculated_fib = fib_decorator(n)

            return (
                f"To use memoization with a decorator pattern to compute Fibonacci({n}) efficiently:\n\n"
                "Python's `functools.lru_cache` decorator provides a convenient way to apply memoization to recursive functions. It automatically caches the results of function calls, turning an exponential time complexity into linear time by avoiding redundant computations. This is a clean and Pythonic way to implement top-down dynamic programming.\n\n"
                "```python\n"
                "from functools import lru_cache\n\n"
                "@lru_cache(maxsize=None) # maxsize=None for unlimited cache, or specify a number\n"
                "def fibonacci_decorated(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    return fibonacci_decorated(n - 1) + fibonacci_decorated(n - 2)\n"
                "```\n\n"
                f"Input: {n}\n"
                f"Output: {calculated_fib}\n"
                "Time Complexity: O(n)\n"
                "Space Complexity: O(n)"
            )
            
    # 11. Write a program that computes the nth Fibonacci number where n is up to {{max_n}} using efficient recursion with caching.
    elif "computes the nth fibonacci number where n is up to" in q and "using efficient recursion with caching" in q:
        match = re.search(r"up to (\d+)", q)
        if match:
            max_n = int(match.group(1))
            # Example for demonstration, we will calculate for max_n
            
            _memo_for_caching_test = {}
            def fib_cached_recursive_test(n_val):
                if n_val in _memo_for_caching_test:
                    return _memo_for_caching_test[n_val]
                if n_val <= 1:
                    result = n_val
                else:
                    result = fib_cached_recursive_test(n_val - 1) + fib_cached_recursive_test(n_val - 2)
                _memo_for_caching_test[n_val] = result
                return result

            # Calculate for max_n as example
            fib_val_at_maxn = fib_cached_recursive_test(max_n)

            return (
                f"To compute the {max_n}th Fibonacci number using efficient recursion with caching:\n\n"
                "Efficient recursion with caching (memoization) stores the results of already computed subproblems in a cache (e.g., a dictionary or array). Before making a recursive call, the function checks if the result for the given input is already in the cache. If it is, the cached value is returned directly, avoiding redundant computations and reducing the time complexity from exponential to linear.\n\n"
                "```python\n"
                "fib_cache = {}\n"
                "def fib_efficient_recursion(n):\n"
                "    if n in fib_cache:\n"
                "        return fib_cache[n]\n"
                "    if n <= 1:\n"
                "        result = n\n"
                "    else:\n"
                "        result = fib_efficient_recursion(n - 1) + fib_efficient_recursion(n - 2)\n"
                "    fib_cache[n] = result\n"
                "    return result\n"
                "```\n\n"
                f"Input: n = {max_n}\n"
                f"Output: Fibonacci({max_n}) = {fib_val_at_maxn}\n"
                "Time Complexity: O(n)\n"
                "Space Complexity: O(n) for the cache and recursion stack"
            )

    # 12. Calculate Fibonacci({{n}}) with memoization and analyze its time and space complexity.
    elif "calculate fibonacci" in q and "with memoization and analyze its time and space complexity" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            n = int(match.group(1))
            _memo_for_analysis = {}
            def fib_memo_analysis(n_val):
                if n_val in _memo_for_analysis:
                    return _memo_for_analysis[n_val]
                if n_val <= 1:
                    result = n_val
                else:
                    result = fib_memo_analysis(n_val - 1) + fib_memo_analysis(n_val - 2)
                _memo_for_analysis[n_val] = result
                return result

            fib_result = fib_memo_analysis(n)

            return (
                f"To calculate Fibonacci({n}) with memoization and analyze its time and space complexity:\n\n"
                "Memoization (top-down dynamic programming) stores the results of each subproblem as it's computed. When a subproblem is encountered again, its result is retrieved from the cache instead of being recomputed. This significantly reduces the total number of computations.\n\n"
                "```python\n"
                "memo = {}\n"
                "def fibonacci_memo_analysis(n):\n"
                "    if n in memo:\n"
                "        return memo[n]\n"
                "    if n <= 1:\n"
                "        result = n\n"
                "    else:\n"
                "        result = fibonacci_memo_analysis(n - 1) + fibonacci_memo_analysis(n - 2)\n"
                "    memo[n] = result\n"
                "    return result\n"
                "```\n\n"
                f"Input: {n}\n"
                f"Output: Fibonacci({n}) = {fib_result}\n\n"
                "**Time Complexity Analysis:**\n"
                "* **Without Memoization (Naive Recursion):** O(2^n). This is because many subproblems are recomputed multiple times, leading to an exponential call tree.\n"
                "* **With Memoization:** O(n). Each Fibonacci number from F(0) to F(n) is computed only once. Each computation takes constant time (sum and lookup). Therefore, the total time is linear with respect to 'n'.\n\n"
                "**Space Complexity Analysis:**\n"
                "* **With Memoization:** O(n). This space is primarily used for the memoization table (to store results for F(0) up to F(n)) and for the recursion stack (which can go up to 'n' calls deep in the worst case before hitting base cases)."
            )

    # 13. Optimize Fibonacci calculation for repeated queries using precomputation up to n = {{max_n}}.
    elif "optimize fibonacci calculation for repeated queries using precomputation up to n" in q:
        match = re.search(r"up to n = (\d+)", q)
        if match:
            max_n = int(match.group(1))
            
            # Perform precomputation
            precomputed_fib_values = [0] * (max_n + 1)
            if max_n >= 0:
                precomputed_fib_values[0] = 0
            if max_n >= 1:
                precomputed_fib_values[1] = 1
            for i in range(2, max_n + 1):
                precomputed_fib_values[i] = precomputed_fib_values[i-1] + precomputed_fib_values[i-2]

            return (
                f"To optimize Fibonacci calculation for repeated queries using precomputation up to n = {max_n}:\n\n"
                "Precomputation involves calculating and storing a range of Fibonacci numbers (e.g., from F(0) up to F({max_n})) beforehand, typically in an array or list. For subsequent queries within this precomputed range, the Fibonacci number can be retrieved in O(1) (constant) time by simply looking up the precomputed value. This approach is highly efficient for scenarios where many queries for different Fibonacci numbers are expected within a known maximum limit.\n\n"
                "```python\n"
                "fib_precomputed_array = []\n"
                "def precompute_fibonacci(max_n):\n"
                "    global fib_precomputed_array # For demonstration\n"
                "    fib_precomputed_array = [0] * (max_n + 1)\n"
                "    if max_n >= 0:\n"
                "        fib_precomputed_array[0] = 0\n"
                "    if max_n >= 1:\n"
                "        fib_precomputed_array[1] = 1\n"
                "    for i in range(2, max_n + 1):\n"
                "        fib_precomputed_array[i] = fib_precomputed_array[i-1] + fib_precomputed_array[i-2]\n\n"
                "def get_precomputed_fib(n):\n"
                "    if n < 0 or n >= len(fib_precomputed_array):\n"
                "        return 'Error: n out of precomputed range'\n"
                "    return fib_precomputed_array[n]\n\n"
                "# Example usage:\n"
                "# precompute_fibonacci({max_n})\n"
                "# value = get_precomputed_fib(k)\n"
                "```\n\n"
                f"Precomputed values up to F({max_n}): {precomputed_fib_values}\n"
                f"Time for precomputation: O(max_n)\n"
                f"Time per query (after precomputation): O(1)\n"
                f"Space Complexity: O(max_n)"
            )

    # 14. Create a function to determine whether a number {{num}} is a Fibonacci number in O(1) time using mathematical properties.
    elif "determine whether a number" in q and "is a fibonacci number in o(1) time using mathematical properties" in q:
        match = re.search(r"number (\d+)", q)
        if match:
            num = int(match.group(1))
            is_fib = is_fibonacci_num_global(num)
            return (
                f"To determine whether a number {num} is a Fibonacci number in O(1) time using mathematical properties:\n\n"
                "A positive integer 'n' is a Fibonacci number if and only if either `(5*n^2 + 4)` or `(5*n^2 - 4)` is a perfect square. This property allows checking membership in the Fibonacci sequence in O(1) time, assuming `is_perfect_square` is efficient (which involves a square root operation, typically constant time for standard integer sizes).\n\n"
                "```python\n"
                "import math\n\n"
                "def is_perfect_square(k):\n"
                "    if k < 0: return False\n"
                "    s = int(math.sqrt(k))\n"
                "    return s * s == k\n\n"
                "def is_fibonacci_number_o1(num):\n"
                "    return is_perfect_square(5 * num * num + 4) or \\\n"
                "           is_perfect_square(5 * num * num - 4)\n"
                "```\n\n"
                f"Input: {num}\n"
                f"Output: {is_fib}\n"
                "Time Complexity: O(1) (due to constant number of arithmetic ops and square root)\n"
                "Space Complexity: O(1)"
            )

    # 15. Generate a Fibonacci sequence up to {{n}} where values are stored using BigInteger types to handle overflow.
    elif "generate a fibonacci sequence up to" in q and "values are stored using biginteger types to handle overflow" in q:
        match = re.search(r"up to (\d+)", q)
        if match:
            n = int(match.group(1))
            
            # Python integers automatically handle arbitrary precision, so no explicit BigInteger type is needed.
            # We'll just demonstrate regular Python integer arithmetic.
            fib_sequence_bigint = []
            a, b = 0, 1
            for i in range(n + 1):
                fib_sequence_bigint.append(a)
                a, b = b, a + b
            
            # Trim the sequence to requested length if it exceeds n+1 (for demonstration up to F_n)
            fib_sequence_bigint = fib_sequence_bigint[:n+1]

            return (
                f"To generate a Fibonacci sequence up to {n} where values are stored using BigInteger types to handle overflow:\n\n"
                "In Python, integers inherently support arbitrary precision, meaning they automatically handle numbers of any size, effectively behaving as 'BigInteger' types. This simplifies handling large Fibonacci numbers, as explicit BigInteger libraries are not required. The standard iterative approach will work correctly for very large 'n' until memory limits are hit.\n\n"
                "```python\n"
                "def generate_fibonacci_bigint(n):\n"
                "    if n < 0: return []\n"
                "    fib_seq = []\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n + 1): # Generate up to F(n)\n"
                "        fib_seq.append(a)\n"
                "        a, b = b, a + b\n"
                "    return fib_seq\n"
                "```\n\n"
                f"Input: {n}\n"
                f"Output (first {n+1} Fibonacci numbers):\n{fib_sequence_bigint}\n"
                f"Fibonacci({n}) (full value): {fib_sequence_bigint[n]}\n"
                "Time Complexity: O(n * log(F_n)) (due to arithmetic operations on growing numbers, where log(F_n) is proportional to the number of digits)\n"
                "Space Complexity: O(n * log(F_n)) (to store numbers with growing digits)"
            )

    # 16. Implement a circular buffer-based solution to store last two Fibonacci numbers for constant space calculation of Fibonacci({{n}}).
    elif "circular buffer-based solution to store last two fibonacci numbers for constant space calculation" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            n = int(match.group(1))
            calculated_fib = fib_two_vars_global(n) # Reusing the O(1) space iterative solution

            return (
                f"To implement a circular buffer-based solution to store last two Fibonacci numbers for constant space calculation of Fibonacci({n}):\n\n"
                "A circular buffer of size 2, or simply two variables, is the canonical way to achieve O(1) space complexity for Fibonacci numbers. It continually updates the two most recent Fibonacci values, discarding older ones. This is the most memory-efficient iterative method.\n\n"
                "```python\n"
                "def fibonacci_circular_buffer(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    # Effectively a circular buffer of size 2\n"
                "    prev = 0 \n"
                "    curr = 1\n"
                "    for _ in range(2, n + 1):\n"
                "        # Next Fibonacci is prev + curr\n"
                "        # Shift: prev becomes curr, curr becomes next_fib\n"
                "        prev, curr = curr, prev + curr\n"
                "    return curr\n"
                "```\n\n"
                f"Input: {n}\n"
                f"Output: {calculated_fib}\n"
                "Time Complexity: O(n)\n"
                "Space Complexity: O(1)"
            )

    # 17. Compute and return the {{n}}th Fibonacci number in binary format using efficient logic.
    elif "return the" in q and "th fibonacci number in binary format using efficient logic" in q:
        match = re.search(r"the (\d+)th Fibonacci number in binary format", q)
        if match:
            n = int(match.group(1))
            
            # Get the Fibonacci number (can be large)
            # Use matrix exponentiation for potentially very large 'n' first
            fib_value = fibonacci_matrix_exponentiation(n)
            
            # Convert to binary
            binary_format = bin(fib_value)
            
            return (
                f"To compute and return the {n}th Fibonacci number in binary format using efficient logic:\n\n"
                "For efficiency, especially for large 'n', first compute the nth Fibonacci number using a fast algorithm like matrix exponentiation (O(log n)) or fast doubling (O(log n)). Once the numerical value is obtained, convert it to its binary string representation. Python's built-in integer type handles arbitrary precision, simplifying the numerical calculation aspect even for very large numbers.\n\n"
                "```python\n"
                "# (Requires fibonacci_matrix_exponentiation or fib_fast_doubling from above)\n"
                "def fibonacci_to_binary(n):\n"
                "    # Step 1: Compute the Fibonacci number efficiently\n"
                "    fib_num = fibonacci_matrix_exponentiation(n) # Or fib_fast_doubling(n)[0]\n"
                "    \n"
                "    # Step 2: Convert to binary format\n"
                "    return bin(fib_num)\n"
                "```\n\n"
                f"Input: {n}\n"
                f"Fibonacci({n}) (decimal): {fib_value}\n"
                f"Output (binary format): {binary_format}\n"
                "Time Complexity: O(log n) (for Fibonacci calculation) + O(log(F_n)) (for binary conversion, where log(F_n) is proportional to number of bits)\n"
                "Space Complexity: O(log(F_n)) (for storing the binary string)"
            )

    # 18. Write a program to compute the number of Fibonacci numbers that are less than or equal to {{x}}.
    elif "compute the number of fibonacci numbers that are less than or equal to" in q:
        match = re.search(r"less than or equal to (\d+)", q)
        if match:
            x = int(match.group(1))
            
            count = 0
            a, b = 0, 1
            while a <= x:
                count += 1
                a, b = b, a + b
            
            return (
                f"To compute the number of Fibonacci numbers that are less than or equal to {x}:\n\n"
                "This iterative program generates Fibonacci numbers sequentially and increments a counter for each number that does not exceed the given limit 'x'. Since Fibonacci numbers grow exponentially, this loop will terminate quickly (in logarithmic time with respect to 'x').\n\n"
                "```python\n"
                "def count_fibonacci_up_to_x(x):\n"
                "    if x < 0: return 0\n"
                "    count = 0\n"
                "    a, b = 0, 1\n"
                "    while a <= x:\n"
                "        count += 1\n"
                "        a, b = b, a + b\n"
                "    return count\n"
                "```\n\n"
                f"Input: {x}\n"
                f"Output: {count}\n"
                "Time Complexity: O(log x) (since Fibonacci numbers grow exponentially)\n"
                "Space Complexity: O(1)"
            )

    # 19. Use memoization and iterative fallback to compute Fibonacci({{n}}) and track subproblem reuse frequency.
    elif "use memoization and iterative fallback to compute fibonacci" in q and "track subproblem reuse frequency" in q:
        match = re.search(r"fibonacci\((\d+)\)", q)
        if match:
            n = int(match.group(1))
            
            _memo_fallback_instance = {}
            _subproblem_reuse_freq = {} # Dictionary to store frequency

            def fib_memo_fallback(n_val):
                if n_val in _memo_fallback_instance:
                    # Track reuse
                    _subproblem_reuse_freq[n_val] = _subproblem_reuse_freq.get(n_val, 0) + 1
                    return _memo_fallback_instance[n_val]
                
                # If n is small enough for direct calculation or base case
                if n_val <= 10: # Example threshold for fallback, adjust as needed
                    result = n_val if n_val <= 1 else (fib_memo_fallback(n_val - 1) + fib_memo_fallback(n_val - 2))
                else: # Fallback to iterative for larger subproblems if not in cache
                    # This "fallback" means if it's not in cache, use iterative for a specific larger part
                    # For a true iterative fallback, it's usually memoized top-down with direct iterative sub-calls
                    # Or a combined approach. For this question, we'll demonstrate recursive memoization and track.
                    # A true "iterative fallback" would mean: if n is large, directly call iterative solution.
                    # Let's adjust to be a standard memoized solution, and track reuse. The "fallback" part might be a misinterpretation here.
                    # Or, it could mean "if recursion depth gets too high, switch to iterative".
                    # Let's stick to simple memoization with tracking.
                    result = fib_memo_fallback(n_val - 1) + fib_memo_fallback(n_val - 2)

                _memo_fallback_instance[n_val] = result
                return result

            # Run the computation
            fib_result = fib_memo_fallback(n)

            reuse_output = "\n".join([f"F({k}): {count} reuses" for k, count in sorted(_subproblem_reuse_freq.items())])

            return (
                f"To compute Fibonacci({n}) using memoization and track subproblem reuse frequency:\n\n"
                "This approach combines the efficiency of memoization with tracking the number of times each pre-computed subproblem is accessed from the cache. The 'iterative fallback' concept could imply switching to an iterative calculation for larger subproblems within the recursive calls, or simply using an iterative solution as the default efficient method after memoization. Here, we demonstrate classic memoization with reuse tracking.\n\n"
                "```python\n"
                "fib_cache = {}\n"
                "reuse_frequency = {} # Tracks how many times a subproblem result is looked up from cache\n\n"
                "def fibonacci_memo_track_reuse(n):\n"
                "    if n in fib_cache:\n"
                "        reuse_frequency[n] = reuse_frequency.get(n, 0) + 1\n"
                "        return fib_cache[n]\n"
                "    \n"
                "    if n <= 1:\n"
                "        result = n\n"
                "    else:\n"
                "        # In a true 'iterative fallback', here you might decide to use an iterative loop\n"
                "        # for fibonacci_memo_track_reuse(n-1) and fibonacci_memo_track_reuse(n-2)\n"
                "        # if n is beyond a certain threshold to limit recursion depth.\n"
                "        result = fibonacci_memo_track_reuse(n - 1) + fibonacci_memo_track_reuse(n - 2)\n"
                "    \n"
                "    fib_cache[n] = result\n"
                "    return result\n"
                "```\n\n"
                f"Input: {n}\n"
                f"Output: Fibonacci({n}) = {fib_result}\n"
                f"Subproblem Reuse Frequencies:\n{reuse_output}\n"
                "Time Complexity: O(n)\n"
                "Space Complexity: O(n)"
            )

    # 20. Build a reusable Fibonacci module that supports multiple modes: recursive, memoized, iterative, and matrix-based.
    elif "build a reusable fibonacci module that supports multiple modes" in q:
        # No specific numerical input needed, just describe the module structure
        return (
            "To build a reusable Fibonacci module that supports multiple modes (recursive, memoized, iterative, and matrix-based):\n\n"
            "A well-designed module encapsulates different algorithmic approaches for the same problem, allowing users to choose the most suitable method based on their requirements (e.g., performance, memory, simplicity). This promotes code reusability and maintainability.\n\n"
            "```python\n"
            "# fibonacci_module.py\n\n"
            "import functools\n\n"
            "class Fibonacci:\n"
            "    def __init__(self):\n"
            "        self._memo = {} # For memoized mode\n"
            "        self._precomputed = [] # For precomputation mode\n\n"
            "    # --- Recursive Mode (Naive) ---\n"
            "    def recursive(self, n):\n"
            "        if n < 0: raise ValueError('Input must be non-negative')\n"
            "        if n <= 1: return n\n"
            "        return self.recursive(n - 1) + self.recursive(n - 2)\n\n"
            "    # --- Memoized Mode (Recursive with Caching / Top-Down DP) ---\n"
            "    @functools.lru_cache(maxsize=None) # Using built-in decorator for simplicity\n"
            "    def memoized(self, n):\n"
            "        if n < 0: raise ValueError('Input must be non-negative')\n"
            "        if n <= 1: return n\n"
            "        return self.memoized(n - 1) + self.memoized(n - 2)\n"
            "    \n"
            "    # Manual memoization (alternative to decorator)\n"
            "    def memoized_manual(self, n):\n"
            "        if n < 0: raise ValueError('Input must be non-negative')\n"
            "        if n in self._memo: return self._memo[n]\n"
            "        if n <= 1: result = n\n"
            "        else: result = self.memoized_manual(n-1) + self.memoized_manual(n-2)\n"
            "        self._memo[n] = result\n"
            "        return result\n"
            "    \n"
            "    def reset_memo(self): # To clear cache for memoized_manual\n"
            "        self._memo = {}\n"
            "        self.memoized.cache_clear() # Clear lru_cache for memoized method\n\n"
            "    # --- Iterative Mode (O(1) Space / Bottom-Up DP) ---\n"
            "    def iterative(self, n):\n"
            "        if n < 0: raise ValueError('Input must be non-negative')\n"
            "        if n <= 1: return n\n"
            "        a, b = 0, 1\n"
            "        for _ in range(2, n + 1):\n"
            "            a, b = b, a + b\n"
            "        return b\n\n"
            "    # --- Matrix Exponentiation Mode (O(log n) Time) ---\n"
            "    def _multiply_matrices(self, A, B):\n"
            "        x = A[0][0] * B[0][0] + A[0][1] * B[1][0]\n"
            "        y = A[0][0] * B[0][1] + A[0][1] * B[1][1]\n"
            "        z = A[1][0] * B[0][0] + A[1][1] * B[1][0]\n"
            "        w = A[1][0] * B[0][1] + A[1][1] * B[1][1]\n"
            "        return [[x, y], [z, w]]\n\n"
            "    def _power_matrix(self, M, n):\n"
            "        result = [[1, 0], [0, 1]] # Identity matrix\n"
            "        M_copy = M\n"
            "        while n > 0:\n"
            "            if n % 2 == 1: result = self._multiply_matrices(result, M_copy)\n"
            "            M_copy = self._multiply_matrices(M_copy, M_copy)\n"
            "            n //= 2\n"
            "        return result\n\n"
            "    def matrix_based(self, n):\n"
            "        if n < 0: raise ValueError('Input must be non-negative')\n"
            "        if n <= 1: return n\n"
            "        matrix = [[1, 1], [1, 0]]\n"
            "        result_matrix = self._power_matrix(matrix, n - 1)\n"
            "        return result_matrix[0][0]\n"
            "```\n\n"
            "**How to use (in another file):**\n"
            "```python\n"
            "# main.py\n"
            "# from fibonacci_module import Fibonacci\n"
            "# fib_calculator = Fibonacci()\n"
            "\n"
            "# print(f'Recursive F(10): {fib_calculator.recursive(10)}')\n"
            "# print(f'Memoized F(100): {fib_calculator.memoized(100)}')\n"
            "# print(f'Iterative F(100): {fib_calculator.iterative(100)}')\n"
            "# print(f'Matrix F(100000): {fib_calculator.matrix_based(100000)}')\n"
            "```\n"
            "This structure provides a clean API for different Fibonacci computation strategies."
        )
    

# --- NEW ADDITION START HERE ---
# --- Application Questions ---
def answer_application_fibonacci(level, question):
    q = question.lower()

    if level == "Level 1":
        # 1. In a population growth model, rabbits reproduce following the Fibonacci sequence. What is the population after {{n}} months?
        if "population growth model" in q and "rabbits reproduce" in q:
            match = re.search(r"after (\d+) months", q)
            if match:
                n = int(match.group(1))
                population = fib_two_vars_global(n)
                return (
                    f"In a population growth model where rabbits reproduce following the Fibonacci sequence, the population after {n} months would be:\n\n"
                    f"The Fibonacci sequence often models rabbit population growth (assuming ideal conditions and starting with one pair of newborn rabbits that become fertile after one month). F(n) represents the number of rabbit pairs after 'n' months.\n"
                    f"Using the standard Fibonacci sequence (0, 1, 1, 2, 3, 5, ... where F(0)=0, F(1)=1):\n"
                    f"After {n} months, the population (in pairs) is: **{population}**."
                )

        # 2. How can the Fibonacci series help in modeling the number of petals in flowers over {{x}} generations?
        elif "petals in flowers" in q and "generations" in q:
            match = re.search(r"over (\d+) generations", q)
            if match:
                x = int(match.group(1))
                first_few_fibs = get_all_fib_less_than_x_global(100) # Get a few to show pattern
                return (
                    f"The Fibonacci series can help model the number of petals in flowers, which often exhibit Fibonacci numbers (e.g., 3, 5, 8, 13 petals). While not directly about 'generations' of petals, it relates to the growth and arrangement of plant structures (phyllotaxis) over growth cycles.\n"
                    "For example, common petal counts are 3, 5, 8, 13, 21, 34, etc., which are all Fibonacci numbers.\n"
                    f"Over {x} generations, one might observe these patterns emerging in petal counts as the plant species evolves or as new flowers bloom based on Fibonacci principles of growth efficiency."
                )

        # 3. You are stacking tiles in a Fibonacci pattern. How many tiles will you need after {{n}} steps?
        elif "stacking tiles" in q and "fibonacci pattern" in q:
            match = re.search(r"after (\d+) steps", q)
            if match:
                n = int(match.group(1))
                tiles_at_n = fib_two_vars_global(n)
                return (
                    f"If you are stacking tiles in a Fibonacci pattern where the number of tiles at each step follows the sequence (0, 1, 1, 2, 3, ...):\n"
                    f"After {n} steps, the number of tiles (representing the nth term or total tiles, depending on problem definition) would be: **{tiles_at_n}**."
                )
        
        # 4. In a board game, players advance based on Fibonacci numbers. What is the total advancement after {{turns}} turns?
        elif "board game" in q and "advance based on fibonacci numbers" in q:
            match = re.search(r"after (\d+) turns", q)
            if match:
                turns = int(match.group(1))
                fib_turns_plus_2 = fib_two_vars_global(turns + 2)
                total_advancement_formula = fib_turns_plus_2 - 1

                return (
                    f"In a board game where players advance based on Fibonacci numbers (e.g., Turn 1: F1 steps, Turn 2: F2 steps, etc.):\n"
                    f"The total advancement after {turns} turns would be the sum of the first {turns} Fibonacci numbers (F1 to F{turns}).\n"
                    f"The sum of Fibonacci numbers from F1 to F{turns} is given by the identity: `F_{{turns+2}} - 1`.\n"
                    f"Therefore, the total advancement is: **{total_advancement_formula}**."
                )

        # 5. In nature photography, the Fibonacci spiral is used for composition. How is Fibonacci relevant in arranging {{n}} objects?
        elif "nature photography" in q and "fibonacci spiral" in q:
            match = re.search(r"arranging (\d+) objects", q)
            if match:
                n = int(match.group(1))
                return (
                    f"In nature photography, the Fibonacci spiral (derived from golden ratio and Fibonacci sequence) is used as a compositional guideline for aesthetic balance.\n"
                    f"When arranging {n} objects, applying Fibonacci principles suggests placing key elements at points corresponding to the spiral's intersections or along its curves. This creates a visually appealing and harmonious layout, often found naturally in seashells, hurricanes, and galaxies."
                )

        # 6. If the cost of each item follows a Fibonacci sequence, what is the total cost of {{count}} items?
        elif "cost of each item follows a fibonacci sequence" in q:
            match = re.search(r"total cost of (\d+) items", q)
            if match:
                count = int(match.group(1))
                total_cost = sum_fib_seq_global(count) # This sums F0 to F_{count-1}
                return (
                    f"If the cost of each item follows the Fibonacci sequence (e.g., item 1 costs F0, item 2 costs F1, ..., item {count} costs F_{{count-1}}):\n"
                    f"The total cost of {count} items would be the sum of the first {count} Fibonacci numbers (from F0 up to F_{{count-1}}).\n"
                    f"This sum is: **{total_cost}**."
                )

        # 7. A storybook shows patterns in tiles using Fibonacci numbers. What is the pattern after {{steps}} steps?
        elif "storybook shows patterns in tiles" in q and "after" in q and "steps" in q:
            match = re.search(r"after (\d+) steps", q)
            if match:
                steps = int(match.group(1))
                pattern_values = []
                a, b = 0, 1
                for i in range(steps + 1):
                    pattern_values.append(a)
                    a, b = b, a + b
                return (
                    f"In a storybook showing patterns in tiles using Fibonacci numbers, the pattern after {steps} steps might refer to the sequence of tile counts at each step.\n"
                    f"The pattern would be the Fibonacci sequence up to the {steps}th term (assuming F0 is step 0, F1 is step 1, etc.):\n"
                    f"Pattern after {steps} steps: **{pattern_values}**."
                )

        # 8. In a math puzzle app, you get Fibonacci points for every level. What will your score be after {{levels}} levels?
        elif "math puzzle app" in q and "fibonacci points for every level" in q:
            match = re.search(r"after (\d+) levels", q)
            if match:
                levels = int(match.group(1))
                total_score = fib_two_vars_global(levels + 2) - 1 # Sum of F1 to F_levels
                return (
                    f"In a math puzzle app where you get Fibonacci points for every level (e.g., Level 1: F1 points, Level 2: F2 points, etc.):\n"
                    f"Your total score after {levels} levels would be the sum of Fibonacci numbers from F1 to F{levels}.\n"
                    f"Using the identity `Sum(F_i for i=1 to N) = F_{{N+2}} - 1`:\n"
                    f"Your total score after {levels} levels is: **{total_score}**."
                )

        # 9. A child's toy robot moves in Fibonacci steps. How far will it go after {{n}} moves?
        elif "toy robot moves in fibonacci steps" in q and "after" in q and "moves" in q:
            match = re.search(r"after (\d+) moves", q)
            if match:
                n = int(match.group(1))
                total_distance = fib_two_vars_global(n + 2) - 1 # Sum F1 to F_n
                return (
                    f"If a child's toy robot moves in Fibonacci steps (e.g., 1st move: F1 steps, 2nd move: F2 steps, etc.):\n"
                    f"The total distance it will go after {n} moves is the sum of Fibonacci numbers from F1 to F{n}.\n"
                    f"Total distance: **{total_distance}**."
                )
        
        # 10. How can Fibonacci be used to schedule a basic fitness plan for {{days}} days with gradually increasing exercise time?
        elif "fitness plan" in q and "gradually increasing exercise time" in q:
            match = re.search(r"for (\d+) days", q)
            if match:
                days = int(match.group(1))
                fib_series = [fib_two_vars_global(i) for i in range(days)] # F0 to F_{days-1}
                return (
                    f"Fibonacci can be used to schedule a fitness plan by incrementing exercise time in a Fibonacci pattern. This provides a gradual, non-linear increase that avoids sudden jumps in intensity.\n"
                    f"For a {days}-day plan, exercise time could be scheduled as:\n"
                    f"Day 1: F1 units of time (1 unit)\n"
                    f"Day 2: F2 units of time (1 unit)\n"
                    f"Day 3: F3 units of time (2 units)\n"
                    f"Day 4: F4 units of time (3 units)\n"
                    f"...up to Day {days}: F{days} units of time.\n"
                    f"Example for {days} days (units of time): **{[fib_two_vars_global(i) for i in range(1, days + 1)]}**."
                )

        # 11. A coding game gives you Fibonacci bonus points. What are your total bonus points after {{rounds}} rounds?
        elif "coding game" in q and "fibonacci bonus points" in q:
            match = re.search(r"after (\d+) rounds", q)
            if match:
                rounds = int(match.group(1))
                total_bonus_points = fib_two_vars_global(rounds + 2) - 1 # Sum F1 to F_rounds
                return (
                    f"In a coding game awarding Fibonacci bonus points (e.g., Round 1: F1 points, Round 2: F2 points, etc.):\n"
                    f"Your total bonus points after {rounds} rounds would be the sum of Fibonacci numbers from F1 to F{rounds}.\n"
                    f"Total bonus points: **{total_bonus_points}**."
                )
        
        # 12. Your piggy bank savings follow a Fibonacci pattern. How much do you save after {{weeks}} weeks?
        elif "piggy bank savings follow a fibonacci pattern" in q:
            match = re.search(r"after (\d+) weeks", q)
            if match:
                weeks = int(match.group(1))
                total_savings = fib_two_vars_global(weeks + 2) - 1 # Sum F1 to F_weeks
                return (
                    f"If your piggy bank savings follow a Fibonacci pattern (e.g., Week 1: F1 saved, Week 2: F2 saved, etc.):\n"
                    f"The total amount you save after {weeks} weeks would be the sum of Fibonacci numbers from F1 to F{weeks}.\n"
                    f"Total savings: **{total_savings}**."
                )

        # 13. In a design contest, each layer of decoration follows the Fibonacci rule. How many decorations are used in {{layers}} layers?
        elif "each layer of decoration follows the fibonacci rule" in q:
            match = re.search(r"in (\d+) layers", q)
            if match:
                layers = int(match.group(1))
                total_decorations = fib_two_vars_global(layers + 2) - 1 # Sum F1 to F_layers
                return (
                    f"In a design contest where each layer of decoration follows the Fibonacci rule (e.g., Layer 1: F1 decorations, Layer 2: F2 decorations, etc.):\n"
                    f"The total number of decorations used in {layers} layers would be the sum of Fibonacci numbers from F1 to F{layers}.\n"
                    f"Total decorations: **{total_decorations}**."
                )

        # 14. A staircase puzzle grows with Fibonacci steps. How many steps are needed to build it after {{n}} levels?
        elif "staircase puzzle grows with fibonacci steps" in q and "after" in q and "levels" in q:
            match = re.search(r"after (\d+) levels", q)
            if match:
                n = int(match.group(1))
                total_steps = fib_two_vars_global(n + 2) - 1 # Sum F1 to F_n
                return (
                    f"If a staircase puzzle grows with Fibonacci steps (e.g., Level 1 needs F1 steps, Level 2 needs F2 steps, etc.):\n"
                    f"The total number of steps needed to build it after {n} levels is the sum of Fibonacci numbers from F1 to F{n}.\n"
                    f"Total steps: **{total_steps}**."
                )

        # 15. A pattern of colored lights in Fibonacci order is being set up. What will be the position of the {{n}}th light?
        elif "pattern of colored lights in fibonacci order" in q and "position of the" in q:
            match = re.search(r"position of the (\d+)th light", q)
            if match:
                n = int(match.group(1))
                position_value = fib_two_vars_global(n)
                return (
                    f"If a pattern of colored lights is set up in Fibonacci order, the position of the {n}th light could correspond to the {n}th Fibonacci number.\n"
                    f"Position of the {n}th light: **{position_value}**."
                )

    elif level == "Level 2":
        # 1. A computer animation algorithm uses Fibonacci numbers to determine frame sequences. What frame pattern is generated after {{frames}} frames?
        if "computer animation algorithm uses fibonacci numbers" in q and "frame pattern" in q:
            match = re.search(r"after (\d+) frames", q)
            if match:
                frames = int(match.group(1))
                frame_pattern = [fib_two_vars_global(i) for i in range(frames)]
                return (
                    f"A computer animation algorithm using Fibonacci numbers for frame sequences might generate frames where the number of elements, or a specific property, follows the sequence.\n"
                    f"After {frames} frames, the generated pattern could be the first {frames} Fibonacci numbers (F0 to F{{frames-1}}):\n"
                    f"Frame pattern: **{frame_pattern}**."
                )

        # 2. A business uses Fibonacci-based scaling to project growth. What is the expected value after {{n}} intervals?
        elif "business uses fibonacci-based scaling to project growth" in q:
            match = re.search(r"after (\d+) intervals", q)
            if match:
                n = int(match.group(1))
                expected_value = fib_two_vars_global(n)
                return (
                    f"When a business uses Fibonacci-based scaling to project growth, it implies that the growth factor or cumulative value at each interval follows the Fibonacci sequence. This models exponential, yet natural, growth.\n"
                    f"After {n} intervals, the expected value (often related to F_n) would be: **{expected_value}**."
                )

        # 3. In project planning, Fibonacci numbers are used to estimate task sizes. How would you assign story points for {{tasks}} tasks based on Fibonacci values?
        elif "project planning" in q and "estimate task sizes" in q:
            match = re.search(r"for (\d+) tasks", q)
            if match:
                tasks = int(match.group(1))
                story_points = [fib_two_vars_global(i) for i in range(1, tasks + 1)] # F1 to F_tasks
                return (
                    f"In project planning, Fibonacci numbers (1, 2, 3, 5, 8, 13, ...) are commonly used for agile story point estimation because they reflect the increasing uncertainty and complexity of larger tasks.\n"
                    f"For {tasks} tasks, you would assign story points from the Fibonacci sequence, typically starting from F1:\n"
                    f"Assigned Story Points: **{story_points}** (corresponding to tasks of increasing complexity)."
                )

        # 4. A museum display features Fibonacci-based lighting intervals. How should the lights be scheduled over {{n}} seconds?
        elif "museum display features fibonacci-based lighting intervals" in q:
            match = re.search(r"over (\d+) seconds", q)
            if match:
                n = int(match.group(1))
                fib_intervals = []
                current_sum = 0
                i = 0
                while current_sum + fib_two_vars_global(i) <= n:
                    if fib_two_vars_global(i) > 0:
                        fib_intervals.append(fib_two_vars_global(i))
                    current_sum += fib_two_vars_global(i)
                    i += 1
                return (
                    f"For a museum display with Fibonacci-based lighting intervals over {n} seconds, the lighting could be scheduled such that each interval's duration is a Fibonacci number.\n"
                    f"This creates a rhythm that feels natural and aesthetically pleasing, aligning with the Golden Ratio often found in art and nature.\n"
                    f"Example schedule (intervals in seconds): **{fib_intervals}** (total sum: {current_sum} seconds)."
                )

        # 5. In a budgeting app, monthly savings follow Fibonacci increments. Calculate total savings over {{months}} months.
        elif "budgeting app" in q and "monthly savings follow fibonacci increments" in q:
            match = re.search(r"over (\d+) months", q)
            if match:
                months = int(match.group(1))
                total_savings = fib_two_vars_global(months + 2) - 1
                return (
                    f"In a budgeting app where monthly savings follow Fibonacci increments (e.g., Month 1: F1, Month 2: F2, etc.):\n"
                    f"The total savings over {months} months would be the sum of Fibonacci numbers from F1 to F{months}.\n"
                    f"Total savings: **{total_savings}**."
                )

        # 6. A gardener plants trees in Fibonacci order each season. How many trees are planted by the {{season}}th season?
        elif "gardener plants trees in fibonacci order each season" in q:
            match = re.search(r"by the (\d+)th season", q)
            if match:
                season = int(match.group(1))
                total_trees = fib_two_vars_global(season + 2) - 1
                return (
                    f"If a gardener plants trees in Fibonacci order each season (e.g., Season 1: F1 trees, Season 2: F2 trees, etc.):\n"
                    f"The total number of trees planted by the {season}th season would be the sum of Fibonacci numbers from F1 to F{season}.\n"
                    f"Total trees: **{total_trees}**."
                )

        # 7. A robot follows a Fibonacci stepping rule. If it starts at 0, how far has it gone after {{steps}} steps?
        elif "robot follows a fibonacci stepping rule" in q and "how far has it gone after" in q:
            match = re.search(r"after (\d+) steps", q)
            if match:
                steps = int(match.group(1))
                total_distance = fib_two_vars_global(steps + 2) - 1
                return (
                    f"If a robot follows a Fibonacci stepping rule (e.g., 1st step is F1 distance, 2nd step is F2 distance, etc.):\n"
                    f"The total distance it has gone after {steps} steps is the sum of Fibonacci numbers from F1 to F{steps}.\n"
                    f"Total distance: **{total_distance}**."
                )
        
        # 8. In digital art, Fibonacci numbers decide brush stroke intervals. What stroke spacing would you expect after {{iterations}} iterations?
        elif "digital art" in q and "fibonacci numbers decide brush stroke intervals" in q:
            match = re.search(r"after (\d+) iterations", q)
            if match:
                iterations = int(match.group(1))
                stroke_spacings = [fib_two_vars_global(i) for i in range(1, iterations + 1)]
                return (
                    f"In digital art, when Fibonacci numbers decide brush stroke intervals, it often means the spacing or size of strokes follow the sequence, creating visually harmonious compositions.\n"
                    f"After {iterations} iterations, you would expect stroke spacings (starting from F1) to be: **{stroke_spacings}**."
                )

        # 9. A model simulates virus spread where new infections follow Fibonacci growth. What is the infected count after {{days}} days?
        elif "model simulates virus spread" in q and "new infections follow fibonacci growth" in q:
            match = re.search(r"after (\d+) days", q)
            if match:
                days = int(match.group(1))
                infected_count = fib_two_vars_global(days)
                return (
                    f"In a model simulating virus spread where new infections follow Fibonacci growth (e.g., total infected count is F_n for day n, or new cases are F_n):\n"
                    f"The infected count after {days} days (as per simple Fibonacci growth model) would be: **{infected_count}**."
                )

        # 10. Fibonacci numbers are used to design a music beat pattern. What will be the tempo variation after {{bars}} bars?
        elif "music beat pattern" in q and "tempo variation" in q:
            match = re.search(r"after (\d+) bars", q)
            if match:
                bars = int(match.group(1))
                tempo_variations = [fib_two_vars_global(i) for i in range(1, bars + 1)]
                return (
                    f"When Fibonacci numbers are used to design a music beat pattern, it often relates to rhythmic complexity, phrase lengths, or tempo changes, providing a sense of natural progression.\n"
                    f"After {bars} bars, the tempo variation (e.g., beats per minute, or relative tempo unit) for each bar might follow the Fibonacci sequence (F1 to F{bars}):\n"
                    f"Tempo variation pattern: **{tempo_variations}**."
                )

        # 11. In a smart lighting system, Fibonacci numbers are used to adjust brightness dynamically. Describe the brightness pattern over {{n}} cycles.
        elif "smart lighting system" in q and "adjust brightness dynamically" in q:
            match = re.search(r"over (\d+) cycles", q)
            if match:
                n = int(match.group(1))
                brightness_pattern = [fib_two_vars_global(i) for i in range(1, n + 1)]
                return (
                    f"In a smart lighting system, adjusting brightness dynamically using Fibonacci numbers creates a non-linear, often subtle and natural, progression of light intensity.\n"
                    f"Over {n} cycles, the brightness pattern (e.g., intensity units) could follow the Fibonacci sequence (F1 to F{n}):\n"
                    f"Brightness pattern: **{brightness_pattern}**."
                )

        # 12. A sequence of Fibonacci numbers is used to determine seating in a stadium for distancing. What are the positions for the first {{n}} seats?
        elif "seating in a stadium for distancing" in q and "positions for the first" in q:
            match = re.search(r"first (\d+) seats", q)
            if match:
                n = int(match.group(1))
                seat_positions = [fib_two_vars_global(i) for i in range(1, n + 1)] # F1 to F_n
                return (
                    f"When Fibonacci numbers are used to determine seating positions for distancing, it implies a non-uniform spacing that grows with the Fibonacci sequence. This could optimize capacity while maintaining relative distancing.\n"
                    f"The positions for the first {n} seats (e.g., distance from a reference point, or indices in a larger sequence) could be: **{seat_positions}**."
                )

        # 13. A game awards Fibonacci coins after each round. How many coins does a player earn after {{rounds}} rounds?
        elif "game awards fibonacci coins after each round" in q:
            match = re.search(r"after (\d+) rounds", q)
            if match:
                rounds = int(match.group(1))
                total_coins = fib_two_vars_global(rounds + 2) - 1
                return (
                    f"In a game that awards Fibonacci coins after each round (e.g., Round 1: F1 coins, Round 2: F2 coins, etc.):\n"
                    f"A player earns a total of **{total_coins}** coins after {rounds} rounds (sum of F1 to F{rounds})."
                )
        
        # 14. In a supply chain model, stock replenishment follows a Fibonacci strategy. How many units should be ordered after {{week}} weeks?
        elif "supply chain model" in q and "stock replenishment follows a fibonacci strategy" in q:
            match = re.search(r"after (\d+) weeks", q)
            if match:
                week = int(match.group(1))
                units_ordered = fib_two_vars_global(week)
                return (
                    f"In a supply chain model, if stock replenishment follows a Fibonacci strategy, it suggests that the quantity ordered at each interval (e.g., week) corresponds to a Fibonacci number.\n"
                    f"After {week} weeks, the number of units that should be ordered (for that specific week) would be: **{units_ordered}**."
                )

        # 15. A mobile app gamifies habits using Fibonacci scoring. How does a user’s progress evolve after {{days}} of streaks?
        elif "mobile app gamifies habits using fibonacci scoring" in q:
            match = re.search(r"after (\d+) of streaks", q)
            if match:
                days = int(match.group(1))
                progress_points_per_day = [fib_two_vars_global(i) for i in range(1, days + 1)]
                total_progress_points = fib_two_vars_global(days + 2) - 1
                return (
                    f"In a mobile app that gamifies habits using Fibonacci scoring, a user's progress might evolve by accumulating points or increasing streak bonuses based on the Fibonacci sequence.\n"
                    f"For {days} days of streaks, points earned per day (starting from F1) could be: **{progress_points_per_day}**.\n"
                    f"Total accumulated progress points: **{total_progress_points}**."
                )

        # 16. A drone flies increasing distances using Fibonacci numbers for each battery charge cycle. What distance is covered after {{charges}} charges?
        elif "drone flies increasing distances using fibonacci numbers" in q:
            match = re.search(r"after (\d+) charges", q)
            if match:
                charges = int(match.group(1))
                total_distance = fib_two_vars_global(charges + 2) - 1
                return (
                    f"If a drone flies increasing distances using Fibonacci numbers for each battery charge cycle (e.g., 1st charge: F1 distance, 2nd charge: F2 distance, etc.):\n"
                    f"The total distance covered after {charges} charges would be the sum of Fibonacci numbers from F1 to F{charges}.\n"
                    f"Total distance: **{total_distance}**."
                )

        # 17. A digital counter shows Fibonacci intervals between events. What is the value displayed on the {{event}}th event?
        elif "digital counter shows fibonacci intervals between events" in q:
            match = re.search(r"on the (\d+)th event", q)
            if match:
                event = int(match.group(1))
                displayed_value = fib_two_vars_global(event)
                return (
                    f"If a digital counter shows Fibonacci intervals between events, the value displayed on the {event}th event could correspond to the {event}th Fibonacci number.\n"
                    f"Value displayed: **{displayed_value}**."
                )
        
        # 18. In a memory training app, items are repeated in Fibonacci intervals. When is the {{nth}} repetition scheduled?
        elif "memory training app" in q and "items are repeated in fibonacci intervals" in q:
            match = re.search(r"when is the (\d+)th repetition scheduled", q)
            if match:
                nth = int(match.group(1))
                interval_for_repetition = fib_two_vars_global(nth)
                return (
                    f"In a memory training app where items are repeated in Fibonacci intervals, this implies a spaced repetition system where review intervals grow non-linearly.\n"
                    f"The {nth} repetition could be scheduled after an interval corresponding to the {nth} Fibonacci number (e.g., F{nth} time units after the previous repetition):\n"
                    f"Scheduled interval for {nth}th repetition: **{interval_for_repetition}**."
                )

        # 19. An ecosystem simulation uses Fibonacci numbers to model prey population. What is the population size at generation {{n}}?
        elif "ecosystem simulation uses fibonacci numbers to model prey population" in q:
            match = re.search(r"at generation (\d+)", q)
            if match:
                n = int(match.group(1))
                population_size = fib_two_vars_global(n)
                return (
                    f"In an ecosystem simulation using Fibonacci numbers to model prey population, the population size at generation {n} directly corresponds to the {n}th Fibonacci number, representing growth under ideal conditions similar to the classic rabbit problem.\n"
                    f"Population size at generation {n}: **{population_size}**."
                )

        # 20. A robot builds layers in Fibonacci thickness. What is the total thickness after {{layers}} layers?
        elif "robot builds layers in fibonacci thickness" in q:
            match = re.search(r"after (\d+) layers", q)
            if match:
                layers = int(match.group(1))
                total_thickness = fib_two_vars_global(layers + 2) - 1
                return (
                    f"If a robot builds layers with Fibonacci thickness (e.g., Layer 1: F1 thickness, Layer 2: F2 thickness, etc.):\n"
                    f"The total thickness after {layers} layers would be the sum of Fibonacci numbers from F1 to F{layers}.\n"
                    f"Total thickness: **{total_thickness}**."
                )

    elif level == "Level 3":
        # 1. In a financial model, a company invests following a Fibonacci pattern for {{years}} years. How can you predict the ROI pattern and its long-term sustainability?
        if "financial model" in q and "invests following a fibonacci pattern" in q:
            match = re.search(r"for (\d+) years", q)
            if match:
                years = int(match.group(1))
                fib_values_up_to_years = [fib_two_vars_global(i) for i in range(years + 1)]
                return (
                    f"In a financial model where a company invests following a Fibonacci pattern for {years} years, the investment amounts or their returns would grow according to the Fibonacci sequence.\n"
                    f"**ROI Pattern Prediction:** The Return on Investment (ROI) pattern would exhibit a similar, accelerating growth curve, as Fibonacci numbers approximate exponential growth (close to $\\phi^n / \\sqrt{5}$ for large n). Each year's ROI would be disproportionately higher than the previous, reflecting larger investment increments.\n"
                    f"**Long-term Sustainability:** Pure Fibonacci growth in investment (or ROI) is **not sustainable indefinitely** without external capital or market expansion. It demands an ever-increasing base, which eventually outstrips available resources or market capacity. While appealing in early stages, it would require increasingly massive capital infusions and market returns. Real-world financial systems need to consider compounding interest, market saturation, and resource limits."
                )

        # 2. You are designing a memory-efficient data structure where access time grows in a Fibonacci sequence. How would this affect lookup and insertion operations over {{n}} elements?
        elif "memory-efficient data structure where access time grows in a fibonacci sequence" in q:
            match = re.search(r"over (\d+) elements", q)
            if match:
                n = int(match.group(1))
                fib_access_time_at_n = fib_two_vars_global(n)
                return (
                    f"If a memory-efficient data structure has access time that grows in a Fibonacci sequence (i.e., access time for 'k' elements is proportional to F_k), this would have significant implications for lookup and insertion operations, especially as the number of elements {n} increases.\n"
                    f"**Impact on Lookup/Insertion:** Since Fibonacci numbers grow exponentially (F_n is roughly $\\phi^n / \\sqrt{5}$), the access time would quickly become exponential (O(F_n)). This is highly inefficient compared to standard data structures that offer logarithmic (e.g., balanced trees, hash tables) or even linear (e.g., arrays, linked lists) access times.\n"
                    f"For {n} elements, the access time would be proportional to F({n}) = **{fib_access_time_at_n}** units. This would make the data structure impractical for any but very small datasets.\n"
                    f"**Optimization:** To make it practical, the 'Fibonacci growth' would likely refer to a *property* of the structure (like memory consumption or number of nodes at a certain depth) rather than direct access time. Access time should ideally be optimized to O(log n) or O(n) for scalability."
                )

        # 3. In a music generation tool, chord progressions are structured using Fibonacci intervals. How can you generate a melody for {{bars}} bars using this pattern?
        elif "music generation tool, chord progressions are structured using fibonacci intervals" in q:
            match = re.search(r"for (\d+) bars", q)
            if match:
                bars = int(match.group(1))
                fib_sequence = [fib_two_vars_global(i) for i in range(1, bars + 1)]
                return (
                    f"In a music generation tool, structuring chord progressions using Fibonacci intervals means leveraging the inherent mathematical harmony of the sequence (e.g., 3-5-8 progression in beats, or using intervals like major/minor thirds, perfect fifths related to ratios). This often results in compositions perceived as natural and pleasing.\n"
                    f"To generate a melody for {bars} bars using this pattern:\n"
                    f"* **Rhythmic Structure:** Assign duration or number of notes per phrase based on Fibonacci numbers. For example, Bar 1 has F1 beats, Bar 2 has F2 beats, or a phrase is F3 beats long, followed by F5 beats.\n"
                    f"* **Melodic Intervals:** Use Fibonacci numbers to determine semitone intervals between notes (e.g., 1 semitone (minor 2nd), 2 (major 2nd), 3 (minor 3rd), 5 (perfect 4th), 8 (minor 6th)). This can create a non-diatonic but mathematically ordered melody.\n"
                    f"* **Phrase Grouping:** Group bars or phrases into lengths corresponding to Fibonacci numbers, creating a hierarchy that sounds organic.\n"
                    f"Example pattern for {bars} bars (e.g., length in beats or number of notes): **{fib_sequence}**."
                )

        # 4. A simulation tracks resource consumption using Fibonacci-based intervals. How would this affect scalability and resource planning for {{cycles}} operation cycles?
        elif "simulation tracks resource consumption using fibonacci-based intervals" in q:
            match = re.search(r"for (\d+) operation cycles", q)
            if match:
                cycles = int(match.group(1))
                fib_consumption_at_n = fib_two_vars_global(cycles)
                return (
                    f"If a simulation tracks resource consumption using Fibonacci-based intervals (meaning resource consumption at each cycle 'n' is proportional to F_n, or increases by F_n), this implies an accelerating, near-exponential rate of consumption.\n"
                    f"**Scalability Impact:** This model suggests **poor scalability** in the long term. For {cycles} operation cycles, the consumption would be proportional to F({cycles}) = **{fib_consumption_at_n}**. As 'cycles' increase, resource demands will quickly outstrip available supply, leading to bottlenecks, failures, or excessively high costs.\n"
                    f"**Resource Planning:** Resource planning would need to account for this exponential growth. It requires forecasting extremely high demands for future cycles and implementing aggressive scaling strategies (e.g., cloud auto-scaling, dynamic resource provisioning). However, this model is typically unsustainable beyond a small number of cycles without external intervention or a change in the consumption pattern."
                )

        # 5. You are building a recursive AI planner where decisions unfold in Fibonacci time steps. What is the total time complexity for {{steps}} steps and how do you optimize it?
        elif "recursive ai planner where decisions unfold in fibonacci time steps" in q:
            match = re.search(r"for (\d+) steps", q)
            if match:
                steps = int(match.group(1))
                return (
                    f"If a recursive AI planner's decisions unfold in Fibonacci time steps (e.g., processing for step 'k' takes F_k time, or the number of paths explored grows like Fibonacci with depth 'k'), it implies an exponential growth in computational cost with each step.\n"
                    f"**Total Time Complexity for {steps} steps:** This would likely lead to an **exponential time complexity** similar to the naive recursive Fibonacci algorithm (O(2^n) or worse, depending on branching). Each decision could branch into sub-decisions, forming a tree where the number of nodes at depth 'k' or the cost of processing at depth 'k' is related to F_k.\n"
                    f"**Optimization:**\n"
                    f"* **Memoization/Dynamic Programming:** Crucial for efficiency. Store the results of sub-decisions or sub-problems to avoid recomputing them. This reduces complexity to polynomial (e.g., O(n) or O(n^2) depending on state).\n"
                    f"* **Iterative Approaches:** Convert recursive logic to iterative loops to reduce stack overhead.\n"
                    f"* **Pruning/Heuristics:** In AI planning, techniques like alpha-beta pruning, A* search with heuristics, or limited-depth search can drastically cut down the search space and time without exploring all Fibonacci branches.\n"
                    f"* **Problem Formulation:** Re-evaluate if a Fibonacci time step model truly captures the problem or if a more efficient polynomial-time model can be used."
                )

        # 6. In an art installation, LED pulses follow a Fibonacci pattern in color intensity. How would you synchronize the lighting pattern for {{zones}} zones?
        elif "art installation, led pulses follow a fibonacci pattern in color intensity" in q:
            match = re.search(r"for (\d+) zones", q)
            if match:
                zones = int(match.group(1))
                return (
                    f"In an art installation where LED pulses follow a Fibonacci pattern in color intensity (e.g., intensity levels are F1, F2, F3, F5, etc.), this creates a dynamic visual experience with a natural, escalating rhythm.\n"
                    f"**To synchronize the lighting pattern for {zones} zones:**\n"
                    f"* **Central Controller:** Use a master controller (e.g., a microcontroller like Arduino/Raspberry Pi or a DMX controller) to generate the Fibonacci intensity values.\n"
                    f"* **Distributed Control (Advanced):** Each zone could have a micro-controller receiving a synchronization signal (e.g., a clock pulse) and then independently calculating its current Fibonacci intensity based on a global cycle counter.\n"
                    f"* **Mapping:** Map Fibonacci numbers to a perceptible range of LED brightness (e.g., 0-255 for RGB). You might use F_k for a relative intensity and then scale it. For instance, F1=1, F2=1, F3=2, F4=3, F5=5, F6=8, F7=13... if max intensity is 255, divide F_k by largest F_n in pattern and multiply by 255.\n"
                    f"* **Timing/Phase Shift:** Synchronize either by having all zones use the same Fibonacci number at the same time, or introduce phase shifts (e.g., Zone 1 is at F_k, Zone 2 is at F_{{k+1}}, etc.) for a ripple effect."
                )

        # 7. A cryptographic algorithm embeds Fibonacci sequences in key generation. How secure is this approach against brute-force over {{n}} key spaces?
        elif "cryptographic algorithm embeds fibonacci sequences in key generation" in q:
            match = re.search(r"over (\d+) key spaces", q)
            if match:
                n = int(match.group(1))
                return (
                    f"If a cryptographic algorithm embeds Fibonacci sequences in key generation, its security against brute-force depends entirely on *how* the sequence is used.\n"
                    f"**Potential Weaknesses:**\n"
                    f"* **Predictability:** The standard Fibonacci sequence is fully deterministic and easily computable. If the key is directly F_n or a simple combination of F_n, it's weak. An attacker can quickly generate Fibonacci numbers and test them.\n"
                    f"* **Pisano Periods:** For keys derived from F_n modulo M, the sequence repeats (Pisano period). An attacker can exploit this cycle if the modulus is small.\n"
                    f"**Possible Strengths (if used correctly):**\n"
                    f"* **Large Indices:** If 'n' is very large (e.g., 10^18), computing F_n (even with O(log n) methods) can be computationally intensive, but not truly 'random' enough for modern crypto.\n"
                    f"* **PRNG Seeding:** Using Fibonacci numbers as seeds or internal states for a cryptographically secure pseudo-random number generator (CSPRNG) *could* be secure, provided the CSPRNG itself is robust.\n"
                    f"* **Non-linear Transformations:** If the Fibonacci sequence is used in conjunction with non-linear functions, chaotic maps, or combined with other random elements in a complex way, it might contribute to security.\n"
                    f"**Conclusion for {n} key spaces:** Without additional strong cryptographic primitives, directly embedding standard Fibonacci sequences into key generation is **generally considered insecure** against brute-force attacks. The predictability of the sequence makes it vulnerable once the pattern is identified. Modern cryptography relies on properties like computational hardness of factoring large numbers or discrete logarithms, not predictable sequences."
                )

        # 8. You're analyzing a Fibonacci-based load balancing strategy in distributed systems. How does it perform under varying loads up to {{tasks}} tasks?
        elif "fibonacci-based load balancing strategy in distributed systems" in q:
            match = re.search(r"up to (\d+) tasks", q)
            if match:
                tasks = int(match.group(1))
                return (
                    f"A Fibonacci-based load balancing strategy in distributed systems might assign tasks to servers (or resources) in a way that aligns with Fibonacci numbers, perhaps to manage ratios of load, or to prioritize certain servers.\n"
                    f"**Performance under Varying Loads (up to {tasks} tasks):**\n"
                    f"* **Predictability:** The strategy would be highly predictable. If a server is assigned F_k tasks, the next might get F_{{k+1}}, etc. This predictability can be both a strength (easy to understand and implement) and a weakness (easily exploitable by malicious actors).\n"
                    f"* **Uneven Distribution:** Fibonacci numbers grow exponentially. This implies that load could become **highly unbalanced** very quickly. Some servers might be assigned F_k tasks while others are assigned F_{{k+1}} or F_{{k+2}} tasks, leading to significant load disparities. For {tasks} tasks, the imbalance might be severe, with a few servers heavily loaded and many underutilized.\n"
                    f"* **Scalability Challenges:** While simple, this strategy would likely not scale well under high and dynamic loads. It wouldn't adapt to changing server availability or fluctuating task sizes efficiently. Dynamic load balancing algorithms usually aim for more uniform distribution or rely on real-time feedback."
                )
        
        # 9. A genetic algorithm uses Fibonacci steps for mutation frequency. How does this influence evolution speed across {{generations}} generations?
        elif "genetic algorithm uses fibonacci steps for mutation frequency" in q:
            match = re.search(r"across (\d+) generations", q)
            if match:
                generations = int(match.group(1))
                mutation_frequencies_example = [fib_two_vars_global(i) for i in range(1, min(generations + 1, 10))]
                
                return (
                    f"If a genetic algorithm uses Fibonacci steps for mutation frequency, it means the rate at which genetic variations are introduced into the population changes according to the Fibonacci sequence (e.g., increases/decreases by F_k in each generation or over periods).\n"
                    f"**Influence on Evolution Speed across {generations} Generations:**\n"
                    f"* **Accelerated Search (Early):** If mutation frequency *increases* with Fibonacci numbers (e.g., F1, F2, F3,...), the algorithm would quickly introduce a large number of variations. This could lead to a very fast initial exploration of the solution space and potentially rapid convergence to a local optimum.\n"
                    f"* **Loss of Convergence/Diversity (Late):** As mutation frequency continues to grow exponentially, it would likely become too high. This can lead to 'genetic drift' where beneficial traits are lost, prevent convergence to an optimal solution, or make the population too chaotic.\n"
                    f"* **Controlled Exploration (Inverse/Ratio):** If mutation frequency *decreases* or is controlled by *ratios* derived from Fibonacci, it could allow for wider exploration initially and then finer tuning. For example, using `1/F_k` for frequency, or applying the Golden Ratio for adaptive mutation.\n"
                    f"In most genetic algorithms, mutation rates are typically small and often adaptively controlled, rather than exponentially increasing, to balance exploration and exploitation."
                )

        # 10. You're creating an e-learning module where quiz difficulty increases in Fibonacci order. How would you distribute {{n}} questions across levels?
        elif "e-learning module where quiz difficulty increases in fibonacci order" in q:
            match = re.search(r"distribute (\d+) questions across levels", q)
            if match:
                n = int(match.group(1))
                
                distribution_plan = []
                current_questions_distributed = 0
                level_num = 1
                while current_questions_distributed < n:
                    fib_q_count = fib_two_vars_global(level_num)
                    if fib_q_count == 0:
                        level_num += 1
                        continue
                    
                    if current_questions_distributed + fib_q_count <= n:
                        distribution_plan.append(f"Level {level_num}: {fib_q_count} questions")
                        current_questions_distributed += fib_q_count
                    else:
                        remaining_questions = n - current_questions_distributed
                        if remaining_questions > 0:
                            distribution_plan.append(f"Level {level_num}: {remaining_questions} questions (remaining)")
                        current_questions_distributed = n
                    level_num += 1

                return (
                    f"In an e-learning module where quiz difficulty increases in Fibonacci order, this could mean that the number of questions per level, or the 'difficulty score' assigned to each level, follows the Fibonacci sequence. This creates a natural, escalating challenge for learners.\n"
                    f"To distribute {n} questions across levels based on increasing difficulty:\n"
                    f"You could assign questions per level as F1, F2, F3, F5, F8, ... (1, 1, 2, 3, 5, 8, ... questions per level).\n"
                    f"Example distribution plan for {n} questions:\n"
                    f"**{'; '.join(distribution_plan)}**."
                )

        # 11. In a simulation game, resources are unlocked using Fibonacci logic. Design a strategy to optimize unlocking {{levels}} levels with minimal waste.
        elif "simulation game, resources are unlocked using fibonacci logic" in q:
            match = re.search(r"unlocking (\d+) levels", q)
            if match:
                levels = int(match.group(1))
                return (
                    f"In a simulation game, if resources are unlocked using Fibonacci logic (e.g., cost to unlock a level, or number of resources unlocked per level, follows Fibonacci), the strategy should anticipate the escalating costs/rewards.\n"
                    f"**Strategy to optimize unlocking {levels} levels with minimal waste:**\n"
                    f"* **Prioritize Value:** If costs increase exponentially (F_k), prioritize levels that provide the highest return on investment (ROI) relative to their Fibonacci cost. Focus on critical path unlocks.\n"
                    f"* **Resource Accumulation:** Since costs grow quickly, implement mechanics for players to accumulate resources at an accelerating rate or through multiple income streams to match the Fibonacci expenditure.\n"
                    f"* **Tiered Unlocks:** Design the game such that players gain access to more efficient resource generation or new gameplay loops at certain Fibonacci milestones (e.g., Level F5, F8, F13) to 'jump-start' progression and avoid grind.\n"
                    f"* **Alternative Paths:** Offer alternative, non-Fibonacci paths for unlocking, such as achievements, daily bonuses, or microtransactions, to allow players to bypass exceptionally high Fibonacci-gated costs.\n"
                    f"* **Dynamic Balancing:** The game designers might need to dynamically adjust the Fibonacci 'multiplier' or the base values to keep the game engaging and prevent players from hitting insurmountable walls of progression, thus minimizing 'waste' (player frustration, churn)."
                )

        # 12. A delivery service adjusts routes based on Fibonacci time windows. How do you plan deliveries for {{packages}} packages under this rule?
        elif "delivery service adjusts routes based on fibonacci time windows" in q:
            match = re.search(r"for (\d+) packages", q)
            if match:
                packages = int(match.group(1))
                return (
                    f"If a delivery service adjusts routes based on Fibonacci time windows (e.g., delivery time slots are F1, F2, F3 hours after order placement, or task duration follows Fibonacci intervals), this creates a non-uniform scheduling challenge.\n"
                    f"**Planning Deliveries for {packages} packages under this rule:**\n"
                    f"* **Categorize Packages:** Group packages by their requested Fibonacci time window (e.g., 1-hour window, 1-hour window, 2-hour window, 3-hour window, 5-hour window, etc.).\n"
                    f"* **Dynamic Routing Optimization:** Use a dynamic routing algorithm that can accommodate variable time windows. Traditional Traveling Salesperson Problem (TSP) variants with Time Windows (TSP-TW) can be adapted. The 'Fibonacci' nature means the windows aren't arbitrary but follow a predictable mathematical pattern.\n"
                    f"* **Resource Allocation:** Allocate delivery vehicles and drivers dynamically. Longer Fibonacci windows can allow a single driver to cover more packages, while shorter ones demand dedicated resources.\n"
                    f"* **Customer Communication:** Clearly communicate the evolving time windows to customers, as the Fibonacci growth means later deliveries might have significantly wider (and less precise) time estimates.\n"
                    f"* **Strategic Bundling:** Prioritize bundling packages with similar time windows or geographically close destinations, especially for larger Fibonacci windows, to maximize efficiency."
                )

        # 13. In an IoT network, energy-saving intervals follow the Fibonacci sequence. How would this affect synchronization across {{devices}} devices?
        elif "iot network, energy-saving intervals follow the fibonacci sequence" in q:
            match = re.search(r"across (\d+) devices", q)
            if match:
                devices = int(match.group(1))
                return (
                    f"If an IoT network's energy-saving intervals follow the Fibonacci sequence (e.g., devices enter low-power mode for F1, F2, F3, F5... time units, or report data at F1, F2, F3... intervals), this creates a variable, non-linear duty cycle.\n"
                    f"**Effect on Synchronization Across {devices} Devices:**\n"
                    f"* **Loss of Coherence:** Unless carefully managed, using individual Fibonacci intervals would quickly lead to devices desynchronizing. Each device's 'sleep' or 'report' cycle would grow independently, making it difficult to predict when a device is active and available for communication.\n"
                    f"* **Increased Latency:** Communication with a specific device might incur high latency, as you'd have to wait for its next active Fibonacci interval.\n"
                    f"* **Complexity in Data Aggregation:** Aggregating data from multiple sensors would become complex. A central hub would need a sophisticated schedule to 'wake up' or listen for data from devices based on their individual Fibonacci intervals.\n"
                    f"**Synchronization Strategy:**\n"
                    f"To maintain synchronization, devices would need a shared, synchronized clock and a master schedule. For example, all {devices} devices could use a single global Fibonacci counter for their sleep cycles, ensuring they all 'wake up' at the same F_k time or enter a shared low-power state. Alternatively, devices could communicate their individual Fibonacci state to a central coordinator. Otherwise, this pattern leads to rapid desynchronization."
                )

        # 14. A healthcare prediction model uses Fibonacci increments to track patient recovery stages. How effective is this for {{conditions}} chronic conditions?
        elif "healthcare prediction model uses fibonacci increments to track patient recovery stages" in q:
            match = re.search(r"for (\d+) chronic conditions", q)
            if match:
                conditions = int(match.group(1))
                return (
                    f"If a healthcare prediction model uses Fibonacci increments to track patient recovery stages (e.g., milestones are F1, F2, F3, F5 days/weeks from start; or improvement metrics are assessed at F_k increments), it implies an accelerating recovery process.\n"
                    f"**Effectiveness for {conditions} chronic conditions:**\n"
                    f"* **Limited Applicability:** This model assumes a consistent, predictable, and accelerating recovery pattern. Many chronic conditions involve fluctuating symptoms, plateaus, or non-linear improvements, which may not align with strict Fibonacci increments.\n"
                    f"* **Over-optimistic for Slow Recovery:** For conditions with very slow or unpredictable recovery, a Fibonacci model might set unrealistic expectations for rapid improvement, potentially causing patient frustration or misguiding treatment plans.\n"
                    f"* **Useful for Specific Accelerating Phases:** It could be effective for specific phases of recovery that are known to accelerate (e.g., initial wound healing or early physical therapy where progress can be rapid). However, it's less likely to fit the entire trajectory of a chronic condition.\n"
                    f"* **Data-Driven Adjustment:** For this to be effective, it would need to be highly data-driven, potentially using machine learning to adapt the Fibonacci 'base' or 'scaling factor' to each specific condition or even individual patient, rather than applying a rigid sequence."
                )

# --- Optimization Functions ---
def answer_optimization_fibonacci(level, question):
    q = question.lower()

    if level == "Level 1":
        # 1. Why is recursion without memoization inefficient for computing Fibonacci numbers?
        if "recursion without memoization inefficient" in q:
            return (
                "Recursion without memoization is inefficient for computing Fibonacci numbers because it leads to **redundant calculations**.\n"
                "When `fib(n)` is called, it recursively calls `fib(n-1)` and `fib(n-2)`. Both of these branches will, in turn, compute `fib(n-3)`, `fib(n-4)`, and so on, multiple times. This leads to an exponential number of function calls, specifically O(2^n) time complexity."
            )

        # 2. How does memoization improve the efficiency of calculating the {nth} Fibonacci number?
        elif "memoization improve the efficiency" in q:
            match = re.search(r"the (\d+)th Fibonacci number", q)
            n_str = match.group(1) if match else "{nth}"
            return (
                f"Memoization improves the efficiency of calculating the {n_str}th Fibonacci number by **storing the results of already computed subproblems** in a cache (e.g., a dictionary or array).\n"
                "When the function is called with inputs for which it has already computed a result, it retrieves the stored value instead of re-calculating it. This transforms the time complexity from exponential (O(2^n)) to linear (O(n)) by ensuring each Fibonacci number is computed only once."
            )

        # 3. What is the time complexity of the naive recursive Fibonacci algorithm?
        elif "time complexity of the naive recursive fibonacci algorithm" in q:
            return (
                "The time complexity of the naive (unoptimized) recursive Fibonacci algorithm is **O(2^n)**."
                "This exponential complexity arises because of the repeated computation of the same subproblems without storing their results."
            )

        # 4. How can using a loop instead of recursion improve Fibonacci calculation performance?
        elif "using a loop instead of recursion improve fibonacci calculation performance" in q:
            return (
                "Using a loop (iterative approach) instead of recursion significantly improves Fibonacci calculation performance by:\n"
                "* **Eliminating redundant calculations:** Iterative methods build the solution from the base cases upwards (bottom-up dynamic programming), computing each Fibonacci number only once.\n"
                "* **Avoiding recursion overhead:** Loops do not incur the overhead of function call stack management, which can be substantial for deep recursive calls.\n"
                "This leads to a time complexity of **O(n)**, which is much faster than the O(2^n) of naive recursion."
            )

        # 5. What is the space complexity of a recursive Fibonacci implementation without memoization?
        elif "space complexity of a recursive fibonacci implementation without memoization" in q:
            return (
                "The space complexity of a recursive Fibonacci implementation without memoization is **O(n)**."
                "This space is consumed by the call stack, as each recursive call adds a new stack frame until the base case is reached. For `fib(n)`, the recursion depth can go up to `n`."
            )

        # 6. How does dynamic programming help in reducing redundant calculations in Fibonacci series?
        elif "dynamic programming help in reducing redundant calculations in fibonacci series" in q:
            return (
                "Dynamic programming (DP) helps in reducing redundant calculations in the Fibonacci series by either **memoization (top-down)** or **tabulation (bottom-up)**.\n"
                "* **Memoization:** Stores the results of subproblems in a cache as they are computed. If a subproblem is encountered again, its result is looked up instead of recomputed.\n"
                "* **Tabulation:** Builds a table (array) of results from the base cases upwards, ensuring that when a larger subproblem needs a smaller one, that smaller one has already been computed and stored.\n"
                "Both methods ensure that each unique Fibonacci number is computed only once, reducing time complexity from O(2^n) to O(n)."
            )

        # 7. Which approach is faster for large Fibonacci numbers: recursion or iteration?
        elif "which approach is faster for large fibonacci numbers: recursion or iteration" in q:
            return (
                "For large Fibonacci numbers, **iteration (or iterative dynamic programming)** is significantly faster than naive recursion.\n"
                "Iterative solutions have a time complexity of O(n) and avoid the overhead of recursive calls, making them much more efficient as 'n' grows. Naive recursion is O(2^n) and quickly becomes impractical due to redundant calculations."
            )

        # 8. Why does the naive recursive method compute the same Fibonacci values multiple times?
        elif "naive recursive method compute the same fibonacci values multiple times" in q:
            return (
                "The naive recursive method computes the same Fibonacci values multiple times due to **overlapping subproblems**.\n"
                "For example, to compute `fib(5)`, it calls `fib(4)` and `fib(3)`. `fib(4)` in turn calls `fib(3)` and `fib(2)`. Notice `fib(3)` is computed twice. This redundancy grows exponentially as 'n' increases, leading to a massive number of repeated calculations of smaller Fibonacci numbers."
            )

        # 9. What is the benefit of using a bottom-up approach in Fibonacci series computation?
        elif "benefit of using a bottom-up approach in fibonacci series computation" in q:
            return (
                "The primary benefits of using a bottom-up approach (tabulation) in Fibonacci series computation are:\n"
                "* **Efficiency:** It computes each Fibonacci number exactly once, building results from the base cases upwards, leading to O(n) time complexity.\n"
                "* **Avoids recursion overhead:** It uses loops, avoiding stack overflow issues common with deep recursion for large 'n'.\n"
                "* **Clear memory management:** It often allows for O(1) space optimization by only storing the two most recent values, rather than an entire DP table or recursion stack."
            )

        # 10. How can storing computed values help in optimizing the Fibonacci algorithm?
        elif "storing computed values help in optimizing the fibonacci algorithm" in q:
            return (
                "Storing computed values (which is the core idea behind memoization and dynamic programming) optimizes the Fibonacci algorithm by **eliminating redundant calculations**.\n"
                "Instead of recalculating a Fibonacci number every time it's needed, the algorithm can simply look up its value in a storage (like an array or dictionary). This reduces the time complexity from exponential O(2^n) to linear O(n), as each distinct Fibonacci number is computed only once."
            )

        # 11. What causes exponential time growth in recursive Fibonacci implementations?
        elif "causes exponential time growth in recursive fibonacci implementations" in q:
            return (
                "Exponential time growth in recursive Fibonacci implementations is caused by **overlapping subproblems** and the lack of memoization.\n"
                "The computation tree for `fib(n)` repeatedly branches and re-calculates the same smaller Fibonacci numbers multiple times (e.g., `fib(n-2)` is calculated by both `fib(n-1)` and `fib(n)`). This leads to a computational cost that doubles roughly with each increment of 'n', resulting in O(2^n) complexity."
            )

        # 12. How many recursive calls are made to compute Fibonacci(5) using naive recursion?
        elif "how many recursive calls are made to compute fibonacci(5) using naive recursion" in q:
            # F(5) -> F(4) + F(3)
            # F(4) -> F(3) + F(2)
            # F(3) -> F(2) + F(1)
            # F(2) -> F(1) + F(0)
            # Calls: F(5), F(4), F(3)(1), F(2)(1), F(1)(1), F(0)(1), F(3)(2), F(2)(2), F(1)(2), F(0)(2), F(1)(3)
            # Total unique Fib calls + redundant calls:
            # F(5): 1
            # F(4): 1
            # F(3): 2
            # F(2): 3
            # F(1): 5
            # F(0): 3
            # Total calls = 1 + 1 + 2 + 3 + 5 + 3 = 15
            # More systematically: Number of calls for fib(n) = 2*F(n+1) - 1. For n=5: 2*F(6) - 1 = 2*8 - 1 = 15.
            return (
                "To compute Fibonacci(5) using naive recursion, **15 recursive calls** are made.\n"
                "This includes redundant calls for subproblems like `fib(3)` and `fib(2)` multiple times.\n"
                "The formula for the number of calls for `fib(n)` is `2 * F(n+1) - 1`."
            )

        # 13. What change would reduce repeated calls in recursive Fibonacci implementation?
        elif "what change would reduce repeated calls in recursive fibonacci implementation" in q:
            return (
                "To reduce repeated calls in a recursive Fibonacci implementation, you should implement **memoization**.\n"
                "Memoization involves storing the results of function calls in a cache (e.g., a dictionary or array) and returning the cached result directly if the same input is encountered again. This ensures each unique Fibonacci number is computed only once, reducing redundancy."
            )

        # 14. How does the use of an auxiliary array improve Fibonacci performance?
        elif "use of an auxiliary array improve fibonacci performance" in q:
            return (
                "The use of an auxiliary array (often called a DP table or memoization table) improves Fibonacci performance by **storing previously computed Fibonacci numbers**.\n"
                "This prevents redundant calculations: whenever a Fibonacci number is needed, the algorithm first checks if it's already in the array. If so, it's retrieved in O(1) time. Otherwise, it's computed, stored, and then returned. This approach transforms the exponential time complexity of naive recursion to linear time (O(n))."
            )

        # 15. Can the Fibonacci problem be optimized to linear time? If so, how?
        elif "can the fibonacci problem be optimized to linear time" in q:
            return (
                "Yes, the Fibonacci problem can be optimized to linear time (O(n)).\n"
                "This can be achieved using:\n"
                "* **Memoization (Top-down Dynamic Programming):** By storing the results of subproblems in a cache (e.g., a dictionary) and returning cached results for repeated calls.\n"
                "* **Tabulation (Bottom-up Dynamic Programming):** By iteratively building up an array of Fibonacci numbers from F0 and F1 to F(n).\n"
                "Both methods ensure each Fibonacci number is calculated only once, leading to O(n) time complexity."
            )

    elif level == "Level 2":
        # 1. Compare the time and space complexity of top-down (with memoization) vs bottom-up approaches in computing Fibonacci numbers up to {{n}}.
        if "compare the time and space complexity of top-down (with memoization) vs bottom-up approaches" in q:
            match = re.search(r"up to (\d+)", q)
            n_val = match.group(1) if match else "n"
            return (
                f"Comparison of Top-Down (Memoization) vs. Bottom-Up (Tabulation) for Fibonacci numbers up to {n_val}:\n\n"
                "| Feature         | Top-Down (Memoization)           | Bottom-Up (Tabulation)           |\n"
                "|-----------------|----------------------------------|----------------------------------|\n"
                "| **Time Complexity** | O(n) | O(n)                     |\n"
                "| **Space Complexity**| O(n) (for memoization table + recursion stack) | O(n) (for DP table), can be O(1) with optimization |\n"
                "| **Approach** | Recursive, cache results         | Iterative, build table from base |\n"
                "| **Memory Access** | Potentially scattered (dict hash) | Sequential (array access)        |\n"
                "| **Readability** | Often more intuitive/direct       | Sometimes requires more thought  |\n\n"
                "**Conclusion:** Both achieve O(n) time complexity. Bottom-up (tabulation) often has slightly better constant factors and can achieve O(1) space, making it generally more memory-efficient and faster for very large 'n' where stack depth is a concern."
            )

        # 2. What are the trade-offs between recursion with memoization and an iterative loop in terms of memory usage for computing Fibonacci({{n}})?
        elif "trade-offs between recursion with memoization and an iterative loop in terms of memory usage" in q:
            match = re.search(r"fibonacci\((\d+)\)", q)
            n_val = match.group(1) if match else "n"
            return (
                f"Trade-offs in memory usage for Fibonacci({n_val}):\n\n"
                "* **Recursion with Memoization:**\n"
                "    * **Memory Usage:** O(n) space complexity.\n"
                "    * **Reason:** It uses memory for both the memoization table (to store `n` results) and for the recursion call stack. For very large `n`, the recursion stack can lead to a `RecursionError` (stack overflow) in some languages or environments, even with memoization.\n"
                "* **Iterative Loop (Bottom-Up DP):**\n"
                "    * **Memory Usage:** O(n) if using a full array, but can be optimized to **O(1)** space complexity by only storing the two previous Fibonacci numbers.\n"
                "    * **Reason:** It avoids recursion overhead, so no large call stack. The O(1) optimization only needs two variables, making it extremely memory-efficient for calculating a single Fibonacci number.\n\n"
                "**Conclusion:** The iterative loop (especially the O(1) space version) is significantly more memory-efficient and generally preferred for large `n` values to avoid stack depth issues and minimize memory footprint."

            )

        # 3. Why is bottom-up dynamic programming more memory-efficient than top-down with memoization for Fibonacci calculation?
        elif "bottom-up dynamic programming more memory-efficient than top-down with memoization" in q:
            return (
                "Bottom-up dynamic programming (tabulation) is generally more memory-efficient than top-down with memoization for Fibonacci calculation primarily because:\n"
                "* **No Recursion Stack:** Bottom-up uses iterative loops, completely avoiding the overhead of the recursion call stack. Top-down, while memoized, still builds a call stack proportional to 'n'.\n"
                "* **Space Optimization:** Bottom-up can easily be optimized to O(1) space complexity by only keeping track of the two previous Fibonacci numbers needed to compute the next one. In contrast, top-down memoization typically requires storing all `n` computed results in its cache (e.g., a dictionary or array of size 'n').\n"
                "This makes the iterative, O(1) space solution the most memory-efficient for computing a single Fibonacci number."
            )

        # 4. How does using only two variables instead of an array optimize the space complexity in Fibonacci computation?
        elif "using only two variables instead of an array optimize the space complexity" in q:
            return (
                "Using only two variables (e.g., `a` and `b`, representing `F(i-1)` and `F(i)`) instead of a full array optimizes the space complexity in Fibonacci computation to **O(1) (constant space)**.\n"
                "This works because to compute `F(i+1)`, you only need `F(i)` and `F(i-1)`. You don't need `F(i-2)`, `F(i-3)`, etc. By updating these two variables in a loop, you effectively 'slide a window' over the sequence, discarding old values that are no longer needed, thereby achieving constant memory usage regardless of how large 'n' gets."
            )

        # 5. How would replacing a full DP table with a sliding window approach optimize Fibonacci series calculation?
        elif "replacing a full dp table with a sliding window approach optimize fibonacci series calculation" in q:
            return (
                "Replacing a full DP table (O(n) space) with a sliding window approach (using just two variables) optimizes Fibonacci series calculation by reducing the space complexity to **O(1) (constant space)**.\n"
                "Since `F(i)` only depends on `F(i-1)` and `F(i-2)`, we only need to store these two most recent values. As we calculate `F(i)`, we update the variables to hold `F(i-1)` and `F(i)`, effectively 'sliding' the window of required values forward. This drastically cuts down memory usage for large 'n' values without affecting the O(n) time complexity."
            )

        # 6. What is the optimized iterative method to compute Fibonacci numbers with O(1) space and O(n) time complexity?
        elif "optimized iterative method to compute fibonacci numbers with o(1) space and o(n) time complexity" in q:
            return (
                "The optimized iterative method to compute Fibonacci numbers with O(1) space and O(n) time complexity uses **two variables** (a sliding window approach).\n"
                "```python\n"
                "def fib_o1_on(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1 # Initialize F(0) and F(1)\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b # Update a to old b, b to new sum\n"
                "    return b # b will hold F(n)\n"
                "```\n"
                "This method is highly efficient for practical 'n' values, balancing time and space."
            )

        # 7. How does the time complexity of matrix exponentiation compare to DP methods for calculating large Fibonacci numbers like F({{n}})?
        elif "time complexity of matrix exponentiation compare to dp methods" in q:
            match = re.search(r"f\((\d+)\)", q)
            n_str = match.group(1) if match else "n"
            return (
                f"Comparison of time complexity for F({n_str}):\n"
                "* **DP Methods (Memoization/Tabulation):** O(n) time complexity. They compute each Fibonacci number up to F(n) sequentially.\n"
                "* **Matrix Exponentiation:** O(log n) time complexity. This method uses binary exponentiation (repeated squaring) on a 2x2 matrix to find F(n) much faster, especially for very large 'n'.\n"
                "**Conclusion:** For extremely large values of 'n' (e.g., n > 1000), matrix exponentiation is significantly faster than linear time DP methods. For smaller 'n', the overhead of matrix operations might make iterative DP competitive or even slightly faster due to constant factors."
            )

        # 8. In what scenarios is it worth switching from DP to matrix exponentiation for computing Fibonacci numbers?
        elif "scenarios is it worth switching from dp to matrix exponentiation" in q:
            return (
                "It is worth switching from linear-time DP methods (memoization/tabulation) to matrix exponentiation for computing Fibonacci numbers in scenarios where:\n"
                "* **'n' is extremely large:** Specifically when `n` is large enough that O(n) becomes too slow (e.g., `n` up to 10^9, 10^18, or beyond). Matrix exponentiation's O(log n) complexity shines in these cases.\n"
                "* **Modular arithmetic is required:** When you need to compute `F(n) % M` for a very large `n`, matrix exponentiation allows applying the modulo operation at each step of matrix multiplication, preventing integer overflow and maintaining efficiency.\n"
                "* **Performance-critical systems:** In applications where even minor optimizations for very large inputs are crucial, despite matrix exponentiation's higher constant factor for smaller `n`."
            )

        # 9. How does the recursive Fibonacci method behave when memoization is implemented using a dictionary instead of a list?
        elif "recursive fibonacci method behave when memoization is implemented using a dictionary instead of a list" in q:
            return (
                "When memoization is implemented using a **dictionary** instead of a list (or array) for recursive Fibonacci:\n"
                "* **Behavior:** The core behavior of avoiding redundant calculations and achieving O(n) time complexity remains the same.\n"
                "* **Flexibility:** Dictionaries offer more flexibility, as keys (the `n` value) do not need to be contiguous or start from 0. This is useful if Fibonacci numbers for non-sequential or sparse indices are computed.\n"
                "* **Memory Usage:** For a dense sequence of `n` values (F0 to Fn), a list might be slightly more memory-efficient due to contiguous allocation. A dictionary might have a slightly larger memory footprint per entry due to hash table overhead. However, both are O(n) space complexity.\n"
                "* **Lookup Speed:** Both provide average O(1) lookup time for integers, making them efficient choices for memoization."
            )

        # 10. Explain how tail recursion could improve optimization for Fibonacci series. Does it eliminate stack overhead?
        elif "tail recursion could improve optimization for fibonacci series" in q and "eliminate stack overhead" in q:
            return (
                "**How Tail Recursion Works:** Tail recursion is a specific form of recursion where the recursive call is the *very last operation* performed in the function. This means the current function's stack frame is no longer needed after the recursive call is made.\n"
                "**Improvement for Fibonacci:** A tail-recursive Fibonacci function can pass accumulated results as arguments, like `fib(n, a, b)`, where `a` and `b` carry the last two Fibonacci numbers.\n"
                "**Does it eliminate stack overhead?**:\n"
                "* **Yes, if optimized by compiler/interpreter:** In languages that support **Tail Call Optimization (TCO)** (e.g., Scheme, Haskell, some C++ compilers with specific flags), the compiler can transform tail-recursive calls into iterative loops. When TCO is applied, the stack frame is reused instead of new ones being pushed, effectively eliminating stack overhead (O(1) space for the call stack).\n"
                "* **No, in Python and many other languages:** Python's interpreter, by design, does *not* perform TCO. Therefore, a tail-recursive function in Python will still build a call stack proportional to `n`, leading to O(n) space complexity and potential `RecursionError` for large `n`.\n"
                "So, while tail recursion *can* improve optimization in some environments, it does **not** eliminate stack overhead in Python."
            )
        
        # 11. Why might using Python's `@lru_cache` decorator be better than manually coding memoization for Fibonacci?
        elif "python's @lru_cache decorator be better than manually coding memoization" in q:
            return (
                "Using Python's `@functools.lru_cache` decorator for memoization is often better than manually coding it for Fibonacci because:\n"
                "* **Conciseness and Readability:** It provides a clean, single-line way to add caching without cluttering the function's logic.\n"
                "* **Correctness:** It's a built-in, well-tested implementation, reducing the chance of bugs in manual caching logic (e.g., forgetting to check the cache, or correctly updating it).\n"
                "* **LRU Strategy:** By default, it implements a Least Recently Used (LRU) eviction strategy (if `maxsize` is set), which is beneficial for managing cache memory in applications where memory is limited and older entries might become irrelevant.\n"
                "* **Performance:** The C-implemented `lru_cache` is highly optimized for performance.\n"
                "* **Ease of Use:** No need to manage a global dictionary or pass a memoization table through function arguments.\n"
                "It simplifies the code and provides robust, optimized caching."
            )

        # 12. Given limited memory, how would you compute Fibonacci({{n}}) efficiently while avoiding stack overflow and excess allocation?
        elif "limited memory" in q and "avoiding stack overflow and excess allocation" in q:
            match = re.search(r"fibonacci\((\d+)\)", q)
            n_val = match.group(1) if match else "n"
            return (
                f"Given limited memory, to compute Fibonacci({n_val}) efficiently while avoiding stack overflow and excess allocation, the **iterative approach with O(1) space complexity** is the most suitable.\n"
                "This method only requires two variables (`a` and `b`) to store the previous two Fibonacci numbers, consuming a constant amount of memory regardless of the size of 'n'. It uses a loop, so there's no recursion stack to overflow. This is highly efficient both in time (O(n)) and memory (O(1)).\n"
                "```python\n"
                "def fibonacci_limited_memory(n):\n"
                "    if n < 0: return 0\n"
                "    if n <= 1: return n\n"
                "    prev = 0\n"
                "    curr = 1\n"
                "    for _ in range(2, n + 1):\n"
                "        prev, curr = curr, prev + curr\n"
                "    return curr\n"
                "```"
            )

        # 13. If memory is constrained, which Fibonacci approach is most suitable: recursion with memoization, bottom-up DP, or iterative two-variable?
        elif "if memory is constrained" in q and "most suitable" in q:
            return (
                "If memory is constrained, the **iterative two-variable approach** is the most suitable Fibonacci computation method.\n"
                "* **Iterative Two-Variable:** Offers O(1) (constant) space complexity, as it only stores the two previous Fibonacci numbers needed for the next calculation. This is the most memory-efficient.\n"
                "* **Bottom-Up DP (with full array):** Has O(n) space complexity for storing the entire DP table. Better than naive recursion, but worse than two-variable.\n"
                "* **Recursion with Memoization:** Also has O(n) space complexity (for the cache and the recursion stack). For very large 'n', it can also lead to stack overflow, which is a critical concern under memory constraints.\n"
                "Therefore, the iterative two-variable solution is the clear winner for memory-constrained environments."
            )

        # 14. How does reducing the number of stored intermediate Fibonacci values affect both speed and memory usage?
        elif "reducing the number of stored intermediate fibonacci values affect both speed and memory usage" in q:
            return (
                "Reducing the number of stored intermediate Fibonacci values significantly impacts both speed and memory usage:\n"
                "* **Memory Usage:** It directly reduces memory consumption. For instance, moving from a full O(n) DP table to an O(1) two-variable approach drastically cuts down space requirements.\n"
                "* **Speed:**\n"
                "    * **Positive Impact:** For very large 'n', especially in memory-constrained systems, reducing memory usage can actually improve speed by reducing cache misses and avoiding swapping to disk, leading to faster execution. The O(1) iterative method maintains O(n) time complexity while optimizing space.\n"
                "    * **Potential Negative Impact (on specific operations):** If the optimization means *recalculating* values that were previously cached (e.g., if you only store a few values but need to jump around the sequence), it would slow down performance for those specific queries. However, for computing a single `F(n)`, reducing storage from O(n) to O(1) generally has no negative impact on time complexity (it remains O(n)) and can improve constant factors by being more cache-friendly."
            )

        # 15. What optimizations can be applied when computing a range of Fibonacci numbers from F({{start}}) to F({{end}}) efficiently?
        elif "computing a range of fibonacci numbers from f({{start}}) to f({{end}}) efficiently" in q:
            match = re.search(r"f\((\d+)\) to f\((\d+)\)", q)
            start_str = match.group(1) if match else "start"
            end_str = match.group(2) if match else "end"
            return (
                f"When computing a range of Fibonacci numbers from F({start_str}) to F({end_str}) efficiently, you can apply these optimizations:\n"
                "* **Iterative (Bottom-Up) with O(n) Space:** The most straightforward and efficient approach for generating a range. Create an array of size `end + 1` and fill it iteratively from F0 up to F(`end`). This is O(`end`) time and O(`end`) space.\n"
                "    ```python\n"
                "    def get_fib_range(start_idx, end_idx):\n"
                "        if end_idx < 0: return []\n"
                "        dp = [0] * (end_idx + 1)\n"
                "        if end_idx >= 1: dp[1] = 1\n"
                "        for i in range(2, end_idx + 1): dp[i] = dp[i-1] + dp[i-2]\n"
                "        return dp[max(0, start_idx):end_idx+1]\n"
                "    ```\n"
                "* **Iterative (Bottom-Up) with O(1) Space (for individual queries):** If memory is a strict concern and you only need *individual* numbers in the range, you can compute each `F(i)` from `start` to `end` using the O(1) space iterative method. However, this re-computes from F0 each time, so total time would be O((`end` - `start`) * `end`), which is worse for large ranges. This is generally only if you need to calculate a *single* `F(n)` but not the whole range.\n"
                "* **Precomputation:** If this range needs to be computed multiple times or `end` is relatively small, precompute all Fibonacci numbers up to `end` once and then retrieve them in O(1) time per query. This is O(`end`) precomputation time + O(1) per query.\n"
                "* **Matrix Exponentiation (for sparse/very large ranges):** If `start` and `end` are extremely large and the range is sparse, or if only `F(end)` is needed quickly, matrix exponentiation is O(log `end`). You can also use it to compute `F(start)` and then iterate the remaining short range or use an identity to find `F(end)` from `F(start)`."
            )

    elif level == "Level 3":
        # 1. How does matrix exponentiation reduce the time complexity of computing Fibonacci({{n}}) to O(log n), and how can it be implemented efficiently?
        if "matrix exponentiation reduce the time complexity" in q and "o(log n)" in q:
            match = re.search(r"fibonacci\((\d+)\)", q)
            n_str = match.group(1) if match else "n"
            return (
                f"Matrix exponentiation reduces the time complexity of computing Fibonacci({n_str}) to **O(log n)** by leveraging the properties of matrix multiplication and binary exponentiation (also known as exponentiation by squaring).\n"
                "**How it works:**\n"
                "1.  **Matrix Identity:** The core idea is that `[[F(k+1), F(k)], [F(k), F(k-1)]]` can be obtained by raising the base matrix `M = [[1, 1], [1, 0]]` to the power of `k` (i.e., `M^k`). So, `F(n)` can be found from `M^(n-1)`.\n"
                "2.  **Binary Exponentiation:** Instead of multiplying the matrix `n-1` times linearly, binary exponentiation calculates `M^X` in `O(log X)` time. It does this by repeatedly squaring the matrix and conditionally multiplying results based on the binary representation of `X`.\n"
                "    * If `X` is even, `M^X = (M^(X/2))^2`.\n"
                "    * If `X` is odd, `M^X = M * (M^((X-1)/2))^2`.\n"
                "This recursive (or iterative) halving of the exponent leads to logarithmic time complexity for matrix powering, and thus for Fibonacci computation.\n"
                "**Efficient Implementation:**\n"
                "The `multiply_matrices` and `power` functions (as implemented in the global helpers) efficiently handle the matrix operations.\n"
                "```python\n"
                "def multiply_matrices(A, B):\n"
                "    # ... (as defined in global helpers) ...\n"
                "    pass\n"
                "def power(M, n):\n"
                "    # ... (as defined in global helpers) ...\n"
                "    pass\n"
                "def fibonacci_matrix_exponentiation(n):\n"
                "    # ... (as defined in global helpers) ...\n"
                "    pass\n"
                "```"
            )

        # 2. Explain how fast doubling technique optimizes the Fibonacci calculation. What is its time complexity compared to other methods?
        elif "fast doubling technique optimizes the fibonacci calculation" in q:
            return (
                "The **fast doubling technique** (also known as 'doubling identities') optimizes Fibonacci calculation by using specific identities to compute F(2k) and F(2k+1) directly from F(k) and F(k+1). This allows the computation to effectively 'double' the index at each recursive step, similar to binary exponentiation.\n"
                "**Identities Used:**\n"
                "* `F(2k) = F(k) * (2 * F(k+1) - F(k))`\n"
                "* `F(2k+1) = F(k+1)^2 + F(k)^2`\n"
                "**Optimization:** By recursively calling `fib_fast_doubling(n // 2)` and then using these identities, the problem size is halved at each step, leading to logarithmic time complexity.\n"
                "**Time Complexity Comparison:**\n"
                "* **Fast Doubling:** O(log n).\n"
                "* **Naive Recursion:** O(2^n).\n"
                "* **Memoized/Tabulated DP:** O(n).\n"
                "* **Matrix Exponentiation:** O(log n).\n"
                "**Conclusion:** Fast doubling offers the same asymptotic time complexity (O(log n)) as matrix exponentiation but can be faster in practice for extremely large numbers as it involves fewer (though slightly more complex) arithmetic operations than general 2x2 matrix multiplication."
            )

        # 3. For large-scale Fibonacci computations like F({{n}}), how would you choose between matrix exponentiation and fast doubling for performance-critical systems?
        elif "large-scale fibonacci computations" in q and "choose between matrix exponentiation and fast doubling" in q:
            match = re.search(r"f\((\d+)\)", q)
            n_str = match.group(1) if match else "n"
            return (
                f"For large-scale Fibonacci computations like F({n_str}) in performance-critical systems, both matrix exponentiation and fast doubling offer O(log n) time complexity, but the choice often depends on practical considerations:\n"
                "* **Fast Doubling (Identities):**\n"
                "    * **Pros:** Often has smaller constant factors (fewer multiplications in Python's arbitrary-precision integers) making it theoretically and sometimes practically faster for very large 'n'. The identities are derived specifically for Fibonacci, so they're highly optimized for the problem. Can be slightly less memory intensive for the recursion stack.\n"
                "    * **Cons:** The identities can be slightly less intuitive to implement than matrix multiplication, and the recursive nature might still hit recursion limits in some Python environments for very large 'n' without explicit iteration or memoization.\n"
                "* **Matrix Exponentiation:**\n"
                "    * **Pros:** Conceptually elegant and easily extendable to other linear recurrence relations. The matrix multiplication logic is generic and reusable. It's also suitable for modular arithmetic.\n"
                "    * **Cons:** Might have slightly larger constant factors due to the general matrix multiplication operations compared to the specialized Fibonacci identities.\n"
                "**Choice for Performance-Critical Systems:**\n"
                "For pure speed with extremely large 'n' (where `log n` is still a large number), **fast doubling is often marginally preferred** due to its tighter constant factors. However, for most practical applications, both are highly efficient and the difference might be negligible. If the problem also involves modular arithmetic (common in competitive programming), both adapt well. The choice can also come down to developer familiarity and ease of implementation."
            )

        # 4. What are the numerical stability concerns when using Binet’s Formula to compute Fibonacci({{n}}) for large n, and how can you mitigate them?
        elif "numerical stability concerns when using binet’s formula" in q:
            match = re.search(r"fibonacci\((\d+)\)", q)
            n_str = match.group(1) if match else "n"
            return (
                f"Binet's Formula for F({n_str}) is: `F(n) = (phi^n - psi^n) / sqrt(5)`, where `phi = (1 + sqrt(5)) / 2` (golden ratio) and `psi = (1 - sqrt(5)) / 2`.\n"
                "**Numerical Stability Concerns for Large `n`:**\n"
                "1.  **Floating-Point Precision:** The primary concern is the use of floating-point numbers (`sqrt(5)`, `phi`, `psi`). These have limited precision. As `n` grows, `phi^n` becomes very large, and `psi^n` becomes very small (approaching zero). The term `psi^n` might eventually become indistinguishable from zero due to floating-point underflow, leading to `F(n) ≈ phi^n / sqrt(5)`.\n"
                "2.  **Accumulation of Error:** Even if `psi^n` doesn't underflow, the massive scale difference between `phi^n` and `psi^n` means that any tiny relative error in `phi^n` gets amplified to potentially significant absolute errors, especially when subtracting a near-zero `psi^n`. The final result, which is an integer, might end up as `X.9999999999` or `Y.0000000001` instead of `X` or `Y`, making it impossible to correctly round to the nearest integer for very large `n`.\n"
                "**Mitigation Strategies:**\n"
                "* **Avoid for Large `n`:** For exact integer results for large `n`, Binet's formula with standard floating-point types should be avoided entirely. Iterative, matrix exponentiation, or fast doubling methods are preferred as they use integer arithmetic.\n"
                "* **High-Precision Floating-Point Libraries:** If Binet's formula *must* be used, employ libraries that support arbitrary-precision floating-point arithmetic (e.g., Python's `Decimal` module or `mpmath` library). This allows `sqrt(5)`, `phi`, `psi`, and `phi^n` to be calculated with enough precision to ensure the small `psi^n` term can still correctly influence the result, and the final value can be accurately rounded to the nearest integer.\n"
                "* **Rounding:** For moderate `n`, if `F(n)` is known to be an integer, `round(phi**n / math.sqrt(5))` might work, but this becomes unreliable for very large `n`."
            )

        # 5. You need to compute Fibonacci numbers modulo {{mod}}. How does modular arithmetic impact the optimization of large Fibonacci sequences?
        elif "fibonacci numbers modulo" in q:
            match = re.search(r"modulo (\d+)", q)
            mod = int(match.group(1)) if match else "mod"
            return (
                f"When computing Fibonacci numbers modulo {mod}, modular arithmetic profoundly impacts optimization, especially for large sequences:\n"
                "1.  **Prevents Overflow:** This is the most crucial impact. Fibonacci numbers grow exponentially and quickly exceed standard integer limits. By applying the modulo operation at each step of arithmetic, all intermediate results remain within the range `[0, mod-1]`, preventing integer overflow even for extremely large `n`.\n"
                "2.  **Enables O(log n) Algorithms:** Modular arithmetic is seamlessly integrated into matrix exponentiation and fast doubling methods. Each matrix multiplication or arithmetic operation (addition, subtraction, multiplication) within these algorithms can be performed modulo `mod`. This allows computing `F(n) % mod` in **O(log n)** time complexity, which is essential for very large `n` that would be impossible otherwise.\n"
                "3.  **Pisano Period:** For any given modulus `M`, the Fibonacci sequence modulo `M` is periodic (repeats itself). This period is called the Pisano period, denoted `pi(M)`. If `n` is extremely large, one can compute `F(n) % M = F(n % pi(M)) % M`. Finding `pi(M)` can be complex, but for repeated queries with the same `M`, it can turn an O(log n) problem into effectively O(1) (after an initial computation of `pi(M)` and then `F(n % pi(M))`).\n"
                "**Overall:** Modular arithmetic makes the computation of `F(n)` for astronomical `n` values feasible, as it bounds the size of numbers and allows the use of very fast logarithmic algorithms."
            )

        # 6. How can memoization be adapted for concurrent Fibonacci computation in a multi-threaded environment?
        elif "memoization be adapted for concurrent fibonacci computation in a multi-threaded environment" in q:
            return (
                "Adapting memoization for concurrent Fibonacci computation in a multi-threaded environment requires careful handling to ensure thread safety and data consistency:\n"
                "1.  **Thread-Safe Cache:** The memoization cache (e.g., a dictionary) must be protected from race conditions. Multiple threads trying to read from or write to the same cache entry simultaneously can lead to incorrect results or corrupt data.\n"
                "    * **Locking:** Use a `threading.Lock` or `multiprocessing.Lock` (for multiprocessing) to ensure that only one thread can access the cache at a time. This is a common approach:\n"
                "        ```python\n"
                "        import threading\n"
                "        memo = {}\n"
                "        memo_lock = threading.Lock()\n"
                "        def fib_concurrent(n):\n"
                "            with memo_lock:\n"
                "                if n in memo: return memo[n]\n"
                "            if n <= 1: result = n\n"
                "            else: result = fib_concurrent(n-1) + fib_concurrent(n-2)\n"
                "            with memo_lock:\n"
                "                memo[n] = result\n"
                "            return result\n"
                "        ```\n"
                "2.  **Concurrent Data Structures:** Use thread-safe data structures for the cache directly (if available in the language/library, e.g., `ConcurrentHashMap` in Java, or specialized concurrent dictionaries). Python's `dict` is *not* inherently thread-safe for concurrent writes by default.\n"
                "3.  **Process-based Parallelism:** For true CPU-bound parallelism in Python (due to GIL), `multiprocessing` might be preferred over `threading`. Each process would have its own memory space, simplifying memoization within that process. However, sharing a single, large memoization table across processes would then require shared memory (e.g., `multiprocessing.Manager.dict()`) which still needs synchronization.\n"
                "4.  **Granularity of Locking:** The `lru_cache` decorator in Python is generally thread-safe for basic operations, but the underlying `dict` access might be protected by Python's Global Interpreter Lock (GIL). However, fine-grained locking around critical cache access points is better for clarity and cross-language applicability."
            )

        # 7. For an embedded system with {{memory_limit}} KB memory, how would you implement an optimized Fibonacci calculator?
        elif "embedded system with" in q and "kb memory" in q and "optimized fibonacci calculator" in q:
            match = re.search(r"with (\d+) kb memory", q)
            memory_limit_kb = int(match.group(1)) if match else "X"
            return (
                f"For an embedded system with a limited memory budget of {memory_limit_kb} KB, implementing an optimized Fibonacci calculator requires prioritizing space efficiency heavily:\n"
                "1.  **Iterative O(1) Space Method:** This is the most suitable approach. It only requires two variables to store the previous two Fibonacci numbers (`prev` and `curr`), making its memory footprint constant and extremely small (a few bytes for integers, regardless of 'n'). This avoids array allocations and recursion stack overhead.\n"
                "    ```python\n"
                "    def fib_embedded_o1(n):\n"
                "        if n < 0: return 0\n"
                "        if n <= 1: return n\n"
                "        a, b = 0, 1\n"
                "        for _ in range(2, n + 1):\n"
                "            a, b = b, a + b\n"
                "        return b\n"
                "    ```\n"
                "2.  **Avoid Recursion:** Recursive solutions, even with memoization, would build a call stack that could quickly exceed `memory_limit_kb` for even moderately large `n`.\n"
                "3.  **Avoid DP Tables:** Full DP tables (O(n) space) would also consume too much memory as `n` grows, as each entry would store a Fibonacci number.\n"
                "4.  **Integer Overflow Handling:** If the target `n` can produce Fibonacci numbers larger than native integer types, ensure the language's integer handling (like Python's arbitrary precision) or manual big integer arithmetic is used carefully, as these can increase per-number memory usage. For very large `n`, modular arithmetic might be the only option if memory is extremely tight and only the last digits are needed.\n"
                "**Conclusion:** The O(1) iterative solution is the only practical choice for such memory-constrained environments."
            )

        # 8. Describe a caching mechanism to store and retrieve previously computed Fibonacci values in a long-running server application.
        elif "caching mechanism to store and retrieve previously computed fibonacci values in a long-running server application" in q:
            return (
                "In a long-running server application, an effective caching mechanism for Fibonacci values would need to balance lookup speed, memory usage, and possibly cache eviction strategies:\n"
                "1.  **In-memory Cache (Dictionary/LRU Cache):**\n"
                "    * **Mechanism:** Use a Python `dict` or `functools.lru_cache` (for thread safety and eviction) to store `n: F(n)` pairs.\n"
                "    * **Pros:** Very fast O(1) average lookup time. Simple to implement.\n"
                "    * **Cons:** Cache size can grow indefinitely, consuming server RAM if `maxsize` is not set or if `n` values are unbounded. Requires thread-safe access in a multi-threaded server.\n"
                "    * **Eviction Strategy:** `lru_cache` automatically implements LRU (Least Recently Used) eviction, which is good for keeping frequently accessed items in cache while discarding older, less used ones when `maxsize` is reached.\n"
                "2.  **Precomputation Cache:** If the range of `n` values is known and not excessively large (e.g., up to 10^5 or 10^6), precompute all values up front into a list/array during application startup. Subsequent lookups are O(1) array access. This is ideal if memory allows and all values in the range are likely to be queried.\n"
                "3.  **External Cache (Redis/Memcached):**\n"
                "    * **Mechanism:** For very large-scale applications or distributed microservices, use an external key-value store like Redis or Memcached.\n"
                "    * **Pros:** Provides centralized caching, persistence (in some configs), scalability, and offloads memory pressure from individual application servers. Supports distributed memoization.\n"
                "    * **Cons:** Adds network latency for each lookup. Requires setting up and managing an external service.\n"
                "**Choice:** For most single-server Python applications, `lru_cache` is highly recommended. For distributed systems, an external cache is often necessary."
            )

        # 9. How can tail-recursive optimization be leveraged in languages that support it to reduce stack usage in Fibonacci computation?
        elif "tail-recursive optimization be leveraged in languages that support it" in q and "reduce stack usage" in q:
            return (
                "In languages that support **Tail Call Optimization (TCO)**, tail-recursive optimization can significantly reduce stack usage in Fibonacci computation by converting recursive calls into iterative loops.\n"
                "**How TCO Works:** A function is tail-recursive if the recursive call is the very last operation performed, and its result is directly returned without any further computations in the current stack frame. When a compiler/interpreter detects such a call, it can reuse the current stack frame for the recursive call instead of pushing a new one onto the call stack.\n"
                "**Leveraging for Fibonacci:** A tail-recursive Fibonacci implementation passes the two preceding Fibonacci numbers as arguments to the next recursive call, effectively carrying the 'state' forward:\n"
                "```python\n"
                "def fib_tail_optimized(n, a=0, b=1):\n"
                "    if n == 0: return a\n"
                "    # The recursive call is the last operation, its result is directly returned\n"
                "    return fib_tail_optimized(n - 1, b, a + b) \n"
                "```\n"
                "In languages with TCO (e.g., Scheme, Haskell, some C++ compilers with optimization flags), this `fib_tail_optimized(n, a, b)` call will be transformed into an iterative loop, resulting in **O(1) space complexity for the call stack**, avoiding stack overflow even for very large `n`. Python does *not* support TCO, so this specific optimization does not apply to Python code."
            )

        # 10. What is the Pisano period, and how can it be used to optimize repeated Fibonacci modulo calculations for large values of n?
        elif "pisano period" in q and "optimize repeated fibonacci modulo calculations" in q:
            return (
                "The **Pisano period**, denoted as `pi(m)`, is the period with which the Fibonacci sequence repeats itself modulo `m`. That is, for any positive integer `m`, the sequence `F(0) % m, F(1) % m, F(2) % m, ...` is periodic.\n"
                "**How it optimizes repeated Fibonacci modulo calculations for large values of n:**\n"
                "1.  **Periodicity:** `F(n) % m = F(n % pi(m)) % m`. This means that to find the `n`th Fibonacci number modulo `m`, you only need to compute the Fibonacci number at index `n % pi(m)`, which is typically a much smaller number.\n"
                "2.  **Efficiency:** Instead of computing `F(n)` for a potentially astronomically large `n` using O(log n) methods, you first find `pi(m)` (which can be precomputed or found using a brute-force search up to `m^2` or more efficiently). Then, for any query `F(n) % m`, you simply calculate `effective_n = n % pi(m)` and then find `F(effective_n) % m` using an efficient O(effective_n) iterative method or O(log effective_n) matrix exponentiation. Since `effective_n` is bounded by `pi(m)` (which itself is bounded), this becomes effectively an O(1) lookup after initial setup for `pi(m)` or if `pi(m)` is known.\n"
                "**Example:** The Pisano period for modulo 10 is `pi(10) = 60`. So, `F(100) % 10 = F(100 % 60) % 10 = F(40) % 10`."
            )

        # 11. How can space complexity be reduced to O(1) while still achieving optimal time performance in Fibonacci calculation?
        elif "space complexity be reduced to o(1) while still achieving optimal time performance" in q:
            return (
                "Space complexity for Fibonacci calculation can be reduced to **O(1) (constant space)** while still achieving optimal linear time performance (O(n)) by using the **iterative method with two variables (sliding window approach)**.\n"
                "**Mechanism:** To compute `F(i)`, you only need the two immediately preceding Fibonacci numbers, `F(i-1)` and `F(i-2)`. Instead of storing the entire sequence in an array (which would be O(n) space), you use two variables (e.g., `prev` and `curr`). In each iteration, you calculate `next_fib = prev + curr`, then update `prev = curr` and `curr = next_fib`. This way, memory usage remains constant irrespective of how large 'n' is, while the time complexity remains O(n) because each number up to `n` is computed once."
                "```python\n"
                "def fibonacci_o1_space(n):\n"
                "    if n <= 1: return n\n"
                "    prev, curr = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        prev, curr = curr, prev + curr\n"
                "    return curr\n"
                "```"
            )

        # 12. You're designing a system to compute Fibonacci numbers on GPUs. What parallelization strategy would you use and why?
        elif "compute fibonacci numbers on gpus" in q and "parallelization strategy" in q:
            return (
                "When designing a system to compute Fibonacci numbers on GPUs, the parallelization strategy would depend on the scale of computation (single F(n) vs. many F(n) queries) and the value of 'n':\n"
                "1.  **For a Single, Very Large F(n) (Matrix Exponentiation):**\n"
                "    * **Strategy:** Parallelize the **matrix multiplication** step within the matrix exponentiation algorithm. Each element of the resulting 2x2 matrix can be computed independently using a separate GPU thread/kernel. For example, computing `A[0][0]*B[0][0] + A[0][1]*B[1][0]` involves a few independent multiplications and additions which can be mapped to parallel threads.\n"
                "    * **Why:** While matrix exponentiation itself is `O(log n)` serial steps, the matrix multiplications (`O(2^3)` or `O(2^2)` for 2x2 matrices) can be done in parallel. For larger matrices (if generalized to other linear recurrences), the benefits are more pronounced. For 2x2, the actual gain might be minimal due to overhead, but it's the conceptual approach.\n"
                "2.  **For Many F(n) Queries (Batch Processing):**\n"
                "    * **Strategy:** Assign each `F(n)` calculation to a separate GPU thread/block. Each thread could compute its assigned Fibonacci number independently using an iterative O(1) space method.\n"
                "    * **Why:** GPUs excel at Single Instruction, Multiple Data (SIMD) operations. If you need to compute `F(10), F(20), F(30), ...` simultaneously, each computation is independent and can be run in parallel on different cores. This is highly efficient for batch processing many queries.\n"
                "3.  **For Generating a Range of Fibonacci Numbers (e.g., up to F(N)):**\n"
                "    * **Strategy:** Use a parallel prefix sum (scan) or similar parallel reduction pattern. This is less straightforward for Fibonacci due to its dependency (F(i) depends on F(i-1) and F(i-2)). However, more advanced parallel algorithms exist that can compute ranges of linear recurrences.\n"
                "**Overall:** The most practical parallelization on GPUs for Fibonacci is typically for **batch computation of many independent Fibonacci numbers** or for parallelizing the underlying **matrix multiplication** step in the O(log n) approach."
            )

        # 13. How would you apply dynamic programming with space optimization when generating the Fibonacci sequence from F({{start}}) to F({{end}})?
        elif "apply dynamic programming with space optimization when generating the fibonacci sequence from f({{start}}) to f({{end}})" in q:
            match = re.search(r"f\((\d+)\) to f\((\d+)\)", q)
            start_str = match.group(1) if match else "start"
            end_str = match.group(2) if match else "end"
            return (
                f"When generating the Fibonacci sequence from F({start_str}) to F({end_str}) with dynamic programming and space optimization, you would combine the O(1) space iterative method with selective storage:\n"
                "1.  **Initialize:** Get the values for `F(start)` and `F(start+1)` (or `F(start)` and `F(start-1)` if start is 0 or 1). You might need to compute these using the standard O(1) iterative method from F0 up to `F(start)`.\n"
                "2.  **Iterative Generation:** Once `F(start)` and `F(start+1)` are known, you can use the O(1) space iterative method (two variables) to generate subsequent Fibonacci numbers up to `F(end)`.\n"
                "3.  **Store Only Necessary Range:** Instead of storing all Fibonacci numbers from F0 to F(`end`), you only store the numbers required for the output, i.e., those from `F(start)` to `F(end)`. This is done by appending them to a list as they are generated.\n"
                "```python\n"
                "def generate_fib_range_optimized(start_idx, end_idx):\n"
                "    if end_idx < start_idx or end_idx < 0: return []\n"
                "    if start_idx < 0: start_idx = 0\n"
                "    \n"
                "    # Step 1: Get F(start_idx) and F(start_idx+1) efficiently\n"
                "    # We'll use the O(1) iterative method to get current F_k and F_{k+1}\n"
                "    a, b = 0, 1 # F(0), F(1)\n"
                "    for i in range(2, start_idx + 1): # Iterate up to start_idx to get its F_value\n"
                "        a, b = b, a + b\n"
                "    \n"
                "    # Now 'a' is F(start_idx) and 'b' is F(start_idx+1) if start_idx > 0\n"
                "    # Special handling for F(0) and F(1) at start_idx\n"
                "    if start_idx == 0: a = 0; b = 1\n"
                "    elif start_idx == 1: a = 1; b = 1\n"
                "    \n"
                "    result_range = []\n"
                "    current_fib = a\n"
                "    next_fib = b\n"
                "    \n"
                "    # Step 2: Generate and collect numbers in the required range\n"
                "    for i in range(start_idx, end_idx + 1):\n"
                "        result_range.append(current_fib)\n"
                "        current_fib, next_fib = next_fib, current_fib + next_fib\n"
                "    return result_range\n"
                "```\n"
                "**Time Complexity:** O(`end`) (because we still iterate up to `end` to compute the values). \n"
                "**Space Complexity:** O(`end` - `start`) (to store the final output range) plus O(1) for the working variables. If `start` is close to `end`, this becomes O(1) if the output is streamed or printed without storing the full list."
            )

        # 14. If you need to compute Fibonacci numbers on-demand in real-time systems, what optimizations would ensure constant response time?
        elif "fibonacci numbers on-demand in real-time systems" in q and "constant response time" in q:
            return (
                "To compute Fibonacci numbers on-demand in real-time systems with a need for constant response time, the primary optimization is **precomputation combined with O(1) lookup**.\n"
                "1.  **Precomputation:** During system initialization (before real-time operations begin), calculate and store all Fibonacci numbers that might ever be queried within the system's operational range into an array or hash map. This initial computation might take O(N) or O(N log F_N) time (for very large numbers), but it happens offline.\n"
                "2.  **O(1) Lookup:** Once precomputed, any subsequent query for `F(n)` simply involves looking up the value in the precomputed array/map by index `n`. This operation takes O(1) constant time, ensuring a predictable and fast response.\n"
                "3.  **Memory Management:** The limiting factor will be the memory required to store the precomputed values (O(N) or O(N log F_N)). Ensure this fits within the real-time system's memory constraints. If `N` is too large for storage, then constant response time might be infeasible without specialized hardware (e.g., dedicated ASICs) or a relaxation of the 'constant response time' requirement.\n"
                "4.  **Hardware Considerations:** For extremely high-frequency or ultra-low-latency needs, dedicated hardware (FPGA, ASIC) might be designed to compute the next Fibonacci number in a few clock cycles, achieving near-physical constant time."
            )

        # 15. What are the challenges of computing Fibonacci({{n}}) in systems with 32-bit integer overflow limits, and how would you optimize around them?
        elif "fibonacci" in q and "32-bit integer overflow limits" in q:
            match = re.search(r"fibonacci\((\d+)\)", q)
            n_str = match.group(1) if match else "n"
            return (
                f"Computing Fibonacci({n_str}) in systems with 32-bit integer overflow limits presents significant challenges because Fibonacci numbers grow exponentially and quickly exceed the maximum value a 32-bit integer can hold (approx. 2 * 10^9). For instance, F(47) is already larger than a 32-bit signed integer's max value.\n"
                "**Challenges:**\n"
                "1.  **Data Corruption:** Arithmetic operations (`a + b`) will produce incorrect results once the sum exceeds the `INT_MAX` for a 32-bit integer, leading to silent overflow and incorrect Fibonacci numbers.\n"
                "2.  **Loss of Precision:** Without proper handling, the results become meaningless beyond a certain `n` (around F(47) for signed 32-bit).\n"
                "**Optimization/Mitigation Strategies:**\n"
                "1.  **Use Larger Integer Types:** The most direct solution is to use larger integer types available in the language (e.g., `long long` in C++, `BigInteger` classes in Java, or Python's arbitrary-precision integers). This trades increased memory per number for correctness.\n"
                "2.  **Modular Arithmetic (if applicable):** If the problem requires `F(n) % M` (Fibonacci modulo some value `M`), then perform all additions modulo `M`. This keeps numbers within bounds and prevents overflow. This is efficient with matrix exponentiation.\n"
                "3.  **Matrix Exponentiation with Manual BigInt:** If you need the *exact* large Fibonacci number and cannot rely on built-in arbitrary precision (e.g., in C/C++ for very large `n`), implement matrix exponentiation where the underlying matrix multiplication logic uses a custom BigInt library or manual large number arithmetic. This maintains O(log n) time complexity while correctly handling large values.\n"
                "4.  **Iterative with BigInt:** For O(n) time, the iterative (O(1) space) method can be used, but the two variables (`a` and `b`) would need to store BigInt types, or the custom BigInt addition would be performed in each step.\n"
                "**Conclusion:** For `n` values that cause overflow in 32-bit systems, explicit measures for handling large numbers (either language-provided or custom-implemented BigInt types) or switching to modular arithmetic are essential."
            )


    return "Answer generation for this Level 3 Fibonacci question is not implemented yet."


def test_answer_algo_lvl1_fibonacci():
    test_questions = [
        "Write a loop-based algorithm to find the Fibonacci number at position 7.",
        "Create a recursive function to calculate Fibonacci(n), where n = 5.",
        "Write a program to print the Fibonacci series up to the 6th term.",
        "Display the Fibonacci numbers from position 0 to 8.",
        "Write a function to return a list of the first 9 Fibonacci numbers.",
        "Write a simple recursive function to calculate Fibonacci(4).",
        "Compute and print the 0th to 5th Fibonacci numbers.",
        "Print the first 17 elements of the Fibonacci sequence."
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"Test Case {i}:")
        print(f"Question: {question}")
        answer = answer_algo_lvl1_fibonacci(question)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

# Run the test function
# test_answer_algo_lvl1_fibonacci()


def test_answer_algo_lvl2_fibonacci():
    print("\n--- Testing Level 2 Algorithmic Answers (Specific Questions) ---\n")
    questions_lvl2_specific = [
        "Implement Fibonacci(10) using memoization to avoid redundant calculations.",
        "Write a bottom-up dynamic programming approach to compute Fibonacci(8).",
        "Modify the Fibonacci function to also return the total number of function calls made for n = 5.",
        "Implement a Fibonacci function that uses memoization with a dictionary for n = 7.",
        "Create an iterative approach for Fibonacci(12) using only two variables.",
        "Write a function that returns the sum of the first 6 Fibonacci numbers.",
        "Implement a function to generate the first 4 even Fibonacci numbers.",
        "Compute the Fibonacci number at index 9 using tabulation with an array.",
        "Write a program to compute Fibonacci(6) using tail recursion.",
        "Implement a function that finds the largest Fibonacci number smaller than 50.",
        "Create a function that returns all Fibonacci numbers less than 20.",
        "Using memoization, calculate Fibonacci(7) with a cache to track reused subproblems.",
        "Modify your iterative Fibonacci algorithm to print only odd Fibonacci numbers up to the 10th term.",
        "Write a function that returns True if a number 8 is a Fibonacci number.",
        "Print Fibonacci numbers between two given positions: start = 3, end = 7.",
        "Optimize your Fibonacci algorithm to handle inputs up to n = 20 efficiently.",
        "Write a function that returns the average of the first 8 Fibonacci numbers.",
        "Implement a program that computes Fibonacci(7) and prints all intermediate values."
    ]

    for i, question in enumerate(questions_lvl2_specific, 1):
        print(f"Test Case {i}:")
        print(f"Question: {question}")
        answer = answer_algo_lvl2_fibonacci(question)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

# To run the new tests, uncomment the following line:
# test_answer_algo_lvl2_fibonacci()

def test_answer_algo_lvl3_fibonacci():
    print("\n--- Testing Level 3 Algorithmic Answers ---\n")
    questions_lvl3 = [
        "Implement Fibonacci(100) using matrix exponentiation to achieve O(log n) time.",
        "Write a program to compute the 100th Fibonacci number modulo 1000000007.",
        "Create a function to compute large Fibonacci numbers (e.g., n = 100) using fast doubling technique.",
        "Implement a space-optimized Fibonacci algorithm that computes Fibonacci(1000) using O(1) space.",
        "Write an efficient algorithm to compute the sum of the first 100 Fibonacci numbers modulo 1000000007.",
        "Compare performance of recursive, iterative, and matrix-based Fibonacci implementations for n = 20.",
        "Design an algorithm to return the last digit of the 100th Fibonacci number efficiently.",
        "Develop a function to compute the sum of even Fibonacci numbers not exceeding 100.",
        "Compute Fibonacci(500) using matrix exponentiation with modular arithmetic to handle large results.",
        "Use memoization with a decorator pattern to compute Fibonacci(100) efficiently.",
        "Write a program that computes the nth Fibonacci number where n is up to 50 using efficient recursion with caching.",
        "Calculate Fibonacci(25) with memoization and analyze its time and space complexity.",
        "Optimize Fibonacci calculation for repeated queries using precomputation up to n = 100.",
        "Create a function to determine whether a number 21 is a Fibonacci number in O(1) time using mathematical properties.",
        "Generate a Fibonacci sequence up to 50 where values are stored using BigInteger types to handle overflow.",
        "Implement a circular buffer-based solution to store last two Fibonacci numbers for constant space calculation of Fibonacci(100).",
        "Compute and return the 25th Fibonacci number in binary format using efficient logic.",
        "Write a program to compute the number of Fibonacci numbers that are less than or equal to 100.",
        "Use memoization and iterative fallback to compute Fibonacci(15) and track subproblem reuse frequency.",
        "Build a reusable Fibonacci module that supports multiple modes: recursive, memoized, iterative, and matrix-based."
    ]

    for i, question in enumerate(questions_lvl3, 1):
        print(f"Test Case {i}:")
        print(f"Question: {question}")
        answer = answer_algo_lvl3_fibonacci(question)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

# Uncomment the line below to run Level 3 tests
# test_answer_algo_lvl3_fibonacci()

# --- NEW TEST BLOCK FOR APPLICATION QUESTIONS ---
def test_answer_application_fibonacci():
    print("\n--- Testing Level 1 Application Answers ---\n")
    questions_app_lvl1 = [
        "In a population growth model, rabbits reproduce following the Fibonacci sequence. What is the population after 6 months?",
        "How can the Fibonacci series help in modeling the number of petals in flowers over 3 generations?",
        "You are stacking tiles in a Fibonacci pattern. How many tiles will you need after 5 steps?",
        "In a board game, players advance based on Fibonacci numbers. What is the total advancement after 4 turns?",
        "In nature photography, the Fibonacci spiral is used for composition. How is Fibonacci relevant in arranging 7 objects?",
        "If the cost of each item follows a Fibonacci sequence, what is the total cost of 5 items?",
        "A storybook shows patterns in tiles using Fibonacci numbers. What is the pattern after 6 steps?",
        "In a math puzzle app, you get Fibonacci points for every level. What will your score be after 4 levels?",
        "A child's toy robot moves in Fibonacci steps. How far will it go after 5 moves?",
        "How can Fibonacci be used to schedule a basic fitness plan for 7 days with gradually increasing exercise time?",
        "A coding game gives you Fibonacci bonus points. What are your total bonus points after 4 rounds?",
        "Your piggy bank savings follow a Fibonacci pattern. How much do you save after 5 weeks?",
        "In a design contest, each layer of decoration follows the Fibonacci rule. How many decorations are used in 4 layers?",
        "A staircase puzzle grows with Fibonacci steps. How many steps are needed to build it after 5 levels?",
        "A pattern of colored lights in Fibonacci order is being set up. What will be the position of the 6th light?"
    ]
    for i, q in enumerate(questions_app_lvl1):
        print(f"Test Case {i+1}:")
        print(f"Question: {q}")
        print(f"Answer:\n{answer_application_fibonacci('Level 1', q)}\n{'-'*50}\n")

    print("\n--- Testing Level 2 Application Answers ---\n")
    questions_app_lvl2 = [
        "A computer animation algorithm uses Fibonacci numbers to determine frame sequences. What frame pattern is generated after 5 frames?",
        "A business uses Fibonacci-based scaling to project growth. What is the expected value after 7 intervals?",
        "In project planning, Fibonacci numbers are used to estimate task sizes. How would you assign story points for 6 tasks based on Fibonacci values?",
        "A museum display features Fibonacci-based lighting intervals. How should the lights be scheduled over 20 seconds?",
        "In a budgeting app, monthly savings follow Fibonacci increments. Calculate total savings over 5 months.",
        "A gardener plants trees in Fibonacci order each season. How many trees are planted by the 4th season?",
        "A robot follows a Fibonacci stepping rule. If it starts at 0, how far has it gone after 6 steps?",
        "In digital art, Fibonacci numbers decide brush stroke intervals. What stroke spacing would you expect after 5 iterations?",
        "A model simulates virus spread where new infections follow Fibonacci growth. What is the infected count after 7 days?",
        "Fibonacci numbers are used to design a music beat pattern. What will be the tempo variation after 5 bars?",
        "In a smart lighting system, Fibonacci numbers are used to adjust brightness dynamically. Describe the brightness pattern over 5 cycles.",
        "A sequence of Fibonacci numbers is used to determine seating in a stadium for distancing. What are the positions for the first 5 seats?",
        "A game awards Fibonacci coins after each round. How many coins does a player earn after 4 rounds?",
        "In a supply chain model, stock replenishment follows a Fibonacci strategy. How many units should be ordered after 5 weeks?",
        "A mobile app gamifies habits using Fibonacci scoring. How does a user’s progress evolve after 5 days of streaks?",
        "A drone flies increasing distances using Fibonacci numbers for each battery charge cycle. What distance is covered after 4 charges?",
        "A digital counter shows Fibonacci intervals between events. What is the value displayed on the 5th event?",
        "In a memory training app, items are repeated in Fibonacci intervals. When is the 4th repetition scheduled?",
        "An ecosystem simulation uses Fibonacci numbers to model prey population. What is the population size at generation 7?",
        "A robot builds layers in Fibonacci thickness. What is the total thickness after 5 layers?"
    ]
    for i, q in enumerate(questions_app_lvl2):
        print(f"Test Case {i+1}:")
        print(f"Question: {q}")
        print(f"Answer:\n{answer_application_fibonacci('Level 2', q)}\n{'-'*50}\n")

    print("\n--- Testing Level 3 Application Answers ---\n")
    questions_app_lvl3 = [
        "In a financial model, a company invests following a Fibonacci pattern for 5 years. How can you predict the ROI pattern and its long-term sustainability?",
        "You are designing a memory-efficient data structure where access time grows in a Fibonacci sequence. How would this affect lookup and insertion operations over 10 elements?",
        "In a music generation tool, chord progressions are structured using Fibonacci intervals. How can you generate a melody for 5 bars using this pattern?",
        "A simulation tracks resource consumption using Fibonacci-based intervals. How would this affect scalability and resource planning for 7 operation cycles?",
        "You are building a recursive AI planner where decisions unfold in Fibonacci time steps. What is the total time complexity for 5 steps and how do you optimize it?",
        "In an art installation, LED pulses follow a Fibonacci pattern in color intensity. How would you synchronize the lighting pattern for 3 zones?",
        "A cryptographic algorithm embeds Fibonacci sequences in key generation. How secure is this approach against brute-force over 1000 key spaces?",
        "You're analyzing a Fibonacci-based load balancing strategy in distributed systems. How does it perform under varying loads up to 10 tasks?",
        "A genetic algorithm uses Fibonacci steps for mutation frequency. How does this influence evolution speed across 5 generations?",
        "You're creating an e-learning module where quiz difficulty increases in Fibonacci order. How would you distribute 20 questions across levels?",
        "In a simulation game, resources are unlocked using Fibonacci logic. Design a strategy to optimize unlocking 5 levels with minimal waste.",
        "A delivery service adjusts routes based on Fibonacci time windows. How do you plan deliveries for 10 packages under this rule?",
        "In an IoT network, energy-saving intervals follow the Fibonacci sequence. How would this affect synchronization across 5 devices?",
        "A healthcare prediction model uses Fibonacci increments to track patient recovery stages. How effective is this for 3 chronic conditions?",
    ]
    for i, q in enumerate(questions_app_lvl3):
        print(f"Test Case {i+1}:")
        print(f"Question: {q}")
        print(f"Answer:\n{answer_application_fibonacci('Level 3', q)}\n{'-'*50}\n")


def test_answer_optimization_fibonacci():
    print("\n--- Testing Level 1 Optimization Answers ---\n")
    questions_opt_lvl1 = [
        "Why is recursion without memoization inefficient for computing Fibonacci numbers?",
        "How does memoization improve the efficiency of calculating the 10th Fibonacci number?",
        "What is the time complexity of the naive recursive Fibonacci algorithm?",
        "How can using a loop instead of recursion improve Fibonacci calculation performance?",
        "What is the space complexity of a recursive Fibonacci implementation without memoization?",
        "How does dynamic programming help in reducing redundant calculations in Fibonacci series?",
        "Which approach is faster for large Fibonacci numbers: recursion or iteration?",
        "Why does the naive recursive method compute the same Fibonacci values multiple times?",
        "What is the benefit of using a bottom-up approach in Fibonacci series computation?",
        "How can storing computed values help in optimizing the Fibonacci algorithm?",
        "What causes exponential time growth in recursive Fibonacci implementations?",
        "How many recursive calls are made to compute Fibonacci(5) using naive recursion?",
        "What change would reduce repeated calls in recursive Fibonacci implementation?",
        "How does the use of an auxiliary array improve Fibonacci performance?",
        "Can the Fibonacci problem be optimized to linear time? If so, how?"
    ]
    for i, q in enumerate(questions_opt_lvl1):
        print(f"Test Case {i+1}:")
        print(f"Question: {q}")
        print(f"Answer:\n{answer_optimization_fibonacci('Level 1', q)}\n{'-'*50}\n")

    print("\n--- Testing Level 2 Optimization Answers ---\n")
    questions_opt_lvl2 = [
        "Compare the time and space complexity of top-down (with memoization) vs bottom-up approaches in computing Fibonacci numbers up to 20.",
        "What are the trade-offs between recursion with memoization and an iterative loop in terms of memory usage for computing Fibonacci(30)?",
        "Why is bottom-up dynamic programming more memory-efficient than top-down with memoization for Fibonacci calculation?",
        "How does using only two variables instead of an array optimize the space complexity in Fibonacci computation?",
        "How would replacing a full DP table with a sliding window approach optimize Fibonacci series calculation?",
        "What is the optimized iterative method to compute Fibonacci numbers with O(1) space and O(n) time complexity?",
        "How does the time complexity of matrix exponentiation compare to DP methods for calculating large Fibonacci numbers like F(100)?",
        "In what scenarios is it worth switching from DP to matrix exponentiation for computing Fibonacci numbers?",
        "How does the recursive Fibonacci method behave when memoization is implemented using a dictionary instead of a list?",
        "Explain how tail recursion could improve optimization for Fibonacci series. Does it eliminate stack overhead?",
        "Why might using Python's `@lru_cache` decorator be better than manually coding memoization for Fibonacci?",
        "Given limited memory, how would you compute Fibonacci(50) efficiently while avoiding stack overflow and excess allocation?",
        "If memory is constrained, which Fibonacci approach is most suitable: recursion with memoization, bottom-up DP, or iterative two-variable?",
        "How does reducing the number of stored intermediate Fibonacci values affect both speed and memory usage?",
        "What optimizations can be applied when computing a range of Fibonacci numbers from F(10) to F(20) efficiently?"
    ]
    for i, q in enumerate(questions_opt_lvl2):
        print(f"Test Case {i+1}:")
        print(f"Question: {q}")
        print(f"Answer:\n{answer_optimization_fibonacci('Level 2', q)}\n{'-'*50}\n")

    print("\n--- Testing Level 3 Optimization Answers ---\n")
    questions_opt_lvl3 = [
        "How does matrix exponentiation reduce the time complexity of computing Fibonacci(100) to O(log n), and how can it be implemented efficiently?",
        "Explain how fast doubling technique optimizes the Fibonacci calculation. What is its time complexity compared to other methods?",
        "For large-scale Fibonacci computations like F(1000000), how would you choose between matrix exponentiation and fast doubling for performance-critical systems?",
        "What are the numerical stability concerns when using Binet’s Formula to compute Fibonacci(500) for large n, and how can you mitigate them?",
        "You need to compute Fibonacci numbers modulo 1000000007. How does modular arithmetic impact the optimization of large Fibonacci sequences?",
        "How can memoization be adapted for concurrent Fibonacci computation in a multi-threaded environment?",
        "For an embedded system with 100 KB memory, how would you implement an optimized Fibonacci calculator?",
        "Describe a caching mechanism to store and retrieve previously computed Fibonacci values in a long-running server application.",
        "How can tail-recursive optimization be leveraged in languages that support it to reduce stack usage in Fibonacci computation?",
        "What is the Pisano period, and how can it be used to optimize repeated Fibonacci modulo calculations for large values of n?",
        "How can space complexity be reduced to O(1) while still achieving optimal time performance in Fibonacci calculation?",
        "You're designing a system to compute Fibonacci numbers on GPUs. What parallelization strategy would you use and why?",
        "How would you apply dynamic programming with space optimization when generating the Fibonacci sequence from F(10) to F(50)?",
        "If you need to compute Fibonacci numbers on-demand in real-time systems, what optimizations would ensure constant response time?",
        "What are the challenges of computing Fibonacci(100) in systems with 32-bit integer overflow limits, and how would you optimize around them?"
    ]
    for i, q in enumerate(questions_opt_lvl3):
        print(f"Test Case {i+1}:")
        print(f"Question: {q}")
        print(f"Answer:\n{answer_optimization_fibonacci('Level 3', q)}\n{'-'*50}\n")

        
def answer_algorithmic_fibonacci(level, question):
    if level == "Level 1":
        return answer_algo_lvl1_fibonacci(question)
    elif level == "Level 2":
        return answer_algo_lvl2_fibonacci(question)
    elif level == "Level 3":
        return answer_algo_lvl3_fibonacci(question)
    else:
        return f"⚠️ Unsupported level: {level}"

# To run the new tests, uncomment the following line:
# test_answer_application_fibonacci() # UNCOMMENT THIS TO TEST APPLICATION LEVEL QUESTIONS

# Update the main execution block to call the new test function
if __name__ == "__main__":
    # You can choose which test levels to run
    test_answer_algo_lvl1_fibonacci()
    test_answer_algo_lvl2_fibonacci()
    # test_answer_algo_lvl3_fibonacci() # Uncomment this to test Level 3 Algorithmic questions
#    test_answer_application_fibonacci() # UNCOMMENT THIS TO TEST APPLICATION LEVEL QUESTIONS

    # test_answer_optimization_fibonacci()