import re
import math
import time # For performance comparisons

# --- HELPER FUNCTIONS (START) ---

def scalar_mult_cost(p_i, p_k, p_j):
    """Calculates the scalar multiplication cost for multiplying A(p_i x p_k) and B(p_k x p_j)."""
    return p_i * p_k * p_j

def parse_dims_string(dims_str):
    """Parses a dimension string like '30x35, 35x15, 15x5' into a list of integers [30, 35, 15, 5]."""
    # Handles strings like "30x35, 35x15, 15x5" or "[30,35,15,5]"
    
    # Check for array-like string e.g., [30,35,15,5]
    array_match = re.search(r'\[(\d+(?:,\s*\d+)*)\]', dims_str)
    if array_match:
        return list(map(int, array_match.group(1).split(',')))
    
    # Otherwise, parse as "AxB, BxC" format
    dims = []
    pairs = dims_str.split(',')
    
    # Extract first dimension from the first pair
    first_pair_match = re.search(r'(\d+)x(\d+)', pairs[0])
    if first_pair_match:
        dims.append(int(first_pair_match.group(1)))
        dims.append(int(first_pair_match.group(2)))
    
    # Extract only the second dimension from subsequent pairs (as first is already in dims)
    for i in range(1, len(pairs)):
        next_pair_match = re.search(r'\d+x(\d+)', pairs[i])
        if next_pair_match:
            dims.append(int(next_pair_match.group(1)))
            
    return dims

# Core MCM Dynamic Programming Solver
def matrix_chain_order_dp(p_dims):
    """
    Computes the minimum number of scalar multiplications and the optimal split points
    for a given chain of matrices using dynamic programming.
    
    Args:
        p_dims (list): A list of matrix dimensions, e.g., [10, 100, 5, 50] for A1(10x100), A2(100x5), A3(5x50).
                       Length of p_dims is n+1 where n is the number of matrices.

    Returns:
        tuple: (m, s, min_cost)
               m (list of lists): The cost table m[i][j] storing minimum scalar multiplications
                                  needed to compute matrix A_i ... A_j.
               s (list of lists): The split table s[i][j] storing the optimal split point k.
               min_cost (int): The overall minimum scalar multiplication cost.
    """
    n = len(p_dims) - 1 # Number of matrices
    
    # m[i][j] stores the minimum scalar multiplications needed to compute matrix A_i ... A_j
    m = [[0 for _ in range(n + 1)] for _ in range(n + 1)] #
    
    # s[i][j] stores the optimal split point k for matrix A_i ... A_j
    s = [[0 for _ in range(n + 1)] for _ in range(n + 1)] #

    # l is chain length
    for l in range(2, n + 1): #
        for i in range(1, n - l + 2): #
            j = i + l - 1 #
            m[i][j] = float('inf') # Initialize with infinity
            for k in range(i, j): # k is the split point
                # cost = m[i][k] + m[k+1][j] + p_dims[i-1]*p_dims[k]*p_dims[j]
                cost = m[i][k] + m[k+1][j] + scalar_mult_cost(p_dims[i-1], p_dims[k], p_dims[j]) #
                if cost < m[i][j]: #
                    m[i][j] = cost #
                    s[i][j] = k #
                    
    return m, s, m[1][n] #

def print_optimal_parenthesization(s_table, i, j):
    """
    Helper function to reconstruct the optimal parenthesization string.
    """
    if i == j:
        return f"A{i}"
    k = s_table[(i, j)] if isinstance(s_table, dict) else s_table[i][j]
    left = print_optimal_parenthesization(s_table, i, k)
    right = print_optimal_parenthesization(s_table, k + 1, j)
    return f"({left} x {right})"


def reconstruct_mcm_order(s_table, i, j, matrix_names=None):
    """Reconstructs the optimal multiplication order."""
    if matrix_names is None:
        matrix_names = [f"A{x}" for x in range(len(s_table))] # Placeholder names if not provided
    
    if i == j:
        return matrix_names[i-1] # Adjust for 0-indexed matrix_names if needed
    else:
        k = s_table[i][j]
        left_part = reconstruct_mcm_order(s_table, i, k, matrix_names)
        right_part = reconstruct_mcm_order(s_table, k + 1, j, matrix_names)
        return f"({left_part} x {right_part})"

def print_matrix_table(table, title, num_matrices):
    """Formats and prints a 2D matrix table (cost or split)."""
    header = "      " + " ".join(f"{j:<5}" for j in range(1, num_matrices + 1))
    separator = "      " + "------" * num_matrices
    rows = [f"{i:<5}| " + " ".join(f"{table[i][j]:<5}" for j in range(1, num_matrices + 1)) for i in range(1, num_matrices + 1)]
    return f"{title}:\n{header}\n{separator}\n" + "\n".join(rows)

# --- HELPER FUNCTIONS (END) ---


# --- MAIN ANSWER GENERATION FUNCTIONS (START) ---

def answer_algorithmic_mcm(level, question):
    q = question.lower()

    if level == "Level 1":
        # 1. Calculate the scalar multiplication cost for A1 ({{d1}}x{{d2}}) and A2 ({{d2}}x{{d3}}).
        if "calculate the scalar multiplication cost for a1" in q and "and a2" in q:
            match = re.search(r"a1 \((\d+)x(\d+)\) and a2 \((\d+)x(\d+)\)", q)
            if match:
                d1, d2 = int(match.group(1)), int(match.group(2))
                d2_prime, d3 = int(match.group(3)), int(match.group(4))

                # ✅ Step 1: Check for dimension compatibility
                if d2 != d2_prime:
                    return (
                        f"⚠️ Error: Matrix dimensions are incompatible.\n"
                        f"You can only multiply A1 ({d1}x{d2}) with A2 ({d2_prime}x{d3}) if their inner dimensions match.\n"
                        f"Here, d2 = {d2} ≠ {d2_prime}, so multiplication is not defined."
                    )

                # ✅ Step 2: Calculate the scalar multiplication cost
                cost = d1 * d2 * d3

                return (
                    f"🧮 **Scalar Multiplication Cost Calculation**\n\n"
                    f"You're multiplying two matrices:\n"
                    f"- A1: {d1}x{d2}\n"
                    f"- A2: {d2}x{d3}\n\n"
                    f"📘 **Formula:**\n"
                    f"To multiply A1 (MxN) by A2 (NxP), the cost is M × N × P scalar multiplications.\n\n"
                    f"📊 **Substituting values:**\n"
                    f"{d1} × {d2} × {d3} = **{cost}** scalar multiplications\n\n"
                    f"✅ **Answer:** {cost} scalar multiplications are required to multiply A1 and A2."
                )

        # 2. Determine the total scalar operations for multiplying A1 x A2 x A3 using dimensions {{dims}}.
        elif "total scalar operations for multiplying a1 x a2 x a3 using dimensions" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                
                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str

                # Two possible parenthesizations for 3 matrices: ((A1xA2)xA3) and (A1x(A2xA3))
                # Cost of ((A1xA2)xA3) = (p0*p1*p2) + (p0*p2*p3)
                cost1 = scalar_mult_cost(p[0], p[1], p[2]) + scalar_mult_cost(p[0], p[2], p[3])
                
                # Cost of (A1x(A2xA3)) = (p1*p2*p3) + (p0*p1*p3)
                cost2 = scalar_mult_cost(p[1], p[2], p[3]) + scalar_mult_cost(p[0], p[1], p[3])
                
                min_cost = min(cost1, cost2)
                return (
                    f"To determine the total scalar operations for multiplying A1 x A2 x A3 using dimensions {dims_str}:\n"
                    f"The dimensions are P = {p}. (A1: {p[0]}x{p[1]}, A2: {p[1]}x{p[2]}, A3: {p[2]}x{p[3]}).\n"
                    f"Possible parenthesizations and their costs:\n"
                    f"1.  ((A1 x A2) x A3):\n"
                    f"    Cost of (A1 x A2) = {p[0]}*{p[1]}*{p[2]} = {p[0]*p[1]*p[2]}\n"
                    f"    Cost of (result x A3) = {p[0]}*{p[2]}*{p[3]} = {p[0]*p[2]*p[3]}\n"
                    f"    Total Cost = {p[0]*p[1]*p[2]} + {p[0]*p[2]*p[3]} = {cost1}\n"
                    f"2.  (A1 x (A2 x A3)):\n"
                    f"    Cost of (A2 x A3) = {p[1]}*{p[2]}*{p[3]} = {p[1]*p[2]*p[3]}\n"
                    f"    Cost of (A1 x result) = {p[0]}*{p[1]}*{p[3]} = {p[0]*p[1]*p[3]}\n"
                    f"    Total Cost = {p[1]*p[2]*p[3]} + {p[0]*p[1]*p[3]} = {cost2}\n"
                    f"The minimum scalar operations required is: **{min_cost}**."
                )
        
        # 3. List all possible parenthesis options for 3 matrices with dimensions {{dims}} and calculate the cost of each.
        elif "list all possible parenthesis options for 3 matrices" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)

                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str

                # Cost of ((A1xA2)xA3) = (p0*p1*p2) + (p0*p2*p3)
                cost1 = scalar_mult_cost(p[0], p[1], p[2]) + scalar_mult_cost(p[0], p[2], p[3])
                
                # Cost of (A1x(A2xA3)) = (p1*p2*p3) + (p0*p1*p3)
                cost2 = scalar_mult_cost(p[1], p[2], p[3]) + scalar_mult_cost(p[0], p[1], p[3])
                
                return (
                    f"For 3 matrices (A1, A2, A3) with dimensions {dims_str} (P={p}):\n"
                    f"There are two possible ways to parenthesize the multiplication:\n"
                    f"1.  **((A1 x A2) x A3)**:\n"
                    f"    - Multiply A1({p[0]}x{p[1]}) x A2({p[1]}x{p[2]}) first: Cost = {p[0]}*{p[1]}*{p[2]} = {p[0]*p[1]*p[2]}\n"
                    f"    - Then multiply (A1A2)({p[0]}x{p[2]}) x A3({p[2]}x{p[3]}): Cost = {p[0]}*{p[2]}*{p[3]} = {p[0]*p[2]*p[3]}\n"
                    f"    - Total Cost = {cost1}\n"
                    f"2.  **(A1 x (A2 x A3))**:\n"
                    f"    - Multiply A2({p[1]}x{p[2]}) x A3({p[2]}x{p[3]}) first: Cost = {p[1]}*{p[2]}*{p[3]} = {p[1]*p[2]*p[3]}\n"
                    f"    - Then multiply A1({p[0]}x{p[1]}) x (A2A3)({p[1]}x{p[3]}): Cost = {p[0]}*{p[1]}*{p[3]} = {p[0]*p[1]*p[3]}\n"
                    f"    - Total Cost = {cost2}"
                )

        # 4. Which parenthesization is cheaper for A1 x A2 x A3 given dimensions {{dims}}?
        elif "which parenthesization is cheaper for a1 x a2 x a3" in q:
            match = re.search(r"dimensions (.+)\?", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)

                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str
                
                cost1 = scalar_mult_cost(p[0], p[1], p[2]) + scalar_mult_cost(p[0], p[2], p[3])
                cost2 = scalar_mult_cost(p[1], p[2], p[3]) + scalar_mult_cost(p[0], p[1], p[3])
                
                cheaper_option = "((A1 x A2) x A3)" if cost1 <= cost2 else "(A1 x (A2 x A3))"
                cheaper_cost = min(cost1, cost2)

                return (
                    f"For A1 x A2 x A3 with dimensions {dims_str} (P={p}):\n"
                    f"Cost of ((A1 x A2) x A3) = {cost1}\n"
                    f"Cost of (A1 x (A2 x A3)) = {cost2}\n"
                    f"The cheaper parenthesization is: **{cheaper_option}** with a cost of **{cheaper_cost}**."
                )

        # 5. Explain the steps for multiplying matrices A1, A2, and A3 using ((A1 x A2) x A3) with dimensions {{dims}}.
        elif "steps for multiplying matrices a1, a2, and a3 using ((a1 x a2) x a3)" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)

                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str
                
                cost_A1A2 = scalar_mult_cost(p[0], p[1], p[2])
                cost_intermediate_A1A2_A3 = scalar_mult_cost(p[0], p[2], p[3])
                total_cost = cost_A1A2 + cost_intermediate_A1A2_A3

                return (
                    f"To multiply matrices A1, A2, and A3 using the parenthesization ((A1 x A2) x A3) with dimensions {dims_str} (P={p}):\n"
                    f"1.  **First Multiplication: (A1 x A2)**\n"
                    f"    - Multiply A1 (dimensions {p[0]}x{p[1]}) by A2 (dimensions {p[1]}x{p[2]}).\n"
                    f"    - This results in an intermediate matrix (A1A2) with dimensions {p[0]}x{p[2]}.\n"
                    f"    - Scalar multiplication cost for this step: {p[0]} * {p[1]} * {p[2]} = **{cost_A1A2}**.\n"
                    f"2.  **Second Multiplication: ((A1A2) x A3)**\n"
                    f"    - Multiply the intermediate matrix (A1A2) (dimensions {p[0]}x{p[2]}) by A3 (dimensions {p[2]}x{p[3]}).\n"
                    f"    - This results in the final matrix with dimensions {p[0]}x{p[3]}.\n"
                    f"    - Scalar multiplication cost for this step: {p[0]} * {p[2]} * {p[3]} = **{cost_intermediate_A1A2_A3}**.\n"
                    f"Total scalar multiplication cost: {cost_A1A2} + {cost_intermediate_A1A2_A3} = **{total_cost}**."
                )

        # 6. Given matrices A1, A2, and A3 with dimensions {{dims}}, compute cost using A1 x (A2 x A3).
        elif "given matrices a1, a2, and a3 with dimensions" in q and "compute cost using a1 x (a2 x a3)" in q:
            match = re.search(r"dimensions (.+), compute cost using a1 x \(a2 x a3\)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)

                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str
                
                cost_A2A3 = scalar_mult_cost(p[1], p[2], p[3])
                cost_A1_intermediate_A2A3 = scalar_mult_cost(p[0], p[1], p[3])
                total_cost = cost_A2A3 + cost_A1_intermediate_A2A3

                return (
                    f"To compute the cost for A1 x (A2 x A3) with dimensions {dims_str} (P={p}):\n"
                    f"1.  **First Multiplication: (A2 x A3)**\n"
                    f"    - Multiply A2 (dimensions {p[1]}x{p[2]}) by A3 (dimensions {p[2]}x{p[3]}).\n"
                    f"    - This results in an intermediate matrix (A2A3) with dimensions {p[1]}x{p[3]}.\n"
                    f"    - Scalar multiplication cost for this step: {p[1]} * {p[2]} * {p[3]} = **{cost_A2A3}**.\n"
                    f"2.  **Second Multiplication: (A1 x (A2A3))**\n"
                    f"    - Multiply A1 (dimensions {p[0]}x{p[1]}) by the intermediate matrix (A2A3) (dimensions {p[1]}x{p[3]}).\n"
                    f"    - This results in the final matrix with dimensions {p[0]}x{p[3]}.\n"
                    f"    - Scalar multiplication cost for this step: {p[0]} * {p[1]} * {p[3]} = **{cost_A1_intermediate_A2A3}**.\n"
                    f"Total scalar multiplication cost: {cost_A2A3} + {cost_A1_intermediate_A2A3} = **{total_cost}**."
                )

        # 7. Determine the minimum number of scalar multiplications needed to multiply matrices A1 x A2 x A3 with dimensions {{dims}}.
        elif "minimum number of scalar multiplications needed to multiply matrices a1 x a2 x a3" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)

                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str
                
                cost1 = scalar_mult_cost(p[0], p[1], p[2]) + scalar_mult_cost(p[0], p[2], p[3])
                cost2 = scalar_mult_cost(p[1], p[2], p[3]) + scalar_mult_cost(p[0], p[1], p[3])
                
                min_cost = min(cost1, cost2)
                return (
                    f"To determine the minimum number of scalar multiplications for A1 x A2 x A3 with dimensions {dims_str} (P={p}):\n"
                    f"Cost of ((A1 x A2) x A3) = {cost1}\n"
                    f"Cost of (A1 x (A2 x A3)) = {cost2}\n"
                    f"The minimum cost is: **{min_cost}**."
                )

        # 8. How would the multiplication cost change if A2 was transposed in a matrix chain with dimensions {{dims}}?
        elif "multiplication cost change if a2 was transposed in a matrix chain with dimensions" in q:
            match = re.search(r"dimensions (.+)\?", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)

                if len(p) < 3: # Need at least 2 matrices to transpose A2
                    return "Error: Insufficient matrices to transpose A2. Need at least A1 x A2."
                
                # Original dimensions: A1(p[0]xp[1]), A2(p[1]xp[2]), A3(p[2]xp[3]) (if 3 matrices)
                # Transposing A2(p[1]xp[2]) -> A2_T(p[2]xp[1])
                # This breaks the chain A1(p[0]xp[1]) x A2_T(p[2]xp[1]) unless p[1] == p[2]
                
                if p[1] != p[2]:
                    return (
                        f"Given matrix chain dimensions {dims_str} (P={p}):\n"
                        f"If A2 ({p[1]}x{p[2]}) was transposed to A2_T ({p[2]}x{p[1]}), the original chain A1 x A2 would become A1({p[0]}x{p[1]}) x A2_T({p[2]}x{p[1]}).\n"
                        f"Since the inner dimensions ({p[1]} and {p[2]}) do not match, the multiplication A1 x A2_T would become **invalid**. The chain cannot be multiplied in this scenario."
                    )
                else:
                    # If p[1] == p[2], then A2_T has dimensions p[1]xp[2].
                    # The dimensions remain effectively the same for multiplication purposes (p[0]xp[1]) x (p[1]xp[2])
                    # However, if dimensions are defined as p_0, p_1, p_2, p_3. A2 is p_1 x p_2.
                    # A2_T is p_2 x p_1.
                    # The chain becomes: A1(p_0 x p_1) x A2_T(p_2 x p_1) x A3(p_1 x p_3)
                    # This would only be valid if p_1 == p_2 for A1 x A2_T, and p_1 == p_1 for A2_T x A3.
                    # The most common interpretation of "transposed A2" in MCM context is that if A2 was square, its dimensions for multiplication stay the same.
                    # If not square, it likely makes the chain invalid.
                    
                    # For simplicity, if A2 is square (p[1] == p[2]), dimensions remain effectively same.
                    # If not square, the multiplication becomes invalid.
                    # For a non-square A2, transposing breaks the chain unless very specific dimensions.
                    return (
                        f"Given matrix chain dimensions {dims_str} (P={p}):\n"
                        f"If A2 ({p[1]}x{p[2]}) was transposed, its new dimensions would be {p[2]}x{p[1]}.\n"
                        f"For the original chain to remain valid (A1 x A2_T x A3...),\n"
                        f"A1({p[0]}x{p[1]}) must be compatible with A2_T({p[2]}x{p[1]}), which means {p[1]} == {p[2]}.\n"
                        f"If {p[1]} != {p[2]}: The multiplication chain would become **invalid** (e.g., A1x(A2_T) cannot be performed).\n"
                        f"If {p[1]} == {p[2]}: The dimensions for multiplication compatibility remain effectively the same (p[0]xp[1] by p[1]xp[2]). In this specific case, the multiplication cost would **not change** from the original setup, as the dimensions for the scalar multiplications remain the same."
                    )

        # 9. Find and compare the scalar multiplication cost for both parenthesizations: ((A1 x A2) x A3) vs A1 x (A2 x A3) with dimensions {{dims}}.
        elif "find and compare the scalar multiplication cost for both parenthesizations" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)

                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str
                
                cost1 = scalar_mult_cost(p[0], p[1], p[2]) + scalar_mult_cost(p[0], p[2], p[3])
                cost2 = scalar_mult_cost(p[1], p[2], p[3]) + scalar_mult_cost(p[0], p[1], p[3])
                
                comparison_result = "Cost of ((A1 x A2) x A3) = {cost1}\nCost of (A1 x (A2 x A3)) = {cost2}\n".format(cost1=cost1, cost2=cost2)
                if cost1 < cost2:
                    comparison_result += "((A1 x A2) x A3) is cheaper."
                elif cost2 < cost1:
                    comparison_result += "(A1 x (A2 x A3)) is cheaper."
                else:
                    comparison_result += "Both parenthesizations have the same cost."

                return (
                    f"For dimensions {dims_str} (P={p}), comparing both parenthesizations:\n"
                    f"{comparison_result}"
                )

        # 10. What values of k (split index) should be considered to compute optimal multiplication of A1 to A3 for dimensions {{dims}}?
        elif "what values of k (split index) should be considered" in q:
            match = re.search(r"dimensions (.+)\?", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                
                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str

                # For A_i ... A_j, k ranges from i to j-1.
                # For A1 to A3 (i=1, j=3), k ranges from 1 to 3-1 = 2.
                # So k can be 1 or 2.
                # k=1 splits (A1) x (A2A3)
                # k=2 splits (A1A2) x (A3)
                
                k_values = [1, 2]
                return (
                    f"To compute optimal multiplication of A1 to A3 with dimensions {dims_str} (P={p}):\n"
                    f"The possible split points `k` (where the chain is divided into (A_i...A_k) x (A_k+1...A_j)) should be considered from `i` to `j-1`.\n"
                    f"For A1 to A3 (i=1, j=3), the values of `k` to consider are: **{k_values}**.\n"
                    f"- `k=1`: Splits as (A1) x (A2 x A3)\n"
                    f"- `k=2`: Splits as (A1 x A2) x (A3)"
                )
        
        # 11. Describe the scalar operations in the multiplication of A1 ({{d1}}x{{d2}}) and A2 ({{d2}}x{{d3}}).
        elif "describe the scalar operations in the multiplication of a1" in q and "and a2" in q:
            match = re.search(r"a1 \((\d+)x(\d+)\) and a2 \((\d+)x(\d+)\)", q)
            if match:
                d1, d2 = int(match.group(1)), int(match.group(2))
                d2_prime, d3 = int(match.group(3)), int(match.group(4))
                
                if d2 != d2_prime:
                    return "Error: Matrix dimensions are incompatible for multiplication (inner dimensions must match)."
                
                cost = scalar_mult_cost(d1, d2, d3)
                return (
                    f"For the multiplication of matrix A1 ({d1}x{d2}) and A2 ({d2}x{d3}):\n"
                    f"The resulting matrix will have dimensions {d1}x{d3}.\n"
                    f"To compute each element `C[i][j]` of the resulting matrix C:\n"
                    f"  `C[i][j] = sum(A1[i][k] * A2[k][j])` for k from 0 to {d2-1}.\n"
                    f"This involves {d2} multiplications and {d2-1} additions for each element. (Focusing on multiplications for scalar cost).\n"
                    f"Since there are {d1} * {d3} elements in the resulting matrix, the total number of scalar multiplications is:\n"
                    f"{d1} * {d2} * {d3} = **{cost}**."
                )

        # 12. Compute and explain the total cost for multiplying A1 x A2 with dimensions {{d1}}x{{d2}} and {{d2}}x{{d3}}.
        elif "compute and explain the total cost for multiplying a1 x a2 with dimensions" in q:
            match = re.search(r"dimensions (\d+)x(\d+) and (\d+)x(\d+)", q)
            if match:
                d1, d2 = int(match.group(1)), int(match.group(2))
                d2_prime, d3 = int(match.group(3)), int(match.group(4))
                
                if d2 != d2_prime:
                    return "Error: Matrix dimensions are incompatible for multiplication (inner dimensions must match)."
                
                cost = scalar_mult_cost(d1, d2, d3)
                return (
                    f"To compute the total cost for multiplying A1 ({d1}x{d2}) and A2 ({d2}x{d3}):\n"
                    f"The cost of multiplying two matrices A(MxN) and B(NxP) is M*N*P scalar multiplications.\n"
                    f"Here, M={d1}, N={d2}, P={d3}.\n"
                    f"Total Cost = {d1} * {d2} * {d3} = **{cost}**.\n"
                    f"This cost represents the minimum number of basic (scalar) multiplications required to produce the elements of the resulting matrix."
                )

        # 13. If matrix A1 has dimensions {{d1}}x{{d2}}, and A2 has dimensions {{d2}}x{{d3}}, what is the multiplication cost?
        elif "if matrix a1 has dimensions" in q and "a2 has dimensions" in q:
            match = re.search(r"a1 has dimensions (\d+)x(\d+), and a2 has dimensions (\d+)x(\d+)", q)
            if match:
                d1, d2 = int(match.group(1)), int(match.group(2))
                d2_prime, d3 = int(match.group(3)), int(match.group(4))
                
                if d2 != d2_prime:
                    return "Error: Matrix dimensions are incompatible for multiplication (inner dimensions must match)."
                
                cost = scalar_mult_cost(d1, d2, d3)
                return (
                    f"If matrix A1 has dimensions {d1}x{d2} and A2 has dimensions {d2}x{d3}:\n"
                    f"The multiplication cost is calculated as (rows of A1) * (common dimension) * (columns of A2).\n"
                    f"Cost = {d1} * {d2} * {d3} = **{cost}**."
                )
        
        # 14. For three matrices with dimensions {{dims}}, calculate the multiplication cost for both possible parenthesizations.
        elif "for three matrices with dimensions" in q and "calculate the multiplication cost for both possible parenthesizations" in q:
            match = re.search(r"dimensions (.+), calculate the multiplication cost for both possible parenthesizations", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)

                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str
                
                cost1 = scalar_mult_cost(p[0], p[1], p[2]) + scalar_mult_cost(p[0], p[2], p[3])
                cost2 = scalar_mult_cost(p[1], p[2], p[3]) + scalar_mult_cost(p[0], p[1], p[3])
                
                return (
                    f"For three matrices with dimensions {dims_str} (P={p}):\n"
                    f"The two possible parenthesizations are ((A1 x A2) x A3) and (A1 x (A2 x A3)).\n"
                    f"Cost of ((A1 x A2) x A3) = **{cost1}**.\n"
                    f"Cost of (A1 x (A2 x A3)) = **{cost2}**."
                )

        # 15. Show the step-by-step scalar operations for ((A1 x A2) x A3) with matrix dimensions {{dims}}.
        elif "show the step-by-step scalar operations for ((a1 x a2) x a3)" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)

                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str
                
                cost_A1A2 = scalar_mult_cost(p[0], p[1], p[2])
                cost_intermediate_A1A2_A3 = scalar_mult_cost(p[0], p[2], p[3])
                total_cost = cost_A1A2 + cost_intermediate_A1A2_A3

                return (
                    f"Step-by-step scalar operations for ((A1 x A2) x A3) with dimensions {dims_str} (P={p}):\n"
                    f"1.  **Operation: A1 x A2**\n"
                    f"    - Dimensions: A1({p[0]}x{p[1]}) and A2({p[1]}x{p[2]})\n"
                    f"    - Resulting matrix (A1A2) dimensions: {p[0]}x{p[2]}\n"
                    f"    - Scalar multiplications: {p[0]} * {p[1]} * {p[2]} = **{cost_A1A2}**\n"
                    f"2.  **Operation: (A1A2) x A3**\n"
                    f"    - Dimensions: (A1A2)({p[0]}x{p[2]}) and A3({p[2]}x{p[3]})\n"
                    f"    - Resulting matrix dimensions: {p[0]}x{p[3]}\n"
                    f"    - Scalar multiplications: {p[0]} * {p[2]} * {p[3]} = **{cost_intermediate_A1A2_A3}**\n"
                    f"Total scalar operations: {cost_A1A2} + {cost_intermediate_A1A2_A3} = **{total_cost}**."
                )

        # 16. Describe how to find the cost of matrix chain multiplication using only 3 matrices with dimensions {{dims}}.
        elif "describe how to find the cost of matrix chain multiplication using only 3 matrices" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)

                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str
                
                return (
                    f"To find the cost of matrix chain multiplication using only 3 matrices (A1, A2, A3) with dimensions {dims_str} (P={p}):\n"
                    f"1.  Identify the two possible parenthesizations: `((A1 x A2) x A3)` and `(A1 x (A2 x A3))`.\n"
                    f"2.  Calculate the scalar multiplication cost for each parenthesization:\n"
                    f"    - Cost of `((A1 x A2) x A3)` = `(p[0]*p[1]*p[2]) + (p[0]*p[2]*p[3])`\n"
                    f"    - Cost of `(A1 x (A2 x A3))` = `(p[1]*p[2]*p[3]) + (p[0]*p[1]*p[3])`\n"
                    f"    (where `p` is the dimension array [d1, d2, d3, d4] corresponding to A1(d1xd2), A2(d2xd3), A3(d3xd4))\n"
                    f"3.  The minimum of these two calculated costs will be the optimal (lowest) cost."
                )

        # 17. What is the multiplication order that gives the lowest cost for matrices with dimensions {{dims}}?
        elif "multiplication order that gives the lowest cost" in q:
            match = re.search(r"dimensions (.+)\?", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)

                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str
                
                cost1 = scalar_mult_cost(p[0], p[1], p[2]) + scalar_mult_cost(p[0], p[2], p[3])
                cost2 = scalar_mult_cost(p[1], p[2], p[3]) + scalar_mult_cost(p[0], p[1], p[3])
                
                if cost1 <= cost2:
                    optimal_order = "((A1 x A2) x A3)"
                else:
                    optimal_order = "(A1 x (A2 x A3))"
                
                return (
                    f"For matrices with dimensions {dims_str} (P={p}):\n"
                    f"The cost of ((A1 x A2) x A3) is {cost1}.\n"
                    f"The cost of (A1 x (A2 x A3)) is {cost2}.\n"
                    f"The multiplication order that gives the lowest cost is: **{optimal_order}**."
                )

        # 18. How many total multiplications are required to multiply matrices A1 ({{d1}}x{{d2}}) and A2 ({{d2}}x{{d3}})?
        elif "how many total multiplications are required to multiply matrices a1" in q and "and a2" in q:
            match = re.search(r"a1 \((\d+)x(\d+)\) and a2 \((\d+)x(\d+)\)", q)
            if match:
                d1, d2 = int(match.group(1)), int(match.group(2))
                d2_prime, d3 = int(match.group(3)), int(match.group(4))
                
                if d2 != d2_prime:
                    return "Error: Matrix dimensions are incompatible for multiplication (inner dimensions must match)."
                
                cost = scalar_mult_cost(d1, d2, d3)
                return (
                    f"To multiply matrices A1 ({d1}x{d2}) and A2 ({d2}x{d3}):\n"
                    f"The total number of scalar multiplications required is (rows of A1) * (common dimension) * (columns of A2).\n"
                    f"Total multiplications = {d1} * {d2} * {d3} = **{cost}**."
                )

        # 19. Compute both ((A1 x A2) x A3) and (A1 x (A2 x A3)) for dimensions {{dims}} and identify which is optimal.
        elif "compute both ((a1 x a2) x a3) and (a1 x (a2 x a3))" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)

                if len(p) != 4:
                    return "Error: Expected dimensions for 3 matrices (e.g., [d1, d2, d3, d4]). Found: " + dims_str
                
                cost1 = scalar_mult_cost(p[0], p[1], p[2]) + scalar_mult_cost(p[0], p[2], p[3])
                cost2 = scalar_mult_cost(p[1], p[2], p[3]) + scalar_mult_cost(p[0], p[1], p[3])
                
                optimal_order = "((A1 x A2) x A3)" if cost1 <= cost2 else "(A1 x (A2 x A3))"
                optimal_cost = min(cost1, cost2)

                return (
                    f"For dimensions {dims_str} (P={p}):\n"
                    f"1.  Cost of ((A1 x A2) x A3): **{cost1}**.\n"
                    f"2.  Cost of (A1 x (A2 x A3)): **{cost2}**.\n"
                    f"The optimal parenthesization is **{optimal_order}** with a cost of **{optimal_cost}**."
                )

        # 20. Explain the effect of matrix shape on multiplication cost for dimensions {{dims}}.
        elif "explain the effect of matrix shape on multiplication cost" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                
                if len(p) < 3: # Need at least 2 matrices to demonstrate shape effect
                    return "Error: Insufficient matrices to demonstrate shape effect. Need at least 2 matrices."

                return (
                    f"The shape of matrices (i.e., their dimensions) has a **profound effect** on the scalar multiplication cost in matrix chain multiplication.\n"
                    f"For multiplying A(MxN) by B(NxP), the cost is M*N*P.\n"
                    f"Given a chain of matrices like A({p[0]}x{p[1]}), B({p[1]}x{p[2]}), C({p[2]}x{p[3]}) (from {dims_str}):\n"
                    f"The critical factor is the **common intermediate dimension**.\n"
                    f"For example, if you have dimensions like (10x100), (100x5), (5x50):\n"
                    f"  - ((A1xA2)xA3): (10*100*5) + (10*5*50) = 5000 + 2500 = 7500\n"
                    f"  - (A1x(A2xA3)): (100*5*50) + (10*100*50) = 25000 + 50000 = 75000\n"
                    f"In this example, the order matters greatly due to the intermediate dimensions. A large common dimension (like 100 in the second case, when 100x5 multiplies by 50) can lead to a very high cost. Optimal parenthesization chooses an order that minimizes these intermediate products."
                )

    elif level == "Level 2":
        # 1. Use dynamic programming to find the minimum multiplication cost for matrix chain {{dims}}.
        if "use dynamic programming to find the minimum multiplication cost for matrix chain" in q:
            match = re.search(r"dimensions (.+)", q)
            if match:
                dims_str = match.group(1).strip().rstrip('.')  # Clean the string
                p = parse_dims_string(dims_str)

                if len(p) < 2:
                    return "⚠️ Error: Invalid input. You need at least two dimensions to define a matrix chain."

                m, s, min_cost = matrix_chain_order_dp(p)

                return (
                    f"🧮 **Matrix Chain Multiplication (MCM) using Dynamic Programming**\n\n"
                    f"📘 **Given Dimensions:** {dims_str} → Interpreted as P = {p}\n"
                    f"This means you are multiplying matrices:\n"
                    f"A1: {p[0]}x{p[1]}, A2: {p[1]}x{p[2]}, ..., An: {p[len(p)-2]}x{p[len(p)-1]}\n\n"
                    f"🎯 **Goal:** Find the optimal way to parenthesize the chain to minimize total scalar multiplications.\n\n"
                    f"💡 **Approach:**\n"
                    f"- Use a DP table `m[i][j]` to store the minimum number of multiplications needed for matrices Ai to Aj\n"
                    f"- Try all possible places to split the chain and take the minimum\n"
                    f"- Time Complexity: O(n³), Space Complexity: O(n²)\n\n"
                    f"✅ **Answer:** The minimum number of scalar multiplications required is: **{min_cost}**"
                )



        # 2. Show how the cost table (m[i][j]) is built for dimensions {{dims}} step-by-step.
        elif "show how the cost table (m[i][j]) is built for dimensions" in q:
            match = re.search(r"dimensions (.+)\s*step-by-step", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 1:
                    return "Error: Invalid dimensions. Need at least one matrix."

                m = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
                s = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
                
                build_log = []
                build_log.append("Initialize m[i][j] table with 0s for single matrix chains (l=1) and infinity for others.")

                for l in range(2, n + 1): # l is chain length
                    build_log.append(f"\n--- Calculating for chain length l = {l} ---")
                    for i in range(1, n - l + 2): #
                        j = i + l - 1 #
                        m[i][j] = float('inf') #
                        
                        build_log.append(f"Calculating m[{i}][{j}] (for matrices A{i} to A{j}):")
                        for k in range(i, j): # k is the split point
                            cost = m[i][k] + m[k+1][j] + scalar_mult_cost(p[i-1], p[k], p[j]) #
                            build_log.append(f"  Split at k={k}: (A{i}..A{k}) x (A{k+1}..A{j})")
                            build_log.append(f"  Cost = m[{i}][{k}]({m[i][k]}) + m[{k+1}][{j}]({m[k+1][j]}) + p[{i-1}]({p[i-1]})*p[{k}]({p[k]})*p[{j}]({p[j]}) = {cost}")
                            if cost < m[i][j]:
                                m[i][j] = cost
                                s[i][j] = k
                                build_log.append(f"  New minimum found! m[{i}][{j}] = {cost}, s[{i}][{j}] = {k}")
                            else:
                                build_log.append(f"  Cost {cost} is not better than current m[{i}][{j}] ({m[i][j]})")
                        build_log.append(f"Final m[{i}][{j}] = {m[i][j]}")

                final_m_table_str = print_matrix_table(m, "Final Cost Table (m)", n)
                
                return (
                    f"To show how the cost table m[i][j] is built for dimensions {dims_str} (P={p}) step-by-step:\n"
                    f"The table is filled diagonally for increasing chain lengths (l).\n"
                    f"Steps:\n"
                    + "\n".join(build_log) + "\n\n"
                    + final_m_table_str
                )

        # 3. Trace m[1][3] calculation in matrix chain multiplication using dimensions {{dims}}.
        elif "trace m[1][3] calculation in matrix chain multiplication" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 3: # Need at least 3 matrices for m[1][3]
                    return "Error: Dimensions must represent at least 3 matrices to calculate m[1][3]."

                # Perform actual DP computation to get intermediate values
                m_full, s_full, _ = matrix_chain_order_dp(p)

                trace_log = []
                trace_log.append(f"Tracing calculation of m[1][3] for dimensions {dims_str} (P={p}):")
                trace_log.append("m[1][3] represents the minimum cost to multiply matrices A1 to A3.")
                trace_log.append("It is calculated for chain length l = 3 (j - i + 1 = 3 => 3 - 1 + 1 = 3).")
                
                i, j = 1, 3
                min_cost_m13 = float('inf')
                optimal_k_m13 = 0

                trace_log.append(f"\nEvaluating m[{i}][{j}] (A{i} to A{j}):")
                for k in range(i, j): # k can be 1 or 2
                    cost_curr_split = m_full[i][k] + m_full[k+1][j] + scalar_mult_cost(p[i-1], p[k], p[j]) #
                    trace_log.append(f"  Considering split point k = {k}:")
                    trace_log.append(f"    Cost = m[{i}][{k}]({m_full[i][k]}) + m[{k+1}][{j}]({m_full[k+1][j]}) + p[{i-1}]({p[i-1]})*p[{k}]({p[k]})*p[{j}]({p[j]})")
                    trace_log.append(f"    Cost = {cost_curr_split}")
                    if cost_curr_split < min_cost_m13:
                        min_cost_m13 = cost_curr_split
                        optimal_k_m13 = k
                        trace_log.append(f"    New minimum found! Current m[1][3] = {min_cost_m13}, split s[1][3] = {optimal_k_m13}")
                    else:
                        trace_log.append(f"    Cost ({cost_curr_split}) is not better than current minimum ({min_cost_m13}).")

                trace_log.append(f"\nFinal m[1][3] = {m_full[1][3]} (Optimal split at k={s_full[1][3]})")
                
                return "\n".join(trace_log)

        # 4. Which value of k (split index) results in the minimum multiplication cost for matrices A2 to A4 with {{dims}}?
        elif "value of k (split index) results in the minimum multiplication cost for matrices a2 to a4" in q:
            match = re.search(r"dimensions (.+)\?", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 4: # Need at least 4 matrices for A2 to A4
                    return "Error: Dimensions must represent at least 4 matrices to calculate optimal split for A2 to A4."
                
                # For A2 to A4, i=2, j=4
                # k ranges from i to j-1, so k from 2 to 3.
                m, s, _ = matrix_chain_order_dp(p) #
                
                optimal_k_value = s[2][4] #
                min_cost_a2a4 = m[2][4] #

                return (
                    f"For matrices A2 to A4 (corresponding to p_1 to p_4 in dimensions P={p}):\n"
                    f"This corresponds to calculating m[2][4]. The possible split points `k` are from 2 to 3.\n"
                    f"Using dynamic programming, the value of `k` that results in the minimum multiplication cost is: **{optimal_k_value}**.\n"
                    f"The minimum cost for A2..A4 is: **{min_cost_a2a4}**."
                )

        # 5. Write a bottom-up DP function to compute matrix chain multiplication cost for dimensions {{dims}}.
        elif "write a bottom-up dp function to compute matrix chain multiplication cost for dimensions" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p_example = parse_dims_string(dims_str)
                
                # Reuse the global DP solver as it's a bottom-up implementation.
                m, s, min_cost_example = matrix_chain_order_dp(p_example)
                
                return (
                    f"Here's a bottom-up dynamic programming function to compute the matrix chain multiplication cost for given dimensions (P={p_example}):\n\n"
                    "```python\n"
                    "def scalar_mult_cost(p_i, p_k, p_j):\n"
                    "    return p_i * p_k * p_j\n\n"
                    "def matrix_chain_order_dp(p_dims):\n"
                    "    n = len(p_dims) - 1 # Number of matrices\n"
                    "    m = [[0 for _ in range(n + 1)] for _ in range(n + 1)] # Cost table\n"
                    "    s = [[0 for _ in range(n + 1)] for _ in range(n + 1)] # Split table\n"
                    "\n"
                    "    for l in range(2, n + 1): # l is chain length\n"
                    "        for i in range(1, n - l + 2): # i is start index\n"
                    "            j = i + l - 1 # j is end index\n"
                    "            m[i][j] = float('inf') # Initialize cost with infinity\n"
                    "            for k in range(i, j): # k is split point\n"
                    "                cost = m[i][k] + m[k+1][j] + scalar_mult_cost(p_dims[i-1], p_dims[k], p_dims[j]) #\n"
                    "                if cost < m[i][j]:\n"
                    "                    m[i][j] = cost\n"
                    "                    s[i][j] = k\n"
                    "    return m, s, m[1][n]\n"
                    "```\n\n"
                    f"Input dimensions: {dims_str}\n"
                    f"Output Minimum Cost: **{min_cost_example}**."
                )

        # 6. Describe the recursive approach to matrix chain multiplication and compare it with DP for dimensions {{dims}}.
        elif "describe the recursive approach to matrix chain multiplication and compare it with dp" in q:
            match = re.search(r"dimensions (.+)\.", q)
            dims_str = match.group(1) if match else "sample_dims"
            
            # Helper for naive recursive MCM (for comparison)
            memo_recursive_mcm = {}
            def mcm_recursive_naive(p_dims_rec, i, j):
                if i == j:
                    return 0
                if (i, j) in memo_recursive_mcm: # Using memoization for practical demonstration
                    return memo_recursive_mcm[(i, j)]
                
                min_ops = float('inf')
                for k in range(i, j):
                    cost = mcm_recursive_naive(p_dims_rec, i, k) + \
                           mcm_recursive_naive(p_dims_rec, k + 1, j) + \
                           p_dims_rec[i-1] * p_dims_rec[k] * p_dims_rec[j]
                    min_ops = min(min_ops, cost)
                memo_recursive_mcm[(i, j)] = min_ops
                return min_ops

            # Note: For actual comparison, the 'naive' should not use memo. But it's too slow.
            # So, we'll explain the conceptual difference.
            return (
                f"**Recursive Approach to Matrix Chain Multiplication:**\n"
                f"The problem `MCM(i, j)` (finding the optimal cost to multiply matrices A_i through A_j) can be recursively defined. For each possible split point `k` between `i` and `j`, we find the minimum cost by adding the costs of the two subproblems (`MCM(i, k)` and `MCM(k+1, j)`) plus the cost of multiplying the two resulting matrices.\n"
                f"Base Case: If `i == j` (single matrix), the cost is 0.\n"
                f"Recurrence: `MCM(i, j) = min_{{i <= k < j}} (MCM(i, k) + MCM(k+1, j) + P[i-1]*P[k]*P[j])`\n"
                f"Time Complexity: O(3^n) or O(4^n) (exponential) due to redundant calculations of overlapping subproblems if not memoized.\n\n"
                f"**Comparison with Dynamic Programming:**\n"
                f"Dynamic programming (DP) improves upon the recursive approach by solving the problem in a bottom-up fashion (tabulation) or top-down with memoization. Both DP methods eliminate redundant calculations.\n"
                f"| Feature             | Recursive (Naive)                  | Dynamic Programming (Tabulation/Memoization) |\n"
                f"|---------------------|------------------------------------|--------------------------------------------|\n"
                f"| **Time Complexity** | O(3^n) (exponential) | O(n^3) (polynomial)         |\n"
                f"| **Redundancy** | Computes overlapping subproblems repeatedly | Solves each subproblem only once |\n"
                f"| **Memory** | O(n) (recursion stack)             | O(n^2) (DP table)         |\n"
                f"| **Approach** | Top-down, divide and conquer       | Bottom-up (tabulation) or top-down with memoization |\n\n"
                f"DP is significantly more efficient because MCM exhibits both optimal substructure and overlapping subproblems, making it a classic DP candidate."
            )

        # 7. Given dimensions {{dims}}, simulate how the DP table is filled for chain length 3.
        elif "simulate how the dp table is filled for chain length 3" in q:
            match = re.search(r"dimensions (.+), simulate how the dp table is filled for chain length 3", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 3:
                    return "Error: Dimensions must represent at least 3 matrices (for chain length 3)."
                
                m = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
                s = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
                
                sim_log = []
                sim_log.append(f"Simulating DP table fill for dimensions {dims_str} (P={p}) for chain length 3:")

                # Calculate for l=2 (chain length 2) first as prerequisites
                sim_log.append("\n--- First, calculate for chain length l = 2 (A_i to A_{i+1}) ---")
                for i in range(1, n):
                    j = i + 1
                    cost = scalar_mult_cost(p[i-1], p[i], p[j])
                    m[i][j] = cost
                    s[i][j] = i
                    sim_log.append(f"  m[{i}][{j}] (A{i} to A{j}) = p[{i-1}]*p[{i}]*p[{j}] = {p[i-1]}*{p[i]}*{p[j]} = {cost}")

                # Now, specifically for l=3
                l = 3
                sim_log.append(f"\n--- Calculating for chain length l = {l} ---")
                for i in range(1, n - l + 2):
                    j = i + l - 1
                    m[i][j] = float('inf')
                    
                    sim_log.append(f"Calculating m[{i}][{j}] (for matrices A{i} to A{j}):")
                    for k in range(i, j): # k is the split point
                        cost = m[i][k] + m[k+1][j] + scalar_mult_cost(p[i-1], p[k], p[j])
                        sim_log.append(f"  Split at k={k}: (A{i}..A{k}) x (A{k+1}..A{j})")
                        sim_log.append(f"  Cost = m[{i}][{k}]({m[i][k]}) + m[{k+1}][{j}]({m[k+1][j]}) + p[{i-1}]({p[i-1]})*p[{k}]({p[k]})*p[{j}]({p[j]}) = {cost}")
                        if cost < m[i][j]:
                            m[i][j] = cost
                            s[i][j] = k
                            sim_log.append(f"  New minimum found! m[{i}][{j}] = {cost}, s[{i}][{j}] = {k}")
                        else:
                            sim_log.append(f"  Cost {cost} is not better than current m[{i}][{j}] ({m[i][j]})")
                    sim_log.append(f"Final m[{i}][{j}] = {m[i][j]}")

                final_m_table_str = print_matrix_table(m, "Intermediate Cost Table (m) after l=3 calculation", n)
                
                return (
                    f"Simulation steps:\n"
                    + "\n".join(sim_log) + "\n\n"
                    + final_m_table_str
                )

        # 8. Construct a partial DP table with dimensions {{dims}} and explain cell m[2][4].
        elif "construct a partial dp table with dimensions" in q and "explain cell m[2][4]" in q:
            match = re.search(r"dimensions (.+)\s*and explain cell m\[2\]\[4\]", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 4: # Need at least 4 matrices for m[2][4] (A2 to A4)
                    return "Error: Dimensions must represent at least 4 matrices for cell m[2][4]."

                # Compute full DP table to accurately explain m[2][4]
                m, s, _ = matrix_chain_order_dp(p)

                explanation_log = []
                explanation_log.append(f"Constructing partial DP table for dimensions {dims_str} (P={p}) and explaining m[2][4]:")
                explanation_log.append(f"\nCell m[2][4] represents the minimum scalar multiplications needed to compute the product of matrices A2 through A4 (A2 x A3 x A4).")
                explanation_log.append(f"Its value is calculated by considering all possible split points k between i=2 and j=4 (i.e., k=2 or k=3).")
                
                explanation_log.append(f"\nCalculation for m[2][4]:")
                for k in range(2, 4): # k=2 or k=3
                    cost = m[2][k] + m[k+1][4] + scalar_mult_cost(p[2-1], p[k], p[4]) #
                    explanation_log.append(f"  If split at k={k}: (A2..A{k}) x (A{k+1}..A4)")
                    explanation_log.append(f"  Cost = m[2][{k}]({m[2][k]}) + m[{k+1}][4]({m[k+1][4]}) + p[{2-1}]({p[2-1]})*p[{k}]({p[k]})*p[4]({p[4]}) = {cost}")
                
                explanation_log.append(f"\nFinal m[2][4] = {m[2][4]} (optimal split at s[2][4]={s[2][4]})")

                # Show a partial table (e.g., up to n)
                partial_m_table_str = print_matrix_table(m, "Complete Cost Table (m)", n)
                
                return "\n".join(explanation_log) + "\n\n" + partial_m_table_str

        # 9. Explain why matrix chain multiplication solves overlapping subproblems using {{dims}}.
        elif "explain why matrix chain multiplication solves overlapping subproblems" in q:
            match = re.search(r"dimensions (.+)\.", q)
            dims_str = match.group(1) if match else "sample_dims"
            return (
                "Matrix Chain Multiplication solves **overlapping subproblems** because the optimal solution to a larger problem reuses optimal solutions to smaller, identical subproblems multiple times.\n"
                f"Consider a chain of matrices A1 x A2 x A3 x A4 with dimensions {dims_str} (P={parse_dims_string(dims_str)}):\n"
                f"- To find the optimal cost for A1..A4, we consider splits at k=1, 2, 3.\n"
                f"- Split at k=1: (A1) x (A2A3A4). Requires optimal costs for A2..A4.\n"
                f"- Split at k=2: (A1A2) x (A3A4). Requires optimal costs for A1..A2 and A3..A4.\n"
                f"- Split at k=3: (A1A2A3) x (A4). Requires optimal costs for A1..A3.\n"
                f"Notice that calculating A1..A4 might need A1..A3, and A1..A3 in turn needs A1..A2. If we just use simple recursion, `A1..A2` would be computed many times when finding the optimal `A1..A3`, `A1..A4`, and so on. Dynamic programming explicitly stores the solutions to these `m[i][j]` subproblems to avoid recalculation."
            )
        
        # 10. How does the cost table for dimensions {{dims}} avoid redundant calculations?
        elif "how does the cost table for dimensions" in q and "avoid redundant calculations" in q:
            match = re.search(r"dimensions (.+)\s*avoid redundant calculations", q)
            dims_str = match.group(1) if match else "sample_dims"
            return (
                f"The cost table (m[i][j]) in Matrix Chain Multiplication avoids redundant calculations by applying the principle of **dynamic programming (specifically, memoization or tabulation)**.\n"
                f"Each cell `m[i][j]` stores the minimum scalar multiplication cost for the subproblem of multiplying matrices `A_i` through `A_j`.\n"
                f"When calculating `m[i][j]`, the algorithm looks up the costs of smaller subproblems (like `m[i][k]` and `m[k+1][j]`) directly from the table, rather than recomputing them. Since each `m[i][j]` value is computed only once and stored, any future need for that subproblem's solution is met by a simple O(1) lookup, thus preventing exponential redundant calculations."
            )

        # 11. Find m[1][4] and its split point for matrix dimensions {{dims}} using DP.
        elif "find m[1][4] and its split point for matrix dimensions" in q:
            match = re.search(r"dimensions (.+)\s*using dp", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 4: # Need at least 4 matrices for m[1][4]
                    return "Error: Dimensions must represent at least 4 matrices to calculate m[1][4]."
                
                m, s, _ = matrix_chain_order_dp(p) #

                m_1_4 = m[1][4] #
                s_1_4 = s[1][4] #

                return (
                    f"For matrix dimensions {dims_str} (P={p}) using Dynamic Programming:\n"
                    f"m[1][4] represents the minimum cost to multiply matrices A1 to A4.\n"
                    f"m[1][4] = **{m_1_4}**.\n"
                    f"The optimal split point (k) for m[1][4] is: **{s_1_4}**."
                )

        # 12. Given matrix sizes {{dims}}, list subproblems solved while computing optimal cost.
        elif "list subproblems solved while computing optimal cost" in q:
            match = re.search(r"matrix sizes (.+), list subproblems solved while computing optimal cost", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1  # Number of matrices

                if n < 1:
                    return "⚠️ Error: Invalid input. You need at least one matrix to perform multiplication."

                # 🧮 Step-by-step: Build the list of subproblems
                subproblems = []
                for length in range(1, n + 1):  # length = chain length
                    for i in range(1, n - length + 2):
                        j = i + length - 1
                        if length == 1:
                            subproblems.append(f"m[{i}][{j}] (A{i}) = 0  ➜ Single matrix, no multiplication")
                        else:
                            subproblems.append(f"m[{i}][{j}] (A{i} to A{j})  ➜ Multiply chain A{i}...A{j}")

                return (
                    f"📘 **Matrix Chain Multiplication Subproblems**\n\n"
                    f"🧾 **Given dimensions:** {dims_str} → Interpreted as P = {p}\n"
                    f"This means you have {n} matrices: A1 to A{n}\n\n"
                    f"💡 **What are subproblems?**\n"
                    f"In MCM, we build a DP table `m[i][j]` that stores the minimum cost to multiply matrices A_i through A_j.\n"
                    f"These subproblems are solved in increasing order of chain length.\n\n"
                    f"🧠 **Subproblems solved:**\n"
                    f"{chr(10).join(subproblems)}"
                )
                
        # 13. Use tabulation to find optimal order of A1 to A4 with dimensions {{dims}}.
        elif "use tabulation to find optimal order of a1 to a4" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 4: # Need at least 4 matrices for A1 to A4
                    return "Error: Dimensions must represent at least 4 matrices."
                
                m, s, min_cost = matrix_chain_order_dp(p) #
                optimal_order_str = print_optimal_parenthesization(s, 1, n) # Reconstruct from s table

                return (
                    f"Using tabulation (bottom-up dynamic programming) for optimal order of A1 to A4 with dimensions {dims_str} (P={p}):\n"
                    f"The minimum multiplication cost is: **{min_cost}**.\n"
                    f"The optimal parenthesization (multiplication order) is: **{optimal_order_str}**."
                )

        # 14. Fill in DP values m[i][i+1] and m[i][i+2] for matrix dimensions {{dims}}.
        elif "fill in dp values m[i][i+1] and m[i][i+2]" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 2:
                    return "Error: Dimensions must represent at least 2 matrices to fill m[i][i+1]."
                
                m, _, _ = matrix_chain_order_dp(p) # Get filled DP table
                
                fill_log = []
                fill_log.append(f"For matrix dimensions {dims_str} (P={p}):")
                fill_log.append("\n**Values for m[i][i+1] (chain length 2):**")
                for i in range(1, n):
                    j = i + 1
                    cost = scalar_mult_cost(p[i-1], p[i], p[j])
                    fill_log.append(f"  m[{i}][{j}] (A{i} x A{j}) = p[{i-1}]*p[{i}]*p[{j}] = {p[i-1]}*{p[i]}*{p[j]} = {cost}")
                
                if n >= 3:
                    fill_log.append("\n**Values for m[i][i+2] (chain length 3):**")
                    for i in range(1, n - 1):
                        j = i + 2
                        # Manually calculate min cost for l=3 (i.e., m[i][i+2])
                        cost_option1 = m[i][i] + m[i+1][i+2] + scalar_mult_cost(p[i-1], p[i], p[i+2])
                        cost_option2 = m[i][i+1] + m[i+2][i+2] + scalar_mult_cost(p[i-1], p[i+1], p[i+2])
                        
                        m_i_iplus2_val = min(cost_option1, cost_option2)
                        fill_log.append(f"  m[{i}][{j}] (A{i} x A{i+1} x A{i+2}) = min( ({m[i][i]} + {m[i+1][i+2]} + {p[i-1]}*{p[i]}*{p[i+2]}), ({m[i][i+1]} + {m[i+2][i+2]} + {p[i-1]}*{p[i+1]}*{p[i+2]}) ) = {m_i_iplus2_val}")
                else:
                    fill_log.append("\nNo m[i][i+2] values as chain is too short.")

                return "\n".join(fill_log)

        # 15. What is the benefit of filling diagonals in the DP table for matrix dimensions {{dims}}?
        # elif "benefit of filling diagonals in the dp table" in q:
        #     match = re.search(r"dimensions (.+)\?", q)
        #     dims_str = match.group(1) if match else "sample_dims"
        #     return (
        #         f"The dynamic programming approach for Matrix Chain Multiplication fills the DP table (m[i][j]) **diagonally**.\n"
        #         f"**Benefit:** This diagonal filling order corresponds to processing subproblems in increasing order of their **chain length**.\n"
        #         f"1.  **Chain Length 1 (main diagonal):** `m[i][i]` is 0 (cost of multiplying a single matrix). These are the base cases.\n"
        #         f"2.  **Chain Length 2 (diagonal above main):** `m[i][i+1]` is `P[i-1]*P[i]*P[i+1]` (cost of multiplying two matrices A_i x A_{i+1}). These calculations only depend on base cases.\n"
        #         f"3.  **Increasing Chain Lengths:** When computing `m[i][j]` (for a chain of length `l = j - i + 1`), its value depends on `m[i][k]` and `m[k+1][j]`. Since `k` is between `i` and `j`, both `m[i][k]` and `m[k+1][j]` represent costs of shorter chains (smaller 'l' values). By filling diagonally, all necessary shorter chain costs are guaranteed to be already computed and available in the table, satisfying the optimal substructure property.\n"
        #         f"This systematic approach ensures that every subproblem is solved exactly once and in the correct order of dependency, avoiding redundant calculations and achieving O(n^3) time complexity."
        #     )

        # 16. Write a function to construct both the cost and split table for matrix chain {{dims}}.
        elif "write a function to construct both the cost and split table for matrix chain" in q:
            match = re.search(r"matrix chain (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p_example = parse_dims_string(dims_str)
                
                # Reuse global DP solver which returns m and s tables
                m_table, s_table, min_cost_example = matrix_chain_order_dp(p_example) #
                n_matrices = len(p_example) - 1

                m_table_str = print_matrix_table(m_table, "Cost Table (m)", n_matrices)
                s_table_str = print_matrix_table(s_table, "Split Table (s)", n_matrices)

                return (
                    f"Here's a function to construct both the cost (m) and split (s) tables for a matrix chain with dimensions {dims_str} (P={p_example}):\n\n"
                    "```python\n"
                    "def scalar_mult_cost(p_i, p_k, p_j):\n"
                    "    return p_i * p_k * p_j\n\n"
                    "def matrix_chain_order_dp(p_dims):\n"
                    "    n = len(p_dims) - 1\n"
                    "    m = [[0 for _ in range(n + 1)] for _ in range(n + 1)] # Cost table\n"
                    "    s = [[0 for _ in range(n + 1)] for _ in range(n + 1)] # Split table\n"
                    "    for l in range(2, n + 1):\n"
                    "        for i in range(1, n - l + 2):\n"
                    "            j = i + l - 1\n"
                    "            m[i][j] = float('inf')\n"
                    "            for k in range(i, j):\n"
                    "                cost = m[i][k] + m[k+1][j] + scalar_mult_cost(p_dims[i-1], p_dims[k], p_dims[j])\n"
                    "                if cost < m[i][j]:\n"
                    "                    m[i][j] = cost\n"
                    "                    s[i][j] = k\n"
                    "    return m, s, m[1][n]\n"
                    "```\n\n"
                    f"The computed tables are:\n"
                    f"{m_table_str}\n"
                    f"{s_table_str}\n"
                    f"Minimum cost: **{min_cost_example}**."
                )

        # 17. Explain how to reconstruct the optimal parenthesization using the split table for dimensions {{dims}}.
        elif "explain how to reconstruct the optimal parenthesization using the split table" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 1:
                    return "Error: Invalid dimensions. Need at least one matrix."
                
                m, s, _ = matrix_chain_order_dp(p) # Compute s table

                optimal_order_str = print_optimal_parenthesization(s, 1, n)

                return (
                    f"To reconstruct the optimal parenthesization using the split table `s` for dimensions {dims_str} (P={p}):\n"
                    f"The `s[i][j]` entry stores the index `k` where the optimal split occurs for the matrix chain `A_i ... A_j`.\n"
                    f"1.  Start with the full chain `A1 ... A_n` (i.e., `s[1][n]`). The value `s[1][n]` gives the primary split point `k`.\n"
                    f"2.  This means the optimal multiplication is `(A1 ... A_k) x (A_k+1 ... A_n)`.\n"
                    f"3.  Recursively apply this process to the two sub-chains: `(A1 ... A_k)` and `(A_k+1 ... A_n)`, using `s[1][k]` and `s[k+1][n]` respectively.\n"
                    f"4.  The recursion stops when `i == j`, indicating a single matrix `A_i` which needs no further parenthesization.\n"
                    f"**Example Reconstruction (for the given dimensions):**\n"
                    f"Optimal parenthesization: **{optimal_order_str}**."
                )

        # 18. Compare time complexity of recursive vs DP solution for matrix chain {{dims}}.
        elif "compare time complexity of recursive vs dp solution for matrix chain" in q:
            match = re.search(r"dimensions (.+)\.", q)
            dims_str = match.group(1) if match else "sample_dims"
            return (
                f"Comparison of time complexity for Matrix Chain Multiplication (MCM) for dimensions {dims_str}:\n"
                f"* **Recursive (Naive) Solution:** O(3^n) or O(4^n) (exponential). This approach is highly inefficient due to its redundant computation of overlapping subproblems, leading to many repeated calculations for the same sub-chains.\n"
                f"* **Dynamic Programming (DP) Solution (Tabulation or Memoization):** O(n^3) (polynomial). This approach solves each subproblem only once and stores its result. The three nested loops (for chain length `l`, start index `i`, and split point `k`) result in cubic time complexity. This is significantly faster than the naive recursive solution for larger 'n' values.\n"
                f"For example, for `n=10`, recursive might take millions of operations, while DP takes around 10^3 = 1000 operations (ignoring constants)."
            )

        # 19. What is the cost difference between recursive and DP approach on matrix chain {{dims}}?
        elif "cost difference between recursive and dp approach on matrix chain" in q:
            match = re.search(r"dimensions (.+)\?", q)
            if match:
                dims_str = match.group(1).strip().rstrip('.')
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n > 15:
                    return (
                        f"⚠️ For matrix chain dimensions {dims_str}:\n"
                        f"The cost difference between recursion and DP becomes too large to compute directly for N > 15.\n"
                        f"Naive recursion has exponential complexity O(3^N), while DP runs in polynomial time O(N³).\n"
                        f"This means DP is exponentially faster and avoids millions or billions of redundant calculations."
                    )

                memo = {}
                def recursive_cost(p, i, j):
                    if i == j:
                        return 0, 1
                    if (i, j) in memo:
                        return memo[(i, j)]
                    min_cost = float('inf')
                    calls = 1
                    for k in range(i, j):
                        cost1, calls1 = recursive_cost(p, i, k)
                        cost2, calls2 = recursive_cost(p, k+1, j)
                        total = cost1 + cost2 + scalar_mult_cost(p[i-1], p[k], p[j])
                        if total < min_cost:
                            min_cost = total
                        calls += calls1 + calls2
                    memo[(i, j)] = (min_cost, calls)
                    return memo[(i, j)]

                _, total_calls = recursive_cost(p, 1, n)
                _, _, dp_cost = matrix_chain_order_dp(p)

                return (
                    f"🧮 **Cost Comparison: Recursive vs DP Approach**\n\n"
                    f"📘 Dimensions: {dims_str} → Interpreted as P = {p}\n"
                    f"📏 Number of matrices: {n}\n\n"
                    f"🌀 **Recursive approach (with memo):**\n"
                    f"- Estimated function calls: **{total_calls}**\n"
                    f"- Time Complexity: ~O(3^N) (inefficient)\n\n"
                    f"✅ **DP approach:**\n"
                    f"- Scalar multiplication cost: **{dp_cost}**\n"
                    f"- Time Complexity: O(N³)\n\n"
                    f"📊 **Insight:**\n"
                    f"Both methods find the same optimal scalar multiplication cost, but recursion does it with **far more redundant work**.\n"
                    f"This highlights the power of dynamic programming in solving overlapping subproblems efficiently."
                )

        # 20. Demonstrate the reduction in scalar operations using DP for matrix chain {{dims}}.
        elif "demonstrate the reduction in scalar operations using dp for matrix chain" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 3: # Need at least 3 matrices to show reduction beyond trivial
                    return "Error: Dimensions must represent at least 3 matrices to demonstrate reduction."

                # Calculate optimal cost using DP
                m, s, dp_min_cost = matrix_chain_order_dp(p) #

                # Calculate costs for all possible naive parenthesizations for small n
                if n == 3: # For 3 matrices, there are only 2 ways
                    cost1 = scalar_mult_cost(p[0], p[1], p[2]) + scalar_mult_cost(p[0], p[2], p[3])
                    cost2 = scalar_mult_cost(p[1], p[2], p[3]) + scalar_mult_cost(p[0], p[1], p[3])
                    naive_max_cost = max(cost1, cost2) # Max of the two naive options
                    # If DP finds one option, and other is worse, reduction is naive_max_cost - dp_min_cost.
                    reduction = naive_max_cost - dp_min_cost
                    reduction_info = f"Max of 2 naive parenthesizations ({cost1}, {cost2}) is {naive_max_cost}. "
                    reduction_info += f"Reduction: {naive_max_cost} - {dp_min_cost} = **{reduction}**."

                else: # For n > 3, brute-forcing all parenthesizations is exponential and complex.
                    # We describe the general case based on time complexity.
                    reduction_info = "For chains larger than 3 matrices, brute-forcing all parenthesizations (which is exponential) is computationally infeasible.\n"
                    reduction_info += "DP's reduction is from an **exponential number of scalar operations in a brute-force approach** to a **polynomial O(N^3) number of scalar operations**.\n"
                    reduction_info += f"For this chain, the optimal cost is **{dp_min_cost}**. This represents the minimum possible multiplications given the dimensions, a value found efficiently by DP."
                
                return (
                    f"For matrix chain dimensions {dims_str} (P={p}), demonstrating the reduction in scalar operations using Dynamic Programming (DP):\n"
                    f"DP ensures that the optimal order is found by systematically evaluating each subproblem once, avoiding redundant calculations.\n"
                    f"The minimum scalar multiplication cost found by DP is: **{dp_min_cost}**.\n\n"
                    f"**Reduction Explanation:**\n"
                    f"{reduction_info}\n"
                    f"The primary benefit of DP is not always a direct numerical 'reduction' in total operations (as sometimes the naive best is chosen), but rather the **guarantee of finding the absolute minimum cost in polynomial time (O(N^3))** when a brute-force check would be exponential."
                )

    elif level == "Level 3":
        # 1. Build a function that returns both cost and optimal order of multiplication for {{dims}} using memoization.
        if "returns both cost and optimal order of multiplication" in q and "using memoization" in q:
            match = re.search(r"multiplication for (.+) using memoization", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                memo_cost = {} # Stores (i, j) -> min_cost
                memo_split = {} # Stores (i, j) -> k_split

                def mcm_memoized(i, j):
                    if i == j:
                        return 0 # Base case
                    if (i, j) in memo_cost: # Check memoa
                        return memo_cost[(i, j)]
                    
                    min_q = float('inf')
                    best_k = -1

                    for k in range(i, j):
                        cost = mcm_memoized(i, k) + mcm_memoized(k + 1, j) + scalar_mult_cost(p[i-1], p[k], p[j])
                        if cost < min_q:
                            min_q = cost
                            best_k = k
                    
                    memo_cost[(i, j)] = min_q # Store cost
                    memo_split[(i, j)] = best_k # Store split point
                    return min_q

                optimal_cost = mcm_memoized(1, n)
                optimal_order_str = print_optimal_parenthesization(memo_split, 1, n)

                return (
                    f"To compute both the minimum cost and optimal multiplication order for dimensions {dims_str} (P={p}) using memoization (top-down dynamic programming):\n\n"
                    f"This approach uses a dictionary (or 2D array) to store the results of subproblems (both cost and split point) as they are computed. This avoids redundant calculations, turning the exponential recursive calls into a polynomial-time solution (O(n^3)).\n"
                    "```python\n"
                    "memo_cost = {}\n"
                    "memo_split = {}\n"
                    "\n"
                    "def scalar_mult_cost(p_i, p_k, p_j):\n"
                    "    return p_i * p_k * p_j\n\n"
                    "def mcm_memoized(i, j, p_dims):\n"
                    "    if i == j:\n"
                    "        return 0\n"
                    "    if (i, j) in memo_cost:\n"
                    "        return memo_cost[(i, j)]\n"
                    "    \n"
                    "    min_q = float('inf')\n"
                    "    best_k = -1\n"
                    "\n"
                    "    for k in range(i, j):\n"
                    "        cost = mcm_memoized(i, k, p_dims) + \\\n"
                    "               mcm_memoized(k + 1, j, p_dims) + \\\n"
                    "               scalar_mult_cost(p_dims[i-1], p_dims[k], p_dims[j])\n"
                    "        if cost < min_q:\n"
                    "            min_q = cost\n"
                    "            best_k = k\n"
                    "    \n"
                    "    memo_cost[(i, j)] = min_q\n"
                    "    memo_split[(i, j)] = best_k\n"
                    "    return min_q\n\n"
                    "def print_optimal_parenthesization(s_table, i, j):\n"
                    "    # ... (helper as defined globally) ...\n"
                    "    pass\n"
                    "```\n\n"
                    f"Minimum cost: **{optimal_cost}**.\n"
                    f"Optimal multiplication order: **{optimal_order_str}**."
                )

        # 2. Track and print full DP and split tables for matrix chain {{dims}}.
        elif "track and print full dp and split tables for matrix chain" in q:
            match = re.search(r"matrix chain (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                m, s, min_cost = matrix_chain_order_dp(p) # Get filled tables

                m_table_output = print_matrix_table(m, "Cost Table (m)", n)
                s_table_output = print_matrix_table(s, "Split Table (s)", n)

                return (
                    f"For matrix chain {dims_str} (P={p}), the full DP cost (m) and split (s) tables are:\n\n"
                    f"{m_table_output}\n"
                    f"{s_table_output}\n"
                    f"The overall minimum cost is **{min_cost}**."
                )
        
        # 3. Write a top-down memoized matrix chain solver for dimensions {{dims}} and print recursion trace.
        elif "write a top-down memoized matrix chain solver" in q and "print recursion trace" in q:
            match = re.search(r"dimensions (.+) and print recursion trace", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                memo_cost_trace = {}
                memo_split_trace = {}
                trace_log = []

                def mcm_memoized_trace(i, j, level=0):
                    indent = "  " * level
                    trace_log.append(f"{indent}Calling mcm_memoized_trace({i}, {j})")
                    if i == j:
                        trace_log.append(f"{indent}Base Case: i == j ({i}), returning 0.")
                        return 0
                    if (i, j) in memo_cost_trace:
                        trace_log.append(f"{indent}Memoized: Found ({i}, {j}) in cache, returning {memo_cost_trace[(i,j)]}.")
                        return memo_cost_trace[(i, j)]
                    
                    min_q = float('inf')
                    best_k = -1

                    trace_log.append(f"{indent}Calculating for A{i}..A{j}:")
                    for k in range(i, j):
                        trace_log.append(f"{indent}  Considering split at k={k}: (A{i}..A{k}) x (A{k+1}..A{j})")
                        cost_left = mcm_memoized_trace(i, k, level + 1)
                        cost_right = mcm_memoized_trace(k + 1, j, level + 1)
                        current_cost = cost_left + cost_right + scalar_mult_cost(p[i-1], p[k], p[j])
                        trace_log.append(f"{indent}  Subproblem costs for k={k}: left={cost_left}, right={cost_right}, mult_cost={scalar_mult_cost(p[i-1], p[k], p[j])}. Total={current_cost}")
                        
                        if current_cost < min_q:
                            min_q = current_cost
                            best_k = k
                            trace_log.append(f"{indent}  New minimum for A{i}..A{j} found: {min_q} (split k={best_k})")
                    
                    memo_cost_trace[(i, j)] = min_q
                    memo_split_trace[(i, j)] = best_k
                    trace_log.append(f"{indent}Storing m[{i}][{j}]={min_q}, s[{i}][{j}]={best_k}. Returning {min_q}.")
                    return min_q

                optimal_cost = mcm_memoized_trace(1, n)
                optimal_order_str = print_optimal_parenthesization(memo_split_trace, 1, n)

                return (
                    f"Top-down memoized matrix chain solver for dimensions {dims_str} (P={p}) with recursion trace:\n\n"
                    "The trace shows how memoization avoids recomputing subproblems:\n"
                    + "\n".join(trace_log) + "\n\n"
                    f"Minimum cost: **{optimal_cost}**.\n"
                    f"Optimal multiplication order: **{optimal_order_str}**."
                )

        # 4. Explain overlapping subproblems in matrix chain multiplication with {{dims}} and how memoization helps.
        elif "explain overlapping subproblems in matrix chain multiplication" in q and "how memoization helps" in q:
            match = re.search(r"dimensions (.+) and how memoization helps", q)
            dims_str = match.group(1) if match else "sample_dims"
            return (
                f"**Overlapping Subproblems in Matrix Chain Multiplication (MCM):**\n"
                f"MCM exhibits overlapping subproblems because the recursive definition repeatedly asks for the solution to the same subproblems.\n"
                f"Consider finding the optimal parenthesization for A1 x A2 x A3 x A4 x A5 (dimensions like {dims_str}):\n"
                f"- To find optimal (A1..A5), we might consider splits like (A1..A2)x(A3..A5) and (A1..A3)x(A4..A5).\n"
                f"- To compute (A1..A3), we might need (A1)x(A2A3) and (A1A2)x(A3).\n"
                f"- Both (A1..A4) and (A1..A5) might need the optimal solution for (A1..A2) (if they are both computed via splits that include this subproblem).\n"
                f"This means the same subproblem, like `m[1][2]` (cost of A1 x A2), can be needed multiple times by different larger problems. Without optimization, these are recomputed many times, leading to exponential time complexity.\n\n"
                f"**How Memoization Helps:**\n"
                f"Memoization (a top-down dynamic programming technique) addresses this by **storing the results of each subproblem (`m[i][j]`) in a cache (e.g., a dictionary or 2D array) as soon as they are computed**.\n"
                f"Before performing a recursive computation for `m[i][j]`, the algorithm first checks if the result is already in the cache. If it is, the cached value is returned directly. If not, it computes the value, stores it, and then returns it. This ensures that each unique subproblem is computed only once, reducing the time complexity from exponential to polynomial (O(n^3))."
            )

        # 5. For a matrix chain of n = {{n}}, generate dimensions and compute minimum cost using bottom-up DP.
        elif "generate dimensions and compute minimum cost using bottom-up dp" in q:
            match = re.search(r"n = (\d+)", q)
            if match:
                n = int(match.group(1))
                
                if n < 1: return "Error: Number of matrices must be at least 1."

                # Generate random dimensions for n matrices
                # (n+1) dimensions are needed for n matrices
                import random
                generated_dims = [random.randint(10, 100) for _ in range(n + 1)]

                m, s, min_cost = matrix_chain_order_dp(generated_dims) #
                optimal_order_str = print_optimal_parenthesization(s, 1, n)

                return (
                    f"For a matrix chain of n = {n} matrices, with randomly generated dimensions (P={generated_dims}):\n\n"
                    f"Using bottom-up dynamic programming:\n"
                    f"Minimum multiplication cost: **{min_cost}**.\n"
                    f"Optimal multiplication order: **{optimal_order_str}**."
                )

        # 6. Develop a function to compute optimal parenthesization depth for dimensions {{dims}}.
        elif "develop a function to compute optimal parenthesization depth" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 1: return "Error: Invalid dimensions. Need at least one matrix."
                
                _, s, _ = matrix_chain_order_dp(p) # Get the split table

                # Recursive helper to find depth
                def get_parenthesization_depth(s_table, i, j):
                    if i == j:
                        return 0 # Depth for a single matrix is 0
                    k = s_table[i][j]
                    left_depth = get_parenthesization_depth(s_table, i, k)
                    right_depth = get_parenthesization_depth(s_table, k + 1, j)
                    return 1 + max(left_depth, right_depth) # Add 1 for the current multiplication level

                optimal_depth = get_parenthesization_depth(s, 1, n)

                return (
                    f"To compute the optimal parenthesization depth for dimensions {dims_str} (P={p}):\n\n"
                    f"The depth represents the number of nested multiplications. A function recursively traverses the optimal split table (s) to determine the maximum depth of parenthesization.\n"
                    "```python\n"
                    "def get_optimal_parenthesization_depth(s_table, i, j):\n"
                    "    if i == j:\n"
                    "        return 0 # Base case: single matrix, depth is 0\n"
                    "    k = s_table[i][j] # Optimal split point\n"
                    "    left_depth = get_optimal_parenthesization_depth(s_table, i, k)\n"
                    "    right_depth = get_optimal_parenthesization_depth(s_table, k + 1, j)\n"
                    "    return 1 + max(left_depth, right_depth) # Add 1 for the current multiplication\n"
                    "```\n\n"
                    f"The optimal parenthesization depth is: **{optimal_depth}**."
                )
        
        # 7. Print DP table after each chain length iteration for matrix chain {{dims}}.
        elif "print dp table after each chain length iteration" in q:
            match = re.search(r"matrix chain (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 1: return "Error: Invalid dimensions. Need at least one matrix."

                m = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
                s = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
                
                iteration_logs = []

                iteration_logs.append("Initial m[i][i] values (chain length 1 = 0 cost):")
                iteration_logs.append(print_matrix_table(m, "m (Chain Length 1)", n))

                for l in range(2, n + 1): # l is chain length
                    iteration_logs.append(f"\n--- After calculating for chain length l = {l} ---")
                    for i in range(1, n - l + 2):
                        j = i + l - 1
                        m[i][j] = float('inf')
                        for k in range(i, j):
                            cost = m[i][k] + m[k+1][j] + scalar_mult_cost(p[i-1], p[k], p[j])
                            if cost < m[i][j]:
                                m[i][j] = cost
                                s[i][j] = k
                    iteration_logs.append(print_matrix_table(m, f"m (After Chain Length {l} Calculation)", n))
                    # Optionally, also print s table for this iteration
                    # iteration_logs.append(print_matrix_table(s, f"s (After Chain Length {l} Calculation)", n))

                return (
                    f"For matrix chain dimensions {dims_str} (P={p}), printing DP table after each chain length iteration:\n"
                    + "\n".join(iteration_logs) + "\n\n"
                    f"Final minimum cost: **{m[1][n]}**."
                )

        # 8. Track total scalar multiplications avoided using DP on chain {{dims}}.
        elif "track total scalar multiplications avoided using dp" in q:
            match = re.search(r"matrix chain (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n <= 2: return "Error: Need at least 3 matrices to show significant avoidance."
                
                # Compute DP cost
                _, _, dp_min_cost = matrix_chain_order_dp(p) #

                # Estimate theoretical naive cost for comparison
                # This is a general estimate as exact naive cost involves huge recursion.
                # Number of distinct subproblems = n^2
                # Each subproblem takes O(n) splits. Total O(n^3) for DP.
                # Naive is exponential, ~Catalan number related.
                # For n matrices, the number of ways to parenthesize is C_{n-1} (Catalan number).
                # C_n = (1/(n+1)) * (2n choose n)
                # For n=3 (2 matrices), C_2 = 2 ways. (A1A2)A3, A1(A2A3).
                # For n=4 (3 matrices), C_3 = 5 ways.
                # For n=5 (4 matrices), C_4 = 14 ways.
                
                # The question asks about scalar multiplications *avoided*.
                # This means, if we picked the worst possible naive multiplication order, how much worse would it be?
                # Or, how many *computations of m[i][j] * k_splits* are avoided vs. the DP's actual calculations.
                
                # A more straightforward answer to "avoided" relates to complexity:
                # DP avoids recomputing m[i][j] repeatedly. Each cell takes O(N) to compute.
                # There are O(N^2) cells.
                
                # Let's illustrate with the performance comparison previously shown.
                # The "reduction in scalar operations" is the *difference* between the worst possible arbitrary parenthesization and the optimal one.
                
                # For general N, it's hard to define "total scalar multiplications avoided" without a specific naive method to compare against.
                # The main avoidance is in *function calls* or *re-evaluations*.
                # For the MCM problem, the number of distinct scalar multiplications performed *in total* is exactly the optimal cost.
                # The "avoided" part refers to avoiding redundant *recalculations* of subproblems.
                
                return (
                    f"For matrix chain dimensions {dims_str} (P={p}), tracking total scalar multiplications avoided using DP:\n"
                    f"The minimum scalar multiplication cost found by DP is: **{dp_min_cost}**.\n\n"
                    f"Dynamic Programming avoids redundant calculations (overlapping subproblems).\n"
                    f"Without DP, a naive recursive algorithm would recompute the optimal cost for the same sub-chains numerous times. For 'N' matrices, this could lead to a time complexity roughly proportional to O(4^N) or O(3^N) evaluations.\n"
                    f"DP ensures that each unique `m[i][j]` subproblem is computed only once. Instead of an exponential number of recomputations, DP performs approximately O(N^3) basic operations (additions and multiplications) to fill its table.\n"
                    f"Thus, the 'avoided multiplications' are not about different final costs, but about avoiding the *vast number of redundant calculations* that an unoptimized recursive solution would perform to arrive at the same optimal minimum cost."
                )

        # 9. Use dynamic programming to simulate matrix chain computation for {{dims}} and visualize each iteration.
        elif "use dynamic programming to simulate matrix chain computation" in q and "visualize each iteration" in q:
            match = re.search(r"dimensions (.+) and visualize each iteration", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 1: return "Error: Invalid dimensions. Need at least one matrix."

                m = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
                s = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
                
                sim_visual_log = []

                sim_visual_log.append("Initial m[i][i] values (chain length 1 = 0 cost):")
                sim_visual_log.append(print_matrix_table(m, "m (Chain Length 1)", n))

                for l in range(2, n + 1): # l is chain length
                    sim_visual_log.append(f"\n--- Iteration for chain length l = {l} ---")
                    for i in range(1, n - l + 2):
                        j = i + l - 1
                        m[i][j] = float('inf')
                        current_cell_log = []
                        current_cell_log.append(f"  Calculating m[{i}][{j}] (A{i} to A{j}):")
                        for k in range(i, j): # k is the split point
                            cost = m[i][k] + m[k+1][j] + scalar_mult_cost(p[i-1], p[k], p[j])
                            current_cell_log.append(f"    Split at k={k}: Cost = m[{i}][{k}]({m[i][k]}) + m[{k+1}][{j}]({m[k+1][j]}) + mult_cost = {cost}")
                            if cost < m[i][j]:
                                m[i][j] = cost
                                s[i][j] = k
                                current_cell_log.append(f"    New minimum for m[{i}][{j}] = {cost}, s[{i}][{j}] = {k}")
                        current_cell_log.append(f"  Final value for m[{i}][{j}] after considering all splits: {m[i][j]}")
                        sim_visual_log.append("\n".join(current_cell_log))
                    sim_visual_log.append(f"\nDP table state after chain length {l} iteration:")
                    sim_visual_log.append(print_matrix_table(m, f"m (After l={l})", n))
                    # sim_visual_log.append(print_matrix_table(s, f"s (After l={l})", n)) # Uncomment to also show split table

                return (
                    f"Simulation of Matrix Chain Multiplication DP for dimensions {dims_str} (P={p}), visualizing each iteration:\n"
                    + "\n".join(sim_visual_log) + "\n\n"
                    f"Final minimum cost: **{m[1][n]}**."
                )
        
        # 10. Find the most balanced split point in the matrix chain {{dims}} to reduce depth.
        elif "find the most balanced split point in the matrix chain" in q and "to reduce depth" in q:
            match = re.search(r"matrix chain (.+) to reduce depth", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 2: return "Error: Need at least 2 matrices for a split."

                _, s, _ = matrix_chain_order_dp(p) # Get optimal split table

                # The "most balanced split point to reduce depth" is not necessarily the 'optimal' split point for cost.
                # For depth, it's a split that makes the two subproblems of roughly equal size.
                # For A_i ... A_j, the size is (j-i+1). A balanced split k would make (k-i+1) approx (j-(k+1)+1).
                # (k-i+1) approx (j-k). So k-i+1 = j-k => 2k = i+j-1 => k = (i+j-1)/2
                # We need to find the split k for the *entire* chain (A1..An) that is closest to (1+n-1)/2 or n/2.
                
                midpoint_k = math.floor((1 + n) / 2) # Midpoint of matrix indices 1 to n

                # For optimal splits, we always use the 's' table.
                # However, the question asks for 'most balanced split point to reduce depth', which implies a different criterion.
                # This is not about the cheapest cost, but about tree balance.
                # The 's' table stores the optimal *cost* split.
                
                # Let's return the k that is closest to the middle of the chain.
                # For i..j, a balanced split k_bal = floor((i+j)/2).
                # For A1..An, it would be k_bal = floor((1+n)/2).
                
                balanced_k_val = math.floor((1 + n) / 2) # The k that would roughly split the chain in half

                # Compare this 'balanced k' to the MCM-optimal k (from s[1][n])
                mcm_optimal_k = s[1][n] #
                
                return (
                    f"For the matrix chain {dims_str} (P={p}):\n"
                    f"The overall chain is A1 to A{n}.\n"
                    f"A split point `k` determines the two sub-chains (A1..Ak) and (A_{{k+1}}..A{n}).\n"
                    f"To reduce the depth of the parenthesization tree (make it more balanced), the split point `k` should ideally be close to the middle of the chain.\n"
                    f"The most balanced split point would be `k` approximately: **{balanced_k_val}**.\n"
                    f"This contrasts with the cost-optimal split point (from dynamic programming) which is `s[1][{n}] = {mcm_optimal_k}`. The optimal split for cost might not necessarily be the most balanced for depth."
                )

        # 11. Extend the DP solution to track nested multiplication levels for matrix chain {{dims}}.
        elif "extend the dp solution to track nested multiplication levels" in q:
            match = re.search(r"matrix chain (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 1: return "Error: Invalid dimensions. Need at least one matrix."

                m, s, min_cost = matrix_chain_order_dp(p) # Get cost and split tables

                # Recursive function to get depth (nested levels)
                def get_nested_levels(s_table, i, j):
                    if i == j:
                        return 0 # Single matrix, 0 nested levels
                    k = s_table[i][j] # Optimal split point
                    left_level = get_nested_levels(s_table, i, k)
                    right_level = get_nested_levels(s_table, k + 1, j)
                    return 1 + max(left_level, right_level) # Add 1 for the current multiplication

                optimal_nested_levels = get_nested_levels(s, 1, n)

                return (
                    f"To extend the DP solution to track nested multiplication levels for matrix chain {dims_str} (P={p}):\n\n"
                    f"After computing the optimal split table `s[i][j]` using dynamic programming, a recursive helper function can be used to traverse `s` and determine the maximum nesting depth.\n"
                    "```python\n"
                    "def get_nested_levels(s_table, i, j):\n"
                    "    if i == j:\n"
                    "        return 0\n"
                    "    k = s_table[i][j]\n"
                    "    left_level = get_nested_levels(s_table, i, k)\n"
                    "    right_level = get_nested_levels(s_table, k + 1, j)\n"
                    "    return 1 + max(left_level, right_level)\n"
                    "```\n\n"
                    f"The minimum multiplication cost is: {min_cost}.\n"
                    f"The maximum nested multiplication levels (depth of optimal parenthesization) is: **{optimal_nested_levels}**."
                )

        # 12. Compare three different parenthesis orders and their costs for matrix dimensions {{dims}}.
        elif "compare three different parenthesis orders and their costs" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 3: # Need at least 3 matrices for 3 different parenthesizations
                    return "Error: Need at least 3 matrices (e.g., A1x...xA3) to compare multiple distinct parenthesizations beyond the basic two for N=3."
                
                m, s, min_cost = matrix_chain_order_dp(p) # Get optimal cost for reference

                comparison_details = []

                # Option 1: Standard left-to-right (A1 x A2) x ...
                cost_l_to_r = 0
                current_p = p[0]
                for i in range(1, n):
                    cost_l_to_r += current_p * p[i] * p[i+1]
                    current_p = current_p # Resulting rows remain original first matrix rows
                comparison_details.append({"order": "Left-to-Right: (((A1 x A2) x A3) ... x An)", "cost": cost_l_to_r})

                # Option 2: Standard right-to-left A1 x (A2 x (A3 ...))
                cost_r_to_l = 0
                current_p_col = p[n] # Last matrix columns
                for i in range(n-1, 0, -1):
                    cost_r_to_l += p[i-1] * p[i] * current_p_col
                    current_p_col = current_p_col # Resulting columns remain original last matrix columns
                comparison_details.append({"order": "Right-to-Left: (A1 x (...(An-1 x An)...))", "cost": cost_r_to_l})

                # Option 3: Optimal (from DP)
                optimal_order_str = print_optimal_parenthesization(s, 1, n)
                comparison_details.append({"order": f"Optimal (found by DP): {optimal_order_str}", "cost": min_cost})
                
                # If n=3, there are only 2 distinct parenthesizations, so Left-to-Right is one, Right-to-Left is the other.
                # If n > 3, there are many. We chose standard ones for easy comparison.
                
                return (
                    f"For matrix dimensions {dims_str} (P={p}), comparing three different parenthesization orders and their costs:\n"
                    f"The choice of parenthesization significantly impacts the total scalar multiplication cost.\n"
                    f"1.  **Left-to-Right Parenthesization:** `((A1 x A2) x A3) ... x An`\n"
                    f"    Cost = **{comparison_details[0]['cost']}**\n"
                    f"2.  **Right-to-Left Parenthesization:** `A1 x (A2 x (A3 ... x An))`\n"
                    f"    Cost = **{comparison_details[1]['cost']}**\n"
                    f"3.  **Optimal Parenthesization (Dynamic Programming):** `{optimal_order_str}`\n"
                    f"    Cost = **{comparison_details[2]['cost']}**\n\n"
                    f"The optimal cost of {min_cost} is found using DP and is guaranteed to be the lowest."
                )

        # 13. Estimate time complexity of bottom-up DP for n = {{n}} and dimensions {{dims}}.
        elif "estimate time complexity of bottom-up dp for n" in q and "dimensions" in q:
            match = re.search(r"n = (\d+) and dimensions (.+)\.", q)
            if match:
                n = int(match.group(1))
                dims_str = match.group(2)
                p_parsed = parse_dims_string(dims_str) # Just to show it's parsed
                
                return (
                    f"The time complexity of the bottom-up dynamic programming solution for Matrix Chain Multiplication for `n = {n}` matrices (with dimensions {dims_str}) is **O(n^3)**.\n"
                    f"This complexity arises from three nested loops in the algorithm:\n"
                    f"1.  The outermost loop iterates over the `chain length (l)` from 2 to `n`.\n"
                    f"2.  The next loop iterates over the `starting index (i)` of the chain.\n"
                    f"3.  The innermost loop iterates over the `split point (k)` within the current chain.\n"
                    f"Each iteration involves a constant number of arithmetic operations (additions and multiplications). Thus, the total operations are cubic with respect to the number of matrices `n`."
                )

        # 14. Given matrix chain {{dims}}, write a complete recursive, memoized, and tabulated version and compare them.
        elif "write a complete recursive, memoized, and tabulated version and compare them" in q:
            match = re.search(r"matrix chain (.+), write a complete recursive, memoized, and tabulated version and compare them", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                comparison_results = []

                # --- Naive Recursive ---
                def mcm_naive_recursive_compare(i, j):
                    if i == j: return 0
                    min_q = float('inf')
                    for k in range(i, j):
                        cost = mcm_naive_recursive_compare(i, k) + mcm_naive_recursive_compare(k + 1, j) + scalar_mult_cost(p[i-1], p[k], p[j])
                        min_q = min(min_q, cost)
                    return min_q
                
                start_time_naive = time.perf_counter()
                try:
                    # Limit N for naive recursion to avoid excessive runtime
                    if n > 15: # Heuristic limit
                        naive_cost = "Too large to compute (RecursionError/long runtime)"
                        naive_time = "N/A"
                    else:
                        naive_cost = mcm_naive_recursive_compare(1, n)
                        naive_time = (time.perf_counter() - start_time_naive) * 1000
                except RecursionError:
                    naive_cost = "RecursionError"
                    naive_time = "N/A"
                comparison_results.append({
                    "Method": "Naive Recursive", "Cost": naive_cost, "Time": naive_time,
                    "Complexity": "O(3^N) (Exponential)"
                })

                # --- Memoized (Top-Down DP) ---
                memo_mcm_compare = {}
                memo_s_compare = {}
                def mcm_memoized_compare(i, j):
                    if i == j: return 0
                    if (i, j) in memo_mcm_compare: return memo_mcm_compare[(i, j)]
                    min_q = float('inf')
                    best_k = -1
                    for k in range(i, j):
                        cost = mcm_memoized_compare(i, k) + mcm_memoized_compare(k + 1, j) + scalar_mult_cost(p[i-1], p[k], p[j])
                        if cost < min_q:
                            min_q = cost
                            best_k = k
                    memo_mcm_compare[(i, j)] = min_q
                    memo_s_compare[(i, j)] = best_k
                    return min_q
                
                start_time_memoized = time.perf_counter()
                memoized_cost = mcm_memoized_compare(1, n)
                memoized_time = (time.perf_counter() - start_time_memoized) * 1000
                comparison_results.append({
                    "Method": "Memoized (Top-Down DP)", "Cost": memoized_cost, "Time": memoized_time,
                    "Complexity": "O(N^3) (Polynomial)"
                })

                # --- Tabulated (Bottom-Up DP) ---
                start_time_tabulated = time.perf_counter()
                m_tab, s_tab, tabulated_cost = matrix_chain_order_dp(p)
                tabulated_time = (time.perf_counter() - start_time_tabulated) * 1000
                comparison_results.append({
                    "Method": "Tabulated (Bottom-Up DP)", "Cost": tabulated_cost, "Time": tabulated_time,
                    "Complexity": "O(N^3) (Polynomial)"
                })
                
                output_str = f"For matrix chain dimensions {dims_str} (P={p}), comparing different implementations:\n\n"
                output_str += "```python\n"
                output_str += "# Scalar Multiplication Cost Function (Common to all)\n"
                output_str += "def scalar_mult_cost(p_i, p_k, p_j): return p_i * p_k * p_j\n\n"
                output_str += "# 1. Naive Recursive Version\n"
                output_str += "def mcm_naive_recursive(p_dims, i, j):\n"
                output_str += "    if i == j: return 0\n"
                output_str += "    min_q = float('inf')\n"
                output_str += "    for k in range(i, j):\n"
                output_str += "        cost = mcm_naive_recursive(p_dims, i, k) + \\\n"
                output_str += "               mcm_naive_recursive(p_dims, k + 1, j) + \\\n"
                output_str += "               scalar_mult_cost(p_dims[i-1], p_dims[k], p_dims[j])\n"
                output_str += "        min_q = min(min_q, cost)\n"
                output_str += "    return min_q\n\n"
                output_str += "# 2. Memoized (Top-Down DP) Version\n"
                output_str += "memo_cost = {}\n"
                output_str += "memo_split = {}\n"
                output_str += "def mcm_memoized(p_dims, i, j):\n"
                output_str += "    if i == j: return 0\n"
                output_str += "    if (i, j) in memo_cost: return memo_cost[(i, j)]\n"
                output_str += "    min_q = float('inf')\n"
                output_str += "    best_k = -1\n"
                output_str += "    for k in range(i, j):\n"
                output_str += "        cost = mcm_memoized(p_dims, i, k) + \\\n"
                output_str += "               mcm_memoized(p_dims, k + 1, j) + \\\n"
                output_str += "               scalar_mult_cost(p_dims[i-1], p_dims[k], p_dims[j])\n"
                output_str += "        if cost < min_q:\n"
                output_str += "            min_q = cost\n"
                output_str += "            best_k = k\n"
                output_str += "    memo_cost[(i, j)] = min_q\n"
                output_str += "    memo_split[(i, j)] = best_k\n"
                output_str += "    return min_q\n\n"
                output_str += "# 3. Tabulated (Bottom-Up DP) Version (using a standard m, s table)\n"
                output_str += "def mcm_tabulated(p_dims):\n"
                output_str += "    n = len(p_dims) - 1\n"
                output_str += "    m = [[0 for _ in range(n + 1)] for _ in range(n + 1)]\n"
                output_str += "    s = [[0 for _ in range(n + 1)] for _ in range(n + 1)]\n"
                output_str += "    for l in range(2, n + 1):\n"
                output_str += "        for i in range(1, n - l + 2):\n"
                output_str += "            j = i + l - 1\n"
                output_str += "            m[i][j] = float('inf')\n"
                output_str += "            for k in range(i, j):\n"
                output_str += "                cost = m[i][k] + m[k+1][j] + scalar_mult_cost(p_dims[i-1], p_dims[k], p_dims[j])\n"
                output_str += "                if cost < m[i][j]:\n"
                output_str += "                    m[i][j] = cost\n"
                output_str += "                    s[i][j] = k\n"
                output_str += "    return m[1][n]\n"
                output_str += "```\n\n"
                
                output_str += "Performance Comparison:\n"
                output_str += "| Method                   | Cost  | Time (ms)    | Theoretical Complexity |\n"
                output_str += "|--------------------------|-------|--------------|------------------------|\n"
                for res in comparison_results:
                    time_val = f"{res['Time']:.4f}" if isinstance(res['Time'], float) else str(res['Time'])
                    output_str += f"| {res['Method']:<24} | {str(res['Cost']):<5} | {time_val:<12} | {res['Complexity']:<22} |\n"

                output_str += "\n**Observations:**\n"
                output_str += "* **Naive Recursive:** Suffers from exponential time complexity due to redundant computations and recursion depth issues, quickly becoming impractical for even small 'N'.\n"
                output_str += "* **Memoized (Top-Down DP):** Significantly improves performance by caching subproblem results, reducing time complexity to O(N^3). It still uses recursion, so stack depth can be an issue for very large N.\n"
                output_str += "* **Tabulated (Bottom-Up DP):** Achieves the same O(N^3) time complexity as memoized DP but avoids recursion overhead, making it generally more robust and slightly faster in practice for large N, as it avoids stack overflow."
                
                return output_str

        # 15. Demonstrate trade-offs between time and space for bottom-up MCM using dimensions {{dims}}.
        elif "demonstrate trade-offs between time and space for bottom-up mcm" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                # Time: O(N^3) is fixed for standard bottom-up
                # Space: O(N^2) for m and s tables
                m_size_bytes = (n + 1) * (n + 1) * 8 # Assuming 8 bytes per int (python ints can be larger)
                s_size_bytes = (n + 1) * (n + 1) * 8

                return (
                    f"For bottom-up Matrix Chain Multiplication with dimensions {dims_str} (P={p}):\n"
                    f"**Time Complexity:** O(N^3). This is determined by the three nested loops used to fill the DP table.\n"
                    f"**Space Complexity:** O(N^2). This is primarily due to the storage required for two `N x N` tables: the `m` table (for minimum costs) and the `s` table (for optimal split points).\n\n"
                    f"**Trade-offs:**\n"
                    f"* **Time vs. Space:** The O(N^2) space is a trade-off for achieving the optimal O(N^3) time complexity. By storing all intermediate subproblem results, redundant computations are avoided.\n"
                    f"* **Scalability:** For large `N`, `N^2` space can become significant. While `N^3` time is acceptable up to `N` around a few hundreds, `N^2` space can be a bottleneck for very large `N` if memory is constrained. This is why more advanced (and complex) space-optimized versions exist, but they might lose the ability to reconstruct the full parenthesization directly from the table."
                )

        # 16. Visualize the fill process of DP table diagonally for matrix chain {{dims}}.
        elif "visualize the fill process of dp table diagonally" in q:
            match = re.search(r"matrix chain (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 1: return "Error: Invalid dimensions. Need at least one matrix."

                m = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
                s = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
                
                visual_log = []

                visual_log.append("Initialization: Fill m[i][i] with 0 (cost of single matrix):")
                visual_log.append(print_matrix_table(m, "m (Chain Length 1)", n))

                for l in range(2, n + 1): # l is chain length
                    visual_log.append(f"\n--- Filling Diagonal for Chain Length l = {l} ---")
                    for i in range(1, n - l + 2):
                        j = i + l - 1
                        m[i][j] = float('inf') # Initialize before finding min
                        visual_log.append(f"Calculating m[{i}][{j}] (A{i} to A{j}):")
                        for k in range(i, j):
                            cost = m[i][k] + m[k+1][j] + scalar_mult_cost(p[i-1], p[k], p[j])
                            visual_log.append(f"  Considering split k={k}: Cost = {cost}")
                            if cost < m[i][j]:
                                m[i][j] = cost
                                s[i][j] = k
                                visual_log.append(f"  Updated m[{i}][{j}] to {m[i][j]}, split s[{i}][{j}] to {s[i][j]}")
                        visual_log.append(f"  Final m[{i}][{j}] = {m[i][j]} for chain A{i}..A{j}")
                    visual_log.append(f"Table state after completing diagonal for chain length {l}:")
                    visual_log.append(print_matrix_table(m, f"m (After Chain Length {l} Complete)", n))
                
                return (
                    f"Visualizing the diagonal fill process of the DP table for matrix chain {dims_str} (P={p}):\n"
                    f"The DP table is filled by increasing chain length `l`, which corresponds to filling diagonals starting from the main diagonal.\n"
                    + "\n".join(visual_log) + "\n\n"
                    f"Final minimum cost: **{m[1][n]}**."
                )

        # 17. Design a function that uses memory-efficient rolling arrays for MCM with dimensions {{dims}}.
        elif "design a function that uses memory-efficient rolling arrays for mcm" in q:
            match = re.search(r"dimensions (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 1: return "Error: Invalid dimensions. Need at least one matrix."

                # For 0/1 Knapsack, rolling array is O(W). For MCM, it's a bit harder.
                # Standard MCM DP m[i][j] depends on m[i][k] and m[k+1][j].
                # This doesn't simply map to two rows.
                # True space optimization for MCM is not simply rolling arrays like 0/1 Knapsack,
                # as it depends on values from the same diagonal and previous diagonals.
                # The "optimal substructure" for MCM refers to shorter chains, not just previous items.
                # Space optimization is usually O(N) by using a specialized divide and conquer (Hirschberg's algorithm variant)
                # or by storing only necessary splits.
                
                # However, a common simplification is using two arrays if only the *cost* is needed (not reconstruction of s).
                # But MCM is more complex as it depends on m[i][k] and m[k+1][j]
                
                # This question is difficult to implement a *true* memory-efficient rolling array
                # that is still O(N^2) or O(N) but better than O(N^2) of full table.
                # A standard O(N^2) DP solution IS already efficient in terms of its time complexity.
                # Let's describe the conceptual challenge and common simplification.

                return (
                    f"For Matrix Chain Multiplication with dimensions {dims_str} (P={p}), designing a function that uses memory-efficient rolling arrays is complex.\n"
                    f"**Standard MCM DP space complexity is O(N^2)** due to the need for the `m` (cost) and `s` (split) tables. Each `m[i][j]` computation relies on values from previous columns and rows/diagonals.\n"
                    f"Unlike problems like 0/1 Knapsack (where O(W) space is achieved by only needing the previous 'row'), MCM's dependencies are more intricate (`m[i][k]` and `m[k+1][j]`), preventing a direct `O(N)` row-by-row or column-by-column rolling array optimization for the full problem.\n"
                    f"**True space optimization for MCM (to O(N) space) is typically achieved using more advanced techniques**:\n"
                    f"* **Divide and Conquer with Recursion Optimization:** Algorithms like Hirschberg's algorithm (used for LCS space optimization) can be adapted. These involve computing only the optimal split point for the middle of a subproblem, then recursively solving the halves, using only O(N) space to compute results at each level.\n"
                    f"* **Limited Table Storage:** If only the minimum cost is needed and not the parenthesization, one might be able to reduce space, but reconstructing the parenthesization still often requires the full `s` table (O(N^2)).\n"
                    f"Therefore, while simple rolling arrays don't apply directly to reduce from O(N^2) to O(N) in MCM, specialized algorithms can achieve O(N) space but are more complex to implement and may have higher constant factors in time."
                )

        # 18. Write test cases to verify cost table entries for various chain lengths in {{dims}}.
        elif "write test cases to verify cost table entries for various chain lengths" in q:
            match = re.search(r"various chain lengths in (.+)\.", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 1: return "Error: Invalid dimensions. Need at least one matrix."

                # Get the full DP table
                m, _, _ = matrix_chain_order_dp(p) #

                test_case_examples = []
                test_case_examples.append(f"Dimensions: {dims_str} (P={p}, N={n} matrices)")
                
                test_case_examples.append("\n# Test for Chain Length 1 (Base Cases):")
                for i in range(1, n + 1):
                    test_case_examples.append(f"assert mcm_cost_table[{i}][{i}] == 0 # Cost of single matrix A{i}")
                
                if n >= 2:
                    test_case_examples.append("\n# Test for Chain Length 2 (A_i x A_{i+1}):")
                    for i in range(1, n):
                        j = i + 1
                        expected_cost = scalar_mult_cost(p[i-1], p[i], p[j])
                        test_case_examples.append(f"assert mcm_cost_table[{i}][{j}] == {expected_cost} # Cost of A{i} x A{j}")

                if n >= 3:
                    test_case_examples.append("\n# Test for Chain Length 3 (A_i x A_{i+1} x A_{i+2}):")
                    for i in range(1, n - 1):
                        j = i + 2
                        expected_cost = m[i][j] # Directly from the computed table
                        test_case_examples.append(f"assert mcm_cost_table[{i}][{j}] == {expected_cost} # Cost of A{i}..A{j}")

                test_case_examples.append("\n# Test for Overall Minimum Cost:")
                test_case_examples.append(f"assert mcm_cost_table[1][{n}] == {m[1][n]} # Overall min cost for A1..A{n}")

                return (
                    f"To write test cases to verify cost table entries for various chain lengths in {dims_str}:\n\n"
                    f"You would compare the computed `m[i][j]` values against known correct values or against values calculated by hand for small, representative subproblems.\n"
                    f"Example Python assertions (assuming `mcm_cost_table` is the result of `matrix_chain_order_dp(p)[0]`):\n"
                    "```python\n"
                    + "\n".join(test_case_examples) + "\n"
                    "```"
                )

        # 19. Determine which chain of {{dims}} leads to maximum savings using dynamic programming.
        elif "determine which chain of" in q and "leads to maximum savings using dynamic programming" in q:
            match = re.search(r"chain of (.+) leads to maximum savings using dynamic programming", q)
            if match:
                dims_str = match.group(1)
                p = parse_dims_string(dims_str)
                n = len(p) - 1

                if n < 3: return "Error: Need at least 3 matrices to determine savings from different parenthesizations."

                m, _, min_cost_dp = matrix_chain_order_dp(p) # Optimal cost

                # Max savings means (Cost of worst parenthesization) - (Cost of optimal parenthesization)
                # It's hard to compute "worst parenthesization" generally for N>3 without brute-forcing all exponential options.
                # For N=3, there are only 2 options, we can pick the max.
                
                max_savings = 0
                if n == 3: # For 3 matrices, there are only 2 options.
                    cost1 = scalar_mult_cost(p[0], p[1], p[2]) + scalar_mult_cost(p[0], p[2], p[3])
                    cost2 = scalar_mult_cost(p[1], p[2], p[3]) + scalar_mult_cost(p[0], p[1], p[3])
                    worst_cost = max(cost1, cost2)
                    max_savings = worst_cost - min_cost_dp
                    savings_info = f"For 3 matrices, the highest cost is {worst_cost}. Savings = {worst_cost} - {min_cost_dp} = **{max_savings}**."
                else:
                    savings_info = f"For N > 3 matrices, calculating the absolute 'worst' parenthesization cost is exponentially complex. Dynamic Programming primarily offers savings by reducing calculations from exponential to polynomial (O(N^3)) to find the optimal cost. The optimal cost found is {min_cost_dp}."
                    savings_info += "\nWhile there's no single 'worst' chain identified computationally, the *theoretical* savings (avoiding exponential runtime) are immense."

                return (
                    f"To determine which chain of {dims_str} leads to maximum savings using dynamic programming:\n"
                    f"Dynamic programming finds the global minimum cost. 'Maximum savings' refers to the difference between a suboptimal (or worst-case) parenthesization and this optimal cost.\n"
                    f"{savings_info}\n"
                    f"The actual matrix chain itself (the specific dimensions) dictates the potential for savings. Chains with very small intermediate dimensions or very large common dimensions often have the greatest difference between optimal and suboptimal parenthesizations."
                )

        # # 20. For large matrix chains with dimensions {{dims}}, optimize both computation and memory usage.
        # elif "optimize both computation and memory usage" in q:
        #     match = re.search(r"dimensions (.+)\.", q)
        #     if match:
        #         dims_str = match.group(1)
        #         p = parse_dims_string(dims_str)
        #         n = len(p) - 1

        #         return (
        #             f"For large matrix chains with dimensions {dims_str} (P={p}), optimizing both computation and memory usage is critical:\n"
        #             f"**1. Computational Optimization (Time Complexity):**\n"
        #             f"    * **Matrix Chain Multiplication DP:** The standard dynamic programming approach (O(N^3) time, O(N^2) space) is the primary method for polynomial time optimization. This is the optimal polynomial time solution.\n"
        #             f"    * **Advanced Algorithms:** For extremely large N (N > few hundreds), specialized algorithms like the "Four Russians" technique (for faster matrix multiplication or MCM) can reduce the complexity further (e.g., O(N^3 / log N)). However, these are highly complex.\n"
        #             f"**2. Memory Optimization (Space Complexity):**\n"
        #             f"    * **Standard DP is O(N^2) Space:** This space is used for the `m` (cost) and `s` (split) tables. For very large N, this can be significant.\n"
        #             f"    * **O(N) Space Optimization:** This is a more challenging problem for MCM than for other DPs (like Knapsack or LCS). It often involves: \n"
        #             f"        * **Divide and Conquer Techniques:** Adapting ideas from Hirschberg's algorithm. This involves recursively finding the optimal split point for the middle of a subproblem, then computing subproblems on either side with limited memory. This allows for O(N) space but makes reconstruction of the full parenthesization more complex or requires re-computation.\n"
        #             f"        * **Not Storing Split Table:** If only the minimum cost is required and not the actual parenthesization, the `s` table (which is O(N^2)) can be discarded, but `m` table remains O(N^2).\n"
        #             f"**Combined Optimization Strategy:**\n"
        #             f"* For optimal balance, use the **standard O(N^3) time, O(N^2) space DP solution** as a baseline.\n"
        #             f"* If `N^2` space is prohibitive, explore **O(N) space algorithms** (e.g., divide and conquer based) if exact optimal parenthesization is not needed, or if its reconstruction can be handled by recursive calls.\n"
        #             f"* For extremely large `N` (rare in practice unless `N` is astronomical), consider highly specialized algorithms for time complexity reduction, but they often come with increased implementation complexity."
        #         )

    
# --- NEW ADDITION START HERE ---
# --- Application Functions ---
def answer_application_mcm(level, question):
    q = question.lower()

    if level == "Level 1":
        # 1. How can Matrix Chain Multiplication improve performance in {{field}} systems with repeated matrix transformations?
        if "improve performance in" in q and "systems with repeated matrix transformations" in q:
            match = re.search(r"in (.+) systems with repeated matrix transformations", q)
            field = match.group(1) if match else "various"
            return (
                f"Matrix Chain Multiplication (MCM) can significantly improve performance in {field} systems with repeated matrix transformations by **finding the most efficient order of multiplication**.\n"
                f"Different parenthesizations of a matrix chain (e.g., (A x B) x C vs. A x (B x C)) can lead to drastically different numbers of scalar multiplications, and thus different computation times. MCM finds the optimal order that minimizes these operations, leading to substantial speedups in systems with frequent matrix computations."
            )

        # 2. Why is matrix multiplication order important in real-time {{field}} applications?
        elif "matrix multiplication order important in real-time" in q:
            match = re.search(r"real-time (.+) applications", q)
            field = match.group(1) if match else "critical"
            return (
                f"Matrix multiplication order is crucial in real-time {field} applications because these systems often have strict latency requirements.\n"
                f"Even small delays can be critical. An inefficient multiplication order can lead to a very high number of scalar operations, causing unacceptable computational overhead and latency in time-sensitive processes like sensor data processing, control loops, or graphics rendering. MCM ensures computations are performed with the fewest possible operations, meeting real-time deadlines."
            )

        # 3. Give an example of how MCM helps in optimizing calculations in {{context}} involving linear transformations.
        elif "mcm helps in optimizing calculations in" in q and "involving linear transformations" in q:
            match = re.search(r"in (.+) involving linear transformations", q)
            context = match.group(1) if match else "graphics"
            return (
                f"In {context} involving linear transformations, Matrix Chain Multiplication (MCM) helps optimize calculations by determining the most efficient sequence of matrix products.\n"
                f"**Example (3D Graphics):** When rendering a 3D scene, an object might undergo multiple linear transformations: translation, rotation, scaling, and projection onto a 2D screen. Each of these can be represented by a matrix (T, R, S, P).\n"
                f"Instead of computing `P x S x R x T x Vertex`, which might be suboptimal, MCM determines the best order to multiply the transformation matrices (e.g., `(P x (S x R)) x T`) to minimize the total number of scalar multiplications, thereby speeding up the rendering pipeline."
            )

        # 4. Describe how MCM reduces computation time in simple image processing pipelines.
        elif "mcm reduces computation time in simple image processing pipelines" in q:
            return (
                "In simple image processing pipelines, operations like resizing, rotation, and color adjustments can often be represented as matrix transformations applied sequentially to image data.\n"
                "MCM reduces computation time by identifying the optimal order to compose these transformation matrices before applying them to the actual image pixels. Multiplying a chain of 2x2 or 3x3 matrices in the most efficient order (e.g., (M1 x M2) x M3 vs. M1 x (M2 x M3)) minimizes the total number of scalar multiplications required to get the single combined transformation matrix. This combined matrix is then applied once per pixel, leading to significant time savings, especially for high-resolution images."
            )

        # 5. In what scenarios in {{field}} can MCM save computation cost by reordering matrix multiplication?
        elif "in what scenarios in" in q and "can mcm save computation cost by reordering matrix multiplication" in q:
            match = re.search(r"in (.+) can mcm save computation cost", q)
            field = match.group(1) if match else "scientific computing"
            return (
                f"In {field}, MCM can save computation cost by reordering matrix multiplication in scenarios involving sequential linear operations.\n"
                f"**Examples:**\n"
                f"* **Computer Graphics/Vision:** Chaining multiple view, model, projection, and texture transformation matrices.\n"
                f"* **Robotics/Control Systems:** Series of joint transformations (forward/inverse kinematics) or state-space model updates.\n"
                f"* **Data Analytics/Machine Learning:** Sequential feature transformations, principal component analysis (PCA) steps, or layer computations in simplified neural networks.\n"
                f"* **Physics Simulations:** Repeated application of transformation matrices to a system's state over time.\n"
                f"In all these fields, if you have a product of three or more non-square matrices, there's a high chance MCM can find a more efficient multiplication order."
            )

        # 6. How does MCM help reduce energy consumption in embedded systems like {{field}}?
        elif "mcm help reduce energy consumption in embedded systems like" in q:
            match = re.search(r"systems like (.+)\?", q)
            field = match.group(1) if match else "sensor networks"
            return (
                f"In embedded systems like {field}, MCM helps reduce energy consumption by minimizing the computational load.\n"
                f"Matrix multiplications are computationally intensive. By finding the optimal order, MCM reduces the total number of scalar multiplications. Fewer operations translate directly to:\n"
                f"* **Less CPU/DSP activity:** The processor spends less time actively computing.\n"
                f"* **Faster execution:** The task completes quicker, allowing the system to return to a low-power sleep state sooner.\n"
                f"Both factors directly contribute to lower energy consumption, which is critical for battery-powered embedded devices."
            )

        # 7. Explain the role of MCM in speeding up analytics workflows involving chained matrix operations.
        elif "role of mcm in speeding up analytics workflows involving chained matrix operations" in q:
            return (
                "In analytics workflows, especially in fields like statistics, data science, or engineering, data transformations are often expressed as chained matrix operations (e.g., scaling, rotation, dimensionality reduction).\n"
                "MCM plays a vital role by optimizing the order of these chained matrix multiplications. Instead of executing them in a naive sequence, MCM identifies the parenthesization that results in the minimum number of scalar multiplications for the entire chain. This direct reduction in computational burden leads to faster execution of complex analytical models and queries, allowing for quicker insights and real-time processing of large datasets."
            )

        # 8. What kind of performance improvements can be achieved using MCM in robotic arm control systems?
        elif "performance improvements can be achieved using mcm in robotic arm control systems" in q:
            return (
                "In robotic arm control systems, MCM can significantly improve performance related to real-time kinematics and dynamics calculations.\n"
                "Robotic arms perform sequences of transformations (e.g., joint rotations, translations, end-effector poses) which are represented by chained transformation matrices. MCM finds the optimal order to multiply these matrices, leading to:\n"
                f"* **Reduced Latency:** Faster computation of forward and inverse kinematics. This means quicker reaction times for the robot.\n"
                f"* **Smoother Motion:** More frequent and precise updates to joint commands, leading to fluid and accurate movements.\n"
                f"* **Lower Power Consumption:** Less computational load on the embedded controller.\n"
                f"* **Increased Throughput:** The controller can process more commands or higher-frequency sensor data, enabling more complex tasks."
            )

        # 9. Describe a basic use case in {{field}} where optimal matrix ordering improves speed.
        elif "basic use case in" in q and "optimal matrix ordering improves speed" in q:
            match = re.search(r"in (.+) where optimal matrix ordering improves speed", q)
            field = match.group(1) if match else "computer graphics"
            return (
                f"A basic use case in {field} where optimal matrix ordering improves speed is in **rendering pipelines**.\n"
                f"Imagine you have a 3D object (represented by vertices) that needs to be scaled (S), then rotated (R), then translated (T), and finally projected (P) onto the screen. Each of these is a matrix transformation (e.g., 4x4 matrices).\n"
                f"If you apply them as `P x T x R x S x Vertex`, the cost of multiplying `P x T`, then `(PT) x R`, etc., might be very high depending on the dimensions of the intermediate matrices. MCM would analyze the dimensions of `P, T, R, S` and suggest an order like `P x (T x (R x S))` that minimizes the total number of scalar multiplications required to get the combined transformation matrix. This single optimized matrix is then applied to all vertices, leading to significant speed improvement in rendering."
            )

        # 10. How can MCM be used to simplify logic in matrix-heavy spreadsheet computations?
        elif "mcm be used to simplify logic in matrix-heavy spreadsheet computations" in q:
            return (
                "In matrix-heavy spreadsheet computations (e.g., financial models, engineering calculations), users often chain multiple matrix operations. MCM can simplify logic by abstracting away the complex order-of-operations decision.\n"
                f"Instead of manually experimenting with different parenthesizations to find the fastest one, a spreadsheet tool or a custom function could integrate MCM. The user would simply input the sequence of matrices, and the system would internally use MCM to compute the optimal multiplication order. This means the user doesn't need to understand the underlying computational cost complexities, simplifying the logic and ensuring efficient calculation automatically."
            )

        # 11. Why might a graphics engine benefit from applying MCM for rendering transformations?
        elif "graphics engine benefit from applying mcm for rendering transformations" in q:
            return (
                "A graphics engine benefits significantly from applying Matrix Chain Multiplication (MCM) for rendering transformations because transformations (translation, rotation, scaling, projection) are represented by matrices, and rendering involves applying sequences of these to many vertices.\n"
                f"The benefit comes from:\n"
                f"* **Minimizing Redundant Operations:** MCM calculates the most efficient order to combine multiple transformation matrices into a single resultant matrix. This minimizes the total scalar multiplications. For example, if you have 1000 vertices, it's vastly more efficient to do `(Matrix1 x Matrix2 x Matrix3)` once optimally, then apply the combined matrix to 1000 vertices, rather than multiplying individual matrices 1000 times for each vertex.\n"
                f"* **Increased Frame Rates:** Fewer operations per frame means faster rendering, leading to higher frame rates and smoother animations.\n"
                f"* **Reduced Latency:** Crucial for real-time interactive graphics and virtual reality applications."
            )

        # 12. How can MCM assist in optimizing pipelines involving multiple coordinate transformations in {{field}}?
        elif "mcm assist in optimizing pipelines involving multiple coordinate transformations in" in q:
            match = re.search(r"transformations in (.+)\?", q)
            field = match.group(1) if match else "GIS"
            return (
                f"In {field}, pipelines often involve sequences of coordinate transformations (e.g., changing coordinate systems, projections, rotations) represented by matrices. MCM assists by optimizing the order in which these transformation matrices are multiplied together.\n"
                f"This creates a single, most efficient combined transformation matrix. This optimization is then applied once to each coordinate, reducing the overall computational cost significantly compared to applying each transformation sequentially without optimizing their composition. This is particularly beneficial when dealing with large datasets of points or objects."
            )

        # 13. Illustrate how MCM is beneficial in real-time sensor data fusion applications.
        elif "mcm is beneficial in real-time sensor data fusion applications" in q:
            return (
                "In real-time sensor data fusion, data from multiple sensors (e.g., GPS, IMU, lidar) often needs to be transformed into a common coordinate frame or filtered using Kalman filters, which involve numerous matrix operations.\n"
                f"MCM is beneficial by:\n"
                f"* **Optimizing Transformation Chains:** If sensor readings undergo a series of transformations (e.g., sensor_to_robot, robot_to_world, world_to_display), MCM can find the most efficient order to combine these transformation matrices. This reduces the processing time for each incoming sensor data packet.\n"
                f"* **Reducing Computational Load:** Fewer operations mean less CPU/DSP cycles consumed, freeing up resources for other real-time tasks and potentially lowering power consumption.\n"
                f"* **Meeting Deadlines:** Faster matrix computations ensure that fused data is available within strict real-time deadlines, critical for autonomous systems (e.g., self-driving cars, drones) where delayed information can lead to unsafe conditions."
            )

        # 14. What is the practical advantage of using MCM in video rendering systems?
        elif "practical advantage of using mcm in video rendering systems" in q:
            return (
                "The practical advantage of using MCM in video rendering systems is the **significant acceleration of complex transformations**, leading to smoother, higher-quality video output at faster processing speeds.\n"
                f"Video rendering involves applying many transformations (e.g., camera perspective, object movement, lighting) to millions of vertices and pixels per frame. By pre-optimizing the order of these matrix multiplications using MCM, the system can reduce the total computational burden for every frame. This translates to:\n"
                f"* **Faster Render Times:** Videos can be rendered in less time, improving production efficiency.\n"
                f"* **Higher Fidelity:** More complex transformations can be afforded within a given time budget.\n"
                f"* **Real-time Playback:** Enabling live previews or interactive rendering with complex scenes that would otherwise lag."
            )

        # 15. In what ways can MCM help reduce delays in software pipelines that handle real-time signal processing?
        elif "mcm help reduce delays in software pipelines that handle real-time signal processing" in q:
            return (
                "In real-time signal processing pipelines (e.g., audio processing, sensor data filtering, telecommunications), signals are often transformed through a series of linear operations, each representable by a matrix.\n"
                f"MCM helps reduce delays by:\n"
                f"* **Minimizing Operation Count:** By finding the optimal parenthesization of chained matrix multiplications, MCM directly reduces the total number of scalar operations. This means less CPU cycles per signal sample.\n"
                f"* **Lower Latency:** Faster computation translates directly to lower processing latency. In real-time systems, reduced latency ensures that signals are processed and outputted quickly enough to meet system requirements (e.g., preventing audio dropouts, enabling responsive control).\n"
                f"* **Resource Efficiency:** Reduced computational demands free up processing power, allowing the system to handle higher data rates or more complex algorithms within the same hardware constraints, avoiding delays caused by resource bottlenecks."
            )

        # 16. Explain how MCM reduces computational overhead in simulations with repeated matrix usage.
        elif "mcm reduces computational overhead in simulations with repeated matrix usage" in q:
            return (
                "In simulations (e.g., scientific, engineering, financial), the state of a system often evolves through repeated application of linear transformations, meaning the same set of matrices might be multiplied many times over simulation steps.\n"
                f"MCM reduces computational overhead by:\n"
                f"* **Pre-optimizing Chain Products:** Instead of performing a sequence of matrix multiplications naively at each simulation step, MCM determines the single most efficient order to combine the transformation matrices into a resultant matrix. This optimization is performed once.\n"
                f"* **Reusing Optimized Result:** The resulting optimal combined matrix is then efficiently reused for all subsequent simulation steps, reducing the per-step computational overhead from a complex chain multiplication to a single (optimized) matrix-vector or matrix-matrix product. This significantly speeds up the entire simulation run."
            )

        # 17. How does the use of MCM reduce latency in real-time control applications like {{field}}?
        elif "mcm reduce latency in real-time control applications like" in q:
            match = re.search(r"applications like (.+)\?", q)
            field = match.group(1) if match else "industrial automation"
            return (
                f"In real-time control applications like {field}, low latency is paramount for system stability and responsiveness. MCM reduces latency by directly minimizing the time spent on matrix computations.\n"
                f"Control systems frequently use linear algebra to model system states, predict behavior, and calculate control outputs. These often involve multiplying sequences of matrices (e.g., system dynamics, sensor transformations, control gains). MCM ensures that these crucial matrix multiplications are performed in the most computationally efficient order. Fewer scalar operations mean calculations complete faster, reducing the delay between input (sensor data) and output (actuator commands), thereby improving the real-time performance and responsiveness of the control loop."
            )

        # 18. What is the value of MCM in accelerating the computation of matrix-based animations?
        elif "value of mcm in accelerating the computation of matrix-based animations" in q:
            return (
                "The value of MCM in accelerating matrix-based animations lies in its ability to significantly **speed up the transformation pipeline for every animated object and frame**.\n"
                f"Animations involve objects undergoing continuous transformations (e.g., hierarchical rotations, translations, scaling). Each frame requires computing a combined transformation matrix for potentially thousands of objects by multiplying multiple individual transformation matrices. MCM optimizes this multiplication chain, reducing the total scalar operations needed to derive each object's final pose matrix.\n"
                f"This acceleration leads to:\n"
                f"* **Higher Frame Rates:** Smoother and more fluid animation playback.\n"
                f"* **More Complex Scenes:** Ability to animate more objects or use more detailed transformations without performance degradation.\n"
                f"* **Faster Rendering:** Reducing the time required to compute and render each frame, improving artist workflow and final output times."
            )

        # 19. Describe how MCM helps simplify calculations in physics simulations in {{context}}.
        elif "mcm helps simplify calculations in physics simulations in" in q:
            match = re.search(r"simulations in (.+)\.", q)
            context = match.group(1) if match else "engineering"
            return (
                f"In physics simulations within {context}, system states (e.g., positions, velocities, forces) are often updated via linear transformations, represented by matrices. MCM helps simplify calculations by automatically determining the most efficient way to combine sequential transformations.\n"
                f"For example, if a particle's state is updated through multiple forces or environmental interactions, each modeled by a matrix, the total update involves a chain of matrix multiplications. MCM finds the optimal parenthesization of this chain. This means the simulation code doesn't need complex manual optimization for these matrix products; instead, it can rely on MCM to perform the underlying calculations as efficiently as possible, simplifying the implementation and maintenance of the simulation's core logic."
            )

        # 20. In the context of {{field}}, how does MCM support smoother and faster matrix computation workflows?
        elif "context of" in q and "mcm support smoother and faster matrix computation workflows" in q:
            match = re.search(r"context of (.+), how does mcm support smoother and faster matrix computation workflows", q)
            field = match.group(1) if match else "data science"
            return (
                f"In the context of {field}, MCM supports smoother and faster matrix computation workflows by ensuring that sequences of matrix operations are executed with optimal efficiency.\n"
                f"Workflows in {field} often involve applying a series of linear algebraic steps (e.g., feature scaling, dimensionality reduction, model transformations) to data. If these steps are represented by matrices, MCM calculates the computationally cheapest order to multiply these matrices together. This means:\n"
                f"* **Reduced Processing Time:** Tasks complete faster, allowing for more iterations, larger datasets, or real-time responsiveness.\n"
                f"* **Smoother User Experience:** Applications that rely on these computations feel more fluid and responsive, as delays due to inefficient matrix operations are minimized.\n"
                f"* **Simplified Optimization:** Developers can chain matrix operations logically without needing to manually optimize their order, as MCM handles this automatically, leading to cleaner and more maintainable code."
            )

    elif level == "Level 2":
        # 1. Design a pipeline that applies MCM optimization to reduce computation time in a {{field}} workload.
        if "design a pipeline that applies mcm optimization to reduce computation time in a" in q:
            match = re.search(r"in a (.+) workload", q)
            field = match.group(1) if match else "scientific simulation"
            return (
                f"**Pipeline Design for MCM Optimization in a {field} Workload:**\n"
                f"Objective: Reduce computation time by optimizing chained matrix multiplications.\n"
                f"**Pipeline Stages:**\n"
                f"1.  **Matrix Identification & Dimension Extraction:**\n"
                f"    * Identify sequences of matrices that are multiplied together (e.g., `M1 x M2 x M3 x ...`).\n"
                f"    * Extract the dimensions `[p0, p1, p2, ..., pn]` for these matrices (where `Mi` is `pi-1 x pi`).\n"
                f"2.  **MCM Optimization (Offline/Preprocessing):**\n"
                f"    * Apply the Matrix Chain Multiplication (MCM) dynamic programming algorithm to the extracted dimensions.\n"
                f"    * This step computes the `m` (cost) table and the `s` (split point) table, yielding the minimum multiplication cost and the optimal parenthesization.\n"
                f"    * This is typically done once as a preprocessing step if the matrix dimensions are static.\n"
                f"3.  **Optimal Combined Matrix Computation:**\n"
                f"    * Using the optimal parenthesization from the `s` table, perform the actual matrix multiplications in the optimized order.\n"
                f"    * This results in a single, combined transformation matrix that is mathematically equivalent to the original chain but computed with minimal operations.\n"
                f"4.  **Application to Data:**\n"
                f"    * Apply this single, optimally combined matrix to the vectors or other matrices in the {field} workload.\n"
                f"    * This final step is where the bulk of the computational savings are realized, as a single optimized matrix product replaces a series of potentially inefficient ones.\n\n"
                f"**Example (Physics Simulation):** Instead of `(Rotate x Scale x Translate) x N_particles`, MCM finds `(R x (S x T))` optimally once, then applies `RST` to all particles. This makes the simulation faster."
            )
        
        # 2. In {{application}} pipelines, how does choosing optimal matrix order improve throughput?
        elif "application pipelines, how does choosing optimal matrix order improve throughput" in q:
            match = re.search(r"in (.+) pipelines, how does choosing optimal matrix order improve throughput", q)
            application = match.group(1) if match else "computer graphics"
            return (
                f"In {application} pipelines, choosing the optimal matrix order (via MCM) significantly improves throughput by reducing the total computational load for each unit of work processed.\n"
                f"Throughput refers to the amount of work completed per unit of time (e.g., frames per second, data points processed per second). By finding the multiplication sequence that requires the fewest scalar operations, MCM ensures that each transformation or calculation block completes faster. This means:\n"
                f"* **More Operations per Second:** The system can perform more matrix multiplications or transformations in the same amount of time.\n"
                f"* **Faster Processing of Batches:** Larger batches of data (e.g., multiple images, sensor readings) can be processed quicker.\n"
                f"* **Reduced Bottlenecks:** Computational stages that involve chained matrix products become less of a bottleneck, allowing data to flow through the pipeline more smoothly and rapidly, thus increasing overall system throughput."
            )

        # 3. Describe how MCM is used in multi-layer transformations in computer graphics rendering engines.
        elif "mcm is used in multi-layer transformations in computer graphics rendering engines" in q:
            return (
                "In computer graphics rendering engines, MCM is used in multi-layer transformations by optimizing the sequence of matrix multiplications applied to objects in a scene.\n"
                f"Objects in a 3D scene often undergo several transformations before being displayed:\n"
                f"1.  **Model Transformation (M):** Converts object's local coordinates to world coordinates.\n"
                f"2.  **View Transformation (V):** Converts world coordinates to camera/view coordinates.\n"
                f"3.  **Projection Transformation (P):** Converts 3D view coordinates to 2D screen coordinates.\n"
                f"A vertex's final position is calculated as `P x V x M x Vertex`. MCM analyzes the dimensions of `P`, `V`, and `M` (e.g., typically 4x4 matrices) to find the most efficient order to compute the combined transformation matrix `(PVM) = P x (V x M)` or `(P x V) x M`. This optimal `PVM` matrix is then applied to all vertices of the object, minimizing the total number of scalar operations per object and significantly accelerating the rendering process."
            )

        # 4. How can MCM be integrated into real-time robotics to speed up decision-making cycles?
        elif "mcm be integrated into real-time robotics to speed up decision-making cycles" in q:
            return (
                "MCM can be integrated into real-time robotics to speed up decision-making cycles by optimizing the underlying kinematic and dynamic computations that are often matrix-heavy.\n"
                f"**Integration Points:**\n"
                f"* **Kinematics (Forward/Inverse):** Robot joint transformations (e.g., Denavit-Hartenberg parameters) involve multiplying sequences of homogeneous transformation matrices. MCM can pre-optimize these chains to compute end-effector poses or joint angles faster.\n"
                f"* **Sensor Fusion:** Combining data from multiple sensors (e.g., cameras, lidar, IMU) often requires transforming data through several coordinate frames (e.g., `sensor_to_robot x robot_to_world`). MCM can optimize these chains.\n"
                f"* **Path Planning/Collision Detection:** Iterative algorithms might involve repeated application of Jacobian matrices or inverse dynamics matrices. MCM can ensure the most efficient computation of these matrices.\n"
                f"By reducing the scalar operations for these critical calculations, MCM shortens the computational time within the robot's control loop. This allows the robot to react faster to environmental changes, process more complex sensor inputs, and execute more agile maneuvers, thus speeding up its overall decision-making cycles."
            )

        # 5. Explain the benefits of MCM in embedded systems processing high-frequency sensor inputs.
        elif "benefits of mcm in embedded systems processing high-frequency sensor inputs" in q:
            return (
                "In embedded systems processing high-frequency sensor inputs (e.g., in drones, industrial control, automotive systems), MCM offers crucial benefits:\n"
                f"1.  **Reduced Latency:** Sensor data often undergoes rapid transformations (e.g., coordinate system changes, filtering, fusion) involving matrix multiplications. MCM ensures these sequences are computed with the minimum number of scalar operations, leading to faster processing of each sensor reading and reduced latency in the control loop.\n"
                f"2.  **Lower Power Consumption:** Fewer computational cycles directly translate to less power drawn by the processor. This is vital for battery-powered or energy-constrained embedded devices.\n"
                f"3.  **Efficient Resource Usage:** Optimizing matrix chains frees up CPU/DSP cycles and memory, allowing the embedded system to handle higher sampling rates, more complex algorithms, or other critical real-time tasks within limited hardware resources.\n"
                f"4.  **Reliability:** By minimizing computational load, MCM helps prevent bottlenecks and missed deadlines, enhancing the overall reliability of real-time operations."
            )

        # 6. How does MCM affect runtime efficiency in an AI pipeline for audio processing?
        elif "mcm affect runtime efficiency in an ai pipeline for audio processing" in q:
            return (
                "In an AI pipeline for audio processing (e.g., for speech recognition, noise cancellation, or audio synthesis), matrix operations are fundamental (e.g., FFTs, convolutions, neural network layer computations).\n"
                f"MCM can significantly affect runtime efficiency by optimizing sequences of these matrix transformations, especially when they are chained. For instance, if audio features are extracted, then normalized, then passed through a linear layer—each being a matrix operation—MCM ensures these combined operations are performed with minimal scalar multiplications. This leads to:\n"
                f"* **Faster Inference/Processing:** Audio segments can be processed more quickly.\n"
                f"* **Lower Latency:** Essential for real-time audio applications.\n"
                f"* **Reduced Computational Load:** Less strain on the processor, potentially enabling more complex AI models or parallel processing, and reducing power consumption."
            )

        # 7. Propose a way to use MCM in optimizing forward-pass layers of a neural network.
        elif "propose a way to use mcm in optimizing forward-pass layers of a neural network" in q:
            return (
                "In a neural network's forward pass, data flows through layers often involving matrix multiplications (`Output = Input x WeightMatrix`). When multiple linear layers are stacked without non-linear activation functions in between, their weight matrices can be combined.\n"
                f"**Proposed MCM Optimization for Forward-Pass Layers:**\n"
                f"1.  **Identify Linear Chains:** Look for sequences of linear layers (e.g., `Dense1 -> Dense2 -> Dense3`) where there are no non-linear activation functions, pooling, or other non-linear operations between them.\n"
                f"2.  **Extract Dimensions:** For such a chain `Input x W1 x W2 x W3`, extract the dimensions of the input vector/matrix and each weight matrix. E.g., `(1xN1) x (N1xN2) x (N2xN3) x (N3xN4)`.\n"
                f"3.  **Apply MCM:** Use the MCM algorithm to find the optimal parenthesization for multiplying `W1 x W2 x W3`.\n"
                f"4.  **Precompute Combined Weight Matrix:** Compute the optimal combined weight matrix `W_combined = (W1 x W2 x W3)` once, offline, following the optimal parenthesization found by MCM.\n"
                f"5.  **Optimize Forward Pass:** During actual inference, replace the sequence of multiplications `Input x W1 x W2 x W3` with a single, more efficient multiplication `Input x W_combined`.\n"
                f"This significantly reduces the scalar multiplications in the forward pass, especially beneficial for inference on edge devices or in high-throughput data centers."
            )

        # 8. Describe how optimal parenthesization using MCM is applied in climate simulation models.
        elif "optimal parenthesization using mcm is applied in climate simulation models" in q:
            return (
                "Climate simulation models involve complex systems of equations, often solved using numerical methods that rely heavily on linear algebra and repeated matrix operations.\n"
                f"Optimal parenthesization using MCM is applied by:\n"
                f"* **Modeling Physical Processes:** Representing various atmospheric, oceanic, or land surface processes as matrix transformations (e.g., diffusion, advection, radiative transfer). These matrices are often applied sequentially over time steps.\n"
                f"* **Optimizing Time Step Calculations:** If multiple matrix transformations are applied within a single time step to update the climate state (e.g., `State_next = M_rad x M_conv x M_diff x State_current`), MCM is used to find the most efficient order to multiply `M_rad x M_conv x M_diff`.\n"
                f"* **Reducing Iteration Cost:** This reduces the computational cost of each time step, which is crucial as climate simulations run for thousands or millions of time steps. Even a small per-step optimization translates to massive overall computational savings.\n"
                f"* **Resource Management:** Faster computation means simulations can be run quicker, allowing for more experiments, higher resolutions, or longer simulation periods within available supercomputing resources."
            )

        # 9. Use MCM to explain how matrix caching and ordering can reduce redundancy in repeated transformations.
        elif "matrix caching and ordering can reduce redundancy in repeated transformations" in q:
            return (
                "Matrix caching and optimal ordering (via MCM) work together to reduce redundancy in repeated transformations:\n"
                f"1.  **Redundancy in Naive Approach:** Without optimization, `A x B x C` is computed as `(A x B)` then `(A B) x C`. If `A x B x C` is part of a larger chain like `(A x B x C) x D`, and also `E x (A x B x C)`, then `A x B x C` might be computed redundantly multiple times if not cached.\n"
                f"2.  **MCM for Optimal Ordering:** MCM's core function is to find the minimum scalar multiplications for a chain `A x B x C` by trying all valid parenthesizations and selecting the cheapest. This ensures that the *single most efficient way* to compute `A x B x C` is identified.\n"
                f"3.  **Matrix Caching:** Once MCM identifies the optimal way to compute a sub-product (e.g., `A x B`), and its result (the intermediate matrix `AB`) is calculated, that **intermediate matrix `AB` can be cached**. If `AB` is needed again by other parts of the computation (e.g., `(A x B) x D`), it is retrieved from the cache in O(1) time rather than recomputed.\n"
                f"**Combined Effect:** MCM finds the fastest way to compute a transformation, and caching prevents that optimized result from being recomputed if the same transformation is needed again. This dual strategy dramatically reduces computational redundancy in complex systems involving repeated matrix transformations."
            )

        # 10. Explain MCM's role in reducing floating-point operations in computational biology workflows.
        elif "mcm's role in reducing floating-point operations in computational biology workflows" in q:
            return (
                "In computational biology, many analyses involve large datasets represented as matrices, undergoing transformations like sequence alignment scoring, phylogenetic tree construction, or molecular dynamics simulations. These often involve extensive floating-point matrix operations.\n"
                f"MCM reduces floating-point operations by:\n"
                f"* **Optimizing Linear Chains:** If a workflow involves a sequence of linear transformations on data matrices (e.g., `Data x Transform1 x Transform2 x Output`), MCM determines the optimal parenthesization for `Transform1 x Transform2`. This leads to the most efficient way to get a single combined transformation matrix.\n"
                f"* **Minimizing Scalar Multiplications:** Each matrix multiplication corresponds to many floating-point scalar multiplications. By minimizing these in the chain, MCM directly reduces the total floating-point arithmetic. This is crucial for numerical stability and speed in computationally intensive biological simulations and analyses.\n"
                f"* **Improved Performance:** Fewer floating-point operations mean faster execution, which is vital for processing massive genomic or proteomic datasets, and for running complex simulations within reasonable timeframes."
            )

        # 11. Show how matrix transformation order impacts cloud-based rendering workloads using MCM.
        elif "matrix transformation order impacts cloud-based rendering workloads using mcm" in q:
            return (
                "In cloud-based rendering workloads, matrix transformation order (optimized by MCM) significantly impacts cost and efficiency. Cloud rendering involves massive parallel computation across distributed servers.\n"
                f"**Impact:**\n"
                f"1.  **Computational Cost:** Each scalar multiplication consumes CPU cycles (or GPU cycles). MCM finds the order that minimizes these operations for each object's transformation chain. In a cloud environment, fewer cycles mean lower CPU/GPU usage, directly translating to **reduced operational costs** (pay-per-use).\n"
                f"2.  **Render Time/Latency:** Faster per-object transformation due to optimal ordering accelerates the overall rendering process. This means frames can be rendered quicker, improving real-time streaming experiences or shortening batch render farm queues.\n"
                f"3.  **Resource Utilization:** Optimal ordering allows the cloud infrastructure to process more rendering tasks with the same number of virtual machines/containers, leading to higher utilization of provisioned resources and better throughput.\n"
                f"**Example:** If a scene has millions of vertices and complex shaders, optimizing the initial model-view-projection matrix for each object (e.g., `P x V x M`) using MCM ensures that the absolute minimum computational work is sent down the pipeline, saving costs across thousands of cloud-based render nodes."
            )

        # 12. How can MCM be used in compiler optimizations for nested matrix expressions?
        elif "mcm be used in compiler optimizations for nested matrix expressions" in q:
            return (
                "MCM can be a powerful tool in compiler optimizations for nested matrix expressions by automatically generating the most efficient machine code for matrix products.\n"
                f"**Compiler Optimization Process:**\n"
                f"1.  **Parse Expression Tree:** The compiler first parses mathematical expressions like `A * (B * C) * D` into an abstract syntax tree (AST).\n"
                f"2.  **Identify Matrix Chains:** It then identifies sub-trees that represent a chain of matrix multiplications.\n"
                f"3.  **Extract Dimensions:** For each identified chain, the compiler extracts the dimensions of the matrices involved.\n"
                f"4.  **Apply MCM:** The compiler applies the MCM dynamic programming algorithm to these dimensions to determine the optimal parenthesization (multiplication order).\n"
                f"5.  **Code Generation:** Instead of generating code that performs multiplications in the user-specified (potentially suboptimal) order, the compiler generates assembly or machine code that executes the multiplications in the MCM-optimized order.\n"
                f"This ensures that the compiled program runs faster, especially for scientific or numerical computing applications that frequently deal with matrix operations, without the programmer needing to manually optimize the multiplication sequence."
            )

        # 13. Illustrate how MCM reduces matrix operation costs in augmented reality rendering.
        elif "mcm reduces matrix operation costs in augmented reality rendering" in q:
            return (
                "In Augmented Reality (AR) rendering, objects (virtual content) need to be precisely aligned with the real world. This involves extensive real-time matrix operations for camera tracking, pose estimation, and virtual object placement.\n"
                f"MCM reduces matrix operation costs by:\n"
                f"* **Optimizing Transformation Chains:** An AR pipeline might involve a chain of transformations like `(CameraPose x VirtualObjectModel x DisplayProjection)`. Each frame, this entire chain needs to be recomputed for every virtual object.\n"
                f"* **Minimizing Per-Frame Computations:** MCM determines the most efficient order to multiply these matrices. This minimizes the total number of scalar operations needed to get the final combined transformation matrix per object, per frame.\n"
                f"* **Achieving Real-time Frame Rates:** Reduced computation cost directly translates to faster frame generation. This is crucial for AR, as low latency and high frame rates are essential for a convincing and immersive augmented reality experience, preventing motion sickness and maintaining visual stability."
            )

        # 14. Apply MCM to identify the least expensive execution path in multi-step matrix computations.
        elif "apply mcm to identify the least expensive execution path in multi-step matrix computations" in q:
            return (
                "MCM is applied to identify the least expensive execution path in multi-step matrix computations by formulating the problem as finding the optimal parenthesization of a matrix chain.\n"
                f"**Steps to Apply MCM:**\n"
                f"1.  **Define the Chain:** Represent the multi-step computation as a sequence of matrix multiplications: `A1 x A2 x A3 x ... x An`.\n"
                f"2.  **Extract Dimensions:** Obtain the dimensions of each matrix. If `Ai` is `p_i-1 x p_i`, you'll have a dimension array `P = [p0, p1, ..., pn]`.\n"
                f"3.  **Run MCM DP:** Apply the dynamic programming algorithm for MCM, which fills a cost table `m[i][j]` (minimum cost to multiply `A_i` to `A_j`) and a split table `s[i][j]` (the optimal split point `k`).\n"
                f"4.  **Identify Least Expensive Path:** The value `m[1][n]` gives the minimum total scalar multiplication cost for the entire chain. The `s` table then allows you to reconstruct the exact sequence of multiplications (the parenthesization) that achieves this minimum cost, representing the 'least expensive execution path'.\n"
                f"This approach guarantees finding the globally optimal (least expensive) path, unlike greedy heuristics which might get stuck in local optima."
            )

        # 15. Discuss how MCM reduces both CPU time and memory usage in a mobile machine learning model.
        elif "mcm reduces both cpu time and memory usage in a mobile machine learning model" in q:
            return (
                "MCM can reduce both CPU time and memory usage in a mobile machine learning model, particularly in the inference phase involving linear layers or transformations.\n"
                f"**1. CPU Time Reduction (Computation):**\n"
                f"    * Mobile ML models often have stacked linear layers (e.g., dense layers in a small feedforward network) or feature transformation steps. If there are no non-linear activations between consecutive linear layers, their weight matrices can be theoretically combined into a single matrix multiplication `Input x W1 x W2 x W3`.\n"
                f"    * MCM finds the optimal parenthesization for `W1 x W2 x W3`, minimizing the total scalar multiplications needed to get the combined weight matrix. This combined matrix is precomputed offline or at model load.\n"
                f"    * During inference, the model then performs only `Input x W_combined`, which is significantly faster than three separate multiplications, reducing CPU cycles and accelerating inference.\n"
                f"**2. Memory Usage Reduction:**\n"
                f"    * By combining multiple weight matrices into one, MCM can potentially reduce the number of intermediate feature maps (tensors) that need to be stored in memory during the forward pass. For example, `(Input x W1) x W2` creates an intermediate `InputW1` tensor. If MCM recommends `Input x (W1 x W2)`, `(W1 x W2)` is computed once, and `Input x W_combined` can be calculated, potentially reducing peak memory if intermediates were large.\n"
                f"    * The combined weight matrix itself might be smaller than the sum of individual matrices if there were redundant dimensions or efficient compression techniques are used, further saving memory."
            )

        # 16. Explain how MCM helps in optimizing transformations in computer vision with matrix sequences.
        elif "mcm helps in optimizing transformations in computer vision with matrix sequences" in q:
            return (
                "In computer vision, tasks like image registration, object tracking, and camera calibration often involve a sequence of geometric transformations (e.g., rotation, translation, scaling, projection, perspective correction), each represented by a matrix.\n"
                f"MCM helps optimize these transformations by:\n"
                f"* **Identifying Optimal Composition:** It finds the most computationally efficient order to multiply these transformation matrices together into a single, combined transformation matrix. For instance, `T_total = T_proj x T_view x T_model`. MCM will determine if `(T_proj x T_view) x T_model` or `T_proj x (T_view x T_model)` is cheaper.\n"
                f"* **Reducing Per-Pixel/Per-Feature Cost:** This optimized combined matrix is then applied once to each pixel, point cloud coordinate, or feature vector. This drastically reduces the total number of scalar multiplications compared to applying individual transformations sequentially, leading to faster image processing and real-time performance.\n"
                f"* **Efficiency for High-Resolution Data:** This optimization is crucial for high-resolution images or large video streams where even small per-operation savings lead to significant overall performance gains."
            )

        # 17. For matrix sequences in {{field}}, describe how MCM determines execution cost savings.
        elif "for matrix sequences in" in q and "describe how mcm determines execution cost savings" in q:
            match = re.search(r"sequences in (.+), describe how mcm determines execution cost savings", q)
            field = match.group(1) if match else "finance"
            return (
                f"In {field}, when dealing with sequences of matrix operations, MCM determines execution cost savings by comparing the scalar multiplication cost of different parenthesizations against the absolute minimum cost.\n"
                f"**How MCM Determines Savings:**\n"
                f"1.  **Calculates Optimal Cost:** The MCM dynamic programming algorithm computes the minimum possible number of scalar multiplications required to perform a given chain of matrix multiplications. Let this be `C_optimal`.\n"
                f"2.  **Implicitly Compares to Suboptimal:** For any chain of `N` matrices where `N > 2`, there are multiple ways to parenthesize the multiplication, and their costs vary significantly. MCM explores all these possibilities (efficiently, via DP) and selects the one with `C_optimal`.\n"
                f"3.  **Savings = C_suboptimal - C_optimal:** The 'savings' is the difference between the cost of an arbitrary (often left-to-right or right-to-left naive) parenthesization `C_suboptimal` and `C_optimal`. This `C_suboptimal` can be orders of magnitude larger than `C_optimal`.\n"
                f"**Example (Financial Modeling):** If a financial model applies a series of risk adjustments (`R`), currency conversions (`C`), and portfolio rebalancing (`P`) matrices (`R x C x P`), MCM determines if `(R x C) x P` or `R x (C x P)` is cheaper. The difference between the chosen cheaper one and the more expensive one represents the direct scalar multiplication savings for each portfolio update."
            )

        # 18. In simulation software, how does MCM help when many intermediate matrices are created and multiplied?
        elif "simulation software, how does mcm help when many intermediate matrices are created and multiplied" in q:
            return (
                "In simulation software, when many intermediate matrices are created and multiplied (e.g., in fluid dynamics, structural analysis, or climate models), MCM helps by ensuring that the *most efficient* intermediate matrices are formed.\n"
                f"**How it helps:**\n"
                f"1.  **Strategic Intermediate Formation:** MCM's optimal parenthesization guides the order of multiplications, dictating which intermediate matrices are formed. It avoids forming large, computationally expensive intermediate matrices that might then be multiplied by another matrix, incurring huge costs. Instead, it prioritizes combinations that result in smaller intermediate products.\n"
                f"2.  **Memory Management (Implicit):** While MCM primarily optimizes computation, forming smaller intermediate matrices can also implicitly lead to better memory usage and cache performance, as less large temporary data needs to be stored and accessed.\n"
                f"3.  **Computational Efficiency:** The overall goal is to reduce the total scalar multiplications, which inherently guides the choice of intermediate matrices to those that minimize computation in the overall chain. This is crucial for running complex simulations within reasonable timeframes."
            )

        # 19. Compare two execution paths of matrix operations in {{field}} and show where MCM optimization helps.
        elif "compare two execution paths of matrix operations in" in q and "show where mcm optimization helps" in q:
            match = re.search(r"operations in (.+) and show where mcm optimization helps", q)
            field = match.group(1) if match else "signal processing"
            return (
                f"In {field}, matrix operations are fundamental. Let's compare two execution paths for multiplying matrices A(10x100), B(100x5), C(5x50) and show where MCM optimization helps:\n"
                f"**Scenario:** You need to compute `A x B x C`.\n"
                f"**Dimensions:** A (10x100), B (100x5), C (5x50)\n"
                f"**Execution Path 1: Left-to-Right ((A x B) x C)**\n"
                f"1.  Compute `(A x B)`: A(10x100) x B(100x5) -> Result(10x5)\n"
                f"    Cost = 10 * 100 * 5 = 5000 scalar multiplications.\n"
                f"2.  Compute `(Result x C)`: Result(10x5) x C(5x50) -> Final(10x50)\n"
                f"    Cost = 10 * 5 * 50 = 2500 scalar multiplications.\n"
                f"    Total Cost = 5000 + 2500 = **7500**.\n"
                f"**Execution Path 2: Right-to-Left (A x (B x C))**\n"
                f"1.  Compute `(B x C)`: B(100x5) x C(5x50) -> Result(100x50)\n"
                f"    Cost = 100 * 5 * 50 = 25000 scalar multiplications.\n"
                f"2.  Compute `(A x Result)`: A(10x100) x Result(100x50) -> Final(10x50)\n"
                f"    Cost = 10 * 100 * 50 = 50000 scalar multiplications.\n"
                f"    Total Cost = 25000 + 50000 = **75000**.\n"
                f"**Where MCM Helps:**\n"
                f"MCM would identify that Path 1 (cost 7500) is significantly cheaper than Path 2 (cost 75000). It ensures the system always chooses the path with the minimal scalar multiplications, leading to computational savings of `75000 - 7500 = 67500` scalar multiplications for this simple chain. This translates directly to faster processing and improved performance in {field} applications."
            )

        # 20. Evaluate the time reduction from applying MCM in a tabular data transformation engine.
        elif "evaluate the time reduction from applying mcm in a tabular data transformation engine" in q:
            return (
                "In a tabular data transformation engine, operations like pivots, joins, and aggregations can sometimes involve matrix multiplications (e.g., for statistical operations, embedding lookup, or custom transformations). MCM's application here can yield significant time reduction.\n"
                f"**Evaluation of Time Reduction:**\n"
                f"1.  **Identify Chainable Operations:** First, identify sequences of transformations that can be represented as matrix multiplications (e.g., a data table (matrix) undergoes `Transform1 -> Transform2 -> Transform3`).\n"
                f"2.  **Compute Optimal Cost:** Apply MCM to find the optimal parenthesization for these chained matrices. This yields `C_optimal`, the minimum scalar multiplications.\n"
                f"3.  **Compare to Naive/Heuristic:** Compare `C_optimal` against the cost of a typical naive (e.g., left-to-right) execution `C_naive`.\n"
                f"4.  **Time Reduction:** The direct time reduction per application of the chain is proportional to `(C_naive - C_optimal) / C_naive`. If `C_naive` is orders of magnitude larger, the reduction is substantial. For example, if `C_naive` is 100,000 and `C_optimal` is 5,000, that's a 95% reduction in multiplications for that step.\n"
                f"**Real-world Impact:** This means complex data transformations complete much faster, allowing the engine to process larger datasets, reduce query latency, and improve overall data processing throughput, which is crucial for big data analytics and real-time reporting."
            )

    elif level == "Level 3":
        # 1. Design a memory-efficient MCM algorithm suitable for use in mobile edge AI processors.
        if "design a memory-efficient mcm algorithm suitable for use in mobile edge ai processors" in q:
            return (
                "Designing a memory-efficient MCM algorithm for mobile edge AI processors (which have severe memory and computational constraints) is critical. The standard O(N^2) space DP for MCM might be too much for large N.\n"
                f"**Algorithm Design:**\n"
                f"1.  **Prioritize O(N) Space:** Instead of storing the full `m` and `s` tables (O(N^2) space), implement a space-optimized version that uses O(N) memory.\n"
                f"2.  **Hirschberg's Algorithm Variant:** A common approach for O(N) space DP is a divide-and-conquer strategy similar to Hirschberg's algorithm for LCS. This recursively finds the optimal split point for the middle of the current chain. It only needs to store `m[i][k]` and `m[k+1][j]` for the current subproblem, not the entire table. The challenge is reconstructing the full parenthesization, which might require re-computation or additional recursion.\n"
                f"3.  **Compute Cost Only:** If only the optimal cost is needed (not the parenthesization itself), further memory savings are possible. The `s` table can be completely omitted. Even for the `m` table, creative rolling array approaches (though complex for MCM's dependencies) might be explored.\n"
                f"4.  **Integer vs. Floating Point:** Ensure that dimensions and costs are handled with appropriate integer types to avoid overhead of arbitrary-precision integers unless explicitly required for very large numbers.\n"
                f"**Trade-off:** This O(N) space optimization usually increases the constant factor of time complexity, making it slightly slower than O(N^2) space solutions for smaller N, but it avoids memory exhaustion for very large N."
            )

        # 2. Apply MCM to optimize matrix operations across a distributed computing cluster performing real-time analytics.
        elif "apply mcm to optimize matrix operations across a distributed computing cluster performing real-time analytics" in q:
            return (
                "Applying MCM to optimize matrix operations across a distributed computing cluster involves strategizing both the MCM computation itself and the subsequent distributed matrix multiplications.\n"
                f"**Strategy for Distributed Optimization:**\n"
                f"1.  **Centralized MCM Optimization (Small N):** If the number of matrices `N` in the chain is relatively small (e.g., N < 1000, so N^3 is manageable), the MCM dynamic programming algorithm can be run on a single powerful node (or coordinator) in the cluster. It computes the optimal parenthesization (`s` table) for the entire chain.\n"
                f"2.  **Distributed Matrix Multiplication (Large Matrices):** Once the optimal multiplication order is determined, the actual matrix multiplications (e.g., `A x B`) are performed across the cluster. For very large matrices, matrix multiplication itself can be parallelized using distributed computing frameworks (e.g., Apache Spark, Dask, MPI). This involves partitioning matrices into blocks and distributing the block multiplications to different nodes.\n"
                f"3.  **Pipeline Stages:** The overall workflow becomes:\n"
                f"    * **Phase 1 (Centralized/Local):** MCM computation to get `s` table. (O(N^3) time, O(N^2) space).\n"
                f"    * **Phase 2 (Distributed):** Perform the matrix products according to the optimal parenthesization. Each individual `Matrix_A x Matrix_B` operation is parallelized across the cluster.\n"
                f"**Benefits:** Ensures that the computationally most intensive parts (individual matrix products for large matrices) are parallelized, while the overall chain is executed in the most efficient order. This reduces overall execution time and resource consumption in real-time distributed analytics."
            )

        # 3. Demonstrate how to apply MCM in the training phase of deep neural networks with multiple transformation layers.
        elif "demonstrate how to apply mcm in the training phase of deep neural networks with multiple transformation layers" in q:
            return (
                "Applying MCM directly in the *training phase* of Deep Neural Networks (DNNs) with *multiple transformation layers* (e.g., fully connected/dense layers) can be done by optimizing sequential linear operations, especially those without non-linear activations in between.\n"
                f"**Application Steps:**\n"
                f"1.  **Identify Linear Blocks:** During model compilation or optimization, identify consecutive `Dense` layers where the activation function is linear or missing (e.g., `Input -> Dense1 -> Dense2 -> Dense3 -> Output` with `linear` activation between Dense1/2 and Dense2/3).\n"
                f"2.  **Extract Weight Matrix Dimensions:** For such a block (e.g., `Dense1`, `Dense2`, `Dense3`), extract the dimensions of the weight matrices `W1`, `W2`, `W3` (e.g., `W1` is `input_dim x hidden1_dim`, `W2` is `hidden1_dim x hidden2_dim`, etc.).\n"
                f"3.  **Apply MCM (Offline):** Run the MCM algorithm on the dimensions of `W1, W2, W3` to find the optimal parenthesization for their product `W_combined = W1 x W2 x W3`. This step is done once.\n"
                f"4.  **Replace Layers with Combined Weight:** Conceptually, replace the sequence of `Dense1 -> Dense2 -> Dense3` with a single `Dense_combined` layer whose weight matrix is `W_combined`.\n"
                f"**Impact on Training:**\n"
                f"* **Forward Pass Optimization:** In the forward pass, the model now performs fewer matrix multiplications (`Input x W_combined` instead of `(Input x W1) x W2 x W3`), reducing computation time per iteration.\n"
                f"* **Backward Pass Optimization (Complex):** Optimizing the backward pass with combined matrices is more complex as it requires derivatives with respect to the original `W1, W2, W3`. This might involve storing intermediate results or using specialized backpropagation rules for combined layers. For simple cases, the time saved in the forward pass might outweigh the complexity in the backward pass.\n"
                f"This approach is more feasible for inference or very specific training architectures, as the non-linearity in most layers prevents such simple matrix chaining."
            )

        # 4. Evaluate the impact of MCM optimization in an online graphics engine for video games.
        elif "evaluate the impact of mcm optimization in an online graphics engine for video games" in q:
            return (
                "In an online graphics engine for video games, MCM optimization has a critical impact on performance, ultimately affecting the player experience and server load.\n"
                f"**Impact Evaluation:**\n"
                f"1.  **Client-Side (Rendering Performance):**\n"
                f"    * **Higher Frame Rates:** MCM optimizes sequences of transformation matrices (model, view, projection) for every object being rendered. Fewer scalar operations per vertex lead to faster frame generation, crucial for fluid gameplay (e.g., 60 FPS or higher).\n"
                f"    * **Reduced Latency:** Faster rendering reduces input-to-display latency, making controls feel more responsive.\n"
                f"    * **Enabling Richer Scenes:** More complex scenes with more objects or detailed transformations can be rendered within real-time budgets.\n"
                f"2.  **Server-Side (if server-rendered/cloud gaming):**\n"
                f"    * **Reduced Server Load/Cost:** For cloud-based gaming or server-side rendering, optimizing matrix computations directly reduces CPU/GPU cycles on the server, translating to lower operational costs (less compute per user/stream).\n"
                f"    * **Increased Throughput:** Servers can handle more concurrent players or render streams efficiently.\n"
                f"    * **Lower Bandwidth (if combined matrix sent):** If combined matrices are sent to thin clients, overall data can be optimized.\n"
                f"**Overall:** MCM is a fundamental optimization for graphics-intensive applications like video games, ensuring computational efficiency at the core of the rendering pipeline and directly contributing to a smoother, more immersive online experience."
            )

        # 5. Develop a hybrid algorithm using MCM to balance computation and memory in large-scale scientific simulations.
        elif "develop a hybrid algorithm using mcm to balance computation and memory in large-scale scientific simulations" in q:
            return (
                "Developing a hybrid algorithm using MCM for large-scale scientific simulations involves combining the time-optimal DP with space-saving techniques.\n"
                f"**Hybrid Algorithm Design:**\n"
                f"1.  **Phase 1: Cost-Optimal MCM (Standard DP for `m` table):**\n"
                f"    * Compute the `m` (cost) table using the standard O(N^3) time, O(N^2) space dynamic programming algorithm. This phase is computationally intensive but guarantees finding the global minimum cost.\n"
                f"    * During this phase, **DO NOT** store the full `s` (split point) table if `N^2` space is a primary concern for the `s` table specifically.\n"
                f"2.  **Phase 2: Space-Optimized Parenthesization (Recursive/Divide & Conquer):**\n"
                f"    * To reconstruct the optimal parenthesization (and thus perform the multiplications) without the O(N^2) `s` table, use a recursive, space-optimized divide-and-conquer approach. This approach is similar to Hirschberg's algorithm.\n"
                f"    * For a chain `A_i ... A_j`, to find the optimal split `k`, iterate through all `k` from `i` to `j-1`. For each `k`, recursively find `m[i][k]` and `m[k+1][j]` (which are already computed in the `m` table from Phase 1) and calculate `m[i][k] + m[k+1][j] + P[i-1]P[k]P[j]`.\n"
                f"    * This re-computes `k` but avoids storing `s` table. The recursion depth for finding `k` is O(log N), and each step involves looking up values in the `m` table. The space used for recursion stack is O(N).\n"
                f"**Balance Achieved:**\n"
                f"* **Computation (Time):** Retains the optimal O(N^3) time complexity for finding the minimal cost (from Phase 1). The reconstruction in Phase 2 adds O(N^2) to O(N^3) (depending on implementation detail) but is usually faster.\n"
                f"* **Memory:** Reduces overall memory usage from O(N^2) (for `m` and `s`) to O(N^2) for just the `m` table (or O(N) for more advanced variants). This balance is critical for large `N` where `N^2` storage for two tables is prohibitive but `N^2` for one is acceptable."
            )

        # 6. How does MCM contribute to latency reduction in real-time robotics motion planning pipelines?
        elif "mcm contribute to latency reduction in real-time robotics motion planning pipelines" in q:
            return (
                "In real-time robotics motion planning pipelines, latency reduction is paramount for safe, precise, and responsive robot operation. MCM contributes by optimizing the underlying matrix algebra, which is central to these pipelines.\n"
                f"**Contribution to Latency Reduction:**\n"
                f"1.  **Kinematics & Dynamics:** Motion planning heavily relies on forward and inverse kinematics (calculating robot pose from joint angles, or vice-versa) and dynamics (forces, torques). These involve sequences of matrix multiplications (e.g., Denavit-Hartenberg transforms).\n"
                f"2.  **Collision Detection:** Algorithms might use bounding box transformations or other geometric checks that involve matrix products.\n"
                f"3.  **State Estimation:** Sensor fusion (e.g., Kalman filters) combines noisy sensor data using matrix operations.\n"
                f"MCM ensures that each of these chained matrix computations is performed in the most computationally efficient order. By minimizing the total number of scalar operations, MCM directly reduces the processing time for each planning cycle. This shorter computation time means:\n"
                f"* **Faster Reactivity:** The robot can perceive its environment, calculate a new plan, and execute movements with minimal delay.\n"
                f"* **Increased Control Loop Frequency:** Allows the control loop to run at higher frequencies, leading to smoother and more accurate motion.\n"
                f"* **Improved Safety:** Quicker response to unexpected obstacles or dynamic environments enhances safety in autonomous systems."
            )

        # 7. Design an automated matrix optimizer using MCM that fits into a compiler backend for scientific code.
        elif "design an automated matrix optimizer using mcm that fits into a compiler backend for scientific code" in q:
            return (
                "Designing an automated matrix optimizer using MCM for a compiler backend involves recognizing matrix operations in source code and replacing them with optimized equivalents.\n"
                f"**Design Components & Integration:**\n"
                f"1.  **Frontend/Intermediate Representation (IR) Analysis:**\n"
                f"    * During the frontend (parsing/lexing) or early backend phases, the compiler builds an Intermediate Representation (IR) (e.g., Abstract Syntax Tree, Control Flow Graph).\n"
                f"    * Identify nodes or patterns in the IR that represent matrix multiplication chains (e.g., `A * B * C`, or `dot(A, dot(B, C))`). Extract the symbolic dimensions of these matrices.\n"
                f"2.  **MCM Optimization Pass:**\n"
                f"    * Implement an MCM optimization pass as a compiler pass (e.g., before code generation).\n"
                f"    * For each identified matrix chain `M1 x M2 x ... x Mn`, retrieve its dimensions `[p0, p1, ..., pn]`.\n"
                f"    * Invoke the MCM dynamic programming algorithm to compute the `m` (cost) and `s` (split point) tables.\n"
                f"    * Store the optimal parenthesization (reconstructed from `s`) as a data structure or modified IR node.\n"
                f"3.  **Code Generation Phase:**\n"
                f"    * When the code generator encounters a matrix multiplication chain, it checks if an optimized order exists (from the MCM pass).\n"
                f"    * If so, it generates machine code (or lower-level IR) that performs the multiplications according to the **optimal parenthesization**, minimizing scalar operations.\n"
                f"    * It might also consider generating calls to highly optimized linear algebra libraries (e.g., BLAS/LAPACK) for the individual matrix-matrix multiplications.\n"
                f"**Benefits:** The optimizer automatically ensures that matrix-heavy scientific code runs with optimal efficiency, without manual intervention from the programmer, leveraging the compiler's deep understanding of the code structure."
            )

        # 8. Create a benchmark suite to test MCM optimization across different matrix transformation workloads.
        elif "create a benchmark suite to test mcm optimization across different matrix transformation workloads" in q:
            return (
                "Creating a benchmark suite to test MCM optimization involves designing various matrix workloads and measuring performance improvements.\n"
                f"**Benchmark Suite Components:**\n"
                f"1.  **Workload Generation Module:**\n"
                f"    * **Random Chains:** Generate random matrix chains of varying lengths (`N`) and dimensions (`P`). Ensure dimensions are compatible.\n"
                f"    * **Specific Patterns:** Include workloads with known challenging patterns (e.g., very 'thin' matrices in the middle of a chain like `(1000x1) x (1x1000) x (1000x1000)`).\n"
                f"    * **Real-world Mimicry:** Create dimensions that simulate typical matrix operations in target domains (e.g., 4x4 for graphics, 1000x1000 for large data).\n"
                f"2.  **Implementation Module:**\n"
                f"    * **Optimized MCM:** The MCM algorithm that generates the optimal parenthesization.\n"
                f"    * **Naive/Baseline:** A simple left-to-right (or right-to-left) matrix multiplication implementation as a baseline for comparison.\n"
                f"    * **Actual Matrix Multiplier:** A function that performs matrix-matrix multiplication given the chosen order (e.g., using NumPy for Python).\n"
                f"3.  **Performance Measurement Module:**\n"
                f"    * Use `time.perf_counter()` or similar high-resolution timers to measure execution time.\n"
                f"    * Measure for both the optimized MCM execution and the baseline execution.\n"
                f"    * Consider measuring CPU cycles, memory usage (peak), and cache misses if possible.\n"
                f"4.  **Reporting Module:**\n"
                f"    * Calculate metrics: Speedup ratio (`Time_Naive / Time_Optimized`), percentage reduction in time (`(Time_Naive - Time_Optimized) / Time_Naive * 100`).\n"
                f"    * Present results in tables and graphs (e.g., speedup vs. N, speedup vs. dimension variance).\n"
                f"**Test Cases (Examples):**\n"
                f"* Small N, varied dimensions (e.g., `N=4, P=[10,100,5,50,20]` to check basic correctness and small savings).\n"
                f"* Large N, consistent dimensions (e.g., `N=20, P=[10,20,...,20,10]` to test scalability).\n"
                f"* Chains with bottleneck dimensions (e.g., `[100, 1, 100, 1, 100]`).\n"
                f"The benchmark suite allows quantifying the practical benefits of MCM across different real-world scenarios."
            )

        # 9. Discuss the performance trade-offs of using MCM in a mixed CPU-GPU compute environment.
        elif "performance trade-offs of using mcm in a mixed cpu-gpu compute environment" in q:
            return (
                "In a mixed CPU-GPU compute environment, using MCM involves trade-offs between the overhead of optimization and the acceleration of execution:\n"
                f"**1. CPU Overhead for MCM Optimization:**\n"
                f"    * The MCM dynamic programming algorithm itself runs on the CPU and has a time complexity of O(N^3).\n"
                f"    * For very large `N` (number of matrices), this CPU-side optimization might become a bottleneck. If the matrix chain needs to be re-optimized frequently (e.g., dimensions change dynamically), the CPU overhead can negate the GPU gain.\n"
                f"**2. GPU Acceleration of Matrix Multiplication:**\n"
                f"    * GPUs excel at highly parallel matrix-matrix multiplications. The individual `A x B` products determined by MCM's optimal order can be offloaded to the GPU for massive speedups.\n"
                f"    * This is where the primary performance gain lies, especially for large matrices.\n"
                f"**3. Data Transfer Overhead (CPU-GPU):**\n"
                f"    * Transferring matrices between CPU host memory and GPU device memory incurs significant overhead. MCM optimizes the *number of multiplications*, but if it leads to frequent, small intermediate transfers, this overhead might hurt performance.\n"
                f"    * **Trade-off:** MCM should aim to minimize the *total data transferred* to the GPU, not just the number of operations. Ideally, the entire optimized chain should be performed on the GPU if possible.\n"
                f"**4. Dynamic vs. Static Chains:**\n"
                f"    * **Static Chains:** If matrix dimensions are fixed, MCM can be run once offline. The optimal order is hardcoded. This is ideal.\n"
                f"    * **Dynamic Chains:** If dimensions change, MCM must be re-run. The CPU overhead might be too high for real-time systems. In such cases, simplified heuristics or a threshold (e.g., only run MCM if N is below a certain value) might be used.\n"
                f"**Conclusion:** MCM is highly beneficial when the GPU performs the actual large matrix multiplications, but it's crucial to consider the CPU overhead of running MCM itself and the data transfer costs between CPU and GPU memory. The optimal strategy often involves running MCM on the CPU offline, and then executing the optimally ordered matrix products on the GPU."
            )

        # 10. Implement a recursive-memoized MCM strategy that adapts based on matrix shape characteristics.
        elif "implement a recursive-memoized mcm strategy that adapts based on matrix shape characteristics" in q:
            return (
                "Implementing a recursive-memoized MCM strategy that adapts based on matrix shape characteristics would involve adding heuristics to the split point selection, potentially favoring certain matrix shapes (e.g., multiplying a tall-skinny by a fat-short matrix first) even if not immediately optimal by pure scalar count, to potentially reduce intermediate matrix size or improve cache locality.\n"
                f"**Adaptive Strategy (Conceptual):**\n"
                f"1.  **Standard Memoization:** Use the classic recursive MCM with memoization (O(N^3) time, O(N^2) space) as the base.\n"
                f"2.  **Adaptive Split Point Heuristic:** Within the `min_{{i <= k < j}}` loop, introduce a heuristic that biases `k` (the split point) based on the dimensions `P[i-1], P[k], P[j]`.\n"
                f"    * **Cost Calculation:** Calculate `cost = mcm_memoized(i, k) + mcm_memoized(k + 1, j) + P[i-1]*P[k]*P[j]`.\n"
                f"    * **Heuristic Adjustment:** Instead of just `if cost < min_q`, you might add a small penalty/bonus based on the shape of the resulting intermediate matrix (e.g., `(P[i-1] x P[k])` or `(P[k] x P[j])`). For instance:\n"
                f"        * Prefer splitting such that intermediate matrices are square-ish or small to improve cache performance.\n"
                f"        * Avoid very large intermediate dimensions (`P[k]`) even if it seems computationally fine at first, as it could lead to memory pressure.\n"
                f"3.  **Example Adaptation:** If `P[k]` is extremely large, even if `P[i-1]*P[k]*P[j]` is not the highest, forming an intermediate matrix `P[i-1]xP[k]` could be very memory-intensive. The adaptive strategy might penalize forming such intermediate matrices.\n"
                f"**Trade-offs:** This 'adaptation' moves away from strictly minimizing scalar operations and might not find the *globally optimal* multiplication count. It trades theoretical optimality for practical considerations like cache performance or peak memory usage, which are crucial on mobile/edge processors."
            )

        # 11. Analyze the speedup obtained when applying MCM to deep learning inference tasks on edge devices.
        elif "analyze the speedup obtained when applying mcm to deep learning inference tasks on edge devices" in q:
            return (
                "Analyzing the speedup from MCM in deep learning inference tasks on edge devices involves quantifying the reduction in operations and execution time.\n"
                f"**Application Context:** On edge devices (mobile, IoT), resources are constrained. Inference involves forward passes through a neural network. MCM applies when there are sequential linear layers (e.g., `Dense` layers) without non-linear activations in between, allowing their weight matrices to be combined.\n"
                f"**Speedup Analysis:**\n"
                f"1.  **Identify Chain:** For a sequence of linear layers `L1, L2, ..., Lm`, this translates to matrix multiplication `Input x W1 x W2 x ... x Wm`.\n"
                f"2.  **Calculate Operations (Baseline):** The naive (sequential) multiplication cost is `(Input_dims * W1_dims) + (W1_dims * W2_dims) + ...`.\n"
                f"3.  **Apply MCM (Optimization):** Use MCM to find the optimal order to multiply `W1 x W2 x ... x Wm`, resulting in a combined weight matrix `W_combined` and `C_optimal` scalar multiplications. The inference then becomes `Input x W_combined` with cost `C_inference = Input_dims * W_combined_dims + C_optimal`.\n"
                f"4.  **Quantify Speedup:**\n"
                f"    * **Operation Reduction:** Calculate `(Total_Naive_Operations - Total_Optimized_Operations) / Total_Naive_Operations * 100%`. This directly reflects the reduction in computational work.\n"
                f"    * **Execution Time Speedup:** Measure actual inference time: `Time_Naive / Time_Optimized`. This will be influenced by memory access patterns, cache performance, and hardware specifics.\n"
                f"**Impact on Edge Devices:** A significant speedup (e.g., 2x, 5x, 10x) directly translates to:\n"
                f"* **Lower Latency:** Faster real-time responses.\n"
                f"* **Reduced Power Consumption:** Longer battery life or less heat generation.\n"
                f"* **Higher Throughput:** More inferences per second.\n"
                f"* **Enabling Larger Models:** Allows running more complex models within tight performance budgets."
            )

        # 12. Devise an MCM-based algorithm to select optimal matrix execution order in a graph of linear operations.
        elif "devise an mcm-based algorithm to select optimal matrix execution order in a graph of linear operations" in q:
            return (
                "Devising an MCM-based algorithm for a *graph* of linear operations (where dependencies are more complex than a simple chain) requires extending the MCM concept.\n"
                f"**Algorithm Design:**\n"
                f"1.  **Graph Representation:** Represent the linear operations as a Directed Acyclic Graph (DAG) where:\n"
                f"    * **Nodes:** Represent individual matrices or intermediate results.\n"
                f"    * **Edges:** Represent multiplication operations (e.g., an edge from `A` and `B` to `C` means `A x B = C`).\n"
                f"2.  **Identify Chains:** Traverse the DAG to identify all possible linear chains of matrix multiplications from input nodes to output nodes. A chain exists where an output of one multiplication is a direct input to the next.\n"
                f"3.  **Apply MCM to Sub-Chains:** For each identifiable linear sub-chain, apply the standard MCM dynamic programming algorithm to find its optimal internal parenthesization and minimum cost. Replace these sub-chains with their optimally combined intermediate matrices (conceptually).\n"
                f"4.  **Dynamic Programming on DAG:** This is the most complex step. The problem effectively becomes one of optimal scheduling or topological sort with cost minimization. Use a DP approach on the DAG:\n"
                f"    * Define `dp[Node_X]` as the minimum cost to compute Node_X (which represents a matrix) from its inputs.\n"
                f"    * Process nodes in topological order. For each node `X` that is the result of `A x B`, `dp[X] = dp[A] + dp[B] + Cost_of_AxB`. If `X` is the result of a *chain* of operations (after step 3), use the `C_optimal` for that chain.\n"
                f"5.  **Reconstruct Global Order:** Backtrack through the DP states on the DAG to reconstruct the global optimal execution order. This combined approach ensures that both individual chains are optimized by MCM, and their overall sequence in the graph is also optimized for minimum total cost."
            )

        # 13. Explain how MCM affects throughput in real-time augmented reality rendering applications.
        elif "explain how mcm affects throughput in real-time augmented reality rendering applications" in q:
            return (
                "MCM significantly affects throughput in real-time Augmented Reality (AR) rendering applications by directly reducing the computational workload per frame, allowing the system to process more frames per second.\n"
                f"**Impact on Throughput:**\n"
                f"1.  **Reduced Per-Frame Cost:** AR rendering involves complex matrix transformations (e.g., camera pose, virtual object model, projection) that are applied to thousands or millions of vertices per frame. MCM optimizes the order of these matrix multiplications, minimizing the scalar operations needed to compute the final transformation matrix for each object.\n"
                f"2.  **Faster Frame Generation:** Lower computational cost per object means the entire frame can be composed and rendered more quickly. This directly increases the **frame rate (frames per second)**, which is the primary measure of throughput in rendering.\n"
                f"3.  **Improved Responsiveness:** Higher throughput allows the AR system to update the virtual overlay more frequently and with less latency, enhancing the sense of immersion and preventing visual inconsistencies that can cause motion sickness.\n"
                f"4.  **Resource Efficiency:** By completing the rendering tasks faster, the system's CPU/GPU resources are utilized more efficiently, allowing for more complex AR scenes or lower power consumption."
            )

        # 14. Evaluate MCM for high-resolution multi-camera processing in autonomous driving software.
        elif "evaluate mcm for high-resolution multi-camera processing in autonomous driving software" in q:
            return (
                "Evaluating MCM for high-resolution multi-camera processing in autonomous driving software highlights its potential for critical real-time performance gains, but also its limitations.\n"
                f"**Evaluation:**\n"
                f"1.  **Application:** In autonomous driving, multiple high-resolution cameras capture vast amounts of image data. This data needs to be continuously processed, rectified, transformed to a common vehicle coordinate system, fused, and used for perception (object detection, lane keeping). Many of these steps involve matrix operations (e.g., camera intrinsics/extrinsics, homographies, transformations between sensor frames).\n"
                f"2.  **Benefits of MCM:**\n"
                f"    * **Reduced Latency & Increased Throughput:** If a pixel or a feature point goes through a chain of 3D transformations (`Camera_to_IMU x IMU_to_Vehicle x Vehicle_to_World`), MCM can optimize this chain to minimize the scalar multiplications for each point. This directly reduces the processing time per camera frame, crucial for real-time decision-making.\n"
                f"    * **Resource Efficiency:** Less computation per frame means lower power consumption and less demand on the powerful but resource-limited embedded processors in the vehicle.\n"
                f"3.  **Limitations & Challenges:**\n"
                f"    * **Static Chains:** MCM is most effective for *static* chains of matrix multiplications where dimensions don't change frequently. In dynamic vision systems, this might be applicable to fixed calibration matrices, but less so for constantly changing scene graphs.\n"
                f"    * **GPU Parallelism:** Modern autonomous driving often leverages GPUs for massive parallel processing of pixel data. While MCM optimizes the *order* of matrix multiplication, the actual matrix-matrix or matrix-vector products themselves would still benefit most from GPU acceleration. MCM helps set up the most efficient products to run on the GPU.\n"
                f"    * **Non-linear Operations:** Many vision algorithms involve non-linear steps (e.g., deep neural networks, non-linear optimization), where MCM's linear algebra optimization isn't directly applicable.\n"
                f"**Conclusion:** MCM is valuable for optimizing the linear algebraic components of multi-camera processing, contributing to lower latency and higher throughput, but it's part of a larger optimization strategy that also includes GPU acceleration and specialized non-linear algorithms."
            )

        # 15. Demonstrate a tool that uses MCM to visualize optimal execution trees for matrix chains.
        elif "demonstrate a tool that uses mcm to visualize optimal execution trees for matrix chains" in q:
            return (
                "A tool that uses MCM to visualize optimal execution trees for matrix chains would provide a graphical representation of the most efficient multiplication order and its structure.\n"
                f"**Tool Demonstration (Conceptual):**\n"
                f"1.  **Input:** The user inputs the dimensions of a matrix chain (e.g., `[10, 100, 5, 50]`).\n"
                f"2.  **MCM Computation:** The tool internally runs the MCM dynamic programming algorithm to compute the `m` (cost) and `s` (split point) tables.\n"
                f"3.  **Tree Construction (from `s` table):** The tool then uses the `s` table to recursively build a binary execution tree:\n"
                f"    * The root of the tree represents the final multiplication of the entire chain (e.g., `A1..A4`).\n"
                f"    * Its children represent the two sub-chains determined by the optimal split point `k` (e.g., `(A1..Ak)` and `(Ak+1..A4)`).\n"
                f"    * Leaf nodes represent individual matrices (`A_i`).\n"
                f"    * Each internal node represents a matrix multiplication operation.\n"
                f"4.  **Visualization:** The tool would display this tree graphically.\n"
                f"    * Nodes might be labeled with the matrix names (e.g., A1), intermediate results (e.g., A1A2), or the multiplication cost of that step.\n"
                f"    * Edges could indicate the flow of operations.\n"
                f"    * The overall cost (`m[1][n]`) is displayed prominently.\n"
                f"**Benefits of Visualization:**\n"
                f"* **Intuitive Understanding:** Helps users quickly grasp why certain orders are more efficient than others by showing the dimensions of intermediate products and their associated costs.\n"
                f"* **Debugging/Learning:** Aids in understanding the MCM algorithm and debugging complex matrix expression optimizations.\n"
                f"* **Performance Analysis:** Provides immediate visual feedback on the impact of different matrix dimensions on the optimal order and total cost, demonstrating the value of MCM."
            )

        # 16. Compare recursive, tabulated, and hybrid MCM approaches in the context of dynamic workloads.
        elif "compare recursive, tabulated, and hybrid mcm approaches in the context of dynamic workloads" in q:
            return (
                "Comparing MCM approaches for dynamic workloads (where matrix dimensions or chains change frequently) involves considering their re-computation overhead and adaptability:\n"
                f"**1. Recursive (Memoized) MCM:**\n"
                f"    * **Pros:** Natural fit for dynamic inputs because memoization caches results on-demand. If parts of the chain repeat or are slightly modified, existing memoized results might be reused.\n"
                f"    * **Cons:** O(N^3) re-computation time for entirely new chains. Cache management (clearing/invalidating) is needed if dimensions change drastically to prevent stale results. Can hit recursion depth limits for large N.\n"
                f"**2. Tabulated (Bottom-Up) MCM:**\n"
                f"    * **Pros:** O(N^3) re-computation time. No recursion stack issues. Efficient for known ranges.\n"
                f"    * **Cons:** Typically recomputes the entire table even for small changes. Less adaptable to partial re-optimizations.\n"
                f"**3. Hybrid MCM (e.g., Online/Adaptive):**\n"
                f"    * **Pros:** A true 'hybrid' for dynamic workloads would aim to re-optimize only affected subproblems. For example, if a small part of a large chain changes, it might only re-run MCM for that affected sub-chain and intelligently update the larger results. Or, it could combine greedy heuristics with DP for very long chains. Some approaches use sampling or approximate MCM for very large chains.\n"
                f"    * **Cons:** Significantly more complex to design and implement. Might not guarantee global optimality for all dynamic changes.\n"
                f"**Conclusion for Dynamic Workloads:**\n"
                f"For dynamically changing workloads:\n"
                f"* If `N` is small to moderate, **memoized MCM** is often the easiest to implement and can offer good performance by reusing cached subproblems implicitly.\n"
                f"* If `N` is large and frequent re-optimization is needed, **custom hybrid algorithms** (e.g., those that identify affected subproblems and only re-optimize those) or more advanced online optimization techniques become necessary, though they involve higher implementation complexity."
            )

        # 17. Create a profiling tool that leverages MCM to analyze matrix performance bottlenecks in simulations.
        elif "create a profiling tool that leverages mcm to analyze matrix performance bottlenecks in simulations" in q:
            return (
                "A profiling tool leveraging MCM would pinpoint exactly where matrix operations are causing bottlenecks in simulations by identifying suboptimal multiplication sequences.\n"
                f"**Tool Design:**\n"
                f"1.  **Code Instrumentation:**\n"
                f"    * Modify the simulation code (or use a runtime hook) to detect matrix multiplication operations (`*`, `dot`, `matmul`).\n"
                f"    * When a chain of multiplications is encountered, log the dimensions of the matrices in the chain.\n"
                f"    * Track the actual execution time for each identified matrix multiplication chain.\n"
                f"2.  **MCM Analysis Module:**\n"
                f"    * For each logged matrix chain (dimensions), apply the MCM dynamic programming algorithm to calculate its optimal cost (`C_optimal`) and optimal parenthesization (`s` table).\n"
                f"    * Calculate the actual scalar multiplication cost (`C_actual`) of the sequence as it ran in the simulation.\n"
                f"3.  **Bottleneck Identification & Reporting:**\n"
                f"    * **Quantify Suboptimality:** Calculate `Savings = C_actual - C_optimal`. A large positive `Savings` indicates a potential bottleneck.\n"
                f"    * **Identify Offending Chains:** Highlight chains where `Savings` are highest or where `C_actual` contributes significantly to total simulation time.\n"
                f"    * **Suggest Optimization:** For each bottleneck, recommend the optimal parenthesization from the `s` table, possibly generating a code snippet for the optimized order.\n"
                f"    * **Visualize:** Show execution trees for actual (suboptimal) and optimal sequences, making the bottleneck visually clear.\n"
                f"**Impact:** This tool would allow simulation developers to quickly identify and rectify inefficiencies in their matrix-heavy code, leading to faster simulations and better resource utilization."
            )

        # 18. How does MCM integrate into modern AI compilers to improve graph-level matrix execution?
        elif "mcm integrate into modern ai compilers to improve graph-level matrix execution" in q:
            return (
                "Modern AI compilers (e.g., for TensorFlow, PyTorch, ONNX) optimize neural network graphs. MCM integrates by improving the execution of linear algebra subgraphs.\n"
                f"**Integration Process:**\n"
                f"1.  **Graph Representation (DAG):** Neural networks are represented as computation graphs (DAGs). Compiler passes analyze these graphs.\n"
                f"2.  **Linear Block Identification:** The compiler identifies subgraphs or sequences of nodes that represent chained linear operations (e.g., `MatMul -> MatMul -> MatMul`) where intermediate non-linear activations (ReLU, Sigmoid, etc.) are absent.\n"
                f"3.  **Dimension Inference:** For these linear blocks, the compiler infers the dimensions of the input tensors and the weight matrices of each operation.\n"
                f"4.  **MCM Optimization Pass:** A dedicated compiler pass invokes the MCM algorithm on the dimensions of these identified linear chains. This computes the optimal parenthesization for multiplying the weight matrices together.\n"
                f"5.  **Graph Rewriting/Fusion:** The compiler then 'rewrites' or 'fuses' the original linear layers into a single, optimized operation. Instead of multiple `matmul` operations, it inserts a single `matmul` with a precomputed, combined weight matrix (`W_combined`) that was derived using the MCM-optimized order.\n"
                f"**Benefit:** This reduces the number of operations in the execution graph, leading to faster inference time (especially on edge devices or for high-throughput serving), lower memory footprint, and improved utilization of underlying hardware (CPUs, GPUs, TPUs) by generating more efficient low-level code."
            )

        # 19. In what ways does MCM support adaptive execution planning in large-scale simulations of physical systems?
        # elif "mcm support adaptive execution planning in large-scale simulations of physical systems" in q:
        #     return (
        #         "MCM supports adaptive execution planning in large-scale simulations of physical systems by providing optimal cost information that can be leveraged for dynamic decision-making at runtime.\n"
        #         f"**Support for Adaptive Planning:**\n"
        #         f"1.  **Cost-Based Scheduling:** In simulations where different computational paths (sequences of matrix operations) might arise based on changing physical conditions (e.g., phase transitions, material properties), MCM can be used to pre-calculate or rapidly re-calculate the optimal cost for each potential path. The simulation engine can then adaptively choose the cheapest path to execute.\n"
        #         f"2.  **Dynamic Resource Allocation:** Knowledge of optimal costs allows for more precise resource allocation. If a specific chain of matrix operations is identified as the bottleneck at a given simulation state, MCM can determine its minimal cost. This information can then be used to dynamically allocate more computational resources (e.g., CPU cores, GPU time, network bandwidth in distributed settings) to that specific subproblem, ensuring efficient resource usage.\n"
        #         f"3.  **Load Balancing:** In distributed simulations, if parts of the simulation involve matrix chains with varying complexities, MCM can help identify the optimal execution cost for each part. This enables a more intelligent load balancing strategy, distributing tasks to nodes in a way that minimizes overall wall-clock time, rather than just raw operation counts.\n"
        #         f"4.  **Thresholding & Approximation:** For extremely large and dynamic chains, MCM can provide the exact optimal cost. An adaptive planner might use this to decide if a chain is "too expensive" even when optimized, triggering a switch to approximate methods or lower-fidelity models for that specific simulation region.\n"
        #         f"This allows simulations to dynamically adjust their computational strategy based on the current state of the system and available resources, leading to higher performance and better utilization of computational power."
        #     )

        # 20. Design a robust API layer that auto-applies MCM when matrix expressions are dynamically chained.
        elif "design a robust api layer that auto-applies mcm when matrix expressions are dynamically chained" in q:
            return (
                "Designing a robust API layer that auto-applies MCM when matrix expressions are dynamically chained requires intercepting matrix operations and intelligently optimizing them transparently to the user.\n"
                f"**API Layer Design (e.g., for a scientific computing library):**\n"
                f"1.  **Expression Tree Builder/Wrapper:**\n"
                f"    * Instead of immediate execution, overrideload matrix multiplication operators (`*`, `@`) to build an expression tree (DAG) representing the sequence of operations.\n"
                f"    * Example: `result = A @ B @ C` would build a tree `(A -> B -> C)`.\n"
                f"2.  **Deferred Execution/Optimization:**\n"
                f"    * Execution is deferred until a `.compute()` or `.evaluate()` method is explicitly called, or until a non-matrix operation (e.g., printing, saving) forces evaluation.\n"
                f"    * At the point of evaluation, the API layer traverses the expression tree.\n"
                f"3.  **MCM Optimization Pass:**\n"
                f"    * Identify contiguous linear chains of matrix multiplications within the expression tree.\n"
                f"    * For each chain, extract matrix dimensions and apply the MCM dynamic programming algorithm to determine the optimal parenthesization.\n"
                f"    * Cache the optimal order and potentially the combined intermediate matrices if they are large and reusable.\n"
                f"4.  **Optimized Execution Plan Generation:**\n"
                f"    * Based on the MCM results, the API generates an optimized execution plan (e.g., a sequence of calls to efficient low-level matrix multiplication routines like BLAS/LAPACK).\n"
                f"    * This might involve fusing multiple operations into a single combined call.\n"
                f"5.  **Execution:** The optimized plan is then executed.\n"
                f"**Robustness Considerations:**\n"
                f"* **Dimension Compatibility Checks:** Validate matrix dimensions at each step of chain building.\n"
                f"* **Error Handling:** Gracefully handle cases where optimization is not possible or if `N` is too large for O(N^3) MCM (e.g., fall back to default order or suggest alternative algorithms).\n"
                f"* **Performance Profiling/Logging:** Allow users to inspect whether MCM was applied and what speedup was achieved.\n"
                f"* **Integration with Hardware Accelerators:** The optimized plan should leverage available hardware (GPUs, TPUs) for actual matrix multiplications.\n"
                f"This provides a 'just-in-time' optimization for matrix computations, making it transparent and powerful for users working with dynamic matrix expressions."
            )

    return "No answer available for this MCM application question level."

# --- MAIN ANSWER GENERATION FUNCTIONS (END) ---


# --- TEST BLOCKS (START) ---

def test_answer_algorithmic_mcm():
    print("\n--- Testing Level 1 Algorithmic MCM Answers ---\n")
    questions_algo_lvl1 = [
        "Calculate the scalar multiplication cost for A1 (10x100) and A2 (100x5).",
        "Determine the total scalar operations for multiplying A1 x A2 x A3 using dimensions 10x100, 100x5, 5x50.",
        "List all possible parenthesis options for 3 matrices with dimensions 10x100, 100x5, 5x50 and calculate the cost of each.",
        "Which parenthesization is cheaper for A1 x A2 x A3 given dimensions [30, 35, 15, 5]?",
        "Explain the steps for multiplying matrices A1, A2, and A3 using ((A1 x A2) x A3) with dimensions 10x100, 100x5, 5x50.",
        "Given matrices A1, A2, and A3 with dimensions 10x100, 100x5, 5x50, compute cost using A1 x (A2 x A3).",
        "Determine the minimum number of scalar multiplications needed to multiply matrices A1 x A2 x A3 with dimensions 10x100, 100x5, 5x50.",
        "How would the multiplication cost change if A2 was transposed in a matrix chain with dimensions 10x20, 20x30, 30x40?", # A2 is not square
        "Find and compare the scalar multiplication cost for both parenthesizations: ((A1 x A2) x A3) vs A1 x (A2 x A3) with dimensions 10x20, 20x5, 5x30.",
        "What values of k (split index) should be considered to compute optimal multiplication of A1 to A3 for dimensions [10, 20, 5, 30]?",
        "Describe the scalar operations in the multiplication of A1 (10x20) and A2 (20x30).",
        "Compute and explain the total cost for multiplying A1 x A2 with dimensions 10x20 and 20x30.",
        "If matrix A1 has dimensions 5x10, and A2 has dimensions 10x15, what is the multiplication cost?",
        "For three matrices with dimensions 2x3, 3x4, 4x5, calculate the multiplication cost for both possible parenthesizations.",
        "Show the step-by-step scalar operations for ((A1 x A2) x A3) with matrix dimensions 5x10, 10x2, 2x8.",
        "Describe how to find the cost of matrix chain multiplication using only 3 matrices with dimensions 7x8, 8x9, 9x10.",
        "What is the multiplication order that gives the lowest cost for matrices with dimensions [2, 3, 4, 5]?",
        "How many total multiplications are required to multiply matrices A1 (10x20) and A2 (20x30)?",
        "Compute both ((A1 x A2) x A3) and (A1 x (A2 x A3)) for dimensions 10x10, 10x10, 10x10 and identify which is optimal.",
        "Explain the effect of matrix shape on multiplication cost for dimensions 10x2, 2x500, 500x100."
    ]

    for i, question in enumerate(questions_algo_lvl1, 1):
        print(f"Test Case {i}:")
        print(f"Question: {question}")
        answer = answer_algorithmic_mcm('Level 1', question)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

def test_answer_algorithmic_mcm_lvl2():
    print("\n--- Testing Level 2 Algorithmic MCM Answers ---\n")
    questions_algo_lvl2 = [
        "Use dynamic programming to find the minimum multiplication cost for matrix chain 10x100, 100x5, 5x50, 50x10.",
        "Show how the cost table (m[i][j]) is built for dimensions 30x35, 35x15, 15x5, 5x10 step-by-step.",
        "Trace m[1][3] calculation in matrix chain multiplication using dimensions 10x20, 20x5, 5x15.",
        "Which value of k (split index) results in the minimum multiplication cost for matrices A2 to A4 with 10x20, 20x5, 5x15, 15x25?",
        "Write a bottom-up DP function to compute matrix chain multiplication cost for dimensions [10, 20, 5, 15].",
        "Describe the recursive approach to matrix chain multiplication and compare it with DP for dimensions [10, 20, 5, 15].",
        "Given dimensions 10x10, 10x20, 20x5, 5x15, simulate how the DP table is filled for chain length 3.",
        "Construct a partial DP table with dimensions [10, 20, 5, 15, 25] and explain cell m[2][4].",
        "Explain why matrix chain multiplication solves overlapping subproblems using 10x20, 20x5, 5x15, 15x25.",
        "How does the cost table for dimensions 10x20, 20x5, 5x15, 15x25 avoid redundant calculations?",
        "Find m[1][4] and its split point for matrix dimensions 10x20, 20x5, 5x15, 15x25 using DP.",
        "Given matrix sizes [10, 20, 5, 15, 25], list subproblems solved while computing optimal cost.",
        "Use tabulation to find optimal order of A1 to A4 with dimensions [10, 20, 5, 15, 25].",
        "Fill in DP values m[i][i+1] and m[i][i+2] for matrix dimensions [10, 20, 5, 15, 25].",
        "What is the benefit of filling diagonals in the DP table for matrix dimensions [10, 20, 5, 15, 25]?",
        "Write a function to construct both the cost and split table for matrix chain [10, 20, 5, 15, 25].",
        "Explain how to reconstruct the optimal parenthesization using the split table for dimensions [10, 20, 5, 15, 25].",
        "Compare time complexity of recursive vs DP solution for matrix chain [10, 20, 5, 15, 25].",
        "What is the cost difference between recursive and DP approach on matrix chain [10, 20, 5, 15, 25]?",
        "Demonstrate the reduction in scalar operations using DP for matrix chain [10, 20, 5, 15, 25]."
    ]
    for i, question in enumerate(questions_algo_lvl2, 1):
        print(f"Test Case {i}:")
        print(f"Question: {question}")
        answer = answer_algorithmic_mcm('Level 2', question)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

def test_answer_algorithmic_mcm_lvl3():
    print("\n--- Testing Level 3 Algorithmic MCM Answers ---\n")
    questions_algo_lvl3 = [
        "Build a function that returns both cost and optimal order of multiplication for [10, 20, 5, 30] using memoization.",
        "Track and print full DP and split tables for matrix chain [30, 35, 15, 5, 10].",
        "Write a top-down memoized matrix chain solver for dimensions [10, 20, 5, 30] and print recursion trace.",
        "Explain overlapping subproblems in matrix chain multiplication with [10, 20, 5, 30] and how memoization helps.",
        "For a matrix chain of n = 5, generate dimensions and compute minimum cost using bottom-up DP.",
        "Develop a function to compute optimal parenthesization depth for dimensions [10, 20, 5, 30, 40].",
        "Print DP table after each chain length iteration for matrix chain [10, 20, 5, 30].",
        "Track total scalar multiplications avoided using DP on chain [10, 20, 5, 30].",
        "Use dynamic programming to simulate matrix chain computation for [10, 20, 5, 30] and visualize each iteration.",
        "Find the most balanced split point in the matrix chain [10, 20, 5, 30, 40] to reduce depth.",
        "Extend the DP solution to track nested multiplication levels for matrix chain [10, 20, 5, 30, 40].",
        "Compare three different parenthesis orders and their costs for matrix dimensions [10, 20, 5, 30, 40].",
        "Estimate time complexity of bottom-up DP for n = 5 and dimensions [10, 20, 5, 30, 40].",
        "Given matrix chain [10, 20, 5, 30], write a complete recursive, memoized, and tabulated version and compare them.",
        "Demonstrate trade-offs between time and space for bottom-up MCM using dimensions [10, 20, 5, 30].",
        "Visualize the fill process of DP table diagonally for matrix chain [10, 20, 5, 30].",
        "Design a function that uses memory-efficient rolling arrays for MCM with dimensions [10, 20, 5, 30].",
        "Write test cases to verify cost table entries for various chain lengths in [10, 20, 5, 30, 40].",
        "Determine which chain of [10, 20, 5, 30] leads to maximum savings using dynamic programming.",
        "For large matrix chains with dimensions [10, 20, 5, 30, 40], optimize both computation and memory usage."
    ]
    for i, question in enumerate(questions_algo_lvl3, 1):
        print(f"Test Case {i}:")
        print(f"Question: {question}")
        answer = answer_algorithmic_mcm('Level 3', question)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

def test_answer_application_mcm():
    print("\n--- Testing Level 1 Application MCM Answers ---\n")
    questions_app_lvl1 = [
        "How can Matrix Chain Multiplication improve performance in graphics systems with repeated matrix transformations?",
        "Why is matrix multiplication order important in real-time control systems applications?",
        "Give an example of how MCM helps in optimizing calculations in 3D rendering involving linear transformations.",
        "Describe how MCM reduces computation time in simple image processing pipelines.",
        "In what scenarios in scientific computing can MCM save computation cost by reordering matrix multiplication?",
        "How does MCM help reduce energy consumption in embedded systems like automotive ECUs?",
        "Explain the role of MCM in speeding up analytics workflows involving chained matrix operations.",
        "What kind of performance improvements can be achieved using MCM in robotic arm control systems?",
        "Describe a basic use case in physics simulations where optimal matrix ordering improves speed.",
        "How can MCM be used to simplify logic in matrix-heavy spreadsheet computations?",
        "Why might a graphics engine benefit from applying MCM for rendering transformations?",
        "How can MCM assist in optimizing pipelines involving multiple coordinate transformations in CAD software?",
        "Illustrate how MCM is beneficial in real-time sensor data fusion applications.",
        "What is the practical advantage of using MCM in video rendering systems?",
        "In what ways can MCM help reduce delays in software pipelines that handle real-time signal processing?",
        "Explain how MCM reduces computational overhead in simulations with repeated matrix usage.",
        "How does the use of MCM reduce latency in real-time control applications like industrial automation?",
        "What is the value of MCM in accelerating the computation of matrix-based animations?",
        "Describe how MCM helps simplify calculations in physics simulations in game development.",
        "In the context of machine learning, how does MCM support smoother and faster matrix computation workflows?"
    ]
    for i, q in enumerate(questions_app_lvl1, 1):
        print(f"Test Case {i}:")
        print(f"Question: {q}")
        answer = answer_application_mcm('Level 1', q)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

    print("\n--- Testing Level 2 Application MCM Answers ---\n")
    questions_app_lvl2 = [
        "Design a pipeline that applies MCM optimization to reduce computation time in a scientific simulation workload.",
        "In deep learning pipelines, how does choosing optimal matrix order improve throughput?",
        "Describe how MCM is used in multi-layer transformations in computer graphics rendering engines.",
        "How can MCM be integrated into real-time robotics to speed up decision-making cycles?",
        "Explain the benefits of MCM in embedded systems processing high-frequency sensor inputs.",
        "How does MCM affect runtime efficiency in an AI pipeline for audio processing?",
        "Propose a way to use MCM in optimizing forward-pass layers of a neural network.",
        "Describe how optimal parenthesization using MCM is applied in climate simulation models.",
        "Use MCM to explain how matrix caching and ordering can reduce redundancy in repeated transformations.",
        "Explain MCM's role in reducing floating-point operations in computational biology workflows.",
        "Show how matrix transformation order impacts cloud-based rendering workloads using MCM.",
        "How can MCM be used in compiler optimizations for nested matrix expressions?",
        "Illustrate how MCM reduces matrix operation costs in augmented reality rendering.",
        "Apply MCM to identify the least expensive execution path in multi-step matrix computations.",
        "Discuss how MCM reduces both CPU time and memory usage in a mobile machine learning model.",
        "Explain how MCM helps in optimizing transformations in computer vision with matrix sequences.",
        "For matrix sequences in financial modeling, describe how MCM determines execution cost savings.",
        "In simulation software, how does MCM help when many intermediate matrices are created and multiplied?",
        "Compare two execution paths of matrix operations in signal processing and show where MCM optimization helps.",
        "Evaluate the time reduction from applying MCM in a tabular data transformation engine."
    ]
    for i, q in enumerate(questions_app_lvl2, 1):
        print(f"Test Case {i}:")
        print(f"Question: {q}")
        answer = answer_application_mcm('Level 2', q)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

    print("\n--- Testing Level 3 Application MCM Answers ---\n")
    questions_app_lvl3 = [
        "Design a memory-efficient MCM algorithm suitable for use in mobile edge AI processors.",
        "Apply MCM to optimize matrix operations across a distributed computing cluster performing real-time analytics.",
        "Demonstrate how to apply MCM in the training phase of deep neural networks with multiple transformation layers.",
        "Evaluate the impact of MCM optimization in an online graphics engine for video games.",
        "Develop a hybrid algorithm using MCM to balance computation and memory in large-scale scientific simulations.",
        "How does MCM contribute to latency reduction in real-time robotics motion planning pipelines?",
        "Design an automated matrix optimizer using MCM that fits into a compiler backend for scientific code.",
        "Create a benchmark suite to test MCM optimization across different matrix transformation workloads.",
        "Discuss the performance trade-offs of using MCM in a mixed CPU-GPU compute environment.",
        "Implement a recursive-memoized MCM strategy that adapts based on matrix shape characteristics.",
        "Analyze the speedup obtained when applying MCM to deep learning inference tasks on edge devices.",
        "Devise an MCM-based algorithm to select optimal matrix execution order in a graph of linear operations.",
        "Explain how MCM affects throughput in real-time augmented reality rendering applications.",
        "Evaluate MCM for high-resolution multi-camera processing in autonomous driving software.",
        "Demonstrate a tool that uses MCM to visualize optimal execution trees for matrix chains.",
        "Compare recursive, tabulated, and hybrid MCM approaches in the context of dynamic workloads.",
        "Create a profiling tool that leverages MCM to analyze matrix performance bottlenecks in simulations.",
        "How does MCM integrate into modern AI compilers to improve graph-level matrix execution?",
        "In what ways does MCM support adaptive execution planning in large-scale simulations of physical systems?",
        "Design a robust API layer that auto-applies MCM when matrix expressions are dynamically chained."
    ]
    for i, q in enumerate(questions_app_lvl3, 1):
        print(f"Test Case {i}:")
        print(f"Question: {q}")
        answer = answer_application_mcm('Level 3', q)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

def test_answer_optimization_mcm():
    print("\n--- Testing Level 1 Optimization MCM Answers ---\n")
    questions_opt_lvl1 = [
        "Why do we need to optimize the order of matrix multiplication in Matrix Chain Multiplication problems?",
        "What is meant by 'optimal order' in Matrix Chain Multiplication?",
        "How does changing the order of multiplying matrices affect performance?",
        "What is the main goal of optimization in Matrix Chain Multiplication?",
        "Why is it not always efficient to multiply matrices in the given input order?",
        "In what way does optimization in MCM reduce computation cost?",
        "Can optimization in MCM reduce the number of scalar multiplications? How?",
        "What would happen if we did not optimize matrix multiplication order in a long matrix chain?",
        "Why is brute-force not suitable for optimizing matrix multiplication order in large problems?",
        "What role does dynamic programming play in optimizing MCM problems?",
        "How can optimizing matrix order improve the efficiency of basic algorithms?",
        "Does optimization always lead to the same result when solving MCM problems? Why or why not?",
        "Is the order of matrix multiplication relevant to the final result in MCM problems?",
        "What is the simplest example where optimizing matrix multiplication reduces the number of operations?",
        "Why is MCM considered a classic optimization problem in dynamic programming?"
    ]
    for i, q in enumerate(questions_opt_lvl1, 1):
        print(f"Test Case {i}:")
        print(f"Question: {q}")
        answer = answer_application_mcm('Level 1', q)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

    print("\n--- Testing Level 2 Optimization MCM Answers ---\n")
    questions_opt_lvl2 = [
        "Explain how dynamic programming helps in optimizing the matrix multiplication sequence in MCM.",
        "How can we construct the cost matrix and split matrix while solving MCM using optimization?",
        "How does the number of possible parenthesizations grow with the number of matrices in MCM, and how does optimization help?",
        "What is the significance of minimizing scalar multiplications in the MCM optimization process?",
        "Describe how memoization can be used as an optimization technique in recursive MCM solutions.",
        "Why is it necessary to store intermediate results during optimization in the MCM problem?",
        "How would you identify the optimal split point in a chain of matrices using the DP table?",
        "Explain the role of base cases in the dynamic programming solution for MCM optimization.",
        "How is time complexity improved when using an optimized approach to solve MCM compared to brute force?",
        "How can we represent the optimal multiplication sequence once the DP table is computed in MCM?",
        "How does overlapping subproblems concept apply in MCM optimization?",
        "What is the difference between top-down and bottom-up optimization in solving MCM?",
        "What are the space and time complexities of the optimized dynamic programming solution to MCM?",
        "How can you visualize the optimization process in MCM using a parenthesis tree or recursion tree?",
        "What is the significance of minimizing multiplication cost in real-world systems that use MCM?",
        "Can the MCM optimization logic be extended to non-square matrices? What challenges arise?",
        "Why is MCM considered a classic example of dynamic programming optimization problems?",
        "How does the order of matrices affect the cost matrix values during MCM optimization?"
    ]
    for i, q in enumerate(questions_opt_lvl2, 1):
        print(f"Test Case {i}:")
        print(f"Question: {q}")
        answer = answer_application_mcm('Level 1', q)
        print(f"Answer:\n{answer}\n{'-'*50}\n")

    print("\n--- Testing Level 3 Optimization MCM Answers ---\n")
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
    for i, q in enumerate(questions_opt_lvl3, 1):
        print(f"Test Case {i}:")
        print(f"Question: {q}")
        answer = answer_application_mcm('Level 1', q)
        print(f"Answer:\n{answer}\n{'-'*50}\n")


# Main execution block to run tests
if __name__ == "__main__":
    # Uncomment the test function calls below to run specific test suites.
    
    # Run Algorithmic MCM tests (all levels)
    test_answer_algorithmic_mcm()
    test_answer_algorithmic_mcm_lvl2()
    # test_answer_algorithmic_mcm_lvl3()

    # Run Application MCM tests (all levels)
    # test_answer_application_mcm()

    # Run Optimization MCM tests (all levels)
    # test_answer_optimization_mcm()