def answer_conceptual_fibonacci_lvl1(question):
    q = question.lower().strip()

    if "what is the fibonacci sequence" in q:
        return (
            "🔢 The Fibonacci sequence is a series of numbers where each term is the sum of the two previous terms.\n\n"
            "👉 It starts with 0 and 1: 0, 1, 1, 2, 3, 5, 8, 13, ...\n"
            "📈 It grows gradually and appears in nature, art, and algorithms!"
        )

    elif "each term in the fibonacci sequence is calculated" in q:
        return (
            "🧮 Each term is calculated by adding the two terms just before it.\n\n"
            "📝 Formula: F(n) = F(n-1) + F(n-2)\n"
            "✔️ Example: F(4) = F(3) + F(2) = 2 + 1 = 3"
        )

    elif "determine whether a number belongs to the fibonacci sequence" in q:
        return (
            "🔍 A number is a Fibonacci number if one of these is a perfect square:\n\n"
            "➕ 5×n² + 4\n➖ 5×n² - 4\n\n"
            "📘 Example: For n=8 → 5×64±4 = 324 or 316 → 324 is 18² → ✔️ 8 is Fibonacci!"
        )

    elif "first two base cases of the fibonacci sequence" in q:
        return (
            "🟢 Base cases:\n\n"
            "F(0) = 0\n"
            "F(1) = 1\n\n"
            "✅ These are starting values used in recursion or iteration."
        )

    elif "0th fibonacci number" in q:
        return "🔢 F(0) = 0\n\nIt's the starting point of the Fibonacci sequence."

    elif "1st fibonacci number" in q:
        return "🔢 F(1) = 1\n\nThis is the second number in the Fibonacci sequence."

    elif "first 5 numbers in the fibonacci series" in q:
        return "📊 First 5 Fibonacci numbers: 0, 1, 1, 2, 3"

    elif "define the recursive relation used to calculate fibonacci" in q:
        return (
            "🔁 Recursive Relation:\n\n"
            "F(n) = F(n-1) + F(n-2)\n"
            "🔹 Base Cases: F(0) = 0, F(1) = 1"
        )

    elif "base case in a recursive fibonacci" in q:
        return (
            "🟨 Base cases stop recursion:\n\n"
            "F(0) = 0 and F(1) = 1\n\n"
            "⛔ Without them, the recursion would never end!"
        )

    elif "time complexity of the naive recursive fibonacci" in q:
        return (
            "⏱ Naive recursion has time complexity O(2^n)\n\n"
            "⚠️ Because it recalculates the same subproblems repeatedly."
        )

    elif "fibonacci numbers grow slowly compared to exponential" in q:
        return (
            "📉 Fibonacci grows slower than 2^n.\n\n"
            "✅ It increases steadily, but each term only adds the last two — not doubling or squaring.\n"
            "That's why it's slower than exponential growth."
        )

    elif "what number comes after" in q:
        return (
            "➕ Add the two numbers to get the next one!\n\n"
            "Example: If given 5 and 8 → 5 + 8 = 13\n"
            "✅ Answer: 13"
        )

    elif "fill in the blank: 0, 1, 1, 2, 3," in q:
        return "✏️ Next number is 5\n\nBecause 2 + 3 = 5."

    elif "which number in the fibonacci sequence is closest to" in q:
        return (
            "🔍 To find the closest Fibonacci number to a given number, compare it with nearby terms.\n\n"
            "✅ Use precomputed Fibonacci list or generate terms until the closest match is found."
        )

    elif "can the fibonacci series contain duplicate numbers in its first" in q:
        return (
            "♻️ Yes, but only once:\n\n"
            "F(1) = 1 and F(2) = 1 → This is the only repeated number in early terms.\n"
            "After that, all terms are unique."
        )

    elif "is the number" in q and "fibonacci" in q:
        return (
            "🔍 Use the perfect square check:\n\n"
            "A number n is Fibonacci if (5n² + 4) or (5n² - 4) is a perfect square.\n\n"
            "🧮 This works without generating the whole sequence."
        )

    elif "role of addition in the fibonacci sequence" in q:
        return (
            "➕ Addition is the core of Fibonacci!\n\n"
            "Each number is the sum of the two before it: F(n) = F(n-1) + F(n-2)"
        )

    elif "what does the fibonacci sequence start with" in q:
        return (
            "📍 It starts with 0 and 1.\n\n"
            "This gives a foundation to apply the recurrence relation and build up the sequence."
        )

    elif "why is the fibonacci sequence considered a recurrence relation" in q:
        return (
            "🔁 Because each term depends on previous terms.\n\n"
            "A recurrence relation defines terms based on earlier ones, which is exactly how Fibonacci works!"
        )

    elif "what mathematical operation defines the fibonacci sequence" in q:
        return "➕ Addition defines the sequence — each term adds the two before it."

    elif "how does each term in the fibonacci series relate to its previous terms" in q:
        return (
            "🧩 Each term is the sum of the previous two.\n\n"
            "F(n) = F(n-1) + F(n-2)"
        )

    elif "fib(0) = 0 and fib(1) = 1" in q:
        return "✅ Fib(2) = Fib(1) + Fib(0) = 1 + 0 = 1"

    elif "property makes the fibonacci sequence a good example of recursion" in q:
        return (
            "🔄 It naturally defines terms using earlier ones.\n\n"
            "✅ Simple base cases and clear recurrence make it perfect for recursion examples."
        )

    elif "pattern do you observe in the parity" in q:
        return (
            "🧮 The even/odd pattern in Fibonacci: Odd, Odd, Even, Odd, Odd, Even...\n\n"
            "♻️ It repeats every 3 numbers!"
        )

    elif "how does the fibonacci sequence differ from an arithmetic progression" in q:
        return (
            "📏 Arithmetic Progression (AP): Adds a constant value.\n"
            "📈 Fibonacci: Adds the two previous terms.\n\n"
            "✅ AP: 2, 4, 6, 8...\n✅ Fibonacci: 0, 1, 1, 2, 3..."
        )

    elif "describe the growth of fibonacci numbers in simple terms" in q:
        return (
            "📈 Fibonacci numbers grow steadily and gradually speed up.\n\n"
            "They don’t double, but get faster over time."
        )

    elif "why are the first two terms in the fibonacci sequence necessary" in q:
        return (
            "🔑 They form the base of the sequence.\n\n"
            "Without them, you can't apply the recurrence relation to build the rest."
        )

    elif "can a fibonacci number be negative in the standard sequence" in q:
        return (
            "❌ No. In the standard Fibonacci sequence, all numbers are non-negative.\n\n"
            "But there's a concept of 'Negafibonacci' for negative indices!"
        )

    elif "how many base cases are needed in a recursive fibonacci function" in q:
        return (
            "🔢 Two base cases are needed:\n\n"
            "F(0) = 0 and F(1) = 1\n\n"
            "🛑 They stop the recursion from going infinitely backward."
        )

    elif "fibonacci sequence unique for a given starting pair" in q:
        return (
            "✅ Yes, if the first two terms are fixed, the sequence is uniquely determined.\n\n"
            "The recurrence relation produces the rest."
        )

    elif "does the fibonacci sequence ever repeat values" in q:
        return (
            "♻️ Only the number 1 is repeated (at F(1) and F(2)).\n\n"
            "After that, all terms are unique."
        )

    elif "what would happen if the fibonacci formula added three previous terms" in q:
        return (
            "➕ You’d get the **Tribonacci sequence**:\n\n"
            "F(n) = F(n-1) + F(n-2) + F(n-3)\n\n"
            "✅ It grows faster than the regular Fibonacci sequence."
        )

    elif "why the fibonacci sequence is important in understanding recursion" in q:
        return (
            "🔁 It clearly shows how recursion works:\n\n"
            "- Base cases\n"
            "- Recursive calls\n"
            "- Overlapping subproblems\n\n"
            "📘 That’s why it’s used in teaching recursion concepts."
        )

    else:
        return "❌ Question not recognized in Fibonacci Level 1 Conceptual bank."

def answer_conceptual_fibonacci_lvl2(question):
    q = question.lower().strip()

    if "naive recursive fibonacci algorithm lead to redundant calculations" in q:
        return (
            "🔁 The naive recursive method recalculates the same Fibonacci values multiple times.\n\n"
            "For example, to compute F(5), it separately computes F(4) and F(3), but F(3) is computed again within F(4).\n"
            "This creates a lot of unnecessary repeated work.\n"
            "❌ That’s why the time complexity grows exponentially."
        )

    elif "memoization improve the efficiency" in q:
        return (
            "📦 Memoization stores results of previous Fibonacci calls in a lookup table.\n\n"
            "This prevents recalculating the same values, saving time and reducing recursion depth.\n"
            "It transforms an exponential-time algorithm into a linear one.\n"
            "✅ Much more efficient for large input sizes."
        )

    elif "time complexity of calculating fibonacci numbers with memoization" in q:
        return (
            "⏱ With memoization, the time complexity is O(n).\n\n"
            "Each Fibonacci number from F(0) to F(n) is calculated once and stored.\n"
            "Subsequent calls use the stored results directly.\n"
            "✅ This makes it scalable even for large n."
        )

    elif "space complexity change when using memoization vs iteration" in q:
        return (
            "💾 Memoization uses O(n) space to store previously computed values in a cache.\n\n"
            "Iterative solutions, on the other hand, can be optimized to use O(1) space.\n"
            "So, iteration is better in memory-constrained environments.\n"
            "✅ Tradeoff: memoization is easier to understand; iteration is more memory efficient."
        )

    elif "pros and cons of recursive vs iterative" in q:
        return (
            "🧠 Recursive Fibonacci is elegant and mirrors the mathematical definition.\n\n"
            "But it's slow and memory-heavy unless optimized with memoization.\n"
            "✅ Iterative is faster and uses less memory.\n"
            "❌ Recursive can cause stack overflow for large n."
        )

    elif "modify the fibonacci base cases to fib(0) = 1 and fib(1) = 1" in q:
        return (
            "🔁 Changing base cases to F(0) = 1 and F(1) = 1 shifts the entire sequence.\n\n"
            "Instead of 0, 1, 1, 2, 3..., it becomes 1, 1, 2, 3, 5...\n"
            "This variation is still valid and is called the 'shifted Fibonacci' sequence.\n"
            "📌 It depends on application context."
        )

    elif "why is dynamic programming a good fit" in q:
        return (
            "✅ Dynamic Programming (DP) is ideal for Fibonacci because it solves overlapping subproblems.\n\n"
            "Instead of recalculating, DP stores and reuses results.\n"
            "This makes the algorithm both fast and efficient.\n"
            "📘 Fibonacci is one of the simplest introductions to DP."
        )

    elif "difference between top-down and bottom-up" in q:
        return (
            "🔼 Top-down: uses recursion + memoization (starts from F(n), works down).\n"
            "🔽 Bottom-up: uses iteration (starts from F(0), builds up to F(n)).\n\n"
            "Top-down is easier to understand; bottom-up is often more space-efficient."
        )

    elif "overlapping subproblems appear in the naive fibonacci" in q:
        return (
            "🔁 Overlapping subproblems happen when the same function calls repeat.\n\n"
            "For example, F(5) calls F(4) and F(3), but F(3) is called again inside F(4).\n"
            "This redundancy is what dynamic programming avoids."
        )

    elif "storing previously computed fibonacci values" in q:
        return (
            "📦 Storing computed values prevents the need to go deeper into recursive calls.\n\n"
            "This reduces the recursion tree size dramatically.\n"
            "✅ It limits the depth to O(n), saving both time and memory."
        )

    elif "why is the fibonacci problem often used to introduce dynamic programming" in q:
        return (
            "📘 The Fibonacci problem is simple, yet shows the power of dynamic programming clearly.\n\n"
            "It has overlapping subproblems and optimal substructure.\n"
            "It teaches memoization and bottom-up techniques in a beginner-friendly way."
        )

    elif "explain the fibonacci recurrence to someone with no programming background" in q:
        return (
            "📖 Imagine you’re counting rabbits: every month, new rabbits are born from existing ones.\n\n"
            "The number of rabbits in a month is just the sum of the last two months.\n"
            "That’s the Fibonacci idea — each number is built from the two before it."
        )

    elif "importance of avoiding recomputation" in q:
        return (
            "🚫 Recomputing values wastes time and computing power.\n\n"
            "By caching previous Fibonacci values, we solve each only once.\n"
            "✅ This greatly improves performance, especially for large n."
        )

    elif "recursion still be preferred over iteration" in q:
        return (
            "🔁 Recursion may be preferred for educational purposes, cleaner code, or functional languages.\n\n"
            "In small input ranges or when readability matters more than performance, recursion is okay.\n"
            "However, it's less efficient for large n."
        )

    elif "fibonacci sequence relates to tree recursion" in q:
        return (
            "🌳 The naive Fibonacci recursion forms a binary tree.\n\n"
            "Each node spawns two subcalls: F(n-1) and F(n-2).\n"
            "This tree grows rapidly, leading to exponential complexity."
        )

    elif "changing the recurrence relation affect the growth pattern" in q:
        return (
            "🔁 Changing the recurrence changes how fast the numbers grow.\n\n"
            "For example, adding 3 previous terms creates the 'Tribonacci' sequence, which grows faster.\n"
            "Each modification defines a different growth curve."
        )

    elif "introduced a negative index in the fibonacci sequence" in q:
        return (
            "➖ For negative indices, there's a concept called 'Negafibonacci'.\n\n"
            "The pattern becomes: F(-n) = (-1)^{n+1} × F(n)\n"
            "So Fibonacci numbers can be extended to negative indices mathematically."
        )

    elif "inefficient to solve fibonacci problems with plain recursion" in q:
        return (
            "⚠️ Plain recursion does not remember past results.\n\n"
            "It recalculates the same values repeatedly.\n"
            "This makes it very slow for large n — with time complexity O(2^n)."
        )

    elif "exponential growth pattern of fibonacci values" in q:
        return (
            "📈 Fibonacci numbers grow exponentially with n.\n\n"
            "In fact, F(n) ≈ φ^n / √5, where φ ≈ 1.618 (the golden ratio).\n"
            "Each term gets roughly 1.6× bigger than the last."
        )

    elif "real-world limitations of using recursive fibonacci algorithms" in q:
        return (
            "🚫 Recursive Fibonacci is inefficient for real software systems.\n\n"
            "It can cause stack overflows, high memory use, and long delays.\n"
            "✅ For real applications, iterative or dynamic programming solutions are preferred."
        )

    else:
        return "❌ Question not recognized in Fibonacci Level 2 Conceptual bank."

def answer_conceptual_fibonacci_lvl3(question):
    q = question.lower().strip()

    if "matrix exponentiation approach to fibonacci" in q:
        return (
            "📐 Matrix exponentiation transforms the Fibonacci recurrence into matrix form,\n"
            "using the identity: [[F(n)], [F(n-1)]] = [[1,1],[1,0]]^(n-1).\n"
            "By applying fast exponentiation to this 2x2 matrix, we reduce the number of operations\n"
            "from linear O(n) to logarithmic O(log n), which is significantly faster for large values of n.\n"
            "This approach is highly efficient and often used when working with large Fibonacci computations\n"
            "in competitive programming and cryptography, especially under modular arithmetic."
        )

    elif "golden ratio in the closed-form expression" in q:
        return (
            "🌟 The golden ratio φ ≈ 1.618 is a key component of Binet’s formula: F(n) = (φ^n - ψ^n)/√5.\n"
            "As n increases, the influence of ψ ≈ -0.618 vanishes due to its negative fractional exponent.\n"
            "This makes φ^n/√5 a close approximation of F(n).\n"
            "The appearance of φ links Fibonacci numbers with natural growth patterns, spirals, and art.\n"
            "It also reveals the exponential nature of Fibonacci's growth."
        )

    elif "derivation and meaning of binet" in q:
        return (
            "📘 Binet’s formula is derived by solving the Fibonacci recurrence relation\n"
            "using characteristic equations in linear algebra.\n"
            "It expresses F(n) = (φ^n - ψ^n)/√5, where φ and ψ are roots of the equation x² = x + 1.\n"
            "Though it uses irrational numbers, the formula produces exact integers due to cancellation.\n"
            "It offers a direct way to compute Fibonacci numbers without loops or recursion."
        )

    elif "exponential time complexity when solved with plain recursion" in q:
        return (
            "⚠️ The plain recursive algorithm calculates F(n) = F(n-1) + F(n-2),\n"
            "but it recomputes the same values over and over.\n"
            "The number of calls doubles with each step, forming an exponential tree of calls.\n"
            "This leads to a time complexity of O(2^n), making it very inefficient for large n."
        )

    elif "overlapping subproblems and optimal substructure" in q:
        return (
            "🔁 Overlapping subproblems occur when the same function (like F(n-1))\n"
            "is called multiple times in different parts of the recursion.\n"
            "🧱 Optimal substructure means the solution to F(n) can be constructed\n"
            "by combining solutions of F(n-1) and F(n-2).\n"
            "These properties make the problem suitable for dynamic programming."
        )

    elif "intuition behind reducing the fibonacci problem to logarithmic time" in q:
        return (
            "🧠 Instead of adding terms one by one, matrix exponentiation jumps forward by\n"
            "using powers of a transformation matrix.\n"
            "This exploits the mathematical structure of the recurrence to combine steps.\n"
            "It works like binary exponentiation—cutting the problem size in half each time."
        )

    elif "tail recursion" in q:
        return (
            "🔚 In tail recursion, the recursive call is the last operation in the function.\n"
            "By passing the results of F(n-1) and F(n-2) as arguments,\n"
            "we avoid stacking new calls.\n"
            "If the language supports tail call optimization, it transforms recursion into a loop,\n"
            "saving memory and preventing stack overflow."
        )

    elif "last two fibonacci values" in q:
        return (
            "💾 Storing only the last two Fibonacci numbers is sufficient\n"
            "because F(n) depends only on F(n-1) and F(n-2).\n"
            "This reduces space from O(n) to O(1) while still computing correct results.\n"
            "It's especially useful when memory is limited, such as in embedded devices."
        )

    elif "floating-point arithmetic in binet" in q:
        return (
            "🔢 Binet’s formula uses irrational numbers and exponentiation,\n"
            "which involves floating-point approximations.\n"
            "For large n, rounding errors accumulate and can result in incorrect integers.\n"
            "That’s why it’s better to use integer-based algorithms for high-precision needs."
        )

    elif "closest fibonacci number less than a given integer" in q:
        return (
            "🔍 Generate Fibonacci numbers iteratively until one exceeds n.\n"
            "Keep track of the last valid number.\n"
            "Alternatively, use binary search on a precomputed list of Fibonacci numbers\n"
            "to find the closest one below the target value."
        )

    elif "compute fibonacci modulo" in q:
        return (
            "🔢 When computing large Fibonacci numbers with modulo (like 10^9+7),\n"
            "apply modulo at each step: F(n) = (F(n-1) + F(n-2)) % mod.\n"
            "This avoids integer overflow and keeps results bounded.\n"
            "It also allows using fast matrix exponentiation with modular arithmetic."
        )

    elif "handle very large numbers using biginteger" in q:
        return (
            "📈 Languages like Python handle big integers natively.\n"
            "For others like Java, use BigInteger class.\n"
            "The algorithm remains the same (DP or iterative), but with big number data types\n"
            "to avoid overflow and ensure accurate results for huge n."
        )

    elif "fibonacci sequence show up in algorithm analysis" in q:
        return (
            "🔍 In AVL trees, the minimum number of nodes grows as a Fibonacci sequence.\n"
            "Fibonacci also appears in dynamic programming, search trees, and recursion analysis.\n"
            "It’s a benchmark for understanding how recursive growth behaves."
        )

    elif "generalized to higher-order recurrences" in q:
        return (
            "🔁 Yes. The Fibonacci sequence can be extended to Tribonacci, Tetranacci, etc.,\n"
            "where each term depends on more previous terms.\n"
            "This adds complexity but models more detailed systems.\n"
            "It also increases memory and base case requirements."
        )

    elif "justify the use of dynamic programming for fibonacci" in q:
        return (
            "💡 Explain that dynamic programming remembers answers to small problems\n"
            "so we don’t solve them again.\n"
            "It’s like writing down past answers so we only work once.\n"
            "This reduces time drastically and shows the benefit of planning ahead."
        )

    elif "memoization differs from tabulation" in q:
        return (
            "📚 Memoization is top-down: it uses recursion and caches answers.\n"
            "Tabulation is bottom-up: it fills a table from base cases up.\n"
            "Memoization is easier to write, while tabulation uses less stack and is often faster."
        )

    elif "minimum space complexity achievable" in q:
        return (
            "📦 Since only the last two values are used at each step,\n"
            "we can use two variables instead of an array.\n"
            "This brings space complexity down to O(1).\n"
            "It’s the most efficient approach in terms of memory."
        )

    elif "sliding window array optimize space" in q:
        return (
            "🔄 A sliding window keeps track of just the last two (or k) values needed.\n"
            "We overwrite values as we move forward, saving space.\n"
            "It’s a key technique in optimizing space in DP."
        )

    elif "iterative fibonacci preferred in embedded systems" in q:
        return (
            "💾 Embedded systems have limited memory and no stack support.\n"
            "Iterative Fibonacci runs with constant space and no recursion.\n"
            "It’s efficient, safe, and predictable—perfect for hardware-level programming."
        )

    elif "dp principles from fibonacci to unrelated problems" in q:
        return (
            "📘 Dynamic programming in Fibonacci teaches breaking problems into\n"
            "smaller overlapping subproblems.\n"
            "This idea applies in text processing, route planning, or finance.\n"
            "Fibonacci is often the first step to mastering DP logic."
        )

    else:
        return "❌ Question not recognized in Fibonacci Level 3 Conceptual bank."


def answer_conceptual_fibonacci(level, question):
    if level == "Level 1":
        return answer_conceptual_fibonacci_lvl1(question)
    elif level == "Level 2":
        return answer_conceptual_fibonacci_lvl2(question)
    elif level == "Level 3":
        return answer_conceptual_fibonacci_lvl3(question)
    else:
        return "No answer for this level."
