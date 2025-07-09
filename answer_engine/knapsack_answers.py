import re
import math

# --- HELPER FUNCTIONS (START) ---
# These helper functions are crucial for the main answer generation logic.

def parse_items_string(items_str):
    """
    Parses a string representation of items into a list of dictionaries.
    Supports formats like "A(w=1,v=10), B(w=2,v=20)" or "(1,10), (2,20)".
    """
    items_list = []
    # Try parsing as simple (weight, value) tuples first
    item_tuples = re.findall(r'\(([^,]+?),\s*([^)]+?)\)', items_str)
    if item_tuples:
        is_simple_tuple = True
        for w_str, v_str in item_tuples:
            if not (w_str.strip().isdigit() and v_str.strip().isdigit()):
                is_simple_tuple = False
                break
        
        if is_simple_tuple:
            for i, (w, v) in enumerate(item_tuples):
                items_list.append({'name': f"Item{i+1}", 'weight': int(w), 'value': int(v)})
            return items_list

    # Fallback to parsing named items like A(w=1,v=10)
    # This regex is more robust for "weight=" or "w=" and "value=" or "v="
    item_pattern = re.compile(r'([A-Za-z]?)(\w*)\(w(?:eight)?=(\d+),\s*v(?:alue)?=(\d+)\)')
    
    current_item_idx = 0
    for match in item_pattern.finditer(items_str):
        name_prefix = match.group(1)
        name_body = match.group(2)
        weight = int(match.group(3))
        value = int(match.group(4))
        
        # Construct a name: use prefix, then body, or default if neither.
        name = name_prefix + name_body if name_prefix or name_body else f"Item{current_item_idx+1}"
        items_list.append({'name': name, 'weight': weight, 'value': value})
        current_item_idx += 1
        
    return items_list


def knapsack_dp_solve(capacity, items):
    """
    Solves the 0/1 Knapsack problem using bottom-up dynamic programming.
    Returns the filled DP table and the maximum value.
    """
    n = len(items)
    # dp[i][w] will store the maximum value that can be attained with capacity `w`
    # and considering first `i` items.
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)] #

    for i in range(1, n + 1):
        weight_i = items[i-1]['weight']
        value_i = items[i-1]['value']
        for w in range(capacity + 1):
            # Case 1: Item `i-1` (current item) is not included
            val_not_included = dp[i-1][w] #

            # Case 2: Item `i-1` is included (if it fits)
            val_included = -1 # Use -1 to represent impossible inclusion
            if weight_i <= w:
                val_included = value_i + dp[i-1][w - weight_i] #
            
            dp[i][w] = max(val_not_included, val_included) #
    
    return dp, dp[n][capacity] # Return full DP table and max value


def knapsack_reconstruct_items(dp_table, items, capacity):
    """
    Reconstructs the list of items selected to achieve the maximum value from the DP table.
    Assumes the DP table was filled using knapsack_dp_solve.
    """
    n = len(items)
    selected_items = []
    w = capacity
    
    # Iterate backwards from the last item and the max capacity
    for i in range(n, 0, -1):
        # If the value at dp[i][w] is different from dp[i-1][w], it means item (i-1) was included.
        # This is because if item (i-1) was NOT included, dp[i][w] would be equal to dp[i-1][w].
        if dp_table[i][w] != dp_table[i-1][w]:
            current_item_weight = items[i-1]['weight']
            current_item_value = items[i-1]['value']
            
            # This condition ensures the item was truly taken, not just that values diverged
            if w >= current_item_weight and dp_table[i][w] == current_item_value + dp_table[i-1][w - current_item_weight]:
                selected_items.append(items[i-1]['name']) # Add the name of the item
                w -= items[i-1]['weight'] # Reduce capacity by the weight of the included item
    
    selected_items.reverse() # Reverse to get items in their original order
    return selected_items

# --- HELPER FUNCTIONS (END) ---


# --- MAIN ANSWER GENERATION FUNCTIONS (START) ---

def answer_algorithmic_knapsack(level, question):
    q = question.lower()

    if level == "Level 1":
        # 1. Given a knapsack with capacity {{capacity}} and one item of weight {{weight}} and value {{value}}, what is the maximum value that can be obtained?
        if "one item of weight" in q:
            match = re.search(r"capacity (\d+) and one item of weight (\d+) and value (\d+)", q)
            if match:
                capacity = int(match.group(1))
                weight = int(match.group(2))
                value = int(match.group(3))

                can_include = weight <= capacity
                max_value = value if can_include else 0

                return (
                    f"🔍 **Problem:**\n"
                    f"You are given a knapsack with capacity `{capacity}` and one item with weight `{weight}` and value `{value}`.\n"
                    f"You need to find the **maximum value** that can be obtained.\n\n"
                    f"📘 **Understanding:**\n"
                    f"This is a basic case of the 0/1 Knapsack problem, where you are allowed to either include the item (if it fits) or exclude it.\n\n"
                    f"📐 **Check if item fits:**\n"
                    f"- Does `{weight} <= {capacity}`? {'✅ Yes' if can_include else '❌ No'}\n\n"
                    f"📦 **Decision:**\n"
                    f"- If the item fits in the knapsack, take it.\n"
                    f"- If not, the maximum value is 0.\n\n"
                    f"🧠 **Computation:**\n"
                    f"```python\n"
                    f"if weight <= capacity:\n"
                    f"    max_value = value\n"
                    f"else:\n"
                    f"    max_value = 0\n"
                    f"```\n\n"
                    f"✅ **Final Answer:** `{max_value}`"
                )

        # 2. If you have a knapsack of capacity {{capacity}} and two items: item1(weight={{w1}}, value={{v1}}), item2(weight={{w2}}, value={{v2}}), what is the best value you can carry?
        elif "two items: item1" in q:
            match = re.search(r"capacity (\d+).*item1\(weight=(\d+),\s*value=(\d+)\).*item2\(weight=(\d+),\s*value=(\d+)\)", q)
            if match:
                capacity = int(match.group(1))
                w1, v1 = int(match.group(2)), int(match.group(3))
                w2, v2 = int(match.group(4)), int(match.group(5))

                items = [{'name': 'item1', 'weight': w1, 'value': v1}, {'name': 'item2', 'weight': w2, 'value': v2}]
                _, max_val = knapsack_dp_solve(capacity, items)
                return (
                    f"Given knapsack capacity {capacity} and two items: Item1(weight={w1}, value={v1}), Item2(weight={w2}, value={v2}).\n"
                    f"We consider all combinations that fit:\n"
                    f"- No items: Value = 0\n"
                    f"- Only Item1: Value = {v1} (if {w1} <= {capacity})\n"
                    f"- Only Item2: Value = {v2} (if {w2} <= {capacity})\n"
                    f"- Both (Item1 + Item2): Value = {v1 + v2} (if {w1 + w2} <= {capacity})\n"
                    f"The best value you can carry is: **{max_val}**."
                )
        
        # 3. Determine whether to include an item with weight {{weight}} and value {{value}} in a knapsack of capacity {{capacity}}.
        if "determine whether to include an item with weight" in q:
            match = re.search(r"weight (\d+) and value (\d+) in a knapsack of capacity (\d+)", q)
            if match:
                w, v, c = map(int, match.groups())
                include = w <= c

                return (
                    f"🔍 **Problem:**\n"
                    f"Check if you can include an item with weight `{w}` and value `{v}` in a knapsack of capacity `{c}`.\n\n"
                    f"📘 **Logic:**\n"
                    f"```python\n"
                    f"if {w} <= {c}:\n"
                    f"    return True  # include\n"
                    f"else:\n"
                    f"    return False  # exclude\n"
                    f"```\n\n"
                    f"✅ **Final Answer:** {'Include' if include else 'Do not include'} the item."
                )

        # 4. Find the optimal value that can be achieved by choosing between two items (weight={{w1}}, value={{v1}}) and (weight={{w2}}, value={{v2}}) with knapsack capacity {{capacity}}.
        if "find the optimal value" in q and "two items" in q:
            match = re.search(r"\(weight=(\d+), value=(\d+)\).*\(weight=(\d+), value=(\d+)\).*capacity (\d+)", q, re.DOTALL)
            if match:
                w1, v1, w2, v2, c = map(int, match.groups())
                explanation = []
                options = []
                if w1 <= c:
                    options.append((v1, "Item1"))
                    explanation.append(f"✅ Item1 fits → Value = {v1}")
                else:
                    explanation.append(f"❌ Item1 too heavy")
                if w2 <= c:
                    options.append((v2, "Item2"))
                    explanation.append(f"✅ Item2 fits → Value = {v2}")
                else:
                    explanation.append(f"❌ Item2 too heavy")
                if w1 + w2 <= c:
                    options.append((v1 + v2, "Item1 + Item2"))
                    explanation.append(f"✅ Both fit together → Value = {v1 + v2}")
                else:
                    explanation.append(f"❌ Together too heavy")
                if not options:
                    return "❌ None of the items fit. Maximum value = 0."
                max_val, selected = max(options)
                return (
                    f"🔍 **Problem:** Choose optimal value between two items with knapsack capacity = {c}\n\n"
                    f"📘 **Options:**\n{chr(10).join(explanation)}\n\n"
                    f"🧠 **Algorithm Steps:**\n"
                    f"1. Check if Item1 fits\n"
                    f"2. Check if Item2 fits\n"
                    f"3. Check if both fit\n"
                    f"4. Choose max value from valid options\n\n"
                    f"✅ **Final Answer:**\n- Value = {max_val}, Selection = {selected}"
                )
        
        # 5. You are given 1 item with weight {{weight}} and value {{value}}. Can it fit into a knapsack of capacity {{capacity}}?
        if "can it fit into a knapsack" in q:
            match = re.search(r"weight (\d+) and value (\d+).*capacity (\d+)", q)
            if match:
                w, v, c = map(int, match.groups())
                fits = w <= c
                return (
                    f"🔍 **Problem:** Can an item with weight = {w}, value = {v} fit into a knapsack of capacity = {c}?\n\n"
                    f"📘 **Logic:**\n"
                    f"```python\n"
                    f"if {w} <= {c}:\n"
                    f"    return True\n"
                    f"else:\n"
                    f"    return False\n"
                    f"```\n\n"
                    f"✅ **Final Answer:** {'Yes, it fits.' if fits else 'No, it does not fit.'}"
                )

        # 6. If you are allowed to select only one item, which should you choose to maximize value: (weight={{w1}}, value={{v1}}) or (weight={{w2}}, value={{v2}}) for capacity {{capacity}}?
        elif "select only one item" in q:
            match = re.search(r"\(weight=(\d+),\s*value=(\d+)\) or \(weight=(\d+),\s*value=(\d+)\) for capacity (\d+)", q)
            if match:
                w1, v1 = int(match.group(1)), int(match.group(2))
                w2, v2 = int(match.group(3)), int(match.group(4))
                capacity = int(match.group(5))

                value1 = v1 if w1 <= capacity else -1 # Use -1 to indicate not fitting
                value2 = v2 if w2 <= capacity else -1
                
                if value1 == -1 and value2 == -1:
                    choice = "Neither, no item fits."
                    max_val = 0
                elif value1 >= value2:
                    choice = f"Item1 (weight={w1}, value={v1})"
                    max_val = value1
                else:
                    choice = f"Item2 (weight={w2}, value={v2})"
                    max_val = value2

                return (
                    f"Given knapsack capacity {capacity} and allowed to select only one item (Item1: w={w1}, v={v1}; Item2: w={w2}, v={v2}):\n"
                    f"Value if Item1 selected: {v1 if w1 <= capacity else 'Does not fit'}\n"
                    f"Value if Item2 selected: {v2 if w2 <= capacity else 'Does not fit'}\n"
                    f"To maximize value, you should choose: **{choice}**. Maximum value: **{max_val}**."
                )

        # 7. What is the total value if you put item1(weight={{w1}}, value={{v1}}) into a knapsack with capacity {{capacity}}?
        elif "put item1" in q and "into a knapsack with capacity" in q:
            match = re.search(r"item1\(weight=(\d+),\s*value=(\d+)\) into a knapsack with capacity (\d+)", q)
            if match:
                w1, v1 = int(match.group(1)), int(match.group(2))
                capacity = int(match.group(3))
                
                total_val = v1 if w1 <= capacity else 0
                message = "The item fits." if w1 <= capacity else "The item does not fit."
                return (
                    f"If you put Item1(weight={w1}, value={v1}) into a knapsack with capacity {capacity}:\n"
                    f"{message} Total value: **{total_val}**."
                )
        
        # 8. Out of three items (w,v): ({{w1}},{{v1}}), ({{w2}},{{v2}}), and ({{w3}},{{v3}}), which two would you choose for capacity {{capacity}}?
        elif "out of three items" in q and "which two would you choose" in q:
            match = re.search(r"\((\d+),(\d+)\), \((\d+),(\d+)\), and \((\d+),(\d+)\), which two would you choose for capacity (\d+)", q)
            if match:
                w1, v1 = int(match.group(1)), int(match.group(2))
                w2, v2 = int(match.group(3)), int(match.group(4))
                w3, v3 = int(match.group(5)), int(match.group(6))
                capacity = int(match.group(7))

                items = [
                    {'name': 'Item1', 'weight': w1, 'value': v1},
                    {'name': 'Item2', 'weight': w2, 'value': v2},
                    {'name': 'Item3', 'weight': w3, 'value': v3}
                ]

                # Check all combinations of two items
                best_value = 0
                best_pair = "None"
                
                # Item1 + Item2
                if (w1 + w2) <= capacity:
                    if (v1 + v2) > best_value:
                        best_value = v1 + v2
                        best_pair = "Item1 and Item2"
                
                # Item1 + Item3
                if (w1 + w3) <= capacity:
                    if (v1 + v3) > best_value:
                        best_value = v1 + v3
                        best_pair = "Item1 and Item3"
                
                # Item2 + Item3
                if (w2 + w3) <= capacity:
                    if (v2 + v3) > best_value:
                        # Re-check condition to correctly handle equal values if they come after a non-zero best_value
                        if (v2 + v3) > best_value:
                            best_value = v2 + v3
                            best_pair = "Item2 and Item3"
                        # If values are equal, keep current best_pair unless specific tie-breaking (e.g., lighter weight) is needed.
                        # For now, if equal, the first found one will be kept.
                
                return (
                    f"Out of three items (Item1: ({w1},{v1}), Item2: ({w2},{v2}), Item3: ({w3},{v3})) for capacity {capacity}:\n"
                    f"To maximize value by choosing two items, you would choose: **{best_pair}**.\n"
                    f"Resulting total value: **{best_value}**."
                )

        # 9. Can a knapsack of capacity {{capacity}} hold both item A(weight={{w1}}) and item B(weight={{w2}})?
        elif "can a knapsack of capacity" in q and "hold both item a" in q:
            match = re.search(r"capacity (\d+) hold both item a\(weight=(\d+)\) and item b\(weight=(\d+)\)", q)
            if not match:
                return "❌ Could not extract knapsack capacity or item weights from the question."

            capacity = int(match.group(1))
            w1 = int(match.group(2))
            w2 = int(match.group(3))
            total_weight = w1 + w2
            can_hold = total_weight <= capacity

            return (
                f"🔍 **Problem Statement:**\n"
                f"Can a knapsack of capacity `{capacity}` hold both Item A (weight={w1}) and Item B (weight={w2})?\n\n"

                f"📘 **Concept:**\n"
                f"In the 0/1 Knapsack problem, each item is either taken completely or not at all.\n"
                f"To include both items, their combined weight must not exceed the knapsack's capacity.\n\n"

                f"🧠 **Step-by-step Logic:**\n"
                f"1. Compute the total weight: `{w1} + {w2} = {total_weight}`\n"
                f"2. Compare total weight with knapsack capacity: Is `{total_weight} <= {capacity}`?\n"
                f"   - {'✅ Yes' if can_hold else '❌ No'}\n\n"

                f"🧪 **Python Code:**\n"
                "```python\n"
                f"capacity = {capacity}\n"
                f"weight_a = {w1}\n"
                f"weight_b = {w2}\n"
                f"can_hold_both = (weight_a + weight_b) <= capacity\n"
                f"print('Can hold both items?', can_hold_both)\n"
                "```\n\n"

                f"📤 **Result:**\n"
                f"- Total weight of both items: `{total_weight}`\n"
                f"- Knapsack capacity: `{capacity}`\n"
                f"- {'✅ YES' if can_hold else '❌ NO'}, the knapsack {'can' if can_hold else 'cannot'} hold both items."
            )


        # 10. Find the maximum value using only the first {{n}} items in a list of items, given a knapsack of capacity {{capacity}}.
        elif "maximum value using only the first" in q and "items in a list of items" in q:

            match = re.search(r"the first (\d+) items in a list of items, given a knapsack of capacity (\d+)", question)
            if match:
                n, capacity = map(int, match.groups())

                # ✅ Make sure we have at least `n` items
                values = [10, 40, 30, 50, 35, 25, 45, 60, 55, 20, 70]
                weights = [1, 3, 4, 5, 2, 6, 3, 7, 4, 2, 8]

                def knapsack(values, weights, n, capacity):
                    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

                    for i in range(1, n + 1):
                        for w in range(capacity + 1):
                            if weights[i - 1] <= w:
                                dp[i][w] = max(
                                    dp[i - 1][w],
                                    dp[i - 1][w - weights[i - 1]] + values[i - 1]
                                )
                            else:
                                dp[i][w] = dp[i - 1][w]
                    return dp[n][capacity]

                result = knapsack(values, weights, n, capacity)

                return (
                    f"🎯 **0/1 Knapsack Problem**\n"
                    f"- Use only the **first {n} items**.\n"
                    f"- Knapsack Capacity = **{capacity}**\n\n"
                    f"🧾 **Input Lists**:\n"
                    f"- Values: {values[:n]}\n"
                    f"- Weights: {weights[:n]}\n\n"
                    f"⚙️ **Code Logic**:\n"
                    "```python\n"
                    "def knapsack(values, weights, n, capacity):\n"
                    "    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n"
                    "    for i in range(1, n + 1):\n"
                    "        for w in range(capacity + 1):\n"
                    "            if weights[i - 1] <= w:\n"
                    "                dp[i][w] = max(\n"
                    "                    dp[i - 1][w],\n"
                    "                    dp[i - 1][w - weights[i - 1]] + values[i - 1]\n"
                    "                )\n"
                    "            else:\n"
                    "                dp[i][w] = dp[i - 1][w]\n"
                    "    return dp[n][capacity]\n"
                    "```\n\n"
                    f"✅ **Maximum value achievable using first {n} items**: **{result}**"
                )



        # 11. How much unused space remains if item(weight={{weight}}) is added to a knapsack of capacity {{capacity}}?
        elif "unused space remains if item(weight" in q:
            match = re.search(r"item\(weight=(\d+)\) is added to a knapsack of capacity (\d+)", q)
            if match:
                weight = int(match.group(1))
                capacity = int(match.group(2))
                
                if weight <= capacity:
                    unused_space = capacity - weight
                    return (
                        f"🔍 **Problem:**\n"
                        f"How much unused space remains if an item of weight `{weight}` is added to a knapsack of capacity `{capacity}`?\n\n"

                        f"📘 **Concept:**\n"
                        f"The unused space is the remaining room after placing the item. If the item fits, it's `capacity - weight`.\n"
                        f"If it doesn't fit, the knapsack remains empty.\n\n"

                        f"🧠 **Calculation:**\n"
                        f"- Item weight = {weight}\n"
                        f"- Knapsack capacity = {capacity}\n"
                        f"- Remaining space = {capacity} - {weight} = {unused_space}\n\n"

                        f"🧪 **Code:**\n"
                        f"```python\n"
                        f"capacity = {capacity}\n"
                        f"item_weight = {weight}\n"
                        f"unused = capacity - item_weight if item_weight <= capacity else capacity\n"
                        f"print(unused)\n"
                        f"```\n\n"

                        f"✅ **Final Answer:** Unused space = **{unused_space}**."
                    )
                else:
                    exceeded = weight - capacity
                    return (
                        f"🔍 **Problem:**\n"
                        f"Can an item of weight `{weight}` be added to a knapsack of capacity `{capacity}`?\n\n"
                        f"❌ The item is **too large** to fit.\n\n"
                        f"📘 **Note:** If forced, it would exceed the capacity by `{exceeded}` units.\n\n"
                        f"✅ **Final Answer:** Item doesn't fit. Unused space = **{capacity}**."
                    )


        # 12. Can both items with weights {{w1}} and {{w2}} be selected if the capacity is {{capacity}}?
        elif "can both items with weights" in q and "be selected if the capacity is" in q:
            match = re.search(r"weights (\d+) and (\d+) be selected if the capacity is (\d+)", q)
            if match:
                w1 = int(match.group(1))
                w2 = int(match.group(2))
                capacity = int(match.group(3))
                total = w1 + w2
                can_fit = total <= capacity

                return (
                    f"🔍 **Problem:**\n"
                    f"Can both items with weights `{w1}` and `{w2}` be selected if knapsack capacity is `{capacity}`?\n\n"
                    
                    f"🧠 **Logic:**\n"
                    f"- Combined weight: `{w1} + {w2} = {total}`\n"
                    f"- Compare with capacity: Is `{total} <= {capacity}`?\n"
                    f"- {'✅ Yes' if can_fit else '❌ No'}\n\n"

                    f"🧪 **Code:**\n"
                    "```python\n"
                    f"capacity = {capacity}\n"
                    f"total_weight = {w1} + {w2}\n"
                    f"can_take_both = total_weight <= capacity\n"
                    f"```\n\n"

                    f"✅ **Final Answer:** {'Yes, both can be selected.' if can_fit else 'No, they cannot both be selected.'}"
                )

        
        # 13. Which of the following fits best into the knapsack (capacity={{capacity}}): item A({{w1}},{{v1}}), item B({{w2}},{{v2}})?
        elif ("fits best into the knapsack" in q or "which of the following fits best" in q) and "item a" in q and "item b" in q:
            match = re.search(r"capacity=?(\d+)\)?[:\s]*item a\((\d+),(\d+)\), item b\((\d+),(\d+)\)", q)
            if match:
                capacity = int(match.group(1))
                w1, v1 = int(match.group(2)), int(match.group(3))
                w2, v2 = int(match.group(4)), int(match.group(5))

                items = [
                    {'name': 'A', 'weight': w1, 'value': v1},
                    {'name': 'B', 'weight': w2, 'value': v2}
                ]

                best_item = "None"
                max_value = 0
                for item in items:
                    if item['weight'] <= capacity and item['value'] > max_value:
                        best_item = item['name']
                        max_value = item['value']

                if best_item == "None":
                    return (
                        f"🔍 **Problem:**\n"
                        f"Which of Item A (weight={w1}, value={v1}) or Item B (weight={w2}, value={v2}) fits best in a knapsack of capacity {capacity}?\n\n"
                        f"📘 **Concept:**\n"
                        f"We want the item with the highest value that can fit within the given capacity.\n\n"
                        f"🧠 **Step-by-step:**\n"
                        f"- Check if each item fits:\n"
                        f"  - A: {'Fits' if w1 <= capacity else 'Too heavy'}\n"
                        f"  - B: {'Fits' if w2 <= capacity else 'Too heavy'}\n"
                        f"- Among the items that fit, choose the one with the highest value.\n\n"
                        f"❌ Neither item fits.\n"
                        f"✅ **Final Answer:** No item can be selected."
                    )
                else:
                    return (
                        f"🔍 **Problem:**\n"
                        f"Which item fits best into a knapsack of capacity {capacity} — A({w1},{v1}) or B({w2},{v2})?\n\n"
                        f"📘 **Concept:**\n"
                        f"Only items whose weight ≤ capacity are considered. From those, choose the item with maximum value.\n\n"
                        f"🧠 **Decision:**\n"
                        f"- A: {'Fits' if w1 <= capacity else 'Too heavy'}, Value = {v1}\n"
                        f"- B: {'Fits' if w2 <= capacity else 'Too heavy'}, Value = {v2}\n\n"
                        f"✅ **Final Answer:** Best choice is **Item {best_item}** with value **{max_value}**."
                    )



        
        # 14. Given two items (w,v): ({{w1}},{{v1}}), ({{w2}},{{v2}}) and capacity {{capacity}}, what is the best value achievable if only one can be selected?
        elif "given two items (w,v)" in q and "only one can be selected" in q:
            match = re.search(r"\((\d+),(\d+)\), \((\d+),(\d+)\) and capacity (\d+)", q)
            if match:
                w1, v1 = int(match.group(1)), int(match.group(2))
                w2, v2 = int(match.group(3)), int(match.group(4))
                cap = int(match.group(5))

                val1 = v1 if w1 <= cap else 0
                val2 = v2 if w2 <= cap else 0

                return (
                    f"🔍 **Problem:**\n"
                    f"Choose the better item from (w,v): ({w1},{v1}) or ({w2},{v2}) for capacity {cap}, selecting only one.\n\n"

                    f"🧠 **Comparison:**\n"
                    f"- Item1: {'Fits' if w1 <= cap else 'Too heavy'} → Value = {val1}\n"
                    f"- Item2: {'Fits' if w2 <= cap else 'Too heavy'} → Value = {val2}\n\n"

                    f"✅ **Final Answer:** Best achievable value = **{max(val1, val2)}**."
                )


        # 15. How many total items can you fit if each has weight {{item_weight}} and the knapsack capacity is {{capacity}}?
        elif "how many total items can you fit if each has weight" in q:
            match = re.search(r"weight (\d+) and the knapsack capacity is (\d+)", q)
            if match:
                item_weight = int(match.group(1))
                capacity = int(match.group(2))
                if item_weight <= 0:
                    return "⚠️ Item weight must be positive."
                num = capacity // item_weight
                return (
                    f"🔍 **Problem:**\n"
                    f"How many items of weight `{item_weight}` fit into capacity `{capacity}`?\n\n"

                    f"🧠 **Logic:**\n"
                    f"- Max count = capacity // weight = {capacity} // {item_weight} = {num}\n\n"

                    f"✅ **Final Answer:** You can fit **{num}** items."
                )


        # 16. If you have item A(weight={{w1}}, value={{v1}}) and item B(weight={{w2}}, value={{v2}}), which gives higher value per weight ratio?
        elif "higher value per weight ratio" in q:
            match = re.search(r"a\(weight=(\d+),\s*value=(\d+)\).*?b\(weight=(\d+),\s*value=(\d+)\)", q)
            if match:
                w1, v1 = int(match.group(1)), int(match.group(2))
                w2, v2 = int(match.group(3)), int(match.group(4))
                r1 = v1 / w1 if w1 > 0 else float('-inf')
                r2 = v2 / w2 if w2 > 0 else float('-inf')

                better = "A" if r1 > r2 else "B" if r2 > r1 else "Both"

                return (
                    f"🔍 **Problem:**\n"
                    f"Compare Item A (weight={w1}, value={v1}) and Item B (weight={w2}, value={v2}) for better value-to-weight ratio.\n\n"
                    f"📘 **Concept:**\n"
                    f"Value-to-weight ratio = value / weight. This helps in greedy selection and fractional knapsack problems.\n\n"
                    f"🧠 **Calculations:**\n"
                    f"- Item A: {v1} / {w1} = {r1:.2f}\n"
                    f"- Item B: {v2} / {w2} = {r2:.2f}\n\n"
                    f"✅ **Final Answer:** {'Item A' if r1 > r2 else 'Item B' if r2 > r1 else 'Both items have equal efficiency'} "
                    f"has the higher value/weight ratio."
                )




        # 17. Choose the best single item from the list: A({{w1}},{{v1}}), B({{w2}},{{v2}}), C({w3},{{v3}}) for a knapsack of capacity {{capacity}}.
        elif "choose the best single item from the list" in q:
            match = re.search(r"a\((\d+),(\d+)\), b\((\d+),(\d+)\), c\((\d+),(\d+)\).*?capacity (\d+)", q)
            if match:
                w1, v1 = int(match.group(1)), int(match.group(2))
                w2, v2 = int(match.group(3)), int(match.group(4))
                w3, v3 = int(match.group(5)), int(match.group(6))
                capacity = int(match.group(7))

                items = [
                    {'name': 'A', 'weight': w1, 'value': v1},
                    {'name': 'B', 'weight': w2, 'value': v2},
                    {'name': 'C', 'weight': w3, 'value': v3}
                ]

                best_item = "None"
                max_value = 0
                for item in items:
                    if item['weight'] <= capacity and item['value'] > max_value:
                        best_item = item['name']
                        max_value = item['value']

                if best_item == "None":
                    return (
                        f"🔍 **Problem:**\n"
                        f"Choose the best item from A({w1},{v1}), B({w2},{v2}), C({w3},{v3}) for capacity {capacity}.\n\n"
                        f"📘 **Concept:**\n"
                        f"From items that fit, pick the one with the highest value.\n\n"
                        f"❌ No item fits.\n"
                        f"✅ **Final Answer:** No item can be selected."
                    )
                else:
                    return (
                        f"🔍 **Problem:**\n"
                        f"Choose the best item from A({w1},{v1}), B({w2},{v2}), C({w3},{v3}) for knapsack capacity {capacity}.\n\n"
                        f"📘 **Concept:**\n"
                        f"From the items that fit, select the one with the highest value.\n\n"
                        f"🧠 **Step-by-step:**\n"
                        + "\n".join(
                            f"- Item {item['name']}: Weight={item['weight']}, Value={item['value']} → "
                            f"{'Fits' if item['weight'] <= capacity else 'Too heavy'}"
                            for item in items
                        ) + "\n\n"
                        f"✅ **Final Answer:** Best item = **Item {best_item}** with value **{max_value}**."
                    )



        # 18. What is the total weight of items selected if you choose item A(weight={{w1}}) and item B(weight={{w2}}) within capacity {{capacity}}?
        elif "total weight of items selected if you choose item a" in q and "and item b" in q:
            match= re.search(r"a\(weight=(\d+)\).*?b\(weight=(\d+)\).*?capacity (\d+)", q)

            if match:
                w1 = int(match.group(1))
                w2 = int(match.group(2))
                capacity = int(match.group(3))
                total = w1 + w2

                if total <= capacity:
                    return (
                        f"🔍 **Problem:**\n"
                        f"What's the total weight of Items A({w1}) and B({w2}) in a knapsack of capacity {capacity}?\n\n"
                        f"🧠 **Logic:**\n"
                        f"- Combined weight: {w1} + {w2} = {total}\n"
                        f"- Compare with capacity: Is {total} ≤ {capacity}?\n"
                        f"- ✅ Yes\n\n"
                        f"✅ **Final Answer:** Total weight = **{total}**."
                    )
                else:
                    return (
                        f"🔍 **Problem:**\n"
                        f"Can you select Items A({w1}) and B({w2}) in knapsack of capacity {capacity}?\n\n"
                        f"❌ Combined weight = {total} exceeds capacity.\n"
                        f"✅ **Final Answer:** Cannot be selected together."
                    )


        # 19. If each item has weight 1 and value 1, how many items can you choose to maximize total value with capacity {{capacity}}?
        elif "each item has weight 1 and value 1" in q:
            match = re.search(r"capacity (\d+)", q)
            if match:
                capacity = int(match.group(1))
                return (
                    f"🔍 **Problem:**\n"
                    f"If each item has weight 1 and value 1, how many can fit in knapsack of capacity {capacity}?\n\n"
                    f"🧠 **Logic:**\n"
                    f"- Each item takes 1 unit of space.\n"
                    f"- Max number of items = capacity // 1 = {capacity}\n\n"
                    f"✅ **Final Answer:** You can fit **{capacity}** items."
                )


        # 20. Can the knapsack of size {{capacity}} be exactly filled using two items with weights {{w1}} and {{w2}}?
        elif "exactly filled using two items with weights" in q:
            match = match = re.search(r"(?:capacity|size) (\d+).*?weights (\d+) and (\d+)", q)
            if match:
                capacity = int(match.group(1))
                w1 = int(match.group(2))
                w2 = int(match.group(3))
                total = w1 + w2
                exact = (total == capacity)

                return (
                    f"🔍 **Problem:**\n"
                    f"Can a knapsack of capacity {capacity} be **exactly** filled using items of weights {w1} and {w2}?\n\n"
                    f"🧠 **Logic:**\n"
                    f"- Total weight = {w1} + {w2} = {total}\n"
                    f"- Does this equal the knapsack capacity? → {'✅ Yes' if exact else '❌ No'}\n\n"
                    f"✅ **Final Answer:** {'Yes' if exact else 'No'}, the total weight {'matches' if exact else 'does not match'} the capacity."
                )


    elif level == "Level 2":
        # 1. Given a knapsack of capacity {{capacity}} and items: A(weight={{w1}}, value={{v1}}), B(weight={{w2}}, value={{v2}}), and C(weight={{w3}}, value={{v3}}), find the maximum value achievable.
        if "find the maximum value achievable" in q and "a(weight=" in q:
            match = re.search(
                r"(?:given a knapsack of )?capacity (\d+).*?a\(weight=(\d+),\s*value=(\d+)\).*?b\(weight=(\d+),\s*value=(\d+)\).*?c\(weight=(\d+),\s*value=(\d+)",
                q
            )
            if match:
                capacity = int(match.group(1))
                w1, v1 = int(match.group(2)), int(match.group(3))
                w2, v2 = int(match.group(4)), int(match.group(5))
                w3, v3 = int(match.group(6)), int(match.group(7))

                items = [{'name': 'A', 'weight': w1, 'value': v1}, {'name': 'B', 'weight': w2, 'value': v2}, {'name': 'C', 'weight': w3, 'value': v3}]
                _, max_val = knapsack_dp_solve(capacity, items)

                return (
                    f"🔍 **Problem:** Find the maximum value for a knapsack of capacity {capacity} using items:\n"
                    f"- A (weight={w1}, value={v1})\n- B (weight={w2}, value={v2})\n- C (weight={w3}, value={v3})\n\n"
                    f"📘 **Concept:** This is a 0/1 Knapsack problem, where each item can be taken once or not at all. "
                    f"We explore combinations to maximize the total value without exceeding the knapsack capacity.\n\n"
                    f"🧠 **Approach:**\n"
                    f"- Try all subsets (A, B, C, AB, AC, BC, ABC)\n"
                    f"- Use dynamic programming to evaluate and store optimal values for capacities.\n\n"
                    f"✅ **Final Answer:** The maximum value achievable is **{max_val}**."
                )


        
        # 2. Choose a combination of items that maximizes value without exceeding the knapsack capacity {{capacity}}. Items: (weight, value) - {{items}}.
        elif "choose a combination of items that maximizes value" in q:
            match = re.search(r"capacity (\d+).*?items: \(weight, value\) - (.+?)(?:\.|$)", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)

                dp_table, max_val = knapsack_dp_solve(capacity, items)
                selected = knapsack_reconstruct_items(dp_table, items, capacity)

                return (
                    f"🔍 **Problem:** Choose a combination of items that gives maximum value with capacity {capacity}.\n\n"
                    f"📘 **Concept:** This is a classic optimization problem. We want the best value from items without exceeding the knapsack's limit.\n\n"
                    f"🧠 **Approach:**\n"
                    f"- Each item has a weight and value.\n"
                    f"- Use dynamic programming to compare options and track best results.\n\n"
                    f"✅ **Result:**\n"
                    f"- Maximum value = **{max_val}**\n"
                    f"- Selected items = **{', '.join(selected)}**"
                )



        # 3. With a knapsack capacity of {{capacity}}, how many items from the list can be included: {{item_list}}?
        elif "how many items from the list can be included" in q:
            match = re.search(r"capacity of (\d+), how many items from the list can be included: (.+)\?", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)

                dp_table, _ = knapsack_dp_solve(capacity, items)
                selected_items = knapsack_reconstruct_items(dp_table, items, capacity)

                return (
                    f"🔍 **Problem:**\nGiven items {items_str} and capacity {capacity}, determine how many can be included.\n\n"
                    f"📘 **Concept:**\nThe goal is to maximize value, but we also count how many items contribute to this max.\n\n"
                    f"✅ **Answer:**\n"
                    f"- Max number of items = **{len(selected_items)}**\n"
                    f"- Selected = **{', '.join(selected_items)}**"
                )

        
        # 4. Find the optimal selection for a 0/1 Knapsack problem with capacity {{capacity}} using the items {{items}}.
        elif "find the optimal selection for a 0/1 knapsack problem" in q:
            match = re.search(r"capacity (\d+) using the items (.+)\.", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)

                dp_table, max_val = knapsack_dp_solve(capacity, items)
                selected_items_names = knapsack_reconstruct_items(dp_table, items, capacity)

                return (
                    f"🔍 **Problem:**\nFind the optimal subset of items {items_str} for capacity {capacity}.\n\n"
                    f"📘 **Method:**\nStandard bottom-up 0/1 knapsack DP. No item can be reused.\n\n"
                    f"✅ **Result:**\n"
                    f"- Max value = **{max_val}**\n"
                    f"- Selected items = **{', '.join(selected_items_names)}**"
                )


        # 5. Determine the maximum value possible by selecting items without repetition. Capacity: {{capacity}}, Items: {{items}}
        elif "maximum value possible by selecting items without repetition" in q:
            match = re.search(r"capacity[:=] (\d+), items: (.+)", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)

                _, max_val = knapsack_dp_solve(capacity, items)

                return (
                    f"🔍 **Problem:** What is the maximum value possible with capacity {capacity} and items {items_str}?\n\n"
                    f"📘 **Note:** Each item can be chosen at most once (0/1 knapsack).\n\n"
                    f"🧠 **Strategy:**\n"
                    f"- Use dynamic programming to build up best values for each sub-capacity.\n"
                    f"- Track combinations that don’t exceed the capacity.\n\n"
                    f"✅ **Final Answer:** Maximum value = **{max_val}**"
                )



        # 6. Which items will be included if we want to maximize the value with knapsack size {{capacity}}? Items: {{items}}
        elif "which items will be included if we want to maximize the value" in q:
            match = re.search(r"knapsack size (\d+)\? items: (.+)", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)

                dp_table, _ = knapsack_dp_solve(capacity, items)
                selected = knapsack_reconstruct_items(dp_table, items, capacity)

                return (
                    f"🔍 **Problem:** From items {items_str}, choose items that maximize value for knapsack size {capacity}.\n\n"
                    f"📘 **Goal:** Find which items will be selected in the optimal solution.\n\n"
                    f"🧠 **Approach:**\n"
                    f"- Use 0/1 Knapsack DP table to compute value.\n"
                    f"- Backtrack from the table to determine selected items.\n\n"
                    f"✅ **Included Items:** **{', '.join(selected)}**"
                )



        # 7. Calculate the best value that can be achieved with items of weights {weights} and values {{values}} for knapsack capacity {{capacity}}.
        elif "calculate the best value that can be achieved with items of weights" in q:
            match = re.search(r"weights (?:\{)?([\d,\s]+)(?:\})? and values (?:\{)?([\d,\s]+)(?:\})? for knapsack capacity (\d+)", q)
            if match:
                weights_str = match.group(1)
                values_str = match.group(2)
                capacity = int(match.group(3))

                weights = list(map(int, weights_str.split(',')))
                values = list(map(int, values_str.split(',')))

                items = [{'name': f"Item{i+1}", 'weight': weights[i], 'value': values[i]} for i in range(len(weights))]

                # Solve using DP
                def knapsack_dp(cap, weights, values, n):
                    dp = [[0 for _ in range(cap + 1)] for _ in range(n + 1)]
                    for i in range(1, n + 1):
                        for w in range(cap + 1):
                            if weights[i - 1] <= w:
                                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
                            else:
                                dp[i][w] = dp[i - 1][w]
                    return dp[n][cap]

                max_val = knapsack_dp(capacity, weights, values, len(weights))

            table_rows = ''.join(f"| Item {i+1} | {weights[i]} | {values[i]} |\n" for i in range(len(weights)))

            return f"""
            🔹 **Question:**  
            Calculate the best value that can be achieved with items of weights {weights} and values {values} for knapsack capacity {capacity}.

            ---

            📦 **Knapsack Capacity:** {capacity}  
            📥 **Items:**

            | Item | Weight | Value |
            |------|--------|--------|
            {table_rows}
            ---

            🔍 **Approach:**  
            - Use the 0/1 Knapsack DP method.  
            - For each item, choose to either include it (if it fits) or exclude it.  
            - Build a DP table of size (n+1) x (capacity+1) and compute the maximum achievable value.

            ---

            ✅ **Maximum Value Achievable:** **{max_val}**

            🧠 Efficient and optimal!
            """




        # 8. Given the capacity of knapsack as {{capacity}}, find if you can pick at least two items from: {{items}} to fully utilize the knapsack.
        elif "pick at least two items from" in q and "to fully utilize the knapsack" in q:
            match = re.search(r"capacity of knapsack as (\d+), find if you can pick at least two items from: (.+) to fully utilize the knapsack", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)
                
                n = len(items)
                
                # dp[i][w][k] = True if weight `w` can be formed with exactly `k` items using first `i` items
                dp_exact_sum_count = [[[False for _ in range(n + 1)] for _ in range(capacity + 1)] for _ in range(n + 1)]
                dp_exact_sum_count[0][0][0] = True # Base case: 0 weight, 0 items

                for i in range(1, n + 1):
                    weight_i = items[i-1]['weight']
                    for w in range(capacity + 1):
                        for k in range(n + 1):
                            # Not including current item
                            dp_exact_sum_count[i][w][k] = dp_exact_sum_count[i-1][w][k]

                            # Including current item (if it fits and we have room for 'k-1' items to make previous sum)
                            if k > 0 and weight_i <= w:
                                dp_exact_sum_count[i][w][k] = dp_exact_sum_count[i][w][k] or dp_exact_sum_count[i-1][w - weight_i][k - 1]
                
                can_fully_utilize_at_least_two = False
                for k in range(2, n + 1): # Check for 2 or more items
                    if dp_exact_sum_count[n][capacity][k]:
                        can_fully_utilize_at_least_two = True
                        break

                return (
                    f"Given knapsack capacity {capacity} and items {items_str}:\n"
                    f"Can you pick at least two items to fully utilize the knapsack? **{'Yes' if can_fully_utilize_at_least_two else 'No'}**."
                )

        # 9. Select two items from the list: {{items}} to maximize value in a knapsack of capacity {{capacity}}.
        elif "select two items from the list" in q and "to maximize value" in q:
            match = re.search(r"select two items from the list: (.+) to maximize value in a knapsack of capacity (\d+)", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items = parse_items_string(items_str)
                
                best_value = 0
                best_pair = "no pair"
                
                n = len(items)
                for i in range(n):
                    for j in range(i + 1, n): # Iterate to get unique pairs
                        w_sum = items[i]['weight'] + items[j]['weight']
                        v_sum = items[i]['value'] + items[j]['value']
                        
                        if w_sum <= capacity and v_sum > best_value:
                            best_value = v_sum
                            best_pair = f"{items[i].get('name', f'Item {i+1}')} and {items[j].get('name', f'Item {j+1}')}"
                
                return (
                    f"From the list {items_str}, to maximize value by selecting two items for a knapsack of capacity {capacity}:\n"
                    f"The best pair is: **{best_pair}**.\n"
                    f"Maximum value for this pair: **{best_value}**."
                )

        # 10. If you have a knapsack of capacity {capacity}, which item combinations from {{items}} will yield a value above {{value_threshold}}?
        elif "yield a value above" in q:
            match = re.search(r"capacity (\d+).*?combinations from (.+) will yield a value above (\d+)", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                value_threshold = int(match.group(3))
                items = parse_items_string(items_str)

                dp_values = [0] * (capacity + 1)
                for item in items:
                    for w in range(capacity, item['weight'] - 1, -1):
                        dp_values[w] = max(dp_values[w], item['value'] + dp_values[w - item['weight']])

                results = [f"Value {val} at weight {w}" for w, val in enumerate(dp_values) if val > value_threshold]
                summary = "No combinations yield a value above the threshold." if not results else "Combinations:\n- " + "\n- ".join(results)

                return (
                    f"🔍 **Problem:** Identify combinations of items from {items_str} in a knapsack of capacity {capacity} "
                    f"that yield value above threshold **{value_threshold}**.\n\n"
                    f"🧠 **Approach:** We use dynamic programming to find the max value at every possible weight.\n\n"
                    f"📊 **Analysis:**\n{summary}"
                )


        # 11. Determine the maximum value for fractional selection not allowed. Capacity: {{capacity}}, Items: {{items}}
        elif "maximum value for fractional selection not allowed" in q:
            match = re.search(r"capacity: (\d+), items: (.+)", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)

                _, max_val = knapsack_dp_solve(capacity, items)

                return (
                    f"🔍 **Problem:** Find max value with items {items_str}, knapsack capacity {capacity}, and no fractional items.\n\n"
                    f"📘 **Explanation:** Since this is 0/1 Knapsack, you can include each item only once. We use dynamic programming to determine the optimal subset.\n\n"
                    f"✅ **Answer:** Maximum value = **{max_val}**."
                )

        
        # 12. From the following items: {{items}}, determine which combination offers the highest value within a capacity of {{capacity}}.
        elif "determine which combination offers the highest value within a capacity" in q:
            match = re.search(r"items: (.+), determine which combination offers the highest value within a capacity of (\d+)", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items = parse_items_string(items_str)

                dp_table, max_val = knapsack_dp_solve(capacity, items)
                selected_items_names = knapsack_reconstruct_items(dp_table, items, capacity)

                return (
                    f"🔍 **Goal:** From items {items_str}, pick the best combination for capacity {capacity}.\n\n"
                    f"🧠 **Method:** We solve using dynamic programming to ensure we check all subsets efficiently.\n\n"
                    f"✅ **Answer:**\n"
                    f"- Selected items: **{', '.join(selected_items_names)}**\n"
                    f"- Total value: **{max_val}**"
                )


        # 13. How many combinations of items are possible within capacity {{capacity}} for item weights {{weights}}?
        elif "how many combinations of items are possible within capacity" in q.lower() and "item weights" in q.lower():
            match = re.search(r"capacity\s+(\d+).*?item weights\s*(?:\{)?([\d,\s]+)(?:\})?", q, re.IGNORECASE)
            if match:
                capacity = int(match.group(1))
                weights_str = match.group(2)
                weights = list(map(int, weights_str.split(',')))

                # Initialize DP array
                dp = [0] * (capacity + 1)
                dp[0] = 1  # One way to make 0

                for weight in weights:
                    for w in range(capacity, weight - 1, -1):
                        dp[w] += dp[w - weight]

                total = sum(dp) - 1  # Exclude empty set if needed

                return (
                    f"🔍 **Question:** How many combinations of items are possible within capacity {capacity} for item weights {weights}?\n\n"
                    f"📦 **Knapsack Capacity:** {capacity}\n"
                    f"⚖️ **Item Weights:** {weights}\n\n"
                    f"🧠 **Approach:** Dynamic Programming counts all subsets summing to each weight up to {capacity}.\n"
                    f"✅ **Answer:** Total valid combinations = **{total}**"
                )
            else:
                return "⚠️ Invalid question format. Please use: capacity 7 for item weights {20, 30}"




        # 14. List the possible values that can be achieved with combinations from items: {{items}}, within a capacity of {{capacity}}.
        elif "list the possible values that can be achieved with combinations from items" in q:
            match = re.search(r"items: (.+), within a capacity of (\d+)", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items = parse_items_string(items_str)
                
                dp_table, _ = knapsack_dp_solve(capacity, items)
                
                possible_values = set()
                # Collect all unique values from the last row of the DP table
                for w in range(capacity + 1):
                    possible_values.add(dp_table[len(items)][w])
                
                sorted_values = sorted(list(possible_values))
                
                return (
                    f"From items {items_str}, within capacity {capacity}, the possible unique values that can be achieved are:\n"
                    f"**{sorted_values}**."
                )

        # 15. Estimate the maximum number of items that can be selected from the list {{items}} within a capacity of {{capacity}}.
        elif "estimate the maximum number of items that can be selected" in q:
            match = re.search(r"from the list (.+) within a capacity of (\d+)", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items = parse_items_string(items_str)
                
                n = len(items)
                # dp[w] = max number of items for weight w. Initialize with -1 (unreachable)
                dp_max_items = [-1] * (capacity + 1)
                dp_max_items[0] = 0 # 0 items for 0 weight is achievable
                
                for item in items:
                    for w in range(capacity, item['weight'] - 1, -1):
                        if dp_max_items[w - item['weight']] != -1: # If previous state was reachable
                            dp_max_items[w] = max(dp_max_items[w], dp_max_items[w - item['weight']] + 1)
                
                max_items_count = max(dp_max_items) if dp_max_items else 0
                
                return (
                    f"From the list {items_str}, within capacity {capacity}, the maximum number of items that can be selected (assuming each item is unique) is: **{max_items_count}**."
                )

        # 16. Determine which of the items from {{items}} are not worth including due to low value-to-weight ratio. Capacity: {{capacity}}.
        elif "not worth including due to low value-to-weight ratio" in q:
            match = re.search(r"items from (.+) are not worth including due to low value-to-weight ratio\. capacity: (\d+)", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items_raw = parse_items_string(items_str)

                items_with_ratios = []
                for item in items_raw:
                    ratio = item['value'] / item['weight'] if item['weight'] > 0 else float('inf')
                    items_with_ratios.append({'name': item.get('name', f"Item{len(items_with_ratios)+1}"), 'weight': item['weight'], 'value': item['value'], 'ratio': ratio})

                dp_table, _ = knapsack_dp_solve(capacity, items_raw)
                selected = set(knapsack_reconstruct_items(dp_table, items_raw, capacity))

                excluded_items = [f"{item['name']} (w={item['weight']}, v={item['value']}, ratio={item['ratio']:.2f})" for item in items_with_ratios if item['name'] not in selected]

                explanation = "All items potentially contribute to the optimal solution." if not excluded_items else "These items were not selected in the optimal solution:\n- " + "\n- ".join(excluded_items)

                return (
                    f"🔍 **Analysis:** We check which items from {items_str} are **not worth including** for capacity {capacity}.\n\n"
                    f"📘 **Insight:** Items with low value-to-weight ratio are often excluded in optimal 0/1 Knapsack.\n\n"
                    f"✅ **Result:**\n{explanation}"
                )


        # 17. From the set {{items}}, identify which pair maximizes total value within the knapsack limit {{capacity}}.
        elif "identify which pair maximizes total value within the knapsack limit" in q:
            match = re.search(r"set (.+), identify which pair maximizes total value within the knapsack limit (\d+)", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items = parse_items_string(items_str)
                
                best_value = 0
                best_pair_names = "no pair"
                
                n = len(items)
                for i in range(n):
                    for j in range(i + 1, n): # Iterate to get unique pairs
                        w_sum = items[i]['weight'] + items[j]['weight']
                        v_sum = items[i]['value'] + items[j]['value']
                        
                        if w_sum <= capacity and v_sum > best_value:
                            best_value = v_sum
                            best_pair_names = f"{items[i].get('name', f'Item {i+1}')} and {items[j].get('name', f'Item {j+1}')}"
                
                return (
                    f"From the set {items_str}, to maximize total value by selecting a pair of items within the knapsack limit {capacity}:\n"
                    f"The pair that maximizes total value is: **{best_pair_names}**.\n"
                    f"Total value achieved: **{best_value}**."
                )

        # 18. What is the difference in total value if you exclude the heaviest item from selection in knapsack capacity {{capacity}}?
        elif "difference in total value if you exclude the heaviest item" in q:
            match = re.search(r"capacity (\d+).*?items: (.+)", q, re.IGNORECASE)

            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items_raw = parse_items_string(items_str)
                
                if not items_raw:
                    return "No items provided."

                # Calculate optimal value with all items
                _, optimal_value_all = knapsack_dp_solve(capacity, items_raw)

                # Find the heaviest item (prioritize first occurring if multiple same max weight)
                heaviest_item = max(items_raw, key=lambda item: item['weight'])
                
                # Create a new list excluding the heaviest item
                items_excluded_heaviest = [item for item in items_raw if item != heaviest_item]
                
                # Calculate optimal value without the heaviest item
                _, optimal_value_without_heaviest = knapsack_dp_solve(capacity, items_excluded_heaviest)
                
                value_difference = optimal_value_all - optimal_value_without_heaviest
                
                return (
                    f"Given knapsack capacity {capacity} and items {items_str}:\n"
                    f"Original optimal value (including heaviest item): {optimal_value_all}.\n"
                    f"Heaviest item: {heaviest_item.get('name', 'N/A')} (weight={heaviest_item['weight']}, value={heaviest_item['value']}).\n"
                    f"Optimal value excluding the heaviest item: {optimal_value_without_heaviest}.\n"
                    f"The difference in total value is: **{value_difference}**."
                )

        # 19. Can you exactly fill a knapsack of size {{capacity}} with the following item weights: {{weights}}?
        elif "exactly fill a knapsack of size" in q and "item weights" in q:
            match = re.search(r"size (\d+) with the following item weights: (?:\{)?([\d,\s]+)(?:\})?", q)
            if match:
                capacity = int(match.group(1))
                weights_str = match.group(2)
                weights = list(map(int, weights_str.split(',')))
                
                # Subset Sum logic
                dp = [False] * (capacity + 1)
                dp[0] = True
                
                for weight in weights:
                    for w in range(capacity, weight - 1, -1):
                        dp[w] = dp[w] or dp[w - weight]
                
                can_exactly_fill = "Yes" if dp[capacity] else "No"
                
                return (
                    f"🧮 **Problem:** Can you exactly fill a knapsack of size `{capacity}` with item weights `{weights}`?\n\n"
                    f"📦 **Subset Sum Result:** **{can_exactly_fill}**\n\n"
                    f"👉 This is a classic Subset Sum problem, solved using dynamic programming where we check whether any subset of items adds up exactly to the capacity."
                )


        # 20. Out of all item combinations from {{items}}, which one gives highest value under the limit {{capacity}} while leaving minimum unused space?
        elif "gives highest value under the limit" in q and "leaving minimum unused space" in q:
            match = re.search(r"combinations from (.+?) which one gives highest value.*?limit (\d+)", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items = parse_items_string(items_str)

                dp_table, max_total_value = knapsack_dp_solve(capacity, items)

                # Find the weight at which max value is achieved using the most capacity
                best_weight = -1
                for w in range(capacity, -1, -1):
                    if dp_table[len(items)][w] == max_total_value:
                        best_weight = w
                        break

                # Backtrack to get selected items
                selected = []
                w_temp = best_weight
                for i in range(len(items), 0, -1):
                    item = items[i-1]
                    if w_temp >= item['weight'] and \
                    dp_table[i][w_temp] == item['value'] + dp_table[i-1][w_temp - item['weight']]:
                        selected.append(item['name'])
                        w_temp -= item['weight']
                selected.reverse()

                unused = capacity - best_weight

                return (
                    f"🔍 **Problem:** Choose a combination from {items_str} for knapsack capacity {capacity} "
                    f"that gives the highest value with **minimum unused space**.\n\n"
                    f"📘 **Concept:** From all valid combinations, we choose the one with max value and largest total weight.\n\n"
                    f"✅ **Result:**\n"
                    f"- Value = **{max_total_value}**\n"
                    f"- Used capacity = **{best_weight}**\n"
                    f"- Unused space = **{unused}**\n"
                    f"- Items selected = **{', '.join(selected)}**"
                )

    elif level == "Level 3":
        # 1. Given a knapsack capacity of {{capacity}}, and items {{items}}, find the maximum value achievable using dynamic programming.
        if "maximum value achievable using dynamic programming" in q:
            match = re.search(r"capacity of (\d+), and items (.+), find the maximum value achievable using dynamic programming", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)
                
                _, max_val = knapsack_dp_solve(capacity, items)
                
                return (
                    f"Given a knapsack capacity of {capacity} and items {items_str}:\n"
                    f"Using dynamic programming for the 0/1 Knapsack problem, the maximum value achievable is: **{max_val}**."
                )

        # 2. Using 0/1 Knapsack DP approach, find which items are selected for capacity {capacity} with the item list {{items}}.
        elif "find which items are selected for capacity" in q:
            match = re.search(r"capacity (\d+) with the item list (.+)", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)
                
                dp_table, _ = knapsack_dp_solve(capacity, items)
                selected_items_names = knapsack_reconstruct_items(dp_table, items, capacity)
                
                return (
                    f"Using the 0/1 Knapsack dynamic programming approach for capacity {capacity} and item list {items_str}:\n"
                    f"The items selected are: **{', '.join(selected_items_names)}**."
                )
        
        # 3. Calculate the DP table for solving the knapsack problem with capacity {{capacity}} and weights {{weights}}, values {{values}}.
        elif "calculate the dp table for solving the knapsack problem" in q:
            match = re.search(
                r"capacity (\d+)\s+and weights(?:\s*\{)?([\d,\s]+)(?:\})?\s*,\s*values(?:\s*\{)?([\d,\s]+)(?:\})?",
                q, re.IGNORECASE
            )
            if match:
                capacity = int(match.group(1))
                weights = list(map(int, match.group(2).split(",")))
                values = list(map(int, match.group(3).split(",")))
                n = len(weights)

                # Check mismatch
                if len(weights) != len(values):
                    return "⚠️ **Error:** Mismatch between number of weights and values."

                # Initialize DP table
                dp = [[0] * (capacity + 1) for _ in range(n + 1)]

                # Fill DP table
                for i in range(1, n + 1):
                    for w in range(capacity + 1):
                        if weights[i - 1] <= w:
                            dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
                        else:
                            dp[i][w] = dp[i - 1][w]

                # Format DP table
                dp_table_str = "🧮 **DP Table (rows = items, columns = capacities):**\n\n"
                dp_table_str += "| i\\w | " + " | ".join(str(w) for w in range(capacity + 1)) + " |\n"
                dp_table_str += "|-----" + "|".join(["------"] * (capacity + 1)) + "|\n"
                for i in range(n + 1):
                    dp_table_str += f"| {i}   | " + " | ".join(str(dp[i][w]) for w in range(capacity + 1)) + " |\n"

                return (
                    f"🔍 **Problem:**\n"
                    f"Calculate the DP table for knapsack with:\n"
                    f"- Capacity: **{capacity}**\n"
                    f"- Weights: `{weights}`\n"
                    f"- Values: `{values}`\n\n"
                    f"📘 **Approach:**\n"
                    f"We use 0/1 Knapsack Dynamic Programming:\n"
                    f"- `dp[i][w]` = max value using first `i` items with total weight ≤ `w`\n\n"
                    f"{dp_table_str}\n"
                    f"✅ **Final Answer:**\n"
                    f"Maximum value achievable: **{dp[n][capacity]}**"
                )



        # 4. Given item list {{items}}, compute both maximum value and the items chosen for knapsack size {{capacity}}.
        elif "compute both maximum value and the items chosen" in q:
            match = re.search(r"item list (.+), compute both maximum value and the items chosen for knapsack size (\d+)", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items = parse_items_string(items_str)

                dp_table, max_val = knapsack_dp_solve(capacity, items)
                selected_items_names = knapsack_reconstruct_items(dp_table, items, capacity)

                return (
                    f"Given item list {items_str} for knapsack size {capacity}:\n"
                    f"Maximum value achievable: **{max_val}**.\n"
                    f"Items chosen: **{', '.join(selected_items_names)}**."
                )

        
        # 5. Which subset of items from {{items}} results in the maximum value without exceeding capacity {{capacity}}, and what is the remaining space?
        elif "subset of items from" in q and "remaining space" in q:
            match = re.search(r"subset of items from (.+?) results .*? capacity (\d+)", q)

            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items = parse_items_string(items_str)

                dp_table, max_val = knapsack_dp_solve(capacity, items)
                selected_items_names = knapsack_reconstruct_items(dp_table, items, capacity)
                total_weight_used = sum(item['weight'] for item in items if item['name'] in selected_items_names)
                remaining_space = capacity - total_weight_used

                return (
                    f"Given item list {items_str} and capacity {capacity}:\n"
                    f"Maximum value: **{max_val}**\n"
                    f"Selected items: **{', '.join(selected_items_names)}**\n"
                    f"Remaining space in knapsack: **{remaining_space}**"
                )

            else:
                return "Could not parse the item list or capacity."


        # 6. For the knapsack instance with capacity {{capacity}} and items {{items}}, construct and fill the DP matrix step-by-step.
        elif "construct and fill the dp matrix step-by-step" in q:
            match = re.search(r"capacity (\d+) and items (.+), construct and fill the dp matrix step-by-step", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)
                
                n = len(items)
                dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
                steps_log = []

                # Initializing row 0 and col 0 (implicitly 0s)
                steps_log.append("Initialize DP table with zeros. Row 0 (no items) and Column 0 (0 capacity) will remain 0.")
                
                for i in range(1, n + 1):
                    weight_i = items[i-1]['weight']
                    value_i = items[i-1]['value']
                    item_name = items[i-1].get('name', f"Item {i}")
                    steps_log.append(f"\nProcessing {item_name} (weight={weight_i}, value={value_i}):")
                    for w in range(capacity + 1):
                        val_not_included = dp[i-1][w]
                        val_included = -1 # Sentinel for not possible
                        
                        if weight_i <= w:
                            val_included = value_i + dp[i-1][w - weight_i]
                            steps_log.append(f"  - For capacity {w}: Compare (Exclude {item_name}: dp[{i-1}][{w}]={val_not_included}) vs (Include {item_name}: {value_i} + dp[{i-1}][{w - weight_i}]={val_included})")
                        else:
                            steps_log.append(f"  - For capacity {w}: {item_name} (weight={weight_i}) does not fit. Take value from dp[{i-1}][{w}]={val_not_included}")
                        
                        dp[i][w] = max(val_not_included, val_included)
                        steps_log.append(f"    -> dp[{i}][{w}] = {dp[i][w]}")

                table_str = "Final DP Table:\n"
                table_str += "Cap " + " ".join(f"{w:3}" for w in range(capacity + 1)) + "\n"
                table_str += "---" + "----" * (capacity + 1) + "\n"
                for r_idx, row in enumerate(dp):
                    item_info = f"Item {r_idx}" if r_idx > 0 else "Base"
                    table_str += f"{item_info:<3} " + " ".join(f"{cell:3}" for cell in row) + "\n"

                return (
                    f"To construct and fill the DP matrix step-by-step for knapsack capacity {capacity} and items {items_str}:\n\n"
                    "The 0/1 Knapsack DP table `dp[i][w]` represents the maximum value achievable with the first `i` items and a capacity of `w`.\n"
                    "**Steps:**\n"
                    + "\n".join(steps_log) + "\n\n"
                    f"{table_str}\n"
                    f"Maximum value: **{dp[n][capacity]}**."
                )
        
        # 7. Given capacity {{capacity}}, explain how the solution changes if item {{exclude_item}} is excluded from {{items}}.
        elif "explain how the solution changes if item" in q and "is excluded from" in q:
            match = re.search(r"capacity (\d+), explain how the solution changes if item (.+) is excluded from (.+)", q)
            if match:
                capacity = int(match.group(1))
                exclude_item_name = match.group(2).strip() # .strip() to clean up name
                items_str = match.group(3)
                items_raw = parse_items_string(items_str)
                
                # Calculate optimal value with all items
                _, optimal_value_all = knapsack_dp_solve(capacity, items_raw)
                selected_items_all = knapsack_reconstruct_items(knapsack_dp_solve(capacity, items_raw)[0], items_raw, capacity)

                # Create a new list excluding the specified item
                items_excluded = [item for item in items_raw if item.get('name', '') != exclude_item_name]
                
                # Calculate optimal value without the excluded item
                _, optimal_value_excluded = knapsack_dp_solve(capacity, items_excluded)
                selected_items_excluded = knapsack_reconstruct_items(knapsack_dp_solve(capacity, items_excluded)[0], items_excluded, capacity)

                return (
                    f"Given capacity {capacity}, and items {items_str}:\n"
                    f"**Original Solution (with all items):**\n"
                    f"  Maximum Value: {optimal_value_all}\n"
                    f"  Selected Items: {', '.join(selected_items_all)}\n\n"
                    f"**If Item '{exclude_item_name}' is excluded:**\n"
                    f"  The dynamic programming algorithm would be run on the reduced set of items, effectively ignoring '{exclude_item_name}' in all calculations.\n"
                    f"  New Maximum Value: **{optimal_value_excluded}**.\n"
                    f"  New Selected Items: **{', '.join(selected_items_excluded)}**.\n\n"
                    f"The solution changes because the optimal choice that might have included '{exclude_item_name}' is no longer possible. The algorithm finds the best possible value from the remaining items, which may be lower than the original optimal value, but is optimal for the new item set. This demonstrates the 'optimal substructure' property of Knapsack."
                )

        # 8. Compute the optimal value for two knapsacks of capacity {{cap1}} and {{cap2}}, where the items {{items}} must be divided optimally.
        elif "compute the optimal value for two knapsacks of capacity" in q and "items" in q and "must be divided optimally" in q:
            match = re.search(r"capacity (\d+) and (\d+), where the items (.+) must be divided optimally", q)
            if match:
                cap1 = int(match.group(1))
                cap2 = int(match.group(2))
                items_str = match.group(3)
                items = parse_items_string(items_str)

                return (
                    f"🔍 **Problem:**\n"
                    f"Compute the maximum total value using two knapsacks with capacities {cap1} and {cap2}, where items {items_str} are divided optimally.\n\n"
                    f"📘 **Concept:**\n"
                    f"This is a **Two-Knapsack** variation of the 0/1 Knapsack Problem. Each item can go in at most one knapsack or be left out.\n\n"
                    f"🧠 **Approach:**\n"
                    f"- Use a 3D DP: `dp[i][c1][c2]` = max value using first `i` items, with `c1` capacity in knapsack 1, `c2` in knapsack 2.\n"
                    f"- Try 3 choices for each item: exclude it, put it in knapsack 1, or put it in knapsack 2.\n\n"
                    f"⚙️ **Time Complexity:** O(n × cap1 × cap2)\n\n"
                    f"✏️ **Note:** This is an explanation-only block. For actual computation, implement the 3D DP as described."
                )



        # 9. Using space-optimized DP, solve the 0/1 knapsack problem with capacity {{capacity}} and items {{items}}.
        elif "using space-optimized dp, solve the 0/1 knapsack problem" in q:
            match = re.search(r"capacity (\d+) and items (.+)", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)
                
                n = len(items)
                # Space-optimized DP: using 1D array
                # dp[w] will store the maximum value for a given weight `w`.
                dp = [0] * (capacity + 1) #
                
                for item in items:
                    weight_i = item['weight']
                    value_i = item['value']
                    # Iterate backwards to ensure each item is considered only once per 'row' simulation.
                    # This is crucial for the 0/1 property.
                    for w in range(capacity, weight_i - 1, -1): #
                        dp[w] = max(dp[w], value_i + dp[w - weight_i]) #
                
                max_val = dp[capacity]
                
                return (
                    f"Using space-optimized DP for the 0/1 knapsack problem with capacity {capacity} and items {items_str}:\n"
                    f"This approach uses a 1D DP array of size `capacity + 1`, reducing space complexity from O(N*W) to O(W).\n"
                    "```python\n"
                    "def knapsack_optimized_space(capacity, items):\n"
                    "    dp = [0] * (capacity + 1) # O(W) space\n"
                    "    for item in items:\n"
                    "        weight_i = item['weight']\n"
                    "        value_i = item['value']\n"
                    "        # Iterate backwards to ensure item is used at most once\n"
                    "        for w in range(capacity, weight_i - 1, -1):\n"
                    "            dp[w] = max(dp[w], value_i + dp[w - weight_i])\n"
                    "    return dp[capacity]\n"
                    "```\n\n"
                    f"Maximum value achievable: **{max_val}**.\n"
                    "Time Complexity: O(N * W)\n"
                    "Space Complexity: O(W)"
                )

        # 10. If the total weight must not exceed {{capacity}}, but at least {{min_items}} items must be selected from {{items}}, what is the optimal value?
        # elif "at least" in q.lower() and "items must be selected" in q.lower():
        #     match = re.search(r"not exceed (\d+), but at least (\d+) items must be selected from (.+), what is the optimal value", q, re.IGNORECASE)
        #     if match:
        #         capacity = int(match.group(1))
        #         min_items = int(match.group(2))
        #         items_str = match.group(3)
        #         items = parse_items_string(items_str)

        #         n = len(items)

        #         if n < min_items:
        #             return (
        #                 f"⚠️ **Invalid Input:** Cannot select at least {min_items} items from only {n} available.\n"
        #                 f"Please provide at least {min_items} items."
        #             )

        #         # Initialize 3D DP table
        #         dp = [[[-float('inf')] * (n + 1) for _ in range(capacity + 1)] for _ in range(n + 1)]
        #         dp[0][0][0] = 0  # base case: 0 items, 0 weight, 0 selected = value 0

        #         for i in range(1, n + 1):
        #             wi = items[i - 1]['weight']
        #             vi = items[i - 1]['value']
        #             for w in range(capacity + 1):
        #                 for k in range(n + 1):
        #                     # Case 1: Don't take the item
        #                     dp[i][w][k] = dp[i - 1][w][k]
        #                     # Case 2: Take the item (if it fits and builds from valid previous state)
        #                     if w >= wi and k > 0 and dp[i - 1][w - wi][k - 1] != -float('inf'):
        #                         dp[i][w][k] = max(dp[i][w][k], vi + dp[i - 1][w - wi][k - 1])

        #         # Check all combinations where we selected ≥ min_items and weight ≤ capacity
        #         optimal_value = max(
        #             dp[n][w][k]
        #             for w in range(capacity + 1)
        #             for k in range(min_items, n + 1)
        #             if dp[n][w][k] != -float('inf')
        #         ) if any(
        #             dp[n][w][k] != -float('inf')
        #             for w in range(capacity + 1)
        #             for k in range(min_items, n + 1)
        #         ) else -float('inf')

        #         if optimal_value == -float('inf'):
        #             return (
        #                 f"🔍 **Problem:** Select at least {min_items} items from {items_str} with total weight ≤ {capacity}.\n\n"
        #                 f"📘 **Constraint:** This problem requires selecting a minimum number of items.\n\n"
        #                 f"⚠️ **Result:** No valid combination satisfies both constraints."
        #             )
        #         else:
        #             return (
        #                 f"🔍 **Problem:** Select at least {min_items} items from {items_str} with total weight ≤ {capacity}.\n\n"
        #                 f"📘 **Approach:**\n"
        #                 f"- Used 3D Dynamic Programming\n"
        #                 f"- `dp[i][w][k]`: max value using first `i` items, total weight ≤ `w`, selecting exactly `k` items\n"
        #                 f"- Final answer: max over all `k ≥ {min_items}` and `w ≤ {capacity}`\n\n"
        #                 f"✅ **Optimal Value:** {optimal_value}"
        #             )


        # 11. How does the result change if values of items in {{items}} are increased by 10% and capacity is {{capacity}}?
        elif "result change if values of items in" in q and "are increased by 10%" in q:
            match = re.search(r"values of items in (.+) are increased by 10% and capacity is (\d+)", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items_original = parse_items_string(items_str)
                
                # Calculate original max value
                _, original_max_val = knapsack_dp_solve(capacity, items_original)
                
                # Create new items list with 10% increased values
                items_increased_value = []
                for item in items_original:
                    # Use math.ceil to ensure value increases even for small fractional parts
                    new_value = math.ceil(item['value'] * 1.10)
                    items_increased_value.append({'name': item.get('name', 'N/A'), 'weight': item['weight'], 'value': new_value})
                
                _, new_max_val = knapsack_dp_solve(capacity, items_increased_value)
                
                # Precompute the formatted item strings outside the f-string
            formatted_items = [
                f"{item.get('name', 'N/A')}(w={item['weight']},v={item['value']})"
                for item in items_increased_value
]

            return (
                f"Given items {items_str} and capacity {capacity}:\n"
                f"Original maximum value: {original_max_val}.\n"
                f"If values of items are increased by 10% (rounded up):\n"
                f"  New item values: {formatted_items}.\n"
                f"  The new maximum value achievable is: **{new_max_val}**.\n"
                f"The result is directly affected by the increased values. "
                f"The selection of items (which items are included) might change if increasing values makes previously 'unprofitable' items now beneficial within the capacity constraints."
        )


        # 12. Given items with close weight-value ratios in {{items}}, find which are most beneficial under capacity {{capacity}}.
        elif "items with close weight-value ratios" in q:
            match = re.search(r"in (.+), find which are most beneficial under capacity (\d+)", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items = parse_items_string(items_str)
                
                # Solve full 0/1 Knapsack
                dp_table, max_val = knapsack_dp_solve(capacity, items)
                selected_items_names = knapsack_reconstruct_items(dp_table, items, capacity)
                
                # Calculate ratios for display
                item_ratios_info = []
                for item in items:
                    ratio = item['value'] / item['weight'] if item['weight'] > 0 else float('inf')
                    item_ratios_info.append(f"{item.get('name', 'N/A')}(w={item['weight']},v={item['value']},ratio={ratio:.2f})")

            ratios_formatted = "- " + ";\n- ".join(item_ratios_info)
            return (
            f"Given items with close weight-value ratios {items_str} and capacity {capacity}:\n"
            f"**Value-to-Weight Ratios:**\n{ratios_formatted}\n\n"
            f"For the 0/1 Knapsack problem, even if items have similar value-to-weight ratios, the optimal solution is not always found by a greedy choice. Dynamic programming considers all combinations to find the true optimum.\n"
            f"The items that are most beneficial (i.e., those that are part of the optimal solution) are: **{', '.join(selected_items_names)}**.\n"
            f"The maximum value achieved is: **{max_val}**.\n"
            f"This shows that even with close ratios, the specific weights and capacity constraints dictate the truly 'most beneficial' subset."
        )


        # 13. Use memoization to solve the knapsack problem for capacity {{capacity}} and the item set {{items}}.
        elif "use memoization to solve the knapsack problem" in q:
            match = re.search(r"capacity (\d+) and the item set (.+)", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)
                n = len(items)

                memo_cache_for_this_run = {} # (index, current_capacity) -> max_value

                def knapsack_memo_recursive(idx, current_cap):
                    if idx == 0 or current_cap == 0: # Base cases
                        return 0
                    
                    if (idx, current_cap) in memo_cache_for_this_run: # Check cache
                        return memo_cache_for_this_run[(idx, current_cap)]
                    
                    item_weight = items[idx-1]['weight']
                    item_value = items[idx-1]['value']
                    
                    # Option 1: Don't include the current item (idx-1)
                    val_not_included = knapsack_memo_recursive(idx - 1, current_cap) #
                    
                    # Option 2: Include the current item (idx-1) if it fits
                    val_included = -1 # Default if cannot include (ensures max() works correctly)
                    if item_weight <= current_cap:
                        val_included = item_value + knapsack_memo_recursive(idx - 1, current_cap - item_weight) #
                    
                    result = max(val_not_included, val_included) #
                    memo_cache_for_this_run[(idx, current_cap)] = result # Store result
                    return result

                max_val = knapsack_memo_recursive(n, capacity)

                # Reconstruct selected items from memo table (similar to bottom-up traceback)
                selected_items_names = []
                current_w = capacity
                
                for i in range(n, 0, -1):
                    item_weight = items[i-1]['weight']
                    item_value = items[i-1]['value']
                    
                    # Check if the value at memo[(i, current_w)] came from including item (i-1)
                    # Compare current state's max value to the value if item (i-1) was excluded.
                    # If it's greater, then item (i-1) must have been included.
                    
                    val_if_excluded = memo_cache_for_this_run.get((i-1, current_w), -1) # Value if item was excluded
                    
                    val_if_included_path = -1
                    if current_w >= item_weight:
                        val_if_included_path = item_value + memo_cache_for_this_run.get((i-1, current_w - item_weight), -1)

                    # If the current optimal value (memo_cache_for_this_run[(i, current_w)]) was achieved by including this item
                    # This implies: (value_if_included_path == memo_cache_for_this_run[(i, current_w)]) AND (value_if_included_path >= value_if_excluded)
                    if memo_cache_for_this_run.get((i, current_w), -1) == val_if_included_path and val_if_included_path >= val_if_excluded:
                         selected_items_names.append(items[i-1].get('name', f'Item {i}'))
                         current_w -= item_weight
                selected_items_names.reverse() # Present in original order

                return (
                    f"To solve the knapsack problem for capacity {capacity} and item set {items_str} using memoization:\n\n"
                    "Memoization (top-down dynamic programming) solves overlapping subproblems by storing their results in a cache (e.g., a dictionary). This prevents redundant computations, making the exponential recursive approach efficient.\n"
                    "```python\n"
                    "memo = {}\n"
                    "def knapsack_memo(idx, current_cap):\n"
                    "    if idx == 0 or current_cap == 0:\n"
                    "        return 0\n"
                    "    if (idx, current_cap) in memo:\n"
                    "        return memo[(idx, current_cap)]\n"
                    "    \n"
                    "    item_weight = items[idx-1]['weight']\n"
                    "    item_value = items[idx-1]['value']\n"
                    "    \n"
                    "    val_not_included = knapsack_memo(idx - 1, current_cap)\n"
                    "    val_included = -1 # Use -1 or very small number to represent not possible\n"
                    "    if item_weight <= current_cap:\n"
                    "        val_included = item_value + knapsack_memo(idx - 1, current_cap - item_weight)\n"
                    "        \n"
                    "    result = max(val_not_included, val_included)\n"
                    "    memo[(idx, current_cap)] = result\n"
                    "    return result\n"
                    "```\n\n"
                    f"Maximum value achievable: **{max_val}**.\n"
                    f"Selected items: **{', '.join(selected_items_names)}**.\n"
                    "Time Complexity: O(N * W)\n"
                    "Space Complexity: O(N * W) (for memoization table and recursion stack)"
                )

        # 14. Modify your solution to the knapsack problem with capacity {{capacity}} so that the value of each item is doubled. What is the new result?
        # elif "value of each item is doubled" in q.lower():
        #     import re

        #     # Extract capacity and item list from the question
        #     match = re.search(
        #         r"capacity\s*(\d+)[^\n]*?value of each item is doubled(?:.*?items:\s*(.+))?",
        #         q,
        #         re.IGNORECASE
        #     )

        #     if match:
        #         capacity = int(match.group(1))
        #         items_str = match.group(2)

        #         if not items_str or not items_str.strip():
        #             return (
        #                 "⚠️ Missing Items:\n"
        #                 "Please provide items in the format:\n"
        #                 "`items: A(weight=2, value=10), B(weight=5, value=20)`"
        #             )

        #         # Try to parse the item list using helper function
        #         items_original = parse_items_string(items_str)
        #         if not items_original:
        #             return (
        #                 "⚠️ Invalid Format:\n"
        #                 "Could not parse item list. Make sure each item is like A(weight=3, value=15)."
        #             )

        #         # Double the values
        #         items_doubled = [
        #             {
        #                 'name': item.get('name', f'Item{i+1}'),
        #                 'weight': item['weight'],
        #                 'value': item['value'] * 2
        #             }
        #             for i, item in enumerate(items_original)
        #         ]

        #         # Solve the knapsack problem with doubled values
        #         _, new_max_val = knapsack_dp_solve(capacity, items_doubled)

        #         # Format the new items for display
        #         formatted_items = ", ".join(
        #             f"{item['name']}(w={item['weight']}, v={item['value']})"
        #             for item in items_doubled
        #         )

        #         return (
        #             f"🔍 Problem:\n"
        #             f"You have a knapsack with capacity {capacity} and a list of items: {items_str}.\n"
        #             f"What happens if the value of each item is doubled?\n\n"
        #             f"📘 Concept:\n"
        #             f"Doubling all item values increases their contribution to the total value.\n"
        #             f"The selection strategy (maximize value within capacity) may change accordingly.\n\n"
        #             f"🧠 Modified Items:\n"
        #             f"{formatted_items}\n\n"
        #             f"✅ New Maximum Value: {new_max_val}"
        #         )


        # 15. Find the optimal solution for the given set {{items}} and capacity {{capacity}}, ensuring at least one item weighs more than {{min_weight}}.
        elif "at least one item weighs more than" in q:
            match = re.search(r"set (.+) and capacity (\d+), ensuring at least one item weighs more than (\d+)", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                min_weight_threshold = int(match.group(3))
                items_raw = parse_items_string(items_str)
                
                n = len(items_raw)
                
                # dp[w][0] = max value for weight w, without using a heavy item
                # dp[w][1] = max value for weight w, using at least one heavy item
                # Initialize with negative infinity to represent unreachable states
                dp_current_iter = [[-float('inf')] * 2 for _ in range(capacity + 1)]
                dp_prev_iter = [[-float('inf')] * 2 for _ in range(capacity + 1)]
                
                dp_prev_iter[0][0] = 0 # Base case: 0 value for 0 weight, no heavy item yet

                for item in items_raw:
                    weight_i = item['weight']
                    value_i = item['value']
                    is_current_item_heavy = (1 if weight_i > min_weight_threshold else 0)

                    # Copy values from previous row (don't include current item)
                    for w_copy in range(capacity + 1):
                        dp_current_iter[w_copy][0] = dp_prev_iter[w_copy][0]
                        dp_current_iter[w_copy][1] = dp_prev_iter[w_copy][1]

                    # Now consider including current item, iterating backwards to avoid using current item multiple times
                    for w in range(capacity, weight_i - 1, -1):
                        # Path where we previously did NOT have a heavy item (dp_prev_iter[...][0])
                        if dp_prev_iter[w - weight_i][0] != -float('inf'):
                            # New value if we include current item
                            new_val = value_i + dp_prev_iter[w - weight_i][0]
                            if is_current_item_heavy == 1: # If current item is heavy, update has_heavy=1 state
                                dp_current_iter[w][1] = max(dp_current_iter[w][1], new_val)
                            else: # If current item is not heavy, update has_heavy=0 state
                                dp_current_iter[w][0] = max(dp_current_iter[w][0], new_val)

                        # Path where we previously DID have a heavy item (dp_prev_iter[...][1])
                        # Including any item in this path will keep the has_heavy=1 state
                        if dp_prev_iter[w - weight_i][1] != -float('inf'):
                            new_val = value_i + dp_prev_iter[w - weight_i][1]
                            dp_current_iter[w][1] = max(dp_current_iter[w][1], new_val)
                    
                    # Update dp_prev_iter for the next item's iteration
                    for w_update in range(capacity + 1):
                        dp_prev_iter[w_update][0] = dp_current_iter[w_update][0]
                        dp_prev_iter[w_update][1] = dp_current_iter[w_update][1]

                # The final answer is the max value from any weight with has_heavy flag set to 1
                optimal_value = -float('inf')
                for w in range(capacity + 1):
                    optimal_value = max(optimal_value, dp_prev_iter[w][1]) # Check only for results that included a heavy item

                if optimal_value == -float('inf'):
                    return (
                        f"For items {items_str}, capacity {capacity}, ensuring at least one item weighs more than {min_weight_threshold}:\n"
                        f"It is **not possible** to find a solution that satisfies the constraints."
                    )
                else:
                    return (
                        f"For items {items_str}, capacity {capacity}, ensuring at least one item weighs more than {min_weight_threshold}:\n"
                        f"The optimal value achievable under these constraints is: **{optimal_value}**."
                    )

        # 16. Determine which item(s) can be removed from the solution without decreasing the maximum achievable value for capacity {{capacity}}. Items: {{items}}.
        elif "removed from the solution without decreasing the maximum achievable value" in q:
            match = re.search(r"capacity (\d+)\. items: (.+)", q)
            if match:
                capacity = int(match.group(1))
                items_str = match.group(2)
                items = parse_items_string(items_str)

                dp_table_full, max_val_full = knapsack_dp_solve(capacity, items)
                removable_items = []

                for i, item in enumerate(items):
                    remaining_items = [itm for j, itm in enumerate(items) if j != i]
                    _, value_without = knapsack_dp_solve(capacity, remaining_items)

                    if value_without == max_val_full:
                        removable_items.append(item.get('name', f"Item{i+1}"))

                if removable_items:
                    item_list = ", ".join(removable_items)
                    result = f"✅ **Removable Items:** {item_list}\nThese items can be excluded **without changing the maximum achievable value**."
                else:
                    result = "❌ **No items can be removed** without reducing the maximum achievable value."

                return (
                    f"🔍 **Problem:**\nGiven knapsack capacity = {capacity} and items = {items_str}, identify which items "
                    f"can be removed from the optimal solution without lowering the total value.\n\n"
                    f"📘 **Concept:**\nSometimes, more than one combination of items leads to the same maximum value. "
                    f"An item is **removable** if there's an alternative optimal combination that doesn't include it.\n\n"
                    f"🧪 **Approach:**\n- Compute the original maximum value with all items.\n"
                    f"- Try removing each item one-by-one and recompute.\n"
                    f"- If value stays the same, the item is not essential.\n\n"
                    f"🎯 **Original Maximum Value:** {max_val_full}\n\n"
                    f"{result}"
                )


        # 17. Find the optimal subset of {{items}} for a knapsack of capacity {{capacity}} using top-down dynamic programming.
        elif "find the optimal subset of" in q and "using top-down dynamic programming" in q:
            match = re.search(r"subset of (.+) for a knapsack of capacity (\d+) using top-down dynamic programming", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items = parse_items_string(items_str)
                n = len(items)

                memo_for_top_down = {} # Reset memo for each call

                def knapsack_memo_top_down(idx, current_cap):
                    if idx == 0 or current_cap == 0:
                        return 0
                    if (idx, current_cap) in memo_for_top_down:
                        return memo_for_top_down[(idx, current_cap)]
                    
                    item_weight = items[idx-1]['weight']
                    item_value = items[idx-1]['value']
                    
                    val_not_included = knapsack_memo_top_down(idx - 1, current_cap)
                    val_included = -1
                    if item_weight <= current_cap:
                        val_included = item_value + knapsack_memo_top_down(idx - 1, current_cap - item_weight)
                    
                    result = max(val_not_included, val_included)
                    memo_for_top_down[(idx, current_cap)] = result
                    return result

                max_val = knapsack_memo_top_down(n, capacity)

                # Reconstruct selected items from memo table (similar to bottom-up traceback)
                selected_items_names = []
                current_w = capacity
                
                for i in range(n, 0, -1):
                    item_weight = items[i-1]['weight']
                    item_value = items[i-1]['value']
                    
                    # Check if the value at memo[(i, current_w)] came from including item (i-1)
                    # Compare current state's max value to the value if item (i-1) was excluded.
                    # If it's greater, then item (i-1) must have been included.
                    
                    val_if_excluded = memo_for_top_down.get((i-1, current_w), -1) # Value if item was excluded
                    
                    val_if_included_path = -1
                    if current_w >= item_weight:
                        val_if_included_path = item_value + memo_for_top_down.get((i-1, current_w - item_weight), -1)

                    # If the current optimal value (memo_for_top_down[(i, current_w)]) was achieved by including this item
                    # This implies: (value_if_included_path == memo_for_top_down[(i, current_w)]) AND (value_if_included_path >= value_if_excluded)
                    if memo_for_top_down.get((i, current_w), -1) == val_if_included_path and val_if_included_path >= val_if_excluded:
                         selected_items_names.append(items[i-1].get('name', f'Item {i}'))
                         current_w -= item_weight
                selected_items_names.reverse() # Present in original order

                return (
                    f"To find the optimal subset of items {items_str} for a knapsack of capacity {capacity} using top-down dynamic programming (memoization):\n"
                    f"Maximum value achievable: **{max_val}**.\n"
                    f"Optimal subset of items: **{', '.join(selected_items_names)}**."
                )

        # 18. How does the optimal solution differ when the knapsack capacity changes from {{capacity1}} to {{capacity2}}? Items: {{items}}
        elif "optimal solution differ when the knapsack capacity changes from" in q:
            match = re.search(r"changes from (\d+) to (\d+)\? items: (.+)", q)
            if match:
                cap1 = int(match.group(1))
                cap2 = int(match.group(2))
                items_str = match.group(3)
                items = parse_items_string(items_str)

                # Compute both solutions
                dp_table1, val1 = knapsack_dp_solve(cap1, items)
                items_sel1 = knapsack_reconstruct_items(dp_table1, items, cap1)

                dp_table2, val2 = knapsack_dp_solve(cap2, items)
                items_sel2 = knapsack_reconstruct_items(dp_table2, items, cap2)

                return (
                    f"🔍 **Problem:**\nCompare the optimal knapsack solutions when capacity changes from {cap1} to {cap2}, using the same items.\n\n"
                    f"📘 **Concept:**\nIn 0/1 Knapsack, increasing the capacity generally allows for a greater total value — but not always a change in the item set.\n\n"
                    f"📊 **Results:**\n"
                    f"- 🧺 **Capacity {cap1}** → Value: **{val1}**, Items: {', '.join(items_sel1)}\n"
                    f"- 🧺 **Capacity {cap2}** → Value: **{val2}**, Items: {', '.join(items_sel2)}\n\n"
                    f"🧠 **Insight:**\nEven a small change in capacity can lead to a different set of selected items and a higher value. This helps illustrate how tight capacity constraints impact optimization choices."
                )


        # 19. For capacity {{capacity}}, evaluate and return all optimal item combinations with total weight below {{max_weight}}. Items: {{items}}
        elif "return all optimal item combinations with total weight below" in q:
            match = re.search(r"capacity (\d+), evaluate and return all optimal item combinations with total weight below (\d+)\. items: (.+)", q)
            if match:
                capacity = int(match.group(1))
                max_weight = int(match.group(2))
                items_str = match.group(3)
                items = parse_items_string(items_str)

                dp_table, max_val = knapsack_dp_solve(capacity, items)
                n = len(items)
                unique_combos = set()

                def backtrack(i, w, path):
                    if i == 0:
                        val = sum(item['value'] for name in path for item in items if item['name'] == name)
                        wt = sum(item['weight'] for name in path for item in items if item['name'] == name)
                        if val == max_val and wt <= max_weight:
                            unique_combos.add(tuple(sorted(path)))
                        return
                    if dp_table[i][w] == dp_table[i-1][w]:
                        backtrack(i-1, w, path)
                    if w >= items[i-1]['weight']:
                        prev_val = dp_table[i-1][w - items[i-1]['weight']]
                        if dp_table[i][w] == items[i-1]['value'] + prev_val:
                            backtrack(i-1, w - items[i-1]['weight'], path + [items[i-1]['name']])

                for w in range(capacity + 1):
                    if dp_table[n][w] == max_val:
                        backtrack(n, w, [])

                combinations_str = ", ".join(f"({', '.join(combo)})" if combo else "(empty set)"
                                            for combo in sorted(unique_combos))

                return (
                    f"🔍 **Problem:**\nFind all item combinations that yield the **maximum value** ({max_val}) for a knapsack of capacity {capacity}, "
                    f"with the constraint that total weight must be ≤ {max_weight}.\n\n"
                    f"📘 **Concept:**\nThis explores all **equally optimal paths**, not just one solution.\n\n"
                    f"✅ **Valid Combinations:**\n{combinations_str if combinations_str else '(None found)'}"
                )


        # 20. Identify all suboptimal selections from {{items}} that do not contribute to the max value under capacity {{capacity}}.
        elif "identify all suboptimal selections from" in q and "do not contribute to the max value" in q:
            match = re.search(r"selections from (.+) that do not contribute to the max value under capacity (\d+)", q)
            if match:
                items_str = match.group(1)
                capacity = int(match.group(2))
                items = parse_items_string(items_str)
                
                dp_table, max_val_overall = knapsack_dp_solve(capacity, items)
                
                # A set to store names of items that ARE part of *any* optimal solution
                contributing_items_names = set()
                
                # Function to get all item subsets that yield a specific target_value at a specific weight
                # This function is used to explore all paths leading to the optimal value.
                def get_all_item_subsets_for_target(k_idx, w_target, target_val_check):
                    if k_idx == 0 or w_target == 0:
                        return [[]] if dp_table[k_idx][w_target] == target_val_check else []
                    
                    if dp_table[k_idx][w_target] < target_val_check: # Pruning
                        return []

                    res_subsets = []
                    # Option 1: Exclude item k-1
                    # If current value at (k_idx, w_target) can be reached by excluding item (k_idx-1)
                    if dp_table[k_idx-1][w_target] >= 0 and dp_table[k_idx-1][w_target] == target_val_check: # Check if value matches
                        res_subsets.extend(get_all_item_subsets_for_target(k_idx-1, w_target, target_val_check))
                    
                    # Option 2: Include item k-1
                    item_w = items[k_idx-1]['weight']
                    item_v = items[k_idx-1]['value']
                    
                    if w_target >= item_w:
                        # Check if including this item leads to target_val_check
                        if dp_table[k_idx][w_target] == item_v + dp_table[k_idx-1][w_target - item_w]:
                            prev_subsets = get_all_item_subsets_for_target(k_idx-1, w_target - item_w, target_val_check - item_v)
                            for subset in prev_subsets:
                                res_subsets.append(subset + [items[k_idx-1].get('name', f'Item {k_idx}')])
                    return res_subsets

                # Iterate through all possible weights in the last row of DP table that yield max_val_overall
                for w_opt in range(capacity + 1):
                    if dp_table[len(items)][w_opt] == max_val_overall:
                        # Get all item subsets for this specific (max_val_overall, w_opt) pair
                        current_optimal_subsets_list = get_all_item_subsets_for_target(len(items), w_opt, max_val_overall)
                        for subset_list in current_optimal_subsets_list:
                            for item_name in subset_list:
                                contributing_items_names.add(item_name) # Add to the set of contributing items
                
                # Get all item names present in the original list
                all_item_names_in_list = {item.get('name', f'Item {idx+1}') for idx, item in enumerate(items)}
                
                # Suboptimal items are those in the original list but not found in any optimal solution
                suboptimal_items_not_contributing = list(all_item_names_in_list - contributing_items_names)
                
                if not suboptimal_items_not_contributing:
                    suboptimal_str = "All items potentially contribute to an optimal solution, or no items fit."
                else:
                    suboptimal_str = "Items that are *not* part of any optimal combination (suboptimal selections):\n- " + "\n- ".join(suboptimal_items_not_contributing)

                return (
                    f"For items {items_str} and capacity {capacity}:\n"
                    f"Overall maximum value achievable: {max_val_overall}.\n"
                    f"Items are considered 'suboptimal' in this context if they are not part of *any* combination that achieves the maximum value.\n"
                    f"{suboptimal_str}"
                )
            
def answer_implementation_knapsack(level, question):
    q = question.lower()

    if level == "Level 1":
        # 1. Recursive function
        if "recursive function to solve the knapsack problem" in q:
            return (
                f"🔍 **Problem:**\n"
                f"Write a recursive solution for the classic 0/1 Knapsack problem.\n\n"
                f"📘 **Concept:**\n"
                f"The recursive approach explores two options for each item: include it or exclude it.\n\n"
                f"🧠 **Recursive Logic:**\n"
                f"- Base Case: If no items or capacity is 0 → value is 0\n"
                f"- If current item weight > remaining capacity → skip it\n"
                f"- Else, take the max of:\n"
                f"  - Including the item (value + subproblem with reduced capacity)\n"
                f"  - Excluding the item\n\n"
                f"```python\n"
                f"def knapsack_recursive(wt, val, n, W):\n"
                f"    if n == 0 or W == 0:\n"
                f"        return 0\n"
                f"    if wt[n-1] > W:\n"
                f"        return knapsack_recursive(wt, val, n-1, W)\n"
                f"    else:\n"
                f"        return max(\n"
                f"            val[n-1] + knapsack_recursive(wt, val, n-1, W - wt[n-1]),\n"
                f"            knapsack_recursive(wt, val, n-1, W)\n"
                f"        )\n"
                f"```\n"
                f"⚠️ This is a basic version without optimization, with exponential time complexity O(2^n)."
            )

        # 2. DP with 2D array
        elif "knapsack problem using dynamic programming with a 2d array" in q:
            return (
                f"🔍 **Problem:**\n"
                f"Implement 0/1 Knapsack using bottom-up DP with a 2D table.\n\n"
                f"📘 **Concept:**\n"
                f"Use a DP table where `dp[i][w]` stores the max value with the first `i` items and capacity `w`.\n\n"
                f"```python\n"
                f"def knapsack_dp(wt, val, n, W):\n"
                f"    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]\n"
                f"    for i in range(1, n + 1):\n"
                f"        for w in range(W + 1):\n"
                f"            if wt[i-1] <= w:\n"
                f"                dp[i][w] = max(val[i-1] + dp[i-1][w - wt[i-1]], dp[i-1][w])\n"
                f"            else:\n"
                f"                dp[i][w] = dp[i-1][w]\n"
                f"    return dp[n][W]\n"
                f"```\n"
                f"✅ Time Complexity: O(n × W)"
            )

        # 3. Function that returns the maximum value
        elif "returns the maximum value that can be put in a knapsack of given capacity" in q:
            return (
                f"🔍 **Goal:**\n"
                f"Return the maximum value that fits into a knapsack of a given capacity.\n\n"
                f"📘 **Approach:**\n"
                f"Use bottom-up DP with a 2D table. Same logic as classic 0/1 knapsack.\n\n"
                f"```python\n"
                f"def max_knapsack_value(weights, values, capacity):\n"
                f"    n = len(weights)\n"
                f"    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n"
                f"    for i in range(1, n + 1):\n"
                f"        for w in range(capacity + 1):\n"
                f"            if weights[i-1] <= w:\n"
                f"                dp[i][w] = max(dp[i-1][w], values[i-1] + dp[i-1][w - weights[i-1]])\n"
                f"            else:\n"
                f"                dp[i][w] = dp[i-1][w]\n"
                f"    return dp[n][capacity]\n"
                f"```\n"
                f"📤 **Returns:** Maximum achievable value"
            )

        # 4. Code to initialize DP table
        elif "initialize the dp table for the knapsack problem" in q:
            return (
                f"🔍 **Task:**\n"
                f"Create a 2D DP table of size `(n+1) x (W+1)` initialized to 0.\n\n"
                f"📘 **Why:**\n"
                f"Each cell `dp[i][w]` stores the best value using `i` items and capacity `w`.\n\n"
                f"```python\n"
                f"n = 4  # number of items\n"
                f"W = 10  # max capacity\n"
                f"dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]\n"
                f"```\n"
                f"✅ **DP table initialized.** You can now start filling it."
            )

        # 5. Recursion without memoization
        elif "basic knapsack solution using recursion without memoization" in q:
            return (
                f"🔍 **Problem:**\n"
                f"Implement recursive 0/1 Knapsack **without memoization**.\n\n"
                f"📘 **Concept:**\n"
                f"At each step, either take the item or skip it. This version does not cache results.\n\n"
                f"```python\n"
                f"def knapsack_basic(wt, val, n, W):\n"
                f"    if n == 0 or W == 0:\n"
                f"        return 0\n"
                f"    if wt[n-1] > W:\n"
                f"        return knapsack_basic(wt, val, n-1, W)\n"
                f"    else:\n"
                f"        return max(\n"
                f"            val[n-1] + knapsack_basic(wt, val, n-1, W - wt[n-1]),\n"
                f"            knapsack_basic(wt, val, n-1, W)\n"
                f"        )\n"
                f"```\n"
                f"⚠️ **Warning:** This solution is inefficient for large `n`."
            )


        elif "initializes a knapsack value table with all zeros" in q:
            return (
                "You can initialize a 2D table using list comprehension:\n\n"
                "```python\n"
                "def initialize_knapsack_table(n, W):\n"
                "    return [[0 for _ in range(W + 1)] for _ in range(n + 1)]\n"
                "\n"
                "# Example:\n"
                "dp = initialize_knapsack_table(4, 10)\n"
                "```"
            )

        elif "fill the first row of the knapsack dp table assuming zero items" in q:
            return (
                "🔍 **Explanation:**\n"
                "In a dynamic programming table for the Knapsack problem, each row represents the inclusion of items up to index `i`.\n"
                "The **first row** corresponds to using **0 items** — so no matter what the capacity is, the maximum value we can get is `0`.\n"
                "That's why we initialize the entire first row to `0`.\n\n"
                "📘 **Code to initialize the first row:**\n"
                "```python\n"
                "for w in range(W + 1):\n"
                "    dp[0][w] = 0  # 0 items means 0 value for any capacity\n"
                "```\n"
                "This ensures that all decisions involving '0 items' are handled correctly when we fill the rest of the table."
            )

        elif "fill the first column of the dp table assuming zero capacity" in q:
            return (
                "🔍 **Explanation:**\n"
                "In the Knapsack DP table, each column represents a specific knapsack capacity.\n"
                "The **first column** corresponds to capacity `0`, meaning the knapsack can't hold anything.\n"
                "So no matter how many items we have, the maximum value is `0`.\n"
                "That's why we initialize the first column to `0`.\n\n"
                "📘 **Code to initialize the first column:**\n"
                "```python\n"
                "for i in range(n + 1):\n"
                "    dp[i][0] = 0  # 0 capacity means we can’t pick any item\n"
                "```\n"
                "This step sets a correct base for filling the rest of the table when considering item inclusion."
            )

        elif "prints the selected items used to achieve the maximum value" in q:
            return (
                "You can trace back from the last cell to find which items were included:\n\n"
                "```python\n"
                "def trace_selected_items(dp, wt, val, n, W):\n"
                "    i, w = n, W\n"
                "    while i > 0 and w > 0:\n"
                "        if dp[i][w] != dp[i-1][w]:\n"
                "            print(f\"Item {i-1} selected\")\n"
                "            w -= wt[i-1]\n"
                "        i -= 1\n"
                "```"
        )

        elif "solve the 0/1 knapsack problem using memoization with a dictionary" in q:
            return (
                "Use top-down recursion with memoization to avoid redundant calls:\n\n"
                "```python\n"
                "def knapsack_memo(wt, val, n, W, memo=None):\n"
                "    if memo is None:\n"
                "        memo = {}\n"
                "    if (n, W) in memo:\n"
                "        return memo[(n, W)]\n"
                "    if n == 0 or W == 0:\n"
                "        result = 0\n"
                "    elif wt[n-1] > W:\n"
                "        result = knapsack_memo(wt, val, n-1, W, memo)\n"
                "    else:\n"
                "        result = max(\n"
                "            val[n-1] + knapsack_memo(wt, val, n-1, W - wt[n-1], memo),\n"
                "            knapsack_memo(wt, val, n-1, W, memo)\n"
                "        )\n"
                "    memo[(n, W)] = result\n"
                "    return result\n"
                "```"
            )

        elif "convert a recursive knapsack solution into a top-down memoized version" in q:
            return (
                "Here's how to memoize the classic recursive knapsack:\n\n"
                "```python\n"
                "def knapsack_memo(wt, val, n, W, memo={}):\n"
                "    if (n, W) in memo:\n"
                "        return memo[(n, W)]\n"
                "    if n == 0 or W == 0:\n"
                "        result = 0\n"
                "    elif wt[n-1] > W:\n"
                "        result = knapsack_memo(wt, val, n-1, W, memo)\n"
                "    else:\n"
                "        result = max(\n"
                "            val[n-1] + knapsack_memo(wt, val, n-1, W - wt[n-1], memo),\n"
                "            knapsack_memo(wt, val, n-1, W, memo)\n"
                "        )\n"
                "    memo[(n, W)] = result\n"
                "    return result\n"
                "```"
            )
        
        elif "print the maximum value in the last row of a dp table" in q:
            return (
                "The maximum value will be the last cell of the table:\n\n"
                "```python\n"
                "def max_value_from_dp(dp):\n"
                "    return dp[-1][-1]  # last row, last column\n"
                "```"
            )
        
        elif "create a list of item weights and values from tuples like" in q:
            return (
                "Separate weights and values from item tuples:\n\n"
                "```python\n"
                "items = [(2, 10), (3, 20), (4, 30)]\n"
                "weights = [w for w, _ in items]\n"
                "values = [v for _, v in items]\n"
                "```"
            )
        
        elif "initialize a 1d array to use for space-optimized knapsack implementation" in q:
            return (
                "Initialize a 1D DP array with size W+1:\n\n"
                "```python\n"
                "W = 10\n"
                "dp = [0] * (W + 1)\n"
                "```"
            )


        elif "solves the 0/1 knapsack problem using a 1d dp array" in q:
            return (
                "This version reduces space from O(n×W) to O(W):\n\n"
                "```python\n"
                "def knapsack_1d(wt, val, n, W):\n"
                "    dp = [0] * (W + 1)\n"
                "    for i in range(n):\n"
                "        for w in range(W, wt[i] - 1, -1):\n"
                "            dp[w] = max(dp[w], val[i] + dp[w - wt[i]])\n"
                "    return dp[W]\n"
                "```"
            )


    if level == "Level 2":
        if "implement the knapsack problem using recursion with memoization" in q:
            return (
                "To solve the 0/1 Knapsack problem using recursion with memoization, we use a dictionary to store subproblem results.\n"
                "```python\n"
                "def knapsack_memo(wt, val, n, W, memo={}):\n"
                "    if (n, W) in memo:\n"
                "        return memo[(n, W)]\n"
                "    if n == 0 or W == 0:\n"
                "        return 0\n"
                "    if wt[n-1] > W:\n"
                "        result = knapsack_memo(wt, val, n-1, W, memo)\n"
                "    else:\n"
                "        result = max(\n"
                "            knapsack_memo(wt, val, n-1, W, memo),\n"
                "            val[n-1] + knapsack_memo(wt, val, n-1, W - wt[n-1], memo)\n"
                "        )\n"
                "    memo[(n, W)] = result\n"
                "    return result\n"
                "```"
            )

        elif "bottom-up dynamic programming solution to the knapsack problem using tabulation" in q:
            return (
                "This solution builds a DP table from the ground up using tabulation.\n"
                "```python\n"
                "def knapsack_tabulation(wt, val, n, W):\n"
                "    dp = [[0] * (W + 1) for _ in range(n + 1)]\n"
                "    for i in range(1, n + 1):\n"
                "        for w in range(W + 1):\n"
                "            if wt[i-1] <= w:\n"
                "                dp[i][w] = max(val[i-1] + dp[i-1][w - wt[i-1]], dp[i-1][w])\n"
                "            else:\n"
                "                dp[i][w] = dp[i-1][w]\n"
                "    return dp[n][W]\n"
                "```"
            )

        elif "modify your knapsack implementation to return the list of selected items" in q:
            return (
                "This approach traces back from the DP table to identify selected items.\n"
                "```python\n"
                "def knapsack_with_items(wt, val, n, W):\n"
                "    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]\n"
                "    for i in range(1, n + 1):\n"
                "        for w in range(W + 1):\n"
                "            if wt[i-1] <= w:\n"
                "                dp[i][w] = max(dp[i-1][w], val[i-1] + dp[i-1][w - wt[i-1]])\n"
                "            else:\n"
                "                dp[i][w] = dp[i-1][w]\n"
                "    # Trace selected items\n"
                "    w = W\n"
                "    items = []\n"
                "    for i in range(n, 0, -1):\n"
                "        if dp[i][w] != dp[i-1][w]:\n"
                "            items.append(i-1)\n"
                "            w -= wt[i-1]\n"
                "    return dp[n][W], items\n"
                "```"
            )

        elif "solve the knapsack problem and print both the maximum value and the selected item indices" in q:
            return (
                "This combines DP and trace-back to print the optimal value and chosen indices.\n"
                "```python\n"
                "def knapsack_print_selected(wt, val, n, W):\n"
                "    dp = [[0]*(W+1) for _ in range(n+1)]\n"
                "    for i in range(1, n+1):\n"
                "        for w in range(W+1):\n"
                "            if wt[i-1] <= w:\n"
                "                dp[i][w] = max(val[i-1] + dp[i-1][w - wt[i-1]], dp[i-1][w])\n"
                "            else:\n"
                "                dp[i][w] = dp[i-1][w]\n"
                "    max_val = dp[n][W]\n"
                "    selected = []\n"
                "    w = W\n"
                "    for i in range(n, 0, -1):\n"
                "        if dp[i][w] != dp[i-1][w]:\n"
                "            selected.append(i-1)\n"
                "            w -= wt[i-1]\n"
                "    print('Maximum Value:', max_val)\n"
                "    print('Selected Item Indices:', selected)\n"
                "```"
            )

        elif "tabulation-based knapsack solution that minimizes space usage by only keeping two rows" in q:
            return (
                "To save space, we only store the current and previous rows.\n"
                "```python\n"
                "def knapsack_2rows(wt, val, n, W):\n"
                "    dp = [[0] * (W + 1) for _ in range(2)]\n"
                "    for i in range(1, n + 1):\n"
                "        for w in range(W + 1):\n"
                "            if wt[i-1] <= w:\n"
                "                dp[i % 2][w] = max(val[i - 1] + dp[(i - 1) % 2][w - wt[i - 1]], dp[(i - 1) % 2][w])\n"
                "            else:\n"
                "                dp[i % 2][w] = dp[(i - 1) % 2][w]\n"
                "    return dp[n % 2][W]\n"
                "```"
)


        elif "build a reusable function in python that solves knapsack using memoization and handles multiple test cases" in q:
            return (
                "This function can be called repeatedly with different inputs.\n"
                "```python\n"
                "def knapsack_multi(wt, val, n, W):\n"
                "    memo = {}\n"
                "    def helper(n, W):\n"
                "        if (n, W) in memo:\n"
                "            return memo[(n, W)]\n"
                "        if n == 0 or W == 0:\n"
                "            return 0\n"
                "        if wt[n-1] > W:\n"
                "            result = helper(n-1, W)\n"
                "        else:\n"
                "            result = max(helper(n-1, W), val[n-1] + helper(n-1, W - wt[n-1]))\n"
                "        memo[(n, W)] = result\n"
                "        return result\n"
                "    return helper(n, W)\n"
                "```"
            )

        elif "implement the knapsack using dynamic programming in a programming language of your choice" in q:
            return (
                "Here's a Python implementation with time complexity explanation:\n"
                "```python\n"
                "def knapsack(wt, val, n, W):\n"
                "    dp = [[0 for _ in range(W+1)] for _ in range(n+1)]\n"
                "    for i in range(1, n+1):\n"
                "        for w in range(W+1):\n"
                "            if wt[i-1] <= w:\n"
                "                dp[i][w] = max(dp[i-1][w], val[i-1] + dp[i-1][w - wt[i-1]])\n"
                "            else:\n"
                "                dp[i][w] = dp[i-1][w]\n"
                "    return dp[n][W]\n"
                "# Time Complexity: O(n*W)\n"
                "```"
            )

        elif "read from user input" in q:
            return (
                "This version accepts weights and values from user input:\n"
                "```python\n"
                "n = int(input(\"Enter number of items: \"))\n"
                "wt = list(map(int, input(\"Enter weights: \").split()))\n"
                "val = list(map(int, input(\"Enter values: \").split()))\n"
                "W = int(input(\"Enter knapsack capacity: \"))\n"
                "# Call knapsack(wt, val, n, W)\n"
                "```\n"
            )


        elif "uses a dictionary for memoization instead of a dp table" in q:
            return (
                "This avoids using a 2D list entirely and replaces it with a dictionary cache:\n"
                "(Same as memoized version shown earlier.)\n"
                "```python\n"
                "def knapsack_dict(wt, val, n, W, memo={}):\n"
                "    if (n, W) in memo:\n"
                "        return memo[(n, W)]\n"
                "    if n == 0 or W == 0:\n"
                "        return 0\n"
                "    if wt[n-1] > W:\n"
                "        result = knapsack_dict(wt, val, n-1, W, memo)\n"
                "    else:\n"
                "        result = max(knapsack_dict(wt, val, n-1, W, memo), val[n-1] + knapsack_dict(wt, val, n-1, W - wt[n-1], memo))\n"
                "    memo[(n, W)] = result\n"
                "    return result\n"
                "```"
            )

        elif "randomly generated within a given range" in q:
            return (
                "You can generate random values and weights using the random module:\n"
                "```python\n"
                "import random\n"
                "n = 5\n"
                "W = 15\n"
                "wt = [random.randint(1, 10) for _ in range(n)]\n"
                "val = [random.randint(10, 100) for _ in range(n)]\n"
                "print(\"Weights:\", wt)\n"
                "print(\"Values:\", val)\n"
                "# Then call knapsack(wt, val, n, W)\n"
                "```\n"
    )

    
    if level == "Level 3":
        if "space-optimized version of the knapsack problem using a 1d array" in q:
            return (
                "🔍 **Goal:** Optimize space in Knapsack by using a single 1D array.\n\n"
                "📘 **Idea:** We update values in reverse to preserve previous values in the same array.\n\n"
                "```python\n"
                "def knapsack_1d(wt, val, n, W):\n"
                "    dp = [0] * (W + 1)\n"
                "    for i in range(n):\n"
                "        for w in range(W, wt[i] - 1, -1):\n"
                "            dp[w] = max(dp[w], val[i] + dp[w - wt[i]])\n"
                "    return dp[W]\n"
                "```\n\n"
                "✅ **Space Complexity:** O(W)"
            )

        elif "print all possible optimal item combinations that give the same maximum value" in q:
            return (
                "🔍 **Advanced Goal:** Track all item sets that yield the same optimal value.\n\n"
                "📘 **Approach:** Build a DP table and backtrack all paths that lead to max value.\n\n"
                "This is complex and requires tracking item choices during table build.\n"
                "Hint: At each cell, store not only value but possible item sets.\n"
                "Due to combinatorial nature, this is recommended only for small n."
            )

        elif "knapsack problem using bitmasking for small input sizes" in q:
            return (
                "🔍 **Use Case:** For small n (<=20), bitmasking is faster.\n\n"
                "📘 **Idea:** Try all subsets and keep track of the best one.\n\n"
                "```python\n"
                "def knapsack_bitmask(wt, val, n, W):\n"
                "    max_val = 0\n"
                "    for mask in range(1 << n):\n"
                "        total_w = total_v = 0\n"
                "        for i in range(n):\n"
                "            if mask & (1 << i):\n"
                "                total_w += wt[i]\n"
                "                total_v += val[i]\n"
                "        if total_w <= W:\n"
                "            max_val = max(max_val, total_v)\n"
                "    return max_val\n"
                "```\n\n"
                "⚠️ Works best for small inputs due to O(2^n) complexity."
            )

        elif "handle multiple knapsacks" in q:
            return (
                "🔍 **Multi-Knapsack Variant:** Distribute items among multiple bags.\n\n"
                "📘 **Idea:** Use 3D DP table if 2 knapsacks: dp[i][w1][w2]\n\n"
                "This increases time and space complexity to O(n×W1×W2), so use carefully.\n"
                "Advanced variants may also allow fractional distribution or constraints."
            )

        elif "solve the knapsack problem for large inputs (up to 10^5 items)" in q:
            return (
                "🔍 **Large Input Strategy:**\n"
                "Use 1D array + early pruning, greedy filtering, or monotonic queues.\n\n"
                "📘 **Tip:** Avoid 2D DP for n > 10^4. Consider using\n"
                " - Meet-in-the-middle\n"
                " - Sparse DP\n"
                " - Scaling/approximation methods\n"
                " - C++ for performance-critical cases\n\n"
                "For values instead of weights, value-based DP (W very large, values small) can help."
            )

        elif "class-based implementation of knapsack" in q:
            return (
                "🔍 **OOP Style:** Encapsulate Knapsack logic in a class.\n\n"
                "```python\n"
                "class KnapsackSolver:\n"
                "    def __init__(self):\n"
                "        self.items = []\n"
                "    def add_item(self, weight, value):\n"
                "        self.items.append((weight, value))\n"
                "    def remove_item(self, index):\n"
                "        self.items.pop(index)\n"
                "    def solve(self, W):\n"
                "        n = len(self.items)\n"
                "        wt, val = zip(*self.items)\n"
                "        dp = [0]*(W+1)\n"
                "        for i in range(n):\n"
                "            for w in range(W, wt[i]-1, -1):\n"
                "                dp[w] = max(dp[w], val[i] + dp[w - wt[i]])\n"
                "        return dp[W]\n"
                "```"
            )

        elif "using functools.lru_cache" in q:
            return (
                "🔍 **Memoization using Decorators:** Python’s functools.lru_cache makes it easy.\n\n"
                "```python\n"
                "from functools import lru_cache\n"
                "@lru_cache(maxsize=None)\n"
                "def knapsack(n, W):\n"
                "    if n == 0 or W == 0: return 0\n"
                "    if weights[n-1] > W:\n"
                "        return knapsack(n-1, W)\n"
                "    return max(knapsack(n-1, W), values[n-1] + knapsack(n-1, W - weights[n-1]))\n"
                "```\n\n"
                "Pass `weights`, `values` as globals or wrap in closure/class for full encapsulation."
            )

        elif "reads item weights/values from a file" in q:
            return (
                "🔍 **File I/O + DP Logic:** Read inputs from a text file, parse, then solve.\n\n"
                "```python\n"
                "def read_items(filename):\n"
                "    with open(filename) as f:\n"
                "        n, W = map(int, f.readline().split())\n"
                "        wt, val = [], []\n"
                "        for _ in range(n):\n"
                "            w, v = map(int, f.readline().split())\n"
                "            wt.append(w)\n"
                "            val.append(v)\n"
                "    return wt, val, n, W\n"
                "```\n"
                "Call your standard knapsack(dp) function afterward."
            )

        elif "visualization tool for the knapsack dp table" in q:
            return (
                "🔍 **Visualization Idea:** Use color or grid output to highlight decisions.\n\n"
                "📘 Tools: Use matplotlib (heatmap), seaborn, or even text UI.\n"
                "Print `*` when value is updated.\n\n"
                "```python\n"
                "for i in range(n+1):\n"
                "    for w in range(W+1):\n"
                "        print(f'{dp[i][w]:3}', end='*') if changed else print(f'{dp[i][w]:3}', end=' ')\n"
                "    print()\n"
                "```\n"
    )


        elif "selecting one item makes some others unavailable" in q:
            return (
                "🔍 **Conflict Knapsack:** Some items are mutually exclusive (e.g., batteries + solar panel).\n\n"
                "📘 **Approach:**\n"
                "Use conflict graph. Before including item `i`, check that no conflicting items are selected.\n\n"
                "🔧 **Model Conflicts:**\n"
                "Store conflicts as a dict: `conflict_map = {i: [j1, j2]}`\n"
                "While tracing combinations, exclude those where conflicts exist.\n\n"
                "⚠️ This adds complexity and requires constraint-aware DP or backtracking."
            )


    return "Answer generation for this Knapsack implementation question is not implemented yet."

# --- MAIN ANSWER GENERATION FUNCTIONS (END) ---


# --- TEST BLOCKS (START) ---

def test_answer_algorithmic_knapsack():
    print("\n--- Testing Level 1 Algorithmic Knapsack Answers ---\n")
    questions_algo_lvl1 = [
        "Given a knapsack with capacity 10 and one item of weight 5 and value 10, what is the maximum value that can be obtained?",
        "If you have a knapsack of capacity 7 and two items: item1(weight=3, value=30), item2(weight=4, value=50), what is the best value you can carry?",
        "Determine whether to include an item with weight 8 and value 100 in a knapsack of capacity 5.",
        "Find the optimal value that can be achieved by choosing between two items (weight=2, value=10) and (weight=3, value=25) with knapsack capacity 4.",
        "You are given 1 item with weight 6 and value 70. Can it fit into a knapsack of capacity 5?",
        "If you are allowed to select only one item, which should you choose to maximize value: (weight=5, value=100) or (weight=6, value=90) for capacity 8?",
        "What is the total value if you put item1(weight=7, value=80) into a knapsack with capacity 5?",
        "Out of three items (w,v): (2,30), (3,40), and (4,50), which two would you choose for capacity 6?",
        "Can a knapsack of capacity 8 hold both item A(weight=3) and item B(weight=6)?",
        "Find the maximum value using only the first 3 items in a list of items, given a knapsack of capacity 10.", # Conceptual, will provide conceptual answer
        "How much unused space remains if item(weight=4) is added to a knapsack of capacity 10?",
        "Can both items with weights 5 and 4 be selected if the capacity is 10?",
        "Which of the following fits best into the knapsack (capacity=7): item A(4,60), item B(3,40)?",
        "Given two items (w,v): (5,50), (4,45) and capacity 8, what is the best value achievable if only one can be selected?",
        "How many total items can you fit if each has weight 2 and the knapsack capacity is 7?",
        "If you have item A(weight=2, value=10) and item B(weight=3, value=12), which gives higher value per weight ratio?",
        "Choose the best single item from the list: A(5,100), B(6,120), C(4,80) for a knapsack of capacity 7.",
        "What is the total weight of items selected if you choose item A(weight=3) and item B(weight=4) within capacity 10?",
        "If each item has weight 1 and value 1, how many items can you choose to maximize total value with capacity 5?",
        "Can the knapsack of size 10 be exactly filled using two items with weights 4 and 6?"
    ]
    for i, q in enumerate(questions_algo_lvl1):
        print(f"Test Case {i+1}:")
        print(f"Question: {q}")
        print(f"Answer:\n{answer_algorithmic_knapsack('Level 1', q)}\n{'-'*50}\n")

    print("\n--- Testing Level 2 Algorithmic Knapsack Answers ---\n")
    questions_algo_lvl2 = [
        "Given a knapsack of capacity 7 and items: A(weight=1, value=10), B(weight=3, value=30), and C(weight=4, value=50), find the maximum value achievable.",
        "Choose a combination of items that maximizes value without exceeding the knapsack capacity 5. Items: (weight, value) - (2,30), (3,40), (1,20).", # Corrected item string format
        "With a knapsack capacity of 6, how many items from the list can be included: (1,10), (2,20), (3,30)?", # Corrected item string format
        "Find the optimal selection for a 0/1 Knapsack problem with capacity 5 using the items (2,3), (3,4), (4,5).", # Corrected item string format
        "Determine the maximum value possible by selecting items without repetition. Capacity: 8, Items: (3,4), (4,5), (5,6).", # Corrected item string format
        "Which items will be included if we want to maximize the value with knapsack size 10? Items: (5,10), (4,40), (6,30).", # Corrected item string format
        "Calculate the best value that can be achieved with items of weights {2,3,4} and values {3,4,5} for knapsack capacity 5.",
        "Given the capacity of knapsack as 5, find if you can pick at least two items from: (1,1), (2,2), (3,3) to fully utilize the knapsack.", # Corrected item string format
        "Select two items from the list: (2,10), (3,20), (4,25) to maximize value in a knapsack of capacity 6.", # Corrected item string format
        "If you have a knapsack of capacity 7, which item combinations from (1,10), (2,20), (3,25) will yield a value above 30?", # Corrected item string format
        "Determine the maximum value for fractional selection not allowed. Capacity: 6, Items: (2,10), (3,15), (4,20).", # Corrected item string format
        "From the following items: (1,5), (2,10), (3,12), determine which combination offers the highest value within a capacity of 4.", # Corrected item string format
        "How many combinations of items are possible within capacity 5 for item weights {1,2,3}?",
        "List the possible values that can be achieved with combinations from items: (1,10), (2,20), (3,30), within a capacity of 4.", # Corrected item string format
        "Estimate the maximum number of items that can be selected from the list (1,10), (1,5), (2,15) within a capacity of 3.", # Corrected item string format
        "Determine which of the items from A(w=5,v=10), B(w=1,v=1), C(w=4,v=8) are not worth including due to low value-to-weight ratio. Capacity: 6.",
        "From the set (2,10), (3,15), (4,25), identify which pair maximizes total value within the knapsack limit 5.", # Corrected item string format
        "What is the difference in total value if you exclude the heaviest item from selection in knapsack capacity 7? Items: A(w=3,v=30), B(w=4,v=50).",
        "Can you exactly fill a knapsack of size 7 with the following item weights: {2,3,5}?",
        "Out of all item combinations from (2,10), (3,15), (4,20), which one gives highest value under the limit 6 while leaving minimum unused space?" # Corrected item string format
    ]
    for i, q in enumerate(questions_algo_lvl2):
        print(f"Test Case {i+1}:")
        print(f"Question: {q}")
        print(f"Answer:\n{answer_algorithmic_knapsack('Level 2', q)}\n{'-'*50}\n")

    print("\n--- Testing Level 3 Algorithmic Knapsack Answers ---\n")
    questions_algo_lvl3 = [
        "Given a knapsack capacity of 5, and items A(w=2,v=35), B(w=3,v=40), C(w=1,v=10), find the maximum value achievable using dynamic programming.",
        "Using 0/1 Knapsack DP approach, find which items are selected for capacity 5 with the item list A(w=2,v=30), B(w=3,v=40), C(w=1,v=10).",
        "Calculate the DP table for solving the knapsack problem with capacity 5 and weights {2,3,1}, values {30,40,10}.",
        "Given item list A(w=2,v=30), B(w=3,v=40), C(w=1,v=10), compute both maximum value and the items chosen for knapsack size 5.",
        "Which subset of items from A(w=2,v=30), B(w=3,v=40), C(w=1,v=10) results in the maximum value without exceeding capacity 5, and what is the remaining space?",
        "For the knapsack instance with capacity 5 and items A(w=2,v=30), B(w=3,v=40), C(w=1,v=10), construct and fill the DP matrix step-by-step.",
        "Given capacity 5, explain how the solution changes if item B is excluded from A(w=2,v=30), B(w=3,v=40), C(w=1,v=10).",
        "Compute the optimal value for two knapsacks of capacity 3 and 4, where the items (w=1,v=10), (w=2,v=20), (w=3,v=30), (w=4,v=40) must be divided optimally.",
        "Using space-optimized DP, solve the 0/1 knapsack problem with capacity 5 and items A(w=2,v=30), B(w=3,v=40), C(w=1,v=10).",
        "If the total weight must not exceed 5, but at least 2 items must be selected from A(w=2,v=30), B(w=3,v=40), C(w=1,v=10), what is the optimal value?",
        "How does the result change if values of items in A(w=2,v=30), B(w=3,v=40), C(w=1,v=10) are increased by 10% and capacity is 5?",
        "Given items with close weight-value ratios in A(w=3,v=30), B(w=4,v=40), C(w=5,v=50), find which are most beneficial under capacity 7.",
        "Use memoization to solve the knapsack problem for capacity 5 and the item set A(w=2,v=30), B(w=3,v=40), C(w=1,v=10).",
        "Modify your solution to the knapsack problem with capacity 5 so that the value of each item is doubled. What is the new result? Items: A(w=2,v=30), B(w=3,v=40), C(w=1,v=10).",
        "Find the optimal solution for the given set A(w=2,v=30), B(w=3,v=40), C(w=1,v=10) and capacity 5, ensuring at least one item weighs more than 2.",
        "Determine which item(s) can be removed from the solution without decreasing the maximum achievable value for capacity 5. Items: A(w=2,v=30), B(w=3,v=40), C(w=1,v=10).",
        "Find the optimal subset of A(w=2,v=30), B(w=3,v=40), C(w=1,v=10) for a knapsack of capacity 5 using top-down dynamic programming.",
        "How does the optimal solution differ when the knapsack capacity changes from 5 to 6? Items: A(w=2,v=30), B(w=3,v=40), C(w=1,v=10).",
        "For capacity 5, evaluate and return all optimal item combinations with total weight below 4. Items: A(w=2,v=30), B(w=3,v=40), C(w=1,v=10).",
        "Identify all suboptimal selections from A(w=2,v=30), B(w=3,v=40), C(w=1,v=10) that do not contribute to the max value under capacity 5."
    ]
    for i, q in enumerate(questions_algo_lvl3):
        print(f"Test Case {i+1}:")
        print(f"Question: {q}")
        print(f"Answer:\n{answer_algorithmic_knapsack('Level 3', q)}\n{'-'*50}\n")

def test_answer_implementation_knapsack():

    print("\n--- Testing Level 1 Implementation Knapsack Answers ---\n")

    questions_impl_lvl1 = [
        "Write a recursive function to solve the Knapsack problem.",
        "Implement the Knapsack problem using dynamic programming with a 2D array.",
        "Create a function that returns the maximum value that can be put in a knapsack of given capacity using Knapsack logic.",
        "Write code to initialize the DP table for the Knapsack problem.",
        "Implement a basic Knapsack solution using recursion without memoization.",
        "Write a Python function that initializes a knapsack value table with all zeros.",
        "Implement a loop to fill the first row of the knapsack DP table assuming zero items.",
        "Implement a loop to fill the first column of the DP table assuming zero capacity.",
        "Modify a given DP table so that it prints the selected items used to achieve the maximum value.",
        "Write a function to solve the 0/1 Knapsack problem using memoization with a dictionary.",
        "Convert a recursive knapsack solution into a top-down memoized version.",
        "Write a Python function to print the maximum value in the last row of a DP table for the knapsack.",
        "Write code to create a list of item weights and values from tuples like (weight, value).",
        "Initialize a 1D array to use for space-optimized knapsack implementation.",
        "Create a function that solves the 0/1 Knapsack problem using a 1D DP array."
    ]

    for i, q in enumerate(questions_impl_lvl1):
            print(f"Test Case {i+1}:")
            print(f"Question: {q}")
            print("Answer:")
            print(answer_implementation_knapsack("Level 1", q))
            print('-' * 50 + '\n')

    print("\n--- Testing Level 2 Implementation Knapsack Answers ---\n")
    questions_impl_lvl2 = [
        "Implement the Knapsack problem using recursion with memoization.",
        "Write a bottom-up dynamic programming solution to the Knapsack problem using tabulation.",
        "Modify your Knapsack implementation to return the list of selected items.",
        "Write code to solve the Knapsack problem and print both the maximum value and the selected item indices.",
        "Implement a tabulation-based Knapsack solution that minimizes space usage by only keeping two rows.",
        "Build a reusable function in Python that solves Knapsack using memoization and handles multiple test cases.",
        "Implement the Knapsack using dynamic programming in a programming language of your choice and analyze time complexity.",
        "Write code to solve the Knapsack problem where weights and values are read from user input.",
        "Implement a Knapsack solution that uses a dictionary for memoization instead of a DP table.",
        "Implement a Knapsack algorithm where values and weights are randomly generated within a given range.",
        "Implement the Knapsack problem using recursion with memoization.",
        "Write a bottom-up dynamic programming solution to the Knapsack problem using tabulation.",
        "Modify your Knapsack implementation to return the list of selected items.",
        "Write code to solve the Knapsack problem and print both the maximum value and the selected item indices.",
        "Implement a tabulation-based Knapsack solution that minimizes space usage by only keeping two rows.",
        "Build a reusable function in Python that solves Knapsack using memoization and handles multiple test cases.",
        "Implement the Knapsack using dynamic programming in a programming language of your choice and analyze time complexity.",
        "Write code to solve the Knapsack problem where weights and values are read from user input.",
        "Implement a Knapsack solution that uses a dictionary for memoization instead of a DP table.",
        "Implement a Knapsack algorithm where values and weights are randomly generated within a given range."
    ]

    for i, q in enumerate(questions_impl_lvl2):
            print(f"Test Case {i+1}:")
            print(f"Question: {q}")
            print("Answer:")
            print(answer_implementation_knapsack("Level 2", q))
            print('-' * 50 + '\n')

    print("\n--- Testing Level 3 Implementation Knapsack Answers ---\n")
    questions_impl_lvl3 = [
        "Implement a space-optimized version of the Knapsack problem using a 1D array.",
        "Write a program to solve the Knapsack problem and print all possible optimal item combinations that give the same maximum value.",
        "Implement the Knapsack problem using bitmasking for small input sizes and compare its performance with standard DP.",
        "Modify your DP-based Knapsack code to handle multiple knapsacks (multi-knapsack variant).",
        "Solve the Knapsack problem for large inputs (up to 10^5 items) with time and space optimizations.",
        "Build a class-based implementation of Knapsack with item insertion, deletion, and re-evaluation support.",
        "Implement a recursive + memoized Knapsack solution using functools.lru_cache and validate performance.",
        "Write a Python program that reads item weights/values from a file and solves the Knapsack problem efficiently.",
        "Implement a visualization tool for the Knapsack DP table that highlights the decision process.",
        "Solve a modified Knapsack problem where selecting one item makes some others unavailable (conflict constraints)."
    ]

    for i, q in enumerate(questions_impl_lvl3):
            print(f"Test Case {i+1}:")
            print(f"Question: {q}")
            print("Answer:")
            print(answer_implementation_knapsack("Level 3", q))
            print('-' * 50 + '\n')

# --- Main execution block to run tests ---
if __name__ == "__main__":
    # test_answer_algorithmic_knapsack()
    test_answer_implementation_knapsack()