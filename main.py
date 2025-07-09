import streamlit as st
import json
import os
import random
import re
import string
from answer_dispatcher import get_answer

import importlib.util
import os
if "objective_state" not in st.session_state:
    st.session_state["objective_state"] = {
        "mcq_opened": False,
        "tf_opened": False,
        "mcq_reset": False,
        "tf_reset": False
    }

if "objective_expanded" not in st.session_state:
    st.session_state["objective_expanded"] = {"mcq": False, "tf": False}
if "objective_reset" not in st.session_state:
    st.session_state["objective_reset"] = {"mcq": False, "tf": False}
# if "generate_clicked" not in st.session_state:
#     st.session_state["generate_clicked"] = False


def load_objective_questions(topic, category, level):
    import importlib.util
    import os

    def normalize_name(name):
        return name.lower().replace(" ", "_")

    filepath = f"objective/{normalize_name(topic)}_{normalize_name(category)}.py"
    if not os.path.exists(filepath):
        return [], []

    try:
        spec = importlib.util.spec_from_file_location("objective_module", filepath)
        objective_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(objective_module)

        mcqs = getattr(objective_module, "mcq_questions", {}).get(level, [])
        tfqs = getattr(objective_module, "tf_questions", {}).get(level, [])
        return mcqs, tfqs
    except Exception as e:
        return [], []

def normalize_name(name):
    return name.lower().replace(" ", "_")

def load_theory(topic, category, level):
    theory_file = f"info/{normalize_name(topic)}_{normalize_name(category)}.py"
    if not os.path.exists(theory_file):
        return None

    try:
        spec = importlib.util.spec_from_file_location("theory_module", theory_file)
        theory_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(theory_module)

        theory_dict = getattr(theory_module, "theory_content", {})
        return theory_dict.get(level, "🔍 No theory available for this level.")
    except Exception as e:
        return f"⚠️ Error loading theory: {str(e)}"

def display_theory_if_available(topic, category, level):
    theory_text = load_theory(topic, category, level)
    if theory_text:
        with st.expander("📘 Click to view theory for this topic and level", expanded=False):
            st.markdown(theory_text)

# -------------------------------
# Configurable string rules by difficulty level
# -------------------------------
STRING_LENGTH_BY_LEVEL = {
    "Level 1": (2, 4),
    "Level 2": (3, 5),
    "Level 3": (4, 8)
}

# -------------------------------
# Curated LCS string bank
# -------------------------------
lcs_string_bank = {
    "PRINTER": ["PRINT", "INTER", "PRNTR", "PINTER", "PAINTER", "PINTEREST", "PRINTERS", "PRIT", "REPRINTER", "PRITE"],
    "COMMUNICATION": ["COMMUTATION", "CONNOTATION", "COMMUNICATE", "COMMUNICATIONS", "COMMUNAL", "CATION", "COMMUTE", "COMMISSION", "COMIC", "COMMUNION"],
    "RESEARCH": ["SEARCH", "RESEEK", "RESECT", "RECHECK", "REACT", "REACH", "RESET", "RESELL", "RSEARCH", "REAP"],
    "DYNAMIC": ["DYNMAIC", "DYNMIC", "DYNAMICALLY", "DYNMICITY", "DYNA", "DYNNAMIC", "DYANMIC", "DYNAMICS", "DNAMIC", "DYMC"],
    "DATABASE": ["DATA", "DATABANK", "DATACENTER", "DATABOARD", "DATABASES", "DBASE", "DATAPOOL", "DATABOX", "DATASHARE", "DBASEX"],
    "FUNCTION": ["FUNCT", "FNCTION", "FUNC", "FUNCTIONAL", "FUNTION", "FUNKTION", "FUNKCION", "FUNCLIB", "FUNTIONALITY", "FUNDAMENTAL"],
    "GRAPHICS": ["GRAPHIC", "GRAPH", "GRPHCS", "GRPH", "GRAFX", "GRAPHS", "GRPHX", "GRAHIC", "GRAFIX", "GRAFIK"],
    "LANGUAGE": ["LANG", "LINGO", "LANGS", "LANGUAGES", "LNGUAGE", "LNGAUGE", "LANGUE", "LANGUG", "LANUGE", "LNGG"],
    "EXPERIMENT": ["EXPERIENCE", "EXPERIMNT", "EXPER", "EXPRMNT", "EXPMNT", "EXPRT", "EXPMNTAL", "EXPERM", "EXPRMNTAL", "EXP"],
    "VARIABLE": ["VAR", "VARNAME", "VARIANT", "VARIABLES", "VARIBL", "VARIBALE", "VBL", "VARE", "VRBLE", "VARL"]
}

# -------------------------------
# Compute actual LCS from two strings
# -------------------------------
def compute_lcs(X, Y):
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    i, j = m, n
    lcs = []
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs.append(X[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return ''.join(reversed(lcs))

# -------------------------------
# Use the bank to generate high-quality LCS strings
# -------------------------------
def generate_lcs_pair_from_bank(level):
    if level == "Level 1":
        min_len, max_len = 3, 5
    elif level == "Level 2":
        min_len, max_len = 6, 8
    else:
        min_len, max_len = 8, 12

    candidates = [key for key in lcs_string_bank if min_len <= len(key) <= max_len]
    if not candidates:
        candidates = list(lcs_string_bank.keys())

    attempts = 0
    while attempts < 10:
        base = random.choice(candidates)
        variants = [v for v in lcs_string_bank[base] if min_len <= len(v) <= max_len]
        if len(variants) >= 2:
            s1, s2 = random.sample(variants, 2)
            return s1, s2, compute_lcs(s1, s2)
        attempts += 1

    # Fallback: pick any base and any two variants regardless of length
    base = random.choice(list(lcs_string_bank.keys()))
    variants = lcs_string_bank[base]
    if len(variants) >= 2:
        s1, s2 = random.sample(variants, 2)
        return s1, s2, compute_lcs(s1, s2)

    # Fallback to old method if everything fails
    return generate_lcs_pair(level)


# -------------------------------
# Placeholder generator
# -------------------------------
def generate_placeholder_value(name, level):
    name = name.lower()
    LCS_KEYS = ["string1", "string2", "a", "b", "s1", "s2", "str1", "str2", "str_a", "str_b"]

    if name in LCS_KEYS or name == "expected_lcs":
        if "lcs_pair" not in generate_placeholder_value.cache:
            s1, s2, lcs = generate_lcs_pair_from_bank(level)
            generate_placeholder_value.cache["lcs_pair"] = (s1, s2, lcs)
        s1, s2, lcs = generate_placeholder_value.cache["lcs_pair"]
        if name in ["string1", "a", "s1", "str1", "str_a"]:
            return s1
        elif name in ["string2", "b", "s2", "str2", "str_b"]:
            return s2
        else:
            return lcs
        
    if name == "function_name":
        return random.choice(["lcs", "lcs_dp", "lcs_bottom_up", "compute_lcs"])


    if "string" in name or "str" in name or "text" in name or "input" in name:
        return generate_string(level)
    
    if name in {"s", "s1", "s2"}:
    # Generate a random uppercase string to simulate palindromic checks
        length = random.randint(4, 8) if level == "Level 3" else random.randint(3, 6)
        return ''.join(random.choices(string.ascii_lowercase, k=length))

    
    if name == "digits":
    # Generate a random numeric string for decoding problems
        return ''.join(random.choices("123456789", k=random.randint(4, 7)))

    if name == "expression":
    # Example boolean expression for Boolean Parenthesization
        return random.choice(["T|F", "T&F|T", "T^F|T", "F&T^F", "T^T|F"])


    if name in {"index1", "index2", "last_row", "last_col"}:
        return str(random.randint(1, 5))
    
    if name in {"start", "end"}:
        return str(random.randint(0, 5))
    
    if name == "graph":
    # Simple adjacency matrix of size 6x6 with random weights
        size = 6
        graph = [[0 if i == j else random.randint(1, 10) for j in range(size)] for i in range(size)]
        return graph



    if any(key in name for key in ["n", "k", "num", "count", "index", "idx", "length", "len", "size", "val", "pos",
                                    "x", "limit", "terms", "last", "start", "end", "mod", "days", "levels", "steps", "layers",
                                    "bars", "charges", "frames", "years", "devices", "cycles", "capacity", "weight", "value",
                                    "max_weight", "budget"]):
        if "mod" in name:
            return str(random.choice([97, 1000, 10007, 1007]))
        return str(random.randint(2, 20))

    if name.lower() in {"w1", "w2", "v1", "v2"}:
        return str(random.randint(1, 25)) if "w" in name else str(random.randint(10, 100))

    if name in {"char1", "char2"}:
        return random.choice(string.ascii_uppercase)

    if "dna" in name or "sequence" in name:
        return ''.join(random.choices("ACGT", k=random.randint(6, 12)))

    if "melody" in name:
        return '-'.join(random.choices(["A", "B", "C", "D", "E", "F", "G"], k=6))

    if "code" in name:
        return "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)"

    if "paper" in name or "doc" in name or "file" in name:
        return f"{name}_{random.randint(1, 99)}.txt"

    if "answer" in name:
        return random.choice(["The longest common subsequence is ABCD.", "O(n*m) where n and m are string lengths.", "It avoids recomputation using memoization."])

    if re.fullmatch(r"w\d", name.lower()) or "weight" in name.lower():
        return str(random.randint(1, 25))

    if re.fullmatch(r"v\d", name.lower()) or "value" in name.lower():
        return str(random.randint(10, 100))

    if "chat" in name or "log" in name:
        return f"chatlog_{random.randint(10, 99)}.txt"

    if "typed" in name or "correct" in name:
        return random.choice(["dynamc", "dynmaic", "dynamic"])

    if "dims" in name:
        return [random.randint(2, 10) for _ in range(random.randint(3, 5))]

    if "items" in name or "item_list" in name:
        items = [f"{chr(65 + i)}(weight={random.randint(1,10)}, value={random.randint(10,100)})" for i in range(random.randint(3, 5))]
        return ", ".join(items)

    if name.lower() in {"cap1", "cap2"}:
        return str(random.randint(5, 25))

    if re.fullmatch(r"d\d", name.lower()):
        return str(random.randint(2, 100))

    if name.lower() in {"capacity", "cap", "cap1", "cap2"}:
        return str(random.randint(2, 20))

    if name.lower() == "m":
        return str(random.randint(2, 20))

    if name.lower() == "arr":
        return [random.randint(1, 30) for _ in range(random.randint(4, 8))]
    
    # Matrix dimensions for MCM (e.g., d1, d2, d3)
    if re.fullmatch(r"d\d", name.lower()):
        return str(random.randint(2, 20))  # Keep values small for LCM demo

    # Chain of matrix dimensions (e.g., dims)
    if name.lower() == "dims":
        length = random.choice([4, 5])  # e.g., 3–4 matrices = 4–5 dims
        return [random.randint(2, 20) for _ in range(length)]
    
    if re.search(r"\b(field|application|domain|scenario|context)\b", name.lower()) or \
   any(kw in name.lower() for kw in ["field", "application", "context", "domain", "scenario"]):
        return random.choice([
        "robotic surgery", "autonomous drones", "power grid control", "real-time stock trading",
        "autonomous vehicles", "manufacturing systems", "weather forecasting", "signal processing",
        "video game AI", "IoT-based automation", "cyber-physical systems", "spacecraft navigation"
    ])  

    if name in {"problem_name"}:
        return random.choice([
            "Fibonacci", "Factorial", "Subset Sum", "Knapsack", "Edit Distance", 
            "Climbing Stairs", "Palindrome Partition", "Unique Paths"
        ])

    if name in {"problem_type"}:
        return random.choice([
            "combinatorial", "recursive", "path-finding", "optimization"
        ])

    if name in {"nth", "n", "target"}:
        return str(random.randint(5, 20))

    if name == "arr" or name == "nums":
        return [random.randint(1, 15) for _ in range(random.randint(4, 7))]

    if name == "coins":
        return [1, 2, 5]  # simple coin set

    if name == "prices" or name == "values":
        return [random.randint(2, 20) for _ in range(4)]

    if name == "weights":
        return [random.randint(1, 10) for _ in range(4)]

    if name in {"pattern", "text", "expr", "source", "word1", "word2", "str1", "str2"}:
        return ''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 8)))

    if name == "grid":
        return [[random.randint(1, 9) for _ in range(3)] for _ in range(3)]

    if name == "dimensions":
        return [random.randint(2, 10) for _ in range(random.randint(3, 5))]

    if name == "items":
        items = []
        for i in range(4):
            weight = random.randint(1, 5)
            value = random.randint(10, 50)
            items.append(f"Item{i+1}(weight={weight}, value={value})")
        return ", ".join(items)


    # NEW: For questions using 'steps' like city visits, parentheses, etc.
    if name == "steps":
        return str(random.randint(4, 20))

# NEW: For grammar parsing questions
    if name == "rules":
        return {
        "S": ["AB", "BC"],
        "A": ["BA", "a"],
        "B": ["CC", "b"],
        "C": ["AB", "a"]
    }

# NEW: For general DP problem identifiers like 'DP_PROBLEM_XXX'
    if "dp_problem" in name:
        return random.choice([
        "coin change", "edit distance", "longest palindromic subsequence", "matrix chain multiplication"
    ])

# NEW: For third parameter in recursive states like f(i, k, t)
    if name == "t":
        return str(random.randint(1, 10))

# NEW: For parsing or grammar string
    if name == "str":
        return ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))

# NEW: For integer partition problems
    if name == "target" or name == "max_int":
        return str(random.randint(5, 20))

# Ensure 'graph' is handled correctly
    if name == "graph":
        size = 5
        return [[0 if i == j else random.randint(1, 10) for j in range(size)] for i in range(size)]



    return f"{name.upper()}_{random.randint(100, 999)}"



def generate_string(level):
    min_len, max_len = STRING_LENGTH_BY_LEVEL.get(level, (2, 4))
    return ''.join(random.choices(string.ascii_uppercase, k=random.randint(min_len, max_len)))

#took this function below of genrate function before it was above the genrate_string function
generate_placeholder_value.cache = {}


def fill_placeholders(template, level):
    generate_placeholder_value.cache = {}
    placeholders = re.findall(r"\{\{(.*?)\}\}", template)
    for ph in placeholders:
        val = generate_placeholder_value(ph.strip(), level)
        replacement = "[" + ', '.join(map(str, val)) + "]" if isinstance(val, list) else str(val)
        template = template.replace(f"{{{{{ph}}}}}", replacement)
    return template


# -------------------------------
# Flatten nested template structure
# -------------------------------
def flatten_templates(raw_json):
    flat_templates = []
    for topic, categories in raw_json.items():
        for category, levels in categories.items():
            for level, templates in levels.items():
                for template in templates:
                    flat_templates.append({
                        "template": template,
                        "category": category,
                        "difficulty": level,
                        "topic": topic
                    })
    return flat_templates

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="📘 DP Question Generator", layout="centered")
st.title("📘 Auto Question Generator (Dynamic Programming Topics)")

# Load JSON files
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    st.error("❌ Folder named 'data' not found.")
    st.stop()

json_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
if not json_files:
    st.error("❌ No .json files found in 'data/' directory.")
    st.stop()

topic_file = st.selectbox("Select Topic", json_files)
with open(os.path.join(DATA_DIR, topic_file), "r", encoding="utf-8") as f:
    raw_json = json.load(f)

templates = flatten_templates(raw_json)

# Extract available options
topics = sorted(set(t["topic"] for t in templates))
selected_topic = st.selectbox("Select Subtopic", topics)

levels = ["Level 1", "Level 2", "Level 3"]
selected_level = st.selectbox("Select Difficulty", levels)

categories = sorted(set(t["category"] for t in templates if t["difficulty"] == selected_level and t["topic"] == selected_topic))
selected_category = st.selectbox("Select Category", categories)

num_questions = st.slider("Number of Questions", 1, 10, 3)

# Reset state if user changes topic/category/level
current_combo = (selected_topic, selected_category, selected_level)
if "previous_combo" in st.session_state:
    if st.session_state["previous_combo"] != current_combo:
        st.session_state["generate_clicked"] = False
else:
    st.session_state["previous_combo"] = current_combo

# Always update the current combo
st.session_state["previous_combo"] = current_combo

# BEFORE THE BUTTON
if "generate_clicked" not in st.session_state:
    st.session_state["generate_clicked"] = False

# BUTTON
if st.button("🚀 Generate Questions"):
    st.session_state["generate_clicked"] = True

# MAIN LOGIC (MOVED OUTSIDE BUTTON)
if st.session_state["generate_clicked"]:
    filtered = [
        t for t in templates
        if t["difficulty"] == selected_level and t["category"] == selected_category and t["topic"] == selected_topic
    ]
    if not filtered:
        st.warning("No templates match your selection.")
    else:
        st.subheader("🧠 Generated Questions")
        display_theory_if_available(selected_topic, selected_category, selected_level)

        if "mcqs" not in st.session_state or "tfqs" not in st.session_state:
            mcqs, tfqs = load_objective_questions(selected_topic, selected_category, selected_level)
            st.session_state["mcqs"] = mcqs
            st.session_state["tfqs"] = tfqs
        else:
            mcqs = st.session_state["mcqs"]
            tfqs = st.session_state["tfqs"]

        # === MCQs ===
                # === MCQs with Pagination ===
        if mcqs:
            with st.expander("📝 Practice MCQs", expanded=False):
                st.markdown("### 🎯 Multiple Choice Questions")

                # --- Pagination Setup ---
                questions_per_page = 10
                total_questions = len(mcqs)
                total_pages = (total_questions - 1) // questions_per_page + 1

                if "mcq_page" not in st.session_state:
                    st.session_state["mcq_page"] = 0

                current_page = st.session_state["mcq_page"]
                start_index = current_page * questions_per_page
                end_index = min(start_index + questions_per_page, total_questions)

                for i in range(start_index, end_index):
                    item = mcqs[i]
                    options = ["-- Select an option --"] + item["options"]
                    user_choice = st.radio(f"Q{i+1}: {item['question']}", options, key=f"mcq_{i}")
                    if user_choice != "-- Select an option --":
                        if user_choice == item["answer"]:
                            st.success("✅ Correct!")
                        else:
                            st.error(f"❌ Incorrect. Correct Answer: {item['answer']}")

                # --- Navigation Buttons ---
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("⬅️ Previous", key="mcq_prev") and current_page > 0:
                        st.session_state["mcq_page"] -= 1
                        st.rerun()
                with col3:
                    if st.button("Next ➡️", key="mcq_next") and current_page < total_pages - 1:
                        st.session_state["mcq_page"] += 1
                        st.rerun()

                with col2:
                    st.markdown(f"<center>Page {current_page + 1} of {total_pages}</center>", unsafe_allow_html=True)

        # === True/False ===
        # === True/False with Pagination ===
        if tfqs:
            with st.expander("✔️ Practice True/False", expanded=False):
                st.markdown("### ✔️ True/False Questions")

                # --- Pagination Setup ---
                tf_per_page = 10
                total_tf = len(tfqs)
                total_tf_pages = (total_tf - 1) // tf_per_page + 1

                if "tf_page" not in st.session_state:
                    st.session_state["tf_page"] = 0

                current_tf_page = st.session_state["tf_page"]
                start_tf = current_tf_page * tf_per_page
                end_tf = min(start_tf + tf_per_page, total_tf)

                for i in range(start_tf, end_tf):
                    item = tfqs[i]
                    options = ["-- Select --", "True", "False"]
                    user_tf = st.radio(f"Q{i+1}: {item['question']}", options, key=f"tf_{i}")
                    correct = "True" if item["answer"] else "False"
                    if user_tf != "-- Select --":
                        if user_tf == correct:
                            st.success("✅ Correct!")
                        else:
                            st.error(f"❌ Incorrect. Correct Answer: {correct}")

                # --- Navigation Buttons ---
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("⬅️ Previous", key="tf_prev") and current_tf_page > 0:
                        st.session_state["tf_page"] -= 1
                        st.rerun()
                with col3:
                    if st.button("Next ➡️", key="tf_next") and current_tf_page < total_tf_pages - 1:
                        st.session_state["tf_page"] += 1
                        st.rerun()
                with col2:
                    st.markdown(f"<center>Page {current_tf_page + 1} of {total_tf_pages}</center>", unsafe_allow_html=True)

        # === Generated Questions and Answers ===
        from answer_dispatcher import get_answer
        generated = set()
        count = 0
        while count < num_questions and len(generated) < len(filtered) * 5:
            template = random.choice(filtered)
            q = fill_placeholders(template["template"], selected_level)
            if q not in generated:
                st.markdown(f"### ❓ Question {count + 1}")
                st.markdown(f"<div style='font-size:18px; font-weight:600;'>{q}</div>", unsafe_allow_html=True)
                generated.add(q)
                count += 1

                answer = get_answer(selected_topic, selected_category, selected_level, q)
                with st.expander("💡 View Answer", expanded=False):
                    st.code(answer, language="markdown")

