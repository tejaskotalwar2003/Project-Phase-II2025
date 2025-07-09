import importlib
import os
import sys
# 🔁 Import conceptual functions




# Add answer_engine/ to path
ANSWER_ENGINE_PATH = os.path.join(os.path.dirname(__file__), "answer_engine")
if ANSWER_ENGINE_PATH not in sys.path:
    sys.path.append(ANSWER_ENGINE_PATH)

# Topic to module
TOPIC_MODULE_MAP = {
    "Fibonacci": "fibonacci_answers",
    "LCS": "lcs_answers",
    "Knapsack": "knapsack_answers",
    "MCM": "mcm_answer",
    "Memoization": "memoization_answer"
}

# Dispatcher map: (Topic, Category) → (module, function, args)
DISPATCH_TABLE = {
    ("Fibonacci", "Algorithmic"): ("fibonacci_answers", "answer_algorithmic_fibonacci", ["level", "question"]),
    ("Fibonacci", "Application"): ("fibonacci_answers", "answer_application_fibonacci", ["level", "question"]),
    ("Fibonacci", "Optimization"): ("fibonacci_answers", "answer_optimization_fibonacci", ["level", "question"]),
    ("Fibonacci", "Conceptual"): ("conceptual_answers.fibonacci_conceptual_answers", "answer_conceptual_fibonacci", ["level", "question"]),
    

    ("LCS", "Algorithmic"): ("lcs_answers", "generate_answer", ["category", "level", "question"]),
    ("LCS", "Application"): ("lcs_answers", "answer_app_lcs", [ "level", "question"]),
    ("LCS", "Implementation"): ("lcs_answers", "answer_impl_lcs", [ "level", "question"]),
    ("LCS", "Optimization"): ("lcs_answers", "answer_opt_lcs", ["level", "question"]),
    ("LCS", "Conceptual"): ("conceptual_answers.lcs_conceptual_answers", "answer_conceptual_lcs", ["level", "question"]),

    ("Knapsack", "Algorithmic"): ("knapsack_answers", "answer_algorithmic_knapsack", ["level", "question"]),
    ("Knapsack", "Implementation"): ("knapsack_answers", "answer_implementation_knapsack", ["level", "question"]),
    ("Knapsack", "Conceptual"): ("conceptual_answers.knapsack_conceptual_answers", "answer_conceptual_knapsack", ["level", "question"]),

    ("MCM", "Algorithmic"): ("mcm_answer", "answer_algorithmic_mcm", ["level", "question"]),
    ("MCM", "Application"): ("mcm_answer", "answer_application_mcm", ["level", "question"]),
    ("MCM", "Optimization"): ("mcm_answer", "answer_optimization_mcm", ["level", "question"]),

    ("Memoization", "Conceptual"): ("conceptual_answers.memoization_conceptual_answers", "answer_conceptual_memoization", ["level","question"]),
    ("Memoization", "Application"): ("memoization_answer", "answer_memoization_application", ["level","question"]),
    ("Memoization", "Quantitative"): ("memoization_answer", "answer_memoization_quantitative", ["level","question"])
    }


def get_answer(topic, category, level, question):
    try:
        key = (topic, category)
        if key not in DISPATCH_TABLE:
            return f"⚠️ No answer function mapped for topic `{topic}` and category `{category}`."

        module_name, func_name, arg_order = DISPATCH_TABLE[key]

        module = importlib.import_module(module_name)
        func = getattr(module, func_name, None)

        if not func:
            return f"⚠️ Function `{func_name}` not found in `{module_name}`."

        # Prepare arguments dynamically
        args = {
            "topic": topic,
            "category": category,
            "level": level,
            "question": question
        }

        selected_args = [args[arg] for arg in arg_order]
        return func(*selected_args)

    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"
