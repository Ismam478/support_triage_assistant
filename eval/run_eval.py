from triage.graph import build_graph
from triage.drafter import check_rules
from langgraph.checkpoint.memory import InMemorySaver
import json


def load_questions():
    with open("eval/holdout.json", "r") as f:
        questions = json.load(f)
    return questions

def run_eval():
    checkpointer = InMemorySaver()
    app = build_graph(checkpointer=checkpointer)

    questions = load_questions()

    rows = []
    errors = []
    for q in questions:
        try:
            config = {'configurable': {'thread_id': f"eval-{q['id']}"}, 'recursion_limit': 12}

            response = app.invoke({"messages": [{"role": "user", "content": q["query"]}], "customer_id": q["customer_id"], "facts": [], "draft": "", "feedback": "", "redrafts": 0}, config=config)
            

            if '__interrupt__' in response:

                state = app.get_state(config=config).values
                
                ai = [m for m in response["messages"] if m.type == "ai"]
                tools_used = []
                for msg in ai:
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            tools_used.append(tc["name"])

                rules_result = check_rules(state["draft"], state["facts"])
                rules_error = 0
                if rules_result != []:
                    rules_error += 1

                result = {
                    "id": q["id"],
                    "query": q["query"],
                    "rules_error_count": rules_error,
                    "customer_id": state["customer_id"],
                    "expected_category": q["expected_category"],
                    "category": state["category"],
                    "expected_urgency": q["expected_urgency"],
                    "urgency": state["urgency"],
                    "expected_tools": q["expected_tools"],
                    "tools_used" : tools_used,
                    "draft": state["draft"],
                    "facts": state["facts"],
                    "tricky": q["tricky"],
                    "llm_calls": state["llm_calls"],
                    "input_tokens": state["input_tokens"],
                    "output_tokens": state["output_tokens"]
                }
                rows.append(result)


        except Exception as e:
            print(f"Error processing question id: {q['id']}\nError: {e}")    
            errors.append({"id": q["id"], "error": e})
            continue
        
    return rows, errors
    


def eval_counts(rows, errors):
    length = len(rows)
    length_errors = len(errors)
    counts = {
        "total": length,
        'total_errors': length_errors,
        "billing_category_accuracy": 0,
        "total_billing_questions": 0,
        "tech_category_accuracy": 0,
        "total_tech_questions": 0,
        "general_category_accuracy": 0,
        "total_general_questions": 0,
        "urgency_accuracy": 0,
        "total_urgency_questions": 0,
        "total_expected_tools": 0,
        "tools_used_accuracy": 0,
        "tricky_accuracy": 0,
        "total_tricky_questions": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "avg_cost_per_question": 0,
        "avg_rules_errors_per_question": 0,
    }

    for r in rows:
        if r["expected_category"] == "billing":
            counts["total_billing_questions"] += 1
        elif r["expected_category"] == "tech":
            counts["total_tech_questions"] += 1
        elif r["expected_category"] == "general":
            counts["total_general_questions"] += 1

        if r["expected_urgency"]:
            counts["total_urgency_questions"] += 1
        if r["tricky"]:
            counts["total_tricky_questions"] += 1

        for tool in r["expected_tools"]:
            counts["total_expected_tools"] += 1

    for r in rows:
        if r["category"] == r["expected_category"]:
            if r["category"] == "billing":
                counts["billing_category_accuracy"] += 1
            elif r["category"] == "tech":
                counts["tech_category_accuracy"] += 1
            elif r["category"] == "general":
                counts["general_category_accuracy"] += 1
    
        if r["urgency"] == r["expected_urgency"]:
            counts["urgency_accuracy"] += 1

        if set(r["expected_tools"]) <= set(r["tools_used"]):
            counts["tools_used_accuracy"] += 1

        if r["tricky"] and r["category"] == r["expected_category"] and r["urgency"] == r["expected_urgency"] and set(r["expected_tools"]) <= set(r["tools_used"]):
            counts["tricky_accuracy"] += 1

        if r["rules_error_count"] > 0:
            counts["avg_rules_errors_per_question"] += 1
        
        counts["input_tokens"] += r["input_tokens"]
        counts["output_tokens"] += r["output_tokens"]

    counts["avg_rules_errors_per_question"] = counts["avg_rules_errors_per_question"] / len(rows) 
    counts["avg_cost_per_question"] = ((0.05 * counts["input_tokens"]/100000) + (0.4 * counts["output_tokens"] / 1000000)) / len(rows)

    return counts


if __name__ == "__main__":
    print(f"{'-'*15} Starting evaluation {'-'*15}")
    rows = run_eval()
    rows, errors = run_eval()
    counts = eval_counts(rows, errors)

    try:
        print("\nEvaluation Counts:")
        print(f"Total Questions: {counts['total']}")
        print(f"Total Errors: {counts['total_errors']}")
        print(f"Billing Category Accuracy: {counts['billing_category_accuracy']}/{counts['total_billing_questions']}")
        print(f"Tech Category Accuracy: {counts['tech_category_accuracy']}/{counts['total_tech_questions']}")
        print(f"General Category Accuracy: {counts['general_category_accuracy']}/{counts['total_general_questions']}")
        print(f"Urgency Accuracy: {counts['urgency_accuracy']}/{counts['total_urgency_questions']}")
        print(f"Tools Used Accuracy: {counts['tools_used_accuracy']}/{counts['total_expected_tools']}")
        print(f"Tricky Accuracy: {counts['tricky_accuracy']}/{counts['total_tricky_questions']}")
        print(f"Average Cost per Question: ${counts['avg_cost_per_question']:.6f}")
        print(f"Average Rules Errors per Question: {counts['avg_rules_errors_per_question']:.4f}")


        with open("eval/results_2.json", "w") as f:
            json.dump(rows, f, indent=4)
        print("Results saved to eval/results_2.json")
    
    except Exception as e:
        with open("eval/results_2.json", "w") as f:
                    json.dump(rows, f, indent=4)
                    print("Results saved to eval/results_2.json")
        print(f"Error : {e}")
