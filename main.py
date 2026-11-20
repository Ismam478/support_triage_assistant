from triage.graph import build_graph

def main():
    app = build_graph()
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        config = {'recursion_limit': 12}
        results = app.invoke({"messages": [{"role": "user", "content": query}], "customer_id": input("Enter Customer ID: ")}, config=config)

        for m in results["messages"]:
            print(type(m).__name__, getattr(m, "tool_calls", ""), str(m.content)[:80])
        print("Category:", results["category"])
        print("Urgency:", results["urgency"])
        print("Facts:", results["facts"])
        print("Input Tokens:", results["input_tokens"])
        print("Output Tokens:", results["output_tokens"])
        print("LLM Calls:", results["llm_calls"])


if __name__ == "__main__":
    main()
