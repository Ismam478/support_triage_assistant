from triage.graph import build_graph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from pathlib import Path

from triage.nodes import review_loop

def main():

    here = Path(__file__).parent

    with SqliteSaver.from_conn_string((here/ "checkpoints.sqlite")) as checkpointer:
        checkpointer.setup()

        app = build_graph(checkpointer=checkpointer)
        
        #### Saving the graph as a PNG or Mermaid file
        drawing = app.get_graph()
        try:
            (here / "graph.png").write_bytes(drawing.draw_mermaid_png())
            print("Saved graph.png")
        except Exception as e:
            (here / "graph.mmd").write_text(drawing.draw_mermaid())
            print(f"PNG failed ({type(e).__name__}); saved graph.mmd instead")

        while True:
            query = input("Enter your question (or 'exit' to quit): ")
            if query.lower() == 'exit':
                print("Exiting....")
                break

            customer_id = input("Enter customer ID: ")
            config = {'configurable': {'thread_id': customer_id}, 'recursion_limit': 12}

            before = app.get_state(config).values
            try:
                # results = app.invoke({"messages": [{"role": "user", "content": query}], "customer_id": customer_id, "facts": [], "draft": "", "feedback": "", "redrafts": 0}, config=config)
                for chunk in app.stream({"messages": [{"role": "user", "content": query}], "customer_id": customer_id, "facts": [], "draft": "", "feedback": "", "redrafts": 0}, config=config, stream_mode="updates"):
                    print(f"\n{chunk}")
                    if chunk.get("drafter"):
                        draft = chunk["drafter"]["draft"]
                    

                # draft = results["draft"]
                # for m in results["messages"]:
                #     print(type(m).__name__, getattr(m, "tool_calls", ""), str(m.content)[:80])
                # print("\nCategory:", results["category"])
                # print("Urgency:", results["urgency"])
                
                draft = review_loop(draft, app, config)
                after = app.get_state(config).values
            
            except GraphRecursionError as e:
                print(f"GraphRecursionError: {e}")
                continue

            calls = after["llm_calls"] - before.get("llm_calls", 0)
            input_tokens = after["input_tokens"] - before.get("input_tokens", 0)
            output_tokens = after["output_tokens"] - before.get("output_tokens", 0)

            print("Drafted Response:", after["draft"])
            print("Redrafts:", after["redrafts"])
            print("Facts:", after["facts"])
            print("Input Tokens:", after["input_tokens"])
            print("Input Tokens (this iteration):", input_tokens)
            print("Output Tokens:", after["output_tokens"])
            print("Output Tokens (this iteration):", output_tokens)
            print("LLM Calls:", after["llm_calls"])
            print("LLM Calls (this iteration):", calls)

            total_cost_per_token = (input_tokens * (0.05/1000000)) + (output_tokens * (0.40/1000000)) # According to GPT-5-nano pricing (https://developers.openai.com/api/docs/models/gpt-5-nano)
            print("Total Cost (this iteration): $", total_cost_per_token)


if __name__ == "__main__":
    main()
