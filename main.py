from triage.graph import build_graph
from langgraph.checkpoint.sqlite import SqliteSaver
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

            results = app.invoke({"messages": [{"role": "user", "content": query}], "customer_id": customer_id, "facts": [], "draft": "", "feedback": "", "redrafts": 0}, config=config)
            draft = results["draft"]
            for m in results["messages"]:
                print(type(m).__name__, getattr(m, "tool_calls", ""), str(m.content)[:80])
            print("\nCategory:", results["category"])
            print("Urgency:", results["urgency"])
            

            draft =review_loop(draft, app, config)

            print("Drafted Response:", draft)
            print("Facts:", results["facts"])
            print("Input Tokens:", results["input_tokens"])
            print("Output Tokens:", results["output_tokens"])
            print("LLM Calls:", results["llm_calls"])


if __name__ == "__main__":
    main()
