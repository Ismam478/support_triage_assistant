from triage.state import TicketSchema
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

def dummy_node(state: TicketSchema) -> dict:
    return {"redrafts": 3, "llm_calls": 1}

def build_test_graph(checkpointer=None):
    graph = StateGraph(TicketSchema)
    graph.add_node("dummy", dummy_node)
    graph.add_edge(START, "dummy")
    graph.add_edge("dummy", END)

    app = graph.compile(checkpointer=checkpointer)
    return app

def test_graph_execution():
    checkpointer = InMemorySaver()
    app = build_test_graph(checkpointer=checkpointer)
    initial_state = {"messages": [], "customer_id": "C001", "redrafts": 0}
    config = {"configurable": {"thread_id": "C001"}, "recursion_limit": 5}
    i = 0
    while i < 2:
        if i == 0:

            result = app.invoke(initial_state, config=config)
            after_redrafts = app.get_state(config).values["redrafts"]
            after_llm_calls = app.get_state(config).values["llm_calls"]
            i += 1

            assert after_redrafts == 3
            assert after_llm_calls == 1
        if i == 1:
            result = app.invoke(initial_state, config=config)
            
            initial_redrafts = list(app.get_state_history(config))[1].values["redrafts"]
            initial_llm_calls = list(app.get_state_history(config))[1].values["llm_calls"]
            
            after_redrafts = app.get_state(config).values["redrafts"]
            after_llm_calls = app.get_state(config).values["llm_calls"]
            i += 1
            
            assert initial_redrafts == 0 # redrafts reset to 0 for the new invocation
            assert initial_llm_calls == 1 # llm_calls stays the same for the new invocation
            assert after_redrafts == 3 # after calling the dummy node, redrafts becomes 3
            assert after_llm_calls == 2 # llm_calls increments by 1 for the new invocation