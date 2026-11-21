from triage.nodes import router_node, specialist_node, custom_tool_node, facts_node, if_tool_call, approval_node, write_email_to_txt
from triage.drafter import drafter_node
from langgraph.graph import StateGraph, START, END
from triage.state import TicketSchema

def build_graph(checkpointer=None):

    graph = StateGraph(TicketSchema)

    graph.add_node("router", router_node)
    graph.add_node("tool_node", custom_tool_node)
    graph.add_node("specialist", specialist_node)
    graph.add_node("facts", facts_node)
    graph.add_node("drafter", drafter_node)
    graph.add_node("approval", approval_node)
    graph.add_node("write", write_email_to_txt)

    graph.add_edge(START, "router")
    graph.add_edge("router", "specialist")
    graph.add_conditional_edges("specialist", if_tool_call, {"tool_node": "tool_node", "facts": "facts"})
    graph.add_edge("tool_node", "specialist")
    graph.add_edge("facts", "drafter")
    graph.add_edge("drafter", "approval")
    graph.add_edge("write", END)

    app = graph.compile(checkpointer=checkpointer)
    return app