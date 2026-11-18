from triage.state import TicketSchema, RouterSchema
from langchain_core.prompts import ChatPromptTemplate
from triage.llm import get_llm
from typing import Literal

BILLING = "Queries related to money, invoices, subscriptions, payments, or refunds"
TECH = "Queries related to bugs, errors, system outages, login issues, or how to use a specific software feature"
GENERAL = "Anything else that doesn't fit into billing or technical support. This could be questions about company policy, opening hours, business inquiries, or general feedback"


def router_node(state: TicketSchema) -> dict:
    """
    Reads the user ticket and categorizes it into three possible nodes: billing, tech, or general. 
    """
    messages = state.messages
    llm = get_llm().with_structured_output(RouterSchema)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are a customer support triage assistant. Your task is to read the user's ticket categorize it into one of three possible categiories: Billing: {BILLING}, Tech: {TECH}, General: {GENERAL}. Also define the urgency of the ticket as either low or high."),
            ("user", "Here is the user's ticket: {messages}. Please categorize it into one of the three category: billing, tech, or general.") 
        ]
    )
    chain = prompt | llm

    result = chain.invoke({"messages": messages})

    category = result.category
    urgency = result.urgency

    return {"category": category, "urgency": urgency}

