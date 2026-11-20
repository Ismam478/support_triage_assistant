from langchain_core.messages import SystemMessage, ToolMessage
from triage.state import TicketSchema, RouterSchema
from langchain_core.prompts import ChatPromptTemplate
from triage.tools import list_orders, lookup_orders, search_faq
from triage.llm import get_llm
from typing import Literal

BILLING = "Queries related to money, invoices, subscriptions, payments, or refunds"
TECH = "Queries related to bugs, errors, system outages, login issues, or how to use a specific software feature"
GENERAL = "Anything else that doesn't fit into billing or technical support. This could be questions about company policy, opening hours, business inquiries, or general feedback"

BILLING_TOOLS = [list_orders, lookup_orders]
TOOLS_BOX = [search_faq]

def router_node(state: TicketSchema) -> dict:
    """
    Reads the user ticket and categorizes it into three possible nodes: billing, tech, or general. 
    """
    messages = state.messages[-1].content
    if not messages:
        raise ValueError("No messages found in the ticket.")
    llm = get_llm().with_structured_output(RouterSchema)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are a customer support triage assistant. Your task is to read the user's ticket categorize it into one of three possible categories: Billing: {BILLING}, Tech: {TECH}, General: {GENERAL}. Also define the urgency of the ticket. High: money was taken or may be wrongly taken (double charge, unauthorized payment), account compromised, or the customer cannot use the product at all. Low: everything else, including routine password resets and general questions."),
            ("user", "Here is the user's ticket: {messages}. Please categorize it into one of the three category: billing, tech, or general.") 
        ]
    )
    chain = prompt | llm

    result = chain.invoke({"messages": messages})

    category = result.category
    urgency = result.urgency

    return {"category": category, "urgency": urgency, "llm_calls": 1}


def custom_tool_node(state:TicketSchema) -> dict:
    '''
    This is the custom tool node, which takes the tool call from LLM and invokes the tool and returns the ToolMessage generated from the tool.
    '''

    last_message = state.messages[-1]
    category = state.category
    if category == "billing":
        tool_box = BILLING_TOOLS
    elif category == "tech" or category == "general":
        tool_box = TOOLS_BOX

    tools_by_name = {tool.name: tool for tool in tool_box}

    tool_results = []

    if last_message.tool_calls:
        for preferred_tool in last_message.tool_calls:
            tool_name = preferred_tool['name']
            tool_args = preferred_tool['args']
            tool_id = preferred_tool['id']
            try:
                selected_tool = tools_by_name[tool_name]
                tool_response = selected_tool.invoke(tool_args)

                tool_results.append(ToolMessage(content=str(tool_response), tool_call_id=tool_id))       

            except Exception as e:
                tool_results.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_id, status="error"))       
    return {'messages': tool_results}

def if_tool_call(state:TicketSchema) -> Literal["tool_node", "end"]:
    last_message = state.messages[-1]
    if last_message.tool_calls:
        return 'tool_node'
    else:
        return 'end'

INSTRUCTIONS = {
    "billing": "You are a billing specialist. Output only facts, as a bullet list. Do not offer to do anything, do not ask questions, do not address the customer. Your task is to read the user's ticket and use the available tools to provide only the accurate information. Do not make up any information. If the information is not available, state what you dont know. If no order id is given, call list_orders, then call lookup_orders for each recent order (at most 3), and compare the charges.",
    "tech": "You are tech support specialist. Output only facts, as a bullet list. Do not offer to do anything, do not ask questions, do not address the customer. Your task is to read the user's ticket and use the available tools to provide only the accurate information. Do not make up any information. If the information is not available, state what you dont know. ",
    "general": "You are a general support specialist. Output only facts, as a bullet list. Do not offer to do anything, do not ask questions, do not address the customer. Your task is to read the user's ticket and use the available tools to provide only the accurate information. Do not make up any information. If the information is not available, state what you dont know. "
}

def specialist_node(state: TicketSchema):

    messages = state.messages
    tools = BILLING_TOOLS if state.category == "billing" else TOOLS_BOX
    instructions = INSTRUCTIONS[state.category]

    system = SystemMessage(
    content=f"Instructions: {instructions}\nThe customer id is {state.customer_id}. "
            f"Pass this exact id in every tool call."
    )

    response = get_llm(reasoning_effort="low").bind_tools(tools).invoke([system] + messages)
    
    usage = response.usage_metadata or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return {"messages": [response], "llm_calls": 1, "input_tokens": input_tokens, "output_tokens": output_tokens}

def facts_node(state: TicketSchema) -> dict:
    return {"facts": [state.messages[-1].content]}
