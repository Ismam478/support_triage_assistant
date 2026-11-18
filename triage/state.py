from operator import add
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langgraph.graph import add_messages


class TicketSchema(BaseModel):
    messages: Annotated[list, add_messages] = Field(..., description="The customer's messages related to the ticket.")
    customer_id: str = Field(..., description="The unique identifier for the customer.")
    category: Literal["billing", "tech", "general"] = Field(..., description="The category of the ticket.")
    urgency: Literal["low", 'high'] = Field(..., description="The urgency level of the ticket.")
    facts: str = Field(..., description="The facts related to the ticket.")
    draft: str = Field(..., description="The draft response waiting for approval.")
    feedback: str = Field(..., description="The feedback provided by the customer.")
    redrafts: list[str, add_messages] = Field(..., description="The list of redrafted responses for the ticket.", default_factory=list)

class RouterSchema(BaseModel):
    category: Literal["billing", "tech", "general"] = Field(..., description="The category of the ticket.")
    urgency: Literal["low", 'high'] = Field(..., description="The urgency level of the ticket.")


