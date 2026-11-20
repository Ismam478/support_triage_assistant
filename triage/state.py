from operator import add
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langgraph.graph import add_messages


class TicketSchema(BaseModel):
    messages: Annotated[list, add_messages] = Field(..., description="The customer's messages related to the ticket.")
    customer_id: str = Field(..., description="The unique identifier for the customer.")
    category: Literal["billing", "tech", "general"] | None= Field(description="The category of the ticket.", default=None)
    urgency: Literal["low", 'high'] = Field(description="The urgency level of the ticket.", default="low")
    facts: list = Field(description="The facts related to the ticket.", default_factory=list)
    draft: str = Field(description="The draft response waiting for approval.", default="")
    feedback: str = Field(description="The feedback provided by the human agent .", default="")
    redrafts: Annotated[int, add] = Field(description="The number of times the response has been redrafted.", default=0)
    llm_calls: Annotated[int, add] = Field(description="The number of LLM calls made for the ticket.", default=0)
    input_tokens: Annotated[int, add] = Field(description="The number of input tokens used for the ticket.", default=0)
    output_tokens: Annotated[int, add] = Field(description="The number of output tokens generated for the ticket.", default=0)

class RouterSchema(BaseModel):
    category: Literal["billing", "tech", "general"] = Field(..., description="The category of the ticket.")
    urgency: Literal["low", 'high'] = Field(..., description="The urgency level of the ticket.")

