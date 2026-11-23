from triage.state import current_ticket
from triage.drafter import check_rules
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage



def test_current_ticket_serialization():
    messages = [
        HumanMessage(content="Hello, I have a problem with my order A-1234."),
        AIMessage(content="Hello! I'm here to help you with your order A-1234."),
        HumanMessage(content="I received the wrong item. I ordered A-1234 but got A-5678."),
        AIMessage(content="I'm sorry to hear that. Let's get this sorted out for you.", tool_call_id="tool_call_1"),
        ToolMessage(content="Order A-1234 was shipped on 2023-01-01. The customer received item A-5678 instead of A-1234.", tool_call_id="tool_call_1")]
    current = current_ticket(messages)
    assert len(current) == 3
    assert current == [HumanMessage(content="I received the wrong item. I ordered A-1234 but got A-5678."), AIMessage(content="I'm sorry to hear that. Let's get this sorted out for you.", tool_call_id="tool_call_1"), ToolMessage(content="Order A-1234 was shipped on 2023-01-01. The customer received item A-5678 instead of A-1234.", tool_call_id="tool_call_1")]

def test_check_rules_min_length():
    facts = ["Order A-1234 was shipped on 2023-01-01.", "The customer received item A-5678 instead of A-1234."]
    draft = "Hello, We apologize for the inconvenience. It seems there was a mix-up with your order A-1234. Our agent will assist you in resolving this issue. NovaGear Support"
    result = check_rules(draft, facts)
    word_count = len(draft.split())
    assert result == [f"Too short: {word_count} words. Write at least 50 words."]

def test_check_rules_max_length():
    facts = ["Order A-1234 was shipped on 2023-01-01.", "The customer received item A-5678 instead of A-1234."]
    draft = "Hello, We sincerely apologize for the inconvenience and frustration caused by the mix-up with your recent order, A-1234. At NovaGear, we strive to ensure every package arrives accurately and on time, and we deeply regret that we fell short of our usual standards in this instance. One of our dedicated support agents is ready to assist you in resolving this issue right away. To help us expedite the replacement process, please reply with your full order details—including the specific item you originally purchased and a brief description or photo of what you received instead. Once we verify this information, we will immediately arrange for the correct item to be dispatched to you at no additional cost, along with instructions for returning the incorrect product if necessary. Our team will follow up with you as soon as possible, and we will make sure everything is sorted out shortly. Thank you for your patience and understanding. Best regards, NovaGear Support"
    result = check_rules(draft, facts)
    word_count = len(draft.split())
    assert result == [f"Too long: {word_count} words. Write at most 120 words."]

def test_check_rules_order_id():
    facts = ["Order A-1234 was shipped on 2023-01-01.", "The customer received item A-5678 instead of A-1234."]
    draft = "Hello, we sincerely apologize for the inconvenience and frustration caused by a recent mix-up with your order A-9999. We strive to get every package right, and we regret falling short here. One of our dedicated support agents will reach out and assist you in resolving this issue promptly. NovaGear Support"
    result = check_rules(draft, facts)
    assert result == ["The draft mentions order A-9999, which is not in the facts.", "The draft does not mention any order from the facts."]

def test_check_rules_fact_id():
    facts = ["Order A-1234 was shipped on 2023-01-01.", "The customer received item A-5678 instead of A-1234."]
    draft = "Hello, we sincerely apologize for the inconvenience and any frustration this has caused you. One of our dedicated support agents will assist you in resolving this issue right away. Please provide us with your full order details, and we will ensure that the correct item is sent to you promptly. Our team will review your information and get back to you as soon as possible. Thank you for your patience, NovaGear Support"
    result = check_rules(draft, facts)
    assert result == ["The draft does not mention any order from the facts."]

def test_check_rules_dollar_amount():
    facts = ["Order A-1234 was shipped on 2023-01-01. The total charge was $49.", "The customer received item A-5678 instead of A-1234."]
    draft = "Hello, we sincerely apologize for the inconvenience and frustration caused by a mix-up with your order A-1234. We noticed the total charge for your order was $99, which is completely incorrect. One of our dedicated support agents will assist you in resolving this billing issue promptly. Thank you, NovaGear Support"
    result = check_rules(draft, facts)
    assert result == ["The draft mentions $99, which is not in the facts."]

def test_check_rules_question():
    facts = ["Order A-1234 was shipped on 2023-01-01.", "The customer received item A-5678 instead of A-1234."]
    draft = "Hello, we sincerely apologize for the inconvenience and frustration caused by a mix-up with your recent order A-1234. To help us look into this right away, can you please provide us with your full order details? One of our dedicated agents will assist you in resolving this issue. NovaGear Support"
    result = check_rules(draft, facts)
    assert result == ["The draft contains a question. Do not ask questions."]

def test_check_rules_sign_off():
    facts = ["Order A-1234 was shipped on 2023-01-01.", "The customer received item A-5678 instead of A-1234."]
    draft = "Hello, we sincerely apologize for the inconvenience and frustration caused by a recent mix-up with your order A-1234. We are committed to making things right for you as quickly as possible. One of our dedicated customer support agents will reach out and assist you in resolving this issue promptly. Thank you for your patience and understanding."
    result = check_rules(draft, facts)
    assert result == ["End with the sign-off NovaGear Support."]

def test_check_rules_good_draft():
    facts = ["Order A-1234 was shipped on 2023-01-01. The total charge was $49.", "The customer received item A-5678 instead of A-1234."]
    draft = "Hello, we sincerely apologize for the inconvenience and frustration caused by a mix-up with your recent order A-1234. Your total charge for this order was $49, and we understand how disappointing it must be to receive the wrong item. One of our dedicated customer support agents will reach out to you promptly to assist in resolving this issue. Best regards, NovaGear Support"
    result = check_rules(draft, facts)
    assert result == []