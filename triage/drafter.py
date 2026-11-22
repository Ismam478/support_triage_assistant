from triage.state import TicketSchema, current_ticket
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from triage.llm import get_llm
import re

MIN_WORDS = 50
MAX_WORDS = 120

def check_rules(draft: str, facts: list[str]) -> list[str]:
    """
    Checks the draft against the facts and returns a list of rules that are violated.
    """
    problems = []
    facts_text = "\n".join(facts)

    # 1. Length
    word_count = len(draft.split())
    if word_count < MIN_WORDS:
        problems.append(f"Too short: {word_count} words. Write at least {MIN_WORDS}.")
    if word_count > MAX_WORDS:
        problems.append(f"Too long: {word_count} words. Write at most {MAX_WORDS}.")

    # 2. Order ids
    draft_ids = set(re.findall(r"A-\d{4}", draft))
    fact_ids = set(re.findall(r"A-\d{4}", facts_text))

    for order_id in sorted(draft_ids - fact_ids):
        problems.append(f"The draft mentions order {order_id}, which is not in the facts.")

    if fact_ids and not (draft_ids & fact_ids):
        problems.append("The draft does not mention any order from the facts.")

    # 3. Dollar amounts
    fact_numbers = set(re.findall(r"\d+", facts_text))
    for amount in re.findall(r"\$(\d+)", draft):
        if amount not in fact_numbers:
            problems.append(f"The draft mentions ${amount}, which is not in the facts.")

    if "?" in draft:
        problems.append("The draft contains a question. Do not ask questions.")

    if not draft.rstrip().endswith("NovaGear Support"):
        problems.append("End with the sign-off NovaGear Support.")

    if re.search(r"\b(True|False)\b|Customer ID", draft):
        problems.append("Do not copy raw fields like 'Customer ID' or True/False. Explain in plain words.")

    if re.search(r"^\s*subject:", draft, re.IGNORECASE):
        problems.append("Remove the subject line.")

    if not re.match(r"\s*(Hi|Hello|Dear)\b", draft):
        problems.append("Start with a greeting such as Hello.")

    if re.search(r"(\bC\d{3}\b)", draft):
        problems.append("Do not mention the csutomer id (C123) in the draft.")
        

    return problems

def drafter_node(state: TicketSchema) -> dict:
    """
    This is the drafter node, which takes the facts and generates a draft response.
    """
    messages = state.messages
    text_ticket = current_ticket(messages)[0].content
              
    facts = state.facts
    category = state.category
    urgency = state.urgency
    feedback = state.feedback
    redrafts = state.redrafts
    llm = get_llm()

    if feedback and redrafts <= 3:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a {category} customer support email drafter(Make sure to follow the email layout). Use only the feedback to improve the draft. Everything else is the same as before. "),
                ("user", "User's ticket: {text_ticket}\nUser's urgency: {urgency}. Do not make up any single information. The following feedback was given on your previous draft: {feedback}. Please revise your draft to comply with this feedback: {draft}.")
            ]
        )
        chain = prompt | llm
        response = chain.invoke({"text_ticket": text_ticket, "urgency": urgency, "feedback": feedback, "draft": state.draft, "category": category})
        draft = response.content
        input_tokens = response.usage_metadata["input_tokens"]
        output_tokens = response.usage_metadata["output_tokens"]

        problems = check_rules(draft, facts)
        # return {'draft': draft, "llm_calls": 1, 'input_tokens': input_tokens, 'output_tokens': output_tokens, "feedback": ""}
    
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a {category} customer support email drafter(Make sure to follow the email layout). Use only the facts below: {facts}. Mention only the order(s) that relate to the customer's problem, and ignore other orders. Structure: greeting, one or two sentences about what we found, what happens next (including the timeline if the facts give one), and the sign-off \"NovaGear Support\". Do not list fields. Do not ask questions or offer anything. {MIN_WORDS} to {MAX_WORDS} words. If the facts don't contain the answer, say a team member will follow up. If urgency is high, acknowledge that in the first sentence."),
                ("user", "User's ticket: {text_ticket}\nUser's urgency: {urgency}. Do not make up any single information.")
            ]
        )

        
        chain = prompt | llm
        response = chain.invoke({"text_ticket": text_ticket, "urgency": urgency, "category": category, "facts": "\n".join(facts), "MIN_WORDS": MIN_WORDS, "MAX_WORDS": MAX_WORDS})
        
        draft = response.content
        input_tokens = response.usage_metadata["input_tokens"]
        output_tokens = response.usage_metadata["output_tokens"]

        problems = check_rules(draft, facts)
    
    count = 0
    if problems:
        if problems and feedback:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a {category} customer support email drafter(Make sure to follow the email layout). Use the facts below: {facts}. Use feedback to improve the draft feedback: {feedback}. {MIN_WORDS} to {MAX_WORDS} words. Keep everything else same as before."),
                    ("user", "User's ticket: {text_ticket}\nUser's urgency: {urgency}. Do not make up any single information. The following rules were violated in your previous draft: {problems}. Please revise your draft: {draft} to comply with these rules.")
                ]
            )
            while count < 3 and problems:
                count += 1
                redraft_chain = prompt | llm
                response = redraft_chain.invoke({"text_ticket": text_ticket, "urgency": urgency, "problems": problems, "draft": draft, "category": category, "facts": "\n".join(facts),"feedback": feedback, "MIN_WORDS": MIN_WORDS, "MAX_WORDS": MAX_WORDS})
                draft = response.content
                input_tokens += response.usage_metadata["input_tokens"]
                output_tokens += response.usage_metadata["output_tokens"]
                problems = check_rules(draft, facts)
        
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a {category} customer support email drafter(Make sure to follow the email layout). Use only the facts below: {facts}. Mention only the order(s) that relate to the customer's problem, and ignore other orders. Structure: greeting, one or two sentences about what we found, what happens next (including the timeline if the facts give one), and the sign-off \"NovaGear Support\". Do not list fields. Do not ask questions or offer anything. {MIN_WORDS} to {MAX_WORDS} words. If the facts don't contain the answer, say a team member will follow up. If urgency is high, acknowledge that in the first sentence."),
                    ("user", "User's ticket: {text_ticket}\nUser's urgency: {urgency}. Do not make up any single information. The following rules were violated in your previous draft: {problems}. Please revise your draft: {draft} to comply with these rules.")
                ]
            )
            while count < 3 and problems:
                count += 1
                redraft_chain = prompt | llm
                response = redraft_chain.invoke({"text_ticket": text_ticket, "urgency": urgency, "problems": problems, "draft": draft, "category": category, "facts": "\n".join(facts), "MIN_WORDS": MIN_WORDS, "MAX_WORDS": MAX_WORDS})
                draft = response.content
                input_tokens += response.usage_metadata["input_tokens"]
                output_tokens += response.usage_metadata["output_tokens"]
                problems = check_rules(draft, facts)
    
    if problems:
        print(f"\nProblems found: {problems}")

        
    return {'draft': draft, 'llm_calls': count+1, 'input_tokens': input_tokens, 'output_tokens': output_tokens, "feedback": ""}
