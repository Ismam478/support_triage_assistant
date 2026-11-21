from langchain_core.tools import tool
from triage.rag import query_vector_retriever, get_faq_vector_store
import json
from pathlib import Path

here = Path(__file__).parent
ORDER_FILE_PATH = here.parent / "data" / "orders.json"

@tool
def list_orders(customer_id: str) -> list[dict]:
    """
    Lists all orders for a given customer ID.
    """
    
    with open(ORDER_FILE_PATH, "r") as file:
        orders = json.load(file)

    customer_orders = [{'id': order['id'], 'date': order['date'], 'total': order['total'], 'status': order['status']} for order in orders if order["customer_id"] == customer_id]

    return customer_orders

@tool
def lookup_order(order_id: str, customer_id: str) -> dict:
    """
    Looks up a specific order by its ID.
    """
    
    with open(ORDER_FILE_PATH, "r") as file:
        orders = json.load(file)

    for order in orders:
        if order["id"] == order_id and order["customer_id"] == customer_id:
            return order

    return {"error": "Order not found"}

@tool
def search_faq(query: str) -> list[str]:
    """
    Searches the FAQ for relevant answers based on a query.
    """
    vector_store, embeddings_model = get_faq_vector_store()
    results = query_vector_retriever(query=query, k=2, vector_store=vector_store, embeddings_model=embeddings_model)

    return [result.page_content for result in results]