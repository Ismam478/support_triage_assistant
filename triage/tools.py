from langchain_core.tools import tool
from triage.rag import query_vector_retriever
import json

@tool
def list_orders(customer_id: str) -> list[dict]:
    """
    Lists all orders for a given customer ID.
    """
    from pathlib import Path
    here = Path(__file__).parent
    orders_file = here / "orders.json"
    
    with open(orders_file, "r") as file:
        orders = json.load(file)

    customer_orders = [{'id': order['id'], 'date': order['date'], 'total': order['total'], 'status': order['status']} for order in orders if order["customer_id"] == customer_id]

    return customer_orders

@tool
def lookup_orders(order_id: str) -> dict:
    """
    Looks up a specific order by its ID.
    """
    from pathlib import Path
    here = Path(__file__).parent
    orders_file = here / "orders.json"
    
    with open(orders_file, "r") as file:
        orders = json.load(file)

    for order in orders:
        if order["id"] == order_id:
            return order

    return {"error": "Order not found"}

@tool
def search_faq(query: str) -> list[str]:
    """
    Searches the FAQ for relevant answers based on a query.
    """
    results = query_vector_retriever(query=query, k=2)

    return [result.page_content for result in results]