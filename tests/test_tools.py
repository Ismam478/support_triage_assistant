from triage.tools import list_orders, lookup_orders

def test_list_orders_returns_only_that_customers_orders():
    result = list_orders.invoke({"customer_id": "C002"})
    assert len(result) == 2                       

def test_lookup_returns_the_order_for_its_owner():
    result = lookup_orders.invoke({"order_id": "A-1002", "customer_id": "C002"})
    assert result["charges"] == [49, 49]

def test_lookup_hides_another_customers_order():
    result = lookup_orders.invoke({"order_id": "A-1001", "customer_id": "C002"})
    assert result == {"error": "Order not found"}

def test_lookup_unknown_order_returns_an_error():
    result = lookup_orders.invoke({"order_id": "A-9999", "customer_id": "C002"})
    assert result == {"error": "Order not found"}